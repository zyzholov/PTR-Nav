#!/usr/bin/env python3
"""Run the four modular launchers in an isolated ROS domain and retain logs."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import yaml
import zipfile


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src/isaac_sim_hunter_livo2/scripts'))
from navigation_courses import (COMPOSITE_SCENARIOS, CROSSING_SCENARIOS, RAMP_SCENARIOS,
                                course_distance, default_obstacle_trigger_distance, minimum_course_distance)
from boundary_courses import load_spec,goal_offset
from lidar_mount import lidar_to_imu, mount_pitch_degrees


ISAAC_MAPPING_CONFIG = ROOT / "src/FAST_LIO/config/isaac_hunter_mid360_pointcloud.yaml"


def write_mount_mapping_config(log_dir, pitch_deg):
    if pitch_deg is None:
        return ISAAC_MAPPING_CONFIG
    config = yaml.safe_load(ISAAC_MAPPING_CONFIG.read_text())
    rotation, translation = lidar_to_imu(pitch_deg)
    mapping = config['/**']['ros__parameters']['mapping']
    mapping.update(extrinsic_R=rotation, extrinsic_T=translation)
    output = log_dir / ISAAC_MAPPING_CONFIG.name
    output.write_text(yaml.safe_dump(config, sort_keys=False))
    return output


def motion_rate_limits(backend):
    """Receive-time qualification, distinct from controller held-map expiry."""
    if backend=='gpu':return 5.0,.30
    if backend=='cpu':return .5,2.0
    raise ValueError('unsupported terrain backend')


def compute_stage_environment(environment,stage,pipe_directory):
    """Opt in only numeric compute launchers to a caller-owned private MPS.

    Isaac/RTX and mapping/observer processes retain ordinary CUDA operation.
    This helper neither creates a daemon nor changes GPU/driver modes.
    """
    result=dict(environment)
    if pipe_directory is None:return result
    for key in tuple(result):
        if key.startswith('CUDA_MPS_'):result.pop(key)
    if stage in ('terrain','planning'):
        result['CUDA_MPS_PIPE_DIRECTORY']=str(pipe_directory)
    return result


def save_source_snapshot(output, hashes):
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, expected in hashes.items():
            if Path(name).parts[0] not in ('src', 'scripts', 'config'):
                continue
            payload = (ROOT / name).read_bytes()
            if hashlib.sha256(payload).hexdigest() != expected:
                raise RuntimeError(f'source changed while freezing qualification: {name}')
            archive.writestr(name, payload)
    return hashlib.sha256(output.read_bytes()).hexdigest()


def interrupt_run(signum, _frame):
    # SIGTERM must enter finally too: launchers own separate process groups.
    # A repeated termination signal must not interrupt their bounded cleanup.
    signal.signal(signum, signal.SIG_IGN)
    raise InterruptedError(f"received {signal.Signals(signum).name}")


def configure_transport(env):
    """Use exactly the launcher transport selection before freezing metadata."""
    process=subprocess.run(['bash','-ec',
        'source scripts/lib/runtime.sh\nhunter_configure_sim_transport\n'
        'python3 -c \'import json,os; print(json.dumps(dict(os.environ)))\''],
        cwd=ROOT,env=env,text=True,capture_output=True,check=True)
    return json.loads(process.stdout)


def goal_probe_command(args, log_dir, *, target=None):
    returning=target is not None
    command=["python3", "scripts/tools/hunter_goal_probe.py",
        "--distance",str(args.trial_distance if args.trial_distance is not None else 2.0),
        "--reach-distance","0.30","--timeout",str(args.trial_timeout),
        "--check-visualization","--output",str(log_dir/("return_goal_trace.json" if returning else "goal_trace.json"))]
    if returning:command.extend(["--target",*[str(v) for v in target],"--behind"])
    else:
        command.extend(["--lateral",str(args.goal_lateral)])
        if args.reverse_goal:command.append("--behind")
    if args.gui and getattr(args,'rviz_capture','required')=='required':
        command.extend(["--rviz-snapshot",str(log_dir/("rviz_return.png" if returning else "rviz_navigation.png"))])
    return command


def simulation_start_command(args):
    command = ["scripts/sim/start_simulation.sh", f"headless:={str(not args.gui).lower()}",
               "--duration", "300", "--runtime-profile-period", "5",
               "--physics-transform-output", args.physics_transform_output,
               "--joint-action-interface", args.joint_action_interface]
    if getattr(args,'independent_model_floor',False):command.insert(1,'navigation_profile:=neos_joint_sim')
    if args.lidar_mount_pitch is not None:
        command.extend(['--lidar-mount-pitch', str(args.lidar_mount_pitch)])
    if args.rotary_lidar_config is not None:
        command.extend(['--lidar-config', args.rotary_lidar_config])
    if args.operating_speed_limit is not None:
        command.extend(['--cmd-vel-max-speed',str(args.operating_speed_limit)])
    if args.wheel_drive_mode is not None:
        command.extend(['--wheel-drive-mode', args.wheel_drive_mode])
    for name in ('wheel_velocity_kp', 'wheel_velocity_ki'):
        value = getattr(args, name)
        if value is not None:
            command.extend(['--'+name.replace('_', '-'), str(value)])
    if args.trial_scenario != 'none':
        command.extend(['--scene-profile', 'light', '--map-asset', 'none',
                        '--nav-test-scenario', args.trial_scenario,
                        '--nav-test-trigger-distance', str(args.obstacle_trigger_distance)])
    if args.trial_scenario in RAMP_SCENARIOS:
        command.extend(['--nav-test-ramp-rise', str(args.ramp_rise)])
        if args.ramp_direction == 'downhill':
            command.extend(['--spawn-x', '7', '--spawn-y', '0', '--spawn-yaw-deg', '180'])
    if args.trial_scenario=='boundary':command.extend(['--boundary-spec',args.boundary_spec])
    return command


def navigation_trial_command(args, log_dir):
    if args.trial_scenario=='boundary':
        command=['python3','scripts/sim/checks/boundary_trial.py','--spec',args.boundary_spec,
                '--output',str(log_dir/'navigation_data.json'),'--timeout',str(args.trial_timeout)]
        if args.capture_motion_grid:command.extend(['--motion-grid-output',str(log_dir/'motion_grids.json')])
        return command
    command = ['python3', 'scripts/sim/checks/navigation_trial.py',
               '--scenario', args.trial_scenario,
               '--distance', str(args.trial_distance if args.trial_distance is not None else
                   9.0 if args.trial_scenario in ('emerge', 'crossing', 'clearance') else course_distance(args.trial_scenario)),
               '--lateral', str(args.goal_lateral), '--reach-distance', '0.30',
               '--output', str(log_dir / 'navigation_data.json'), '--timeout', str(args.trial_timeout),
               '--evaluate-front-sign']
    if args.reverse_goal:
        command.extend(['--goal-mode', 'behind'])
    if args.require_dynamic_encounter:
        command.append('--require-dynamic-encounter')
    if args.require_clearance_recovery:
        command.append('--require-clearance-recovery')
    if args.clearance_observation_window is not None:
        command.extend(['--clearance-observation-window',str(args.clearance_observation_window)])
    if args.capture_motion_grid:
        command.extend(['--motion-grid-output', str(log_dir / 'motion_grids.json')])
    if args.trial_scenario in RAMP_SCENARIOS:
        command.extend(['--ramp-direction', args.ramp_direction, '--ramp-rise', str(args.ramp_rise)])
    return command


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-clearance-recovery',action='store_true')
    from clearance_metrics import EARLY_RECOVERY_WINDOW_S
    parser.add_argument('--clearance-observation-window',type=float,choices=(EARLY_RECOVERY_WINDOW_S,))
    parser.add_argument("--domain", type=int, default=87)
    parser.add_argument("--lidar-mount-pitch", type=mount_pitch_degrees,
                        help="Compare an alternate MID360 mount with matching FAST-LIO extrinsics")
    parser.add_argument("--rotary-lidar-config", choices=("OS0_REV6_128ch10hz512res",),
                        help="Use the installed Isaac Ouster wide-view profile for a diagnostic comparison")
    parser.add_argument('--dds-publication-mode',choices=('default','asynchronous','synchronous','auto'),default='default',
        help='Explicit transport qualification; does not change navigation or risk budgets')
    parser.add_argument("--gui", action="store_true")
    parser.add_argument('--rviz-capture',choices=('required','advisory'),default='required',
        help='Advisory retains screenshot failures separately; all ROS visualization, navigation and timing checks remain required')
    parser.add_argument("--physics-transform-output",choices=("usd","fabric"),default="fabric")
    parser.add_argument('--joint-action-interface',choices=('targets','controller'),default='targets')
    parser.add_argument('--wheel-drive-mode', choices=('velocity', 'velocity_pi', 'effort'), default=None,
                        help='Explicit motor-servo qualification; keeps scene default otherwise')
    parser.add_argument('--wheel-velocity-kp', type=float, default=None)
    parser.add_argument('--wheel-velocity-ki', type=float, default=None)
    parser.add_argument("--min-realtime-factor",type=float,default=.95,
        help="Minimum sensor wall-clock throughput relative to nominal; not a hard-deadline proof")
    parser.add_argument("--rviz-view",choices=("risk","height","cloud","source"),default="risk")
    parser.add_argument("--terrain-backend", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument('--gp-fit-device',choices=('cpu','cuda'),default=None,
        help='Explicit GP fit device; geometry retains the terrain backend. Default follows terrain backend')
    parser.add_argument('--terrain-executor',choices=('thread_pool','single_ingress'),default='thread_pool')
    parser.add_argument('--terrain-update-hz',type=float,default=8.,
        help='Terrain maximum service cadence, 1..10Hz for the 10Hz LiDAR; source-age gates remain unchanged')
    parser.add_argument('--motion-prediction-model',choices=('reachable','gaussian_joint_cuda','gaussian_joint_cpu'),default='reachable')
    parser.add_argument('--motion-gaussian-dynamics',choices=('cv','ca'),default='cv')
    parser.add_argument('--motion-raw-surface-fit',action='store_true')
    parser.add_argument('--motion-component-link-cells',type=int,choices=(1,2,3),
                        help='Connect sparse measured returns for tracking only, at native grid resolution')
    parser.add_argument('--motion-gaussian-jerk-density',type=float,default=.04)
    parser.add_argument('--sim-motion-observation-calibration',action='store_true')
    parser.add_argument('--motion-gaussian-acceleration-envelope',action='store_true',
        help='Explicit bounded longitudinal mean-acceleration stress model; original probability limits retained')
    parser.add_argument('--dynamic-yield-delay',type=float,default=None)
    parser.add_argument('--joint-dynamic-risk-limit',type=float,default=None)
    parser.add_argument('--compute-mps-pipe-directory',default=None,
                        help='Explicit caller-owned MPS socket directory for terrain/planning only; Isaac/RTX excluded')
    parser.add_argument("--prediction-backend", choices=("none", "feature_sgp"), default="none")
    parser.add_argument("--startup-support",choices=("unknown","stable_stance"),default="unknown")
    parser.add_argument('--directional-startup-support',action='store_true',help='Explicit sim-only fixed stance with one-sided measured plane and attitude checks')
    parser.add_argument('--gp-extrapolation-distance',type=float,default=0.,help='Opt-in shared-GP extrapolation distance, maximum .75m, uncertainty gated')
    parser.add_argument("--gp-refine-observed", action="store_true")
    parser.add_argument("--gp-complete-supported",action="store_true")
    parser.add_argument('--ground-history-s',type=float,default=0.,help='Opt-in raw-ground history qualification, maximum 2 seconds')
    parser.add_argument('--gp-support-distance',type=float,default=.35,help='Raw support distance in metres; extended interpolation is qualification-only')
    parser.add_argument('--probability-model',choices=('marginal','shared_gp'),default='marginal')
    parser.add_argument('--condition-full-residual',action='store_true',help='Qualify the cell-local Gaussian residual kernel conditioned on real observations; legacy model remains default')
    parser.add_argument('--gp-detail-sources',type=int,default=64)
    parser.add_argument('--gp-posterior-radius',type=float,default=3.,
        help='Condition real ground in a 1..5m half-width tile; never changes measured support or uncertainty gates')
    parser.add_argument('--gp-basis-spacing',type=float,default=1.,
        help='Global low-rank kernel basis spacing .5..2m; local residual and detail kernels remain separate')
    parser.add_argument('--gp-refinement-radius',type=float,default=0.,
        help='Optional covariance tightening near fresh odometry; outside keeps its conservative bound')
    parser.add_argument('--gp-residual-sources',type=int,default=0)
    parser.add_argument('--gp-residual-lengthscale',type=float,default=.35)
    parser.add_argument('--bounded-pending',action='store_true',help='Explicit simulation bounded native floor likelihood family; contract 9')
    parser.add_argument('--independent-model-floor',action='store_true',
        help='Explicit simulation Gaussian nugget prior; arbitrary floor correlation is no longer claimed')
    parser.add_argument('--terrain-input-deadline',type=float,default=.16,
                        help='Discard old pending terrain before costly processing; output/safety deadlines remain unchanged')
    parser.add_argument('--reverse-recovery',action='store_true',help='Opt-in certified reverse arc with a jointly certified forward exit')
    parser.add_argument('--capture-motion-grid',action='store_true',help='Archive raw live obstacle masks for deterministic motion-guard replay')
    parser.add_argument('--operating-speed-limit',type=float,default=None,
                        help='Explicit unloaded simulation speed domain; sets BOTH simulator clamp and planner limit, below the NEOS datasheet maximum')
    parser.add_argument('--speed-dead-time',type=float,default=None)
    parser.add_argument('--reverse-speed-limit',type=float,default=None)
    parser.add_argument('--steering-dead-time',type=float,default=None)
    parser.add_argument('--capture-dynamic-refusal',action='store_true',
                        help='Passively retain first source-matched dynamic refusal; serialize only after the trial stops')
    parser.add_argument('--capture-dynamic-refusal-after-action',action='store_true',
                        help='Capture a refusal after an accepted moving avoidance action, rather than startup input holds')
    parser.add_argument('--capture-dynamic-refusal-latest',action='store_true',
                        help='Keep the last complete matched refusal before the assessor ends the trial')
    parser.add_argument('--obstacle-memory-topic', default='none',
        help='Experimental accumulated obstacle input; disabled by default after sloped dynamic-ghost counterexample')
    parser.add_argument('--obstacle-memory-role', choices=('hard', 'advisory'), default='hard',
        help='Advisory history changes route cost only, never collision or probability qualification')
    parser.add_argument('--motion-capture-only',action='store_true',help='Stationary, explicitly triggered sensor test; no navigation goal')
    parser.add_argument("--dynamic-clear-enabled", choices=("true", "false"), default="true")
    parser.add_argument('--dynamic-filter-backend',choices=('none','dufomap','freedom'),default='none')
    parser.add_argument('--dynamic-registration-filter',action='store_true',help='Experimental causal FreeDOM EKF-association exclusion; full scan remains published')
    parser.add_argument('--record-mapping-input',action='store_true',help='Retain actual registered scans, odometry and maps for deterministic dynamic-map replay')
    parser.add_argument("--terrain-dynamic-clear-enabled", choices=("true", "false"), default="true")
    parser.add_argument("--disable-live-obstacles", action="store_true")
    parser.add_argument("--enable-map-expansion", action="store_true",
                        help="Experimental A/B mode; may erase adjacent static obstacles")
    parser.add_argument("--timeout", type=float, default=90.0, help="Per-stage wall-time limit")
    parser.add_argument("--skip-goal", action="store_true", help="Check data flow without moving the vehicle")
    parser.add_argument('--check-mid360-scan',action='store_true',help='Stationary actual-return scan diversity qualification before any goal')
    parser.add_argument("--goal-lateral", type=float, default=0.0, help="Goal offset to the vehicle's left")
    parser.add_argument("--reverse-goal", action="store_true", help="Test a close goal behind the vehicle")
    parser.add_argument("--return-to-start",action="store_true",help="After forward arrival, reverse to the same recorded start")
    parser.add_argument("--trial-scenario", choices=("none", "boundary", "detour", "ramp", "step", "emerge", *CROSSING_SCENARIOS, "clearance", "clearance_guard", *COMPOSITE_SCENARIOS), default="none")
    parser.add_argument('--boundary-spec',default=None)
    parser.add_argument("--ramp-direction", choices=("uphill", "downhill"), default="uphill")
    parser.add_argument('--ramp-rise', type=float, default=.45)
    parser.add_argument("--trial-timeout", type=float, default=45.0)
    parser.add_argument("--trial-distance", type=float, default=None)
    parser.add_argument("--obstacle-trigger-distance",type=float,default=None)
    parser.add_argument("--require-dynamic-encounter",action="store_true",
        help="Crossing must threaten a held-velocity projection, not just move far ahead")
    parser.add_argument("--fault-stale-map", action="store_true", help="Pause terrain publishing during motion and verify planner stop")
    args = parser.parse_args(argv)
    if args.rotary_lidar_config and args.check_mid360_scan:
        parser.error("MID360 scan verification requires the MID360 profile")
    if args.compute_mps_pipe_directory is not None:
        pipe=Path(args.compute_mps_pipe_directory).resolve()
        if not pipe.is_dir() or not (pipe/'control').is_socket():
            parser.error('compute MPS requires an existing caller-owned control socket; no daemon is started implicitly')
        args.compute_mps_pipe_directory=str(pipe)
    if (not math.isfinite(args.motion_gaussian_jerk_density) or not 0<=args.motion_gaussian_jerk_density<=4 or
        args.motion_gaussian_dynamics=='ca' and (args.motion_prediction_model=='reachable' or args.motion_gaussian_acceleration_envelope)):
        parser.error('CA requires Gaussian joint risk, a separate finite white-jerk density [0,4], and no bounded-mean CV envelope')
    if args.motion_gaussian_acceleration_envelope and args.motion_prediction_model=='reachable':
        parser.error('mean acceleration envelope requires a Gaussian motion prediction model')
    if args.dynamic_registration_filter and args.dynamic_filter_backend!='freedom':
        parser.error('dynamic registration exclusion requires the FreeDOM source-label backend')
    if args.operating_speed_limit is not None:
        speed_max=json.loads((ROOT/'config/vehicles/neos.json').read_text())['max_speed_kmh']/3.6
        if not math.isfinite(args.operating_speed_limit) or not 0<args.operating_speed_limit<=speed_max:
            parser.error('operating speed must be positive and within the NEOS datasheet limit')
    if any(v is not None and (not math.isfinite(v) or not 0<=v<=1)
           for v in (args.speed_dead_time,args.steering_dead_time)):
        parser.error('pure actuator dead times must be finite within 0..1 seconds; slew is modeled separately')
    if args.reverse_speed_limit is not None and (not math.isfinite(args.reverse_speed_limit) or not .1<=args.reverse_speed_limit<=.8):
        parser.error('reverse speed must remain within the existing planner range 0.1..0.8 m/s')
    gains = [args.wheel_velocity_kp, args.wheel_velocity_ki]
    if (any(v is not None and (not math.isfinite(v) or v < 0) for v in gains)
            or args.wheel_velocity_kp is not None and args.wheel_drive_mode != 'effort'
            or args.wheel_velocity_ki is not None and args.wheel_drive_mode not in ('effort','velocity_pi')):
        parser.error('finite nonnegative PI gains require effort; integral gain also supports velocity_pi')
    if (args.trial_scenario=='boundary') != (args.boundary_spec is not None):parser.error('boundary scenario requires --boundary-spec')
    if args.dynamic_yield_delay is not None and (not math.isfinite(args.dynamic_yield_delay) or not 0<=args.dynamic_yield_delay<=1):
        parser.error('dynamic response model delay must be finite within 0..1 seconds')
    if args.joint_dynamic_risk_limit is not None and (not math.isfinite(args.joint_dynamic_risk_limit) or not 0<args.joint_dynamic_risk_limit<=.05):
        parser.error('dynamic risk sub-budget cannot exceed the configured 5% combined state limit')
    if args.boundary_spec:
        try:args.boundary_case=load_spec(args.boundary_spec)
        except (ValueError,OSError) as exc:parser.error(str(exc))
    if args.obstacle_trigger_distance is None:
        args.obstacle_trigger_distance = default_obstacle_trigger_distance(args.trial_scenario)
    if args.trial_scenario in (*COMPOSITE_SCENARIOS, 'ramp_crossing') and (
            args.reverse_goal or args.goal_lateral != 0. or
            args.trial_distance is not None and args.trial_distance < minimum_course_distance(args.trial_scenario)):
        parser.error('composite course qualification requires a forward axial goal beyond the complete course')
    if args.probability_model=='shared_gp' and (not args.gp_complete_supported or args.prediction_backend!='feature_sgp'):
        parser.error('shared_gp requires feature_sgp and explicit GP completion')
    if args.condition_full_residual and args.probability_model!='shared_gp':
        parser.error('--condition-full-residual requires the explicit shared_gp model')
    if args.bounded_pending and (not args.independent_model_floor or not args.condition_full_residual):
        parser.error('bounded pending requires independent model floor and full residual')
    if args.independent_model_floor and args.probability_model!='shared_gp':
        parser.error('independent model floor requires explicit simulation shared_gp')
    if not math.isfinite(args.terrain_update_hz) or not 1.<=args.terrain_update_hz<=10.:
        parser.error('terrain update cadence must be 1..10Hz')
    if not math.isfinite(args.gp_posterior_radius) or not 1.<=args.gp_posterior_radius<=5.:
        parser.error('shared posterior radius must be 1..5m')
    if not math.isfinite(args.gp_basis_spacing) or not .5<=args.gp_basis_spacing<=2.:
        parser.error('shared global basis spacing must be .5..2m')
    if (not math.isfinite(args.gp_refinement_radius) or
            args.gp_refinement_radius!=0 and not 2.5<=args.gp_refinement_radius<=5. or
            args.gp_refinement_radius>0 and args.probability_model!='shared_gp'):
        parser.error('optional shared-GP refinement radius must be 0 or 2.5..5m')
    if (not 0<=args.gp_residual_sources<=256 or not math.isfinite(args.gp_residual_lengthscale) or
        not .1<=args.gp_residual_lengthscale<=.8 or args.gp_residual_sources>0 and not args.condition_full_residual):
        parser.error('correlated residual needs the full conditional model, 0..256 inducing sites and 0.1..0.8m lengthscale')
    if not math.isfinite(args.terrain_input_deadline) or not 0<args.terrain_input_deadline<=.16:
        parser.error('terrain input qualification deadline must be in (0,0.16] seconds')
    if args.directional_startup_support and args.startup_support!='stable_stance':
        parser.error('directional startup requires --startup-support stable_stance')
    if (not math.isfinite(args.gp_extrapolation_distance) or not 0<=args.gp_extrapolation_distance<=min(.75,args.gp_support_distance)
            or args.gp_extrapolation_distance>0 and args.probability_model!='shared_gp'):
        parser.error('bounded extrapolation requires shared_gp and a distance in [0,min(.75,support distance)]')
    if args.motion_capture_only and (not args.skip_goal or args.trial_scenario!='crossing'):
        parser.error('--motion-capture-only requires --skip-goal --trial-scenario crossing')
    if (not 1 <= args.domain <= 100 or not math.isfinite(args.timeout) or args.timeout <= 0
            or not math.isfinite(args.trial_timeout) or args.trial_timeout <= 2
            or not math.isfinite(args.min_realtime_factor) or not 0<args.min_realtime_factor<=1
            or not math.isfinite(args.goal_lateral)
            or args.trial_distance is not None and
            (not math.isfinite(args.trial_distance) or args.trial_distance <= 0.65)):
        parser.error("use a free domain between 1 and 100, positive timeout, and finite goal-lateral")
    if args.fault_stale_map and (args.trial_scenario != "none" or args.skip_goal):
        parser.error("--fault-stale-map requires a goal and no trial scenario")
    if args.return_to_start and (args.reverse_goal or args.skip_goal or args.fault_stale_map or args.trial_scenario!="none"):
        parser.error("--return-to-start requires a forward goal and no fault/scenario")
    if args.gp_complete_supported and args.prediction_backend!="feature_sgp":
        parser.error("--gp-complete-supported requires --prediction-backend feature_sgp")
    if not 0<=args.gp_detail_sources<=128:parser.error('--gp-detail-sources must be within [0,128]')
    if (not math.isfinite(args.ground_history_s) or not 0<=args.ground_history_s<=2 or
            args.ground_history_s>0 and not args.gp_complete_supported):
        parser.error('--ground-history-s requires current GP completion and a finite age within [0,2]')
    if (not math.isfinite(args.gp_support_distance) or not 0<args.gp_support_distance<=1.25 or
            args.gp_support_distance!=.35 and not args.gp_complete_supported):
        parser.error('--gp-support-distance requires current GP completion and a finite distance in (0,1.25]')
    if args.gp_support_distance>.75 and args.probability_model!='shared_gp':
        parser.error('support beyond .75m requires --probability-model shared_gp')
    if not math.isfinite(args.obstacle_trigger_distance) or not .5<=args.obstacle_trigger_distance<=3.5:
        parser.error("obstacle trigger must be within [0.5,3.5] metres")
    if args.trial_scenario == 'ramp_crossing' and args.obstacle_trigger_distance <= 1.5:
        parser.error('ramp_crossing trigger must be on the slope, beyond 1.5 metres from spawn')
    if args.require_clearance_recovery and args.trial_scenario not in CROSSING_SCENARIOS:
        parser.error('observed-ground recovery requires a crossing fixture')
    if args.clearance_observation_window is not None and not args.require_clearance_recovery:
        parser.error('a finite near-field window requires the observed-ground recovery protocol')
    if args.require_dynamic_encounter and args.trial_scenario not in CROSSING_SCENARIOS:
        parser.error('dynamic encounter gate requires --trial-scenario crossing or ramp_crossing')
    if args.ramp_direction == 'downhill' and args.trial_scenario not in RAMP_SCENARIOS:
        parser.error('--ramp-direction downhill requires a ramp scenario')
    if not math.isfinite(args.ramp_rise) or not .1 <= args.ramp_rise <= 1.2:
        parser.error('--ramp-rise must be finite and within [0.1, 1.2] metres')
    return args


def main():
    args = parse_args()

    env = os.environ.copy()
    env.update(ROS_DOMAIN_ID=str(args.domain), RMW_IMPLEMENTATION="rmw_fastrtps_cpp",
               MIN_REALTIME_FACTOR=str(args.min_realtime_factor))
    if args.dds_publication_mode != 'default':
        env['RMW_FASTRTPS_PUBLICATION_MODE'] = args.dds_publication_mode.upper()
        env['HUNTER_ISAAC_DDS_PUBLICATION_MODE'] = args.dds_publication_mode.upper()
    else:
        mode=env.get('HUNTER_ISAAC_DDS_PUBLICATION_MODE') or env.get('RMW_FASTRTPS_PUBLICATION_MODE') or 'ASYNCHRONOUS'
        env['RMW_FASTRTPS_PUBLICATION_MODE']=mode.upper()
    env=configure_transport(env)
    log_dir = ROOT / "log" / ("modular_smoke_" + time.strftime("%Y%m%d_%H%M%S"))
    log_dir.mkdir(parents=True)
    sys.path.insert(0, str(ROOT / "scripts/sim/experiments"))
    from run_matrix import method_hashes
    if args.boundary_spec:
        frozen=log_dir/'boundary_spec.json';frozen.write_text(json.dumps(args.boundary_case,indent=2)+'\n')
        args.boundary_spec=str(frozen.resolve())
    (log_dir / "run_metadata.json").write_text(json.dumps({
        "arguments": vars(args), "method_hashes": method_hashes(),
        "ros_domain_id": args.domain,
        'compute_mps':{'pipe_directory':args.compute_mps_pipe_directory,
                       'stages':['terrain','planning'] if args.compute_mps_pipe_directory else [],
                       'simulation_excluded':bool(args.compute_mps_pipe_directory)},
        'transport_environment':{name:env.get(name) for name in
            ('HUNTER_ISAAC_DDS_PUBLICATION_MODE','HUNTER_ISAAC_DDS_PROFILE','RMW_FASTRTPS_PUBLICATION_MODE','RMW_FASTRTPS_USE_QOS_FROM_XML','FASTRTPS_DEFAULT_PROFILES_FILE')},
    }, indent=2) + "\n")
    processes = []
    stages = {}
    logs = []
    checks = []
    goal_stage_started = False

    def capture_rviz(name):
        process=start(name,["python3","scripts/sim/checks/rviz_snapshot.py",
            "--output",str(log_dir/(name+".png"))],ros=True)
        try:code=process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.send_signal(signal.SIGINT)
            code=124
        required=args.rviz_capture=='required'
        checks.append(dict(check=name,exit_code=code,required=required))
        if code:
            print(f'[smoke] {name} failed: exit={code}, required={required}; capture log retained',flush=True)
            if required:raise RuntimeError(f'{name} failed with exit code {code}')

    def start(name, command, ros=False):
        log_file = (log_dir / f"{name}.log").open("w")
        logs.append(log_file)
        if ros:
            command = ["bash", "-c", 'unset PYTHONHOME PYTHONPATH; source /opt/ros/humble/setup.bash && source install/setup.bash && exec "$@"', "probe", *command]
        stage_env=compute_stage_environment(env,name,args.compute_mps_pipe_directory)
        process = subprocess.Popen(command, cwd=ROOT, env=stage_env, stdout=log_file,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        processes.append(process)
        return process

    def check(name, command, timeout):
        nonlocal goal_stage_started
        print(f"[smoke] {name}", flush=True)
        monitor_name="map_rates_during_return" if name=="return_goal" else "map_rates_during_goal"
        streams_file="return_motion_streams.json" if name=="return_goal" else "motion_streams.json"
        ready_file=log_dir/(monitor_name+".ready.json")
        # Import/setup of the independent observer can briefly contend with
        # FAST-LIO. Load it with publishing gated BEFORE timing the mission.
        # The same monitor thresholds/arming policy apply after preparation;
        # the unmodified startup logs and explicit preparation marker remain.
        prepared_boundary=name=="navigation_trial" and args.trial_scenario=="boundary"
        process=None
        refusal_capture=None
        if name=='navigation_trial' and (args.capture_dynamic_refusal or args.capture_dynamic_refusal_after_action or args.capture_dynamic_refusal_latest):
            refusal_capture=start('dynamic_refusal_capture',[
                'python3','scripts/sim/checks/capture_dynamic_rejection.py',str(log_dir),
                '--output',str(log_dir/'dynamic_refusal'),'--status-only',
                '--timeout',str(min(600.,args.trial_timeout+100.)),
                *(['--after-executed-action'] if args.capture_dynamic_refusal_after_action else []),
                *(['--latest'] if args.capture_dynamic_refusal_latest else [])],ros=True)
        if prepared_boundary:
            prepared_file=log_dir/'observer_prepared.json';start_file=log_dir/'mission_monitor_armed'
            command=[*command,'--prepared-file',str(prepared_file),'--start-file',str(start_file)]
            process=start(name,command,ros=True)
            preparation_deadline=time.monotonic()+35
            while not prepared_file.is_file() and process.poll() is None and time.monotonic()<preparation_deadline:
                time.sleep(.05)
            if not prepared_file.is_file():raise RuntimeError('boundary observer did not initialize; goal not sent')
        minimum_rate,maximum_source_age=motion_rate_limits(args.terrain_backend)
        monitor=(start(monitor_name,["python3","scripts/sim/checks/map_rates.py",
            "--output",str(log_dir/streams_file),
            "--ready-file",str(ready_file),
            "--duration",str(args.trial_timeout+120), "--sensors", "--upstream", "--raw-headers",
            "--sensor-min-realtime-factor",str(args.min_realtime_factor),
            "--max-source-age",str(maximum_source_age),
            "--min-rate",str(minimum_rate)],ros=True)
            if name in ("goal","return_goal","navigation_trial") else None)
        if monitor is not None:
            ready_deadline=time.monotonic()+20
            while not ready_file.is_file() and monitor.poll() is None and time.monotonic()<ready_deadline:
                time.sleep(.05)
            if not ready_file.is_file():
                raise RuntimeError(f"{monitor_name} did not arm before motion; goal not sent")
        if name in ('goal','return_goal','navigation_trial'):
            goal_stage_started = True
        if prepared_boundary:
            start_file.write_text('stream monitor armed; release single mission goal\n')
        else:process = start(name, command, ros=True)
        deadline=time.monotonic()+timeout
        while process.poll() is None:
            exited=[(stage,p.poll()) for stage,p in stages.items() if p.poll() is not None]
            if exited:raise RuntimeError(f'owned pipeline process exited during {name}: {exited}')
            if time.monotonic()>=deadline:raise subprocess.TimeoutExpired(command,timeout)
            time.sleep(.1)
        code=process.returncode
        checks.append({"check": name, "exit_code": code})
        print((log_dir / f"{name}.log").read_text()[-5000:], end="", flush=True)
        if monitor is not None:
            if monitor.poll() is None:
                monitor.send_signal(signal.SIGINT)
            monitor_code=monitor.wait(timeout=15)
            checks.append({"check":monitor_name,"exit_code":monitor_code})
            print((log_dir/(monitor_name+".log")).read_text()[-5000:],end="",flush=True)
        if name in ("goal","return_goal","navigation_trial"):
            terminal_name=("probability_after_goal" if name=="goal" else
                           "probability_after_return" if name=="return_goal" else "probability_after_trial")
            terminal=start(terminal_name,["python3","scripts/sim/checks/probability_contract_probe.py",
                "--output",str(log_dir/terminal_name),"--timeout","10"],ros=True)
            terminal_code=terminal.wait(timeout=15)
            checks.append({"check":terminal_name,"exit_code":terminal_code})
            print((log_dir/(terminal_name+".log")).read_text()[-5000:],end="",flush=True)
            if args.gui and name=="navigation_trial":
                capture_rviz("rviz_after_trial")
            if not code and (monitor_code or terminal_code):
                raise RuntimeError("goal arrived but qualification evidence checks failed")
        if code:
            if refusal_capture is not None:
                capture_code=refusal_capture.wait(timeout=15)
                checks.append({'check':'dynamic_refusal_capture','exit_code':capture_code})
            raise RuntimeError(f"{name} failed with exit code {code}")
        if refusal_capture is not None:
            capture_code=refusal_capture.wait(timeout=15)
            checks.append({'check':'dynamic_refusal_capture','exit_code':capture_code})

    def wait_topic(topic,data_type=None):
        check("wait_" + topic.strip("/").replace("/", "_"),
              ["python3", "scripts/tools/hunter_stack_wait.py", topic, "--timeout", str(args.timeout),
               *(["--type",data_type] if data_type else [])],
              args.timeout + 10)
        for name, process in stages.items():
            if process.poll() is not None:
                raise RuntimeError(f"{name} exited early with code {process.returncode}")

    result = 1
    failure_reason = None
    previous_sigterm = signal.signal(signal.SIGTERM, interrupt_run)
    try:
        print(f"[smoke] Domain={args.domain}; logs={log_dir}", flush=True)
        metadata_path = log_dir / 'run_metadata.json'
        metadata = json.loads(metadata_path.read_text())
        metadata['source_snapshot_sha256'] = save_source_snapshot(
            log_dir / 'source_snapshot.zip', metadata['method_hashes'])
        mapping_config = write_mount_mapping_config(log_dir, args.lidar_mount_pitch)
        mapping_args = []
        if args.lidar_mount_pitch is not None:
            mapping_args = [f"config_path:={mapping_config.parent}", f"config_file:={mapping_config.name}"]
            metadata["alternate_mapping_config_sha256"] = hashlib.sha256(mapping_config.read_bytes()).hexdigest()
        metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
        check("runtime_installation", ["python3", "scripts/tools/check_runtime_installation.py",
                                       "--output", str(log_dir / "runtime_installation.json")], 15)
        simulation_command = simulation_start_command(args)
        profile_args=['navigation_profile:=neos_joint_sim'] if args.independent_model_floor else []
        stages["simulation"] = start("simulation", simulation_command)
        wait_topic("/livox/lidar")
        wait_topic("/imu/data_raw")
        stages["mapping"] = start("mapping", ["scripts/sim/start_mapping.sh", *profile_args, *mapping_args, "rviz:=false",
                                             f"dynamic_expand_enabled:={str(args.enable_map_expansion).lower()}",
                                             f"dynamic_filter_backend:={args.dynamic_filter_backend}",
                                             f"dynamic_registration_filter:={str(args.dynamic_registration_filter).lower()}",
                                             f"dynamic_clear_enabled:={args.dynamic_clear_enabled}"])
        if args.record_mapping_input:
            stages['mapping_record']=start('mapping_record',['ros2','bag','record',
                '--output',str(log_dir/'mapping_inputs'),'/clock','/Odometry',
                '/cloud_registered','/cloud_registered_visual','/Laser_map',
                '/car_0_terrain_analyzer/fused_map','/car_0_fsgp_bgk/fused_map',
                '/terrain_nav/topology_memory','/terrain_nav/topology_route'],ros=True)
        wait_topic("/Odometry")
        wait_topic("/Laser_map")
        if args.check_mid360_scan:
            check('mid360_scan',['python3','scripts/sim/checks/mid360_scan_probe.py',
                                '--output',str(log_dir/'mid360_scan.json')],20)
        parameter_command = ["python3", "scripts/sim/checks/mapping_parameters.py",
                             "--config", str(mapping_config)]
        parameter_command.extend(["--dynamic-clear-enabled", args.dynamic_clear_enabled])
        parameter_command.extend(['--dynamic-filter-backend',args.dynamic_filter_backend])
        if args.dynamic_registration_filter:parameter_command.append('--dynamic-registration-filter')
        if args.enable_map_expansion:
            parameter_command.append("--enable-map-expansion")
        check("mapping_parameters", parameter_command, 25)
        configuration=yaml.safe_load((ROOT/"src/fsgp_bgk_se2/rviz/hunter_terrain_navigation.rviz").read_text())
        for panel in configuration["Panels"]:
            if panel["Name"]=="Navigation Status":panel["ViewMode"]=("risk","height","cloud","source").index(args.rviz_view)
        (log_dir/"hunter_terrain_navigation.rviz").write_text(yaml.safe_dump(configuration,sort_keys=False))
        stages["terrain"] = start("terrain", ["scripts/sim/start_terrain_analysis.sh", *profile_args, f"terrain_backend:={args.terrain_backend}",
                                      f'fsgp_processing_executor:={args.terrain_executor}',
                                      f"prediction_backend:={args.prediction_backend}",
                                      f"startup_support_mode:={args.startup_support}",
                                      f"startup_support_directional:={str(args.directional_startup_support).lower()}",
                                      "gp_device:=" + (args.gp_fit_device or ("cpu" if args.terrain_backend == "cpu" else "cuda")),
                                      f"gp_refine_observed:={str(args.gp_refine_observed).lower()}",
                                      f"gp_complete_supported:={str(args.gp_complete_supported).lower()}",
                                      f"probability_ground_history_s:={args.ground_history_s}",
                                      f'probability_model:={args.probability_model}',
                                      f'shared_gp_detail_sources:={args.gp_detail_sources}',
                                      f'shared_gp_refinement_radius_m:={args.gp_refinement_radius}',
                                      f'shared_gp_posterior_radius_m:={args.gp_posterior_radius}',
                                      f'fsgp_max_update_rate_hz:={args.terrain_update_hz}',
                                      f'shared_gp_basis_spacing_m:={args.gp_basis_spacing}',
                                      f'shared_gp_condition_full_residual:={str(args.condition_full_residual).lower()}',
                                      f'shared_gp_residual_inducing_points:={args.gp_residual_sources}',
                                      f'shared_gp_residual_lengthscale_m:={args.gp_residual_lengthscale}',
                                      f'shared_gp_independent_model_floor:={str(args.independent_model_floor).lower()}',
                                      f'shared_gp_bounded_pending:={str(args.bounded_pending).lower()}',
                                      f'fsgp_max_input_source_age_sec:={args.terrain_input_deadline}',
                                      f"probability_gp_max_distance_m:={args.gp_support_distance}",
                                      f"probability_gp_max_extrapolation_m:={args.gp_extrapolation_distance}",
                                      f"terrain_dynamic_clear_enabled:={args.terrain_dynamic_clear_enabled}",
                                      f'obstacle_memory_topic:={args.obstacle_memory_topic}',
                                      f'obstacle_memory_role:={args.obstacle_memory_role}',
                                      "fsgp_gpu_backend:=" + ("cpu" if args.terrain_backend == "cpu" else "cupy"),
                                      f"rviz:={str(args.gui).lower()}",
                                      f"rviz_cfg:={log_dir/'hunter_terrain_navigation.rviz'}",
                                      "grid_clouds:=true"])
        wait_topic("/car_0_terrain_analyzer/fused_map")
        check("wait_fsgp_collision_layer", ["python3", "scripts/tools/hunter_stack_wait.py",
                                             "/car_0_fsgp_bgk/fused_map", "--type", "grid",
                                             "--layer", "collision_risk", "--timeout", str(args.timeout)],
              args.timeout + 10)
        if args.prediction_backend == "feature_sgp":
            check("wait_gp_posterior", ["python3", "scripts/tools/hunter_stack_wait.py",
                                        "/car_0_fsgp_bgk/fused_map", "--type", "grid", "--layer", "gp_latent_var",
                                        "--timeout", str(args.timeout)], args.timeout + 10)
        check("probability_contract_probe", ["python3", "scripts/sim/checks/probability_contract_probe.py",
                                             "--output", str(log_dir / "probability_inputs")], 25)
        check("visualization_products", ["python3", "scripts/sim/checks/visualization_probe.py",
                                         "--timeout", "20"], 30)
        if args.gui:
            capture_rviz("rviz_snapshot")
        check("map_rates_before_goal", ["python3", "scripts/sim/checks/map_rates.py", "--min-rate",
                                        "0.5" if args.terrain_backend=="cpu" else "2.0", "--raw-headers"], 20)
        planning_map="/car_0_fsgp_bgk/fused_map"
        if args.fault_stale_map:
            stages["map_relay"]=start("map_relay",["python3","scripts/sim/checks/map_fault_relay.py"],ros=True)
            planning_map="/qualification/fused_map"
            wait_topic(planning_map,"grid")
        stages["planning"] = start("planning", ["scripts/sim/start_planning_control.sh",
                                                *profile_args,
                                                *([f'max_speed:={args.operating_speed_limit}'] if args.operating_speed_limit is not None else []),
                                                *([f'reverse_max_speed:={args.reverse_speed_limit}'] if args.reverse_speed_limit is not None else []),
                                                *([f'dynamic_yield_speed_dead_time_sec:={args.speed_dead_time}'] if args.speed_dead_time is not None else []),
                                                *([f'dynamic_yield_steering_dead_time_sec:={args.steering_dead_time}'] if args.steering_dead_time is not None else []),
                                                *([f'dynamic_yield_actuator_delay_sec:={args.dynamic_yield_delay}'] if args.dynamic_yield_delay is not None else []),
                                                *([f'live_motion_joint_risk_limit:={args.joint_dynamic_risk_limit}'] if args.joint_dynamic_risk_limit is not None else []),
                                                f"live_motion_prediction_model:={args.motion_prediction_model}",
                                                f'live_motion_gaussian_constant_acceleration:={str(args.motion_gaussian_dynamics=="ca").lower()}',
                                                *(['live_motion_raw_surface_fit:=true'] if args.motion_raw_surface_fit else []),
                                                *([f'live_motion_component_link_cells:={args.motion_component_link_cells}']
                                                  if args.motion_component_link_cells is not None else []),
                                                f'live_motion_gaussian_jerk_density:={args.motion_gaussian_jerk_density}',
                                                f"live_motion_sim_observation_calibration:={str(args.sim_motion_observation_calibration).lower()}",
                                                *(['live_motion_gaussian_acceleration_envelope:=true'] if args.motion_gaussian_acceleration_envelope else []),
                                                f"fused_map_topic:={planning_map}",
                                                f"chance_allow_startup_prior:={str(args.startup_support=='stable_stance').lower()}",
                                                f"chance_allow_gp_completion:={str(args.gp_complete_supported).lower()}",
                                                f'chance_allow_joint_model:={str(args.probability_model=="shared_gp").lower()}',
                                                f'chance_allow_independent_model_floor:={str(args.independent_model_floor).lower()}',
                                                f'chance_allow_bounded_pending:={str(args.bounded_pending).lower()}',
                                                f"reverse_recovery_enable:={str(args.reverse_recovery).lower()}",
                                                f"live_obstacle_enable:={str(not args.disable_live_obstacles).lower()}"])
        if args.trial_scenario == "clearance_guard":
            check("map_guard_trial", ["python3", "scripts/sim/checks/map_guard_trial.py",
                                       "--timeout", str(args.trial_timeout)], args.trial_timeout + 20)
        elif args.motion_capture_only:
            check('motion_capture',['python3','scripts/sim/checks/motion_grid_capture.py',
                '--output',str(log_dir/'motion_grids.json')],30)
        elif not args.skip_goal:
            if args.fault_stale_map:
                check("stale_map", ["python3", "scripts/sim/checks/stale_map_trial.py",
                                    "--pause-service", "/qualification/pause_map_relay",
                                    "--output", str(log_dir / "stale_map_data.json")], 55)
                planning_log = (log_dir / "planning.log").read_text()
                if "reason=stale odometry or map" in planning_log or "reason=stale live cloud" in planning_log:
                    raise RuntimeError("planner searched repeatedly on stale inputs")
                if "inputs fresh again: requesting a replacement path" not in planning_log:
                    raise RuntimeError("planner did not explicitly replace its stale path")
            elif args.trial_scenario == "none":
                check("goal", goal_probe_command(args,log_dir), args.trial_timeout+25)
                if args.return_to_start:
                    reference=json.loads((log_dir/"goal_trace.json").read_text())["summary"]["start"]
                    check("return_goal",goal_probe_command(args,log_dir,target=reference),args.trial_timeout+25)
            else:
                trial_command = navigation_trial_command(args, log_dir)
                check("navigation_trial", trial_command, args.trial_timeout + 100)
                if args.trial_scenario in ("emerge", *CROSSING_SCENARIOS, "clearance"):
                    planning_log = (log_dir / "planning.log").read_text()
                    simulation_log = (log_dir / "simulation.log").read_text()
                    event = ("navigation test obstacle has left the lane" if args.trial_scenario == "clearance"
                             else "navigation test obstacle has crossed the lane"
                             if args.trial_scenario in CROSSING_SCENARIOS else "navigation test obstacle is now blocking the lane")
                    if event not in simulation_log:
                        raise RuntimeError("moving obstacle did not enter the lane")
                    # Truth clearance, live observations, arrival and settling
                    # are checked in navigation_trial. An already safe path
                    # need not trigger one particular emergency log message.
        check("rates", ["scripts/sim/checks/rates.sh"], 60)
        check("map_rates", ["python3", "scripts/sim/checks/map_rates.py", "--min-rate",
                            "0.5" if args.terrain_backend == "cpu" else "2.0", "--raw-headers"], 20)
        check("drift", ["scripts/sim/checks/slam_drift.sh", "10"], 20)
        for name, process in stages.items():
            if process.poll() is not None:
                raise RuntimeError(f"{name} exited early with code {process.returncode}")
        result = 0
    except (RuntimeError, subprocess.TimeoutExpired, KeyboardInterrupt, InterruptedError) as exc:
        failure_reason = str(exc) or type(exc).__name__
        print(f"[smoke] FAILED: {failure_reason}", file=sys.stderr, flush=True)
    finally:
        # A second interrupt while waiting for owned launchers must not skip
        # process-group cleanup or leave Isaac running without its controller.
        previous_sigint=signal.signal(signal.SIGINT,signal.SIG_IGN)
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        # Let ros2 launch forward one interrupt. Sending it to the whole group
        # as well double-interrupts Python nodes during resource destruction.
        # Kill only owned process groups after the bounded graceful interval.
        for process in reversed(processes):
            try:
                process.send_signal(signal.SIGINT)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 8
        for process in reversed(processes):
            try:
                process.wait(timeout=max(0.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
        for process in reversed(processes):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        for log_file in logs:
            log_file.close()
        scope=("stationary_sensor_motion_qualification_not_navigation" if args.motion_capture_only else
            "stale_map_fault_qualification" if args.fault_stale_map else
            "stationary_data_flow_not_navigation" if args.skip_goal else
            "goal_qualification_attempt" if goal_stage_started else "pre_navigation_validation")
        (log_dir / "result.json").write_text(json.dumps({"passed": result == 0, "scope":scope,
                                                          "visual_capture_required":args.rviz_capture=='required',
                                                          "visual_capture_passed":all(c['exit_code']==0 for c in checks if c['check'] in ('rviz_snapshot','rviz_after_trial')) if args.gui else None,
                                                          "goal_tested":goal_stage_started,"goal_stage_started":goal_stage_started,
                                                          "scenario": args.trial_scenario,
                                                          "failure_reason": failure_reason,
                                                          "checks": checks}, indent=2) + "\n")
        print(f"[smoke] {'PASS' if result == 0 else 'FAIL'}; scope={scope}; logs={log_dir}", flush=True)
        signal.signal(signal.SIGTERM, previous_sigterm)
        signal.signal(signal.SIGINT, previous_sigint)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
