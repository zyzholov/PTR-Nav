from pathlib import Path
from types import SimpleNamespace
import importlib.util
import signal
import sys
import pytest
import json
import numpy as np

spec=importlib.util.spec_from_file_location("pipeline_smoke",Path(__file__).with_name("pipeline_smoke.py"))
pipeline=importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)

from navigation_courses import CROSSING_SCENARIOS, RAMP_SCENARIOS
import navigation_trial
from boundary_trial import numeric_json

def test_explicit_operating_speed_reaches_the_simulator_without_changing_risk_or_response():
    default=pipeline.parse_args([])
    assert default.operating_speed_limit is None
    assert '--cmd-vel-max-speed' not in pipeline.simulation_start_command(default)
    args=pipeline.parse_args(['--operating-speed-limit','1.8'])
    command=pipeline.simulation_start_command(args)
    assert command[command.index('--cmd-vel-max-speed')+1]=='1.8'
    assert args.dynamic_yield_delay==default.dynamic_yield_delay
    assert args.joint_dynamic_risk_limit==default.joint_dynamic_risk_limit
    for value in ('0','-1','nan','inf','6'):
        with pytest.raises(SystemExit):pipeline.parse_args(['--operating-speed-limit',value])

def test_separate_input_dead_times_are_not_response_thresholds_or_slew_changes():
    args=pipeline.parse_args(['--speed-dead-time','0','--steering-dead-time','.06'])
    assert args.speed_dead_time==0. and args.steering_dead_time==.06
    for flag in ('--speed-dead-time','--steering-dead-time'):
        for value in ('-.1','nan','inf','1.01'):
            with pytest.raises(SystemExit):pipeline.parse_args([flag,value])


def test_explicit_reverse_limit_uses_the_existing_range_without_changing_forward_speed_or_risk():
    default=pipeline.parse_args([])
    assert default.reverse_speed_limit is None
    args=pipeline.parse_args(['--reverse-speed-limit','.8'])
    assert args.reverse_speed_limit==.8
    assert args.operating_speed_limit==default.operating_speed_limit
    assert args.joint_dynamic_risk_limit==default.joint_dynamic_risk_limit
    for value in ('0','-.1','nan','inf','.81'):
        with pytest.raises(SystemExit):pipeline.parse_args(['--reverse-speed-limit',value])


@pytest.mark.parametrize('stage',['simulation','mapping','terrain','planning','navigation_trial'])
def test_private_mps_only_reaches_compute_launchers_and_never_mutates_the_parent(stage):
    original={'ROS_DOMAIN_ID':'88','CUDA_MPS_PIPE_DIRECTORY':'inherited',
              'CUDA_MPS_ACTIVE_THREAD_PERCENTAGE':'25','UNCHANGED':'value'}
    env=pipeline.compute_stage_environment(original,stage,'/tmp/owned-mps')
    assert env['ROS_DOMAIN_ID']=='88' and env['UNCHANGED']=='value'
    assert original['CUDA_MPS_PIPE_DIRECTORY']=='inherited'
    assert 'CUDA_MPS_ACTIVE_THREAD_PERCENTAGE' not in env
    if stage in ('terrain','planning'):assert env['CUDA_MPS_PIPE_DIRECTORY']=='/tmp/owned-mps'
    else:assert 'CUDA_MPS_PIPE_DIRECTORY' not in env
    assert pipeline.compute_stage_environment(original,stage,None)==original


def test_gp_fit_device_is_explicit_and_does_not_replace_the_geometry_backend():
    assert pipeline.parse_args([]).gp_fit_device is None
    args=pipeline.parse_args(['--gp-fit-device','cpu'])
    assert args.gp_fit_device=='cpu' and args.terrain_backend=='gpu'
    with pytest.raises(SystemExit):pipeline.parse_args(['--gp-fit-device','invalid'])


def test_acceleration_posterior_is_an_explicit_separate_model_with_jerk_units():
    assert pipeline.parse_args([]).motion_gaussian_dynamics=='cv'
    for args in (['--motion-gaussian-dynamics','ca'],
                 ['--motion-gaussian-dynamics','ca','--motion-prediction-model','gaussian_joint_cuda','--motion-gaussian-acceleration-envelope'],
                 ['--motion-gaussian-jerk-density','nan'],['--motion-gaussian-jerk-density','-1']):
        with pytest.raises(SystemExit):pipeline.parse_args(args)
    args=pipeline.parse_args(['--motion-gaussian-dynamics','ca','--motion-prediction-model','gaussian_joint_cuda'])
    assert args.motion_gaussian_dynamics=='ca' and args.motion_gaussian_jerk_density==.04


def test_compute_mps_requires_an_explicit_existing_control_socket(tmp_path):
    assert pipeline.parse_args([]).compute_mps_pipe_directory is None
    with pytest.raises(SystemExit):pipeline.parse_args(['--compute-mps-pipe-directory',str(tmp_path)])
    import socket
    with socket.socket(socket.AF_UNIX) as control:
        control.bind(str(tmp_path/'control'))
        assert pipeline.parse_args(['--compute-mps-pipe-directory',str(tmp_path)]).compute_mps_pipe_directory==str(tmp_path)


def test_terrain_numeric_thread_budget_reaches_real_libraries_and_preserves_user_overrides():
    import os,subprocess
    names=['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS']
    clean={k:v for k,v in os.environ.items()if k not in names and k not in ('HUNTER_TERRAIN_CPU_THREADS','HUNTER_FSGP_CPU_THREADS')}
    program='source scripts/lib/runtime.sh\nhunter_configure_terrain_threads || exit $?\nexec python3 -c "$1"'
    code='import json,os; print(json.dumps({k:os.environ[k] for k in '+repr(names)+'}))'
    def run(env):return subprocess.run(['bash','-c',program,'thread-test',code],cwd=pipeline.ROOT,env=env,text=True,capture_output=True)
    result=run(clean);assert result.returncode==0
    assert set(json.loads(result.stdout).values())=={'1'}
    override=run(dict(clean,HUNTER_TERRAIN_CPU_THREADS='2',OPENBLAS_NUM_THREADS='3'))
    assert override.returncode==0
    assert json.loads(override.stdout)=={k:('3'if k=='OPENBLAS_NUM_THREADS'else'2')for k in names}
    legacy=run(dict(clean,HUNTER_TERRAIN_CPU_THREADS='2',HUNTER_FSGP_CPU_THREADS='4',OPENBLAS_NUM_THREADS='3'))
    assert legacy.returncode==0
    assert json.loads(legacy.stdout)=={k:('3'if k=='OPENBLAS_NUM_THREADS'else'4')for k in names}
    for env in (dict(clean,HUNTER_TERRAIN_CPU_THREADS='0'),dict(clean,OMP_NUM_THREADS='invalid')):
        assert run(env).returncode!=0
    pytest.importorskip('threadpoolctl')
    probe='import json,numpy,scipy.linalg,torch; from threadpoolctl import threadpool_info; '+\
        'print(json.dumps(dict(torch_threads=torch.get_num_threads(),pools=[p["num_threads"] for p in threadpool_info()])))'
    actual=subprocess.run(['bash','-c',program,'thread-test',probe],cwd=pipeline.ROOT,env=clean,text=True,capture_output=True,check=True)
    budget=json.loads(actual.stdout);assert budget['torch_threads']==1 and set(budget['pools'])=={1}


def test_registration_exclusion_requires_its_source_backend_and_explicit_enablement():
    assert not pipeline.parse_args([]).dynamic_registration_filter
    with pytest.raises(SystemExit):pipeline.parse_args(['--dynamic-registration-filter'])
    with pytest.raises(SystemExit):pipeline.parse_args(['--dynamic-registration-filter','--dynamic-filter-backend','dufomap'])
    args=pipeline.parse_args(['--dynamic-registration-filter','--dynamic-filter-backend','freedom'])
    assert args.dynamic_registration_filter and args.dynamic_filter_backend=='freedom'


def test_raw_motion_surface_is_an_explicit_qualification_choice():
    assert not pipeline.parse_args([]).motion_raw_surface_fit
    assert pipeline.parse_args(['--motion-raw-surface-fit']).motion_raw_surface_fit


def test_sensor_matched_service_cadence_keeps_the_original_source_deadlines(tmp_path):
    assert pipeline.parse_args([]).terrain_update_hz == 8.
    assert pipeline.parse_args(['--terrain-update-hz','10']).terrain_update_hz == 10.
    profile=json.loads((Path(__file__).resolve().parents[3]/'scripts/sim/config/neos_joint_sim.json').read_text())
    terrain=dict(v.split(':=',1)for v in profile['stages']['terrain'])
    # Explicit NEOS simulation profile follows the 10Hz producer; generic
    # pipeline defaults remain 8Hz. Neither number relaxes source deadlines
    # or asserts measured end-to-end throughput.
    assert float(terrain['fsgp_max_update_rate_hz']) == 10.
    assert float(terrain['fsgp_max_input_source_age_sec']) == .16
    assert float(terrain['fsgp_max_output_source_age_sec']) == .30
    import subprocess
    result=subprocess.run([sys.executable,str(Path(__file__).parents[1]/'experiments/capability_qualification.py'),
        '--cases','single_turning','--terrain-update-hz','10','--dry-run','--output',str(tmp_path/'unused')],
        capture_output=True,text=True,check=True)
    command=json.loads(result.stdout)[0]['command']
    assert command[command.index('--terrain-update-hz')+1] == '10.0'
    assert not (tmp_path/'unused').exists()


@pytest.mark.parametrize('value',['nan','inf','0','.9','10.1'])
def test_invalid_service_cadence_cannot_start_a_trial(value):
    with pytest.raises(SystemExit):pipeline.parse_args(['--terrain-update-hz',value])


def test_gpu_motion_receive_age_matches_the_preset_output_deadline_not_the_old_350ms():
    profile=json.loads((Path(__file__).resolve().parents[3]/'scripts/sim/config/neos_joint_sim.json').read_text())
    terrain=dict(v.split(':=',1)for v in profile['stages']['terrain'])
    rate,age=pipeline.motion_rate_limits('gpu')
    assert rate==5. and age==float(terrain['fsgp_max_output_source_age_sec'])==.30
    assert pipeline.motion_rate_limits('cpu')==(.5,2.)
    with pytest.raises(ValueError):pipeline.motion_rate_limits('unknown')


def test_boundary_observer_serializes_numpy_measurements_without_erasing_invalid_values():
    record=dict(accepted=np.bool_(False),summary=dict(physical_ok=np.bool_(True),distance=np.float64(.13)))
    decoded=json.loads(json.dumps(record,default=numeric_json,allow_nan=False))
    assert decoded==dict(accepted=False,summary=dict(physical_ok=True,distance=.13))
    with pytest.raises(ValueError):json.dumps(dict(distance=np.float64(float('nan'))),default=numeric_json,allow_nan=False)


def test_boundary_motion_archive_reaches_the_observer():
    args=pipeline.parse_args(['--trial-scenario','boundary','--boundary-spec',
        str(Path(__file__).parents[1]/'experiments/boundary_cases/long_corridor.json'),'--capture-motion-grid'])
    command=pipeline.navigation_trial_command(args,Path('run'))
    assert command[command.index('--motion-grid-output')+1]=='run/motion_grids.json'


def test_motor_servo_qualification_preserves_scene_default_and_does_not_increase_power():
    default=pipeline.simulation_start_command(pipeline.parse_args([]))
    assert '--wheel-drive-mode' not in default
    args=pipeline.parse_args(['--wheel-drive-mode','effort','--wheel-velocity-kp','60',
                             '--wheel-velocity-ki','40'])
    command=pipeline.simulation_start_command(args)
    assert command[command.index('--wheel-drive-mode')+1]=='effort'
    assert float(command[command.index('--wheel-velocity-kp')+1])==60.
    assert float(command[command.index('--wheel-velocity-ki')+1])==40.
    assert '--wheel-motor-effort' not in command and '--powered-wheels' not in command
    implicit=pipeline.simulation_start_command(pipeline.parse_args(['--wheel-drive-mode','velocity_pi',
        '--wheel-velocity-ki','40']))
    assert implicit[implicit.index('--wheel-drive-mode')+1]=='velocity_pi'
    for extra in (['--wheel-velocity-kp','60'],['--wheel-drive-mode','effort','--wheel-velocity-ki','nan']):
        with pytest.raises(SystemExit):pipeline.parse_args(extra)


def test_neos_positive_grade_qualification_checks_the_maximum_not_average_grade(tmp_path):
    import subprocess
    tool=Path(__file__).parents[1]/'experiments/capability_qualification.py'
    for case in ['grade09up','grade09down','grade12turn']:
        result=subprocess.run([sys.executable,str(tool),'--cases',case,'--dry-run',
            '--output',str(tmp_path/case)],capture_output=True,text=True)
        assert result.returncode==2 and 'exceeds NEOS' in result.stderr
        assert not (tmp_path/case).exists()
    result=subprocess.run([sys.executable,str(tool),'--cases','grade045up','grade045down','--dry-run',
        '--output',str(tmp_path/'accepted')],capture_output=True,text=True,check=True)
    for item in json.loads(result.stdout):
        cmd=item['command'];assert float(cmd[cmd.index('--ramp-rise')+1])==.45


def test_small_dense_stability_protocol_declares_entry_quality_before_running(tmp_path):
    import subprocess
    tool=Path(__file__).parents[1]/'experiments/capability_qualification.py'
    args=[sys.executable,str(tool),'--cases','dense','--stability-repeats','3','--dry-run',
        '--output',str(tmp_path/'trials')]
    result=subprocess.run(args,capture_output=True,text=True,check=True)
    plan=json.loads(result.stdout)
    assert len(plan)==3 and len({p['run_label']for p in plan})==3
    assert all(p['case']=='dense' and p['entrance_quality_limits']==dict(
        first_path_s=6.,first_row_s=30.,approach_distance_m=8.)for p in plan)
    assert all(p['command']==plan[0]['command']for p in plan)
    result=subprocess.run([sys.executable,str(tool),'--cases','chicane','--stability-repeats','3',
        '--dry-run','--output',str(tmp_path/'bad')],capture_output=True,text=True)
    assert result.returncode==2 and not (tmp_path/'bad').exists()


@pytest.mark.parametrize('flags',[
    ['--directional-startup-support'],
    ['--gp-extrapolation-distance','.1'],
    ['--gp-extrapolation-distance','nan'],
    ['--gp-extrapolation-distance','-.1'],
    ['--gp-extrapolation-distance','.76'],
])
def test_tilted_sensor_assumptions_require_explicit_bounded_simulation_model(flags):
    with pytest.raises(SystemExit):pipeline.parse_args(flags)


def test_full_residual_model_is_explicit_and_invalid_for_marginal_model():
    assert not pipeline.parse_args([]).condition_full_residual
    with pytest.raises(SystemExit):pipeline.parse_args(['--condition-full-residual'])
    args=pipeline.parse_args(['--condition-full-residual','--probability-model','shared_gp',
        '--prediction-backend','feature_sgp','--gp-complete-supported'])
    assert args.condition_full_residual


@pytest.mark.parametrize('extra',[
    ['--independent-model-floor'],
    ['--gp-residual-sources','64'],['--gp-residual-sources','257'],
    ['--gp-residual-lengthscale','nan'],['--terrain-input-deadline','nan'],
    ['--terrain-input-deadline','0'],['--terrain-input-deadline','.17']])
def test_new_covariance_and_freshness_choices_cannot_silently_relax_the_contract(extra):
    with pytest.raises(SystemExit):pipeline.parse_args(extra)


@pytest.mark.parametrize('extra',[[],['--skip-goal'],['--trial-scenario','crossing']])
def test_stationary_motion_capture_cannot_masquerade_as_navigation(monkeypatch,extra):
    monkeypatch.setattr(sys,'argv',['pipeline_smoke','--motion-capture-only',*extra])
    with pytest.raises(SystemExit) as exit_info:pipeline.main()
    assert exit_info.value.code==2


def test_termination_enters_exception_cleanup_and_masks_repeated_signals(monkeypatch):
    handlers=[]
    monkeypatch.setattr(pipeline.signal,"signal",lambda number,handler:handlers.append((number,handler)))
    with pytest.raises(InterruptedError,match="received SIGTERM"):
        pipeline.interrupt_run(signal.SIGTERM,None)
    assert handlers==[(signal.SIGTERM,signal.SIG_IGN)]


@pytest.mark.parametrize('extra', [['--trial-distance','5'], ['--reverse-goal'], ['--goal-lateral','3']])
def test_partial_or_bypassed_course_cannot_be_called_composite_navigation(monkeypatch, extra):
    monkeypatch.setattr(sys, 'argv', ['pipeline_smoke', '--trial-scenario', 'staggered', *extra])
    with pytest.raises(SystemExit) as exit_info:
        pipeline.main()
    assert exit_info.value.code == 2


def test_requested_goal_distance_timeout_and_lateral_reach_the_probe():
    args=SimpleNamespace(trial_distance=3.,trial_timeout=47.,goal_lateral=.75,reverse_goal=False,gui=True)
    command=pipeline.goal_probe_command(args,Path("run"))
    assert command[command.index("--distance")+1]=="3.0"
    assert command[command.index("--timeout")+1]=="47.0"
    assert command[command.index("--lateral")+1]=="0.75"
    assert "--check-visualization" in command and "--rviz-snapshot" in command


def test_advisory_capture_keeps_visual_topics_required_but_does_not_abort_goal_on_focus():
    args=pipeline.parse_args(['--gui','--rviz-capture','advisory'])
    command=pipeline.goal_probe_command(args,Path('run'))
    assert '--check-visualization' in command and '--rviz-snapshot' not in command
    assert pipeline.parse_args(['--gui']).rviz_capture=='required'


def test_default_distance_and_exact_return_target_remain_separate():
    args=SimpleNamespace(trial_distance=None,trial_timeout=45.,goal_lateral=.75,reverse_goal=False,gui=False)
    command=pipeline.goal_probe_command(args,Path("run"),target=[-.2,.1])
    assert command[command.index("--distance")+1]=="2.0"
    assert command[command.index("--target")+1:command.index("--target")+3]==["-0.2","0.1"]
    assert "--behind" in command and "--lateral" not in command


def test_source_snapshot_keeps_exact_runtime_sources_and_rejects_a_changed_file(tmp_path, monkeypatch):
    import hashlib
    import zipfile
    root = tmp_path / 'project'
    source = root / 'src' / 'node.py'
    source.parent.mkdir(parents=True)
    source.write_text('version = 1\n')
    monkeypatch.setattr(pipeline, 'ROOT', root)
    hashes = {'src/node.py': hashlib.sha256(source.read_bytes()).hexdigest(), 'install/large_binary': 'unused'}
    target = tmp_path / 'source.zip'
    assert pipeline.save_source_snapshot(target, hashes) == hashlib.sha256(target.read_bytes()).hexdigest()
    with zipfile.ZipFile(target) as archive:
        assert archive.namelist() == ['src/node.py']
        assert archive.read('src/node.py') == source.read_bytes()
    source.write_text('version = 2\n')
    with pytest.raises(RuntimeError, match='source changed'):
        pipeline.save_source_snapshot(tmp_path / 'changed.zip', hashes)


@pytest.mark.parametrize('scenario', RAMP_SCENARIOS)
@pytest.mark.parametrize('direction', ['uphill', 'downhill'])
@pytest.mark.parametrize('rise', [.1, .45, 1.2])
def test_ramp_commands_preserve_direction_rise_and_full_forward_goal(scenario, direction, rise):
    args = pipeline.parse_args(['--trial-scenario', scenario, '--ramp-direction', direction,
                                '--ramp-rise', str(rise)])
    simulation = pipeline.simulation_start_command(args)
    trial = pipeline.navigation_trial_command(args, Path('run'))
    assert simulation[simulation.index('--nav-test-scenario')+1] == scenario
    assert float(simulation[simulation.index('--nav-test-ramp-rise')+1]) == rise
    assert trial[trial.index('--ramp-direction')+1] == direction
    assert float(trial[trial.index('--ramp-rise')+1]) == rise
    assert float(trial[trial.index('--distance')+1]) == 7.
    assert trial[trial.index('--lateral')+1] == '0.0'
    assert '--goal-mode' not in trial
    if direction == 'downhill':
        assert simulation[simulation.index('--spawn-x')+1] == '7'
        assert simulation[simulation.index('--spawn-y')+1] == '0'
        assert simulation[simulation.index('--spawn-yaw-deg')+1] == '180'
    else:
        assert '--spawn-x' not in simulation and '--spawn-yaw-deg' not in simulation
    parsed_trial = navigation_trial.parse_args(trial[2:])
    assert parsed_trial.scenario == scenario and parsed_trial.ramp_direction == direction
    assert parsed_trial.ramp_rise == rise and parsed_trial.goal_mode == 'forward'
    assert args.obstacle_trigger_distance == (2. if scenario == 'ramp_crossing' else 1.10)


@pytest.mark.parametrize('scenario', CROSSING_SCENARIOS)
def test_dynamic_encounter_gate_is_available_for_both_crossings(scenario):
    args = pipeline.parse_args(['--trial-scenario', scenario, '--require-dynamic-encounter'])
    command = pipeline.navigation_trial_command(args, Path('run'))
    assert navigation_trial.parse_args(command[2:]).require_dynamic_encounter
    if scenario == 'crossing':
        assert command[command.index('--distance')+1] == '9.0'
        assert args.obstacle_trigger_distance == 1.10


@pytest.mark.parametrize('scenario', ['none', 'detour', 'crossing', 'corridor'])
def test_downhill_and_encounter_options_reject_unrelated_scenarios(scenario):
    with pytest.raises(SystemExit) as error:
        pipeline.parse_args(['--trial-scenario', scenario, '--ramp-direction', 'downhill'])
    assert error.value.code == 2
    if scenario not in ('none', 'crossing'):
        with pytest.raises(SystemExit) as error:
            navigation_trial.parse_args(['--scenario', scenario, '--ramp-direction', 'downhill'])
        assert error.value.code == 2
    if scenario != 'crossing':
        with pytest.raises(SystemExit) as error:
            pipeline.parse_args(['--trial-scenario', scenario, '--require-dynamic-encounter'])
        assert error.value.code == 2


@pytest.mark.parametrize('scenario', ['ramp_detour', 'ramp_crossing'])
@pytest.mark.parametrize('direction', ['uphill', 'downhill'])
@pytest.mark.parametrize('extra', [['--trial-distance', '6'], ['--reverse-goal'], ['--goal-lateral', '.1']])
def test_ramp_obstacle_courses_reject_partial_or_bypassed_goals(scenario, direction, extra):
    with pytest.raises(SystemExit) as error:
        pipeline.parse_args(['--trial-scenario', scenario, '--ramp-direction', direction, *extra])
    assert error.value.code == 2
    trial_extra = {'--trial-distance': ['--distance', '6'], '--reverse-goal': ['--goal-mode', 'behind'],
                   '--goal-lateral': ['--lateral', '.1']}[extra[0]]
    with pytest.raises(SystemExit) as error:
        navigation_trial.parse_args(['--scenario', scenario, '--ramp-direction', direction, *trial_extra])
    assert error.value.code == 2


@pytest.mark.parametrize('value', ['nan', 'inf', '-inf', '0', '-.1', '.099', '1.201'])
def test_ramp_rise_is_finite_and_bounded_in_smoke_and_trial(value):
    for parse, scenario_option in ((pipeline.parse_args, '--trial-scenario'),
                                   (navigation_trial.parse_args, '--scenario')):
        with pytest.raises(SystemExit) as error:
            parse([scenario_option, 'ramp_crossing', '--ramp-rise='+value])
        assert error.value.code == 2


@pytest.mark.parametrize('value', ['nan', 'inf', '.49', '.5', '1.1', '1.5', '3.51'])
def test_ramp_crossing_numeric_trigger_must_start_on_the_slope(value):
    with pytest.raises(SystemExit) as error:
        pipeline.parse_args(['--trial-scenario', 'ramp_crossing', '--obstacle-trigger-distance', value])
    assert error.value.code == 2


@pytest.mark.parametrize('value', ['1.6', '2.0', '3.5'])
def test_explicit_ramp_crossing_distance_reaches_the_scene(value):
    args = pipeline.parse_args(['--trial-scenario', 'ramp_crossing', '--obstacle-trigger-distance', value])
    command = pipeline.simulation_start_command(args)
    assert float(command[command.index('--nav-test-trigger-distance')+1]) == float(value)


def test_default_ramp_rise_and_invalid_trial_encounter_do_not_launch_ros():
    assert pipeline.parse_args(['--trial-scenario', 'ramp']).ramp_rise == .45
    assert navigation_trial.parse_args(['--scenario', 'ramp']).ramp_rise == .45
    with pytest.raises(SystemExit) as error:
        navigation_trial.parse_args(['--scenario', 'ramp_detour', '--require-dynamic-encounter'])
    assert error.value.code == 2


def test_deadend_recovery_qualification_requires_truth_entry_closure_and_reverse():
    from boundary_trial import deadend_recovery_evidence
    # columns: wall,x,y,z,pitch,roll,stamp,yaw; the gate is stamp,x,y.
    rows=[[0,0,0,0,0,0,0,0],[1,2.2,0,0,0,0,1,0],[2,1.2,0,0,0,0,2,0],[3,0,0,0,0,0,3,0]]
    gate=[[0,4.85,3.],[1,4.85,2.9],[2,4.85,.1]]
    assert deadend_recovery_evidence(rows,gate)['exercised']
    forward=[r[:] for r in rows];forward[2][1]=3.;forward[3][1]=3.5
    assert not deadend_recovery_evidence(forward,gate)['exercised']
    assert not deadend_recovery_evidence(rows,gate[:1])['exercised']


def test_bounded_pending_protocol_requires_both_model_permissions_and_is_default_off():
    assert not pipeline.parse_args([]).bounded_pending
    for flags in [ ['--bounded-pending'],['--bounded-pending','--independent-model-floor'] ]:
        with pytest.raises(SystemExit):pipeline.parse_args(flags)
    args=pipeline.parse_args(['--bounded-pending','--independent-model-floor','--condition-full-residual',
        '--probability-model','shared_gp','--gp-complete-supported','--prediction-backend','feature_sgp'])
    assert args.bounded_pending and args.independent_model_floor and args.condition_full_residual
