#!/usr/bin/env python3
"""Directed physical capability checks; no random campaign or pooled success rate.

Each fixture receives one final external goal. Existing per-fixture safety,
source-age, physical and arrival/stop checks are retained. A failure never
becomes a pass because the other fixtures pass. Freeze source while running.
"""
import argparse
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from run_matrix import method_hashes

ROOT=Path(__file__).resolve().parents[3]
BOUNDARY='scripts/sim/experiments/boundary_cases/'
CASES={
    'single_crossing':('无遮挡单物体横穿：停车可能占用其路径','arrive_and_interact',['--trial-scenario','boundary','--boundary-spec',BOUNDARY+'single_crossing.json','--obstacle-trigger-distance','3.5','--trial-timeout','100']),
    'single_head_on':('无遮挡单物体迎面接近后停在路中','arrive_and_interact',['--trial-scenario','boundary','--boundary-spec',BOUNDARY+'single_head_on.json','--obstacle-trigger-distance','3.5','--trial-timeout','100']),
    'single_varying_speed':('连续变速横穿；未来速度不输入导航','arrive_and_interact',['--trial-scenario','boundary','--boundary-spec',BOUNDARY+'single_varying_speed.json','--obstacle-trigger-distance','3.5','--trial-timeout','100']),
    'single_stop_go':('横穿中有限制动、停留再起步','arrive_and_interact',['--trial-scenario','boundary','--boundary-spec',BOUNDARY+'single_stop_go.json','--obstacle-trigger-distance','3.5','--trial-timeout','100']),
    'single_turning':('可见物体切向连续九十度转弯','arrive_and_interact',['--trial-scenario','boundary','--boundary-spec',BOUNDARY+'single_turning.json','--obstacle-trigger-distance','3.5','--trial-timeout','100']),
    'deadend':('死胡同单最终目标：停车倒出后重新寻路','arrive',['--trial-scenario','boundary','--boundary-spec',BOUNDARY+'deadend.json','--trial-timeout','120']),
    'deadend_entry_observed_approach':('已有更长进场观测的封闭巷道回退对照','arrive_and_recover',['--trial-scenario','boundary','--boundary-spec',BOUNDARY+'deadend_entry_observed_approach.json','--obstacle-trigger-distance','3.5','--trial-timeout','120']),
    'deadend_entry':('驶入后封闭巷道：沿已走区域倒出并绕行单最终目标','arrive_and_recover',['--trial-scenario','boundary','--boundary-spec',BOUNDARY+'deadend_entry.json','--obstacle-trigger-distance','2.2','--trial-timeout','120']),
    'dense':('密集障碍内单目标导航','arrive',['--trial-scenario','boundary','--boundary-spec',BOUNDARY+'dense.json','--trial-timeout','120']),
    'chicane':('窄门连续曲率变化','arrive',['--trial-scenario','boundary','--boundary-spec',BOUNDARY+'chicane.json','--trial-timeout','120']),
    'rubble3':('3厘米连续全宽台沿与碎块','arrive',['--trial-scenario','boundary','--boundary-spec',BOUNDARY+'rubble_3cm.json','--trial-timeout','120']),
    'rubble6':('6厘米连续全宽台沿与碎块能力检查','arrive',['--trial-scenario','boundary','--boundary-spec',BOUNDARY+'rubble_6cm.json','--trial-timeout','120']),
    'corridor35':('启动即设35米走廊外最终目标','arrive',['--trial-scenario','boundary','--boundary-spec',BOUNDARY+'side_goal_corridor_35m.json','--trial-timeout','180']),
    'grade045up':('最大坡度9.58度的0.45米高差上坡绕柱','arrive',['--trial-scenario','ramp_detour','--ramp-rise','.45','--trial-timeout','120']),
    'grade045down':('最大坡度9.58度的0.45米高差下坡绕柱','arrive',['--trial-scenario','ramp_detour','--ramp-rise','.45','--ramp-direction','downhill','--trial-timeout','120']),
    'grade09up':('0.9米高差上坡绕柱','arrive',['--trial-scenario','ramp_detour','--ramp-rise','.9','--trial-timeout','120']),
    'grade09down':('0.9米高差下坡绕柱','arrive',['--trial-scenario','ramp_detour','--ramp-rise','.9','--ramp-direction','downhill','--trial-timeout','120']),
    'grade12turn':('1.2米高差上坡与转弯组合','arrive',['--trial-scenario','ramp_detour','--ramp-rise','1.2','--trial-timeout','140']),
    'multi_dynamic':('相反方向横穿与纵向移动三障碍','arrive',['--trial-scenario','boundary','--boundary-spec',BOUNDARY+'multi_dynamic.json','--obstacle-trigger-distance','2.5','--trial-timeout','120']),
    'multi_dynamic_varying':('三物体连续变速、反向横穿与纵向移动','arrive',['--trial-scenario','boundary','--boundary-spec',BOUNDARY+'multi_dynamic_varying.json','--obstacle-trigger-distance','2.5','--trial-timeout','120']),
    'multi_dynamic_six':('六物体持续变速与反向运动；仅传感器观测','arrive',['--trial-scenario','boundary','--boundary-spec',BOUNDARY+'multi_dynamic_six.json','--obstacle-trigger-distance','2.5','--trial-timeout','120']),
    'urgent_crossing':('有实际逼近证据的横穿避障','arrive',['--trial-scenario','crossing','--trial-distance','9','--obstacle-trigger-distance','2.5','--require-dynamic-encounter','--trial-timeout','100']),
    'new_obstacle':('新出现未知障碍后重规划','arrive',['--trial-scenario','emerge','--obstacle-trigger-distance','2.5','--trial-timeout','100']),
    'clearance':('障碍离开后点云及概率足迹快速恢复','arrive_and_clear',['--trial-scenario','clearance','--trial-timeout','100']),
    'observed_clearance':('已有实测支撑道路上的实际横穿与四秒近场恢复','arrive_and_clear_measured_nearfield',['--trial-scenario','crossing','--trial-distance','9','--obstacle-trigger-distance','2.5','--require-dynamic-encounter','--require-clearance-recovery','--clearance-observation-window','4','--trial-timeout','100']),
    'blocked_step':('高台阶拒绝通行并保留安全距离','safe_refusal',['--trial-scenario','step','--trial-timeout','35']),
}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',nargs='+',choices=tuple(CASES),required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--domain',type=int,default=88)
    p.add_argument('--gui',action='store_true');p.add_argument('--dry-run',action='store_true')
    p.add_argument('--dynamic-registration-filter',action='store_true',help='Explicit FreeDOM EKF exclusion qualification; no raw scan filtering')
    p.add_argument('--rviz-capture',choices=('required','advisory'),default='required')
    p.add_argument('--gp-residual-sources',type=int,default=0)
    p.add_argument('--gp-residual-lengthscale',type=float,default=.35)
    p.add_argument('--gp-refinement-radius',type=float,default=0.)
    p.add_argument('--gp-posterior-radius',type=float,default=3.)
    p.add_argument('--gp-basis-spacing',type=float,default=1.)
    p.add_argument('--gp-fit-device',choices=('cpu','cuda'),default=None)
    p.add_argument('--terrain-executor',choices=('thread_pool','single_ingress'),default='thread_pool')
    p.add_argument('--terrain-update-hz',type=float,default=8.)
    p.add_argument('--motion-prediction-model',choices=('reachable','gaussian_joint_cuda','gaussian_joint_cpu'),default='reachable')
    p.add_argument('--motion-gaussian-dynamics',choices=('cv','ca'),default='cv')
    p.add_argument('--motion-raw-surface-fit',action='store_true')
    p.add_argument('--motion-component-link-cells',type=int,choices=(1,2,3))
    p.add_argument('--capture-dynamic-refusal',action='store_true')
    p.add_argument('--capture-dynamic-refusal-after-action',action='store_true')
    p.add_argument('--capture-dynamic-refusal-latest',action='store_true')
    p.add_argument('--operating-speed-limit',type=float,default=None)
    p.add_argument('--lidar-mount-pitch',type=float,
                   help='Compare a MID360 mounting angle with the corresponding FAST-LIO extrinsics')
    p.add_argument('--rotary-lidar-config',choices=('OS0_REV6_128ch10hz512res',),
                   help='Installed Isaac wide-view rotary reference for diagnosis')
    p.add_argument('--speed-dead-time',type=float,default=None)
    p.add_argument('--reverse-speed-limit',type=float,default=None)
    p.add_argument('--steering-dead-time',type=float,default=None)
    p.add_argument('--motion-gaussian-jerk-density',type=float,default=.04)
    p.add_argument('--sim-motion-observation-calibration',action='store_true')
    p.add_argument('--motion-gaussian-acceleration-envelope',action='store_true')
    p.add_argument('--dynamic-yield-delay',type=float,default=None)
    p.add_argument('--joint-dynamic-risk-limit',type=float,default=None)
    p.add_argument('--obstacle-memory-topic',choices=('none','/Laser_map'),default='/Laser_map')
    p.add_argument('--terrain-input-deadline',type=float,default=.16)
    p.add_argument('--compute-mps-pipe-directory',default=None,
                   help='Explicit existing private MPS for compute launchers, never Isaac/RTX')
    p.add_argument('--independent-model-floor',action='store_true')
    p.add_argument('--bounded-pending',action='store_true')
    p.add_argument('--dense-entry-quality',action='store_true',help='Also qualify initial path/first-row timing and actual approach distance')
    p.add_argument('--stability-repeats',type=int,choices=(1,2,3),default=1,
        help='Small cold-start repeats of dense or the two simple single-body fixtures; not a randomized campaign')
    a=p.parse_args()
    if a.motion_gaussian_acceleration_envelope and a.motion_prediction_model=='reachable':
        p.error('mean acceleration envelope requires a Gaussian motion prediction model')
    if len(a.cases)!=len(set(a.cases)):p.error('each directed fixture occurs once; randomized/repeat studies require a separate reviewed protocol')
    if a.stability_repeats>1 and a.cases not in (['dense'],['single_crossing'],['single_head_on'],['single_crossing','single_head_on']):
        p.error('small stability repetitions apply only to dense or the same simple single-body fixtures; a broader study needs its own reviewed protocol')
    if a.dense_entry_quality and 'dense' not in a.cases:p.error('dense entry quality requires a dense fixture')
    common=[sys.executable,'scripts/sim/checks/pipeline_smoke.py','--domain',str(a.domain),
        '--prediction-backend','feature_sgp','--gp-complete-supported','--probability-model','shared_gp',
        '--condition-full-residual','--startup-support','stable_stance','--directional-startup-support',
        '--gp-support-distance','1.25','--gp-extrapolation-distance','.75','--reverse-recovery',
        '--obstacle-memory-topic',a.obstacle_memory_topic,'--obstacle-memory-role','advisory','--dynamic-filter-backend','freedom',
        '--wheel-drive-mode','velocity_pi','--wheel-velocity-ki','40','--capture-motion-grid']
    common+=['--gp-residual-sources',str(a.gp_residual_sources),'--gp-residual-lengthscale',str(a.gp_residual_lengthscale)]
    common+=['--gp-refinement-radius',str(a.gp_refinement_radius)]
    common+=['--gp-posterior-radius',str(a.gp_posterior_radius)]
    common+=['--gp-basis-spacing',str(a.gp_basis_spacing)]
    if a.gp_fit_device is not None:common+=['--gp-fit-device',a.gp_fit_device]
    common+=['--terrain-executor',a.terrain_executor]
    common+=['--terrain-update-hz',str(a.terrain_update_hz)]
    common+=['--motion-prediction-model',a.motion_prediction_model]
    common+=['--motion-gaussian-dynamics',a.motion_gaussian_dynamics,
             '--motion-gaussian-jerk-density',str(a.motion_gaussian_jerk_density)]
    if a.dynamic_registration_filter:common+=['--dynamic-registration-filter']
    if a.sim_motion_observation_calibration:common+=['--sim-motion-observation-calibration']
    if a.motion_raw_surface_fit:common+=['--motion-raw-surface-fit']
    if a.motion_component_link_cells is not None:common+=['--motion-component-link-cells',str(a.motion_component_link_cells)]
    if a.capture_dynamic_refusal:common+=['--capture-dynamic-refusal']
    if a.capture_dynamic_refusal_after_action:common+=['--capture-dynamic-refusal-after-action']
    if a.capture_dynamic_refusal_latest:common+=['--capture-dynamic-refusal-latest']
    if a.operating_speed_limit is not None:common+=['--operating-speed-limit',str(a.operating_speed_limit)]
    if a.lidar_mount_pitch is not None:common+=['--lidar-mount-pitch',str(a.lidar_mount_pitch)]
    if a.rotary_lidar_config is not None:common+=['--rotary-lidar-config',a.rotary_lidar_config]
    if a.speed_dead_time is not None:common+=['--speed-dead-time',str(a.speed_dead_time)]
    if a.reverse_speed_limit is not None:common+=['--reverse-speed-limit',str(a.reverse_speed_limit)]
    if a.steering_dead_time is not None:common+=['--steering-dead-time',str(a.steering_dead_time)]
    if a.motion_gaussian_acceleration_envelope:common+=['--motion-gaussian-acceleration-envelope']
    if a.dynamic_yield_delay is not None:common+=['--dynamic-yield-delay',str(a.dynamic_yield_delay)]
    if a.joint_dynamic_risk_limit is not None:common+=['--joint-dynamic-risk-limit',str(a.joint_dynamic_risk_limit)]
    common+=['--terrain-input-deadline',str(a.terrain_input_deadline)]
    if a.compute_mps_pipe_directory is not None:
        common+=['--compute-mps-pipe-directory',a.compute_mps_pipe_directory]
    common+=['--rviz-capture',a.rviz_capture]
    if a.independent_model_floor:common+=['--independent-model-floor']
    if a.bounded_pending:
        if not a.independent_model_floor:p.error('bounded pending requires independent model floor')
        common+=['--bounded-pending']
    if a.gui:common+=['--gui','--rviz-view','risk']
    plan=[dict(case=k,description=CASES[k][0],expectation=CASES[k][1],command=common+CASES[k][2]) for k in a.cases]
    quality=a.dense_entry_quality or (a.stability_repeats>1 and 'dense' in a.cases)
    if quality:
        sys.path.insert(0,str(ROOT/'scripts/sim/checks'))
        from navigation_progress_audit import DEFAULT_LIMITS
        for item in plan:
            if item['case']=='dense':item['entrance_quality_limits']=dict(DEFAULT_LIMITS)
    if a.stability_repeats>1:
        plan=[dict(item,repetition=i,run_label=f"{item['case']}_r{i}")
              for i in range(1,a.stability_repeats+1) for item in plan]
    vehicle=json.loads((ROOT/'config/vehicles/neos.json').read_text())
    for item in plan:
        command=item['command']
        if '--ramp-rise' in command and item['expectation']=='arrive':
            rise=float(command[command.index('--ramp-rise')+1])
            # Navigation course: cubic smoothstep over 4m. Its derivative
            # peaks at 1.5*rise/4, not the misleading average rise/length.
            maximum=math.degrees(math.atan(1.5*rise/4.))
            if maximum>vehicle['max_climb_deg']+1e-8:
                p.error(f"{item['case']}: {maximum:.2f}deg exceeds NEOS {vehicle['max_climb_deg']:.1f}deg; "
                    'legacy course cannot be an arrival qualification; use grade045up/down or a separately reviewed refusal case')
    # Validate the WHOLE protocol before starting any physical process.
    # A malformed later fixture must not consume earlier qualification runs.
    sys.path.insert(0,str(ROOT/'scripts/sim/checks'))
    from pipeline_smoke import parse_args as validate_pipeline_args
    from process_ownership import domain_conflicts,run_cancellable
    for item in plan:validate_pipeline_args(item['command'][2:])
    if a.dry_run:print(json.dumps(plan,ensure_ascii=False,indent=2));return 0
    conflicts=domain_conflicts(a.domain)
    if conflicts:raise RuntimeError(f'ROS domain {a.domain} already has participants; no test launched: {conflicts}')
    # Exercise the ACTUAL observer entry before consuming expensive physics.
    # Argument-schema validation alone cannot catch clobbered import names.
    for item in plan:
        if '--boundary-spec' in item['command']:
            manifest=item['command'][item['command'].index('--boundary-spec')+1]
            subprocess.run(['bash','-ec','source /opt/ros/humble/setup.bash\nsource install/setup.bash\nexec "$@"',
                'preflight',sys.executable,'scripts/sim/checks/boundary_trial.py','--spec',manifest,
                '--output','/dev/null','--validate-only'],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,timeout=15)
    a.output.mkdir(parents=True,exist_ok=False);initial=method_hashes();rows=[]
    def save():
        (a.output/'qualification.json').write_text(json.dumps(dict(scope=__doc__,method_hashes=initial,
            planned_runs=len(plan),attempted_runs=len(rows),complete=len(rows)==len(plan),runs=rows),ensure_ascii=False,indent=2)+'\n')
    for item in plan:
        conflicts=domain_conflicts(a.domain)
        if conflicts:raise RuntimeError(f'previous pipeline left participants; no next test launched: {conflicts}')
        if method_hashes()!=initial:raise RuntimeError('source changed between physical checks; stop to preserve version attribution')
        print('START '+item['case']+' '+item['description'],flush=True)
        log=a.output/(item.get('run_label',item['case'])+'.log')
        with log.open('w') as out:code,cancelled=run_cancellable(item['command'],out,cwd=ROOT)
        row={**item,'pipeline_exit_code':code,'exit_code':130 if cancelled else code,
             'cancelled':cancelled,'source_unchanged':method_hashes()==initial}
        match=re.search(r'\[smoke\] Domain=\d+; logs=(.+)',log.read_text())
        if match:
            run=Path(match[1]);row['run_directory']=str(run)
            if (run/'result.json').exists():row['result']=json.loads((run/'result.json').read_text())
            if (run/'navigation_data.json').exists():row['navigation_summary']=json.loads((run/'navigation_data.json').read_text())['summary']
            if item['expectation']=='arrive_and_interact':
                from single_dynamic_audit import audit as audit_interaction
                try:interaction=audit_interaction(run)
                except (ValueError,KeyError,FileNotFoundError) as exc:
                    interaction=dict(accepted=False,error=str(exc))
                (run/'single_dynamic_interaction.json').write_text(json.dumps(interaction,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
                row['interaction_quality']=interaction
                if not interaction['accepted']:row['exit_code']=code or 1
            if 'entrance_quality_limits' in item:
                from navigation_progress_audit import audit
                try:
                    entry=audit(json.loads((run/'navigation_data.json').read_text()),item['entrance_quality_limits'])
                except (ValueError,KeyError,FileNotFoundError) as exc:
                    entry=dict(entrance_quality_passed=False,error=str(exc))
                (run/'navigation_progress.json').write_text(json.dumps(entry,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
                row['entrance_quality']=entry
                if not entry['entrance_quality_passed']:row['exit_code']=code or 1
        rows.append(row);save();print('FINISHED '+item.get('run_label',item['case'])+' exit='+str(row['exit_code']),flush=True)
        if not row['source_unchanged']:raise RuntimeError('source changed DURING a check; result retained but not qualified')
        if cancelled or row['exit_code']!=0:break # Diagnose first failure; no unattended unsafe repeats.
    return int(any(r['exit_code']!=0 for r in rows))


if __name__=='__main__':raise SystemExit(main())
