from copy import deepcopy
from types import SimpleNamespace

import pytest
from sensor_msgs_py.point_cloud2 import create_cloud_xyz32
from std_msgs.msg import Header

from clearance_metrics import clearance_acceptance, sustained_clearance
from navigation_trial import Trial


def stationary_fixture(stamp=1.10):
    from nav_msgs.msg import Odometry
    odom=Odometry();odom.header.frame_id='map'
    odom.header.stamp.sec=int(stamp);odom.header.stamp.nanosec=round((stamp-int(stamp))*1e9)
    odom.pose.pose.orientation.w=1.;odom.pose.pose.position.x=13.;odom.pose.pose.position.y=-1.
    rows=[(0.,3.,2.,0.,0.,0.,1.+i*.005,0.) for i in range(51)]
    return odom,rows


def test_stationary_alignment_uses_acquisition_history_with_delayed_odometry():
    from navigation_trial import stationary_observer_alignment
    odom,rows=stationary_fixture()
    # Latest truth is 150ms newer and has already moved; no receipt-time pairing.
    rows[-1]=(0.,9.,4.,0.,0.,0.,1.25,.8)
    assert stationary_observer_alignment(odom,rows)==pytest.approx((10.,-3.,0.))


@pytest.mark.parametrize('bad',['future','past','gap','truth_motion','odom_motion','angular_motion','nan_twist','nonmonotonic'])
def test_stationary_alignment_rejects_missing_or_unstable_source_pair(bad):
    from navigation_trial import stationary_observer_alignment
    odom,rows=stationary_fixture()
    if bad=='future':rows=rows[:10]
    elif bad=='past':rows=rows[30:]
    elif bad=='gap':rows=[rows[0],rows[-1]]
    elif bad=='truth_motion':rows=[(r[0],r[6],*r[2:]) for r in rows]
    elif bad=='odom_motion':odom.twist.twist.linear.x=.04
    elif bad=='angular_motion':odom.twist.twist.angular.z=.04
    elif bad=='nan_twist':odom.twist.twist.linear.x=float('nan')
    elif bad=='nonmonotonic':rows[4]=rows[3]
    assert stationary_observer_alignment(odom,rows) is None


def test_stationary_alignment_interpolates_and_never_refits_drift():
    from navigation_trial import stationary_observer_alignment
    odom,rows=stationary_fixture(1.1025)
    rows=[(r[0],3.+.01*r[6],*r[2:]) for r in rows]
    alignment=stationary_observer_alignment(odom,rows)
    assert alignment==pytest.approx((10.-.011025,-3.,0.))
    node=SimpleNamespace(observer_map_alignment=alignment,odom=odom,truth_samples=rows)
    odom.pose.pose.position.x+=1.
    Trial._try_observer_alignment(node)
    assert node.observer_map_alignment==alignment


@pytest.mark.parametrize('bad',[None,'coasting','turning','nonzero_command','missing_truth_tail',
    'truth_gap','control_gap','short_window','nonmonotonic','nan'])
def test_refusal_stop_requires_actual_stationary_body_and_covered_command_tail(bad):
    from clearance_metrics import terminal_stop_evidence
    truth=[(0.,0.,0.,.26,0.,0.,i*.025,0.)for i in range(41)]
    controls=[(i*.025,0.,0.)for i in range(41)]
    if bad=='coasting':truth=[(r[0],r[6]*.08,*r[2:])for r in truth]
    elif bad=='turning':truth=[(*r[:7],r[6]*.12)for r in truth]
    elif bad=='nonzero_command':controls[-5]=(.9,.04,0.)
    elif bad=='missing_truth_tail':truth=truth[:-8]
    elif bad=='truth_gap':truth=truth[:25]+truth[32:]
    elif bad=='control_gap':controls=controls[:25]+controls[32:]
    elif bad=='short_window':truth=truth[-8:];controls=controls[-8:]
    elif bad=='nonmonotonic':truth[-2]=truth[-3]
    elif bad=='nan':controls[-1]=(1.,float('nan'),0.)
    assert terminal_stop_evidence(truth,controls)['accepted']==(bad is None)


def test_previously_measured_support_cannot_be_supplied_by_gp_pending_or_outside_cells(monkeypatch):
    import numpy as np
    import navigation_trial as trial
    values={'gp_model_ground_observed':np.ones((12,12)),
            'ground_pending_effective':np.zeros((12,12)),
            **{'probability_'+k:np.full((12,12),v) for k,v in
               (('front',.52),('back',.52),('half_width',.413))}}
    msg=SimpleNamespace(layers=list(values),info=SimpleNamespace(pos_resolution=.2,
        length_x=2.4,length_y=2.4,pose=SimpleNamespace(position=SimpleNamespace(x=0.,y=0.))))
    monkeypatch.setattr(trial,'get_layer_slices',lambda m,k:([values[k]],False,False))
    monkeypatch.setattr(trial,'grid_buffer_to_world',lambda a,m:a)
    assert trial.measured_probe_support(msg,0.,0.,.4)
    values['gp_model_ground_observed'][6,6]=0.  # A GP prediction is not a measurement.
    assert not trial.measured_probe_support(msg,0.,0.,0.)
    values['gp_model_ground_observed'][6,6]=1.
    values['ground_pending_effective'][6,6]=1.
    assert not trial.measured_probe_support(msg,0.,0.,0.)
    values['ground_pending_effective'][6,6]=0.
    assert not trial.measured_probe_support(msg,1.,0.,0.)


def test_observed_clearance_switch_is_preserved_and_cannot_relabel_a_cold_occluded_fixture():
    from pathlib import Path
    import pipeline_smoke as pipeline
    args=pipeline.parse_args(['--trial-scenario','crossing','--require-clearance-recovery'])
    command=pipeline.navigation_trial_command(args,Path('run'))
    assert '--require-clearance-recovery' in command
    assert '--clearance-observation-window' not in command
    args=pipeline.parse_args(['--trial-scenario','crossing','--require-clearance-recovery','--clearance-observation-window','4'])
    command=pipeline.navigation_trial_command(args,Path('run'))
    assert command[command.index('--clearance-observation-window')+1]=='4.0'
    with pytest.raises(SystemExit):pipeline.parse_args(['--clearance-observation-window','4'])
    for scenario in ('clearance','emerge','step'):
        with pytest.raises(SystemExit):pipeline.parse_args(['--trial-scenario',scenario,'--require-clearance-recovery'])


def test_window_allows_the_declared_deadline_and_entire_confirmation_instead_of_censoring_it():
    from clearance_metrics import EARLY_RECOVERY_WINDOW_S
    # 1.5s is still an admissible first clear, then keep the original 2s hold.
    samples=[(i*.5,1. if i<3 else 0.) for i in range(9)]
    early=sustained_clearance([r for r in samples if r[0]<=3.],.2)
    complete=sustained_clearance([r for r in samples if r[0]<=EARLY_RECOVERY_WINDOW_S],.2)
    assert early['first_sustained_clear_s'] is None
    assert complete['first_sustained_clear_s']==1.5 and complete['clear_at_end']


def test_localization_alignment_removes_origin_but_never_refits_drift():
    import math
    from clearance_metrics import localization_agreement
    truth=[(t,t,0.,.30,0.,0.,t,0.) for t in [i*.05 for i in range(61)]]
    # Sensor frame is translated and rotated; timestamps lie between truth frames.
    poses=[(t,10.,20.-t,1.3,-math.pi/2) for t in [.025+i*.05 for i in range(60)]]
    good=localization_agreement(poses,truth)
    assert good['accepted'] and good['xy_max_m']<1e-12 and good['z_max_m']<1e-12
    for index,delta in ((1,.4),(3,.2),(4,.2)):
        broken=[list(row) for row in poses];broken[-1][index]+=delta
        assessment=localization_agreement(broken,truth)
        assert not assessment['accepted']
    poses[-1]=(poses[-2][0],*poses[-1][1:])
    assert not localization_agreement(poses,truth)['accepted']


def test_localization_does_not_interpolate_across_gaps_or_accept_sparse_matches():
    from clearance_metrics import localization_agreement
    poses=[(i*.05,0.,0.,0.,0.) for i in range(100)]
    truth=[(t,0.,0.,0.,0.,0.,t,0.) for t in (0.,.05,4.9,4.95)]
    result=localization_agreement(poses,truth)
    assert not result['accepted'] and result['matched_count']==4
    assert result['matched_fraction']==.04


def test_localization_interpolates_yaw_across_pi_without_full_turn_error():
    import math
    from clearance_metrics import localization_agreement
    truth=[(i*.1,0.,0.,0.,0.,0.,i*.1,math.atan2(math.sin(3.+i*.02),math.cos(3.+i*.02)))
           for i in range(40)]
    poses=[(i*.1+.05,0.,0.,0.,3.+i*.02+.01) for i in range(39)]
    result=localization_agreement(poses,truth)
    assert result['accepted'] and result['yaw_max_rad']<1e-12


def test_underpass_uses_actual_height_and_attitude_while_post_stays_blocked():
    from clearance_metrics import static_course_clearances
    body=[0.,0.,0.,.30,0.,0.,1.,0.]
    panel=dict(name='beam',shape='box',x=0.,y=0.,z=1.,length=.2,width=2.,height=.2)
    planar=static_course_clearances([body],[panel])['beam'][0][1]
    spatial=static_course_clearances([body],[panel],body_z_bounds=(-.30,.35))['beam'][0][1]
    assert planar<0 and spatial==pytest.approx(.25)
    tilted=body.copy();tilted[4]=-.5
    assert static_course_clearances([tilted],[panel],body_z_bounds=(-.30,.35))['beam'][0][1]<.10
    low=dict(panel,z=.5)
    assert static_course_clearances([body],[low],body_z_bounds=(-.30,.35))['beam'][0][1]<0
    post=dict(name='post',shape='cylinder',x=0.,y=0.,z=.5,radius=.05,height=1.)
    assert static_course_clearances([body],[post],body_z_bounds=(-.30,.35))['post'][0][1]<0


@pytest.mark.parametrize("alignment", [(0.,0.,0.),(-.185,.23,.4),(10.,-7.,-1.2)])
@pytest.mark.parametrize("ground", [-3.2,-.48,1.7])
def test_obstacle_count_is_invariant_to_map_origin_and_rotation(ground,alignment):
    from navigation_trial import course_to_map
    points=[(4.+i*.02,1.4,ground) for i in range(100)]
    points += [(5.,0.,ground+.31)]*10 + [(5.,0.,ground+.20)]*7
    points=[(*course_to_map(x,y,alignment),z) for x,y,z in points]
    cloud=create_cloud_xyz32(Header(frame_id="map"),points)
    node=SimpleNamespace(obstacle_x=5.,reference_ground_z=None,vacated_stamp=None,
                         registered_peak_before_vacate=0,observer_map_alignment=alignment)
    node.count_obstacle_points=lambda msg:Trial.count_obstacle_points(node,msg)
    assert node.count_obstacle_points(cloud)==0
    Trial.on_registered(node,cloud)
    assert node.reference_ground_z==pytest.approx(ground)
    assert node.count_obstacle_points(cloud)==10
    assert node.registered_peak_before_vacate==10
    original=node.reference_ground_z
    raised=create_cloud_xyz32(Header(frame_id="map"),[(*course_to_map(5.,0.,alignment),ground+1.)]*50)
    Trial.on_registered(node,raised)
    assert node.reference_ground_z==original


def test_clearing_requires_all_layers_and_no_reappearance():
    good=sustained_clearance([(i*.1,0.) for i in range(40)],.2,max_gap_s=.3)
    layers={name:dict(good) for name in
            ("registered","laser_obstacle","baseline_risk","enhanced_risk","live")}
    assert clearance_acceptance(layers)
    for field,value in (("first_sustained_clear_s",1.6),("first_sustained_clear_s",None),
                        ("clear_at_end",False),("reappearance_count",1)):
        broken=deepcopy(layers);broken["enhanced_risk"][field]=value
        assert not clearance_acceptance(broken)
    del layers["registered"]
    assert not clearance_acceptance(layers)


def test_single_clear_frame_and_unknown_interval_do_not_pass():
    transient=sustained_clearance([(0.,0.),(.1,1.),(.2,None),(.3,0.)],.2)
    assert transient["first_sustained_clear_s"] is None
    assert transient["clear_at_end"] is False


def test_probability_clearance_requires_known_pose_even_when_heuristic_colors_clear():
    good=sustained_clearance([(i*.1,.01) for i in range(40)],.05,max_gap_s=.3)
    layers={name:dict(good) for name in
            ('registered','laser_obstacle','baseline_risk','enhanced_risk','live')}
    assert not clearance_acceptance(layers,require_probability=True)
    layers['probability_pose']=dict(good)
    assert clearance_acceptance(layers,require_probability=True)
    layers['probability_pose']=sustained_clearance([(i*.1,None) for i in range(40)],.05)
    assert not clearance_acceptance(layers,require_probability=True)
    layers['probability_pose']=sustained_clearance([(i*.1,.0504) for i in range(40)],.05)
    assert not clearance_acceptance(layers,require_probability=True)


def test_probability_probe_keeps_raw_precision_and_matches_planner_yaw_bin():
    import math
    import numpy as np
    from se2_grid_msgs.msg import SE2Grid
    from fsgp_bgk_se2.se2_grid_tools import matrix_to_multiarray_like,world_to_grid_buffer
    from navigation_trial import map_probe
    msg=SE2Grid();msg.info.pos_resolution=.2;msg.info.length_x=msg.info.length_y=.8
    msg.info.yaw_resolution=math.pi/4;msg.info.length_yaw=2*math.pi
    for name,values in (('probability_valid',[1.]*8),
                        ('failure_probability_upper',[.0504]+[1.]*7)):
        msg.layers.append(name);msg.has_so2.append(True)
        for value in values:
            msg.data.append(matrix_to_multiarray_like(world_to_grid_buffer(np.full((4,4),value),msg),None))
    probe=map_probe(msg,.1,.1,.6)
    assert .05<probe['failure_probability_upper']<.051
    assert probe['probability_valid']==1


def test_step_clearance_includes_rotated_bumpers_not_only_vehicle_center():
    import math
    from clearance_metrics import rectangle_front_wall_clearance
    shape=dict(front=.58,back=.48,half_width=.4)
    assert rectangle_front_wall_clearance(3.,0.,4.,**shape)==pytest.approx(.42)
    assert rectangle_front_wall_clearance(3.,math.pi,4.,**shape)==pytest.approx(.52)
    assert rectangle_front_wall_clearance(3.,math.pi/2,4.,**shape)==pytest.approx(.60)
    # A center-only 3.55m threshold could previously admit bumper penetration.
    assert rectangle_front_wall_clearance(3.55,0.,4.,**shape)<0
