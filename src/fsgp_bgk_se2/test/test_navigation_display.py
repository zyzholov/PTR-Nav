import math
import numpy as np
from se2_grid_msgs.msg import SE2Grid

from fsgp_bgk_se2.display_geometry import project_path,command_projection
from fsgp_bgk_se2.navigation_display_node import inspect_grid,NavigationDisplay
from fsgp_bgk_se2.se2_grid_tools import matrix_to_multiarray_like,world_to_grid_buffer


def test_bad_serialized_input_cannot_kill_display_or_refresh_old_evidence(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    import fsgp_bgk_se2.navigation_display_node as display
    callback=Mock();old_grid=object();logger=Mock()
    node=SimpleNamespace(grid=old_grid,invalid_display_packets={},get_logger=lambda:logger)
    def broken(*_):raise RuntimeError('conversion failure')
    monkeypatch.setattr(display,'deserialize_grid',broken)
    NavigationDisplay.consume_payload(node,b'invalid',SE2Grid,'/test/grid',callback)
    assert node.grid is old_grid and node.invalid_display_packets=={'/test/grid':1}
    callback.assert_not_called();logger.warn.assert_called_once()
    sample=SE2Grid();sample.info.header.stamp.sec=12
    monkeypatch.setattr(display,'deserialize_grid',lambda *_:sample)
    NavigationDisplay.consume_payload(node,b'next',SE2Grid,'/test/grid',callback)
    callback.assert_called_once_with(sample)
    assert sample.info.header.stamp.sec==12 and node.invalid_display_packets['/test/grid']==1


def test_tracking_error_projects_to_segments_not_only_vertices():
    error,point=project_path(5,1,[(0,0,0),(10,0,2)])
    assert error==1
    np.testing.assert_allclose(point,[5,0,1])
    assert project_path(0,0,[]) is None


def test_projection_supports_forward_reverse_and_rejects_invalid():
    p=command_projection(0,0,0,1,0,.549)
    np.testing.assert_allclose(p[-1],[1,0])
    p=command_projection(0,0,0,-1,.2,.549)
    assert p[-1][0]<0 and p[-1][1]>0
    assert command_projection(0,0,0,math.nan,0,.549)==[]


def test_inspector_preserves_ring_layout_units_and_unknown():
    msg=SE2Grid(); msg.info.pos_resolution=.2
    msg.info.length_x=1.6; msg.info.length_y=1.2
    msg.info.yaw_resolution=math.pi/2; msg.info.length_yaw=2*math.pi
    msg.start_index_row=2; msg.start_index_col=3
    height=np.arange(48,dtype=np.float32).reshape(6,8)*.01
    for name,value in (("elevation",height),("var",np.full_like(height,.0004)),
                       ("valid",np.ones_like(height)),("probability_height_std_floor",np.full_like(height,.005)),
                       ("probability_variance_scale",np.ones_like(height))):
        msg.layers.append(name); msg.has_so2.append(False)
        msg.data.append(matrix_to_multiarray_like(world_to_grid_buffer(value,msg),None))
    q=inspect_grid(msg,-.7,-.5,0)
    assert q["source"]=="observed"
    assert q["elevation"]==0
    assert abs(q["model_sigma"]-math.sqrt(.0004+.005**2))<1e-8
    assert "failure_probability_upper" not in q
    assert inspect_grid(msg,100,0,0) is None


def test_gp_candidate_probe_displays_prediction_without_relabeling_unknown_as_measured():
    msg=SE2Grid();msg.info.pos_resolution=.2;msg.info.length_x=msg.info.length_y=.4
    msg.info.yaw_resolution=math.pi/2;msg.info.length_yaw=2*math.pi
    for name,value in (("elevation",np.nan),("var",np.nan),("valid",0),("terrain_observed",0),
                       ("gp_candidate",1),("gp_mean",.3),("gp_latent_var",.0004),
                       ("probability_height_std_floor",.005),("probability_variance_scale",1)):
        msg.layers.append(name);msg.has_so2.append(False)
        msg.data.append(matrix_to_multiarray_like(np.full((2,2),value,np.float32),None))
    q=inspect_grid(msg,-.1,-.1,0)
    assert q["source"]=="GP candidate (not navigation)"
    assert abs(q["display_height"]-.3)<1e-6
    assert math.isnan(q["elevation"]) and q["valid"]==0
    assert abs(q["model_sigma"]-math.sqrt(.0004+.005**2))<1e-8
    assert "failure_probability_upper" not in q


def test_terrain_is_published_without_odometry():
    import rclpy
    from unittest.mock import Mock
    rclpy.init()
    node=NavigationDisplay()
    try:
        msg=SE2Grid();msg.info.header.frame_id="map"
        msg.info.pos_resolution=.2;msg.info.length_x=msg.info.length_y=.4
        msg.info.yaw_resolution=math.pi/2;msg.info.length_yaw=2*math.pi
        msg.layers=["elevation"];msg.has_so2=[False]
        msg.data=[matrix_to_multiarray_like(np.zeros((2,2),np.float32),None)]
        node.grid=msg
        node.terrain_publishers={k:Mock() for k in node.terrain_publishers}
        node.render()
        node.terrain_publishers["measured_height"].publish.assert_called_once()
        assert node.odom is None
    finally:
        node.destroy_node();rclpy.shutdown()


def test_navigation_prediction_uses_probability_height_without_inventing_raw_measurement():
    msg=SE2Grid();msg.info.pos_resolution=.2;msg.info.length_x=msg.info.length_y=.4
    msg.info.yaw_resolution=math.pi/2;msg.info.length_yaw=2*math.pi
    for name,value in (("elevation",np.nan),("var",np.nan),("valid",0),("terrain_observed",0),
                       ("probability_gp_used",1),("probability_elevation",.24),
                       ("probability_variance",.0003),("gp_candidate",1),
                       ("gp_mean",.24),("gp_latent_var",.0003)):
        msg.layers.append(name);msg.has_so2.append(False)
        msg.data.append(matrix_to_multiarray_like(np.full((2,2),value,np.float32),None))
    q=inspect_grid(msg,-.1,-.1,0)
    assert q["source"]=="GP navigation support"
    assert math.isclose(q["display_height"],.24,abs_tol=1e-6)
    assert math.isnan(q["elevation"]) and q["terrain_observed"]==0


def probability_probe_message(**changes):
    msg=SE2Grid();msg.info.pos_resolution=.2;msg.info.length_x=msg.info.length_y=2.
    msg.info.yaw_resolution=math.pi/2;msg.info.length_yaw=2*math.pi
    values=dict(elevation=-.4,var=.0001,valid=1.,terrain_observed=1.,ground_pending=0.,
        probability_valid=1.,failure_probability_upper=.01,probability_elevation=-.39,
        probability_variance=.0009,probability_height_std_floor=.005,probability_variance_scale=1.,
        collision_risk=0.,probability_front=.68,probability_back=.58,probability_half_width=.5)
    values.update(changes)
    for name,value in values.items():
        a=np.full((10,10),value,np.float32) if np.isscalar(value) else value
        msg.layers.append(name);msg.has_so2.append(False)
        msg.data.append(matrix_to_multiarray_like(world_to_grid_buffer(a,msg),None))
    return msg


def test_probe_keeps_original_pending_evidence_but_explains_separated_column_side():
    q=inspect_grid(probability_probe_message(ground_pending=1.,ground_pending_effective=0.,
        ground_pending_column_side=1.,terrain_observed=0.,probability_gp_used=1.),.1,.1,0)
    assert q['ground_pending']==1. and q['ground_pending_column_side']==1.
    assert q['source']=='GP navigation support' and q['risk_cause_code']=='pose_ok'
    q=inspect_grid(probability_probe_message(ground_pending=1.,ground_pending_effective=1.,
        probability_valid=0.),.1,.1,0)
    assert q['risk_cause_code']=='pending'


def test_probe_uses_actual_continuous_body_and_height_proof_instead_of_the_display_raster():
    occupied=np.zeros((10,10),np.float32);occupied[6,8]=1.
    fine=np.zeros_like(occupied);fine[6,8]=2.
    msg=probability_probe_message(collision_risk=occupied,collision_subcell_bits=fine,
        collision_subcell_contract=3.,collision_subcell_factor=2.,collision_body_front=.68,
        collision_body_back=.58,collision_body_half_width=.5,
        failure_probability_upper=1.,terrain_failure_probability_upper=.0001)
    clear=inspect_grid(msg,0,0,0)
    assert clear['failure_probability_upper']==1.  # original raster retained for diagnosis
    assert clear['body_geometry_known'] and not clear['footprint_collision']
    assert clear['risk_cause_code']=='pose_ok'
    assert abs(clear['effective_pose_probability_upper']-.0001)<1e-8
    hit=inspect_grid(msg,.05,0,0)
    assert hit['body_geometry_known'] and hit['footprint_collision']
    assert hit['risk_cause_code']=='collision' and hit['effective_pose_probability_upper']==1.
    edge=inspect_grid(msg,.9,.9,0)
    assert edge['risk_cause_code']=='geometry_unknown'
    assert not edge['body_geometry_known'] and math.isnan(edge['effective_pose_probability_upper'])


def test_probe_distinguishes_pending_ground_from_obstacle_and_measurement_from_navigation_sigma():
    q=inspect_grid(probability_probe_message(),.1,.1,0)
    assert q['risk_cause_code']=='pose_ok'
    assert q['navigation_sigma']>2*q['model_sigma']
    q=inspect_grid(probability_probe_message(ground_pending=1.,probability_valid=0.,
        ground_pending_elevation=-.35),.1,.1,0)
    assert q['risk_cause_code']=='pending' and q['source']=='ground transition pending'
    assert math.isnan(q['display_height']) and not q['footprint_collision']
    assert abs(q['ground_pending_elevation']+.35)<1e-6
    q=inspect_grid(probability_probe_message(probability_valid=0.),.1,.1,0)
    assert q['risk_cause_code']=='footprint_support' and q['source']=='observed'


def test_probe_detects_occupied_cell_inside_vehicle_footprint_and_uses_actual_budget():
    occupied=np.zeros((10,10),np.float32);occupied[5,7]=1
    q=inspect_grid(probability_probe_message(collision_risk=occupied,failure_probability_upper=1.),.1,.1,0)
    assert q['collision_risk']==0 and q['footprint_collision']
    assert q['risk_cause_code']=='collision'
    msg=probability_probe_message(failure_probability_upper=.06)
    assert inspect_grid(msg,.1,.1,0)['risk_cause_code']=='terrain_limit'
    assert inspect_grid(msg,.1,.1,0,state_limit=.1)['risk_cause_code']=='pose_ok'


def test_probe_exposes_distinct_vehicle_limits_without_changing_risk_or_unknown_support():
    msg=probability_probe_message(probability_contract=7.,probability_max_step=.1,
        probability_max_roughness=.125,probability_ground_clearance=.135,
        probability_underbody_margin=.01,probability_valid=0.,terrain_observed=0.)
    q=inspect_grid(msg,.1,.1,0)
    assert q['probability_contract']==7.
    assert math.isclose(q['probability_max_step'],.1,abs_tol=1e-7)
    assert math.isclose(q['probability_max_roughness'],.125,abs_tol=1e-7)
    assert math.isclose(q['probability_ground_clearance']-q['probability_underbody_margin'],.125,abs_tol=1e-7)
    assert q['risk_cause_code']=='query_support'
    assert q['probability_valid']==0. and math.isclose(q['failure_probability_upper'],.01,abs_tol=1e-7)


def test_phase_gate_keeps_eight_hz_from_ten_hz_without_catchup_bursts():
    from fsgp_bgk_se2.fsgp_bgk_se2_node import map_deadline
    deadline=0.;accepted=[]
    for t in np.arange(100,110,.1):
        next_deadline=map_deadline(t,deadline,.125)
        if next_deadline is not None:accepted.append(t);deadline=next_deadline
    assert 79<=len(accepted)<=81
    assert map_deadline(1000,deadline,.125)==1000.125
    assert map_deadline(1000.01,1000.125,.125) is None
