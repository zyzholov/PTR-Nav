import math
from unittest.mock import patch

import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("torch")
pytest.importorskip("gpytorch")

from se2_grid_msgs.msg import SE2Grid
from std_msgs.msg import Header
from sensor_msgs_py.point_cloud2 import create_cloud_xyz32
from fsgp_bgk_se2.obstacle_columns import vertical_columns
from fsgp_bgk_se2.fsgp_bgk_se2_node import FsgpBgkSe2Node
from fsgp_bgk_se2.se2_grid_tools import (get_layer_slices, grid_buffer_to_world,
                                       matrix_to_multiarray_like, world_to_grid_buffer)


def test_sim_startup_latches_only_the_planner_ack_without_impersonating_goal_readiness():
    from geometry_msgs.msg import PoseStamped
    rclpy.init(args=['--ros-args','-p','use_sim_time:=true','-p','gpu_backend:=cpu',
        '-p','startup_support_mode:=stable_stance','-p','startup_support_directional:=true'])
    node=None
    try:
        node=FsgpBgkSe2Node()
        topics={s.topic_name for s in node.subscriptions}
        assert '/terrain_nav/received_goal' in topics
        assert '/goal_pose' not in topics and '/move_base_simple/goal' not in topics
        assert '/terrain_nav/chance_status' in topics
        assert node.startup_support.require_geometry and node.startup_support.footprint is None
        assert not node.startup_support.armed
        from rclpy.qos import DurabilityPolicy
        assert node.startup_goal_sub.qos_profile.durability==DurabilityPolicy.TRANSIENT_LOCAL
        stale=PoseStamped();stale.header.stamp.sec=5
        node.startup_goal_sub.callback(stale)
        node._qualify_startup_goal(46.) # A retained old mission cannot initialize this stance.
        assert not node.startup_support.armed
        fresh=PoseStamped();fresh.header.stamp.sec=46;fresh.header.stamp.nanosec=50_000_000
        node.startup_goal_sub.callback(fresh) # /clock may arrive after the ack.
        node._qualify_startup_goal(46.) # A source stamp slightly ahead waits for the next map.
        assert not node.startup_support.armed
        assert node.get_clock().now().nanoseconds==0
        node._qualify_startup_goal(46.1)
        assert node.startup_support.armed and node.startup_support.anchor is None
        from diagnostic_msgs.msg import DiagnosticArray,DiagnosticStatus,KeyValue
        msg=DiagnosticArray();status=DiagnosticStatus();status.name='terrain_nav/chance_constraints'
        status.values=[KeyValue(key=k,value=v) for k,v in dict(collision_front_m='.68',collision_back_m='.58',
            collision_protected_width_m='1.0').items()];msg.status=[status]
        node._startup_geometry_callback(msg)
        assert node.startup_support.footprint==(.68,.58,.5)
    finally:
        if node is not None:node.destroy_node()
        rclpy.shutdown()


@pytest.mark.parametrize('reliable',[False,True])
def test_cloud_subscription_policy_is_explicit_and_retains_only_the_latest(reliable):
    from rclpy.qos import ReliabilityPolicy
    rclpy.init(args=['--ros-args','-p','gpu_backend:=cpu','-p','subscribe_cloud:=true',
                    '-p','cloud_reliable:='+str(reliable).lower()])
    node=None
    try:
        node=FsgpBgkSe2Node()
        qos=node.cloud_sub.qos_profile
        assert qos.depth==1
        assert qos.reliability==(ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT)
    finally:
        if node is not None:node.destroy_node()
        rclpy.shutdown()


def test_shared_clearance_reference_uses_the_configured_derived_cloud_policy():
    from rclpy.qos import ReliabilityPolicy
    rclpy.init(args=['--ros-args','-p','gpu_backend:=cpu','-p','use_sim_time:=true',
        '-p','prediction_backend:=feature_sgp','-p','gp_device:=cpu','-p','gp_training_steps:=0',
        '-p','gp_complete_supported:=true','-p','probability_use_gp_completion:=true','-p','probability_model:=shared_gp',
        '-p','cloud_reliable:=true'])
    node=None
    try:
        node=FsgpBgkSe2Node()
        assert node.reference_cloud_sub.qos_profile.depth==1
        assert node.reference_cloud_sub.qos_profile.reliability==ReliabilityPolicy.RELIABLE
    finally:
        if node is not None:node.destroy_node()
        rclpy.shutdown()


@pytest.mark.parametrize('limit',['0.0','-0.1','0.76','0.6'])
def test_extended_support_is_bounded_and_requires_simulation(limit):
    rclpy.init(args=['--ros-args','-p','gpu_backend:=cpu','-p',
                    'probability_gp_max_distance_m:='+limit])
    try:
        with pytest.raises(ValueError):FsgpBgkSe2Node()
    finally:rclpy.shutdown()


def test_explicit_simulation_distance_reaches_current_conditioner():
    rclpy.init(args=['--ros-args','-p','use_sim_time:=true','-p','gpu_backend:=cpu',
                    '-p','prediction_backend:=feature_sgp','-p','gp_device:=cpu','-p','gp_async:=false',
                    '-p','gp_complete_supported:=true','-p','probability_use_gp_completion:=true',
                    '-p','probability_gp_max_distance_m:=0.6'])
    node=FsgpBgkSe2Node()
    try:
        assert node.completion_distance==.6
        assert node.completion_predictor.params.local_nearest_m==.6
    finally:node.destroy_node();rclpy.shutdown()


@pytest.mark.parametrize('full,sources,floor',[(False,0,False),(True,0,False),(True,64,False),(True,64,True)])
def test_shared_model_serialization_keeps_raw_unknown_and_collision_identity(full,sources,floor):
    rclpy.init(args=['--ros-args','-p','use_sim_time:=true','-p','gpu_backend:=cpu',
        '-p','prediction_backend:=feature_sgp','-p','gp_device:=cpu','-p','gp_async:=true',
        '-p','gp_training_steps:=0','-p','yaw_bins:=8','-p','gp_complete_supported:=true',
        '-p','probability_use_gp_completion:=true','-p','probability_model:=shared_gp',
        '-p','probability_gp_max_distance_m:=0.75',
        '-p',f'shared_gp_condition_full_residual:={str(full).lower()}',
        '-p',f'shared_gp_residual_inducing_points:={sources}',
        '-p',f'shared_gp_independent_model_floor:={str(floor).lower()}',
        '-p',f'probability_roughness_limit_m:={.125 if floor else 0.}'])
    node=FsgpBgkSe2Node()
    try:
        assert node.prediction_worker is None
        x,y=np.meshgrid((np.arange(30)-14.5)*.2,(np.arange(30)-14.5)*.2)
        z=.01*x-.01*y;observed=np.hypot(x,y)>.6
        msg=SE2Grid();msg.info.header.frame_id='map';msg.info.pos_resolution=.2
        msg.info.length_x=msg.info.length_y=6.;msg.info.length_yaw=2*math.pi
        msg.info.yaw_resolution=2*math.pi/8
        risk=np.zeros(z.shape);risk[5,5]=1; risk[17,15]=1
        for name,matrix in (('elevation',np.where(observed,z,np.nan)),('var',np.full(z.shape,.0001)),
                            ('valid',observed.astype(float)),('risk',risk)):
            msg.layers.append(name);msg.has_so2.append(False)
            msg.data.append(matrix_to_multiarray_like(world_to_grid_buffer(matrix,msg),None))
        obstacle=np.array([[x[5,5],y[5,5],z[5,5]+.6],
                           [x[5,5]+.01,y[5,5],z[5,5]+.65]])
        node.latest_cloud=create_cloud_xyz32(Header(frame_id='map'),
            np.r_[np.c_[x[observed],y[observed],z[observed]],obstacle])
        _,out=node._process_map(msg)
        def layer(name):
            a,_,_=get_layer_slices(out,name);return [grid_buffer_to_world(v,out) for v in a]
        assert len(out.layers)==len(set(out.layers))
        assert layer('probability_contract')[0][15,15]==(8 if floor else 6)
        assert layer('probability_floor_covariance_model')[0][15,15]==float(floor)
        assert layer('probability_step_model')[0][15,15]==1
        assert layer('probability_conditioned_local_residual')[0][15,15]==1
        assert layer('probability_conditioned_full_residual')[0][15,15]==float(full)
        assert layer('probability_residual_inducing_points')[0][15,15]==sources
        assert layer('probability_joint_model')[0][15,15]==1
        assert layer('probability_startup_prior_enabled')[0][15,15]==0
        assert np.isnan(layer('gp_elevation_observed')[0][15,15])
        assert layer('terrain_observed')[0][15,15]==0
        assert layer('probability_gp_used')[0][15,15]==1
        assert np.isfinite(layer('probability_elevation')[0][15,15])
        assert layer('probability_elevation')[0][15,15]==layer('gp_mean')[0][15,15]
        assert all(a[15,15]==1 for a in layer('probability_valid'))
        assert all(a[15,15]<.05 for a in layer('failure_probability_upper'))
        # This fixture supplies BOTH a real measured floor and raised pole
        # returns in the same coarse column. The floor witness must preserve
        # that measurement provenance while retaining the occupied volume.
        # Erasing it and labeling it GP would reintroduce the wall-seam bug.
        assert layer('probability_joint_used')[0][5,5]==1
        assert layer('probability_gp_used')[0][5,5]==0
        assert layer('gp_model_ground_observed')[0][5,5]==1
        assert layer('gp_observed')[0][5,5]==1
        assert layer('terrain_observed')[0][5,5]==1
        assert layer('probability_predict_ground_under_columns')[0][5,5]==1
        assert layer('collision_risk')[0][5,5]==1
        assert all(a[5,5]==1 for a in layer('failure_probability_upper'))
        assert np.all(layer('collision_subcell_contract')[0]==3)
        height_bounds=layer('terrain_failure_probability_upper')
        assert len(height_bounds)==8
        for height,composite in zip(height_bounds,layer('failure_probability_upper')):
            finite=np.isfinite(height)
            assert np.all(composite[finite]>=height[finite])
            assert height[15,15]==composite[15,15]
        # An upstream pose cost of one is not a point-occupancy measurement.
        # The original heuristic is preserved, while measured flat ground
        # remains available to the independent full-footprint certificate.
        assert layer('baseline_risk')[0][17,15]==1
        assert layer('probability_joint_used')[0][17,15]==1
        # New negative evidence persists through occlusion and needs three
        # distinct measured clearing scans, not repeated map callbacks.
        raw=np.c_[x[observed],y[observed],z[observed]]
        assert observed[18,15]
        pit=np.array([[x[18,15],y[18,15],z[18,15]-.4]])
        def update_cloud(points,sec):
            msg.info.header.stamp.sec=sec
            node.latest_cloud=create_cloud_xyz32(Header(frame_id='map',stamp=msg.info.header.stamp),points)
            return node._process_map(msg)[1]
        out=update_cloud(np.r_[raw,pit],1)
        assert layer('probability_raw_veto')[0][18,15]==1
        out=update_cloud(raw,2)
        assert layer('probability_raw_veto')[0][18,15]==1
        # The voxel-filtered cloud has one ground witness per cell per scan.
        # Second 2 above is already the first clearing witness.
        for sec in (3,3,3):
            out=update_cloud(raw,sec)
            assert layer('probability_raw_veto')[0][18,15]==1
        out=update_cloud(raw,4)
        assert layer('probability_raw_veto')[0][18,15]==0
        assert layer('probability_joint_used')[0][18,15]==1
        node.latest_cloud=None
        _,stale=node._process_map(msg)
        a,_,_=get_layer_slices(stale,'probability_valid')
        assert all(grid_buffer_to_world(v,stale)[15,15]==0 for v in a)
        # A changed floor being confirmed must not be refilled by the GP or
        # reported as a hard obstacle. Its unknown support veto is immediate.
        pending=np.zeros(z.shape);pending[15,15]=1
        msg.layers.append('ground_pending');msg.has_so2.append(False)
        msg.data.append(matrix_to_multiarray_like(world_to_grid_buffer(pending,msg),None))
        out=update_cloud(raw,5)
        assert layer('terrain_observed')[0][15,15]==0
        assert layer('collision_risk')[0][15,15]==0
        assert layer('probability_joint_used')[0][15,15]==0
        assert all(a[15,15]==0 for a in layer('probability_valid'))
    finally:node.destroy_node();rclpy.shutdown()


def test_current_cloud_age_uses_exact_ros_stamp_difference():
    from builtin_interfaces.msg import Time
    rclpy.init(args=["--ros-args","-p","prediction_backend:=none","-p","gpu_backend:=cpu"])
    node=FsgpBgkSe2Node()
    try:
        msg=SE2Grid();msg.info.header.frame_id="map"
        for sec,nanosec in ((53,914999776),(114,414997442),(1792565356,999999998)):
            msg.info.header.stamp=Time(sec=sec,nanosec=nanosec)
            node.latest_cloud=create_cloud_xyz32(Header(frame_id="map",stamp=msg.info.header.stamp),[[0.,0.,0.]])
            _,age=node._current_points(msg)
            assert age==0.
            node.latest_cloud.header.stamp=Time(sec=sec,nanosec=nanosec+1)
            _,age=node._current_points(msg)
            assert age==-1e-9  # Actual future stamps are not clamped to zero.
        msg.info.header.stamp=Time(sec=115,nanosec=100)
        node.latest_cloud.header.stamp=Time(sec=114,nanosec=999999900)
        assert node._current_points(msg)[1]==pytest.approx(2e-7)
    finally:node.destroy_node();rclpy.shutdown()


def test_supported_completion_keeps_observation_identity_and_raw_collision_evidence():
    from types import SimpleNamespace
    rclpy.init(args=["--ros-args","-p","gp_async:=false","-p","prediction_backend:=feature_sgp","-p","gp_device:=cpu",
                    "-p","gpu_backend:=cpu","-p","yaw_bins:=8","-p","gp_complete_supported:=true"])
    node=FsgpBgkSe2Node()
    try:
        shape=(16,16);n=256
        msg=SE2Grid();msg.info.header.frame_id="map";msg.info.pos_resolution=.2
        msg.info.length_x=msg.info.length_y=3.2;msg.info.length_yaw=2*math.pi;msg.info.yaw_resolution=2*math.pi/8
        z=np.full(shape,-.48,np.float32);valid=np.ones(shape,np.float32);valid[7,7]=valid[12,12]=0
        z[valid==0]=np.nan;baseline=np.zeros(shape,np.float32);baseline[12,12]=1
        for name,matrix in (("elevation",z),("valid",valid),("var",np.full(shape,.0004)),("risk",baseline)):
            msg.layers.append(name);msg.has_so2.append(False)
            msg.data.append(matrix_to_multiarray_like(world_to_grid_buffer(matrix,msg),None))
        p=SimpleNamespace(mean=np.full(n,-.48),latent_var=np.full(n,.0009),observation_var=np.full(n,.0013),
            nearest_distance=np.full(n,.1),accepted=np.ones(n,bool),supported=np.ones(n,bool),
            local_conditioned=np.ones(n,bool),device="cpu",fit_ms=1.,predict_ms=1.)
        node.predictor.model=True;node._gp_fit_wall=1e30;node._gp_source_stamp_s=0.
        with patch.object(node.predictor,"predict",return_value=p):
            _,out=node._process_map(msg)
            assert set(node._process_stage_ms)=={
                "decode","posterior","current_gp","bgk_seb","collision","message_base",
                "support","probability","conditional","message_layers"}
            assert all(math.isfinite(ms) and ms>=0 for ms in node._process_stage_ms.values())
            assert out.info.header.stamp==msg.info.header.stamp
            def layer(name):
                arrays,_,_=get_layer_slices(out,name);return grid_buffer_to_world(arrays[0],out)
            assert len(out.layers)==len(set(out.layers))
            assert layer("valid")[7,7]==1 and layer("gp_observed")[7,7]==0
            assert layer("gp_completed")[7,7]==1 and layer("gp_completed")[12,12]==0
            assert layer("elevation")[7,7]==pytest.approx(-.48) and layer("risk")[7,7]<.2
            assert layer("var")[7,7]==pytest.approx(.0009)
            assert np.isnan(layer("gp_elevation_observed")[7,7]) and np.isnan(layer("gp_variance_observed")[7,7])
            node.latest_cloud=create_cloud_xyz32(Header(frame_id="map"),[[-.1,-.1,.6],[-.1,-.1,.65]])
            with patch("fsgp_bgk_se2.fsgp_bgk_se2_node.vertical_columns",wraps=vertical_columns) as detector:
                _,collision=node._process_map(msg)
                assert np.isnan(detector.call_args.kwargs["ground_elevation"][7,7])
            arrays,_,_=get_layer_slices(collision,"collision_risk")
            assert grid_buffer_to_world(arrays[0],collision)[7,7]==1
    finally:
        node.destroy_node();rclpy.shutdown()


def test_gp_layers_do_not_convert_unknown_into_free_space():
    rclpy.init(args=["--ros-args", "-p", "gp_async:=false", "-p", "prediction_backend:=feature_sgp", "-p", "gp_device:=cpu",
                    "-p", "gpu_backend:=cpu", "-p", "gp_training_steps:=1", "-p", "yaw_bins:=8",
                    "-p", "gp_refine_observed:=true"])
    node = FsgpBgkSe2Node()
    try:
        msg = SE2Grid()
        msg.info.header.frame_id = "map"
        msg.info.pos_resolution = 0.2
        msg.info.length_x = msg.info.length_y = 3.2
        msg.info.length_yaw = 2 * math.pi + 0.02
        msg.info.yaw_resolution = msg.info.length_yaw / 8
        valid = np.ones((16, 16), dtype=np.float32)
        valid[6:10, 6:10] = 0
        elevation = np.zeros_like(valid)
        elevation[valid == 0] = np.nan
        baseline = np.zeros_like(valid)
        baseline[3, 3] = 1.0
        for name, matrix in (("elevation", elevation), ("var", np.full_like(valid, 0.001)),
                             ("valid", valid), ("risk", baseline)):
            msg.layers.append(name)
            msg.has_so2.append(False)
            msg.data.append(matrix_to_multiarray_like(world_to_grid_buffer(matrix, msg), None))
        _, out = node._process_map(msg)
        assert "gp_latent_var" in out.layers
        assert "gp_observation_var" in out.layers
        assert "gp_elevation_observed" in out.layers
        def layer(name):
            arrays, _, _ = get_layer_slices(out, name)
            return [grid_buffer_to_world(a, out) for a in arrays]
        np.testing.assert_equal(layer("valid")[0][6:10, 6:10], 0)
        for risk in layer("risk"):
            np.testing.assert_allclose(risk[6:10, 6:10], 0.65)
            assert risk[3, 3] >= 0.999
        assert np.isfinite(layer("gp_mean")[0]).all()
        np.testing.assert_equal(layer("gp_elevation_observed")[0],elevation)
        assert (layer("gp_observation_var")[0] > layer("gp_latent_var")[0]).all()
        # Two near-identical-height returns have no within-cell span, but the
        # node's original observed ground must corroborate the physical column.
        header = Header(frame_id="map")
        node.latest_cloud = create_cloud_xyz32(header, [[-.9, -.9, .6], [-.9, -.9, .65]])
        with patch("fsgp_bgk_se2.fsgp_bgk_se2_node.vertical_columns",
                   wraps=vertical_columns) as detector:
            _, collision_out = node._process_map(msg)
            ground = detector.call_args.kwargs["ground_elevation"]
            assert np.isnan(ground[6:10, 6:10]).all()
            assert ground[3, 3] == 0
        slices, _, _ = get_layer_slices(collision_out, "collision_risk")
        assert grid_buffer_to_world(slices[0], collision_out)[3, 3] == 1
        node.latest_cloud = None
        node.gp_fit_period_s = 1000.0
        with patch.object(node.predictor, "predict", wraps=node.predictor.predict) as predict:
            _, repeated = node._process_map(msg)
            assert "gp_mean" in repeated.layers
            predict.assert_not_called()
            msg.info.pose.position.x += 0.2
            node._process_map(msg)
            predict.assert_called_once()
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_navigation_completion_is_current_preserves_measurements_and_yields_to_new_evidence():
    rclpy.init(args=["--ros-args","-p","prediction_backend:=feature_sgp","-p","gp_device:=cpu",
        "-p","gpu_backend:=cpu","-p","yaw_bins:=8","-p","gp_complete_supported:=true",
        "-p","probability_use_gp_completion:=true"])
    node=FsgpBgkSe2Node()
    try:
        msg=SE2Grid();msg.info.header.frame_id="map";msg.info.header.stamp.sec=10
        msg.info.pos_resolution=.2;msg.info.length_x=msg.info.length_y=4.
        msg.info.length_yaw=2*math.pi;msg.info.yaw_resolution=2*math.pi/8
        x,y=np.meshgrid((np.arange(20)-9.5)*.2,(np.arange(20)-9.5)*.2)
        valid=np.ones(x.shape,bool);valid[10,10]=valid[10,13]=False
        z=np.where(valid,0.,np.nan).astype(np.float32)
        risk=np.where(valid,0.,.65);risk[10,13]=1
        for name,value in (("elevation",z),("var",np.full_like(z,.0001)),
                           ("valid",valid.astype(np.float32)),("risk",risk)):
            msg.layers.append(name);msg.has_so2.append(False)
            msg.data.append(matrix_to_multiarray_like(world_to_grid_buffer(value,msg),None))
        points=np.column_stack((x[valid],y[valid],z[valid]))
        node.latest_cloud=create_cloud_xyz32(Header(frame_id="map",stamp=msg.info.header.stamp),points)
        def layers(out,name):
            return [grid_buffer_to_world(a,out) for a in get_layer_slices(out,name)[0]]
        with patch.object(node,"_prediction_for_map",return_value=None):
            _,out=node._process_map(msg)
            used=layers(out,"probability_gp_used")[0]
            assert used[10,10]==1 and used[10,13]==0
            np.testing.assert_array_equal(layers(out,"elevation")[0][valid],z[valid])
            np.testing.assert_array_equal(layers(out,"terrain_observed")[0],valid)
            assert layers(out,"gp_completion_age")[0][10,10]==0
            # No global posterior is available: current reconditioning alone
            # must work; conversely, history without current corroboration cannot.
            from builtin_interfaces.msg import Time
            node.latest_cloud.header.stamp=Time(sec=9)
            _,stale=node._process_map(msg)
            assert not layers(stale,"probability_gp_used")[0].any()
            # A newly observed low step vetoes completion immediately, even
            # though it is lower than the tall-column detector's threshold.
            extra=np.array([[x[10,10],y[10,10],.25],[x[10,10]+.01,y[10,10]+.01,.26]])
            node.latest_cloud=create_cloud_xyz32(Header(frame_id="map",stamp=msg.info.header.stamp),np.r_[points,extra])
            _,changed=node._process_map(msg)
            assert layers(changed,"probability_gp_used")[0][10,10]==0
            assert np.isnan(layers(changed,"probability_elevation")[0][10,10])
    finally:node.destroy_node();rclpy.shutdown()
