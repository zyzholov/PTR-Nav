"""Mapper-owned obstacle history cannot become observed or predicted ground."""
import math

import numpy as np
import pytest

pytest.importorskip('rclpy')
import rclpy
from builtin_interfaces.msg import Time
from se2_grid_msgs.msg import SE2Grid
from sensor_msgs_py.point_cloud2 import create_cloud_xyz32
from std_msgs.msg import Header

from fsgp_bgk_se2.fsgp_bgk_se2_node import FsgpBgkSe2Node
from fsgp_bgk_se2.se2_grid_tools import (
    get_layer_slices, grid_buffer_to_world, matrix_to_multiarray_like, world_to_grid_buffer)


@pytest.fixture
def node():
    rclpy.init(args=['--ros-args', '-p', 'prediction_backend:=none', '-p', 'gpu_backend:=cpu',
                    '-p', 'yaw_bins:=8', '-p', 'publish_debug_clouds:=false'])
    instance = FsgpBgkSe2Node()
    yield instance
    instance.destroy_node()
    rclpy.shutdown()


def grid(stamp=10):
    msg = SE2Grid()
    msg.info.header = Header(frame_id='map', stamp=Time(sec=stamp))
    msg.info.length_x = msg.info.length_y = 3.2
    msg.info.pos_resolution = .2
    msg.info.length_yaw = 2*math.pi
    msg.info.yaw_resolution = 2*math.pi/8
    for name, matrix in (('elevation', np.zeros((16,16))), ('var', np.full((16,16), .0004)),
                         ('valid', np.ones((16,16))), ('risk', np.zeros((16,16)))):
        msg.layers.append(name)
        msg.has_so2.append(False)
        msg.data.append(matrix_to_multiarray_like(world_to_grid_buffer(matrix, msg), None))
    return msg


def layer(msg, name):
    values, _, _ = get_layer_slices(msg, name)
    return grid_buffer_to_world(values[0], msg)


def cloud(points, sec=10, frame='map'):
    return create_cloud_xyz32(Header(frame_id=frame, stamp=Time(sec=sec)), points)


def test_memory_geometry_only_adds_hazards_without_training_or_inventing_ground(node):
    msg = grid()
    node.latest_cloud = cloud([[.1,.1,0.]])
    _, before = node._process_map(msg)
    assert np.all(layer(before,'historical_obstacle_source_valid')==0)
    node._obstacle_memory_callback(cloud([[.5,.5,.8], [100.,100.,.8]]))
    points, age = node._memory_points(msg)
    assert points.shape == (1,3) and age == 0
    _, after = node._process_map(msg)
    assert np.all(layer(after,'historical_obstacle_source_valid')==1)
    np.testing.assert_allclose(layer(after,'historical_obstacle_source_age_s'),0.)
    assert layer(after, 'collision_memory').sum() == 1
    assert layer(after, 'collision_current').sum() == 0
    np.testing.assert_equal(layer(after, 'elevation'), layer(before, 'elevation'))
    np.testing.assert_equal(layer(after, 'terrain_observed'), layer(before, 'terrain_observed'))
    assert np.all(layer(after, 'collision_risk') >= layer(before, 'collision_risk'))
    assert layer(after, 'failure_probability_upper')[10,10] == 1
    assert node._obstacle_memory_cells == 1


def test_removed_mapper_object_is_not_retained_in_a_second_unbounded_cache(node):
    node._obstacle_memory_callback(cloud([[.5,.5,.8]], sec=9))
    assert layer(node._process_map(grid())[1], 'collision_memory').sum() == 1
    # A new, cleaned full snapshot replaces the old geometry; no local TTL.
    node._obstacle_memory_callback(cloud([[.5,.5,0.]], sec=10))
    assert layer(node._process_map(grid())[1], 'collision_memory').sum() == 0


def test_advisory_history_cannot_change_support_collision_or_probability(node):
    from rclpy.parameter import Parameter
    node.set_parameters([Parameter('obstacle_memory_role', value='advisory')])
    _, before = node._process_map(grid())
    node._obstacle_memory_callback(cloud([[.5,.5,.8]]))
    _, after = node._process_map(grid())
    assert layer(after, 'historical_obstacle_advisory').sum() == 1
    for name in ('collision_risk', 'collision_current', 'terrain_observed', 'elevation',
                 'failure_probability_upper', 'probability_valid'):
        old, _, _ = get_layer_slices(before, name)
        new, _, _ = get_layer_slices(after, name)
        np.testing.assert_equal(new, old)
    # Source expiry removes a preference, not safety evidence or terrain input.
    _, expired = node._process_map(grid(12))
    assert np.all(layer(expired,'historical_obstacle_source_valid')==0)
    assert layer(expired, 'historical_obstacle_advisory').sum() == 0
    np.testing.assert_equal(layer(expired, 'collision_risk'), layer(before, 'collision_risk'))


def test_navigation_overhead_slice_does_not_delete_raw_mapping_cloud_or_low_collision(node):
    from rclpy.parameter import Parameter
    node.set_parameters([Parameter('overhead_filter_enabled', value=True)])
    yy,xx=np.mgrid[:16,:16]
    floor=np.c_[-1.6+(xx.ravel()+.5)*.2,-1.6+(yy.ravel()+.5)*.2,np.zeros(256)]
    raw=cloud(np.r_[floor,[[.5,.5,1.1],[.7,.5,.6],[.7,.5,.5]]])
    original=bytes(raw.data);node.latest_cloud=raw
    _,out=node._process_map(grid())
    assert not layer(out,'collision_current')[10,10]
    assert layer(out,'collision_current')[10,11]
    assert layer(out,'collision_ceiling_elevation')[10,10] == pytest.approx(.75)
    for name,value in (('collision_height_contract',1.),('collision_height_filter_enabled',1.),
                       ('collision_vehicle_height_m',.65),('collision_headroom_margin_m',.10)):
        np.testing.assert_allclose(layer(out,name),value)
    assert bytes(node.latest_cloud.data)==original and raw.width==259


def test_required_memory_failure_cannot_publish_a_new_geometry_free_risk_map(node):
    from rclpy.parameter import Parameter
    node.set_parameters([Parameter('obstacle_memory_required', value=True)])
    with pytest.raises(ValueError, match='required mapper obstacle'):
        node._process_map(grid())
    node._obstacle_memory_callback(cloud([[.5,.5,.8]]))
    assert layer(node._process_map(grid())[1], 'collision_memory').sum() == 1
    with pytest.raises(ValueError, match='required mapper obstacle'):
        node._process_map(grid(12))
    assert node._obstacle_memory_rejections == 2


def test_large_mapper_snapshots_use_reliable_fragment_transport():
    from rclpy.qos import ReliabilityPolicy
    rclpy.init(args=['--ros-args', '-p', 'prediction_backend:=none', '-p', 'gpu_backend:=cpu',
                    '-p', 'obstacle_memory_topic:=/test/accumulated'])
    instance = FsgpBgkSe2Node()
    try:
        assert instance.obstacle_memory_sub.qos_profile.reliability == ReliabilityPolicy.RELIABLE
        assert instance.obstacle_memory_sub.qos_profile.depth == 3
    finally:
        instance.destroy_node()
        rclpy.shutdown()


def test_qualification_does_not_silently_enable_accumulated_dynamic_ghosts():
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'scripts/sim/checks'))
    from pipeline_smoke import parse_args
    assert parse_args([]).obstacle_memory_topic == 'none'
    assert parse_args(['--obstacle-memory-topic', '/Laser_map']).obstacle_memory_topic == '/Laser_map'


def test_none_is_a_disabled_subscription_not_a_relative_ros_topic():
    rclpy.init(args=['--ros-args', '-p', 'prediction_backend:=none', '-p', 'gpu_backend:=cpu',
                    '-p', 'obstacle_memory_topic:=none'])
    instance = FsgpBgkSe2Node()
    try:
        assert instance.obstacle_memory_sub is None
    finally:
        instance.destroy_node()
        rclpy.shutdown()


def test_timestamp_frame_rollback_and_expiry_do_not_relabel_old_evidence(node):
    node._obstacle_memory_callback(cloud([[.5,.5,.8]], sec=9))
    node._obstacle_memory_callback(cloud([[.5,.5,1.]], sec=11))
    points, age = node._memory_points(grid(10))
    assert age == 1 and points[0,2] == pytest.approx(.8)
    msg = grid(12)
    msg.info.header.stamp.nanosec = 250000000
    assert node._memory_points(msg)[0] is not None
    msg.info.header.stamp.nanosec += 1
    assert node._memory_points(msg)[0] is None
    node._obstacle_memory_callback(cloud([[.5,.5,.8]], sec=1, frame='other'))
    assert node._memory_points(grid(1))[0] is None
    node._obstacle_memory_callback(cloud([[.5,.5,.8]], sec=2))
    assert node._memory_points(grid(1))[0] is None
    node._cloud_callback(cloud([[0.,0.,0.]], sec=10))
    node._cloud_callback(cloud([[0.,0.,0.]], sec=1))
    assert not node._obstacle_memory_history


@pytest.mark.parametrize('backend', ['cpu', 'cupy'])
def test_single_thin_static_return_is_preserved_on_cpu_and_gpu(node, backend):
    from rclpy.parameter import Parameter
    if backend == 'cupy':
        cupy = pytest.importorskip('cupy')
        assert cupy.cuda.runtime.getDeviceCount() > 0
    node.set_parameters([Parameter('gpu_backend', value=backend)])
    from fsgp_bgk_se2.bgk_seb import BgkSebEstimator
    node.params.prefer_gpu = backend == 'cupy'
    node.params.gpu_backend = backend
    node.estimator = BgkSebEstimator(node.params)
    node._obstacle_memory_callback(cloud([[.5,.5,.8]]))
    result, out = node._process_map(grid())
    assert result.backend == backend
    assert layer(out, 'collision_memory').sum() == 1
    assert layer(out, 'collision_risk')[10,10] == 1
