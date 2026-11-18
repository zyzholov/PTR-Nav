import threading
import time
import math
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

rclpy = pytest.importorskip('rclpy')
from rclpy.executors import MultiThreadedExecutor,SingleThreadedExecutor
from sensor_msgs_py.point_cloud2 import create_cloud_xyz32
from se2_grid_msgs.msg import SE2Grid
from std_msgs.msg import Header

from fsgp_bgk_se2.fsgp_bgk_se2_node import FsgpBgkSe2Node
from fsgp_bgk_se2.se2_grid_tools import serialized_header


def wait_until(predicate, timeout=3.):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(.01)
    return False


@pytest.mark.parametrize('raw',[False,True])
@pytest.mark.parametrize('executor_mode',['thread_pool','single_ingress'])
def test_ingress_receives_latest_map_while_compute_is_busy_without_parallel_computes(raw,executor_mode):
    rclpy.init(args=['--ros-args', '-p', 'prediction_backend:=none', '-p', 'gpu_backend:=cpu',
                    '-p', 'publish_debug_clouds:=false','-p','raw_map_ingress:='+str(raw).lower(),
                    '-p','processing_executor:='+executor_mode])
    node = FsgpBgkSe2Node()
    feeder = rclpy.create_node('terrain_ingress_test_feeder')
    publisher = feeder.create_publisher(SE2Grid, node.input_fused_map_topic, 1)
    executor = (SingleThreadedExecutor() if executor_mode=='single_ingress' else MultiThreadedExecutor(num_threads=3))
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin)
    entered, release = threading.Event(), threading.Event()
    calls, active = [], []

    def process(msg):
        assert not active
        active.append(True)
        calls.append(msg.info.header.stamp.sec)
        if len(calls) == 1:
            entered.set()
            assert release.wait(timeout=4.)
        active.pop()
        return SimpleNamespace(backend='cpu', backend_reason='test', bgk_mean=np.zeros((2, 2)),
                               final_risk_slices=[np.zeros((2, 2))]), msg

    def message(stamp):
        msg = SE2Grid()
        msg.info.header.stamp.sec = stamp
        return msg

    def pending_stamp():
        value=node._pending_map
        return None if value is None else (serialized_header(value).stamp.sec if raw else value.info.header.stamp.sec)

    try:
        with patch.object(node, '_process_map', side_effect=process), patch.object(node, '_publish_runtime'):
            if executor_mode=='single_ingress':
                node.start_processing_worker()
                assert node.map_timer.is_canceled()
                with pytest.raises(RuntimeError,match='already started'):node.start_processing_worker()
            thread.start()
            assert wait_until(lambda: publisher.get_subscription_count() > 0)
            publisher.publish(message(1))
            assert entered.wait(timeout=3.)
            assert wait_until(lambda: time.monotonic() >= node._next_process_wall)
            publisher.publish(message(2))
            assert wait_until(lambda: pending_stamp()==2)
            publisher.publish(message(3))
            assert wait_until(lambda: pending_stamp()==3)
            assert calls == [1]
            release.set()
            assert wait_until(lambda: calls == [1, 3])
    finally:
        release.set()
        executor.shutdown(timeout_sec=5.)
        if thread.is_alive():
            thread.join(timeout=5.)
        node.destroy_node()
        feeder.destroy_node()
        rclpy.shutdown()


def test_rate_gate_waits_for_a_new_arrival_instead_of_aging_an_early_frame():
    rclpy.init(args=['--ros-args', '-p', 'prediction_backend:=none', '-p', 'gpu_backend:=cpu'])
    node = FsgpBgkSe2Node()
    first, latest = SE2Grid(), SE2Grid()
    node._next_process_wall = 10.125
    try:
        with patch('fsgp_bgk_se2.fsgp_bgk_se2_node.time.monotonic', return_value=10.1):
            node._map_callback(first)
        with patch.object(node, '_process_map', side_effect=RuntimeError('test sentinel')) as process:
            with patch('fsgp_bgk_se2.fsgp_bgk_se2_node.time.monotonic', return_value=10.13):
                node._map_timer_callback()
            process.assert_not_called()
            with patch('fsgp_bgk_se2.fsgp_bgk_se2_node.time.monotonic', return_value=10.2):
                node._map_callback(latest)
                node._map_timer_callback()
            process.assert_called_once_with(latest)
            assert node._pending_map is None
    finally:
        node.destroy_node()
        rclpy.shutdown()


@pytest.mark.parametrize('age,accepted',[(.10,True),(.161,False),(float('nan'),False),(-.2,False)])
@pytest.mark.parametrize('raw',[False,True])
def test_age_bounded_latest_frame_can_arrive_during_compute_without_weakening_deadline(age,accepted,raw):
    from rclpy.serialization import serialize_message
    from fsgp_bgk_se2.grid_cdr import deserialize_grid
    rclpy.init(args=['--ros-args','-p','prediction_backend:=none','-p','gpu_backend:=cpu',
        '-p','publish_debug_clouds:=false','-p','max_input_source_age_sec:=0.16',
        '-p','max_output_source_age_sec:=0.30'])
    node=FsgpBgkSe2Node();msg=SE2Grid();msg.info.header.stamp.sec=10
    node._next_process_wall=10.125
    result=SimpleNamespace(backend='cpu',backend_reason='test',bgk_mean=np.zeros((2,2)),
                          final_risk_slices=[np.zeros((2,2))])
    try:
        with patch('fsgp_bgk_se2.fsgp_bgk_se2_node.time.monotonic',return_value=10.1):
            node._map_callback(serialize_message(msg) if raw else msg)
        with patch.object(node,'_source_age',return_value=age),patch.object(node,'_publish_runtime'), \
             patch.object(node,'_process_map',return_value=(result,msg)) as process, \
             patch.object(node.map_pub,'publish') as publish, \
             patch('fsgp_bgk_se2.fsgp_bgk_se2_node.deserialize_grid',wraps=deserialize_grid) as decode, \
             patch('fsgp_bgk_se2.fsgp_bgk_se2_node.time.monotonic',return_value=10.13):
            node._map_timer_callback()
        assert decode.call_count==int(raw and accepted)
        assert process.call_count==int(accepted) and publish.call_count==int(accepted)
        if accepted:assert process.call_args.args[0]==msg
        assert node._pending_map is None
        assert msg.info.header.stamp.sec==10
        assert node._deadline_input_drops==int(not accepted)
    finally:node.destroy_node();rclpy.shutdown()


def test_cloud_window_uses_latest_nonfuture_same_frame_and_clears_on_time_reset():
    from builtin_interfaces.msg import Time
    rclpy.init(args=['--ros-args', '-p', 'prediction_backend:=none', '-p', 'gpu_backend:=cpu'])
    node = FsgpBgkSe2Node()
    try:
        for sec, height in ((10, 1.), (11, 2.)):
            node._cloud_callback(create_cloud_xyz32(Header(frame_id='map', stamp=Time(sec=sec)),
                                                   [[0., 0., height]]))
        msg = SE2Grid()
        msg.info.header.frame_id = 'map'
        msg.info.header.stamp.sec = 10
        points, age = node._current_points(msg)
        assert age == 0 and points[0, 2] == 1
        msg.info.header.stamp = Time(sec=11, nanosec=100000000)
        points, age = node._current_points(msg)
        assert age == pytest.approx(.1) and points[0, 2] == 2
        msg.info.header.frame_id = 'other'
        assert node._current_points(msg)[0] is None
        node._cloud_callback(create_cloud_xyz32(Header(frame_id='map', stamp=Time(sec=1)),
                                               [[0., 0., 3.]]))
        assert len(node._cloud_history) == 1
        msg.info.header.frame_id = 'map'
        msg.info.header.stamp = Time(sec=1)
        assert node._current_points(msg)[0][0, 2] == 3
        msg.info.header.stamp = Time(sec=2)
        assert node._current_points(msg)[0] is None
    finally:
        node.destroy_node()
        rclpy.shutdown()


@pytest.mark.parametrize('age,limit,expired', [
    (.12,.12,False), (.121,.12,True), (.08,.12,False),
    (float('nan'),.12,True), (-.12,.12,True), (0.,.12,False),
    (100.,0.,False)])
def test_source_deadline_preserves_time_and_rejects_missing_or_reset_clock(age,limit,expired):
    assert FsgpBgkSe2Node._age_expired(age,limit) is expired


@pytest.mark.parametrize('simulation',[False,True])
def test_independent_simulation_sources_can_only_tighten_a_lagged_clock_deadline(simulation):
    import threading
    from types import SimpleNamespace as S
    from builtin_interfaces.msg import Time
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import PointCloud2
    def stamp(value):return Time(sec=int(value),nanosec=round((value-int(value))*1e9))
    odom=Odometry();odom.header.frame_id='map';odom.header.stamp=stamp(10.4)
    cloud=PointCloud2();cloud.header.frame_id='map';cloud.header.stamp=stamp(10.3)
    fake=S(get_clock=lambda:S(now=lambda:S(nanoseconds=10_000_000_000)),
        get_parameter=lambda _:S(value=simulation),_stamp_ns=FsgpBgkSe2Node._stamp_ns,
        _input_lock=threading.Lock(),latest_odom=odom,latest_cloud=cloud,
        _reference_cloud_history=[],frame_id='map')
    actual=FsgpBgkSe2Node._source_age(fake,stamp(9.9))
    assert actual==pytest.approx(.5 if simulation else .1)
    assert FsgpBgkSe2Node._age_expired(actual,.3) is simulation
    # A mismatched-frame/misclocked source supplies no timing witness.
    odom.header.frame_id=cloud.header.frame_id='other'
    assert FsgpBgkSe2Node._source_age(fake,stamp(9.9))==pytest.approx(.1)
    # If every reference is old, use /clock; never make an input younger.
    odom.header.frame_id='map';odom.header.stamp=stamp(9.8)
    assert FsgpBgkSe2Node._source_age(fake,stamp(9.9))==pytest.approx(.1)
    # Nor can a future source validate a future input or missing reset clock.
    odom.header.stamp=stamp(12.)
    assert FsgpBgkSe2Node._source_age(fake,stamp(11.))==pytest.approx(-1.)
    fake.get_clock=lambda:S(now=lambda:S(nanoseconds=0))
    assert math.isnan(FsgpBgkSe2Node._source_age(fake,stamp(11.)))


def test_deadline_drops_are_explicit_without_publishing_stale_maps_or_consuming_ingress_slot():
    rclpy.init(args=['--ros-args','-p','prediction_backend:=none','-p','gpu_backend:=cpu',
                    '-p','max_update_rate_hz:=0.0','-p','max_input_source_age_sec:=0.12',
                    '-p','max_output_source_age_sec:=0.30'])
    node=FsgpBgkSe2Node();msg=SE2Grid();msg.info.header.stamp.sec=10
    result=SimpleNamespace(backend='cpu',backend_reason='test',bgk_mean=np.zeros((2,2)),
                           final_risk_slices=[np.zeros((2,2))])
    try:
        with patch.object(node,'_process_map',return_value=(result,msg)) as process, \
             patch.object(node,'_publish_runtime') as report, \
             patch.object(node.map_pub,'publish') as publish:
            with patch.object(node,'_source_age',return_value=.2):
                node._map_callback(msg);node._map_timer_callback()
            process.assert_not_called();publish.assert_not_called()
            assert node._deadline_input_drops==1 and node._next_process_wall==0.
            assert report.call_args.kwargs==dict(published=False,drop_reason='input deadline')
            # Input, processed, egress admission, then EXACT encoded payload.
            with patch.object(node,'_source_age',side_effect=[.08,.20,.20,.31]):
                node._map_callback(msg);node._map_timer_callback()
            process.assert_called_once_with(msg);publish.assert_not_called()
            assert node._deadline_output_drops==1
            assert report.call_args.kwargs['published'] is False
            assert report.call_args.kwargs['drop_reason']=='output deadline'
            assert report.call_args.kwargs['stages']['egress_encode']>=0
            with patch.object(node,'_source_age',return_value=.08):
                node._map_callback(msg);node._map_timer_callback()
            from rclpy.serialization import deserialize_message
            publish.assert_called_once()
            assert deserialize_message(publish.call_args.args[0],SE2Grid)==msg
            assert msg.info.header.stamp.sec==10
            assert node._deadline_input_drops==node._deadline_output_drops==1
    finally:
        node.destroy_node();rclpy.shutdown()


@pytest.mark.parametrize('limit,reserve,result',[(.3,.05,.25),(.3,0.,.3),(0.,0.,0.)])
def test_delivery_reserve_is_inside_the_original_deadline(limit,reserve,result):
    assert FsgpBgkSe2Node._output_admission_deadline(limit,reserve)==pytest.approx(result)


@pytest.mark.parametrize('limit,reserve',[(0.,.01),(.3,.3),(.3,.31),(.3,-.1),(.3,float('nan')),(float('inf'),0.)])
def test_invalid_delivery_reserve_cannot_disable_or_expand_a_deadline(limit,reserve):
    with pytest.raises(ValueError):FsgpBgkSe2Node._output_admission_deadline(limit,reserve)


@pytest.mark.parametrize('output_age,accepted',[(.24,True),(.251,False),(.29,False),(.31,False)])
def test_result_with_insufficient_transport_headroom_is_dropped_without_restamping(output_age,accepted):
    rclpy.init(args=['--ros-args','-p','prediction_backend:=none','-p','gpu_backend:=cpu',
        '-p','max_update_rate_hz:=0.0','-p','max_input_source_age_sec:=0.08',
        '-p','max_output_source_age_sec:=0.30','-p','output_delivery_reserve_sec:=0.05'])
    node=FsgpBgkSe2Node();msg=SE2Grid();msg.info.header.stamp.sec=10
    result=SimpleNamespace(backend='cpu',backend_reason='test',bgk_mean=np.zeros((2,2)),
                          final_risk_slices=[np.zeros((2,2))])
    try:
        with patch.object(node,'_process_map',return_value=(result,msg)),patch.object(node,'_publish_runtime'), \
             patch.object(node.map_pub,'publish') as publish, \
             patch.object(node,'_source_age',side_effect=[.05,output_age,output_age,output_age]):
            node._map_callback(msg);node._map_timer_callback()
        assert publish.call_count==int(accepted)
        assert node._deadline_output_drops==int(not accepted)
        assert msg.info.header.stamp.sec==10
        assert node.max_output_source_age_sec==.30
    finally:node.destroy_node();rclpy.shutdown()


def test_last_deadline_check_includes_serialization_and_drops_an_expired_payload():
    from rclpy.serialization import serialize_message
    rclpy.init(args=['--ros-args','-p','prediction_backend:=none','-p','gpu_backend:=cpu',
        '-p','publish_debug_clouds:=false','-p','max_update_rate_hz:=0.0',
        '-p','max_input_source_age_sec:=0.08','-p','max_output_source_age_sec:=0.30',
        '-p','output_delivery_reserve_sec:=0.015'])
    node=FsgpBgkSe2Node();msg=SE2Grid();msg.info.header.stamp.sec=10;age=[.02]
    result=SimpleNamespace(backend='cpu',backend_reason='test',bgk_mean=np.zeros((2,2)),
                          final_risk_slices=[np.zeros((2,2))])
    def encode(m):
        age[0]=.296  # Computation was timely, serialization consumed the budget.
        return serialize_message(m)
    try:
        with patch.object(node,'_process_map',return_value=(result,msg)),patch.object(node,'_publish_runtime') as report, \
             patch.object(node,'_source_age',side_effect=lambda _:age[0]), \
             patch('fsgp_bgk_se2.fsgp_bgk_se2_node.serialize_grid',side_effect=encode) as encoder, \
             patch.object(node.map_pub,'publish') as publish:
            node._map_callback(msg);node._map_timer_callback()
        encoder.assert_called_once_with(msg);publish.assert_not_called()
        assert node._deadline_output_drops==1 and msg.info.header.stamp.sec==10
        # Egress has its own immutable job-stage snapshot; the next compute
        # may already own _process_stage_ms on the separate worker thread.
        assert 'egress_encode' in report.call_args.kwargs['stages']
    finally:node.destroy_node();rclpy.shutdown()
