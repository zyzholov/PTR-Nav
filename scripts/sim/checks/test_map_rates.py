import math
import struct

import pytest

from map_rates import ObservationWindow, map_stream_qualified, stream_statistics,serialized_header_stamp


@pytest.mark.parametrize('message_type',['header','imu','odom','cloud','grid'])
def test_raw_stamp_matches_actual_ros_serializer_for_every_observed_schema(message_type):
    from rclpy.serialization import serialize_message
    from std_msgs.msg import Header,Float32MultiArray
    from sensor_msgs.msg import Imu,PointCloud2
    from nav_msgs.msg import Odometry
    from se2_grid_msgs.msg import SE2Grid
    msg={'header':Header,'imu':Imu,'odom':Odometry,'cloud':PointCloud2,'grid':SE2Grid}[message_type]()
    header=msg if message_type=='header' else msg.info.header if message_type=='grid' else msg.header
    header.stamp.sec=123;header.stamp.nanosec=987654321;header.frame_id='地形/map'
    if message_type=='cloud':msg.data=bytes(2_000_000)
    if message_type=='grid':
        from array import array
        layer=Float32MultiArray();layer.data=array('f',[.5])*400_000;msg.data=[layer]
    assert serialized_header_stamp(serialize_message(msg))==(123,987654321)


def test_raw_big_endian_header_and_signed_seconds_are_not_native_endian_guesses():
    payload=b'\x00\x00\x00\x00'+struct.pack('>iII',-1,999999999,4)+b'map\x00'
    assert serialized_header_stamp(payload)==(-1,999999999)


@pytest.mark.parametrize('payload',[
    b'',b'\x00\x01'+bytes(13),
    b'\x00\x07\x00\x00'+struct.pack('<iII',1,2,4)+b'map\x00',
    b'\x00\x01\x00\x00'+struct.pack('<iII',1,1_000_000_000,4)+b'map\x00',
    b'\x00\x01\x00\x00'+struct.pack('<iII',1,2,5)+b'map\x00',
    b'\x00\x01\x00\x00'+struct.pack('<iII',1,2,4)+b'map!'])
def test_invalid_or_unsupported_raw_header_cannot_silently_be_counted(payload):
    with pytest.raises(ValueError):serialized_header_stamp(payload)


def samples(count=20, age=.1):
    return [(i*.1, i//10, (i%10)*100000000, age) for i in range(count)]


def test_map_requires_fresh_data_not_repeated_timestamps():
    stats = stream_statistics(samples(), 2.)
    assert map_stream_qualified(stats, 5, .35)
    repeated = [(i*.1, 0, 0, .1) for i in range(20)]
    assert not map_stream_qualified(stream_statistics(repeated, 2.), 5, .35)


def test_high_frequency_stale_map_is_not_fast_navigation():
    assert not map_stream_qualified(stream_statistics(samples(age=.4), 2.), 5, .35)
    assert not map_stream_qualified(stream_statistics(samples(age=math.nan), 2.), 5, .35)
    assert not map_stream_qualified(stream_statistics(samples(age=-.2), 2.), 5, .35)


@pytest.mark.parametrize('age,passed',[(.2999999,True),(.30,True),(.3000001,False),(.305,False),(.35,False)])
def test_current_gpu_limit_rejects_samples_that_legacy_350ms_would_accept(age,passed):
    from pipeline_smoke import motion_rate_limits
    rate,limit=motion_rate_limits('gpu')
    assert bool(map_stream_qualified(stream_statistics(samples(age=age),2.),rate,limit)) is passed


def test_4999hz_is_not_rounded_up_to_the_required_five_hz():
    rows=[(i/4.999,s,n,a)for i,(_,s,n,a)in enumerate(samples())]
    assert not map_stream_qualified(stream_statistics(rows,rows[-1][0]),5.,.30)


def test_holding_previous_map_can_be_older_than_every_arriving_message():
    stats=stream_statistics([(0.,10,0,.1),(.4,10,400000000,.1)],.4)
    assert stats['age_max']==.1
    assert stats['arrival_gap_max']==.4
    assert math.isclose(stats['held_age_before_update_max'],.5)


def test_empty_and_time_reset_fail():
    assert not map_stream_qualified(stream_statistics([], 2.), 5, .35)
    reset = samples()
    reset[-1] = (1.9, 0, 0, .1)
    assert not map_stream_qualified(stream_statistics(reset, 2.), 5, .35)


def test_discovery_barrier_retains_missing_clock_before_motion():
    window=ObservationWindow(("map","imu"))
    window.record("map",(0.,0,0,math.nan))
    window.record("imu",(.1,0,1,math.nan))
    assert window.armed_at is None
    window.record("map",(.2,1,0,.1))
    assert window.armed_at is None
    window.record("imu",(.3,1,1,.1))
    assert window.armed_at is None
    window.record("map",(.31,1,2,.1))
    window.record("imu",(.32,1,3,.1))
    assert window.armed_at==.32
    assert len(window.startup["map"])==3
    assert len(window.samples["map"])==1
    # Nothing after arming is discarded, including stale/future/clock-reset data.
    window.record("map",(.4,1,2,.5))
    window.record("map",(.5,1,3,math.nan))
    assert len(window.samples["map"])==3
    assert not map_stream_qualified(stream_statistics(window.samples["map"],.5),1,.35)


def test_discovery_is_fixed_sample_count_not_age_cherry_picking():
    window=ObservationWindow(("map",))
    window.record("map",(0.,1,0,.5))
    window.record("map",(.1,1,0,.6))
    assert window.armed_at is None
    window.record("map",(.2,1,1,.5))
    assert window.armed_at==.2
    assert not map_stream_qualified(stream_statistics(window.samples["map"],.2),1,.35)
    assert len(window.startup["map"])==3


def test_default_rviz_tracks_fastlio_body_and_keeps_world_overview():
    from pathlib import Path
    import yaml

    root = Path(__file__).resolve().parents[3]
    rviz = yaml.safe_load((root/'src/fsgp_bgk_se2/rviz/hunter_terrain_navigation.rviz').read_text())
    mapping = yaml.safe_load((root/'src/FAST_LIO/config/isaac_hunter_mid360_pointcloud.yaml').read_text())
    view = rviz['Visualization Manager']['Views']
    body = mapping['/**']['ros__parameters']['frame']['body_frame']
    assert view['Current']['Target Frame'] == body
    assert any(saved['Target Frame'] == 'map' for saved in view['Saved'])
