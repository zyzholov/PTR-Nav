import pytest
from rclpy.serialization import serialize_message,deserialize_message
from nav_msgs.msg import Odometry,Path,OccupancyGrid
from sensor_msgs.msg import PointCloud2
from se2_grid_msgs.msg import SE2Grid
from serialized_header import leading_stamp_ns


@pytest.mark.parametrize('kind',[Odometry,Path,OccupancyGrid,PointCloud2,SE2Grid])
def test_raw_diagnostic_stamp_matches_independent_ros_deserialization(kind):
    m=kind();header=m.info.header if kind is SE2Grid else m.header
    header.stamp.sec=123;header.stamp.nanosec=987654321;header.frame_id='map'
    payload=serialize_message(m);decoded=deserialize_message(payload,kind)
    h=decoded.info.header if kind is SE2Grid else decoded.header
    assert leading_stamp_ns(payload)==h.stamp.sec*10**9+h.stamp.nanosec


@pytest.mark.parametrize('payload',[b'',b'\x00\x06\x00\x00'+b'\x00'*8,
    b'\x00\x01\x00\x00'+b'\x00'*4+(10**9).to_bytes(4,'little')])
def test_raw_diagnostic_rejects_unknown_encoding_or_invalid_header(payload):
    with pytest.raises(ValueError):leading_stamp_ns(payload)
