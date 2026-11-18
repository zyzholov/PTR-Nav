"""Wire interoperability against ROS's actual CDR encoder/decoder and RMW."""
from array import array
import struct

import numpy as np
import pytest
from rclpy.serialization import serialize_message,deserialize_message
from se2_grid_msgs.msg import SE2Grid
from std_msgs.msg import Float32MultiArray,MultiArrayDimension
from fsgp_bgk_se2.grid_cdr import serialize_grid,deserialize_grid,supported_schema


def message(seed):
    rng=np.random.default_rng(seed);m=SE2Grid()
    m.info.header.frame_id=('map','测量坐标系','ab','abcde','')[seed%5]
    m.info.header.stamp.sec=-123;m.info.header.stamp.nanosec=123456789
    for name,value in zip(('pos_resolution','yaw_resolution','length_x','length_y','length_yaw'),rng.random(5)):
        setattr(m.info,name,float(value))
    for name,value in zip(('x','y','z'),rng.standard_normal(3)):setattr(m.info.pose.position,name,float(value))
    for name,value in zip(('x','y','z','w'),rng.standard_normal(4)):setattr(m.info.pose.orientation,name,float(value))
    m.layers=['elevation','risk','中文高程'][:seed%4];m.has_so2=[True,False,True][:seed%4]
    m.start_index_row=65535;m.start_index_col=17
    for i in range(seed%7):
        a=Float32MultiArray()
        a.layout.dim=[MultiArrayDimension(label='column_index'+'x'*i,size=3,stride=6),
                      MultiArrayDimension(label='row_index',size=2,stride=2)][:i%3]
        a.layout.data_offset=i
        a.data=array('f',rng.standard_normal(i*17).astype(np.float32))
        if len(a.data)>=6:
            # Include a noncanonical NaN payload, signed zero, subnormal and infinities.
            a.data[:6]=array('f',np.array([0x7fc01234,0x80000000,1,0x7f800000,0xff800000,0],np.uint32).view(np.float32))
        m.data.append(a)
    return m


def require_equal(a,b):
    assert a.info==b.info and a.layers==b.layers
    np.testing.assert_array_equal(a.has_so2,b.has_so2)
    assert (a.start_index_row,a.start_index_col)==(b.start_index_row,b.start_index_col)
    assert len(a.data)==len(b.data)
    for aa,bb in zip(a.data,b.data):
        assert aa.layout==bb.layout
        assert aa.data.tobytes()==bb.data.tobytes()


@pytest.mark.parametrize('seed',range(35))
@pytest.mark.parametrize('order',['little','big'])
def test_standard_ros_decoder_accepts_both_endiannesses_and_all_nested_values(seed,order):
    assert supported_schema(),'specialized path must actually run in this test'
    m=message(seed);payload=serialize_grid(m,byte_order=order)
    require_equal(m,deserialize_message(payload,SE2Grid))
    require_equal(m,deserialize_grid(payload))
    reference=serialize_message(m)
    require_equal(m,deserialize_grid(reference))
    assert len(payload)==len(reference)
    if order=='little':
        # Padding bytes are not meaningful, so compare ROS's canonical
        # re-encoding as well as every field and primitive's original bits.
        assert serialize_message(deserialize_message(payload,SE2Grid))==reference


def test_empty_grid_and_unrecognized_schema_fall_back_without_dropping_fields(monkeypatch):
    m=SE2Grid();require_equal(m,deserialize_message(serialize_grid(m),SE2Grid))
    monkeypatch.setattr('fsgp_bgk_se2.grid_cdr.supported_schema',lambda:False)
    assert serialize_grid(message(13))==serialize_message(message(13))
    with pytest.raises(ValueError):
        # Test the explicit byte-order guard on a supported schema.
        monkeypatch.setattr('fsgp_bgk_se2.grid_cdr.supported_schema',lambda:True)
        serialize_grid(m,byte_order='unknown')


@pytest.mark.parametrize('order',['little','big'])
def test_reused_headers_follow_mutated_dimensions_offsets_lengths_and_data_bits(order):
    from fsgp_bgk_se2.grid_cdr import _array_header
    _array_header.cache_clear()
    m=message(13)
    for i in range(4):
        # Repeated equal layouts exercise hits; mutation of any header field
        # must select its new encoding. Primitive data must NEVER be cached.
        for _ in range(2):require_equal(m,deserialize_message(serialize_grid(m,byte_order=order),SE2Grid))
        for a in m.data:
            a.layout.data_offset+=1
            for dim in a.layout.dim:
                dim.label+='地';dim.size+=1;dim.stride+=3
            a.data.append(float(i));a.data[0]=-float(i+1)
    assert _array_header.cache_info().hits>0
    # Large/exotic layouts are encoded normally without retaining cache keys.
    serialize_grid(m,byte_order=order)  # Warm the last mutations of OTHER arrays.
    a=m.data[0];a.layout.dim=[MultiArrayDimension(label='界'*80,size=1,stride=1)]*5
    before=_array_header.cache_info().currsize
    require_equal(m,deserialize_message(serialize_grid(m,byte_order=order),SE2Grid))
    assert _array_header.cache_info().currsize==before


def test_specialized_decoder_owns_arrays_and_never_aliases_or_caches_payload():
    payload=bytearray(serialize_grid(message(13)))
    decoded=deserialize_grid(payload);reference=message(13)
    payload[:]=b'\xff'*len(payload)
    require_equal(decoded,reference)
    another=deserialize_grid(serialize_message(message(13)))
    decoded.data[1].data[0]=123.
    require_equal(another,reference)


def test_decoder_rejects_all_truncations_and_impossible_counts_without_allocating():
    payload=serialize_grid(message(13))
    # Every nested field boundary and primitive-array truncation is covered.
    for stop in range(len(payload)):
        with pytest.raises(Exception):deserialize_grid(payload[:stop])
    with pytest.raises(ValueError):deserialize_grid(payload+b'\0')
    # First string length follows the encapsulation and source timestamp.
    broken=bytearray(payload);struct.pack_into('<I',broken,12,0xffffffff)
    with pytest.raises(ValueError):deserialize_grid(broken)
    broken=bytearray(payload);struct.pack_into('<I',broken,12,0)
    with pytest.raises(ValueError):deserialize_grid(broken)
    broken=bytearray(payload);length=struct.unpack_from('<I',broken,12)[0];broken[16+length-1]=1
    with pytest.raises(ValueError):deserialize_grid(broken)


def test_unknown_decoder_schema_uses_standard_ros_without_losing_fields(monkeypatch):
    monkeypatch.setattr('fsgp_bgk_se2.grid_cdr.supported_schema',lambda:False)
    require_equal(deserialize_grid(serialize_message(message(13))),message(13))
