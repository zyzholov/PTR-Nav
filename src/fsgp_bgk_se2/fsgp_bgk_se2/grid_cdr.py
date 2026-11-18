"""CDR1 encoding for the installed SE2Grid IDL, using primitive buffers.

Produces an ordinary ROS serialized message, not a private transport format.
Alignment is relative to the payload AFTER the four-byte encapsulation.
Unknown schemas use the standard ROS encoder; no fields may be omitted.
"""
from array import array
from functools import lru_cache
import struct
import sys

from rclpy.serialization import serialize_message,deserialize_message
from se2_grid_msgs.msg import SE2Grid,SE2GridInfo
from geometry_msgs.msg import Pose,Point,Quaternion
from std_msgs.msg import Header,Float32MultiArray,MultiArrayLayout,MultiArrayDimension
from builtin_interfaces.msg import Time


@lru_cache(maxsize=1)
def supported_schema():
    # Exact primitive types are part of the wire contract too. A future IDL
    # revision must not silently select this specialized encoder.
    schemas={
        SE2Grid:dict(info='se2_grid_msgs/SE2GridInfo',layers='sequence<string>',has_so2='sequence<boolean>',
                     data='sequence<std_msgs/Float32MultiArray>',start_index_row='uint16',start_index_col='uint16'),
        SE2GridInfo:dict(header='std_msgs/Header',pos_resolution='double',yaw_resolution='double',
                        length_x='double',length_y='double',length_yaw='double',pose='geometry_msgs/Pose'),
        Header:dict(stamp='builtin_interfaces/Time',frame_id='string'),Time:dict(sec='int32',nanosec='uint32'),
        Pose:dict(position='geometry_msgs/Point',orientation='geometry_msgs/Quaternion'),
        Point:dict(x='double',y='double',z='double'),Quaternion:dict(x='double',y='double',z='double',w='double'),
        Float32MultiArray:dict(layout='std_msgs/MultiArrayLayout',data='sequence<float>'),
        MultiArrayLayout:dict(dim='sequence<std_msgs/MultiArrayDimension>',data_offset='uint32'),
        MultiArrayDimension:dict(label='string',size='uint32',stride='uint32')}
    return all(list(t.get_fields_and_field_types().items())==list(fields.items())for t,fields in schemas.items())


@lru_cache(maxsize=128)
def _array_header(dimensions,offset,count,endian):
    """Immutable CDR metadata only; primitive buffers are never cached.

    Every Float32MultiArray begins on a four-byte boundary. Its strings,
    sizes, offset and count also have four-byte alignment, so the same
    relative header can be reused at any array position in the message.
    """
    parts=[struct.pack(endian+'I',len(dimensions))]
    for label,size,stride in dimensions:
        encoded=label.encode('utf-8')+b'\0'
        parts.extend((struct.pack(endian+'I',len(encoded)),encoded,
                      b'\0'*((-len(encoded))%4),struct.pack(endian+'II',size,stride)))
    parts.append(struct.pack(endian+'II',offset,count))
    return b''.join(parts)


def serialize_grid(msg,*,byte_order='little'):
    if type(msg) is not SE2Grid or not supported_schema():return serialize_message(msg)
    if byte_order not in ('little','big'):raise ValueError('CDR byte order must be little or big')
    endian='<' if byte_order=='little' else '>'
    # Join immutable metadata and primitive views ONCE. Repeatedly extending
    # a multi-megabyte bytearray can reallocate/copy the map, then bytes()
    # copies it again. The final joined bytes owns all data independently.
    parts=[b'\0\1\0\0' if byte_order=='little' else b'\0\0\0\0'];byte_offset=4
    def append(value):
        nonlocal byte_offset
        parts.append(value);byte_offset+=len(value)
    def align(n):
        padding=(4-byte_offset)%n
        if padding:append(b'\0'*padding)
    def pack(fmt,*values,alignment=4):
        align(alignment);append(struct.pack(endian+fmt,*values))
    def string(value):
        payload=value.encode('utf-8')+b'\0'
        pack('I',len(payload));append(payload)
    info=msg.info;stamp=info.header.stamp
    pack('iI',stamp.sec,stamp.nanosec);string(info.header.frame_id)
    position,orientation=info.pose.position,info.pose.orientation
    pack('12d',info.pos_resolution,info.yaw_resolution,info.length_x,info.length_y,info.length_yaw,
         position.x,position.y,position.z,orientation.x,orientation.y,orientation.z,orientation.w,alignment=8)
    pack('I',len(msg.layers))
    for name in msg.layers:string(name)
    pack('I',len(msg.has_so2));append(bytes(msg.has_so2))
    pack('I',len(msg.data))
    for value in msg.data:
        dimensions=tuple((d.label,d.size,d.stride)for d in value.layout.dim)
        # Keep the cache bounded to small normal map layouts, including the
        # source offset/count in its key. Exotic layouts still encode fully.
        header=_array_header if len(dimensions)<=4 and all(len(d[0])<=64 for d in dimensions) else _array_header.__wrapped__
        align(4);append(header(dimensions,value.layout.data_offset,len(value.data),endian))
        # Standard generated ROS setters own an array('f'). Preserve its
        # exact float bits (including NaNs), dimensions, offset and ordering.
        values=value.data
        if not isinstance(values,array) or values.typecode!='f' or values.itemsize!=4:
            return serialize_message(msg)
        if byte_order!=sys.byteorder:
            values=array('f',values);values.byteswap()
        append(memoryview(values).cast('B'))
    pack('HH',msg.start_index_row,msg.start_index_col,alignment=2)
    return b''.join(parts)


def deserialize_grid(payload):
    """Decode the exact CDR1 schema with owned primitive arrays.

    No private ROS attributes or alternate wire format. Unsupported schemas
    and encapsulations use ROS; malformed supported CDR is rejected. Bounds
    are checked BEFORE allocation/iteration so forged lengths cannot create
    an unbounded Python sequence. Array float bits are copied, never cast.
    """
    data=memoryview(payload).cast('B')
    if len(data)<4:raise ValueError('truncated CDR encapsulation')
    if not supported_schema() or bytes(data[:4]) not in (b'\0\1\0\0',b'\0\0\0\0'):
        return deserialize_message(payload,SE2Grid)
    endian='<' if data[1]==1 else '>';position=4

    def take(length):
        nonlocal position
        if length<0 or length>len(data)-position:raise ValueError('truncated SE2Grid CDR')
        start=position;position+=length;return data[start:position]

    def unpack(fmt,alignment=4):
        nonlocal position
        take((4-position)%alignment)
        size=struct.calcsize(endian+fmt)
        return struct.unpack(endian+fmt,take(size))

    def count(minimum=1):
        value=unpack('I')[0]
        if value>(len(data)-position)//minimum:raise ValueError('invalid SE2Grid CDR sequence length')
        return value

    def string():
        size=count()
        if not size:raise ValueError('CDR string lacks terminator')
        value=take(size)
        if value[-1]!=0:raise ValueError('CDR string lacks terminator')
        return bytes(value[:-1]).decode('utf-8')

    result=SE2Grid();info=result.info
    sec,nanosec=unpack('iI');info.header.stamp.sec=sec;info.header.stamp.nanosec=nanosec
    info.header.frame_id=string()
    values=unpack('12d',8)
    for name,value in zip(('pos_resolution','yaw_resolution','length_x','length_y','length_yaw'),values[:5]):
        setattr(info,name,value)
    for name,value in zip(('x','y','z'),values[5:8]):setattr(info.pose.position,name,value)
    for name,value in zip(('x','y','z','w'),values[8:]):setattr(info.pose.orientation,name,value)
    result.layers=[string() for _ in range(count(5))]
    boolean_values=take(count())
    if any(v>1 for v in boolean_values):raise ValueError('invalid CDR boolean')
    result.has_so2=[bool(v) for v in boolean_values]
    arrays=[]
    for _ in range(count(12)):
        value=Float32MultiArray();dimensions=[]
        for _ in range(count(16)):
            label=string();size,stride=unpack('II')
            dimensions.append(MultiArrayDimension(label=label,size=size,stride=stride))
        value.layout.dim=dimensions
        offset=unpack('I')[0];length=count(4)
        elements=array('f');elements.frombytes(take(length*4))
        if (endian=='<')!=(sys.byteorder=='little'):elements.byteswap()
        value.layout.data_offset=offset;value.data=elements;arrays.append(value)
    result.data=arrays
    result.start_index_row,result.start_index_col=unpack('HH',2)
    if position!=len(data):raise ValueError('trailing SE2Grid CDR data')
    return result
