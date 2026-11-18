"""Publication must preserve received evidence and avoid duplicate buffers."""
from types import SimpleNamespace
import numpy as np
import pytest
from rclpy.serialization import serialize_message,deserialize_message
from se2_grid_msgs.msg import SE2Grid
from fsgp_bgk_se2.fsgp_bgk_se2_node import FsgpBgkSe2Node
from fsgp_bgk_se2.se2_grid_tools import matrix_to_multiarray_like,layer_refs,multiarray_to_matrix
from fsgp_bgk_se2.grid_cdr import serialize_grid


@pytest.mark.parametrize('value',[0.,-0.,1.,np.nan,np.inf,-np.inf,.03])
@pytest.mark.parametrize('row_major',[False,True])
def test_constant_metadata_skips_spatial_work_but_preserves_wire_bits_and_ownership(value,row_major):
    from fsgp_bgk_se2.se2_grid_tools import world_to_grid_buffer
    msg=SE2Grid();msg.start_index_row=3;msg.start_index_col=2
    world=np.full((5,7),value,np.float32)
    template=matrix_to_multiarray_like(np.zeros((7,5),np.float32),None)
    if row_major:template.layout.dim=template.layout.dim[::-1];template.layout.dim[0].label='row_index'
    old=SE2Grid();new=SE2Grid()
    owner=SimpleNamespace(_append_layer=lambda *args:FsgpBgkSe2Node._append_layer(None,*args))
    FsgpBgkSe2Node._append_layer(None,old,'scalar',(False,[world_to_grid_buffer(world,msg)]),template)
    FsgpBgkSe2Node._append_constant_layer(owner,new,'scalar',value,world.shape,template)
    assert serialize_message(new)==serialize_message(old)
    second=SE2Grid();FsgpBgkSe2Node._append_constant_layer(owner,second,'scalar',value,world.shape,template)
    assert new.data[0].data is not second.data[0].data
    original=serialize_message(second);new.data[0].data[0]=123.
    assert serialize_message(second)==original


@pytest.mark.parametrize('encoder',[serialize_message,serialize_grid])
def test_serialized_ros_publication_preserves_exact_payload_independent_of_sender_buffers(encoder):
    import rclpy,time
    rclpy.init();sender=rclpy.create_node('serialized_map_sender');receiver=rclpy.create_node('serialized_map_receiver')
    received=[];topic='/qualification/serialized_terrain'
    pub=sender.create_publisher(SE2Grid,topic,1)
    receiver.create_subscription(SE2Grid,topic,received.append,1)
    m=SE2Grid();m.info.header.stamp.sec=12;m.info.header.frame_id='map'
    m.start_index_row=3;m.start_index_col=7;m.layers=['height'];m.has_so2=[False]
    values=np.arange(6400,dtype=np.float32).reshape(80,80);values[4,5]=np.nan
    m.data=[matrix_to_multiarray_like(values,None)];payload=encoder(m)
    m.data[0].data[0]=-999.  # The wire payload must already own the original.
    try:
        deadline=time.monotonic()+5
        while not received and time.monotonic()<deadline:
            pub.publish(payload);rclpy.spin_once(receiver,timeout_sec=.05)
        assert received and serialize_message(received[-1])==payload
        assert received[-1].info.header.stamp.sec==12
    finally:sender.destroy_node();receiver.destroy_node();rclpy.shutdown()


def test_schema_cache_never_reuses_buffers_and_tracks_yaw_append_and_reorder_changes():
    def message():
        m=SE2Grid();m.layers=['height','risk'];m.has_so2=[False,True]
        m.info.yaw_resolution=np.pi;m.info.length_yaw=2*np.pi;return m
    a,b=message(),message();first=layer_refs(a)
    assert layer_refs(b) is first and first['risk'].data_indices==(1,2)
    with pytest.raises(TypeError):first['height']=first['risk']
    a.layers.append('variance');a.has_so2.append(False)
    assert layer_refs(a)['variance'].data_indices==(3,)
    a.info.yaw_resolution=np.pi/2
    assert layer_refs(a)['variance'].data_indices==(5,)
    b.layers=['risk','height'];b.has_so2=[True,False]
    assert layer_refs(b)['height'].data_indices==(2,)
    assert first['height'].data_indices==(0,)


def test_packing_new_array_does_not_mutate_received_layout_offset():
    template=matrix_to_multiarray_like(np.ones((3,2),np.float32),None)
    template.layout.data_offset=7
    before=serialize_message(template)
    output=matrix_to_multiarray_like(np.full((3,2),2.,np.float32),template)
    assert output.layout is not template.layout and output.layout.data_offset==0
    assert serialize_message(template)==before
    assert np.all(multiarray_to_matrix(output)==2.)


@pytest.mark.parametrize('row_major',[False,True])
@pytest.mark.parametrize('view',['normal','sliced','transpose','big_endian'])
def test_packed_float_buffer_keeps_storage_bits_and_owns_its_data(row_major,view):
    from fsgp_bgk_se2.se2_grid_tools import ROW_LABEL
    base=np.array([0x7fc01234,0x80000000,1,0x7f800000,0xff800000,0],np.uint32).view(np.float32).reshape(2,3)
    matrix={'normal':base,'sliced':base[:,::-1],'transpose':base.T,
            'big_endian':base.astype('>f4')}[view]
    template=matrix_to_multiarray_like(np.zeros(matrix.shape,np.float32),None)
    if row_major:template.layout.dim=template.layout.dim[::-1];template.layout.dim[0].label=ROW_LABEL
    result=matrix_to_multiarray_like(matrix,template)
    order='C' if row_major else 'F'
    expected=np.asarray(matrix,np.float32).tobytes(order=order)
    assert result.data.tobytes()==expected
    matrix.flat[0]=123.
    assert result.data.tobytes()==expected


def test_unchanged_map_evidence_is_shared_read_only_and_serializes_independently():
    msg=SE2Grid();msg.info.pos_resolution=.2;msg.info.length_x=.6;msg.info.length_y=.4
    msg.info.yaw_resolution=np.pi;msg.info.length_yaw=2*np.pi
    msg.info.header.stamp.sec=12;msg.info.header.frame_id='map'
    template=matrix_to_multiarray_like(np.arange(6,dtype=np.float32).reshape(3,2),None)
    msg.layers=['elevation','ground_observed','risk'];msg.has_so2=[False,False,True]
    msg.data=[template,matrix_to_multiarray_like(np.ones((3,2),np.float32),None),
              matrix_to_multiarray_like(np.zeros((3,2),np.float32),None),
              matrix_to_multiarray_like(np.ones((3,2),np.float32),None)]
    before=serialize_message(msg);scalar=np.zeros((2,3),np.float32)
    result=SimpleNamespace(final_risk_slices=[scalar.copy(),scalar.copy()],baseline_risk=scalar,
        bgk_mean=scalar,bgk_var=scalar,bgk_traversability=scalar)
    node=SimpleNamespace(include_diagnostic_layers=False,frame_id='')
    node._append_layer=lambda *args:FsgpBgkSe2Node._append_layer(node,*args)
    output=FsgpBgkSe2Node._build_output_message(node,msg,result,scalar,template)
    refs=layer_refs(output)
    assert output.data[refs['elevation'].data_indices[0]] is msg.data[0]
    assert output.data[refs['ground_observed'].data_indices[0]] is msg.data[1]
    assert output.info is not msg.info and serialize_message(msg)==before
    # ROS serialization owns the resulting bytes; future input lifetimes do
    # not affect the delivered map. Risk overrides still get fresh buffers.
    restored=deserialize_message(serialize_message(output),SE2Grid)
    for name in ('elevation','ground_observed'):
        a=refs[name].data_indices[0]
        assert np.array_equal(multiarray_to_matrix(restored.data[a]),multiarray_to_matrix(output.data[a]))
    assert all(output.data[i] is not msg.data[2] for i in refs['risk'].data_indices)


def test_packed_array_fast_and_schema_fallback_preserve_layout_and_all_float_bits(monkeypatch):
    import fsgp_bgk_se2.se2_grid_tools as grid
    from std_msgs.msg import Float32MultiArray,MultiArrayDimension
    from rclpy.serialization import serialize_message
    assert grid._packed_array_schema_supported()
    values=np.array([0x7fc01234,0x80000000,1,0x7f800000,0xff800000,0],np.uint32).view(np.float32).reshape(2,3)
    for label in ['row_index','column_index']:
        template=Float32MultiArray();template.layout.dim=[MultiArrayDimension(label=label,size=2,stride=6),
            MultiArrayDimension(label='other',size=3,stride=3)];template.layout.data_offset=7
        source=values.copy()
        fast=grid.matrix_to_multiarray_like(source,template)
        with monkeypatch.context()as patch:
            patch.setattr(grid,'_packed_array_schema_supported',lambda:False)
            fallback=grid.matrix_to_multiarray_like(source,template)
        assert serialize_message(fast)==serialize_message(fallback)
        assert fast.layout is not template.layout and fast.layout.data_offset==0
        assert template.layout.data_offset==7
        before=fast.data.tobytes();source[:]=0
        assert fast.data.tobytes()==before
    for shape in [(0,3),(2,0),(2,3)]:
        a=np.zeros(shape,np.float32)
        fast=grid.matrix_to_multiarray_like(a,None)
        with monkeypatch.context()as patch:
            patch.setattr(grid,'_packed_array_schema_supported',lambda:False)
            fallback=grid.matrix_to_multiarray_like(a,None)
        assert serialize_message(fast)==serialize_message(fallback)


@pytest.mark.parametrize('shape',[(3,7),(7,3),(0,3),(3,0),(1,1)])
@pytest.mark.parametrize('shift',[(0,0),(1,2),(-2,4),(999,-999)])
def test_direct_ring_permutation_preserves_every_bit_and_owns_its_storage(shape,shift):
    from fsgp_bgk_se2.se2_grid_tools import _roll2d_owned
    source=np.arange(shape[0]*shape[1],dtype=np.uint32).reshape(shape)
    if source.size:source.flat[0]=0x7fc01234
    for a in [source.view(np.float32),source.T.view(np.float32),source[::-1,::2].view(np.float32)]:
        actual=_roll2d_owned(a,*shift);expected=np.roll(a,shift,axis=(0,1))
        assert actual.dtype==expected.dtype and actual.shape==expected.shape
        np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
        assert not np.shares_memory(actual,a)
