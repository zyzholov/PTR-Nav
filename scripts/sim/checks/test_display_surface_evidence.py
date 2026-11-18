import importlib.util
from pathlib import Path

import numpy as np
import pytest
from se2_grid_msgs.msg import SE2Grid
from fsgp_bgk_se2.se2_grid_tools import matrix_to_multiarray_like,world_to_grid_buffer

spec=importlib.util.spec_from_file_location('visualization_probe',Path(__file__).with_name('visualization_probe.py'))
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)


def test_probe_accepts_only_measured_or_qualified_heights_without_filling_unknown():
    msg=SE2Grid();msg.info.pos_resolution=.2;msg.info.length_x=msg.info.length_y=.4
    msg.start_index_row=msg.start_index_col=1
    measured=np.array([[1.,np.nan],[np.nan,4.]],np.float32)
    observed=np.array([[1.,0.],[0.,0.]],np.float32)
    for name,value in [('gp_elevation_observed',measured),('terrain_observed',observed),
                       ('probability_elevation',np.full((2,2),2.,np.float32)),
                       ('probability_gp_used',np.array([[1,1],[0,0]],np.float32))]:
        msg.layers.append(name);msg.has_so2.append(False)
        msg.data.append(matrix_to_multiarray_like(world_to_grid_buffer(value,msg),None))
    expected,qualified=probe.surface_evidence(msg)
    assert len(expected)==2
    assert abs(expected[(-.1,-.1)]-1.02)<1e-6
    assert abs(expected[(.1,-.1)]-2.02)<1e-6
    assert qualified=={(.1,-.1)}


def test_flat_measured_evidence_has_zero_span_not_a_fabricated_height_range():
    msg=SE2Grid();msg.info.pos_resolution=.2;msg.info.length_x=msg.info.length_y=.4
    for name,value in [('gp_elevation_observed',np.full((2,2),-.48,np.float32)),
                       ('terrain_observed',np.ones((2,2),np.float32))]:
        msg.layers.append(name);msg.has_so2.append(False)
        msg.data.append(matrix_to_multiarray_like(value,None))
    expected,qualified=probe.surface_evidence(msg)
    assert len(expected)==4 and not qualified
    assert max(expected.values())-min(expected.values())==0
    assert next(iter(expected.values()))==pytest.approx(-.46)


def test_surface_sampling_requires_one_original_stamp_across_all_five_products():
    from rclpy.serialization import serialize_message
    from sensor_msgs.msg import PointCloud2
    from visualization_msgs.msg import Marker
    names=('grid','elevation','risk','surface','height_surface');samples=probe.SerializedSamples(names)
    types=(SE2Grid,PointCloud2,PointCloud2,Marker,Marker)
    for name,kind in zip(names,types):
        msg=kind();head=msg.info.header if name=='grid' else msg.header
        head.stamp.sec=12;head.stamp.nanosec=345;head.frame_id='map'
        matched=samples.add(name,serialize_message(msg))
        if name!=names[-1]:assert matched is None
    assert matched[0]==(12,345)
    assert len(matched[1])==5


def test_serialized_pairing_requires_exact_original_stamp_and_stays_bounded():
    from rclpy.serialization import serialize_message,deserialize_message
    from sensor_msgs.msg import PointCloud2
    samples=probe.SerializedSamples(('raw','rgb','nav'),depth=3)
    def payload(sec):
        msg=PointCloud2();msg.header.stamp.sec=sec;msg.header.stamp.nanosec=123
        msg.header.frame_id='map';return serialize_message(msg)
    for sec in range(10,20):
        assert samples.add('raw',payload(sec)) is None
    assert len(samples.pending['raw'])==3
    assert samples.add('rgb',payload(18)) is None
    assert samples.add('nav',payload(19)) is None
    key,matched=samples.add('nav',payload(18))
    assert key==(18,123)
    assert all(deserialize_message(p,PointCloud2).header.stamp.sec==18 for p in matched.values())
    assert samples.add('rgb',payload(20))[0]==(18,123)  # never pairs 19 and 20 as one scan


def test_raw_lidar_triple_checks_xyz_and_density_without_retimestamping():
    from types import SimpleNamespace
    from rclpy.serialization import serialize_message
    from sensor_msgs.msg import PointCloud2,PointField
    from std_msgs.msg import Header
    from sensor_msgs_py.point_cloud2 import create_cloud
    header=Header();header.stamp.sec=42;header.stamp.nanosec=17;header.frame_id='map'
    fields=[PointField(name=a,offset=4*i,datatype=PointField.FLOAT32,count=1) for i,a in enumerate(('x','y','z','rgb'))]
    cloud=create_cloud(header,fields,[(1.,2.,3.,1.)])
    obj=SimpleNamespace(lidar_received={n:0 for n in ('live','raw_live','nav_live')},lidar_matches={},
                        serialized_lidar={'live':probe.SerializedSamples(('live','raw_live','nav_live'))},
                        lidar_pending={'live':{},'raw_live':{}},navigation_scans={},density_matches=[],lidar_checked={})
    for name in ('_lidar_cb','_navigation_scan_cb','_check_density'):
        setattr(obj,name,getattr(probe.VisualizationProbe,name).__get__(obj))
    raw=serialize_message(cloud)
    for n in ('live','raw_live'):
        probe.VisualizationProbe._raw_lidar_cb(obj,n,raw)
        assert not obj.lidar_matches
    probe.VisualizationProbe._raw_lidar_cb(obj,'nav_live',raw)
    assert obj.lidar_checked['live']==(42,17)
    assert obj.density_matches==[(1,1)]
    cloud=create_cloud(header,fields,[(99.,2.,3.,1.)])
    obj.lidar_matches.clear();obj.serialized_lidar['live']=probe.SerializedSamples(('live','raw_live','nav_live'))
    obj.lidar_checked.clear()
    probe.VisualizationProbe._raw_lidar_cb(obj,'live',serialize_message(cloud))
    probe.VisualizationProbe._raw_lidar_cb(obj,'raw_live',raw)
    with pytest.raises(RuntimeError,match='changed measured coordinates'):
        probe.VisualizationProbe._raw_lidar_cb(obj,'nav_live',raw)
