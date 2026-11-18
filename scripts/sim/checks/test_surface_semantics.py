"""Exercise the installed C++ renderer over ROS, independently of navigation."""
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
import pytest
import rclpy
from ament_index_python.packages import get_package_prefix
from se2_grid_msgs.msg import SE2Grid
from visualization_msgs.msg import Marker
from fsgp_bgk_se2.se2_grid_tools import matrix_to_multiarray_like,world_to_grid_buffer


def test_draped_cloud_omits_pose_without_height_instead_of_flattening_to_default_z():
    from sensor_msgs.msg import PointCloud2
    from sensor_msgs_py.point_cloud2 import read_points
    exe=Path(get_package_prefix('se2_grid_converter'))/'lib/se2_grid_converter/se2_grid_to_pointcloud_node'
    params=dict(input_topic='/qualification/drape/input',output_topic='/qualification/drape/cloud',
        layer_name='failure_probability_upper',valid_layer_name='probability_valid',
        height_layer_name='probability_elevation',drape_on_height='true',z_offset=.035,z_value=.035)
    cmd=[str(exe),'--ros-args']
    for k,v in params.items():cmd+=['-p',f'{k}:={v}']
    proc=subprocess.Popen(cmd,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    rclpy.init();node=rclpy.create_node('drape_fidelity_test');received=[]
    node.create_subscription(PointCloud2,'/qualification/drape/cloud',received.append,1)
    publisher=node.create_publisher(SE2Grid,'/qualification/drape/input',1)
    m=SE2Grid();m.info.header.frame_id='map';m.info.header.stamp.sec=12
    m.info.pos_resolution=.2;m.info.length_x=m.info.length_y=.4
    m.info.yaw_resolution=m.info.length_yaw=2*np.pi
    for n,a in (('failure_probability_upper',np.full((2,2),.01)),('probability_valid',np.ones((2,2))),
                ('probability_elevation',np.array([[1.,np.nan],[1.,np.nan]]))):
        m.layers.append(n);m.has_so2.append(False)
        m.data.append(matrix_to_multiarray_like(world_to_grid_buffer(a,m),None))
    try:
        deadline=time.monotonic()+10
        while not received and time.monotonic()<deadline:
            assert proc.poll() is None,'renderer exited'
            publisher.publish(m);rclpy.spin_once(node,timeout_sec=.05)
        assert received and received[-1].header.stamp.sec==12
        points=read_points(received[-1],field_names=('x','y','z'),skip_nans=True)
        assert len(points)==2
        np.testing.assert_allclose(points['z'],1.035,atol=1e-6)
    finally:
        node.destroy_node();rclpy.shutdown();proc.send_signal(signal.SIGINT)
        try:proc.wait(timeout=5)
        except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=5)


@pytest.mark.parametrize('occupied_style',['cells','boundary'])
def test_raw_terrain_surface_does_not_reapply_body_expansion_and_keeps_full_pose_diagnostic(occupied_style):
    exe=Path(get_package_prefix('se2_grid_converter'))/'lib/se2_grid_converter/se2_grid_to_pointcloud_node'
    params=dict(input_topic='/qualification/surface/input',output_topic='/qualification/surface/cloud',
        layer_name='elevation',valid_layer_name='valid',publish_surface='true',
        surface_topic='/qualification/surface/terrain',body_pose_surface_topic='/qualification/surface/pose',
        surface_risk_layer='failure_probability_upper',surface_probability_colors='true',
        surface_use_navigation_height='true',surface_body_collision_overlay='false',occupied_display_style=occupied_style)
    cmd=[str(exe),'--ros-args']
    for k,v in params.items():cmd+=['-p',f'{k}:={v}']
    proc=subprocess.Popen(cmd,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    rclpy.init();node=rclpy.create_node('surface_semantics_test');received={}
    node.create_subscription(Marker,'/qualification/surface/terrain',lambda m:received.update(terrain=m),1)
    node.create_subscription(Marker,'/qualification/surface/pose',lambda m:received.update(pose=m),1)
    node.create_subscription(Marker,'/qualification/surface/terrain/occupied_cells',lambda m:received.update(occupied=m),1)
    publisher=node.create_publisher(SE2Grid,'/qualification/surface/input',1)
    msg=SE2Grid();msg.info.header.frame_id='map';msg.info.header.stamp.sec=1
    msg.info.pos_resolution=.2;msg.info.length_x=msg.info.length_y=2.
    msg.info.yaw_resolution=2*np.pi;msg.info.length_yaw=2*np.pi
    occupied=np.zeros((10,10),np.float32);occupied[5,5]=1
    for name,value in [('elevation',np.zeros_like(occupied)),('gp_elevation_observed',np.zeros_like(occupied)),
        ('valid',np.ones_like(occupied)),('terrain_observed',np.ones_like(occupied)),
        ('collision_current',occupied),('collision_risk',occupied),
        ('terrain_failure_probability_upper',np.full_like(occupied,.01)),
        ('failure_probability_upper',np.ones_like(occupied)),
        ('collision_witness_z_min',np.where(occupied,.2,np.nan)),
        ('collision_witness_z_max',np.where(occupied,.7,np.nan))]:
        msg.layers.append(name);msg.has_so2.append(False)
        msg.data.append(matrix_to_multiarray_like(world_to_grid_buffer(value,msg),None))
    try:
        deadline=time.monotonic()+10
        while time.monotonic()<deadline and len(received)<3:
            assert proc.poll() is None,'renderer exited'
            publisher.publish(msg);rclpy.spin_once(node,timeout_sec=.05)
        assert len(received)==3,'missing primary, diagnostic surface or occupied extent'
        primary,pose=received['terrain'],received['pose']
        assert primary.ns=='terrain_surface' and pose.ns=='body_pose_risk_surface'
        assert primary.action==pose.action==Marker.ADD
        assert len(primary.points)==len(pose.points)>100
        primary_red=sum(c.r>1.4*c.g for c in primary.colors)
        pose_red=sum(c.r>1.4*c.g for c in pose.colors)
        # One raw occupied cell affects only its triangle vertices. The other
        # surface intentionally receives a full map of unsafe body poses.
        assert 0<primary_red<len(primary.colors)//10
        assert pose_red==len(pose.colors)
        np.testing.assert_allclose([[p.x,p.y,p.z] for p in primary.points],
                                   [[p.x,p.y,p.z] for p in pose.points],atol=1e-7)
        # An obstacle with no terrain height must remain visibly occupied.
        # Its real point Z extents may be drawn, but never become a GP floor.
        msg.info.header.stamp.sec=2
        for name in ('elevation','gp_elevation_observed','terrain_observed','valid'):
            value=np.zeros((10,10),np.float32) if name in ('elevation','gp_elevation_observed') else np.ones((10,10),np.float32)
            value[5,5]=np.nan if name in ('elevation','gp_elevation_observed') else 0.
            msg.data[msg.layers.index(name)]=matrix_to_multiarray_like(world_to_grid_buffer(value,msg),None)
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            publisher.publish(msg);rclpy.spin_once(node,timeout_sec=.05)
            if all(v.header.stamp.sec==2 for v in received.values()):break
        else:pytest.fail('missing source-matched second display frame')
        raw=received['occupied'];assert raw.action==Marker.ADD and len(raw.points)==(24 if occupied_style=='cells' else 8)
        xyz=np.array([[p.x,p.y,p.z] for p in raw.points]);assert np.isfinite(xyz).all()
        np.testing.assert_allclose(xyz.max(axis=0)-xyz.min(axis=0),[.2,.2,.5 if occupied_style=='cells' else 0.],atol=1e-6)
        assert raw.header.stamp.sec==2
        assert raw.color.r>raw.color.g
    finally:
        node.destroy_node();rclpy.shutdown();proc.send_signal(signal.SIGINT)
        try:proc.wait(timeout=5)
        except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=5)
        proc.stderr.close()
