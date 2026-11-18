#!/usr/bin/env python3
"""Capture the ORIGINAL evidence of a large downward native ground innovation.

Diagnostic observer only: never publishes commands, goals or map values.
Its extra subscriber/file I/O excludes this run from performance qualification.
"""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.serialization import deserialize_message
from se2_grid_msgs.msg import SE2Grid
from sensor_msgs.msg import PointCloud2,Imu
from nav_msgs.msg import Odometry
from fsgp_bgk_se2.se2_grid_tools import serialized_header,get_layer_slices,grid_buffer_to_world


def fault_cells(height,variance,pending,candidate,observed,x,y,pose,max_distance=2.):
    arrays=[np.asarray(a)for a in (height,variance,pending,candidate,observed,x,y)]
    if any(a.shape!=arrays[0].shape for a in arrays):raise ValueError('fault audit geometry mismatch')
    if pose is None:return np.empty((0,2),int)
    h,v,p,z,seen,x,y=arrays
    mask=(np.isfinite(h)&np.isfinite(v)&np.isfinite(z)&(v>=0)&(v<=.0004)&
        (seen>.5)&(p>.5)&(z<h-.20)&(np.hypot(x-pose[0],y-pose[1])<=max_distance))
    return np.argwhere(mask)


class Capture(Node):
    def __init__(self,output):
        super().__init__('native_ground_fault_capture');self.output=output
        self.sources={name:deque(maxlen=count)for name,count in [('lidar',40),('registered',40),('imu',800),('odom',40)]}
        self.pose=None;self.saved=False
        self.map_count=0;self.missing_field_maps=0
        self.received={name:0 for name in self.sources}
        self.max_downward_m=0.
        for name,topic,kind in [('lidar','/livox/lidar',PointCloud2),('registered','/cloud_registered',PointCloud2),
                               ('imu','/imu/data_raw',Imu),('odom','/Odometry',Odometry)]:
            self.create_subscription(kind,topic,lambda payload,n=name:self.source(n,payload),qos_profile_sensor_data,raw=True)
        self.create_subscription(SE2Grid,'/car_0_terrain_analyzer/fused_map',self.map,qos_profile_sensor_data,raw=True)

    def source(self,name,payload):
        self.received[name]+=1
        header=serialized_header(payload);stamp=header.stamp.sec*1_000_000_000+header.stamp.nanosec
        self.sources[name].append((stamp,bytes(payload)))
        if name=='odom':
            m=deserialize_message(payload,Odometry);p=m.pose.pose.position;self.pose=(p.x,p.y)

    def map(self,payload):
        if self.saved:return
        self.map_count+=1
        m=deserialize_message(payload,SE2Grid)
        def layer(n):
            a=get_layer_slices(m,n)[0]
            return grid_buffer_to_world(a[0],m)if a else None
        fields=[layer(n)for n in ['elevation','var','ground_pending','ground_pending_elevation','ground_observed']]
        if any(a is None for a in fields):
            self.missing_field_maps+=1
            return
        h,v,p,z,seen=fields;r=m.info.pos_resolution
        x,y=np.meshgrid(m.info.pose.position.x-m.info.length_x/2+(np.arange(h.shape[1])+.5)*r,
                        m.info.pose.position.y-m.info.length_y/2+(np.arange(h.shape[0])+.5)*r)
        cells=fault_cells(h,v,p,z,seen,x,y,self.pose)
        if self.pose is not None:
            near=(np.hypot(x-self.pose[0],y-self.pose[1])<=2.)&(p>.5)&np.isfinite(h)&np.isfinite(z)
            if near.any():self.max_downward_m=max(self.max_downward_m,float(np.max(h[near]-z[near])))
        if not len(cells):return
        self.saved=True;self.output.mkdir(parents=True,exist_ok=False)
        (self.output/'map.cdr').write_bytes(payload)
        stamp=m.info.header.stamp.sec*1_000_000_000+m.info.header.stamp.nanosec
        rows=[]
        for rr,cc in cells:
            rows.append(dict(row=int(rr),column=int(cc),x=float(x[rr,cc]),y=float(y[rr,cc]),
                old_height=float(h[rr,cc]),pending_height=float(z[rr,cc]),variance=float(v[rr,cc])))
        manifest={}
        for name,packets in self.sources.items():
            manifest[name]=[]
            for i,(s,data) in enumerate(packets):
                file=f'{name}_{i:04d}.cdr';(self.output/file).write_bytes(data)
                manifest[name].append(dict(file=file,stamp_ns=s,sha256=hashlib.sha256(data).hexdigest(),
                    map_stamp_delta_ns=s-stamp))
        report=dict(scope=__doc__,map_stamp_ns=stamp,map_sha256=hashlib.sha256(payload).hexdigest(),
            pose=self.pose,cells=rows,sources=manifest)
        (self.output/'event.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(saved=str(self.output),stamp_ns=stamp,cells=rows)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--timeout',type=float,default=240.)
    a=p.parse_args();rclpy.init();n=Capture(a.output);deadline=time.monotonic()+a.timeout
    try:
        while rclpy.ok() and not n.saved and time.monotonic()<deadline:rclpy.spin_once(n,timeout_sec=.05)
        return 0 if n.saved else 2
    finally:
        a.output.mkdir(parents=True,exist_ok=True)
        coverage=dict(scope=__doc__,event_saved=n.saved,maps_received=n.map_count,
            maps_missing_fields=n.missing_field_maps,source_counts=n.received,
            maximum_nearby_pending_downward_m=n.max_downward_m,
            streams_seen=bool(n.map_count and all(n.received.values())))
        (a.output/'coverage.json').write_text(json.dumps(coverage,indent=2)+'\n')
        n.destroy_node();rclpy.shutdown()


if __name__=='__main__':raise SystemExit(main())
