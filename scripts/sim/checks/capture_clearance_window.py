#!/usr/bin/env python3
"""Observe a near-field stop and archive sensor/floor history for local replay.

No publications, goals, truth inputs or navigation configuration changes.
Clouds are cropped to +/-6m around the observed pose; this is a local diagnostic,
not a complete planner replay or timing benchmark.
"""
import argparse
from collections import deque
import json
from pathlib import Path
import time

import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.serialization import serialize_message
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from se2_grid_msgs.msg import SE2Grid
from diagnostic_msgs.msg import DiagnosticArray
from fsgp_bgk_se2.fsgp_bgk_se2_node import FsgpBgkSe2Node
from fsgp_bgk_se2.se2_grid_tools import get_layer_slices,grid_buffer_to_world


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--timeout',type=float,default=150.)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    rclpy.init();node=rclpy.create_node('clearance_window_observer')
    buffers={key:deque(maxlen=150) for key in ('maps','nav','full')}
    state={'pose':None,'first':None,'stopped':None,'map':None,'status':None}
    stamp=lambda h:h.stamp.sec+h.stamp.nanosec*1e-9
    def odom(msg):
        p=msg.pose.pose.position;state['pose']=(p.x,p.y)
        if state['first'] is None:state['first']=(p.x,p.y)
    def cloud(key,msg):
        if state['pose'] is None:return
        values=FsgpBgkSe2Node._decode_points(msg)
        if values is None:return
        x,y=state['pose'];keep=(abs(values[:,0]-x)<6)&(abs(values[:,1]-y)<6)
        buffers[key].append((stamp(msg.header),msg.header.frame_id,values[keep]))
    names=('gp_elevation_observed','gp_variance_observed','ground_observed',
           'collision_ceiling_elevation','collision_current','probability_raw_veto',
           'collision_floor_reference_max_age_s','collision_floor_reference_full_scan')
    def grid(msg):
        if any(name not in msg.layers for name in names):return
        arrays=[grid_buffer_to_world(get_layer_slices(msg,name)[0][0],msg).copy() for name in names]
        buffers['maps'].append((stamp(msg.info.header),msg.info.header.frame_id,
            (msg.info.pose.position.x-msg.info.length_x/2,msg.info.pose.position.y-msg.info.length_y/2),
            msg.info.pos_resolution,np.stack(arrays)))
        state['map']=msg
    def status(msg):
        for s in msg.status:
            if s.name!='terrain_nav/chance_constraints':continue
            values={kv.key:kv.value for kv in s.values}
            state['status']={'message':s.message,'values':values,'stamp':stamp(msg.header)}
            pose,first=state['pose'],state['first']
            eligible=(pose is not None and first is not None and np.linalg.norm(np.subtract(pose,first))>2.
                and abs(float(values.get('control_state_speed_mps',values.get('measured_speed_mps','inf'))))<.03
                and values.get('command_accepted')=='0')
            if not eligible:state['stopped']=None
            elif state['stopped'] is None:state['stopped']=time.monotonic()
    node.create_subscription(Odometry,'/Odometry',odom,qos_profile_sensor_data)
    node.create_subscription(PointCloud2,'/cloud_registered',lambda msg:cloud('nav',msg),qos_profile_sensor_data)
    node.create_subscription(PointCloud2,'/cloud_registered_visual',lambda msg:cloud('full',msg),qos_profile_sensor_data)
    node.create_subscription(SE2Grid,'/car_0_fsgp_bgk/fused_map',grid,qos_profile_sensor_data)
    node.create_subscription(DiagnosticArray,'/terrain_nav/chance_status',status,10)
    deadline=time.monotonic()+args.timeout;trigger=False
    while time.monotonic()<deadline:
        rclpy.spin_once(node,timeout_sec=.05)
        if state['stopped'] is not None and time.monotonic()-state['stopped']>=1.:
            trigger=True;break
    arrays={};metadata={'scope':__doc__,'triggered':trigger,'layer_names':names,'status':state['status']}
    for key,rows in buffers.items():
        metadata[key]=[]
        for i,row in enumerate(rows):
            name=f'{key}_{i}';arrays[name]=row[-1]
            metadata[key].append({'key':name,'stamp':row[0],'frame':row[1],
                **({'origin':row[2],'resolution':row[3]} if key=='maps' else {})})
    np.savez(args.output/'window.npz',**arrays)
    (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    if state['map'] is not None:(args.output/'map.cdr').write_bytes(serialize_message(state['map']))
    print(json.dumps({'triggered':trigger,'counts':{key:len(rows) for key,rows in buffers.items()},
                      'output':str(args.output)}),flush=True)
    node.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
