#!/usr/bin/env python3
"""Archive time-matched measured/GP maps after a physical obstacle leaves.

Passive diagnostic only: no goals, commands, map edits or truth-fed planning.
Additional subscriptions make this unsuitable as a latency qualification.
"""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import time

import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.serialization import deserialize_message
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from se2_grid_msgs.msg import SE2Grid
from fsgp_bgk_se2.se2_grid_tools import get_layer_slices
from serialized_header import leading_stamp_ns


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--timeout',type=float,default=150.)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    rclpy.init();node=rclpy.create_node('vacated_source_recorder')
    queues={k:deque(maxlen=30) for k in ('front','cloud','map','odom')}
    state={'vacated_ns':None};rows=[];seen=set();targets=(.1,.5,1.,2.,4.)
    def obstacle(msg):
        if state['vacated_ns'] is None and abs(msg.pose.position.y)>1.5:
            state['vacated_ns']=msg.header.stamp.sec*10**9+msg.header.stamp.nanosec
    node.create_subscription(PoseStamped,'/isaac/nav_test_obstacle/pose',obstacle,10)
    for key,cls,topic in (('front',SE2Grid,'/car_0_terrain_analyzer/fused_map'),
                          ('map',SE2Grid,'/car_0_fsgp_bgk/fused_map'),
                          ('cloud',PointCloud2,'/cloud_registered'),('odom',Odometry,'/Odometry')):
        def retain(b,k=key):queues[k].append((leading_stamp_ns(b),b))
        node.create_subscription(cls,topic,retain,1 if cls is SE2Grid else qos_profile_sensor_data,raw=True)
    deadline=time.monotonic()+a.timeout
    try:
        while time.monotonic()<deadline and len(rows)<len(targets):
            rclpy.spin_once(node,timeout_sec=.02)
            if state['vacated_ns'] is None or not queues['odom']:continue
            for stamp,b in list(queues['map']):
                age=(stamp-state['vacated_ns'])*1e-9
                if age<targets[len(rows)] or stamp in seen:continue
                msg=deserialize_message(b,SE2Grid)
                completion_age=get_layer_slices(msg,'gp_completion_age')[0]
                source=stamp-round(float(completion_age[0].flat[0])*1e9) if completion_age else stamp
                front=[data for t,data in queues['front'] if abs(t-stamp)<=5]
                cloud=[data for t,data in queues['cloud'] if abs(t-source)<=5]
                if not front or not cloud:continue
                dest=a.output/f'event_{len(rows)+1:03d}';dest.mkdir()
                payloads={'map.cdr':b,'front.cdr':front[-1],'completion_cloud.cdr':cloud[-1],
                          'odom.cdr':min(queues['odom'],key=lambda row:abs(row[0]-stamp))[1]}
                for name,payload in payloads.items():(dest/name).write_bytes(payload)
                row=dict(directory=str(dest),target_age_s=targets[len(rows)],age_s=age,
                         map_stamp_ns=stamp,conditioning_stamp_ns=source,
                         sha256={name:hashlib.sha256(data).hexdigest() for name,data in payloads.items()})
                (dest/'capture.json').write_text(json.dumps(row,indent=2)+'\n')
                rows.append(row);seen.add(stamp);print(json.dumps(row),flush=True)
                if len(rows)==len(targets):break
        (a.output/'captures.json').write_text(json.dumps(dict(scope=__doc__,
            complete=len(rows)==len(targets),vacated_ns=state['vacated_ns'],captures=rows),indent=2)+'\n')
        return int(len(rows)!=len(targets))
    finally:node.destroy_node();rclpy.shutdown()


if __name__=='__main__':raise SystemExit(main())
