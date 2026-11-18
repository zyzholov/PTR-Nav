#!/usr/bin/env python3
"""Capture a source-matched braking rejection, without goals or commands.

Raw bounded caches read only the leading stamp, avoiding repeated full-map
Python decoding. This is instrumented diagnosis, not a latency qualification.
"""
import argparse
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import re
import time
import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.serialization import deserialize_message
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from se2_grid_msgs.msg import SE2Grid
from fsgp_bgk_se2.se2_grid_tools import get_layer_slices
from serialized_header import leading_stamp_ns

NUMBER=r'[-+0-9.eE]+'
PATTERN=re.compile(r'terrain braking rejected: source_ns=(\d+) now_ns=(\d+) root=\(('+NUMBER+'),('+NUMBER+'),('+NUMBER+'),('+NUMBER+'),('+NUMBER+r')\) cmd=\(('+NUMBER+'),('+NUMBER+r')\) sample=\(('+NUMBER+'),('+NUMBER+'),('+NUMBER+r')\) t=('+NUMBER+r') cause=(\w+) support=(\d+) upper=([^ ]+) body=(\d+)')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--timeout',type=float,default=180.)
    p.add_argument('--min-x',type=float,default=.25);a=p.parse_args()
    if not math.isfinite(a.min_x) or not math.isfinite(a.timeout) or a.timeout<=0:p.error('finite diagnostic selection required')
    if a.output.exists():raise ValueError('diagnostic output exists')
    rclpy.init();node=rclpy.create_node('capture_braking_rejection')
    maps=deque(maxlen=10);clouds=deque(maxlen=30);odom=deque(maxlen=60);fronts=deque(maxlen=20)
    def retain(queue,payload):queue.append((leading_stamp_ns(payload),payload))
    node.create_subscription(SE2Grid,'/car_0_fsgp_bgk/fused_map',lambda b:retain(maps,b),1,raw=True)
    node.create_subscription(SE2Grid,'/car_0_terrain_analyzer/fused_map',lambda b:retain(fronts,b),1,raw=True)
    node.create_subscription(PointCloud2,'/cloud_registered',lambda b:retain(clouds,b),qos_profile_sensor_data,raw=True)
    node.create_subscription(Odometry,'/Odometry',lambda b:retain(odom,b),qos_profile_sensor_data,raw=True)
    deadline=time.monotonic()+a.timeout;offset=0;carry='';pending=deque(maxlen=12)
    try:
        while time.monotonic()<deadline:
            rclpy.spin_once(node,timeout_sec=.02)
            log=a.run/'planning.log'
            if log.exists():
                with log.open() as f:f.seek(offset);carry+=f.read();offset=f.tell()
                lines=carry.split('\n');carry=lines.pop()
                for line in lines:
                    event=PATTERN.search(line)
                    if event and float(event[3])>=a.min_x:pending.append(event)
            for event in pending:
                stamp=int(event[1]);exact=[b for t,b in maps if t==stamp]
                if not exact or not odom:continue
                msg=deserialize_message(exact[-1],SE2Grid)
                age=get_layer_slices(msg,'gp_completion_age')[0]
                source=stamp-round(float(age[0].flat[0])*1e9) if age else stamp
                conditioning=[b for t,b in clouds if abs(t-source)<=5]
                front=[b for t,b in fronts if t==stamp]
                if not conditioning or not front:continue
                data={'map.cdr':exact[-1],'completion_cloud.cdr':conditioning[-1],
                      'front.cdr':front[-1],
                      'odom.cdr':min(odom,key=lambda v:abs(v[0]-int(event[2])))[1]}
                a.output.mkdir(parents=True)
                for name,b in data.items():(a.output/name).write_bytes(b)
                record=dict(scope=__doc__,kind='braking',source_run=str(a.run.resolve()),
                    map_source_stamp_ns=stamp,evaluation_clock_ns=int(event[2]),
                    root=[float(event[i]) for i in (3,4,5,6,7)],command=[float(event[i]) for i in (8,9)],
                    rejected_pose=[float(event[i]) for i in (10,11,12)],horizon_time_s=float(event[13]),
                    cause=event[14],reported_support=bool(int(event[15])),reported_upper=event[16],reported_body=bool(int(event[17])),
                    original_log=event[0],sha256={name:hashlib.sha256(b).hexdigest() for name,b in data.items()},
                    limitation='exact map/conditioning cloud, logged evaluated state/command; detector coast/actuator-history state is not reconstructed')
                (a.output/'braking.json').write_text(json.dumps(record,indent=2)+'\n')
                print(json.dumps(record),flush=True);return 0
        raise RuntimeError('no source-matched braking rejection captured')
    finally:node.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
