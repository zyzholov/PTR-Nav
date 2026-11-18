#!/usr/bin/env python3
"""Capture exact source maps of a bounded set of rejections; no goal/commands.

This is an instrumented diagnostic, not a latency or navigation qualification.
The planning log supplies the map source stamp; original messages are retained.
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
from rclpy.serialization import deserialize_message
from rclpy.qos import qos_profile_sensor_data
from se2_grid_msgs.msg import SE2Grid
from nav_msgs.msg import Path as RosPath,Odometry,OccupancyGrid
from sensor_msgs.msg import PointCloud2
from fsgp_bgk_se2.se2_grid_tools import get_layer_slices
from serialized_header import leading_stamp_ns


PATTERN=re.compile(r'terrain retention rejected: cause=(\w+) source_ns=(\d+) now_ns=(\d+) points=(\d+) end=\(([^,]+),([^\)]+)\) pose=\(([^,]+),([^,]+),([^\)]+)\) nearest=(\d+) remaining=([\d.]+)')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--timeout',type=float,default=90.)
    p.add_argument('--min-x',type=float,help='Diagnostic selection in map frame; never a navigation input')
    p.add_argument('--max-x',type=float,help='Upper bound for diagnostic selection only')
    p.add_argument('--cause',choices=('any','prefix_certificate'),default='any')
    p.add_argument('--count',type=int,choices=range(1,6),default=1)
    a=p.parse_args()
    if a.min_x is not None and not math.isfinite(a.min_x):p.error('--min-x must be finite')
    if a.max_x is not None and (not math.isfinite(a.max_x) or a.min_x is not None and a.max_x<a.min_x):p.error('invalid --max-x')
    if a.output.exists():raise ValueError('output exists')
    rclpy.init();node=rclpy.create_node('capture_retention_rejection');maps=deque(maxlen=20);paths=deque(maxlen=8)
    clouds=deque(maxlen=25);odom=deque(maxlen=60);live=deque(maxlen=25)
    ns=lambda s:s.sec*10**9+s.nanosec
    def grid(payload):
        maps.append((leading_stamp_ns(payload),payload))
    def retain(payload,cls,queue):
        m=deserialize_message(payload,cls) if cls is RosPath else None
        queue.append((leading_stamp_ns(payload),m,payload))
    node.create_subscription(SE2Grid,'/car_0_fsgp_bgk/fused_map',grid,1,raw=True)
    node.create_subscription(RosPath,'/terrain_nav/path',lambda b:retain(b,RosPath,paths),1,raw=True)
    node.create_subscription(Odometry,'/Odometry',lambda b:retain(b,Odometry,odom),qos_profile_sensor_data,raw=True)
    node.create_subscription(PointCloud2,'/cloud_registered',lambda b:retain(b,PointCloud2,clouds),qos_profile_sensor_data,raw=True)
    node.create_subscription(OccupancyGrid,'/terrain_nav/live_obstacles_raw',lambda b:retain(b,OccupancyGrid,live),1,raw=True)
    deadline=time.monotonic()+a.timeout;lines=0;pending=deque(maxlen=128);seen=set();captured=[]
    def save_index():
        if a.count>1 and captured:
            (a.output/'captures.json').write_text(json.dumps(dict(requested=a.count,
                captured=captured,complete=len(captured)==a.count,
                scope='passive instrumented diagnosis, not timing qualification'),indent=2)+'\n')
    try:
        while time.monotonic()<deadline:
            rclpy.spin_once(node,timeout_sec=.02)
            file=a.run/'planning.log'
            if file.exists():
                with file.open() as f:
                    f.seek(lines);new=f.read();lines=f.tell()
                for match in PATTERN.finditer(new):
                    if (match[1]!='capture_or_remaining' and (a.cause=='any' or match[1]==a.cause) and
                            (a.min_x is None or float(match[7])>=a.min_x) and
                            (a.max_x is None or float(match[7])<=a.max_x)):pending.append(match)
            for event in pending:
                stamp=int(event[2]);matches=[b for t,b in maps if t==stamp]
                if stamp in seen:continue
                candidates=[(m,b) for _,m,b in paths if len(m.poses)==int(event[4]) and
                    abs(m.poses[-1].pose.position.x-float(event[5]))<2e-5 and
                    abs(m.poses[-1].pose.position.y-float(event[6]))<2e-5]
                if not matches or not candidates or not odom:continue
                payloads={'map.cdr':matches[-1],'path.cdr':candidates[-1][1]}
                msg=deserialize_message(matches[-1],SE2Grid)
                age=get_layer_slices(msg,'gp_completion_age')[0]
                source=stamp-round(float(age[0].flat[0])*1e9) if age else stamp
                conditioning=[b for t,_,b in clouds if abs(t-source)<=5]
                if not conditioning:continue
                payloads['completion_cloud.cdr']=conditioning[-1]
                payloads['odom.cdr']=min(odom,key=lambda q:abs(q[0]-int(event[3])))[2]
                if live:payloads['live_raw.cdr']=min(live,key=lambda q:abs(q[0]-int(event[3])))[2]
                destination=a.output
                if a.count>1:
                    a.output.mkdir(parents=True,exist_ok=True)
                    destination=a.output/f'event_{len(captured)+1:03d}'
                destination.mkdir(parents=True)
                for name,data in payloads.items():(destination/name).write_bytes(data)
                record=dict(scope=__doc__,source_run=str(a.run.resolve()),cause=event[1],
                    diagnostic_min_x=a.min_x,
                    map_source_stamp_ns=stamp,evaluation_clock_ns=int(event[3]),
                    evaluated_pose=[float(event[i]) for i in (7,8,9)],nearest_index=int(event[10]),remaining_m=float(event[11]),
                    original_log=event[0],sha256={name:hashlib.sha256(data).hexdigest() for name,data in payloads.items()},
                    limitation='map/path/cloud are source matched; recorded live grid is nearest in source time and cannot reconstruct all tracker or hold-state history')
                (destination/'rejection.json').write_text(json.dumps(record,indent=2)+'\n')
                seen.add(stamp);captured.append(dict(directory=str(destination),source_stamp_ns=stamp,
                    cause=event[1],pose=record['evaluated_pose']))
                save_index();print(json.dumps(record),flush=True)
                if len(captured)>=a.count:return 0
        if captured:
            save_index();print(json.dumps(dict(partial_capture=True,captured=len(captured),requested=a.count)),flush=True)
            return 0
        raise RuntimeError('no source-matched useful rejection was captured')
    finally:node.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
