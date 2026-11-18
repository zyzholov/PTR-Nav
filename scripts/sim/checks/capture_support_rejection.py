#!/usr/bin/env python3
"""Capture one exact source map/scan at rejected support; no goals or commands.

This extra subscriber is instrumented diagnosis, not a latency qualification.
The logged evaluated pose is retained; a later stopped pose cannot replace it.
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
PATTERN=re.compile(r'Probability current pose rejected before search: known=(\w+) upper=([^ ]+) limit=([^ ]+) source_ns=(\d+) now_ns=(\d+) pose=\(('+NUMBER+'),('+NUMBER+'),('+NUMBER+r')\)')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--timeout',type=float,default=180.);p.add_argument('--min-x',type=float,default=.5)
    a=p.parse_args()
    if not math.isfinite(a.timeout) or a.timeout<=0 or not math.isfinite(a.min_x):p.error('finite timeout and pose selection required')
    if a.output.exists():raise ValueError('capture output exists')
    rclpy.init();node=rclpy.create_node('capture_support_rejection')
    maps=deque(maxlen=10);clouds=deque(maxlen=40);odoms=deque(maxlen=60)
    def retain(queue,payload):queue.append((leading_stamp_ns(payload),payload))
    for kind,topic,queue,qos in [(SE2Grid,'/car_0_fsgp_bgk/fused_map',maps,1),
            (PointCloud2,'/cloud_registered',clouds,qos_profile_sensor_data),
            (Odometry,'/Odometry',odoms,qos_profile_sensor_data)]:
        node.create_subscription(kind,topic,lambda payload,q=queue:retain(q,payload),qos,raw=True)
    deadline=time.monotonic()+a.timeout;offset=0;carry='';pending=deque(maxlen=16)
    try:
        while time.monotonic()<deadline:
            rclpy.spin_once(node,timeout_sec=.02)
            log=a.run/'planning.log'
            if log.exists():
                with log.open() as f:f.seek(offset);carry+=f.read();offset=f.tell()
                lines=carry.split('\n');carry=lines.pop()
                for line in lines:
                    event=PATTERN.search(line)
                    if event and event[1]=='false' and float(event[6])>=a.min_x:pending.append(event)
            for event in pending:
                stamp=int(event[4]);exact=[b for t,b in maps if t==stamp]
                if not exact or not odoms:continue
                msg=deserialize_message(exact[-1],SE2Grid)
                age=get_layer_slices(msg,'gp_completion_age')[0]
                cloud_age=float(age[0].flat[0]) if age else 0.
                source=stamp-round(cloud_age*1e9)
                scans=[b for t,b in clouds if abs(t-source)<=5]
                if not scans:continue
                selected=min(odoms,key=lambda r:abs(r[0]-int(event[5])))[1]
                odom=deserialize_message(selected,Odometry)
                payloads={'map.cdr':exact[-1],'completion_cloud.cdr':scans[-1],'odom.cdr':selected}
                a.output.mkdir(parents=True)
                for name,value in payloads.items():(a.output/name).write_bytes(value)
                record=dict(scope=__doc__,source_run=str(a.run.resolve()),original_log=event[0],
                    evaluation_clock_ns=int(event[5]),map_sha256=hashlib.sha256(exact[-1]).hexdigest(),
                    pose=[float(event[6]),float(event[7]),odom.pose.pose.position.z,float(event[8])],
                    completion_cloud=dict(source_age_s=cloud_age,sha256=hashlib.sha256(scans[-1]).hexdigest(),
                        scope='exact conditioning stamp; original shared posterior still requires numerical replay verification'),
                    sha256={name:hashlib.sha256(value).hexdigest()for name,value in payloads.items()})
                (a.output/'diagnostic.json').write_text(json.dumps(record,indent=2)+'\n')
                print(json.dumps(record),flush=True);return 0
        raise RuntimeError('no exact support-rejection source was captured')
    finally:node.destroy_node();rclpy.shutdown()


if __name__=='__main__':raise SystemExit(main())
