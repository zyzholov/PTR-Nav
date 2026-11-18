#!/usr/bin/env python3
"""One passive source-matched terrain snapshot, including pre-navigation failure.

No goals, commands or truth subscriptions. Additional subscriptions make the
owning run an instrumented diagnosis, not a timing qualification.
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
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from se2_grid_msgs.msg import SE2Grid
from fsgp_bgk_se2.se2_grid_tools import get_layer_slices
from serialized_header import leading_stamp_ns


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-after',type=float,default=30.)
    p.add_argument('--timeout',type=float,default=150.)
    a=p.parse_args()
    if a.output.exists():p.error('output exists')
    rclpy.init();node=rclpy.create_node('terrain_source_snapshot')
    queues={k:deque(maxlen=20) for k in ('map','front','cloud','odom')}
    for key,cls,topic in (('map',SE2Grid,'/car_0_fsgp_bgk/fused_map'),
            ('front',SE2Grid,'/car_0_terrain_analyzer/fused_map'),
            ('cloud',PointCloud2,'/cloud_registered'),('odom',Odometry,'/Odometry')):
        def retain(b,k=key):queues[k].append((leading_stamp_ns(b),b))
        node.create_subscription(cls,topic,retain,1 if cls is SE2Grid else qos_profile_sensor_data,raw=True)
    deadline=time.monotonic()+a.timeout
    try:
        while time.monotonic()<deadline:
            rclpy.spin_once(node,timeout_sec=.02)
            for stamp,payload in list(queues['map']):
                if stamp*1e-9<a.source_after or not queues['odom']:continue
                msg=deserialize_message(payload,SE2Grid)
                ages=get_layer_slices(msg,'gp_completion_age')[0]
                source=stamp-round(float(ages[0].flat[0])*1e9) if ages else stamp
                front=[b for t,b in queues['front'] if t==stamp]
                cloud=[b for t,b in queues['cloud'] if abs(t-source)<=5]
                odom_stamp,odom=min(queues['odom'],key=lambda r:abs(r[0]-stamp))
                if not front or not cloud or abs(odom_stamp-stamp)>120_000_000:continue
                data={'map.cdr':payload,'front.cdr':front[-1],'completion_cloud.cdr':cloud[-1],'odom.cdr':odom}
                a.output.mkdir(parents=True)
                for name,b in data.items():(a.output/name).write_bytes(b)
                metadata=dict(scope=__doc__,map_source_stamp_ns=stamp,conditioning_stamp_ns=source,
                    odom_stamp_ns=odom_stamp,sha256={name:hashlib.sha256(b).hexdigest()for name,b in data.items()})
                (a.output/'capture.json').write_text(json.dumps(metadata,indent=2)+'\n')
                print(json.dumps(metadata),flush=True);return 0
        raise RuntimeError('no source-matched terrain snapshot')
    finally:node.destroy_node();rclpy.shutdown()


if __name__=='__main__':raise SystemExit(main())
