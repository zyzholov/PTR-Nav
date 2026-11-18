#!/usr/bin/env python3
"""Extract source-stamped diagnostic inputs; never publishes navigation data."""
import argparse
import json
from pathlib import Path
import sqlite3

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from rosgraph_msgs.msg import Clock


def source_stamp(message):
    header=message.info.header if hasattr(message,'layers') else message.header
    return header.stamp.sec+1e-9*header.stamp.nanosec


def extract(bag,stamp,output):
    databases=list(Path(bag).glob('*.db3'))
    if len(databases)!=1:raise ValueError('expected exactly one closed sqlite bag segment')
    if output.exists():raise ValueError('output must not exist')
    con=sqlite3.connect(f'file:{databases[0]}?mode=ro',uri=True)
    topics={name:(identifier,kind) for identifier,name,kind in con.execute('select id,name,type from topics')}
    clock_id=topics['/clock'][0]
    clocks=[]
    for receipt,data in con.execute('select timestamp,data from messages where topic_id=?',(clock_id,)):
        message=deserialize_message(data,Clock);clocks.append((abs(message.clock.sec+message.clock.nanosec*1e-9-stamp),receipt))
    difference,receipt=min(clocks)
    if difference>.05:raise ValueError(f'requested clock not present: {difference}s away')
    requested={'map.cdr':'/car_0_fsgp_bgk/fused_map','odom.cdr':'/Odometry',
        'completion_cloud.cdr':'/cloud_registered_visual','topology_memory.cdr':'/terrain_nav/topology_memory',
        'topology_route.cdr':'/terrain_nav/topology_route'}
    records={};payloads={}
    for filename,topic in requested.items():
        identifier,kind=topics[topic];cls=get_message(kind)
        candidates=[]
        for wall,data in con.execute('select timestamp,data from messages where topic_id=? and timestamp between ? and ? order by timestamp',
                                    (identifier,receipt-int(3e9),receipt+int(.5e9))):
            message=deserialize_message(data,cls);time=source_stamp(message)
            if time<=stamp+1e-6:candidates.append((time,wall,data))
        if not candidates:raise ValueError(f'no causal source-stamped {topic} near requested clock')
        time,wall,data=max(candidates,key=lambda value:value[:2])
        if stamp-time>2.5:raise ValueError(f'stale diagnostic {topic}: {stamp-time}s')
        payloads[filename]=data;records[filename]=dict(topic=topic,source_stamp_s=time,age_s=stamp-time)
    output.mkdir(parents=True)
    for filename,data in payloads.items():(output/filename).write_bytes(data)
    (output/'snapshot.json').write_text(json.dumps(dict(scope='causal recorded diagnostic; no truth feedback or physical execution',
        requested_stamp_s=stamp,bag=str(bag),records=records),indent=2)+'\n')
    con.close()
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag',type=Path);parser.add_argument('--stamp',type=float,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(extract(args.bag,args.stamp,args.output),indent=2))


if __name__=='__main__':main()
