#!/usr/bin/env python3
"""Export measured, timestamp-matched SLAM scans/poses for native filter replay.

No simulator truth, object labels, future poses, or ROI enters this stream.
The binary stream is little endian: magic PTRMAP01; repeated timestamp f64,
origin 3xf32, sensor rotation row-major 9xf32, count u32, world XYZ Nx3xf32.
"""
import argparse
import json
from pathlib import Path
import sqlite3
import struct
import numpy as np
from rclpy.serialization import deserialize_message
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2


def stamp(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def rotation(q):
    x,y,z,w=q.x,q.y,q.z,q.w
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('bag',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--config',type=Path,default=Path('src/FAST_LIO/config/isaac_hunter_mid360_pointcloud.yaml'))
    a=p.parse_args()
    import yaml
    root=yaml.safe_load(a.config.read_text())['/**']['ros__parameters']
    conf=root['mapping'];frame=root['frame']
    rt=np.array(conf['extrinsic_R']).reshape(3,3);tr=np.array(conf['extrinsic_T'])
    publish_r=np.array(frame['body_pose_R']).reshape(3,3);publish_t=np.array(frame['body_pose_T'])
    dbs=sorted(a.bag.glob('*.db3'));poses={};scans=[]
    for db in dbs:
        with sqlite3.connect(db) as con:
            topics=dict(con.execute('select id,name from topics'))
            for tid,data in con.execute('select topic_id,data from messages order by timestamp'):
                if topics[tid]=='/Odometry':
                    m=deserialize_message(data,Odometry);s=round(stamp(m),6)
                    if m.child_frame_id!=frame['body_frame']:raise ValueError('unexpected odometry body frame')
                    r=rotation(m.pose.pose.orientation)@publish_r.T;v=m.pose.pose.position
                    poses[s]=(np.array([v.x,v.y,v.z])+r@(tr-publish_t),r@rt)
                elif topics[tid]=='/cloud_registered_visual':
                    m=deserialize_message(data,PointCloud2)
                    if m.header.frame_id!='map':raise ValueError('world cloud must use map frame')
                    if m.height!=1 or m.is_bigendian:raise ValueError('unsupported cloud layout')
                    offsets={f.name:f.offset for f in m.fields}
                    xyz=np.column_stack([np.ndarray((m.width,),dtype='<f4',buffer=m.data,
                        offset=offsets[k],strides=(m.point_step,)) for k in ('x','y','z')])
                    scans.append((stamp(m),xyz[np.isfinite(xyz).all(axis=1)]))
    matched=0
    with a.output.open('xb') as out:
        out.write(b'PTRMAP01')
        for s,xyz in sorted(scans,key=lambda x:x[0]):
            pose=poses.get(round(s,6))
            if pose is None:continue  # exact source timestamp only, never future/nearest pose
            origin,r=pose
            out.write(struct.pack('<d12fI',s,*origin,*r.ravel(),len(xyz)))
            out.write(xyz.astype('<f4').tobytes());matched+=1
    meta=dict(scans=len(scans),matched=matched,missing=len(scans)-matched,
              input=str(a.bag),config=str(a.config),scope='measured exact-time LiDAR pose and corrected world returns; no truth')
    a.output.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta))
    if matched!=len(scans):raise RuntimeError('pose/cloud timestamp mismatch')


if __name__=='__main__':main()
