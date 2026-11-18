#!/usr/bin/env python3
"""Observe actual RTX returns, angular diversity and accumulated-map size.

Run while stationary, in the launcher's ROS domain. No generated point samples
are published. Counts are valid hits, not the 200,000 emitted shots/second.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data, QoSProfile
from sensor_msgs.msg import PointCloud2


def xyz_array(msg):
    fields = {f.name: f for f in msg.fields}
    if not all(n in fields and fields[n].datatype == 7 for n in ('x','y','z')):
        raise ValueError('requires FLOAT32 XYZ')
    dtype = np.dtype({'names': ['x','y','z'], 'formats': [('>f4' if msg.is_bigendian else '<f4')]*3,
                     'offsets': [fields[n].offset for n in ('x','y','z')], 'itemsize': msg.point_step})
    data = np.ndarray((msg.height,msg.width), dtype=dtype, buffer=msg.data,
                      strides=(msg.row_step,msg.point_step))
    xyz = np.column_stack([data[n].ravel() for n in ('x','y','z')])
    return xyz[np.isfinite(xyz).all(axis=1) & (np.linalg.norm(xyz,axis=1)>.1)]


def angular_cells(xyz):
    az = np.mod(np.degrees(np.arctan2(xyz[:,1],xyz[:,0])),360.)
    el = np.degrees(np.arctan2(xyz[:,2],np.hypot(xyz[:,0],xyz[:,1])))
    # Quarter-degree bins; deliberately coarser than 0.01deg ray noise.
    return set((np.floor(az*4).astype(np.int64)*1000 + np.floor((el+90)*4).astype(np.int64)).tolist())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--duration',type=float,default=8.)
    args=parser.parse_args()
    rclpy.init(); node=rclpy.create_node('mid360_scan_qualification')
    frames=[]; maps=[]; all_cells=set(); first_cells=set(); arrivals=[]
    def raw(msg):
        nonlocal first_cells
        xyz=xyz_array(msg); cells=angular_cells(xyz)
        if not frames: first_cells=cells
        new=len(cells-all_cells); all_cells.update(cells); arrivals.append(time.monotonic())
        frames.append(dict(stamp=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9,
                           frame=msg.header.frame_id, valid_returns=len(xyz),
                           new_angular_cells=new, angular_cells=len(cells),
                           outside_first_frame_fraction=len(cells-first_cells)/max(1,len(cells))))
    node.create_subscription(PointCloud2,'/livox/lidar',raw,qos_profile_sensor_data)
    node.create_subscription(PointCloud2,'/Laser_map',lambda m:maps.append(dict(
        stamp=m.header.stamp.sec+m.header.stamp.nanosec*1e-9,points=m.width*m.height)),QoSProfile(depth=1))
    deadline=time.monotonic()+args.duration
    try:
        while time.monotonic()<deadline: rclpy.spin_once(node,timeout_sec=.05)
    finally: node.destroy_node(); rclpy.shutdown()
    rate=(len(frames)-1)/(arrivals[-1]-arrivals[0]) if len(frames)>1 else 0.
    summary=dict(frames=len(frames),wall_hz=rate,valid_returns_mean=float(np.mean([f['valid_returns'] for f in frames])) if frames else 0.,
                 union_angular_cells=len(all_cells),
                 angular_diversity_ratio=len(all_cells)/max(1,len(first_cells)),
                 map_points_latest=maps[-1]['points'] if maps else None,
                 scope='stationary_actual_return_diversity_not_navigation_or_hardware_fidelity')
    passed=len(frames)>=30 and rate>=9.5 and summary['angular_diversity_ratio']>=3.
    Path(args.output).write_text(json.dumps(dict(passed=passed,summary=summary,frames=frames,maps=maps),indent=2)+'\n')
    print(json.dumps(dict(passed=passed,**summary),indent=2))
    return 0 if passed else 1


if __name__=='__main__': raise SystemExit(main())
