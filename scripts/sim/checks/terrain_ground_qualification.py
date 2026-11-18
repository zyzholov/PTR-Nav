#!/usr/bin/env python3
"""Exercise the installed CUDA terrain frontend with controlled real input messages."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.serialization import serialize_message
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import create_cloud_xyz32
from se2_grid_msgs.msg import SE2Grid
from std_msgs.msg import Header

from fsgp_bgk_se2.se2_grid_tools import get_layer_2d,grid_buffer_to_world,matrix_shape


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[3]
    command=[str(root/'install/terrain_analyzer/lib/terrain_analyzer/terrain_analyzer_gpu'),
        '--ros-args','--params-file',str(root/'src/uneven_mapping/terrain_analyzer/params/hunter_fastlio_gpu.yaml'),
        '-p','use_sim_time:=false','-p','elevation.preserve_measured_ground:=true',
        '-r','__ns:=/ground_qualification']
    log=(args.output/'frontend.log').open('w')
    worker=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    rclpy.init();node=rclpy.create_node('terrain_ground_qualification')
    odom_pub=node.create_publisher(Odometry,'/ground_qualification/odom',5)
    cloud_pub=node.create_publisher(PointCloud2,'/ground_qualification/cloud',1)
    received=[]
    node.create_subscription(SE2Grid,'/ground_qualification/fused_map',received.append,1)
    records=[];counter=0
    axis=(np.arange(30)-14.5)*.2
    x,y=np.meshgrid(axis,axis);base=.10*x+.03*y
    patch=(x>.6)&(x<1.4)&(np.abs(y)<.4)

    def send(z,offset=0.):
        nonlocal counter
        counter+=1
        odom=Odometry();odom.header.frame_id='map'
        odom.header.stamp.sec=10+counter//10;odom.header.stamp.nanosec=(counter%10)*100000000
        odom.pose.pose.position.x=offset;odom.pose.pose.position.z=.5
        odom.pose.pose.orientation.w=1.
        odom.pose.covariance[14]=1e-5
        points=np.c_[x.ravel()+offset,y.ravel(),z.ravel()]
        odom_pub.publish(odom)
        # The CUDA node requires matching odometry to have arrived first.
        until=time.monotonic()+.025
        while time.monotonic()<until:rclpy.spin_once(node,timeout_sec=.005)
        cloud_pub.publish(create_cloud_xyz32(Header(frame_id='map',stamp=odom.header.stamp),points))
        deadline=time.monotonic()+3
        while time.monotonic()<deadline:
            if worker.poll() is not None:raise RuntimeError('CUDA frontend exited')
            rclpy.spin_once(node,timeout_sec=.02)
            if received and received[-1].info.header.stamp==odom.header.stamp:return received[-1]
        raise RuntimeError('no timestamp-matched CUDA map')

    def check(name,msg,expected,offset=0.):
        shape=matrix_shape(msg.data[0]);resolution=msg.info.pos_resolution
        left=msg.info.pose.position.x-msg.info.length_x/2
        bottom=msg.info.pose.position.y-msg.info.length_y/2
        cols=np.floor((x+offset-left)/resolution).astype(int)
        rows=np.floor((y-bottom)/resolution).astype(int)
        height=grid_buffer_to_world(get_layer_2d(msg,'elevation',np.nan,shape)[0],msg)[rows,cols]
        observed=grid_buffer_to_world(get_layer_2d(msg,'ground_observed',0.,shape)[0],msg)[rows,cols]
        error=np.abs(height-expected)
        row=dict(case=name,passed=bool(np.isfinite(error).all() and np.max(error)<.015 and (observed>.5).all()),
                 max_height_error_m=float(np.nanmax(error)),measured_fraction=float((observed>.5).mean()))
        records.append(row)
        (args.output/(name+'.cdr')).write_bytes(serialize_message(msg))
        print(json.dumps(row),flush=True)

    try:
        deadline=time.monotonic()+20
        while time.monotonic()<deadline and (not cloud_pub.get_subscription_count() or not odom_pub.get_subscription_count()):
            if worker.poll() is not None:raise RuntimeError('CUDA frontend failed during startup')
            rclpy.spin_once(node,timeout_sec=.1)
        for _ in range(3):msg=send(base)
        check('measured_continuous_slope',msg,base)
        raised=np.where(patch,base+.45,base)
        for _ in range(5):msg=send(raised)
        check('passing_column_preserves_measured_ground',msg,base)
        for _ in range(3):msg=send(base)
        check('vacated_column_ground',msg,base)
        # Rolling into wholly unknown terrain must retain a real static step.
        stepped=np.where(x>.6,base+.45,base)
        for _ in range(3):msg=send(stepped,30.)
        check('newly_measured_static_step',msg,stepped,30.)
        lowered=np.where(patch,stepped-.4,stepped)
        for _ in range(3):msg=send(lowered,30.)
        check('new_lower_surface_is_not_ignored',msg,lowered,30.)
        # Initially unknown tile: ONE high obstacle return poisons a single
        # cell, while surrounding real ground is already observed. A single
        # later scan contains one real return per cell on the lower plane.
        # This exercises the two-kernel enclosure rule, not the old three-scan
        # temporal confirmation or a neighbor-filled observation.
        single=(np.abs(x-.9)<1e-5)&(np.abs(y+.1)<1e-5)
        assert np.count_nonzero(single)==1
        send(np.where(single,base+.4,base),60.)
        msg=send(base,60.)
        check('single_scan_enclosed_ground_retires_initial_high_return',msg,base,60.)
        # The SAME one-cell drop from ordinary surrounding ground is a new
        # pit. Its mixed patch fails planarity; support must remain pending.
        msg=send(np.where(single,base-.4,base),60.)
        shape=matrix_shape(msg.data[0]);resolution=msg.info.pos_resolution
        left=msg.info.pose.position.x-msg.info.length_x/2;bottom=msg.info.pose.position.y-msg.info.length_y/2
        col=int(np.floor((x[single][0]+60.-left)/resolution));row=int(np.floor((y[single][0]-bottom)/resolution))
        pending=grid_buffer_to_world(get_layer_2d(msg,'ground_pending',0.,shape)[0],msg)
        height=grid_buffer_to_world(get_layer_2d(msg,'elevation',np.nan,shape)[0],msg)
        records.append(dict(case='new_pit_single_scan_remains_pending',passed=bool(pending[row,col]>.5 and
            abs(height[row,col]-base[single][0])<.015),pending=float(pending[row,col])))
        (args.output/'new_pit_single_scan_remains_pending.cdr').write_bytes(serialize_message(msg))
        # A wider initially observed column lacks a lower 3x3 historical ring.
        # Only the later REAL scan supplies lower heights in those cells.
        disk=np.hypot(x,y)<.55
        poisoned=np.where(disk,.4,0.)
        for _ in range(3):msg=send(poisoned,90.)
        check('wide_old_column_before_reobservation',msg,poisoned,90.)
        msg=send(np.zeros_like(base),90.)
        check('one_current_floor_scan_retires_wide_old_column',msg,np.zeros_like(base),90.)
        witness=grid_buffer_to_world(get_layer_2d(msg,'ground_reobserved',0.,matrix_shape(msg.data[0]))[0],msg)
        cols=np.floor((x+90-msg.info.pose.position.x+msg.info.length_x/2)/msg.info.pos_resolution).astype(int)
        rows=np.floor((y-msg.info.pose.position.y+msg.info.length_y/2)/msg.info.pos_resolution).astype(int)
        if not (witness[rows[disk],cols[disk]]>.5).all():raise RuntimeError('wide floor replacement lacks real current witnesses')
        # A NEW deep pit is not an old obstacle side returning to its floor.
        msg=send(np.where(disk,-.4,0.),90.)
        pending=grid_buffer_to_world(get_layer_2d(msg,'ground_pending',0.,matrix_shape(msg.data[0]))[0],msg)
        if not (pending[rows[disk],cols[disk]]>.5).all():raise RuntimeError('new deep pit bypassed confirmation')
        if (grid_buffer_to_world(get_layer_2d(msg,'ground_reobserved',0.,matrix_shape(msg.data[0]))[0],msg)[rows[disk],cols[disk]]>.5).any():
            raise RuntimeError('new pit mislabeled as old floor reobservation')
        check('new_deep_pit_not_cleared_by_floor_witness',msg,np.zeros_like(base),90.)
        result=dict(passed=all(r['passed'] for r in records),
            scope='controlled real-message CUDA frontend checks; no physical navigation or calibration claim',
            command=command,cases=records)
        (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
        return int(not result['passed'])
    finally:
        if worker.poll() is None:
            os.killpg(worker.pid,signal.SIGINT)
            try:worker.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(worker.pid,signal.SIGKILL);worker.wait()
        log.close();node.destroy_node();rclpy.shutdown()


if __name__=='__main__':raise SystemExit(main())
