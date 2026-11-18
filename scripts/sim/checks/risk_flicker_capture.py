#!/usr/bin/env python3
"""Stationary source-layer diagnosis; not navigation or probability calibration.

Run beside pipeline_smoke --skip-goal with a boundary fixture. A goal at the
current measured pose arms the explicitly enabled simulation stance. Any
nonzero drive command or >5cm truth motion aborts this observation-only check.
"""
import argparse
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import read_points
from geometry_msgs.msg import PoseStamped,Twist
from se2_grid_msgs.msg import SE2Grid
from fsgp_bgk_se2.se2_grid_tools import layer_refs,multiarray_to_matrix,grid_buffer_to_world
import yaml

ROOT=Path(__file__).resolve().parents[3]
NAMES=('collision_risk','gp_elevation_observed','gp_variance_observed','probability_elevation',
       'probability_variance','probability_gp_used','startup_prior_used','gp_completion_discontinuity_veto',
       'probability_raw_veto','probability_joint_used','probability_initial_affine_basis',
       'probability_detail_max_sources','probability_observation_halo_m','ground_pending')


def rotation(q):
    v=np.array([q.x,q.y,q.z],float);w=q.w;n=np.dot(v,v)+w*w
    if not math.isfinite(n) or n<1e-12:raise ValueError('invalid pose rotation')
    v=v/math.sqrt(n);w/=math.sqrt(n)
    x,y,z=v;cross=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3)+2*w*cross+2*cross@cross


def stamp(msg):return msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
def xyz(msg):
    p=read_points(msg,field_names=('x','y','z'),skip_nans=True)
    return np.c_[p['x'],p['y'],p['z']]


def point_counts(points):
    x,y,z=points.T
    body=(x>=-.38)&(x<=.48)&(abs(y)<=.30)&(z>=-.16)&(z<=.04)
    raised=(x>.48)&(x<1.2)&(abs(y)<.6)&(z>-.20)&(z<.4)
    floor=(x>-.3)&(x<.6)&(abs(y)<.30)&(z>-.32)&(z<-.22)
    return dict(points=len(points),body_box=int(body.sum()),front_raised=int(raised.sum()),near_floor=int(floor.sum()))


class Capture(Node):
    def __init__(self):
        super().__init__('stationary_risk_sources')
        self.odom=None;self.truth=None;self.grid=None;self.clock=None;self.odoms=deque(maxlen=40)
        self.clouds=deque(maxlen=10);self.raw=None;self.command=(0.,0.);self.max_command=0.
        self.create_subscription(Odometry,'/Odometry',self.on_odom,qos_profile_sensor_data)
        self.create_subscription(Odometry,'/isaac/ground_truth/odometry',lambda m:setattr(self,'truth',m),qos_profile_sensor_data)
        self.create_subscription(SE2Grid,'/car_0_fsgp_bgk/fused_map',lambda m:setattr(self,'grid',m),1)
        self.create_subscription(PointCloud2,'/cloud_registered',self.clouds.append,qos_profile_sensor_data)
        self.create_subscription(PointCloud2,'/livox/lidar',lambda m:setattr(self,'raw',m),qos_profile_sensor_data)
        self.create_subscription(Twist,'/cmd_vel',self.on_command,10)
        self.create_subscription(Clock,'/clock',lambda m:setattr(self,'clock',m.clock),1)
        self.goal=self.create_publisher(PoseStamped,'/goal_pose',10)

    def on_odom(self,msg):self.odom=msg;self.odoms.append(msg)
    def on_command(self,msg):
        self.command=(msg.linear.x,msg.angular.z);self.max_command=max(self.max_command,abs(msg.linear.x))

    def inputs_fresh(self):
        if self.clock is None or self.odom is None or self.truth is None or self.grid is None:return False
        now=self.clock.sec+self.clock.nanosec*1e-9
        s=self.grid.info.header.stamp;grid_time=s.sec+s.nanosec*1e-9
        return now>0 and all(0<=now-t<=.6 for t in (stamp(self.odom),stamp(self.truth),grid_time))

    def sample(self,msg,Rbl,Tbl):
        refs=layer_refs(msg)
        def layer(n,k=0):
            ref=refs[n]
            return grid_buffer_to_world(multiarray_to_matrix(msg.data[ref.data_indices[min(k,len(ref.data_indices)-1)]]),msg)
        yaw=math.atan2(rotation(self.odom.pose.pose.orientation)[1,0],rotation(self.odom.pose.pose.orientation)[0,0])
        k=int((yaw%(2*math.pi))/msg.info.yaw_resolution)
        position=self.odom.pose.pose.position;res=msg.info.pos_resolution
        x0=msg.info.pose.position.x-msg.info.length_x/2;y0=msg.info.pose.position.y-msg.info.length_y/2
        arrays={n:layer(n) for n in NAMES if n in refs}
        bounds=np.stack([layer('failure_probability_upper',j) for j in range(len(refs['failure_probability_upper'].data_indices))])
        points=[]
        for dx in (.6,1.,2.5):
            x=position.x+dx*math.cos(yaw);y=position.y+dx*math.sin(yaw)
            col=int(math.floor((x-x0)/res));row=int(math.floor((y-y0)/res))
            if not 0<=row<bounds.shape[1] or not 0<=col<bounds.shape[2]:raise RuntimeError('probe outside map')
            points.append(dict(offset_m=dx,xy=[x0+(col+.5)*res,y0+(row+.5)*res],
                values={n:float(a[row,col]) if math.isfinite(a[row,col]) else None for n,a in arrays.items()},
                probability_by_yaw=[float(v) if math.isfinite(v) else None for v in bounds[:,row,col]]))
        current=np.stack([arrays[n] if n in arrays else np.full(bounds.shape[1:],np.nan) for n in NAMES])
        # Keep measured values at the native resolution; no display smoothing.
        ri=max(0,int((position.y-2.-y0)/res));rj=min(bounds.shape[1],ri+21)
        ci=max(0,int((position.x-1.5-x0)/res));cj=min(bounds.shape[2],ci+31)
        record=dict(stamp_s=msg.info.header.stamp.sec+msg.info.header.stamp.nanosec*1e-9,
            yaw_rad=yaw,yaw_index=k,probes=points,crop_origin=[x0+ci*res,y0+ri*res],resolution=res)
        if self.raw is not None:record['raw_lidar_base_counts']=point_counts(xyz(self.raw)@Rbl.T+Tbl)
        if self.clouds and self.odoms:
            cloud=self.clouds[-1];odom=min(self.odoms,key=lambda o:abs(stamp(o)-stamp(cloud)))
            age=abs(stamp(odom)-stamp(cloud));record['registered_pose_alignment_s']=age
            if age<=.05:
                p=odom.pose.pose.position;t=np.array([p.x,p.y,p.z]);R=rotation(odom.pose.pose.orientation)
                record['registered_base_counts']=point_counts((xyz(cloud)-t)@R)
        return record,current[:,ri:rj,ci:cj],bounds[[0,35,k],ri:rj,ci:cj]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--duration',type=float,default=15.)
    parser.add_argument('--timeout',type=float,default=90.)
    args=parser.parse_args()
    if not 5<=args.duration<=25:parser.error('duration must be 5..25 seconds')
    args.output.mkdir(parents=True,exist_ok=False)
    cfg=yaml.safe_load((ROOT/'src/FAST_LIO/config/isaac_hunter_mid360_pointcloud.yaml').read_text())['/**']['ros__parameters']
    Ril=np.array(cfg['mapping']['extrinsic_R']).reshape(3,3);Til=np.array(cfg['mapping']['extrinsic_T'])
    Rib=np.array(cfg['frame']['body_pose_R']).reshape(3,3);Tib=np.array(cfg['frame']['body_pose_T'])
    Rbl=Rib.T@Ril;Tbl=Rib.T@(Til-Tib)
    rclpy.init();node=Capture();rows=[];arrays=[];prob=[];failure=None;start=time.monotonic()
    try:
        while time.monotonic()-start<args.timeout:
            rclpy.spin_once(node,timeout_sec=.05)
            planners=node.get_subscriptions_info_by_topic('/goal_pose')
            if node.inputs_fresh() and any(p.node_name=='terrain_nav_node' for p in planners):break
        else:raise RuntimeError('no complete stationary pipeline')
        original=node.truth.pose.pose.position;start_truth=np.array([original.x,original.y,original.z])
        goal=PoseStamped();goal.header.frame_id=node.odom.header.frame_id;goal.header.stamp=node.clock
        goal.pose=node.odom.pose.pose;node.goal.publish(goal)
        # Observe the transition as well as the later model, without driving.
        start=time.monotonic();last=None
        while time.monotonic()-start<args.duration:
            rclpy.spin_once(node,timeout_sec=.02)
            p=node.truth.pose.pose.position
            if np.linalg.norm(np.array([p.x,p.y,p.z])-start_truth)>.05 or node.max_command>.01:
                raise RuntimeError('stationary contract violated; no navigation result is inferred')
            if node.grid is not None:
                key=(node.grid.info.header.stamp.sec,node.grid.info.header.stamp.nanosec)
                if key!=last:
                    record,a,b=node.sample(node.grid,Rbl,Tbl);rows.append(record);arrays.append(a);prob.append(b);last=key
        if len(rows)<40:raise RuntimeError('insufficient distinct source frames')
        # No claim that the front is repaired if the intended assumption never
        # activated (e.g. a stale diagnostic goal). Keep the failed trace too.
        if not any(q['values'].get('probability_initial_affine_basis')==1.
                   for r in rows for q in r['probes']):
            raise RuntimeError('explicit initial stance was never qualified; front comparison invalid')
    except Exception as exc:failure=str(exc)
    finally:
        if failure:node.create_publisher(Twist,'/cmd_vel',10).publish(Twist())
        node.destroy_node();rclpy.shutdown()
    payload=dict(scope='stationary diagnostic, not navigation or empirical probability calibration',
        accepted=failure is None,failure=failure,maximum_speed_command_mps=node.max_command,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        layer_names=NAMES,probability_array_slices=['yaw0','yaw35','current'],samples=rows,
        geometry_scope='box counts exclude wheels/mast; registered counts require <=50ms pose alignment; relative probes track the stationary car, not exact ground-truth object IDs')
    (args.output/'trace.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    if arrays:np.savez_compressed(args.output/'layers.npz',layers=np.stack(arrays),probability=np.stack(prob))
    print(json.dumps(dict(accepted=not failure,failure=failure,frames=len(rows),output=str(args.output))),flush=True)
    return int(failure is not None)


if __name__=='__main__':raise SystemExit(main())
