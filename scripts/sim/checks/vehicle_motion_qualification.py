#!/usr/bin/env python3
"""Directed steering/straight-line physics check, without mapping or planner.

Run in an isolated simulation domain. Commands act through the ordinary wheel
controller. Independent ground-truth poses fit circles about the chassis
centre AND the rear axle. No pose-setting, force assist, or authored routes.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from rclpy.qos import qos_profile_sensor_data

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src/isaac_sim_hunter_livo2/scripts'))
from vehicle_geometry import geometry,load_spec,chassis_vertical_bounds


def circle_fit(xy):
    a=np.asarray(xy,dtype=float)
    if len(a)<30 or not np.isfinite(a).all():raise ValueError('insufficient finite circle samples')
    shift=a.mean(axis=0);a=a-shift
    centre,_,rank,_=np.linalg.lstsq(np.c_[2*a,np.ones(len(a))],np.sum(a*a,axis=1),rcond=None)
    if rank<3:raise ValueError('circle is unobservable')
    radius=math.sqrt(centre[2]+sum(centre[:2]**2))
    residual=np.linalg.norm(a-centre[:2],axis=1)-radius
    return dict(radius_m=radius,rms_m=float(np.sqrt(np.mean(residual**2))),samples=len(a))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('output already exists')
    rclpy.init();node=rclpy.create_node('vehicle_physics_qualification')
    state={'clock':0.,'pose':None};records=[];commands=[]
    def pose(m):
        q=m.pose.pose.orientation;v=m.pose.pose.position
        yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
        t=m.header.stamp.sec+m.header.stamp.nanosec*1e-9
        state['pose']=(t,v.x,v.y,v.z,yaw,m.twist.twist.linear.x)
    def clock(m):state['clock']=m.clock.sec+m.clock.nanosec*1e-9
    node.create_subscription(Odometry,'/isaac/ground_truth/odometry',pose,qos_profile_sensor_data)
    node.create_subscription(Clock,'/clock',clock,qos_profile_sensor_data)
    publisher=node.create_publisher(Twist,'/cmd_vel',10);g=geometry();spec=load_spec()
    phases=[('settle',0.,0.,2.),('straight',.4,0.,6.),('stop',0.,0.,2.),
            ('left',.35,g['max_steer'],20.),('stop',0.,0.,2.),
            ('right',.35,-g['max_steer'],20.),('stop',0.,0.,2.),
            ('reverse_left',-.3,g['max_steer'],12.),('stop',0.,0.,2.),
            ('reverse_right',-.3,-g['max_steer'],12.),('final_stop',0.,0.,3.)]
    started=time.monotonic();phase_start=None;index=0;last_publish=0.;last_record=-1.;failure=None
    try:
        while time.monotonic()-started<150.:
            rclpy.spin_once(node,timeout_sec=.01)
            if state['pose'] is None or state['clock']<=0 or publisher.get_subscription_count()<1:continue
            others=[x.node_name for x in node.get_publishers_info_by_topic('/cmd_vel') if x.node_name!=node.get_name()]
            if others:raise RuntimeError('another command publisher is active: '+str(others))
            now=state['clock']
            if phase_start is None:phase_start=now
            if now-phase_start>=phases[index][3]:
                index+=1;phase_start=now
                if index==len(phases):break
                print('PHASE '+phases[index][0],flush=True)
            name,speed,steer,_=phases[index]
            if time.monotonic()-last_publish>=.02:
                cmd=Twist();cmd.linear.x=speed;cmd.angular.z=steer;publisher.publish(cmd)
                commands.append([now,name,speed,steer]);last_publish=time.monotonic()
            if state['pose'][0]!=last_record:
                records.append([name,now-phase_start,*state['pose']]);last_record=state['pose'][0]
        else:failure='wall timeout or missing physics input'
        checks={}
        for name in ('left','right','reverse_left','reverse_right'):
            points=np.asarray([r[2:] for r in records if r[0]==name and r[1]>=3.],float)
            if len(points)<30:checks[name]={'passed':False,'reason':'missing settled samples'};continue
            yaw=np.unwrap(points[:,4]);base=points[:,1:3]
            heading=np.c_[np.cos(yaw),np.sin(yaw)]
            centre=base+g['centre_x']*heading;rear=base-spec['rear_axle_offset_m']*heading
            cf,rf=circle_fit(centre),circle_fit(rear)
            expected_sign=1 if name in ('left','reverse_right') else -1
            passed=abs(cf['radius_m']-spec['center_turn_radius_m'])<=.10 and cf['rms_m']<=.03 and (yaw[-1]-yaw[0])*expected_sign>.8
            checks[name]=dict(passed=bool(passed),chassis_centre=cf,rear_axle=rf,yaw_change_rad=float(yaw[-1]-yaw[0]))
        straight=np.asarray([r[2:]for r in records if r[0]=='straight' and r[1]>=1.5],float)
        if len(straight)>30:
            yaw0=straight[0,4];d=straight[-1,1:3]-straight[0,1:3]
            lateral=float(-math.sin(yaw0)*d[0]+math.cos(yaw0)*d[1]);turn=float(np.unwrap(straight[:,4])[-1]-yaw0)
            checks['straight']=dict(passed=abs(lateral)<.04 and abs(turn)<.04,lateral_drift_m=lateral,yaw_drift_rad=turn)
        else:checks['straight']=dict(passed=False,reason='missing samples')
        final=[r for r in records if r[0]=='final_stop' and r[1]>=2.]
        checks['stopped']=dict(passed=bool(final) and max(abs(r[-1])for r in final)<.03)
        if final:
            # This directed fixture uses the explicitly authored -20mm floor.
            base=float(np.median([r[5] for r in final]));bottom,top=chassis_vertical_bounds()
            clearance=base+bottom+.02;height=base+top+.02
            checks['static_clearance']=dict(passed=abs(clearance-spec['ground_clearance_m'])<.005,
                measured_m=clearance,target_m=spec['ground_clearance_m'])
            checks['static_height']=dict(passed=abs(height-spec['height_m'])<.005,
                measured_m=height,target_m=spec['height_m'])
        result=dict(scope=__doc__,specification=spec,geometry=g,checks=checks,
            passed=failure is None and all(x['passed']for x in checks.values()),failure=failure,
            columns=['phase','phase_elapsed_s','stamp_s','x','y','z','yaw','body_speed'],poses=records,commands=commands,
            limitation='flat-ground empty-load sandbox; payload/COM/inertia and actual chassis radius-reference definition are uncalibrated')
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items()if k not in ('poses','commands')},ensure_ascii=False),flush=True)
        return 0 if result['passed'] else 1
    finally:
        for _ in range(5):publisher.publish(Twist());rclpy.spin_once(node,timeout_sec=.02)
        node.destroy_node();rclpy.shutdown()


if __name__=='__main__':raise SystemExit(main())
