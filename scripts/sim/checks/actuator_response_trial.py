#!/usr/bin/env python3
"""Short unloaded simulation response qualification; NOT a navigation trial.

Commands depend only on elapsed simulation time. Independent physical poses
are recorded for evaluation, never used to steer or generate a route. There
must be no other command publisher or autonomous planner in this ROS domain.
"""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

SCHEDULE=((0.,0.,0.),(2.,1.,0.),(4.5,1.,.20),(5.5,1.,0.),
          (7.,1.,-.20),(8.,1.,0.),(9.5,0.,0.))

def response_schedule(speed=1.,steering=.2):
    if not math.isfinite(speed) or not 0<speed<=20./3.6:raise ValueError('speed outside NEOS domain')
    if not math.isfinite(steering) or not 0<steering<=.31432913231137:raise ValueError('steering outside NEOS domain')
    return tuple((t,speed if v else 0.,s*(steering/.2))for t,v,s in SCHEDULE)

class ResponseTrial(Node):
    def __init__(self,speed=1.,steering=.2):
        super().__init__('simulation_actuator_response_trial',parameter_overrides=[Parameter('use_sim_time',value=True)])
        self.commands=[];self.truth=[];self.start=None;self.done=False;self.armed=False
        self.schedule=response_schedule(speed,steering)
        self.publisher=self.create_publisher(Twist,'/cmd_vel',10)
        self.subscription=self.create_subscription(Odometry,'/isaac/ground_truth/odometry',self.observe,50)
        self.timer=self.create_timer(.02,self.control)
    def observe(self,msg):
        p=msg.pose.pose.position;q=msg.pose.pose.orientation
        yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
        stamp=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
        if not self.truth or stamp>self.truth[-1][0]:self.truth.append([stamp,p.x,p.y,p.z,yaw])
    def control(self):
        if self.done or not self.armed:self.publisher.publish(Twist());return
        if not self.truth:return
        stamp=self.get_clock().now().nanoseconds*1e-9
        if self.start is None:self.start=stamp
        elapsed=stamp-self.start;speed=steer=0.
        for beginning,v,s in self.schedule:
            if elapsed>=beginning:speed,steer=v,s
        if stamp-self.truth[-1][0]>.2:speed=steer=0.;self.done=True
        message=Twist();message.linear.x=speed;message.angular.z=steer;self.publisher.publish(message)
        self.commands.append([stamp,speed,steer])
        if elapsed>=12.5:self.done=True
    def stop(self):
        for _ in range(5):self.publisher.publish(Twist());rclpy.spin_once(self,timeout_sec=.02)

def analyze(truth,commands,start,steering=.2):
    poses=np.asarray(truth);times=poses[:,0];yaw=np.unwrap(poses[:,4])
    speed=np.gradient(poses[:,1],times)*np.cos(yaw)+np.gradient(poses[:,2],times)*np.sin(yaw)
    rate=np.gradient(yaw,times);events=[]
    for elapsed,kind,request in ((4.5,'left',steering),(7.,'right',-steering),(9.5,'brake',0.)):
        stamp=start+elapsed;before=(times>=stamp-.2)&(times<=stamp-.05)
        baseline=float(np.median(speed[before] if kind=='brake' else rate[before]))
        expected=-baseline if kind=='brake' else float(np.median(speed[before]))*math.tan(request)/.43-baseline
        measured=speed if kind=='brake' else rate
        onset=None
        for i in np.where((times>=stamp)&(times<=stamp+.6))[0]:
            following=np.where((times>=times[i])&(times<=times[i]+.055))[0]
            if len(following)>=2 and np.all(np.sign(expected)*(measured[following]-baseline)>=max(.04,.15*abs(expected))):
                onset=float(times[i]-stamp);break
        events.append(dict(kind=kind,request_stamp_s=stamp,baseline=baseline,expected_change=expected,
            first_sustained_response_s=onset,central_difference_time_reserve_s=.05))
    stopped=speed[times>=start+11.5]
    return dict(events=events,maximum_speed_mps=float(np.max(abs(speed))),
        terminal_speed_max_mps=float(np.max(abs(stopped))) if len(stopped)else None,
        scope=__doc__,limitation='One near-flat unloaded run; publication is not plant reception; not a worst-case DDS, terrain, real-vehicle or navigation guarantee.')

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--speed',type=float,default=1.)
    parser.add_argument('--steering',type=float,default=.2)
    args=parser.parse_args()
    if args.output.exists():parser.error('output already exists')
    if not math.isfinite(args.speed) or not 0<args.speed<=20./3.6:parser.error('speed must be within the NEOS datasheet maximum')
    if not math.isfinite(args.steering) or not 0<args.steering<=.31432913231137:parser.error('steering outside NEOS domain')
    rclpy.init();node=ResponseTrial(args.speed,args.steering);began=time.monotonic()
    try:
        while time.monotonic()-began<1.:rclpy.spin_once(node,timeout_sec=.05)
        others=[p.node_name for p in node.get_publishers_info_by_topic('/cmd_vel') if p.node_name!=node.get_name()]
        if others:raise RuntimeError('command publishers already present: '+str(others))
        node.armed=True
        while rclpy.ok() and not node.done and time.monotonic()-began<50:rclpy.spin_once(node,timeout_sec=.05)
        node.stop()
        result=analyze(node.truth,node.commands,node.start,args.steering) if node.start and len(node.truth)>10 else dict(error='no valid physical observations')
        result['requested_speed_mps']=args.speed
        result['requested_steering_rad']=args.steering
        args.output.write_text(json.dumps(dict(result=result,truth=node.truth,commands=node.commands,start=node.start),indent=2)+'\n')
        print(json.dumps(result,indent=2))
        return 0 if node.done and all(e['first_sustained_response_s']is not None for e in result.get('events',[])) and len(result.get('events',[]))==3 else 1
    finally:node.destroy_node();rclpy.shutdown()
if __name__=='__main__':raise SystemExit(main())
