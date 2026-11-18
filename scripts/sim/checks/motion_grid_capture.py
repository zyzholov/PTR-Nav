#!/usr/bin/env python3
"""Stationary sensor qualification. Never publishes goals or vehicle commands."""
import argparse
import json
import math
from pathlib import Path
import time

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid,Odometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool

from navigation_trial import motion_grid_sample,yaw_from_quaternion


class Capture(Node):
    def __init__(self):
        super().__init__('measured_motion_grid_capture')
        self.frames=[];self.tracking_frames=[];self.obstacle=[];self.robot=[];self.odom=[]
        self.trigger=self.create_publisher(Bool,'/isaac/nav_test_trigger',10)
        self.create_subscription(OccupancyGrid,'/terrain_nav/live_obstacles_raw',self.on_grid,10)
        self.create_subscription(OccupancyGrid,'/terrain_nav/motion_observations',self.on_tracking_grid,10)
        self.create_subscription(PoseStamped,'/isaac/nav_test_obstacle/pose',self.on_obstacle,10)
        self.create_subscription(Odometry,'/isaac/ground_truth/odometry',self.on_robot,10)
        self.create_subscription(Odometry,'/Odometry',self.on_odom,10)

    def on_grid(self,msg):
        if len(self.frames)>=1500:raise ValueError('motion capture overflow')
        self.frames.append(motion_grid_sample(msg))

    def on_tracking_grid(self,msg):
        if len(self.tracking_frames)>=1500:raise ValueError('tracking observation capture overflow')
        self.tracking_frames.append(motion_grid_sample(msg))

    def on_obstacle(self,msg):
        self.obstacle.append([msg.header.stamp.sec+1e-9*msg.header.stamp.nanosec,
                              msg.pose.position.x,msg.pose.position.y])

    def on_robot(self,msg):
        self.robot.append([msg.header.stamp.sec+1e-9*msg.header.stamp.nanosec,
                           msg.pose.pose.position.x,msg.pose.pose.position.y,
                           yaw_from_quaternion(msg.pose.pose.orientation)])

    def on_odom(self,msg):
        self.odom.append([msg.header.stamp.sec+1e-9*msg.header.stamp.nanosec,
                          msg.pose.pose.position.x,msg.pose.pose.position.y,
                          yaw_from_quaternion(msg.pose.pose.orientation)])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('output must not exist')
    rclpy.init();node=Capture()
    try:
        deadline=time.monotonic()+15
        while time.monotonic()<deadline:
            rclpy.spin_once(node,timeout_sec=.05)
            if len(node.frames)>=20 and node.obstacle and node.robot and node.odom and node.trigger.get_subscription_count():break
        else:raise RuntimeError(f'missing capture inputs: grids={len(node.frames)}, obstacle={len(node.obstacle)}, '
            f'robot={len(node.robot)}, odom={len(node.odom)}, trigger_subscribers={node.trigger.get_subscription_count()}')
        trigger_stamp=node.frames[-1]['stamp_s']
        node.trigger.publish(Bool(data=True))
        end=time.monotonic()+8
        while time.monotonic()<end:rclpy.spin_once(node,timeout_sec=.05)
        moved=max(math.hypot(p[1]-node.robot[0][1],p[2]-node.robot[0][2]) for p in node.robot)
        crossed=any(y<-2.5 for _,_,y in node.obstacle)
        accepted=moved<.05 and crossed and len(node.frames)>=60
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as output:
            json.dump({'schema_version':1,'scope':'stationary measured-column capture; not prediction or navigation acceptance',
                'overflow':False,'accepted':accepted,'trigger_stamp_s':trigger_stamp,
                'robot_motion_max_m':moved,'frames':node.frames,
                'tracking_frames':node.tracking_frames,
                'tracking_scope':'actual tracker input; upper witnesses do not authorize ground/free space',
                'obstacle_truth_assessment_only':node.obstacle,'robot_truth_assessment_only':node.robot,
                'robot_odom_assessment_only':node.odom},output,allow_nan=False)
        print(json.dumps({'accepted':accepted,'scope':'stationary sensor capture only; not prediction acceptance',
            'frames':len(node.frames),'obstacle_crossed':crossed,'robot_motion_max_m':moved}))
        return 0 if accepted else 1
    finally:node.destroy_node();rclpy.shutdown()


if __name__=='__main__':raise SystemExit(main())
