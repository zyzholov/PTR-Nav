#!/usr/bin/env python3
"""Verify that terrain_nav stops a moving vehicle after risk-map publication stalls."""

import argparse
import math
import os
import signal
import time

os.environ["RMW_IMPLEMENTATION"] = os.environ.get("HUNTER_ISAAC_RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node


def yaw_from_quaternion(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class Probe(Node):
    def __init__(self):
        super().__init__("hunter_stale_map_trial")
        self.odom = None
        self.commands = []
        self.truth_speeds = []
        self.create_subscription(Odometry, "/Odometry", self.on_odom, 10)
        self.create_subscription(Odometry, "/isaac/ground_truth/odometry", self.on_truth, 10)
        self.create_subscription(Twist, "/cmd_vel", self.on_command, 10)
        self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)

    def on_odom(self, msg):
        self.odom = msg

    def on_command(self, msg):
        self.commands.append((time.monotonic(), msg.linear.x))

    def on_truth(self, msg):
        velocity = msg.twist.twist.linear
        self.truth_speeds.append((time.monotonic(), math.hypot(velocity.x, velocity.y)))


def run(node, terrain_pgid):
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.odom is not None and node.goal_pub.get_subscription_count() > 0:
            break
    else:
        print("FAIL: no odometry or goal subscriber", flush=True)
        return 1

    p = node.odom.pose.pose.position
    q = node.odom.pose.pose.orientation
    yaw = yaw_from_quaternion(q)
    goal = PoseStamped()
    goal.header.frame_id = "map"
    goal.pose.position.x = p.x + 5.0 * math.cos(yaw)
    goal.pose.position.y = p.y + 5.0 * math.sin(yaw)
    goal.pose.orientation = q

    deadline = time.monotonic() + 12.0
    while time.monotonic() < deadline:
        goal.header.stamp = node.odom.header.stamp
        node.goal_pub.publish(goal)
        rclpy.spin_once(node, timeout_sec=0.1)
        if sum(abs(v) > 0.2 for _, v in node.commands) >= 5:
            break
    else:
        print("FAIL: planner did not drive before map fault", flush=True)
        return 1

    stopped_at = time.monotonic()
    try:
        os.killpg(terrain_pgid, signal.SIGSTOP)
        while time.monotonic() - stopped_at < 2.5:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        os.killpg(terrain_pgid, signal.SIGCONT)

    after = [(t - stopped_at, v) for t, v in node.commands if t >= stopped_at]
    late = [abs(v) for dt, v in after if dt >= 1.5]
    late_truth = [v for t, v in node.truth_speeds if t - stopped_at >= 2.0]
    print(f"pre_fault_nonzero={sum(abs(v) > 0.2 for t, v in node.commands if t < stopped_at)} "
          f"post_fault_samples={len(after)} late_samples={len(late)} "
          f"late_max_cmd_speed={max(late) if late else None} "
          f"late_truth_speed={late_truth[-1] if late_truth else None}", flush=True)
    if len(late) < 5 or max(late) > 0.03:
        print("FAIL: planner continued commanding motion with a stale risk map", flush=True)
        return 1
    if not late_truth or late_truth[-1] > 0.15:
        print("FAIL: vehicle was still moving after the stale-map stop", flush=True)
        return 1
    print("PASS: stale risk map caused /cmd_vel to stop", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terrain-pgid", type=int, required=True)
    args = parser.parse_args()
    rclpy.init()
    node = Probe()
    try:
        return run(node, args.terrain_pgid)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
