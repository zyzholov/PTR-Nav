#!/usr/bin/env python3
"""Verify that terrain_nav stops a moving vehicle after risk-map publication stalls."""

import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ["RMW_IMPLEMENTATION"] = os.environ.get("HUNTER_ISAAC_RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_srvs.srv import SetBool


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


def run(node, pause_service, output=None):
    client=node.create_client(SetBool,pause_service)
    if not client.wait_for_service(timeout_sec=5):raise RuntimeError("missing qualification map relay")
    def pause(value):
        request=SetBool.Request();request.data=value
        future=client.call_async(request)
        deadline=time.monotonic()+3
        while not future.done() and time.monotonic()<deadline:rclpy.spin_once(node,timeout_sec=.05)
        if not future.done() or not future.result().success:raise RuntimeError("map relay did not acknowledge fault state")
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
        pause(True)
        while time.monotonic() - stopped_at < 2.5:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        pause(False)
    resumed_at = time.monotonic()

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

    resume_start = node.odom.pose.pose.position
    resume_xy = (resume_start.x, resume_start.y)
    resumed_commands = []
    progress = 0.0
    deadline = resumed_at + 12.0
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        resumed_commands = [(t, v) for t, v in node.commands if t >= resumed_at and abs(v) > 0.2]
        p = node.odom.pose.pose.position
        progress = math.hypot(p.x - resume_xy[0], p.y - resume_xy[1])
        if len(resumed_commands) >= 5 and progress > 0.15:
            break

    recovered = len(resumed_commands) >= 5 and progress > 0.15
    metrics = {
        "fault_scope":"only risk-map delivery, sensors and CUDA processes remain running",
        "passed": recovered,
        "late_max_command_mps": max(late),
        "late_truth_speed_mps": late_truth[-1],
        "resume_first_motion_wall_s": resumed_commands[0][0] - resumed_at if resumed_commands else None,
        "resume_progress_m": progress,
        "pause_duration_wall_s": resumed_at - stopped_at,
    }
    if output:
        Path(output).write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)
    if not recovered:
        print("FAIL: fresh risk map did not recover motion without a new goal", flush=True)
        return 1
    print("PASS: stale risk map stopped motion; fresh data replanned and recovered", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pause-service",default="/qualification/pause_map_relay")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rclpy.init()
    node = Probe()
    try:
        return run(node, args.pause_service, args.output)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
