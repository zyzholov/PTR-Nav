#!/usr/bin/env python3
"""Publish a forward goal and verify that the vehicle actually moves toward it."""

import argparse
import math
import os
from pathlib import Path
import subprocess
import time

# Keep the probe on the same DDS implementation as the Isaac/ROS launcher.
# This must be set before importing rclpy because the middleware is selected
# during rclpy initialization.
os.environ["RMW_IMPLEMENTATION"] = os.environ.get("HUNTER_ISAAC_RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path as NavPath
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from visualization_msgs.msg import Marker, MarkerArray


class GoalProbe(Node):
    def __init__(self):
        super().__init__("hunter_goal_probe")
        self.last_odom = None
        self.last_truth = None
        self.odom_received = 0.0
        self.truth_received = 0.0
        self.nonzero_commands = 0
        self.steering_commands = 0
        self.max_speed_command = 0.0
        self.min_speed_command = 0.0
        self.max_steering_command = 0.0
        self.first_nonzero_command_wall = None
        self.command_times = []
        self.path_seen = False
        self.goal_seen = False
        self.path_cleared = False
        self.goal_cleared = False
        self.visualization_error = None
        self.create_subscription(Odometry, "/Odometry", self.on_odom, 10)
        self.create_subscription(Odometry, "/isaac/ground_truth/odometry", self.on_truth, 10)
        self.create_subscription(Twist, "/cmd_vel", self.on_command, 10)
        self.create_subscription(NavPath, "/terrain_nav/path", self.on_path, 1)
        self.create_subscription(
            MarkerArray, "/terrain_nav/active_goal", self.on_goal_marker,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)
        self.test_start_wall = None

    def on_odom(self, msg):
        self.last_odom = msg
        self.odom_received = time.monotonic()

    def on_truth(self, msg):
        self.last_truth = msg
        self.truth_received = time.monotonic()

    def on_command(self, msg):
        if self.test_start_wall is None or time.monotonic() < self.test_start_wall:
            return
        speed = abs(msg.linear.x)
        command_wall = time.monotonic()
        self.command_times.append(command_wall)
        if len(self.command_times) > 250:
            self.command_times.pop(0)
        self.min_speed_command = min(self.min_speed_command, float(msg.linear.x))
        steering = abs(msg.angular.z)
        self.max_speed_command = max(self.max_speed_command, speed)
        self.max_steering_command = max(self.max_steering_command, steering)
        if speed > 0.03:
            self.nonzero_commands += 1
            if self.first_nonzero_command_wall is None:
                self.first_nonzero_command_wall = command_wall
        if steering > 0.03:
            self.steering_commands += 1

    def on_path(self, msg):
        if self.test_start_wall is None:
            return
        if msg.poses:
            self.path_seen = True
            self.path_cleared = False
            if msg.header.frame_id != "map" or not all(
                math.isfinite(value)
                for pose in msg.poses
                for value in (pose.pose.position.x, pose.pose.position.y, pose.pose.position.z)
            ):
                self.visualization_error = "planned path has an invalid frame or non-finite position"
        elif self.path_seen:
            self.path_cleared = True

    def on_goal_marker(self, msg):
        if self.test_start_wall is None:
            return
        for marker in msg.markers:
            if marker.action == Marker.DELETEALL:
                self.goal_cleared = self.goal_seen
            elif marker.action == Marker.ADD:
                self.goal_seen = True
                self.goal_cleared = False
                if marker.header.frame_id != "map" or not math.isfinite(marker.pose.position.z):
                    self.visualization_error = "active goal has an invalid frame or non-finite height"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance", type=float, default=2.0)
    parser.add_argument("--lateral", type=float, default=0.0, help="Goal offset to the vehicle's left")
    parser.add_argument("--behind", action="store_true", help="Place the goal behind the vehicle and require reverse motion")
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--reach-distance", type=float, default=0.60)
    parser.add_argument("--odom-only", action="store_true", help="Explicitly skip the Isaac truth check")
    parser.add_argument("--check-visualization", action="store_true",
                        help="Require a valid path/goal display and clear them on arrival")
    parser.add_argument("--rviz-snapshot", type=Path,
                        help="Capture RViz while this goal has an active path")
    parser.add_argument(
        "--settle-time",
        type=float,
        default=2.0,
        help="Wall time to let odometry initialize before choosing the test start pose",
    )
    parser.add_argument(
        "--min-motion",
        type=float,
        default=0.25,
        help="Minimum XY displacement required before a goal can be reported as reached",
    )
    parser.add_argument(
        "--min-commands",
        type=int,
        default=5,
        help="Minimum nonzero cmd_vel samples required for a successful closed-loop check",
    )
    parser.add_argument(
        "--max-first-command-latency",
        type=float,
        default=2.0,
        help="Maximum wall latency from goal publication to the first nonzero cmd_vel",
    )
    args = parser.parse_args()
    if (
        not all(math.isfinite(v) for v in (args.distance, args.lateral, args.reach_distance, args.timeout, args.settle_time, args.min_motion))
        or args.reach_distance <= 0
        or args.distance <= args.reach_distance
        or args.timeout <= 0
        or args.settle_time < 0
        or args.min_motion <= 0
        or args.min_commands < 1
        or not math.isfinite(args.max_first_command_latency)
        or args.max_first_command_latency <= 0
    ):
        parser.error(
            "distance must exceed reach-distance; timeout, min-motion and min-commands must be positive; "
            "settle-time cannot be negative"
        )

    rclpy.init()
    node = GoalProbe()
    try:
        deadline = time.monotonic() + min(15.0, args.timeout)
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
            now = time.monotonic()
            if (now - node.odom_received < 1.0
                    and (args.odom_only or now - node.truth_received < 1.0)
                    and node.goal_pub.get_subscription_count() > 0):
                break
        else:
            print("missing live odometry/truth or /goal_pose subscriber; is the full stack running?", flush=True)
            return 1

        # FAST-LIO can publish a frame initialization jump during its first
        # scans.  Establish the test origin only after that transient window.
        settle_deadline = time.monotonic() + args.settle_time
        while time.monotonic() < settle_deadline:
            rclpy.spin_once(node, timeout_sec=0.1)

        odom = node.last_odom
        if odom.header.frame_id != "map":
            print(f"unsupported odometry frame {odom.header.frame_id!r}; goal requires map coordinates", flush=True)
            return 1
        p = odom.pose.pose.position
        q = odom.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        signed_distance = -args.distance if args.behind else args.distance
        goal = PoseStamped()
        goal.header = odom.header
        goal.header.frame_id = "map"
        goal.pose.position.x = p.x + signed_distance * math.cos(yaw) - args.lateral * math.sin(yaw)
        goal.pose.position.y = p.y + signed_distance * math.sin(yaw) + args.lateral * math.cos(yaw)
        goal.pose.position.z = p.z
        goal.pose.orientation = q
        print(f"goal=({goal.pose.position.x:.2f},{goal.pose.position.y:.2f}) start=({p.x:.2f},{p.y:.2f})", flush=True)

        start_x = float(p.x)
        start_y = float(p.y)
        truth_start = node.last_truth
        truth_start_xy = None
        truth_goal_xy = None
        if truth_start is not None:
            truth_p = truth_start.pose.pose.position
            truth_start_xy = (float(truth_p.x), float(truth_p.y))
            truth_q = truth_start.pose.pose.orientation
            truth_yaw = math.atan2(
                2.0 * (truth_q.w * truth_q.z + truth_q.x * truth_q.y),
                1.0 - 2.0 * (truth_q.y * truth_q.y + truth_q.z * truth_q.z),
            )
            # The map and Isaac world origins/yaws differ. Construct the same
            # relative forward target in each frame, never compare raw poses.
            truth_goal_xy = (
                truth_start_xy[0] + signed_distance * math.cos(truth_yaw) - args.lateral * math.sin(truth_yaw),
                truth_start_xy[1] + signed_distance * math.sin(truth_yaw) + args.lateral * math.cos(truth_yaw),
            )
            print(
                f"truth_start=({truth_start_xy[0]:.2f},{truth_start_xy[1]:.2f}) "
                "(used as an independent motion check)",
                flush=True,
            )
        node.test_start_wall = time.monotonic()

        # Publish for a short discovery window instead of relying on a single
        # sample being delivered while the planner is joining the graph.
        publish_until = time.monotonic() + 2.0
        while time.monotonic() < publish_until:
            node.goal_pub.publish(goal)
            rclpy.spin_once(node, timeout_sec=0.05)

        if args.rviz_snapshot is not None:
            capture = Path(__file__).resolve().parents[1] / "sim/checks/rviz_snapshot.py"
            subprocess.run(
                ["python3", str(capture), "--output", str(args.rviz_snapshot)],
                check=True, timeout=15,
            )

        deadline = time.monotonic() + args.timeout
        best_distance = float("inf")
        best_motion = 0.0
        best_truth_motion = 0.0
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
            now = time.monotonic()
            if now - node.odom_received > 1.0 or (not args.odom_only and now - node.truth_received > 1.0):
                print("odometry/truth stopped updating during the goal check", flush=True)
                return 1
            if node.last_odom is None:
                continue
            current = node.last_odom.pose.pose.position
            distance = math.hypot(current.x - goal.pose.position.x, current.y - goal.pose.position.y)
            best_distance = min(best_distance, distance)
            motion = math.hypot(current.x - start_x, current.y - start_y)
            best_motion = max(best_motion, motion)

            truth_motion = 0.0
            truth_error = float("inf")
            if truth_start_xy is not None and node.last_truth is not None:
                truth_p = node.last_truth.pose.pose.position
                truth_motion = math.hypot(
                    float(truth_p.x) - truth_start_xy[0],
                    float(truth_p.y) - truth_start_xy[1],
                )
                truth_error = math.hypot(truth_p.x - truth_goal_xy[0], truth_p.y - truth_goal_xy[1])
                best_truth_motion = max(best_truth_motion, truth_motion)

            if distance <= args.reach_distance:
                command_ok = node.nonzero_commands >= args.min_commands
                direction_ok = (not args.behind) or node.min_speed_command < -0.05
                steering_ok = args.lateral == 0.0 or node.steering_commands >= args.min_commands
                motion_ok = motion >= args.min_motion
                truth_ok = args.odom_only or (truth_motion >= args.min_motion and truth_error <= args.reach_distance)
                first_latency = (
                    node.first_nonzero_command_wall - node.test_start_wall
                    if node.first_nonzero_command_wall is not None else float("inf")
                )
                if command_ok and direction_ok and steering_ok and motion_ok and truth_ok and first_latency <= args.max_first_command_latency:
                    if args.check_visualization:
                        clear_deadline = time.monotonic() + 4.0
                        while time.monotonic() < clear_deadline and not (
                            node.path_cleared and node.goal_cleared
                        ):
                            rclpy.spin_once(node, timeout_sec=0.1)
                        if node.visualization_error or not (
                            node.path_seen and node.goal_seen and node.path_cleared and node.goal_cleared
                        ):
                            print(
                                f"visualization failed: path_seen={node.path_seen} "
                                f"goal_seen={node.goal_seen} path_cleared={node.path_cleared} "
                                f"goal_cleared={node.goal_cleared} error={node.visualization_error}",
                                flush=True,
                            )
                            return 1
                        print("PASS path/goal markers published and cleared on arrival", flush=True)
                    command_rate = 0.0
                    if len(node.command_times) >= 2:
                        span = node.command_times[-1] - node.command_times[0]
                        if span > 1e-6:
                            command_rate = (len(node.command_times) - 1) / span
                    print(
                        f"reached: error={distance:.2f}m nav_motion={motion:.2f}m "
                        f"truth_motion={truth_motion:.2f}m truth_error={truth_error:.2f}m "
                        f"commands={node.nonzero_commands} steering_commands={node.steering_commands} "
                        f"truth_checked={not args.odom_only} min_cmd_v={node.min_speed_command:.2f} "
                        f"first_cmd_latency={first_latency:.3f}s cmd_rate={command_rate:.1f}Hz",
                        flush=True,
                    )
                    return 0
        print(
            f"not reached: best_error={best_distance:.2f}m nonzero_commands={node.nonzero_commands} "
            f"max_command={node.max_speed_command:.2f}m/s "
            f"steering_commands={node.steering_commands} "
            f"max_steering={node.max_steering_command:.2f}rad nav_motion={best_motion:.2f}m "
            f"truth_motion={best_truth_motion:.2f}m min_motion={args.min_motion:.2f}m "
            f"min_cmd_v={node.min_speed_command:.2f} first_cmd_latency="
            f"{(node.first_nonzero_command_wall - node.test_start_wall):.3f}s" if node.first_nonzero_command_wall is not None else
            f"min_cmd_v={node.min_speed_command:.2f} first_cmd_latency=none",
            flush=True,
        )
        return 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
