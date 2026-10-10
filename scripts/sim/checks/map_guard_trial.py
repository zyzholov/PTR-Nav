#!/usr/bin/env python3
"""Check dynamic ghost clearing without erasing a briefly occluded static post."""

import argparse
import json
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool


def high_points(msg, x_min, x_max, y_min, y_max):
    if not msg.point_step or not msg.data:
        return 0
    offsets = {field.name: field.offset for field in msg.fields}
    if not all(name in offsets for name in ("x", "y", "z")):
        return 0
    dtype = ">f4" if msg.is_bigendian else "<f4"
    count = msg.width * msg.height
    x = np.ndarray((count,), dtype=dtype, buffer=msg.data,
                   offset=offsets["x"], strides=(msg.point_step,))
    y = np.ndarray((count,), dtype=dtype, buffer=msg.data,
                   offset=offsets["y"], strides=(msg.point_step,))
    z = np.ndarray((count,), dtype=dtype, buffer=msg.data,
                   offset=offsets["z"], strides=(msg.point_step,))
    return int(np.count_nonzero(
        (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max) & (z > -0.18)))


def stamp_seconds(msg):
    stamp = msg.header.stamp
    return stamp.sec + stamp.nanosec * 1e-9


class MapGuardTrial(Node):
    def __init__(self):
        super().__init__("hunter_map_guard_trial")
        self.poses = []
        self.maps = []
        self.registered = []
        self.trigger_pub = self.create_publisher(Bool, "/isaac/nav_test_trigger", 10)
        self.create_subscription(PoseStamped, "/isaac/nav_test_obstacle/pose", self.on_pose, 10)
        self.create_subscription(PointCloud2, "/Laser_map", self.on_map, 2)
        self.create_subscription(PointCloud2, "/cloud_registered", self.on_registered, 2)

    def on_pose(self, msg):
        self.poses.append((stamp_seconds(msg), msg.pose.position.x, msg.pose.position.y))

    def on_cloud(self, msg, output):
        output.append((stamp_seconds(msg),
                       high_points(msg, 1.85, 4.15, -0.55, 0.55),
                       high_points(msg, 2.45, 3.55, 1.25, 2.25)))

    def on_map(self, msg):
        self.on_cloud(msg, self.maps)

    def on_registered(self, msg):
        self.on_cloud(msg, self.registered)


def first_time(poses, predicate):
    return next((stamp for stamp, x, y in poses if predicate(x, y)), None)


def evaluate(node):
    move = first_time(node.poses, lambda x, y: y > 0.20)
    hold = first_time(node.poses, lambda x, y: x < 1.75 and 1.55 < y < 1.65)
    leave_hold = first_time(node.poses, lambda x, y: y > 1.75)
    vacated = first_time(node.poses, lambda x, y: y > 1.50)
    finished = first_time(node.poses, lambda x, y: y > 4.40)
    if None in (move, hold, leave_hold, vacated, finished):
        return None

    before = [sample for sample in node.maps if sample[0] < move - 0.1]
    during = [sample for sample in node.maps if hold <= sample[0] < leave_hold]
    after = [sample for sample in node.maps if sample[0] >= finished + 0.5]
    registered_before = [sample for sample in node.registered if sample[0] < move - 0.1]
    registered_during = [sample for sample in node.registered if hold <= sample[0] < leave_hold]
    if not before or not during or not after or not registered_before or not registered_during:
        return None

    baseline_ghost = max(sample[1] for sample in before[-3:])
    baseline_guard = max(sample[2] for sample in before[-3:])
    guard_during_min = min(sample[2] for sample in during)
    guard_after_min = min(sample[2] for sample in after)
    guard_moving_min = min(sample[2] for sample in node.maps
                           if move <= sample[0] <= finished + 1.0)
    registered_guard_before = max(sample[2] for sample in registered_before[-20:])
    registered_guard_during_min = min(sample[2] for sample in registered_during)
    ghost_clear = [round(stamp - vacated, 2) for stamp, ghost, _ in node.maps
                   if 0.0 <= stamp - vacated <= 1.5 and
                   ghost <= max(1, baseline_ghost * 0.2)]
    summary = {
        "baseline_ghost": baseline_ghost,
        "baseline_guard": baseline_guard,
        "guard_during_min": guard_during_min,
        "guard_after_min": guard_after_min,
        "guard_moving_min": guard_moving_min,
        "registered_guard_before": registered_guard_before,
        "registered_guard_during_min": registered_guard_during_min,
        "ghost_clear_delay_s": min(ghost_clear, default=None),
        "hold_map_samples": len(during),
        "map_samples": len(node.maps),
        "guard_map_trace": [[round(stamp, 2), guard] for stamp, _, guard in node.maps],
    }
    passed = (
        baseline_ghost >= 5 and baseline_guard >= 5 and len(during) >= 1 and
        guard_moving_min >= 0.7 * baseline_guard and
        guard_during_min >= 0.7 * baseline_guard and
        guard_after_min >= 0.7 * baseline_guard and
        registered_guard_before >= 5 and
        registered_guard_during_min <= 0.5 * registered_guard_before and
        summary["ghost_clear_delay_s"] is not None
    )
    return summary, passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=55.0)
    args = parser.parse_args()
    rclpy.init()
    node = MapGuardTrial()
    deadline = time.monotonic() + args.timeout
    try:
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if len(node.maps) >= 3 and first_time(node.poses, lambda x, y: y > 0.20) is None:
                node.trigger_pub.publish(Bool(data=True))
            finished = first_time(node.poses, lambda x, y: y > 4.40)
            if finished is not None and any(sample[0] >= finished + 2.0 for sample in node.maps):
                break
        result = evaluate(node)
        if result is None:
            print("FAIL: missing obstacle phases or cloud samples", flush=True)
            return 1
        summary, passed = result
        print(json.dumps(summary), flush=True)
        print("PASS" if passed else "FAIL", flush=True)
        return 0 if passed else 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
