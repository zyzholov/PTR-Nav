#!/usr/bin/env python3
"""Wait for one usable ROS message from a Hunter stack stage."""

import argparse
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu, PointCloud2


TOPIC_TYPES = {
    "/clock": "clock",
    "/livox/lidar": "cloud",
    "/livox/imu": "imu",
    "/imu/data_raw": "imu",
    "/Odometry": "odom",
    "/isaac/ground_truth/odometry": "odom",
    "/cloud_registered": "cloud",
    "/Laser_map": "cloud",
    "/car_0_terrain_analyzer/fused_map": "grid",
    "/car_0_fsgp_bgk/fused_map": "grid",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("topic")
    parser.add_argument("--type", choices=("clock", "cloud", "imu", "odom", "grid"))
    parser.add_argument("--layer", help="Required layer for SE2Grid messages")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    kind = args.type or TOPIC_TYPES.get(args.topic)
    if kind is None or args.timeout <= 0:
        parser.error("custom topics require --type; timeout must be positive")
    topic_types = {"clock": Clock, "cloud": PointCloud2, "imu": Imu, "odom": Odometry}
    se2_grid_type = None
    if kind == "grid":
        try:
            from se2_grid_msgs.msg import SE2Grid
        except ImportError:
            print(
                "[hunter_stack] ERROR: fused-map checks require the workspace "
                "environment (source install/setup.bash).",
                flush=True,
            )
            return 2
        se2_grid_type = SE2Grid
        topic_types["grid"] = SE2Grid

    rclpy.init()
    node = rclpy.create_node("hunter_stack_wait")
    qos = QoSProfile(depth=10)
    qos.reliability = (
        ReliabilityPolicy.BEST_EFFORT
        if kind in ("clock", "cloud", "imu")
        else ReliabilityPolicy.RELIABLE
    )
    received = []

    def callback(msg):
        if isinstance(msg, PointCloud2):
            if msg.width * msg.height == 0 or not msg.data:
                return
            detail = f"points={msg.width * msg.height}"
        elif se2_grid_type is not None and isinstance(msg, se2_grid_type):
            needed = args.layer or ("bgk_mean" if "fsgp_bgk" in args.topic else "risk")
            if needed not in msg.layers or not msg.data:
                return
            detail = f"layers={len(msg.layers)} required={needed}"
        else:
            detail = "sample received"
        received.append(detail)

    subscription = node.create_subscription(topic_types[kind], args.topic, callback, qos)
    deadline = time.monotonic() + args.timeout
    try:
        while not received and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
        if received:
            print(f"[hunter_stack] Live: {args.topic} ({received[0]})", flush=True)
            return 0
        print(f"[hunter_stack] ERROR: no usable {args.topic} after {args.timeout:g}s", flush=True)
        return 1
    except KeyboardInterrupt:
        return 130
    finally:
        node.destroy_subscription(subscription)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
