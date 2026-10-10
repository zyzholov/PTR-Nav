#!/usr/bin/env python3
"""Verify that optional D455 streams contain nonblank image/depth samples."""

import argparse
from pathlib import Path
import time

import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, PointCloud2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--save-color", type=Path, help="Write the received RGB frame to a PNG")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("timeout must be positive")

    rclpy.init()
    node = rclpy.create_node("hunter_camera_frame_probe")
    samples = {}
    topics = (
        ("color", Image, "/camera/camera/color/image_raw"),
        ("depth", Image, "/camera/camera/depth/image_rect_raw"),
        ("points", PointCloud2, "/camera/camera/depth/color/points"),
        ("info", CameraInfo, "/camera/camera/color/camera_info"),
    )
    subscriptions = []
    for name, msg_type, topic in topics:
        subscriptions.append(node.create_subscription(
            msg_type, topic, lambda msg, key=name: samples.setdefault(key, msg), qos_profile_sensor_data
        ))
    try:
        deadline = time.monotonic() + args.timeout
        while len(samples) < len(topics) and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if len(samples) < len(topics):
            print(f"missing camera streams: {sorted(set(name for name, _, _ in topics) - samples.keys())}")
            return 1

        color = samples["color"]
        depth = samples["depth"]
        points = samples["points"]
        info = samples["info"]
        rgb = np.frombuffer(color.data, dtype=np.uint8)
        depth_values = np.frombuffer(depth.data, dtype=np.float32)
        finite_depth = depth_values[np.isfinite(depth_values) & (depth_values > 0.0)]
        point_count = int(points.width) * int(points.height)
        rgb_std = float(rgb.std()) if rgb.size else 0.0
        depth_fraction = finite_depth.size / depth_values.size if depth_values.size else 0.0
        print(
            f"color={color.width}x{color.height} {color.encoding} std={rgb_std:.1f}; "
            f"depth={depth.width}x{depth.height} {depth.encoding} "
            f"valid={depth_fraction:.1%} range="
            f"{float(finite_depth.min()) if finite_depth.size else 0.0:.2f}-"
            f"{float(finite_depth.max()) if finite_depth.size else 0.0:.2f}m; "
            f"point_slots={point_count}; camera_fx={info.k[0]:.1f}"
        )
        if args.save_color and color.encoding == "rgb8" and rgb.size >= color.height * color.step:
            from PIL import Image as PILImage

            frame = rgb[: color.height * color.step].reshape(color.height, color.step)
            frame = frame[:, : color.width * 3].reshape(color.height, color.width, 3)
            args.save_color.parent.mkdir(parents=True, exist_ok=True)
            PILImage.fromarray(frame).save(args.save_color)
            print(f"saved_color={args.save_color}")
        valid = (
            color.encoding == "rgb8"
            and rgb.size >= color.width * color.height * 3
            and rgb_std > 3.0
            and depth.encoding == "32FC1"
            and depth_values.size >= depth.width * depth.height
            and depth_fraction > 0.01
            and point_count > 100
            and info.k[0] > 0.0
        )
        return 0 if valid else 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
