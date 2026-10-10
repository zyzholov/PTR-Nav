#!/usr/bin/env python3
"""Check that both terrain maps continue to publish fresh SE2Grid data."""

import argparse
import time

import rclpy
from rclpy.qos import QoSProfile
from se2_grid_msgs.msg import SE2Grid


TOPICS = (
    "/car_0_terrain_analyzer/fused_map",
    "/car_0_fsgp_bgk/fused_map",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--min-rate", type=float, default=2.0)
    args = parser.parse_args()
    if args.duration <= 0 or args.min_rate <= 0:
        parser.error("duration and min-rate must be positive")

    rclpy.init()
    node = rclpy.create_node("hunter_map_rate_probe")
    received = {topic: [] for topic in TOPICS}
    subscriptions = []
    for topic in TOPICS:
        def callback(msg, name=topic):
            stamp = msg.info.header.stamp
            received[name].append((time.monotonic(), stamp.sec, stamp.nanosec))

        subscriptions.append(node.create_subscription(SE2Grid, topic, callback, QoSProfile(depth=1)))

    try:
        deadline = time.monotonic() + args.duration
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        finished = time.monotonic()
        passed = True
        for topic, samples in received.items():
            interval = samples[-1][0] - samples[0][0] if len(samples) >= 2 else 0.0
            rate = (len(samples) - 1) / interval if interval > 0 else 0.0
            fresh = [sample for i, sample in enumerate(samples)
                     if i == 0 or sample[1:] != samples[i - 1][1:]]
            fresh_interval = fresh[-1][0] - fresh[0][0] if len(fresh) >= 2 else 0.0
            fresh_rate = (len(fresh) - 1) / fresh_interval if fresh_interval > 0 else 0.0
            age = finished - samples[-1][0] if samples else float("inf")
            stamps_monotonic = all(samples[i][1:] >= samples[i - 1][1:] for i in range(1, len(samples)))
            print(
                f"{topic}: samples={len(samples)} wall_rate={rate:.2f}Hz "
                f"fresh_stamps={len(fresh)} fresh_rate={fresh_rate:.2f}Hz last_age={age:.2f}s"
            )
            passed &= rate >= args.min_rate and fresh_rate >= args.min_rate and stamps_monotonic
            passed &= age <= max(1.5, 2.0 / args.min_rate)
        return 0 if passed else 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
