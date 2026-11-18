#!/usr/bin/env python3
"""Bounded, read-only capture of the first live-path veto and its actual witness.

Source-matched map/cloud/path are retained. The logged rectangle is the exact
runtime witness (including held/forecast evidence); a one-cloud replay cannot
reconstruct that history. This instrumented check is not a timing benchmark.
"""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import re
import time

import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.serialization import deserialize_message
from nav_msgs.msg import Odometry, Path as RosPath
from sensor_msgs.msg import PointCloud2
from se2_grid_msgs.msg import SE2Grid
from fsgp_bgk_se2.se2_grid_tools import get_layer_slices
from serialized_header import leading_stamp_ns


PATTERN = re.compile(
    r'terrain ahead rejected: layer=live map_ns=(\d+) cloud_ns=(\d+) now_ns=(\d+) '
    r'root=\(([^)]+)\) sample=\(([^)]+)\) points=(\d+) end=\(([^)]+)\) '
    r'nearest=(\d+) rectangle=\(([^)]+)\) present=(\d+) forecast=(\d+)')


def numbers(value):
    return [float(x) for x in value.split(',')]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--timeout', type=float, default=150.)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('diagnostic output exists')
    rclpy.init()
    node = rclpy.create_node('capture_ahead_rejection')
    maps, clouds, odom, paths = (deque(maxlen=n) for n in (15, 35, 80, 20))

    def retain(queue, payload):
        queue.append((leading_stamp_ns(payload), payload))

    node.create_subscription(SE2Grid, '/car_0_fsgp_bgk/fused_map',
                             lambda b: retain(maps, b), 1, raw=True)
    node.create_subscription(PointCloud2, '/cloud_registered',
                             lambda b: retain(clouds, b), qos_profile_sensor_data, raw=True)
    node.create_subscription(Odometry, '/Odometry',
                             lambda b: retain(odom, b), qos_profile_sensor_data, raw=True)
    node.create_subscription(RosPath, '/terrain_nav/path',
                             lambda b: retain(paths, b), 1, raw=True)
    deadline = time.monotonic() + a.timeout
    offset, carry, pending = 0, '', deque(maxlen=12)
    try:
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.02)
            log = a.run / 'planning.log'
            if log.exists():
                with log.open() as f:
                    f.seek(offset)
                    carry += f.read()
                    offset = f.tell()
                lines = carry.split('\n')
                carry = lines.pop()
                for line in lines:
                    event = PATTERN.search(line)
                    if event:
                        pending.append(event)
            for event in pending:
                stamp, cloud_stamp, clock = (int(event[i]) for i in (1, 2, 3))
                map_data = next((b for t, b in reversed(maps) if t == stamp), None)
                live_data = next((b for t, b in reversed(clouds) if abs(t-cloud_stamp) <= 5), None)
                candidates = []
                end = numbers(event[7])
                for _, data in reversed(paths):
                    msg = deserialize_message(data, RosPath)
                    if len(msg.poses) == int(event[6]) and msg.poses and all(
                            abs(v-w) < 2e-5 for v, w in zip(
                                (msg.poses[-1].pose.position.x, msg.poses[-1].pose.position.y), end)):
                        candidates.append(data)
                        break
                if map_data is None or live_data is None or not candidates or not odom:
                    continue
                grid = deserialize_message(map_data, SE2Grid)
                age = get_layer_slices(grid, 'gp_completion_age')[0]
                conditioning_stamp = stamp-round(float(age[0].flat[0])*1e9) if age else stamp
                conditioning = next((b for t, b in reversed(clouds)
                                     if abs(t-conditioning_stamp) <= 5), None)
                if conditioning is None:
                    continue
                payloads = {'map.cdr': map_data, 'live_cloud.cdr': live_data,
                            'completion_cloud.cdr': conditioning, 'path.cdr': candidates[0],
                            'odom.cdr': min(odom, key=lambda q: abs(q[0]-clock))[1]}
                a.output.mkdir(parents=True)
                for name, data in payloads.items():
                    (a.output/name).write_bytes(data)
                record = dict(scope=__doc__, source_run=str(a.run.resolve()),
                              map_source_stamp_ns=stamp, cloud_source_stamp_ns=cloud_stamp,
                              evaluation_clock_ns=clock, evaluated_pose=numbers(event[4]),
                              rejected_pose_and_distance=numbers(event[5]),
                              nearest_index=int(event[8]), actual_runtime_rectangle=numbers(event[9]),
                              present=bool(int(event[10])), forecast=bool(int(event[11])),
                              original_log=event[0],
                              sha256={k: hashlib.sha256(b).hexdigest() for k, b in payloads.items()})
                (a.output/'ahead.json').write_text(json.dumps(record, indent=2)+'\n')
                print(json.dumps(record), flush=True)
                return 0
        raise RuntimeError('no source-matched live path rejection captured')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
