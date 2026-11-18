#!/usr/bin/env python3
"""Check that both terrain maps continue to publish fresh SE2Grid data."""

import argparse
import json
import math
from pathlib import Path
import signal
import statistics
import time

import rclpy
from diagnostic_msgs.msg import DiagnosticArray
from rclpy.qos import QoSProfile
from rclpy.parameter import Parameter
from rclpy.signals import SignalHandlerOptions
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, PointCloud2
from nav_msgs.msg import Odometry
from se2_grid_msgs.msg import SE2Grid
from fsgp_bgk_se2.se2_grid_tools import serialized_header


TOPICS = (
    "/car_0_terrain_analyzer/fused_map",
    "/car_0_fsgp_bgk/fused_map",
)


def serialized_header_stamp(payload):
    """Read only the leading Header of these Humble CDR1 message types.

    Imu, Odometry and PointCloud2 start with Header; SE2Grid starts with
    info.header. Do not deserialize megabytes of unused grid/point arrays in a
    timing observer. Unsupported encapsulation fails, never guesses offsets.
    This is a transport/stamp observation, not sensor-content validation.
    """
    stamp=serialized_header(payload).stamp
    return stamp.sec,stamp.nanosec


class ObservationWindow:
    """Keep discovery evidence separate; arm before any commanded motion."""
    def __init__(self, topics):
        self.startup = {topic: [] for topic in topics}
        self.samples = {topic: [] for topic in topics}
        self.armed_at = None
        self.discovery_stamps = {topic: set() for topic in topics}

    def record(self, topic, sample):
        if self.armed_at is not None:
            self.samples[topic].append(sample)
            return
        self.startup[topic].append(sample)
        if math.isfinite(sample[3]):
            self.discovery_stamps[topic].add(sample[1:3])
        # The first large DDS sample can predate reader discovery. Require
        # exactly two distinct clocked samples per stream before arming. Do
        # not wait for a favourable age: a stale second sample still fails.
        if all(len(self.discovery_stamps[name])>=2 and math.isfinite(rows[-1][3])
               for name,rows in self.startup.items()):
            self.armed_at = sample[0]
            self.samples = {name: [rows[-1]] for name, rows in self.startup.items()}


def stream_statistics(samples, finished):
    interval = samples[-1][0] - samples[0][0] if len(samples) >= 2 else 0.0
    rate = (len(samples) - 1) / interval if interval > 0 else 0.0
    fresh = [sample for i, sample in enumerate(samples)
             if i == 0 or sample[1:3] != samples[i - 1][1:3]]
    fresh_interval = fresh[-1][0] - fresh[0][0] if len(fresh) >= 2 else 0.0
    fresh_rate = (len(fresh) - 1) / fresh_interval if fresh_interval > 0 else 0.0
    stamp_span = (samples[-1][1] - samples[0][1] +
                  (samples[-1][2] - samples[0][2]) * 1e-9) if len(samples) >= 2 else 0.0
    ages = sorted(s[3] for s in samples if math.isfinite(s[3]))
    gaps = [b[0]-a[0] for a,b in zip(samples,samples[1:])]
    held_ages = [b[1]+b[2]*1e-9+b[3]-a[1]-a[2]*1e-9
                 for a,b in zip(samples,samples[1:]) if math.isfinite(b[3])]
    return dict(count=len(samples), rate=rate, fresh_rate=fresh_rate, fresh_count=len(fresh),
        stamp_rate=(len(samples) - 1) / stamp_span if stamp_span > 0 else 0.0,
        arrival_age=finished - samples[-1][0] if samples else math.inf,
        monotonic=all(samples[i][1:3] >= samples[i - 1][1:3] for i in range(1, len(samples))),
        age_count=len(ages), age_min=min(ages) if ages else math.nan,
        age_p50=statistics.median(ages) if ages else math.nan,
        age_p95=ages[math.ceil(.95*len(ages))-1] if ages else math.nan,
        age_max=max(ages) if ages else math.nan,
        arrival_gap_max=max(gaps) if gaps else math.nan,
        held_age_before_update_max=max(held_ages) if held_ages else math.nan)


def map_stream_qualified(stats, minimum_rate, maximum_age=None):
    passed = (stats["rate"] >= minimum_rate and stats["fresh_rate"] >= minimum_rate
              and stats["monotonic"] and stats["arrival_age"] <= max(1.5, 2.0 / minimum_rate))
    if maximum_age is not None:
        passed &= (stats["age_count"] >= .9 * stats["count"] and stats["age_count"] > 0
                   and stats["age_min"] >= -.11 and stats["age_max"] <= maximum_age)
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--min-rate", type=float, default=2.0)
    parser.add_argument("--max-source-age", type=float, default=None)
    parser.add_argument("--sensors", action="store_true", help="Also check LiDAR and external IMU during the same window")
    parser.add_argument('--upstream',action='store_true',help='Record registered cloud and FAST-LIO odometry to localize latency spikes')
    observation=parser.add_mutually_exclusive_group()
    observation.add_argument('--raw-headers',action='store_true',
        help='Diagnostic comparison: observe raw CDR headers without deserializing map/cloud arrays')
    observation.add_argument('--typed-messages',action='store_false',dest='raw_headers',help=argparse.SUPPRESS)
    parser.set_defaults(raw_headers=False)
    parser.add_argument("--sensor-min-realtime-factor", type=float, default=.95)
    parser.add_argument("--output", type=Path, help="Retain every arrival/stamp/age, including startup and failures")
    parser.add_argument("--ready-file", type=Path, help="Signal armed observation before the caller sends a goal")
    parser.add_argument("--startup-timeout", type=float, default=15.)
    args = parser.parse_args()
    if (not math.isfinite(args.duration) or args.duration <= 0
            or not math.isfinite(args.min_rate) or args.min_rate <= 0
            or args.max_source_age is not None and (not math.isfinite(args.max_source_age) or args.max_source_age <= 0)
            or not math.isfinite(args.sensor_min_realtime_factor) or not 0<args.sensor_min_realtime_factor<=1
            or not math.isfinite(args.startup_timeout) or args.startup_timeout<=0):
        parser.error("use finite positive duration, rate, age and a realtime factor in (0,1]")

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    running = True

    def stop(_signum, _frame):
        nonlocal running
        running = False

    previous_handler = signal.signal(signal.SIGINT, stop)
    node = rclpy.create_node("hunter_map_rate_probe",parameter_overrides=[Parameter("use_sim_time",value=True)])
    sensor_types = {"/livox/lidar": (PointCloud2, 10.), "/imu/data_raw": (Imu, 200.)} if args.sensors else {}
    if args.upstream:
        sensor_types.update({'/cloud_registered':(PointCloud2,10.),
            '/cloud_registered_visual':(PointCloud2,10.),'/Odometry':(Odometry,10.)})
    window = ObservationWindow((*TOPICS, *sensor_types))
    subscriptions = []
    runtime_samples = []
    def runtime_callback(msg):
        for status in msg.status:
            if status.name == 'fsgp_bgk_se2/runtime':
                runtime_samples.append(dict(wall_monotonic_s=time.monotonic(),
                    stamp_sec=msg.header.stamp.sec, stamp_nanosec=msg.header.stamp.nanosec,
                    values={item.key:(float(item.value) if math.isfinite(float(item.value)) else None)
                            for item in status.values}))
    subscriptions.append(node.create_subscription(
        DiagnosticArray, '/car_0_fsgp_bgk/runtime', runtime_callback, QoSProfile(depth=20)))
    for topic in window.samples:
        is_map = topic in TOPICS
        def callback(msg, name=topic):
            if not args.raw_headers:
                stamp=msg.info.header.stamp if name in TOPICS else msg.header.stamp
                sec,nanosec=stamp.sec,stamp.nanosec
            else:sec,nanosec=serialized_header_stamp(msg)
            clock=node.get_clock().now().nanoseconds*1e-9
            age=clock-sec-nanosec*1e-9 if clock>0 else math.nan
            window.record(name,(time.monotonic(),sec,nanosec,age))

        subscriptions.append(node.create_subscription(SE2Grid if is_map else sensor_types[topic][0],
            topic, callback, QoSProfile(depth=1) if is_map or topic in ('/cloud_registered','/cloud_registered_visual','/Odometry')
            else qos_profile_sensor_data,raw=args.raw_headers))

    try:
        startup_deadline=time.monotonic()+args.startup_timeout
        while running and window.armed_at is None and time.monotonic()<startup_deadline:
            rclpy.spin_once(node,timeout_sec=.1)
        if window.armed_at is not None and args.ready_file is not None:
            args.ready_file.write_text(json.dumps({"armed_at":window.armed_at})+"\n")
        deadline = time.monotonic() + args.duration
        while running and window.armed_at is not None and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        finished = time.monotonic()
        passed = window.armed_at is not None
        summaries = {}
        for topic, samples in window.samples.items():
            stats = stream_statistics(samples, finished)
            summaries[topic] = stats
            latency=(f"source_age_p50={stats['age_p50']:.3f}s "
                     f"source_age_p95={stats['age_p95']:.3f}s "
                     f"source_age_max={stats['age_max']:.3f}s") if stats["age_count"] else "source_age=unavailable"
            print(
                f"{topic}: samples={stats['count']} wall_rate={stats['rate']:.2f}Hz "
                f"fresh_stamps={stats['fresh_count']} fresh_rate={stats['fresh_rate']:.2f}Hz "
                f"stamp_rate={stats['stamp_rate']:.2f}Hz last_arrival_age={stats['arrival_age']:.2f}s {latency}"
            )
            if topic in TOPICS:
                passed &= map_stream_qualified(stats, args.min_rate, args.max_source_age)
            else:
                nominal = sensor_types[topic][1]
                passed &= map_stream_qualified(stats, nominal * args.sensor_min_realtime_factor,args.max_source_age)
                passed &= stats["stamp_rate"] >= .95 * nominal
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({"scope":"continuous armed observation; discovery retained separately before commanded motion",
                "arming_policy":"two_distinct_clocked_samples_per_stream_v2_no_age_filter",
                'subscription_mode':'raw_cdr1_header_only' if args.raw_headers else 'full_typed_messages',
                "passed":bool(passed),"arguments":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                "armed_at":window.armed_at,
                "columns":["wall_monotonic_s","stamp_sec","stamp_nanosec","source_age_sim_s"],
                "statistic_definitions":{
                    'age_max':'maximum source age at message arrival, not between arrivals',
                    'arrival_gap_max':'maximum wall-clock interval between received messages',
                    'held_age_before_update_max':'maximum reconstructed simulated age of the previous message immediately before the next arrival; not continuous-time proof'},
                "summaries":{topic:{k:None if isinstance(v,float) and not math.isfinite(v) else v
                    for k,v in stats.items()} for topic,stats in summaries.items()},
                "samples":{topic:[[a,b,c,d if math.isfinite(d) else None] for a,b,c,d in rows]
                    for topic,rows in window.samples.items()},
                "startup_samples":{topic:[[a,b,c,d if math.isfinite(d) else None] for a,b,c,d in rows]
                    for topic,rows in window.startup.items()},
                "runtime_diagnostics":runtime_samples},indent=2,allow_nan=False)+"\n")
        return 0 if passed else 1
    finally:
        node.destroy_node()
        rclpy.shutdown()
        signal.signal(signal.SIGINT, previous_handler)


if __name__ == "__main__":
    raise SystemExit(main())
