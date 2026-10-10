#!/usr/bin/env python3
"""Closed-loop Isaac navigation trial with independent truth and map-age gates."""

import argparse
from bisect import bisect_left
import json
import math
import os
import time

os.environ["RMW_IMPLEMENTATION"] = os.environ.get("HUNTER_ISAAC_RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")

import rclpy
import numpy as np
from fsgp_bgk_se2.se2_grid_tools import get_layer_slices, grid_buffer_to_world
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import PointCloud2
from se2_grid_msgs.msg import SE2Grid


def yaw_from_quaternion(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def pitch_roll(q):
    roll = math.atan2(2.0 * (q.w * q.x + q.y * q.z), 1.0 - 2.0 * (q.x * q.x + q.y * q.y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (q.w * q.y - q.z * q.x))))
    return pitch, roll


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


class Trial(Node):
    def __init__(self, scenario):
        super().__init__("hunter_navigation_trial")
        self.scenario = scenario
        self.obstacle_x = 3.0 if scenario == "detour" else 5.0
        self.odom = None
        self.truth = None
        self.clock_ns = 0
        self.odom_wall = 0.0
        self.truth_wall = 0.0
        self.map_wall = 0.0
        self.map_intervals = []
        self.map_ages = []
        self.map_samples = 0
        self.last_map = None
        self.commands = 0
        self.steering_commands = 0
        self.last_command = (0.0, 0.0)
        self.truth_samples = []
        self.obstacle_pose_samples = []
        self.vacated_stamp = None
        self.ghost_map_samples = []
        self.ghost_laser_samples = []
        self.ghost_laser_obstacle_samples = []
        self.ghost_laser_height_samples = []
        self.ghost_map_before = []
        self.ghost_laser_before = []
        self.ghost_laser_obstacle_before = []
        self.ghost_live_samples = []
        self.registered_peak_before_vacate = 0
        self.registered_after_vacate = []
        self.latest_ground_xy = None
        self.ghost_ground_distance_samples = []
        self.live_center_samples = 0
        self.live_peak_cells = 0
        self.command_history = []
        self.truth_start = None
        self.create_subscription(Odometry, "/Odometry", self.on_odom, 10)
        self.create_subscription(Odometry, "/isaac/ground_truth/odometry", self.on_truth, 10)
        self.create_subscription(Twist, "/cmd_vel", self.on_command, 10)
        self.create_subscription(Clock, "/clock", self.on_clock, 10)
        self.create_subscription(SE2Grid, "/car_0_fsgp_bgk/fused_map", self.on_map, 2)
        self.create_subscription(OccupancyGrid, "/terrain_nav/live_obstacles", self.on_live, 2)
        self.create_subscription(PoseStamped, "/isaac/nav_test_obstacle/pose", self.on_obstacle, 10)
        self.create_subscription(PointCloud2, "/Laser_map", self.on_laser_map, 2)
        self.create_subscription(PointCloud2, "/cloud_registered", self.on_registered, 2)
        self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)

    def on_odom(self, msg):
        self.odom = msg
        self.odom_wall = time.monotonic()

    def on_truth(self, msg):
        self.truth = msg
        self.truth_wall = time.monotonic()
        p = msg.pose.pose.position
        pitch, roll = pitch_roll(msg.pose.pose.orientation)
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.truth_samples.append((self.truth_wall, p.x, p.y, p.z, pitch, roll, stamp))

    def on_obstacle(self, msg):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.obstacle_pose_samples.append((stamp, msg.pose.position.x, msg.pose.position.y))
        if self.vacated_stamp is None and (
                abs(msg.pose.position.y) > 1.5 if self.scenario == "clearance"
                else msg.pose.position.y < -1.5):
            self.vacated_stamp = stamp

    def count_obstacle_points(self, msg, min_z=-0.18):
        if msg.point_step == 0:
            return 0
        offsets = {field.name: field.offset for field in msg.fields}
        if not all(name in offsets for name in ("x", "y", "z")):
            return 0
        dtype = ">f4" if msg.is_bigendian else "<f4"
        shape = (msg.width * msg.height,)
        coordinates = {
            name: np.ndarray(shape, dtype=dtype, buffer=msg.data,
                             offset=offsets[name], strides=(msg.point_step,))
            for name in ("x", "y", "z")
        }
        center_high = np.count_nonzero(
            (np.abs(coordinates["x"] - self.obstacle_x) < 1.15) &
            (np.abs(coordinates["y"]) < 0.95) &
            (coordinates["z"] > min_z))
        return int(center_high)

    def on_registered(self, msg):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.vacated_stamp is None or stamp < self.vacated_stamp:
            self.registered_peak_before_vacate = max(
                self.registered_peak_before_vacate, self.count_obstacle_points(msg))
        elif msg.point_step > 0:
            age = stamp - self.vacated_stamp
            if 0.0 <= age <= 2.0 and len(self.registered_after_vacate) < 25:
                offsets = {field.name: field.offset for field in msg.fields}
                dtype = ">f4" if msg.is_bigendian else "<f4"
                shape = (msg.width * msg.height,)
                x = np.ndarray(shape, dtype=dtype, buffer=msg.data,
                               offset=offsets["x"], strides=(msg.point_step,))
                y = np.ndarray(shape, dtype=dtype, buffer=msg.data,
                               offset=offsets["y"], strides=(msg.point_step,))
                z = np.ndarray(shape, dtype=dtype, buffer=msg.data,
                               offset=offsets["z"], strides=(msg.point_step,))
                roi = (np.abs(x - self.obstacle_x) < 1.15) & (np.abs(y) < 0.95)
                ground = roi & (z < -0.30)
                nearby_ground = (np.abs(x - self.obstacle_x) < 3.0) & (np.abs(y) < 3.0) & (z < -0.30)
                self.latest_ground_xy = np.column_stack((x[nearby_ground], y[nearby_ground]))
                self.registered_after_vacate.append(
                    (round(age, 2), int(np.count_nonzero(roi & (z > -0.18))),
                     int(np.count_nonzero(ground))))

    def on_laser_map(self, msg):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        center_high = self.count_obstacle_points(msg)
        obstacle_high = self.count_obstacle_points(msg, min_z=0.30)
        if self.vacated_stamp is None or stamp < self.vacated_stamp:
            self.ghost_laser_before.append(center_high)
            self.ghost_laser_obstacle_before.append(obstacle_high)
        else:
            age = round(stamp - self.vacated_stamp, 2)
            self.ghost_laser_samples.append((age, center_high))
            self.ghost_laser_obstacle_samples.append((age, obstacle_high))
            if 0.0 <= age <= 2.0 and center_high >= 20:
                offsets = {field.name: field.offset for field in msg.fields}
                dtype = ">f4" if msg.is_bigendian else "<f4"
                shape = (msg.width * msg.height,)
                x = np.ndarray(shape, dtype=dtype, buffer=msg.data,
                               offset=offsets["x"], strides=(msg.point_step,))
                y = np.ndarray(shape, dtype=dtype, buffer=msg.data,
                               offset=offsets["y"], strides=(msg.point_step,))
                z = np.ndarray(shape, dtype=dtype, buffer=msg.data,
                               offset=offsets["z"], strides=(msg.point_step,))
                heights = z[(np.abs(x - self.obstacle_x) < 1.15) &
                            (np.abs(y) < 0.95) & (z > -0.18)]
                bins = np.histogram(heights, bins=(-0.18, 0.3, 0.8, 1.3, 2.0))[0]
                self.ghost_laser_height_samples.append((age, bins.tolist()))
                if self.latest_ground_xy is not None and len(self.latest_ground_xy) and 0.8 <= age <= 2.0:
                    ghost_xy = np.column_stack((x[(np.abs(x - self.obstacle_x) < 1.15) &
                                                   (np.abs(y) < 0.95) & (z > -0.18)],
                                                y[(np.abs(x - self.obstacle_x) < 1.15) &
                                                  (np.abs(y) < 0.95) & (z > -0.18)]))
                    d2 = np.sum((ghost_xy[:, None, :] - self.latest_ground_xy[None, :, :]) ** 2, axis=2)
                    nearest = np.sqrt(np.min(d2, axis=1))
                    self.ghost_ground_distance_samples.append(
                        (age, round(float(np.median(nearest)), 2),
                         round(float(np.percentile(nearest, 90)), 2),
                         int(np.count_nonzero(nearest <= 0.4)), len(nearest)))

    def on_command(self, msg):
        self.last_command = (msg.linear.x, msg.angular.z)
        if self.truth and self.truth_start:
            p = self.truth.pose.pose.position
            self.command_history.append((time.monotonic(), msg.linear.x, p.x - self.truth_start[0]))
        if abs(msg.linear.x) > 0.03:
            self.commands += 1
        if abs(msg.angular.z) > 0.03:
            self.steering_commands += 1

    def on_clock(self, msg):
        self.clock_ns = msg.clock.sec * 1_000_000_000 + msg.clock.nanosec

    def on_map(self, msg):
        now = time.monotonic()
        if self.map_wall:
            self.map_intervals.append(now - self.map_wall)
        self.map_wall = now
        self.map_samples += 1
        self.last_map = msg
        probe = map_probe(msg, self.obstacle_x - 0.55, 0.0, 0.0)
        stamp_sec = msg.info.header.stamp.sec + msg.info.header.stamp.nanosec * 1e-9
        if self.vacated_stamp is not None and stamp_sec >= self.vacated_stamp:
            self.ghost_map_samples.append(
                (round(stamp_sec - self.vacated_stamp, 2), probe.get("valid"),
                 probe.get("baseline_risk_local_max")))
        else:
            self.ghost_map_before.append(probe.get("baseline_risk_local_max"))
        stamp = msg.info.header.stamp
        stamp_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
        if stamp_ns and self.clock_ns >= stamp_ns:
            self.map_ages.append((self.clock_ns - stamp_ns) / 1_000_000_000)

    def on_live(self, msg):
        self.live_peak_cells = max(self.live_peak_cells, sum(value == 100 for value in msg.data))
        resolution = msg.info.resolution
        if resolution <= 0.0:
            return
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.vacated_stamp is not None and stamp >= self.vacated_stamp:
            ix = int(math.floor((5.0 - msg.info.origin.position.x) / resolution))
            iy = int(math.floor((0.0 - msg.info.origin.position.y) / resolution))
            if 0 <= ix < msg.info.width and 0 <= iy < msg.info.height:
                occupied = msg.data[iy * msg.info.width + ix] == 100
                self.ghost_live_samples.append((round(stamp - self.vacated_stamp, 2), occupied))
        for x in (4.2, 4.6, 5.0, 5.4, 5.8):
            for y in (-0.8, -0.4, 0.0, 0.4, 0.8):
                ix = int(math.floor((x - msg.info.origin.position.x) / resolution))
                iy = int(math.floor((y - msg.info.origin.position.y) / resolution))
                if 0 <= ix < msg.info.width and 0 <= iy < msg.info.height:
                    if msg.data[iy * msg.info.width + ix] == 100:
                        self.live_center_samples += 1
                        return


def map_probe(msg, x, y, yaw):
    if msg is None:
        return {}
    resolution = msg.info.pos_resolution
    left = msg.info.pose.position.x - msg.info.length_x * 0.5
    bottom = msg.info.pose.position.y - msg.info.length_y * 0.5
    ix = int(math.floor((x - left) / resolution))
    iy = int(math.floor((y - bottom) / resolution))
    out = {"cell": [ix, iy]}
    for name in ("valid", "baseline_risk", "risk", "elevation"):
        slices, has_so2, _ = get_layer_slices(msg, name)
        if not slices:
            continue
        yaw_index = int(round((yaw % (2.0 * math.pi)) / max(msg.info.yaw_resolution, 1e-6))) % len(slices)
        matrix = grid_buffer_to_world(slices[yaw_index if has_so2 else 0], msg)
        if not (0 <= iy < matrix.shape[0] and 0 <= ix < matrix.shape[1]):
            continue
        value = matrix[iy, ix]
        out[name] = round(float(value), 3) if np.isfinite(value) else None
        window = matrix[max(0, iy - 3): iy + 4, max(0, ix - 3): ix + 4]
        finite = window[np.isfinite(window)]
        out[name + "_local_max"] = round(float(finite.max()), 3) if finite.size else None
    return out


def run(node, args):
    deadline = time.monotonic() + 25.0
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        now = time.monotonic()
        if (node.odom and node.truth and node.map_samples >= 2
                and now - node.odom_wall < 1.0 and now - node.truth_wall < 1.0
                and node.goal_pub.get_subscription_count() > 0):
            break
    else:
        print("FAIL: live odometry, truth, enhanced map or goal subscriber missing", flush=True)
        return 1

    settle_until = time.monotonic() + 2.0
    while time.monotonic() < settle_until:
        rclpy.spin_once(node, timeout_sec=0.1)

    if node.odom.header.frame_id != "map":
        print(f"FAIL: odometry frame {node.odom.header.frame_id!r} is not map", flush=True)
        return 1

    op = node.odom.pose.pose.position
    oq = node.odom.pose.pose.orientation
    oyaw = yaw_from_quaternion(oq)
    tp = node.truth.pose.pose.position
    tq = node.truth.pose.pose.orientation
    tyaw = yaw_from_quaternion(tq)
    truth_start = (tp.x, tp.y, tp.z)
    node.truth_start = truth_start
    truth_goal = (tp.x + args.distance * math.cos(tyaw), tp.y + args.distance * math.sin(tyaw))

    goal = PoseStamped()
    goal.header.frame_id = "map"
    goal.pose.position.x = op.x + args.distance * math.cos(oyaw)
    goal.pose.position.y = op.y + args.distance * math.sin(oyaw)
    goal.pose.position.z = op.z
    goal.pose.orientation = oq

    node.truth_samples.clear()
    node.commands = 0
    node.steering_commands = 0
    start_wall = time.monotonic()
    for _ in range(2):
        goal.header.stamp = node.odom.header.stamp
        node.goal_pub.publish(goal)
        until = time.monotonic() + 0.25
        while time.monotonic() < until:
            rclpy.spin_once(node, timeout_sec=0.05)

    reached = False
    deadline = start_wall + args.timeout
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        now = time.monotonic()
        if now - node.odom_wall > 1.0 or now - node.truth_wall > 1.0 or now - node.map_wall > 1.0:
            print("FAIL: odometry, truth or risk map stopped updating", flush=True)
            return 1
        p = node.truth.pose.pose.position
        if math.hypot(p.x - truth_goal[0], p.y - truth_goal[1]) <= args.reach_distance:
            reached = True
            if args.scenario != "step":
                break

    samples = node.truth_samples
    if not samples:
        print("FAIL: no truth trajectory", flush=True)
        return 1
    forward = []
    lateral = []
    obstacle_distance = []
    obstacle_x = 5.0 if args.scenario in ("emerge", "crossing", "clearance") else 3.0
    for _, x, y, *_ in samples:
        dx, dy = x - truth_start[0], y - truth_start[1]
        f = dx * math.cos(tyaw) + dy * math.sin(tyaw)
        l = -dx * math.sin(tyaw) + dy * math.cos(tyaw)
        forward.append(f)
        lateral.append(l)
        obstacle_distance.append(math.hypot(f - obstacle_x, l))
    final = samples[-1]
    truth_error = math.hypot(final[1] - truth_goal[0], final[2] - truth_goal[1])
    map_age_p95 = percentile(node.map_ages, 0.95)
    physical_separations = []
    obstacle_times = [sample[0] for sample in node.obstacle_pose_samples]
    for sample in samples:
        nearest = bisect_left(obstacle_times, sample[6])
        candidates = (index for index in (nearest - 1, nearest) if 0 <= index < len(obstacle_times))
        closest = min(candidates, key=lambda index: abs(obstacle_times[index] - sample[6]), default=None)
        if closest is not None and abs(obstacle_times[closest] - sample[6]) <= 0.12:
            obstacle = node.obstacle_pose_samples[closest]
            physical_separations.append(math.hypot(sample[1] - obstacle[1], sample[2] - obstacle[2]))
    summary = {
        "scenario": args.scenario,
        "reached": reached,
        "truth_error_m": round(truth_error, 3),
        "truth_forward_max_m": round(max(forward), 3),
        "truth_lateral_max_m": round(max(map(abs, lateral)), 3),
        "truth_z_rise_max_m": round(max(s[3] for s in samples) - truth_start[2], 3),
        "truth_pitch_max_rad": round(max(abs(s[4]) for s in samples), 3),
        "truth_roll_max_rad": round(max(abs(s[5]) for s in samples), 3),
        "obstacle_center_min_m": round(min(obstacle_distance), 3),
        "physical_obstacle_min_m": round(min(physical_separations), 3) if physical_separations else None,
        "physical_obstacle_samples": len(physical_separations),
        "nonzero_commands": node.commands,
        "steering_commands": node.steering_commands,
        "final_command_mps": round(node.last_command[0], 3),
        "risk_map_samples": node.map_samples,
        "risk_map_period_p95_s": round(percentile(node.map_intervals, 0.95), 3) if node.map_intervals else None,
        "risk_map_age_p95_s": round(map_age_p95, 3) if map_age_p95 is not None else None,
        "trajectory_samples": len(samples),
        "live_center_samples": node.live_center_samples,
        "live_peak_cells": node.live_peak_cells,
        "ghost_laser_after_vacate": node.ghost_laser_samples[-12:],
        "ghost_laser_obstacle_after_vacate": node.ghost_laser_obstacle_samples[-12:],
        "ghost_laser_height_samples": node.ghost_laser_height_samples,
        "ghost_ground_distance_samples": node.ghost_ground_distance_samples,
        "ghost_risk_after_vacate": node.ghost_map_samples[-20:],
        "ghost_laser_peak_before_vacate": max(node.ghost_laser_before, default=0),
        "ghost_laser_obstacle_peak_before_vacate": max(node.ghost_laser_obstacle_before, default=0),
        "registered_peak_before_vacate": node.registered_peak_before_vacate,
        "registered_after_vacate": node.registered_after_vacate,
        "static_laser_last": node.ghost_laser_before[-1] if node.ghost_laser_before else 0,
        "ghost_live_after_vacate": node.ghost_live_samples[:15],
        "ghost_risk_peak_before_vacate": max(
            (value for value in node.ghost_map_before if value is not None), default=0.0),
        "front_stop_commands": sum(0.5 <= forward_m <= 2.5 and abs(v) < 0.03
                                   for _, v, forward_m in node.command_history),
        "map_probe": map_probe(
            node.last_map,
            op.x + (4.0 if args.scenario == "step" else obstacle_x) * math.cos(oyaw),
            op.y + (4.0 if args.scenario == "step" else obstacle_x) * math.sin(oyaw),
            oyaw,
        ),
        "forward_map_profile": {
            str(distance): map_probe(
                node.last_map,
                op.x + distance * math.cos(oyaw),
                op.y + distance * math.sin(oyaw),
                oyaw,
            )
            for distance in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
        },
        "lateral_map_profile": {
            f"{forward_m:+.1f},{left_m:+.1f}": (
                lambda probe: [probe.get("valid"), probe.get("baseline_risk"), probe.get("risk")]
            )(
                map_probe(
                    node.last_map,
                    op.x + forward_m * math.cos(oyaw) - left_m * math.sin(oyaw),
                    op.y + forward_m * math.sin(oyaw) + left_m * math.cos(oyaw),
                    oyaw,
                )
            )
            for forward_m in (1.5, 2.5, 3.5)
            for left_m in (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)
        } if args.scenario == "detour" else {},
    }
    common_ok = node.map_samples >= 10 and node.commands >= 5 and map_age_p95 is not None and map_age_p95 <= 0.7
    if args.scenario in ("detour", "emerge"):
        passed = (common_ok and reached and truth_error <= args.reach_distance
                  and min(obstacle_distance) >= 1.05 and max(map(abs, lateral)) >= 0.75
                  and node.steering_commands >= 5)
        if args.scenario == "detour":
            passed = (passed and summary["ghost_laser_peak_before_vacate"] >= 5
                      and summary["static_laser_last"] >=
                      0.5 * summary["ghost_laser_peak_before_vacate"])
        if args.scenario == "emerge":
            passed = (passed and node.live_center_samples >= 3 and len(physical_separations) >= 20
                      and min(physical_separations) >= 1.05)
    elif args.scenario == "ramp":
        passed = (common_ok and reached and truth_error <= args.reach_distance
                  and summary["truth_z_rise_max_m"] >= 0.28
                  and summary["truth_pitch_max_rad"] <= 0.55
                  and summary["truth_roll_max_rad"] <= 0.45)
    elif args.scenario == "crossing":
        passed = (common_ok and reached and truth_error <= args.reach_distance
                  and node.live_center_samples >= 3 and summary["front_stop_commands"] >= 1
                  and len(physical_separations) >= 20 and min(physical_separations) >= 1.05)
    elif args.scenario == "clearance":
        laser_peak = summary["ghost_laser_peak_before_vacate"]
        laser_clear = [age for age, points in node.ghost_laser_samples
                       if age >= 0.0 and points <= max(1, laser_peak * 0.2)]
        obstacle_peak = summary["ghost_laser_obstacle_peak_before_vacate"]
        obstacle_clear = [age for age, points in node.ghost_laser_obstacle_samples
                          if age >= 0.0 and points <= max(1, obstacle_peak * 0.2)]
        # A local terrain map may legitimately mark a vacated cell unknown
        # once the robot has moved away.  Unknown is handled explicitly by
        # terrain_nav (allow_unknown/unknown_cost); it is not a dynamic-risk
        # residual. Track both the risk-value clearance and the known-cell
        # clearance so the test does not confuse the two states.
        risk_clear = [age for age, _valid, risk in node.ghost_map_samples
                      if age >= 0.0 and risk is not None and risk <= 0.2]
        known_risk_clear = [age for age, valid, risk in node.ghost_map_samples
                            if age >= 0.0 and valid is not None and valid >= 0.5
                            and risk is not None and risk <= 0.2]
        live_clear = [age for age, occupied in node.ghost_live_samples
                      if age >= 0.0 and not occupied]
        summary["ghost_laser_clear_delay_s"] = min(laser_clear, default=None)
        summary["ghost_laser_obstacle_clear_delay_s"] = min(obstacle_clear, default=None)
        summary["ghost_risk_clear_delay_s"] = min(risk_clear, default=None)
        summary["ghost_risk_known_clear_delay_s"] = min(known_risk_clear, default=None)
        summary["ghost_live_clear_delay_s"] = min(live_clear, default=None)
        passed = (common_ok and reached and truth_error <= args.reach_distance
                  and len(physical_separations) >= 20 and min(physical_separations) >= 1.60
                  and node.vacated_stamp is not None and laser_peak >= 5
                  and obstacle_peak >= 5
                  and summary["ghost_risk_peak_before_vacate"] >= 0.8
                  and summary["ghost_laser_obstacle_clear_delay_s"] is not None
                  and summary["ghost_laser_obstacle_clear_delay_s"] <= 1.5
                  and summary["ghost_risk_clear_delay_s"] is not None
                  and summary["ghost_risk_clear_delay_s"] <= 1.5
                  and summary["ghost_live_clear_delay_s"] is not None
                  and summary["ghost_live_clear_delay_s"] <= 1.5)
    else:
        passed = (node.map_samples >= 10 and map_age_p95 is not None and map_age_p95 <= 0.7
                  and not reached and 1.5 <= max(forward) <= 3.55
                  and summary["map_probe"].get("risk_local_max", 0.0) >= 0.95
                  and summary["static_laser_last"] >=
                  0.5 * summary["ghost_laser_peak_before_vacate"]
                  and abs(node.last_command[0]) <= 0.1)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    print("PASS" if passed else "FAIL", flush=True)
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("detour", "ramp", "step", "emerge", "crossing", "clearance"), required=True)
    parser.add_argument("--distance", type=float, default=7.0)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--reach-distance", type=float, default=0.65)
    args = parser.parse_args()
    if args.distance <= args.reach_distance or args.timeout <= 2.0:
        parser.error("distance must exceed reach-distance and timeout must exceed 2 s")
    rclpy.init()
    node = Trial(args.scenario)
    try:
        return run(node, args)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
