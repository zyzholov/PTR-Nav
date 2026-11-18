#!/usr/bin/env python3
"""Closed-loop Isaac navigation trial with independent truth and map-age gates."""

import argparse
from bisect import bisect_left
import json
import math
import os
from pathlib import Path
import sys
import time

from clearance_metrics import (EARLY_RECOVERY_WINDOW_S, clearance_acceptance, sustained_clearance, sampled_body_clearances,
    held_velocity_encounter, event_stop_response,rectangle_front_wall_clearance, static_course_clearances,
    localization_agreement,moving_obstacle_observation,terminal_stop_evidence)

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src/isaac_sim_hunter_livo2/scripts'))
from navigation_courses import (COMPOSITE_SCENARIOS, CROSSING_SCENARIOS, RAMP_SCENARIOS, OVERHEAD_SCENARIOS,
                                course_obstacles, minimum_course_distance, front_sign_obstacles)
from vehicle_geometry import swept_bounds

os.environ["RMW_IMPLEMENTATION"] = os.environ.get("HUNTER_ISAAC_RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")

import rclpy
import numpy as np
from fsgp_bgk_se2.se2_grid_tools import get_layer_slices, grid_buffer_to_world
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Bool
from diagnostic_msgs.msg import DiagnosticArray
from nav_msgs.msg import OccupancyGrid, Odometry, Path as NavPath
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


def course_to_map(x,y,alignment):
    tx,ty,angle=alignment;c,s=math.cos(angle),math.sin(angle)
    return tx+c*x-s*y,ty+s*x+c*y


def map_to_course(x,y,alignment):
    tx,ty,angle=alignment;c,s=math.cos(angle),math.sin(angle)
    return c*(x-tx)+s*(y-ty),-s*(x-tx)+c*(y-ty)


def stationary_observer_alignment(odom,truth):
    """Pair the delayed odometry with historical truth, never receipt time.

    Returns one evaluator-only transform. No extrapolation or drift fitting.
    Both ends of a short truth bracket must support a stationary acquisition.
    """
    if odom is None or odom.header.frame_id!='map' or len(truth)<2:return None
    twist=odom.twist.twist
    if not all(math.isfinite(v) for v in (twist.linear.x,twist.linear.y,twist.angular.z)):return None
    if math.hypot(twist.linear.x,twist.linear.y)>.03 or abs(twist.angular.z)>.03:return None
    rows=truth[-256:]
    if any(not all(math.isfinite(row[k]) for k in (1,2,6,7)) for row in rows):return None
    times=[row[6] for row in rows]
    if any(b<=a for a,b in zip(times,times[1:])):return None
    stamp=odom.header.stamp.sec+1e-9*odom.header.stamp.nanosec
    if not math.isfinite(stamp) or stamp<times[0] or stamp>times[-1]:return None
    j=max(1,bisect_left(times,stamp));a,b=rows[j-1],rows[j]
    dt=b[6]-a[6]
    if dt>.05:return None
    turn=math.atan2(math.sin(b[7]-a[7]),math.cos(b[7]-a[7]))
    if math.hypot(b[1]-a[1],b[2]-a[2])/dt>.03 or abs(turn/dt)>.03:return None
    f=(stamp-a[6])/dt;tx=a[1]+f*(b[1]-a[1]);ty=a[2]+f*(b[2]-a[2])
    angle=yaw_from_quaternion(odom.pose.pose.orientation)-(a[7]+f*turn)
    p=odom.pose.pose.position;c,s=math.cos(angle),math.sin(angle)
    result=(p.x-c*tx+s*ty,p.y-s*tx-c*ty,math.atan2(math.sin(angle),math.cos(angle)))
    return result if all(math.isfinite(v) for v in result) else None


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def truth_speed(samples):
    if len(samples)<2:return None
    last=samples[-1]
    prior=next((s for s in reversed(samples[:-1]) if .08<=last[6]-s[6]<=.5),None)
    if prior is None:return None
    return math.hypot(last[1]-prior[1],last[2]-prior[2])/(last[6]-prior[6])


def reserve_recovery_evidence(status_samples, truth, yaw):
    poses = np.asarray(truth, float)
    if (poses.ndim != 2 or poses.shape[1] != 8 or len(poses)<2 or
            not np.isfinite(poses).all() or not math.isfinite(yaw) or np.any(np.diff(poses[:,6])<=0)):
        return dict(accepted=False, reason='missing monotonic independent truth')
    forward = (poses[:,1]-poses[0,1])*math.cos(yaw)+(poses[:,2]-poses[0,2])*math.sin(yaw)
    certified = 0; invalid = False; active = None; phases = []
    for row in status_samples:
        for status in row['status']:
            values = status['values']
            state = values.get('clearance_recovery_state')
            if state in ('idle', 'refused'):
                active = None
            if state not in ('retreat', 'hold', 'completed'):
                continue
            try:
                speed = float(values['command_speed_mps'])
                path = float(values['clearance_recovery_path_upper'])
                braking = float(values['clearance_recovery_braking_upper'])
                valid = (values['command_accepted'] == '1' and all(math.isfinite(v) for v in (speed,path,braking))
                         and 0 <= path <= .10 and 0 <= braking <= .05 and
                         (-.300001 <= speed < -.01 if state == 'retreat' else abs(speed)<.03))
            except (KeyError, ValueError, TypeError):
                valid = False
            invalid |= not valid
            stamp = row['stamp_s']
            if not valid:
                active = None
            elif state == 'retreat':
                certified += 1
                if active is None:active = [stamp, 0]
                active[1] += 1
            elif state == 'completed' and active is not None:
                begin, count = active
                if poses[0,6]-.05 <= begin <= stamp <= poses[-1,6]+.05 and count>=2:
                    positions = list(forward[(poses[:,6]>=begin)&(poses[:,6]<=stamp)])
                    positions.append(float(np.interp(stamp, poses[:,6], forward)))
                    reverse = max(0., float(np.interp(begin, poses[:,6], forward))-min(positions))
                    phases.append(dict(begin_s=begin, end_s=stamp, reverse_m=reverse, certified_samples=count))
                active = None
    reverse = max((phase['reverse_m'] for phase in phases), default=0.)
    return dict(certified_status_samples=certified, invalid_status=bool(invalid),
                completed_phases=phases, physical_reverse_m=reverse,
                scope='independent reverse motion only within a certified, successfully completed reserve phase',
                accepted=bool(phases and not invalid and reverse >= .02))


def motion_grid_sample(msg, *, allow_unknown=False):
    info=msg.info
    geometry=[info.width,info.height,info.origin.position.x,info.origin.position.y,info.resolution]
    data=np.asarray(msg.data)
    if (not np.isfinite(geometry).all() or min(info.width,info.height,info.resolution)<=0 or
            data.size!=info.width*info.height or not np.isin(data,(-1,0,75,100) if allow_unknown else (0,100)).all()):
        raise ValueError('invalid raw obstacle grid')
    sample={'stamp_s':msg.header.stamp.sec+1e-9*msg.header.stamp.nanosec,
            'frame':msg.header.frame_id,'geometry':geometry,'occupied':np.flatnonzero(data==100).tolist()}
    if allow_unknown:
        sample['observed_free']=np.flatnonzero(data==0).tolist()
        sample['historical_advisory']=np.flatnonzero(data==75).tolist()
    return sample


class Trial(Node):
    def __init__(self, scenario, capture_motion=False):
        super().__init__("hunter_navigation_trial")
        self.scenario = scenario
        self.obstacle_x = 3.5 if scenario == 'ramp_crossing' else 3.0 if scenario == "detour" else 5.0
        self.odom = None
        self.truth = None
        self.observer_map_alignment = None
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
        self.ghost_enhanced_samples = []
        self.ghost_probability_samples = []
        self.probability_before_vacate = []
        self.obstacle_motion_stamp = None
        self.initial_obstacle_xy = None
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
        self.reference_ground_z = None
        self.ghost_ground_distance_samples = []
        self.live_center_samples = 0
        self.live_peak_cells = 0
        self.command_history = []
        self.truth_start = None
        self.truth_yaw_start = 0.0
        self.latest_path = None
        self.path_history = []
        self.odom_samples = []
        self.localization_poses = []
        self.localization_attitudes = []
        self.localization_truth = []
        self.localization_overflow = False
        self.tracking_errors = []
        self.control_samples = []
        self.safety_status_samples = []
        self.safety_status_overflow = False
        self.recording = False
        self.motion_grids=[]
        self.motion_tracking_grids=[]
        self.topology_grids=[]
        self.topology_routes=[]
        self.motion_grid_overflow=False
        if capture_motion or scenario in CROSSING_SCENARIOS:
            self.create_subscription(OccupancyGrid,"/terrain_nav/live_obstacles_raw",self.on_motion_raw,10)
            self.create_subscription(OccupancyGrid,"/terrain_nav/motion_observations",self.on_motion_tracking,10)
        if capture_motion:
            self.create_subscription(OccupancyGrid,"/terrain_nav/topology_memory",self.on_topology_grid,2)
            self.create_subscription(NavPath,"/terrain_nav/topology_route",self.on_topology_route,2)
        self.create_subscription(Odometry, "/Odometry", self.on_odom, 10)
        self.create_subscription(Odometry, "/isaac/ground_truth/odometry", self.on_truth, 10)
        self.create_subscription(NavPath, "/terrain_nav/path", self.on_path, 10)
        self.create_subscription(Twist, "/cmd_vel", self.on_command, 10)
        self.create_subscription(DiagnosticArray, "/terrain_nav/chance_status", self.on_safety_status, 10)
        self.create_subscription(Clock, "/clock", self.on_clock, 10)
        self.create_subscription(SE2Grid, "/car_0_fsgp_bgk/fused_map", self.on_map, 2)
        self.create_subscription(OccupancyGrid, "/terrain_nav/live_obstacles", self.on_live, 2)
        self.create_subscription(PoseStamped, "/isaac/nav_test_obstacle/pose", self.on_obstacle, 10)
        self.create_subscription(PointCloud2, "/Laser_map", self.on_laser_map, 2)
        self.create_subscription(PointCloud2, "/cloud_registered", self.on_registered, 2)
        self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)

    def on_safety_status(self, msg):
        if not self.recording:
            return
        if len(self.safety_status_samples) >= 10000:
            self.safety_status_overflow = True
            return
        self.safety_status_samples.append(dict(
            stamp_s=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9,
            status=[dict(name=s.name,
                         level=s.level[0] if isinstance(s.level,(bytes,bytearray)) else int(s.level),
                         message=s.message,
                         values={value.key:value.value for value in s.values}) for s in msg.status]))

    def on_odom(self, msg):
        self.odom = msg
        self.odom_wall = time.monotonic()
        self._try_observer_alignment()
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if stamp <= 0.0:
            stamp = self.clock_ns * 1e-9
        twist = msg.twist.twist
        self.odom_samples.append((stamp, twist.linear.x, twist.linear.y, twist.angular.z))
        p = msg.pose.pose.position
        if len(self.localization_poses)<20000:
            self.localization_poses.append((stamp,p.x,p.y,p.z,yaw_from_quaternion(msg.pose.pose.orientation)))
            q=msg.pose.pose.orientation
            self.localization_attitudes.append((stamp,q.x,q.y,q.z,q.w))
        else:
            self.localization_overflow=True
        if self.recording and self.path_history and msg.header.frame_id == "map":
            p = msg.pose.pose.position
            path = next((points for path_stamp, points in reversed(self.path_history)
                         if path_stamp <= stamp), None)
            if path:
                error = point_to_polyline_distance((p.x, p.y), path)
                self.tracking_errors.append((stamp, error))
        if len(self.odom_samples) > 20000:
            del self.odom_samples[:5000]

    def on_truth(self, msg):
        self.truth = msg
        self.truth_wall = time.monotonic()
        p = msg.pose.pose.position
        pitch, roll = pitch_roll(msg.pose.pose.orientation)
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if stamp <= 0.0:
            stamp = self.clock_ns * 1e-9
        yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        self.truth_samples.append((self.truth_wall, p.x, p.y, p.z, pitch, roll, stamp, yaw))
        self._try_observer_alignment()
        if len(self.localization_truth)<20000:
            self.localization_truth.append(self.truth_samples[-1])
        else:
            self.localization_overflow=True

    def _try_observer_alignment(self):
        # Freeze ONE paired stationary alignment for evaluation ROIs only.
        # Never re-fit drift, alter localization, or send this to navigation.
        if self.observer_map_alignment is not None:return
        self.observer_map_alignment=stationary_observer_alignment(self.odom,self.truth_samples)

    def on_path(self, msg):
        if msg.header.frame_id != "map":
            return
        points = [(pose.pose.position.x, pose.pose.position.y)
                  for pose in msg.poses]
        if not points:
            return
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if stamp <= 0.0:
            stamp = self.clock_ns * 1e-9
        self.latest_path = points
        if self.recording:
            self.path_history.append((stamp, points))
        if len(self.path_history) > 500:
            del self.path_history[:100]

    def on_obstacle(self, msg):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        xy=(msg.pose.position.x,msg.pose.position.y)
        if self.initial_obstacle_xy is None:self.initial_obstacle_xy=xy
        elif self.obstacle_motion_stamp is None and math.hypot(xy[0]-self.initial_obstacle_xy[0],xy[1]-self.initial_obstacle_xy[1])>.03:
            self.obstacle_motion_stamp=stamp
        self.obstacle_pose_samples.append((stamp, msg.pose.position.x, msg.pose.position.y))
        if self.vacated_stamp is None and (
                abs(msg.pose.position.y) > 1.5 if self.scenario == "clearance"
                else msg.pose.position.y < -1.5):
            self.vacated_stamp = stamp

    def count_obstacle_points(self, msg, min_z=None):
        if self.observer_map_alignment is None:return 0
        if min_z is None:
            if self.reference_ground_z is None:
                return 0
            min_z = self.reference_ground_z + .30
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
        x,y=map_to_course(coordinates['x'],coordinates['y'],self.observer_map_alignment)
        center_high = np.count_nonzero(
            (np.abs(x - self.obstacle_x) < 1.15) &
            (np.abs(y) < 0.95) &
            (coordinates["z"] > min_z))
        return int(center_high)

    def on_registered(self, msg):
        if self.observer_map_alignment is None:return
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.reference_ground_z is None and msg.point_step > 0:
            offsets = {field.name: field.offset for field in msg.fields}
            if all(name in offsets for name in ("x", "y", "z")):
                dtype = ">f4" if msg.is_bigendian else "<f4"
                xyz = [np.ndarray((msg.width * msg.height,), dtype=dtype, buffer=msg.data,
                                  offset=offsets[name], strides=(msg.point_step,))
                       for name in ("x", "y", "z")]
                x,y,z = xyz
                x,y=map_to_course(x,y,self.observer_map_alignment)
                ring = (np.abs(x-self.obstacle_x) < 3) & (np.abs(y) < 3) & np.isfinite(z)
                if np.count_nonzero(ring) >= 30:
                    # Flat-course qualification only: freeze a measured floor,
                    # not the simulator's world-Z or a predicted ground value.
                    self.reference_ground_z = float(np.percentile(z[ring],20))
        if self.vacated_stamp is None or stamp < self.vacated_stamp:
            self.registered_peak_before_vacate = max(
                self.registered_peak_before_vacate, self.count_obstacle_points(msg))
        elif msg.point_step > 0:
            age = stamp - self.vacated_stamp
            if age >= 0.0:
                offsets = {field.name: field.offset for field in msg.fields}
                dtype = ">f4" if msg.is_bigendian else "<f4"
                shape = (msg.width * msg.height,)
                x = np.ndarray(shape, dtype=dtype, buffer=msg.data,
                               offset=offsets["x"], strides=(msg.point_step,))
                y = np.ndarray(shape, dtype=dtype, buffer=msg.data,
                               offset=offsets["y"], strides=(msg.point_step,))
                z = np.ndarray(shape, dtype=dtype, buffer=msg.data,
                               offset=offsets["z"], strides=(msg.point_step,))
                x,y=map_to_course(x,y,self.observer_map_alignment)
                roi = (np.abs(x - self.obstacle_x) < 1.15) & (np.abs(y) < 0.95)
                if self.reference_ground_z is None:
                    return
                floor = self.reference_ground_z
                ground = roi & (np.abs(z-floor) < .15)
                nearby_ground = (np.abs(x - self.obstacle_x) < 3.0) & (np.abs(y) < 3.0) & (np.abs(z-floor) < .15)
                self.latest_ground_xy = np.column_stack((x[nearby_ground], y[nearby_ground]))
                self.registered_after_vacate.append(
                    (round(age, 2), int(np.count_nonzero(roi & (z > floor+.30))),
                     int(np.count_nonzero(ground))))

    def on_laser_map(self, msg):
        if self.observer_map_alignment is None:return
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        center_high = self.count_obstacle_points(msg)
        obstacle_high = center_high
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
                x,y=map_to_course(x,y,self.observer_map_alignment)
                floor = self.reference_ground_z
                heights = z[(np.abs(x - self.obstacle_x) < 1.15) &
                            (np.abs(y) < 0.95) & (z > floor+.30)] - floor
                bins = np.histogram(heights, bins=(.30, .78, 1.28, 1.78, 2.48))[0]
                self.ghost_laser_height_samples.append((age, bins.tolist()))
                if self.latest_ground_xy is not None and len(self.latest_ground_xy) and 0.8 <= age <= 2.0:
                    ghost_xy = np.column_stack((x[(np.abs(x - self.obstacle_x) < 1.15) &
                                                   (np.abs(y) < 0.95) & (z > floor+.30)],
                                                y[(np.abs(x - self.obstacle_x) < 1.15) &
                                                  (np.abs(y) < 0.95) & (z > floor+.30)]))
                    d2 = np.sum((ghost_xy[:, None, :] - self.latest_ground_xy[None, :, :]) ** 2, axis=2)
                    nearest = np.sqrt(np.min(d2, axis=1))
                    self.ghost_ground_distance_samples.append(
                        (age, round(float(np.median(nearest)), 2),
                         round(float(np.percentile(nearest, 90)), 2),
                         int(np.count_nonzero(nearest <= 0.4)), len(nearest)))

    def on_command(self, msg):
        self.last_command = (msg.linear.x, msg.angular.z)
        if self.recording:
            self.control_samples.append((self.clock_ns * 1e-9, msg.linear.x, msg.angular.z))
        if self.truth and self.truth_start:
            p = self.truth.pose.pose.position
            forward = ((p.x - self.truth_start[0]) * math.cos(self.truth_yaw_start)
                       + (p.y - self.truth_start[1]) * math.sin(self.truth_yaw_start))
            self.command_history.append((time.monotonic(), msg.linear.x, forward))
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
        alignment=self.observer_map_alignment
        probe = map_probe(msg,*course_to_map(self.obstacle_x-.55,0.,alignment),alignment[2]) if alignment else {}
        stamp_sec = msg.info.header.stamp.sec + msg.info.header.stamp.nanosec * 1e-9
        if self.vacated_stamp is not None and stamp_sec >= self.vacated_stamp:
            self.ghost_map_samples.append(
                (round(stamp_sec - self.vacated_stamp, 2), probe.get("valid"),
                 probe.get("baseline_risk_local_max")))
            self.ghost_enhanced_samples.append(
                (round(stamp_sec - self.vacated_stamp, 2), probe.get("valid"),
                 probe.get("risk_local_max")))
            self.ghost_probability_samples.append(
                (round(stamp_sec-self.vacated_stamp,2),probe.get('probability_valid'),
                 probe.get('failure_probability_upper')))
        else:
            self.ghost_map_before.append(probe.get("baseline_risk_local_max"))
            if alignment is not None:
                self.probability_before_vacate.append(dict(stamp_s=stamp_sec,
                    valid=probe.get('probability_valid'),upper=probe.get('failure_probability_upper'),
                    measured_support=measured_probe_support(msg,*course_to_map(self.obstacle_x-.55,0.,alignment),alignment[2])))
        stamp = msg.info.header.stamp
        stamp_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
        if stamp_ns and self.clock_ns >= stamp_ns:
            self.map_ages.append((self.clock_ns - stamp_ns) / 1_000_000_000)

    def on_motion_raw(self,msg):
        if not self.recording:return
        if len(self.motion_grids)>=6000:
            self.motion_grid_overflow=True
            return
        self.motion_grids.append(motion_grid_sample(msg))

    def on_motion_tracking(self,msg):
        if not self.recording:return
        if len(self.motion_tracking_grids)>=6000:
            self.motion_grid_overflow=True;return
        self.motion_tracking_grids.append(motion_grid_sample(msg))

    def on_topology_grid(self,msg):
        if not self.recording:return
        if len(self.topology_grids)>=600:
            self.motion_grid_overflow=True;return
        self.topology_grids.append(motion_grid_sample(msg,allow_unknown=True))

    def on_topology_route(self,msg):
        if not self.recording:return
        if len(self.topology_routes)>=600:
            self.motion_grid_overflow=True;return
        self.topology_routes.append(dict(stamp_s=msg.header.stamp.sec+1e-9*msg.header.stamp.nanosec,
            frame=msg.header.frame_id,poses=[[p.pose.position.x,p.pose.position.y,
                yaw_from_quaternion(p.pose.orientation)] for p in msg.poses]))

    def on_live(self, msg):
        self.live_peak_cells = max(self.live_peak_cells, sum(value == 100 for value in msg.data))
        resolution = msg.info.resolution
        if resolution <= 0.0:
            return
        if self.observer_map_alignment is None:return
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        center_x = self.obstacle_x if self.scenario == 'ramp_crossing' else 5.0
        if self.vacated_stamp is not None and stamp >= self.vacated_stamp:
            px,py=course_to_map(center_x,0.,self.observer_map_alignment)
            ix = int(math.floor((px - msg.info.origin.position.x) / resolution))
            iy = int(math.floor((py - msg.info.origin.position.y) / resolution))
            if 0 <= ix < msg.info.width and 0 <= iy < msg.info.height:
                occupied = msg.data[iy * msg.info.width + ix] == 100
                self.ghost_live_samples.append((round(stamp - self.vacated_stamp, 2), occupied))
        for x in (center_x-.8, center_x-.4, center_x, center_x+.4, center_x+.8):
            for y in (-0.8, -0.4, 0.0, 0.4, 0.8):
                px,py=course_to_map(x,y,self.observer_map_alignment)
                ix = int(math.floor((px - msg.info.origin.position.x) / resolution))
                iy = int(math.floor((py - msg.info.origin.position.y) / resolution))
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
    for name in ("valid", "baseline_risk", "risk", "elevation",
                 "probability_valid", "failure_probability_upper"):
        slices, has_so2, _ = get_layer_slices(msg, name)
        if not slices:
            continue
        yaw_index = int(math.floor((yaw % (2.0 * math.pi)) / max(msg.info.yaw_resolution, 1e-6))) % len(slices)
        matrix = grid_buffer_to_world(slices[yaw_index if has_so2 else 0], msg)
        if not (0 <= iy < matrix.shape[0] and 0 <= ix < matrix.shape[1]):
            continue
        value = matrix[iy, ix]
        out[name] = float(value) if np.isfinite(value) else None
        window = matrix[max(0, iy - 3): iy + 4, max(0, ix - 3): ix + 4]
        finite = window[np.isfinite(window)]
        out[name + "_local_max"] = float(finite.max()) if finite.size else None
    return out


def measured_probe_support(msg,x,y,yaw):
    """Conservative observer-only stencil: every intersecting cell measured.

    Includes cell extent in both body axes. GP/initial assumptions cannot
    satisfy this precondition for a previously measured-ground experiment.
    """
    if 'gp_model_ground_observed' not in msg.layers:return False
    def scalar(name):return float(get_layer_slices(msg,name)[0][0].flat[0])
    front,back,half=(scalar('probability_'+k) for k in ('front','back','half_width'))
    r=msg.info.pos_resolution;c,s=math.cos(yaw),math.sin(yaw)
    left=msg.info.pose.position.x-msg.info.length_x/2;bottom=msg.info.pose.position.y-msg.info.length_y/2
    observed=grid_buffer_to_world(get_layer_slices(msg,'gp_model_ground_observed')[0][0],msg)
    iy,ix=np.indices(observed.shape);dx=left+(ix+.5)*r-x;dy=bottom+(iy+.5)*r-y
    u,v=c*dx+s*dy,-s*dx+c*dy;extent=.5*r*(abs(c)+abs(s))
    mask=(u>=-back-extent)&(u<=front+extent)&(abs(v)<=half+extent)
    radius=math.hypot(max(front,back),half)+r
    if x-radius<left or y-radius<bottom or x+radius>left+observed.shape[1]*r or y+radius>bottom+observed.shape[0]*r:return False
    pending=grid_buffer_to_world(get_layer_slices(msg,'ground_pending_effective')[0][0],msg)
    return bool(np.count_nonzero(mask)>=4 and np.all(observed[mask]>.5) and np.all(pending[mask]<.5))


def polyline_length(points):
    if len(points) < 2:
        return 0.0
    return float(sum(math.hypot(b[0] - a[0], b[1] - a[1])
                     for a, b in zip(points, points[1:])))


def polyline_curvatures(points):
    values = []
    for first, middle, last in zip(points, points[1:], points[2:]):
        v1 = (middle[0] - first[0], middle[1] - first[1])
        v2 = (last[0] - middle[0], last[1] - middle[1])
        length = 0.5 * (math.hypot(*v1) + math.hypot(*v2))
        if length < 1.0e-3:
            continue
        heading1 = math.atan2(v1[1], v1[0])
        heading2 = math.atan2(v2[1], v2[0])
        delta = math.atan2(math.sin(heading2 - heading1),
                           math.cos(heading2 - heading1))
        values.append(abs(delta) / length)
    return values


def point_to_polyline_distance(point, points):
    if not points:
        return None
    if len(points) == 1:
        return math.hypot(point[0] - points[0][0], point[1] - points[0][1])
    best = float("inf")
    for start, end in zip(points, points[1:]):
        dx, dy = end[0] - start[0], end[1] - start[1]
        length_sq = dx * dx + dy * dy
        if length_sq < 1.0e-9:
            projection = start
        else:
            t = max(0.0, min(1.0, ((point[0] - start[0]) * dx +
                                   (point[1] - start[1]) * dy) / length_sq))
            projection = (start[0] + t * dx, start[1] + t * dy)
        best = min(best, math.hypot(point[0] - projection[0],
                                    point[1] - projection[1]))
    return best


def trajectory_metrics(samples, planned_path, direct_goal_distance, tracking_errors=()):
    if len(samples) < 2:
        return {
            "duration_s": None,
            "truth_path_length_m": 0.0,
            "avg_speed_mps": None,
            "max_speed_mps": None,
            "max_yaw_rate_radps": None,
            "tracking_error_mean_m": None,
            "tracking_error_p95_m": None,
            "tracking_error_max_m": None,
        }
    truth_length = 0.0
    speeds = []
    yaw_rates = []
    for first, second in zip(samples, samples[1:]):
        dt = second[6] - first[6]
        distance = math.hypot(second[1] - first[1], second[2] - first[2])
        truth_length += distance
        if dt > 1.0e-4:
            speeds.append(distance / dt)
            dyaw = math.atan2(math.sin(second[7] - first[7]),
                              math.cos(second[7] - first[7]))
            yaw_rates.append(abs(dyaw) / dt)
    tracking = [value for _, value in tracking_errors if value is not None]
    duration = max(0.0, samples[-1][6] - samples[0][6])
    return {
        "duration_s": round(duration, 3),
        "truth_path_length_m": round(truth_length, 3),
        "planned_path_length_m": round(polyline_length(planned_path), 3)
        if planned_path else None,
        "planned_path_points": len(planned_path),
        "path_ratio_to_goal": round(polyline_length(planned_path) /
                                      max(direct_goal_distance, 1.0e-3), 3)
        if planned_path else None,
        "mean_curvature_1pm": round(float(np.mean(polyline_curvatures(planned_path))), 4)
        if len(planned_path) >= 3 and polyline_curvatures(planned_path) else None,
        "max_curvature_1pm": round(max(polyline_curvatures(planned_path)), 4)
        if len(planned_path) >= 3 and polyline_curvatures(planned_path) else None,
        "avg_speed_mps": round(truth_length / duration, 3) if duration > 1.0e-4 else None,
        "max_speed_mps": round(max(speeds), 3) if speeds else None,
        "max_yaw_rate_radps": round(max(yaw_rates), 3) if yaw_rates else None,
        "tracking_error_mean_m": round(float(np.mean(tracking)), 3) if tracking else None,
        "tracking_error_p95_m": round(percentile(tracking, 0.95), 3) if tracking else None,
        "tracking_error_max_m": round(max(tracking), 3) if tracking else None,
    }


def run(node, args):
    preparation = None
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
    if node.observer_map_alignment is None:
        print('FAIL: no source-time-paired stationary evaluation alignment; goal not sent',flush=True)
        return 1

    if args.scenario == 'reserve_overlap':
        deadline = time.monotonic()+20.
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.05)
            p = node.odom.pose.pose.position
            probe = map_probe(node.last_map, p.x, p.y, yaw_from_quaternion(node.odom.pose.pose.orientation))
            valid, upper = probe.get('probability_valid'), probe.get('failure_probability_upper')
            if valid is not None and upper is not None and valid >= .5 and math.isfinite(upper) and upper <= .05:
                break
        else:
            print('FAIL: initial support unqualified before obstacle preparation', flush=True)
            return 1
        # Build a genuinely observed return corridor, rather than assuming
        # ground behind the starting blind spot. The normal planner drives it.
        p = node.odom.pose.pose.position
        yaw = yaw_from_quaternion(node.odom.pose.pose.orientation)
        warmup = PoseStamped()
        warmup.header.frame_id = 'map'
        warmup.header.stamp = node.odom.header.stamp
        warmup.pose.position.x = p.x+2.12*math.cos(yaw)
        warmup.pose.position.y = p.y+2.12*math.sin(yaw)
        warmup.pose.orientation = node.odom.pose.pose.orientation
        origin = node.truth.pose.pose.position
        truth_yaw = yaw_from_quaternion(node.truth.pose.pose.orientation)
        warmup_goal = (origin.x+2.12*math.cos(truth_yaw), origin.y+2.12*math.sin(truth_yaw))
        warmup_index = len(node.truth_samples)
        node.goal_pub.publish(warmup)
        deadline = time.monotonic()+25.
        stopped_since = None
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.05)
            p = node.truth.pose.pose.position
            speed = truth_speed(node.truth_samples)
            ready = (math.hypot(p.x-warmup_goal[0],p.y-warmup_goal[1])<=.3 and
                     speed is not None and speed<.05 and abs(node.last_command[0])<.03)
            stopped_since = (stopped_since or time.monotonic()) if ready else None
            if stopped_since is not None and time.monotonic()-stopped_since>=.5:break
        else:
            print('FAIL: observed-corridor warmup did not arrive and settle', flush=True)
            return 1
        warmup_truth = list(node.truth_samples[warmup_index:])
        trigger = node.create_publisher(Bool, '/isaac/nav_test_trigger', 10)
        deadline = time.monotonic()+5.
        while trigger.get_subscription_count() == 0 and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.05)
        if not trigger.get_subscription_count():
            print('FAIL: reserve obstacle trigger subscriber missing', flush=True)
            return 1
        truth_index, obstacle_index = len(node.truth_samples), len(node.obstacle_pose_samples)
        before = (node.truth.pose.pose.position.x, node.truth.pose.pose.position.y)
        trigger.publish(Bool(data=True))
        deadline = time.monotonic()+4.
        while time.monotonic() < deadline:rclpy.spin_once(node, timeout_sec=.05)
        truth = node.truth_samples[truth_index:]
        obstacle = node.obstacle_pose_samples[obstacle_index:]
        stable = (len(truth)>=20 and len(obstacle)>=20 and
                  max(math.hypot(row[1]-before[0], row[2]-before[1]) for row in truth)<.05 and
                  all(abs(row[1]-3.1)<.02 and abs(row[2])<.02 for row in obstacle[-15:]))
        preparation = dict(scope='stationary preparation before navigation; actual moving box poses, no map truth supplied',
                           initial_support_probe=probe, warmup_goal=warmup_goal, warmup_truth=warmup_truth,
                           truth=truth, obstacle=obstacle, accepted=stable)
        if not stable:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.with_suffix('.preparation_failed.json').open('x') as output:
                json.dump(preparation, output, allow_nan=False)
            print('FAIL: obstacle preparation moved robot or did not settle', flush=True)
            return 1

    op = node.odom.pose.pose.position
    oq = node.odom.pose.pose.orientation
    oyaw = yaw_from_quaternion(oq)
    tp = node.truth.pose.pose.position
    tq = node.truth.pose.pose.orientation
    tyaw = yaw_from_quaternion(tq)
    truth_start = (tp.x, tp.y, tp.z)
    node.truth_start = truth_start
    node.truth_yaw_start = tyaw
    direction = -1.0 if args.goal_mode == "behind" else 1.0
    truth_goal = (tp.x + direction * args.distance * math.cos(tyaw) - args.lateral * math.sin(tyaw),
                  tp.y + direction * args.distance * math.sin(tyaw) + args.lateral * math.cos(tyaw))

    goal = PoseStamped()
    goal.header.frame_id = "map"
    goal.pose.position.x = op.x + direction * args.distance * math.cos(oyaw) - args.lateral * math.sin(oyaw)
    goal.pose.position.y = op.y + direction * args.distance * math.sin(oyaw) + args.lateral * math.cos(oyaw)
    goal.pose.position.z = op.z
    goal.pose.orientation = oq

    node.truth_samples.clear()
    node.odom_samples.clear()
    node.path_history.clear()
    node.tracking_errors.clear()
    node.control_samples.clear()
    node.command_history.clear()
    node.latest_path = None
    node.recording = True
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
    input_stream_failure = None
    settled=False;settled_since=None
    deadline = start_wall + args.timeout
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        now = time.monotonic()
        if now - node.odom_wall > 1.0 or now - node.truth_wall > 1.0 or now - node.map_wall > 1.0:
            print("FAIL: odometry, truth or risk map stopped updating", flush=True)
            input_stream_failure = dict(reason='odometry, truth or risk map stopped updating',
                odom_wall_age_s=now-node.odom_wall, truth_wall_age_s=now-node.truth_wall,
                risk_wall_age_s=now-node.map_wall, stamp_s=node.clock_ns*1e-9)
            break
        p = node.truth.pose.pose.position
        if math.hypot(p.x - truth_goal[0], p.y - truth_goal[1]) <= args.reach_distance:
            reached = True
            speed=truth_speed(node.truth_samples)
            if speed is not None and speed<.05 and abs(node.last_command[0])<.03:
                if settled_since is None:settled_since=node.clock_ns*1e-9
                settled=node.clock_ns*1e-9-settled_since>=.5
            else:settled_since=None
            if args.scenario != "step" and settled:
                break
        else:settled_since=None

    samples = node.truth_samples
    if not samples:
        print("FAIL: no truth trajectory", flush=True)
        return 1
    forward = []
    lateral = []
    obstacle_distance = []
    obstacle_x = (3.5 if args.scenario == 'ramp_crossing' else
                  5.0 if args.scenario in ("emerge", "crossing", "clearance") else 3.0)
    for _, x, y, *_ in samples:
        dx, dy = x - truth_start[0], y - truth_start[1]
        f = dx * math.cos(tyaw) + dy * math.sin(tyaw)
        l = -dx * math.sin(tyaw) + dy * math.cos(tyaw)
        forward.append(f)
        lateral.append(l)
        obstacle_distance.append(math.hypot(f - obstacle_x, l))
    final = samples[-1]
    truth_error = math.hypot(final[1] - truth_goal[0], final[2] - truth_goal[1])
    planned_path = node.path_history[0][1] if node.path_history else []
    motion_metrics = trajectory_metrics(
        samples, planned_path, math.hypot(truth_goal[0] - truth_start[0],
                                          truth_goal[1] - truth_start[1]), node.tracking_errors)
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
    # Raised Isaac perimeter colliders extend beyond the wheel-contact stencil.
    # Enclose bumpers and turned wheels for a separate physical-truth metric.
    body_geometry=dict(**swept_bounds(),
        radius=.85 if args.scenario=="clearance" else .65)
    body_clearances=([] if args.scenario == 'reserve_overlap' else
        sampled_body_clearances(samples,node.obstacle_pose_samples,**body_geometry))
    body_min=min((s[1] for s in body_clearances),default=None)
    body_alignment_fraction=len(body_clearances)/len(samples)
    body_clearance_ok=(len(body_clearances)>=20 and body_alignment_fraction>=.95
        and body_min is not None and body_min>=.10)
    encounter=(None if args.scenario == 'reserve_overlap' else
        held_velocity_encounter(samples,node.obstacle_pose_samples,**body_geometry))
    event_response=(event_stop_response(samples,node.control_samples,encounter["event_stamp_s"])
        if encounter is not None else None)
    evidence_begin=node.obstacle_motion_stamp
    evidence_end=node.vacated_stamp
    if encounter is not None and evidence_begin is not None:
        evidence_begin=max(evidence_begin,encounter['event_stamp_s']-.5)
        evidence_end=encounter['event_stamp_s']+.6
    dynamic_observation=moving_obstacle_observation(node.motion_grids,node.obstacle_pose_samples,
        node.observer_map_alignment,radius=body_geometry['radius'],start_s=evidence_begin,end_s=evidence_end)
    summary = {
        "scenario": args.scenario,
        "ramp_direction": args.ramp_direction if args.scenario in RAMP_SCENARIOS else None,
        "ramp_rise_m": args.ramp_rise if args.scenario in RAMP_SCENARIOS else None,
        "goal_mode": args.goal_mode,
        "goal_distance_m": args.distance,
        "goal_lateral_m": args.lateral,
        "reach_distance_m":args.reach_distance,
        "reached": reached,
        "settled":settled,
        "final_truth_speed_mps":truth_speed(samples),
        "truth_error_m": round(truth_error, 3),
        "truth_forward_max_m": round(max(forward), 3),
        "truth_lateral_max_m": round(max(map(abs, lateral)), 3),
        "truth_z_rise_max_m": round(max(s[3] for s in samples) - truth_start[2], 3),
        "truth_z_drop_max_m": round(truth_start[2] - min(s[3] for s in samples), 3),
        "truth_pitch_max_rad": round(max(abs(s[4]) for s in samples), 3),
        "truth_roll_max_rad": round(max(abs(s[5]) for s in samples), 3),
        "obstacle_center_min_m": round(min(obstacle_distance), 3)
            if args.scenario in ("detour", "emerge", *CROSSING_SCENARIOS, "clearance") else None,
        "physical_obstacle_min_m": round(min(physical_separations), 3) if physical_separations else None,
        "physical_obstacle_samples": len(physical_separations),
        "sampled_body_clearance_min_m":body_min,
        "body_clearance_alignment_fraction":body_alignment_fraction,
        "body_clearance_geometry_m":body_geometry,
        "body_clearance_scope":("not applicable to box; use course_clearance_minima_m" if
            args.scenario == 'reserve_overlap' else
            "oriented enclosing rectangle vs cylinder; bracketed truth samples, not continuous contact proof"),
        "held_velocity_encounter":encounter,
        "dynamic_encounter_required":args.require_dynamic_encounter,
        "dynamic_raw_observation":dynamic_observation,
        "event_stop_response":event_response,
        "nonzero_commands": node.commands,
        "steering_commands": node.steering_commands,
        "final_command_mps": round(node.last_command[0], 3),
        "risk_map_samples": node.map_samples,
        "risk_map_period_p95_s": round(percentile(node.map_intervals, 0.95), 3) if node.map_intervals else None,
        "risk_map_age_p95_s": round(map_age_p95, 3) if map_age_p95 is not None else None,
        "trajectory_samples": len(samples),
        "reverse_command_count": sum(v < -0.03 for _, v, _ in node.command_history),
        "max_abs_pitch_rad": round(max(abs(s[4]) for s in samples), 3),
        "max_abs_roll_rad": round(max(abs(s[5]) for s in samples), 3),
        "live_center_samples": node.live_center_samples,
        "live_peak_cells": node.live_peak_cells,
        "ghost_laser_after_vacate": node.ghost_laser_samples[-12:],
        "ghost_laser_obstacle_after_vacate": node.ghost_laser_obstacle_samples[-12:],
        "ghost_laser_height_samples": node.ghost_laser_height_samples,
        "ghost_ground_distance_samples": node.ghost_ground_distance_samples,
        "ghost_risk_after_vacate": node.ghost_map_samples[-20:],
        "ghost_laser_peak_before_vacate": max(node.ghost_laser_before, default=0),
        "ghost_laser_obstacle_peak_before_vacate": max(node.ghost_laser_obstacle_before, default=0),
        "obstacle_point_filter": {"reference_ground_z_m": node.reference_ground_z,
                                  "minimum_height_above_ground_m": .30,
                                  "reference": "frozen 20th percentile of measured flat-course ROI"},
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
    summary.update(motion_metrics)
    summary["path_update_count"] = len(node.path_history)
    summary["planned_path_metric_scope"]="first published rolling segment, not a complete mission path"
    summary["tracking_error_frame"] = "map: estimated odometry vs causal active path"
    summary["duration_timebase"] = "simulation time"
    common_ok = node.map_samples >= 10 and node.commands >= 5 and map_age_p95 is not None and map_age_p95 <= 0.7
    course_clearances = None
    if args.scenario in COMPOSITE_SCENARIOS:
        fixtures = course_obstacles(args.scenario)
        course_clearances = static_course_clearances(samples, fixtures,
            body_z_bounds=(-.30,.35) if args.scenario in OVERHEAD_SCENARIOS else None)
        minima = {name: min(value for _, value in rows) for name, rows in course_clearances.items()}
        gaps = [b[6]-a[6] for a, b in zip(samples, samples[1:])]
        summary['course_geometry_world_m'] = fixtures
        summary['course_clearance_minima_m'] = minima
        summary['course_truth_max_gap_s'] = max(gaps, default=None)
        summary['course_clearance_scope'] = 'sampled planar enclosing body vs all fixed fixtures; not continuous 3D contact proof'
        passed = (common_ok and reached and truth_error <= args.reach_distance
                  and len(samples) >= 20 and gaps and 0. < min(gaps) and max(gaps) <= .12
                  and minima and min(minima.values()) >= .10
                  and summary['truth_pitch_max_rad'] <= .55 and summary['truth_roll_max_rad'] <= .45)
        if args.scenario in OVERHEAD_SCENARIOS:
            inside=all(abs(s[2])+.4*abs(math.cos(s[7]))+.58*abs(math.sin(s[7]))<=1.1 for s in samples)
            summary['overhead_inside_channel']=inside
            summary['overhead_bottom_world_m']=fixtures[-1]['z']-fixtures[-1]['height']/2
            summary['course_clearance_scope']='sampled full-body/payload conservative 3D separation lower bound; no planar-only beam veto, not continuous mesh distance'
            if args.scenario=='overhead_block':
                stopped=truth_speed(samples)
                passed=(common_ok and not reached and len(samples)>=20 and gaps and
                    0<min(gaps) and max(gaps)<=.12 and minima and min(minima.values())>=.10 and
                    .5<=max(forward)<=3.2 and stopped is not None and stopped<.03 and
                    abs(node.last_command[0])<.03)
                summary['negative_fixture_stopped_speed_mps']=stopped
            passed=passed and inside
        if args.scenario == 'corridor':
            throat_samples = [s for s in samples if 3. <= s[1] <= 6.]
            traversed_inside = (len(throat_samples) >= 20 and
                all(abs(s[2]) + .4*abs(math.cos(s[7])) + .58*abs(math.sin(s[7])) <= 1.1
                    for s in throat_samples))
            summary['corridor_traversed_inside'] = traversed_inside
            passed = passed and traversed_inside
        if args.scenario == 'reserve_overlap':
            evidence = reserve_recovery_evidence(node.safety_status_samples, samples, tyaw)
            summary['reserve_recovery_evidence'] = evidence
            passed = passed and evidence['accepted']
    elif args.scenario in ("detour", "emerge"):
        passed = (common_ok and reached and truth_error <= args.reach_distance
                  and min(obstacle_distance) >= 1.05 and max(map(abs, lateral)) >= 0.75
                  and node.steering_commands >= 5)
        if args.scenario == "detour":
            passed = (passed and body_clearance_ok and summary["ghost_laser_peak_before_vacate"] >= 5
                      and summary["static_laser_last"] >=
                      0.5 * summary["ghost_laser_peak_before_vacate"])
        if args.scenario == "emerge":
            passed = (passed and node.live_center_samples >= 3 and len(physical_separations) >= 20
                      and min(physical_separations) >= 1.05 and body_clearance_ok)
    elif args.scenario == "ramp":
        passed = common_ok and reached and truth_error <= args.reach_distance
    elif args.scenario in CROSSING_SCENARIOS:
        passed = (common_ok and reached and truth_error <= args.reach_distance
                  and dynamic_observation['accepted'] and not node.motion_grid_overflow
                  and len(physical_separations) >= 20 and min(physical_separations) >= 1.05
                  and body_clearance_ok)
        if args.require_dynamic_encounter:
            passed=(passed and encounter is not None and encounter["minimum_clearance_m"]<.10
                and event_response is not None
                and event_response["first_zero_command_delay_s"] is not None
                and event_response["first_zero_command_delay_s"]<=.6
                and event_response["first_sustained_truth_stop_delay_s"] is not None)
        else:passed=passed and summary["front_stop_commands"]>=1
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
                  and body_clearance_ok
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
        # HighStep occupies x=[4,10], y=[-9,9]. All sampled body corners must
        # remain in its front-face band for this plane-distance check to apply.
        step_geometry=swept_bounds()
        in_face_band=all(abs(s[2])+max(step_geometry['front'],step_geometry['back'])+
                         step_geometry['half_width']<9. for s in samples)
        step_clearances=[(s[6],rectangle_front_wall_clearance(s[1],s[7],4.,**step_geometry))
                         for s in samples]
        summary['step_front_face_clearance_min_m']=min((v for _,v in step_clearances),default=None)
        summary['step_clearance_geometry']=dict(**step_geometry,wall_x_m=4.,wall_half_width_m=9.,
            sampled_body_within_face_band=in_face_band,scope='sampled physical truth; HighStep front face, not sensor-map height')
        summary['terminal_stop']=terminal_stop_evidence(samples,node.control_samples)
        summary['refusal_protocol']='observed_high_step_and_terminal_stop_v2'
        # Refusal must not require driving an arbitrary 1.5m towards a hazard.
        # Establish that this mission actually moved on a planned prefix,
        # then require independent terminal rest AND the original clearance,
        # obstacle evidence and non-crossing checks. Log progress separately.
        passed = (node.map_samples >= 10 and map_age_p95 is not None and map_age_p95 <= 0.7
                  and not reached and .10 <= max(forward) <= 3.55 and node.path_history
                  and in_face_band and len(step_clearances)>=20
                  and summary['step_front_face_clearance_min_m']>=.10
                  and summary["map_probe"].get("risk_local_max", 0.0) >= 0.95
                  and summary["static_laser_last"] >=
                  0.5 * summary["ghost_laser_peak_before_vacate"]
                  and summary['terminal_stop']['accepted'])
    if args.scenario in RAMP_SCENARIOS:
        passed = (passed and summary['truth_z_drop_max_m' if args.ramp_direction == 'downhill'
                                     else 'truth_z_rise_max_m'] >= .28
                  and summary['truth_pitch_max_rad'] <= .55 and summary['truth_roll_max_rad'] <= .45)
    if args.scenario not in ('step','overhead_block'):passed=passed and settled
    temporal=None;whole_record_temporal=None;window_coverage=None
    if args.scenario=="clearance" or args.require_clearance_recovery:
        series={
            "registered":([(t,count) for t,count,_ in node.registered_after_vacate],1.,.3),
            "laser_obstacle":(node.ghost_laser_obstacle_samples,
                max(1.,summary["ghost_laser_obstacle_peak_before_vacate"]*.2),1.5),
            "baseline_risk":([(t,risk if valid is not None and valid>=.5 else None)
                     for t,valid,risk in node.ghost_map_samples],.2,.75),
            "enhanced_risk":([(t,risk if valid is not None and valid>=.5 else None)
                     for t,valid,risk in node.ghost_enhanced_samples],.2,.75),
            "probability_pose":([(t,upper if valid is not None and valid>=.5 else None)
                     for t,valid,upper in node.ghost_probability_samples],.05,.75),
            "live":([(t,float(occupied)) for t,occupied in node.ghost_live_samples],.5,.3)}
        temporal={}
        for name,(samples_,threshold,gap) in series.items():
            try:
                temporal[name]=sustained_clearance(samples_,threshold,max_gap_s=gap)
            except ValueError as exc:
                temporal[name]={"invalid_temporal_order":str(exc)}
        whole_record_temporal=temporal
        if args.clearance_observation_window is not None:
            temporal={};window=args.clearance_observation_window;window_coverage={}
            for name,(samples_,threshold,gap) in series.items():
                selected=[row for row in samples_ if 0<=row[0]<=window]
                temporal[name]=sustained_clearance(selected,threshold,max_gap_s=gap)
                # Never turn missing observations or early observer exit into
                # evidence of recovery. Laser map runs at 2 Hz, other layers
                # retain their declared tighter sampling-gap requirements.
                window_coverage[name]=bool(selected and selected[-1][0]>=window-min(gap,.6))
            passed=passed and all(window_coverage.values())
        passed = passed and clearance_acceptance(temporal,require_probability=True)
    observed_recovery=None
    if args.require_clearance_recovery:
        prior=[r for r in node.probability_before_vacate if node.obstacle_motion_stamp is not None and
            node.obstacle_motion_stamp-2.<=r['stamp_s']<node.obstacle_motion_stamp and r['measured_support'] and
            r['valid'] is not None and r['valid']>=.5 and r['upper'] is not None and 0<=r['upper']<=.05]
        prior_stamps=sorted({r['stamp_s'] for r in prior})
        precondition=bool(len(prior_stamps)>=2 and prior_stamps[-1]-prior_stamps[0]>=.1)
        observed_recovery=dict(prior_measured_support=precondition,prior_qualified_source_stamps_s=prior_stamps,
            motion_onset_s=node.obstacle_motion_stamp,clearing_limit_s=1.5,
            observation_window_s=args.clearance_observation_window,window_coverage=window_coverage,
            scope='separate observed-ground crossing case; does not replace the initially occluded cold-obstacle case')
        passed=passed and precondition
    background_clearances = None
    background_clearances_planar = None
    if args.evaluate_front_sign:
        background_clearances_planar = static_course_clearances(samples, front_sign_obstacles(),**swept_bounds())
        background_clearances = static_course_clearances(samples, front_sign_obstacles(),body_z_bounds=(-.30,.35),**swept_bounds())
        minima = {name:min(value for _,value in rows) for name,rows in background_clearances.items()}
        summary['front_sign_clearance_minima_m'] = minima
        summary['front_sign_clearance_scope'] = 'sampled conservative separation lower bound using full body/sensor z=[-0.30,+0.35]m in base frame and actual pitch/roll; not continuous mesh distance or full-world coverage'
        passed = passed and len(samples) >= 20 and min(minima.values(), default=-math.inf) >= .10
    if args.motion_grid_output:
        args.motion_grid_output.parent.mkdir(parents=True,exist_ok=True)
        with args.motion_grid_output.open('x') as output:
            json.dump({'schema_version':1,'scope':'raw measured columns before persistence, inflation and motion forecast; zero is not terrain safety',
                'overflow':node.motion_grid_overflow,'frames':node.motion_grids,
                'tracking_frames':node.motion_tracking_grids,
                'tracking_scope':'actual tracker input: body columns plus measured upper witnesses around existing rigid identities; not collision/free-space evidence',
                'topology_grids':node.topology_grids,'topology_routes':node.topology_routes,
                'topology_scope':'optimistic global routing diagnostics; unknown routing is not an executable or certified path'},output,allow_nan=False)
        if node.motion_grid_overflow or len(node.motion_grids)<2:passed=False
    if node.safety_status_overflow:passed=False
    if input_stream_failure is not None:passed=False
    localization=localization_agreement(node.localization_poses,node.localization_truth)
    if node.localization_overflow:localization.update(accepted=False,reason='localization recorder overflow')
    summary['localization_accepted']=localization['accepted']
    summary['localization_xy_max_m']=localization.get('xy_max_m')
    summary['localization_z_max_m']=localization.get('z_max_m')
    summary['localization_yaw_max_rad']=localization.get('yaw_max_rad')
    passed=passed and localization['accepted']
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "schema_version": 10, "summary": summary,"accepted":bool(passed),
            "evaluation_map_alignment":dict(tx_ty_yaw=node.observer_map_alignment,
                scope='fixed stationary paired truth-to-map transform for observer ROIs only; never a navigation input'),
            "localization_assessment":localization,
            "localization_pose_columns":['stamp_s','x','y','z','yaw'],
            "localization_poses":node.localization_poses,
            "localization_attitude_columns":["stamp_s","qx","qy","qz","qw"],
            "localization_attitudes":node.localization_attitudes,
            "localization_truth":node.localization_truth,
            "input_stream_failure":input_stream_failure,
            "safety_status":node.safety_status_samples,
            "preparation":preparation,
            "safety_status_overflow":node.safety_status_overflow,
            "qualification_version": ("observed_ground_early_recovery_raw_encounter_v4" if args.clearance_observation_window is not None
                else "observed_ground_crossing_sustained_all_layers_raw_v2" if args.require_clearance_recovery
                else "dynamic_encounter_body_clearance_raw_v2" if args.require_dynamic_encounter
                else "sustained_all_layers_probability_pose_plus_arrival_v3" if args.scenario=="clearance"
                else "physical_body_and_settled_arrival_v1"),
            "clearance_temporal_diagnostics":temporal,
            "clearance_whole_record_diagnostics":whole_record_temporal,
            "observed_ground_recovery":observed_recovery,
            "probability_before_vacate":node.probability_before_vacate,
            "clearance_series":{
                "laser":node.ghost_laser_samples,
                "laser_obstacle":node.ghost_laser_obstacle_samples,
                "risk_columns":["age_s","valid","risk"],
                "baseline_risk":node.ghost_map_samples,"enhanced_risk":node.ghost_enhanced_samples,
                "probability_pose":node.ghost_probability_samples,
                "probability_columns":["age_s","probability_valid","failure_probability_upper"],
                "probability_probe":{"x_m":node.obstacle_x-.55,"y_m":0.,"yaw_rad":0.,
                    "coordinates_frame":"authored course; transformed to map with evaluation_map_alignment",
                    "map_xy_yaw":([*course_to_map(node.obstacle_x-.55,0.,node.observer_map_alignment),node.observer_map_alignment[2]]
                        if node.observer_map_alignment is not None else None),
                    "state_limit":.05,"scope":"producer full-footprint pose bound at fixed lane pose; no extra spatial dilation"},
                "live":node.ghost_live_samples,"registered":node.registered_after_vacate},
            "step_front_face_clearances":step_clearances if args.scenario=='step' else None,
            "course_body_clearances": course_clearances,
            "front_sign_body_clearances": background_clearances,
            "front_sign_planar_projection_clearances": background_clearances_planar,
            "truth_start": truth_start, "truth_goal": truth_goal,
            "map_start": [op.x, op.y, op.z, oyaw], "truth_yaw_start": tyaw,
            "truth_columns": ["wall_s", "x", "y", "z", "pitch", "roll", "stamp_s", "yaw"],
            "truth": samples, "path_history": node.path_history,
            "tracking_errors": node.tracking_errors,
            "control_columns": ["stamp_s", "linear_mps", "steering_rad"],
            "controls": node.control_samples, "obstacle": node.obstacle_pose_samples,
            "sampled_body_clearances":body_clearances,
            "odom_twist": node.odom_samples,
        }, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    print("PASS" if passed else "FAIL", flush=True)
    return 0 if passed else 1


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("detour", "ramp", "step", "emerge", *CROSSING_SCENARIOS, "clearance", *COMPOSITE_SCENARIOS), required=True)
    parser.add_argument("--require-dynamic-encounter",action="store_true")
    parser.add_argument('--require-clearance-recovery',action='store_true',
        help='Crossing only: require measured-safe support before motion and all-layer sustained recovery <=1.5s')
    parser.add_argument('--clearance-observation-window',type=float,choices=(EARLY_RECOVERY_WINDOW_S,),
        help='Explicit 4s near-field window: 1.5s clearing + 2s confirmation + .5s publication; whole record retained')
    parser.add_argument('--evaluate-front-sign', action='store_true',
        help='Require >=.10m sampled planar body clearance from both known background sign components')
    parser.add_argument('--ramp-direction', choices=('uphill', 'downhill'), default='uphill')
    parser.add_argument('--ramp-rise', type=float, default=.45)
    parser.add_argument("--distance", type=float, default=7.0)
    parser.add_argument("--lateral", type=float, default=0.0)
    parser.add_argument("--goal-mode", choices=("forward", "behind"), default="forward")
    parser.add_argument("--output", type=Path)
    parser.add_argument('--motion-grid-output',type=Path)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--reach-distance", type=float, default=0.65)
    args = parser.parse_args(argv)
    if args.require_clearance_recovery and args.scenario not in CROSSING_SCENARIOS:
        parser.error('observed-ground recovery requires a crossing fixture; initially occluded clearance remains separate')
    if args.clearance_observation_window is not None and not args.require_clearance_recovery:
        parser.error('a finite near-field window requires the separate observed-ground recovery protocol')
    if args.scenario in (*COMPOSITE_SCENARIOS, 'ramp_crossing') and (args.goal_mode != 'forward' or args.lateral != 0. or
            args.distance < minimum_course_distance(args.scenario)):
        parser.error('composite course goal must lie beyond the complete forward course')
    if args.ramp_direction == 'downhill' and args.scenario not in RAMP_SCENARIOS:
        parser.error('--ramp-direction downhill requires a ramp scenario')
    if args.require_dynamic_encounter and args.scenario not in CROSSING_SCENARIOS:
        parser.error('dynamic encounter gate requires --scenario crossing or ramp_crossing')
    if not math.isfinite(args.ramp_rise) or not .1 <= args.ramp_rise <= 1.2:
        parser.error('--ramp-rise must be finite and within [0.1, 1.2] metres')
    if (not all(math.isfinite(v) for v in (args.distance, args.lateral, args.timeout, args.reach_distance))
            or args.reach_distance <= 0. or args.distance <= args.reach_distance or args.timeout <= 2.0):
        parser.error("distance must exceed reach-distance and timeout must exceed 2 s")
    return args


def main():
    args = parse_args()
    rclpy.init()
    node = Trial(args.scenario,capture_motion=args.motion_grid_output is not None)
    try:
        return run(node, args)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
