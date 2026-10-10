#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

LINEAR_X="${1:-0.8}"
ANGULAR_Z="${2:-0.25}"
DURATION="${3:-7.0}"
ANGULAR_MODE="${4:-steer}"

unset PYTHONHOME
unset PYTHONPATH

if [[ -f /opt/ros/humble/setup.bash ]]; then
  source /opt/ros/humble/setup.bash
fi
if [[ -f "$ROOT_DIR/install/setup.bash" ]]; then
  source "$ROOT_DIR/install/setup.bash"
fi

set -u

ISAAC_RMW_IMPLEMENTATION="${HUNTER_ISAAC_RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
if [[ -n "${RMW_IMPLEMENTATION:-}" && "$RMW_IMPLEMENTATION" != "$ISAAC_RMW_IMPLEMENTATION" ]]; then
  echo "[hunter_cmd_probe] Overriding RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION -> $ISAAC_RMW_IMPLEMENTATION for Isaac ROS2 bridge compatibility"
fi
export RMW_IMPLEMENTATION="$ISAAC_RMW_IMPLEMENTATION"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"

echo "[hunter_cmd_probe] command: linear.x=${LINEAR_X} angular.z=${ANGULAR_Z} duration=${DURATION}s mode=${ANGULAR_MODE}"

python3 - "$LINEAR_X" "$ANGULAR_Z" "$DURATION" "$ANGULAR_MODE" <<'PY'
import math
import os
import statistics
import sys
import time
from bisect import bisect_left
from dataclasses import dataclass

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, PointCloud2


linear_x = float(sys.argv[1])
angular_z = float(sys.argv[2])
duration = max(0.1, float(sys.argv[3]))
angular_mode = sys.argv[4]
wheel_base = 0.549185
cmd_hz = 30.0


@dataclass
class OdomSample:
    wall: float
    stamp: float
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float
    vx: float
    wz: float


def stamp_seconds(msg):
    stamp = msg.header.stamp
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def quat_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def quat_rpy(q):
    sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
    cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (q.w * q.y - q.z * q.x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)
    yaw = quat_yaw(q)
    return roll, pitch, yaw


def wrap_pi(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def rate(times):
    if len(times) < 2:
        return 0.0
    span = times[-1] - times[0]
    return (len(times) - 1) / span if span > 0.0 else 0.0


def summarize_odom(label, samples, start_wall, end_wall):
    window = [s for s in samples if start_wall <= s.wall <= end_wall]
    if len(window) < 2:
        return f"[hunter_cmd_probe] {label}: samples={len(samples)}, not enough samples in command window"
    first = window[0]
    last = window[-1]
    dx = last.x - first.x
    dy = last.y - first.y
    dxy = math.hypot(dx, dy)
    dz = last.z - first.z
    dyaw = wrap_pi(last.yaw - first.yaw)
    mean_roll = statistics.fmean(s.roll for s in window)
    mean_pitch = statistics.fmean(s.pitch for s in window)
    max_roll = max(abs(s.roll) for s in window)
    max_pitch = max(abs(s.pitch) for s in window)
    mean_vx = statistics.fmean(s.vx for s in window)
    mean_wz = statistics.fmean(s.wz for s in window)
    sim_span = max(0.0, last.stamp - first.stamp)
    wall_span = max(0.0, last.wall - first.wall)
    integrated_vx = 0.0
    for previous, current in zip(window, window[1:]):
        dt = current.stamp - previous.stamp
        if 0.0 < dt < 0.5:
            integrated_vx += 0.5 * (previous.vx + current.vx) * dt
    return (
        f"[hunter_cmd_probe] {label}: samples={len(samples)}, window={len(window)}, "
        f"sim/wall={sim_span:.3f}/{wall_span:.3f}s rtf={sim_span / max(1e-6, wall_span):.3f} "
        f"dxy={dxy:.3f}m dxyz=({dx:+.3f},{dy:+.3f},{dz:+.3f})m dyaw={dyaw:+.3f}rad "
        f"mean_rp=({math.degrees(mean_roll):+.1f},{math.degrees(mean_pitch):+.1f})deg "
        f"end_rp=({math.degrees(last.roll):+.1f},{math.degrees(last.pitch):+.1f})deg "
        f"max_abs_rp=({math.degrees(max_roll):.1f},{math.degrees(max_pitch):.1f})deg "
        f"mean_twist_vx={mean_vx:+.3f} mean_twist_wz={mean_wz:+.3f} integrated_vx={integrated_vx:+.3f}m"
    )


def odom_delta(samples, start_wall, end_wall):
    window = [s for s in samples if start_wall <= s.wall <= end_wall]
    if len(window) < 2:
        return None
    first = window[0]
    last = window[-1]
    return math.hypot(last.x - first.x, last.y - first.y), wrap_pi(last.yaw - first.yaw)


def interpolate_odom(samples, stamp):
    if not samples:
        return None
    stamps = [sample.stamp for sample in samples]
    index = bisect_left(stamps, stamp)
    if index <= 0:
        return samples[0]
    if index >= len(samples):
        return samples[-1]
    before = samples[index - 1]
    after = samples[index]
    span = after.stamp - before.stamp
    if span <= 1e-9:
        return before
    ratio = max(0.0, min(1.0, (stamp - before.stamp) / span))
    yaw_delta = wrap_pi(after.yaw - before.yaw)
    return OdomSample(
        wall=before.wall + ratio * (after.wall - before.wall),
        stamp=stamp,
        x=before.x + ratio * (after.x - before.x),
        y=before.y + ratio * (after.y - before.y),
        z=before.z + ratio * (after.z - before.z),
        roll=before.roll + ratio * wrap_pi(after.roll - before.roll),
        pitch=before.pitch + ratio * wrap_pi(after.pitch - before.pitch),
        yaw=wrap_pi(before.yaw + ratio * yaw_delta),
        vx=before.vx + ratio * (after.vx - before.vx),
        wz=before.wz + ratio * (after.wz - before.wz),
    )


def aligned_odom_deltas(truth, estimate, start_wall, end_wall):
    truth_window = [s for s in truth if start_wall <= s.wall <= end_wall]
    estimate_window = [s for s in estimate if start_wall <= s.wall <= end_wall]
    if len(truth_window) < 2 or len(estimate_window) < 2:
        return None
    common_start = max(truth_window[0].stamp, estimate_window[0].stamp)
    common_end = min(truth_window[-1].stamp, estimate_window[-1].stamp)
    if common_end <= common_start:
        return None
    truth_start = interpolate_odom(truth_window, common_start)
    truth_end = interpolate_odom(truth_window, common_end)
    estimate_start = interpolate_odom(estimate_window, common_start)
    estimate_end = interpolate_odom(estimate_window, common_end)
    return {
        "span": common_end - common_start,
        "truth_xy": math.hypot(truth_end.x - truth_start.x, truth_end.y - truth_start.y),
        "truth_yaw": wrap_pi(truth_end.yaw - truth_start.yaw),
        "estimate_xy": math.hypot(
            estimate_end.x - estimate_start.x,
            estimate_end.y - estimate_start.y,
        ),
        "estimate_yaw": wrap_pi(estimate_end.yaw - estimate_start.yaw),
    }


def odom_motion_metrics(samples, start_wall, end_wall):
    window = [s for s in samples if start_wall <= s.wall <= end_wall]
    if len(window) < 2:
        return None
    integrated_vx = 0.0
    for previous, current in zip(window, window[1:]):
        dt = current.stamp - previous.stamp
        if 0.0 < dt < 0.5:
            integrated_vx += 0.5 * (previous.vx + current.vx) * dt
    return {
        "sim_span": max(0.0, window[-1].stamp - window[0].stamp),
        "integrated_vx": integrated_vx,
        "dyaw": wrap_pi(window[-1].yaw - window[0].yaw),
    }


rclpy.init()
node = rclpy.create_node("hunter_command_response_probe")

discovery_deadline = time.monotonic() + 0.75
while time.monotonic() < discovery_deadline:
    rclpy.spin_once(node, timeout_sec=0.05)
existing_cmd_publishers = node.get_publishers_info_by_topic("/cmd_vel")
if existing_cmd_publishers and os.environ.get("HUNTER_CMD_PROBE_ALLOW_MULTIPLE_PUBLISHERS", "0") != "1":
    names = ", ".join(sorted({info.node_name for info in existing_cmd_publishers}))
    print(
        f"[hunter_cmd_probe] ERROR: /cmd_vel already has {len(existing_cmd_publishers)} publisher(s): {names}. "
        "Stop terrain_nav or launch it with nav:=false before measuring chassis response.",
        file=sys.stderr,
    )
    node.destroy_node()
    rclpy.shutdown()
    raise SystemExit(2)

sensor_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=300)
sensor_qos.reliability = ReliabilityPolicy.BEST_EFFORT
sensor_qos.durability = DurabilityPolicy.VOLATILE

cmd_pub = node.create_publisher(Twist, "/cmd_vel", 10)
truth_samples = []
fast_samples = []
imu_samples = []
lidar_times = []
lidar_stamp_times = []
lidar_points = []
lidar_fields = {"value": None}
map_times = []
map_stamp_times = []
map_points = []
command_subscriber_lost = False


def make_odom_cb(dst):
    def cb(msg):
        p = msg.pose.pose.position
        roll, pitch, yaw = quat_rpy(msg.pose.pose.orientation)
        dst.append(
            OdomSample(
                time.monotonic(),
                stamp_seconds(msg),
                float(p.x),
                float(p.y),
                float(p.z),
                roll,
                pitch,
                yaw,
                float(msg.twist.twist.linear.x),
                float(msg.twist.twist.angular.z),
            )
        )
    return cb


def imu_cb(msg):
    imu_samples.append(
        (
            time.monotonic(),
            stamp_seconds(msg),
            float(msg.angular_velocity.x),
            float(msg.angular_velocity.y),
            float(msg.angular_velocity.z),
            float(msg.linear_acceleration.x),
            float(msg.linear_acceleration.y),
            float(msg.linear_acceleration.z),
        )
    )


def lidar_cb(msg):
    lidar_times.append(time.monotonic())
    lidar_stamp_times.append(stamp_seconds(msg))
    lidar_points.append(int(msg.width) * int(msg.height))
    if lidar_fields["value"] is None:
        lidar_fields["value"] = ",".join(field.name for field in msg.fields)


def map_cb(msg):
    map_times.append(time.monotonic())
    map_stamp_times.append(stamp_seconds(msg))
    map_points.append(int(msg.width) * int(msg.height))


node.create_subscription(Odometry, "/isaac/ground_truth/odometry", make_odom_cb(truth_samples), 100)
node.create_subscription(Odometry, "/Odometry", make_odom_cb(fast_samples), 100)
node.create_subscription(Imu, "/livox/imu", imu_cb, sensor_qos)
node.create_subscription(PointCloud2, "/livox/lidar", lidar_cb, sensor_qos)
map_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=20)
map_qos.reliability = ReliabilityPolicy.RELIABLE
map_qos.durability = DurabilityPolicy.VOLATILE
node.create_subscription(PointCloud2, "/Laser_map", map_cb, map_qos)

time.sleep(0.2)
sub_count = len(node.get_subscriptions_info_by_topic("/cmd_vel"))
print(f"[hunter_cmd_probe] /cmd_vel subscriptions before command: {sub_count}")
if sub_count == 0:
    print(
        "[hunter_cmd_probe] ERROR: no /cmd_vel subscriber was discovered; "
        "the Isaac process may have exited or the ROS middleware/domain does not match.",
        file=sys.stderr,
    )
    node.destroy_node()
    rclpy.shutdown()
    raise SystemExit(2)

zero = Twist()
cmd = Twist()
cmd.linear.x = linear_x
cmd.angular.z = angular_z

warmup_until = time.monotonic() + 1.0
while time.monotonic() < warmup_until:
    cmd_pub.publish(zero)
    rclpy.spin_once(node, timeout_sec=0.02)

start_wall = time.monotonic()
end_wall = start_wall + duration
next_pub = start_wall
period = 1.0 / cmd_hz
while time.monotonic() < end_wall:
    now = time.monotonic()
    if now >= next_pub:
        cmd_pub.publish(cmd)
        next_pub += period
    rclpy.spin_once(node, timeout_sec=0.01)

stop_until = time.monotonic() + 1.0
while time.monotonic() < stop_until:
    cmd_pub.publish(zero)
    rclpy.spin_once(node, timeout_sec=0.02)

sub_count_after = len(node.get_subscriptions_info_by_topic("/cmd_vel"))
if sub_count_after == 0:
    command_subscriber_lost = True
    print("[hunter_cmd_probe] ERROR: /cmd_vel subscriber disappeared during the command window", file=sys.stderr)

if angular_mode == "yaw_rate":
    nominal_wz = angular_z
else:
    nominal_wz = linear_x * math.tan(angular_z) / wheel_base if abs(linear_x) > 1e-6 else 0.0

print(f"[hunter_cmd_probe] /cmd_vel subscriptions after command: {sub_count_after}")
truth_motion = odom_motion_metrics(truth_samples, start_wall, end_wall)
if truth_motion is None:
    print(f"[hunter_cmd_probe] nominal command yaw_rate={nominal_wz:+.3f}rad/s")
elif angular_mode == "steer":
    target_curvature = math.tan(angular_z) / wheel_base
    measured_curvature = truth_motion["dyaw"] / max(1e-6, abs(truth_motion["integrated_vx"]))
    motion_expected_dyaw = truth_motion["integrated_vx"] * target_curvature
    print(
        f"[hunter_cmd_probe] nominal yaw_rate={nominal_wz:+.3f}rad/s, "
        f"sim-time ideal_dyaw={nominal_wz * truth_motion['sim_span']:+.3f}rad, "
        f"motion-based expected/actual dyaw={motion_expected_dyaw:+.3f}/{truth_motion['dyaw']:+.3f}rad, "
        f"target/actual curvature={target_curvature:+.3f}/{measured_curvature:+.3f} 1/m"
    )
else:
    print(
        f"[hunter_cmd_probe] nominal command yaw_rate={nominal_wz:+.3f}rad/s, "
        f"sim-time ideal_dyaw={nominal_wz * truth_motion['sim_span']:+.3f}rad"
    )
print(summarize_odom("Isaac truth /isaac/ground_truth/odometry", truth_samples, start_wall, end_wall))
print(summarize_odom("FAST-LIO /Odometry", fast_samples, start_wall, end_wall))

aligned_delta = aligned_odom_deltas(truth_samples, fast_samples, start_wall, end_wall)
if aligned_delta and abs(aligned_delta["truth_yaw"]) > 1e-3:
    print(
        f"[hunter_cmd_probe] common sim-time span={aligned_delta['span']:.3f}s; "
        "FAST-LIO/Isaac yaw ratio="
        f"{aligned_delta['estimate_yaw'] / aligned_delta['truth_yaw']:+.3f}, "
        f"xy ratio={aligned_delta['estimate_xy'] / max(1e-6, aligned_delta['truth_xy']):+.3f}"
    )

imu_window = [s for s in imu_samples if start_wall <= s[0] <= end_wall]
if imu_window:
    mean_gx = statistics.fmean(s[2] for s in imu_window)
    mean_gy = statistics.fmean(s[3] for s in imu_window)
    mean_gz = statistics.fmean(s[4] for s in imu_window)
    mean_az = statistics.fmean(s[7] for s in imu_window)
    print(
        f"[hunter_cmd_probe] /livox/imu: samples={len(imu_samples)}, "
        f"wall/sim rate={rate([s[0] for s in imu_samples]):.1f}/{rate([s[1] for s in imu_samples]):.1f}Hz, "
        f"command-window mean gyro=({mean_gx:+.3f},{mean_gy:+.3f},{mean_gz:+.3f}) "
        f"mean_acc_z={mean_az:+.3f}"
    )
else:
    print(f"[hunter_cmd_probe] /livox/imu: samples={len(imu_samples)}, no IMU samples in command window")

if lidar_times:
    mean_points = statistics.fmean(lidar_points) if lidar_points else 0.0
    print(
        f"[hunter_cmd_probe] /livox/lidar: samples={len(lidar_times)}, "
        f"wall/sim rate={rate(lidar_times):.2f}/{rate(lidar_stamp_times):.2f}Hz, points min/mean/max="
        f"{min(lidar_points)}/{mean_points:.1f}/{max(lidar_points)}, fields={lidar_fields['value']}"
    )
else:
    print("[hunter_cmd_probe] /livox/lidar: no samples")

if map_times:
    print(
        f"[hunter_cmd_probe] /Laser_map accumulated: samples={len(map_times)}, "
        f"wall/sim rate={rate(map_times):.2f}/{rate(map_stamp_times):.2f}Hz, "
        f"points first/last/min/max={map_points[0]}/{map_points[-1]}/{min(map_points)}/{max(map_points)}"
    )
else:
    print("[hunter_cmd_probe] /Laser_map: no accumulated map samples")

node.destroy_node()
rclpy.shutdown()
if command_subscriber_lost:
    raise SystemExit(2)
PY
