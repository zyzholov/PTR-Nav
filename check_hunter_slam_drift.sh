#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DURATION="${1:-20}"

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
  echo "[hunter_slam_drift] Overriding RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION -> $ISAAC_RMW_IMPLEMENTATION for Isaac ROS2 bridge compatibility"
fi
export RMW_IMPLEMENTATION="$ISAAC_RMW_IMPLEMENTATION"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"

FASTLIO_ODOM_TOPIC="${FASTLIO_ODOM_TOPIC:-/Odometry}"
ISAAC_ODOM_TOPIC="${ISAAC_ODOM_TOPIC:-/isaac/ground_truth/odometry}"
LIDAR_TOPIC="${LIDAR_TOPIC:-/livox/lidar}"
IMU_TOPIC="${IMU_TOPIC:-/livox/imu}"
CMD_TOPIC="${CMD_TOPIC:-/cmd_vel}"

echo "[hunter_slam_drift] Sampling for ${DURATION}s"
echo "[hunter_slam_drift] FAST-LIO odom: ${FASTLIO_ODOM_TOPIC}"
echo "[hunter_slam_drift] Isaac truth : ${ISAAC_ODOM_TOPIC}"
echo "[hunter_slam_drift] LiDAR/IMU   : ${LIDAR_TOPIC}, ${IMU_TOPIC}"

python3 - "$DURATION" "$FASTLIO_ODOM_TOPIC" "$ISAAC_ODOM_TOPIC" "$LIDAR_TOPIC" "$IMU_TOPIC" "$CMD_TOPIC" <<'PY'
import math
import statistics
import sys
import time
from dataclasses import dataclass

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, PointCloud2


duration = float(sys.argv[1])
fast_odom_topic = sys.argv[2]
isaac_odom_topic = sys.argv[3]
lidar_topic = sys.argv[4]
imu_topic = sys.argv[5]
cmd_topic = sys.argv[6]


@dataclass
class OdomSample:
    wall: float
    stamp: float
    x: float
    y: float
    z: float
    yaw: float


def stamp_seconds(msg):
    stamp = msg.header.stamp
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def quat_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


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


def nearest_by_stamp(samples, stamp, max_dt):
    best = None
    best_dt = max_dt
    for sample in samples:
        dt = abs(sample.stamp - stamp)
        if dt <= best_dt:
            best = sample
            best_dt = dt
    return best


def summarize_odom_alignment(fast_samples, isaac_samples):
    pairs = []
    for fast in fast_samples:
        truth = nearest_by_stamp(isaac_samples, fast.stamp, 0.12)
        if truth is not None:
            pairs.append((fast, truth))
    if len(pairs) < 3:
        return "[hunter_slam_drift] WARN: not enough timestamp-aligned FAST-LIO/Isaac odom pairs for drift comparison"

    f0, t0 = pairs[0]
    xy_errors = []
    z_errors = []
    yaw_errors = []
    for fast, truth in pairs:
        fdx = fast.x - f0.x
        fdy = fast.y - f0.y
        fdz = fast.z - f0.z
        tdx = truth.x - t0.x
        tdy = truth.y - t0.y
        tdz = truth.z - t0.z
        xy_errors.append(math.hypot(fdx - tdx, fdy - tdy))
        z_errors.append(abs(fdz - tdz))
        yaw_errors.append(abs(wrap_pi((fast.yaw - f0.yaw) - (truth.yaw - t0.yaw))))

    return (
        "[hunter_slam_drift] odom drift relative to Isaac truth: "
        f"pairs={len(pairs)}, xy_last={xy_errors[-1]:.3f}m, xy_max={max(xy_errors):.3f}m, "
        f"z_last={z_errors[-1]:.3f}m, z_max={max(z_errors):.3f}m, "
        f"yaw_last={math.degrees(yaw_errors[-1]):.2f}deg, yaw_max={math.degrees(max(yaw_errors)):.2f}deg"
    )


rclpy.init()
node = rclpy.create_node("hunter_slam_drift_probe")

sensor_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=100)
sensor_qos.reliability = ReliabilityPolicy.BEST_EFFORT
sensor_qos.durability = DurabilityPolicy.VOLATILE

fast_odom = []
isaac_odom = []
lidar_times = []
lidar_stamp_times = []
imu_times = []
imu_stamp_times = []
cmd_times = []
point_counts = []
point_fields = None
point_step = None
last_cmd = None
last_imu = None


def fast_odom_cb(msg):
    p = msg.pose.pose.position
    fast_odom.append(OdomSample(time.monotonic(), stamp_seconds(msg), p.x, p.y, p.z, quat_yaw(msg.pose.pose.orientation)))


def isaac_odom_cb(msg):
    p = msg.pose.pose.position
    isaac_odom.append(OdomSample(time.monotonic(), stamp_seconds(msg), p.x, p.y, p.z, quat_yaw(msg.pose.pose.orientation)))


def lidar_cb(msg):
    global point_fields, point_step
    lidar_times.append(time.monotonic())
    lidar_stamp_times.append(stamp_seconds(msg))
    point_counts.append(int(msg.width) * int(msg.height))
    if point_fields is None:
        point_fields = ",".join(field.name for field in msg.fields)
        point_step = int(msg.point_step)


def imu_cb(msg):
    global last_imu
    imu_times.append(time.monotonic())
    imu_stamp_times.append(stamp_seconds(msg))
    last_imu = (
        float(msg.angular_velocity.x),
        float(msg.angular_velocity.y),
        float(msg.angular_velocity.z),
        float(msg.linear_acceleration.x),
        float(msg.linear_acceleration.y),
        float(msg.linear_acceleration.z),
    )


def cmd_cb(msg):
    global last_cmd
    cmd_times.append(time.monotonic())
    last_cmd = (float(msg.linear.x), float(msg.angular.z))


node.create_subscription(Odometry, fast_odom_topic, fast_odom_cb, 50)
node.create_subscription(Odometry, isaac_odom_topic, isaac_odom_cb, 50)
node.create_subscription(PointCloud2, lidar_topic, lidar_cb, sensor_qos)
node.create_subscription(Imu, imu_topic, imu_cb, sensor_qos)
node.create_subscription(Twist, cmd_topic, cmd_cb, 20)

end_time = time.monotonic() + max(0.1, duration)
while time.monotonic() < end_time:
    rclpy.spin_once(node, timeout_sec=0.05)

fast_pubs = node.get_publishers_info_by_topic(fast_odom_topic)
truth_pubs = node.get_publishers_info_by_topic(isaac_odom_topic)

node.destroy_node()
rclpy.shutdown()

print(f"[hunter_slam_drift] {lidar_topic}: count={len(lidar_times)}, rate={rate(lidar_times):.3f}Hz")
if point_counts:
    mean_points = statistics.fmean(point_counts)
    print(
        f"[hunter_slam_drift] {lidar_topic}: points min/mean/max="
        f"{min(point_counts)}/{mean_points:.1f}/{max(point_counts)}, fields={point_fields}, point_step={point_step}"
    )
    if point_fields and "time" not in point_fields.split(",") and "t" not in point_fields.split(","):
        print("[hunter_slam_drift] WARN: PointCloud2 has no per-point time/t field; Isaac FAST-LIO config should keep synthetic_point_time=false")
print(f"[hunter_slam_drift] {imu_topic}: count={len(imu_times)}, rate={rate(imu_times):.3f}Hz")
if last_imu is not None:
    print(
        f"[hunter_slam_drift] last IMU: gyro=({last_imu[0]:+.3f},{last_imu[1]:+.3f},{last_imu[2]:+.3f}) rad/s, "
        f"acc=({last_imu[3]:+.3f},{last_imu[4]:+.3f},{last_imu[5]:+.3f}) m/s^2"
    )
print(f"[hunter_slam_drift] {cmd_topic}: count={len(cmd_times)}, rate={rate(cmd_times):.3f}Hz")
if last_cmd is not None:
    print(f"[hunter_slam_drift] last cmd_vel: linear.x={last_cmd[0]:+.3f} m/s, angular.z={last_cmd[1]:+.3f}")

print(f"[hunter_slam_drift] {fast_odom_topic}: samples={len(fast_odom)}, pubs={len(fast_pubs)}, rate={rate([s.wall for s in fast_odom]):.3f}Hz")
if len(fast_pubs) > 1:
    print(f"[hunter_slam_drift] WARN: {fast_odom_topic} has {len(fast_pubs)} publishers. Keep FAST-LIO on /Odometry and Isaac truth on /isaac/ground_truth/odometry.")
print(f"[hunter_slam_drift] {isaac_odom_topic}: samples={len(isaac_odom)}, pubs={len(truth_pubs)}, rate={rate([s.wall for s in isaac_odom]):.3f}Hz")
print(summarize_odom_alignment(fast_odom, isaac_odom))

failed = False
if len(lidar_times) < 2:
    print(f"[hunter_slam_drift] ERROR: no usable LiDAR samples from {lidar_topic}", file=sys.stderr)
    failed = True
if len(imu_times) < 2:
    print(f"[hunter_slam_drift] ERROR: no usable IMU samples from {imu_topic}", file=sys.stderr)
    failed = True
if len(fast_odom) < 2:
    print(f"[hunter_slam_drift] WARN: no FAST-LIO odometry yet; start ./run_hunter_fastlio.sh after Isaac sensors are publishing")
if len(isaac_odom) < 2:
    print(f"[hunter_slam_drift] WARN: no Isaac truth odometry yet; start Isaac with --drive-mode cmd_vel")

sys.exit(2 if failed else 0)
PY
