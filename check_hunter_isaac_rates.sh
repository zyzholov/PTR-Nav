#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f /opt/ros/humble/setup.bash ]]; then
  source /opt/ros/humble/setup.bash
fi
if [[ -f "$ROOT_DIR/install/setup.bash" ]]; then
  source "$ROOT_DIR/install/setup.bash"
fi

set -u

ISAAC_RMW_IMPLEMENTATION="${HUNTER_ISAAC_RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
if [[ -n "${RMW_IMPLEMENTATION:-}" && "$RMW_IMPLEMENTATION" != "$ISAAC_RMW_IMPLEMENTATION" ]]; then
  echo "[hunter_isaac_rates] Overriding RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION -> $ISAAC_RMW_IMPLEMENTATION for Isaac ROS2 bridge compatibility"
fi
export RMW_IMPLEMENTATION="$ISAAC_RMW_IMPLEMENTATION"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"

LIDAR_MIN_HZ="${LIDAR_MIN_HZ:-9.0}"
IMU_MIN_HZ="${IMU_MIN_HZ:-180.0}"
LIDAR_WARN_MIN_POINTS="${LIDAR_WARN_MIN_POINTS:-10000}"

measure_rate() {
  local topic="$1"
  local expected="$2"
  local minimum="$3"
  local seconds="$4"
  local window="$5"

  echo "[hunter_isaac_rates] Measuring $topic, expected ${expected}Hz, minimum ${minimum}Hz ..."
  local output
  local rc=0
  output="$(timeout "$((seconds + 5))" python3 - "$topic" "$seconds" 2>&1 <<'PY'
import sys
import time

import rclpy
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, PointCloud2

topic = sys.argv[1]
seconds = float(sys.argv[2])
msg_type = Imu if topic.endswith("imu") or "imu" in topic else PointCloud2

rclpy.init()
node = rclpy.create_node("hunter_isaac_rate_probe")
qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10)
qos.reliability = ReliabilityPolicy.BEST_EFFORT
qos.durability = DurabilityPolicy.VOLATILE

samples = []
point_counts = []
point_fields = None
point_step = None

def cb(msg):
    global point_fields, point_step
    samples.append(time.monotonic())
    if hasattr(msg, "width") and hasattr(msg, "height"):
        point_counts.append(int(msg.width) * int(msg.height))
        if point_fields is None:
            point_fields = ",".join(field.name for field in msg.fields)
            point_step = int(msg.point_step)

node.create_subscription(msg_type, topic, cb, qos)
end_time = time.monotonic() + seconds
while time.monotonic() < end_time:
    rclpy.spin_once(node, timeout_sec=0.1)

node.destroy_node()
rclpy.shutdown()

if len(samples) < 2:
    print(f"no rate sample: count={len(samples)}")
    sys.exit(2)

duration = samples[-1] - samples[0]
rate = (len(samples) - 1) / duration if duration > 0.0 else 0.0
print(f"average rate: {rate:.6f}")
print(f"sample count: {len(samples)}")
if point_counts:
    point_mean = sum(point_counts) / len(point_counts)
    print(f"point fields: {point_fields}")
    print(f"point step: {point_step}")
    print(f"point count min/mean/max: {min(point_counts)}/{point_mean:.1f}/{max(point_counts)}")
PY
)" || rc=$?
  if [[ "$rc" -ne 0 && "$rc" -ne 124 ]]; then
    echo "[hunter_isaac_rates] ERROR: failed to measure $topic" >&2
    sed -n '1,12p' <<<"$output" >&2
    return 1
  fi

  local rate
  rate="$(awk '/average rate:/ {rate=$3} END {print rate}' <<<"$output")"
  if [[ -z "$rate" ]]; then
    echo "[hunter_isaac_rates] ERROR: no rate sample for $topic" >&2
    sed -n '1,12p' <<<"$output" >&2
    return 1
  fi

  awk -v topic="$topic" '/sample count:/ {
    print "[hunter_isaac_rates] " topic " " $0
  }
  /point fields:/ {
    print "[hunter_isaac_rates] " topic " " $0
  }
  /point step:/ {
    print "[hunter_isaac_rates] " topic " " $0
  }
  /point count min\/mean\/max:/ {
    print "[hunter_isaac_rates] " topic " " $0
  }' <<<"$output"

  if [[ "$topic" == "/livox/lidar" ]]; then
    local stats
    local mean_points
    stats="$(awk '/point count min\/mean\/max:/ {print $4; exit}' <<<"$output")"
    mean_points="$(cut -d/ -f2 <<<"$stats")"
    if [[ -n "$mean_points" ]]; then
      awk -v mean="$mean_points" -v minimum="$LIDAR_WARN_MIN_POINTS" 'BEGIN {
        if (mean + 0.0 < minimum + 0.0) {
          printf("[hunter_isaac_rates] WARNING: /livox/lidar mean point count %.1f is below %.1f; mapping will look sparse. Use the default mapping_dense profile or raise LIDAR_WARN_MIN_POINTS only for debug scenes.\n", mean, minimum) > "/dev/stderr";
        }
      }'
    fi
  fi

  awk -v topic="$topic" -v rate="$rate" -v minimum="$minimum" 'BEGIN {
    if (rate + 0.0 < minimum + 0.0) {
      printf("[hunter_isaac_rates] ERROR: %s rate %.3fHz is below %.3fHz\n", topic, rate, minimum) > "/dev/stderr";
      exit 1;
    }
    printf("[hunter_isaac_rates] OK: %s rate %.3fHz\n", topic, rate);
  }'
}

measure_rate "/livox/lidar" "10" "$LIDAR_MIN_HZ" 18 40
measure_rate "/imu/data_raw" "200" "$IMU_MIN_HZ" 10 400
measure_rate "/livox/imu" "200" "$IMU_MIN_HZ" 10 400

echo "[hunter_isaac_rates] OK: Isaac LiDAR/IMU rates are within thresholds"
