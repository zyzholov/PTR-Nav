#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC_DIR="$ROOT_DIR/src"
FASTLIO_SRC_DIR="$SRC_DIR/FAST_LIO"

CONFIG_PATH="$FASTLIO_SRC_DIR/config"
CONFIG_FILE="auto"
RVIZ="true"
LIDAR_WARN_MIN_POINTS="${HUNTER_FASTLIO_WARN_MIN_POINTS:-10000}"
EXTRA_ARGS=()

for arg in "$@"; do
  case "$arg" in
    -h|--help)
      cat <<'EOF'
Usage: scripts/sim/start_mapping.sh [rviz:=true|false] [config_file:=auto|FILE] [dynamic_clear_enabled:=true|false] [--show-args]
Starts FAST-LIO only after receiving simulation clock, LiDAR and external IMU.
PointCloud2 and Livox CustomMsg select different configurations automatically.
EOF
      exit 0
      ;;
    config_path:=*)
      CONFIG_PATH="${arg#config_path:=}"
      ;;
    config_file:=*)
      CONFIG_FILE="${arg#config_file:=}"
      ;;
    rviz:=*)
      RVIZ="${arg#rviz:=}"
      ;;
    *)
      EXTRA_ARGS+=("$arg")
      ;;
  esac
done

unset PYTHONHOME
unset PYTHONPATH
source /opt/ros/humble/setup.bash

if [[ -f "/home/z/workspace_zyz/driver/ws_livox/install/setup.bash" ]]; then
  source "/home/z/workspace_zyz/driver/ws_livox/install/setup.bash"
fi

if [[ -f "$ROOT_DIR/install/setup.bash" ]]; then
  source "$ROOT_DIR/install/setup.bash"
elif [[ -f "$ROOT_DIR/install/local_setup.bash" ]]; then
  source "$ROOT_DIR/install/local_setup.bash"
elif [[ -f "$SRC_DIR/install/setup.bash" ]]; then
  source "$SRC_DIR/install/setup.bash"
elif [[ -f "$SRC_DIR/install/local_setup.bash" ]]; then
  source "$SRC_DIR/install/local_setup.bash"
fi

if [[ "${RMW_IMPLEMENTATION:-}" != "rmw_fastrtps_cpp" ]]; then
  echo "[hunter_fastlio] Overriding RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-<unset>} -> rmw_fastrtps_cpp for Isaac ROS2 bridge compatibility"
  export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
fi
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"

source "$ROOT_DIR/scripts/lib/runtime.sh"
if hunter_show_args "${EXTRA_ARGS[@]}"; then
  echo "[hunter_fastlio] Wrapper config: path=$CONFIG_PATH file=$CONFIG_FILE rviz=$RVIZ"
  echo "[hunter_fastlio] file=auto selects isaac_hunter_mid360_pointcloud.yaml for Isaac PointCloud2."
  echo "[hunter_fastlio] ROS launch defaults below are not the wrapper's runtime selection."
  exec ros2 launch fast_lio mapping_hunter_sim.launch.py --show-args
fi

echo "[hunter_fastlio] Waiting for /clock, /livox/lidar and /livox/imu ..."
ready=0
for _ in $(seq 1 60); do
  topics="$(ros2 topic list --no-daemon 2>/dev/null || true)"
  if grep -q '^/clock$' <<< "$topics" && \
     grep -q '^/livox/lidar$' <<< "$topics" && \
     grep -q '^/livox/imu$' <<< "$topics"; then
    ready=1
    break
  fi
  sleep 1
done

if [[ $ready -ne 1 ]]; then
  echo "[hunter_fastlio] ERROR: timed out waiting for Isaac/Gazebo sensor topics." >&2
  exit 1
fi

wait_topic_sample() {
  local topic="$1"
  local field="${2:-}"
  local output=""
  local attempt=0
  for attempt in $(seq 1 5); do
    cmd=(ros2 topic echo --no-daemon --once --qos-profile sensor_data --spin-time 2 --no-arr --no-str)
    if [[ -n "$field" ]]; then
      cmd+=(--field "$field")
    fi
    cmd+=("$topic")
    if output="$(timeout 10 "${cmd[@]}" 2>&1)"; then
      return 0
    fi
    if (( attempt == 5 )); then
      echo "[hunter_fastlio] ERROR: topic $topic is visible but no data was received." >&2
      sed -n '1,8p' <<<"$output" >&2
      return 1
    fi
    sleep 1
  done
}

for spec in "/clock:" "/livox/lidar:header" "/livox/imu:header"; do
  topic="${spec%%:*}"
  field="${spec#*:}"
  wait_topic_sample "$topic" "$field"
done

lidar_msg_type="$(timeout 5 ros2 topic info -v /livox/lidar 2>/dev/null | awk -F': ' '/^Type:/ {print $2; exit}')"
if [[ -z "$lidar_msg_type" ]]; then
  echo "[hunter_fastlio] ERROR: could not determine /livox/lidar message type." >&2
  exit 1
fi

if [[ "$CONFIG_FILE" == "auto" ]]; then
  case "$lidar_msg_type" in
    sensor_msgs/msg/PointCloud2)
      CONFIG_FILE="isaac_hunter_mid360_pointcloud.yaml"
      ;;
    livox_ros_driver2/msg/CustomMsg|livox_ros_driver/msg/CustomMsg)
      CONFIG_FILE="gazebo_hunter_mid360.yaml"
      ;;
    *)
      echo "[hunter_fastlio] ERROR: unsupported /livox/lidar type: $lidar_msg_type" >&2
      echo "  Expected sensor_msgs/msg/PointCloud2 from Isaac, or livox_ros_driver2/msg/CustomMsg from Gazebo/Livox." >&2
      exit 1
      ;;
  esac
  echo "[hunter_fastlio] Detected /livox/lidar type: $lidar_msg_type -> using $CONFIG_FILE"
else
  selected_config="$CONFIG_PATH/$CONFIG_FILE"
  if [[ "$lidar_msg_type" == "sensor_msgs/msg/PointCloud2" ]] && \
     [[ -f "$selected_config" ]] && \
     grep -Eq '^[[:space:]]*lidar_type:[[:space:]]*1([[:space:]]|$)' "$selected_config"; then
    echo "[hunter_fastlio] ERROR: /livox/lidar is PointCloud2, but $CONFIG_FILE uses lidar_type: 1 (Livox CustomMsg)." >&2
    echo "  Use config_file:=isaac_hunter_mid360_pointcloud.yaml, or leave config_file unset for auto-detection." >&2
    exit 1
  fi
  echo "[hunter_fastlio] Detected /livox/lidar type: $lidar_msg_type -> using explicit $CONFIG_FILE"
fi

if [[ ! -f "$CONFIG_PATH/$CONFIG_FILE" ]]; then
  echo "[hunter_fastlio] ERROR: FAST_LIO config not found: $CONFIG_PATH/$CONFIG_FILE" >&2
  exit 1
fi

odom_info="$(timeout 5 ros2 topic info --no-daemon --verbose /Odometry 2>/dev/null || true)"
existing_odom_publishers="$(awk '/Publisher count:/ {print $3; exit}' <<<"$odom_info")"
if [[ -n "$existing_odom_publishers" && "$existing_odom_publishers" -gt 0 ]]; then
  echo "[hunter_fastlio] ERROR: /Odometry already has ${existing_odom_publishers} publisher(s) before FAST_LIO starts." >&2
  echo "  FAST_LIO also publishes /Odometry. Isaac truth odometry should use /isaac/ground_truth/odometry." >&2
  echo "  If RViz/control looks jumpy, stop the old publisher or restart Isaac with scripts/sim/start_simulation.sh defaults." >&2
  exit 2
fi

if [[ "$lidar_msg_type" == "sensor_msgs/msg/PointCloud2" ]]; then
  width="$(timeout 10 ros2 topic echo --no-daemon --once --qos-profile sensor_data --spin-time 2 --field width /livox/lidar 2>/dev/null | awk '/^[0-9]+$/ {print $1; exit}' || true)"
  height="$(timeout 10 ros2 topic echo --no-daemon --once --qos-profile sensor_data --spin-time 2 --field height /livox/lidar 2>/dev/null | awk '/^[0-9]+$/ {print $1; exit}' || true)"
  if [[ -n "$width" ]]; then
    if [[ -z "$height" ]]; then
      height=1
    fi
    first_cloud_points=$((width * height))
    echo "[hunter_fastlio] First /livox/lidar PointCloud2 sample: ${first_cloud_points} points (${width}x${height})"
    if (( first_cloud_points < LIDAR_WARN_MIN_POINTS )); then
      echo "[hunter_fastlio] WARNING: /livox/lidar looks sparse for FAST-LIO mapping." >&2
      echo "  Current sample has ${first_cloud_points} points; warning threshold is ${LIDAR_WARN_MIN_POINTS}." >&2
      echo "  Start Isaac with the default mapping profile or --lidar-quality mapping_dense for a denser 10Hz cloud." >&2
    fi
  fi
fi

echo "[hunter_fastlio] Launching FAST_LIO with ${CONFIG_FILE} (use_sim_time:=true, rviz:=${RVIZ})"
exec ros2 launch fast_lio mapping_hunter_sim.launch.py \
  use_sim_time:=true \
  config_path:="$CONFIG_PATH" \
  config_file:="$CONFIG_FILE" \
  rviz:="$RVIZ" \
  "${EXTRA_ARGS[@]}"
