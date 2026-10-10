#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  scripts/real/record_bag.sh [mode] [key:=value ...] [-- extra ros2 bag args]

Modes:
  core      Raw sensor + SLAM output + terrain grid + planner/MPC/chassis feedback. Default.
  raw       Only Livox, IMU, TF, and rosout. Best for FAST-LIO offline replay.
  camera    Only RealSense color image stream + camera_info, TF, and rosout.
  mapping   Raw sensor + FAST-LIO point clouds/maps. Good for SLAM debugging.
  planning  SLAM output + terrain + planner/MPC/chassis. Smaller than core.
  debug     core + RViz/debug marker/cloud topics, without global /Laser_map.
  full      debug + global /Laser_map and extra low-level chassis comm topics. Very large.

Keys:
  mode:=core|raw|camera|mapping|planning|debug|full
  camera:=false|true       Append RealSense color image stream to any mode.
  camera_depth:=false|true Append RealSense depth image stream and depth_to_color extrinsics.
  car_id:=0
  output:=/path/to/bag_dir
  storage:=auto|mcap|sqlite3
  compression:=none|file|message
  split_duration:=600
  max_cache_size:=1073741824
  qos_file:=/path/to/qos.yaml

Environment:
  BAG_ROOT              Default: <workspace>/bags
  LIVOX_ROS2_SETUP      Optional external Livox overlay setup.bash
  HEXMAN_SDK_SETUP      Default: /home/z/workspace_zyz/driver/hexman/sdk_neos_ws/install/setup.bash

Examples:
  scripts/real/record_bag.sh
  scripts/real/record_bag.sh raw
  scripts/real/record_bag.sh camera
  scripts/real/record_bag.sh core camera:=true
  scripts/real/record_bag.sh full output:=/data/bags/test_full
  scripts/real/record_bag.sh core -- --max-bag-size 21474836480
EOF
}

source /opt/ros/humble/setup.bash
unset PYTHONHOME
unset PYTHONPATH

if [[ -n "${LIVOX_ROS2_SETUP:-}" && -f "$LIVOX_ROS2_SETUP" ]]; then
  source "$LIVOX_ROS2_SETUP"
elif [[ -f "/home/z/workspace_zyz/driver/ws_livox/install/setup.bash" ]]; then
  source "/home/z/workspace_zyz/driver/ws_livox/install/setup.bash"
fi

HEXMAN_SDK_SETUP="${HEXMAN_SDK_SETUP:-/home/z/workspace_zyz/driver/hexman/sdk_neos_ws/install/setup.bash}"
if [[ -f "$HEXMAN_SDK_SETUP" ]]; then
  source "$HEXMAN_SDK_SETUP"
fi

if [[ -f "$ROOT_DIR/install/setup.bash" ]]; then
  source "$ROOT_DIR/install/setup.bash"
elif [[ -f "$ROOT_DIR/install/local_setup.bash" ]]; then
  source "$ROOT_DIR/install/local_setup.bash"
else
  echo "[record_real_bag] ERROR: $ROOT_DIR/install not found. Build from workspace root first:" >&2
  echo "  cd $ROOT_DIR && colcon build --symlink-install" >&2
  exit 1
fi

if [[ -z "${RMW_IMPLEMENTATION:-}" ]]; then
  export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
fi

MODE="${BAG_MODE:-core}"
CAR_ID="${CAR_ID:-0}"
RECORD_CAMERA="${BAG_RECORD_CAMERA:-false}"
RECORD_CAMERA_DEPTH="${BAG_RECORD_CAMERA_DEPTH:-false}"
BAG_ROOT="${BAG_ROOT:-$ROOT_DIR/bags}"
OUTPUT=""
STORAGE="${BAG_STORAGE:-auto}"
COMPRESSION="${BAG_COMPRESSION:-none}"
SPLIT_DURATION="${BAG_SPLIT_DURATION:-600}"
MAX_CACHE_SIZE="${BAG_MAX_CACHE_SIZE:-1073741824}"
QOS_FILE="${BAG_QOS_FILE:-$ROOT_DIR/config/real_rosbag_qos_overrides.yaml}"
EXTRA_ARGS=()
PASSTHROUGH=0
POSITIONAL_MODE_SET=0

for arg in "$@"; do
  if [[ "$PASSTHROUGH" -eq 1 ]]; then
    EXTRA_ARGS+=("$arg")
    continue
  fi
  case "$arg" in
    -h|--help)
      usage
      exit 0
      ;;
    --)
      PASSTHROUGH=1
      ;;
    mode:=*)
      MODE="${arg#mode:=}"
      POSITIONAL_MODE_SET=1
      ;;
    camera:=*)
      RECORD_CAMERA="${arg#camera:=}"
      ;;
    camera_depth:=*)
      RECORD_CAMERA_DEPTH="${arg#camera_depth:=}"
      ;;
    car_id:=*)
      CAR_ID="${arg#car_id:=}"
      ;;
    output:=*)
      OUTPUT="${arg#output:=}"
      ;;
    storage:=*)
      STORAGE="${arg#storage:=}"
      ;;
    compression:=*)
      COMPRESSION="${arg#compression:=}"
      ;;
    split_duration:=*)
      SPLIT_DURATION="${arg#split_duration:=}"
      ;;
    max_cache_size:=*)
      MAX_CACHE_SIZE="${arg#max_cache_size:=}"
      ;;
    qos_file:=*)
      QOS_FILE="${arg#qos_file:=}"
      ;;
    raw|camera|core|mapping|planning|debug|full)
      if [[ "$POSITIONAL_MODE_SET" -eq 0 ]]; then
        MODE="$arg"
        POSITIONAL_MODE_SET=1
      else
        EXTRA_ARGS+=("$arg")
      fi
      ;;
    *)
      EXTRA_ARGS+=("$arg")
      ;;
  esac
done

if [[ "$STORAGE" == "auto" ]]; then
  if ros2 bag record -h 2>/dev/null | grep -q "mcap"; then
    STORAGE="mcap"
  else
    STORAGE="sqlite3"
  fi
fi

case "$MODE" in
  raw|camera|core|mapping|planning|debug|full) ;;
  *)
    echo "[record_real_bag] ERROR: unknown mode '$MODE'" >&2
    usage >&2
    exit 2
    ;;
esac

if [[ "$MODE" == "camera" ]]; then
  RECORD_CAMERA=true
fi

mkdir -p "$BAG_ROOT"
if [[ -z "$OUTPUT" ]]; then
  OUTPUT="$BAG_ROOT/real_${MODE}_$(date +%Y%m%d_%H%M%S)"
fi

COMMON_TOPICS=(
  /tf
  /tf_static
  /rosout
)

RAW_TOPICS=(
  /livox/lidar
  /imu_raw
)

FASTLIO_CORE_TOPICS=(
  /Odometry
  /aft_mapped_to_init
  /path
  /cloud_registered
)

FASTLIO_LOCAL_TOPICS=(
  /cloud_local_map
  /LIVO2/imu_propagate
  /rgb_img
  /cloud_visual_sub_map_before
)

FASTLIO_HEAVY_TOPICS=(
  /Laser_map
  /cloud_effected
  /cloud_registered_body
)

TERRAIN_CORE_TOPICS=(
  /car_${CAR_ID}_terrain_analyzer/fused_map
  /car_${CAR_ID}_terrain_analyzer/sdf_map
)

TERRAIN_DEBUG_TOPICS=(
  /car_${CAR_ID}_terrain_analyzer/elevation_cloud
  /car_${CAR_ID}_terrain_analyzer/sdf_cloud
  /car_${CAR_ID}_terrain_analyzer/risk_cloud
  /car_${CAR_ID}_terrain_analyzer/obstacle_cloud
  /car_${CAR_ID}_terrain_analyzer/overhead_cloud
  /car_${CAR_ID}_terrain_analyzer/elevation_grid_cloud
  /car_${CAR_ID}_terrain_analyzer/sdf_grid_cloud
  /car_${CAR_ID}_terrain_analyzer/risk_grid_cloud
)

PLANNER_TOPICS=(
  /goal_pose
  /move_base_simple/goal
  /planner/trajectory
  /planner/dptrajectory
  /visualization/refinedTraj_kinoastar
)

MPC_TOPICS=(
  /cmd_vel
  /cmd_vel_stamp
  /desired_state
  /track_err
)

MPC_DEBUG_TOPICS=(
  /following_path
  /predict_path
  /reference_path
  /vehicle_history
)

CHASSIS_TOPICS=(
  /odom
  /vehicle_path
  /xtopic_vehicle/device_state_json
)

CHASSIS_LOW_LEVEL_TOPICS=(
  /xtopic_vehicle/ctrl_json
  /xtopic_comm/com_send_xstd
  /xtopic_comm/com_recv_xstd_vehicle
)

VIS_TOPICS=(
  /vehicle_model
)

CAMERA_COLOR_TOPICS=(
  /camera/camera/color/image_raw
  /camera/camera/color/camera_info
)

CAMERA_COLOR_DEBUG_TOPICS=(
  /camera/camera/color/metadata
)

CAMERA_DEPTH_TOPICS=(
  /camera/camera/depth/image_rect_raw
  /camera/camera/depth/camera_info
  /camera/camera/depth/metadata
  /camera/camera/extrinsics/depth_to_color
  /camera/camera/depth/color/points
)

TOPICS=("${COMMON_TOPICS[@]}")
case "$MODE" in
  raw)
    TOPICS+=("${RAW_TOPICS[@]}")
    ;;
  camera)
    TOPICS+=("${CAMERA_COLOR_TOPICS[@]}")
    ;;
  mapping)
    TOPICS+=("${RAW_TOPICS[@]}" "${FASTLIO_CORE_TOPICS[@]}" "${FASTLIO_LOCAL_TOPICS[@]}" "${FASTLIO_HEAVY_TOPICS[@]}")
    ;;
  planning)
    TOPICS+=("${FASTLIO_CORE_TOPICS[@]}" "${TERRAIN_CORE_TOPICS[@]}" "${PLANNER_TOPICS[@]}" "${MPC_TOPICS[@]}" "${CHASSIS_TOPICS[@]}")
    ;;
  core)
    TOPICS+=("${RAW_TOPICS[@]}" "${FASTLIO_CORE_TOPICS[@]}" "${TERRAIN_CORE_TOPICS[@]}" "${PLANNER_TOPICS[@]}" "${MPC_TOPICS[@]}" "${CHASSIS_TOPICS[@]}")
    ;;
  debug)
    TOPICS+=("${RAW_TOPICS[@]}" "${FASTLIO_CORE_TOPICS[@]}" "${FASTLIO_LOCAL_TOPICS[@]}" "${TERRAIN_CORE_TOPICS[@]}" "${TERRAIN_DEBUG_TOPICS[@]}" "${PLANNER_TOPICS[@]}" "${MPC_TOPICS[@]}" "${MPC_DEBUG_TOPICS[@]}" "${CHASSIS_TOPICS[@]}" "${VIS_TOPICS[@]}")
    ;;
  full)
    TOPICS+=("${RAW_TOPICS[@]}" "${FASTLIO_CORE_TOPICS[@]}" "${FASTLIO_LOCAL_TOPICS[@]}" "${FASTLIO_HEAVY_TOPICS[@]}" "${TERRAIN_CORE_TOPICS[@]}" "${TERRAIN_DEBUG_TOPICS[@]}" "${PLANNER_TOPICS[@]}" "${MPC_TOPICS[@]}" "${MPC_DEBUG_TOPICS[@]}" "${CHASSIS_TOPICS[@]}" "${CHASSIS_LOW_LEVEL_TOPICS[@]}" "${VIS_TOPICS[@]}")
    ;;
esac

case "$RECORD_CAMERA" in
  true|1|yes|on)
    if [[ "$MODE" != "camera" ]]; then
      TOPICS+=("${CAMERA_COLOR_TOPICS[@]}")
    fi
    ;;
  false|0|no|off) ;;
  *)
    echo "[record_real_bag] ERROR: camera must be true or false" >&2
    exit 2
    ;;
esac

case "$RECORD_CAMERA_DEPTH" in
  true|1|yes|on)
    RECORD_CAMERA=true
    TOPICS+=("${CAMERA_DEPTH_TOPICS[@]}")
    ;;
  false|0|no|off) ;;
  *)
    echo "[record_real_bag] ERROR: camera_depth must be true or false" >&2
    exit 2
    ;;
esac

if [[ "$MODE" == "debug" || "$MODE" == "full" ]]; then
  case "$RECORD_CAMERA" in
    true|1|yes|on)
      TOPICS+=("${CAMERA_COLOR_DEBUG_TOPICS[@]}")
      ;;
  esac
fi

if [[ ! -f "$QOS_FILE" ]]; then
  echo "[record_real_bag] ERROR: QoS override file not found: $QOS_FILE" >&2
  exit 1
fi

CMD=(
  ros2 bag record
  --include-unpublished-topics
  --qos-profile-overrides-path "$QOS_FILE"
  --storage "$STORAGE"
  --output "$OUTPUT"
  --max-cache-size "$MAX_CACHE_SIZE"
)

if [[ "$SPLIT_DURATION" != "0" && "$SPLIT_DURATION" != "none" ]]; then
  CMD+=(--max-bag-duration "$SPLIT_DURATION")
fi

case "$COMPRESSION" in
  none) ;;
  file|message)
    CMD+=(--compression-mode "$COMPRESSION" --compression-format zstd)
    ;;
  *)
    echo "[record_real_bag] ERROR: compression must be none, file, or message" >&2
    exit 2
    ;;
esac

CMD+=("${EXTRA_ARGS[@]}")
CMD+=("${TOPICS[@]}")

echo "[record_real_bag] mode: $MODE"
echo "[record_real_bag] car_id: $CAR_ID"
echo "[record_real_bag] camera: $RECORD_CAMERA"
echo "[record_real_bag] camera_depth: $RECORD_CAMERA_DEPTH"
echo "[record_real_bag] output: $OUTPUT"
echo "[record_real_bag] storage: $STORAGE"
echo "[record_real_bag] compression: $COMPRESSION"
echo "[record_real_bag] split_duration: $SPLIT_DURATION"
echo "[record_real_bag] qos_file: $QOS_FILE"
echo "[record_real_bag] topics (${#TOPICS[@]}):"
printf '  %s\n' "${TOPICS[@]}"
echo "[record_real_bag] Starting recorder. Press Ctrl-C to stop."

exec "${CMD[@]}"
