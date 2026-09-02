#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /opt/ros/humble/setup.bash
unset PYTHONHOME
unset PYTHONPATH

if [[ -n "${LIVOX_ROS2_SETUP:-}" && -f "$LIVOX_ROS2_SETUP" ]]; then
  source "$LIVOX_ROS2_SETUP"
elif [[ -f "/home/z/workspace_zyz/driver/ws_livox/install/setup.bash" ]]; then
  source "/home/z/workspace_zyz/driver/ws_livox/install/setup.bash"
fi

if [[ -f "$ROOT_DIR/install/setup.bash" ]]; then
  source "$ROOT_DIR/install/setup.bash"
elif [[ -f "$ROOT_DIR/install/local_setup.bash" ]]; then
  source "$ROOT_DIR/install/local_setup.bash"
else
  echo "[run_real_mapping] ERROR: $ROOT_DIR/install not found. Build from workspace root first:" >&2
  echo "  cd $ROOT_DIR && colcon build --symlink-install" >&2
  exit 1
fi

if [[ -z "${RMW_IMPLEMENTATION:-}" ]]; then
  export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
fi

if [[ "${CLEAN_STALE_REAL_MAPPING:-1}" == "1" ]]; then
  pkill -f 'fastlio_mapping' >/dev/null 2>&1 || true
  pkill -f 'terrain_analyzer' >/dev/null 2>&1 || true
fi

LAUNCH_ARGS=("$@")
SHOW_ARGS=0
for arg in "${LAUNCH_ARGS[@]}"; do
  if [[ "$arg" == "--show-args" ]]; then
    SHOW_ARGS=1
  fi
done

add_default_arg() {
  local key="$1"
  local value="$2"
  local arg
  for arg in "${LAUNCH_ARGS[@]}"; do
    if [[ "$arg" == "$key":=* ]]; then
      return
    fi
  done
  LAUNCH_ARGS+=("$key:=$value")
}

if [[ $SHOW_ARGS -eq 0 ]]; then
  add_default_arg use_sim_time false
  add_default_arg fastlio true
  add_default_arg terrain true
  add_default_arg rviz true
  add_default_arg grid_clouds false
  add_default_arg lid_topic /livox/lidar
  add_default_arg imu_topic /imu/data_raw
fi

echo "[run_real_mapping] Launching NEOS real mapping/localization stack"
echo "[run_real_mapping] Extra launch args: ${LAUNCH_ARGS[*]}"
exec ros2 launch fast_lio neos_mid360_mapping.launch.py "${LAUNCH_ARGS[@]}"
