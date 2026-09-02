#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="/home/z/workspace_zyz/seb_clean"
SRC_DIR="$ROOT_DIR/src"

pkill -9 -x gzserver >/dev/null 2>&1 || true
pkill -9 -x gzclient >/dev/null 2>&1 || true
pkill -9 -x robot_state_publisher >/dev/null 2>&1 || true
pkill -9 -f 'spawn_entity.py' >/dev/null 2>&1 || true
pkill -9 -f 'ros2 launch hunter_se_gazebo' >/dev/null 2>&1 || true
pkill -9 -f 'keyboard_teleop_ros2.py' >/dev/null 2>&1 || true
pkill -9 -f 'ros2 topic pub' >/dev/null 2>&1 || true
pkill -9 -f 'mpc_controller_node' >/dev/null 2>&1 || true
pkill -9 -f 'plan_node' >/dev/null 2>&1 || true
sleep 1

source /opt/ros/humble/setup.bash
unset PYTHONHOME
unset PYTHONPATH

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

if [[ -z "${RMW_IMPLEMENTATION:-}" ]]; then
  export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
fi

LAUNCH_ARGS=("$@")
HAS_DRIVE_MODE_ARG=0
HAS_MAP_OR_WORLD_ARG=0
for arg in "${LAUNCH_ARGS[@]}"; do
  if [[ "$arg" == drive_mode:=* ]]; then
    HAS_DRIVE_MODE_ARG=1
  fi
  if [[ "$arg" == map:=* || "$arg" == world:=* ]]; then
    HAS_MAP_OR_WORLD_ARG=1
  fi
done

if [[ $HAS_DRIVE_MODE_ARG -eq 0 ]]; then
  LAUNCH_ARGS+=("drive_mode:=hunter_joint")
fi

if [[ $HAS_MAP_OR_WORLD_ARG -eq 0 ]]; then
  LAUNCH_ARGS+=("map:=mili")
fi

exec ros2 launch hunter_se_gazebo hunter_se_empty_world.launch.py "${LAUNCH_ARGS[@]}"
