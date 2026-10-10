#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="/home/z/workspace_zyz/seb_clean"
SRC_DIR="$ROOT_DIR/src"

pkill -9 -f 'keyboard_teleop_ros2.py' >/dev/null 2>&1 || true
pkill -9 -f 'mpc_controller_node' >/dev/null 2>&1 || true
pkill -9 -f 'plan_node' >/dev/null 2>&1 || true
pkill -9 -f 'ros2 topic pub' >/dev/null 2>&1 || true
sleep 0.2

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

PKG_PREFIX="$(ros2 pkg prefix hunter_se_control)"
TELEOP_BIN="$PKG_PREFIX/lib/hunter_se_control/keyboard_teleop_ros2.py"

if [[ ! -x "$TELEOP_BIN" ]]; then
  echo "ERROR: keyboard teleop executable not found: $TELEOP_BIN" >&2
  exit 1
fi

exec "$TELEOP_BIN" \
  --topic /hunter_se/cmd_vel \
  --odom-topic /hunter_se/wheel_odom \
  --max-steer 0.55 \
  --step-steer 0.08 \
  --yaw-hold-timeout 0.0 \
  --no-center-steer-on-speed-step \
  "$@"
