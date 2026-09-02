#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$ROOT_DIR/src"
LAUNCH_ARGS=("$@")
SHOW_ARGS=0

source /opt/ros/humble/setup.bash
unset PYTHONHOME
unset PYTHONPATH

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
  LIVO2_RVIZ_CFG="$(ros2 pkg prefix fast_livo2_integration 2>/dev/null || true)"
  if [[ -n "$LIVO2_RVIZ_CFG" ]]; then
    LIVO2_RVIZ_CFG="$LIVO2_RVIZ_CFG/share/fast_livo2_integration/rviz/hunter_livo2_mapping.rviz"
  fi
  add_default_arg use_sim_time true
  add_default_arg odom_topic /Odometry
  add_default_arg mpc_odom_topic /Odometry
  add_default_arg cloud_topic /cloud_registered
  add_default_arg planner_backend terrain_nav
  add_default_arg terrain true
  add_default_arg planner true
  add_default_arg mpc false
  add_default_arg rviz true
  if [[ -n "$LIVO2_RVIZ_CFG" && -f "$LIVO2_RVIZ_CFG" ]]; then
    add_default_arg rviz_cfg "$LIVO2_RVIZ_CFG"
  fi
fi

echo "[hunter_livo2_nav] Launching terrain/nav stack on FAST-LIVO2 topics"
echo "[hunter_livo2_nav] Extra launch args: ${LAUNCH_ARGS[*]}"
exec ros2 launch plan_manage hunter_gazebo_fastlio_nav.launch.py "${LAUNCH_ARGS[@]}"
