#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /opt/ros/humble/setup.bash
unset PYTHONHOME
unset PYTHONPATH

HEXMAN_SDK_SETUP="${HEXMAN_SDK_SETUP:-/home/z/workspace_zyz/driver/hexman/sdk_neos_ws/install/setup.bash}"
if [[ -f "$HEXMAN_SDK_SETUP" ]]; then
  source "$HEXMAN_SDK_SETUP"
else
  echo "[run_real_nav] ERROR: hexman NEOS SDK setup not found: $HEXMAN_SDK_SETUP" >&2
  exit 1
fi

if [[ -f "$ROOT_DIR/install/setup.bash" ]]; then
  source "$ROOT_DIR/install/setup.bash"
elif [[ -f "$ROOT_DIR/install/local_setup.bash" ]]; then
  source "$ROOT_DIR/install/local_setup.bash"
else
  echo "[run_real_nav] ERROR: $ROOT_DIR/install not found. Build from workspace root first:" >&2
  echo "  cd $ROOT_DIR && colcon build --symlink-install" >&2
  exit 1
fi

if [[ -z "${RMW_IMPLEMENTATION:-}" ]]; then
  export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
fi

if [[ "${CLEAN_STALE_REAL_NAV:-1}" == "1" ]]; then
  pkill -f 'terrain_nav_node' >/dev/null 2>&1 || true
  pkill -f 'plan_node' >/dev/null 2>&1 || true
  pkill -f 'mpc_controller_node' >/dev/null 2>&1 || true
  pkill -f 'xnode_vehicle' >/dev/null 2>&1 || true
  pkill -f 'xnode_comm' >/dev/null 2>&1 || true
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
  add_default_arg terrain false
  add_default_arg planner true
  add_default_arg planner_backend terrain_nav
  add_default_arg mpc false
  add_default_arg vehicle_model true
  add_default_arg rviz true
  add_default_arg grid_clouds false
  add_default_arg vehicle_driver hexman
  add_default_arg odom_topic /Odometry
  add_default_arg mpc_odom_topic /Odometry
  add_default_arg cmd_vel_topic /cmd_vel
  add_default_arg chassis_odom_topic /odom
fi

echo "[run_real_nav] Launching NEOS real planning/control stack"
echo "[run_real_nav] Extra launch args: ${LAUNCH_ARGS[*]}"
exec ros2 launch plan_manage neos_planning_control.launch.py "${LAUNCH_ARGS[@]}"
