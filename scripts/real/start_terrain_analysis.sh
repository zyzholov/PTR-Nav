#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source /opt/ros/humble/setup.bash
unset PYTHONHOME PYTHONPATH

if [[ -n "${LIVOX_ROS2_SETUP:-}" && -f "$LIVOX_ROS2_SETUP" ]]; then
  source "$LIVOX_ROS2_SETUP"
elif [[ -f "/home/z/workspace_zyz/driver/ws_livox/install/setup.bash" ]]; then
  source "/home/z/workspace_zyz/driver/ws_livox/install/setup.bash"
fi
source "$ROOT_DIR/install/setup.bash"

export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
ARGS=("use_sim_time:=false" "fastlio:=false" "terrain:=true" "rviz:=false" "grid_clouds:=false")
ARGS+=("$@")

echo "[real_terrain_analysis] Starting terrain analyzer only from the NEOS real-sensor launch"
exec ros2 launch fast_lio neos_mid360_mapping.launch.py "${ARGS[@]}"
