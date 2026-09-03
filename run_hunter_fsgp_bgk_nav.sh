#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /opt/ros/humble/setup.bash
unset PYTHONHOME
unset PYTHONPATH

if [[ -f "$ROOT_DIR/install/setup.bash" ]]; then
  source "$ROOT_DIR/install/setup.bash"
elif [[ -f "$ROOT_DIR/install/local_setup.bash" ]]; then
  source "$ROOT_DIR/install/local_setup.bash"
fi

if [[ -z "${RMW_IMPLEMENTATION:-}" ]]; then
  export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
fi

echo "[hunter_fsgp_bgk_nav] Launching terrain_nav on /car_0_fsgp_bgk/fused_map"
exec ros2 launch terrain_nav hunter_gazebo_terrain_nav.launch.py \
  fused_map_topic:=/car_0_fsgp_bgk/fused_map \
  risk_query_mode:=se2_yaw \
  se2_risk_layer:=risk \
  hard_risk_layer:=baseline_risk \
  "$@"
