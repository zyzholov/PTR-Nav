#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source /opt/ros/humble/setup.bash
unset PYTHONHOME PYTHONPATH
source "$ROOT_DIR/install/setup.bash"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"

echo "[real_fsgp_bgk] Starting FSGP-BGK/SEB on /car_0_terrain_analyzer/fused_map"
exec ros2 launch fsgp_bgk_se2 fsgp_bgk_se2.launch.py \
  use_sim_time:=false \
  input_fused_map_topic:=/car_0_terrain_analyzer/fused_map \
  output_fused_map_topic:=/car_0_fsgp_bgk/fused_map \
  "$@"
