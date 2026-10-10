#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/scripts/lib/runtime.sh"
for arg in "$@"; do
  case "$arg" in
    -h|--help)
      cat <<'EOF'
Usage: scripts/sim/start_planning_control.sh [key:=value ...] [--show-args]
  odom_topic:=/Odometry
  fused_map_topic:=/car_0_fsgp_bgk/fused_map
  cmd_topic:=/cmd_vel
Starts terrain_nav only. The shared RViz is owned by start_terrain_analysis.sh;
this module intentionally does not open a second RViz. Stop keyboard control
before starting this module.
EOF
      exit 0 ;;
  esac
done
hunter_source_ros sim
if ! hunter_show_args "$@"; then
  ODOM_TOPIC="$(hunter_arg_value odom_topic /Odometry "$@")"
  MAP_TOPIC="$(hunter_arg_value fused_map_topic /car_0_fsgp_bgk/fused_map "$@")"
  CMD_TOPIC="$(hunter_arg_value cmd_topic /cmd_vel "$@")"
  python3 "$ROOT_DIR/scripts/tools/hunter_stack_wait.py" "$ODOM_TOPIC" --type odom --timeout 45
  python3 "$ROOT_DIR/scripts/tools/hunter_stack_wait.py" "$MAP_TOPIC" --type grid --layer risk --timeout 45
  hunter_require_idle_command_topic "$CMD_TOPIC"
  echo "[planning_control] Input: $MAP_TOPIC + $ODOM_TOPIC; output: $CMD_TOPIC"
fi
exec ros2 launch terrain_nav fsgp_bgk_terrain_nav.launch.py "$@"
