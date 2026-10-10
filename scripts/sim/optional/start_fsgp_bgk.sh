#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "$ROOT_DIR/scripts/lib/runtime.sh"
for arg in "$@"; do
  case "$arg" in
    -h|--help)
      cat <<'EOF'
Usage: scripts/sim/optional/start_fsgp_bgk.sh [gpu_backend:=auto|cupy|cpu] [launch args]
Start only BGK/SEB after start_terrain_analysis.sh enhance:=false.
Do not run this beside an already enabled enhancer.
EOF
      exit 0 ;;
    terrain_backend:=*|fsgp_gpu_backend:=*)
      echo "[fsgp_bgk] This entry uses gpu_backend:=auto|cupy|cpu." >&2
      exit 2 ;;
  esac
done
hunter_source_ros sim
if ! hunter_show_args "$@"; then
  INPUT_TOPIC="$(hunter_arg_value input_fused_map_topic /car_0_terrain_analyzer/fused_map "$@")"
  python3 "$ROOT_DIR/scripts/tools/hunter_stack_wait.py" "$INPUT_TOPIC" --type grid --layer risk --timeout 45
fi
exec ros2 launch fsgp_bgk_se2 fsgp_bgk_se2.launch.py "$@"
