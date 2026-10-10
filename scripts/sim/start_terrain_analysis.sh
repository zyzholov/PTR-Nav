#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/scripts/lib/runtime.sh"
BACKEND="${HUNTER_TERRAIN_BACKEND:-gpu}"
ARGS=()
for arg in "$@"; do
  case "$arg" in
    -h|--help)
      cat <<'EOF'
Usage: scripts/sim/start_terrain_analysis.sh [key:=value ...] [--show-args]
  terrain_backend:=gpu|cpu   Native terrain front-end (GPU with CPU fallback).
  terrain_dynamic_clear_enabled:=true|false  Enable terrain free-ray clearing.
  terrain_dynamic_clear_frames:=3             Consecutive frames required.
  enhance:=true|false        Run BGK/SEB after the front-end (default true).
  fsgp_gpu_backend:=auto|cupy|cpu
  rviz:=true|false          Shared RViz for terrain + planning (default true).
  grid_clouds:=true|false   Colored elevation/risk/SDF clouds for RViz.
  vehicle_model:=true|false Show the vehicle marker/model in RViz.
  odom_topic:=/Odometry cloud_topic:=/cloud_registered
With enhance:=false, run scripts/sim/optional/start_fsgp_bgk.sh separately.
EOF
      exit 0 ;;
    terrain_backend:=*) BACKEND="${arg#*:=}" ;;
    gpu_backend:=*)
      echo "[terrain_analysis] Use terrain_backend:=gpu|cpu or fsgp_gpu_backend:=auto|cupy|cpu." >&2
      exit 2 ;;
    *) ARGS+=("$arg") ;;
  esac
done
case "$BACKEND" in gpu|cpu) ;; *) echo "Invalid terrain_backend: $BACKEND" >&2; exit 2 ;; esac
hunter_source_ros sim
if hunter_show_args "${ARGS[@]}"; then
  exec ros2 launch fsgp_bgk_se2 terrain_analysis.launch.py "terrain_backend:=$BACKEND" "${ARGS[@]}"
fi

ODOM_TOPIC="$(hunter_arg_value odom_topic /Odometry "${ARGS[@]}")"
CLOUD_TOPIC="$(hunter_arg_value cloud_topic /cloud_registered "${ARGS[@]}")"
python3 "$ROOT_DIR/scripts/tools/hunter_stack_wait.py" "$ODOM_TOPIC" --type odom --timeout 45
python3 "$ROOT_DIR/scripts/tools/hunter_stack_wait.py" "$CLOUD_TOPIC" --type cloud --timeout 30

if [[ "$BACKEND" == gpu ]] && {
  ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1 ||
  [[ ! -x "$(ros2 pkg prefix terrain_analyzer)/lib/terrain_analyzer/terrain_analyzer_gpu" ]];
}; then
  echo "[terrain_analysis] CUDA front-end unavailable; falling back to CPU" >&2
  BACKEND=cpu
fi
echo "[terrain_analysis] Input: $CLOUD_TOPIC + $ODOM_TOPIC; backend=$BACKEND"
echo "[terrain_analysis] BGK/SEB enhancement: $(hunter_arg_value enhance true "${ARGS[@]}")"
echo "[terrain_analysis] Shared RViz: $(hunter_arg_value rviz true "${ARGS[@]}") (terrain + planning displays)"
exec ros2 launch fsgp_bgk_se2 terrain_analysis.launch.py "terrain_backend:=$BACKEND" "${ARGS[@]}"
