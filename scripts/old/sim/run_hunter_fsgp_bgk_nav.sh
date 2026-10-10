#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

source /opt/ros/humble/setup.bash
unset PYTHONHOME
unset PYTHONPATH

if [[ -f "$ROOT_DIR/install/setup.bash" ]]; then
  source "$ROOT_DIR/install/setup.bash"
elif [[ -f "$ROOT_DIR/install/local_setup.bash" ]]; then
  source "$ROOT_DIR/install/local_setup.bash"
fi

if [[ "${RMW_IMPLEMENTATION:-}" != "rmw_fastrtps_cpp" ]]; then
  echo "[hunter_fsgp_bgk_nav] Overriding RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-<unset>} -> rmw_fastrtps_cpp for Isaac ROS2 bridge compatibility"
  export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
fi

CMD_TOPIC="${HUNTER_FSGP_BGK_CMD_TOPIC:-/cmd_vel}"
TERRAIN_BACKEND="${HUNTER_TERRAIN_BACKEND:-gpu}"
NAV_ENABLED="true"

for arg in "$@"; do
  case "$arg" in
    terrain_backend:=*)
      TERRAIN_BACKEND="${arg#terrain_backend:=}"
      ;;
    nav:=*)
      NAV_ENABLED="${arg#nav:=}"
      ;;
  esac
done

if [[ "$TERRAIN_BACKEND" == "gpu" ]]; then
  TERRAIN_PREFIX="$(ros2 pkg prefix terrain_analyzer 2>/dev/null || true)"
  GPU_EXECUTABLE="$TERRAIN_PREFIX/lib/terrain_analyzer/terrain_analyzer_gpu"
  if [[ ! -x "$GPU_EXECUTABLE" ]] || ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
    echo "[hunter_fsgp_bgk_nav] CUDA terrain backend unavailable; falling back to CPU" >&2
    TERRAIN_BACKEND="cpu"
  fi
fi

if [[ "$NAV_ENABLED" == "true" ]]; then
  echo "[hunter_fsgp_bgk_nav] Launching ${TERRAIN_BACKEND} terrain analyzer -> FSGP-BGK/SEB -> terrain_nav"
  echo "[hunter_fsgp_bgk_nav] Autonomous control enabled; terrain_nav publishes ${CMD_TOPIC}"
else
  echo "[hunter_fsgp_bgk_nav] Launching ${TERRAIN_BACKEND} terrain analyzer -> FSGP-BGK/SEB (terrain_nav disabled)"
  echo "[hunter_fsgp_bgk_nav] Manual control is available; no node in this launch publishes ${CMD_TOPIC}"
fi
echo "[hunter_fsgp_bgk_nav] Enhanced map: /car_0_fsgp_bgk/fused_map"
exec ros2 launch fsgp_bgk_se2 hunter_fsgp_bgk_nav.launch.py \
  terrain_backend:="${TERRAIN_BACKEND}" \
  cmd_topic:="${CMD_TOPIC}" \
  "$@"
