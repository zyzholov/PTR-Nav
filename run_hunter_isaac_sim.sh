#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_ROOT="${ISAAC_SIM_ROOT:-/home/z/workspace_zyz/Isaac_Sim}"
SCENE_SCRIPT="$ROOT_DIR/src/isaac_sim_hunter_livo2/scripts/hunter_livo2_isaac_scene.py"
HEADLESS="false"
EXTRA_ARGS=()
VISUAL_ONLY="false"

for arg in "$@"; do
  case "$arg" in
    headless:=*)
      HEADLESS="${arg#headless:=}"
      ;;
    mapping:=true|profile:=mapping|sensor_profile:=mapping)
      EXTRA_ARGS+=("--sensor-profile" "mapping")
      ;;
    mapping:=false|profile:=visual|sensor_profile:=visual)
      EXTRA_ARGS+=("--sensor-profile" "visual")
      ;;
    --visual-only)
      VISUAL_ONLY="true"
      EXTRA_ARGS+=("$arg")
      ;;
    *)
      EXTRA_ARGS+=("$arg")
      ;;
  esac
done

if [[ ! -x "$ISAAC_ROOT/python.sh" ]]; then
  echo "[hunter_isaac] ERROR: Isaac Sim python.sh was not found at $ISAAC_ROOT" >&2
  echo "  Set ISAAC_SIM_ROOT=/path/to/Isaac_Sim if needed." >&2
  exit 1
fi

if [[ ! -f "$SCENE_SCRIPT" ]]; then
  echo "[hunter_isaac] ERROR: scene script not found: $SCENE_SCRIPT" >&2
  exit 1
fi

unset PYTHONHOME
unset PYTHONPATH
ISAAC_RMW_IMPLEMENTATION="${HUNTER_ISAAC_RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
if [[ -n "${RMW_IMPLEMENTATION:-}" && "$RMW_IMPLEMENTATION" != "$ISAAC_RMW_IMPLEMENTATION" ]]; then
  echo "[hunter_isaac] Overriding RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION -> $ISAAC_RMW_IMPLEMENTATION for Isaac ROS2 bridge compatibility"
fi
export RMW_IMPLEMENTATION="$ISAAC_RMW_IMPLEMENTATION"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
export PYTHONUNBUFFERED=1

if [[ -f "$ISAAC_ROOT/setup_ros_env.sh" ]]; then
  source "$ISAAC_ROOT/setup_ros_env.sh"
fi
BRIDGE_ROS_DISTRO="${ROS_DISTRO:-humble}"
BRIDGE_ROS_LIB="$ISAAC_ROOT/exts/isaacsim.ros2.bridge/$BRIDGE_ROS_DISTRO/lib"
if [[ -d "$BRIDGE_ROS_LIB" ]]; then
  case ":${LD_LIBRARY_PATH:-}:" in
    *":$BRIDGE_ROS_LIB:"*) ;;
    *) export LD_LIBRARY_PATH="$BRIDGE_ROS_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
  esac
else
  echo "[hunter_isaac] WARNING: Isaac ROS2 bridge library path not found: $BRIDGE_ROS_LIB" >&2
fi

ISAAC_ARGS=()
if [[ "$HEADLESS" == "true" ]]; then
  ISAAC_ARGS+=("--headless")
fi

echo "[hunter_isaac] Launching Isaac Sim scene from $ISAAC_ROOT"
echo "[hunter_isaac] Extra args: ${EXTRA_ARGS[*]}"
echo "[hunter_isaac] ROS env: ROS_DISTRO=${ROS_DISTRO:-unset} RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-unset} ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-unset} ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY:-unset}"
echo "[hunter_isaac] Isaac ROS2 bridge libs: $BRIDGE_ROS_LIB"
if [[ "$VISUAL_ONLY" == "true" ]]; then
  echo "[hunter_isaac] WARNING: --visual-only disables /clock, /livox/lidar, /imu/data_raw, /livox/imu, and D455 ROS topics." >&2
fi
exec "$ISAAC_ROOT/python.sh" "$SCENE_SCRIPT" "${ISAAC_ARGS[@]}" "${EXTRA_ARGS[@]}"
