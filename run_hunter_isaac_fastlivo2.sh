#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$ROOT_DIR/src"
RVIZ="true"
EXTRA_ARGS=()

for arg in "$@"; do
  case "$arg" in
    rviz:=*)
      RVIZ="${arg#rviz:=}"
      ;;
    *)
      EXTRA_ARGS+=("$arg")
      ;;
  esac
done

source /opt/ros/humble/setup.bash
unset PYTHONHOME

if [[ -f "/home/z/workspace_zyz/driver/ws_livox/install/setup.bash" ]]; then
  source "/home/z/workspace_zyz/driver/ws_livox/install/setup.bash"
fi

if [[ -f "$ROOT_DIR/install/setup.bash" ]]; then
  source "$ROOT_DIR/install/setup.bash"
elif [[ -f "$ROOT_DIR/install/local_setup.bash" ]]; then
  source "$ROOT_DIR/install/local_setup.bash"
elif [[ -f "$SRC_DIR/install/setup.bash" ]]; then
  source "$SRC_DIR/install/setup.bash"
elif [[ -f "$SRC_DIR/install/local_setup.bash" ]]; then
  source "$SRC_DIR/install/local_setup.bash"
fi

ISAAC_RMW_IMPLEMENTATION="${HUNTER_ISAAC_RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
if [[ -n "${RMW_IMPLEMENTATION:-}" && "$RMW_IMPLEMENTATION" != "$ISAAC_RMW_IMPLEMENTATION" ]]; then
  echo "[hunter_isaac_livo2] Overriding RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION -> $ISAAC_RMW_IMPLEMENTATION for Isaac ROS2 bridge compatibility"
fi
export RMW_IMPLEMENTATION="$ISAAC_RMW_IMPLEMENTATION"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"

if ! ros2 pkg prefix fast_livo >/dev/null 2>&1; then
  echo "[hunter_isaac_livo2] ERROR: ROS 2 package 'fast_livo' was not found." >&2
  exit 1
fi

if ! ros2 pkg prefix isaac_sim_hunter_livo2 >/dev/null 2>&1; then
  echo "[hunter_isaac_livo2] ERROR: package 'isaac_sim_hunter_livo2' was not found. Build the workspace first." >&2
  exit 1
fi

echo "[hunter_isaac_livo2] Waiting for Isaac Sim sensor data ..."
echo "[hunter_isaac_livo2] ROS env: ROS_DISTRO=${ROS_DISTRO:-unset} RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-unset} ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-unset} ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY:-unset}"
if ! "$ROOT_DIR/check_hunter_isaac_topics.sh" 120; then
  echo "[hunter_isaac_livo2] ERROR: timed out waiting for Isaac Sim sensor data." >&2
  echo "  Start Isaac first with: ./run_hunter_isaac_sim.sh headless:=true" >&2
  exit 1
fi

echo "[hunter_isaac_livo2] Launching FAST-LIVO2 on Isaac Sim topics (rviz:=${RVIZ})"
exec ros2 launch isaac_sim_hunter_livo2 hunter_isaac_livo2_mapping.launch.py \
  rviz:="$RVIZ" \
  "${EXTRA_ARGS[@]}"
