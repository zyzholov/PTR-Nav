#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
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
unset PYTHONPATH

if [[ -f "/home/z/workspace_zyz/driver/ws_livox/install/setup.bash" ]]; then
  source "/home/z/workspace_zyz/driver/ws_livox/install/setup.bash"
fi

if [[ -n "${FAST_LIVO2_SETUP:-}" && -f "$FAST_LIVO2_SETUP" ]]; then
  source "$FAST_LIVO2_SETUP"
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

if [[ -z "${RMW_IMPLEMENTATION:-}" ]]; then
  export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
fi

if ! ros2 pkg prefix fast_livo >/dev/null 2>&1; then
  echo "[hunter_fastlivo2] ERROR: ROS 2 package 'fast_livo' was not found." >&2
  echo "  Import/build a Humble FAST-LIVO2 port, or source it with FAST_LIVO2_SETUP=/path/to/install/setup.bash" >&2
  echo "  Suggested sources are listed in third_party/fast_livo2_humble.repos" >&2
  exit 1
fi

echo "[hunter_fastlivo2] Waiting for /clock, /livox/lidar, /livox/imu and D455 color image ..."
ready=0
for _ in $(seq 1 75); do
  topics="$(ros2 topic list --no-daemon 2>/dev/null || true)"
  if grep -q '^/clock$' <<< "$topics" && \
     grep -q '^/livox/lidar$' <<< "$topics" && \
     grep -q '^/livox/imu$' <<< "$topics" && \
     grep -q '^/camera/camera/color/image_raw$' <<< "$topics"; then
    ready=1
    break
  fi
  sleep 1
done

if [[ $ready -ne 1 ]]; then
  echo "[hunter_fastlivo2] ERROR: timed out waiting for Gazebo LiDAR/IMU/D455 topics." >&2
  echo "  Start legacy Gazebo with: scripts/old/sim/run_hunter_gazebo.sh enable_d455:=true" >&2
  exit 1
fi

for spec in "/clock:" "/livox/lidar:header" "/livox/imu:header" "/camera/camera/color/image_raw:header"; do
  topic="${spec%%:*}"
  field="${spec#*:}"
  cmd=(ros2 topic echo --no-daemon --once --qos-profile sensor_data --spin-time 2 --no-arr --no-str)
  if [[ -n "$field" ]]; then
    cmd+=(--field "$field")
  fi
  cmd+=("$topic")
  if ! output="$(timeout 10 "${cmd[@]}" 2>&1)"; then
    echo "[hunter_fastlivo2] ERROR: topic $topic is visible but no data was received." >&2
    sed -n '1,8p' <<<"$output" >&2
    exit 1
  fi
done

echo "[hunter_fastlivo2] Launching FAST-LIVO2 (use_sim_time:=true, rviz:=${RVIZ})"
exec ros2 launch fast_livo2_integration hunter_livo2_mapping.launch.py \
  use_sim_time:=true \
  rviz:="$RVIZ" \
  "${EXTRA_ARGS[@]}"
