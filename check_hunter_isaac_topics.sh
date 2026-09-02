#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMEOUT_SEC="${1:-45}"

if [[ -f /opt/ros/humble/setup.bash ]]; then
  source /opt/ros/humble/setup.bash
fi
if [[ -f "$ROOT_DIR/install/setup.bash" ]]; then
  source "$ROOT_DIR/install/setup.bash"
fi

set -u

ISAAC_RMW_IMPLEMENTATION="${HUNTER_ISAAC_RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
if [[ -n "${RMW_IMPLEMENTATION:-}" && "$RMW_IMPLEMENTATION" != "$ISAAC_RMW_IMPLEMENTATION" ]]; then
  echo "[hunter_isaac_topics] Overriding RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION -> $ISAAC_RMW_IMPLEMENTATION for Isaac ROS2 bridge compatibility"
fi
export RMW_IMPLEMENTATION="$ISAAC_RMW_IMPLEMENTATION"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"

EXPECTED_TOPICS=(
  "/clock"
  "/livox/lidar"
  "/imu/data_raw"
  "/livox/imu"
  "/camera/camera/color/image_raw"
  "/camera/camera/color/camera_info"
  "/camera/camera/depth/image_rect_raw"
  "/camera/camera/depth/color/points"
)

SAMPLE_CHECKS=(
  "/clock:"
  "/livox/lidar:header"
  "/imu/data_raw:header"
  "/livox/imu:header"
  "/camera/camera/color/image_raw:header"
  "/camera/camera/color/camera_info:header"
  "/camera/camera/depth/image_rect_raw:header"
  "/camera/camera/depth/color/points:header"
)

echo "[hunter_isaac_topics] Waiting up to ${TIMEOUT_SEC}s for Isaac sensor topics and first samples"
echo "[hunter_isaac_topics] ROS env: ROS_DISTRO=${ROS_DISTRO:-unset} RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-unset} ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-unset} ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY:-unset}"

deadline=$((SECONDS + TIMEOUT_SEC))
last_topics=""
missing=("${EXPECTED_TOPICS[@]}")

while (( SECONDS < deadline )); do
  last_topics="$(ros2 topic list --no-daemon 2>/dev/null || true)"
  missing=()
  for topic in "${EXPECTED_TOPICS[@]}"; do
    if ! grep -qx "$topic" <<<"$last_topics"; then
      missing+=("$topic")
    fi
  done

  if (( ${#missing[@]} == 0 )); then
    echo "[hunter_isaac_topics] OK: all expected Isaac sensor topics are visible; checking data ..."
    ros2 topic list --no-daemon -t | sort | grep -E '/clock|/livox|/imu/data_raw|/camera/camera'
    break
  fi

  sleep 1
done

if (( ${#missing[@]} != 0 )); then
  echo "[hunter_isaac_topics] ERROR: missing expected topics after ${TIMEOUT_SEC}s:" >&2
  for topic in "${missing[@]}"; do
    echo "  - $topic" >&2
  done
  echo "[hunter_isaac_topics] Last visible topics from live discovery:" >&2
  printf '%s\n' "$last_topics" | sort >&2
  echo "[hunter_isaac_topics] Make sure Isaac was not launched with --visual-only and that this terminal uses the same RMW_IMPLEMENTATION/ROS_DOMAIN_ID." >&2
  exit 2
fi

sample_timeout=10
failures=0
for spec in "${SAMPLE_CHECKS[@]}"; do
  topic="${spec%%:*}"
  field="${spec#*:}"
  cmd=(ros2 topic echo --no-daemon --once --qos-profile sensor_data --spin-time 2 --no-arr --no-str)
  if [[ -n "$field" ]]; then
    cmd+=(--field "$field")
  fi
  cmd+=("$topic")

  if output="$(timeout "$sample_timeout" "${cmd[@]}" 2>&1)"; then
    echo "[hunter_isaac_topics] DATA OK: $topic"
  else
    failures=$((failures + 1))
    echo "[hunter_isaac_topics] DATA ERROR: no sample from $topic within ${sample_timeout}s" >&2
    sed -n '1,8p' <<<"$output" >&2
  fi
done

if (( failures > 0 )); then
  echo "[hunter_isaac_topics] ERROR: ${failures} topic(s) are visible but not publishing data." >&2
  echo "[hunter_isaac_topics] This usually means Isaac is paused/not playing, launched with --visual-only, or the ROS RMW/Domain does not match." >&2
  exit 3
fi

echo "[hunter_isaac_topics] OK: all expected Isaac topics are publishing data"
