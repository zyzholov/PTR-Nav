#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC_DIR="$ROOT_DIR/src"

unset PYTHONHOME
unset PYTHONPATH
source /opt/ros/humble/setup.bash

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
  echo "[hunter_isaac_keyboard] Overriding RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION -> $ISAAC_RMW_IMPLEMENTATION for Isaac ROS2 bridge compatibility"
fi
export RMW_IMPLEMENTATION="$ISAAC_RMW_IMPLEMENTATION"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"

TELEOP_BIN="$ROOT_DIR/src/simulator/fake_ugv/hunter_se/hunter_se_control/scripts/keyboard_teleop_ros2.py"
if [[ ! -x "$TELEOP_BIN" ]]; then
  PKG_PREFIX="$(ros2 pkg prefix hunter_se_control)"
  TELEOP_BIN="$PKG_PREFIX/lib/hunter_se_control/keyboard_teleop_ros2.py"
fi

if [[ ! -x "$TELEOP_BIN" ]]; then
  echo "[hunter_isaac_keyboard] ERROR: keyboard teleop executable not found." >&2
  echo "  Checked source tree and hunter_se_control install prefix." >&2
  exit 1
fi

echo "[hunter_isaac_keyboard] Publishing keyboard Twist commands to /cmd_vel"
echo "[hunter_isaac_keyboard] angular.z is used as Hunter/Gazebo steering angle by default"
echo "[hunter_isaac_keyboard] Default mode: w/s latch speed, a/d latch steering angle, x center, space stop"
echo "[hunter_isaac_keyboard] Uses /isaac/ground_truth/odometry feedback when Isaac is running in cmd_vel physics mode"
echo "[hunter_isaac_keyboard] Default speed feedback gain is ${HUNTER_ISAAC_KEYBOARD_SPEED_KP:-0.0} (set HUNTER_ISAAC_KEYBOARD_SPEED_KP to override)"
echo "[hunter_isaac_keyboard] Isaac scene must be started with: --drive-mode cmd_vel --cmd-vel-angular-mode steer"

SHOW_HELP="false"
for arg in "$@"; do
  case "$arg" in
    -h|--help)
      SHOW_HELP="true"
      ;;
  esac
done

if [[ "$SHOW_HELP" != "true" && "${HUNTER_ISAAC_KEYBOARD_WAIT_FOR_CMD_VEL:-1}" != "0" ]]; then
  WAIT_TIMEOUT="${HUNTER_ISAAC_KEYBOARD_WAIT_TIMEOUT:-30}"
  echo "[hunter_isaac_keyboard] Waiting up to ${WAIT_TIMEOUT}s for an Isaac /cmd_vel subscription ..."
  FOUND_CMD_VEL_SUB="false"
  DEADLINE=$((SECONDS + WAIT_TIMEOUT))
  while (( SECONDS < DEADLINE )); do
    if timeout 2s ros2 topic info --no-daemon --verbose /cmd_vel 2>/dev/null | grep -Eq 'Subscription count: [1-9][0-9]*'; then
      FOUND_CMD_VEL_SUB="true"
      break
    fi
    sleep 1
  done
  if [[ "$FOUND_CMD_VEL_SUB" == "true" ]]; then
    echo "[hunter_isaac_keyboard] Detected Isaac /cmd_vel subscriber."
  else
    echo "[hunter_isaac_keyboard] WARNING: no /cmd_vel subscriber detected yet; starting teleop anyway." >&2
    echo "[hunter_isaac_keyboard]          Check that Isaac was launched with --drive-mode cmd_vel and FastDDS." >&2
  fi
fi

if [[ "$SHOW_HELP" != "true" ]]; then
  CMD_TOPIC_INFO="$(timeout 3s ros2 topic info --no-daemon --verbose /cmd_vel 2>/dev/null || true)"
  CMD_PUBLISHER_COUNT="$(awk '/^Publisher count:/ {print $3; exit}' <<< "$CMD_TOPIC_INFO")"
  CMD_PUBLISHER_COUNT="${CMD_PUBLISHER_COUNT:-0}"
  if (( CMD_PUBLISHER_COUNT > 0 )); then
    echo "[hunter_isaac_keyboard] ERROR: detected ${CMD_PUBLISHER_COUNT} existing /cmd_vel publisher(s)." >&2
    echo "[hunter_isaac_keyboard] Stop autonomous control or launch the enhanced stack with nav:=false." >&2
    echo "[hunter_isaac_keyboard] Set HUNTER_ISAAC_KEYBOARD_ALLOW_MULTIPLE_PUBLISHERS=1 only for deliberate controller tests." >&2
    if [[ "${HUNTER_ISAAC_KEYBOARD_ALLOW_MULTIPLE_PUBLISHERS:-0}" != "1" ]]; then
      exit 2
    fi
    echo "[hunter_isaac_keyboard] WARNING: multiple command publishers explicitly allowed." >&2
  fi
fi

exec "$TELEOP_BIN" \
  --topic /cmd_vel \
  --odom-topic "${HUNTER_ISAAC_KEYBOARD_ODOM_TOPIC:-/isaac/ground_truth/odometry}" \
  --max-linear "${HUNTER_ISAAC_KEYBOARD_MAX_LINEAR:-1.00}" \
  --max-steer "${HUNTER_ISAAC_KEYBOARD_MAX_STEER:-0.50}" \
  --step-linear "${HUNTER_ISAAC_KEYBOARD_STEP_LINEAR:-0.10}" \
  --step-steer "${HUNTER_ISAAC_KEYBOARD_STEP_STEER:-0.08}" \
  --yaw-hold-timeout "${HUNTER_ISAAC_KEYBOARD_YAW_HOLD_TIMEOUT:-0.0}" \
  --no-center-yaw-on-speed-step \
  --speed-kp "${HUNTER_ISAAC_KEYBOARD_SPEED_KP:-0.0}" \
  "$@"
