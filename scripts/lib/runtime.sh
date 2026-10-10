#!/usr/bin/env bash
# Shared environment and argument handling for modular ROS launchers.

hunter_source_ros() {
  unset PYTHONHOME PYTHONPATH
  if [[ ! -f /opt/ros/humble/setup.bash || ! -f "$ROOT_DIR/install/setup.bash" ]]; then
    echo "[hunter] ROS Humble or workspace install is missing. Build the workspace first." >&2
    return 2
  fi
  source /opt/ros/humble/setup.bash
  source "$ROOT_DIR/install/setup.bash"
  if [[ "${1:-sim}" == sim ]]; then
    export RMW_IMPLEMENTATION="${HUNTER_ISAAC_RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
  else
    export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
  fi
  export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
  export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
}

hunter_arg_value() {
  local key="$1" value="$2" arg
  shift 2
  for arg in "$@"; do
    [[ "$arg" != "$key":=* ]] || value="${arg#*:=}"
  done
  printf '%s\n' "$value"
}

hunter_show_args() {
  local arg
  for arg in "$@"; do
    case "$arg" in --show-args|-s) return 0 ;; esac
  done
  return 1
}

hunter_require_idle_command_topic() {
  local topic="$1" info count attempt
  for attempt in 1 2 3; do
    if info="$(timeout 8 ros2 topic info --no-daemon --verbose "$topic" 2>&1)"; then
      break
    fi
    if [[ "$info" != *"Unknown topic"* ]]; then
      echo "[hunter] Cannot inspect $topic. Check simulator readiness and ROS_DOMAIN_ID." >&2
      printf '%s\n' "$info" >&2
      return 2
    fi
    sleep 1
  done
  if [[ "$info" == *"Unknown topic"* ]]; then
    echo "[hunter] $topic has no discovered endpoints yet; continuing after odometry/map checks."
    return 0
  fi
  count="$(awk '/^Publisher count:/ {print $3; exit}' <<< "$info")"
  if [[ ! "$count" =~ ^[0-9]+$ ]] || (( count > 0 )); then
    echo "[hunter] Refusing to start another controller on $topic. Existing endpoints:" >&2
    printf '%s\n' "$info" >&2
    return 2
  fi
}
