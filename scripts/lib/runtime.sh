#!/usr/bin/env bash
# Shared environment and argument handling for modular ROS launchers.

hunter_configure_terrain_threads() {
  # Small CPU matrices/preprocessing accompany the GPU workload. Letting
  # every BLAS/OpenMP runtime claim all host cores delays ROS ingress/egress,
  # physics and mapping. Keep explicit user library settings authoritative.
  local terrain_threads="${HUNTER_FSGP_CPU_THREADS:-${HUNTER_TERRAIN_CPU_THREADS:-1}}" variable value
  [[ "$terrain_threads" =~ ^[1-9][0-9]*$ ]] || {
    echo "[hunter] Terrain/FSGP CPU thread budget must be a positive integer" >&2; return 2;
  }
  for variable in OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS NUMEXPR_NUM_THREADS VECLIB_MAXIMUM_THREADS; do
    value="${!variable:-$terrain_threads}"
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
      echo "[hunter] Invalid numeric thread count for $variable" >&2; return 2;
    }
    export "$variable=$value"
  done
}

hunter_configure_sim_transport() {
  [[ "${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}" == rmw_fastrtps_cpp ]] || return 0
  local mode="${HUNTER_ISAAC_DDS_PUBLICATION_MODE:-${RMW_FASTRTPS_PUBLICATION_MODE:-ASYNCHRONOUS}}"
  mode="${mode^^}"
  case "$mode" in
    ASYNCHRONOUS|SYNCHRONOUS|AUTO) export RMW_FASTRTPS_PUBLICATION_MODE="$mode" ;;
    *) echo "[hunter] Invalid Fast DDS publication mode: $mode" >&2; return 2 ;;
  esac
  # Enhanced grids and accumulated clouds exceed the default SHM segment.
  # Respect an existing user XML; builtin disables only this automatic choice.
  case "${HUNTER_ISAAC_DDS_PROFILE:-large_maps}" in
    large_maps)
      if [[ -z "${FASTRTPS_DEFAULT_PROFILES_FILE:-}" ]]; then
        local profile
        profile="$(cd "$(dirname "${BASH_SOURCE[0]}")/../sim/config" && pwd)/fastdds_large_maps.xml"
        [[ -r "$profile" ]] || { echo "[hunter] Missing DDS profile: $profile" >&2; return 2; }
        export FASTRTPS_DEFAULT_PROFILES_FILE="$profile"
      fi
      ;;
    builtin) ;;
    *) echo "[hunter] Invalid HUNTER_ISAAC_DDS_PROFILE (use large_maps or builtin)" >&2; return 2 ;;
  esac
}

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
    hunter_configure_sim_transport
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
