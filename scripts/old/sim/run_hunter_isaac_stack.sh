#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HEADLESS=false
NAV=true
RVIZ=false
GRID_CLOUDS=false
DISPLAY=responsive
SIM_EXTRA=()

while (($#)); do
  case "$1" in
    --help|-h)
      printf '%s\n' \
        'Usage: ./run_hunter_isaac_stack.sh [headless:=true|false] [nav:=true|false] [rviz:=true|false] [grid_clouds:=true|false] [display:=responsive|quality] [Isaac scene options...]' \
        'Default: Mars Gale, physical cmd_vel/steer, 10 Hz LiDAR, 200 Hz external IMU, GPU terrain/FSGP and navigation.'
      exit 0
      ;;
    headless:=*) HEADLESS="${1#headless:=}" ;;
    nav:=*) NAV="${1#nav:=}" ;;
    rviz:=*) RVIZ="${1#rviz:=}" ;;
    grid_clouds:=*) GRID_CLOUDS="${1#grid_clouds:=}" ;;
    display:=*) DISPLAY="${1#display:=}" ;;
    mapping:=*|profile:=*|sensor_profile:=*|--visual-only|--no-lidar-enabled|--no-imu-enabled|--fix-robot-base)
      echo "[hunter_stack] ERROR: $1 conflicts with the mapping stack" >&2
      exit 2
      ;;
    --drive-mode|--cmd-vel-angular-mode|--sensor-profile|--cmd-vel-topic|--motion-model|--legacy-imu-topic)
      echo "[hunter_stack] ERROR: $1 is fixed for the sensor/control chain" >&2
      exit 2
      ;;
    --drive-mode=*|--cmd-vel-angular-mode=*|--sensor-profile=*|--cmd-vel-topic=*|--motion-model=*|--legacy-imu-topic=*)
      echo "[hunter_stack] ERROR: $1 is fixed for the sensor/control chain" >&2
      exit 2
      ;;
    *) SIM_EXTRA+=("$1") ;;
  esac
  shift
done

if [[ "$DISPLAY" != responsive && "$DISPLAY" != quality ]]; then
  echo "[hunter_stack] ERROR: display must be responsive or quality" >&2
  exit 2
fi

for value in "$HEADLESS" "$NAV" "$RVIZ" "$GRID_CLOUDS"; do
  if [[ "$value" != true && "$value" != false ]]; then
    echo "[hunter_stack] ERROR: boolean options must be true or false" >&2
    exit 2
  fi
done

export RMW_IMPLEMENTATION="${HUNTER_ISAAC_RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"

LOG_DIR="$(mktemp -d /tmp/hunter-isaac-stack.XXXXXX)"
sim_pid= lio_pid= nav_pid=
group_alive() {
  [[ -n "$1" ]] && kill -0 -- "-$1" 2>/dev/null
}
cleanup() {
  trap - EXIT INT TERM
  for pid in "$nav_pid" "$lio_pid" "$sim_pid"; do
    if group_alive "$pid"; then
      kill -TERM -- "-$pid" 2>/dev/null || true
    fi
  done
  local deadline=$((SECONDS + 12))
  while ((SECONDS < deadline)); do
    local alive=false
    for pid in "$nav_pid" "$lio_pid" "$sim_pid"; do
      if group_alive "$pid"; then alive=true; fi
    done
    if [[ "$alive" == false ]]; then break; fi
    sleep 0.2
  done
  for pid in "$nav_pid" "$lio_pid" "$sim_pid"; do
    if [[ -n "$pid" ]]; then
      if group_alive "$pid"; then kill -KILL -- "-$pid" 2>/dev/null || true; fi
      wait "$pid" 2>/dev/null || true
    fi
  done
  echo "[hunter_stack] Logs: $LOG_DIR"
}
trap cleanup EXIT INT TERM

wait_sample() {
  local topic="$1" timeout_sec="$2" owner_pid="$3"
  if python3 "$ROOT_DIR/scripts/tools/hunter_stack_wait.py" "$topic" --timeout "$timeout_sec"; then
    return 0
  fi
  if ! kill -0 "$owner_pid" 2>/dev/null; then
    echo "[hunter_stack] ERROR: owning process exited while waiting for $topic" >&2
  fi
  local log_file="$LOG_DIR/terrain_nav.log"
  case "$topic" in
    /clock|/livox/*) log_file="$LOG_DIR/isaac.log" ;;
    /Odometry|/Laser_map) log_file="$LOG_DIR/fastlio.log" ;;
  esac
  if [[ -f "$log_file" ]]; then
    echo "[hunter_stack] Recent $(basename "$log_file") output:" >&2
    tail -n 18 "$log_file" >&2
  fi
  return 1
}

echo "[hunter_stack] Starting Isaac Sim; logs: $LOG_DIR"
GUI_RENDER_ARGS=()
if [[ "$HEADLESS" == false ]]; then
  if [[ "$DISPLAY" == responsive ]]; then
    # Keep the full PBR Mars skin and collision model while matching the
    # mapping render cadence to the 10 Hz LiDAR. Rendering at 20 Hz with the
    # derived frameSkip=1 produces partial ~19k-point clouds in this RTX path.
    GUI_RENDER_ARGS=(--camera-width 960 --camera-height 540 --render-hz 10 --render-quality performance)
  else
    GUI_RENDER_ARGS=(--camera-width 1280 --camera-height 720 --render-hz 10 --render-quality balanced)
  fi
  echo "[hunter_stack] GUI display=$DISPLAY; Isaac options after display:= override its render settings"
fi
cpu_governor="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || true)"
if [[ "$cpu_governor" == powersave ]]; then
  echo "[hunter_stack] WARNING: CPU governor is powersave; 200 Hz physics may miss real time"
fi
ps -eo pid,user,pcpu,comm --sort=-pcpu 2>/dev/null | awk 'NR > 1 && $3 + 0 >= 300 && shown++ < 3 {print "[hunter_stack] WARNING: high-CPU process: pid=" $1 " user=" $2 " cpu=" $3 "% name=" $4}'
setsid "$ROOT_DIR/scripts/sim/start_simulation.sh" "headless:=$HEADLESS" mapping:=true \
  --scene-profile mars_gale --drive-mode cmd_vel --cmd-vel-angular-mode steer \
  --obstacle-density sparse --realtime \
  "${GUI_RENDER_ARGS[@]}" "${SIM_EXTRA[@]}" \
  >"$LOG_DIR/isaac.log" 2>&1 &
sim_pid=$!
set +u
source /opt/ros/humble/setup.bash
unset PYTHONHOME PYTHONPATH
source "$ROOT_DIR/install/setup.bash"
set -u
for package in fast_lio terrain_analyzer fsgp_bgk_se2 terrain_nav; do
  if ! ros2 pkg prefix "$package" >/dev/null 2>&1; then
    echo "[hunter_stack] ERROR: missing ROS package $package; build the workspace first" >&2
    exit 1
  fi
done
wait_sample /clock 120 "$sim_pid"
wait_sample /livox/lidar 30 "$sim_pid"
wait_sample /livox/imu 30 "$sim_pid"

echo "[hunter_stack] Starting FAST-LIO"
setsid "$ROOT_DIR/scripts/sim/start_mapping.sh" "rviz:=$RVIZ" >"$LOG_DIR/fastlio.log" 2>&1 &
lio_pid=$!
wait_sample /Odometry 90 "$lio_pid"
wait_sample /Laser_map 60 "$lio_pid"

echo "[hunter_stack] Starting GPU terrain, FSGP-BGK/SEB and terrain_nav (nav=$NAV)"
setsid "$ROOT_DIR/scripts/old/sim/run_hunter_fsgp_bgk_nav.sh" "nav:=$NAV" rviz:=false \
  "grid_clouds:=$GRID_CLOUDS" "vehicle_model:=$RVIZ" \
  >"$LOG_DIR/terrain_nav.log" 2>&1 &
nav_pid=$!
wait_sample /car_0_terrain_analyzer/fused_map 60 "$nav_pid"
wait_sample /car_0_fsgp_bgk/fused_map 60 "$nav_pid"

echo "[hunter_stack] Ready: Isaac -> FAST-LIO -> terrain_analyzer -> FSGP-BGK/SEB -> terrain_nav"
echo "[hunter_stack] Commands: /cmd_vel; goal: /goal_pose; logs: $LOG_DIR"
exited_pid=
if wait -n -p exited_pid "$sim_pid" "$lio_pid" "$nav_pid"; then
  exit_status=0
else
  exit_status=$?
fi
# Bash can clear the -p variable when wait is interrupted by SIGINT.  Keep the
# shutdown path compatible with set -u so Ctrl-C does not report a false error.
exited_pid="${exited_pid:-}"
if [[ "$exited_pid" != "$sim_pid" ]]; then
  echo "[hunter_stack] ERROR: ROS stack process $exited_pid exited unexpectedly" >&2
  exit_status=1
else
  echo "[hunter_stack] Isaac Sim exited; stopping the remaining processes"
fi
exit "$exit_status"
