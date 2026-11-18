#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/scripts/lib/runtime.sh"
source "$ROOT_DIR/scripts/lib/sim_navigation_profile.sh"
hunter_sim_navigation_profile planning "$@"
set -- "${HUNTER_STAGE_ARGS[@]}"
for arg in "$@"; do
  case "$arg" in
    -h|--help)
      cat <<'EOF'
Usage: scripts/sim/start_planning_control.sh [key:=value ...] [--show-args]
  navigation_profile:=neos_joint_sim  Matching explicit model permissions.
  odom_topic:=/Odometry
  fused_map_topic:=/car_0_fsgp_bgk/fused_map
  cmd_topic:=/cmd_vel
  tracking_controller:=auto|hybrid|sampling_mpc
  live_motion_prediction_model:=reachable|gaussian_joint_cuda|gaussian_joint_cpu
  live_motion_gaussian_acceleration_envelope:=auto|true|false  Optional bounded longitudinal mean acceleration.
  live_motion_gaussian_constant_acceleration:=auto|true|false
    Explicit CA position/velocity/acceleration posterior; separate from bounded-mean CV.
  live_motion_gaussian_jerk_density:=auto|number
    CA white-jerk density in m^2/s^5, an uncalibrated model assumption.
    Gaussian joint mode is an explicit uncalibrated simulation qualification model.
  live_motion_sim_observation_calibration:=true|false
    Opt-in measured 5cm-grid cylinder repeatability model; simulation only.
  dynamic_observed_only:=auto|true|false
    Simulation YAML defaults to true: predict currently measured targets only.
    Missing targets do not latch dynamic waiting; measured occupancy still applies.
    false restores retained-conflict handling; auto preserves the supplied YAML.
  max_speed:=auto|number
    Explicit operating speed in m/s; set the simulator --cmd-vel-max-speed to match.
    auto preserves YAML. The NEOS datasheet maximum is a limit, not a qualified speed.
  reverse_max_speed:=auto|number  Existing planner range 0.1..0.8 m/s; auto preserves YAML.
  dynamic_yield_actuator_delay_sec:=auto|number  auto preserves the YAML.
  dynamic_yield_speed_dead_time_sec:=auto|number
  dynamic_yield_steering_dead_time_sec:=auto|number
    Pure input dead times, excluding explicit acceleration/steering slew.
    -1 inherits the common delay; auto preserves YAML. Explicit fits are nominal models.
  live_motion_joint_risk_limit:=auto|number     Also shares the combined terrain/dynamic limits.
  unclassified_encounter_speed_mps:=auto|number  Cautious approach to nearby compact returns; 0 disables.
  live_motion_track_known_slow_centers:=auto|true|false
    Continue measured identities through slowdown; never refresh from missing returns.
  unclassified_encounter_model_release:=auto|true|false
    Exact fresh modeled moving bodies use full risk proofs; unknown compact cues stay capped.
  chance_allow_gp_completion:=true|false  Defaults to use_sim_time; real use remains opt-in.
  chance_allow_startup_prior:=true|false  Explicit sim-only fixed-stance permission.
  chance_allow_joint_model:=true|false  Explicit sim-only shared-GP covariance permission; default false.
  reverse_recovery_enable:=true|false    Automatic paired or bounded dead-end retreat; sim default true.
  live_cloud_topic:=/cloud_registered_visual  Full corrected scan for Isaac collision/motion detection.
                                            Use /cloud_registered for mappers without that output.
  params:=/path/to/config.yaml               Motion profile is explicit in this YAML.
  Default prediction speed domain: 4.5 m/s; a lower value is an explicit model assumption.
  auto preserves YAML selection (default hybrid). sampling_mpc is NOT MPPI.
Starts terrain_nav only. The shared RViz is owned by start_terrain_analysis.sh;
this module intentionally does not open a second RViz. Stop keyboard control
before starting this module.
EOF
      exit 0 ;;
  esac
done
hunter_source_ros sim
if ! hunter_show_args "$@"; then
  ODOM_TOPIC="$(hunter_arg_value odom_topic /Odometry "$@")"
  MAP_TOPIC="$(hunter_arg_value fused_map_topic /car_0_fsgp_bgk/fused_map "$@")"
  CMD_TOPIC="$(hunter_arg_value cmd_topic /cmd_vel "$@")"
  python3 "$ROOT_DIR/scripts/tools/hunter_stack_wait.py" "$ODOM_TOPIC" --type odom --timeout 45
  CHANCE_MODE="$(hunter_arg_value chance_constrained true "$@")"
  MAP_LAYER=risk
  if [[ "$CHANCE_MODE" == "true" ]]; then MAP_LAYER=failure_probability_upper; fi
  python3 "$ROOT_DIR/scripts/tools/hunter_stack_wait.py" "$MAP_TOPIC" --type grid --layer "$MAP_LAYER" --timeout 45
  hunter_require_idle_command_topic "$CMD_TOPIC"
  echo "[planning_control] Input: $MAP_TOPIC ($MAP_LAYER) + $ODOM_TOPIC; output: $CMD_TOPIC"
  echo "[planning_control] Prediction=$(hunter_arg_value live_motion_prediction_model reachable "$@"); Gaussian joint modes are uncalibrated qualification models."
  MOTION_PARAMS="$(hunter_arg_value params "$ROOT_DIR/src/planner/terrain_nav/config/hunter_gazebo.yaml" "$@")"
  python3 - "$MOTION_PARAMS" "$(hunter_arg_value dynamic_yield_actuator_delay_sec auto "$@")" "$(hunter_arg_value live_motion_joint_risk_limit auto "$@")" "$(hunter_arg_value dynamic_observed_only auto "$@")" "$(hunter_arg_value dynamic_yield_speed_dead_time_sec auto "$@")" "$(hunter_arg_value dynamic_yield_steering_dead_time_sec auto "$@")" <<'PY'
import sys,yaml
with open(sys.argv[1]) as f:document=yaml.safe_load(f)
config=document.get('terrain_nav_node',document.get('/**',{})).get('ros__parameters',{})
delay=config.get('dynamic_yield_actuator_delay_sec',.35) if sys.argv[2]=='auto' else sys.argv[2]
risk=config.get('live_motion_joint_risk_limit',.02) if sys.argv[3]=='auto' else sys.argv[3]
observed=config.get('dynamic_observed_only',False) if sys.argv[4]=='auto' else sys.argv[4]=='true'
print('[planning_control] Motion model: prediction speed domain={} m/s; center-candidate actions={}; actuator delay={} s; dynamic sub-budget={}; configuration={}'.format(
    config.get('live_motion_prediction_maximum_speed_mps',4.5),
    config.get('live_motion_center_candidate_actions',False),
    delay,risk,sys.argv[1]),flush=True)
def dead_time(argument,key):
    value=float(config.get(key,-1.) if argument=='auto' else argument)
    return float(delay) if value<0 else value
print('[planning_control] Input dead times: speed={} s; steering={} s; acceleration and steering slew are modeled separately.'.format(
    dead_time(sys.argv[5],'dynamic_yield_speed_dead_time_sec'),
    dead_time(sys.argv[6],'dynamic_yield_steering_dead_time_sec')),flush=True)
print('[planning_control] Dynamic target scope: {}'.format(
    'current measured targets only; existing occupancy checks remain' if observed else 'retain unresolved conflicts across missing observations'),flush=True)
PY
fi
LIVE_CLOUD_TOPIC="$(hunter_arg_value live_cloud_topic /cloud_registered_visual "$@")"
REVERSE_RECOVERY="$(hunter_arg_value reverse_recovery_enable true "$@")"
PLANNING_ARGS=()
LAUNCH_DISPLAY_ARGS=()
for arg in "$@"; do
  case "$arg" in
    live_cloud_topic:=*|reverse_recovery_enable:=*) ;;
    --show-args|-s) LAUNCH_DISPLAY_ARGS=(--show-args) ;;
    *) PLANNING_ARGS+=("$arg") ;;
  esac
done
exec ros2 launch terrain_nav fsgp_bgk_terrain_nav.launch.py "live_cloud_topic:=$LIVE_CLOUD_TOPIC" \
  "reverse_recovery_enable:=$REVERSE_RECOVERY" "${PLANNING_ARGS[@]}" "${LAUNCH_DISPLAY_ARGS[@]}"
