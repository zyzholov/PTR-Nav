#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/scripts/lib/runtime.sh"
source "$ROOT_DIR/scripts/lib/sim_navigation_profile.sh"
hunter_sim_navigation_profile terrain "$@"
set -- "${HUNTER_STAGE_ARGS[@]}"
BACKEND="${HUNTER_TERRAIN_BACKEND:-gpu}"
ARGS=()
HAS_GP_COMPLETION=false
SHOW_ARGS=false
for arg in "$@"; do
  case "$arg" in
    -h|--help)
      cat <<'EOF'
Usage: scripts/sim/start_terrain_analysis.sh [key:=value ...] [--show-args]
  navigation_profile:=neos_joint_sim  Explicit simulation contract-8 preset.
  terrain_backend:=gpu|cpu   Native terrain front-end (GPU with CPU fallback).
  HUNTER_TERRAIN_CPU_THREADS=1  Environment default for BLAS/OpenMP preprocessing; explicit library thread settings take precedence.
  terrain_dynamic_clear_enabled:=true|false  Enable terrain free-ray clearing.
  terrain_dynamic_clear_frames:=3             Consecutive frames required.
  enhance:=true|false        Run BGK/SEB after the front-end (default true).
  fsgp_gpu_backend:=auto|cupy|cpu
  fsgp_processing_executor:=thread_pool|single_ingress  NEOS preset uses one ROS ingress executor and one compute owner.
  fsgp_max_input_source_age_sec:=0.16  GPU sim ingress deadline (0 disables).
  fsgp_max_output_source_age_sec:=0.30 GPU sim result deadline (0 disables).
  fsgp_output_delivery_reserve_sec:=0.015 Reserve transport time AFTER encoding inside the source deadline.
  gp_complete_supported:=true|false  Bounded GP support (sim default true).
  probability_model:=marginal|shared_gp  Shared covariance is explicit simulation qualification only.
  shared_gp_condition_full_residual:=false  Opt-in cell-local Gaussian residual, conditioned only on real observations.
  shared_gp_posterior_radius_m:=3.0  Half-width of conditioned terrain tile (1..5m); separate from missing-ground support limits.
  shared_gp_basis_spacing_m:=1.0  Global kernel basis spacing (.5..2m); changes model capacity, not grid resolution.
  probability_gp_max_distance_m:=0.75  Upper bound on measured support distance, not unrestricted extrapolation.
  probability_use_gp_completion:=false  Inspect completion without navigation use.
  startup_support_mode:=unknown|stable_stance  Fixed sim-only initial assumption; default unknown.
  startup_support_directional:=true  Explicit one-sided initial-plane check for tilted MID360; sim only.
  probability_gp_max_extrapolation_m:=0.75  Optional bounded extrapolation; default 0, shared_gp only.
  rviz:=true|false          Shared RViz for terrain + planning (default true).
  grid_clouds:=true|false   Colored elevation/risk/SDF clouds for RViz.
  vehicle_model:=true|false Show the vehicle marker/model in RViz.
  odom_topic:=/Odometry cloud_topic:=/cloud_registered
  overhead_filter_enabled:=true|false  Measured-floor-relative collision slice (sim default true).
  vehicle_envelope_height_m:=0.65      Entire vehicle/payload above its supporting plane.
  overhead_clearance_margin_m:=0.10    Extra vertical headroom; unknown floor is not clipped.
  obstacle_memory_topic:=none|/Laser_map  Experimental obstacle history (default none).
  obstacle_memory_role:=hard|advisory   Hard occupancy or soft route cost, never ground support.
                                    Accumulated ghosts on slopes are not qualified.
  obstacle_memory_max_age_sec:=1.25 Never use a future, wrong-frame or stale memory cloud.
With enhance:=false, run scripts/sim/optional/start_fsgp_bgk.sh separately.
EOF
      exit 0 ;;
    terrain_backend:=*) BACKEND="${arg#*:=}" ;;
    --show-args|-s) SHOW_ARGS=true ;;
    gp_complete_supported:=*) HAS_GP_COMPLETION=true; ARGS+=("$arg") ;;
    gpu_backend:=*)
      echo "[terrain_analysis] Use terrain_backend:=gpu|cpu or fsgp_gpu_backend:=auto|cupy|cpu." >&2
      exit 2 ;;
    *) ARGS+=("$arg") ;;
  esac
done
case "$BACKEND" in gpu|cpu) ;; *) echo "Invalid terrain_backend: $BACKEND" >&2; exit 2 ;; esac
hunter_configure_terrain_threads
if [[ "$HAS_GP_COMPLETION" == false ]]; then
  GP_COMPLETION=true
  if [[ "$(hunter_arg_value prediction_backend feature_sgp "${ARGS[@]}")" == none ]]; then GP_COMPLETION=false; fi
  ARGS+=("gp_complete_supported:=$GP_COMPLETION")
fi
ARGS+=("overhead_filter_enabled:=$(hunter_arg_value overhead_filter_enabled true "${ARGS[@]}")")
hunter_source_ros sim
if [[ "$SHOW_ARGS" == true ]]; then
  exec ros2 launch fsgp_bgk_se2 terrain_analysis.launch.py "terrain_backend:=$BACKEND" "${ARGS[@]}" --show-args
fi
python3 "$ROOT_DIR/scripts/tools/check_runtime_installation.py"

ODOM_TOPIC="$(hunter_arg_value odom_topic /Odometry "${ARGS[@]}")"
CLOUD_TOPIC="$(hunter_arg_value cloud_topic /cloud_registered "${ARGS[@]}")"
python3 "$ROOT_DIR/scripts/tools/hunter_stack_wait.py" "$ODOM_TOPIC" --type odom --timeout 45
python3 "$ROOT_DIR/scripts/tools/hunter_stack_wait.py" "$CLOUD_TOPIC" --type cloud --timeout 30
MEMORY_TOPIC="$(hunter_arg_value obstacle_memory_topic none "${ARGS[@]}")"
MEMORY_ROLE="$(hunter_arg_value obstacle_memory_role hard "${ARGS[@]}")"
MEMORY_REQUIRED=false
if [[ -n "$MEMORY_TOPIC" && "$MEMORY_TOPIC" != none ]]; then
  [[ "$MEMORY_ROLE" != hard ]] || MEMORY_REQUIRED=true
  python3 "$ROOT_DIR/scripts/tools/hunter_stack_wait.py" "$MEMORY_TOPIC" --type cloud --timeout 30
fi

if [[ "$BACKEND" == gpu ]] && {
  ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1 ||
  [[ ! -x "$(ros2 pkg prefix terrain_analyzer)/lib/terrain_analyzer/terrain_analyzer_gpu" ]];
}; then
  echo "[terrain_analysis] CUDA front-end unavailable; falling back to CPU" >&2
  BACKEND=cpu
fi
INPUT_DEADLINE=0.0
OUTPUT_DEADLINE=0.0
if [[ "$BACKEND" == gpu ]] && [[ "$(hunter_arg_value fsgp_gpu_backend auto "${ARGS[@]}")" != cpu ]]; then
  INPUT_DEADLINE=0.16
  OUTPUT_DEADLINE=0.30
fi
ARGS+=("fsgp_max_input_source_age_sec:=$(hunter_arg_value fsgp_max_input_source_age_sec "$INPUT_DEADLINE" "${ARGS[@]}")")
ARGS+=("fsgp_max_output_source_age_sec:=$(hunter_arg_value fsgp_max_output_source_age_sec "$OUTPUT_DEADLINE" "${ARGS[@]}")")
ARGS+=("obstacle_memory_topic:=$MEMORY_TOPIC")
ARGS+=("obstacle_memory_required:=$(hunter_arg_value obstacle_memory_required "$MEMORY_REQUIRED" "${ARGS[@]}")")
echo "[terrain_analysis] Input: $CLOUD_TOPIC + $ODOM_TOPIC; backend=$BACKEND"
echo "[terrain_analysis] Numeric CPU threads: OMP=$OMP_NUM_THREADS OpenBLAS=$OPENBLAS_NUM_THREADS MKL=$MKL_NUM_THREADS NumExpr=$NUMEXPR_NUM_THREADS"
echo "[terrain_analysis] Obstacle memory: $MEMORY_TOPIC; role=$MEMORY_ROLE; required=$(hunter_arg_value obstacle_memory_required false "${ARGS[@]}"); not GP training ground or calibrated probability"
echo "[terrain_analysis] Height slice: enabled=$(hunter_arg_value overhead_filter_enabled true "${ARGS[@]}"); vehicle=$(hunter_arg_value vehicle_envelope_height_m 0.65 "${ARGS[@]}")m + headroom=$(hunter_arg_value overhead_clearance_margin_m 0.10 "${ARGS[@]}")m; measured local floor required"
echo "[terrain_analysis] BGK/SEB enhancement: $(hunter_arg_value enhance true "${ARGS[@]}")"
echo "[terrain_analysis] Source deadlines: input=$(hunter_arg_value fsgp_max_input_source_age_sec 0 "${ARGS[@]}")s output=$(hunter_arg_value fsgp_max_output_source_age_sec 0 "${ARGS[@]}")s; drops reported on /car_0_fsgp_bgk/runtime"
echo "[terrain_analysis] Shared RViz: $(hunter_arg_value rviz true "${ARGS[@]}") (terrain + planning displays)"
echo "[terrain_analysis] GP completion: $(hunter_arg_value gp_complete_supported false "${ARGS[@]}"); model=$(hunter_arg_value probability_model marginal "${ARGS[@]}") (simulation qualification, not calibrated deployment risk)"
exec ros2 launch fsgp_bgk_se2 terrain_analysis.launch.py "terrain_backend:=$BACKEND" "${ARGS[@]}"
