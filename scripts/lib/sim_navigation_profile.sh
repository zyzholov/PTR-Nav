#!/usr/bin/env bash
# Explicit simulation presets. Arguments supplied by the caller come LAST.
# No environment latch and no dependency on a previous module's shell.
hunter_sim_navigation_profile() {
  local stage="$1" profile=none arg serialized
  shift
  HUNTER_STAGE_ARGS=()
  for arg in "$@"; do
    case "$arg" in
      navigation_profile:=*) profile="${arg#*:=}" ;;
      *) HUNTER_STAGE_ARGS+=("$arg") ;;
    esac
  done
  case "$stage" in simulation|mapping|terrain|planning) ;; *) return 2 ;; esac
  case "$profile" in
    none) return 0 ;;
    neos_joint_sim) ;;
    *) echo "[hunter] Unknown simulation navigation profile: $profile" >&2; return 2 ;;
  esac
  serialized="$(python3 - "$ROOT_DIR/scripts/sim/config/$profile.json" "$stage" <<'PY'
import json,sys
with open(sys.argv[1]) as f:profile=json.load(f)
values=profile['stages'][sys.argv[2]]
if not values or any(not isinstance(v,str) or '\n' in v or '\r' in v or '\0' in v for v in values):
    raise ValueError('invalid profile argument array')
print('\n'.join(values))
PY
  )" || return 2
  local defaults=()
  mapfile -t defaults <<< "$serialized"
  HUNTER_STAGE_ARGS=("${defaults[@]}" "${HUNTER_STAGE_ARGS[@]}")
  echo "[hunter] $profile / $stage: simulation joint Gaussian model; fixed initial stance and independent 5mm cell nugget are explicit assumptions, not calibrated real-vehicle risk." >&2
}
