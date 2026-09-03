#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /opt/ros/humble/setup.bash
unset PYTHONHOME
unset PYTHONPATH

if [[ -f "$ROOT_DIR/install/setup.bash" ]]; then
  source "$ROOT_DIR/install/setup.bash"
elif [[ -f "$ROOT_DIR/install/local_setup.bash" ]]; then
  source "$ROOT_DIR/install/local_setup.bash"
fi

if [[ -z "${RMW_IMPLEMENTATION:-}" ]]; then
  export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
fi

echo "[hunter_fsgp_bgk] Launching FSGP-BGK/SEB SE2Grid enhancer"
exec ros2 launch fsgp_bgk_se2 fsgp_bgk_se2.launch.py "$@"
