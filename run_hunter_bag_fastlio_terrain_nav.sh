#!/usr/bin/env bash
set -e

WS_DIR="/home/z/workspace_zyz/seb_clean"
DEFAULT_RAW_BAG="/home/z/workspace_zyz/bags/rosbag2_2026_07_07-19_37_13"
DEFAULT_REFERENCE_BAG="/home/z/workspace_zyz/bags/rosbag2_2026_07_07-21_53_43"

RAW_BAG="$DEFAULT_RAW_BAG"
REFERENCE_BAG="$DEFAULT_REFERENCE_BAG"
PLAY_REFERENCE="true"
EXTRA_ARGS=()

for arg in "$@"; do
  case "$arg" in
    raw_bag:=*)
      RAW_BAG="${arg#raw_bag:=}"
      ;;
    reference_bag:=*)
      REFERENCE_BAG="${arg#reference_bag:=}"
      PLAY_REFERENCE="true"
      ;;
    *)
      EXTRA_ARGS+=("$arg")
      ;;
  esac
done

source /opt/ros/humble/setup.bash
if [ -f "$WS_DIR/install/setup.bash" ]; then
  source "$WS_DIR/install/setup.bash"
fi

LAUNCH_ARGS=(
  "raw_bag_path:=$RAW_BAG"
  "play_reference_bag:=$PLAY_REFERENCE"
)

if [ -n "$REFERENCE_BAG" ]; then
  LAUNCH_ARGS+=("reference_bag_path:=$REFERENCE_BAG")
fi

cd "$WS_DIR"
exec ros2 launch plan_manage hunter_bag_fastlio_terrain_nav.launch.py \
  "${LAUNCH_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
