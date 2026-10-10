#!/usr/bin/env bash
set -e

WS_DIR="/home/z/workspace_zyz/seb_clean"
CONDA_SH="/home/z/anaconda3/etc/profile.d/conda.sh"
CONDA_ENV="yolov8-seg-new"
ULTRALYTICS_ROOT="/home/z/workspace_zyz/yolo_v8_man_detection/ultralytics-main"

if [ -f "$CONDA_SH" ]; then
  # shellcheck source=/home/z/anaconda3/etc/profile.d/conda.sh
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
else
  echo "[WARN] Conda setup script not found: $CONDA_SH" >&2
fi

# Keep the local ROS package and YOLO source visible even when they are not installed
# into the selected conda environment.
export PYTHONPATH="$WS_DIR/src/yolo_person_tracker:$ULTRALYTICS_ROOT:${PYTHONPATH:-}"

# Prefer Python wheel CUDA libraries from the selected conda env when they exist.
if [ -n "${CONDA_PREFIX:-}" ]; then
  NVIDIA_LIB_ROOT="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia"
  for subdir in cuda_cupti nccl cublas cusparse cudnn cuda_runtime; do
    if [ -d "$NVIDIA_LIB_ROOT/$subdir/lib" ]; then
      export LD_LIBRARY_PATH="$NVIDIA_LIB_ROOT/$subdir/lib:${LD_LIBRARY_PATH:-}"
    fi
  done
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi

# Source ROS after conda so rclpy and message packages remain discoverable.
source /opt/ros/humble/setup.bash
if [ -f "$WS_DIR/install/setup.bash" ]; then
  source "$WS_DIR/install/setup.bash"
fi

cd "$WS_DIR"

ROS_PARAM_ARGS=()
PASSTHROUGH_ARGS=()
for arg in "$@"; do
  if [[ "$arg" == *":="* ]]; then
    ROS_PARAM_ARGS+=("-p" "$arg")
  else
    PASSTHROUGH_ARGS+=("$arg")
  fi
done

exec python -m yolo_person_tracker.person_tracker_node \
  --ros-args "${ROS_PARAM_ARGS[@]}" "${PASSTHROUGH_ARGS[@]}"
