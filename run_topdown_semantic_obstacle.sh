#!/usr/bin/env bash
set -e

WS_DIR="/home/z/workspace_zyz/seb_clean"
CONDA_SH="/home/z/anaconda3/etc/profile.d/conda.sh"
CONDA_ENV="yolov8-seg-new"
ULTRALYTICS_ROOT="/home/z/workspace_zyz/yolo_v8_man_detection/ultralytics-main"

MEDIA_ID=""
ROS_PARAM_ARGS=()
PASSTHROUGH_ARGS=()

for arg in "$@"; do
  case "$arg" in
    media:=*)
      MEDIA_ID="${arg#media:=}"
      ;;
    *)
      if [[ "$arg" == *":="* ]]; then
        ROS_PARAM_ARGS+=("-p" "$arg")
      else
        PASSTHROUGH_ARGS+=("$arg")
      fi
      ;;
  esac
done

if [ -n "$MEDIA_ID" ]; then
  case "$MEDIA_ID" in
    1)
      ROS_PARAM_ARGS+=("-p" "video_path:=/home/z/Videos/tw/媒体1.mp4" "-p" "target_class_names:=bicycle" "-p" "enable_bicycle_color_fallback:=true")
      ;;
    2)
      ROS_PARAM_ARGS+=("-p" "video_path:=/home/z/Videos/tw/媒体2.mp4" "-p" "target_class_names:=bicycle" "-p" "enable_bicycle_color_fallback:=true")
      ;;
    3)
      ROS_PARAM_ARGS+=("-p" "video_path:=/home/z/Videos/tw/媒体3.mp4" "-p" "target_class_names:=person" "-p" "enable_bicycle_color_fallback:=false" "-p" "conf_thres:=0.08" "-p" "imgsz:=1280")
      ;;
    4)
      ROS_PARAM_ARGS+=("-p" "video_path:=/home/z/Videos/tw/媒体4.mp4" "-p" "target_class_names:=person" "-p" "enable_bicycle_color_fallback:=false" "-p" "conf_thres:=0.08" "-p" "imgsz:=1280")
      ;;
    *)
      echo "[ERROR] media must be 1, 2, 3, or 4; got: $MEDIA_ID" >&2
      exit 2
      ;;
  esac
fi

ROS_PARAM_ARGS+=("-p" "draw_semantic_path:=false" "-p" "draw_track_history:=false")

if [ -f "$CONDA_SH" ]; then
  # shellcheck source=/home/z/anaconda3/etc/profile.d/conda.sh
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
else
  echo "[WARN] Conda setup script not found: $CONDA_SH" >&2
fi

export PYTHONPATH="$WS_DIR/src/yolo_person_tracker:$ULTRALYTICS_ROOT:${PYTHONPATH:-}"

if [ -n "${CONDA_PREFIX:-}" ]; then
  NVIDIA_LIB_ROOT="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia"
  for subdir in cuda_cupti nccl cublas cusparse cudnn cuda_runtime; do
    if [ -d "$NVIDIA_LIB_ROOT/$subdir/lib" ]; then
      export LD_LIBRARY_PATH="$NVIDIA_LIB_ROOT/$subdir/lib:${LD_LIBRARY_PATH:-}"
    fi
  done
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi

source /opt/ros/humble/setup.bash
if [ -f "$WS_DIR/install/setup.bash" ]; then
  source "$WS_DIR/install/setup.bash"
fi

cd "$WS_DIR"
exec python -m yolo_person_tracker.topdown_obstacle_node \
  --ros-args "${ROS_PARAM_ARGS[@]}" "${PASSTHROUGH_ARGS[@]}"
