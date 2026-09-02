# yolo_person_tracker

ROS2 node for demo video based person detection, ID tracking, contour ghost trajectory rendering, and annotated video publishing.

## Run

Build from the workspace root:

```bash
cd /home/z/workspace_zyz/seb_clean
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select yolo_person_tracker
```

Run with the requested conda environment:

```bash
cd /home/z/workspace_zyz/seb_clean
./run_yolo_person_tracker.sh
```

Useful debug run without opening a window:

```bash
./run_yolo_person_tracker.sh show_window:=false loop_video:=false max_frames:=100
```

Realtime defaults use GPU `device:=0`, `half:=true`, asynchronous inference, and full-resolution frames. If the target board is overloaded, lower only the display/inference frame size while keeping the original video timing:

```bash
./run_yolo_person_tracker.sh frame_scale:=0.667
```

The history overlay uses lightweight contour ghosts. Useful tuning:

```bash
./run_yolo_person_tracker.sh ghost_alpha:=0.24 ghost_max_count:=10
```

If you need to override the weight file:

```bash
./run_yolo_person_tracker.sh model_path:=/path/to/best.pt
```

## Published Topics

- `/yolo_person_tracker/annotated_image` (`sensor_msgs/Image`): BGR image with boxes, track IDs, history trail, and predicted path.
- `/yolo_person_tracker/detections` (`std_msgs/String`): JSON payload with bbox, confidence, velocity, and sampled trajectory pixel positions.
- `/yolo_person_tracker/primary_person` (`geometry_msgs/PointStamped`): largest active person center in pixel coordinates, confidence in `z`.
- `/yolo_person_tracker/person_safety_stop` (`std_msgs/Bool`): true when the primary person occupies a large image area.
- `/yolo_person_tracker/status` (`std_msgs/String`): lightweight JSON status.

These are private node topics, so they remain grouped under the `yolo_person_tracker` node name unless remapped.

## Environment Check

The default script activates `yolov8-seg-new` and executes the node with that environment's Python. The YOLO backend requires:

```bash
source /home/z/anaconda3/etc/profile.d/conda.sh
conda activate yolov8-seg-new
python -c "import torch, torchvision, ultralytics; print(torch.__version__)"
```

If this command fails, fix the conda environment first or pass `backend:=hog` only for ROS/node debugging. The HOG backend is not the requested YOLO detector and should not be used for the final demo.
