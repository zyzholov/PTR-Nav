# FAST-LIVO2 + RealSense D455 Integration

## What Changed

- Hunter Gazebo now has an optional D455-like RGB-D camera, enabled by default.
- FAST-LIVO2 is integrated through a small adapter package:
  `src/fast_livo2_integration`.
- This is not the third-party FAST-LIVO2 algorithm package itself. The actual
  ROS 2 FAST-LIVO2 port is expected to provide package `fast_livo` and
  executable `fastlivo_mapping`.
- The FAST-LIVO2 repository is cloned under `src/FAST-LIVO2-ROS2`, with Hunter
  simulation configs added directly under `fast_livo/config`.
- FAST-LIVO2's native odometry topic `/aft_mapped_to_init` is remapped to the
  existing project topic `/Odometry`.
- The existing terrain analyzer and planner can keep using:
  `/cloud_registered` and `/Odometry`.
- The adapter launch starts a `livo2_chain_check.py` monitor by default. It
  reports whether the sensor inputs, FAST-LIVO2 outputs, terrain outputs and
  required TF transforms are alive.
- The default RViz config is now
  `fast_livo2_integration/rviz/hunter_livo2_mapping.rviz`, with fixed frame
  `map` and displays for odometry, registered cloud, path, D455 color image and
  terrain clouds.

## External FAST-LIVO2 Package

This workspace expects a ROS 2 package named `fast_livo` with executable
`fastlivo_mapping`. A Humble-compatible source set is listed in:

```bash
third_party/fast_livo2_humble.repos
```

Typical import flow:

```bash
cd /home/z/workspace_zyz/seb_clean
vcs import src < third_party/fast_livo2_humble.repos
```

FAST-LIVO2 also needs `livox_ros_driver2` and Sophus. This workspace already
sources the existing Livox overlay from `/home/z/workspace_zyz/driver/ws_livox`
when present. Install Sophus following the FAST-LIVO2 ROS2 port you choose.

## Simulation Run

Terminal 1:

```bash
scripts/old/sim/run_hunter_gazebo.sh enable_d455:=true
```

Terminal 2:

```bash
scripts/sim/optional/start_fastlivo2.sh
```

Terminal 3:

```bash
scripts/old/sim/run_hunter_livo2_terrain_nav.sh
```

The simulated D455 topics are:

```text
/camera/camera/color/image_raw
/camera/camera/color/camera_info
/camera/camera/depth/image_rect_raw
/camera/camera/depth/camera_info
/camera/camera/depth/color/points
```

Expected FAST-LIVO2 output contract after remapping:

```text
/Odometry
/cloud_registered
/path
map -> camera_init -> aft_mapped TF
```

If the monitor logs `required chain incomplete`, check the first topic marked
`missing`, `waiting`, or `stale`.

## Calibration Notes

Simulation extrinsics are in the FAST-LIVO2 package:

```text
src/FAST-LIVO2-ROS2/config/hunter_mid360_d455_sim.yaml
src/FAST-LIVO2-ROS2/config/camera_d455_sim.yaml
```

Real-robot placeholder config is in:

```text
src/fast_livo2_integration/config/neos_mid360_d455_real.template.yaml
```

Do not use the real template for closed-loop runs until `extrin_calib.Rcl` and
`extrin_calib.Pcl` are replaced with measured LiDAR-to-D455 color optical
extrinsics. FAST-Calib is the recommended route from the FAST-LIVO2 authors.
