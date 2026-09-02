# FAST-LIVO2 + D455 Integration

This package adapts a ROS 2 FAST-LIVO2 package named `fast_livo` to this
workspace's existing Hunter SE / terrain mapping topic contract.

It is not the FAST-LIVO2 algorithm package itself. The algorithm package should
provide executable `fastlivo_mapping`.

FAST-LIVO2 configuration now lives in the algorithm package:

- `fast_livo/config/hunter_mid360_d455_sim.yaml`
- `fast_livo/config/camera_d455_sim.yaml`
- `fast_livo/launch/mapping_hunter_mid360_d455_sim.launch.py`

Expected inputs:
- `/livox/lidar`
- `/livox/imu`
- `/camera/camera/color/image_raw`

Adapted outputs:
- `/cloud_registered`
- `/Odometry` remapped from FAST-LIVO2's `/aft_mapped_to_init`
- `map -> camera_init -> aft_mapped` TF for RViz fixed-frame visualization

Simulation launch:

```bash
./run_hunter_gazebo.sh enable_d455:=true
./run_hunter_fastlivo2.sh
ros2 launch plan_manage hunter_gazebo_fastlio_nav.launch.py
```

`hunter_livo2_mapping.launch.py` starts `livo2_chain_check.py` and the dedicated
`rviz/hunter_livo2_mapping.rviz` view by default.

The simulation extrinsics in `config/hunter_mid360_d455_sim.yaml` match the
mounts in `hunter_se_gazebo.xacro`. Replace `extrin_calib.Rcl/Pcl` and the
camera intrinsics before using a real D455.
