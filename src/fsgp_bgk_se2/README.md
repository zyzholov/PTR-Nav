# fsgp_bgk_se2

ROS 2 integration package for FSGP-BGK/SEB traversability enhancement.

This package keeps the existing FAST-LIO2 and `terrain_analyzer` chain intact,
then republishes an enhanced `SE2Grid` map with yaw-dependent `risk` layers:

```text
FAST-LIO2
  |-- /cloud_registered, /Odometry -> terrain_analyzer -> /car_0_terrain_analyzer/fused_map --+
  |                                                                                           |
  +-- optional raw cloud/odom ---------------------------------------------------------------+
                                                                                              v
                                                                                     fsgp_bgk_se2_node
                                                                                              |
                                                                                              v
                                                                                  /car_0_fsgp_bgk/fused_map
```

The first implementation uses a GPU-capable BGK-style Gaussian smoother plus a
SEB yaw sweep over the Hunter footprint. It uses CuPy when available and falls
back to NumPy on CPU.

## Run

```bash
ros2 launch fsgp_bgk_se2 fsgp_bgk_se2.launch.py
```

To make `terrain_nav` consume the enhanced map:

```bash
ros2 launch terrain_nav hunter_gazebo_terrain_nav.launch.py \
  fused_map_topic:=/car_0_fsgp_bgk/fused_map
```
