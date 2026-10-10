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

The first implementation uses a CPU/CuPy adaptive BGK-style Gaussian smoother
plus a SEB yaw sweep over the Hunter footprint. `gpu_backend:=auto` selects
CuPy for the normal `104x104x36` local map and retains CPU for tiny maps where
GPU kernel launch overhead dominates. Use `gpu_backend:=cupy` to force GPU
testing.

## Run

Start the terrain front-end and FSGP-BGK/SEB enhancement in one terminal:

```bash
scripts/sim/start_terrain_analysis.sh terrain_backend:=gpu fsgp_gpu_backend:=cupy
```

To inspect the two processes separately, disable enhancement in the main
terrain script and use the optional launcher in another terminal:

```bash
scripts/sim/start_terrain_analysis.sh terrain_backend:=gpu enhance:=false
scripts/sim/optional/start_fsgp_bgk.sh gpu_backend:=cupy
```

The supported terrain script uses
`ros2 launch fsgp_bgk_se2 terrain_analysis.launch.py` internally.

Start planning/control separately:

```bash
scripts/sim/start_planning_control.sh
```

For manual driving, keep both analysis stages running and then:

```bash
scripts/sim/start_keyboard_control.sh
```
