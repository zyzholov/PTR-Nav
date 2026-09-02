# Isaac Sim FAST-LIVO2 Backend

This package adds an Isaac Sim backend for the Hunter FAST-LIVO2 mapping chain.
It is intentionally separate from the Gazebo backend.

## Status

- Isaac Sim root tested at `/home/z/workspace_zyz/Isaac_Sim`.
- Version tested locally: `5.1.0-rc.19`.
- ROS 2 bridge uses Humble/FastDDS.
- The scene imports the project Hunter SE model from
  `src/simulator/fake_ugv/hunter_se/hunter_se_description`, including the STL
  body/wheel meshes and articulation joints.
- The Isaac sensor frames use a compact stacked mount:
  - Livox MID360 frame `livox`, mounted on `base_link` at
    `xyz="0.18 0.0 0.250"` with a 25 degree pitch.
  - LiDAR frame local offset under `livox_base` is `xyz="-0.012 0.0 0.047"`.
  - IMU frame `imu_link`, mounted below the LiDAR at `xyz="0.18 0.0 0.205"`.
  - D455 camera link, mounted below the LiDAR at `xyz="0.225 0.0 0.205"` with the optical center
    0.02 m forward of the link.
- Sensor visual detail is separate from ROS sensor publishing. `cad` mode uses
  the local Livox MID360 DAE mesh, and D455/IMU use lightweight Isaac geometry
  with realistic proportions. The MID360 body visual is yaw-corrected by
  default without changing the ROS LiDAR frame.
- The default `rich` world is a 150 m project scene: generated rolling terrain,
  road network, textured wall/road materials, existing project DAE landmarks
  from `src/simulator/fake_ugv/maps/mili_tech`, tree groups, and a roadside
  visual marker. It is no longer the small 18 m debug plane.
- `--scene-profile light` keeps a small debug scene. `--scene-profile rubicon`
  loads the high-detail Rubicon terrain asset via the Isaac asset converter.
- `--scene-profile mars_gale` and `--scene-profile offroad` load the local
  Mars Gale crater mesh from `src/simulator/fake_ugv/maps/Mars_Gale`. The Hunter
  starts directly on the sampled terrain by default, with a controlled obstacle
  field of upright barrier posts, standing trunks, outer posts, and procedural Mars rocks for
  camera/LiDAR initialization.
  The source STL spans about 571 m x 422 m with roughly 93 m of relief.
- `--scene-profile offroad_nav` loads the community Offroad-Nav forest/rock USD
  terrain from `/home/z/workspace_zyz/third_party/Offroad-Nav/assets`. GUI auto
  uses `medium.usd`; headless auto uses `hard.usd`.
  The scene references only `/map/GeneratedForest`, so the original Offroad-Nav
  Barakuda robot, ROS1 graph, and old LiDAR profiles are not imported into this
  ROS2/Humble simulation. A local dense obstacle field is added on top for
  repeatable SLAM/planning tests.
- The scene publishes:
  - `/clock`
  - `/livox/lidar` as `sensor_msgs/msg/PointCloud2`
  - `/imu/data_raw` as the primary external `sensor_msgs/msg/Imu`
  - `/livox/imu` as a compatibility alias for older launch/config files
  - `/camera/camera/color/image_raw`
  - `/camera/camera/color/camera_info`
  - `/camera/camera/depth/image_rect_raw`
  - `/camera/camera/depth/color/points`
  - `/isaac/ground_truth/odometry` when `--drive-mode cmd_vel` is enabled

For the current FAST-LIO-only Isaac test chain, use `run_hunter_fastlio.sh`.
It auto-detects whether `/livox/lidar` is Isaac `sensor_msgs/msg/PointCloud2`
or Gazebo/Livox `CustomMsg`. Isaac PointCloud2 selects
`src/FAST_LIO/config/isaac_hunter_mid360_pointcloud.yaml`; Gazebo CustomMsg
selects `src/FAST_LIO/config/gazebo_hunter_mid360.yaml`. FAST-LIO owns
`/Odometry`; Isaac's measured/ground-truth vehicle odometry is kept on
`/isaac/ground_truth/odometry` so RViz and control feedback do not mix two
different odometry sources.

The older FAST-LIVO2 chain and D455 topics are kept for later camera-fusion
work, but they are not the default mapping path while calibration/sync is being
kept simple.

## Run

Terminal 1:

```bash
./run_hunter_isaac_sim.sh headless:=true --scene-profile mars_gale --mars-visual-detail fast --sensor-visual-detail procedural --drive-mode circle --realtime
```

Terminal 2:

```bash
./run_hunter_fastlio.sh
```

Current FAST-LIO-only mapping path:

```bash
./run_hunter_isaac_sim.sh headless:=true mapping:=true --scene-profile mars_gale --drive-mode cmd_vel --obstacle-density sparse --render-hz 10 --realtime
./run_hunter_fastlio.sh
```

Mapping while watching the rendered Isaac scene:

```bash
./run_hunter_isaac_sim.sh mapping:=true --scene-profile mars_gale --drive-mode cmd_vel --cmd-vel-angular-mode steer --mars-visual-detail full --mars-surface-skin full --sensor-visual-detail cad --viewer-camera close --obstacle-density sparse --camera-width 1280 --camera-height 720 --render-hz 10 --realtime
./run_hunter_fastlio.sh
./run_hunter_isaac_keyboard.sh
./check_hunter_slam_drift.sh 20
```

Higher-fidelity LiDAR diagnostics:

```bash
# Low-load debug mode. This keeps 10 Hz but is visibly sparse in RViz.
./run_hunter_isaac_sim.sh headless:=true mapping:=true --scene-profile mars_gale --drive-mode stationary --lidar-quality realtime --obstacle-density sparse --render-hz 10 --realtime

# 20k-ray CSV solidState MID-360 approximation generated from the Gazebo
# livox_laser_simulation_ros2 mid360.csv table. It matches the scan table more
# closely than the rotary surrogate, but was slower than 10 Hz on the test GPU.
./run_hunter_isaac_sim.sh headless:=true mapping:=true --scene-profile mars_gale --drive-mode stationary --lidar-quality mid360_csv --realtime

# Complete Isaac RTX scan buffers. Denser per frame, but much slower on this GPU.
./run_hunter_isaac_sim.sh headless:=true mapping:=true --scene-profile mars_gale --drive-mode stationary --lidar-quality dense --lidar-full-scan --realtime

# Local MID-360 FOV/rate surrogate. It matches 360 x [-7, 52] deg and ~20k
# points/frame, but still uses Isaac RTX rotary scanning rather than Livox's
# true non-repetitive scan pattern.
./run_hunter_isaac_sim.sh headless:=true mapping:=true --scene-profile mars_gale --drive-mode stationary --lidar-quality mid360 --lidar-full-scan --realtime
```

Optional downstream terrain stack:

```bash
./run_hunter_livo2_terrain_nav.sh
```

Useful Isaac arguments:

```bash
# Full default scene: Hunter SE URDF/STL, 150 m rich project map, DAE
# landmarks, tree groups, Livox visual mesh, IMU body, D455 body, and a parked
# vehicle.
./run_hunter_isaac_sim.sh headless:=true --duration 600 --realtime

# High-detail Rubicon terrain profile. This uses the 190 MB Rubicon DAE and can
# take longer on first conversion.
./run_hunter_isaac_sim.sh headless:=true --scene-profile rubicon --duration 600 --realtime

# Local Mars Gale off-road crater profile. This is the preferred large
# off-road test scene when you want to use the existing project map assets.
./run_hunter_isaac_sim.sh headless:=true --scene-profile mars_gale --duration 600 --realtime

# Community Offroad-Nav forest/rock USD profile. This uses Offroad-Nav as a
# visual/geometric asset source only; the ROS2 sensor and FAST-LIVO2 chain stays
# in this package.
./run_hunter_isaac_sim.sh headless:=true --scene-profile offroad_nav --duration 600 --realtime

# Interactive Isaac UI inspection. GUI mode should use the textured Mars skin
# and CAD sensor visuals while checking the vehicle and camera view.
./run_hunter_isaac_sim.sh --scene-profile mars_gale --mars-visual-detail full --mars-surface-skin full --sensor-visual-detail cad --obstacle-density sparse --camera-width 1280 --camera-height 720 --render-hz 10 --realtime

# Fast visual check: no RTX LiDAR, D455, ROS clock, or IMU graph, but keep
# high-quality terrain and sensor visuals.
./run_hunter_isaac_sim.sh --scene-profile mars_gale --visual-only --mars-visual-detail full --mars-surface-skin full --sensor-visual-detail cad --viewer-camera close --obstacle-density sparse --camera-width 1280 --camera-height 720 --render-hz 10 --realtime

# Interactive scene with sensor topics enabled. Do not add --visual-only when
# checking /clock, /livox/lidar, /imu/data_raw, /livox/imu, or D455 topics.
./run_hunter_isaac_sim.sh --scene-profile mars_gale --mars-visual-detail full --mars-surface-skin full --sensor-visual-detail cad --viewer-camera close --obstacle-density sparse --camera-width 1280 --camera-height 720 --render-hz 10 --realtime

# In another terminal, verify that the ROS 2 bridge topics are visible.
./check_hunter_isaac_topics.sh 45

# Same scene from an overview/top-down camera.
./run_hunter_isaac_sim.sh --scene-profile mars_gale --viewer-camera overview --obstacle-density sparse --camera-width 960 --camera-height 540 --render-hz 10 --realtime
./run_hunter_isaac_sim.sh --scene-profile mars_gale --viewer-camera top --obstacle-density sparse --camera-width 960 --camera-height 540 --render-hz 10 --realtime

# Higher-quality screenshots/recording after the scene is confirmed.
./run_hunter_isaac_sim.sh --scene-profile offroad_nav --renderer PathTracing --obstacle-density sparse --camera-width 1280 --camera-height 720 --render-hz 5 --realtime
./run_hunter_isaac_sim.sh --scene-profile mars_gale --mars-visual-detail full --mars-surface-skin full --sensor-visual-detail cad --renderer PathTracing --obstacle-density sparse --camera-width 1280 --camera-height 720 --render-hz 5 --realtime

# Repeatable FAST-LIVO2 mapping demo path. Start FAST-LIVO2 during the initial
# stationary window, before the circle motion begins.
./run_hunter_isaac_sim.sh headless:=true --scene-profile mars_gale --mars-visual-detail fast --sensor-visual-detail procedural --drive-mode circle --duration 600 --realtime

# ROS2 teleop/control mode. Publish geometry_msgs/Twist to /cmd_vel; linear.x
# drives forward/backward and angular.z is the Hunter/Gazebo steering angle by
# default. Isaac drives the physical wheel/steering articulation and applies
# speed/steering limits.
./run_hunter_isaac_sim.sh --scene-profile mars_gale --drive-mode cmd_vel --cmd-vel-angular-mode steer --mars-visual-detail full --mars-surface-skin full --sensor-visual-detail cad --obstacle-density sparse --camera-width 1280 --camera-height 720 --render-hz 10 --realtime
./run_hunter_isaac_keyboard.sh

# Lighter debug scene. This still uses the real Hunter SE model, but skips the
# rich terrain/landmark assets and decorative sensor meshes.
./run_hunter_isaac_sim.sh headless:=true --scene-profile light --map-asset none --no-landmarks --no-sensor-meshes --obstacle-density none --realtime
```

## Notes

- Do not import system ROS 2 `rclpy` inside Isaac Sim Python. Isaac Sim 5.1
  uses Python 3.11 while ROS 2 Humble on Ubuntu 22.04 uses Python 3.10.
  The scene uses Isaac's bundled `rclpy` for `/cmd_vel` and measured
  `/isaac/ground_truth/odometry`. Isaac's native ROS 2 Bridge remains enabled
  for clock, IMU, LiDAR, and camera publishers.
- The Isaac scripts force `RMW_IMPLEMENTATION=rmw_fastrtps_cpp` by default,
  because Isaac's internal Humble bridge is most reliable with FastDDS. They
  print the effective `RMW_IMPLEMENTATION`, `ROS_DOMAIN_ID`, and
  `ROS_LOCALHOST_ONLY` at startup. If you override it with
  `HUNTER_ISAAC_RMW_IMPLEMENTATION`, use the same value for Isaac, FAST-LIVO2,
  `ros2 topic pub`, and `check_hunter_isaac_topics.sh`.
- By default, `--drive-mode stationary` keeps Hunter parked so the vehicle does
  not unexpectedly slide around the scene.
- `--drive-mode circle` enables the repeatable SLAM test trajectory. Its first
  45 simulated seconds keep the Hunter still for IMU initialization; override
  that with `--stationary-duration`.
- `--drive-mode cmd_vel` subscribes to `geometry_msgs/Twist` on `/cmd_vel` by
  default through Isaac's bundled `rclpy`. `angular.z` is treated as the front
  steering angle, matching the existing Gazebo Hunter command semantics. Add
  `--cmd-vel-angular-mode yaw_rate` only when the upstream controller really
  publishes yaw-rate commands. `--cmd-vel-backend bridge` is kept as a
  diagnostic fallback, but the default `rclpy` backend is recommended for
  keyboard/control commands because it has command timeout protection and logs
  when multiple publishers are driving `/cmd_vel`.
- The default `--motion-model physics` drives the imported Hunter articulation
  through wheel velocity targets and steering joints, with a bounded root velocity assist
  to suppress unreal lateral slip, straight-line yaw drift, and parked creeping
  on rough terrain. The default assist mode is `strict`, so manual driving and
  mapping runs track the commanded speed/steering angle closely while the model
  still uses articulation joints and scene collisions. Use
  `--physics-root-assist-mode contact` only when deliberately tuning raw
  wheel/terrain contact; on the imported Hunter URDF it is noticeably slower
  and turns more weakly.
  `--motion-model kinematic` is kept only as a sensor-pipeline debug fallback.
- `./run_hunter_isaac_keyboard.sh` waits for the Isaac `/cmd_vel` subscriber by
  default and publishes pure target speed/steering commands. It reads Isaac
  measured speed from `/isaac/ground_truth/odometry`, leaving FAST-LIO's
  `/Odometry` output separate. Its Isaac defaults are `max_linear=1.00 m/s`,
  `max_steer=0.50 rad`, `step_linear=0.10 m/s`, `step_steer=0.08 rad`,
  latched steering
  (`HUNTER_ISAAC_KEYBOARD_YAW_HOLD_TIMEOUT=0.0`), and no speed-step auto-center.
  Press `x` to center steering and `space` to stop. Set
  `HUNTER_ISAAC_KEYBOARD_SPEED_KP=0.15` only when you intentionally want simple
  odometry feedback while tuning speed tracking.
- Use `--realtime` during live FAST-LIO runs so Isaac does not advance
  simulated sensor time much faster than the mapper can consume it.
- Isaac Sim 5.1 does not ship a Livox MID-360 RTX profile in the local
  `exts/isaacsim.sensors.rtx/data/lidar_configs` tree. The live mapping
  default uses Isaac's Ouster 128-channel 10 Hz profile as a practical dense
  PointCloud2 source. In the local light test scene it measured about
  16k/23k/29k min/mean/max points per frame at 10.3 Hz.
- `--lidar-quality realtime` keeps the older 32-channel 512-res profile for
  low-load debugging. It stays near 10 Hz but is visibly sparse in RViz.
- `--lidar-quality mid360_csv` is a 20k-ray solidState MID-360 approximation
  generated from the Gazebo `mid360.csv` scan table. It loaded successfully,
  but in the local light test scene measured about 3.9k effective return
  points per frame at 8.3 Hz, so it is useful for profile experiments rather
  than as the default live mapper input.
- `--lidar-full-scan` publishes complete RTX scan buffers. Locally this gave
  dense 32k-point Ouster frames, or 20k-point `mid360` surrogate frames, but the
  wall-clock rate dropped to roughly 1.7 Hz in the test scene. Use it to inspect
  LiDAR geometry, not as the default live FAST-LIO mapping mode on this machine.
- The local `LivoxMid360Surrogate10hz` profile follows the public MID-360
  envelope: 10 Hz typical frame rate, 200k points/s first return, horizontal
  360 deg, vertical -7 to 52 deg, 0.1 m blind zone, and about 40 m/70 m range
  for 10%/80% reflectivity. It is still a rotary RTX approximation, not a real
  Livox non-repetitive scanner. For that exact scan pattern, the existing
  Gazebo Livox plugin and `mid360.csv` scan table are closer.
- FAST-LIO's stock `src/FAST_LIO/config/mid360.yaml` is left matching the
  upstream ROS2 branch. The Isaac-specific config lowers downsampling
  (`point_filter_num: 1`, `filter_size_surf: 0.20`, `filter_size_map: 0.25`)
  and enables `/Laser_map` plus `/cloud_local_map`, so sparse Isaac maps should
  be debugged from the simulator point cloud first rather than from FAST-LIO
  downsampling.
- `src/FAST_LIO/config/isaac_hunter_mid360_pointcloud.yaml` enables a
  near-body self-filter box in the generic PointCloud2 preprocessing path. This
  removes close Isaac self-reflections from the Hunter body/sensor mount before
  FAST-LIO sees the scan.
- `./check_hunter_isaac_rates.sh` prints LiDAR/IMU rates, PointCloud2 fields,
  and min/mean/max points per LiDAR frame. If mean points fall below
  `LIDAR_WARN_MIN_POINTS` (default 10000), the script warns that mapping will
  look sparse.
- `./check_hunter_slam_drift.sh 20` compares FAST-LIO `/Odometry` against
  Isaac `/isaac/ground_truth/odometry`, and prints LiDAR/IMU rates, point cloud
  fields, point counts, latest command, and relative XY/z/yaw drift. Run it
  while Isaac and FAST-LIO are both active when a map starts to smear.
- `./run_hunter_fastlio.sh` samples the first Isaac PointCloud2 frame and warns
  when it is below `HUNTER_FASTLIO_WARN_MIN_POINTS` (default 10000). It also
  warns if `/Odometry` already has a publisher before FAST-LIO starts, because
  mixed odometry publishers make RViz/control diagnostics misleading.
- The robot is the project Hunter SE model, not a box-only placeholder. The
  optional circle mode is a deterministic SLAM test path with wheel/steering
  joint targets. The `cmd_vel` mode uses the same physical articulation control
  path without changing the sensor topics or FAST-LIO launch chain. Use the
  circle mode first when validating mapping drift, then switch to
  keyboard/manual control.
- The Offroad-Nav repository is useful as an Isaac Sim terrain asset source, but
  its upstream runtime stack targets Isaac Sim 4.5/ROS Noetic, and its ROS2 port
  is still under development. Keep using the local ROS2 bridge graph and
  FAST-LIO launch files here. Upstream: https://github.com/LARIAD/Offroad-Nav
- The Offroad-Nav `Grass_Instancer` is disabled at load time for Isaac Sim 5.1
  because it can emit invalid prototype warnings and stall the simulation. The
  scene still keeps the Offroad-Nav terrain and adds local upright posts, rocks,
  and trees for nearby mapping features.
- The Mars Gale profile uses the original Gazebo visual/collision split:
  with `--mars-surface-skin none`, `gale_crater_patch1.stl` can be drawn as
  visual terrain and `gale_crater_collision_161.obj` is the physics/raycast
  mesh. With `--mars-surface-skin full`, the textured generated skin is the
  visible terrain and the raw Mars mesh is kept hidden as collision so it does
  not cover the texture as a grey shell. A high-friction sand physics material
  is bound to the collision mesh.
- `--mars-visual-detail auto` expands to `full` in the Isaac GUI and `fast` in
  headless runs. Use `full` when judging scene quality or taking screenshots, and
  use `fast` for live mapping or on runs where the GUI becomes unresponsive.
- `--mars-surface-skin auto` expands to `full` in the Isaac GUI and `none` in
  headless runs. `full` adds a smooth-normal Mars surface that follows the
  collision height grid instead of floating above it, with a PBR
  albedo/normal/roughness texture set for granular sand, rust-red mineral
  streaks, and small gravel. Small pebbles near the route add local texture.
  Keep it off for the fastest repeatable mapping runs.
- `--sensor-visual-detail cad` loads the local Livox MID360 CAD mesh and adds
  only a small front-window/status trim. The previous fake black top ring is not
  used. `procedural` keeps a close lightweight MID360 shape, and
  `none`/`--no-sensor-meshes` hides decorative sensor bodies without changing the
  ROS frames or topics.
- D455 and external IMU visuals sit below the MID360 on a compact bracket. The matching
  Isaac FAST-LIVO2 extrinsics are in
  `src/FAST-LIVO2-ROS2/config/hunter_isaac_d455_sim.yaml`.
- `--livox-visual-yaw-deg 180` is the default visual correction for the MID360
  body. It only rotates the decorative mesh; the actual RTX LiDAR prim, ROS
  frame, and FAST-LIVO2 extrinsics remain unchanged. Set
  `--livox-visual-yaw-deg 0` to compare against the raw CAD orientation.
- `--visual-only` skips RTX LiDAR, D455, ROS clock, and IMU graph creation for
  fast inspection of the vehicle, cameras, and terrain.
- The fast Mars terrain applies generated XY UVs, the original sand texture
  material, and double-sided viewport styling to avoid black terrain when the
  Gazebo collision OBJ has no texture coordinates. Use
  `--mars-surface-skin full` for the PBR visual surface.
- The Mars Hunter spawn height, obstacles, initial cameras, and optional debug
  overlays sample the collision OBJ height grid, so they sit on the same terrain
  surface instead of using fixed z values.
- The default Hunter base spawn height is `--robot-z 0.33`, leaving a small
  tire clearance at spawn so the imported wheel collision cylinders settle onto
  the collision mesh instead of starting clipped into the visible Mars skin.
- The Hunter wheel collision prims get a high-friction tire contact material,
  while chassis collision prims get a separate lower-friction material.
- Extra invisible base-link body, bumper, side, and corner collision boxes are
  added after the URDF import because the Gazebo Hunter URDF only gives the
  visual body a thin lower collision plate. These helper colliders keep upright
  posts and walls from passing through the vehicle body.
- The scene explicitly re-authors the Hunter URDF link masses in Isaac and
  enables collision/contact offsets on static terrain, rocks, posts, and walls.
  If you are deliberately testing raw wheel-contact behavior, use
  `--physics-root-assist-mode contact`; use the default `strict` mode for repeatable sensor
  data when obstacle collision is not the thing being evaluated.
- The artificial start pad, compacted route surface, and yellow route center
  guide are hidden by default. Add `--show-start-pad`, `--show-route-surface`, or
  `--show-route-guide` only when debugging those overlays.
- The RTX 5060 in this machine has 8 GB VRAM, so the default camera resolution
  is `640x360` and the lidar render product is small.
- `--obstacle-density none|sparse|medium|dense` controls the extra off-road
  upright barrier posts, standing trunks, rear posts, outer posts, and rocks.
  `sparse` is already visually dense; use `medium`/`dense` for heavier mapping
  stress tests.
- The imported Hunter STL meshes are generated by Isaac under `/visuals` and
  `/meshes`; the scene binds high-contrast body/wheel materials to those source
  meshes so the vehicle remains visible on the bright Mars/off-road terrain.
- Some Gazebo-only complex DAE assets, notably the tank/fire/soldier models,
  are not loaded by default because Isaac Sim 5.1's converter emits broken
  absolute `/props` USD references for them. The default scene uses the stable
  converted project assets so the camera/LiDAR stage starts without missing
  asset spam.
