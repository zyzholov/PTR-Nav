# Isaac Sim + FAST-LIVO2 Integration

The Isaac Sim backend lives in `src/isaac_sim_hunter_livo2`.

The default scene imports the real project Hunter SE URDF/STL model from
`src/simulator/fake_ugv/hunter_se/hunter_se_description`, places the Livox
MID360 above a compact IMU/D455 stack, and loads project map assets from
`src/simulator/fake_ugv/maps` by default. It is not a box-only equivalent scene.
The Isaac-specific FAST-LIVO2 config is `hunter_isaac_d455_sim.yaml`; the older
Gazebo mount config is kept separate.

Two off-road profiles are available:

```bash
# Local 571 m x 422 m Mars Gale crater mesh from this workspace. The Hunter
# starts directly on the sampled terrain; debug pad/path overlays are off by default.
./run_hunter_isaac_sim.sh headless:=true --scene-profile mars_gale --duration 600 --realtime

# Community Isaac Sim off-road forest/rock terrain from Offroad-Nav. Only the
# GeneratedForest prim is referenced, so the original ROS1 robot/sensor graph is
# not loaded into this ROS2 scene.
./run_hunter_isaac_sim.sh headless:=true --scene-profile offroad_nav --duration 600 --realtime
```

For interactive visual inspection in the Isaac UI, start without
`headless:=true` and keep the real-time renderer. GUI mode now defaults to the
PBR-textured Mars surface skin and CAD-quality sensor visuals:

```bash
./run_hunter_isaac_sim.sh --scene-profile mars_gale --mars-visual-detail full --mars-surface-skin full --sensor-visual-detail cad --viewer-camera close --obstacle-density sparse --camera-width 1280 --camera-height 720 --render-hz 10 --realtime
```

The default `--viewer-camera close` starts focused on the Hunter. Use
`--viewer-camera overview` or `--viewer-camera top` to inspect the full terrain
and surrounding Mars valley.

For first visual checks, use `--visual-only` so Isaac does not create RTX LiDAR,
D455, or ROS graphs while you are just checking the vehicle and terrain:

```bash
./run_hunter_isaac_sim.sh --scene-profile mars_gale --visual-only --mars-visual-detail full --mars-surface-skin full --sensor-visual-detail cad --viewer-camera close --obstacle-density sparse --camera-width 1280 --camera-height 720 --render-hz 10 --realtime
```

Do not use `--visual-only` when checking ROS topics. The normal GUI command
publishes `/clock`, `/livox/lidar`, `/imu/data_raw`, compatibility `/livox/imu`,
and D455 camera topics. For
ROS2 chassis control, start with `--drive-mode cmd_vel` and publish
`geometry_msgs/Twist` to `/cmd_vel`:

```bash
./run_hunter_isaac_sim.sh --scene-profile mars_gale --drive-mode cmd_vel --cmd-vel-angular-mode steer --cmd-vel-max-speed 0.35 --cmd-vel-max-steer 0.45 --mars-visual-detail full --mars-surface-skin full --sensor-visual-detail cad --obstacle-density sparse --camera-width 1280 --camera-height 720 --render-hz 10 --realtime
./check_hunter_isaac_topics.sh 45
./run_hunter_isaac_keyboard.sh
```

The Isaac scripts force `RMW_IMPLEMENTATION=rmw_fastrtps_cpp` by default,
because Isaac's internal Humble bridge is most reliable with FastDDS. They print
the effective `RMW_IMPLEMENTATION`, `ROS_DOMAIN_ID`, and `ROS_LOCALHOST_ONLY`.
If you override the RMW with `HUNTER_ISAAC_RMW_IMPLEMENTATION`, use the same
value for Isaac, FAST-LIVO2, `ros2 topic pub`, and
`check_hunter_isaac_topics.sh`; otherwise Isaac may be publishing while
`ros2 topic list` appears empty.

Use PathTracing only for screenshots/recording after the scene is confirmed,
because it is much heavier than the live mapping setup:

```bash
./run_hunter_isaac_sim.sh --scene-profile offroad_nav --renderer PathTracing --obstacle-density sparse --camera-width 1280 --camera-height 720 --render-hz 5 --realtime
./run_hunter_isaac_sim.sh --scene-profile mars_gale --mars-visual-detail full --mars-surface-skin full --sensor-visual-detail cad --renderer PathTracing --obstacle-density sparse --camera-width 1280 --camera-height 720 --render-hz 5 --realtime
```

The Offroad-Nav assets live at `/home/z/workspace_zyz/third_party/Offroad-Nav`
and are used as scene assets only. The upstream project targets Isaac Sim 4.5
and ROS1 Noetic, with a ROS2 port still under development, so the ROS2
FAST-LIVO2/D455/Livox chain remains the local one in this workspace. GUI auto
uses the upstream `medium.usd` scene; headless auto uses `hard.usd`. The upstream
`Grass_Instancer` is disabled at load time because it produces invalid prototype
warnings and can stall Isaac Sim 5.1; local upright posts, rocks, and trees are
still added for nearby SLAM features. Source:
https://github.com/LARIAD/Offroad-Nav

The Mars Gale profile now keeps a strict visual/physics split. With
`--mars-surface-skin full`, the UV-textured generated skin is the visible
terrain; the raw Mars mesh is not drawn over it as a grey shell. Physics/raycast
collision uses the lighter `gale_crater_collision_161.obj` with high-friction
sand material. The Hunter spawn height, obstacles, cameras, and optional debug
overlays are sampled against that collision mesh so they sit on the same terrain
surface.

`--mars-visual-detail auto` expands to `full` in the Isaac GUI and `fast` in
headless runs. `fast` uses the Gazebo collision OBJ as the visible terrain too.
It also generates XY UVs, applies the lightweight sand texture material, and
forces double-sided viewport styling so the OBJ does not render black in RTX
Real-Time. Use `full` with `--mars-surface-skin full` when judging visual
quality; use `fast` when running live mapping or when the GUI becomes
unresponsive.

`--mars-surface-skin auto` expands to `full` in the Isaac GUI and `none` in
headless runs. `full` creates a smooth-normal surface that follows the collision
height grid instead of floating above it, with a PBR albedo/normal/roughness
texture set for granular sand, rust-red mineral streaks, and small gravel;
nearby pebbles provide small visual detail. Use `--mars-surface-skin full` for
visual inspection/screenshots, and keep `none` for the fastest repeatable
mapping runs.
The artificial start pad, compacted route surface, and yellow route guide are
hidden by default; add `--show-start-pad`, `--show-route-surface`, or
`--show-route-guide` only when you need those debug overlays.

The Isaac sensor stack is:

```text
Livox base:  xyz=[0.18,  0.0, 0.250], pitch=25 deg
LiDAR frame: local xyz=[-0.012, 0.0, 0.047]
IMU frame:   xyz=[0.18,  0.0, 0.205]
D455 link:   xyz=[0.225, 0.0, 0.205], optical center +0.02 m on x
```

The default Hunter base spawn height is `--robot-z 0.30`, matching the imported
wheel mesh radius so the vehicle starts above the terrain instead of clipped into
the visible Mars skin.

Sensor appearance is controlled separately. `--sensor-visual-detail cad` loads
the local Livox MID360 CAD mesh from `src/simulator/livox_mid360/.../mid360.dae`
and only adds a small front window/status trim; the fake black top ring is not
used. D455 and IMU sit below the MID360 on a compact visual bracket. Use
`--sensor-visual-detail procedural` to skip the MID360 CAD reference for faster
GUI checks, or `none`/`--no-sensor-meshes` when only ROS topics matter.
`--livox-visual-yaw-deg` defaults to `180` to make the MID360 body visual face
the same way as the mounted LiDAR; this is visual-only and does not change the
ROS LiDAR frame or FAST-LIVO2 extrinsics. If you need to compare against the raw
CAD orientation, set `--livox-visual-yaw-deg 0`.

The Mars obstacle field no longer imports Rubicon detail rock assets, because
their converted bounds can become huge in Isaac and obscure the vehicle. Mars
rocks are now generated as controlled local meshes, while upright barrier posts,
standing trunks, and outer posts stay on the sampled terrain surface.

Run Isaac Sim sensors:

```bash
./run_hunter_isaac_sim.sh headless:=true --scene-profile mars_gale --mars-visual-detail fast --sensor-visual-detail procedural --duration 600 --realtime
```

Run FAST-LIVO2 on the Isaac topics:

```bash
./run_hunter_isaac_fastlivo2.sh
```

The scene publishes `/clock`, `/livox/lidar`, primary external `/imu/data_raw`,
compatibility `/livox/imu`, D455 color/depth topics, and uses FAST-LIVO2 config:

```text
src/FAST-LIVO2-ROS2/config/hunter_isaac_d455_sim.yaml
src/FAST-LIVO2-ROS2/config/camera_d455_isaac.yaml
```

FAST-LIVO2 publishes `/Odometry`, `/cloud_registered`, `/path`, `/rgb_img`,
and TF outputs when the Isaac sensor topics are live. Use `rviz:=true` on
`run_hunter_isaac_fastlivo2.sh` for the configured visualization chain.

The imported Hunter SE articulation can either stay parked, follow the
repeatable mapping circle, or accept `/cmd_vel` through `--drive-mode cmd_vel`.
By default `angular.z` is interpreted as the Hunter/Gazebo steering angle, and
the command mode drives the physical wheel/steering articulation through a
conservative torque controller.

Useful Isaac arguments:

```bash
./run_hunter_isaac_sim.sh headless:=true --duration 600 --render-hz 15 --stationary-duration 60 --realtime
./run_hunter_isaac_sim.sh headless:=true --scene-profile mars_gale --mars-visual-detail fast --sensor-visual-detail procedural --obstacle-density medium --duration 600 --realtime
./run_hunter_isaac_sim.sh headless:=true --scene-profile offroad_nav --obstacle-density sparse --duration 600 --realtime
./run_hunter_isaac_sim.sh --scene-profile mars_gale --mars-visual-detail full --mars-surface-skin full --sensor-visual-detail cad --obstacle-density sparse --camera-width 1280 --camera-height 720 --render-hz 10 --realtime
./run_hunter_isaac_sim.sh --scene-profile mars_gale --drive-mode cmd_vel --cmd-vel-angular-mode steer --mars-visual-detail full --mars-surface-skin full --sensor-visual-detail cad --obstacle-density sparse --camera-width 1280 --camera-height 720 --render-hz 10 --realtime
./run_hunter_isaac_sim.sh --scene-profile offroad_nav --map-asset offroad_nav_medium --visual-only --sensor-visual-detail cad --viewer-camera close --camera-width 960 --camera-height 540 --render-hz 10 --realtime
```

`--realtime` is recommended for live FAST-LIVO2 testing so Isaac does not push
simulated sensor time much faster than the mapper can consume it.

For faster debugging without changing the robot model:

```bash
./run_hunter_isaac_sim.sh headless:=true --scene-profile light --map-asset none --no-landmarks --no-sensor-meshes --obstacle-density none --realtime
```
