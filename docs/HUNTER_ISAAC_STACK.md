# Hunter Isaac Simulation Stack

## Data and Control Flow

```text
Isaac Sim Hunter SE articulation + Mars Gale collision/visual terrain
  |-- RTX LiDAR /livox/lidar (PointCloud2, 10 Hz simulation time)
  |-- external IMU /imu/data_raw, /livox/imu (same sensor, 200 Hz)
  |-- /clock and /isaac/ground_truth/odometry (diagnostics only)
  +-- optional D455 RGB/depth topics (--camera-enabled)
          |
          v
FAST-LIO2 /Odometry + /cloud_registered (current frame, 10 Hz)
          |                     +-- /Laser_map (bounded cumulative map, 1 Hz, RViz)
          |                     +-- terrain_nav /terrain_nav/live_obstacles (short-lived hard obstacle layer)
          v
terrain_analyzer_gpu (CUDA; CPU fallback via terrain_backend:=cpu)
  /car_0_terrain_analyzer/fused_map (local SE2Grid)
          v
FSGP-BGK / SEB (CuPy; CPU fallback)
  /car_0_fsgp_bgk/fused_map (yaw-aware risk)
          v
terrain_nav: /goal_pose -> /cmd_vel -> Hunter wheel/steering articulation
```

The current planner is **not** a persistent global + local planner. It runs
an A* guide over the current rolling risk grid, then a vehicle-kinematic
local search/curve planner and a path tracker. A remote goal is advanced
through local subgoals as the grid moves. Long routes through areas outside
the rolling grid, dead ends, and global route recovery remain unimplemented.
The enhanced SE2 risk is a soft directional cost; the geometric
`baseline_risk` and live obstacle layer are hard collision gates.

In simulation, `terrain_nav` also consumes the fresh registered cloud in
`map` frame. It marks local height discontinuities with at least two returns,
retains them for up to 1.0 s through brief occlusion, and inflates them by 0.25 m
before checking the vehicle footprint. A newly intersecting path is stopped
and replanned; every planned path is checked again before a drive command.
Two successive frames with low ground returns and no tall return clear a
previously occupied cell without waiting for the hold timeout.
If the live cloud becomes older than 0.9 s, the planner commands an immediate
stop instead of silently treating the obstacle layer as clear. This is a
local, non-tracking obstacle gate, not a motion predictor. The node default
is disabled for real hardware until its cloud frame, timing, and extrinsics
have been verified; the Isaac simulation config enables it.
If a drive command above 0.25 m/s produces less than 0.12 m odometry progress
for 4 s, the controller also latches an immediate stop until a new goal arrives.

`/Laser_map` is for cumulative visualization. Terrain analysis deliberately uses
`/cloud_registered` and the corresponding `/Odometry`, not the much larger
global map. The FAST-LIO `map` frame has an arbitrary initial yaw relative to
Isaac world; drift diagnostics align the initial poses before comparing motion.
In the Isaac configuration, FAST-LIO's local IKD tree and cumulative map remove
an old 3D voxel only after three independent free-ray scans with a farther
return. The experimental lower-ground replacement and neighborhood expansion
paths are disabled by default because a lower return beside an occluded static
surface is not proof that the old surface is gone. Current occupied hits reset
the free evidence; occlusion alone does not clear a cell. The IKD tree is
updated per scan, while the RViz cumulative map refreshes at 1 Hz. The CUDA
terrain front-end also requires repeated free-ray evidence and a supported
neighboring ground level before any optional stale-elevation replacement. The
direct cleaner is enabled in the Isaac profile; ground replacement and
expansion must be explicitly enabled for an A/B trial. The navigation live
layer remains reactive, not a dynamic-object tracker or motion predictor.
The cumulative map cannot reliably clear an old obstacle within a fixed time
when the sensor gets no free-space or ground observations through its old
position. Some moving-obstacle trials retained old points for about 5 s even
while the live navigation risk had already cleared. A temporal object tracker
or another independent observation is needed for a stronger guarantee; do not
use `/Laser_map` alone as the immediate collision layer.
The planner publishes `Twist.angular.z` as a front steering angle, and Isaac
must run with `--cmd-vel-angular-mode steer`. Its speed/steering clamps match
the terrain navigation config (1.33 m/s and 0.40 rad).

The current Isaac RTX `PointCloud2` has xyz but no per-point time field.
`synthetic_point_time: false` avoids artificial deskew during turns. This is a
simulator approximation, not a sensor-fidelity claim about the real MID360.
The external IMU is a separately placed simulated sensor, not the LiDAR IMU.
FAST-LIVO2/D455 is retained as an optional branch, not part of this default
LiDAR-IMU navigation stack.

## Dynamic-map audit follow-up (2026-10-03)

The FAST-LIO Isaac parameter service is now checked by
`scripts/sim/checks/mapping_parameters.py`; the GPU smoke run compared 68
parameters and reported no mismatch. The conservative default policy is
`dynamic_clear_enabled=true`, `dynamic_clear_frames=3`,
`dynamic_ground_replace_enabled=false`, `dynamic_expand_enabled=false`, and
`dynamic_ground_min_height=0.30`. This is intentional: repeated viewpoint-
diverse guard trials showed that the aggressive ground/expansion modes could
lower the static-obstacle minimum, so they are not presented as production
ghost removal.

The latest guard regression passed at
`log/modular_smoke_20261003_233119`: static guard minimum 4/5, registered
obstacle points dropped to zero while the obstacle was present in the guard
fixture, cumulative-map observation delay 0.74 s, LiDAR 10 Hz, external IMU
200 Hz, terrain 10.03 Hz, and enhanced risk 5.02 Hz. The ordinary clearance
trial at `log/modular_smoke_20261003_233427` reached its goal and cleared the
risk/live layers in 0.09/0.19 s, but was correctly rejected by the strict
cumulative-map assertion because that fixture did not produce a sufficient
high-obstacle baseline in the selected ROI. This is a test limitation and a
remaining cumulative-map coverage gap, not evidence that the global map is
always clear.

The parameter audit is included in `scripts/sim/checks/pipeline_smoke.py`, so
future full-stack runs fail early when a YAML parameter is undeclared or a
launch override is ignored. No GitHub commit or push was made during this
iteration.

## RViz terrain-navigation viewer regression (2026-10-08)

The shared terrain/planning viewer now uses
`fsgp_bgk_se2/rviz/hunter_terrain_navigation.rviz`, rather than the legacy ARC
viewer. Its default layers are the draped FSGP elevation and risk clouds,
live obstacle occupancy, planner path/reference, active goal, odometry, the
vehicle model, and the cumulative `/Laser_map` display. The accumulated map is
shown with a fixed light-blue color so it remains visible even when a LiDAR
message has missing or flat intensity values; it can be disabled for a clean
risk-only view.
The saved overview is an 18 m orbit; `Vehicle Follow` is a 7 m close view.

The SE2Grid converter now honors rectangular native dimensions, circular
`start_index_*`, storage labels, data offsets, exact layer names, and the
current odometry yaw bin. Risk cells are draped from `bgk_mean` instead of
being rendered on a fixed floating plane. This is checked by
`scripts/sim/checks/visualization_probe.py`.

Regression results from the final source are recorded below. An earlier
`095618` detour failure is retained in the log history as a tuning trial, not
as the current result.

* `colcon test --packages-select terrain_nav fsgp_bgk_se2 se2_grid_converter`:
  27 tests passed across the planner, SE2Grid converter, and FSGP/SEB layers.
* Headless reverse-goal stack `log/modular_smoke_20261008_150240`: the vehicle
  issued `-0.55 m/s`, moved 1.49 m, and finished with 0.51 m truth error;
  FAST-LIO maximum XY discrepancy was 0.003 m.
* Static-column detour `log/modular_smoke_20261008_150756`: reached the goal
  with 1.99 m lateral bypass; the obstacle probe measured
  `baseline_risk_local_max=1.0` and `risk_local_max=1.0`.
* Dynamic crossing `log/modular_smoke_20261008_150504`: reached the goal after
  live-obstacle stop/replan, with 469 live occupied cells at peak and risk-map
  age P95 of 0.08 s. Sensor rates remained 10 Hz LiDAR and 200 Hz IMU.
* GUI stack `log/modular_smoke_20261008_151102`: shared RViz loaded the final
  configuration and all visualization products passed. The GUI real-time
  factor was 0.89--0.91, while the headless runs stayed near 1.00; this is the
  current rendering performance boundary on this host.

### Six-issue regression closure (2026-10-08)

The follow-up regression used the current source and isolated ROS domains:

* The shared RViz view loaded `/Laser_map` and the FSGP risk/elevation products.
  The visualization probe measured 4316 common draped cells, and the GUI run
  retained the full sensor chain at approximately 10 Hz LiDAR, 200 Hz external
  IMU, 10 Hz terrain, and 5 Hz enhanced-map output.
* The vehicle marker lifetime is now persistent (`0.0`) and uses the odometry
  message frame. This prevents short `/Odometry` transport gaps from making
  the model disappear. Both modular and legacy combined launch paths use this
  setting.
* FSGP-BGK now appends `collision_risk` from the synchronized registered cloud.
  Thin vertical columns are retained as hard risk instead of being smoothed
  away. The final static-column run measured `baseline_risk_local_max=1.0` and
  `risk_local_max=1.0`, then detoured with 1.99 m maximum lateral offset:
  `log/modular_smoke_20261008_150756`.
* Planning latency in the detour log was 56--76 ms for the terrain lattice and
  85--156 ms for the bounded guide search. The guide search is now capped at
  65 ms (or 60% of the planner budget), and stale map/odometry data causes a
  stop instead of a late command.
* A close rear goal produced a real negative command (`min_cmd_v=-0.55 m/s`),
  moved the vehicle about 1.49 m, and finished with 0.51 m truth error in
  `log/modular_smoke_20261008_150240`. The reverse arc still requires a fresh
  live scan to validate unknown cells; it does not blindly drive through them.
* A crossing dynamic obstacle triggered a live-obstacle stop/replan, reached
  the goal, and its live occupancy peak was 469 cells in
  `log/modular_smoke_20261008_150504`. A fully blocking `emerge` obstacle is
  intentionally treated differently: the vehicle stops safely rather than
  claiming a path exists. It is not guaranteed to invent a bypass when the
  lane is completely sealed.

The simulator supports physical dynamic-obstacle fixtures through
`--nav-test-scenario crossing|emerge|clearance`; the obstacle is moved as a
collidable USD body and its pose is published on
`/isaac/nav_test_obstacle/pose`. These are regression fixtures, not a general
multi-agent behavior simulator. No GitHub commit or push was made.

## Start and Inspect

Start each boundary in a separate terminal:

```bash
# 1. Isaac Sim environment and sensors
scripts/sim/start_simulation.sh

# 2. FAST-LIO2 odometry and cumulative map
scripts/sim/start_mapping.sh rviz:=true

# 3. CUDA terrain-analysis front-end, FSGP-BGK/SEB, and the shared RViz
scripts/sim/start_terrain_analysis.sh terrain_backend:=gpu fsgp_gpu_backend:=cupy

# 4. Planner and vehicle control; this stage does not start another RViz
scripts/sim/start_planning_control.sh
```

The terrain stage owns the single navigation RViz window. It starts the
SE2Grid-to-PointCloud converters, vehicle marker publisher, and the shared
`hunter_terrain_navigation.rviz` configuration. Planning/control only publishes
the planner path, goal marker, live obstacle grid, and `/cmd_vel`; it does not
open a second RViz. To disable the shared window, pass `rviz:=false` to step 3.

For manual driving, keep the terrain stage running, omit step 4, and run this
instead:

```bash
scripts/sim/start_keyboard_control.sh
```

The simulator uses a 960x540 RTX viewport at 10 Hz by default, with the full
Mars visual skin and terrain collision enabled. Add `--camera-enabled` to step
1 for optional D455 RGB/depth topics. In a separate GUI sensor test, the
960x540 RGB image contained visible textured terrain, 63.5% of 960x540 depth
pixels were valid, and the aligned point cloud contained 518400 pixel slots
(not a claim that every XYZ was valid).
The captured RGB frame is at `log/d455_gui_rgb_sample.png`. The headless
mapping profile clamps the optional camera to 640x360 and uses a simplified
surface skin; it should not be judged as the high-fidelity GUI appearance.
The camera-enabled GUI simulator alone settled near `RTF=0.9` on this host,
while the no-camera full stack held about `RTF=1.0`. Camera-enabled full-stack
mapping throughput was not measured in that test.
Use `--duration 120` with step 1 for a
bounded headless run, for example:

```bash
scripts/sim/start_simulation.sh headless:=true --duration 120
```

The boundaries intentionally use explicit topics:

```text
FAST-LIO /cloud_registered + /Odometry
  -> terrain_analyzer /car_0_terrain_analyzer/fused_map
  -> fsgp_bgk_se2 /car_0_fsgp_bgk/fused_map
  -> terrain_nav /cmd_vel
```

`terrain_analyzer` builds local elevation, variance, SDF, obstacle and an
already heading-indexed baseline-risk layer from the current registered cloud.
FSGP-BGK/SEB consumes that grid, applies BGK-style weighted smoothing and
evaluates vehicle support, pitch, roll and steps over heading slices using SEB.
The adapter has not ported the upstream sparse Gaussian process (SGP). It
publishes enhanced heading-aware risk; it
does not replace the terrain front-end or the planner. `terrain_nav` is kept
separate so it is possible to inspect maps without a second `/cmd_vel`
publisher.

Use these checks while the relevant modules are running:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
scripts/sim/checks/topics.sh 45
scripts/sim/checks/rates.sh
python3 scripts/sim/checks/map_rates.py --duration 8 --min-rate 2
scripts/sim/checks/topics.sh 45 camera  # only when --camera-enabled is active
python3 scripts/sim/checks/camera_frames.py --save-color log/d455_rgb_sample.png
scripts/sim/checks/slam_drift.sh 20
scripts/sim/checks/pipeline_smoke.py --timeout 90
python3 scripts/sim/checks/pipeline_smoke.py --domain 98 --trial-scenario detour
python3 scripts/sim/checks/pipeline_smoke.py --domain 98 --trial-scenario ramp
python3 scripts/sim/checks/pipeline_smoke.py --domain 98 --trial-scenario step
python3 scripts/sim/checks/pipeline_smoke.py --domain 98 --trial-scenario emerge --trial-timeout 60
python3 scripts/sim/checks/pipeline_smoke.py --domain 98 --trial-scenario crossing --trial-timeout 60
python3 scripts/sim/checks/pipeline_smoke.py --domain 98 --trial-scenario clearance --trial-timeout 65
python3 scripts/sim/checks/pipeline_smoke.py --domain 98 --fault-stale-map
```

The smoke command uses a separate ROS domain, starts and stops the four
modules itself, records logs under `log/modular_smoke_*`, and requires a
forward goal to produce both FAST-LIO and independent Isaac-truth motion.
Add `--gui --skip-goal --domain 88` to test the rendered viewport without
moving the vehicle.
Use `--goal-lateral 1.0 --domain 97` for a short turning goal. The probe
requires nonzero steering commands and independent Isaac-truth arrival.
The `--trial-scenario` runs use a small light scene with physical
collision fixtures: a centered cylinder, a 0.45 m continuous ramp, a
0.5 m high step, or a kinematic cylinder that enters/crosses the lane after
the vehicle begins moving. They run one at a time in an isolated ROS domain and check
truth trajectory, map freshness and scenario-specific safety. They are
functional tests, not visual-fidelity benchmarks for Mars Gale.

The old all-in-one launcher remains only as an archive under
`scripts/old/sim/`; it is not part of the supported startup path.

The probe scripts force `rmw_fastrtps_cpp`, matching the Isaac launcher. This
is important on hosts whose interactive shell defaults to CycloneDDS. The
goal probe publishes the goal during a short discovery window, and the command
probe fails if Isaac has already exited or if `/cmd_vel` has no subscriber.

## Verification and Limits

The live headless chain has been checked through nonempty LiDAR/IMU, FAST-LIO
odometry and cumulative map, native CUDA terrain output, CuPy FSGP-BGK/SEB
output, SE2 yaw-aware cost construction, planner commands and physical
steering. The goal probe has an explicit motion gate so a closed-loop arrival is
only accepted after the vehicle has actually displaced; rerun it after starting
all four stages. These are short-run results in one Mars Gale area, not a
long-duration localization or collision-safety guarantee.

On 2026-09-29 the isolated four-stage headless smoke test passed. It observed
38.0k-38.4k LiDAR points/scan at 10.0 Hz, external IMU at 200.0 Hz, a
nonempty cumulative `/Laser_map`, 9-layer CUDA terrain output, and 12-layer
CuPy BGK/SEB output. The 2 m forward goal reached 0.54 m FAST-LIO error;
Isaac truth recorded 1.56 m motion and 0.44 m error, with 48 nonzero commands.
The 10 s stationary drift window measured 0.003 m maximum aligned XY error.
The steady simulation profile was about RTF 1.00. These numbers describe
this single short run and do not establish long-route reliability.

A separate headless turning smoke test targeted 2 m forward and 1 m left.
It passed with 56 nonzero steering commands, 1.87 m Isaac-truth motion, and
0.43 m truth error; LiDAR/IMU and both terrain maps continued updating at
about 10/200/10/5 Hz. The post-arrival 10 s odometry check had 0.005 m maximum
aligned XY error. This confirms one short turn, not slope traversal or
obstacle-collision safety.

The CPU fallback also passed a four-stage closed-loop smoke test on the same
host. Its baseline terrain map published at 14.93 Hz with 10.07 Hz distinct
input timestamps, and the CPU BGK/SEB map published at 7.50 Hz. A 2 m forward
goal produced 1.53 m independent Isaac-truth motion. CPU and CUDA terrain
front-ends expose different diagnostic layers (6 vs 9 baseline layers in this
test); both provide the risk/elevation/validity inputs required by BGK/SEB,
but the test does not prove numerical equivalence of the two front-ends.

After cumulative realtime pacing was added, the tested headless stack held
about `RTF=1.00` with roughly 4.6-4.8 ms work per 5 ms physics step. A separate
control test measured a 0.30 rad steering command at 0.8 m/s as target/actual
curvature `0.563/0.540 1/m`, with FAST-LIO/Isaac yaw ratio `0.979`. A straight
command measured actual curvature `0.002 1/m`. The 10 Hz LiDAR stream produced
about 38k points per frame, while both external IMU topics held 200 Hz
simulation timestamps and about 199-201 Hz wall rate. `/Laser_map` produced
four bounded cumulative samples during a 6 s command probe at about 0.48 Hz.

The host still reports a `powersave` CPU governor, so the simulator startup
warns about it; the scripts do not change system policy. The Python environment also has
both `cupy-cuda11x` and `cupy-cuda12x` installed. CuPy is running on the GPU,
but a clean environment with one CUDA-matched CuPy package is recommended for
deployment. Isaac still emits expected extension-deprecation and RTX sensor
warnings; they do not prevent the tested ROS data path from running.

A 2026-09-29 GUI smoke test reached every module readiness boundary. With
the 960x540 RTX viewport, DLSS balanced and full PBR Mars visual skin, its
steady-state profile was roughly `RTF=1.00` and LiDAR wall rate 10.01 Hz on
this host. A second GUI test with vehicle motion observed a 10.03 Hz baseline
terrain map and 5.01 Hz enhanced map, each with changing source timestamps.
The captured viewport in `log/modular_smoke_20260929_144542/gui_window_capture.png`
shows the Hunter and textured ground; this confirms nonblank rendering, not
photorealistic equivalence with the official Isaac demo scenes.
`start_simulation.sh` keeps GUI rendering at the LiDAR's 10 Hz
cadence so the mapping path retains the dense non-full-scan cloud
instead of producing partial ~19k-point clouds. Use `headless:=true` for
mapping benchmarks and `display:=quality` only when visual fidelity is more
important than throughput.
Before the environment-order fix, two full-stack headless attempts timed out
before `/clock`; one full-surface attempt spent 84 seconds between Isaac's
`app ready` and `Simulation App Startup Complete`. After the fix, the same
full-surface path completed Isaac startup in about 12.5 seconds and reached
all module topic readiness checks. The module scripts print the failing stage's logs on topic
timeout. This is a local repeat test, not a guarantee against every Isaac
startup hang.

Run `python3 scripts/tools/benchmark_terrain_risk.py` for repeatable flat/ramp/step,
unknown-region, heading-specific hazard and CPU/CuPy parity checks. The latest
104x104x36 run after slope-corrected step scoring measured about 66-68 ms on
CPU and 3.2-3.3 ms on warm CuPy, with CPU/CuPy risk parity within 1e-5
absolute tolerance. These are local map-kernel timings, not
end-to-end ROS latencies. The enhanced risk preserves the input baseline risk;
the current planner uses it as a soft heading-aware cost, while geometric
baseline risk remains lethal. The bridge uses a
BGK-style smoother rather than the upstream sparse GP. It is not a validated
replacement for the original SEB navigation stack on diverse terrain.

### Physical Navigation Trials (2026-09-29)

The isolated four-stage light-scene trials passed the following measured
gates, using independent Isaac truth rather than FAST-LIO odometry for
vehicle position:

| Fixture | Truth result | Risk-map P95 period / source age |
| --- | --- | --- |
| Cylinder, radius 0.65 m at 3 m | Two runs reached the 7 m goal; lateral deviations 2.11/2.02 m, minimum cylinder-center distances 1.345/1.366 m, final truth errors 0.637/0.639 m | 0.235/0.228 s; 0.085/0.095 s |
| Smooth ramp, rise 0.45 m | Reached the 7 m goal, maximum truth height gain 0.453 m, maximum pitch 0.172 rad, truth error 0.648 m | 0.231 s; 0.090 s |
| High step, rise 0.50 m | Advanced 3.315 m toward the step front at 4 m and stopped without climbing; known local risk reached 1.0 | 0.229 s; 0.070 s |

The tests also required 10 Hz LiDAR, 200 Hz external IMU, about 10 Hz native
terrain maps and about 5 Hz enhanced maps. The headless simulator stayed near
real time. A separate default Mars Gale turning smoke test after the final
planner rebuild reached its 2 m forward / 1 m lateral goal with 0.43 m
Isaac-truth error. Logs are under `log/modular_smoke_20260929_201028`
(ramp), `..._201650` (step), `..._201907` and `..._202234` (detour), and
`..._202504` (Mars turn).

The previous heightfield winding made the simulated vehicle fall through the
ramp; that winding is now upward. SEB previously treated the total footprint
height span on a normal slope as a hard step. It now removes local pitch/roll
before scoring the residual height span, while still assigning hard risk to a
step. Short collision-checked partial paths let the planner approach observed
free space without allowing arbitrary unknown cells. The planner also sends
an immediate zero command when odometry or risk map age exceeds 0.5/1.0 s,
respectively. In the forced map-dropout trial, the late command speed was
0.0 m/s and independent Isaac-truth body speed was 0.003 m/s after stopping
(`log/modular_smoke_20260929_203256`). This confirms a bounded dropout case,
not every possible sensor failure mode.

On 2026-09-30, the current live-obstacle revision passed isolated full-stack
detour and late-emerge runs. The 7 m static-cylinder run reached within
0.645 m of truth goal, deviated 1.949 m laterally, and kept at least 1.436 m
between vehicle and cylinder centers (`log/modular_smoke_20260930_092617`).
The 9 m late-emerge run reached within 0.635 m, with a 1.238 m minimum
distance to the final cylinder center (`..._092846`). The crossing run reached
within 0.649 m after a stop; its synchronized vehicle/obstacle truth poses
showed 2.480 m minimum center separation (`..._093933`). LiDAR/external IMU,
baseline terrain, and enhanced risk held about 10/200/10/5 Hz in those runs.
The synchronized obstacle pose topic is test-only:
`/isaac/nav_test_obstacle/pose`.
The later truth-pose-gated late-emerge rerun reached within 0.639 m and
kept 1.359 m minimum vehicle/obstacle center separation
(`log/modular_smoke_20260930_094139`). A default Mars Gale short turning
goal also passed after these changes: 0.46 m Isaac-truth goal error, 62 drive
commands and 60 steering commands, with 10/200/10/5 Hz LiDAR/external
IMU/terrain/enhanced-risk rates (`..._094416`).

The final no-path replan throttle was checked again on 2026-09-30. Static
detour, late-emerge and crossing all reached their goals, with minimum
truth center separations of 1.362, 1.382 and 2.363 m respectively. The
repeated high-step run stopped before the step at 3.358 m forward progress.
The post-trial `/cmd_vel` check returned to about 25 Hz, compared with
13.6 Hz in the earlier step run that repeatedly replanned without a path;
the earlier false 0.6 s live-cloud age warnings were absent. Logs are in
`log/modular_smoke_20260930_100650` (detour), `..._100125` (emerge),
`..._100400` (crossing), and `..._095844` (step). The ramp rerun gained
0.452 m elevation with 0.166 rad maximum pitch (`..._094637`). A forced
terrain-map dropout commanded 0 m/s and left about 0.003 m/s Isaac-truth
motion (`..._095406`). These are bounded short trials, not a guarantee that
planning and control remain at 25 Hz during every long or cluttered run.

Limits: the light-scene fixtures do not prove Mars Gale long-route navigation,
real MID360 timing fidelity, arbitrary dynamic-obstacle avoidance, or equivalence to the
upstream sparse-GP FSGP-BGK method. The step trial demonstrates stopping before an
untraversable rise; it does not establish obstacle-contact force limits.

For a short closed-loop check, source this workspace in a second terminal and
run `python3 scripts/tools/hunter_goal_probe.py --distance 2.0`. The probe now
waits for FAST-LIO initialization, requires nonzero planner commands and
requires measured vehicle displacement; when Isaac truth odometry is present it
also checks independent ground-truth displacement. A coordinate jump in
`/Odometry` can therefore no longer be reported as a successful arrival.
The terrain front-end logs are normally 6-12 ms per update after warm-up and
the FSGP-BGK/SEB node selects CuPy automatically for the 104x104x36 map. These
earlier short-goal observations do not establish long-route or real-world
collision safety; the physical fixture trials above add bounded simulation
evidence for basic bypass, ramp traversal and step rejection.

The grid bridge now explicitly converts the native circular-buffer layout of
`SE2Grid` into world-axis matrices before BGK/SEB processing and converts the
result back before publishing. Run the regression suite after changing that
bridge:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
PYTHONPATH="src/fsgp_bgk_se2:${PYTHONPATH:-}" \
  /usr/bin/python3 -m pytest -q \
  src/fsgp_bgk_se2/test/test_se2_grid_tools.py \
  src/fsgp_bgk_se2/test/test_bgk_seb.py
```

## Dynamic-Map Audit (2026-09-30)

The simulator's `clearance` fixture is a physical 1.7 m diameter cylinder
which stands in the lane and then moves out. The trial asserts that it was
observed, that the vehicle reached its goal without close contact, and that
`/Laser_map`, enhanced terrain risk, and the live obstacle layer each clear
within 1.5 s. This is deliberately a strict regression test, not a claim of
general dynamic-object tracking.

The quick-clear cases passed with cumulative-map delays of 0.40, 0.48, 0.59,
and 1.13 s during tuning. The final conservative build also passed one run
at 0.61 s (`log/modular_smoke_20260930_170429`). However, other runs with a different
vehicle viewpoint failed: 90-117 high map points remained for roughly
4.8-5.6 s, even though current registered scans had zero high returns in the
old region and the navigation risk cleared in about 0.4-0.8 s. Those failed
runs are retained at `log/modular_smoke_20260930_161704`,
`log/modular_smoke_20260930_162256`, and
`log/modular_smoke_20260930_163336`. Widening the ray-clearing seed and
using full-scan ground evidence did not reliably close this gap. An
experimental seedless clearing rule was removed because it did not fix the
failure and could incorrectly erase unseen static surfaces. The cumulative
FAST-LIO map therefore **does not yet meet** a hard 1.5 s ghost-removal
requirement in all tested viewpoints. The slow-clear failures were captured
during tuning, so the final build still needs repeated viewpoint-diverse
trials before any stronger conclusion is possible.

The static-cylinder rerun kept all 154 measured high map points and reached
the goal (`log/modular_smoke_20260930_165306`). The high-step rerun retained
112/112 points, marked hard risk, and stopped before the step
(`..._164946`). A dynamic crossing trial reached its goal after a live-layer
emergency replan, with 2.439 m minimum truth center separation
(`..._165717`). A four-module, stationary headless Mars Gale run maintained
about 10/200/10/5 Hz LiDAR/IMU/terrain/enhanced-map rates and RTF 1.00
(`..._165515`). GUI responsiveness and long-duration moving Mars navigation
were not measured in this audit.
After removing the seedless rule, the final high-step rerun retained 106/106
high points and stopped before the step (`..._170735`). In that run the new
stuck watchdog fired after 4 s of 0.55 m/s command and only 0.03 m measured
progress, then latched zero speed until a new goal. The earlier static-cylinder
run used the seedless rule. A final-build static-cylinder run subsequently
reached the goal with 1.440 m minimum center separation and retained 168/168
high map points (`..._171212`). Static retention outside these fixtures
still needs wider scene coverage.

### Repeated revalidation (2026-09-30)

The accumulated-map cleaner now expands only from a compact group of direct
free-ray deletions observed in at least two scan frames. Expanded clearing is
limited to the published `/Laser_map` history; the FAST-LIO registration tree
only receives deletions backed by direct free rays. This remains enabled by
the Isaac profile and does not change the real-robot profile.

The final build passed seven consecutive isolated GPU stack trials: three
moving-cylinder clearance runs (cumulative-map clearing in 0.92, 1.26 and
1.07 s; risk in 0.62, 0.56 and 0.17 s; live layer in 0.52, 0.16 and 0.17 s),
one static-cylinder detour (153/153 final high points retained), one high-step
stop (110/110 retained, final command zero), one moving crossing (2.561 m
minimum synchronized truth center separation), and one stationary Mars Gale
throughput run. Mars Gale held RTF 1.00 and approximately 10/200/10/5 Hz for
LiDAR/external IMU/GPU terrain/CuPy enhanced risk. A separate CPU-backend
detour passed with approximately 10/200/10/7.4 Hz for the same chain; its
terrain frontend produced fresh stamps at 10.04 Hz.

All seven final-build runs and their full logs are indexed in
`log/repeat_pipeline_20260930_193243/summary.json`; the CPU run is at
`log/modular_smoke_20260930_194652`. Repeat the tests with:

```bash
python3 scripts/sim/checks/repeat_pipeline.py \
  --case clearance:3 --case detour:1 --case step:1 \
  --case crossing:1 --case mars:1 --first-domain 70
```

The broader seed expansion tested immediately before the final build passed
seven clearance runs but also deleted roughly 2680 ground-supported historical
voxels during a stationary Mars run. The two-frame cluster gate reduced that
count to 375 in the comparable final Mars run (direct deletions were 1515).
These counters do **not** identify which deleted voxels were real static
surfaces; they are a warning to keep testing on varied static geometry, not a
proof of perfect map preservation. A fourth earlier clearance trial was
incorrectly failed by the test harness: a pre-vacate `/Laser_map` message
arrived after the obstacle-pose callback and was counted as post-vacate.
The harness now classifies messages by their timestamps; that trial actually
cleared in 1.02 s and three subsequent timestamp-corrected runs passed.

Remaining limits: these short fixtures do not establish guaranteed ghost
removal when the old volume is completely unobservable, long-route Mars
navigation, GUI frame rate, or static-landmark preservation under prolonged
occlusion. The installed CuPy environment currently contains both
`cupy-cuda11x` and `cupy-cuda12x`; CUDA 12 was used successfully in these
tests, but the duplicate-install warning remains an environment cleanup item.

## Visualization and planner revalidation (2026-10-09)

The shared RViz configuration now uses one 320 px Displays dock and leaves the
remaining window for the 3D viewport. The default view keeps the FAST-LIO
accumulated map, the vehicle model, the terrain surface and the navigation
path visible together. The dense risk PointCloud2 overlay is disabled by
default because risk is now carried by the terrain mesh vertex colours; it can
still be enabled from the Displays panel for diagnostics. The mesh is built
from local finite neighbouring height cells only, rejects height jumps over
0.65 m and never bridges the circular-buffer edge. A check for this product is
included in `scripts/sim/checks/visualization_probe.py`.

The current checks passed:

* headless full chain: LiDAR 10.000 Hz timestamp rate, external IMU 200.000
  Hz, terrain 10.09 Hz, FSGP-BGK 5.10 Hz, FAST-LIO/Isaac stationary XY error
  no more than 0.003 m;
* GUI full chain: RViz snapshot at
  `log/modular_smoke_20261009_153359/rviz_snapshot.png`, 1042 px render
  viewport, 31 FPS in the captured frame, terrain surface 6154 triangles and
  1.21 m height span;
* forward goal: 1.44 m FAST-LIO displacement, 1.56 m truth displacement and
  0.44 m final truth error;
* close reverse goal: minimum command speed -0.55 m/s, 1.45 m truth
  displacement and 0.55 m final truth error.

The GUI run is naturally more expensive than headless mode: one run measured
approximately 9.8 Hz LiDAR and 183 Hz IMU wall delivery while preserving the
correct 10/200 Hz simulation timestamps. Headless mode remains the reference
for real-time sensor throughput. The XWD capture and its viewport-size guard
are implemented in `scripts/sim/checks/rviz_snapshot.py`.

The cumulative-map display was then revalidated with the independent dense
publication path. FAST-LIO still registers the scan with the existing
`point_filter_num=1` and 0.20 m registration surface leaf, while
`publish.map_dense_en=true` accumulates deskewed full-scan returns into a
0.10 m publication voxel. This increased the first `/Laser_map` message from
roughly 15k points to 30-45k points in the same startup window without
changing odometry or terrain input. The headless run held 10.04 Hz LiDAR,
200.20 Hz IMU and 10.01/5.01 Hz terrain/enhanced terrain; the GUI run held
9.68 Hz LiDAR and 195.96 Hz IMU wall delivery, correct 10/200 Hz stamps,
1042 px RViz viewport and 31 FPS. The final dense screenshot is
`log/modular_smoke_20261009_193425/rviz_snapshot.png`.

## Planner and control latency revalidation (2026-10-10)

The navigation node now separates the control loop from the lattice planner:

* `terrain_nav` owns a dedicated planner worker. Goal changes, map changes and
  live-obstacle changes enqueue a plan request instead of running a potentially
  long search inside the control timer. A goal sequence check discards stale
  plans when a newer goal arrives.
* The control timer publishes at 50 Hz (`control_period=0.02`). It reads
  immutable atomic snapshots of the terrain costmap and the live obstacle
  layer, so map copies and point-cloud bookkeeping do not take the control
  mutex. A newly unsafe path still causes an immediate stop while the worker
  computes the replacement path.
* The default planner budget is now 90 ms, the minimum replan interval is
  250 ms, and reference-path smoothing is 20 iterations. These are latency
  settings, not changes to the FAST-LIO2 registration or downsampling
  parameters.

The post-change checks passed:

* headless forward chain: `/cmd_vel` 49.998 Hz, first non-zero response
  0.314 s, LiDAR 10 Hz, external IMU 200 Hz, and stationary FAST-LIO/Isaac
  XY error 0.007 m; log:
  `log/modular_smoke_20261010_091807`;
* dynamic obstacle emerge/vacate: obstacle minimum synchronized truth
  separation 1.43 m, front-stop commands present, no stale laser/live/risk
  ghost after vacating, and `/cmd_vel` 49.999 Hz; log:
  `log/modular_smoke_20261010_092915`;
* GUI full-chain smoke: RViz viewport 1042 px, RGB standard deviation 99.27,
  terrain surface and vehicle/map products visible, with wall delivery near
  10 Hz LiDAR and 200 Hz IMU; log and screenshot:
  `log/modular_smoke_20261010_093227`.

In the dynamic run, live-obstacle replans completed in about 20 ms and control
callbacks stayed around 0.09--0.25 ms. A normal lattice search can still take
roughly 100--160 ms, but it no longer blocks command publication. One earlier
dynamic seed produced a genuine no-path condition and was rerun successfully;
the current short fixtures therefore validate responsiveness and recovery, not
a hard real-time or universal no-failure guarantee. Longer routes, severe
occlusion and dense moving-object scenes still need broader field-style tests.
