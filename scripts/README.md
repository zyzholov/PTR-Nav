# Runtime Scripts

2026-11-16: simulation planning now defaults to `dynamic_observed_only: true`
in its YAML, following the requested scope of currently measured moving targets.
Use `scripts/sim/start_planning_control.sh dynamic_observed_only:=true` to
select it explicitly, or `dynamic_observed_only:=false` for retained-conflict
handling. Missing observations no longer latch dynamic waiting in this mode;
raw/held occupied geometry, unknown terrain and input-age checks still apply.
This is observation-limited avoidance, with no blind-zone collision guarantee.
The visible-target crossing qualification still fails; this change does not
establish usable multi-object navigation. Real-vehicle launchers are unchanged.
See the [scope, commands and retained test results](../docs/OBSERVED_DYNAMIC_SCOPE_20261116.md).

2026-11-05 NEOS: the four module launchers now accept the explicit
`navigation_profile:=neos_joint_sim` preset. See the [four-terminal commands,
model assumptions and current qualification limits](../docs/NEOS_SIM_STARTUP.md).
This preset is simulation-only; the real launchers and generic defaults are
unchanged. Earlier Hunter/Ouster results below do not qualify the new vehicle.

Historical directed work on 2026-10-27: the default became **MID360 tilted 25deg**,
advancing through a finite 40-phase scan table. See the [MID360 fixes and actual
qualifications](../docs/MID360_TILT25_QUALIFICATION_20261027.md). The earlier
[level Ouster qualification](../docs/NAVIGATION_COMPLETION_QUALIFICATION_20261026.md)
does not qualify this new sensor. Explicit simulation startup/extrapolation
options below preserve covariance, contradictions and whole-body chance checks;
generic and real defaults remain unknown. RoboSense 32 is not implemented yet.
The local FreeDOM auxiliary filter is selected explicitly with
`scripts/sim/start_mapping.sh rviz:=false dynamic_filter_backend:=freedom`.
For the historical 2026-10-27 shared-GP simulation setup, keep historical obstacle memory
`obstacle_memory_topic:=/Laser_map obstacle_memory_role:=advisory`; current
returns remain hard collision evidence. See the [dynamic mapping and navigation
qualification](../docs/DYNAMIC_MAPPING_NAV_QUALIFICATION_20261027.md) for the
pinned sources, local fixes, repeat results and retained failures.

The supported runtime entry points are split by platform and by pipeline
boundary. Run each stage in its own terminal when debugging.

Simulation launchers and qualification observers select the same Fast DDS
asynchronous transport and `scripts/sim/config/fastdds_large_maps.xml` profile.
An existing `FASTRTPS_DEFAULT_PROFILES_FILE` takes precedence;
`HUNTER_ISAAC_DDS_PROFILE=builtin` disables only project auto-selection.
Use the same RMW and ROS domain in manual CLI terminals too. See the root
README for the environment commands and current four-stage startup sequence.

2026-10-26: physical checks resumed after the collision-geometry correction.
The final identical revision passed seven of eight directed checks, including
one deliberate step rejection. The remaining run arrived safely but failed
the map freshness criterion. This is not a navigation success-rate estimate;
planning tails and sporadic latency remain unresolved. See the
[runtime qualification](../docs/NAVIGATION_RUNTIME_QUALIFICATION_20261026.md) and
[geometry contract](../docs/COLLISION_GEOMETRY_CONTRACT_20261026.md).
Simulation uses the full oriented body plus one 0.10 m margin, without a
second default hard obstacle reserve. The simulation terrain wrapper enables
the measured-floor height slice; generic/real launch leaves it off. Vehicle
height and headroom metadata must match the planner. The read-only
`checks/audit_collision_geometry.py` replays a frozen map/scan and can plot
the exact removed returns; this is not a physical navigation test.

For the 2026-10-24 conditional-terrain, thin-obstacle, bounded-recovery and
GPU simulation freshness changes, see the [refinement record](../docs/TERRAIN_NAV_REFINEMENT_20261024.md).
The GPU simulation wrapper now uses 0.16 s input / 0.30 s output source-age
deadlines and reports discarded results on `/car_0_fsgp_bgk/runtime`.
Generic launch and CPU fallback leave these extra deadlines disabled by default.
Discarded results are not updates; inspect rate, gaps and held-map age as well.
Planner stale-input stopping remains enabled. Limited checks are not a hard
real-time or general navigation success-rate guarantee.

For the 2026-10-23 nonflat fixes, ramp/crossing fixtures, exact commands and
both passing and failing attempts, see the [nonflat qualification record](../docs/NONFLAT_NAVIGATION_20261023.md).
The steeper downhill off-axis task now has a same-version arrival record, but
takes about 45 seconds. Limited passes are not evidence of general-purpose
safe navigation; the older failed attempt remains archived.

2026-10-22 qualification: explicit shared-GP simulation has static detour,
dynamic crossing, uphill and short Mars forward/reverse arrival records.
These directed checks do not establish a deployment success rate.
Unknown support still blocks startup by default.
An explicitly enabled, fixed initial-stance assumption is available only for
simulation qualification, not real-vehicle use or measured terrain. See
[qualification evidence](../docs/QUALIFICATION_20261020.md). Small checks are
permitted, but batch experiments still require protocol review. The default
tracker is hybrid Pure Pursuit/Stanley, not standard MPPI.

## Isaac Simulation

```text
scripts/sim/start_simulation.sh
  -> scripts/sim/start_mapping.sh
  -> scripts/sim/start_terrain_analysis.sh
  -> scripts/sim/start_planning_control.sh
```

The first script publishes the simulated sensors and `/clock`. The mapping
script starts FAST-LIO2 and publishes `/Odometry`, `/cloud_registered` and the
cumulative `/Laser_map`. The terrain script starts the CUDA/CPU terrain
front-end and the FSGP-BGK/SEB node, publishing both the baseline and enhanced
`SE2Grid` maps. Set `enhance:=false` to disable the latter; then
`scripts/sim/optional/start_fsgp_bgk.sh` can start it separately. The current
adapter uses a bounded GPyTorch feature-SGP branch, not a bit-identical upstream
FSGP-BGK reproduction. The simulation terrain script now defaults to current-scan
local GP completion; background candidates alone are not navigation support.
Use `gp_complete_supported:=false` for observed-only operation, or
`probability_use_gp_completion:=false` for display-only completion. Real-vehicle
and generic-node defaults remain off. The planning script
starts only `terrain_nav` and publishes `/cmd_vel` from `/goal_pose`.

The simulator now uses Isaac Fabric for physical-transform output. Contacts,
mass/inertia, CCD, 200 Hz physics and LiDAR point density are unchanged; this
avoids USD transformation notices on every physics substep. Use
`--physics-transform-output usd` only for compatibility/debugging. This is
not a switch to GPU contact simulation.

The simulation launchers now default Fast DDS publication to `ASYNCHRONOUS`.
Override with `HUNTER_ISAAC_DDS_PUBLICATION_MODE=SYNCHRONOUS` or `AUTO` for
diagnosis. Real-vehicle launchers do not set this policy. This transport setting
is not a hard real-time guarantee and does not change point density or risk limits.

For manual control, run the first three stages but use
`start_keyboard_control.sh` instead of `start_planning_control.sh`. Do not run
both command publishers together.

Checks live under `scripts/sim/checks/`. FAST-LIVO2 remains an optional branch
under `scripts/sim/optional/`.

Controlled courses now include `staggered`, `corridor`, and `ramp_detour`.
Select one using `start_simulation.sh` with `--scene-profile light`,
`--map-asset none`, and `--nav-test-scenario corridor`;
the other three module launchers remain separate.
For isolated automatic qualification, use `checks/pipeline_smoke.py` with the
explicit shared-model options in the [course protocol](../docs/NAVIGATION_SCENARIO_EXPANSION.md).
These fixtures are not ground-truth inputs to navigation. Batch execution still
requires protocol review; `experiments/run_matrix.py --dry-run` only previews jobs.

Shared RViz includes the read-only `ptr_navigation_rviz/StatusPanel` fixed
readout and color legend. Build it with the terrain package. The panel displays
unavailable/stale values explicitly; it publishes no commands or goals.
Select `Measured + predicted height` for the actual elevation mesh and GP
candidate dots, or `Risk navigation` for the fixed probability color scale.
Mint squares are current navigation-qualified GP heights, not global candidates;
amber squares are the explicit simulation-only initial-stance assumption.
The accumulated LiDAR view colors actual `/Laser_map` points by height, through
the read-only `/terrain_display/lidar_map` presentation topic. Live returns
use `/cloud_registered_visual` (full measured scan) and
`/terrain_display/lidar_returns` (range color), not the downsampled navigation
scan. Both preserve XYZ, point count, frame and source timestamp. Closing the
live display releases its own dense-scan subscription. The explicit shared-GP
simulation also consumes the full scan for measured clearance references,
so that independent subscriber may keep publication active. The FAST-LIO startup ROS
parameter `publish.visual_scan_en:=false` disables this optional publication.
Color is not reflectivity.
Risk mode uses faint one-pixel live returns so they do not obscure the surface;
the LiDAR mode emphasizes them. None of these settings changes registration,
terrain input density, collision geometry or risk budgets.

The risk surface uses measured height first, then **navigation-qualified** GP
height where available. The actual native mesh spacing is 0.2 m; interpolated
vertex colors are a presentation choice, not finer terrain measurements. Gold
cell outlines show raw measured occupancy, not the larger forbidden region for
the vehicle's center. Unsupported holes remain unknown. See the
[display and navigation qualification](../docs/NAVIGATION_DISPLAY_QUALIFICATION_20261026.md).
The default orbit follows the FAST-LIO `base_link` position so a moving vehicle
does not leave the viewport; the saved `Terrain Overview` remains world-fixed.
`rviz_recorded_probe.py` is a labeled stationary UI fixture, not a navigation
test. Never interpret its re-timestamped map age as a performance result.

Simulation-only qualification (after simulation + mapping in other terminals):

```bash
scripts/sim/start_terrain_analysis.sh terrain_backend:=gpu fsgp_gpu_backend:=cupy \
  prediction_backend:=feature_sgp gp_device:=cuda \
  probability_model:=shared_gp gp_complete_supported:=true \
  probability_gp_max_distance_m:=1.25 probability_gp_max_extrapolation_m:=0.75 \
  startup_support_mode:=stable_stance startup_support_directional:=true
scripts/sim/start_planning_control.sh chance_allow_gp_completion:=true \
  chance_allow_joint_model:=true chance_allow_startup_prior:=true reverse_recovery_enable:=true
```

Optional simulation route guidance, qualified in repeated three-pillar checks:
add `obstacle_memory_topic:=/Laser_map obstacle_memory_role:=advisory` to the
terrain command above. This is a **soft route preference only**. It neither
supplies ground support nor adds historical ghosts as hard occupancy. The
default remains `none`; `hard` history is not a qualified substitute. Current
sensor occupancy and probability checks still veto an unsafe route. A small
number of directed checks is not general navigation qualification.

The shared model uses one local posterior for height display and navigation.
Only this simulated configuration enables lower-ground preservation in the
CUDA frontend. Its raw returns still veto new obstacles. Ground clearing needs
three distinct measured scans; predictions and repeated timestamps do not clear.

The hypothesis stays within its initial footprint; it is not a ground plate,
does not move with the vehicle, and does not authorize unseen rear ground.
`pipeline_smoke.py --return-to-start` checks a forward goal then reverse to
the recorded starting position. `--fault-stale-map` drops only delivery through
a qualification relay; it must not suspend CUDA processes. Neither check is a
batch experiment or evidence of calibrated real-vehicle risk.

`pipeline_smoke.py` defaults to sensor wall-clock throughput of at least 95%
of nominal. Its full-motion observer also checks the external IMU, LiDAR,
fresh map rate and source age (GPU: at least 5 Hz and at most 0.35 s).
These bounded debug checks are not formal success-rate experiments or proof
of hard real-time deadlines.

## Real Vehicle

```text
scripts/real/start_mapping.sh
  -> scripts/real/start_terrain_analysis.sh
  -> scripts/real/start_fsgp_bgk.sh
  -> scripts/real/start_planning_control.sh
```

`record_bag.sh` records real sensor data with selectable topic profiles.
Real planning defaults to the original terrain map and `flat_2d` risk query.
For an *offline or supervised* enhanced-map trial, start the FSGP stage and
pass `fused_map_topic:=/car_0_fsgp_bgk/fused_map`, `risk_query_mode:=se2_yaw`
and `vehicle_driver:=none` to `scripts/real/start_planning_control.sh`. Do not
enable the vehicle driver until this branch has been validated on recorded
data and with the chassis disabled. These scripts no longer kill existing real nodes by default;
`CLEAN_STALE_REAL_MAPPING=1` and `CLEAN_STALE_REAL_NAV=1` are explicit opt-ins.

## Archived

Historical one-shot launchers, Gazebo-only launchers and experiment scripts
are under `scripts/old/`. They are retained for reference and are not the
supported startup path.
