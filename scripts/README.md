# Runtime Scripts

The supported runtime entry points are split by platform and by pipeline
boundary. Run each stage in its own terminal when debugging.

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
adapter does not run the upstream sparse Gaussian process. The planning script
starts only `terrain_nav` and publishes `/cmd_vel` from `/goal_pose`.

For manual control, run the first three stages but use
`start_keyboard_control.sh` instead of `start_planning_control.sh`. Do not run
both command publishers together.

Checks live under `scripts/sim/checks/`. FAST-LIVO2 remains an optional branch
under `scripts/sim/optional/`.

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
