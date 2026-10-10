# Dynamic Map and Navigation Experiments

This document defines the repeatable experiment protocol for the Isaac Hunter
stack. The experiment driver never changes the production defaults unless an
ablation is explicitly selected. Every run keeps the original stage logs and
stores a machine-readable row in `manifest.json`.

## Metrics

The primary metrics are:

* **navigation**: pass/fail, final truth error, minimum synchronized vehicle /
  obstacle separation, front-stop count and maximum lateral bypass;
* **dynamic clearing**: time until high points disappear from `/Laser_map`,
  time until the terrain/FSGP risk falls below 0.2, and time until the planner
  live layer is clear;
* **real-time behavior**: LiDAR and IMU wall/stamp rates, terrain/FSGP fresh
  rates, risk-map age, planner p95 latency and control-loop p95 latency;
* **mapping quality guards**: pre-vacate obstacle peak, post-vacate residual,
  static guard retention and LiDAR points per scan.

Clearance time is measured from the simulator's obstacle-vacated timestamp,
not from wall-clock process start. A missing clearance observation is recorded
as `null`, never silently converted to zero.

## Current experiment matrix

| Group | Scene cases | Purpose |
|---|---|---|
| `full` | `mars`, `detour`, `ramp`, `step`, `crossing`, `emerge`, `clearance`, `clearance_guard` | Current integrated GPU + CuPy + dynamic-clearing + live-layer system |
| `no_fastlio_dynamic_clear` | `clearance`, `clearance_guard` | Measures the contribution of FAST-LIO2 historical voxel clearing |
| `no_terrain_dynamic_clear` | `clearance`, `clearance_guard` | Measures stale elevation/risk removal without the CUDA free-ray cleaner |
| `no_live_layer` | `crossing`, `emerge`, `detour` | Separates map-based planning from the current-cloud emergency layer |
| `cpu` | `mars`, `detour`, `clearance` | GPU/CPU throughput and latency comparison |

Run a first repeated baseline study:

```bash
python3 scripts/sim/experiments/run_matrix.py \
  --case mars:2 --case detour:2 --case crossing:2 \
  --case clearance:3 --case step:1 --case ramp:1 \
  --ablation full --first-domain 60
```

Then aggregate and draw the result figure:

```bash
python3 scripts/sim/experiments/analyze_matrix.py \
  log/experiment_matrix_YYYYMMDD_HHMMSS
```

The output directory contains `manifest.json`, one log per run,
`results.csv`, `statistics.json`, and `experiment_results.png`.

## First own-method baseline record

The first smoke matrix was run on 2026-10-10 with **only the current full
method** enabled:

`FAST-LIO2 + GPU terrain analyzer + CuPy FSGP-BGK/SEB + terrain_nav`

It is a data-collection run, not a comparison or ablation study. The matrix
used six runs: two `clearance`, and one each of `crossing`, `detour`, `step`
and `mars`. The raw directory is
`log/experiment_matrix_20261010_112130/`.

| Case | Runs | Pipeline gate | Main observation |
|---|---:|---:|---|
| `clearance` | 2 | 0/2 | Physical minimum separation was 2.157--2.226 m; terrain risk cleared in 0.27--0.29 s and the live layer in 0.17--0.19 s, but historical `/Laser_map` points did not clear within the observation window. |
| `crossing` | 1 | 1/1 | Minimum separation 2.671 m; terrain fresh rate 10.06 Hz; FSGP fresh rate 5.08 Hz; planner p95 169.6 ms. |
| `detour` | 1 | 0/1 | The vehicle did not reach the goal; final truth error was 5.691 m. This is a real planning/navigation defect to fix, not a successful result. |
| `step` | 1 | 1/1 | The safety gate correctly stopped before the step; planner p95 was 297.9 ms. The 89.96 ms control p95 is an outlier requiring a repeat. |
| `mars` | 1 | 1/1 | Sensor/terrain pipeline only; terrain fresh rate 10.14 Hz and FSGP fresh rate 5.06 Hz. |

The aggregate gate count is therefore 3/6, but it must not be interpreted as
an algorithm comparison: the sample is too small and all rows use the same
implementation. The clearance result also separates two products: immediate
risk/live clearing was fast and physically safe, while cumulative
`/Laser_map` retained conservative historical points. The figure and machine
readable outputs are:

* `log/experiment_matrix_20261010_112130/manifest.json`
* `log/experiment_matrix_20261010_112130/results.csv`
* `log/experiment_matrix_20261010_112130/statistics.json`
* `log/experiment_matrix_20261010_112130/experiment_results.png`

Before any paper-quality claim, repeat the full-method matrix with at least
10 paired seeds per scenario, fix the `detour` failure and investigate the
`step` control-latency outlier. Only after that should the same recorded bags
be replayed through external baselines.

## Internal ablation protocol (not an external comparison)

The following are **not competing methods**. They are diagnostic ablations of
our own implementation, using identical scene seeds and ROS domains. Their
purpose is to attribute an observed change to one module; they must not be
reported as an algorithm-vs-algorithm comparison:

1. **Current full chain**: FAST-LIO2 dynamic clearing + GPU terrain dynamic
   clearing + FSGP-BGK/SEB collision layer + planner live layer.
2. **Mapping ablation**: disable FAST-LIO2 dynamic clearing while keeping the
   rest of the chain unchanged.
3. **Terrain ablation**: disable CUDA terrain free-ray clearing while keeping
   FAST-LIO2 and the live layer enabled.
4. **Reactive-layer ablation**: disable `terrain_nav` live obstacles to expose
   how much immediate safety depends on the current registered cloud.
5. **Backend comparison**: GPU terrain + CuPy/FSGP versus CPU terrain + CPU
   FSGP. Compare rates and latency separately from navigation quality.

For module-attribution claims, use at least 10 independent runs per cell and report median,
IQR, p95 and pass-rate confidence intervals. Use paired seeds when comparing
ablations; do not compare a difficult crossing run against an easy empty run.
The `clearance_guard` case is important because it tests that ghost removal
does not erase a nearby static obstacle. `step` and `ramp` check that dynamic
cleaning does not weaken geometric terrain rejection.

## External method comparison

Only after our full-method baseline and internal diagnostics are stable should
we compare against other methods. The primary external baselines are:

* **legacy SEB-Naver stack**: the archived original project under
  `../seb_old/seb_8_18`, rebuilt in its own workspace and ROS domain;
* **upstream FSGP-BGK-ROS2**: the official ROS2 implementation under
  `src/FSGP-BGK-ROS2` (or its clean external checkout), without this project's
  SE2/yaw-aware and live-obstacle additions;
* **original terrain-only pipeline**: the legacy terrain analyzer and planner
  without FSGP-BGK/SEB, used as a controlled system-level baseline;
* **published non-directional map baseline**: a 2-D traversability map whose
  score is identical for all vehicle headings, used to test the specific value
  of SEB's heading dependence.

The fair-comparison unit is a fixed recorded run, not a separately generated
simulation run. Record `/livox/lidar`, `/imu/data_raw`, `/Odometry`, Isaac
ground truth, obstacle pose, goals and commands with `rosbag2`; replay the
same bag to every terrain method. For mapping comparisons, either use the same
FAST-LIO2 pose/cloud output for all terrain methods or compare each mapping
method against the same ground-truth trajectory after SE(3) alignment. Do not
mix a better pose estimate with a better terrain method in one score.

The comparison table must include success rate, collision or minimum
separation, path length, final goal error, replan count, p95 replanning
latency, control p95, terrain/FSGP update rate, dynamic-risk clear delay,
historical-map residual points and sensor real-time factor. A successful
navigation trial alone is not sufficient evidence that a dynamic residual was
removed. Use at least 10 paired seeds per scenario, report median/IQR/p95 and
95% confidence intervals, and use paired bootstrap or Wilcoxon signed-rank
tests for latency, residual points and path length.

The current `experiment_matrix_*` directory is deliberately labeled
`study_type=our_method_baseline`; it is not an external comparison result.

## Known interpretation limits

The current `/Laser_map` cleaner requires a farther return and repeated free
rays. If the old obstacle volume is never observed after it moves, retaining
the historical point is conservative and expected. The live obstacle layer and
terrain/FSGP risk are therefore the immediate navigation products; cumulative
`/Laser_map` remains a visualization and mapping-history product. Final
reports must disclose this distinction instead of calling every historical
point a tracking failure.
