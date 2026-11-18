# fsgp_bgk_se2

Current NEOS simulation preset selects `probability_model:=shared_gp`, contract 8,
including the explicit independent 5mm model-floor assumption. Earlier shared
models have different contract numbers; do not mix their qualification records.
one current local posterior supplies both completion and directional probability
bounds. It does not run the asynchronous marginal-model worker described below.

The NEOS preset uses `fsgp_processing_executor:=single_ingress`: one ROS ingress
executor and one compute owner retain only the latest input. Its input window is
0.10s; the original-source output deadline remains 0.30s with 0.015s reserved
after encoding. Generic/real defaults are not switched by this simulation preset.
The standalone executable freezes its initialization object graph once after
warmup, keeping normal collection of later allocations; it releases the owned
freeze on shutdown. No GC policy is changed by importing/constructing the Node.
The fine collision raster and packed witness bounds now use CUDA with an exact
CPU oracle/fallback. Original coordinates, floor vetoes, 2cm witness boxes and
outward rounding remain unchanged. This is an implementation optimization, not
a new probability model or a reduction in obstacle/vehicle size.

Map delivery checks source age AFTER complete ROS message encoding. The
specialized CDR1 encoder joins primitive buffers once and retains every field,
layout, source stamp and float bit; unknown schemas use ROS's standard encoder.
Encoding time and deadline drops are exposed in `/car_0_fsgp_bgk/runtime`.
This does not cache a GP prediction or relax a risk budget. An asynchronous
encoder prototype was rejected after physical throughput regressed; it is not
part of the runtime configuration. See the dated NEOS qualification record.
The latest measured-floor clearance, positive-occupancy/negative-terrain semantics,
CPU/CUDA checks and physical failures/passes are documented in the
[2026-10-26 qualification](../../docs/NAVIGATION_COMPLETION_QUALIFICATION_20261026.md).
Use the four matching commands in the top-level README; the historical examples
below are not a substitute for that navigation configuration.

## Probability Navigation Contract (2026-10-20)

The legacy `risk` score is still heuristic. A separate `failure_probability_upper`
layer propagates observed height moments into directional pitch/roll/residual/step
constraints, with arbitrary-correlation Gaussian feature bounds and a union bound.
Contract 2 additionally reuses marginal-height error events across overlapping
poses and search primitives, rather than charging the same source repeatedly.
The enhanced navigation launch now consumes it through local path and braking
budgets, without converting an 8-bit cost back into a probability. Unsupported
poses fail closed. GP completion needs explicit permission on both sides and is
off by default. This is a geometric model, not calibrated vehicle failure risk.

See [the contract, formulas, visualization key and validation limits](../../docs/PROBABILISTIC_NAVIGATION.md).
Small numerical/interface checks, short GUI runs and read-only RViz fixtures
have run. Unknown starts remain rejected by default. A fixed initial-stance
hypothesis is opt-in for simulation only; its shared-source errors are propagated
in contract 4, not relabeled as measured ground. Short movement is not an arrival
result. See [current qualification](../../docs/QUALIFICATION_20261020.md).
The latest small Isaac qualifications reached short forward/turning goals and
reversed along a previously observed corridor. Unobserved rear goals still
fail closed; these runs do not establish general navigation success rates.
The shared RViz has separate risk-surface and measured-elevation-surface views;
candidate dots do not become measured or navigation-approved heights.
Historical experiments below belong to earlier configurations.

## Experimental Feature SGP Prediction

An actual GPyTorch inducing-point GP now exposes posterior elevation
layers independently of the BGK local-statistic variance. It adapts the vendor
FSGP feature/RQ-kernel structure with bounded observation sampling, a fitted
plane mean, and observed-only features. It is not a bit-identical upstream
FSGP-BGK reproduction; no synthetic flat robot points or query-side PCA fit is
used. The modular launch enables background prediction for display by default;
using predictions as navigation support still needs explicit dual permission.
Latent variance and noisy-observation variance are separate, following
the [GPyTorch posterior contract](https://docs.gpytorch.ai/en/stable/examples/01_Exact_GPs/Simple_GP_Regression.html).

The 2026-10-13 revision uses deterministic XY-stratified measurement selection
at all three budgets (1024 input, 384 training, 96 inducing points), rather
than point-file ordering. `gp_latent_var` is now a **conservative augmented
surface variance**, not just the conditional inducing-GP variance: it includes
fitted-plane coefficient covariance and distance-weighted unresolved local
residual variation, with a sensor-scale floor. `gp_observation_var` adds the
fitted measurement-noise variance. Nominal intervals still require validation;
this correction does not certify a 95% coverage guarantee or safe unknown space.

The collision overlay can use the **unrefined observed** ground height to
corroborate sparse upper returns in adjacent cells. It never accepts GP hidden
heights as measured ground; invalid cells are explicitly NaN. High canopy,
isolated noise, rolling-grid edge wrap-around and CPU/CuPy parity have regression
tests. This is a current-scan occupancy branch, not temporal dynamic-object
tracking or FAST-LIO map deletion.

```bash
scripts/sim/start_terrain_analysis.sh terrain_backend:=gpu fsgp_gpu_backend:=cupy \
  prediction_backend:=feature_sgp gp_device:=cuda
```

Output layers include `gp_mean`, `gp_latent_var`, `gp_observation_var`,
`gp_observed`, `gp_supported`, and `gp_accepted`. `gp_accepted` is a configured
support/standard-deviation gate, not evidence of calibrated safety. Unknown
cells retain their conservative risk and do not become observed/free cells.
GP fitting is capped at one update per second by default; the current model
may therefore use older, causal observations. CPU is available through
`gp_device:=cpu`, while explicit CUDA requests fail clearly if unavailable.
The worker has one in-flight job and no queue. Exact integer-cell window shifts
reindex cached predictions without interpolation; newly entered strips stay
unknown. Source timestamps remain unchanged, and display snapshots expire after
2 seconds. Navigation has a separate, stricter source-age gate (0.25 seconds).
CUDA prediction uses its own stream; CPU remains available.

The simulation-only start qualification requires both explicit switches in
separate terrain/planning terminals, after simulation and mapping are running:

```bash
scripts/sim/start_terrain_analysis.sh terrain_backend:=gpu fsgp_gpu_backend:=cupy startup_support_mode:=stable_stance
scripts/sim/start_planning_control.sh chance_allow_startup_prior:=true
```

The assumption is frozen at the initial footprint, revocable by observations,
and speed-limited nearby. It does not authorize unknown forward ground. The
planner rejects this contract without simulated time and explicit permission.
See [the model and limitations](../../docs/PROBABILISTIC_NAVIGATION.md#51-显式仿真初始支撑假设).

For experimental blending of the GP into **observed** elevation only, add
`gp_refine_observed:=true`. Baseline hazards and current-scan collision overrides
remain enforced. This is off by default until held-out uncertainty and boundary
accuracy are validated. Do not enable unknown-space driving based on these
posterior layers alone.

The 2026-10-15 support correction rejects every candidate outside the convex
hull of measured XY, including nearby diagonal extrapolation. Local neighbors
must enclose the query angularly, not just bracket its X and Y separately.
`gp_supported` now requires both this geometric support and the distance gate.
The raw `gp_mean` remains finite for diagnostics outside support; it is not a
certified completion. Budgets, kernel scales and uncertainty thresholds were
not relaxed, and unknown completion remains off by default.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 scripts/sim/experiments/prediction_holdout.py --scene-count 10 \
  --device cuda --output log/prediction_holdout_20261012
```

This benchmark tests five independently parameterized terrain families and
four observation masks. It measures hidden-cell errors explicitly, with
truth-only scoring and separate coverage/calibration. It is an analytic
controlled experiment, not Isaac LiDAR or real-world accuracy proof.

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

The yaw sweep fits its support plane at the **actual rounded grid samples**
in the vehicle frame, not at ideal offsets. Raw observed height discontinuities
are retained separately before BGK smoothing, so removing grid-induced false
steps does not remove genuinely hazardous steps. CPU and CUDA share this
contract and have parity tests. Risk scores remain heuristic, not calibrated
event probabilities.
When GP refinement is enabled, discontinuity detection still receives the
pre-refinement observed elevation through a separate input; the GP cannot
replace that raw height evidence.

The upstream terrain analyzer's `risk` is already a yaw-dependent **pose**
cost. The ROS node sets `baseline_is_pose_risk=true`: preserve the prior at
the same position/yaw instead of sweeping a second footprint over it.
Standalone estimators default to `false` for point-occupancy inputs.
The enhanced navigation launcher uses `collision_risk` for its 2D hard
obstacle layer and separately checks pose risk at the vehicle center.
CPU/CuPy tests cover both contracts; they are not interchangeable.

`gp_complete_supported` remains an opt-in experiment, not recommended for
deployment. A 4,000-condition paired regression reduced false rejection but
increased aggregate false-safe classifications (1.93% to 2.14% of unsafe
reference states). Low posterior variance is not free-space evidence.
`gp_observed` and `gp_completed` distinguish measurement from prediction;
the controller's ground-memory cache must not use predictions as proof.
Full results and version boundaries are documented in
`docs/reports/terrain_control_expansion_20261013_v2/`.

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
