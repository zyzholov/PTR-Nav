import math
import warnings

import numpy as np
import pytest

from fsgp_bgk_se2.bgk_seb import BgkSebEstimator, BgkSebParams


def make_estimator(yaw_bins=36):
    return BgkSebEstimator(
        BgkSebParams(
            resolution=0.2,
            yaw_bins=yaw_bins,
            prefer_gpu=False,
            footprint_front=0.42,
            footprint_back=0.42,
            footprint_half_width=0.313,
            footprint_margin=0.10,
            min_support_ratio=0.55,
        )
    )


def test_flat_map_is_finite_and_low_risk():
    shape = (48, 48)
    elevation = np.zeros(shape, dtype=np.float32)
    variance = np.full(shape, 0.001, dtype=np.float32)
    valid = np.ones(shape, dtype=bool)
    baseline = [np.zeros(shape, dtype=np.float32) for _ in range(36)]

    result = make_estimator().compute(elevation, variance, valid, baseline)

    risk = np.stack(result.final_risk_slices)
    assert np.isfinite(risk).all()
    assert float(np.max(risk[:, 8:-8, 8:-8])) < 0.10


def test_quadratic_surface_is_not_an_observed_edge_step():
    estimator=make_estimator();xy=np.stack(np.meshgrid(np.arange(48)*.2,np.arange(48)*.2),axis=-1)
    z=(.012*xy[:,:,0]**2+.004*xy[:,:,1]**2).astype(np.float32)
    valid=np.ones_like(z,np.float32)
    risk=estimator._observed_step_risk(z,z,valid)
    assert risk[4:-4,4:-4].max()<.02


def test_pre_limit_geometry_guard_is_not_weakened_by_weighted_fusion():
    estimator=make_estimator();x=np.arange(48,dtype=np.float32)[None,:]*.2
    z=np.broadcast_to(np.tan(.54)*x,(48,48)).copy()
    result=estimator.compute(z,np.full_like(z,.0004),np.ones_like(z,bool),[np.zeros_like(z)])
    assert result.final_risk_slices[0][24,24]==1.


def test_invalid_geometry_guard_is_rejected():
    with pytest.raises(ValueError,match="geometry guard"):
        BgkSebEstimator(BgkSebParams(resolution=.2,yaw_bins=36,geometry_guard_risk=0))


def test_observed_center_with_insufficient_vehicle_support_stays_unknown_risk():
    estimator=make_estimator();shape=(40,40);valid=np.zeros(shape,bool);valid[20,20]=True
    result=estimator.compute(np.zeros(shape,np.float32),np.full(shape,.0004,np.float32),valid,
        [np.zeros(shape,np.float32)])
    assert all(layer[20,20]>=np.float32(estimator.params.unknown_risk) for layer in result.final_risk_slices)


def test_directional_ridge_changes_seb_risk_with_yaw():
    shape = (64, 64)
    elevation = np.zeros(shape, dtype=np.float32)
    elevation[:, 32:] = 0.16
    variance = np.full(shape, 0.001, dtype=np.float32)
    valid = np.ones(shape, dtype=bool)
    baseline = [np.zeros(shape, dtype=np.float32) for _ in range(36)]

    result = make_estimator().compute(elevation, variance, valid, baseline)
    risk = np.stack(result.final_risk_slices)

    forward_bin = 0
    sideways_bin = int(round((math.pi / 2.0) / (2.0 * math.pi) * 36))
    # At the step center both footprints cross the same edge. Directionality
    # is visible at first contact because length and width are different.
    band = (slice(27, 37), slice(27, 38))
    delta = np.abs(risk[forward_bin][band] - risk[sideways_bin][band])
    assert float(np.max(delta)) > 0.05


def test_unknown_cells_keep_unknown_risk():
    shape = (40, 40)
    elevation = np.zeros(shape, dtype=np.float32)
    variance = np.full(shape, 0.001, dtype=np.float32)
    valid = np.ones(shape, dtype=bool)
    valid[:8, :] = False
    baseline = [np.zeros(shape, dtype=np.float32) for _ in range(36)]

    estimator = make_estimator()
    result = estimator.compute(elevation, variance, valid, baseline)
    risk = np.stack(result.final_risk_slices)

    assert np.allclose(risk[:, :8, :], estimator.params.unknown_risk)


def test_enhancement_never_reduces_known_baseline_hazard():
    shape = (48, 48)
    elevation = np.zeros(shape, dtype=np.float32)
    variance = np.full(shape, 0.001, dtype=np.float32)
    valid = np.ones(shape, dtype=bool)
    baseline = [np.zeros(shape, dtype=np.float32) for _ in range(36)]
    baseline[9][24, 24] = 0.92

    result = make_estimator().compute(elevation, variance, valid, baseline)
    risk = np.stack(result.final_risk_slices)

    assert risk[9, 24, 24] >= 0.92
    assert risk[0, 24, 24] < 0.10


@pytest.mark.parametrize("gpu", [False, True])
def test_pose_prior_is_not_inflated_twice_and_keeps_yaw(gpu):
    shape = (40, 40)
    z = np.zeros(shape, np.float32)
    prior = [z.copy() for _ in range(36)]
    prior[9][20, 20] = 1.0
    result = BgkSebEstimator(BgkSebParams(
        resolution=.2, yaw_bins=36, prefer_gpu=gpu, gpu_backend="cupy",
        baseline_is_pose_risk=True)).compute(z, z + .0004, np.ones(shape, bool), prior)
    if gpu:
        assert result.backend == "cupy"
    assert result.final_risk_slices[9][20, 20] == 1.0
    assert result.final_risk_slices[9][20, 21] < .1
    assert result.final_risk_slices[0][20, 20] < .1
    point_result = BgkSebEstimator(BgkSebParams(
        resolution=.2, yaw_bins=36, prefer_gpu=gpu, gpu_backend="cupy",
        baseline_is_pose_risk=False)).compute(z, z + .0004, np.ones(shape, bool), prior)
    assert point_result.final_risk_slices[9][20, 21] == 1.0


def test_ramp_remains_passable_and_step_costs_more_than_flat():
    shape = (64, 64)
    variance = np.full(shape, 0.001, dtype=np.float32)
    valid = np.ones(shape, dtype=bool)
    baseline = [np.zeros(shape, dtype=np.float32) for _ in range(36)]
    estimator = make_estimator()
    flat = np.zeros(shape, dtype=np.float32)
    ramp = np.broadcast_to(
        (np.arange(shape[1], dtype=np.float32) * 0.2 * 0.10)[None, :], shape
    ).copy()
    step = flat.copy()
    step[:, 32:] = 0.30

    flat_risk = np.stack(estimator.compute(flat, variance, valid, baseline).final_risk_slices)
    ramp_risk = np.stack(estimator.compute(ramp, variance, valid, baseline).final_risk_slices)
    step_risk = np.stack(estimator.compute(step, variance, valid, baseline).final_risk_slices)

    assert flat_risk[:, 24:40, 24:40].max() < 0.10
    assert ramp_risk[:, 24:40, 24:40].max() < 0.58
    assert step_risk[:, 28:36, 30:34].max() > ramp_risk[:, 28:36, 30:34].max()
    assert step_risk[:, 28:36, 30:34].max() >= 0.58


def test_steep_but_drivable_ramp_is_not_misclassified_as_step():
    shape = (64, 64)
    x = np.arange(shape[1], dtype=np.float32)[None, :]
    ramp = np.broadcast_to(x * 0.2 * 0.17, shape).copy()
    step = ramp.copy()
    step[:, 32:] += 0.35
    variance = np.full(shape, 0.001, dtype=np.float32)
    valid = np.ones(shape, dtype=bool)
    baseline = [np.zeros(shape, dtype=np.float32) for _ in range(36)]
    estimator = make_estimator()

    ramp_risk = np.stack(estimator.compute(ramp, variance, valid, baseline).final_risk_slices)
    step_risk = np.stack(estimator.compute(step, variance, valid, baseline).final_risk_slices)

    assert ramp_risk[:, 24:40, 24:40].max() < 0.58
    assert step_risk[:, 28:36, 30:34].max() >= 0.58


def test_excessive_slope_remains_a_hard_hazard_after_fusion():
    shape = (64, 64)
    x = np.arange(shape[1], dtype=np.float32)[None, :]
    steep = np.broadcast_to(x * 0.2 * math.tan(0.60), shape).copy()
    baseline = [np.zeros(shape, dtype=np.float32) for _ in range(36)]
    result = make_estimator().compute(
        steep,
        np.full(shape, 0.001, dtype=np.float32),
        np.ones(shape, dtype=bool),
        baseline,
    )
    risk = np.stack(result.final_risk_slices)

    assert risk[:, 20:44, 20:44].max() >= 0.58


@pytest.mark.parametrize("direction", [0.0, 0.37, 0.91])
def test_all_yaws_on_exact_plane_match_physical_pitch_and_roll(direction):
    estimator = make_estimator()
    estimator.params.store_diagnostic_slices = True
    p = estimator.params
    x, y = np.meshgrid(np.arange(48) * p.resolution, np.arange(48) * p.resolution)
    gradient = math.tan(0.35)
    elevation = gradient * (x * math.cos(direction) + y * math.sin(direction))
    result = estimator.compute(elevation.astype(np.float32), np.full(x.shape, 0.001, dtype=np.float32),
                               np.ones(x.shape, dtype=bool), [np.zeros(x.shape, dtype=np.float32)])
    yaw = (np.arange(p.yaw_bins) + 0.5) * 2 * math.pi / p.yaw_bins
    expected = np.maximum(np.clip((np.arctan(np.abs(gradient * np.cos(yaw - direction))) - p.pitch_free_rad) /
                                 (p.max_pitch_rad - p.pitch_free_rad), 0, 1),
                          np.clip((np.arctan(np.abs(gradient * np.sin(yaw - direction))) - p.roll_free_rad) /
                                 (p.max_roll_rad - p.roll_free_rad), 0, 1))
    risk = np.stack(result.seb_risk_slices)[:, 16:32, 16:32]
    np.testing.assert_allclose(risk, np.broadcast_to(expected[:, None, None], risk.shape), atol=3e-5)
    assert np.max(np.stack(result.seb_step_risk_slices)[:, 16:32, 16:32]) < 1e-5


def test_refinement_cannot_erase_a_raw_observed_step():
    estimator = make_estimator()
    shape = (48, 48)
    refined = np.zeros(shape, dtype=np.float32)
    observed = refined.copy()
    observed[:, 24:] = 0.30
    result = estimator.compute(refined, np.full(shape, 0.001, dtype=np.float32),
        np.ones(shape, dtype=bool), [np.zeros(shape, dtype=np.float32)],
        observed_elevation=observed)
    assert np.stack(result.final_risk_slices)[:, 20:28, 22:26].min() >= 0.999


def test_raw_observed_step_rejects_a_mismatched_grid():
    estimator = make_estimator()
    flat = np.zeros((24, 24), dtype=np.float32)
    with pytest.raises(ValueError, match="share one grid"):
        estimator.compute(flat, flat, np.ones_like(flat, dtype=bool), [flat],
                          observed_elevation=np.zeros((24, 23), dtype=np.float32))


def test_auto_backend_keeps_small_map_on_cpu_without_importing_cupy():
    shape = (24, 24)
    params = BgkSebParams(
        resolution=0.2,
        yaw_bins=12,
        prefer_gpu=True,
        gpu_backend="auto",
        auto_gpu_min_work_items=100_000,
    )
    estimator = BgkSebEstimator(params)

    result = estimator.compute(
        np.zeros(shape, dtype=np.float32),
        np.full(shape, 0.001, dtype=np.float32),
        np.ones(shape, dtype=bool),
        baseline_risk_slices=[
            np.zeros(shape, dtype=np.float32) for _ in range(params.yaw_bins)
        ],
    )

    assert result.backend == "cpu"
    assert estimator._cupy is None
    assert "auto selected CPU" in result.backend_reason


@pytest.mark.parametrize("observed_geometry", [False, True])
def test_cuda_backend_matches_cpu_when_available(observed_geometry):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="CuPy may not function correctly because multiple CuPy packages",
        )
        cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device is available")
    except Exception as exc:
        pytest.skip(f"CUDA runtime is unavailable: {type(exc).__name__}")

    rng = np.random.default_rng(20260911)
    shape = (52, 48)
    yaw_bins = 18
    elevation = rng.normal(0.0, 0.12, shape).astype(np.float32)
    elevation[20:31, 23:] += np.float32(0.17)
    variance = rng.uniform(0.001, 0.08, shape).astype(np.float32)
    valid = rng.random(shape) > 0.08
    baseline = [
        rng.uniform(0.0, 0.72, shape).astype(np.float32)
        for _ in range(yaw_bins)
    ]
    params = dict(
        resolution=0.2,
        yaw_bins=yaw_bins,
        footprint_front=0.42,
        footprint_back=0.42,
        footprint_half_width=0.313,
        footprint_margin=0.10,
        min_support_ratio=0.55,
        store_diagnostic_slices=True,
        use_observed_geometry=observed_geometry,
    )

    cpu = BgkSebEstimator(BgkSebParams(prefer_gpu=False, **params)).compute(
        elevation, variance, valid, baseline
    )
    gpu = BgkSebEstimator(
        BgkSebParams(prefer_gpu=True, gpu_backend="cupy", **params)
    ).compute(elevation, variance, valid, baseline)

    if gpu.backend != "cupy":
        pytest.fail(f"CUDA device exists but estimator fell back: {gpu.backend_reason}")

    np.testing.assert_allclose(gpu.bgk_mean, cpu.bgk_mean, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(gpu.bgk_var, cpu.bgk_var, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(
        gpu.bgk_traversability, cpu.bgk_traversability, rtol=2e-5, atol=2e-6
    )
    for gpu_layers, cpu_layers in (
        (gpu.final_risk_slices, cpu.final_risk_slices),
        (gpu.seb_risk_slices, cpu.seb_risk_slices),
        (gpu.seb_pitch_risk_slices, cpu.seb_pitch_risk_slices),
        (gpu.seb_roll_risk_slices, cpu.seb_roll_risk_slices),
        (gpu.seb_step_risk_slices, cpu.seb_step_risk_slices),
        (gpu.seb_support_slices, cpu.seb_support_slices),
    ):
        np.testing.assert_allclose(
            np.stack(gpu_layers), np.stack(cpu_layers), rtol=2e-5, atol=2e-6
        )


@pytest.mark.parametrize("observed_geometry", [False, True])
def test_fused_observed_step_matches_cpu_at_edges_and_missing_support(observed_geometry):
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("no CUDA device")
    rng = np.random.default_rng(20261024)
    cpu = BgkSebEstimator(BgkSebParams(resolution=.2, yaw_bins=36, prefer_gpu=False,
                                     use_observed_geometry=observed_geometry))
    gpu = BgkSebEstimator(BgkSebParams(resolution=.2, yaw_bins=36, prefer_gpu=True, gpu_backend="cupy",
                                     use_observed_geometry=observed_geometry))
    assert gpu.backend_name == "cupy"
    for shape in [(1, 1), (2, 7), (17, 23)]:
        r, c = np.indices(shape)
        plane = (.03*c+.01*r).astype(np.float32)
        for observed in [plane, plane+.001*c*c,
                         plane+(c >= shape[1]//2).astype(np.float32)*.20,
                         rng.normal(0., .1, shape).astype(np.float32)]:
            observed = observed.astype(np.float32)
            for valid in [np.ones(shape, np.float32),
                          (rng.random(shape) > .3).astype(np.float32),
                          np.zeros(shape, np.float32)]:
                expected = cpu._observed_step_risk(observed, plane, valid)
                actual = cp.asnumpy(gpu._observed_step_risk(
                    cp.asarray(observed), cp.asarray(plane), cp.asarray(valid)))
                np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)
