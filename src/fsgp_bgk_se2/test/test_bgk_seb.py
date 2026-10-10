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
    band = (slice(27, 37), slice(29, 35))
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


def test_cuda_backend_matches_cpu_when_available():
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
