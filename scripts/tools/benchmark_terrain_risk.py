#!/usr/bin/env python3
"""Repeatable local terrain-risk sanity and latency check (no ROS required)."""

import argparse
from pathlib import Path
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "fsgp_bgk_se2"))
from fsgp_bgk_se2.bgk_seb import BgkSebEstimator, BgkSebParams  # noqa: E402


def scenario(name, size=104, yaw_bins=36):
    height = np.zeros((size, size), dtype=np.float32)
    variance = np.full_like(height, 0.001)
    valid = np.ones_like(height, dtype=bool)
    baseline = [np.zeros_like(height) for _ in range(yaw_bins)]
    if name == "ramp":
        height[:] = np.arange(size, dtype=np.float32)[None, :] * 0.2 * 0.17
    elif name == "step":
        height[:, size // 2 :] = 0.30
    elif name == "yaw_hazard":
        baseline[yaw_bins // 4][size // 2, size // 2] = 0.92
    elif name == "unknown":
        valid[: size // 4, :] = False
    return height, variance, valid, baseline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "cupy", "both"), default="both")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    outputs = {}
    for backend in (("cpu", "cupy") if args.backend == "both" else (args.backend,)):
        estimator = BgkSebEstimator(
            BgkSebParams(
                resolution=0.2,
                yaw_bins=36,
                prefer_gpu=backend == "cupy",
                gpu_backend=backend,
                footprint_front=0.42,
                footprint_back=0.42,
                footprint_half_width=0.313,
                footprint_margin=0.10,
                min_support_ratio=0.55,
            )
        )
        if backend == "cupy" and estimator.backend_name != "cupy":
            print(f"cupy unavailable: {estimator.backend_reason}")
            continue
        for name in ("flat", "ramp", "step", "yaw_hazard", "unknown"):
            data = scenario(name)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="CuPy may not function correctly")
                started = time.perf_counter()
                result = estimator.compute(*data)
                cold_ms = (time.perf_counter() - started) * 1000.0
                times = []
                for _ in range(args.repeats):
                    started = time.perf_counter()
                    result = estimator.compute(*data)
                    times.append((time.perf_counter() - started) * 1000.0)
            risk = np.stack(result.final_risk_slices)
            outputs[(backend, name)] = risk
            center = (slice(40, 64), slice(40, 64))
            print(
                f"{backend:4} {name:10} median={np.median(times):6.1f} ms "
                f"cold={cold_ms:6.1f} ms center_max={risk[:, center[0], center[1]].max():.3f}"
            )
            assert np.isfinite(risk).all()
            if name == "flat":
                assert risk[:, 40:64, 40:64].max() < 0.1
            elif name == "ramp":
                assert risk[:, 40:64, 40:64].max() < 0.58
            elif name == "step":
                assert risk[:, 48:56, 50:54].max() >= 0.58
            elif name == "yaw_hazard":
                assert risk[9, 52, 52] >= 0.92
                assert risk[0, 52, 52] < 0.1
            elif name == "unknown":
                assert np.allclose(risk[:, :26, :], estimator.params.unknown_risk)
        assert outputs[(backend, "step")][:, 48:56, 50:54].max() > outputs[(backend, "ramp")][:, 48:56, 50:54].max()

    if args.backend == "both" and ("cupy", "flat") in outputs:
        for name in ("flat", "ramp", "step", "yaw_hazard", "unknown"):
            np.testing.assert_allclose(outputs[("cupy", name)], outputs[("cpu", name)], rtol=2e-5, atol=1e-5)
        print("CPU/CuPy risk parity: PASS")
    print("Terrain-risk sanity: PASS")


if __name__ == "__main__":
    main()
