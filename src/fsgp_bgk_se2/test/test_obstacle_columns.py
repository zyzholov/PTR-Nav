from types import SimpleNamespace

import numpy as np
import pytest

from fsgp_bgk_se2.obstacle_columns import preserve_collision_risk, vertical_columns


def ground_cloud(slope=0.0):
    yy, xx = np.mgrid[0:20, 0:20]
    return np.c_[xx.ravel() * .2 + .1, yy.ravel() * .2 + .1,
                 slope * xx.ravel() * .2].astype(np.float32)


@pytest.mark.parametrize("slope", [0.0, 0.25, -0.25])
def test_ground_not_vertical_obstacle(slope):
    mask, _ = vertical_columns(ground_cloud(slope), (20, 20), (0, 0), .2)
    assert not mask.any()


def test_thin_column_survives_smoothing_and_clears_on_next_ground_scan():
    ground = ground_cloud()
    pole = np.c_[np.full(14, 1.9), np.full(14, 2.1), np.linspace(.1, 1.4, 14)]
    occupied, _ = vertical_columns(np.r_[ground, pole], (20, 20), (0, 0), .2)
    assert occupied[10, 9]
    result = SimpleNamespace(baseline_risk=np.zeros((20, 20)),
                             final_risk_slices=[np.zeros((20, 20)) for _ in range(36)])
    preserve_collision_risk(result, occupied)
    assert result.baseline_risk[10, 9] == 1.0
    assert all(r[10, 9] == 1.0 for r in result.final_risk_slices)
    cleared, _ = vertical_columns(ground, (20, 20), (0, 0), .2)
    assert not cleared.any()


def test_gpu_cpu_agree():
    cp = pytest.importorskip("cupy")
    points = np.r_[ground_cloud(), [[1.9, 2.1, .5], [1.9, 2.1, .9]]]
    cpu, _ = vertical_columns(points, (20, 20), (0, 0), .2)
    gpu, _ = vertical_columns(points, (20, 20), (0, 0), .2, xp=cp)
    np.testing.assert_array_equal(cpu, cp.asnumpy(gpu))
