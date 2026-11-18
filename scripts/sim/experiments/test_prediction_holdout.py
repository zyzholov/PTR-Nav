from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prediction_holdout import ground, regression, specification, visible_mask


def grid():
    x, y = np.meshgrid(np.linspace(-3, 3, 40), np.linspace(-3, 3, 40))
    return x, y


def test_geometry_seed_changes_geometry_not_only_sensor_noise():
    x, y = grid()
    first = ground(specification("hills", 1), x, y)
    second = ground(specification("hills", 2), x, y)
    assert not np.allclose(first, second)


def test_block_and_frontier_have_separate_hidden_query_cells():
    x, y = grid()
    spec = specification("step", 7)
    for pattern in ("random", "block", "occlusion", "frontier"):
        visible = visible_mask(pattern, spec, x, y, 8)
        assert visible.any() and (~visible).any()
        observed_xyz = np.column_stack((x[visible], y[visible], ground(spec, x, y)[visible]))
        assert len(observed_xyz) == np.count_nonzero(visible)
    frontier = visible_mask("frontier", spec, x, y, 8)
    assert (x[~frontier] > spec["offset"]).all()


def test_unknown_errors_are_not_silently_excluded():
    truth = np.zeros((4, 4))
    observed = np.ones_like(truth, dtype=bool)
    observed[1:3, 1:3] = False
    prediction = truth.copy()
    prediction[~observed] = 1.0
    assert regression(prediction, truth, observed)["rmse_m"] == 0
    assert regression(prediction, truth, ~observed)["rmse_m"] == 1
