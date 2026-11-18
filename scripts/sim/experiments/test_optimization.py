import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_optimization import cluster_stats, load, observed_cell_mean
from obstacle_regression import make_case
from fsgp_bgk_se2.obstacle_columns import vertical_columns


def test_statistics_group_repeats_before_bootstrap():
    rows = [{"dataset_id": "a", "error": 1.0}] * 50 + [{"dataset_id": "b", "error": 3.0}]
    stats = cluster_stats(rows, "error")
    assert stats["mean"] == 2.0 and stats["n"] == 2
    grouped = cluster_stats(rows, "error", {"a": "source", "b": "source"})
    assert grouped["n"] == 1
    assert grouped["ci95"][0] == grouped["ci95"][1]


def test_plotting_refuses_incomplete_batch(tmp_path):
    path = tmp_path / "batch.json"
    path.write_text(json.dumps({"runs": [], "planned_cases": 20}))
    with pytest.raises(ValueError, match="not finished"):
        load(path, batch=True)


def test_collision_generator_exposes_only_measured_ground_and_clears():
    points, observed, truth, valid, clear = make_case("crosscell_pole", 91)
    assert np.isnan(observed).any() and truth.sum() == 2
    assert len(clear) == np.isfinite(observed).sum()
    assert valid[truth].all()
    no_obstacle, _ = vertical_columns(clear, truth.shape, (0, 0), .2, ground_elevation=observed)
    assert not no_obstacle.any()
    np.testing.assert_equal(points[:len(clear)], clear)


def test_observation_panel_uses_input_measurements_not_reference_heights():
    xy = np.array([[.1, .1], [.3, .1], [.1, .3], [.3, .3]])
    points = np.array([[.08, .08, 4], [.12, .12, 8], [.3, .3, 9]])
    panel = observed_cell_mean(points, xy, (2, 2), .2)
    assert panel[0, 0] == 6 and panel[1, 1] == 9
    assert np.isnan(panel[0, 1]) and np.isnan(panel[1, 0])
