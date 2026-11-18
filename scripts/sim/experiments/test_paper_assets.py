import hashlib
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_paper_assets as paper
from prediction_holdout import MASKS, specification, visible_mask
from fsgp_bgk_se2.feature_sgp import FeatureSgpPredictor


def forbid_fit(*args, **kwargs):
    raise AssertionError("paper rendering must not rerun a predictor")


def test_missingness_diagram_reads_the_archived_masks(tmp_path, monkeypatch):
    monkeypatch.setattr(paper, "ROOT", tmp_path)
    monkeypatch.setattr(FeatureSgpPredictor, "fit", forbid_fit)
    source = tmp_path / "recordings"
    source.mkdir()
    seed = 20271012
    spec = specification("hills", seed)
    axis = (np.arange(40) - 19.5) * .15
    x, y = np.meshgrid(axis, axis)
    expected = []
    for index, kind in enumerate(MASKS):
        visible = visible_mask(kind, spec, x, y, seed + index * 1000000)
        expected.append(~visible)
        np.savez(source / f"hills_{kind}.npz", visible=visible, extent=[-3, 3, -3, 3])
    captured = []
    monkeypatch.setattr(paper, "save", lambda fig, *args: captured.append(fig))
    paper.configure_chinese()
    result = paper.missingness_patterns(source, {"runs": [{"scene": "hills", "geometry_seed": seed}]}, tmp_path)
    fig = captured[0]
    for ax, hidden, row in zip(fig.axes, expected, result["examples"]):
        np.testing.assert_array_equal(ax.images[0].get_array(), hidden.astype(float))
        assert row["hidden_cells"] == int(hidden.sum())
        assert row["observed_cells"] + row["hidden_cells"] == hidden.size
        assert row["source_sha256"] == hashlib.sha256((tmp_path / row["source"]).read_bytes()).hexdigest()
        assert "地形平面" in ax.get_xlabel() and "米" in ax.get_ylabel()
    assert "颜色不表示高度或风险" in fig._suptitle.get_text()
    plt.close(fig)


def test_summary_distinguishes_categories_and_confidence_intervals(tmp_path, monkeypatch):
    monkeypatch.setattr(FeatureSgpPredictor, "fit", forbid_fit)
    rows = [{"scene": "hills", "geometry_seed": seed, "mask": kind,
             "unknown_all_rmse_m": .02 + .02 * seed, "unknown_all_mae_m": .01,
             "unknown_prediction_coverage": .5 + .5 * seed, "unknown_picp95": .6 + .36 * seed}
            for seed in range(2) for kind in MASKS]
    captured = []
    monkeypatch.setattr(paper, "save", lambda fig, *args: captured.append(fig))
    paper.configure_chinese()
    result = paper.prediction_summary(rows, tmp_path)
    fig = captured[0]
    assert "2 个地形" in fig._suptitle.get_text() and "8 组试验" in fig._suptitle.get_text()
    assert "统计置信区间" in fig._suptitle.get_text()
    for ax, key, scale in zip(fig.axes,
                             ("unknown_all_rmse_m", "unknown_prediction_coverage", "unknown_picp95"),
                             (100, 100, 100)):
        assert "横轴为类别" in ax.get_xlabel()
        for kind, annotation in zip(MASKS, ax.texts):
            stats = result[kind][key]
            assert stats["geometry_n"] == 2
            assert annotation.get_position()[1] > scale * stats["ci95"][1]
    assert "候选格子" in fig.axes[1].get_ylabel()
    assert "真值落入区间" in fig.axes[2].get_ylabel()
    assert " & 2 & " in (tmp_path / "holdout_rows.tex").read_text()
    plt.close(fig)
