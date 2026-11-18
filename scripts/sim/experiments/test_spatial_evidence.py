import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_clouds import grid, height_at
from spatial_evidence import observation_pair, reconstruct_spatial_case
from pointcloud_batch import digest
from road_reference_audit import height_score


def test_sparse_grid_centers_do_not_require_unused_nan_neighbors():
    height = np.array([[1., np.nan, 3.], [4., 5., np.nan], [np.nan, 8., 9.]])
    xy = grid(height.shape, [1000.123, -523.91], .2)
    sampled = height_at(height, [1000.123, -523.91], .2, xy).reshape(height.shape)
    np.testing.assert_array_equal(np.isfinite(sampled), np.isfinite(height))
    np.testing.assert_allclose(sampled, height, equal_nan=True)


def test_nan_free_grid_line_works_but_missing_positive_weight_does_not():
    height = np.array([[1., np.nan], [3., np.nan]])
    query = np.array([[.1, .2], [.2, .2], [-.1, .2]])
    result = height_at(height, [0., 0.], .2, query)
    assert result[0] == 2.
    assert np.isnan(result[1]) and np.isnan(result[2])


def test_full_finite_interpolation_is_unchanged():
    xy = grid((12, 11), [-1., -1.], .2)
    height = (.2 * xy[:, 0] - .3 * xy[:, 1]).reshape(12, 11)
    query = np.array([[-.45, -.15], [.12, .75]])
    np.testing.assert_allclose(height_at(height, [-1., -1.], .2, query),
                               .2 * query[:, 0] - .3 * query[:, 1], atol=1e-12)


def test_deletion_reduces_points_not_creates_observed_cells():
    xyz = np.array([[.1, .1, 1.], [.1, .1, 2.], [.3, .1, 3.]])
    before, after = observation_pair(xyz, np.ones(3, bool), np.array([False, True, False]),
                                      (2, 2), np.array([0., 0.]), .2)
    assert np.isfinite(before).sum() == 2 and np.isfinite(after).sum() == 1
    assert not (np.isfinite(after) & ~np.isfinite(before)).any()
    # A minimum-height cell can rise after deleting its lowest point, but is not new coverage.
    assert before[0, 0] == 1. and after[0, 0] == 2.


def test_exact_case_reconstruction_does_not_use_overwritten_snapshot(tmp_path):
    from paper_scale_evaluate import partial_points
    shape = (104, 104); origin = np.array([0., 0.]); xy = grid(shape, origin, .2)
    z = .1 * xy[:, 0]; xyz = np.c_[xy, z].astype(np.float32)
    source = tmp_path / "source.npz"
    np.savez(source, xyz=xyz, is_ground=np.ones(len(xyz), bool), shape=np.array(shape),
        query_origin=origin, truth_origin=origin, truth_resolution=.2, truth_height=z.reshape(shape))
    row = {"id": "first__hole16", "dataset_id": "first", "family": "road", "condition": "hole16", "case_seed": 3}
    keep = partial_points(xyz, "hole16", origin, 3)
    observed = np.ones(shape, bool); observed.ravel()[~keep] = False
    prediction = np.zeros(len(xy))
    np.savez(tmp_path / (row["id"] + ".npz"), height_mean=prediction, latent_var=np.ones(len(xy)),
        reference_height=z.reshape(shape), observed_cells=observed)
    np.savez(tmp_path / "road__hole16__snapshot.npz", mean=np.ones(shape), query_origin=origin)
    catalog = {"datasets": [{"id": "first", "path": "source.npz", "sha256": digest(source)}]}
    d = reconstruct_spatial_case({"settings": {"input_noise_std_m": .02}}, catalog,
        tmp_path / "results.json", tmp_path / "dataset.json", row)
    assert d["legacy_snapshot_matches"] is False
    np.testing.assert_array_equal(d["mean"].ravel(), prediction)
    assert d["counts"]["new_observed_cells"] == 0 and d["counts"]["removed_cells"] > 0
    legacy = reconstruct_spatial_case({"settings": {}}, catalog,
        tmp_path / "results.json", tmp_path / "dataset.json", row)
    assert legacy["observation_height_kind"] == "source_subset_before_unrecorded_noise"
    assert legacy["input_noise_std_m"] is None
    np.testing.assert_array_equal(legacy["observed"], legacy["after_before_noise"])


def test_height_rescore_uses_only_hidden_supported_ground():
    reference = np.array([[1., 1., np.nan]])
    mean = np.array([[100., 2., 100.]])
    result = height_score(mean, reference, np.array([[True, False, False]]), np.ones_like(mean))
    assert result["hidden_cells"] == 1 and result["rmse_m"] == 1. and result["picp95"] == 1.
