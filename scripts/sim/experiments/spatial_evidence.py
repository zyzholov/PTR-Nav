"""Matched spatial views reconstructed from the exact archived case, never a family snapshot."""

import numpy as np

from paper_clouds import grid, height_at, raster
from paper_scale_evaluate import partial_points
from pointcloud_batch import digest


def observation_pair(xyz, ground, keep, shape, origin, resolution):
    before = raster(xyz[ground], shape, origin, resolution)
    after = raster(xyz[ground & keep], shape, origin, resolution)
    if (np.isfinite(after) & ~np.isfinite(before)).any():
        raise ValueError("deleting points introduced an observed cell")
    return before, after


def reconstruct_spatial_case(record, catalog, results_path, dataset_path, row):
    item = next(d for d in catalog["datasets"] if d["id"] == row["dataset_id"])
    source = dataset_path.parent / item["path"]
    if digest(source) != item["sha256"]:
        raise ValueError("spatial source hash mismatch")
    with np.load(source, allow_pickle=False) as a:
        case = {k: a[k].copy() for k in a.files if k != "full_xyz"}
    archive = results_path.parent / (row["id"] + ".npz")
    with np.load(archive, allow_pickle=False) as a:
        saved = {k: a[k].copy() for k in a.files}
    shape, origin = tuple(case["shape"]), case["query_origin"]
    keep = partial_points(case["xyz"], row["condition"], origin, row["case_seed"])
    ground = case["is_ground"].astype(bool)
    before, clean_after = observation_pair(case["xyz"], ground, keep, shape, origin, .2)
    noise_std = record["settings"].get("input_noise_std_m")
    if noise_std is None:
        # Older archives did not record input noise; show the source subset, do not guess Z values.
        observed = clean_after.copy()
    else:
        raw = case["xyz"][keep].copy()
        raw[:, 2] += np.random.default_rng(row["case_seed"] + 100000).normal(0, noise_std, len(raw))
        observed = raster(raw[ground[keep]], shape, origin, .2)
    if not np.array_equal(np.isfinite(observed), saved["observed_cells"]):
        raise ValueError("reconstructed observation mask differs from the actual run")
    if not np.array_equal(np.isfinite(clean_after), np.isfinite(observed)):
        raise ValueError("Z noise changed XY occupancy")
    reference = height_at(case["truth_height"], case["truth_origin"], float(case["truth_resolution"]),
                          grid(shape, origin, .2)).reshape(shape)
    old_reference = saved["reference_height"]
    common = np.isfinite(old_reference) & np.isfinite(reference)
    if common.any() and not np.allclose(old_reference[common], reference[common], atol=1e-5):
        raise ValueError("reference correction unexpectedly changed supported heights")
    snapshot = results_path.parent / (row["family"] + "__" + row["condition"] + "__snapshot.npz")
    matches = None
    if snapshot.exists():
        with np.load(snapshot, allow_pickle=False) as a:
            matches = bool(np.array_equal(a["mean"].ravel(), saved["height_mean"].ravel())
                           and np.allclose(a["query_origin"], origin))
    counts = {"full_ground_points": int(ground.sum()), "partial_ground_points": int((ground & keep).sum()),
        "removed_ground_points": int((ground & ~keep).sum()),
        "before_cells": int(np.isfinite(before).sum()), "after_cells": int(np.isfinite(observed).sum()),
        "new_observed_cells": int((np.isfinite(observed) & ~np.isfinite(before)).sum()),
        "removed_cells": int((np.isfinite(before) & ~np.isfinite(observed)).sum()),
        "legacy_reference_cells": int(np.isfinite(old_reference).sum()),
        "corrected_reference_cells": int(np.isfinite(reference).sum()),
        "legacy_hidden_cells": int((np.isfinite(old_reference) & ~np.isfinite(observed)).sum()),
        "corrected_hidden_cells": int((np.isfinite(reference) & ~np.isfinite(observed)).sum()),
        "lost_reference_cells": int((np.isfinite(reference) & ~np.isfinite(old_reference)).sum())}
    return {"row": row, "origin": origin, "before": before, "after_before_noise": clean_after, "observed": observed,
        "mean": saved["height_mean"].reshape(shape), "std": np.sqrt(saved["latent_var"]).reshape(shape),
        "legacy_reference": old_reference, "corrected_reference": reference, "saved": saved,
        "counts": counts, "legacy_snapshot_matches": matches,
        "source_sha256": digest(source), "archive_sha256": digest(archive), "input_noise_std_m": noise_std,
        "observation_height_kind": "source_subset_before_unrecorded_noise" if noise_std is None else "replayed_recorded_noise"}


def provenance(case):
    return {k: case[k] for k in ("counts", "legacy_snapshot_matches", "source_sha256", "archive_sha256",
                                "input_noise_std_m", "observation_height_kind")} | {
        "case_id": case["row"]["id"], "source_id": case["row"]["dataset_id"],
        "scope": "read-only paired observation/archived prediction; v2 reference is diagnostic, old scores unchanged"}
