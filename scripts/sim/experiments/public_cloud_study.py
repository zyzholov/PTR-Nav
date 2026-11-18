#!/usr/bin/env python3
"""Public-data adapters and current BGK/SEB tests, with explicit proxy truth.

Not the ROS CUDA terrain_analyzer, not paper-identical temporal BGK, and not a
certified traversability probability test. RELLIS heights use annotated ground
in the sensor frame; poses/gravity and an independent survey are unavailable.
"""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import re
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

from pointcloud_batch import atomic_json, digest, read_points, point_cells
from terrain_risk_benchmark import params_from_config
from fsgp_bgk_se2.bgk_seb import BgkSebEstimator
from fsgp_bgk_se2.obstacle_columns import vertical_columns, preserve_collision_risk

ROOT = Path(__file__).resolve().parents[3]
GROUND_NAMES = {"dirt", "grass", "asphalt", "concrete", "mud"}
STRUCTURE_NAMES = {"tree", "pole", "vehicle", "object", "building", "log", "person", "fence", "barrier"}


def load_rellis(points_path, labels_path):
    points = np.fromfile(points_path, dtype="<f4")
    labels = np.fromfile(labels_path, dtype="<u4") & 0xffff
    if points.size % 4 or len(labels) != points.size // 4:
        raise ValueError("KITTI XYZ/intensity and uint32 semantic label lengths disagree")
    points = points.reshape(-1, 4)[:, :3]
    good = np.isfinite(points).all(axis=1) & (np.linalg.norm(points, axis=1) > .5)
    return points[good], labels[good]


def rellis_identity(path):
    sequence = path.parent.parent.name
    if sequence == "Rellis_3D_lidar_example":
        return "rellis_" + path.stem, "rellis_short_example"
    if not re.fullmatch(r"[A-Za-z0-9_-]+", sequence):
        raise ValueError("sequence directory must be an ASCII identifier")
    return f"rellis_{sequence}_{path.stem}", f"rellis_sequence_{sequence}"


def grid_reference(center, size, resolution):
    axis = (np.arange(size) - (size - 1) / 2) * resolution
    x, y = np.meshgrid(axis + center[0], axis + center[1])
    return {"query_xy": np.column_stack((x.ravel(), y.ravel())).astype(np.float32),
            "shape": (size, size), "resolution": resolution}


def envelope(points, ref, min_hits=2):
    ids, inside = point_cells(points, ref)
    count = np.bincount(ids[inside], minlength=len(ref["query_xy"]))
    low = np.full(len(count), np.inf, dtype=np.float32)
    np.minimum.at(low, ids[inside], points[inside, 2])
    valid = np.isfinite(low) & (count >= min_hits)
    return low, valid


def select_lower_points(points, ref, tolerance=.06):
    # Cell-local extraction depends on the remaining input only. Future/hidden
    # reference heights are not used by this extraction or by the predictor.
    ids, inside = point_cells(points, ref)
    low, _ = envelope(points, ref, min_hits=1)
    keep = inside.copy()
    keep[inside] &= points[inside, 2] <= low[ids[inside]] + tolerance
    return points[keep]


def source_stream(raw):
    for path in sorted((raw / "fsgp").glob("*.pcd")):
        points = read_points(path)
        low, high = points[:, :2].min(0), points[:, :2].max(0)
        centers = [low + (high - low) * [fx, fy] for fx, fy in ((.25, .25), (.75, .25), (.25, .75), (.75, .75))]
        yield "fsgp_" + path.stem, points, None, centers, path, None, "fsgp_" + path.stem
    for path in sorted((raw / "rellis").rglob("os1_cloud_node_kitti_bin/*.bin")):
        label = path.parent.parent / "os1_cloud_node_semantickitti_label_id" / (path.stem + ".label")
        points, labels = load_rellis(path, label)
        source_id, cluster_id = rellis_identity(path)
        yield source_id, points, labels, [(-8., 0.), (-14., 0.)], path, label, cluster_id


def prepare(args):
    raw = args.raw.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("new prepared dataset output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    for folder in ("clouds", "references", "raw_patches"):
        (output / folder).mkdir()
    fetch = json.loads((raw / "fetch_manifest.json").read_text())
    ontology = json.loads((raw / "rellis_labels.json").read_text())
    ground_ids = [int(k) for k, v in ontology["labels"].items() if v in GROUND_NAMES]
    datasets, audit, inventory = [], [], []
    seen = set()
    for source_id, points, labels, centers, source_path, label_path, cluster_id in source_stream(raw):
        if source_id in seen:
            raise ValueError("duplicate source identities; do not overwrite scans from different sequences")
        seen.add(source_id)
        inventory.append({"source_id": source_id, "source_path": str(source_path),
                          "source_sha256": digest(source_path), "points": len(points),
                          "label_path": str(label_path) if label_path else None,
                          "label_sha256": digest(label_path) if label_path else None,
                          "bbox_min": points.min(0).tolist(), "bbox_max": points.max(0).tolist()})
        for index, center in enumerate(centers):
            name = f"{source_id}_p{index:02d}"
            ref = grid_reference(center, args.size, args.resolution)
            _, inside = point_cells(points, ref)
            patch = points[inside]
            patch_labels = labels[inside] if labels is not None else np.full(len(patch), -1, dtype=np.int32)
            ground = patch[np.isin(patch_labels, ground_ids)] if labels is not None else select_lower_points(patch, ref)
            height, valid = envelope(ground, ref)
            row = {"id": name, "source_id": source_id, "valid_reference_cells": int(valid.sum()),
                   "reference_fraction": float(valid.mean()), "ground_points": len(ground), "raw_points": len(patch),
                   "center_xy": np.asarray(center).tolist()}
            if valid.sum() < args.min_valid_cells or len(ground) < 100:
                row.update(status="excluded_low_reference_density")
                audit.append(row)
                continue
            np.save(output / "clouds" / f"{name}.npy", ground)
            np.savez_compressed(output / "references" / f"{name}.npz", **ref,
                               height=height, valid=valid)
            np.savez_compressed(output / "raw_patches" / f"{name}.npz", xyz=patch, labels=patch_labels)
            row.update(status="included")
            audit.append(row)
            datasets.append({"id": name, "family": "rellis" if labels is not None else source_id,
                "split": "development", "points": f"clouds/{name}.npy", "reference": f"references/{name}.npz",
                "raw_patch": f"raw_patches/{name}.npz", "reference_kind": "dense_cloud_proxy", "unit": "m",
                "frame_id": "ouster_sensor_unlevelled" if labels is not None else "map",
                "source_id": source_id, "cluster_id": cluster_id,
                "center_xy": np.asarray(center).tolist(),
                "ground_input_mode": "semantic_annotation_assisted" if labels is not None else "cell_lower_envelope",
                "reference_note": "minimum observed ground height per cell; no independent survey or dense hidden-surface truth"})
    atomic_json(output / "dataset_manifest.json", {"schema_version": 1, "datasets": datasets,
        "source": "public examples; overlapping crops / adjacent frames are not independent worlds",
        "fetch_manifest_sha256": digest(raw / "fetch_manifest.json"), "inventory": inventory,
        "selection_audit": audit, "ontology": ontology,
        "grid": {"size": args.size, "resolution_m": args.resolution},
        "raw_source_root": str(raw)})
    print(f"prepared {len(datasets)}/{len(audit)} candidate patches from {len(inventory)} files; {output}")


def semantic_targets(points, labels, ground, ref, ontology):
    names = {int(k): v for k, v in ontology["labels"].items()}
    structure_ids = [k for k, v in names.items() if v in STRUCTURE_NAMES]
    ground_ids = [k for k, v in names.items() if v in GROUND_NAMES]
    size = len(ref["query_xy"])
    target = np.zeros(size, bool)
    negative = np.zeros(size, bool)
    if len(ground) < 8:
        return target.reshape(ref["shape"]), negative.reshape(ref["shape"])
    distance, nearest = cKDTree(ground[:, :2]).query(points[:, :2])
    above = points[:, 2] - ground[nearest, 2]
    hazard_points = np.isin(labels, structure_ids) & (distance <= 1.0) & (above >= .32) & (above <= 1.5)
    ground_points = np.isin(labels, ground_ids)
    ids, inside = point_cells(points, ref)
    target[ids[inside & hazard_points]] = True
    negative[ids[inside & ground_points]] = True
    ambiguous = (labels != 0) & ~ground_points & ~np.isin(labels, structure_ids)
    ambiguous_cells = np.zeros(size, bool)
    ambiguous_cells[ids[inside & ambiguous]] = True
    negative &= ~ambiguous_cells & ~target
    return target.reshape(ref["shape"]), (target | negative).reshape(ref["shape"])


def binary_counts(prediction, target, valid):
    return {"tp": int((prediction & target & valid).sum()),
            "fp": int((prediction & ~target & valid).sum()),
            "fn": int((~prediction & target & valid).sum()),
            "tn": int((~prediction & ~target & valid).sum())}


def analyze(args):
    from pointcloud_batch import read_reference, keep_cells
    import cupy as cp
    args.output.mkdir(parents=True, exist_ok=True)
    catalog = json.loads(args.dataset.read_text())
    batch = json.loads(args.batch.read_text())
    if batch["settings"]["dataset_sha256"] != digest(args.dataset):
        raise ValueError("dataset differs from recorded batch")
    items = {item["id"]: item for item in catalog["datasets"]}
    params = params_from_config("cupy", catalog["grid"]["resolution_m"], 36)
    params.store_diagnostic_slices = True
    estimator = BgkSebEstimator(params)
    rows, fingerprints = [], {}
    algorithm_fingerprints = {name: digest(ROOT / "src/fsgp_bgk_se2/fsgp_bgk_se2" / name)
                              for name in ("obstacle_columns.py", "bgk_seb.py")}
    cached_id, cache = None, None
    for index, trial in enumerate(batch["runs"]):
        if trial["status"] != "ok":
            rows.append({"id": trial["id"], "status": "upstream_not_evaluable" if
                         trial.get("reason") == "no held-out reference cells" or
                         trial.get("error", "").endswith("no held-out reference cells") else "upstream_failed"})
            continue
        item = items[trial["dataset_id"]]
        if item["id"] != cached_id:
            ref = read_reference(args.dataset.parent / item["reference"])
            raw_path = args.dataset.parent / item["raw_patch"]
            with np.load(raw_path, allow_pickle=False) as raw:
                points, labels = raw["xyz"], raw["labels"]
            ground = read_points(args.dataset.parent / item["points"])
            target, label_valid = semantic_targets(points, labels, ground, ref, catalog["ontology"])
            fingerprints[item["id"]] = {"raw_patch_sha256": digest(raw_path)}
            cache = (ref, points, labels, target, label_valid)
            cached_id = item["id"]
        ref, points, labels, target, label_valid = cache
        record = {"id": trial["id"], "dataset_id": item["id"], "cluster_id": item["cluster_id"],
                  "family": item["family"], "kind": trial["kind"], "severity": trial["severity"]}
        try:
            with np.load(args.batch.parent / "cases" / trial["id"] / "prediction.npz", allow_pickle=False) as prediction:
                keep = keep_cells(ref["query_xy"], trial["kind"], trial["severity"], trial["mask_seed"])
                ids, inside = point_cells(points, ref)
                take = inside.copy()
                take[inside] &= keep[ids[inside]]
                observed_raw = points[take]
                observed_ground = prediction["observed_xyz"]
                heights, valid = envelope(observed_ground, ref, min_hits=1)
                shape = ref["shape"]
                elevation = np.where(valid, heights, np.nan).reshape(shape)
                variance = np.full(shape, .0004, np.float32)
                cp.cuda.Stream.null.synchronize()
                start = time.perf_counter()
                occupied_gpu, _ = vertical_columns(observed_raw, shape, ref["query_xy"][0] - ref["resolution"] / 2,
                                                    ref["resolution"], ground_elevation=elevation, xp=cp)
                occupied = cp.asnumpy(occupied_gpu)
                result = estimator.compute(elevation, variance, valid.reshape(shape),
                                            [occupied.astype(np.float32)], observed_elevation=elevation)
                preserve_collision_risk(result, occupied)
                cp.cuda.Stream.null.synchronize()
                elapsed = (time.perf_counter() - start) * 1000
                record.update(status="ok", actual_backend=result.backend, bgk_seb_columns_ms=elapsed,
                              observed_ground_cells=int(valid.sum()), observed_raw_points=len(observed_raw),
                              occupied_cells=int(occupied.sum()),
                              hidden_risk_min=float(np.min(np.stack(result.final_risk_slices)[:, ~valid.reshape(shape)]))
                              if (~valid).any() else None)
                if item["family"] == "rellis":
                    # Both common-observed and all labelled cells are reported.
                    # Deliberately removed cells must not disappear from scores.
                    observed_cells = np.zeros(len(keep), bool)
                    observed_cells[ids[take]] = True
                    record["semantic_common_observed"] = binary_counts(occupied, target,
                        label_valid & observed_cells.reshape(shape))
                    record["semantic_all_labelled"] = binary_counts(occupied, target, label_valid)
                if trial["id"].endswith("r000"):
                    np.savez_compressed(args.output / (trial["id"] + ".npz"), raw_xyz=observed_raw,
                        extent=[ref["query_xy"][:,0].min() - ref["resolution"]/2,
                                ref["query_xy"][:,0].max() + ref["resolution"]/2,
                                ref["query_xy"][:,1].min() - ref["resolution"]/2,
                                ref["query_xy"][:,1].max() + ref["resolution"]/2],
                        reference_height=np.where(ref["valid"], ref["height"], np.nan).reshape(shape),
                        reference_valid=ref["valid"].reshape(shape), elevation=elevation,
                        predicted=prediction["mean"].reshape(shape), latent_var=prediction["latent_var"].reshape(shape),
                        valid=valid.reshape(shape), occupied=occupied,
                        risk_yaw0=result.final_risk_slices[0], label_target=target, label_valid=label_valid)
        except Exception as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        rows.append(record)
        atomic_json(args.output / "results.json", {"scope": __doc__, "parameters": asdict(params),
            "dataset_sha256": digest(args.dataset), "batch_sha256": digest(args.batch),
            "script_sha256": digest(__file__), "algorithm_fingerprints": algorithm_fingerprints,
            "source_fingerprints": fingerprints, "runs": rows})
        if (index + 1) % 50 == 0:
            print(f"[public BGK/SEB] {index + 1}/{len(batch['runs'])}", flush=True)
    failures = sum(r["status"] in ("failed", "upstream_failed") for r in rows)
    skipped = sum(r["status"] == "upstream_not_evaluable" for r in rows)
    print(f"BGK/SEB {len(rows)} cases, {failures} failures, {skipped} no-reference cases: {args.output}")
    return int(failures > 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--raw", type=Path, default=Path("data/public_clouds"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--resolution", type=float, default=.2)
    p.add_argument("--min-valid-cells", type=int, default=100)
    a = sub.add_parser("analyze")
    a.add_argument("--dataset", type=Path, required=True)
    a.add_argument("--batch", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
        return 0
    return analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())
