#!/usr/bin/env python3
"""File-based, causal partial-ground-cloud evaluation of our Feature SGP only.

This is 2.5D surface prediction, not arbitrary 3D scene completion, nor the
ROS terrain front-end. Reference heights never enter feature extraction.
"""

import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src/fsgp_bgk_se2"))
from prediction_holdout import SCENES, ground, specification, regression


class NoHeldOutReference(ValueError):
    """The requested mask contains no evaluable, independently withheld cell."""


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def read_points(path):
    path = Path(path)
    if path.suffix == ".npy":
        points = np.load(path, allow_pickle=False)
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            points = data["xyz"]
    elif path.suffix.lower() in (".pcd", ".ply", ".xyz"):
        import open3d as o3d
        points = np.asarray(o3d.io.read_point_cloud(str(path)).points)
    else:
        raise ValueError("point file must be NPY/NPZ(xyz)/PCD/PLY/XYZ")
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("ground input must be an N x 3 XYZ array")
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 8:
        raise ValueError("not enough finite ground observations")
    return points


def read_reference(path):
    with np.load(path, allow_pickle=False) as data:
        ref = {key: data[key] for key in data.files}
    xy = np.asarray(ref["query_xy"], dtype=np.float32)
    shape = tuple(int(n) for n in ref["shape"])
    z = np.asarray(ref["height"], dtype=np.float32).ravel()
    valid = np.asarray(ref["valid"], dtype=bool).ravel()
    resolution = float(ref["resolution"])
    if (len(shape) != 2 or min(shape) < 4 or xy.shape != (np.prod(shape), 2)
            or z.shape != (len(xy),) or valid.shape != z.shape or resolution <= 0
            or not np.isfinite(xy).all() or not np.isfinite(z[valid]).all()):
        raise ValueError("invalid reference grid contract")
    grid = xy.reshape(*shape, 2)
    if (not np.allclose(np.diff(grid[:, :, 0], axis=1), resolution, atol=1e-4)
            or not np.allclose(np.diff(grid[:, :, 1], axis=0), resolution, atol=1e-4)
            or not np.allclose(grid[:, :, 0], grid[0:1, :, 0], atol=1e-4)
            or not np.allclose(grid[:, :, 1], grid[:, 0:1, 1], atol=1e-4)):
        raise ValueError("reference must be a row-major axis-aligned metric XY grid")
    ref.update(query_xy=xy, shape=shape, height=z, valid=valid, resolution=resolution)
    return ref


def point_cells(points, reference):
    rows, cols = reference["shape"]
    resolution = reference["resolution"]
    origin = reference["query_xy"][0] - resolution * 0.5
    ij = np.floor((points[:, :2] - origin) / resolution).astype(int)
    inside = (ij[:, 0] >= 0) & (ij[:, 0] < cols) & (ij[:, 1] >= 0) & (ij[:, 1] < rows)
    return ij[:, 1] * cols + ij[:, 0], inside


def keep_cells(xy, kind, severity, seed):
    rng = np.random.default_rng(seed)
    if kind == "random":
        return rng.random(len(xy)) >= severity
    if kind == "block":
        center = (xy.min(axis=0) + xy.max(axis=0)) * 0.5
        center += rng.uniform(-0.2, 0.2, 2)
        return (np.abs(xy - center) >= severity * 0.5).any(axis=1)
    if kind == "frontier":
        boundary = float(xy[:, 0].min() + (1 - severity) * np.ptp(xy[:, 0]))
        return xy[:, 0] <= boundary
    raise ValueError("unknown masking pattern")


def partial_input(points, reference, keep, seed, noise_std):
    cells, inside = point_cells(points, reference)
    selected = inside.copy()
    selected[inside] &= keep[cells[inside]]
    observed = points[selected].copy()
    if len(observed) < 8:
        raise ValueError("mask leaves fewer than eight observed points")
    observed[:, 2] += np.random.default_rng(seed).normal(0, noise_std, len(observed)).astype(np.float32)
    observed_cells = np.zeros(len(keep), dtype=bool)
    observed_cells[cells[selected]] = True
    return observed, observed_cells


def score(prediction, reference, observed_cells):
    valid = reference["valid"]
    hidden = valid & ~observed_cells
    if not hidden.any():
        raise NoHeldOutReference("no held-out reference cells")
    results = {"unknown_cells": int(hidden.sum()), "observed_reference_cells": int((valid & observed_cells).sum())}
    for name, mask in (("observed", valid & observed_cells), ("unknown", hidden),
                       ("accepted_unknown", hidden & prediction.accepted)):
        results.update({name + "_" + key: value for key, value in
            regression(prediction.mean, reference["height"], mask).items()})
    error = prediction.mean - reference["height"]
    std = np.sqrt(np.maximum(prediction.latent_var, 1e-10))
    for nominal, factor in ((50, 0.67448975), (80, 1.28155157), (90, 1.64485363), (95, 1.95996398)):
        results[f"picp{nominal}"] = float((np.abs(error[hidden]) <= factor * std[hidden]).mean())
    results["interval_width95_m"] = float((3.91992796 * std[hidden]).mean())
    results["unknown_accepted_coverage"] = float(prediction.accepted[hidden].mean())
    results["unknown_nll"] = float((0.5 * np.log(2 * np.pi * std[hidden] ** 2)
        + 0.5 * (error[hidden] / std[hidden]) ** 2).mean())
    if "boundary" in reference:
        boundary = hidden & reference["boundary"].ravel().astype(bool)
        results.update({"boundary_" + key: value for key, value in
            regression(prediction.mean, reference["height"], boundary).items()})
    return results


def make_synthetic(args):
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("dataset output must be empty")
    (output / "clouds").mkdir(parents=True)
    (output / "references").mkdir()
    datasets = []
    axis = (np.arange(args.size) - (args.size - 1) * 0.5) * args.resolution
    x, y = np.meshgrid(axis, axis)
    for si, scene in enumerate(SCENES):
        for i in range(args.scenes_per_family):
            split_offset = {"development": 0, "calibration": 1000000, "test": 2000000}[args.split]
            geometry_seed = args.seed + split_offset + si * 10000 + i
            spec = specification(scene, geometry_seed)
            name = f"{scene}_{i:03d}"
            # Four surface samples per cell. The reference is independently
            # evaluated at the center, not copied from an input measurement.
            cloud = np.concatenate([np.column_stack((
                (x + dx).ravel(), (y + dy).ravel(), ground(spec, x + dx, y + dy).ravel()))
                for dx in (-args.resolution * 0.25, args.resolution * 0.25)
                for dy in (-args.resolution * 0.25, args.resolution * 0.25)]).astype(np.float32)
            np.save(output / "clouds" / f"{name}.npy", cloud)
            h = ground(spec, x, y)
            eps = 0.05
            jump = np.maximum(np.abs(ground(spec, x + eps, y) - ground(spec, x - eps, y)),
                              np.abs(ground(spec, x, y + eps) - ground(spec, x, y - eps)))
            np.savez_compressed(output / "references" / f"{name}.npz",
                query_xy=np.column_stack((x.ravel(), y.ravel())), height=h.ravel(),
                valid=np.ones(h.size, dtype=bool), shape=np.asarray(x.shape),
                resolution=args.resolution, boundary=jump.ravel() >= 0.18)
            datasets.append({"id": name, "family": scene, "geometry_seed": geometry_seed,
                "split": args.split, "points": f"clouds/{name}.npy",
                "reference": f"references/{name}.npz", "reference_kind": "analytic_ground",
                "frame_id": "map", "unit": "m", "geometry_parameters": spec})
    atomic_json(output / "dataset_manifest.json", {
        "schema_version": 1, "source": "generated analytic ground-only point files; not Isaac or field data",
        "datasets": datasets})
    print(f"created {len(datasets)} independent ground clouds: {output}")


def run_batch(args):
    from fsgp_bgk_se2.feature_sgp import FeatureSgpParams, FeatureSgpPredictor
    predictor_path = ROOT / "src/fsgp_bgk_se2/fsgp_bgk_se2/feature_sgp.py"
    if getattr(args, "predictor_source", None):
        predictor_path = args.predictor_source.resolve()
        spec = importlib.util.spec_from_file_location("frozen_predictor", predictor_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        FeatureSgpParams, FeatureSgpPredictor = module.FeatureSgpParams, module.FeatureSgpPredictor
    import torch
    torch.set_num_threads(1)
    manifest_path = args.dataset.resolve()
    catalog = json.loads(manifest_path.read_text())
    datasets = catalog["datasets"]
    ids = [item["id"] for item in datasets]
    if len(set(ids)) != len(ids) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in ids):
        raise ValueError("dataset IDs must be unique ASCII slugs")
    if args.split != "all":
        datasets = [item for item in datasets if item["split"] == args.split]
    if not datasets:
        raise ValueError("no datasets selected")
    independent_units = len({item.get("cluster_id", item["id"]) for item in datasets})
    params = FeatureSgpParams(device=args.device, training_steps=args.training_steps)
    conditions = [("random", s) for s in (0.30, 0.60)] + [("block", s) for s in (0.8, 1.6)] + [
        ("frontier", s) for s in (0.25, 0.50)]
    if args.protocol == "public15":
        conditions = [("random", 0.15), ("block", 1.6), ("frontier", 0.50)]
    jobs = []
    inputs = {}
    for di, item in enumerate(datasets):
        if item.get("unit") != "m" or not item.get("frame_id"):
            raise ValueError("explicit metric unit and registered frame_id are required")
        if item.get("reference_kind") not in ("analytic_ground", "surveyed_ground", "dense_cloud_proxy"):
            raise ValueError("reference_kind must distinguish independent truth from dense-cloud proxy")
        paths = {key: (manifest_path.parent / item[key]).resolve() for key in ("points", "reference")}
        inputs[item["id"]] = {key: {"path": str(path), "sha256": digest(path)} for key, path in paths.items()}
        for ci, (kind, severity) in enumerate(conditions):
            for repeat in range(args.repeats):
                jobs.append({"id": f"{item['id']}__{kind}_{ci}__r{repeat:03d}",
                    "dataset_id": item["id"], "family": item["family"], "split": item["split"],
                    "reference_kind": item["reference_kind"], "kind": kind, "severity": severity,
                    "mask_seed": args.seed + di * 10000 + ci * 100 + repeat,
                    "noise_seed": args.seed + 10000000 + di * 10000 + ci * 100 + repeat})
    settings = {"dataset_sha256": digest(manifest_path), "inputs": inputs,
        "parameters": asdict(params), "noise_std_m": args.noise_std,
        "predictor_sha256": digest(predictor_path), "predictor_path": str(predictor_path),
        "runner_sha256": digest(__file__)}
    if args.dry_run:
        print(json.dumps({"world_count": independent_units, "dataset_count": len(datasets), "case_count": len(jobs),
                          "settings": settings, "jobs": jobs}, ensure_ascii=False, indent=2))
        return 0
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    saved = output / "batch_manifest.json"
    if saved.exists():
        if not args.resume:
            raise ValueError("output exists; use --resume or a new output directory")
        record = json.loads(saved.read_text())
        if record["settings"] != settings or record["jobs"] != jobs:
            raise ValueError("resume requires unchanged method, input files and job plan")
    else:
        if args.resume:
            raise ValueError("resume requires an existing batch manifest")
        record = {"created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "method": "our Feature SGP only; no external baseline or online-risk release",
            "scope": "ground-cloud files -> conditional height posterior; excludes ROS front-end and navigation",
            "world_count": independent_units, "dataset_count": len(datasets),
            "independent_unit": "catalog cluster_id, or dataset id when no grouping is supplied",
            "planned_cases": len(jobs), "settings": settings,
            "jobs": jobs, "runs": []}
        atomic_json(saved, record)
    completed = {row["id"] for row in record["runs"]}
    # Per-case atomic records survive interruption between manifest checkpoints.
    for job in jobs:
        path = output / "cases" / job["id"] / "metrics.json"
        if job["id"] not in completed and path.exists():
            row = json.loads(path.read_text())
            if any(row.get(key) != value for key, value in job.items()):
                raise ValueError("checkpoint case does not match the frozen job plan")
            if row["status"] == "ok" and not path.with_name("prediction.npz").exists():
                raise ValueError("successful checkpoint is missing prediction arrays")
            record["runs"].append(row)
            completed.add(job["id"])
    cached_dataset = None
    cached_data = None
    for index, job in enumerate(jobs, 1):
        if job["id"] in completed:
            continue
        row = dict(job)
        case_output = output / "cases" / job["id"]
        case_output.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        try:
            if job["dataset_id"] != cached_dataset:
                paths = inputs[job["dataset_id"]]
                cached_data = (read_points(paths["points"]["path"]), read_reference(paths["reference"]["path"]))
                cached_dataset = job["dataset_id"]
            points, ref = cached_data
            keep = keep_cells(ref["query_xy"], job["kind"], job["severity"], job["mask_seed"])
            observed, observed_cells = partial_input(points, ref, keep, job["noise_seed"], args.noise_std)
            if not (ref["valid"] & ~observed_cells).any():
                raise NoHeldOutReference("no held-out reference cells")
            # No reference height is passed to either model API.
            prediction = FeatureSgpPredictor(params).fit(observed).predict(ref["query_xy"])
            row.update(score(prediction, ref, observed_cells))
            row.update(status="ok", device=prediction.device, fit_ms=prediction.fit_ms,
                predict_ms=prediction.predict_ms, source_point_count=len(points),
                input_point_count=len(observed), training_points=prediction.training_points)
            np.savez_compressed(case_output / "prediction.npz", observed_xyz=observed,
                query_xy=ref["query_xy"], reference_height=ref["height"], reference_valid=ref["valid"],
                observed_cells=observed_cells, mean=prediction.mean,
                latent_var=prediction.latent_var, observation_var=prediction.observation_var,
                nearest_distance=prediction.nearest_distance, supported=prediction.supported,
                accepted=prediction.accepted, shape=np.asarray(ref["shape"]))
        except NoHeldOutReference as exc:
            row.update(status="not_evaluable", unknown_cells=0, reason=str(exc))
        except Exception as exc:
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        row["wall_ms_including_io_and_scoring"] = (time.perf_counter() - start) * 1000
        atomic_json(case_output / "metrics.json", row)
        record["runs"].append(row)
        if index % 20 == 0 or index == len(jobs) or row["status"] == "failed":
            atomic_json(saved, record)
        if index % 50 == 0 or row["status"] == "failed":
            print(f"[cloud batch] {index}/{len(jobs)} {job['id']} {row['status']}", flush=True)
    job_order = {job["id"]: index for index, job in enumerate(jobs)}
    record["runs"].sort(key=lambda row: job_order[row["id"]])
    atomic_json(saved, record)
    failures = sum(row["status"] == "failed" for row in record["runs"])
    not_evaluable = sum(row["status"] == "not_evaluable" for row in record["runs"])
    print(f"complete: {len(record['runs'])}/{len(jobs)} cases, {len(datasets)} datasets / "
          f"{independent_units} source groups, {failures} failures, {not_evaluable} not evaluable; {output}")
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("make-synthetic", help="create independent ground-only files and exact references")
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--scenes-per-family", type=int, default=2)
    create.add_argument("--size", type=int, default=32)
    create.add_argument("--resolution", type=float, default=0.20)
    create.add_argument("--seed", type=int, default=2026101200)
    create.add_argument("--split", choices=("development", "calibration", "test"), default="development")
    run = commands.add_parser("run", help="current-method-only batch evaluation")
    run.add_argument("--dataset", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    run.add_argument("--split", choices=("development", "calibration", "test", "all"), default="development")
    run.add_argument("--repeats", type=int, default=2)
    run.add_argument("--seed", type=int, default=20261012)
    run.add_argument("--noise-std", type=float, default=0.02)
    run.add_argument("--training-steps", type=int, default=12)
    run.add_argument("--predictor-source", type=Path,
                     help="frozen local source for paired engineering regression, not an external baseline")
    run.add_argument("--protocol", choices=("default", "public15"), default="default",
                     help="public15: 15%% random hidden cells plus hole/frontier stress; not physical 15%% ray occlusion")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "make-synthetic":
            if args.scenes_per_family < 1 or args.size < 16 or args.resolution <= 0:
                raise ValueError("invalid synthetic grid dimensions")
            make_synthetic(args)
            return 0
        if args.repeats < 1 or args.noise_std < 0 or args.training_steps < 0:
            raise ValueError("invalid repeat/noise/training setting")
        return run_batch(args)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
