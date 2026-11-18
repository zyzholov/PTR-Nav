#!/usr/bin/env python3
"""Paired local column-detector regression; synthetic observations, not contacts."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

from pointcloud_batch import ROOT, atomic_json, digest
from fsgp_bgk_se2.obstacle_columns import vertical_columns
from public_cloud_study import binary_counts

FAMILIES = ("dense_pole", "sparse_pole", "crosscell_pole", "wall",
            "canopy", "ground", "outlier", "step_ground")


def make_case(family, seed, size=48, resolution=.2):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    xy = np.c_[(xx.ravel() + .5) * resolution, (yy.ravel() + .5) * resolution]
    slope = rng.uniform(-.45, .45, 2)
    height = xy @ slope + .04 * np.sin(xy[:, 0])
    if family == "step_ground":
        height += (xy[:, 0] > size * resolution / 2) * rng.uniform(.2, .5)
    measured = height + rng.normal(0, .01, len(height))
    visible = rng.random(len(height)) > rng.uniform(.1, .65)
    points = np.c_[xy[visible], measured[visible]].astype(np.float32)
    observed_ground = np.where(visible, measured, np.nan).reshape(size, size).astype(np.float32)
    target = np.zeros((size, size), bool)
    valid = visible.reshape(size, size).copy()
    ix, iy = rng.integers(5, size - 6, 2)
    center = iy * size + ix
    if family in ("dense_pole", "sparse_pole", "crosscell_pole", "wall", "canopy", "outlier"):
        if family == "dense_pole":
            cells, upper = [center] * 12, np.linspace(.35, 1.3, 12)
        elif family == "sparse_pole":
            cells, upper = [center] * 2, [.6, .64]
        elif family == "crosscell_pole":
            cells, upper = [center, center + 1], [.6, .6]
        elif family == "wall":
            cells, upper = list(range(center - 4, center + 5)), rng.uniform(.5, 1.2, 9)
        elif family == "canopy":
            cells, upper = [center, center + 1], [3.0, 3.1]
        else:
            cells, upper = [center], [.8]
        upper_points = np.c_[xy[cells], height[cells] + upper].astype(np.float32)
        points = np.r_[points, upper_points]
        valid.ravel()[cells] = True
        if family not in ("canopy", "outlier"):
            target.ravel()[cells] = True
    return points, observed_ground, target, valid, points[:int(visible.sum())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--before-source", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--seed", type=int, default=80261013)
    args = parser.parse_args()
    if args.repeats < 1 or (args.output.exists() and any(args.output.iterdir())):
        parser.error("positive repetitions and a new empty output are required")
    import cupy as cp
    spec = importlib.util.spec_from_file_location("columns_before", args.before_source)
    before = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(before)
    args.output.mkdir(parents=True)
    rows = []
    for fi, family in enumerate(FAMILIES):
        for repeat in range(args.repeats):
            seed = args.seed + fi * 10000 + repeat
            points, observed, target, valid, clear_cloud = make_case(family, seed)
            masks = {}
            for version, function in (("before", before.vertical_columns), ("after", vertical_columns)):
                for backend, xp in (("cpu", np), ("cupy", cp)):
                    extra = {"ground_elevation": observed} if version == "after" else {}
                    if xp is cp:
                        cp.cuda.Stream.null.synchronize()
                    start = time.perf_counter()
                    mask, _ = function(points, target.shape, (0, 0), .2, xp=xp, **extra)
                    if xp is cp:
                        mask = cp.asnumpy(mask)
                        cp.cuda.Stream.null.synchronize()
                    elapsed = (time.perf_counter() - start) * 1000
                    clear, _ = function(clear_cloud, target.shape, (0, 0), .2, xp=xp, **extra)
                    if xp is cp:
                        clear = cp.asnumpy(clear)
                    masks[(version, backend)] = mask
                    rows.append({"family": family, "geometry_seed": seed, "repeat": repeat,
                        "version": version, "backend": backend, "latency_ms": elapsed,
                        "clear_scan_false_cells": int(clear.sum()),
                        **binary_counts(mask, target, valid)})
            for version in ("before", "after"):
                if not np.array_equal(masks[(version, "cpu")], masks[(version, "cupy")]):
                    raise AssertionError(f"CPU/GPU mismatch: {family} / {seed} / {version}")
            if repeat == 0:
                np.savez_compressed(args.output / (family + ".npz"), xyz=points, ground=observed,
                    target=target, valid=valid, before=masks[("before", "cupy")], after=masks[("after", "cupy")])
        print(f"[columns] {family}: {args.repeats} independent geometry/noise seeds", flush=True)
        atomic_json(args.output / "results.json", {"scope": __doc__, "date": "2026-10-13",
            "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "before_sha256": digest(args.before_source),
            "after_sha256": digest(ROOT / "src/fsgp_bgk_se2/fsgp_bgk_se2/obstacle_columns.py"),
            "runner_sha256": digest(__file__), "independent_cases": len(FAMILIES) * args.repeats,
            "grid": {"size": 48, "resolution_m": .2},
            "truth": "analytic point associations; ground input contains measured visible cells only",
            "runs": rows})
    print(f"[columns] complete {len(rows)} paired detector calls plus equal clearing calls", flush=True)


if __name__ == "__main__":
    main()
