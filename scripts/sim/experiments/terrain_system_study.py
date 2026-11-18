"""Versioned, source-audited terrain experiments with explicit evaluation domains.

The GP receives only measured ground XYZ. Reference arrays are used for scoring
after prediction. Navigation risk uses measured elevations only, as configured
by default online; diagnostic GP predictions never silently fill unknown cells.
"""

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.spatial import ConvexHull
import torch

from pointcloud_batch import ROOT, atomic_json, digest
from paper_clouds import grid, height_at, raster
from paper_scale_evaluate import partial_points, footprint_reference, complete_generated_collision
from paper_budget_campaign import require_qualified_sources
from controlled_terrain_study import specification, raw_cloud, surface, footprint_truth, environment
from terrain_risk_benchmark import params_from_config
from thesis_statistics import height_diagnostics, threshold_counts
from fsgp_bgk_se2.feature_sgp import FeatureSgpParams, FeatureSgpPredictor
from fsgp_bgk_se2.bgk_seb import BgkSebEstimator
from fsgp_bgk_se2.obstacle_columns import vertical_columns, preserve_collision_risk

BASE = ROOT / "log/terrain_system_v3_20261015"
PROTOCOL = ROOT / "scripts/sim/experiments/protocols/terrain_study_v3.json"


def fingerprints():
    names = ["scripts/sim/experiments/" + n for n in (
        "terrain_system_study.py", "controlled_terrain_study.py", "paper_clouds.py",
        "paper_scale_evaluate.py", "terrain_risk_benchmark.py", "thesis_statistics.py")]
    names += ["src/fsgp_bgk_se2/fsgp_bgk_se2/" + n for n in (
        "feature_sgp.py", "bgk_seb.py", "obstacle_columns.py")]
    names += ["src/fsgp_bgk_se2/config/hunter_fastlio.yaml"]
    return {name: digest(ROOT/name) for name in names}


def evaluation_domains(reference, full_observed, observed):
    reference = np.asarray(reference).ravel()
    full = np.asarray(full_observed, bool).ravel()
    visible = np.asarray(observed, bool).ravel()
    if full.shape != visible.shape or full.shape != reference.shape:
        raise ValueError("domain fields must have identical shapes")
    if (visible & ~full).any():
        raise ValueError("masking cannot add observed cells")
    known = np.isfinite(reference)
    return {"artificial": known & full & ~visible,
            "natural": known & ~full,
            "hidden": known & ~visible}


def domain_metrics(prediction, reference, mask, nominal):
    mask = np.asarray(mask, bool) & np.isfinite(reference)
    n = int(mask.sum())
    if not n:
        return dict(n=0, rmse_m=None, mae_m=None, p95_m=None, bias_m=None,
                    picp=None, width_m=None, crps_m=None)
    error = prediction.mean[mask].astype(float)-reference[mask]
    d = height_diagnostics(prediction.mean, prediction.latent_var,
        np.where(mask, reference, np.nan), np.zeros(len(mask), bool), nominal)
    return dict(n=n, rmse_m=float(np.sqrt(np.mean(error**2))),
        mae_m=float(np.abs(error).mean()), p95_m=float(np.quantile(np.abs(error), .95)),
        bias_m=float(error.mean()), picp=(d["coverage_counts"]/n).tolist(),
        width_m=(d["width_sums"]/n).tolist(), crps_m=d["crps_sum_m"]/n)


def risk_metrics(scores, reference, unsafe, support, threshold):
    counts = threshold_counts(scores, unsafe, support, [threshold])[0]
    tp, fn, fp, tn = map(int, counts)
    error = scores[support]-reference[support]
    return dict(n=int(support.sum()), tp=tp, fn=fn, fp=fp, tn=tn,
        fnr=fn/(tp+fn) if tp+fn else None, fpr=fp/(fp+tn) if fp+tn else None,
        recall=tp/(tp+fn) if tp+fn else None,
        precision=tp/(tp+fp) if tp+fp else None,
        mae=float(np.abs(error).mean()) if len(error) else None)


def prepare(args, protocol):
    target = args.output/"catalog.json"
    if target.exists():
        raise ValueError("preserve frozen catalog; use a new output directory")
    source = args.output/"generated/dataset_manifest.json"
    original = ROOT/"log/terrain_diagnosis_20261015/regression_v2_catalog.json"
    generated = json.loads(source.read_text())["datasets"]
    reused = [d for d in json.loads(original.read_text())["datasets"] if d["family"] in ("road", "indoor")]
    prior_hashes = set()
    for path in (ROOT/"log/paper_aligned_20261015/catalog.json", original):
        prior_hashes.update(d["geometry_sha256"] for d in json.loads(path.read_text())["datasets"] if "geometry_sha256" in d)
    if prior_hashes & {d["geometry_sha256"] for d in generated}:
        raise ValueError("new generated geometry overlaps prior validation")
    rows = []
    for src, items, reuse in ((source, generated, False), (original, reused, True)):
        for item in items:
            row = dict(item)
            row["path"] = os.path.relpath((src.parent/item["path"]).resolve(), args.output.resolve())
            row["reused_public_source"] = reuse
            rows.append(row)
    require_qualified_sources(rows)
    counts = Counter(d["family"] for d in rows)
    p = protocol["public"]
    if counts != Counter(hilly=p["generated_per_family"], forest=p["generated_per_family"],
                         ruin=p["generated_per_family"], road=p["reused_corrected_road_inputs"],
                         indoor=p["reused_indoor_inputs"]):
        raise ValueError("catalog differs from predeclared source counts")
    atomic_json(target, dict(study_id=protocol["study_id"], datasets=rows,
        protocol_sha256=digest(args.protocol), provenance={str(p): digest(p) for p in (source, original)},
        source_groups={f: len({d["cluster_id"] for d in rows if d["family"] == f}) for f in counts}))
    print("[study catalog]", dict(counts), flush=True)


def source_cases(args, protocol):
    path = args.output/"catalog.json"
    catalog = json.loads(path.read_text())
    if catalog["protocol_sha256"] != digest(args.protocol):
        raise ValueError("protocol changed after source freeze")
    params = params_from_config("cupy", protocol["resolution_m"], protocol["yaw_bins"])
    for index, item in enumerate(catalog["datasets"]):
        source = path.parent/item["path"]
        if digest(source) != item["sha256"]:
            raise ValueError("source archive changed")
        with np.load(source) as a:
            case = {k: a[k] for k in a.files if k != "full_xyz"}
        shape = tuple(case["shape"])
        xy = grid(shape, case["query_origin"], protocol["resolution_m"])
        case["truth_collision"] = complete_generated_collision(case, item)
        reference, ids, oracle, unsafe, support = footprint_reference(
            case, xy, params, protocol["seed"]+index)
        yield dict(item=item, raw=case["xyz"], ground=case["is_ground"], shape=shape,
            origin=case["query_origin"], xy=xy, reference=reference, indices=ids,
            reference_risk=oracle, unsafe=unsafe, risk_supported=support)


def mechanism_cases(protocol):
    p = protocol["mechanism"]
    size = p["size"]; res = protocol["resolution_m"]
    shape = (size, size); origin = np.full(2, -size*res/2)
    xy = grid(shape, origin, res)
    axis = np.arange(size//2-10, size//2+11, 2)
    rr, cc = np.meshgrid(axis, axis); ids = (rr*size+cc).ravel()
    params = params_from_config("cupy", res, protocol["yaw_bins"])
    for fi, family in enumerate(p["families"]):
        for index in range(p["seeds_per_family"]):
            seed = protocol["seed"]+fi*10000+index
            spec = specification(family, seed)
            raw, count = raw_cloud(spec, xy)
            reference = surface(spec, xy)
            oracle, unsafe, slopes, spans = footprint_truth(spec, xy[ids], params)
            geometry_hash = hashlib.sha256(raw.tobytes()).hexdigest()
            item = dict(id=f"{family}_{index:03d}", family=family,
                cluster_id="flat_control" if family == "flat" else f"analytic_{family}_{seed}",
                source_kind="analytic", sha256=geometry_hash, geometry_sha256=geometry_hash,
                spec=spec, reference_kind="dense_analytic_footprint")
            yield dict(item=item, raw=raw, ground=np.arange(len(raw)) < count,
                shape=shape, origin=origin, xy=xy, reference=reference,
                indices=ids, reference_risk=oracle, unsafe=unsafe,
                risk_supported=np.ones_like(unsafe), reference_pose=slopes, reference_span=spans)


def evaluate(case, condition, seed, protocol, gp, estimator, folder):
    import cupy as cp
    shape, origin, xy = case["shape"], case["origin"], case["xy"]
    item = case["item"]; res = protocol["resolution_m"]
    raw = case["raw"]
    # The common helper's public windows are 20.8m; analytic windows are smaller.
    mask_origin = np.asarray(origin)+(np.asarray(shape[::-1])*res-20.8)/2
    keep = partial_points(raw, condition, mask_origin, seed)
    noisy = raw.copy()
    noisy[:, 2] += np.random.default_rng(seed+100000).normal(0, protocol["noise_std_m"], len(raw))
    measured = raster(noisy[keep & case["ground"]], shape, origin, res)
    full = raster(raw[case["ground"]], shape, origin, res)
    observed, full_observed = np.isfinite(measured), np.isfinite(full)
    variance = np.full(shape, protocol["noise_std_m"]**2, np.float32)
    ground = noisy[keep & case["ground"]]
    cp.cuda.Stream.null.synchronize(); torch.cuda.synchronize()
    started = time.perf_counter()
    torch.manual_seed(seed)
    prediction = FeatureSgpPredictor(gp).fit(ground).predict(xy)
    torch.cuda.synchronize()
    gp_wall = (time.perf_counter()-started)*1000
    cp.cuda.Stream.null.synchronize(); started = time.perf_counter()
    occupied, _ = vertical_columns(noisy[keep], shape, origin, res, ground_elevation=measured, xp=cp)
    occupied = cp.asnumpy(occupied)
    result = estimator.compute(measured, variance, observed, [occupied.astype(np.float32)], observed_elevation=measured)
    preserve_collision_risk(result, occupied)
    cp.cuda.Stream.null.synchronize()
    risk_wall = (time.perf_counter()-started)*1000
    if prediction.device != "cuda" or result.backend != "cupy":
        raise ValueError("explicit GPU execution must not silently fall back")
    ref = case["reference"]
    domains = evaluation_domains(ref, full_observed, observed)
    metrics = {}
    nominal = np.asarray(protocol["nominal_levels"])
    for label, selected in domains.items():
        metrics[label] = domain_metrics(prediction, ref, selected, nominal)
        metrics[label+"_accepted"] = domain_metrics(prediction, ref, selected & prediction.accepted, nominal)
    bins = protocol["distance_edges_m"]
    distance = [domain_metrics(prediction, ref, domains["hidden"] &
        (prediction.nearest_distance >= lo) & (prediction.nearest_distance < hi), nominal)
        for lo, hi in zip(bins[:-1], bins[1:])]
    center = ground[:, :2].mean(axis=0, dtype=np.float64)
    eq = ConvexHull(ground[:, :2].astype(float)-center).equations
    outside = np.any((xy-center)@eq[:, :2].T+eq[:, 2] > 1e-6, axis=1)
    if np.any(outside & prediction.accepted):
        raise ValueError("regression: exterior query accepted")
    scores = np.stack(result.final_risk_slices).reshape(protocol["yaw_bins"], -1)[:, case["indices"]]
    risk = risk_metrics(scores, case["reference_risk"], case["unsafe"],
        case["risk_supported"], protocol["risk_threshold"])
    row = dict(id=item["id"]+"__"+condition, dataset_id=item["id"], family=item["family"],
        cluster_id=item["cluster_id"], source_sha256=item["sha256"], condition=condition,
        case_seed=seed, status="ok", actual_device=prediction.device, actual_backend=result.backend,
        ground_input_points=len(ground), reference_cells=int(np.isfinite(ref).sum()),
        original_observed_cells=int(full_observed.sum()), current_observed_cells=int(observed.sum()),
        accepted_outside_hull=int((prediction.accepted & outside).sum()),
        unreferenced_hidden_cells=int((~np.isfinite(ref) & ~observed.ravel()).sum()),
        gp_wall_ms=gp_wall, risk_wall_ms=risk_wall, fit_ms=prediction.fit_ms,
        predict_ms=prediction.predict_ms, diagnostics_total_ms=gp_wall+risk_wall,
        height=metrics, distance=distance, risk=risk,
        source_kind=item["source_kind"], reused_public_source=item.get("reused_public_source", False),
        spec=item.get("spec"))
    target = folder/(row["id"]+".npz")
    archive = dict(mean=prediction.mean, variance=prediction.latent_var, reference=ref,
        accepted=prediction.accepted, supported=prediction.supported, nearest=prediction.nearest_distance,
        outside_hull=outside, xy=xy, measured=measured, full_observed=full_observed,
        observed=observed, occupied=occupied, predicted_risk=scores,
        reference_risk=case["reference_risk"], unsafe=case["unsafe"],
        risk_supported=case["risk_supported"], query_indices=case["indices"],
        origin=origin, resolution=res, raw_xyz=noisy[keep])
    np.savez_compressed(target, **archive)
    row.update(archive=target.name, archive_sha256=digest(target))
    return row


def run(args, protocol):
    torch.set_num_threads(1)
    gp = FeatureSgpParams(device="cuda")
    params = params_from_config("cupy", protocol["resolution_m"], protocol["yaw_bins"])
    estimator = BgkSebEstimator(params)
    phase = args.phase
    p = protocol[phase]
    settings = dict(study_id=protocol["study_id"], protocol_sha256=digest(args.protocol),
        source_sha256=fingerprints(), phase=phase, gp=asdict(gp), bgk_seb=asdict(params),
        catalog_sha256=digest(args.output/"catalog.json") if phase == "public" else None,
        planned_cases=p["planned_cases"], risk_input="measured_only", unknown_completion=False,
        noise_std_m=protocol["noise_std_m"], nominal=protocol["nominal_levels"],
        distance_edges_m=protocol["distance_edges_m"])
    folder = args.output/phase; path = folder/"results.json"
    folder.mkdir(parents=True, exist_ok=True)
    if path.exists():
        record = json.loads(path.read_text())
        if not args.resume or record["settings"] != settings:
            raise ValueError("resume requires identical frozen sources, protocol and parameters")
    else:
        if any(folder.iterdir()):
            raise ValueError("unidentified existing files")
        record = dict(settings=settings, environment_start=environment(), runs=[])
        atomic_json(path, record)
    rows = {r["id"]: r for r in record["runs"]}
    # Warm both frameworks before the first recorded component timing.
    warm_xy = grid((32, 32), [-3.2, -3.2], .2)
    for i in range(3):
        torch.manual_seed(protocol["seed"]+i)
        FeatureSgpPredictor(gp).fit(np.c_[warm_xy, .05*warm_xy[:, 0]]).predict(warm_xy)
        estimator.compute(np.zeros((32, 32), np.float32), np.full((32, 32), .0004, np.float32),
            np.ones((32, 32), bool), [np.zeros((32, 32), np.float32)])
    cases = source_cases(args, protocol) if phase == "public" else mechanism_cases(protocol)
    for di, case in enumerate(cases):
        for ci, condition in enumerate(p["conditions"]):
            key = case["item"]["id"]+"__"+condition
            checkpoint = folder/(key+".json")
            if checkpoint.exists() and key not in rows:
                if not args.resume:
                    raise ValueError("existing checkpoint requires explicit resume")
                rows[key] = json.loads(checkpoint.read_text())
            if key in rows:
                if rows[key]["status"] != "ok" or digest(folder/rows[key]["archive"]) != rows[key]["archive_sha256"]:
                    raise ValueError("failed or changed checkpoint requires a new batch")
                continue
            seed = protocol["seed"]+di*100+ci
            row = evaluate(case, condition, seed, protocol, gp, estimator, folder)
            atomic_json(checkpoint, row); rows[key] = row
        record["runs"] = list(rows.values()); atomic_json(path, record)
        print(f"[study {phase}] {len(rows)}/{p['planned_cases']}: {case['item']['id']}", flush=True)
    if len(rows) != p["planned_cases"]:
        raise ValueError("incomplete factorial")
    record["environment_end"] = environment(); atomic_json(path, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "public", "mechanism"))
    parser.add_argument("--output", type=Path, default=BASE)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(); protocol = json.loads(args.protocol.read_text())
    if args.phase == "prepare":
        prepare(args, protocol)
    else:
        run(args, protocol)


if __name__ == "__main__":
    main()
