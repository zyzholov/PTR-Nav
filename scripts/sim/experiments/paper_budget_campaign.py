"""Paper-motivated inducing-budget sensitivity, not an external method comparison."""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from pointcloud_batch import ROOT, atomic_json, digest
from paper_clouds import grid, raster
from paper_scale_evaluate import complete_generated_collision, partial_points, footprint_reference, height_metrics, risk_metrics
from terrain_risk_benchmark import params_from_config
from thesis_statistics import height_diagnostics
from controlled_terrain_study import environment
from fsgp_bgk_se2.feature_sgp import FeatureSgpParams, FeatureSgpPredictor
from fsgp_bgk_se2.bgk_seb import BgkSebEstimator
from fsgp_bgk_se2.obstacle_columns import vertical_columns, preserve_collision_risk

FAMILIES = ("hilly", "forest", "ruin", "road", "indoor")
CONDITIONS = ("point15", "hole16", "frontier50")
BUDGETS = (50, 125, 250, 500)


def require_qualified_sources(datasets):
    problems=[]
    for item in datasets:
        if item["family"] in FAMILIES[:3] and item.get("generator_profile")!="metric_v2":
            problems.append(item["id"]+": metric_v2 generation metadata required")
        shader=item.get("shader_parameters",{})
        if item["family"] in FAMILIES[:3] and shader.get("type")==3 and not 0<shader.get("H",0)<1:
            problems.append(item["id"]+": Swiss amplitude gain must decay")
        if item["family"]=="road" and item.get("registration_pose_source")!="semantic_suma":
            problems.append(item["id"]+": verified SemanticKITTI SuMa registration required")
    if problems:
        raise ValueError("unqualified main-experiment sources; keep old data as diagnostics, rebuild into a new catalog: "+"; ".join(problems[:5]))


def stratified_selection(items, family, n):
    groups = defaultdict(list)
    for item in items:
        if item["family"] == family:
            groups[item["cluster_id"]].append(item)
    selected = []
    for i in range(max((len(g) for g in groups.values()), default=0)):
        for group in sorted(groups):
            ordered = sorted(groups[group], key=lambda r: r["id"])
            if i < len(ordered): selected.append(ordered[i])
            if len(selected) == n: return selected
    raise ValueError("not enough source inputs for declared family")


def prepare(generated_path, public_path, output):
    if output.exists(): raise ValueError("preserve previous catalog")
    generated, public = [json.loads(p.read_text()) for p in (generated_path, public_path)]
    previous = {d["geometry_sha256"] for d in public["datasets"] if d["source_kind"].startswith("epfl")}
    fresh = generated["datasets"]
    if len(fresh) != 300 or len({d["geometry_sha256"] for d in fresh}) != 300:
        raise ValueError("need 300 distinct new generated geometries")
    if previous & {d["geometry_sha256"] for d in fresh}:
        raise ValueError("generated sources overlap earlier holdout")
    rows = []
    for family in FAMILIES:
        path = generated_path if family in FAMILIES[:3] else public_path
        catalog = generated if path == generated_path else public
        for original in stratified_selection(catalog["datasets"], family, 100):
            item = dict(original)
            item["path"] = os.path.relpath((path.parent/item["path"]).resolve(), output.parent.resolve())
            item["split"] = "new_generated" if family in FAMILIES[:3] else "public_regression"
            rows.append(item)
    output.parent.mkdir(parents=True, exist_ok=True)
    require_qualified_sources(rows)
    result = {"study_id": "PTR-PAPER-04", "data_revision":"metric_v2_suma", "datasets": rows,
        "sources": {str(p): digest(p) for p in (generated_path, public_path)},
        "groups": {f: len({r["cluster_id"] for r in rows if r["family"] == f}) for f in FAMILIES},
        "scope": "300 new generated geometries and 200 grouped public regression inputs; not 500 independent real scenes"}
    atomic_json(output, result)
    print("[budget catalog]", len(rows), result["groups"], flush=True)


def method_hashes():
    names = ["scripts/sim/experiments/paper_budget_campaign.py", "scripts/sim/experiments/paper_scale_evaluate.py",
        "scripts/sim/experiments/paper_clouds.py", "scripts/sim/experiments/thesis_statistics.py",
        "src/fsgp_bgk_se2/config/hunter_fastlio.yaml"]
    names += ["src/fsgp_bgk_se2/fsgp_bgk_se2/"+n for n in ("feature_sgp.py", "bgk_seb.py", "obstacle_columns.py")]
    return {n: digest(ROOT/n) for n in names}


def budget_order(seed):
    return np.random.default_rng(seed).permutation(BUDGETS).tolist()


def run(catalog_path, output, resume=False, preflight=False):
    import torch
    import cupy as cp
    torch.set_num_threads(1)
    catalog = json.loads(catalog_path.read_text()); datasets = catalog["datasets"]
    require_qualified_sources(datasets)
    if preflight: datasets = [next(d for d in datasets if d["family"] == f) for f in FAMILIES]
    params = params_from_config("cupy", .2, 36); estimator = BgkSebEstimator(params)
    base = FeatureSgpParams(device="cuda", max_train_points=512)
    settings = {"dataset_sha256": digest(catalog_path), "source_sha256": method_hashes(), "seed": 261015902,
        "selected_ids": [r["id"] for r in datasets], "conditions": list(CONDITIONS), "budgets": list(BUDGETS),
        "base_gp": asdict(base), "bgk_seb": asdict(params), "input_noise_m": .02, "preflight": preflight,
        "reference": "finite-weight v2 independent dense footprint/collision proxy; not tire contact truth",
        "scope": "max_train_points=512 held fixed for all M; local conditioning on; parameter sensitivity, not default deployment or ablation"}
    output.mkdir(parents=True, exist_ok=True); path = output/"results.json"
    if path.exists():
        record = json.loads(path.read_text())
        if not resume or record["settings"] != settings: raise ValueError("strict resume or new directory required")
    else:
        if any(output.iterdir()): raise ValueError("unidentified earlier output")
        record = {"study_id": "PTR-PAPER-04", "settings": settings, "environment_start": environment(),
            "planned_cases": len(datasets)*len(CONDITIONS)*len(BUDGETS), "runs": []}
    rows = {r["id"]: r for r in record["runs"]}; atomic_json(path, record)
    for di, item in enumerate(datasets):
        source = (catalog_path.parent/item["path"]).resolve()
        if digest(source) != item["sha256"]: raise ValueError("source archive changed")
        with np.load(source, allow_pickle=False) as a: case = {k:a[k] for k in a.files if k != "full_xyz"}
        case["truth_collision"] = complete_generated_collision(case, item)
        xy = grid(tuple(case["shape"]), case["query_origin"], .2)
        try:
            reference, ids, oracle, unsafe, supported = footprint_reference(case, xy, params, settings["seed"]+di)
            reference_error = None
        except ValueError as exc:
            reference_error = str(exc)
        for ci, condition in enumerate(CONDITIONS):
            seed = settings["seed"]+di*100+ci
            keep = partial_points(case["xyz"], condition, case["query_origin"], seed)
            raw = case["xyz"][keep].copy()
            raw[:,2] += np.random.default_rng(seed+100000).normal(0,.02,len(raw))
            ground = raw[case["is_ground"][keep]]
            measured = raster(ground, tuple(case["shape"]), case["query_origin"], .2)
            valid = np.isfinite(measured); variance = np.full_like(measured,.0004)
            reference_path = output/(item["id"]+"__"+condition+"__reference.npz")
            if reference_error is None and not reference_path.exists():
                np.savez_compressed(reference_path, reference_height=reference.reshape(measured.shape),
                    query_indices=ids, reference_risk=oracle, unsafe=unsafe, supported=supported,
                    observed=valid, measured=measured, raw_xyz=raw, query_origin=case["query_origin"])
            for order_index, m in enumerate(budget_order(seed)):
                name = item["id"]+"__"+condition+f"__M{m}"
                checkpoint = output/(name+".json")
                if name not in rows and resume and checkpoint.exists(): rows[name] = json.loads(checkpoint.read_text())
                if name in rows:
                    r = rows[name]
                    if r["requested_M"] != m or r["case_seed"] != seed: raise ValueError("checkpoint mismatch")
                    if r["status"] == "ok":
                        if digest(output/r["archive"]) != r["archive_sha256"]: raise ValueError("prediction changed")
                        if digest(output/r["reference"]) != r["reference_sha256"]: raise ValueError("reference changed")
                    continue
                row = {"id": name, "dataset_id": item["id"], "family": item["family"], "split": item["split"],
                    "cluster_id": item["cluster_id"], "condition": condition, "requested_M": m, "case_seed": seed,
                    "budget_order": order_index, "source_sha256": item["sha256"],
                    "input_points": len(raw), "ground_input_points": len(ground)}
                try:
                    if reference_error: raise ValueError("reference geometry: "+reference_error)
                    torch.manual_seed(seed); torch.cuda.synchronize(); cp.cuda.Stream.null.synchronize()
                    start = time.perf_counter()
                    predictor = FeatureSgpPredictor(FeatureSgpParams(**{**asdict(base), "inducing_points": m})).fit(ground)
                    prediction = predictor.predict(xy)
                    risk_start = time.perf_counter()
                    refined = np.where(valid & prediction.accepted.reshape(measured.shape),
                        measured+(variance/(variance+prediction.latent_var.reshape(measured.shape)))*
                        (prediction.mean.reshape(measured.shape)-measured), measured)
                    occupied,_ = vertical_columns(raw, measured.shape, case["query_origin"], .2, ground_elevation=measured, xp=cp)
                    occupied = cp.asnumpy(occupied)
                    risk = estimator.compute(refined, variance, valid, [occupied.astype(np.float32)], observed_elevation=measured)
                    preserve_collision_risk(risk, occupied)
                    cp.cuda.Stream.null.synchronize(); torch.cuda.synchronize()
                    stop = time.perf_counter(); scores = np.stack(risk.final_risk_slices).reshape(36,-1)[:,ids]
                    if prediction.device != "cuda" or risk.backend != "cupy": raise RuntimeError("GPU fallback")
                    h = height_metrics(prediction, reference, valid.ravel())
                    diagnostics = height_diagnostics(prediction.mean,prediction.latent_var,reference,valid.ravel(),np.array([.95]))
                    hidden = np.isfinite(reference) & ~valid.ravel()
                    row.update(status="ok", actual_device=prediction.device, actual_backend=risk.backend,
                        effective_M=int(predictor.model.covar_module.inducing_points.shape[0]),
                        train_points=prediction.training_points,
                        fit_ms=prediction.fit_ms, predict_ms=prediction.predict_ms, risk_ms=(stop-risk_start)*1000,
                        total_wall_ms=(stop-start)*1000, local_hidden_fraction=float(prediction.local_conditioned[hidden].mean()) if hidden.any() else None,
                        hidden_crps_m=diagnostics["crps_sum_m"]/diagnostics["n"] if diagnostics["n"] else None,
                        **h, **risk_metrics(scores,oracle,unsafe,supported))
                    archive = output/(name+".npz")
                    np.savez_compressed(archive, mean=prediction.mean, variance=prediction.latent_var, accepted=prediction.accepted,
                        local_conditioned=prediction.local_conditioned, risk=scores)
                    row.update(archive=archive.name, archive_sha256=digest(archive), reference=reference_path.name,
                        reference_sha256=digest(reference_path))
                except ValueError as exc:
                    row.update(status="not_evaluable" if "reference geometry" in str(exc) or "eight" in str(exc) else "failed", reason=str(exc))
                except Exception as exc:
                    row.update(status="failed",reason=f"{type(exc).__name__}: {exc}")
                atomic_json(checkpoint,row); rows[name] = row
        record["runs"] = list(rows.values())
        atomic_json(path,record)
        if (di+1)%10 == 0 or preflight:
            print(f"[budget] {di+1}/{len(datasets)} inputs, {len(rows)}/{record['planned_cases']} computations, {dict(Counter(r['status'] for r in rows.values()))}", flush=True)
    record["environment_end"] = environment(); atomic_json(path,record)
    if len(rows) != record["planned_cases"]: raise ValueError("partial campaign")
    return int(any(r["status"] == "failed" for r in rows.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("phase", choices=("prepare","run"))
    parser.add_argument("--generated", type=Path, default=ROOT/"log/paper_aligned_20261015/generated/dataset_manifest.json")
    parser.add_argument("--public", type=Path, default=ROOT/"log/optimization_navigation_20261013_v2/holdout_clouds/evaluation_manifest.json")
    parser.add_argument("--catalog", type=Path, default=ROOT/"log/paper_aligned_20261015/catalog.json")
    parser.add_argument("--output", type=Path, default=ROOT/"log/paper_aligned_20261015/budget")
    parser.add_argument("--resume", action="store_true"); parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.phase == "prepare": prepare(args.generated,args.public,args.catalog); return 0
    return run(args.catalog,args.output,args.resume,args.preflight)


if __name__ == "__main__": raise SystemExit(main())
