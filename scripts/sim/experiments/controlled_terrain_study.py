"""Frozen new terrain/noise and scaling experiments; never overwrite earlier batches."""

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch

from pointcloud_batch import ROOT, atomic_json, digest
from paper_clouds import grid, raster
from terrain_risk_benchmark import params_from_config
from thesis_statistics import height_diagnostics
from fsgp_bgk_se2.feature_sgp import FeatureSgpParams, FeatureSgpPredictor
from fsgp_bgk_se2.bgk_seb import BgkSebEstimator
from fsgp_bgk_se2.obstacle_columns import vertical_columns, preserve_collision_risk


NAMES = {"flat": "平地", "slope_safe": "缓坡", "slope_boundary": "坡度边界", "waves": "连续起伏",
    "ridge": "窄脊", "step_low": "阈下台阶", "step_high": "阈上台阶", "column": "细柱",
    "random30": "随机删点30%", "hole16": "1.6米孔洞", "frontier50": "单侧隐藏50%"}
NOMINAL = np.array([.5, .8, .9, .95, .99])


def specification(family, seed):
    rng = np.random.default_rng(seed)
    return {"family": family, "seed": seed, "angle": float(rng.uniform(-np.pi, np.pi)),
        "offset": float(rng.uniform(-.35, .35)), "slope_safe": float(rng.uniform(.1, .3)),
        "slope_boundary": float(rng.uniform(.43, .60)), "amplitude": float(rng.uniform(.06, .14)),
        "wavelength": float(rng.uniform(2.5, 4.5)), "ridge_height": float(rng.uniform(.12, .30)),
        "ridge_width": float(rng.uniform(.10, .30)), "step_low": float(rng.uniform(.08, .17)),
        "step_high": float(rng.uniform(.19, .35)), "column_radius": float(rng.uniform(.06, .12)),
        "column_height": float(rng.uniform(.8, 1.4))}


def surface(spec, xy):
    xy = np.asarray(xy)
    c, s = np.cos(spec["angle"]), np.sin(spec["angle"])
    u = xy[..., 0]*c + xy[..., 1]*s
    v = -xy[..., 0]*s + xy[..., 1]*c
    kind = spec["family"]
    if kind in ("flat", "column"):
        return np.zeros_like(u)
    if kind in ("slope_safe", "slope_boundary"):
        return np.tan(spec[kind])*u
    if kind == "waves":
        return spec["amplitude"]*np.sin(2*np.pi*u/spec["wavelength"])*np.cos(v/1.5)
    if kind == "ridge":
        return spec["ridge_height"]*np.exp(-.5*((u-spec["offset"])/spec["ridge_width"])**2)
    if kind in ("step_low", "step_high"):
        return np.where(u >= spec["offset"], spec[kind], 0.)
    raise ValueError("unknown analytic family")


def missing_mask(xy, pattern, seed):
    if pattern == "random30":
        return np.random.default_rng(seed).random(len(xy)) >= .3
    if pattern == "hole16":
        return (np.abs(xy) >= .8).any(axis=1)
    if pattern == "frontier50":
        return xy[:, 0] <= 0
    raise ValueError("unknown missingness")


def raw_cloud(spec, xy):
    ground = np.c_[xy, surface(spec, xy)].astype(np.float32)
    if spec["family"] != "column":
        return ground, len(ground)
    angle, z = np.meshgrid(np.arange(48)*2*np.pi/48, np.linspace(.2, spec["column_height"], 16))
    r = spec["column_radius"]
    column = np.c_[r*np.cos(angle.ravel())+spec["offset"], r*np.sin(angle.ravel()), z.ravel()]
    return np.r_[ground, column].astype(np.float32), len(ground)


def footprint_truth(spec, centers, params):
    lx, ly = np.meshgrid(np.linspace(-params.footprint_back-params.footprint_margin,
        params.footprint_front+params.footprint_margin, 13),
        np.linspace(-params.footprint_half_width-params.footprint_margin,
        params.footprint_half_width+params.footprint_margin, 11))
    local = np.c_[lx.ravel(), ly.ravel()]
    design = np.c_[local, np.ones(len(local))]; inverse = np.linalg.pinv(design)
    scores, hazards, slopes, spans = [], [], [], []
    for yaw in (np.arange(params.yaw_bins)+.5)*2*np.pi/params.yaw_bins:
        c, s = np.cos(yaw), np.sin(yaw)
        rotation = np.array([[c, -s], [s, c]])
        queries = centers[:, None, :] + (local@rotation.T)[None, :, :]
        z = surface(spec, queries)
        coefficients = z@inverse.T; residual = z-coefficients@design.T
        pitch, roll = np.arctan(np.abs(coefficients[:, 0])), np.arctan(np.abs(coefficients[:, 1]))
        span = np.ptp(residual, axis=1)
        jump = np.zeros(len(centers)); collision = np.zeros(len(centers), bool)
        if spec["family"].startswith("step_"):
            # Analytic edge crossing detects a true discontinuity, not only plane-fit residual.
            u = queries[..., 0]*np.cos(spec["angle"])+queries[..., 1]*np.sin(spec["angle"])
            crossed = (u.min(axis=1) < spec["offset"]) & (u.max(axis=1) >= spec["offset"])
            jump = np.where(crossed, spec[spec["family"]], 0.)
        if spec["family"] == "column":
            relative = (np.array([spec["offset"], 0.])-centers)@rotation
            closest_x = np.clip(relative[:, 0], -params.footprint_back-params.footprint_margin,
                params.footprint_front+params.footprint_margin)
            closest_y = np.clip(relative[:, 1], -params.footprint_half_width-params.footprint_margin,
                params.footprint_half_width+params.footprint_margin)
            collision = np.hypot(relative[:, 0]-closest_x, relative[:, 1]-closest_y) <= spec["column_radius"]
        rough = np.maximum(span, jump)
        risk = np.maximum.reduce((np.clip((pitch-params.pitch_free_rad)/(params.max_pitch_rad-params.pitch_free_rad), 0, 1),
            np.clip((roll-params.roll_free_rad)/(params.max_roll_rad-params.roll_free_rad), 0, 1),
            np.clip((rough-params.step_free_m)/(params.max_step_m-params.step_free_m), 0, 1)))
        scores.append(np.where(collision, 1., risk))
        hazards.append(collision | (pitch >= params.max_pitch_rad) | (roll >= params.max_roll_rad) | (rough >= params.max_step_m))
        slopes.append(np.c_[pitch, roll]); spans.append(rough)
    return np.asarray(scores), np.asarray(hazards), np.asarray(slopes), np.asarray(spans)


def score_height(prediction, reference, observed, smooth, true_gradient):
    hidden = ~observed
    error = prediction.mean[hidden]-reference[hidden]
    d = height_diagnostics(prediction.mean, prediction.latent_var, reference, observed, NOMINAL)
    selected = hidden & smooth
    a = np.c_[-prediction.gradient_xy[selected], np.ones(selected.sum())]
    b = np.c_[-true_gradient[selected], np.ones(selected.sum())]
    cosine = np.sum(a*b, axis=1)/(np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1))
    return {"hidden_cells": len(error), "hidden_rmse_m": float(np.sqrt(np.mean(error**2))) if len(error) else None,
        "hidden_mae_m": float(np.abs(error).mean()) if len(error) else None,
        "hidden_p95_m": float(np.quantile(np.abs(error), .95)) if len(error) else None,
        "accepted_hidden_fraction": float(prediction.accepted[hidden].mean()) if len(error) else None,
        "picp": (d["coverage_counts"]/d["n"]).tolist() if d["n"] else None,
        "width_m": (d["width_sums"]/d["n"]).tolist() if d["n"] else None,
        "crps_m": d["crps_sum_m"]/d["n"] if d["n"] else None,
        "normal_cells": int(selected.sum()),
        "normal_mae_deg": float(np.rad2deg(np.arccos(np.clip(cosine, -1, 1))).mean()) if len(cosine) else None}


def score_risk(predicted, reference, unsafe, threshold):
    hazard = predicted >= threshold
    tp, fn = int((hazard & unsafe).sum()), int((~hazard & unsafe).sum())
    fp, tn = int((hazard & ~unsafe).sum()), int((~hazard & ~unsafe).sum())
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "reference_states": unsafe.size,
        "fnr": fn/(tp+fn) if tp+fn else None, "fpr": fp/(fp+tn) if fp+tn else None,
        "risk_mae": float(np.abs(predicted-reference).mean())}


def fingerprints():
    return {str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__),
        ROOT/"scripts/sim/experiments/terrain_risk_benchmark.py", ROOT/"scripts/sim/experiments/thesis_statistics.py",
        ROOT/"scripts/sim/experiments/paper_clouds.py", ROOT/"src/fsgp_bgk_se2/config/hunter_fastlio.yaml",
        *[ROOT/"src/fsgp_bgk_se2/fsgp_bgk_se2"/n for n in ("feature_sgp.py", "bgk_seb.py", "obstacle_columns.py")]]}


def environment():
    import cupy as cp
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,temperature.gpu,utilization.gpu",
        "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout.strip()
    return {"utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
        "numpy": np.__version__, "torch": torch.__version__, "cupy": cp.__version__,
        "gpu": gpu, "gpu_isolation": "shared desktop GPU; no exclusive frequency/power guarantee"}


def run_terrain(protocol, path, output, resume):
    import cupy as cp
    p = protocol["terrain"]; gp = FeatureSgpParams(device="cuda")
    params = params_from_config("cupy", p["resolution_m"], p["yaw_bins"])
    estimator = BgkSebEstimator(params)
    size, resolution = p["size"], p["resolution_m"]
    shape = (size, size); origin = np.full(2, -size*resolution/2)
    xy = grid(shape, origin, resolution)
    indices = np.arange(size//2-8, size//2+9, 2)
    rr, cc = np.meshgrid(indices, indices); ids = (rr*size+cc).ravel()
    settings = {"protocol_sha256": digest(path), "terrain_plan": p, "source_sha256": fingerprints(), "gp": asdict(gp), "bgk_seb": asdict(params)}
    result_path = output/"results.json"
    if result_path.exists():
        old = json.loads(result_path.read_text())
        if not resume or old["settings"] != settings:
            raise ValueError("new output or unchanged strict resume required")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("unidentified earlier output; preserve it and use a new directory")
        old = {"study_id": protocol["study_id"], "settings": settings, "planned_cases": p["planned_cases"],
            "environment_start": environment(), "runs": []}
    rows = {r["id"]: r for r in old["runs"]}
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(result_path, old)
    for fi, family in enumerate(p["families"]):
        for repeat in range(p["geometries_per_family"]):
            seed = p["seed"]+fi*10000+repeat
            spec = specification(family, seed); raw, ground_count = raw_cloud(spec, xy)
            truth = surface(spec, xy)
            reference, unsafe, slopes, spans = footprint_truth(spec, xy[ids], params)
            delta = .01; dx = np.array([delta, 0.]); dy = dx[::-1]
            gradient = np.c_[(surface(spec, xy+dx)-surface(spec, xy-dx))/(2*delta),
                (surface(spec, xy+dy)-surface(spec, xy-dy))/(2*delta)]
            smooth = np.ones(len(xy), bool)
            if family.startswith("step_"):
                u = xy[:, 0]*np.cos(spec["angle"])+xy[:, 1]*np.sin(spec["angle"])
                smooth &= np.abs(u-spec["offset"]) > .4
            geometry_hash = hashlib.sha256(truth.tobytes()+raw.tobytes()).hexdigest()
            for mi, mask in enumerate(p["masks"]):
                keep = missing_mask(raw[:, :2], mask, seed+mi*100000)
                visible = keep[:ground_count]
                epsilon = np.random.default_rng(seed+mi*100000+500000).normal(0, 1, len(raw))
                for ni, noise in enumerate(p["noise_std_m"]):
                    case_id = f"{family}_{repeat:03d}_{mask}_n{ni}"
                    checkpoint = output/(case_id+".json")
                    if resume and case_id not in rows and checkpoint.exists():
                        r = json.loads(checkpoint.read_text())
                        if r["id"] != case_id or r["spec"] != spec or r["noise_std_m"] != noise:
                            raise ValueError("controlled checkpoint differs from the frozen plan")
                        rows[case_id] = r
                    if case_id in rows:
                        r = rows[case_id]
                        if r["status"] == "ok" and digest(output/(case_id+".npz")) != r["archive_sha256"]:
                            raise ValueError("saved controlled prediction changed")
                        continue
                    row = {"id": case_id, "family": family,
                        "geometry_id": "flat_control" if family == "flat" else f"{family}_{repeat:03d}",
                        "geometry_sha256": geometry_hash, "mask": mask, "noise_std_m": noise, "spec": spec}
                    try:
                        noisy = raw.copy(); noisy[:, 2] += noise*epsilon
                        observed = noisy[:ground_count][visible]
                        torch.manual_seed(seed+mi*100000)
                        prediction = FeatureSgpPredictor(gp).fit(observed).predict(xy, gradients=True)
                        measured = raster(observed, shape, origin, resolution)
                        valid = np.isfinite(measured); variance = np.full(shape, gp.noise_std_m**2, np.float32)
                        refined = measured.copy(); chosen = valid & prediction.accepted.reshape(shape)
                        weight = variance/(variance+prediction.latent_var.reshape(shape))
                        refined[chosen] = ((1-weight)*measured+weight*prediction.mean.reshape(shape))[chosen]
                        cp.cuda.Stream.null.synchronize(); start = time.perf_counter()
                        occupied, _ = vertical_columns(noisy[keep], shape, origin, resolution, ground_elevation=measured, xp=cp)
                        occupied = cp.asnumpy(occupied)
                        result = estimator.compute(refined, variance, valid, [occupied.astype(np.float32)], observed_elevation=measured)
                        preserve_collision_risk(result, occupied)
                        cp.cuda.Stream.null.synchronize(); elapsed = (time.perf_counter()-start)*1000
                        if prediction.device != "cuda" or result.backend != "cupy":
                            raise RuntimeError("GPU fallback is not a successful GPU experiment")
                        scores = np.stack(result.final_risk_slices).reshape(p["yaw_bins"], -1)[:, ids]
                        row.update(status="ok", actual_device=prediction.device, actual_backend=result.backend,
                            input_points=int(keep.sum()), observed_ground_points=len(observed), fit_ms=prediction.fit_ms,
                            predict_with_gradient_ms=prediction.predict_ms, risk_ms=elapsed,
                            **score_height(prediction, truth, visible, smooth, gradient),
                            **score_risk(scores, reference, unsafe, p["risk_threshold"]))
                        archive = output/(case_id+".npz")
                        np.savez_compressed(archive, truth=truth.reshape(shape), observed=measured, refined=refined,
                            mean=prediction.mean.reshape(shape), variance=prediction.latent_var.reshape(shape),
                            accepted=prediction.accepted.reshape(shape), gradient=prediction.gradient_xy,
                            raw_xyz=noisy[keep], occupied=occupied, query_indices=ids, query_xy=xy[ids],
                            predicted_risk=scores, reference_risk=reference, unsafe=unsafe, reference_pose=slopes,
                            reference_span=spans, origin=origin, resolution=resolution)
                        row["archive_sha256"] = digest(archive)
                    except Exception as exc:
                        row.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
                    rows[case_id] = row
                    atomic_json(output/(case_id+".json"), row)
            old["runs"] = list(rows.values()); atomic_json(result_path, old)
            print(f"[controlled terrain] {len(rows)}/{p['planned_cases']}: {dict(Counter(r['status'] for r in rows.values()))}", flush=True)
    if len(rows) != p["planned_cases"]:
        raise ValueError("protocol does not match the complete terrain factorial")
    old["environment_end"] = environment(); atomic_json(result_path, old)
    return int(any(r["status"] != "ok" for r in rows.values()))


def run_scaling(protocol, path, output):
    import cupy as cp
    p = protocol["scaling"]
    output.mkdir(parents=True, exist_ok=True)
    if (output/"results.json").exists():
        raise ValueError("preserve scaling results; use a new output")
    rows = []; cold = []; agreement = []; start_environment = environment()
    for size in p["sizes"]:
        xy = grid((size, size), [-size*.1, -size*.1], .2)
        noise = np.random.default_rng(p["seed"]+size).normal(0, .02, len(xy))
        height = (.1*xy[:, 0]+.04*np.sin(xy[:, 1])+noise).reshape(size, size).astype(np.float32)
        variance = np.full((size, size), .0004, np.float32); valid = np.ones((size, size), bool)
        outputs = {}
        for yaw in p["yaw_bins"]:
            for backend in p["backends"]:
                estimator = BgkSebEstimator(params_from_config(backend, .2, yaw))
                for repeat in range(1+p["warmup_calls"]+p["measured_calls"]):
                    if backend == "cupy": cp.cuda.Stream.null.synchronize()
                    start = time.perf_counter()
                    result = estimator.compute(height, variance, valid, [np.zeros_like(height)], observed_elevation=height)
                    if backend == "cupy": cp.cuda.Stream.null.synchronize()
                    elapsed = (time.perf_counter()-start)*1000
                    if result.backend != backend:
                        raise RuntimeError("scaling backend fallback")
                    entry = {"size": size, "yaw_bins": yaw, "backend": backend, "repeat": repeat, "elapsed_ms": elapsed}
                    if repeat == 0: cold.append(entry)
                    elif repeat > p["warmup_calls"]: rows.append(entry)
                outputs[(yaw, backend)] = np.stack(result.final_risk_slices)
            agreement.append({"size": size, "yaw_bins": yaw,
                "max_abs_delta": float(np.max(np.abs(outputs[(yaw, "cpu")]-outputs[(yaw, "cupy")]))),
                "threshold_disagreements": int(((outputs[(yaw, "cpu")] >= .58) != (outputs[(yaw, "cupy")] >= .58)).sum())})
            print(f"[scaling] {size}x{size}x{yaw}: {len(rows)} measured calls", flush=True)
    atomic_json(output/"results.json", {"study_id": protocol["study_id"], "protocol_sha256": digest(path),
        "source_sha256": fingerprints(), "planned_calls": p["planned_measured_calls"],
        "environment_start": start_environment, "environment_end": environment(),
        "cold_calls": cold, "runs": rows, "agreement": agreement, "scope": p["scope"]})
    if len(rows) != p["planned_measured_calls"]:
        raise ValueError("scaling protocol count mismatch")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("terrain", "scaling"))
    parser.add_argument("--protocol", type=Path, default=ROOT/"scripts/sim/experiments/protocols/supplemental_study_v1.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight", action="store_true", help="three declared component checks, not the full campaign")
    args = parser.parse_args(); protocol = json.loads(args.protocol.read_text())
    if args.preflight:
        if args.phase != "terrain": parser.error("preflight only applies to terrain")
        protocol["terrain"].update(families=["flat", "step_high", "column"], geometries_per_family=1,
            masks=["hole16"], noise_std_m=[.02], planned_cases=3)
    torch.set_num_threads(1)
    return run_terrain(protocol, args.protocol, args.output, args.resume) if args.phase == "terrain" else run_scaling(protocol, args.protocol, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
