#!/usr/bin/env python3
"""Observed-only feature SGP evaluation against withheld analytic terrain truth."""

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src/fsgp_bgk_se2"))
from fsgp_bgk_se2.feature_sgp import FeatureSgpParams, FeatureSgpPredictor
from plot_style import configure_chinese

SCENES = ("slope", "hills", "ridge", "step", "ditch")
MASKS = ("random", "block", "occlusion", "frontier")
NAMES = {"slope": "斜坡", "hills": "起伏山地", "ridge": "窄脊", "step": "台阶",
         "ditch": "沟壑", "random": "随机稀疏", "block": "连续孔洞",
         "occlusion": "几何遮挡", "frontier": "单侧外推"}


def specification(scene, seed):
    rng = np.random.default_rng(seed)
    return {"scene": scene, "angle": float(rng.uniform(-np.pi, np.pi)),
            "slope": float(rng.uniform(0.08, 0.30)), "height": float(rng.uniform(0.20, 0.40)),
            "width": float(rng.uniform(0.18, 0.45)), "offset": float(rng.uniform(-0.4, 0.4)),
            "wavelength": float(rng.uniform(2.5, 4.5))}


def ground(spec, x, y):
    u = x * np.cos(spec["angle"]) + y * np.sin(spec["angle"])
    v = -x * np.sin(spec["angle"]) + y * np.cos(spec["angle"])
    scene = spec["scene"]
    if scene == "slope":
        return 0.1 + spec["slope"] * u
    if scene == "hills":
        return spec["height"] * np.sin(2 * np.pi * u / spec["wavelength"]) * np.cos(v / 1.5)
    if scene == "ridge":
        return 0.03 * v + spec["height"] * np.exp(-0.5 * ((u - spec["offset"]) / spec["width"]) ** 2)
    if scene == "step":
        return np.where(u > spec["offset"], spec["height"], 0.0)
    if scene == "ditch":
        return 0.03 * v - spec["height"] * np.exp(-0.5 * ((u - spec["offset"]) / spec["width"]) ** 2)
    raise ValueError(scene)


def visible_mask(pattern, spec, x, y, seed):
    rng = np.random.default_rng(seed)
    if pattern == "random":
        return rng.random(x.shape) >= 0.30
    if pattern == "block":
        width = rng.uniform(1.0, 2.0)
        return ~((np.abs(x - spec["offset"]) < width / 2) & (np.abs(y) < width / 2))
    if pattern == "frontier":
        return x <= spec["offset"]
    # Exact line samples against analytic terrain plus a solid vertical cylinder.
    # This controlled geometry test is not an RTX/real sensor visibility claim.
    sensor = np.array([float(x.min()) - 0.2, 0.0])
    sensor_z = ground(spec, sensor[0], sensor[1]) + 0.7
    t = np.linspace(0.03, 0.98, 48)
    sx = sensor[0] + (x[..., None] - sensor[0]) * t
    sy = sensor[1] + (y[..., None] - sensor[1]) * t
    ray_z = sensor_z + (ground(spec, x, y)[..., None] - sensor_z) * t
    blocked_ground = ground(spec, sx, sy) > ray_z + 0.005
    cylinder = (sx ** 2 + sy ** 2 < 0.35 ** 2) & (ray_z <= 1.8)
    return ~(blocked_ground | cylinder).any(axis=-1)


def regression(mean, truth, mask):
    error = np.asarray(mean)[mask] - np.asarray(truth)[mask]
    if not len(error):
        return {"n": 0, "rmse_m": None, "mae_m": None, "bias_m": None, "p95_m": None}
    return {"n": int(len(error)), "rmse_m": float(np.sqrt(np.mean(error ** 2))),
            "mae_m": float(np.mean(np.abs(error))), "bias_m": float(np.mean(error)),
            "p95_m": float(np.percentile(np.abs(error), 95))}


def measure(pred, truth, visible, grad_truth):
    hidden = ~visible.ravel()
    z = truth.ravel()
    results = {}
    for name, mask in (("observed", visible.ravel()), ("unknown_all", hidden),
                       ("unknown_supported", hidden & pred.supported),
                       ("unknown_accepted", hidden & pred.accepted),
                       ("unknown_near", hidden & (pred.nearest_distance <= 0.5)),
                       ("unknown_far", hidden & (pred.nearest_distance > 1.0))):
        results.update({name + "_" + key: value for key, value in regression(pred.mean, z, mask).items()})
    results["unknown_prediction_coverage"] = float(np.mean(pred.accepted[hidden])) if hidden.any() else None
    results["hidden_fraction"] = float(hidden.mean())
    std = np.sqrt(np.maximum(pred.latent_var, 1e-10))
    error = pred.mean - z
    results["unknown_picp95"] = float(np.mean(np.abs(error[hidden]) <= 1.959964 * std[hidden]))
    results["unknown_interval_width95_m"] = float(np.mean(3.919928 * std[hidden]))
    results["unknown_nll"] = float(np.mean(0.5 * np.log(2 * np.pi * std[hidden] ** 2)
                                            + 0.5 * (error[hidden] / std[hidden]) ** 2))
    smooth = hidden & (np.linalg.norm(grad_truth, axis=1) < 1.0)
    normals = np.column_stack((-grad_truth, np.ones(len(z))))
    normal_pred = np.column_stack((-pred.gradient_xy, np.ones(len(z))))
    cosine = np.sum(normals * normal_pred, axis=1) / (
        np.linalg.norm(normals, axis=1) * np.linalg.norm(normal_pred, axis=1))
    results["unknown_smooth_normal_mae_deg"] = float(np.mean(np.degrees(np.arccos(
        np.clip(cosine[smooth], -1, 1))))) if smooth.any() else None
    return results


def plot_results(output, rows, snapshots):
    configure_chinese()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, key, title in ((axes[0], "unknown_all_rmse_m", "未观测区域高程 RMSE（米）"),
                           (axes[1], "unknown_prediction_coverage", "未观测区域有效预测覆盖率"),
                           (axes[2], "unknown_picp95", "未观测区域 95% 区间实际覆盖率")):
        values = [[r[key] for r in rows if r["mask"] == pattern and r.get(key) is not None] for pattern in MASKS]
        ax.boxplot([v or [np.nan] for v in values], showmeans=True)
        ax.set_xticks(np.arange(1, len(MASKS) + 1), [NAMES[p] for p in MASKS])
        ax.set_title(title); ax.grid(axis="y", alpha=0.2)
        if key != "unknown_all_rmse_m": ax.set_ylim(0, 1.05)
        if key == "unknown_picp95": ax.axhline(0.95, color="gray", ls=":", label="名义覆盖率")
    fig.suptitle("真实留出评测：模型只接收可见点，真值仅用于评分（仅当前方法）")
    save(fig, output, "01_unknown_prediction_statistics")

    chosen = [s for s in snapshots if s["mask"] == "block"]
    fig, axes = plt.subplots(5, len(chosen), figsize=(4 * len(chosen), 15), squeeze=False, constrained_layout=True)
    for col, sample in enumerate(chosen):
        arrays = (sample["truth"], np.where(sample["visible"], sample["observations"], np.nan),
                  sample["mean"], sample["mean"] - sample["truth"], np.sqrt(sample["variance"]))
        titles = ("完整真值", "可见输入（白色为空洞）", "GP 高程预测", "预测误差", "潜在函数标准差")
        heights = np.concatenate([a[np.isfinite(a)] for a in arrays[:3]])
        limits = (float(heights.min()), float(heights.max()))
        error_max = max(float(np.max(np.abs(arrays[3]))), 1e-6)
        for row, (matrix, title) in enumerate(zip(arrays, titles)):
            im = axes[row, col].imshow(matrix, origin="lower", cmap="RdBu_r" if row == 3 else "viridis",
                                      extent=sample["extent"],
                                      vmin=limits[0] if row < 3 else (-error_max if row == 3 else 0),
                                      vmax=limits[1] if row < 3 else (error_max if row == 3 else None))
            axes[row, col].set_title(NAMES[sample["scene"]] + "：" + title)
            axes[row, col].set_xlabel("横向位置（米）"); axes[row, col].set_ylabel("纵向位置（米）")
            fig.colorbar(im, ax=axes[row, col], shrink=0.75, label="米")
    save(fig, output, "02_prediction_truth_panels")

    fig, axes = plt.subplots(1, len(chosen), figsize=(4 * len(chosen), 4), squeeze=False, constrained_layout=True)
    for ax, sample in zip(axes.flat, chosen):
        middle = sample["truth"].shape[0] // 2
        lo, hi = sample["extent"][:2]
        x = np.linspace(lo, hi, sample["truth"].shape[1], endpoint=False) + (hi - lo) / sample["truth"].shape[1] / 2
        mean = sample["mean"][middle]; std = np.sqrt(sample["variance"][middle])
        ax.plot(x, sample["truth"][middle], color="black", label="地形真值")
        ax.plot(x, mean, color="#247a91", label="预测均值")
        ax.fill_between(x, mean - 1.959964 * std, mean + 1.959964 * std, alpha=0.2, label="名义 95% 区间")
        observed = sample["visible"][middle]
        ax.scatter(x[observed], sample["observations"][middle][observed], s=8, color="#cf5c36", label="可见输入")
        ax.set_title(NAMES[sample["scene"]]); ax.set_xlabel("横向位置（米）"); ax.set_ylabel("高程（米）")
        ax.grid(alpha=0.2); ax.legend(fontsize=8)
    fig.suptitle("跨连续孔洞剖面：预测区间是否覆盖真值，需要由实测判断")
    save(fig, output, "03_unknown_cross_sections")


def save(fig, output, name):
    fig.savefig(output / (name + ".png"), dpi=180, bbox_inches="tight")
    fig.savefig(output / (name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-count", type=int, default=10, help="independent geometry seeds per family")
    parser.add_argument("--size", type=int, default=40)
    parser.add_argument("--resolution", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--training-steps", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.scene_count < 1 or args.size < 16 or args.resolution <= 0:
        parser.error("invalid scene/grid count")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    params = FeatureSgpParams(device=args.device, training_steps=args.training_steps)
    axis = (np.arange(args.size) - (args.size - 1) / 2) * args.resolution
    x, y = np.meshgrid(axis, axis)
    query = np.column_stack((x.ravel(), y.ravel())).astype(np.float32)
    rows, snapshots = [], []
    for scene_index, scene in enumerate(SCENES):
        for repeat in range(args.scene_count):
            geometry_seed = args.seed + scene_index * 10000 + repeat
            spec = specification(scene, geometry_seed)
            truth = ground(spec, x, y)
            eps = 0.05
            gradient = np.column_stack(((ground(spec, x + eps, y) - ground(spec, x - eps, y)).ravel() / (2 * eps),
                                        (ground(spec, x, y + eps) - ground(spec, x, y - eps)).ravel() / (2 * eps)))
            for mask_index, pattern in enumerate(MASKS):
                sensor_seed = geometry_seed + mask_index * 1000000
                visible = visible_mask(pattern, spec, x, y, sensor_seed)
                rng = np.random.default_rng(sensor_seed)
                observations = truth + rng.normal(0, params.noise_std_m, truth.shape)
                observed_xyz = np.column_stack((x[visible], y[visible], observations[visible]))
                predictor = FeatureSgpPredictor(params).fit(observed_xyz)
                pred = predictor.predict(query, gradients=True)
                row = {"scene": scene, "geometry_seed": geometry_seed, "sensor_seed": sensor_seed,
                       "mask": pattern, "device": pred.device, "fit_ms": pred.fit_ms,
                       "predict_ms_including_gradients": pred.predict_ms, "training_points": pred.training_points}
                row.update(measure(pred, truth, visible, gradient))
                rows.append(row)
                if repeat == 0:
                    sample = {"scene": scene, "mask": pattern, "truth": truth, "visible": visible,
                              "observations": observations, "mean": pred.mean.reshape(x.shape),
                              "variance": pred.latent_var.reshape(x.shape),
                              "extent": [-args.size * args.resolution / 2, args.size * args.resolution / 2] * 2}
                    snapshots.append(sample)
                    np.savez_compressed(args.output / f"{scene}_{pattern}.npz", **sample)
                metadata = {"study": "feature_sgp_true_holdout", "scope": "analytic independent geometry, no Isaac rays or original paper reproduction",
                            "parameters": asdict(params), "geometry_count": args.scene_count * len(SCENES),
                            "independent_unit": "scene family + geometry seed; masks correlated within one scene",
                            "predictor_sha256": hashlib.sha256((ROOT / "src/fsgp_bgk_se2/fsgp_bgk_se2/feature_sgp.py").read_bytes()).hexdigest(),
                            "runs": rows}
                temporary = args.output / "results.tmp"
                temporary.write_text(json.dumps(metadata, indent=2) + "\n")
                temporary.replace(args.output / "results.json")
            print(f"[holdout] {scene} geometry {repeat + 1}/{args.scene_count}: 4 masks recorded", flush=True)
    with (args.output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    plot_results(args.output, rows, snapshots)
    print(f"[holdout] complete: {len(rows)} cases; {len(SCENES) * args.scene_count} independent geometries; {args.output}")


if __name__ == "__main__":
    main()
