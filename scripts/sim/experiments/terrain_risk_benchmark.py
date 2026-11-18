#!/usr/bin/env python3
"""Repeatable geometric-oracle benchmark of the current BGK/SEB adapter.

This does not substitute for the upstream FSGP implementation or a point-cloud
front-end test. Inputs are controlled noisy elevation grids, not fabricated ROS
measurements. CPU/CUDA timings include host transfers and result materialization.
"""

import argparse
import csv
import hashlib
from dataclasses import asdict
import json
import math
from pathlib import Path
import platform
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src/fsgp_bgk_se2"))
from fsgp_bgk_se2.bgk_seb import BgkSebEstimator, BgkSebParams
from plot_style import configure_chinese, scenario_name

SCENES = ("flat", "ramp", "directional_slope", "steep_slope", "step", "ridge",
          "rough", "small_obstacle")


def height(scene, x, y, angle=0.0):
    u = x * math.cos(angle) + y * math.sin(angle)
    v = -x * math.sin(angle) + y * math.cos(angle)
    if scene == "flat":
        return np.zeros_like(u)
    if scene in ("ramp", "directional_slope", "steep_slope"):
        slope = {"ramp": 0.17, "directional_slope": math.tan(0.50),
                 "steep_slope": math.tan(0.66)}[scene]
        return slope * u
    if scene == "step":
        return np.where(u >= 0.0, 0.30, 0.0)
    if scene == "ridge":
        return 0.30 * np.exp(-0.5 * (u / 0.16) ** 2)
    if scene == "rough":
        return 0.07 * np.sin(8.0 * u) * np.cos(9.0 * v)
    if scene == "small_obstacle":
        return np.where((u - 0.30) ** 2 + v ** 2 <= 0.28 ** 2, 0.55, 0.0)
    raise ValueError(scene)


def coordinates(size, resolution):
    axis = (np.arange(size) - (size - 1) * 0.5) * resolution
    return np.meshgrid(axis, axis)


def oracle(scene, x, y, params, angle):
    if scene in ("flat", "ramp", "directional_slope", "steep_slope"):
        # Exact analytic plane, independent of rounded estimator grid samples.
        gradient = {"flat": 0.0, "ramp": 0.17, "directional_slope": math.tan(0.50),
                    "steep_slope": math.tan(0.66)}[scene]
        yaws = (np.arange(params.yaw_bins) + 0.5) * (2 * math.pi / params.yaw_bins)
        pitch = np.arctan(np.abs(gradient * np.cos(yaws - angle)))
        roll = np.arctan(np.abs(gradient * np.sin(yaws - angle)))
        score = np.maximum(np.clip((pitch - params.pitch_free_rad) /
                                   (params.max_pitch_rad - params.pitch_free_rad), 0, 1),
                           np.clip((roll - params.roll_free_rad) /
                                   (params.max_roll_rad - params.roll_free_rad), 0, 1))
        hazard = (pitch >= params.max_pitch_rad) | (roll >= params.max_roll_rad)
        shape = (params.yaw_bins,) + x.shape
        return (np.broadcast_to(score[:, None, None], shape).astype(np.float32),
                np.broadcast_to(hazard[:, None, None], shape).copy())
    # Dense analytic surface samples fit the support plane independently of
    # the estimator's rounded grid offsets and edge-average approximation.
    lx, ly = np.meshgrid(np.linspace(-params.footprint_back - params.footprint_margin,
                                     params.footprint_front + params.footprint_margin, 13),
                         np.linspace(-params.footprint_half_width - params.footprint_margin,
                                     params.footprint_half_width + params.footprint_margin, 11))
    lx, ly = lx.ravel(), ly.ravel()
    design = np.column_stack((lx, ly, np.ones_like(lx)))
    inverse = np.linalg.pinv(design)
    risks = []
    hazards = []
    for yaw in (np.arange(params.yaw_bins) + 0.5) * (2.0 * math.pi / params.yaw_bins):
        dx = lx * math.cos(yaw) - ly * math.sin(yaw)
        dy = lx * math.sin(yaw) + ly * math.cos(yaw)
        z = height(scene, x[..., None] + dx, y[..., None] + dy, angle)
        coefficients = z @ inverse.T
        residual = z - coefficients @ design.T
        span = np.ptp(residual, axis=-1)
        pitch = np.arctan(np.abs(coefficients[..., 0]))
        roll = np.arctan(np.abs(coefficients[..., 1]))
        hazard = ((pitch >= params.max_pitch_rad) | (roll >= params.max_roll_rad)
                  | (span >= params.max_step_m))
        pitch_risk = np.clip((pitch - params.pitch_free_rad) /
                             (params.max_pitch_rad - params.pitch_free_rad), 0, 1)
        roll_risk = np.clip((roll - params.roll_free_rad) /
                            (params.max_roll_rad - params.roll_free_rad), 0, 1)
        step_risk = np.clip((span - params.step_free_m) /
                            (params.max_step_m - params.step_free_m), 0, 1)
        risks.append(np.maximum.reduce((pitch_risk, roll_risk, step_risk)))
        hazards.append(hazard)
    return np.asarray(risks, dtype=np.float32), np.asarray(hazards)


def classification(predicted_risk, unsafe, mask, threshold):
    predicted = predicted_risk[:, mask] >= threshold
    target = unsafe[:, mask]
    tp = int(np.count_nonzero(predicted & target))
    fp = int(np.count_nonzero(predicted & ~target))
    fn = int(np.count_nonzero(~predicted & target))
    tn = int(np.count_nonzero(~predicted & ~target))
    ratio = lambda a, b: a / b if b else None
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
            "f1": ratio(2 * tp, 2 * tp + fp + fn), "iou": ratio(tp, tp + fp + fn),
            "false_safe_rate": ratio(fn, tp + fn),
            "false_hazard_rate": ratio(fp, fp + tn), "evaluated_states": tp + fp + fn + tn}


def params_from_config(backend, resolution, yaw_bins):
    config = yaml.safe_load((ROOT / "src/fsgp_bgk_se2/config/hunter_fastlio.yaml").read_text())
    values = config["fsgp_bgk_se2_node"]["ros__parameters"]
    keys = BgkSebParams.__dataclass_fields__
    kwargs = {key: value for key, value in values.items() if key in keys}
    kwargs.update(resolution=resolution, yaw_bins=yaw_bins, prefer_gpu=backend == "cupy",
                  gpu_backend=backend, fusion_mode=values["risk_fusion_mode"],
                  store_diagnostic_slices=False)
    # Offline producers supply point occupancy, unlike the ROS pose-risk prior.
    kwargs["baseline_is_pose_risk"]=False
    return BgkSebParams(**kwargs)


def case_data(scene, size, resolution, seed, noise_std, dropout):
    rng = np.random.default_rng(seed)
    angle = float(rng.uniform(-math.pi, math.pi))
    x, y = coordinates(size, resolution)
    truth = height(scene, x, y, angle).astype(np.float32)
    valid = rng.random(truth.shape) >= dropout
    noisy = truth + rng.normal(0.0, noise_std, truth.shape).astype(np.float32)
    noisy[~valid] = np.nan
    variance = np.full(truth.shape, noise_std ** 2, dtype=np.float32)
    # Controlled component-isolation input. Actual upstream terrain risk and
    # vertical-column collision overlay are NOT part of this synthetic test.
    baseline = [np.zeros(truth.shape, dtype=np.float32)]
    margin = int(math.ceil(1.5 / resolution))
    mask = valid.copy()
    mask[:margin] = mask[-margin:] = False
    mask[:, :margin] = mask[:, -margin:] = False
    return x, y, truth, noisy, variance, valid, baseline, mask, angle


def export_table(path, rows):
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, output, name):
    fig.savefig(output / (name + ".png"), dpi=200, bbox_inches="tight")
    fig.savefig(output / (name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def figures(output, quality, latency, snapshots, oracle_center_aligned=True):
    configure_chinese()
    size = max(row["size"] for row in quality)
    groups = [[r for r in quality if r["scene"] == scene and r["size"] == size and
               r["actual_backend"] == "cpu"] for scene in SCENES]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
    for ax, key, title, unit in zip(axes,
            ("elevation_rmse_m", "elevation_p95_m", "false_hazard_rate"),
            ("高程均方根误差", "高程绝对误差 P95", "安全区域误报率"),
            ("误差（米）", "误差（米）", "比例")):
        values = [[r[key] for r in group if r.get(key) is not None] for group in groups]
        ax.boxplot([v or [np.nan] for v in values], showmeans=True)
        ax.set_xticks(np.arange(1, len(SCENES) + 1), [scenario_name(s) for s in SCENES])
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.tick_params(axis="x", rotation=55)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(f"地形分析精度：当前 BGK/SEB 组件（{size}×{size} 网格，多随机种子）")
    save_figure(fig, output, "01_terrain_accuracy")

    fig, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
    for ax, scene, group in zip(axes.flat, SCENES, groups):
        counts = np.array([[sum(r["tn"] for r in group), sum(r["fp"] for r in group)],
                           [sum(r["fn"] for r in group), sum(r["tp"] for r in group)]])
        proportions = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
        ax.imshow(proportions, cmap="Blues", vmin=0, vmax=1)
        for (i, j), value in np.ndenumerate(counts):
            ax.text(j, i, f"{value:,}\n{proportions[i,j]:.1%}", ha="center", va="center",
                    color="white" if proportions[i, j] > 0.5 else "black")
        ax.set_xticks([0, 1], ["预测安全", "预测危险"])
        ax.set_yticks([0, 1], ["实际安全", "实际危险"])
        ax.set_title(scenario_name(scene))
    fig.suptitle("风险识别混淆矩阵：独立几何真值与当前组件输出")
    save_figure(fig, output, "02_risk_confusion")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    labels = [scenario_name(s) for s in SCENES]
    x = np.arange(len(labels))
    for ax, key, title in (
            (axes[0], "f1", "危险识别 F1（越高越好）"),
            (axes[1], "false_safe_rate", "危险区域漏报率（越低越好）"),
            (axes[2], "false_hazard_rate", "安全区域误报率（越低越好）")):
        values = [[r[key] for r in group if r[key] is not None] for group in groups]
        means = [np.mean(v) if v else np.nan for v in values]
        deviations = [np.std(v, ddof=1) if len(v) > 1 else 0 for v in values]
        ax.bar(x, means, yerr=deviations, capsize=3, color="#247a91")
        for index, v in enumerate(values):
            if not v:
                ax.text(index, 0.04, "不适用", ha="center", rotation=90, fontsize=8)
        ax.set_xticks(x, labels, rotation=55, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("风险分类统计：均值与种子间标准差（仅当前方法，分母为空时不适用）")
    save_figure(fig, output, "07_risk_classification_metrics")

    fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    for ax, scene in zip(axes.flat, SCENES):
        snapshot = snapshots[scene]
        im = ax.imshow(snapshot["risk"][0], origin="lower", cmap="RdYlGn_r", vmin=0, vmax=1,
                       extent=snapshot["extent"])
        ax.contour(snapshot["oracle_unsafe"][0].astype(float), levels=[0.5], colors="black",
                   linewidths=0.7, origin="lower", extent=snapshot["extent"])
        ax.set_title(scenario_name(scene) + f"（首个航向切片 {180 / snapshot['risk'].shape[0]:.1f}°）")
        ax.set_xlabel("横向位置（米）"); ax.set_ylabel("纵向位置（米）")
    fig.colorbar(im, ax=list(axes.flat), shrink=0.8, label="估计风险分数")
    fig.suptitle("风险地图示例：黑色轮廓表示独立几何真值中的危险边界")
    save_figure(fig, output, "03_risk_map_examples")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for scene in ("directional_slope", "step", "ridge"):
        s = snapshots[scene]
        c = s["risk"].shape[1] // 2
        yaws = (np.arange(s["risk"].shape[0]) + 0.5) * 360 / s["risk"].shape[0]
        axes[0].plot(yaws, s["risk"][:, c, c], label=scenario_name(scene))
        axes[1].plot(yaws, s["oracle_risk"][:, c, c], label=scenario_name(scene))
        band = np.max(s["risk"], axis=0) - np.min(s["risk"], axis=0)
        lo, hi = s["extent"][:2]
        resolution = (hi - lo) / band.shape[1]
        axes[2].plot(lo + (np.arange(band.shape[1]) + 0.5) * resolution,
                     band[c], label=scenario_name(scene))
    axes[0].set_title("同一位置的航向相关风险")
    axes[1].set_title("独立几何真值的航向相关风险" if oracle_center_aligned else "历史几何真值（旧起点角口径）")
    axes[2].set_title("中心剖面的跨航向风险差值")
    for ax in axes:
        ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.2); ax.legend(fontsize=8)
        ax.set_xlabel("航向角（度）" if ax is not axes[2] else "横向位置（米）")
    axes[0].set_ylabel("风险分数")
    fig.suptitle("SE(2) 方向性分析：相同地形、不同车辆航向")
    save_figure(fig, output, "04_se2_directionality")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for backend, color in (("cpu", "#247a91"), ("cupy", "#cf5c36")):
        selected = [r for r in latency if r["actual_backend"] == backend]
        if not selected:
            continue
        sizes = sorted({r["size"] for r in selected})
        work = [n * n * selected[0]["yaw_bins"] for n in sizes]
        median = [np.median([r["latency_ms"] for r in selected if r["size"] == n]) for n in sizes]
        p95 = [np.percentile([r["latency_ms"] for r in selected if r["size"] == n], 95) for n in sizes]
        name = "CPU" if backend == "cpu" else "GPU（CUDA）"
        axes[0].plot(work, median, "o-", label=name + " 中位数", color=color)
        axes[0].plot(work, p95, "o--", label=name + " P95", color=color)
        axes[1].plot(work, [1000 / value for value in p95], "o-", label=name, color=color)
    axes[0].axhline(100, ls=":", color="gray", label="10 Hz 时间预算")
    axes[0].set_ylabel("计算延迟（毫秒）"); axes[1].set_ylabel("P95 延迟倒数（Hz）")
    for ax in axes:
        ax.set_xlabel("SE(2) 状态数（行×列×航向数）")
        ax.grid(alpha=0.2); ax.legend(fontsize=8); ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    fig.suptitle("CPU/GPU 计算规模测试：包含数据传输，不等同于完整链路更新率")
    save_figure(fig, output, "05_backend_scaling")

    fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    for ax, scene in zip(axes.flat, SCENES):
        s = snapshots[scene]
        im = ax.imshow(s["mean"] - s["truth"], origin="lower", cmap="RdBu_r",
                       vmin=-0.15, vmax=0.15, extent=s["extent"])
        ax.set_title(scenario_name(scene)); ax.set_xlabel("横向位置（米）"); ax.set_ylabel("纵向位置（米）")
    fig.colorbar(im, ax=list(axes.flat), shrink=0.8, label="估计高程减真实高程（米）")
    fig.suptitle("高程误差分布：展示突变边界平滑误差")
    save_figure(fig, output, "06_elevation_error_maps")


def write_report(output, args, quality, latency, unavailable):
    def metric(scene, key, backend="cpu"):
        values = [row[key] for row in quality if row["scene"] == scene
                  and row["actual_backend"] == backend and row.get(key) is not None]
        return f"{float(np.mean(values)):.4f}" if values else "n/a"

    lines = [
        "# Terrain Analysis Benchmark Report",
        "",
        "> Scope: current BGK/SEB adapter on controlled noisy elevation grids. "
        "This is a component benchmark, not a LiDAR/ROS/end-to-end navigation result.",
        "",
        f"- Date: `{time.strftime('%Y-%m-%dT%H:%M:%S%z')}`",
        f"- Grid sizes: `{args.sizes}`; repetitions per scene/backend/size: `{args.repeats}`",
        f"- yaw bins: `{args.yaw_bins}`; resolution: `{args.resolution} m`; noise std: `{args.noise_std} m`; dropout: `{args.dropout}`",
        f"- CUDA unavailable/fallback records: `{len(unavailable)}`",
        "",
        "## Tested Objects",
        "",
        "The test objects follow the local reference papers: flat and passable ramp for false alarms, steep slope for hazard recall, step/ridge for SE(2) directionality, rough terrain for uncertainty, and a small obstacle for local hazard sensitivity.",
        "",
        "## Figures",
        "",
        "![terrain accuracy](01_terrain_accuracy.png)",
        "![risk confusion](02_risk_confusion.png)",
        "![risk metrics](07_risk_classification_metrics.png)",
        "![risk maps](03_risk_map_examples.png)",
        "![directionality](04_se2_directionality.png)",
        "![backend scaling](05_backend_scaling.png)",
        "![elevation errors](06_elevation_error_maps.png)",
        "",
        "## Mean Metrics (CPU rows)",
        "",
        "| Scene | Elevation RMSE (m) | Elevation P95 (m) | Hazard F1 | False-safe | False-hazard | Risk MAE | Yaw span |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scene in SCENES:
        lines.append(f"| `{scene}` | {metric(scene, 'elevation_rmse_m')} | {metric(scene, 'elevation_p95_m')} | "
                     f"{metric(scene, 'f1')} | {metric(scene, 'false_safe_rate')} | "
                     f"{metric(scene, 'false_hazard_rate')} | {metric(scene, 'risk_mae')} | {metric(scene, 'yaw_span_mean')} |")
    lines.extend([
        "",
        "## Timing",
        "",
        "Timing is measured around `BgkSebEstimator.compute`, after warm-up, and includes array conversion/result materialization. It does not claim that the full ROS terrain pipeline runs at the reciprocal rate.",
        "",
        "| Requested/actual backend | Size | Calls | Median (ms) | P95 (ms) | Max (ms) |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for backend in sorted({row["actual_backend"] for row in latency}):
        for size in sorted({row["size"] for row in latency if row["actual_backend"] == backend}):
            values = [row["latency_ms"] for row in latency if row["actual_backend"] == backend and row["size"] == size]
            lines.append(f"| `{backend}` | {size} | {len(values)} | {np.median(values):.2f} | {np.percentile(values, 95):.2f} | {max(values):.2f} |")
    lines.extend([
        "",
        "## Interpretation Boundary",
        "",
        "Only the current method is reported. The oracle is computed from the analytic terrain and vehicle footprint. Hazard labels mean a hard physical limit is exceeded; soft-score MAE is not probability calibration. These figures measure adapter behavior, not real-world accuracy. Sparse GP training, vertical-column collision overlay and the upstream terrain analyzer are excluded from this component test. Full navigation and measured terrain replay remain separate experiments defined in `docs/EXPERIMENT_PROTOCOL.md`.",
        "",
    ])
    (output / "terrain_benchmark_report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="64,96,104")
    parser.add_argument("--repeats", type=int, default=100, help="independent noise/rotation seeds per scene")
    parser.add_argument("--timing-repeats", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--yaw-bins", type=int, default=36)
    parser.add_argument("--resolution", type=float, default=0.2)
    parser.add_argument("--noise-std", type=float, default=0.02)
    parser.add_argument("--dropout", type=float, default=0.03)
    parser.add_argument("--threshold", type=float, default=0.58)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replot", type=Path, help="regenerate Chinese figures from recorded data; no experiment rerun")
    args = parser.parse_args()
    if args.replot:
        saved = json.loads((args.replot / "results.json").read_text())
        snapshots = {}
        for scene in SCENES:
            with np.load(args.replot / (scene + "_snapshot.npz")) as source:
                snapshots[scene] = {key: source[key] for key in source.files}
        args.output.mkdir(parents=True, exist_ok=True)
        figures(args.output, saved["quality_rows"], saved["timing_rows"], snapshots,
                oracle_center_aligned="yaw_convention" in saved)
        print(f"[terrain] Chinese figures regenerated from {args.replot}; data unchanged")
        return 0
    sizes = sorted(set(int(n) for n in args.sizes.split(",")))
    if (min(sizes) < 24 or args.repeats < 1 or args.timing_repeats < 1 or args.warmup < 0
            or args.yaw_bins < 4 or args.resolution <= 0 or not 0 <= args.dropout < 1
            or not 0 < args.threshold <= 1 or args.noise_std < 0):
        parser.error("invalid grid/noise/timing configuration")
    args.output.mkdir(parents=True, exist_ok=True)
    quality, latency, snapshots, unavailable = [], [], {}, []
    params = params_from_config("cpu", args.resolution, args.yaw_bins)
    for size in sizes:
        for backend in ("cpu", "cupy"):
            estimator = BgkSebEstimator(params_from_config(backend, args.resolution, args.yaw_bins))
            inputs = case_data("ramp", size, args.resolution, args.seed, args.noise_std, args.dropout)
            payload = inputs[3:7]
            for _ in range(max(1, args.warmup)):
                result = estimator.compute(*payload)
            if result.backend != backend:
                unavailable.append({"requested": backend, "size": size, "reason": result.backend_reason})
                print(f"[terrain] unavailable {backend}: {result.backend_reason}", flush=True)
                continue
            for index in range(args.timing_repeats):
                start = time.perf_counter()
                result = estimator.compute(*payload)
                latency.append({"size": size, "yaw_bins": args.yaw_bins,
                                "requested_backend": backend, "actual_backend": result.backend,
                                "repeat": index, "latency_ms": (time.perf_counter() - start) * 1000})
            for scene_index, scene in enumerate(SCENES):
                for repeat in range(args.repeats):
                    seed = args.seed + scene_index * 1000 + repeat
                    x, y, truth, noisy, var, valid, baseline, mask, angle = case_data(
                        scene, size, args.resolution, seed, args.noise_std, args.dropout)
                    result = estimator.compute(noisy, var, valid, baseline)
                    risk = np.stack(result.final_risk_slices)
                    oracle_risk, unsafe = oracle(scene, x, y, params, angle)
                    error = result.bgk_mean[mask] - truth[mask]
                    row = {"scene": scene, "size": size, "yaw_bins": args.yaw_bins,
                           "seed": seed, "repeat": repeat, "actual_backend": result.backend,
                           "angle_rad": angle, "valid_coverage": float(valid.mean()),
                           "elevation_rmse_m": float(np.sqrt(np.mean(error ** 2))),
                           "elevation_mae_m": float(np.mean(np.abs(error))),
                           "elevation_p95_m": float(np.percentile(np.abs(error), 95)),
                           "risk_mae": float(np.mean(np.abs(risk[:, mask] - oracle_risk[:, mask]))),
                           "yaw_span_mean": float(np.mean(np.ptp(risk, axis=0)[mask]))}
                    row.update(classification(risk, unsafe, mask, args.threshold))
                    quality.append(row)
                    if (repeat + 1) % 25 == 0:
                        print(f"[terrain] {size} {backend} {scene}: {repeat + 1}/{args.repeats}", flush=True)
                    if backend == "cpu" and size == sizes[-1] and repeat == 0:
                        extent = [-size * args.resolution / 2, size * args.resolution / 2] * 2
                        snapshots[scene] = {"risk": risk, "oracle_risk": oracle_risk,
                                            "oracle_unsafe": unsafe, "mean": result.bgk_mean,
                                            "truth": truth, "extent": extent}
                        np.savez_compressed(args.output / (scene + "_snapshot.npz"), **snapshots[scene])
            print(f"[terrain] size={size} backend={backend}: accuracy seeds and timings complete", flush=True)
    export_table(args.output / "terrain_quality.csv", quality)
    export_table(args.output / "terrain_latency.csv", latency)
    metadata = {"study": "current_adapter_component_benchmark", "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "scope": "synthetic elevation input -> BGK/SEB adapter; no LiDAR, sparse GP training, terrain_analyzer or planner",
                "yaw_convention": "bin center: (index + 0.5) * 2pi / yaw_bins",
                "parameters": asdict(params), "arguments": {k: str(v) if isinstance(v, Path) else v
                                                             for k, v in vars(args).items()},
                "platform": platform.platform(), "python": platform.python_version(),
                "independent_input_configs": len({(r['scene'], r['size'], r['seed']) for r in quality}),
                "scene_seed_clusters": len({(r['scene'], r['seed']) for r in quality}),
                "adapter_sha256": hashlib.sha256((ROOT / "src/fsgp_bgk_se2/fsgp_bgk_se2/bgk_seb.py").read_bytes()).hexdigest(),
                "unavailable": unavailable, "quality_rows": quality, "timing_rows": latency}
    (args.output / "results.json").write_text(json.dumps(metadata, indent=2) + "\n")
    figures(args.output, quality, latency, snapshots)
    write_report(args.output, args, quality, latency, unavailable)
    print(f"[terrain] {len(quality)} quality runs, {len(latency)} timed calls; results={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
