#!/usr/bin/env python3
"""Chinese, geometry-clustered reports for prediction_holdout recordings."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from prediction_holdout import MASKS, NAMES, SCENES, plot_results
from plot_style import configure_chinese


def clustered_stats(rows, key, seed=20261012):
    groups = defaultdict(list)
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and np.isfinite(value):
            groups[(row["scene"], row["geometry_seed"])].append(value)
    values = np.asarray([np.mean(group) for group in groups.values()])
    if not len(values):
        return {"geometry_n": 0, "mean": None, "ci95": None}
    rng = np.random.default_rng(seed)
    bootstraps = values[rng.integers(0, len(values), (2000, len(values)))].mean(axis=1)
    return {"geometry_n": len(values), "mean": float(values.mean()),
            "median": float(np.median(values)), "std": float(values.std(ddof=1)) if len(values) > 1 else None,
            "ci95": np.percentile(bootstraps, [2.5, 97.5]).tolist()}


def save(fig, output, name):
    fig.savefig(output / (name + ".png"), dpi=180, bbox_inches="tight")
    fig.savefig(output / (name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads((args.source / "results.json").read_text())
    rows = data["runs"]
    args.output.mkdir(parents=True, exist_ok=True)
    configure_chinese()
    snapshots = []
    for scene in SCENES:
        for pattern in MASKS:
            with np.load(args.source / f"{scene}_{pattern}.npz") as source:
                sample = {name: source[name] for name in source.files}
                sample["scene"] = str(sample["scene"])
                sample["mask"] = str(sample["mask"])
                snapshots.append(sample)
    plot_results(args.output, rows, snapshots)
    keys = ("unknown_all_rmse_m", "unknown_all_mae_m", "unknown_all_p95_m",
            "unknown_accepted_rmse_m", "unknown_prediction_coverage", "unknown_picp95",
            "unknown_interval_width95_m", "unknown_nll", "unknown_smooth_normal_mae_deg")
    by_mask = {pattern: {key: clustered_stats([r for r in rows if r["mask"] == pattern], key)
                         for key in keys} for pattern in MASKS}
    by_scene = {scene: {key: clustered_stats([r for r in rows if r["scene"] == scene], key)
                        for key in keys} for scene in SCENES}
    statistics = {"source": str(args.source), "source_predictor_sha256": data.get("predictor_sha256"),
                  "unit": "scene + geometry_seed; masks grouped within geometry",
                  "by_mask": by_mask, "by_scene": by_scene}
    (args.output / "prediction_statistics.json").write_text(json.dumps(statistics, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    x = np.arange(len(MASKS))
    for key, color, name in (("unknown_all_rmse_m", "#cf5c36", "所有未观测位置"),
                             ("unknown_accepted_rmse_m", "#247a91", "候选预测位置")):
        stats = [by_mask[p][key] for p in MASKS]
        offset = -0.15 if key == "unknown_all_rmse_m" else 0.15
        means = [s["mean"] for s in stats]
        lower = [s["mean"] - s["ci95"][0] for s in stats]
        upper = [s["ci95"][1] - s["mean"] for s in stats]
        axes[0].bar(x + offset, means, 0.28, yerr=[lower, upper], capsize=3, color=color, label=name)
    axes[0].set_title("未知区误差及 95% 场景聚类区间")
    axes[0].set_ylabel("平均高程 RMSE（米）"); axes[0].legend(fontsize=8)
    coverage = [by_mask[p]["unknown_prediction_coverage"]["mean"] for p in MASKS]
    axes[1].bar(x, coverage, color="#247a91")
    axes[1].set_title("支撑/标准差门限保留的预测覆盖率")
    axes[1].set_ylabel("比例"); axes[1].set_ylim(0, 1.05)
    for ax in axes:
        ax.set_xticks(x, [NAMES[p] for p in MASKS]); ax.grid(axis="y", alpha=0.2)
    fig.suptitle("误差与覆盖率必须同时报告：拒绝预测不计作准确预测")
    save(fig, args.output, "04_error_coverage_tradeoff")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, key, title in ((axes[0], "unknown_all_rmse_m", "未知区高程误差（米）"),
                           (axes[1], "unknown_smooth_normal_mae_deg", "未知平滑区法向角误差（度）")):
        values = [by_scene[s][key]["mean"] for s in SCENES]
        ax.bar(np.arange(len(SCENES)), values, color="#247a91")
        ax.set_xticks(np.arange(len(SCENES)), [NAMES[s] for s in SCENES]); ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("几何类型分层：不能用大面积平地掩盖窄脊、台阶和沟壑误差")
    save(fig, args.output, "05_geometry_accuracy")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    times = ([r["fit_ms"] for r in rows], [r["predict_ms_including_gradients"] for r in rows])
    axes[0].boxplot(times, showmeans=True)
    axes[0].set_xticks([1, 2], ["特征提取与模型拟合", "推断与空间梯度"])
    axes[0].set_ylabel("耗时（毫秒）"); axes[0].set_title("记录的离线计算耗时（含冷启动样本）")
    axes[1].bar(x, [by_mask[p]["unknown_picp95"]["mean"] for p in MASKS], color="#247a91")
    axes[1].axhline(0.95, ls=":", color="#cf5c36", label="名义 95%")
    axes[1].set_xticks(x, [NAMES[p] for p in MASKS]); axes[1].set_ylim(0, 1.05)
    axes[1].set_title("名义 95% 区间实际覆盖率"); axes[1].legend(fontsize=8)
    for ax in axes: ax.grid(axis="y", alpha=0.2)
    save(fig, args.output, "06_runtime_and_calibration")

    lines = ["# 上游地形预测真实留出实验", "",
        f"- 原始数据：`{args.source}`；记录 `{len(rows)}` 组；独立几何实例 `{len({(r['scene'],r['geometry_seed']) for r in rows})}`。",
        "- 同一几何的四种留出有关联，点、航向、遮挡掩码均不重复计作独立世界。",
        "- 使用真实特征稀疏 GP 适配器，非原版论文完全复现。几何遮挡由解析地形与柱体的射线采样生成，不是 Isaac RTX 传感器实测。",
        "- 模型输入不含隐藏高度；GP 后验区间是待检验的名义区间，不能声称已校准。", "",
        "| 留出类型 | 未知区 RMSE（米） | RMSE 场景聚类 95% 区间 | 候选预测 RMSE（米） | 候选覆盖率 | 实际 PICP95 |",
        "|---|---:|---|---:|---:|---:|"]
    for pattern in MASKS:
        s = by_mask[pattern]
        error = s["unknown_all_rmse_m"]
        lines.append(f"| {NAMES[pattern]} | {error['mean']:.4f} | [{error['ci95'][0]:.4f}, {error['ci95'][1]:.4f}] | "
                     f"{s['unknown_accepted_rmse_m']['mean']:.4f} | {s['unknown_prediction_coverage']['mean']:.1%} | {s['unknown_picp95']['mean']:.1%} |")
    lines.extend(["", "## 实验图", ""])
    for name in ("01_unknown_prediction_statistics", "02_prediction_truth_panels", "03_unknown_cross_sections",
                 "04_error_coverage_tradeoff", "05_geometry_accuracy", "06_runtime_and_calibration"):
        lines.append(f"![{name}]({name}.png)\n")
    lines.extend(["## 必须保留的问题", "",
        "连续孔洞、遮挡和外推不能由随机缺点评测代替。若实际 PICP95 明显低于 95%，说明名义区间偏窄/模型不匹配；候选覆盖率高不是安全可信度高。当前在线默认不把未知预测放行为可通行，仅可选修正已观测高程。", "",
        "下一阶段需要独立验证集上的校准与拒绝门限、危险边界专项及 Isaac/实物可见性数据。新增参数不能在这一份最终测试集上调优后再宣称无偏结果。", "",
        "GP 潜在函数分布和带观测噪声的分布按 [GPyTorch 官方接口](https://docs.gpytorch.ai/en/stable/examples/01_Exact_GPs/Simple_GP_Regression.html) 分开记录。拟合与数据驱动平面参数的不确定性仍是条件估计，不包括所有模型误设。"])
    (args.output / "prediction_report.md").write_text("\n".join(lines) + "\n")
    print(f"[prediction analysis] {len(rows)} cases; figures/statistics/report={args.output}")


if __name__ == "__main__":
    main()
