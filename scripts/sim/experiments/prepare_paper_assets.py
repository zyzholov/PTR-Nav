#!/usr/bin/env python3
"""Replot archived measurements for a readable paper; never rerun a predictor."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, Patch
import numpy as np

from analyze_prediction_holdout import clustered_stats
from analyze_navigation_subject import read_trials, plot_trajectories
from plot_style import configure_chinese, scenario_name, wilson_interval
from prediction_holdout import MASKS, NAMES, SCENES, specification

ROOT = Path(__file__).resolve().parents[3]


def save(fig, output, name):
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def sample(source, scene, kind="block"):
    with np.load(source / f"{scene}_{kind}.npz", allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def missingness_patterns(source, data, output):
    scene = "hills"
    first = next(row for row in data["runs"] if row["scene"] == scene)
    spec = specification(scene, first["geometry_seed"])
    cmap = ListedColormap(["#247a91", "#f2d5a4"])
    titles = {"random": "(a) 随机稀疏：零散位置没有点",
              "block": "(b) 连续孔洞：中间一整块没有点",
              "occlusion": "(c) 几何遮挡：视线被挡住",
              "frontier": "(d) 单侧外推：只观测边界一侧"}
    purposes = {"random": "测零散缺点恢复；周围通常仍有观测",
                "block": "测孔洞填补；空白区域周围仍有观测",
                "occlusion": "测受控遮挡后的预测；不是 RTX 实测",
                "frontier": "测向未观测一侧预测；不是内部插值"}
    fig, axes = plt.subplots(2, 2, figsize=(10.3, 9.0), layout="constrained")
    examples = []
    for ax, kind in zip(axes.flat, MASKS):
        s = sample(source, scene, kind)
        hidden = ~s["visible"].astype(bool)
        ax.imshow(hidden.astype(float), origin="lower", extent=s["extent"],
                  cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        low, high = s["extent"][:2]
        spacing = (high - low) / hidden.shape[1]
        if kind == "occlusion":
            sensor_x = low + spacing / 2 - 0.2
            ax.scatter(sensor_x, 0, marker="*", s=130, color="#292b2c", zorder=4)
            ax.add_patch(Circle((0, 0), 0.35, color="#292b2c", zorder=3))
            ax.annotate("射线起点", (sensor_x, 0), xytext=(-2.8, -1.5),
                        fontsize=10, arrowprops={"arrowstyle": "->", "color": "#292b2c"})
            ax.annotate("遮挡柱体", (0, 0), xytext=(-1.6, 1.4), fontsize=10,
                        arrowprops={"arrowstyle": "->", "color": "#292b2c"})
        if kind == "frontier":
            ax.axvline(spec["offset"], color="#292b2c", ls="--", lw=1.5)
            ax.text(spec["offset"] + 0.15, 2.4, "观测边界", fontsize=10,
                    va="center", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85})
        ax.set_xlim(low - 0.45, high + 0.10)
        ax.set_ylim(s["extent"][2:])
        ax.set_aspect("equal")
        ax.set_xticks([-3, -2, -1, 0, 1, 2, 3]); ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])
        ax.set_title(titles[kind] + f"\n此示例隐藏 {100 * hidden.mean():.1f}% 格子", fontsize=12)
        ax.set_xlabel("地形平面 x（米）\n" + purposes[kind], fontsize=10)
        ax.set_ylabel("地形平面 y（米）")
        examples.append({"mask": kind, "geometry_seed": first["geometry_seed"],
                         "observed_cells": int((~hidden).sum()), "hidden_cells": int(hidden.sum()),
                         "hidden_fraction": float(hidden.mean()),
                         "source": str((source / f"{scene}_{kind}.npz").relative_to(ROOT)),
                         "source_sha256": hashlib.sha256((source / f"{scene}_{kind}.npz").read_bytes()).hexdigest()})
    fig.suptitle("四种缺测方式：同一地形，哪些点允许进入模型？\n"
                 "蓝色＝可见输入；浅橙＝隐藏评分区；颜色不表示高度或风险", fontsize=13)
    fig.legend(handles=[Patch(facecolor="#247a91", label="可见：作为模型输入"),
                        Patch(facecolor="#f2d5a4", label="隐藏：删去输入，只用真值评分")],
               loc="outside lower center", ncol=2, fontsize=11, frameon=False)
    save(fig, output, "missingness_patterns")
    return {"scene": scene, "examples": examples, "source": "archived first-geometry visibility masks"}


def map_panels(source, output, scene):
    s = sample(source, scene)
    hidden = ~s["visible"].astype(bool)
    fields = [s["truth"], np.where(~hidden, s["observations"], np.nan), s["mean"],
              np.where(hidden, 100 * (s["mean"] - s["truth"]), np.nan),
              np.where(hidden, 100 * np.sqrt(s["variance"]), np.nan), hidden.astype(float)]
    titles = ["(a) 完整参考地面", "(b) 模型可见输入；白色为空洞", "(c) GP 预测高程",
              "(d) 只在隐藏区域计误差", "(e) 隐藏区预测标准差", "(f) 输入可见性掩码"]
    low = min(float(np.nanmin(a)) for a in fields[:3])
    high = max(float(np.nanmax(a)) for a in fields[:3])
    error_max = max(1.0, float(np.nanmax(np.abs(fields[3]))))
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 7.1), constrained_layout=True)
    for i, (ax, matrix, title) in enumerate(zip(axes.flat, fields, titles)):
        if i < 3:
            cmap, limits, label = "viridis", (low, high), "高程（米）"
        elif i == 3:
            cmap, limits, label = "RdBu_r", (-error_max, error_max), "预测 − 真值（厘米）"
        elif i == 4:
            cmap, limits, label = "magma", (0, float(np.nanmax(matrix))), "标准差（厘米）"
        else:
            cmap, limits, label = ListedColormap(["#dce3e6", "#bd493f"]), (0, 1), "0=可见；1=隐藏"
        artist = ax.imshow(matrix, origin="lower", extent=s["extent"], cmap=cmap,
                           vmin=limits[0], vmax=limits[1])
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("世界 x（米）"); ax.set_ylabel("世界 y（米）")
        fig.colorbar(artist, ax=ax, shrink=0.72, label=label, ticks=[0, 1] if i == 5 else None)
    fig.suptitle(f"{NAMES[scene]}上的孔洞补全：哪些位置没有输入，预测又错了多少？", fontsize=13)
    save(fig, output, f"surface_{scene}")


def cross_sections(source, output):
    fig, axes = plt.subplots(3, 1, figsize=(8.1, 8.0), constrained_layout=True)
    for ax, scene in zip(axes, ("ridge", "step", "ditch")):
        s = sample(source, scene)
        row = len(s["truth"]) // 2
        size = s["truth"].shape[1]
        low, high = s["extent"][:2]
        x = np.linspace(low, high, size, endpoint=False) + (high - low) / (2 * size)
        visible = s["visible"][row].astype(bool)
        for index in np.flatnonzero(~visible):
            ax.axvspan(x[index] - (high - low) / (2 * size), x[index] + (high - low) / (2 * size),
                       color="#a2a9ae", alpha=0.14, linewidth=0)
        mu = s["mean"][row]
        std = np.sqrt(s["variance"][row])
        ax.plot(x, s["truth"][row], color="#222222", label="完整地形真值")
        ax.plot(x, mu, color="#176e8c", label="模型预测均值")
        ax.fill_between(x, mu - 1.959964 * std, mu + 1.959964 * std,
                        color="#176e8c", alpha=0.18, label="名义 95% 高程区间")
        ax.scatter(x[visible], s["observations"][row][visible], s=16, color="#b55238", label="可见测量点")
        ax.set_title(f"{NAMES[scene]}：灰色竖带内没有点作为模型输入", fontsize=11)
        ax.set_xlabel("剖面 x（米）"); ax.set_ylabel("高程（米）"); ax.grid(alpha=0.2)
    axes[0].legend(ncol=2, fontsize=9)
    fig.suptitle("缺测区的预测是否保留了窄脊、台阶和沟壑？", fontsize=13)
    save(fig, output, "hidden_profiles")


def prediction_summary(rows, output):
    geometry_count = len({(row["scene"], row["geometry_seed"]) for row in rows})
    stats = {kind: {key: clustered_stats([r for r in rows if r["mask"] == kind], key)
                   for key in ("unknown_all_rmse_m", "unknown_all_mae_m", "unknown_prediction_coverage", "unknown_picp95")}
             for kind in MASKS}
    labels = ["随机\n稀疏", "连续\n孔洞", "几何\n遮挡", "单侧\n外推"]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 5.0), constrained_layout=True)
    for ax, key, factor, title, unit in zip(axes,
        ("unknown_all_rmse_m", "unknown_prediction_coverage", "unknown_picp95"), (100, 100, 100),
        ("(a) 高程误差：越低越好", "(b) 隐藏格子被保留多少？", "(c) 高程区间是否可信？"),
        ("平均高程 RMSE（厘米）", "候选格子比例（%）\n分母：全部隐藏格子",
         "真值落入区间比例（%）\n分母：全部隐藏格子")):
        values = [stats[k][key]["mean"] * factor for k in MASKS]
        lo = [max(0, (stats[k][key]["mean"] - stats[k][key]["ci95"][0]) * factor) for k in MASKS]
        hi = [max(0, (stats[k][key]["ci95"][1] - stats[k][key]["mean"]) * factor) for k in MASKS]
        ax.bar(np.arange(4), values, color="#247a91", yerr=[lo, hi], capsize=3)
        for i, value in enumerate(values):
            ax.text(i, value + hi[i] + (0.35 if key == "unknown_all_rmse_m" else 2),
                    f"{value:.1f}", ha="center", fontsize=14)
        ax.set_xticks(range(4), labels, fontsize=12)
        ax.set_xlabel("输入缺测方式（横轴为类别）", fontsize=12)
        ax.set_title(title, fontsize=14); ax.set_ylabel(unit, fontsize=13)
        ax.tick_params(axis="y", labelsize=12); ax.grid(axis="y", alpha=0.2)
        if key != "unknown_all_rmse_m":
            ax.set_ylim(0, 115); ax.set_yticks(np.arange(0, 101, 20))
        else:
            ax.set_ylim(0, max(v + h for v, h in zip(values, hi)) * 1.15)
        if key == "unknown_picp95":
            ax.axhline(95, color="#b55238", ls=":")
            ax.text(.98, 96, "名义目标：95%", transform=ax.get_yaxis_transform(),
                    ha="right", va="bottom", color="#b55238", fontsize=12)
    fig.suptitle(f"同一批 {geometry_count} 个地形 × 4 种缺测方式＝{len(rows)} 组试验\n"
                 "柱高＝各地形指标均值；黑线＝均值的 95% 统计置信区间", fontsize=14)
    save(fig, output, "prediction_summary")
    lines = []
    for kind in MASKS:
        s = stats[kind]
        error = s["unknown_all_rmse_m"]
        lines.append(f"{NAMES[kind]} & {error['geometry_n']} & {100 * error['mean']:.2f} & "
            f"[{100 * error['ci95'][0]:.2f}, {100 * error['ci95'][1]:.2f}] & "
            f"{100 * s['unknown_all_mae_m']['mean']:.2f} & "
            f"{100 * s['unknown_picp95']['mean']:.1f} & {100 * s['unknown_prediction_coverage']['mean']:.1f}" + r" \\")
    (output / "holdout_rows.tex").write_text("\n".join(lines) + "\n" + r"\bottomrule" + "\n")
    return stats


def cloud_stats(rows, key):
    groups = defaultdict(list)
    for row in rows:
        value = row.get(key)
        if row["status"] == "ok" and isinstance(value, (int, float)) and np.isfinite(value):
            groups[row["dataset_id"]].append(value)
    values = np.asarray([np.mean(v) for v in groups.values()])
    if not len(values): return None
    boot = values[np.random.default_rng(20261012).integers(0, len(values), (2000, len(values)))].mean(axis=1)
    return {"mean": float(values.mean()), "ci95": np.percentile(boot, [2.5, 97.5]).tolist(), "world_n": len(values)}


def batch_summary(data, output):
    rows = data["runs"]
    stats = {}
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.5), constrained_layout=True)
    lines = []
    for col, kind in enumerate(("random", "block", "frontier")):
        levels = sorted({r["severity"] for r in rows if r["kind"] == kind})
        label = {"random": "随机隐藏的格子比例（%）", "block": "连续孔洞宽度（米）", "frontier": "单侧未观测宽度比例（%）"}[kind]
        xs = np.asarray(levels) * (1 if kind == "block" else 100)
        chosen = [[r for r in rows if r["kind"] == kind and r["severity"] == level] for level in levels]
        for ax, key, scale, title in ((axes[0, col], "unknown_rmse_m", 100, "隐藏区域高程误差（厘米）"),
                                     (axes[1, col], "picp95", 100, "名义 95% 区间实际覆盖（%）")):
            group_stats = [cloud_stats(group, key) for group in chosen]
            values = [s["mean"] * scale for s in group_stats]
            errors = [[max(0, (s["mean"] - s["ci95"][0]) * scale) for s in group_stats],
                      [max(0, (s["ci95"][1] - s["mean"]) * scale) for s in group_stats]]
            ax.errorbar(xs, values, yerr=errors, color="#247a91", marker="o", capsize=4)
            ax.set_xticks(xs); ax.set_xlabel(label); ax.set_ylabel(title); ax.grid(alpha=0.2)
            if key == "picp95":
                ax.axhline(95, color="#b55238", ls=":"); ax.set_ylim(0, 105)
        axes[0, col].set_title({"random": "随机缺点", "block": "整块缺测", "frontier": "观测边界之外"}[kind])
        for level, group in zip(levels, chosen):
            s = {key: cloud_stats(group, key) for key in ("unknown_rmse_m", "unknown_mae_m", "picp95", "unknown_accepted_coverage")}
            stats[f"{kind}_{level}"] = s
            condition = (f"孔洞 {level:.1f} 米" if kind == "block" else
                         ("随机" if kind == "random" else "前沿") + f"隐藏 {level:.0%}".replace("%", r"\%"))
            count = sum(r["status"] == "ok" for r in group)
            lines.append(f"{condition} & {count}/{len(group)} & {s['unknown_rmse_m']['mean'] * 100:.2f} & "
                f"{s['unknown_mae_m']['mean'] * 100:.2f} & {s['picp95']['mean'] * 100:.1f} & "
                f"{s['unknown_accepted_coverage']['mean'] * 100:.1f}" + r" \\")
    fig.suptitle("文件化不完整点云试验：10 个新地形，120 个掩码/噪声条件", fontsize=12)
    save(fig, output, "batch_severity")
    (output / "batch_rows.tex").write_text("\n".join(lines) + "\n" + r"\bottomrule" + "\n")
    return stats


def yaw_diagnostic(output):
    before = json.loads((ROOT / "docs/figures/seb_direction_20261012/results.json").read_text())
    after = json.loads((ROOT / "docs/figures/seb_direction_fixed_20261012/results.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), constrained_layout=True)
    for ax, index in zip(axes, (2, 4)):
        old, new = before["curves"][index], after["curves"][index]
        yaw = np.degrees(new["yaw"])
        ax.plot(yaw, new["oracle"], color="#222222", label="解析姿态风险参考")
        ax.plot(yaw, old["seb"], "--", color="#b55238", label="修复前 SEB 分量")
        ax.plot(yaw, new["seb"], color="#247a91", label="修复后 SEB 分量")
        ax.set_title(f"同一坡面：{np.degrees(new['slope_rad']):.1f} 度")
        ax.set_xlabel("车辆航向（度）"); ax.set_ylabel("无量纲姿态风险分数")
        ax.set_ylim(0, 1.05); ax.set_xticks([0, 90, 180, 270, 360]); ax.grid(alpha=0.2)
    axes[0].legend(fontsize=9)
    fig.suptitle("测的是方向性几何合同，不是危险事件概率；前后版本为工程诊断", fontsize=12)
    save(fig, output, "seb_direction")


def runtime_pairs(gpu_rows, cpu_rows, output):
    key = lambda row: (row["scene"], row["geometry_seed"], row["mask"])
    cpu_keys = {key(r) for r in cpu_rows}
    gpu = [r for r in gpu_rows if key(r) in cpu_keys]
    fig, axes = plt.subplots(1, 2, figsize=(9.7, 3.8), constrained_layout=True)
    values = {}
    for ax, field, title in ((axes[0], "fit_ms", "特征与 GP 拟合"),
                             (axes[1], "predict_ms_including_gradients", "后验查询与空间梯度")):
        samples = [[r[field] for r in records] for records in (cpu_rows, gpu)]
        medians = [np.median(v) for v in samples]
        p95 = [np.percentile(v, 95) for v in samples]
        x = np.arange(2)
        ax.bar(x - .18, medians, .33, color="#247a91", label="P50")
        ax.bar(x + .18, p95, .33, color="#b55238", label="P95")
        ax.set_xticks(x, ["CPU", "CUDA GPU"]); ax.set_title(title)
        ax.set_ylabel("墙钟耗时（毫秒）"); ax.grid(axis="y", alpha=.2)
        values[field] = {name: {"n": len(v), "p50_ms": float(np.median(v)), "p95_ms": float(np.percentile(v, 95))}
                         for name, v in zip(("cpu", "cuda"), samples)}
    axes[0].legend(fontsize=9)
    fig.suptitle("相同 20 个地形/掩码条件的配对计时；含冷启动，不是整条 ROS 链路", fontsize=12)
    save(fig, output, "runtime_pairs")
    return values


def navigation_tables(output):
    stats = json.loads((ROOT / "docs/figures/gp_navigation_20261012/navigation_statistics.json").read_text())
    lines = []
    for kind in ("ramp", "detour", "crossing", "step"):
        row = stats[kind]
        lo, hi = row["success_wilson95"]
        metrics = row["metrics"]
        lines.append(f"{scenario_name(kind)} & {row['runs']} & {row['reached']} & {row['gate_passed']} & "
            f"[{lo * 100:.1f}, {hi * 100:.1f}] & {metrics['tracking_error_p95_m']['mean']:.3f} & "
            f"{metrics['planner_p95_ms']['mean']:.1f}" + r" \\")
    (output / "navigation_rows.tex").write_text("\n".join(lines) + "\n" + r"\bottomrule" + "\n")
    _, trials = read_trials(ROOT / "log/gp_navigation_fixed_20261012/manifest.json", True)
    plot_trajectories(output, trials, columns=2, legend_fontsize=10)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    configure_chinese()
    plt.rcParams.update({"font.size": 11, "legend.frameon": False})
    gpu_source = ROOT / "log/prediction_holdout_20261012"
    sources = {"gpu_holdout": gpu_source / "results.json",
        "cpu_holdout": ROOT / "log/prediction_holdout_cpu_20261012/results.json",
        "cloud_batch": ROOT / "log/pointcloud_batch_20261012/batch_manifest.json"}
    loaded = {key: json.loads(path.read_text()) for key, path in sources.items()}
    statistics = {"sources": {key: {"path": str(path.relative_to(ROOT)),
                                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                              for key, path in sources.items()},
        "missingness_examples": missingness_patterns(gpu_source, loaded["gpu_holdout"], output),
        "holdout": prediction_summary(loaded["gpu_holdout"]["runs"], output),
        "cloud_batch": batch_summary(loaded["cloud_batch"], output),
        "runtime_pairs": runtime_pairs(loaded["gpu_holdout"]["runs"], loaded["cpu_holdout"]["runs"], output),
        "navigation": navigation_tables(output)}
    for scene in ("hills", "ridge", "step"):
        map_panels(gpu_source, output, scene)
    cross_sections(gpu_source, output)
    yaw_diagnostic(output)
    (output / "derived_statistics.json").write_text(json.dumps(statistics, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote paper figures, auditable statistics and tables: {output}")


if __name__ == "__main__":
    main()
