#!/usr/bin/env python3
"""Read-only, source-clustered figures for the five-category cloud campaign."""

import argparse
from collections import Counter, defaultdict
from importlib.metadata import distribution
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import mpl_toolkits
toolkits = str(distribution("matplotlib").locate_file("mpl_toolkits"))
mpl_toolkits.__path__ = [toolkits] + [p for p in mpl_toolkits.__path__ if p != toolkits]
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.projections import register_projection
register_projection(Axes3D)
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from matplotlib.ticker import FormatStrFormatter

from plot_style import configure_chinese
from pointcloud_batch import atomic_json, digest

FAMILIES = {"hilly": "丘陵", "forest": "森林", "ruin": "废墟", "road": "城市道路", "indoor": "室内"}
CONDITIONS = {"point15": "随机删去15%点", "hole16": "内部1.6米孔洞", "frontier50": "单侧隐藏50%宽度"}
COLORS = ("#217c91", "#bf6540", "#65717c")


def grouped_values(rows, key):
    """First average within each source cluster, never over all correlated poses."""
    groups = defaultdict(list)
    for row in rows:
        value = row.get(key)
        if row["status"] == "ok" and value is not None and np.isfinite(value):
            groups[row["cluster_id"]].append(float(value))
    return np.array([np.mean(values) for _, values in sorted(groups.items())])


def clustered_summary(rows, key, resamples=2000):
    values = grouped_values(rows, key)
    if not len(values):
        return {"mean": None, "low": None, "high": None, "clusters": 0, "cases": 0}
    mean = float(values.mean())
    if len(values) > 1:
        rng = np.random.default_rng(81261013)
        means = values[rng.integers(0, len(values), (resamples, len(values)))].mean(axis=1)
        low, high = map(float, np.quantile(means, [.025, .975]))
    else:
        low = high = mean
    return {"mean": mean, "low": low, "high": high, "clusters": len(values),
            "cases": sum(r["status"] == "ok" and r.get(key) is not None for r in rows)}


def ratio_rows(rows):
    """Confusion ratios use pooled counts inside a cluster, then equal clusters."""
    groups = defaultdict(lambda: Counter())
    for row in rows:
        if row["status"] == "ok" and row.get("reference_states", 0):
            groups[row["cluster_id"]].update({k: row[k] for k in ("tp", "fn", "fp", "tn")})
    output = []
    for cluster, c in sorted(groups.items()):
        positive, negative = c["tp"] + c["fn"], c["fp"] + c["tn"]
        output.append({"status": "ok", "cluster_id": cluster,
            "recall": c["tp"] / positive if positive else None,
            "false_safe": c["fn"] / positive if positive else None,
            "false_hazard": c["fp"] / negative if negative else None})
    return output


def save(fig, output, name):
    fig.savefig(output / (name + ".pdf"), bbox_inches="tight")
    fig.savefig(output / (name + ".png"), bbox_inches="tight", dpi=180)
    plt.close(fig)


def validate_campaign(record, catalog, allow_preflight=False):
    selected = record["settings"]["selected_ids"]
    if len(selected) != len(set(selected)):
        raise ValueError("duplicate source IDs in campaign plan")
    if not allow_preflight and set(selected) != {d["id"] for d in catalog["datasets"]}:
        raise ValueError("partial source campaign; use --allow-preflight only for development figures")
    conditions = record["settings"]["conditions"]
    if set(conditions) != set(CONDITIONS) or len(conditions) != len(CONDITIONS):
        raise ValueError("these figures require all three frozen conditions")
    expected = {source + "__" + kind for source in selected for kind in conditions}
    rows = record["runs"]
    if {r["id"] for r in rows} != expected or len(rows) != len(expected) or record["planned_cases"] != len(expected):
        raise ValueError("incomplete or duplicate campaign: refuse final figures")


def bars(ax, stats, key, factor, ylabel, title):
    for j, (kind, color) in enumerate(zip(CONDITIONS, COLORS)):
        s = [stats[f][kind][key] for f in FAMILIES]
        y = np.array([x["mean"] * factor if x["mean"] is not None else np.nan for x in s])
        lower = np.array([(x["mean"] - x["low"]) * factor if x["mean"] is not None else 0 for x in s])
        upper = np.array([(x["high"] - x["mean"]) * factor if x["mean"] is not None else 0 for x in s])
        ax.bar(np.arange(5) + (j - 1) * .25, y, .23, color=color, label=CONDITIONS[kind],
               yerr=np.maximum(0, [lower, upper]), capsize=2, error_kw={"linewidth": .8})
    ax.set_xticks(range(5), list(FAMILIES.values()))
    ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(axis="y", alpha=.2)
    ax.set_xlabel("点云类别（误差条：按来源分组的95%均值区间）")


def source_views(catalog, dataset_path, output):
    fig = plt.figure(figsize=(13, 8.5), layout="constrained")
    selected = {}
    for i, (family, label) in enumerate(FAMILIES.items()):
        item = next(d for d in catalog["datasets"] if d["family"] == family)
        selected[family] = item["id"]
        with np.load(dataset_path.parent / item["path"]) as a:
            points = a["full_xyz"]
        ids = np.random.default_rng(20261013).choice(len(points), min(len(points), 16000), replace=False)
        points = points[ids]
        ax = fig.add_subplot(2, 3, i + 1, projection="3d")
        cloud = ax.scatter(*points.T, c=points[:, 2], s=.35, cmap="terrain", rasterized=True)
        ax.set_xlabel("x（米）", labelpad=3); ax.set_ylabel("y（米）", labelpad=3); ax.set_zlabel("z", labelpad=1)
        for axis, dimension in ((ax.xaxis, 0), (ax.yaxis, 1), (ax.zaxis, 2)):
            axis.set_ticks(np.linspace(points[:, dimension].min(), points[:, dimension].max(), 3 if dimension < 2 else 2))
            axis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.tick_params(labelsize=8, pad=0)
        ax.set_box_aspect(np.maximum(np.ptp(points, axis=0), .2)); ax.view_init(elev=30, azim=-62)
        ax.set_title(label + "：首个预定样例", fontsize=12)
        fig.colorbar(cloud, ax=ax, shrink=.5, pad=.06, label="高度z（米）")
    ax = fig.add_subplot(2, 3, 6); ax.axis("off")
    ax.text(0, .95, "2500份点云输入，非2500个独立实景\n\n"
        "丘陵 / 森林 / 废墟：各500个新几何\n"
        "室内：500份采样，53个OBJ资源\n"
        "  按49组布局统计，52个几何哈希\n"
        "城市道路：500个位置，11条序列\n\n"
        "EPFL与道路源范围：50×50米\n"
        "模型评价窗口：20.8×20.8米\n"
        "室内保持房间原尺寸，不放大\n\n"
        "城市z未独立校正重力方向\n"
        "颜色是高度，不是风险", va="top", fontsize=11, linespacing=1.5)
    fig.suptitle("实际取得并参与测试的五类点云\n固定展示每类首个输入；点云显示抽样不影响模型输入", fontsize=15)
    save(fig, output, "01_sources")
    return selected


def missingness_views(results, output):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4), layout="constrained")
    for ax, kind in zip(axes, CONDITIONS):
        with np.load(results.parent / ("forest__" + kind + "__snapshot.npz")) as a:
            points, origin = a["raw_xyz"], a["query_origin"]
        ax.scatter(points[:, 0], points[:, 1], c=points[:, 2], s=.08, cmap="terrain", rasterized=True)
        ax.set_xlim(origin[0], origin[0] + 20.8); ax.set_ylim(origin[1], origin[1] + 20.8)
        ax.set_aspect("equal"); ax.set_title(CONDITIONS[kind]); ax.set_xlabel("局部x（米）"); ax.set_ylabel("局部y（米）")
    fig.suptitle("同一森林点云，先删点再提特征：三种难度不同的缺测任务\n"
                 "随机删点不是随机删格；孔洞测试插值；单侧隐藏测试外推边界", fontsize=13)
    save(fig, output, "02_missingness")


def spatial_views(record, catalog, results, dataset, output):
    from spatial_evidence import reconstruct_spatial_case, provenance
    from matplotlib.patches import Rectangle
    selected = {}
    provenance_rows = {}
    for family, label in FAMILIES.items():
        row = next(r for r in record["runs"] if r["family"] == family and r["condition"] == "hole16" and r["status"] == "ok")
        selected[family] = row["id"]
        d = reconstruct_spatial_case(record, catalog, results, dataset, row)
        provenance_rows[family] = provenance(d)
        risk = d["saved"]; n = d["counts"]
        extent = [d["origin"][0], d["origin"][0] + 20.8, d["origin"][1], d["origin"][1] + 20.8]
        hidden = np.isfinite(d["legacy_reference"]) & ~np.isfinite(d["observed"])
        error = np.where(hidden, (d["mean"] - d["legacy_reference"]) * 100, np.nan)
        lo, hi = np.nanmin(d["before"]), np.nanmax(d["before"])
        fig, axes = plt.subplots(2, 3, figsize=(11, 7), layout="constrained")
        fields = (d["before"], d["after_before_noise"], error, d["std"] * 100, None, None)
        titles = (f"(a) 删孔前地面观测：{n['before_cells']}格", f"(b) 删孔后地面观测：{n['after_cells']}格",
                  f"(c) 旧参考支持的隐藏误差：{n['legacy_hidden_cells']}格",
                  "(d) GP高程标准差", "(e) 预测足迹风险：5°", "(f) 旧评分足迹参考：5°")
        for j, (ax, field, title) in enumerate(zip(axes.flat, fields, titles)):
            if j < 2:
                artist = ax.imshow(field, origin="lower", extent=extent, cmap="terrain", vmin=lo, vmax=hi)
                unit = "高度（米）"
                center = d["origin"] + 10.4
                ax.add_patch(Rectangle(center - .8, 1.6, 1.6, fill=False, edgecolor="#b34240", ls="--", lw=1.2))
            elif j == 2:
                limit = max(1, float(np.nanmax(np.abs(field))) if hidden.any() else 1)
                artist = ax.imshow(field, origin="lower", extent=extent, cmap="RdBu_r", vmin=-limit, vmax=limit)
                unit = "预测减参考（厘米）"
            elif j == 3:
                artist = ax.imshow(field, origin="lower", extent=extent, cmap="magma", vmin=0)
                unit = "标准差（厘米）"
            else:
                ids = risk["supported"][0]
                values = risk["predicted_risk"] if j == 4 else risk["reference_risk"]
                artist = ax.scatter(*risk["query_xy"][ids].T, c=values[0, ids],
                    cmap="RdYlGn_r", vmin=0, vmax=1, s=24, edgecolors="none")
                unit = "风险分数（无量纲，非概率）"
            ax.set_xlim(extent[:2]); ax.set_ylim(extent[2:]); ax.set_aspect("equal")
            ax.set_xlabel("局部x（米）"); ax.set_ylabel("局部y（米）"); ax.set_title(title, fontsize=10)
            fig.colorbar(artist, ax=ax, shrink=.73, label=unit, pad=.02)
        fig.suptitle(label + "：同一来源的删孔对照与已保存预测\n"
            "(a)(b)为加噪前源点；红框为1.6米孔洞；风险仅比较同一可评分位置", fontsize=13)
        save(fig, output, "spatial_" + family)
    atomic_json(output / "spatial_provenance.json", provenance_rows)
    return selected


def yaw_views(record, results, output):
    fig, axes = plt.subplots(5, 1, figsize=(10.2, 11.4), layout="constrained")
    selected = {}
    for ax, (family, label) in zip(axes, FAMILIES.items()):
        rows = [r for r in record["runs"] if r["family"] == family and r["condition"] == "point15" and r["status"] == "ok"]
        for row in rows:
            with np.load(results.parent / (row["id"] + ".npz")) as a:
                support = a["supported"]; candidates = np.flatnonzero(support.all(axis=0))
                if not len(candidates):
                    continue
                index = candidates[0]; xy = a["query_xy"][index]
                reference, predicted = a["reference_risk"][:, index], a["predicted_risk"][:, index]
            break
        else:
            ax.text(.5, .5, "没有36航向均有参考的查询位置", ha="center", transform=ax.transAxes)
            continue
        angles = (np.arange(36) + .5) * 10
        ax.plot(angles, reference, "o-", ms=3, lw=1.4, color=COLORS[0], label="完整来源足迹参考")
        ax.plot(angles, predicted, "s--", ms=3, lw=1.3, color=COLORS[1], label="本工程风险输出")
        ax.axhline(.58, color="#65717c", ls=":", lw=1, label="分类阈值0.58")
        ax.set_ylim(-.05, 1.05); ax.set_xlim(0, 360); ax.set_xticks(np.arange(0, 361, 60))
        ax.set_ylabel("风险分数[0,1]"); ax.set_xlabel("航向θ（度，从局部+x轴逆时针）")
        ax.set_title(f"{label}：首个全航向有参考点，x={xy[0]:.2f}米，y={xy[1]:.2f}米", fontsize=11)
        ax.grid(alpha=.2)
        selected[family] = {"run": row["id"], "query_index": int(index), "xy": xy.tolist()}
    axes[0].legend(ncol=3, fontsize=9)
    fig.suptitle("固定位置、改变航向：评价SE(2)风险，而不是二维高程曲线\n"
        "预定顺序首个完整参考位置，不挑选最优曲线；室内/道路参考是几何代理", fontsize=14)
    save(fig, output, "08_yaw_profiles")
    return selected


def difficulty_views(record, catalog, results, output):
    """Describe input difficulty without selecting or excluding by model scores."""
    items = {d["id"]: d for d in catalog["datasets"]}
    values = []
    for row in record["runs"]:
        if row["status"] != "ok" or row["condition"] != "point15":
            continue
        with np.load(results.parent / (row["id"] + ".npz")) as a:
            height = a["reference_height"]
        gradients = np.r_[np.diff(height, axis=0).ravel(), np.diff(height, axis=1).ravel()] / .2
        gradients = gradients[np.isfinite(gradients)]
        values.append({"id": row["dataset_id"], "family": row["family"], "cluster_id": row["cluster_id"],
            "relief_m": float(np.nanmax(height) - np.nanmin(height)),
            "gradient_angle_p95_deg": float(np.rad2deg(np.arctan(np.quantile(np.abs(gradients), .95)))) if len(gradients) else None,
            "mode": items[row["dataset_id"]].get("shader_parameters", {}).get("type"),
            "hidden_rmse_m": row["hidden_rmse_m"]})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout="constrained")
    for ax, key, label, title in zip(axes[:2], ("relief_m", "gradient_angle_p95_deg"),
        ("局部参考高程最大减最小（米）", "相邻地面梯度角P95（度）"),
        ("输入高程起伏跨度", "输入相邻地面变化难度")):
        boxes = [[v[key] for v in values if v["family"] == f and v[key] is not None] for f in FAMILIES]
        ax.boxplot(boxes, positions=np.arange(5), whis=(0, 100), widths=.5,
            medianprops={"color": COLORS[1], "linewidth": 1.7}, boxprops={"color": COLORS[0]})
        ax.set_xticks(range(5), list(FAMILIES.values())); ax.set_title(title); ax.set_ylabel(label); ax.grid(axis="y", alpha=.2)
    axes[0].set_yscale("symlog", linthresh=.1)
    axes[0].set_ylim(0, max(v["relief_m"] for v in values) * 1.2)
    modes = ("分形噪声", "脊状噪声", "团状噪声", "瑞士湍流")
    boxes = [[v["hidden_rmse_m"] * 100 for v in values if v["mode"] == mode and v["hidden_rmse_m"] is not None] for mode in range(4)]
    axes[2].boxplot(boxes, positions=np.arange(4), whis=(0, 100), widths=.5,
        medianprops={"color": COLORS[1], "linewidth": 1.7}, boxprops={"color": COLORS[0]})
    axes[2].set_xticks(range(4), modes, rotation=15); axes[2].set_yscale("log")
    axes[2].set_ylabel("随机删点15%：隐藏高程RMSE（厘米）"); axes[2].set_title("生成模式分层（全部样例保留）")
    axes[2].grid(axis="y", alpha=.2)
    fig.suptitle("输入难度与失败来源：原始生成模式并非都属于温和小丘陵\n"
        "箱体为点云实例四分位数，须线为全部范围，非置信区间；道路z未独立对齐重力", fontsize=13)
    save(fig, output, "09_difficulty")
    atomic_json(output / "difficulty_cases.json", values)
    return {f: {"relief_median_m": float(np.median([v["relief_m"] for v in values if v["family"] == f])),
               "relief_max_m": max(v["relief_m"] for v in values if v["family"] == f)} for f in FAMILIES}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-preflight", action="store_true", help="explicitly allow a small subset; not final report assets")
    parser.add_argument("--spatial-only", action="store_true", help="correct matched spatial illustrations; keep numeric summaries unchanged")
    args = parser.parse_args()
    record, catalog = json.loads(args.results.read_text()), json.loads(args.dataset.read_text())
    if record["settings"]["dataset_sha256"] != digest(args.dataset):
        raise ValueError("recorded inputs differ from supplied catalog")
    rows = record["runs"]
    validate_campaign(record, catalog, args.allow_preflight)
    args.output.mkdir(parents=True, exist_ok=True); configure_chinese()
    # Matplotlib's Type42 embedding can mislabel CFF Noto collections. Use the
    # installed TrueType fallback for portable, searchable Chinese PDF figures.
    portable = Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
    if portable.exists():
        font_manager.fontManager.addfont(str(portable))
        plt.rcParams["font.family"] = ["DejaVu Sans", font_manager.FontProperties(fname=str(portable)).get_name()]
    if args.spatial_only:
        spatial_views(record, catalog, args.results, args.dataset, args.output)
        print("[spatial] matched case illustrations rebuilt; old numeric scores unchanged")
        return
    keys = ("risk_mae", "risk_rmse", "risk_error_variance", "hidden_rmse_m", "hidden_mae_m", "hidden_picp95",
            "hidden_width95_m", "hidden_candidate_coverage", "accepted_hidden_rmse_m")
    stats = {}
    for family in FAMILIES:
        stats[family] = {}
        for kind in CONDITIONS:
            subset = [r for r in rows if r["family"] == family and r["condition"] == kind]
            summary = {key: clustered_summary(subset, key) for key in keys}
            ratios = ratio_rows(subset)
            summary.update({key: clustered_summary(ratios, key) for key in ("recall", "false_safe", "false_hazard")})
            summary["statuses"] = dict(Counter(r["status"] for r in subset))
            summary["without_risk_reference"] = sum(r["status"] == "ok" and not r["reference_states"] for r in subset)
            summary["reference_states"] = sum(r.get("reference_states", 0) for r in subset)
            summary["confusion_counts"] = {k: sum(r.get(k, 0) for r in subset) for k in ("tp", "fn", "fp", "tn")}
            stats[family][kind] = summary
    summary = {"results_sha256": digest(args.results), "dataset_sha256": digest(args.dataset),
        "planned_runs": record["planned_cases"], "statuses": dict(Counter(r["status"] for r in rows)),
        "actual_devices": dict(Counter(r.get("actual_device", "none") for r in rows)),
        "actual_backends": dict(Counter(r.get("actual_backend", "none") for r in rows)),
        "statistics": stats, "aggregation": "case mean inside source cluster, then equal cluster mean; 2000 cluster bootstrap samples"}
    summary["source_examples"] = source_views(catalog, args.dataset, args.output)
    missingness_views(args.results, args.output)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), layout="constrained")
    bars(axes[0], stats, "risk_mae", 1, "通行风险MAE（无量纲）", "分数偏差：越低越好")
    bars(axes[1], stats, "risk_error_variance", 1, "风险误差的经验方差", "误差波动：不是GP后验方差")
    axes[0].legend(fontsize=9)
    fig.suptitle("核心对象：完整参考足迹上的通行风险误差\n参考为Hunter足迹几何代理，与论文缺失的二维真值公式不等价", fontsize=14)
    save(fig, args.output, "03_risk_errors")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), layout="constrained")
    for ax, key, label, title in zip(axes, ("recall", "false_safe", "false_hazard"),
            ("TP/(TP+FN)（%）", "FN/(TP+FN)（%）", "FP/(FP+TN)（%）"),
            ("危险识别：越高越好", "危险被判低风险：越低越好", "可通过参考被保守拒绝：越低越好")):
        bars(ax, stats, key, 100, label, title); ax.set_ylim(0, 105)
    axes[0].legend(fontsize=9)
    fig.suptitle("安全分类与保守代价必须同时报告\n阈值0.58；未观测风险0.65会造成保守拒绝；没有危险分母时不画召回柱", fontsize=13)
    save(fig, args.output, "04_safety")
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), layout="constrained")
    for ax, key, factor, label, title in zip(axes.flat,
        ("hidden_rmse_m", "hidden_picp95", "hidden_width95_m", "hidden_candidate_coverage"), (100, 100, 100, 100),
        ("隐藏高程RMSE（厘米）", "参考落入名义95%区间（%）", "平均95%区间宽度（厘米）", "隐藏格子通过门限（%）"),
        ("高程预测误差", "区间实际覆盖率", "区间有多宽", "候选保留比例（非可通行率）")):
        bars(ax, stats, key, factor, label, title)
    axes[0, 0].legend(fontsize=9)
    axes[0, 1].axhline(95, color=COLORS[1], ls=":", lw=1); axes[0, 1].set_ylim(0, 105); axes[1, 1].set_ylim(0, 105)
    fig.suptitle("辅助对象：没有输入地面点的格子上，高程预测与不确定性\n均值区间和单格95%预测区间不是同一概念；候选不自动写入已观测地图", fontsize=13)
    save(fig, args.output, "05_height_uncertainty")
    timing = {}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), layout="constrained")
    for ax, key, title in zip(axes, ("fit_ms", "predict_ms", "risk_ms"),
             ("GP特征提取及拟合", "GP预测10816个查询", "当前柱状占据及BGK/SEB")):
        timing[key] = {}
        for family, color in zip(FAMILIES, ("#217c91", "#bf6540", "#697283", "#a89533", "#57936a")):
            values = np.sort([r[key] for r in rows if r["family"] == family and r["status"] == "ok"])
            if not len(values):
                continue
            timing[key][family] = {"n": len(values), "p50_ms": float(np.median(values)),
                                  "p95_ms": float(np.quantile(values, .95)), "max_ms": float(values[-1])}
            ax.plot(values, np.arange(1, len(values) + 1) / len(values), color=color, label=FAMILIES[family])
        ax.set_xscale("log"); ax.set_ylim(0, 1.02); ax.set_title(title); ax.set_xlabel("实际耗时（毫秒，对数刻度）")
        ax.set_ylabel("已完成案例的累计比例"); ax.grid(alpha=.2)
    axes[0].legend(fontsize=9)
    fig.suptitle("实测CUDA/CuPy组件耗时分布（包含首轮初始化，非端到端ROS延迟）\n模型每份点云重置；不含文件下载、参考生成、保存、仿真或规划控制", fontsize=13)
    save(fig, args.output, "06_runtime"); summary["runtime"] = timing
    source_counts = {}
    for family in FAMILIES:
        items = [d for d in catalog["datasets"] if d["family"] == family]
        source_counts[family] = {"clouds": len(items), "clusters": len({d["cluster_id"] for d in items}),
            "geometry_hashes": len({d["geometry_sha256"] for d in items})}
    summary["source_counts"] = source_counts
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), layout="constrained")
    x = np.arange(5)
    axes[0].bar(x - .15, [source_counts[f]["clouds"] for f in FAMILIES], .28, label="点云输入数", color=COLORS[0])
    axes[0].bar(x + .15, [source_counts[f]["clusters"] for f in FAMILIES], .28, label="来源统计组数", color=COLORS[1])
    axes[0].set_ylabel("数量"); axes[0].set_title("避免把采样次数当场景数"); axes[0].legend(fontsize=9)
    bars(axes[1], {f: {k: {"support": clustered_summary(
        [dict(r, support=r.get("reference_states", 0) / 4608) for r in rows if r["family"] == f and r["condition"] == k and r["status"] == "ok"], "support")}
        for k in CONDITIONS} for f in FAMILIES}, "support", 100, "计划4608足迹状态中有参考（%）", "参考缺失必须明示")
    axes[1].set_ylim(0, 105)
    balance = {f: {k: {"hazard": clustered_summary(
        [dict(r, hazard=(r["tp"] + r["fn"]) / r["reference_states"]) for r in rows
         if r["family"] == f and r["condition"] == k and r["status"] == "ok" and r["reference_states"]], "hazard")}
        for k in CONDITIONS} for f in FAMILIES}
    bars(axes[2], balance, "hazard", 100, "有参考状态中硬危险（%）", "危险/非危险样本比例")
    axes[2].set_ylim(0, 105)
    summary["reference_hazard_fraction"] = balance
    axes[0].set_xticks(x, list(FAMILIES.values())); axes[0].grid(axis="y", alpha=.2)
    fig.suptitle("统计单位、可评价范围与参考类别比例\n室内地板边界和道路稀疏参考限制评分；全部危险时，高召回不能证明正常区域可规划", fontsize=13)
    save(fig, args.output, "07_support")
    summary["spatial_examples"] = spatial_views(record, catalog, args.results, args.dataset, args.output)
    summary["yaw_examples"] = yaw_views(record, args.results, args.output)
    summary["input_difficulty"] = difficulty_views(record, catalog, args.results, args.output)
    excluded = [r for r in rows if r["status"] != "ok" or not r.get("reference_states", 0)]
    atomic_json(args.output / "exceptions.json", excluded)
    atomic_json(args.output / "summary.json", summary)
    with (args.output / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=("family", "condition", "metric", "mean", "low95", "high95", "clusters", "cases"))
        writer.writeheader()
        for family in FAMILIES:
            for kind in CONDITIONS:
                for key in (*keys, "recall", "false_safe", "false_hazard"):
                    s = stats[family][kind][key]
                    writer.writerow({"family": family, "condition": kind, "metric": key, "mean": s["mean"],
                        "low95": s["low"], "high95": s["high"], "clusters": s["clusters"], "cases": s["cases"]})
    with (args.output / "runs.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader(); writer.writerows(rows)
    counts = summary["statuses"]
    commands = {"PlannedRuns": record["planned_cases"], "PassedRuns": counts.get("ok", 0),
                "FailedRuns": counts.get("failed", 0), "UnevaluableRuns": counts.get("not_evaluable", 0),
                "NoRiskReferenceRuns": sum(r["status"] == "ok" and not r["reference_states"] for r in rows),
                "CudaRuns": summary["actual_devices"].get("cuda", 0),
                "CupyRuns": summary["actual_backends"].get("cupy", 0)}
    (args.output / "counts.tex").write_text("\n".join(rf"\newcommand{{\{key}}}{{{value}}}"
        for key, value in commands.items()) + "\n", encoding="utf-8")
    lines = []
    for family, label in FAMILIES.items():
        subset = [r for r in rows if r["family"] == family]
        counts = Counter(r["status"] for r in subset)
        lines.append(f"{label} & {source_counts[family]['clouds']} & {source_counts[family]['clusters']} & "
            f"{counts['ok']} & {counts['not_evaluable']} & {counts['failed']} & "
            f"{sum(r['status'] == 'ok' and not r['reference_states'] for r in subset)}" + r" \\")
    (args.output / "eligibility_rows.tex").write_text("\n".join(lines) + "\n" + r"\bottomrule" + "\n", encoding="utf-8")
    fmt = lambda s, factor=1: "--" if s["mean"] is None else f"{s['mean'] * factor:.3f}"
    lines = []
    for family, label in FAMILIES.items():
        for kind, condition in CONDITIONS.items():
            s = stats[family][kind]
            condition = condition.replace("%", r"\%")
            lines.append(f"{label} & {condition} & {s['risk_mae']['clusters']} & " +
                " & ".join((fmt(s['risk_mae']), fmt(s['risk_error_variance']), fmt(s['recall'], 100),
                              fmt(s['false_hazard'], 100), fmt(s['hidden_rmse_m'], 100))) + r" \\")
    (args.output / "result_rows.tex").write_text("\n".join(lines) + "\n" + r"\bottomrule" + "\n", encoding="utf-8")
    lines = []
    for family, label in FAMILIES.items():
        values = [timing[k].get(family) for k in ("fit_ms", "predict_ms", "risk_ms")]
        lines.append(label + " & " + " & ".join("--" if s is None else f"{s['p50_ms']:.1f}/{s['p95_ms']:.1f}" for s in values) + r" \\")
    (args.output / "timing_rows.tex").write_text("\n".join(lines) + "\n" + r"\bottomrule" + "\n", encoding="utf-8")
    print(json.dumps({"statuses": summary["statuses"], "figures": len(list(args.output.glob("*.pdf")))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
