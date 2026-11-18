#!/usr/bin/env python3
"""Plot recorded public-cloud results; never fit models while making figures."""

import argparse
from collections import Counter, defaultdict
from importlib.metadata import distribution
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import mpl_toolkits

# Debian preloads an older namespace even in a system-site-packages venv.
# Select mplot3d from the same installed wheel as matplotlib, only for this tool.
toolkits = str(distribution("matplotlib").locate_file("mpl_toolkits"))
mpl_toolkits.__path__ = [toolkits] + [p for p in mpl_toolkits.__path__ if p != toolkits]
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.projections import register_projection
register_projection(Axes3D)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

from pointcloud_batch import atomic_json, digest, read_points
from public_cloud_study import load_rellis
from plot_style import configure_chinese

NAMES = {"fsgp_desert": "沙漠样例", "fsgp_forest": "森林样例", "fsgp_hill": "丘陵样例",
         "fsgp_mountain": "山地样例", "fsgp_quick": "quick样例", "fsgp_snow": "雪地样例",
         "fsgp_vocano": "火山样例", "rellis_short_example": "RELLIS短片段"}
CONDITIONS = {"random": "随机删去15%格子", "block": "1.6米内部孔洞", "frontier": "单侧隐藏50%宽度"}


def save(fig, output, name):
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{name}.png", bbox_inches="tight", dpi=180)
    plt.close(fig)


def mean(rows, key):
    values = [r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])]
    return float(np.mean(values)) if values else None


def verify_disjoint_mask_seeds(batches):
    seen = set()
    for batch in batches:
        for row in batch["runs"]:
            key = (row["dataset_id"], row["kind"], row["severity"], row["mask_seed"])
            if key in seen:
                raise ValueError("overlapping mask seeds across supplied batches; do not count duplicate runs as new masks")
            seen.add(key)


def write_numbers(path, values):
    if any(not key.isascii() or not key.isalpha() for key in values):
        raise ValueError("LaTeX command names must contain ASCII letters only")
    path.write_text("\n".join(rf"\newcommand{{\{key}}}{{{value}}}" for key, value in values.items()) + "\n")


def source_views(catalog, output):
    wanted = ("fsgp_forest", "fsgp_hill", "fsgp_quick", "rellis_000000")
    fig = plt.figure(figsize=(10.5, 9.2), layout="constrained")
    for i, source_id in enumerate(wanted):
        item = next(row for row in catalog["inventory"] if row["source_id"] == source_id)
        if item["label_path"]:
            points, _ = load_rellis(item["source_path"], item["label_path"])
            points = points[np.linalg.norm(points[:, :2], axis=1) <= 25]
        else:
            points = read_points(item["source_path"])
        points = points[np.linspace(0, len(points) - 1, min(len(points), 18000), dtype=int)]
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        cloud = ax.scatter(*points.T, c=points[:, 2], s=.4, cmap="terrain", rasterized=True)
        ax.set_xlabel("x（米）", labelpad=8); ax.set_ylabel("y（米）", labelpad=8)
        ax.set_zlabel("z", labelpad=0)
        for coordinate, dimension in ((ax.xaxis, 0), (ax.yaxis, 1), (ax.zaxis, 2)):
            coordinate.set_ticks(np.linspace(points[:, dimension].min(), points[:, dimension].max(),
                                             2 if dimension == 2 else 3))
            coordinate.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.tick_params(labelsize=10, pad=1)
        ax.set_box_aspect(np.maximum(np.ptp(points, axis=0), .1))
        ax.view_init(elev=28, azim=-60)
        ax.set_title(NAMES.get(source_id, "RELLIS：第1帧，25米内"), fontsize=13)
        fig.colorbar(cloud, ax=ax, shrink=.6, pad=.12, label="z高度（米）")
    fig.suptitle("实际下载的官方点云：可见地形与障碍结构\n"
                 "颜色仅表示高度；RELLIS保留原始传感器坐标，未对齐重力", fontsize=14)
    save(fig, output, "public_sources_3d")


def prediction_summary(rows, items, output):
    groups = list(NAMES)
    stats = {}
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.6), layout="constrained")
    for offset, (kind, color) in enumerate(zip(CONDITIONS, ("#247a91", "#b55238", "#63676b"))):
        buckets = [[r for r in rows if r["status"] == "ok" and r["kind"] == kind
                    and items[r["dataset_id"]]["cluster_id"] == group] for group in groups]
        for group, bucket in zip(groups, buckets):
            stats.setdefault(group, {})[kind] = {"n": len(bucket), "rmse_cm": 100 * mean(bucket, "unknown_rmse_m"),
                "mae_cm": 100 * mean(bucket, "unknown_mae_m"), "picp95_percent": 100 * mean(bucket, "picp95"),
                "candidate_percent": 100 * mean(bucket, "unknown_accepted_coverage")}
        x = np.arange(len(groups)) + (offset - 1) * .25
        axes[0].bar(x, [stats[g][kind]["rmse_cm"] for g in groups], .23, color=color, label=CONDITIONS[kind])
        axes[1].bar(x, [stats[g][kind]["picp95_percent"] for g in groups], .23, color=color)
    labels = [NAMES[g].replace("样例", "\n样例").replace("短片段", "\n短片段") for g in groups]
    for ax in axes:
        ax.set_xticks(np.arange(len(groups)), labels, fontsize=12)
        ax.set_xlabel("点云来源分组（同源裁块、相邻帧和重复掩码不是独立场景）", fontsize=12)
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("隐藏参考格子的平均RMSE（厘米）", fontsize=12)
    axes[1].set_ylabel("95%高程区间实际覆盖比例（%）", fontsize=12)
    axes[1].set_ylim(0, 110); axes[1].axhline(95, ls=":", color="#b55238")
    axes[1].text(.985, 96, "名义95%参照", transform=axes[1].get_yaxis_transform(),
                 ha="right", color="#b55238", fontsize=10)
    axes[0].legend(ncol=3, fontsize=11)
    fig.suptitle("公开点云的地面高程预测：参考是原始点云地面代理，不是独立测量真值\n"
                 "两批缺测种子分别运行；图中合并已完成的条件，不包含无参考案例", fontsize=13)
    save(fig, output, "public_prediction_summary")
    table = []
    for group in groups:
        s = stats[group]
        table.append(f"{NAMES[group]} & {s['random']['n']} & {s['random']['rmse_cm']:.2f} & "
            f"{s['block']['rmse_cm']:.2f} & {s['frontier']['rmse_cm']:.2f} & "
            f"{s['random']['picp95_percent']:.1f}" + r" \\")
    (output / "public_rows.tex").write_text("\n".join(table) + "\n" + r"\bottomrule" + "\n")
    return stats


def layer_views(source, trial_id, output, name):
    with np.load(source / f"{trial_id}.npz", allow_pickle=False) as data:
        d = {k: data[k] for k in data.files}
    fields = [d["reference_height"], d["elevation"], d["predicted"],
              np.where(d["reference_valid"] & ~d["valid"],
                       100 * (d["predicted"] - d["reference_height"]), np.nan),
              d["risk_yaw0"], d["occupied"].astype(float)]
    titles = ["(a) 原始点云的地面高度代理", "(b) 删点后的可见地面", "(c) GP预测高程",
              "(d) 仅在隐藏且有参考处计误差", "(e) BGK/SEB：5度航向风险", "(f) 当前点云近地柱状占据"]
    lo, hi = np.nanmin(d["reference_height"]), np.nanmax(d["reference_height"])
    err = max(1., float(np.nanmax(np.abs(fields[3]))))
    fig, axes = plt.subplots(2, 3, figsize=(10.6, 7.2), layout="constrained")
    for i, (ax, field, title) in enumerate(zip(axes.flat, fields, titles)):
        cmap = "viridis" if i < 3 else "RdBu_r" if i == 3 else "RdYlGn_r" if i == 4 else "Greys"
        vmin, vmax = (lo, hi) if i < 3 else (-err, err) if i == 3 else (0, 1)
        artist = ax.imshow(field, origin="lower", extent=d["extent"], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("原始坐标x（米）"); ax.set_ylabel("原始坐标y（米）")
        label = "高度（米）" if i < 3 else "预测减参考（厘米）" if i == 3 else "风险分数" if i == 4 else "0无占据；1有占据"
        fig.colorbar(artist, ax=ax, shrink=.72, label=label)
    fig.suptitle("同一个公开点云裁块：地面预测与当前BGK/SEB风险输出\n"
                 "白色＝无有效高度参考/观测；高程预测不自动放行未知区域", fontsize=12)
    save(fig, output, name)


def obstacle_summary(rows, output):
    stats = {}
    for kind in CONDITIONS:
        selected = [r for r in rows if r.get("status") == "ok" and r.get("kind") == kind and r.get("family") == "rellis"]
        result = {}
        for scope in ("semantic_common_observed", "semantic_all_labelled"):
            counts = {key: sum(r[scope][key] for r in selected) for key in ("tp", "fp", "fn", "tn")}
            ratio = lambda a, b: a / b if b else None
            counts.update(recall=ratio(counts["tp"], counts["tp"] + counts["fn"]),
                          false_positive_rate=ratio(counts["fp"], counts["fp"] + counts["tn"]))
            result[scope] = counts
        stats[kind] = result
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.5), layout="constrained")
    for ax, key, title in ((axes[0], "recall", "标注近地结构障碍：检出比例"),
                            (axes[1], "false_positive_rate", "标注地面：误报占据比例")):
        for j, scope in enumerate(("semantic_common_observed", "semantic_all_labelled")):
            values = [stats[k][scope][key] for k in CONDITIONS]
            ax.bar(np.arange(3) + (j - .5) * .32, [100 * v if v is not None else np.nan for v in values], .3,
                   color=("#247a91", "#b55238")[j], label=("共同已观测标签区域", "全部有标签区域（含删去处）")[j])
        ax.set_xticks(range(3), ["随机15%", "内部孔洞", "单侧缺测"])
        ax.set_ylabel("百分比（%）"); ax.set_title(title); ax.set_ylim(0, 105); ax.grid(axis="y", alpha=.2)
    axes[0].legend(fontsize=9)
    fig.suptitle("RELLIS近地结构障碍代理：只统计有标签、且满足相对地面高度条件的格子\n"
                 "不是语义分割mIoU、车辆通行真值或动态残影实验；分母为零时不画柱", fontsize=12)
    save(fig, output, "public_semantic_obstacles")
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--batches", type=Path, nargs="+", required=True)
    parser.add_argument("--risk-results", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    configure_chinese()
    plt.rcParams.update({"font.size": 11})
    catalog = json.loads(args.dataset.read_text())
    items = {d["id"]: d for d in catalog["datasets"]}
    batches = [json.loads(path.read_text()) for path in args.batches]
    verify_disjoint_mask_seeds(batches)
    risk = [json.loads(path.read_text()) for path in args.risk_results]
    if any(d["settings"]["dataset_sha256"] != digest(args.dataset) for d in batches):
        raise ValueError("input batch/dataset fingerprint mismatch")
    rows = [r for d in batches for r in d["runs"]]
    risk_rows = [r for d in risk for r in d["runs"]]
    unique_inputs = defaultdict(set)
    for source, batch in zip(args.batches, batches):
        for row in batch["runs"]:
            if row["status"] != "ok":
                continue
            with np.load(source.parent / "cases" / row["id"] / "prediction.npz", allow_pickle=False) as data:
                sha = hashlib.sha256(np.ascontiguousarray(data["observed_xyz"]).tobytes()).hexdigest()
            unique_inputs[row["kind"]].add((row["dataset_id"], sha))
    source_views(catalog, args.output)
    group_stats = prediction_summary(rows, items, args.output)
    layers = args.risk_results[0].parent
    layer_views(layers, "fsgp_forest_p00__random_0__r000", args.output, "public_layers_forest")
    layer_views(layers, "rellis_000000_p00__random_0__r000", args.output, "public_layers_rellis")
    semantic = obstacle_summary(risk_rows, args.output)
    runtime = {"GP拟合": [r["fit_ms"] for r in rows if r["status"] == "ok"],
               "GP查询": [r["predict_ms"] for r in rows if r["status"] == "ok"],
               "占据+BGK/SEB": [r["bgk_seb_columns_ms"] for r in risk_rows if r["status"] == "ok"]}
    fig, ax = plt.subplots(figsize=(8.5, 4.2), layout="constrained")
    ax.boxplot(list(runtime.values()), tick_labels=list(runtime), showfliers=False)
    ax.set_ylabel("单案例墙钟耗时（毫秒）"); ax.grid(axis="y", alpha=.2)
    ax.set_title("6.4米裁块的局部算法计时：含冷启动，不含IO、ROS和渲染")
    save(fig, args.output, "public_runtime")
    atomic_json(args.output / "public_statistics.json", {"source_hashes": {str(p): digest(p)
        for p in [args.dataset, *args.batches, *args.risk_results]},
        "inventory": catalog["inventory"], "selection": Counter(r["status"] for r in catalog["selection_audit"]),
        "patch_count": len(items), "source_group_count": len({x["cluster_id"] for x in items.values()}),
        "unique_evaluable_GP_inputs": {k: len(v) for k, v in unique_inputs.items()},
        "GP_status": Counter(r["status"] for r in rows), "BGK_status": Counter(r["status"] for r in risk_rows),
        "device_GP": Counter(r["device"] for r in rows if r["status"] == "ok"),
        "device_BGK": Counter(r["actual_backend"] for r in risk_rows if r["status"] == "ok"),
        "prediction_by_source": group_stats, "semantic_proxies": semantic,
        "runtime": {name: {"n": len(values), "p50_ms": float(np.median(values)),
                     "p95_ms": float(np.percentile(values, 95)), "p99_ms": float(np.percentile(values, 99))}
                    for name, values in runtime.items()},
        "limits": "dense-cloud proxy / semantic-assisted ground; RELLIS 8 adjacent frames are one short source group"})
    good = [r for r in rows if r["status"] == "ok"]
    no_reference = sum(r["status"] == "not_evaluable" or
                       r.get("error", "").endswith("no held-out reference cells") for r in rows)
    macro = {"PublicPlanned": str(len(rows)), "PublicEvaluable": str(len(good)),
             "PublicNoReference": str(no_reference), "PublicUniqueInputs": str(sum(len(v) for v in unique_inputs.values()))}
    for prefix, family, kind, key, scale in (
        ("PublicFsgpRandomRmse", "fsgp", "random", "unknown_rmse_m", 100),
        ("PublicFsgpFrontierRmse", "fsgp", "frontier", "unknown_rmse_m", 100),
        ("PublicFsgpFrontierPicp", "fsgp", "frontier", "picp95", 100),
        ("PublicRellisRandomRmse", "rellis", "random", "unknown_rmse_m", 100),
        ("PublicRellisRandomPicp", "rellis", "random", "picp95", 100)):
        chosen = [r for r in good if r["kind"] == kind and r["dataset_id"].startswith(family)]
        macro[prefix] = f"{scale * mean(chosen, key):.2f}"
    observed = semantic["random"]["semantic_common_observed"]
    macro["PublicStructureRecall"] = f"{100 * observed['recall']:.2f}"
    macro["PublicStructureMiss"] = f"{100 * (1 - observed['recall']):.2f}"
    macro["PublicGroundFalsePositive"] = f"{100 * observed['false_positive_rate']:.2f}"
    for prefix, name in (("PublicFit", "GP拟合"), ("PublicQuery", "GP查询"), ("PublicRisk", "占据+BGK/SEB")):
        macro[prefix + "Median"] = f"{np.median(runtime[name]):.2f}"
        macro[prefix + "Pninetyfive"] = f"{np.percentile(runtime[name], 95):.2f}"
    write_numbers(args.output / "public_numbers.tex", macro)
    print(f"six public-data figure sets and statistics: {args.output}")


if __name__ == "__main__":
    main()
