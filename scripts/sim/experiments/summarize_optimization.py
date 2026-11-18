#!/usr/bin/env python3
"""Chinese plots of frozen terrain experiments; no fitting or score selection."""

import argparse
from collections import Counter, defaultdict
import json
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pointcloud_batch import ROOT, atomic_json, digest, point_cells
from plot_style import configure_chinese

KINDS = {"random": "随机缺测", "block": "内部连续孔洞", "frontier": "单侧外推"}
FAMILIES = {"slope": "斜坡", "hills": "起伏山地", "ridge": "窄脊", "step": "台阶", "ditch": "沟壑"}
COLUMN_NAMES = {"dense_pole": "密集柱体", "sparse_pole": "同格稀疏柱体", "crosscell_pole": "跨格稀疏柱体",
    "wall": "稀疏墙面", "canopy": "高处树冠", "ground": "平滑地面", "outlier": "孤立离群点", "step_ground": "台阶地面"}
METRICS = ("unknown_rmse_m", "accepted_unknown_rmse_m", "picp95", "interval_width95_m",
           "unknown_accepted_coverage", "unknown_nll")
COLORS = ("#526a80", "#278a84", "#c95a42")


def load(path, batch=False):
    data = json.loads(Path(path).read_text())
    if batch and len(data["runs"]) != data["planned_cases"]:
        raise ValueError(f"batch not finished: {path}")
    return data


def cluster_stats(rows, key, groups=None):
    buckets = defaultdict(list)
    for row in rows:
        value = row.get(key)
        if value is not None and np.isfinite(value):
            buckets[groups[row["dataset_id"]] if groups else row["dataset_id"]].append(value)
    values = np.asarray([np.mean(v) for v in buckets.values()])
    if not len(values):
        return {"mean": None, "ci95": None, "n": 0}
    rng = np.random.default_rng(20261013)
    boot = values[rng.integers(len(values), size=(2000, len(values)))].mean(axis=1)
    return {"mean": float(values.mean()), "ci95": np.percentile(boot, [2.5, 97.5]).tolist(), "n": len(values)}


def save(fig, output, name):
    fig.savefig(output / (name + ".pdf"), bbox_inches="tight")
    fig.savefig(output / (name + ".png"), bbox_inches="tight", dpi=180)
    plt.close(fig)


def observed_cell_mean(points, query_xy, shape, resolution):
    ids, inside = point_cells(points, {"query_xy": query_xy, "shape": shape, "resolution": resolution})
    counts = np.bincount(ids[inside], minlength=np.prod(shape))
    sums = np.bincount(ids[inside], weights=points[inside, 2], minlength=np.prod(shape))
    return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan).reshape(shape)


def bars(ax, stats, label, offset=0, width=.7, multiplier=1, color=COLORS[1]):
    x = np.arange(len(stats)) + offset
    mean = np.array([s["mean"] if s["mean"] is not None else np.nan for s in stats]) * multiplier
    lower = np.array([s["ci95"][0] if s["ci95"] else np.nan for s in stats]) * multiplier
    upper = np.array([s["ci95"][1] if s["ci95"] else np.nan for s in stats]) * multiplier
    ax.bar(x, mean, width, yerr=np.maximum([mean-lower, upper-mean], 0), capsize=3, color=color, label=label)
    ax.grid(axis="y", alpha=.2)


def hidden_summary(batch, output):
    rows = [r for r in batch["runs"] if r["status"] == "ok"]
    conditions = [("random", .3), ("random", .6), ("block", .8), ("block", 1.6), ("frontier", .25), ("frontier", .5)]
    labels = ["随机缺测\n30%格子", "随机缺测\n60%格子", "内部孔洞\n0.8米宽", "内部孔洞\n1.6米宽",
              "单侧隐藏\n25%宽度", "单侧隐藏\n50%宽度"]
    stats = {f"{kind}:{severity}": {key: cluster_stats([r for r in rows if r["kind"] == kind and
        r["severity"] == severity], key) for key in METRICS} for kind, severity in conditions}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7), layout="constrained")
    for key, color, name, offset in (("unknown_rmse_m", COLORS[0], "全部隐藏参考格子", -.19),
        ("accepted_unknown_rmse_m", COLORS[1], "门限保留的候选格子", .19)):
        bars(axes[0], [stats[f"{k}:{s}"][key] for k, s in conditions], name, offset, .34, 100, color)
    for ax, key, title in ((axes[1], "picp95", "名义95%区间实际覆盖率"),
                            (axes[2], "unknown_accepted_coverage", "门限保留的隐藏区比例")):
        bars(ax, [stats[f"{k}:{s}"][key] for k, s in conditions], "当前冻结版本", multiplier=100)
        ax.set_title(title); ax.set_ylabel("比例（%）"); ax.set_ylim(0, 108)
    axes[1].axhline(95, color=COLORS[2], ls=":")
    axes[0].set_title("隐藏高程的预测误差"); axes[0].set_ylabel("每地形平均RMSE（厘米）"); axes[0].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(np.arange(6), labels, fontsize=9); ax.set_xlabel("缺测条件：删除整格输入后再拟合")
    fig.suptitle(f"独立测试集：{batch['world_count']}个地形，{len(rows)}次可评估留出\n"
                 "误差条为地形聚类bootstrap区间，不是单格预测区间")
    save(fig, output, "01_hidden_test")
    matrix = np.array([[cluster_stats([r for r in rows if r["family"] == family and r["kind"] == k
                        and r["severity"] == s], "unknown_rmse_m")["mean"] * 100
                        for k, s in conditions] for family in FAMILIES])
    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0)
    for (i, j), v in np.ndenumerate(matrix):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="black")
    ax.set_xticks(np.arange(6), labels); ax.set_yticks(np.arange(5), list(FAMILIES.values()))
    ax.set_xlabel("缺测方式及强度"); ax.set_ylabel("独立解析地形类型（每类50个）")
    ax.set_title("不同地形的隐藏区高程RMSE：不以平地掩盖突变边界误差")
    fig.colorbar(im, ax=ax, label="平均RMSE（厘米）")
    save(fig, output, "02_geometry_error")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    for ax, key, title, unit in ((axes[0], "interval_width95_m", "高程区间平均宽度", "厘米"),
        (axes[1], "unknown_nll", "隐藏参考高程的高斯负对数评分", "NLL（以米制密度计算，可为负）")):
        bars(ax, [stats[f"{k}:{s}"][key] for k, s in conditions], "当前版本",
             multiplier=100 if key.endswith("_m") else 1)
        ax.set_title(title); ax.set_ylabel(unit); ax.set_xticks(np.arange(6), labels, fontsize=9)
    fig.suptitle("覆盖率不能单独评价：同时检查区间宽度和负对数评分")
    save(fig, output, "03_uncertainty_width")
    return stats


def paired_development(before, after, output):
    old = {r["id"]: r for r in before["runs"] if r["status"] == "ok"}
    rows, differences = [], []
    for r in after["runs"]:
        if r["status"] != "ok" or r["id"] not in old:
            continue
        b = old[r["id"]]
        if any(r[k] != b[k] for k in ("dataset_id", "mask_seed", "noise_seed", "input_point_count")):
            raise ValueError("unpaired development input")
        rows.append(r)
        differences.append({**r, **{k: r[k] - b[k] if r.get(k) is not None and b.get(k) is not None else None
                                   for k in METRICS}})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout="constrained")
    result = {}
    for version, data, offset, color in (("修改前", list(old.values()), -.18, COLORS[0]),
                                          ("修改后", rows, .18, COLORS[1])):
        for ax, key, title, multiplier in ((axes[0], "unknown_rmse_m", "隐藏区高程RMSE（厘米）", 100),
            (axes[1], "picp95", "名义95%区间实际覆盖率（%）", 100),
            (axes[2], "interval_width95_m", "95%区间宽度（厘米）", 100)):
            bars(ax, [cluster_stats([r for r in data if r["kind"] == k], key) for k in KINDS], version,
                 offset, .32, multiplier, color)
            ax.set_title(title); ax.set_xticks(np.arange(3), list(KINDS.values()))
    axes[0].legend(); axes[1].axhline(95, ls=":", color=COLORS[2])
    fig.suptitle("25个开发地形的同输入版本回归：不是外部方法对比，也不是独立测试集")
    save(fig, output, "04_paired_development")
    for kind in KINDS:
        result[kind] = {key: cluster_stats([r for r in differences if r["kind"] == kind], key) for key in METRICS}
    return result


def public_results(batches, dataset, old_risk, new_risk, output):
    items = {r["id"]: r for r in dataset["datasets"]}
    groups = {key: r["cluster_id"] for key, r in items.items()}
    rows = []
    seen = set()
    for batch in batches:
        for r in batch["runs"]:
            key = (r["dataset_id"], r["kind"], r["severity"], r["mask_seed"])
            if key in seen:
                raise ValueError("duplicate public mask job")
            seen.add(key)
            rows.append(r)
    summary = {}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.6), layout="constrained")
    for family, axs in zip(("fsgp", "rellis"), axes):
        selected = [r for r in rows if r["status"] == "ok" and
                    (r["family"].startswith("fsgp_") if family == "fsgp" else r["family"] == "rellis")]
        summary[family] = {}
        for key in METRICS:
            summary[family][key] = {kind: cluster_stats([r for r in selected if r["kind"] == kind], key, groups)
                                   for kind in KINDS}
        for ax, key, title, multiplier in ((axs[0], "unknown_rmse_m", "隐藏区高程RMSE（厘米）", 100),
            (axs[1], "picp95", "名义95%区间实际覆盖率（%）", 100),
            (axs[2], "unknown_accepted_coverage", "候选隐藏区覆盖率（%）", 100)):
            bars(ax, [summary[family][key][k] for k in KINDS], "当前版本", multiplier=multiplier)
            ax.set_title(("作者7张地图" if family == "fsgp" else "RELLIS单短片段") + "\n" + title)
            ax.set_xticks(np.arange(3), list(KINDS.values()), fontsize=9)
            if key != "unknown_rmse_m":
                ax.set_ylim(0, 110)
        axs[1].axhline(95, ls=":", color=COLORS[2])
    fig.suptitle(f"公开点云扩大测试：{len(rows)}次运行；参考为密集地面代理\n"
                 "源地图先平均再等权统计；RELLIS相邻帧只算一个来源，不提供跨场景推断")
    save(fig, output, "05_public_ground")
    counts = {}
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.7), layout="constrained")
    for version, risks, color, offset in (("before", old_risk, COLORS[0], -.18),
                                           ("after", new_risk, COLORS[1], .18)):
        selected = [r for data in risks for r in data["runs"] if r["status"] == "ok" and r["family"] == "rellis"
                    and r["kind"] == "random"]
        values = []
        for scope in ("semantic_common_observed", "semantic_all_labelled"):
            c = {key: sum(r[scope][key] for r in selected) for key in ("tp", "fp", "fn", "tn")}
            counts.setdefault(version, {})[scope] = c
            values.append([c["tp"] / max(1, c["tp"] + c["fn"]), c["fp"] / max(1, c["fp"] + c["tn"])])
        for column, values_for_scope in enumerate(values):
            for row_index, value in enumerate(values_for_scope):
                ax = axes[row_index, column]
                position = 0 if version == "before" else 1
                ax.bar(position, value*100, .6, color=color)
                ax.text(position, value*100+.5 if row_index == 0 else value*100+.015,
                        f"{value*100:.2f}", ha="center", fontsize=10)
    for column, scope_title in enumerate(("共同观测格子", "全部标注格子（含隐藏区）")):
        for row_index, metric_title in enumerate(("结构障碍召回率", "标注地面误报率")):
            ax = axes[row_index, column]
            ax.set_title(scope_title + "\n" + metric_title)
            ax.set_xticks([0, 1], ["修改前", "修改后"]); ax.set_ylabel("比例（%）")
            ax.set_ylim(0, 105 if row_index == 0 else 1.5); ax.grid(axis="y", alpha=.2)
    fig.suptitle("RELLIS同输入碰撞分支回归：仅配对两批随机缺测15%的输入\n"
                 "语义+近地几何代理标签，不是实车通过概率；低误报不能替代高召回")
    save(fig, output, "06_public_columns")
    return {"status": dict(Counter(r["status"] for r in rows)), "source_groups": len(set(groups.values())),
            "by_family": summary, "paired_semantic_counts": counts}


def public_paired(before, after, groups, output):
    identity = lambda r: (r["dataset_id"], r["kind"], r["severity"], r["mask_seed"])
    old = {identity(r): r for b in before for r in b["runs"] if r["status"] == "ok"}
    paired = [r for b in after for r in b["runs"] if r["status"] == "ok" and identity(r) in old]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), layout="constrained")
    result = {}
    for version, rows, color, offset in (("before", [old[identity(r)] for r in paired], COLORS[0], -.18),
        ("after", paired, COLORS[1], .18)):
        result[version] = {kind: {key: cluster_stats([r for r in rows if r["kind"] == kind], key, groups)
            for key in METRICS} for kind in KINDS}
        for ax, key, title in ((axes[0], "unknown_rmse_m", "隐藏参考高程RMSE（厘米）"),
            (axes[1], "picp95", "名义95%区间实际覆盖率（%）"),
            (axes[2], "interval_width95_m", "95%区间宽度（厘米）")):
            bars(ax, [result[version][kind][key] for kind in KINDS], "修改前" if version == "before" else "修改后",
                 offset, .32, 100, color)
            ax.set_title(title); ax.set_xticks(np.arange(3), list(KINDS.values())); ax.set_xlabel("同一缺测输入条件")
    axes[0].legend(); axes[1].axhline(95, ls=":", color=COLORS[2])
    fig.suptitle(f"公开点云配对回归：{len(paired)}对输入、8个来源分组等权\n"
                 "包括7张作者地图及1个RELLIS短片段；不是外部方法对比")
    save(fig, output, "11_public_paired")
    return result


def columns_summary(data, output):
    rows = [r for r in data["runs"] if r["backend"] == "cupy"]
    stats = {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    positive = list(COLUMN_NAMES)[:4]
    negative = list(COLUMN_NAMES)[4:]
    for version, color, offset in (("before", COLORS[0], -.18), ("after", COLORS[1], .18)):
        stats[version] = {}
        for family in COLUMN_NAMES:
            selected = [r for r in rows if r["version"] == version and r["family"] == family]
            c = {key: sum(r[key] for r in selected) for key in ("tp", "fn", "fp", "tn")}
            stats[version][family] = {**c, "recall": c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else None,
                "false_positive": c["fp"] / max(1, c["fp"] + c["tn"])}
        axes[0].bar(np.arange(4) + offset, [stats[version][f]["recall"] * 100 for f in positive],
                    .32, color=color, label="修改前" if version == "before" else "修改后")
        axes[1].bar(np.arange(4) + offset, [stats[version][f]["false_positive"] * 100 for f in negative],
                    .32, color=color)
    for ax, families, title in ((axes[0], positive, "近地障碍召回率（越高越好）"),
        (axes[1], negative, "非障碍返回的格子误报率（越低越好）")):
        ax.set_xticks(np.arange(4), [COLUMN_NAMES[f].replace("柱体", "\n柱体") for f in families], fontsize=9)
        ax.set_ylabel("比例（%）"); ax.set_title(title); ax.grid(axis="y", alpha=.2)
    axes[0].set_ylim(0, 108); axes[0].legend()
    axes[1].set_ylim(0, max(1, axes[1].get_ylim()[1]))
    fig.suptitle(f"受控碰撞分支专项：{data['independent_cases']}组输入，两版本CPU/GPU配对\n"
                 "解析点关联标签；观测地面缺失10%至65%，不输入隐藏地面高程")
    save(fig, output, "07_column_regression")
    fig, axes = plt.subplots(3, 3, figsize=(10, 9), layout="constrained")
    for axs, family in zip(axes, ("sparse_pole", "crosscell_pole", "wall")):
        with np.load(Path(data["arguments"]["output"]) / (family + ".npz")) as source:
            for ax, key, title in zip(axs, ("target", "before", "after"), ("独立点关联标签", "修改前检测", "修改后检测")):
                ax.imshow(source[key], origin="lower", extent=[0, 9.6, 0, 9.6], cmap="Reds", vmin=0, vmax=1)
                ax.set_title(COLUMN_NAMES[family] + "\n" + title); ax.set_xlabel("局部x（米）"); ax.set_ylabel("局部y（米）")
                occupied_y, occupied_x = np.where(source["target"])
                ax.set_xlim(max(0, occupied_x.min()*.2-.8), min(9.6, (occupied_x.max()+1)*.2+.8))
                ax.set_ylim(max(0, occupied_y.min()*.2-.8), min(9.6, (occupied_y.max()+1)*.2+.8))
    fig.suptitle("固定首个种子的障碍占据图：红色为障碍，白色为非障碍/未检测")
    save(fig, output, "08_column_maps")
    return {"by_family": stats, "clear_scan_false_cells": sum(r["clear_scan_false_cells"] for r in data["runs"]),
            "independent_cases": data["independent_cases"], "detector_calls": len(data["runs"])}


def spatial_examples(batch_path, output):
    cases = (("hills_000__block_3__r000", "起伏山地/1.6米孔洞"),
             ("step_000__block_3__r000", "台阶/1.6米孔洞"),
             ("ridge_000__frontier_5__r000", "窄脊/隐藏50%宽度"))
    fig, axes = plt.subplots(3, 4, figsize=(13, 9), layout="constrained")
    for axs, (case, label) in zip(axes, cases):
        with np.load(batch_path.parent / "cases" / case / "prediction.npz") as source:
            shape = tuple(source["shape"])
            observed = source["observed_cells"].reshape(shape)
            xy = source["query_xy"]
            extent = [xy[:, 0].min()-.1, xy[:, 0].max()+.1, xy[:, 1].min()-.1, xy[:, 1].max()+.1]
            truth = source["reference_height"].reshape(shape)
            input_height = observed_cell_mean(source["observed_xyz"], xy, shape, .2)
            fields = (input_height, truth,
                np.where(~observed, (source["mean"].reshape(shape)-truth)*100, np.nan),
                np.where(~observed, np.sqrt(source["latent_var"].reshape(shape))*100, np.nan))
            for ax, field, title, cmap, unit in zip(axs, fields,
                ("输入点的格内高程均值", "完整独立解析参考", "隐藏区预测减参考", "隐藏区预测标准差"),
                ("terrain", "terrain", "RdBu_r", "magma"), ("高程（米）", "高程（米）", "误差（厘米）", "标准差（厘米）")):
                kwargs = {"vmin": -.25, "vmax": .4} if unit == "高程（米）" else {}
                if unit.startswith("误差"):
                    limit = max(5, np.nanmax(np.abs(field))); kwargs = {"vmin": -limit, "vmax": limit}
                im = ax.imshow(field, origin="lower", extent=extent, cmap=cmap, **kwargs)
                ax.set_title(label + "\n" + title, fontsize=9); ax.set_xlabel("局部x（米）"); ax.set_ylabel("局部y（米）")
                fig.colorbar(im, ax=ax, shrink=.8, label=unit)
    fig.suptitle("预先固定首个地形、首个重复的空间案例：白色表示未输入或不在该面板评分范围")
    save(fig, output, "09_spatial_prediction")


def online_checks(output):
    folders = ("modular_smoke_20261013_100702", "modular_smoke_20261013_100951", "modular_smoke_20261013_101212")
    names = ("静态绕障", "缓坡通行", "动态横穿")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3), layout="constrained")
    records, rows = [], []
    for ax, folder, name in zip(axes, folders, names):
        path = ROOT / "log" / folder
        result = load(path / "result.json")
        data = load(path / "navigation_data.json")
        metadata = load(path / "run_metadata.json")
        if metadata["arguments"]["prediction_backend"] != "feature_sgp":
            raise ValueError("online check did not run the optimized GP")
        truth = np.asarray(data["truth"])
        start = np.asarray(data["truth_start"][:2])
        goal = np.asarray(data["truth_goal"])
        ax.plot(truth[:, 1]-start[0], truth[:, 2]-start[1], color=COLORS[1], label="Isaac真值轨迹")
        ax.scatter([0], [0], s=30, color=COLORS[0], label="起点")
        ax.scatter(goal[0]-start[0], goal[1]-start[1], marker="*", s=90, color=COLORS[2], label="目标")
        ax.set_title(name + ("：通过" if result["passed"] else "：失败")); ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("世界x相对起点（米）"); ax.set_ylabel("世界y相对起点（米）"); ax.grid(alpha=.2); ax.legend(fontsize=8)
        s = data["summary"]
        record = {"name": name, "log_directory": str(path), "result": result, "summary": s,
            "source_sha256": {key: digest(path / key) for key in ("result.json", "navigation_data.json", "run_metadata.json")}}
        records.append(record)
        rows.append(f"{name} & {'通过' if result['passed'] else '失败'} & {s['duration_s']:.2f} & "
            f"{s['truth_path_length_m']:.2f} & {s['tracking_error_p95_m']:.3f} & {s['risk_map_age_p95_s']:.3f} \\\\")
    fig.suptitle("冻结算法的三次轻量无窗口闭环接口复测：不是大样本导航成功率实验")
    save(fig, output, "12_online_paths")
    (output / "online_table.tex").write_text("\\begin{tabular}{lrrrrr}\\toprule\n"
        "场景 & 链路检查 & 用时/s & 真值路程/m & 跟踪P95/m & 地图年龄P95/s\\\\\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    configure_chinese()
    paths = {
        "test": ROOT / "log/optimization_test_gp_20261013/batch_manifest.json",
        "development_before": ROOT / "log/optimization_dev_before_20261013/batch_manifest.json",
        "development_after": ROOT / "log/optimization_dev_after_half_20261013/batch_manifest.json",
        "columns": ROOT / "log/optimization_columns_20261013/results.json",
        "public_dataset": ROOT / "log/public_cloud_dataset_20261012/dataset_manifest.json"}
    public_paths = [ROOT / f"log/optimization_public_{name}_20261013/batch_manifest.json"
                    for name in ("paired_a", "paired_b", "expanded")]
    old_risk_paths = [ROOT / f"log/public_cloud_bgk_{name}_20261012/results.json" for name in ("audit", "disjoint")]
    new_risk_paths = [ROOT / f"log/optimization_risk_{name}_20261013/results.json" for name in ("paired_a", "paired_b")]
    data = {key: load(path, "manifest" in path.name and key != "public_dataset") for key, path in paths.items()}
    test = data["test"]
    if test["settings"]["predictor_sha256"] != digest(ROOT / "src/fsgp_bgk_se2/fsgp_bgk_se2/feature_sgp.py"):
        raise ValueError("tested predictor differs from current implementation")
    public = [load(path, True) for path in public_paths]
    campaign_path = ROOT / "log/algorithm_optimization_20261013/campaign.json"
    campaign = load(campaign_path)
    for stage in campaign["stages"]:
        command = stage["command"]
        artifact = ROOT / command[command.index("--output")+1] / (
            "batch_manifest.json" if stage["stage"].startswith("gp_") else "results.json")
        stage["result_sha256"] = digest(artifact)
    campaign["original_campaign_sha256"] = digest(campaign_path)
    atomic_json(campaign_path.with_name("campaign_audit.json"), campaign)
    old_public = []
    for new, old_name in zip(public[:2], ("public_cloud_gp_20261012", "public_cloud_gp_disjoint_20261012")):
        old = load(ROOT / "log" / old_name / "batch_manifest.json", True)
        old_public.append(old)
        if new["jobs"] != old["jobs"] or new["settings"]["inputs"] != old["settings"]["inputs"]:
            raise ValueError("public regression inputs are not paired with the old batches")
    unique_public = defaultdict(set)
    for path, batch in zip(public_paths, public):
        for row in batch["runs"]:
            if row["status"] != "ok":
                continue
            with np.load(path.parent / "cases" / row["id"] / "prediction.npz") as prediction:
                unique_public[row["kind"]].add((row["dataset_id"],
                    hashlib.sha256(prediction["observed_xyz"].tobytes()).hexdigest()))
    stats = {"date": "2026-10-13", "scope": __doc__, "plotter_sha256": digest(__file__), "sources": {str(p): digest(p)
        for p in [*paths.values(), *public_paths, *old_risk_paths, *new_risk_paths]},
        "test_count": len(test["runs"]), "test_world_count": test["world_count"],
        "test_status": dict(Counter(r["status"] for r in test["runs"])),
        "test_parameters": test["settings"]["parameters"],
        "test_by_condition": hidden_summary(test, args.output),
        "development_paired_delta": paired_development(data["development_before"], data["development_after"], args.output),
        "public": public_results(public, data["public_dataset"], [load(p) for p in old_risk_paths],
            [load(p) for p in new_risk_paths], args.output), "columns": columns_summary(data["columns"], args.output)}
    stats["unique_public_inputs_by_condition"] = {key: len(values) for key, values in unique_public.items()}
    stats["public_paired"] = public_paired(old_public, public[:2],
        {item["id"]: item["cluster_id"] for item in data["public_dataset"]["datasets"]}, args.output)
    stats["online_checks"] = online_checks(args.output)
    spatial_examples(paths["test"], args.output)
    times = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for label, batch, color in (("独立解析测试", test, COLORS[0]), ("公开点云测试", {"runs": [r for b in public for r in b["runs"]]}, COLORS[1])):
        for ax, key, title in ((axes[0], "fit_ms", "输入筛选、特征提取与GP拟合"), (axes[1], "predict_ms", "1024格GP推断")):
            values = np.sort([r[key] for r in batch["runs"] if r["status"] == "ok"])
            ax.plot(values, np.arange(1, len(values)+1)/len(values), label=label, color=color)
            times.setdefault(label, {})[key] = {"p50": float(np.median(values)), "p95": float(np.percentile(values, 95)),
                "max": float(values.max()), "n": len(values)}
            ax.set_title(title); ax.set_xlabel("含CUDA同步的计算耗时（毫秒）"); ax.set_ylabel("累积分布比例")
            ax.set_xlim(0, np.percentile(values, 99.5) * 1.2); ax.grid(alpha=.2); ax.legend(fontsize=9)
    fig.suptitle("串行离线计算的耗时分布：不含文件写入、ROS、RTX渲染及规划")
    save(fig, args.output, "10_runtime")
    stats["latency"] = times
    atomic_json(args.output / "statistics.json", stats)
    counts = stats["public"]["paired_semantic_counts"]
    old = counts["before"]["semantic_common_observed"]
    new = counts["after"]["semantic_common_observed"]
    percent = lambda v: f"{100*v:.2f}" + r"\%"
    observations = [
        f"独立解析测试完成{stats['test_count']}次，全部成功，独立地形{stats['test_world_count']}个。"
        f"公开点云安排{sum(b['planned_cases'] for b in public)}次，其中{stats['public']['status'].get('ok',0)}次可评估，"
        f"{stats['public']['status'].get('not_evaluable',0)}次没有隐藏参考，未赋零误差。"
        f"公开点云不同实际输入数组共{sum(stats['unique_public_inputs_by_condition'].values())}组；"
        "单侧边界重复不是新的场景。",
        "RELLIS共同观测分母下，近地结构代理标签召回率由"
        + percent(old['tp']/(old['tp']+old['fn'])) + "提高到" + percent(new['tp']/(new['tp']+new['fn']))
        + "，标注地面误报率由" + percent(old['fp']/(old['fp']+old['tn'])) + "变为"
        + percent(new['fp']/(new['fp']+new['tn'])) + "。召回改善伴随小幅误报增加，二者均须报告。",
        "独立解析测试中，0.8米孔洞和1.6米孔洞的实际区间覆盖率分别为"
        + percent(stats['test_by_condition']['block:0.8']['picp95']['mean']) + "和"
        + percent(stats['test_by_condition']['block:1.6']['picp95']['mean'])
        + "，仍明显低于名义95\\%。这说明隐藏突变或细结构仍有模型误设，不能宣称区间已经全面校准。",
        f"受控碰撞分支完成{stats['columns']['independent_cases']}组输入、{stats['columns']['detector_calls']}次配对检测，"
        f"及同数量的下一帧清除调用。清除帧误占据格子总数{stats['columns']['clear_scan_false_cells']}。"
        "这仅证明该当前帧分支没有历史滞留，不能代替累积点云地图残影测试。",
        f"独立解析测试拟合P50/P95为{times['独立解析测试']['fit_ms']['p50']:.2f}/"
        f"{times['独立解析测试']['fit_ms']['p95']:.2f}毫秒，1024格推断P50/P95为"
        f"{times['独立解析测试']['predict_ms']['p50']:.2f}/{times['独立解析测试']['predict_ms']['p95']:.2f}毫秒。"
        f"拟合最大值{times['独立解析测试']['fit_ms']['max']:.2f}毫秒，包含初始化影响，未删除慢样本。"]
    (args.output / "observations.tex").write_text("\n\n".join(observations) + "\n")
    table = []
    for condition, values in stats["test_by_condition"].items():
        kind, severity = condition.split(":")
        label = KINDS[kind] + (f"{float(severity)*100:.0f}\\%" if kind != "block" else f"{severity}米")
        e = values["unknown_rmse_m"]; ci = e["ci95"]
        table.append(f"{label} & {e['mean']*100:.2f} & [{ci[0]*100:.2f}, {ci[1]*100:.2f}] & "
            f"{values['picp95']['mean']*100:.1f} & {values['interval_width95_m']['mean']*100:.1f} & "
            f"{values['unknown_accepted_coverage']['mean']*100:.1f} \\\\")
    (args.output / "test_rows.tex").write_text(
        "\\begin{tabular}{lrrrrr}\\toprule\n"
        "缺测条件 & RMSE & 均值95\\%区间 & PICP95/\\% & 区间宽度 & 候选覆盖/\\%\\\\\\midrule\n"
        + "\n".join(table) + "\n\\bottomrule\\end{tabular}\n")
    notes = ["# 本地算法优化与扩大测试：2026-10-13", "", f"独立解析测试：{test['world_count']}地形，{len(test['runs'])}次。",
        "统计均先在地形内平均，再在地形之间统计；开发配对不是外部比较；公开地面参考是代理。", "",
        "|条件|隐藏高程RMSE/cm|PICP95/%|95%区间宽度/cm|候选覆盖/%|", "|---|---:|---:|---:|---:|"]
    for key, v in stats["test_by_condition"].items():
        notes.append(f"|{key}|{100*v['unknown_rmse_m']['mean']:.2f}|{100*v['picp95']['mean']:.1f}|"
                     f"{100*v['interval_width95_m']['mean']:.1f}|{100*v['unknown_accepted_coverage']['mean']:.1f}|")
    (args.output.parent / "RESULTS.md").write_text("\n".join(notes) + "\n")
    print(json.dumps({k: stats[k] for k in ("test_count", "test_world_count", "test_status", "latency")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
