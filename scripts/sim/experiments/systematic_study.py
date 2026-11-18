#!/usr/bin/env python3
"""Rebuild a question-led report from recorded evidence, without rerunning algorithms."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from matplotlib import colors
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.ticker import FormatStrFormatter

from plot_paper_scale import (FAMILIES, clustered_summary, font_manager, plt,
                             ratio_rows, save)
from plot_expanded_campaign import CONDITIONS, SCENARIOS, validate_navigation, validate_terrain
from plot_style import configure_chinese, wilson_interval
from pointcloud_batch import atomic_json, digest
from thesis_statistics import box_summary, source_mean_interval, source_values
from plot_thesis_evidence import plain


ROOT = Path(__file__).resolve().parents[3]
SHOW = ("point15", "hole16", "frontier50")
COLORS = ("#24788e", "#c27845", "#42856d", "#74618b", "#9a8734")
BLUE, RED, GRAY = COLORS[0], "#bc4b46", "#626b72"


def read(path):
    return json.loads(path.read_text())


def figure_names(protocol):
    names = ["00_framework"]+[n for e in protocol["experiments"] for n in e["figures"]]+protocol.get("appendix_figures", [])
    if len(set(names)) != len(names):
        raise ValueError("main and appendix figure identities must be unique")
    return names


def require_hash(path, expected):
    if digest(path) != expected:
        raise ValueError(f"recorded evidence changed: {path}")


def complete_jobs(record):
    planned = [r["index"] for r in record["jobs"]]
    actual = [r["index"] for r in record["runs"]]
    if len(set(planned)) != len(planned) or len(actual) != len(planned) or set(actual) != set(planned):
        raise ValueError("incomplete or duplicate physical plan")


def verify_analytic(record, experiment):
    f = experiment["factors"]
    expected = {(b, s, d, r) for b in f["backends"] for s in f["slope_rad"]
                for d in f["direction_rad"] for r in range(f["noise_repeats"])}
    actual = [(r["backend"], r["slope_rad"], r["direction_rad"], r["repeat"])
              for r in record["runs"]]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("analytic plan does not match protocol")
    pairs = defaultdict(dict)
    for row in record["runs"]:
        pairs[(row["geometry_id"], row["sensor_seed"])][row["backend"]] = row
    if any(set(p) != set(f["backends"]) for p in pairs.values()):
        raise ValueError("missing paired backend")
    return pairs


def navigation_summary(plan, record):
    validate_navigation(plan, record)
    cases = {c["id"]: c for c in plan["cases"]}
    stats = {}
    for name in SCENARIOS:
        rows = [r for r in record["runs"] if r["scenario"] == name]
        if not rows:
            raise ValueError("missing navigation family")
        flags = {cases[r["id"]]["expected_stop"] for r in rows}
        if len(flags) != 1 or any(r["expected_stop"] not in flags for r in rows):
            raise ValueError("arrival and rejection targets cannot share a denominator")
        passed = sum(r["status"] == "passed" for r in rows)
        stats[name] = {"n": len(rows), "passed": passed, "expected_stop": flags.pop(),
            "interval": wilson_interval(passed, len(rows)),
            "collision": sum(bool(r.get("collision")) for r in rows),
            "terrain_violation": sum(bool(r.get("unsafe_terrain")) for r in rows),
            "unknown_violation": sum(bool(r.get("unknown_policy_violation")) for r in rows),
            "tracking_p95_m": box_summary([r["tracking_p95_m"] for r in rows])}
    return stats


def completion_summary(record, audit):
    details = audit["completion_details"]
    summary = {}
    for family in FAMILIES:
        summary[family] = {}
        for condition in CONDITIONS:
            rows = [r for r in record["runs"] if r["family"] == family and r["condition"] == condition]
            original = [dict(r, **{k: r.get("observed_only_" + k) for k in ("tp", "fn", "fp", "tn")})
                        for r in rows if r["status"] == "ok"]
            now = {r["cluster_id"]: r for r in ratio_rows(rows)}
            old = {r["cluster_id"]: r for r in ratio_rows(original)}
            if now.keys() != old.keys():
                raise ValueError("unpaired completion sources")
            delta = [{"cluster_id": g, "status": "ok", **{
                "delta_" + k: now[g][k] - old[g][k]
                if now[g][k] is not None and old[g][k] is not None else None
                for k in ("false_safe", "false_hazard")}} for g in sorted(now)]
            scored = [r for r in details if r["family"] == family and r["condition"] == condition]
            summary[family][condition] = {
                **{k: clustered_summary(delta, k) for k in ("delta_false_safe", "delta_false_hazard")},
                **{k: clustered_summary(scored, k) for k in ("filled_gt_coverage", "filled_rmse_m")}}
    ok = [r for r in record["runs"] if r["status"] == "ok"]
    totals = {k: sum(r[k] for r in ok) for k in ("tp", "fn", "fp", "tn",
              "observed_only_tp", "observed_only_fn", "observed_only_fp", "observed_only_tn")}
    return summary, totals


def load_evidence(protocol):
    paths = {k: ROOT / v for k, v in protocol["evidence"].items()}
    data = {k: read(v) for k, v in paths.items()}
    validate_terrain(data["terrain"], data["dataset"])
    require_hash(paths["dataset"], data["terrain"]["settings"]["dataset_sha256"])
    for audit_key, result_key in (("terrain_audit", "terrain"), ("completion_audit", "completion")):
        if data[audit_key]["status"] != "passed":
            raise ValueError("recorded array audit did not pass")
        require_hash(paths[result_key], data[audit_key]["results_sha256"])
    require_hash(paths["terrain"], data["array_statistics"]["results_sha256"])
    if data["array_statistics"]["arrays_verified"] != data["terrain_audit"]["successful_arrays_verified"]:
        raise ValueError("array diagnostics have a different evaluation denominator")
    if len(data["completion"]["runs"]) != data["completion"]["planned_cases"]:
        raise ValueError("incomplete completion campaign")
    for key in ("physical", "dynamic_capture"):
        complete_jobs(data[key])
    if len(data["dynamic_capture"]["runs"]) != 1:
        raise ValueError("single diagnostic cannot be silently promoted to a batch")
    row = data["dynamic_capture"]["runs"][0]
    path = Path(row["run_dir"]) / "navigation_data.json"
    data["dynamic_raw"] = read(path)
    if data["dynamic_raw"].get("schema_version", 0) < 3:
        raise ValueError("dynamic evidence must separate baseline and enhanced risk")
    paths["dynamic_raw"] = path
    data["nav_stats"] = navigation_summary(data["navigation_plan"], data["navigation"])
    data["completion_stats"], data["completion_totals"] = completion_summary(data["completion"], data["completion_audit"])
    experiment = next(e for e in protocol["experiments"] if e["id"] == "S1")
    data["analytic_pairs"] = verify_analytic(data["analytic"], experiment)
    from supplemental_evidence import validate_new_campaign
    validate_new_campaign(data, paths)
    from speed_response_evidence import validate_speed_campaign
    validate_speed_campaign(data["speed_campaign"], data["speed_protocol"], paths["speed_protocol"])
    return data, paths


def style():
    configure_chinese()
    font = Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": ["DejaVu Sans", font_manager.FontProperties(fname=str(font)).get_name()],
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "figure.titlesize": 11, "axes.prop_cycle": plt.cycler(color=COLORS)})


def finish(fig, output, name):
    save(fig, output, name)


def terrain_summary(rows):
    result = {}
    for family in FAMILIES:
        result[family] = {}
        for condition in CONDITIONS:
            selected = [r for r in rows if r["family"] == family and r["condition"] == condition]
            stats = {k: clustered_summary(selected, k) for k in
                     ("hidden_rmse_m", "hidden_mae_m", "risk_mae", "hidden_candidate_coverage")}
            stats.update({k: clustered_summary(ratio_rows(selected), k) for k in ("false_safe", "false_hazard")})
            result[family][condition] = stats
    return result


def paired_source_series(rows, family, conditions, metric):
    groups = [source_values([r for r in rows if r["family"] == family and r["condition"] == c], metric)
              for c in conditions]
    common = sorted(set.intersection(*(set(g) for g in groups)))
    if not common:
        raise ValueError("difficulty response requires common evaluable source groups")
    values = np.asarray([[g[s] for g in groups] for s in common])
    low, high = source_mean_interval(values)
    return {"mean": values.mean(axis=0), "low": low, "high": high, "source_groups": common}


DISTANCE_UPPER = np.array([.2, .4, .8, 1.6, 3.2, 6.4, np.inf])


def grid_distance_errors(mean, reference, observed, resolution=.2):
    from scipy.ndimage import distance_transform_edt
    reference = np.asarray(reference)
    observed = np.asarray(observed, bool)
    mean = np.asarray(mean).reshape(reference.shape)
    if observed.shape != reference.shape or resolution <= 0 or not np.isfinite(mean).all():
        raise ValueError("invalid distance diagnostic fields")
    if not observed.any():
        return None
    hidden = np.isfinite(reference) & ~observed
    distances = distance_transform_edt(~observed, sampling=resolution)[hidden]
    bins = np.searchsorted(DISTANCE_UPPER, distances, side="left")
    errors = mean[hidden].astype(float) - reference[hidden].astype(float)
    count = np.bincount(bins, minlength=7)
    squared = np.bincount(bins, weights=errors ** 2, minlength=7)
    return count, squared


def distance_analysis(data, paths, output):
    conditions = ("hole16", "frontier50")
    pooled = {}; excluded = Counter(); chain = hashlib.sha256(); checked = 0
    for row in data["terrain"]["runs"]:
        if row["status"] != "ok" or row["condition"] not in conditions:
            continue
        path = paths["terrain"].parent / (row["id"] + ".npz")
        chain.update(row["id"].encode()); chain.update(bytes.fromhex(digest(path)))
        with np.load(path, allow_pickle=False) as a:
            ref, obs = a["reference_height"], a["observed_cells"]
            mean = a["height_mean"].reshape(ref.shape)
            hidden = np.isfinite(ref) & ~obs
            error = mean[hidden] - ref[hidden]
            if hidden.sum() != row["hidden_cells"]:
                raise ValueError("distance input has a different hidden-reference denominator")
            if len(error) and (row["hidden_rmse_m"] is None or not np.isclose(
                    np.sqrt(np.mean(error ** 2)), row["hidden_rmse_m"], atol=1e-6)):
                raise ValueError("distance input differs from recorded hidden-ground scores")
            diagnostics = grid_distance_errors(mean, ref, obs)
        checked += 1
        if diagnostics is None:
            excluded[row["family"] + "/" + row["condition"]] += 1
            continue
        key = (row["family"], row["condition"], row["cluster_id"])
        if key not in pooled:
            pooled[key] = (np.zeros(7, dtype=np.int64), np.zeros(7))
        pooled[key][0][:] += diagnostics[0]; pooled[key][1][:] += diagnostics[1]
        if checked % 1000 == 0:
            print(f"[systematic] stratified {checked} saved height arrays", flush=True)
    stats = {}
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 4.0), layout="constrained")
    labels = ["≤0.2", "(0.2,0.4]", "(0.4,0.8]", "(0.8,1.6]", "(1.6,3.2]", "(3.2,6.4]", ">6.4"]
    for ax, condition in zip(axes, conditions):
        for family, color in zip(FAMILIES, COLORS):
            selected = [v for k, v in pooled.items() if k[:2] == (family, condition)]
            entries = []
            for i in range(7):
                values = np.array([np.sqrt(s[i] / n[i]) for n, s in selected if n[i]]) * 100
                if len(values):
                    low, high = source_mean_interval(values[:, None])[:, 0]
                    entry = {"mean_cm": values.mean(), "low_cm": low, "high_cm": high,
                             "source_groups": len(values), "reference_cells": sum(n[i] for n, _ in selected)}
                else:
                    entry = {"mean_cm": None, "low_cm": None, "high_cm": None, "source_groups": 0, "reference_cells": 0}
                entries.append(entry)
            stats[family + "/" + condition] = entries
            ax.plot(range(7), [v["mean_cm"] for v in entries], "o-", color=color, label=FAMILIES[family], ms=3)
        ax.set_xticks(range(7), labels, rotation=35, ha="right", fontsize=7)
        ax.set(xlabel="到窗口内观测格的距离区间（米）", ylabel="分层隐藏高程 RMSE（厘米）", title=CONDITIONS[condition])
        ax.set_yscale("symlog", linthresh=1); ax.set_ylim(bottom=0); ax.grid(alpha=.18)
    axes[0].legend(fontsize=7)
    finish(fig, output, "M1_distance")
    result = plain({"conditions": conditions, "intervals": labels, "statistics": stats,
        "arrays_checked": checked, "array_digest_chain": chain.hexdigest(), "no_window_observation_cases": dict(excluded),
        "definition": "Euclidean grid-center distance to measured ground inside the query window only; not exact raw-point support or visibility; source pooled then source equal; each distance bin may have different sources"})
    atomic_json(output / "distance_diagnostics.json", result)
    return result


def framework(output):
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    ax.set(xlim=(0, 10), ylim=(0, 6)); ax.axis("off")
    boxes = [(0.1, 4.2, "不完整观测\n点云 / 地面观测"), (3.55, 4.2, "GP 高程推断\n均值 / 方差 / 支撑"),
             (7.0, 4.2, "BGK / SEB 风险\n位置＋航向＋碰撞"), (7.0, 1.6, "规划与跟踪\n搜索 / 轨迹 / 安全停车")]
    for x, y, label in boxes:
        ax.add_patch(Rectangle((x, y), 2.8, 1.2, facecolor="#edf3f5", edgecolor=BLUE, linewidth=1.1))
        ax.text(x + 1.4, y + .6, label, ha="center", va="center", linespacing=1.6)
    for start, end in (((2.9, 4.8), (3.55, 4.8)), ((6.35, 4.8), (7, 4.8)), ((8.4, 4.2), (8.4, 2.8))):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=12, color=GRAY))
    ax.text(1.5, 3.3, "参考仅用于独立评分", ha="center", color=RED)
    ax.add_patch(Rectangle((.1, .25), 6.25, 2.5, facecolor="white", edgecolor=GRAY, linestyle="--"))
    ax.text(.35, 2.42, "实验分工", weight="bold")
    ax.text(.35, 1.95, "M1：隐藏高程和分布质量    M2：危险辨识和保守代价")
    ax.text(.35, 1.42, "M3：模型闭环 / 物理闭环    M4：计算与消息时序")
    ax.text(.35, .86, "S1：航向机理   S2：补全代价   S3：持续清除")
    ax.text(.35, .43, "S4：负障碍与多层边界（待测，不是结果）", color=RED)
    ax.text(8.4, .65, "组件证据不能替代\n物理系统验证", ha="center", color=GRAY)
    finish(fig, output, "00_framework")


def observed_surface(data, paths, family, condition):
    from spatial_evidence import reconstruct_spatial_case
    row = next(r for r in data["terrain"]["runs"] if r["family"] == family and r["condition"] == condition and r["status"] == "ok")
    d = reconstruct_spatial_case(data["terrain"], data["dataset"], paths["terrain"], paths["dataset"], row)
    return row, d["origin"], d["observed"], d["mean"], d["legacy_reference"]


def terrain_figures(data, paths, stats, output):
    selection = {}
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.5), layout="constrained")
    for ax, condition in zip(axes, SHOW):
        row, origin, obs, mean, ref = observed_surface(data, paths, "forest", condition)
        selection["mask/" + condition] = row["id"]
        extent = [origin[0], origin[0] + 20.8, origin[1], origin[1] + 20.8]
        ax.imshow(np.isfinite(obs), origin="lower", extent=extent,
                  cmap=colors.ListedColormap(["#f0f1f1", BLUE]), vmin=0, vmax=1)
        ax.set_title(CONDITIONS[condition]); ax.set_xlabel("局部 x（米）"); ax.set_ylabel("局部 y（米）")
    fig.suptitle("M1 观测任务：蓝色为实际地面观测格，浅灰为未观测")
    finish(fig, output, "M1_masks")
    groups = (("point15", "point30", "point60"), ("hole08", "hole16", "hole32"), ("frontier25", "frontier50"))
    xs = ((15, 30, 60), (.8, 1.6, 3.2), (25, 50))
    xlabels = ("按点删除比例（%）", "中心孔洞边长（米）", "单侧隐藏窗口宽度（%）")
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.7), layout="constrained")
    paired = {}
    for ax, cs, x, label in zip(axes, groups, xs, xlabels):
        for family in FAMILIES:
            s = paired_source_series(data["terrain"]["runs"], family, cs, "hidden_rmse_m")
            paired[family + "/" + cs[0]] = s
            line, = ax.plot(x, s["mean"] * 100, "o-", label=FAMILIES[family], ms=3)
            ax.fill_between(x, s["low"] * 100, s["high"] * 100, alpha=.12, color=line.get_color())
        ax.set_xlabel(label); ax.set_ylabel("隐藏高程 RMSE（厘米）"); ax.grid(alpha=.18)
        ax.set_yscale("symlog", linthresh=1); ax.set_ylim(bottom=0); ax.set_xticks(x)
        ax.text(.03, .03, f"室内共同组：{len(s['source_groups'])}", transform=ax.transAxes, fontsize=7, color=GRAY)
    axes[0].legend(fontsize=7)
    atomic_json(output / "paired_difficulty.json", plain(paired))
    finish(fig, output, "M1_difficulty")
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 4.0), layout="constrained")
    boxes = {}
    for ax, condition in zip(axes, SHOW):
        values = [list(source_values([r for r in data["terrain"]["runs"]
                  if r["family"] == f and r["condition"] == condition], "hidden_rmse_m").values()) for f in FAMILIES]
        groups_cm = [np.asarray(v) * 100 for v in values]
        b = ax.boxplot(groups_cm, patch_artist=True, widths=.6, showfliers=True,
            medianprops={"color": "black"}, flierprops={"marker": ".", "markersize": 2, "alpha": .4})
        for patch, color in zip(b["boxes"], COLORS):
            patch.set_facecolor(color); patch.set_alpha(.45)
        ax.set_xticks(range(1, 6), [f"{FAMILIES[f]}\nn={len(v)}" for f, v in zip(FAMILIES, values)], rotation=28, ha="right")
        ax.set_title(CONDITIONS[condition]); ax.set_ylabel("来源组 RMSE（厘米）")
        ax.set_yscale("symlog", linthresh=1); ax.set_ylim(bottom=0); ax.grid(axis="y", alpha=.18)
        boxes[condition] = {f: box_summary(v) for f, v in zip(FAMILIES, groups_cm)}
    finish(fig, output, "M1_distributions")
    fig, axes = plt.subplots(2, 3, figsize=(7.5, 6.1), layout="constrained")
    nominal = np.asarray(data["array_statistics"]["nominal"]) * 100
    for col, condition in enumerate(SHOW):
        for family, color in zip(FAMILIES, COLORS):
            c = data["array_statistics"]["calibration"][family][condition]
            axes[0, col].plot(nominal, np.asarray(c["coverage_mean"]) * 100, "o-", color=color, label=FAMILIES[family], ms=2.5)
            axes[1, col].scatter(c["width_mean_m"][3] * 100, c["coverage_mean"][3] * 100, color=color, s=24)
        axes[0, col].plot([50, 99], [50, 99], "--", color=GRAY)
        axes[0, col].set(xlim=(48, 101), ylim=(0, 103), xlabel="预设名义覆盖 q（%）", ylabel="实际高程覆盖 PICP（%）", title=CONDITIONS[condition])
        axes[1, col].axhline(95, ls="--", color=GRAY)
        axes[1, col].set(xscale="log", xlabel="高程区间总宽度（厘米，q=95%）", ylabel="实际覆盖 PICP95（%）", ylim=(0, 103))
        for ax in axes[:, col]: ax.grid(alpha=.18)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("高程区间校准：名义等级是设定，实际覆盖是评分\n覆盖更高不一定更好，必须同时看区间宽度；旧v1来源参考", fontsize=10)
    finish(fig, output, "M1_uncertainty")
    for family in ("hilly", "ruin"):
        row, origin, observed, mean, ref = observed_surface(data, paths, family, "hole16")
        selection["surface/" + family] = row["id"]
        x, y = np.meshgrid(origin[0] + (np.arange(ref.shape[1]) + .5) * .2,
                           origin[1] + (np.arange(ref.shape[0]) + .5) * .2)
        low, high = min(np.nanmin(ref), np.nanmin(mean)), max(np.nanmax(ref), np.nanmax(mean))
        error = np.abs(mean - ref); norm = colors.Normalize(0, max(.01, np.nanmax(error)))
        fig = plt.figure(figsize=(7.4, 6.3), layout="constrained")
        for i, (field, title) in enumerate(((observed, "输入地面"), (mean, "预测高程"), (ref, "完整参考"), (ref, "绝对预测误差"))):
            ax = fig.add_subplot(2, 2, i + 1, projection="3d")
            kw = {"cmap": "terrain", "vmin": low, "vmax": high} if i < 3 else {"facecolors": plt.cm.magma(norm(error)), "shade": False}
            mesh = ax.plot_surface(x, y, field, rcount=52, ccount=52, linewidth=0, rasterized=True, **kw)
            ax.set(xlim=(x.min(), x.max()), ylim=(y.min(), y.max()), zlim=(low, high))
            ax.set_box_aspect((20.8, 20.8, 4 * max(.2, high - low))); ax.view_init(30, -60)
            ax.set_xticks([x.min() + 5.2, x.max() - 5.2]); ax.set_yticks([y.min() + 5.2, y.max() - 5.2])
            ax.set_zticks([low, high]); ax.zaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            ax.set_xlabel("x（米）", labelpad=0); ax.set_ylabel("y（米）", labelpad=0)
            ax.text2D(.96, .88, "z（米）", transform=ax.transAxes, ha="right", fontsize=8)
            ax.set_title(title); ax.tick_params(labelsize=7, pad=0)
            bar = mesh if i < 3 else plt.cm.ScalarMappable(norm=norm, cmap="magma")
            fig.colorbar(bar, ax=ax, orientation="horizontal", shrink=.7, pad=.1, aspect=22,
                         label="高程（米）" if i < 3 else "绝对误差（米）")
        fig.suptitle(f"M1 {FAMILIES[family]}：预定首个孔洞样例；竖向显示 ×4")
        finish(fig, output, "M1_surface_" + family)
    return selection, boxes


def risk_figures(data, paths, stats, output):
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 7.4), layout="constrained")
    for ax, key, factor, title, cmap, maximum in zip(axes,
        ("risk_mae", "false_safe", "false_hazard"), (1, 100, 100),
        ("风险 MAE（无量纲，越低越好）", "危险漏判（%，越低越好）", "安全参考误拒（%，越低越好）"),
        ("YlGnBu", "OrRd", "YlOrBr"), (1, 100, 100)):
        values = np.array([[stats[f][c][key]["mean"] for c in CONDITIONS] for f in FAMILIES], dtype=float) * factor
        im = ax.imshow(values, aspect="auto", vmin=0, vmax=maximum, cmap=cmap)
        ax.set_xticks(range(8), list(CONDITIONS.values()), fontsize=7)
        ax.set_yticks(range(5), list(FAMILIES.values())); ax.set_title(title)
        for (i, j), v in np.ndenumerate(values):
            if np.isfinite(v): ax.text(j, i, f"{v:.1f}" if factor == 100 else f"{v:.3f}", ha="center", va="center", fontsize=8,
                                     color="white" if im.norm(v) > .58 else "black")
        fig.colorbar(im, ax=ax, shrink=.75, pad=.02)
    finish(fig, output, "M2_risk_safety")
    row = next(r for r in data["terrain"]["runs"] if r["family"] == "hilly" and r["condition"] == "hole16" and r["status"] == "ok")
    with np.load(paths["terrain"].parent / (row["id"] + ".npz"), allow_pickle=False) as a:
        xy, predicted, reference, valid = (a[k].copy() for k in ("query_xy", "predicted_risk", "reference_risk", "supported"))
    fig = plt.figure(figsize=(7.5, 4.3), layout="constrained")
    for i, (field, title) in enumerate(((reference, "独立足迹参考"), (predicted, "本工程预测"))):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        for index in (0, 9, 18, 27):
            mask = valid[index]
            ax.scatter(xy[mask, 0], xy[mask, 1], np.full(mask.sum(), index * 10 + 5),
                       c=field[index, mask], cmap="RdYlGn_r", vmin=0, vmax=1, s=5, rasterized=True)
        ax.set_xlabel("x（米）", labelpad=0); ax.set_ylabel("y（米）", labelpad=0)
        ax.text2D(.96, .90, "航向 θ（度）", transform=ax.transAxes, ha="right", fontsize=8)
        ax.set_zticks([5, 95, 185, 275]); ax.set_title(title); ax.tick_params(labelsize=7, pad=0)
        ax.set_box_aspect((1, 1, 1)); ax.view_init(25, -60)
        fig.colorbar(plt.cm.ScalarMappable(norm=colors.Normalize(0, 1), cmap="RdYlGn_r"), ax=ax,
                     orientation="horizontal", shrink=.75, pad=.12, label="风险分数（无量纲）")
    finish(fig, output, "M2_pose_space")


def navigation_trace(ax, row, case):
    trace = np.asarray(row["trajectory"])
    color = BLUE if row["status"] == "passed" else RED
    ax.plot(trace[:, 1], trace[:, 2], color=color, label="执行轨迹")
    ax.plot(*case["goal"], "*", color=RED, ms=7, label="目标")
    ax.plot(*case["start"][:2], "o", color="black", ms=3)
    for b in case["obstacles"]:
        ax.add_patch(Rectangle((b["x"] - b["hx"], b["y"] - b["hy"]), 2 * b["hx"], 2 * b["hy"], color=GRAY, alpha=.35))
    ax.set_xlabel("地图 x（米）"); ax.set_ylabel("地图 y（米）")
    ax.set_aspect("equal", adjustable="datalim"); ax.grid(alpha=.18)


def navigation_speed(ax, row):
    trace = np.asarray(row["trajectory"])
    ax.plot(trace[:, 0], trace[:, 4], color=BLUE, label="模型实际速度")
    ax.plot(trace[:, 0], trace[:, 6], "--", color=COLORS[1], label="发布速度指令")
    ax.plot(row["sim_elapsed_s"], row["final_speed_mps"], "o", ms=3, color="black", label="单独保存的终点速度")
    ax.set_xlabel("仿真时间（秒）"); ax.set_ylabel("纵向速度（米/秒）"); ax.grid(alpha=.18)


def navigation_figures(data, output):
    rows, stats = data["navigation"]["runs"], data["nav_stats"]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 5.0), layout="constrained", gridspec_kw={"width_ratios": [1.4, 1]})
    for ax, stop in zip(axes, (False, True)):
        names = [n for n in SCENARIOS if stats[n]["expected_stop"] == stop]
        for y, name in enumerate(names):
            s = stats[name]; rate = 100 * s["passed"] / s["n"]
            ax.barh(y, rate, color=BLUE if not stop else COLORS[1], alpha=.85)
            lo, hi = np.asarray(s["interval"]) * 100
            ax.errorbar(rate, y, xerr=[[max(0, rate - lo)], [max(0, hi - rate)]], color="black", capsize=2, lw=.8)
            ax.text(3, y, f"{s['passed']}/{s['n']}", va="center", color="white", fontsize=8)
        ax.set_yticks(range(len(names)), [SCENARIOS[n] for n in names]); ax.invert_yaxis()
        ax.set(xlim=(0, 108), xlabel="任务合格率（%）", title="到达并停车任务" if not stop else "安全拒绝任务")
        ax.grid(axis="x", alpha=.18)
    finish(fig, output, "M3_qualification")
    names = ("turn", "reverse", "detour", "corridor", "emerge", "crossing")
    fig, ax = plt.subplots(figsize=(7.4, 3.7), layout="constrained")
    groups = [[r["tracking_p95_m"] for r in rows if r["scenario"] == n] for n in names]
    ax.boxplot(groups, widths=.55, medianprops={"color": BLUE}, flierprops={"marker": ".", "markersize": 2})
    for i, n in enumerate(names, 1):
        failed = [r["tracking_p95_m"] for r in rows if r["scenario"] == n and r["status"] != "passed"]
        ax.scatter(np.full(len(failed), i), failed, color=RED, s=8, alpha=.5, zorder=3)
    ax.set_xticks(range(1, 7), [SCENARIOS[n] for n in names]); ax.set_ylabel("案例内跟踪误差 P95（米）")
    ax.grid(axis="y", alpha=.18); ax.set_title("全部案例均参与统计；红点为不合格任务")
    finish(fig, output, "M3_tracking")
    cases = {c["id"]: c for c in data["navigation_plan"]["cases"]}
    fig, axes = plt.subplots(4, 2, figsize=(7.5, 8.5), layout="constrained")
    chosen = {}
    for i, name in enumerate(("reverse", "detour", "ramp", "emerge")):
        row = next(r for r in rows if r["scenario"] == name); case = cases[row["id"]]
        chosen[name] = row["id"]
        ax = axes[i, 0]; navigation_trace(ax, row, case)
        ax.set_title(SCENARIOS[name] + ("：合格" if row["status"] == "passed" else "：不合格"))
        navigation_speed(axes[i, 1], row)
    axes[0, 0].legend(fontsize=7); axes[0, 1].legend(fontsize=7)
    finish(fig, output, "M3_examples")
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.2), layout="constrained")
    for i, name in enumerate(("detour", "emerge")):
        row = next(r for r in rows if r["scenario"] == name and r["status"] != "passed")
        chosen["first_failure/" + name] = row["id"]
        navigation_trace(axes[i, 0], row, cases[row["id"]])
        navigation_speed(axes[i, 1], row)
        axes[i, 0].set_title(SCENARIOS[name] + "：首个不合格案例")
        axes[i, 1].set_title(f"终点误差 {row['goal_error_m']:.2f} 米")
    axes[0, 0].legend(fontsize=7); axes[0, 1].legend(fontsize=7)
    finish(fig, output, "M3_failures")
    names = ("detour", "ramp", "step", "emerge", "crossing", "clearance")
    fig, ax = plt.subplots(figsize=(7.4, 3.7), layout="constrained")
    physical = {}
    for i, name in enumerate(names):
        subset = [r for r in data["physical"]["runs"] if r["scenario"] == name]
        physical[name] = {"n": len(subset), "qualified": sum(r["passed"] for r in subset),
                          "reached": sum(bool(r.get("reached")) for r in subset)}
        s = physical[name]
        ax.bar(i - .17, s["qualified"], .32, color=BLUE, label="完整流程合格" if i == 0 else None)
        ax.bar(i + .17, s["reached"], .32, color=COLORS[1], label="目标到达" if i == 0 else None)
        ax.text(i, 2.17, f"n={s['n']}", ha="center", fontsize=8)
    ax.set_xticks(range(6), [SCENARIOS.get(n, "障碍移开") for n in names]); ax.set_yticks([0, 1, 2])
    ax.set(ylim=(0, 2.55), ylabel="案例数（次）", title="Isaac K批次：到达不等于全流程合格；台阶任务要求拒绝")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=.18)
    finish(fig, output, "M3_physical")
    return chosen, physical


def runtime_figures(data, output):
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.6), layout="constrained")
    timing = {}
    for ax, key, title in zip(axes, ("fit_ms", "predict_ms", "risk_ms"), ("特征与GP拟合", "高程查询预测", "占据与BGK/SEB")):
        timing[key] = {}
        for family, color in zip(FAMILIES, COLORS):
            a = np.sort([r[key] for r in data["terrain"]["runs"] if r["family"] == family and r["status"] == "ok"])
            ax.plot(a, np.arange(1, len(a) + 1) / len(a), color=color, label=FAMILIES[family])
            timing[key][family] = {"n": len(a), "p50": np.median(a), "p95": np.quantile(a, .95), "p99": np.quantile(a, .99), "max": a.max()}
        ax.set(xscale="log", xlabel="组件墙钟（毫秒）", ylabel="累计案例比例", title=title); ax.grid(alpha=.18)
    axes[0].legend(fontsize=7); finish(fig, output, "M4_component_timing")
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.7), layout="constrained")
    analytic = data["analytic"]["runs"]
    a = [[r["compute_ms"] for r in analytic if r["backend"] == b] for b in ("cpu", "cupy")]
    axes[0].boxplot(a, tick_labels=["CPU", "GPU"], flierprops={"marker": ".", "markersize": 2})
    axes[0].set(yscale="log", ylabel="墙钟（毫秒）", title="48×48×36 SEB调用")
    for ax, key, title in zip(axes[1:], ("planner_p95_ms", "control_p95_ms"), ("规划调用", "控制回调")):
        for name, color in zip(("straight", "detour", "reverse", "emerge", "crossing"), COLORS):
            v = np.sort([r[key] for r in data["navigation"]["runs"] if r["scenario"] == name])
            ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=color, label=SCENARIOS[name])
        ax.set(xscale="log", xlabel="案例内 P95（毫秒）", ylabel="累计案例比例", title=title)
    axes[1].legend(fontsize=7)
    for ax in axes: ax.grid(alpha=.18)
    finish(fig, output, "M4_backend")
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.6), layout="constrained")
    names = ("detour", "ramp", "step", "emerge", "crossing", "clearance")
    for ax, key, title in zip(axes, ("risk_map_age_p95_s", "planner_p95_ms"), ("消息地图年龄 P95（秒）", "规划调用 P95（毫秒）")):
        for i, name in enumerate(names):
            subset = [r for r in data["physical"]["runs"] if r["scenario"] == name and isinstance(r.get(key), (int, float))]
            ax.scatter([i] * len(subset), [r[key] for r in subset], color=[BLUE if r["passed"] else RED for r in subset], s=24)
        ax.set_xticks(range(6), [SCENARIOS.get(n, "障碍移开") for n in names], rotation=30, ha="right")
        ax.set_ylabel(title); ax.grid(axis="y", alpha=.18)
    finish(fig, output, "M4_physical_timing")
    return timing, {b: box_summary(v) for b, v in zip(("cpu", "cupy"), a)}


def special_figures(data, output):
    record = data["analytic"]
    fig, axes = plt.subplots(3, 2, figsize=(7.5, 6.3), layout="constrained")
    for ax, c in zip(axes.flat, record["curves"]):
        yaw = np.degrees(c["yaw"])
        ax.plot(yaw, c["oracle"], color="black", label="解析参考")
        ax.plot(yaw, c["seb"], color=BLUE, label="SEB几何分量")
        ax.plot(yaw, c["fused"], "--", color=COLORS[1], label="融合输出")
        ax.fill_between(yaw, 0, 1, where=c["unsafe"], alpha=.1, color=RED)
        ax.axhline(.58, ls=":", color=RED)
        ax.set(xlabel="车辆航向（度）", ylabel="风险分数", ylim=(0, 1.05), title=f"坡面倾角 {np.degrees(c['slope_rad']):.1f}°")
        ax.grid(alpha=.18)
    axes[0, 0].legend(fontsize=7); finish(fig, output, "S1_direction")
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 5.8), layout="constrained")
    for ax, metric, title in zip(axes, ("delta_false_hazard", "delta_false_safe"),
        ("误拒变化：负值为减少误拒", "漏判变化：正值为增加危险漏判")):
        values = np.asarray([[data["completion_stats"][f][c][metric]["mean"] for c in CONDITIONS] for f in FAMILIES], float) * 100
        bound = max(.01, np.nanmax(np.abs(values)))
        im = ax.imshow(values, vmin=-bound, vmax=bound, cmap="RdBu_r", aspect="auto")
        ax.set_xticks(range(8), list(CONDITIONS.values()), fontsize=7); ax.set_yticks(range(5), list(FAMILIES.values())); ax.set_title(title)
        for (i, j), v in np.ndenumerate(values):
            if np.isfinite(v): ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=8,
                                      color="white" if abs(v) > .6 * bound else "black")
        fig.colorbar(im, ax=ax, shrink=.8, pad=.02, label="补全开启−仅观测（百分点）")
    finish(fig, output, "S2_completion")
    raw = data["dynamic_raw"]; series = raw["clearance_series"]
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 6.4), layout="constrained", sharex=True)
    for key, label, color in (("registered", "当前注册帧", BLUE), ("laser_obstacle", "FAST-LIO累计高度带", RED)):
        a = np.asarray(series[key], float); axes[0].plot(a[:, 0], a[:, 1], ".-", color=color, label=label)
    axes[0].set_ylabel("原区域高度带点数（个）"); axes[0].legend(fontsize=8)
    for key, label, color in (("baseline_risk", "前端基线风险", BLUE), ("enhanced_risk", "增强后SEB风险", RED)):
        a = np.asarray(series[key], float); valid = a[:, 1].astype(bool)
        values = np.where(valid, a[:, 2], np.nan)
        axes[1].plot(a[:, 0], values, ".-", color=color, label=label)
    axes[1].axhline(.58, color=GRAY, ls="--", label="风险分类门限")
    axes[1].set(ylim=(-.04, 1.07), ylabel="有效风险分数"); axes[1].legend(fontsize=8)
    a = np.asarray(series["live"], float); axes[2].step(a[:, 0], a[:, 1], where="post", color=BLUE)
    axes[2].set(yticks=[0, 1], ylim=(-.1, 1.1), ylabel="实时碰撞占据（0/1）", xlabel="障碍移开后的仿真时间（秒）")
    axes[1].axvspan(6.5, 7.5, color=RED, alpha=.08)
    axes[1].text(6.0, .40, "低值尾段仅1秒\n不足2秒持续判据", ha="right", fontsize=8, color=RED)
    for ax in axes: ax.grid(alpha=.18)
    finish(fig, output, "S3_dynamic")
    pairs = data["analytic_pairs"]
    differences = {k: max(abs(p["cpu"][k] - p["cupy"][k]) for p in pairs.values())
                   for k in ("seb_risk_mae", "fused_risk_mae", "false_safe_count", "false_hazard_count")}
    totals = {b: {k: sum(r[k] for r in record["runs"] if r["backend"] == b)
                  for k in ("false_safe_count", "unsafe_states", "false_hazard_count", "safe_states")} for b in ("cpu", "cupy")}
    return {"paired_inputs": len(pairs), "calculations": len(record["runs"]), "max_difference": differences, "counts": totals}


def tables(data, stats, output):
    lines = []
    for family in FAMILIES:
        for condition in SHOW:
            s = stats[family][condition]; c = data["array_statistics"]["calibration"][family][condition]
            fmt = lambda v: "--" if v is None else f"{v:.2f}"
            condition_label = CONDITIONS[condition].replace('%', r'\%')
            family_label = FAMILIES[family] + ("*" if family == "road" else "")
            lines.append(f"{family_label} & {condition_label} & "
                f"{s['hidden_rmse_m']['clusters']} & {s['hidden_rmse_m']['mean']*100:.2f} & "
                f"{c['coverage_mean'][3]*100:.1f} & {c['width_mean_m'][3]*100:.1f} & {c['crps_mean_m']*100:.2f} & "
                f"{fmt(s['false_safe']['mean']*100 if s['false_safe']['mean'] is not None else None)} & "
                f"{fmt(s['false_hazard']['mean']*100 if s['false_hazard']['mean'] is not None else None)} " + r"\\")
    (output / "terrain_rows.tex").write_text("\n".join(lines) + "\n" + r"\bottomrule" + "\n", encoding="utf-8")
    lines = []
    for name, s in data["nav_stats"].items():
        lines.append(f"{SCENARIOS[name]} & {s['n']} & {s['passed']} & {s['collision']} & {s['terrain_violation']} & "
            f"{s['unknown_violation']} & {s['tracking_p95_m']['median']:.3f} " + r"\\")
    (output / "navigation_rows.tex").write_text("\n".join(lines) + "\n" + r"\bottomrule" + "\n", encoding="utf-8")
    lines = []
    for layer, label in (("registered", "当前注册帧"), ("laser_obstacle", "累计点云高度带"),
                         ("baseline_risk", "前端基线风险"), ("enhanced_risk", "增强SEB风险"), ("live", "实时碰撞层")):
        d = data["dynamic_raw"]["clearance_temporal_diagnostics"][layer]
        t = d["first_sustained_clear_s"]
        value = "未证实" if t is None else f"{t:.2f}"
        lines.append(f"{label} & {d['samples']} & {value} & {d['observation_end_s']:.2f} & {d['reappearance_count']} " + r"\\")
    (output / "dynamic_rows.tex").write_text("\n".join(lines) + "\n" + r"\bottomrule" + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, default=ROOT / "scripts/sim/experiments/protocols/systematic_study_v2.json")
    p.add_argument("--output", type=Path, default=ROOT / "docs/reports/systematic_study/assets")
    p.add_argument("--verify-assets", action="store_true", help="verify frozen inputs and plotted assets without redrawing")
    args = p.parse_args(); protocol = read(args.protocol)
    if protocol.get("schema_version") != 2:
        raise ValueError("V1 is historical; use the archived report or the V2 report protocol")
    args.output.mkdir(parents=True, exist_ok=True)
    data, paths = load_evidence(protocol)
    names = figure_names(protocol)
    if args.verify_assets:
        previous = read(args.output / "results.json")
        require_hash(args.protocol, previous["protocol_sha256"])
        require_hash(Path(__file__), previous["plotter_sha256"])
        for key, path in paths.items():
            require_hash(path, previous["hashes"][key])
        for filename, expected in previous["helper_sha256"].items():
            require_hash(Path(__file__).with_name(filename), expected)
        manifest = read(args.output / "figure_manifest.json")
        figures = manifest["figures"]
        if [f["id"] for f in figures] != names:
            raise ValueError("figure plan changed; redraw required")
        for figure in figures:
            require_hash(args.output / (figure["id"] + ".pdf"), figure["sha256"])
        for filename, expected in manifest["artifacts"].items():
            require_hash(args.output / filename, expected)
        print("[systematic] evidence and figure fingerprints verified")
        return
    stats = terrain_summary(data["terrain"]["runs"])
    style(); framework(args.output)
    from road_reference_audit import audit_road_reference, plot_road_deletion
    road_audit, road_case = audit_road_reference(data["terrain"], data["dataset"], paths["terrain"], paths["dataset"], args.output)
    plot_road_deletion(road_case, args.output)
    from sparse_reference_audit import sparse_reference_audit, plot_scoring_domains
    sparse_audit, sparse_examples = sparse_reference_audit(data["terrain"], data["dataset"],
        paths["terrain"], paths["dataset"], args.output)
    plot_scoring_domains(sparse_examples, args.output)
    selection, boxes = terrain_figures(data, paths, stats, args.output)
    distance = distance_analysis(data, paths, args.output)
    risk_figures(data, paths, stats, args.output)
    examples, physical = navigation_figures(data, args.output)
    timing, backend = runtime_figures(data, args.output)
    analytic = special_figures(data, args.output)
    tables(data, stats, args.output)
    from supplemental_evidence import plot_campaign
    new = plot_campaign(data, paths, args.output)
    from speed_response_evidence import plot_speed_campaign
    speed = plot_speed_campaign(data, paths, args.output)
    from planning_timing_evidence import plot_planning_timing, plot_workload_bars
    planning_timing = plot_planning_timing(data, paths, args.output)
    plot_workload_bars(data, args.output)
    executions = {"joint_terrain_cases": new["new_joint_terrain_cases"],
        "scaling_measured_calls": new["measured_scaling_calls"], "model_navigation_cases": new["new_model_navigation_cases"],
        "paired_speed_model_cases": speed["cases"], "physical_cases": 0}
    summary = plain({"study_id": protocol["study_id"], "reanalysis_only": True,
        "report_revision": {"date": protocol["date"], "new_algorithm_runs": 0,
            "planning_tasks_reanalysed": planning_timing["tasks"], "planning_calls_reanalysed": planning_timing["calls"]},
        "execution_count_scope": "recorded prior batches; no new algorithm executions in the v2 report revision",
        "recorded_prior_executions": executions, "algorithm_runs_during_report_build": 0,
        "terrain": stats, "terrain_statuses": Counter(r["status"] for r in data["terrain"]["runs"]),
        "source_boxplots_cm": boxes, "spatial_selection": selection, "distance_arrays_checked": distance["arrays_checked"],
        "distance_array_chain": distance["array_digest_chain"], "navigation_examples": examples,
        "model_navigation": data["nav_stats"], "physical_navigation": physical, "component_timing": timing,
        "backend_timing_ms": backend, "analytic": analytic, "road_reference_rescored_cases": road_audit["cases"],
        "sparse_reference_conditions_audited": sparse_audit["cases"],
        "completion": data["completion_stats"],
        "completion_state_totals": data["completion_totals"], "hashes": {k: digest(v) for k, v in paths.items()},
        "protocol_sha256": digest(args.protocol), "plotter_sha256": digest(__file__),
        "helper_sha256": {name: digest(Path(__file__).with_name(name)) for name in
            ("thesis_statistics.py", "plot_paper_scale.py", "plot_expanded_campaign.py", "plot_style.py",
             "pointcloud_batch.py", "paper_scale_evaluate.py", "paper_clouds.py", "plot_thesis_evidence.py",
             "spatial_evidence.py", "road_reference_audit.py", "sparse_reference_audit.py",
             "controlled_terrain_study.py", "supplemental_evidence.py", "speed_response_study.py", "speed_response_evidence.py", "planning_timing_evidence.py")}})
    atomic_json(args.output / "results.json", summary)
    # Every plotted quantity keeps its experiment identity instead of a chronological figure index.
    metrics = []
    for family in FAMILIES:
        for condition in CONDITIONS:
            for metric, s in stats[family][condition].items():
                metrics.append({"experiment": "M2" if metric in ("risk_mae", "false_safe", "false_hazard") else "M1",
                    "family": family, "condition": condition, "metric": metric,
                    **{k: s[k] for k in ("mean", "low", "high", "clusters")}})
    with (args.output / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, metrics[0].keys()); w.writeheader(); w.writerows(metrics)
    for name in names:
        if not (args.output / (name + ".pdf")).exists():
            raise ValueError("missing planned figure: " + name)
    artifacts = ["results.json", "metrics.csv", "paired_difficulty.json", "distance_diagnostics.json", "terrain_rows.tex",
                 "navigation_rows.tex", "dynamic_rows.tex", "road_reference_audit.json", "road_reference_cases.csv", "road_reference_rows.tex",
                 "sparse_reference_audit.json", "sparse_reference_cases.csv", "sparse_reference_rows.tex",
                 "supplemental_results.json", "supplemental_counts.tex", "C1_false_safe_states.csv", "C1_rows.tex", "C2_rows.tex", "C3_rows.tex",
                 "speed_response_results.json", "speed_response_cases.csv", "C4_speed_rows.tex", "C4_response_rows.tex", "C4_findings.tex",
                 "planning_timing_results.json", "planning_timing_tasks.csv", "planning_timing_calls.csv", "planning_timing_rows.tex", "planning_timing_counts.tex"
                 ] + [name + extension for name in names for extension in (".pdf", ".png")]
    atomic_json(args.output / "figure_manifest.json", {"study_id": protocol["study_id"], "figures": [
        {"id": name, "sha256": digest(args.output / (name + ".pdf"))} for name in names],
        "artifacts": {name: digest(args.output / name) for name in artifacts}})
    print(json.dumps({"figures": len(names), "recorded_prior_executions": executions, "terrain_conditions": len(data["terrain"]["runs"]),
                      "model_tasks": len(data["navigation"]["runs"]), "physical_tasks": len(data["physical"]["runs"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
