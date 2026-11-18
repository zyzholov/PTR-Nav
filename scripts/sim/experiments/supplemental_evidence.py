"""Audit and explain new controlled runs without mixing their old/reference scopes."""

from collections import Counter
import csv
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from matplotlib.patches import Circle

from controlled_terrain_study import NAMES, NOMINAL, footprint_truth, score_height, score_risk, specification, surface
from fsgp_bgk_se2.bgk_seb import BgkSebParams
from paper_clouds import grid
from plot_paper_scale import clustered_summary, plt, save
from plot_expanded_campaign import SCENARIOS, validate_navigation
from plot_style import wilson_interval
from pointcloud_batch import ROOT, atomic_json, digest
from speed_response_evidence import task_tracking_value


FIGURES = ("C1_quality", "C1_noise", "C1_sections", "C1_uncertainty", "C1_risk_spatial", "C2_scaling", "C3_navigation", "C3_runtime")


def validate_new_campaign(data, paths):
    protocol, terrain = data["supplemental_protocol"], data["new_terrain"]
    p = protocol["terrain"]
    expected = {f"{f}_{r:03d}_{m}_n{n}" for f in p["families"] for r in range(p["geometries_per_family"])
        for m in p["masks"] for n in range(len(p["noise_std_m"]))}
    actual = [r["id"] for r in terrain["runs"]]
    if set(actual) != expected or len(actual) != p["planned_cases"] or terrain["settings"]["terrain_plan"] != p:
        raise ValueError("controlled terrain campaign is partial or has a different protocol")
    if terrain["settings"]["protocol_sha256"] != digest(paths["supplemental_protocol"]):
        raise ValueError("controlled terrain protocol changed")
    for fi, family in enumerate(p["families"]):
        for repeat in range(p["geometries_per_family"]):
            expected_spec = specification(family, p["seed"]+fi*10000+repeat)
            for mask in p["masks"]:
                for ni, noise in enumerate(p["noise_std_m"]):
                    r = next(r for r in terrain["runs"] if r["id"] == f"{family}_{repeat:03d}_{mask}_n{ni}")
                    if r["spec"] != expected_spec or r["mask"] != mask or r["noise_std_m"] != noise:
                        raise ValueError("case label/geometry/noise mismatch")
                    if r["status"] == "ok" and (r["actual_device"] != "cuda" or r["actual_backend"] != "cupy"):
                        raise ValueError("GPU results include a fallback")
    scaling = data["new_scaling"]; s = protocol["scaling"]
    expected = {(size, yaw, backend, repeat) for size in s["sizes"] for yaw in s["yaw_bins"]
        for backend in s["backends"] for repeat in range(s["warmup_calls"]+1, s["warmup_calls"]+s["measured_calls"]+1)}
    actual = [(r["size"], r["yaw_bins"], r["backend"], r["repeat"]) for r in scaling["runs"]]
    if set(actual) != expected or len(actual) != s["planned_measured_calls"]:
        raise ValueError("scaling repetitions are incomplete or duplicated")
    if scaling["protocol_sha256"] != digest(paths["supplemental_protocol"]):
        raise ValueError("scaling protocol changed")
    for record in (terrain["settings"], scaling, data["new_navigation_plan"]["settings"]):
        for name, expected_hash in record["source_sha256"].items():
            if digest(ROOT/name) != expected_hash:
                raise ValueError("new campaign method source changed: "+name)
    plan, nav = data["new_navigation_plan"], data["new_navigation"]
    validate_navigation(plan, nav)
    counts = Counter(c["scenario"] for c in plan["cases"])
    n = protocol["navigation"]
    if counts != Counter({name: n["per_scenario"] for name in n["scenarios"]}) or plan["settings"]["seed"] != n["seed"]:
        raise ValueError("new navigation scenarios/seeds differ from protocol")
    if plan["settings"]["mppi"] != n["mppi"]:
        raise ValueError("new navigation controller differs from protocol")


def audit_terrain(data, paths, output):
    p = data["supplemental_protocol"]["terrain"]
    params = BgkSebParams(**data["new_terrain"]["settings"]["bgk_seb"])
    rows = []; misses = []; chain = hashlib.sha256()
    for r in data["new_terrain"]["runs"]:
        row = dict(r, cluster_id=r["geometry_id"])
        if r["status"] != "ok":
            rows.append(row); continue
        path = paths["new_terrain"].parent/(r["id"]+".npz")
        sha = digest(path)
        if sha != r["archive_sha256"]:
            raise ValueError("controlled arrays changed")
        chain.update((r["id"]+sha).encode())
        with np.load(path, allow_pickle=False) as a:
            xy = grid(a["truth"].shape, a["origin"], float(a["resolution"]))
            truth = surface(r["spec"], xy)
            if not np.array_equal(truth.reshape(a["truth"].shape), a["truth"]):
                raise ValueError("analytic truth differs from the frozen source")
            ref, unsafe, _, _ = footprint_truth(r["spec"], a["query_xy"], params)
            if not np.allclose(ref, a["reference_risk"], atol=1e-10) or not np.array_equal(unsafe, a["unsafe"]):
                raise ValueError("independent footprint oracle changed")
            delta = .01; dx = np.array([delta, 0.]); dy = dx[::-1]
            gradient = np.c_[(surface(r["spec"], xy+dx)-surface(r["spec"], xy-dx))/(2*delta),
                (surface(r["spec"], xy+dy)-surface(r["spec"], xy-dy))/(2*delta)]
            smooth = np.ones(len(xy), bool)
            if r["family"].startswith("step_"):
                u = xy[:, 0]*np.cos(r["spec"]["angle"])+xy[:, 1]*np.sin(r["spec"]["angle"])
                smooth &= np.abs(u-r["spec"]["offset"]) > .4
            prediction = SimpleNamespace(mean=a["mean"].ravel(), latent_var=a["variance"].ravel(),
                accepted=a["accepted"].ravel(), gradient_xy=a["gradient"])
            h = score_height(prediction, truth, np.isfinite(a["observed"]).ravel(), smooth, gradient)
            risk = score_risk(a["predicted_risk"], ref, unsafe, p["risk_threshold"])
            for key, value in (h | risk).items():
                if value is None:
                    if r[key] is not None: raise ValueError("empty metric denominator was zero-filled")
                elif not np.allclose(value, r[key], rtol=1e-6, atol=1e-7):
                    raise ValueError("controlled metric differs from saved arrays: "+key)
            row.update(picp95=r["picp"][3], width95_m=r["width_m"][3])
            for yaw_index, position in np.argwhere((a["predicted_risk"] < p["risk_threshold"]) & unsafe):
                q = a["query_xy"][position]
                misses.append({"case_id": r["id"], "family": r["family"], "mask": r["mask"],
                    "noise_std_m": r["noise_std_m"], "x_m": q[0], "y_m": q[1],
                    "yaw_deg": (yaw_index+.5)*360/p["yaw_bins"],
                    "prediction": float(a["predicted_risk"][yaw_index, position]),
                    "reference": float(ref[yaw_index, position])})
        rows.append(row)
    if misses:
        with (output/"C1_false_safe_states.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, misses[0].keys()); writer.writeheader(); writer.writerows(misses)
    else:
        (output/"C1_false_safe_states.csv").write_text("case_id,family,mask,noise_std_m,x_m,y_m,yaw_deg,prediction,reference\n", encoding="utf-8")
    return rows, {"arrays_verified": sum(r["status"] == "ok" for r in rows), "array_chain_sha256": chain.hexdigest(),
        "false_safe_state_occurrences": len(misses), "unique_geometries": len({r["geometry_sha256"] for r in rows})}


def metric_summary(rows, family, mask, noise, metric):
    return clustered_summary([r for r in rows if r["family"] == family and r["mask"] == mask and r["noise_std_m"] == noise], metric)


def section_indices(shape, origin, resolution, angle):
    # The dominant normal axis crosses the boundary without interpolating stored predictions.
    along_x = abs(np.cos(angle)) >= abs(np.sin(angle))
    fixed_axis = 1 if along_x else 0
    fixed_count = shape[0] if along_x else shape[1]
    varying_count = shape[1] if along_x else shape[0]
    fixed = int(np.argmin(np.abs(origin[fixed_axis]+(np.arange(fixed_count)+.5)*resolution)))
    varying_axis = 1-fixed_axis
    x = origin[varying_axis]+(np.arange(varying_count)+.5)*resolution
    index = (fixed, slice(None)) if along_x else (slice(None), fixed)
    return x, index, "x" if along_x else "y", float(origin[fixed_axis]+(fixed+.5)*resolution)


def quality_figures(rows, data, paths, output):
    p = data["supplemental_protocol"]["terrain"]; families, masks = p["families"], p["masks"]
    colors = plt.cm.tab10(np.arange(8))
    fig, axes = plt.subplots(2, 2, figsize=(8.1, 7.1), layout="constrained")
    for ax, metric, factor, title, label in zip(axes.flat,
        ("hidden_rmse_m", "normal_mae_deg", "fnr", "fpr"), (100, 1, 100, 100),
        ("隐藏高程：均方根误差", "隐藏平滑地面：法向误差", "参考危险：漏判率", "参考安全：误拒率"),
        ("厘米", "度", "%", "%")):
        values = np.array([[metric_summary(rows, f, m, .02, metric)["mean"]
            if metric_summary(rows, f, m, .02, metric)["mean"] is not None else np.nan for m in masks] for f in families])*factor
        cmap = plt.colormaps["viridis"].copy(); cmap.set_bad("#e2e5e7")
        image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=0)
        ax.set_xticks(range(3), [NAMES[m] for m in masks], rotation=22, ha="right", fontsize=8)
        ax.set_yticks(range(8), [NAMES[f] for f in families]); ax.set_title(title)
        for (i, j), value in np.ndenumerate(values):
            ax.text(j, i, "--" if np.isnan(value) else f"{value:.1f}", ha="center", va="center", fontsize=8,
                color="black" if np.isnan(value) or value > np.nanmax(values)*.6 else "white")
        fig.colorbar(image, ax=ax, shrink=.8, label=label)
    fig.suptitle("受控地形联合测试：高程误差小，不代表风险判断可靠\n实际输入噪声2厘米；缺危险参考的格用 --，不是零漏判", fontsize=11)
    save(fig, output, "C1_quality")
    fig, axes = plt.subplots(2, 3, figsize=(8.4, 6.7), layout="constrained")
    for col, mask in enumerate(masks):
        for family, color in zip(families, colors):
            for row_index, metric, factor in ((0, "hidden_rmse_m", 100), (1, "fpr", 100)):
                summaries = [metric_summary(rows, family, mask, n, metric) for n in p["noise_std_m"]]
                y = np.array([s["mean"] if s["mean"] is not None else np.nan for s in summaries])*factor
                ax = axes[row_index, col]
                ax.plot(np.array(p["noise_std_m"])*100, y, "o-", color=color, label=NAMES[family], ms=3)
                if family != "flat":
                    ax.fill_between(np.array(p["noise_std_m"])*100,
                        np.array([s["low"] if s["low"] is not None else np.nan for s in summaries])*factor,
                        np.array([s["high"] if s["high"] is not None else np.nan for s in summaries])*factor, color=color, alpha=.1)
        axes[0, col].set_title(NAMES[mask]); axes[0, col].set_yscale("symlog", linthresh=1)
        axes[0, col].set_ylabel("隐藏高程 RMSE（厘米）")
        axes[1, col].set_ylabel("安全误拒率（%）"); axes[1, col].set_ylim(0, 103)
        for ax in axes[:, col]: ax.set_xlabel("输入噪声标准差（厘米）"); ax.set_xticks([0, 2, 4]); ax.grid(alpha=.18)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=8)
    fig.suptitle("噪声压力响应：同几何、同缺测掩码，仅改变输入噪声\n噪声配置固定2厘米，训练仍更新似然；阴影是几何组均值区间", fontsize=10)
    save(fig, output, "C1_noise")
    chosen = [next(r for r in rows if r["id"] == f+"_000_hole16_n1") for f in ("ridge", "step_high")]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5), layout="constrained")
    for ax, r in zip(axes, chosen):
        with np.load(paths["new_terrain"].parent/(r["id"]+".npz")) as a:
            x, index, axis, fixed = section_indices(a["truth"].shape, a["origin"], float(a["resolution"]), r["spec"]["angle"])
            mean, std, truth, observed = a["mean"][index], np.sqrt(a["variance"][index]), a["truth"][index], a["observed"][index]
            ax.fill_between(x, mean-1.96*std, mean+1.96*std, color="#24788e", alpha=.22, label="名义95%高程区间")
            ax.plot(x, truth, color="#242a30", label="解析参考高度"); ax.plot(x, mean, color="#24788e", label="预测中心")
            visible = np.isfinite(observed); ax.scatter(x[visible], observed[visible], s=10, color="#c27845", label="实际地面输入")
            ax.axvspan(-.8, .8, color="#c27845", alpha=.07); ax.set_xlim(-2, 2)
            other = "y" if axis == "x" else "x"
            ax.set_title(NAMES[r["family"]]+f"：首个几何 / {other}={fixed:.1f}米")
            ax.set_xlabel(f"局部 {axis}（米）"); ax.set_ylabel("高程（米）")
            ax.grid(alpha=.18)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=8)
    fig.suptitle("区间宽度就是蓝色带的竖向厚度，不是横向孔洞宽度\n黑线是否落入蓝带才决定该位置是否命中；固定首样例，不挑最优", fontsize=10)
    save(fig, output, "C1_sections")
    fig, axes = plt.subplots(1, 2, figsize=(8.1, 4.1), layout="constrained")
    for family, color in zip(families, colors):
        selected = [r for r in rows if r["family"] == family and r["mask"] == "hole16" and r["noise_std_m"] == .02 and r["status"] == "ok"]
        groups = {r["geometry_id"] for r in selected}
        coverage = np.mean([np.mean([r["picp"] for r in selected if r["geometry_id"] == g], axis=0) for g in groups], axis=0)
        axes[0].plot(NOMINAL*100, coverage*100, "o-", color=color, label=NAMES[family], ms=3)
        for mask, marker in zip(masks, ("o", "s", "^")):
            selected = [r for r in rows if r["family"] == family and r["mask"] == mask and r["noise_std_m"] == .02 and r["status"] == "ok"]
            axes[1].scatter([r["hidden_rmse_m"]*100 for r in selected], [r["width95_m"]*100 for r in selected],
                color=color, marker=marker, s=13, alpha=.65)
    axes[0].plot([50, 99], [50, 99], "--", color="#626b72")
    axes[0].set(xlabel="预设名义覆盖 q（%）", ylabel="实际高程覆盖 PICP（%）", xlim=(48, 101), ylim=(0, 103), title="1.6米孔洞：区间校准")
    axes[1].set(xlabel="每案例隐藏高程 RMSE（厘米）", ylabel="每案例95%区间总宽（厘米）", title="误差与预测范围是否同时合理")
    axes[1].set_xscale("symlog", linthresh=1); axes[1].set_yscale("log")
    for ax in axes: ax.grid(alpha=.18)
    handles, labels = axes[0].get_legend_handles_labels(); fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=8)
    fig.suptitle("新受控批次的不确定性：实际输入噪声2厘米\n右图圆/方/三角分别为随机删点/孔洞/单侧隐藏；相关案例不当独立世界", fontsize=10)
    save(fig, output, "C1_uncertainty")
    fig = plt.figure(figsize=(8.3, 6.6), layout="constrained")
    for row_index, case_id in enumerate(("column_000_random30_n1", "step_high_000_hole16_n1")):
        r = next(r for r in rows if r["id"] == case_id)
        with np.load(paths["new_terrain"].parent/(case_id+".npz")) as a:
            ax = fig.add_subplot(2, 3, row_index*3+1, projection="3d")
            cloud = a["raw_xyz"]; take = (np.abs(cloud[:, :2]) < 2).all(axis=1); cloud = cloud[take]
            ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], c=cloud[:, 2], cmap="terrain", s=2, rasterized=True)
            ax.set_xlabel("x（米）", labelpad=0); ax.set_ylabel("y（米）", labelpad=0)
            ax.text2D(.9, .9, "z（米）", transform=ax.transAxes, fontsize=8)
            ax.set_title(NAMES[r["family"]]+"：实际输入点云", fontsize=9); ax.set_box_aspect((4, 4, 2))
            ax.set_xlim(-2, 2); ax.set_ylim(-2, 2); ax.view_init(elev=28, azim=-58)
            for col, key, title in ((1, "reference_risk", "独立几何参考"), (2, "predicted_risk", "本工程风险输出")):
                ax = fig.add_subplot(2, 3, row_index*3+col+1)
                q = a["query_xy"]; im = ax.scatter(q[:, 0], q[:, 1], c=a[key][0], s=34, marker="s", cmap="RdYlGn_r", vmin=0, vmax=1)
                if r["family"] == "column":
                    ax.add_patch(Circle((r["spec"]["offset"], 0), r["spec"]["column_radius"], fill=False, color="black", lw=1))
                ax.set_aspect("equal"); ax.set_title(title+"：航向5度", fontsize=9)
                ax.set_xlabel("x（米）"); ax.set_ylabel("y（米）"); ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)
                fig.colorbar(im, ax=ax, shrink=.75, label="风险（无量纲）")
    fig.suptitle("输入对象与车辆足迹风险：细柱、台阶不是只有地面高度\n每图显示固定81查询位置的同一航向；未将稀疏参考插值成整图", fontsize=10)
    save(fig, output, "C1_risk_spatial")


def scaling_figure(data, output):
    p = data["supplemental_protocol"]["scaling"]; rows = data["new_scaling"]["runs"]
    summaries = []; fig, axes = plt.subplots(1, 2, figsize=(8.1, 3.7), layout="constrained")
    for yaw, color in zip(p["yaw_bins"], ("#24788e", "#c27845")):
        for backend, style in (("cpu", "-"), ("cupy", "--")):
            median, tail = [], []
            for size in p["sizes"]:
                a = np.array([r["elapsed_ms"] for r in rows if r["size"] == size and r["yaw_bins"] == yaw and r["backend"] == backend])
                s = {"size": size, "yaw_bins": yaw, "backend": backend, "n": len(a),
                    "p50_ms": float(np.median(a)), "p95_ms": float(np.quantile(a, .95)),
                    "p99_ms": float(np.quantile(a, .99)), "max_ms": float(a.max())}
                summaries.append(s); median.append(s["p50_ms"]); tail.append(s["p95_ms"])
            for ax, values in zip(axes, (median, tail)):
                ax.plot(p["sizes"], values, "o"+style, color=color, label=f"{backend.upper()} / {yaw}航向", ms=4)
                ax.set_yscale("log"); ax.set_xlabel("正方形地图侧长（格）"); ax.set_xticks(p["sizes"]); ax.grid(alpha=.18)
    axes[0].set_ylabel("同步组件耗时中位数（毫秒）"); axes[1].set_ylabel("同步组件耗时 P95（毫秒）")
    handles, labels = axes[0].get_legend_handles_labels(); fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=8)
    fig.suptitle("地图规模与航向数扩展：每工作负载暖机20次后测50次\n含主机传输及结果物化；共享GPU；不包含RTX、GP拟合或ROS队列", fontsize=10)
    save(fig, output, "C2_scaling")
    return summaries


def navigation_figures(data, output):
    rows = data["new_navigation"]["runs"]; summary = {}
    for name in data["supplemental_protocol"]["navigation"]["scenarios"]:
        selected = [r for r in rows if r["scenario"] == name]
        expected = {r["expected_stop"] for r in selected if r["status"] != "error"}
        if len(expected) != 1 or any(r["status"] == "error" for r in selected):
            raise ValueError("navigation errors need explicit nonzero-denominator handling")
        passed = sum(r["status"] == "passed" for r in selected)
        tracking = [task_tracking_value(r) for r in selected if task_tracking_value(r) is not None]
        summary[name] = {"n": len(selected), "passed": passed, "expected_stop": expected.pop(),
            "collision": sum(r["collision"] for r in selected), "terrain_violation": sum(r["unsafe_terrain"] for r in selected),
            "unknown_violation": sum(r["unknown_policy_violation"] for r in selected),
            "tracking_evaluable_cases": len(tracking),
            "tracking_case_p95_median_m": float(np.median(tracking)) if tracking else None,
            "planner_case_p95_median_ms": float(np.median([r['planner_p95_ms'] for r in selected])),
            "planner_max_ms": max(r["planner_max_ms"] for r in selected),
            "maximum_plan_calls_per_task": max(r["plan_count"] for r in selected),
            "maximum_successful_replans_per_task": max(r["replans"] for r in selected),
            "interval": wilson_interval(passed, len(selected))}
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.7), layout="constrained", gridspec_kw={"width_ratios": [1.7, 1]})
    for ax, stop, title in zip(axes, (False, True), ("普通任务：到达并停车", "危险/失效任务：安全拒绝")):
        names = [k for k, s in summary.items() if s["expected_stop"] == stop]
        for i, name in enumerate(names):
            s = summary[name]; ratio = s["passed"]/s["n"]
            ax.barh(i, ratio*100, color="#24788e" if s["passed"] == s["n"] else "#bc4b46")
            ax.errorbar(ratio*100, i, xerr=np.array([[ratio-s["interval"][0]], [s["interval"][1]-ratio]])*100, color="#333a40", capsize=2)
            ax.text(min(ratio*100+2, 101), i, f"{s['passed']}/{s['n']}", va="center", fontsize=8)
        ax.set_yticks(range(len(names)), [SCENARIOS[n] for n in names]); ax.invert_yaxis(); ax.set_xlim(0, 118)
        ax.set_xlabel("任务合格比例（%）"); ax.set_title(title, fontsize=10); ax.grid(axis="x", alpha=.18)
    fig.suptitle("新种子生产模型闭环：14类任务各20次，完整保留失败\n误差条为场景族内Wilson区间；不等于Isaac物理或实车可靠性", fontsize=10)
    save(fig, output, "C3_navigation")
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.7), layout="constrained")
    selected_names = ("straight", "reverse", "detour", "ramp", "step", "emerge", "crossing")
    palette = ("#24788e", "#c27845", "#42856d", "#74618b", "#9a8734", "#bc4b46", "#626b72")
    for name, color in zip(selected_names, palette):
        for ax, key in zip(axes, ("planner_p95_ms", "tracking_p95_m")):
            values = [task_tracking_value(r) if key == "tracking_p95_m" else r[key] for r in rows if r["scenario"] == name]
            a = np.sort([v for v in values if v is not None])
            if not len(a): continue
            ax.step(a, np.arange(1, len(a)+1)/len(a), where="post", label=SCENARIOS[name], color=color,
                linestyle="--" if name in ("emerge", "crossing") else "-")
            ax.grid(alpha=.18); ax.set_ylabel("不超过横轴值的任务比例")
    axes[0].set_xlabel("每任务规划调用 P95（毫秒）"); axes[0].set_xscale("log")
    axes[1].set_xlabel("每任务跟踪距离 P95（米）")
    handles, labels = axes[0].get_legend_handles_labels(); fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=8)
    fig.suptitle("生产模型的规划耗时与跟踪误差：七类预定任务族，每类20任务\n仅可评价跟踪值进入CDF；失败保留，低误差不等于到达", fontsize=10)
    save(fig, output, "C3_runtime")
    return summary


def plot_campaign(data, paths, output):
    validate_new_campaign(data, paths)
    rows, audit = audit_terrain(data, paths, output)
    quality_figures(rows, data, paths, output)
    scaling = scaling_figure(data, output); navigation = navigation_figures(data, output)
    p = data["supplemental_protocol"]["terrain"]; table = []; stats = {}
    for family in p["families"]:
        stats[family] = {}
        for mask in p["masks"]:
            stats[family][mask] = {}
            for noise in p["noise_std_m"]:
                selected = [r for r in rows if r["family"] == family and r["mask"] == mask and r["noise_std_m"] == noise]
                metrics = {k: clustered_summary(selected, k) for k in (
                    "hidden_rmse_m", "hidden_mae_m", "hidden_p95_m", "normal_mae_deg", "picp95", "width95_m", "crps_m", "fnr", "fpr")}
                stats[family][mask][str(noise)] = metrics
                if mask == "hole16" and noise == .02:
                    fmt = lambda key, scale=1: "--" if metrics[key]["mean"] is None else f"{metrics[key]['mean']*scale:.2f}"
                    fnr = metrics["fnr"]
                    table.append(f"{NAMES[family]} & {fmt('hidden_rmse_m',100)} & {fmt('normal_mae_deg')} & " +
                        f"{fmt('picp95',100)} & {fmt('width95_m',100)} & {fmt('fnr',100)} & {fnr['clusters']} & {fmt('fpr',100)} " + r"\\")
    (output/"C1_rows.tex").write_text("\n".join(table)+"\n"+r"\bottomrule"+"\n", encoding="utf-8")
    lines = [f"{SCENARIOS[name]} & {s['n']} & {s['passed']} & {s['collision']} & {s['terrain_violation']} & " +
        ("--" if s["tracking_case_p95_median_m"] is None else f"{s['tracking_case_p95_median_m']:.3f}") + " " + r"\\" for name, s in navigation.items()]
    (output/"C3_rows.tex").write_text("\n".join(lines)+"\n"+r"\bottomrule"+"\n", encoding="utf-8")
    lines = []
    for size in data["supplemental_protocol"]["scaling"]["sizes"]:
        s = {r["backend"]: r for r in scaling if r["size"] == size and r["yaw_bins"] == 36}
        lines.append(f"{size} & {s['cpu']['p50_ms']:.2f} & {s['cpu']['p95_ms']:.2f} & {s['cupy']['p50_ms']:.2f} & " +
            f"{s['cupy']['p95_ms']:.2f} & {s['cpu']['p50_ms']/s['cupy']['p50_ms']:.1f} " + r"\\")
    (output/"C2_rows.tex").write_text("\n".join(lines)+"\n"+r"\bottomrule"+"\n", encoding="utf-8")
    totals = {}
    for stop, label in ((False, "Arrival"), (True, "Rejection")):
        selected = [s for s in navigation.values() if s["expected_stop"] == stop]
        totals["New"+label+"Passed"] = sum(s["passed"] for s in selected)
        totals["New"+label+"Cases"] = sum(s["n"] for s in selected)
    totals.update(NewDetourPassed=navigation["detour"]["passed"], NewEmergePassed=navigation["emerge"]["passed"],
        NewTerrainMisses=audit["false_safe_state_occurrences"],
        NewTerrainViolations=sum(s["terrain_violation"] for s in navigation.values()),
        NewCollisions=sum(s["collision"] for s in navigation.values()),
        NewWorstPlannerMs=f"{max(s['planner_max_ms'] for s in navigation.values()):.2f}",
        NewPeakPlannerCalls=max(s["maximum_plan_calls_per_task"] for s in navigation.values()))
    (output/"supplemental_counts.tex").write_text("\n".join(
        rf"\newcommand{{\{name}}}{{{value}}}" for name, value in totals.items())+"\n", encoding="utf-8")
    summary = {"study_id": data["supplemental_protocol"]["study_id"], "new_joint_terrain_cases": len(rows),
        "terrain_statuses": dict(Counter(r["status"] for r in rows)), "audit": audit, "terrain": stats,
        "measured_scaling_calls": len(data["new_scaling"]["runs"]), "scaling": scaling,
        "scaling_agreement": data["new_scaling"]["agreement"], "new_model_navigation_cases": len(data["new_navigation"]["runs"]),
        "navigation": navigation, "new_physical_cases": 0,
        "hashes": {k: digest(paths[k]) for k in ("supplemental_protocol", "new_terrain", "new_scaling", "new_navigation_plan", "new_navigation")}}
    atomic_json(output/"supplemental_results.json", summary)
    return summary
