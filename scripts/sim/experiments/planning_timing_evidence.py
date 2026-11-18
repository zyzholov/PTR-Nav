"""Call-level timing audit; simulation timestamps are never wall-clock latency."""

from collections import Counter, defaultdict
import csv

import numpy as np

from plot_expanded_campaign import SCENARIOS, validate_navigation
from plot_paper_scale import plt, save
from pointcloud_batch import atomic_json, digest


def native_quantile(values, q):
    values = sorted(values)
    return values[min(len(values)-1, int(q*(len(values)-1)))] if values else None


def audit_task(row):
    calls = row.get("planning", [])
    if row["status"] == "error":
        if calls:
            raise ValueError("partial error trace needs a separately defined scoring rule")
    else:
        if row["plan_count"] != len(calls) or row["replans"] != sum(c["success"] for c in calls):
            raise ValueError("native call counts disagree with their trace")
        elapsed = [c["elapsed_ms"] for c in calls]
        for key, q in (("planner_p95_ms", .95), ("planner_max_ms", 1.)):
            if not np.isclose(row[key], native_quantile(elapsed, q) or 0., rtol=1e-9, atol=1e-9):
                raise ValueError("native planning quantile disagrees with its trace")
    previous = -1.
    for c in calls:
        if (not np.isfinite(c["elapsed_ms"]) or c["elapsed_ms"] < 0 or
                not np.isfinite(c["time_s"]) or c["time_s"] < 0 or c["time_s"] < previous or
                type(c["success"]) is not bool or c["path_points"] < 0):
            raise ValueError("invalid timing trace")
        previous = c["time_s"]
    following = calls[1:]
    control = row.get("control_p95_ms")
    if control is not None and (not np.isfinite(control) or control < 0):
        raise ValueError("invalid saved control quantile")
    return {"id": row["id"], "scenario": row["scenario"], "status": row["status"],
        "expected_stop": row.get("expected_stop"), "goal_error_m": row.get("goal_error_m"),
        "first_call_ms": calls[0]["elapsed_ms"] if calls else None,
        "first_call_returned_success": calls[0]["success"] if calls else None,
        "subsequent_case_p95_ms": float(np.quantile([c["elapsed_ms"] for c in following], .95)) if following else None,
        "control_saved_case_p95_ms": control,
        "calls": len(calls), "subsequent_calls": len(following),
        "returned_success_calls": sum(c["success"] for c in calls),
        "failed_call_fraction": sum(not c["success"] for c in calls)/len(calls) if calls else None,
        "planning_sum_ms": sum(c["elapsed_ms"] for c in calls) if calls else None}


def task_equal_ecdf(calls):
    """Equal weight per contributing task, then equal weight per conditional call."""
    groups = defaultdict(list)
    for c in calls:
        groups[c["task_id"]].append(c["elapsed_ms"])
    if not groups:
        return np.array([]), np.array([]), 0
    values, weights = [], []
    for group in groups.values():
        values.extend(group)
        weights.extend([1/len(groups)/len(group)]*len(group))
    order = np.argsort(values)
    return np.asarray(values)[order], np.cumsum(np.asarray(weights)[order]), len(groups)


def write_csv(path, rows):
    if not rows:
        raise ValueError("empty timing export")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, rows[0].keys())
        writer.writeheader(); writer.writerows(rows)


def plot_planning_timing(data, paths, output):
    plan, record = data["new_navigation_plan"], data["new_navigation"]
    validate_navigation(plan, record)
    tasks, calls = [], []
    for row in record["runs"]:
        tasks.append(audit_task(row))
        for i, c in enumerate(row.get("planning", [])):
            calls.append({"task_id": row["id"], "scenario": row["scenario"], "task_status": row["status"],
                "call_index": i, "phase": "first" if i == 0 else "subsequent",
                "simulation_time_s": c["time_s"], "elapsed_ms": c["elapsed_ms"],
                "returned_success": c["success"], "path_points": c["path_points"], "reason": c["reason"]})
    names = list(dict.fromkeys(c["scenario"] for c in plan["cases"]))
    summary = {}
    for scene in names:
        selected = [t for t in tasks if t["scenario"] == scene]
        s = {"tasks": len(selected), "statuses": dict(Counter(t["status"] for t in selected))}
        for key in ("first_call_ms", "subsequent_case_p95_ms", "control_saved_case_p95_ms", "failed_call_fraction"):
            values = [t[key] for t in selected if t[key] is not None]
            s[key] = {"n": len(values), "median": float(np.median(values)) if values else None,
                "p95": float(np.quantile(values, .95)) if values else None}
        summary[scene] = s
    fig, axes = plt.subplots(1, 3, figsize=(9.1, 6.0), layout="constrained")
    keys = ("first_call_ms", "subsequent_case_p95_ms", "control_saved_case_p95_ms")
    titles = ("首次规划调用：每类20任务", "后续规划/重试：任务内P95", "控制回调：已保存任务内P95")
    for ax, key, title, color in zip(axes, keys, titles, ("#3985B8", "#399C94", "#7A87A5")):
        groups = [[t[key] for t in tasks if t["scenario"] == scene and t[key] is not None] for scene in names]
        boxes = ax.boxplot([g or [np.nan] for g in groups], vert=False, patch_artist=True,
            widths=.55, medianprops={"color": "#203946"}, flierprops={"marker": ".", "markersize": 2})
        for box in boxes["boxes"]: box.set(facecolor=color, alpha=.5)
        for j, scene in enumerate(names, 1):
            bad = [t[key] for t in tasks if t["scenario"] == scene and t["status"] != "passed" and t[key] is not None]
            if bad: ax.scatter(bad, np.full(len(bad), j), s=9, facecolors="none", edgecolors="#C15A52", zorder=3)
            if not groups[j-1]:
                ax.text(.55, j, "无后续调用（n=0）", transform=ax.get_yaxis_transform(), ha="center", va="center", fontsize=7)
        ax.set_yticks(range(1, len(names)+1), [SCENARIOS[n] for n in names] if ax is axes[0] else [""]*len(names), fontsize=8)
        ax.invert_yaxis(); ax.set_xscale("linear" if key == "control_saved_case_p95_ms" else "log")
        ax.grid(axis="x", alpha=.15)
        scale = "线性轴" if key == "control_saved_case_p95_ms" else "对数轴"
        ax.set_title(title, fontsize=9); ax.set_xlabel(f"墙钟耗时（毫秒，{scale}）", fontsize=8)
    fig.suptitle("同一批280任务的分阶段耗时：首次、后续调用与控制不能混算\n每个箱体样本是一个任务；红圈为未合格任务，各阶段可评价数见表", fontsize=10)
    save(fig, output, "P1_phase_timing")

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), layout="constrained")
    distribution_counts = {}
    for ax, scene in zip(axes, ("detour", "emerge")):
        distribution_counts[scene] = {}
        for success, color in ((True, "#3985B8"), (False, "#D47A65")):
            selected = [c for c in calls if c["scenario"] == scene and c["phase"] == "subsequent" and c["returned_success"] == success]
            x, y, n = task_equal_ecdf(selected)
            label = "返回成功" if success else "返回失败"
            distribution_counts[scene][label] = {"tasks": n, "calls": len(selected)}
            if len(x): ax.step(x, y, where="post", color=color, linestyle="-" if success else "--",
                label=f"{label}：{n}任务/{len(x)}调用")
        ax.set_xscale("log"); ax.set_ylim(0, 1.03); ax.grid(alpha=.18)
        ax.set_title(SCENARIOS[scene]); ax.set_xlabel("后续调用墙钟耗时（毫秒）")
        ax.set_ylabel("贡献任务等权累计比例"); ax.legend(fontsize=7)
    fig.suptitle("返回失败的调用也必须计时：先任务内归一，再等权贡献任务\n同一任务的上千次重试不冒充上千个独立任务；成功返回不代表最终到达", fontsize=10)
    save(fig, output, "P2_return_timing")

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), layout="constrained")
    for qualified, color, label in ((True, "#3985B8", "任务合格"), (False, "#D47A65", "任务未合格")):
        selected = [t for t in tasks if (t["status"] == "passed") == qualified and t["planning_sum_ms"] is not None]
        axes[0].scatter([t["calls"] for t in selected], [t["planning_sum_ms"]/1000 for t in selected],
            color=color, alpha=.65, s=14, label=f"{label} n={len(selected)}")
        selected = [t for t in selected if t["expected_stop"] is False and t["calls"] and t["goal_error_m"] is not None]
        axes[1].scatter([100*t["returned_success_calls"]/t["calls"] for t in selected],
            [t["goal_error_m"] for t in selected], color=color, alpha=.65, s=14, label=f"{label} n={len(selected)}")
    axes[0].set(xscale="log", yscale="log", xlabel="每任务规划调用数（对数轴）",
        ylabel="每任务累计规划墙钟（秒，对数轴）", title="重试开销：不是任务行驶时长")
    axes[1].set(xlabel="规划调用返回成功比例（%）", ylabel="最终距目标距离（米）", title="仅普通到达任务：返回成功不等于到达")
    for ax in axes: ax.grid(alpha=.15); ax.legend(fontsize=7)
    fig.suptitle("耗时与有效性联合诊断：不能只统计成功调用或删除未到达任务", fontsize=10)
    save(fig, output, "P3_retry_effort")

    lines = []
    fmt = lambda value: "--" if value is None else f"{value:.2f}"
    for scene, s in summary.items():
        lines.append(f"{SCENARIOS[scene]} & {s['first_call_ms']['n']} & {fmt(s['first_call_ms']['median'])} & " +
            f"{fmt(s['first_call_ms']['p95'])} & {s['subsequent_case_p95_ms']['n']} & " +
            f"{fmt(s['subsequent_case_p95_ms']['median'])} & {fmt(s['control_saved_case_p95_ms']['median'])} " + r"\\")
    (output/"planning_timing_rows.tex").write_text("\n".join(lines)+"\n"+r"\bottomrule"+"\n", encoding="utf-8")
    (output/"planning_timing_counts.tex").write_text(
        rf"\newcommand{{\TimingTaskCount}}{{{len(tasks)}}}"+"\n"+
        rf"\newcommand{{\TimingCallCount}}{{{len(calls)}}}"+"\n", encoding="utf-8")
    result = {"tasks": len(tasks), "calls": len(calls), "phases": dict(Counter(c["phase"] for c in calls)),
        "scenario": summary, "conditional_ecdf_counts": distribution_counts,
        "reanalysis_only": True, "new_algorithm_runs": 0,
        "search_substage_budget_ms": plan["parameters"].get("max_search_time_ms"),
        "quantile_policy": "raw subsequent calls: NumPy linear P95; saved native planner summaries audited with floor index; saved control P95 retains native floor index",
        "limitations": ["sequential first-order model, not DDS/PhysX concurrency", "control raw samples unavailable; saved P95 cannot be independently recomputed",
            "first call not time to first feasible committed path", "subsequent includes failed retries, not only replans after a valid path",
            "return success does not independently certify map freshness or executed path", "no measured goal/event to command wall-clock response"],
        "hashes": {k: digest(paths[k]) for k in ("new_navigation_plan", "new_navigation")}}
    atomic_json(output/"planning_timing_results.json", result)
    write_csv(output/"planning_timing_tasks.csv", tasks); write_csv(output/"planning_timing_calls.csv", calls)
    return result


def plot_workload_bars(data, output):
    record = data["new_scaling"]["runs"]
    sizes = sorted({r["size"] for r in record}); yaws = sorted({r["yaw_bins"] for r in record})
    fig = plt.figure(figsize=(8.8, 4.1), layout="none")
    for column, backend, color in ((0, "cpu", "#438CC4"), (1, "cupy", "#40A69D")):
        ax = fig.add_axes([.01+.51*column, .19, .43, .63], projection="3d")
        for j, yaw in enumerate(yaws):
            heights = []
            for size in sizes:
                values = [r["elapsed_ms"] for r in record if r["backend"] == backend and r["size"] == size and r["yaw_bins"] == yaw]
                if not values: raise ValueError("empty workload cell")
                heights.append(float(np.median(values)))
            ax.bar3d(np.arange(len(sizes))-.25, np.full(len(sizes), j)-.24, np.zeros(len(sizes)),
                .5, .48, heights, color=color, shade=False, edgecolor="white", linewidth=.6)
            for i, height in enumerate(heights): ax.text(i, j, height, f"{height:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(sizes)), [str(s) for s in sizes]); ax.set_yticks(range(len(yaws)), [str(y) for y in yaws])
        ax.set_xlabel("地图侧长（格）", labelpad=3, fontsize=9); ax.set_ylabel("航向数", labelpad=3, fontsize=9)
        ax.text2D(.0, .96, "耗时中位数（毫秒）", transform=ax.transAxes, fontsize=9)
        ax.set_title(("CPU" if backend == "cpu" else "GPU / CuPy")+"：独立纵轴", fontsize=10)
        ax.set_zlim(bottom=0); ax.set_box_aspect((1.4, 1., 1.1)); ax.view_init(elev=25, azim=-58)
    fig.suptitle("计算负载与地形组件耗时：地图规模 × 航向数，而非车辆期望速度\n每柱暖机后50次测量；两图纵轴刻度不同，不按柱子相对高度比较后端", fontsize=10)
    save(fig, output, "P4_workload_3d")
