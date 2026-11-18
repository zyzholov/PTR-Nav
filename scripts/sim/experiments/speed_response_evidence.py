"""Supplementary speed stress study, not a planning computation benchmark."""

from collections import Counter, defaultdict
import csv
import json

import numpy as np

from plot_paper_scale import plt, save
from plot_expanded_campaign import SCENARIOS
from plot_style import wilson_interval
from pointcloud_batch import ROOT, atomic_json, digest


def validate_speed_campaign(record, protocol, path):
    if record["protocol"] != protocol or record["protocol_sha256"] != digest(path):
        raise ValueError("speed protocol changed")
    if digest(record["base_plan_path"]) != record["base_plan_sha256"] or digest(record["binary_path"]) != record["binary_sha256"]:
        raise ValueError("speed base plan/binary changed")
    rows = record["runs"]
    if len(rows) != protocol["planned_cases"] or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("speed campaign is partial or duplicated")
    expected = Counter({(scene, speed): protocol["per_scenario"] for scene in protocol["scenarios"] for speed in protocol["desired_speed_mps"]})
    if Counter((r["scenario"], r["desired_speed_mps"]) for r in rows) != expected:
        raise ValueError("speed cells have different trial denominators")
    groups = defaultdict(list)
    for r in rows: groups[r["paired_id"]].append(r["desired_speed_mps"])
    if len(groups) != protocol["paired_base_tasks"] or any(sorted(s) != protocol["desired_speed_mps"] for s in groups.values()):
        raise ValueError("missing speed pair")


def task_tracking_value(row):
    # Native quantile returns zero for no samples; a run with no successful path has no tracking measurement.
    if row["status"] == "error" or not any(p["success"] and p["path_points"] > 0 for p in row.get("planning", [])):
        return None
    return row.get("tracking_p95_m")


def plot_speed_campaign(data, paths, output):
    record, protocol = data["speed_campaign"], data["speed_protocol"]
    validate_speed_campaign(record, protocol, paths["speed_protocol"])
    cases = {}; manifest_root = paths["speed_campaign"].parent
    for entry in record["plans"]:
        path = manifest_root/entry["path"]
        if digest(path) != entry["sha256"]: raise ValueError("paired speed plan changed")
        plan = json.loads(path.read_text())
        for name, expected in plan["settings"]["source_sha256"].items():
            if digest(ROOT/name) != expected: raise ValueError("speed method source changed")
        for c in plan["cases"]:
            if digest(c["map_file"]) != c["map_sha256"]: raise ValueError("paired map changed")
            cases[c["id"]] = c
    scenes, speeds = protocol["scenarios"], protocol["desired_speed_mps"]
    cells = {}; rows = []
    for r in record["runs"]:
        c = cases[r["id"]]
        if any(r[key] != c[key] for key in ("scenario", "paired_id", "desired_speed_mps")):
            raise ValueError("speed result is mislabeled against its frozen plan")
        if r["status"] not in ("passed", "failed", "error"):
            raise ValueError("unrecognized speed result status")
        trajectory = np.asarray(r.get("trajectory", []), dtype=float)
        measured = np.abs(trajectory[:, 4]) if len(trajectory) else np.array([])
        event = c.get("event_s")
        eligible = False
        if event is not None and len(trajectory):
            before = trajectory[trajectory[:, 0] < event]
            eligible = bool(len(before) and event-before[-1, 0] <= .3 and abs(before[-1, 4]) > .05 and abs(before[-1, 6]) > .05)
        stop = r.get("event_stop_response_s", -1)
        rows.append({"id": r["id"], "paired_id": r["paired_id"], "scenario": r["scenario"],
            "desired_speed_mps": r["desired_speed_mps"], "status": r["status"], "qualified": r["status"] == "passed",
            "collision": r.get("collision"), "terrain_violation": r.get("unsafe_terrain"),
            "tracking_p95_m": task_tracking_value(r), "planner_p95_ms": r.get("planner_p95_ms"),
            "planner_calls": len(r.get("planning", [])),
            "goal_error_m": r.get("goal_error_m"), "sampled_speed_p95_mps": float(np.quantile(measured, .95)) if len(measured) else None,
            "event_response_evaluable": eligible, "eligible_stop_response_s": stop if eligible and stop >= 0 else None})
    for scene in scenes:
        cells[scene] = {}
        for speed in speeds:
            selected = [r for r in rows if r["scenario"] == scene and r["desired_speed_mps"] == speed]
            passed = sum(r["qualified"] for r in selected)
            tracking = [r["tracking_p95_m"] for r in selected if r["tracking_p95_m"] is not None]
            sampled = [r["sampled_speed_p95_mps"] for r in selected if r["sampled_speed_p95_mps"] is not None]
            response = [r["eligible_stop_response_s"] for r in selected if r["eligible_stop_response_s"] is not None]
            planning = [r["planner_p95_ms"] for r in selected if r["planner_calls"] and r["planner_p95_ms"] is not None]
            cells[scene][str(speed)] = {"n": len(selected), "passed": passed, "qualified_rate": passed/len(selected),
                "statuses": dict(Counter(r["status"] for r in selected)),
                "interval": wilson_interval(passed, len(selected)), "tracking_evaluable_cases": len(tracking),
                "tracking_case_p95_median_m": float(np.median(tracking)) if tracking else None,
                "sampled_speed_case_p95_median_mps": float(np.median(sampled)) if sampled else None,
                "planner_case_p95_median_ms": float(np.median(planning)) if planning else None,
                "event_moving_cases": sum(r["event_response_evaluable"] for r in selected),
                "event_stop_observed_cases": len(response),
                "command_stop_response_median_sim_s": float(np.median(response)) if response else None,
                "command_stop_response_p95_sim_s": float(np.quantile(response, .95)) if response else None,
                "collisions": sum(bool(r["collision"]) for r in selected),
                "collision_evaluable_cases": sum(r["collision"] is not None for r in selected),
                "terrain_evaluable_cases": sum(r["terrain_violation"] is not None for r in selected),
                "terrain_violations": sum(bool(r["terrain_violation"]) for r in selected)}
    fig, axes = plt.subplots(2, 3, figsize=(8.5, 5.8), layout="constrained")
    for ax, scene in zip(axes.flat, scenes):
        selected = [cells[scene][str(speed)] for speed in speeds]
        y = np.array([s["qualified_rate"] for s in selected])
        error = np.array([[v-s["interval"][0] for v, s in zip(y, selected)],
            [s["interval"][1]-v for v, s in zip(y, selected)]])
        ax.errorbar(speeds, y*100, yerr=error*100, fmt="o--", color="#438CC4", capsize=3, lw=1.2)
        for speed, value, s in zip(speeds, y, selected):
            ax.annotate(f"{s['passed']}/{s['n']}", (speed, value*100), xytext=(0, 6),
                textcoords="offset points", ha="center", fontsize=8)
        ax.set_ylim(0, 116); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_xticks(speeds)
        ax.set_title(SCENARIOS[scene]); ax.set_xlabel("期望速度（m/s）"); ax.set_ylabel("任务合格率（%）")
        ax.grid(axis="y", alpha=.15)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(.05, .85, "附录：执行速度压力测试\n不是规划计算效率测试\n\n误差条：各格Wilson区间\n同一任务跨速度为配对重复\n\n100%饱和并不证明无速度影响\n不能靠截断纵轴制造差异", va="top", fontsize=9)
    fig.suptitle("配对速度任务：二维分面避免遮挡，饱和场景如实展示\n固定档位执行顺序；连线不是连续速度模型或无偏因果结论", fontsize=10)
    save(fig, output, "C4_speed_rates")
    fig, axes = plt.subplots(2, 3, figsize=(8.5, 6.0), layout="constrained")
    rng = np.random.default_rng(261014704)
    for ax, scene in zip(axes.flat, scenes):
        groups = [[r["tracking_p95_m"] for r in rows if r["scenario"] == scene and r["desired_speed_mps"] == speed and r["tracking_p95_m"] is not None] for speed in speeds]
        ax.boxplot([v or [np.nan] for v in groups], widths=.5, medianprops={"color": "#24788e"}, flierprops={"marker": ".", "markersize": 3})
        for i, speed in enumerate(speeds, 1):
            failed = [r["tracking_p95_m"] for r in rows if r["scenario"] == scene and r["desired_speed_mps"] == speed and not r["qualified"] and r["tracking_p95_m"] is not None]
            if failed: ax.scatter(i+rng.uniform(-.1, .1, len(failed)), failed, color="#bc4b46", s=12, zorder=3)
        ax.set_xticks(range(1, len(speeds)+1), [f"{s}\nn={len(v)}" for s, v in zip(speeds, groups)], fontsize=8)
        ax.set_title(SCENARIOS[scene]); ax.set_xlabel("期望巡航速度（m/s）"); ax.set_ylabel("每任务跟踪 P95（米）"); ax.grid(axis="y", alpha=.18)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(.05, .85, "箱体：25%至75%分位\n中线：中位数\n须：1.5倍四分位距范围内极值\n红点：未合格任务\n\n未产生有效路径的任务\n不以零跟踪误差代替缺测\n低误差但停滞仍不合格", va="top", fontsize=9)
    fig.suptitle("配对速度任务的跟踪误差箱线分布\n一个值是一个任务的P95，不是一个轨迹点；全部可评价失败保留", fontsize=10)
    save(fig, output, "C4_tracking_boxes")
    lines = []
    for scene in scenes:
        values = [cells[scene][str(speed)]["sampled_speed_case_p95_median_mps"] for speed in speeds]
        lines.append(SCENARIOS[scene]+" & "+" & ".join("--" if v is None else f"{v:.3f}" for v in values)+" " + r"\\")
    (output/"C4_speed_rows.tex").write_text("\n".join(lines)+"\n"+r"\bottomrule"+"\n", encoding="utf-8")
    lines = []
    for speed in speeds:
        c = cells["emerge"][str(speed)]
        fmt = lambda k: "--" if c[k] is None else f"{c[k]:.3f}"
        lines.append(f"{speed} & {c['event_moving_cases']} & {c['event_stop_observed_cases']} & " +
            fmt("command_stop_response_median_sim_s")+" & "+fmt("command_stop_response_p95_sim_s")+" " + r"\\")
    (output/"C4_response_rows.tex").write_text("\n".join(lines)+"\n"+r"\bottomrule"+"\n", encoding="utf-8")
    totals = [sum(cells[scene][str(speed)]["passed"] for scene in scenes) for speed in speeds]
    emerge = [cells["emerge"][str(speed)]["passed"] for speed in speeds]
    (output/"C4_findings.tex").write_text("本批次各速度档的合格数依次为"+"、".join(str(v) for v in totals)+
        "（各100次）；突现障碍到达数依次为"+"、".join(str(v) for v in emerge)+
        "（各20次）。零碰撞记录不能抵消未到达失败；该配对批次的速度趋势也不能当作真实车辆的无偏因果结论。\n", encoding="utf-8")
    summary = {"study_id": protocol["study_id"], "cases": len(rows), "paired_base_tasks": len({r["paired_id"] for r in rows}),
        "cells": cells, "qualified": sum(r["qualified"] for r in rows), "new_physical_cases": 0,
        "speed_protocol_sha256": digest(paths["speed_protocol"]), "manifest_sha256": digest(paths["speed_campaign"]),
        "limitations": ["fixed speed order, shared hardware; not unbiased speed causality", "about 5Hz recorded trajectory speed, not full-rate speed extremes", "not Isaac/PhysX", "sample counts for native tracking are unavailable; no-path tracking omitted"]}
    atomic_json(output/"speed_response_results.json", summary)
    with (output/"speed_response_cases.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, rows[0].keys()); w.writeheader(); w.writerows(rows)
    return summary
