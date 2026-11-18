#!/usr/bin/env python3
"""Aggregate an experiment_matrix manifest and draw publication-ready plots."""

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from plot_style import configure_chinese, scenario_name


def finite(rows, key):
    return [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]


def summary(values):
    if not values:
        return {"n": 0, "mean": None, "median": None, "p95": None, "std": None}
    values = np.asarray(values, dtype=float)
    return {"n": int(values.size), "mean": round(float(values.mean()), 4),
            "median": round(float(np.median(values)), 4),
            "p95": round(float(np.percentile(values, 95)), 4),
            "std": round(float(values.std(ddof=1)), 4) if values.size > 1 else 0.0}


def display_name(label):
    return scenario_name(label.split("::", 1)[-1])


def format_number(value, digits=2):
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def format_range(rows, key, suffix="", digits=2):
    values = finite(rows, key)
    if not values:
        return "n/a"
    low, high = min(values), max(values)
    if abs(low - high) < 1e-9:
        return f"{format_number(low, digits)}{suffix}"
    return f"{format_number(low, digits)}--{format_number(high, digits)}{suffix}"


def report_text(data, rows, statistics, figure_link):
    passed = sum(bool(row.get("passed")) for row in rows)
    reached = sum(bool(row.get("reached")) for row in rows if "reached" in row)
    labels = sorted({f"{row.get('ablation', 'unknown')}::{row.get('scenario', 'unknown')}"
                     for row in rows})
    lines = [
        "# Experiment Report: Our Method Baseline",
        "",
        f"- Study type: `{data.get('study_type', 'our_method_baseline')}`",
        f"- Method: `{data.get('method', 'unknown')}`",
        f"- Created: `{data.get('created_at', 'unknown')}`",
        f"- Runs: `{len(rows)}`; pipeline gates passed: `{passed}/{len(rows)}`",
        f"- Goal reached in rows with a goal: `{reached}`",
        "",
        "## Scope",
        "",
        "This report records the current integrated method only. It is a baseline "
        "record for later external comparisons and does not claim superiority over "
        "another algorithm. The run set is small and is not a publication-sized "
        "statistical sample.",
        "",
        "## Result Figure",
        "",
        f"![Our method baseline results]({figure_link})",
        "",
        "## Scenario Results",
        "",
        "| Scenario | Runs | Gate pass | Reached | Truth error (m) | Min separation (m) | Planner p95 (ms) | Risk clear (s) | Live clear (s) | Interpretation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for label in labels:
        group_rows = [row for row in rows
                      if f"{row.get('ablation', 'unknown')}::{row.get('scenario', 'unknown')}" == label]
        passed_group = sum(bool(row.get("passed")) for row in group_rows)
        reached_group = sum(bool(row.get("reached")) for row in group_rows if "reached" in row)
        scenario = label.split("::", 1)[-1]
        if scenario == "clearance":
            interpretation = "风险层和 live 层快速清除；历史 /Laser_map 在窗口内仍有残留"
        elif scenario == "detour":
            interpretation = "未到达目标，需修复静态绕障规划"
        elif scenario == "step":
            interpretation = "安全停止；不是到达目标"
        elif scenario == "mars":
            interpretation = "仅验证传感器与地形更新吞吐"
        else:
            interpretation = "流程通过"
        lines.append(
            f"| `{scenario}` | {len(group_rows)} | {passed_group}/{len(group_rows)} | "
            f"{reached_group} | {format_range(group_rows, 'truth_error_m', ' m')} | "
            f"{format_range(group_rows, 'physical_obstacle_min_m', ' m')} | "
            f"{format_range(group_rows, 'planner_p95_ms', ' ms')} | "
            f"{format_range(group_rows, 'ghost_risk_clear_delay_s', ' s')} | "
            f"{format_range(group_rows, 'ghost_live_clear_delay_s', ' s')} | {interpretation} |"
        )

    lines.extend([
        "",
        "## Throughput and Latency",
        "",
        "The observed terrain update rate is approximately 10 Hz and the FSGP "
        "update rate is approximately 5 Hz in the runs that collected these "
        "measurements. Planner p95 is scenario dependent. The `step` row has a "
        "control p95 of 89.96 ms and should be repeated before making a real-time "
        "claim. LiDAR point-count metrics are absent from this first matrix, so "
        "this report does not draw a conclusion about simulated point density.",
        "",
        "## Findings Requiring Follow-up",
        "",
        "1. The `detour` scenario failed to reach its goal with a final truth "
        "error of 5.691 m. This is the highest-priority navigation issue.",
        "2. In both `clearance` runs the physical separation remained above 2.157 m "
        "and the vehicle reached the goal, but the cumulative `/Laser_map` did not "
        "clear its historical obstacle points within the observation window.",
        "3. Risk and live obstacle clearing were fast in `clearance`: risk "
        "0.27--0.29 s and live layer 0.17--0.19 s.",
        "4. Repeat the `step` case and collect LiDAR/IMU rates before expanding "
        "to ablations or external method comparisons.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "python3 scripts/sim/experiments/analyze_matrix.py \\",
        "  log/experiment_matrix_20261010_112130 \\",
        "  --figure-output docs/figures/our_method_baseline_20261010.png \\",
        "  --report-output docs/EXPERIMENT_REPORT_20261010.md",
        "```",
        "",
        "The machine-readable source files are `manifest.json`, `results.csv` and "
        "`statistics.json` in the experiment output directory.",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--figure-output", type=Path, default=None,
                        help="optional additional path for the PNG figure")
    parser.add_argument("--report-output", type=Path, default=None,
                        help="optional Markdown report path")
    args = parser.parse_args()
    configure_chinese()
    manifest_path = args.manifest / "manifest.json" if args.manifest.is_dir() else args.manifest
    data = json.loads(manifest_path.read_text())
    rows = data.get("runs", [])
    output_dir = manifest_path.parent

    keys = sorted({key for row in rows for key in row})
    with (output_dir / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    groups = {}
    for row in rows:
        group = f"{row.get('ablation', 'unknown')}::{row.get('scenario', 'unknown')}"
        groups[group] = [item for item in rows
                         if f"{item.get('ablation', 'unknown')}::{item.get('scenario', 'unknown')}" == group]
    statistics = {
        "manifest": str(manifest_path),
        "study_type": data.get("study_type", "our_method_baseline"),
        "method": data.get("method", "FAST-LIO2 + GPU terrain analyzer + FSGP-BGK/SEB + terrain_nav"),
        "groups": {},
    }
    for group, items in groups.items():
        statistics["groups"][group] = {
            "runs": len(items),
            "pass_rate": round(sum(bool(item.get("passed")) for item in items) / len(items), 4),
            "truth_error_m": summary(finite(items, "truth_error_m")),
            "physical_obstacle_min_m": summary(finite(items, "physical_obstacle_min_m")),
            "laser_clear_s": summary(finite(items, "ghost_laser_obstacle_clear_delay_s")),
            "risk_clear_s": summary(finite(items, "ghost_risk_clear_delay_s")),
            "live_clear_s": summary(finite(items, "ghost_live_clear_delay_s")),
            "planner_p95_ms": summary(finite(items, "planner_p95_ms")),
            "control_p95_ms": summary(finite(items, "control_p95_ms")),
            "lidar_mean_points": summary(finite(items, "lidar_mean_points")),
            "terrain_fresh_hz": summary(finite(items, "terrain_fresh_hz")),
            "fsgp_fresh_hz": summary(finite(items, "fsgp_fresh_hz")),
        }
    (output_dir / "statistics.json").write_text(json.dumps(statistics, indent=2) + "\n")

    labels = sorted(groups)
    names = [display_name(label) for label in labels]
    x = np.arange(len(labels))
    pass_rates = [statistics["groups"][label]["pass_rate"] for label in labels]
    reached_rates = [sum(bool(row.get("reached")) for row in groups[label]
                         if "reached" in row) / max(1, sum("reached" in row for row in groups[label]))
                     for label in labels]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    axes[0, 0].bar(x - 0.18, pass_rates, width=0.36, color="#2a9d8f", label="链路门限通过")
    axes[0, 0].bar(x + 0.18, reached_rates, width=0.36, color="#457b9d", label="目标到达")
    axes[0, 0].set_title("运行结果")
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].set_ylabel("比例")
    axes[0, 0].set_xticks(x, names, rotation=30, ha="right")
    axes[0, 0].legend(fontsize=8)

    for index, label in enumerate(labels):
        for key, color, title in (("ghost_risk_clear_delay_s", "#e76f51", "风险层"),
                                   ("ghost_live_clear_delay_s", "#f4a261", "实时障碍层")):
            values = finite(groups[label], key)
            if values:
                axes[0, 1].scatter([index] * len(values), values, color=color,
                                   label=title if index == 0 else None, s=48)
    axes[0, 1].set_title("动态残影清除延迟")
    axes[0, 1].set_ylabel("时间（秒）")
    axes[0, 1].set_xticks(x, names, rotation=30, ha="right")
    if axes[0, 1].get_legend_handles_labels()[0]:
        axes[0, 1].legend(fontsize=8)

    for index, label in enumerate(labels):
        planner = finite(groups[label], "planner_p95_ms")
        control = finite(groups[label], "control_p95_ms")
        if planner:
            axes[1, 0].scatter([index - 0.12] * len(planner), planner, color="#f4a261", label="规划 P95" if index == 0 else None)
        if control:
            axes[1, 0].scatter([index + 0.12] * len(control), control, color="#457b9d", label="控制 P95" if index == 0 else None)
    axes[1, 0].set_title("规划与控制延迟")
    axes[1, 0].set_ylabel("延迟（毫秒）")
    axes[1, 0].set_xticks(x, names, rotation=30, ha="right")
    axes[1, 0].legend(fontsize=8)

    for index, label in enumerate(labels):
        for key, color, title in (("terrain_fresh_hz", "#2a9d8f", "地形前端"),
                                   ("fsgp_fresh_hz", "#6c5ce7", "BGK/SEB 组件")):
            values = finite(groups[label], key)
            if values:
                axes[1, 1].scatter([index] * len(values), values, color=color,
                                   label=title if index == 0 else None, s=48)
    axes[1, 1].set_title("地形链路新时间戳更新率")
    axes[1, 1].set_ylabel("Hz")
    axes[1, 1].set_xticks(x, names, rotation=30, ha="right")
    handles, legend_labels = axes[1, 1].get_legend_handles_labels()
    if handles:
        axes[1, 1].legend(fontsize=8)

    for index, label in enumerate(labels):
        for key, color, title in (("truth_error_m", "#e63946", "目标误差"),
                                   ("physical_obstacle_min_m", "#1d3557", "最小中心距离")):
            values = finite(groups[label], key)
            if values:
                axes[0, 2].scatter([index] * len(values), values, color=color,
                                   label=title if index == 0 else None, s=48)
    axes[0, 2].set_title("位姿与安全距离")
    axes[0, 2].set_ylabel("距离（米）")
    axes[0, 2].set_xticks(x, names, rotation=30, ha="right")
    axes[0, 2].legend(fontsize=8)

    for index, label in enumerate(labels):
        for key, color, title in (("ghost_laser_obstacle_peak_before_vacate", "#8d99ae", "历史障碍点峰值"),
                                   ("live_peak_cells", "#ff9f1c", "实时障碍格峰值")):
            values = finite(groups[label], key)
            if values:
                axes[1, 2].scatter([index] * len(values), values, color=color,
                                   label=title if index == 0 else None, s=48)
    axes[1, 2].set_title("动态障碍观测证据")
    axes[1, 2].set_ylabel("数量")
    axes[1, 2].set_xticks(x, names, rotation=30, ha="right")
    axes[1, 2].legend(fontsize=8)

    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    fig.suptitle("当前方法实验总览：链路、地形更新、规划控制与动态障碍", fontsize=15)
    figure_path = output_dir / "experiment_results.png"
    fig.savefig(figure_path, dpi=160)
    extra_figure = args.figure_output
    if extra_figure:
        extra_figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(extra_figure, dpi=160)
    report_path = args.report_output or output_dir / "experiment_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    figure_link = os.path.relpath(extra_figure or figure_path, report_path.parent)
    report_path.write_text(report_text(data, rows, statistics, figure_link) + "\n")
    print(f"wrote {output_dir / 'results.csv'}")
    print(f"wrote {output_dir / 'statistics.json'}")
    print(f"wrote {figure_path}")
    if extra_figure:
        print(f"wrote {extra_figure}")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
