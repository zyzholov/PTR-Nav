#!/usr/bin/env python3
"""Plot the measured navigation/control objects from a run_matrix manifest."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from plot_style import configure_chinese, scenario_name, wilson_interval


COLORS = {"ramp": "#2a9d8f", "detour": "#e76f51", "crossing": "#457b9d",
          "step": "#6c5ce7", "clearance": "#f4a261", "emerge": "#264653"}


def read_trials(manifest, refresh_metrics=False):
    data = json.loads(manifest.read_text())
    trials = []
    for row in data.get("runs", []):
        row = dict(row)
        if refresh_metrics and row.get("run_dir"):
            from run_matrix import parse_log_metrics
            row.update(parse_log_metrics(Path(row["run_dir"])))
        path = Path(row.get("run_dir") or ".")
        if not path.is_absolute():
            path = Path.cwd() / path
        raw_path = path / "navigation_data.json"
        raw = json.loads(raw_path.read_text()) if row.get("run_dir") and raw_path.exists() else {}
        summary = dict(raw.get("summary", {}))
        summary.setdefault("scenario", row.get("scenario", "unknown"))
        summary.setdefault("reached", bool(row.get("reached", False)))
        trials.append({"row": row, "raw": raw, "summary": summary})
    return data, trials


def safe(values):
    return np.asarray([float(v) for v in values if isinstance(v, (int, float))
                       and np.isfinite(v)], dtype=float)


def path_in_truth_frame(points, raw):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or not len(points):
        return points
    if not all(key in raw for key in ("map_start", "truth_start", "truth_yaw_start")):
        return points
    angle = raw["truth_yaw_start"] - raw["map_start"][3]
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return (points - np.asarray(raw["map_start"][:2])) @ rotation.T + raw["truth_start"][:2]


def save(fig, output, name):
    fig.savefig(output / (name + ".png"), dpi=200, bbox_inches="tight")
    fig.savefig(output / (name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_trajectories(output, trials, columns=None, legend_fontsize=8):
    scenarios = sorted({trial["summary"].get("scenario", "unknown") for trial in trials})
    columns = max(1, min(columns or len(scenarios), len(scenarios)))
    rows = int(np.ceil(len(scenarios) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4 * rows),
                             squeeze=False, constrained_layout=True)
    for ax, scenario in zip(axes.flat, scenarios):
        selected = [trial for trial in trials if trial["summary"].get("scenario") == scenario]
        color = COLORS.get(scenario, "#333333")
        if scenario == "detour":
            ax.add_patch(Circle((3.0, 0.0), 0.65, color="#636b70", alpha=0.35,
                               label="静态柱体（场景配置）"))
        elif scenario == "ramp":
            ax.axvspan(1.5, 5.5, color="#a8c7a4", alpha=0.18, label="缓坡段（场景配置）")
        elif scenario == "step":
            ax.axvspan(4.0, 10.0, color="#cc7067", alpha=0.15, label="0.50 米台阶（场景配置）")
            ax.axvline(4.0, color="#aa4848", linewidth=1)
        for index, trial in enumerate(selected, 1):
            truth = np.asarray(trial["raw"].get("truth", []), dtype=float)
            if truth.size:
                ax.plot(truth[:, 1], truth[:, 2], color=color, alpha=0.75,
                        label="真实轨迹" if index == 1 else None)
                ax.plot(truth[0, 1], truth[0, 2], "o", color=color, ms=4)
            for path_index, (_, points) in enumerate(trial["raw"].get("path_history", [])):
                points = path_in_truth_frame(points, trial["raw"])
                if points.ndim == 2 and len(points):
                    ax.plot(points[:, 0], points[:, 1], "--", color=color, alpha=0.12 + 0.35 * (path_index == 0),
                            label="规划路径历史" if index == 1 and path_index == 0 else None)
            goal = trial["raw"].get("truth_goal")
            if goal:
                ax.plot(goal[0], goal[1], "*", color="#d62828", ms=9,
                        label="目标点" if index == 1 else None)
            obstacle = np.asarray(trial["raw"].get("obstacle", []), dtype=float)
            if obstacle.ndim == 2 and len(obstacle) and scenario in ("crossing", "emerge", "clearance"):
                ax.plot(obstacle[:, 1], obstacle[:, 2], ":", color="#30373d", alpha=0.5,
                        label="障碍中心真实轨迹" if index == 1 else None)
        ax.set_title(f"{scenario_name(scenario)}（{len(selected)} 次）")
        ax.set_xlabel("世界 x（米）"); ax.set_ylabel("世界 y（米）")
        ax.set_aspect("equal", adjustable="datalim"); ax.grid(alpha=0.2)
        ax.legend(fontsize=legend_fontsize)
    for ax in axes.flat[len(scenarios):]:
        ax.set_visible(False)
    fig.suptitle("导航轨迹：真实轨迹、规划路径历史与目标点（启动位姿对齐）")
    save(fig, output, "01_navigation_trajectories")


def plot_performance(output, trials):
    scenarios = sorted({trial["summary"].get("scenario", "unknown") for trial in trials})
    x = np.arange(len(scenarios))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    metrics = (
        ("reached", "目标到达率及 95% 置信区间", lambda s: float(bool(s.get("reached"))), (0, 1.05)),
        ("truth_path_length_m", "执行路径长度（米）", lambda s: s.get("truth_path_length_m"), None),
        ("duration_s", "观测时长（仿真秒，含安全停机）", lambda s: s.get("duration_s"), None),
        ("avg_speed_mps", "平均速度（米/秒）", lambda s: s.get("avg_speed_mps"), None),
        ("tracking_error_p95_m", "跟踪误差 P95（米）", lambda s: s.get("tracking_error_p95_m"), None),
        ("max_curvature_1pm", "首段局部规划路径最大曲率（1/米）", lambda s: s.get("max_curvature_1pm"), None),
    )
    for ax, (key, title, getter, ylim) in zip(axes.flat, metrics):
        values = []
        for scenario in scenarios:
            values.append([getter(trial["summary"]) for trial in trials
                           if trial["summary"].get("scenario") == scenario and getter(trial["summary"]) is not None])
        if key == "reached":
            means = [sum(v) / len(v) if v else np.nan for v in values]
            intervals = [wilson_interval(sum(v), len(v)) for v in values]
            ax.bar(x + 1, means, color="#247a91",
                   yerr=[[p - lo for p, (lo, _) in zip(means, intervals)],
                         [hi - p for p, (_, hi) in zip(means, intervals)]], capsize=4)
        else:
            ax.boxplot([v or [np.nan] for v in values], showmeans=True)
        ax.set_xticks(x + 1, [scenario_name(s) for s in scenarios], rotation=35, ha="right")
        ax.set_title(title); ax.grid(axis="y", alpha=0.2)
        if ylim: ax.set_ylim(*ylim)
    fig.suptitle("规划控制统计：所有运行计入到达率，台阶安全停止不算到达")
    save(fig, output, "02_navigation_performance")


def plot_safety(output, trials):
    scenarios = sorted({trial["summary"].get("scenario", "unknown") for trial in trials})
    x = np.arange(len(scenarios))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for ax, key, title in ((axes[0], "max_abs_pitch_rad", "最大俯仰角绝对值（弧度）"),
                           (axes[1], "max_abs_roll_rad", "最大横滚角绝对值（弧度）"),
                           (axes[2], "physical_obstacle_min_m", "与动态障碍中心最小距离（米）")):
        values = [[trial["summary"].get(key) for trial in trials
                   if trial["summary"].get("scenario") == scenario and trial["summary"].get(key) is not None]
                  for scenario in scenarios]
        ax.boxplot([v or [np.nan] for v in values], showmeans=True)
        ax.set_xticks(x + 1, [scenario_name(s) for s in scenarios], rotation=35, ha="right")
        for i, v in enumerate(values, 1):
            if not v:
                ax.text(i, 0.04, "未采集", transform=ax.get_xaxis_transform(), ha="center", fontsize=9)
        ax.set_title(title); ax.grid(axis="y", alpha=0.2)
    fig.suptitle("安全与车体姿态：中心距离不是碰撞接触次数")
    save(fig, output, "03_navigation_safety")


def plot_latency(output, trials):
    scenarios = sorted({trial["summary"].get("scenario", "unknown") for trial in trials})
    x = np.arange(len(scenarios))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    legend_names = {"planner_mean_ms": "规划均值", "planner_p95_ms": "规划 P95",
                    "control_mean_ms": "控制均值", "control_p95_ms": "控制 P95",
                    "terrain_fresh_hz": "地形前端更新率", "fsgp_fresh_hz": "BGK/SEB 更新率"}
    for ax, keys, title in ((axes[0], ("planner_mean_ms", "planner_p95_ms"), "规划计算耗时（毫秒）"),
                            (axes[1], ("control_mean_ms", "control_p95_ms"), "控制计算耗时（毫秒）"),
                            (axes[2], ("terrain_fresh_hz", "fsgp_fresh_hz"), "新时间戳更新率（Hz）")):
        for offset, key in zip((-0.18, 0.18), keys):
            values = []
            for scenario in scenarios:
                values.append([trial["row"].get(key) for trial in trials
                               if trial["summary"].get("scenario") == scenario and trial["row"].get(key) is not None])
            means = [np.mean(v) if v else np.nan for v in values]
            errors = [np.std(v, ddof=1) if len(v) > 1 else 0 for v in values]
            lower=[min(m,e) if np.isfinite(m) else 0 for m,e in zip(means,errors)]
            ax.bar(x + offset, means, width=0.32, yerr=[lower,errors], capsize=3,
                   color="#2a9d8f" if offset < 0 else "#e76f51",
                   label=legend_names[key])
        ax.set_ylim(bottom=0)
        for index,scenario in enumerate(scenarios):
            if not any(trial["row"].get(key) is not None for trial in trials
                       if trial["summary"].get("scenario")==scenario for key in keys):
                ax.text(index,.025,"未采集",transform=ax.get_xaxis_transform(),
                        rotation=90,ha="center",va="bottom",fontsize=8)
        ax.set_xticks(x, [scenario_name(s) for s in scenarios], rotation=35, ha="right")
        ax.set_title(title); ax.grid(axis="y", alpha=0.2); ax.legend(fontsize=8)
    fig.suptitle("规划、控制与地形更新性能：柱高为重复均值，误差线为标准差（下界截至0）")
    save(fig, output, "04_navigation_latency")


def plot_profiles(output, trials):
    scenarios = sorted({trial["summary"].get("scenario", "unknown") for trial in trials})
    fig, axes = plt.subplots(2, max(1, len(scenarios)), figsize=(5 * max(1, len(scenarios)), 7),
                             squeeze=False, constrained_layout=True)
    for column, scenario in enumerate(scenarios):
        trial = next(trial for trial in trials if trial["summary"].get("scenario") == scenario)
        raw = trial["raw"]
        truth = np.asarray(raw.get("truth", []), dtype=float)
        if len(truth) > 1:
            dt = np.diff(truth[:, 6]); ds = np.hypot(np.diff(truth[:, 1]), np.diff(truth[:, 2]))
            valid = dt > 1.0e-4
            axes[0, column].plot(truth[1:, 6] - truth[0, 6], np.divide(ds, dt, out=np.zeros_like(ds), where=valid),
                                 color=COLORS.get(scenario, "#333333"), label="真实速度")
            axes[1, column].plot(truth[:, 6] - truth[0, 6], truth[:, 4], label="俯仰角", color="#e76f51")
            axes[1, column].plot(truth[:, 6] - truth[0, 6], truth[:, 5], label="横滚角", color="#457b9d")
        controls = np.asarray(raw.get("controls", []), dtype=float)
        if len(controls):
            origin = truth[0, 6] if len(truth) else controls[0, 0]
            axes[0, column].plot(controls[:, 0] - origin, np.abs(controls[:, 1]),
                                 alpha=0.55, label="速度指令绝对值")
        axes[0, column].set_title(scenario_name(scenario)); axes[0, column].set_ylabel("速度（米/秒）")
        axes[1, column].set_xlabel("仿真时间（秒）"); axes[1, column].set_ylabel("车体姿态（弧度）")
        axes[0, column].grid(alpha=0.2); axes[1, column].grid(alpha=0.2)
        axes[0, column].legend(fontsize=8); axes[1, column].legend(fontsize=8)
    fig.suptitle("代表性闭环时序（每个场景首轮，仅作定性展示，非重复统计）")
    save(fig, output, "05_navigation_profiles")


def report(output, data, trials):
    scenarios = sorted({trial["summary"].get("scenario", "unknown") for trial in trials})
    lines = ["# 规划控制重复实验报告", "",
             "> 本轮仅记录当前方法，不作外部方法或消融对比。", "",
             f"- 总运行数：`{len(trials)}`；含轨迹记录：`{sum(bool(t['raw']) for t in trials)}`",
             "- 启动失败、缺失轨迹、未到达均计入分母；台阶安全停止与目标到达分开统计。",
             "- 成功率置信区间使用 Wilson 95% 区间；单帧/栅格点不是独立导航样本。",
             "", "## 实验图", "",
             "![trajectories](01_navigation_trajectories.png)",
             "![performance](02_navigation_performance.png)",
             "![safety](03_navigation_safety.png)",
             "![latency](04_navigation_latency.png)",
             "![profiles](05_navigation_profiles.png)",
             "", "## 到达率与性能", "",
             "| 场景 | 总次数 | 到达次数 | 到达率 95% 区间 | 执行路长均值（米） | 观测时长均值（秒） | 平均速度（米/秒） | 跟踪 P95 均值（米） | 规划 P95 均值（毫秒） |",
             "|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    statistics = {}
    for scenario in scenarios:
        selected = [t for t in trials if t["summary"].get("scenario") == scenario]
        summaries = [t["summary"] for t in selected]
        def mean(key):
            values = safe([s.get(key) for s in summaries])
            return f"{np.mean(values):.3f}" if len(values) else "n/a"
        planner = safe([t["row"].get("planner_p95_ms") for t in selected])
        reached = sum(bool(s.get("reached")) for s in summaries)
        low, high = wilson_interval(reached, len(selected))
        planner_text = f"{np.mean(planner):.1f}" if len(planner) else "未采集"
        lines.append(f"| {scenario_name(scenario)} | {len(selected)} | {reached} | "
                     f"[{low:.1%}, {high:.1%}] | {mean('truth_path_length_m')} | "
                     f"{mean('duration_s')} | {mean('avg_speed_mps')} | {mean('tracking_error_p95_m')} | {planner_text} |")
        statistics[scenario] = {"runs": len(selected), "reached": reached,
            "success_rate": reached / len(selected), "success_wilson95": [low, high],
            "gate_passed": sum(bool(t["row"].get("passed")) for t in selected),
            "metrics": {}}
        for key in ("truth_path_length_m", "duration_s", "avg_speed_mps", "tracking_error_p95_m",
                    "max_abs_pitch_rad", "max_abs_roll_rad", "physical_obstacle_min_m", "planner_p95_ms"):
            values = safe([t["summary"].get(key, t["row"].get(key)) for t in selected])
            successes = safe([t["summary"].get(key, t["row"].get(key)) for t in selected
                              if t["summary"].get("reached")])
            statistics[scenario]["metrics"][key] = {
                "observed_n": len(values), "missing_n": len(selected) - len(values),
                "mean": float(np.mean(values)) if len(values) else None,
                "median": float(np.median(values)) if len(values) else None,
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                "p95": float(np.percentile(values, 95)) if len(values) else None,
                "successful_n": len(successes),
                "successful_mean": float(np.mean(successes)) if len(successes) else None}
    lines.extend(["", "## 统计口径与限制", "",
                  "表中的性能均值包含已采集到数据的所有运行（包括停止和超时）；缺失值不是零。成功运行单独均值、标准差、中位数和缺失数存于 `navigation_statistics.json`。台阶观测时长是安全停机窗口，不是完成任务的时间。", "",
                  "动态障碍距离是车体参考点到障碍中心的距离，不是表面净距，也不是碰撞接触次数。累计点云残影需要独立清除实验，不能由到达率推断。首段路径往往是局部子目标，不能把它的路长比当作全程绕行比。", "",
                  "轨迹图按启动位姿将规划路径对齐到仿真真值坐标，只用于定性叠加，不是随时间修正的 SLAM 真值对齐。跟踪误差使用地图坐标中的估计里程计与当时规划路径。", ""])
    lines.extend(["背景柱体、缓坡和台阶是固定轻量测试场景的配置示意，不是推断得到的地图真值。动态障碍轨迹来自原始采集记录。其他地图场景不能套用这些示意。", ""])
    (output / "navigation_subject_report.md").write_text("\n".join(lines))
    (output / "navigation_statistics.json").write_text(json.dumps(statistics, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refresh-metrics", action="store_true", help="reparse logs without overwriting the raw manifest")
    args = parser.parse_args()
    configure_chinese()
    manifest = args.manifest / "manifest.json" if args.manifest.is_dir() else args.manifest
    data, trials = read_trials(manifest, args.refresh_metrics)
    args.output.mkdir(parents=True, exist_ok=True)
    if not trials:
        raise SystemExit("no navigation_data.json files found")
    plot_trajectories(args.output, trials)
    plot_performance(args.output, trials)
    plot_safety(args.output, trials)
    plot_latency(args.output, trials)
    plot_profiles(args.output, trials)
    report(args.output, data, trials)
    (args.output / "navigation_runs.json").write_text(json.dumps([t["row"] for t in trials], indent=2) + "\n")
    print(f"wrote {len(trials)} navigation trials to {args.output}")


if __name__ == "__main__":
    main()
