#!/usr/bin/env python3
"""Aggregate an experiment_matrix manifest and draw publication-ready plots."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
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
    pass_rates = [statistics["groups"][label]["pass_rate"] for label in labels]
    clear_keys = [("laser_clear_s", "FAST-LIO /Laser_map"),
                  ("risk_clear_s", "terrain + FSGP risk"),
                  ("live_clear_s", "planner live layer")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    x = np.arange(len(labels))
    axes[0, 0].bar(x, pass_rates, color="#36b37e")
    axes[0, 0].set_title("Pass rate")
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].set_ylabel("fraction")
    axes[0, 0].set_xticks(x, labels, rotation=35, ha="right")

    for key, title in clear_keys:
        for index, label in enumerate(labels):
            values = finite(groups[label], "ghost_laser_obstacle_clear_delay_s" if key == "laser_clear_s"
                            else "ghost_risk_clear_delay_s" if key == "risk_clear_s"
                            else "ghost_live_clear_delay_s")
            if values:
                axes[0, 1].scatter([index] * len(values), values, label=title if index == 0 else None,
                                   alpha=0.8, s=42)
    axes[0, 1].set_title("Dynamic residual clear delay")
    axes[0, 1].set_ylabel("seconds, lower is better")
    axes[0, 1].set_xticks(x, labels, rotation=35, ha="right")
    axes[0, 1].legend(fontsize=8)

    for index, label in enumerate(labels):
        planner = finite(groups[label], "planner_p95_ms")
        control = finite(groups[label], "control_p95_ms")
        if planner:
            axes[1, 0].scatter([index - 0.12] * len(planner), planner, color="#f4a261", label="planner p95" if index == 0 else None)
        if control:
            axes[1, 0].scatter([index + 0.12] * len(control), control, color="#457b9d", label="control p95" if index == 0 else None)
    axes[1, 0].set_title("Latency")
    axes[1, 0].set_ylabel("milliseconds")
    axes[1, 0].set_xticks(x, labels, rotation=35, ha="right")
    axes[1, 0].legend(fontsize=8)

    for index, label in enumerate(labels):
        values = finite(groups[label], "lidar_mean_points")
        if values:
            axes[1, 1].scatter([index] * len(values), values, color="#6c5ce7", s=42)
    axes[1, 1].set_title("LiDAR density")
    axes[1, 1].set_ylabel("mean points / scan")
    axes[1, 1].set_xticks(x, labels, rotation=35, ha="right")
    fig.suptitle(f"{data.get('method', 'method')} | {data.get('study_type', 'experiment')}", fontsize=14)
    fig.savefig(output_dir / "experiment_results.png", dpi=160)
    print(f"wrote {output_dir / 'results.csv'}")
    print(f"wrote {output_dir / 'statistics.json'}")
    print(f"wrote {output_dir / 'experiment_results.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
