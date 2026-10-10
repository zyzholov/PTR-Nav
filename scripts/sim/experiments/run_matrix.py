#!/usr/bin/env python3
"""Run reproducible Isaac navigation experiments and retain per-run evidence."""

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[3]
SCENARIOS = {"clearance", "clearance_guard", "crossing", "detour", "emerge", "ramp", "step", "mars"}
ABLATIONS = {"full", "no_fastlio_dynamic_clear", "no_terrain_dynamic_clear", "no_live_layer", "cpu"}


def parse_case(value):
    scenario, separator, count = value.partition(":")
    if scenario not in SCENARIOS or not separator or not count.isdecimal() or int(count) < 1:
        raise argparse.ArgumentTypeError("expected SCENARIO:COUNT, e.g. clearance:3")
    return scenario, int(count)


def stop_group(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def find_run_dir(text):
    match = re.search(r"\[smoke\] (?:PASS|FAIL); logs=(\S+)", text)
    return Path(match.group(1)) if match else None


def read_json_line(path):
    if not path or not path.exists():
        return {}
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and "scenario" in value:
                return value
    return {}


def parse_log_metrics(run_dir):
    metrics = {}
    if not run_dir:
        return metrics

    trial = read_json_line(run_dir / ("map_guard_trial.log" if (run_dir / "map_guard_trial.log").exists()
                                      else "navigation_trial.log"))
    for key in (
        "reached", "truth_error_m", "physical_obstacle_min_m", "front_stop_commands",
        "risk_map_period_p95_s", "risk_map_age_p95_s", "ghost_laser_clear_delay_s",
        "ghost_laser_obstacle_clear_delay_s", "ghost_risk_clear_delay_s",
        "ghost_risk_known_clear_delay_s", "ghost_live_clear_delay_s",
        "ghost_laser_peak_before_vacate", "ghost_laser_obstacle_peak_before_vacate",
        "ghost_risk_peak_before_vacate", "static_laser_last", "live_peak_cells",
        "truth_lateral_max_m", "truth_pitch_max_rad", "truth_roll_max_rad",
    ):
        if key in trial:
            metrics[key] = trial[key]

    planning = (run_dir / "planning.log").read_text(errors="replace") if (run_dir / "planning.log").exists() else ""
    planner_ms = [float(value) for value in re.findall(r"planner: result=\w+ reason=.*? elapsed=([0-9.]+)ms", planning)]
    control_ms = [float(value) for value in re.findall(r"control timing: ([0-9.]+)ms", planning)]
    live_replans = len(re.findall(r"planner: result=\w+ reason=new live obstacle", planning))
    metrics["planner_count"] = len(planner_ms)
    metrics["planner_mean_ms"] = sum(planner_ms) / len(planner_ms) if planner_ms else None
    metrics["planner_p95_ms"] = percentile(planner_ms, 0.95)
    metrics["control_mean_ms"] = sum(control_ms) / len(control_ms) if control_ms else None
    metrics["control_p95_ms"] = percentile(control_ms, 0.95)
    metrics["live_replan_count"] = live_replans

    rates = (run_dir / "rates.log").read_text(errors="replace") if (run_dir / "rates.log").exists() else ""
    for topic, name in (("/livox/lidar", "lidar"), ("/imu/data_raw", "imu"), ("/livox/imu", "livox_imu")):
        section_match = re.search(
            r"\[hunter_isaac_rates\] Measuring " + re.escape(topic) +
            r".*?(?=\n\[hunter_isaac_rates\] Measuring |\Z)", rates, re.S)
        block = section_match.group(0) if section_match else ""
        for field, suffix in (("average wall rate", "wall_hz"), ("average stamp rate", "stamp_hz"),
                              ("point count min/mean/max", "points")):
            match = re.search(re.escape(field) + r": ([0-9.]+)(?:/([0-9.]+)/([0-9.]+))?", block)
            if match:
                if suffix == "points":
                    metrics[f"{name}_mean_points"] = float(match.group(2) or match.group(1))
                else:
                    metrics[f"{name}_{suffix}"] = float(match.group(1))

    map_rates = (run_dir / "map_rates.log").read_text(errors="replace") if (run_dir / "map_rates.log").exists() else ""
    for topic, name in (("terrain_analyzer/fused_map", "terrain"), ("fsgp_bgk/fused_map", "fsgp")):
        match = re.search(re.escape(topic) + r": samples=\d+ wall_rate=([0-9.]+)Hz fresh_stamps=\d+ fresh_rate=([0-9.]+)Hz", map_rates)
        if match:
            metrics[f"{name}_wall_hz"] = float(match.group(1))
            metrics[f"{name}_fresh_hz"] = float(match.group(2))
    return metrics


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    index = min(len(values) - 1, max(0, int(round(fraction * (len(values) - 1)))))
    return round(values[index], 4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", type=parse_case, required=True,
                        help="SCENARIO:COUNT; repeat this option for multiple scenarios")
    parser.add_argument("--ablation", choices=sorted(ABLATIONS), default="full")
    parser.add_argument("--first-domain", type=int, default=60)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--trial-timeout", type=float, default=45.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    cases = [scenario for scenario, count in args.case for _ in range(count)]
    if not cases or not 1 <= args.first_domain <= 100 or args.first_domain + len(cases) - 1 > 100:
        parser.error("use a free ROS domain range inside 1..100")

    experiment_dir = args.output or (ROOT / "log" / ("experiment_matrix_" + time.strftime("%Y%m%d_%H%M%S")))
    experiment_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "study_type": "our_method_baseline",
        "method": "FAST-LIO2 + GPU terrain analyzer + FSGP-BGK/SEB + terrain_nav",
        "ablation": args.ablation,
        "cases": args.case,
        "runs": [],
    }
    (experiment_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    for index, scenario in enumerate(cases, 1):
        domain = args.first_domain + index - 1
        command = [sys.executable, "scripts/sim/checks/pipeline_smoke.py",
                   "--domain", str(domain), "--timeout", str(args.timeout),
                   "--trial-timeout", str(args.trial_timeout)]
        if scenario == "mars":
            command.append("--skip-goal")
        else:
            command.extend(["--trial-scenario", scenario])
        if args.ablation == "cpu":
            command.extend(["--terrain-backend", "cpu"])
        if args.ablation == "no_fastlio_dynamic_clear":
            command.extend(["--dynamic-clear-enabled", "false"])
        if args.ablation == "no_terrain_dynamic_clear":
            command.extend(["--terrain-dynamic-clear-enabled", "false"])
        if args.ablation == "no_live_layer":
            command.append("--disable-live-obstacles")

        output_file = experiment_dir / f"{index:02d}_{args.ablation}_{scenario}.log"
        print(f"[experiment] {index}/{len(cases)} ablation={args.ablation} scenario={scenario} domain={domain}", flush=True)
        started = time.monotonic()
        with output_file.open("w") as output:
            process = subprocess.Popen(command, cwd=ROOT, stdout=output,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                returncode = process.wait(timeout=args.timeout + args.trial_timeout + 100.0)
            except subprocess.TimeoutExpired:
                stop_group(process)
                returncode = -1
        text = output_file.read_text(errors="replace")
        run_dir = find_run_dir(text)
        row = {
            "index": index,
            "ablation": args.ablation,
            "scenario": scenario,
            "domain": domain,
            "passed": returncode == 0,
            "returncode": returncode,
            "duration_s": round(time.monotonic() - started, 2),
            "experiment_log": str(output_file),
            "run_dir": str(run_dir) if run_dir else None,
        }
        row.update(parse_log_metrics(run_dir))
        manifest["runs"].append(row)
        (experiment_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"[experiment] {'PASS' if row['passed'] else 'FAIL'} run_dir={run_dir}", flush=True)

    passed = sum(bool(row["passed"]) for row in manifest["runs"])
    print(f"[experiment] {passed}/{len(manifest['runs'])} passed; output={experiment_dir}", flush=True)
    return 0 if passed == len(manifest["runs"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
