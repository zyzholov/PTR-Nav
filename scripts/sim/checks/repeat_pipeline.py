#!/usr/bin/env python3
"""Repeat isolated end-to-end Isaac trials and retain every run's evidence."""

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


def parse_case(value):
    scenario, separator, count = value.partition(":")
    if scenario not in SCENARIOS or not separator or not count.isdecimal() or int(count) < 1:
        raise argparse.ArgumentTypeError("expected SCENARIO:COUNT, e.g. clearance:3")
    return scenario, int(count)


def read_trial(run_dir):
    trial_file = run_dir / ("map_guard_trial.log" if (run_dir / "map_guard_trial.log").exists()
                            else "navigation_trial.log")
    if not trial_file.exists():
        return {}
    for line in trial_file.read_text().splitlines():
        if line.startswith("{"):
            return json.loads(line)
    return {}


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", type=parse_case, required=True,
                        help="SCENARIO:COUNT; repeat this option for more scenarios")
    parser.add_argument("--first-domain", type=int, default=70)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--trial-timeout", type=float, default=65.0)
    args = parser.parse_args()
    cases = [scenario for scenario, count in args.case for _ in range(count)]
    if args.first_domain < 1 or args.first_domain + len(cases) - 1 > 100:
        parser.error("all test domains must be within 1..100")
    evidence_dir = ROOT / "log" / ("repeat_pipeline_" + time.strftime("%Y%m%d_%H%M%S"))
    evidence_dir.mkdir(parents=True)
    runs = []
    try:
        for index, scenario in enumerate(cases, 1):
            domain = args.first_domain + index - 1
            command = [sys.executable, "scripts/sim/checks/pipeline_smoke.py",
                       "--domain", str(domain), "--timeout", str(args.timeout),
                       "--trial-timeout", str(args.trial_timeout)]
            if scenario == "mars":
                command.append("--skip-goal")
            else:
                command.extend(["--trial-scenario", scenario])
            output_file = evidence_dir / f"{index:02d}_{scenario}.log"
            print(f"[repeat] {index}/{len(cases)} {scenario} domain={domain}", flush=True)
            started = time.monotonic()
            with output_file.open("w") as output:
                process = subprocess.Popen(command, cwd=ROOT, stdout=output,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    returncode = process.wait(timeout=args.timeout + args.trial_timeout + 100.0)
                except subprocess.TimeoutExpired:
                    stop_group(process)
                    returncode = -1
                except KeyboardInterrupt:
                    stop_group(process)
                    raise
            output_text = output_file.read_text()
            match = re.search(r"\[smoke\] (?:PASS|FAIL); logs=(\S+)", output_text)
            run_dir = Path(match.group(1)) if match else None
            trial = read_trial(run_dir) if run_dir else {}
            run = {
                "scenario": scenario, "domain": domain, "passed": returncode == 0,
                "duration_s": round(time.monotonic() - started, 1),
                "pipeline_log": str(output_file),
                "run_dir": str(run_dir) if run_dir else None,
                "laser_clear_s": trial.get("ghost_laser_clear_delay_s"),
                "laser_obstacle_clear_s": trial.get("ghost_laser_obstacle_clear_delay_s"),
                "laser_before_peak": trial.get("ghost_laser_peak_before_vacate"),
                "risk_clear_s": trial.get("ghost_risk_clear_delay_s"),
                "risk_known_clear_s": trial.get("ghost_risk_known_clear_delay_s"),
                "live_clear_s": trial.get("ghost_live_clear_delay_s"),
                "guard_clear_s": trial.get("ghost_clear_delay_s"),
                "guard_baseline": trial.get("baseline_guard"),
                "guard_moving_min": trial.get("guard_moving_min"),
                "static_points_last": trial.get("static_laser_last"),
                "static_points_peak": trial.get("ghost_laser_peak_before_vacate"),
                "physical_obstacle_min_m": trial.get("physical_obstacle_min_m"),
                "risk_map_period_p95_s": trial.get("risk_map_period_p95_s"),
            }
            runs.append(run)
            (evidence_dir / "summary.json").write_text(json.dumps(runs, indent=2) + "\n")
            print(f"[repeat] {'PASS' if run['passed'] else 'FAIL'} "
                  f"laser={run['laser_clear_s']} risk={run['risk_clear_s']} "
                  f"live={run['live_clear_s']} log={run['run_dir']}", flush=True)
    except KeyboardInterrupt:
        print("[repeat] interrupted; completed runs remain in summary.json", file=sys.stderr)
        return 130
    passed = sum(run["passed"] for run in runs)
    print(f"[repeat] {passed}/{len(runs)} passed; evidence={evidence_dir}", flush=True)
    return 0 if passed == len(runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
