#!/usr/bin/env python3
"""Serial public-cloud/column campaign; preserve logs and every failed stage."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from pointcloud_batch import ROOT, atomic_json, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true", help="resume unchanged GP jobs; other completed stages are verified")
    args = parser.parse_args()
    output = ROOT / "log/algorithm_optimization_20261013"
    output.mkdir(parents=True, exist_ok=True)
    dataset = "log/public_cloud_dataset_20261012/dataset_manifest.json"
    jobs = []
    for name, seed, repeats in (("paired_a", 20261012, 5), ("paired_b", 40261012, 5), ("expanded", 80261013, 10)):
        result = f"log/optimization_public_{name}_20261013"
        command = [sys.executable, "scripts/sim/experiments/pointcloud_batch.py", "run", "--dataset", dataset,
            "--output", result, "--device", "cuda", "--protocol", "public15", "--repeats", str(repeats),
            "--noise-std", "0", "--seed", str(seed)]
        if args.resume and (ROOT / result / "batch_manifest.json").exists():
            command += ["--resume"]
        jobs.append(("gp_" + name, command))
        # The paired risk branch is intentionally evaluated on the same raw inputs.
        if name != "expanded":
            jobs.append(("risk_" + name, [sys.executable, "scripts/sim/experiments/public_cloud_study.py", "analyze",
                "--dataset", dataset, "--batch", result + "/batch_manifest.json",
                "--output", f"log/optimization_risk_{name}_20261013"]))
    jobs.append(("columns", [sys.executable, "scripts/sim/experiments/obstacle_regression.py",
        "--before-source", "log/algorithm_optimization_20261013/frozen_before/obstacle_columns.py",
        "--output", "log/optimization_columns_20261013", "--repeats", "200"]))
    fingerprints = {name: digest(ROOT / "src/fsgp_bgk_se2/fsgp_bgk_se2" / name)
                    for name in ("feature_sgp.py", "obstacle_columns.py", "bgk_seb.py")}
    existing = output / ("campaign_resume.json" if args.resume else "campaign.json")
    candidates = [existing, output / "campaign_audit.json", output / "campaign.json"] if args.resume else [existing]
    readable = next((p for p in candidates if p.exists()), None)
    previous = json.loads(readable.read_text()) if readable else None
    if previous and (not args.resume or previous["fingerprints"] != fingerprints):
        raise ValueError("use --resume only for the same frozen method")
    records = {r["stage"]: r for r in previous["stages"]} if previous else {}
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    for name, command in jobs:
        result_path = ROOT / command[command.index("--output") + 1] / (
            "batch_manifest.json" if name.startswith("gp_") else "results.json")
        if args.resume and name in records and records[name]["exit_code"] == 0 and not name.startswith("gp_"):
            if not result_path.exists() or digest(result_path) != records[name].get("result_sha256"):
                raise ValueError(f"completed stage artifacts changed: {name}")
            continue
        if args.resume and not name.startswith("gp_") and result_path.exists():
            raise ValueError(f"preserve failed stage {name}; rerun explicitly into a new output directory")
        if any(digest(ROOT / "src/fsgp_bgk_se2/fsgp_bgk_se2" / n) != h for n, h in fingerprints.items()):
            raise ValueError("algorithm changed during frozen campaign")
        print(f"[campaign] {name}: {' '.join(command)}", flush=True)
        logfile = output / (name + ("_resume" if args.resume else "") + ".log")
        with logfile.open("w") as stream:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        records[name] = {"stage": name, "command": command, "log": str(logfile), "exit_code": result.returncode,
                         "result_sha256": digest(result_path) if result_path.exists() else None}
        atomic_json(existing, {"date": "2026-10-13", "fingerprints": fingerprints, "stages": list(records.values())})
        print(logfile.read_text()[-1500:], flush=True)
    failures = [r["stage"] for r in records.values() if r["exit_code"]]
    print(f"[campaign] finished {len(records)} stages; nonzero stages: {failures}", flush=True)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
