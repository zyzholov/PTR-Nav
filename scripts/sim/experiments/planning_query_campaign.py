"""Frozen randomized static queries using the production planner, not closed-loop navigation."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from pointcloud_batch import ROOT, atomic_json, digest
from navigation_sandbox import prepare as prepare_maps, verify_native_build

FAMILIES = ("turn", "reverse", "detour", "corridor", "ramp", "cross_slope", "waves", "steep_ramp", "step", "unknown")


def layout_id(case):
    geometry = {k: case.get(k) for k in ("terrain", "obstacles", "unknown_band", "unknown_edge")}
    return hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest()


def randomize_queries(cases, seed):
    result = []
    for original in cases:
        case = dict(original)
        rng = np.random.default_rng(case["seed"] + seed)
        half_y = .1 if case["scenario"] == "corridor" else .25
        case["start"] = [float(rng.uniform(-.6, .6)), float(rng.uniform(-half_y, half_y)), float(rng.uniform(-.15, .15))]
        case["layout_id"] = layout_id(case)
        result.append(case)
    order = np.random.default_rng(seed).permutation(len(result))
    return [result[i] for i in order]


def prepare(args):
    args.scenarios = list(FAMILIES); args.mppi = False
    prepare_maps(args)
    path = args.output / "cases.json"
    plan = json.loads(path.read_text())
    plan["cases"] = randomize_queries(plan["cases"], args.seed)
    protocol=json.loads(args.protocol.read_text())
    planned_per_scenario=protocol.get("planning",{}).get("per_scenario",100)
    plan["settings"].update(study_id=protocol.get("study_id","PTR-PAPER-03")+"-Q", timing_repeats=args.repeats,
        harness_sha256=digest(Path(__file__)), protocol_sha256=digest(args.protocol),
        scope=__doc__, timing_boundary="executePlannerJob only; excludes map loading/update and independent validation",
        independent_validation="0.05m spatial and 0.05rad angular interpolation, dense footprint and SAT, no physical dynamics",
        preflight=args.per_scenario != planned_per_scenario)
    atomic_json(path, plan)
    print("[query plan]", len(plan["cases"]), "queries,", len({c["layout_id"] for c in plan["cases"]}), "geometries", flush=True)


def run(args):
    path = args.output / "cases.json"; plan = json.loads(path.read_text())
    if digest(args.binary) != plan["settings"]["binary_sha256"]: raise ValueError("binary changed")
    if digest(Path(__file__)) != plan["settings"]["harness_sha256"]: raise ValueError("runner changed")
    if digest(args.protocol) != plan["settings"]["protocol_sha256"]: raise ValueError("protocol changed")
    verify_native_build(args.binary)
    for case in plan["cases"]:
        if digest(args.output / case["map_file"]) != case["map_sha256"]: raise ValueError("map archive changed")
    for name, expected in plan["settings"]["source_sha256"].items():
        if digest(ROOT / name) != expected: raise ValueError("map source changed")
    result = subprocess.run([str(args.binary), str(path), str(args.output / "runs")],
        env={**os.environ, "ROS_DOMAIN_ID": str(args.domain), "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp"})
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, default=ROOT/"log/paper_aligned_20261015/planning")
    parser.add_argument("--seed", type=int, default=261015903); parser.add_argument("--per-scenario", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3); parser.add_argument("--domain", type=int, default=88)
    parser.add_argument("--protocol", type=Path, default=ROOT/"scripts/sim/experiments/protocols/paper_aligned_v1.json")
    parser.add_argument("--binary", type=Path, default=ROOT/"build/terrain_nav/terrain_nav_query_benchmark")
    args = parser.parse_args()
    if args.per_scenario < 1 or args.repeats < 1: parser.error("positive queries and repeats required")
    if args.phase == "prepare": prepare(args); return 0
    return run(args)


if __name__ == "__main__": raise SystemExit(main())
