"""Paired desired-speed experiment using frozen GPU maps and production callbacks."""

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess

from pointcloud_batch import ROOT, atomic_json, digest
from navigation_sandbox import verify_native_build


def make_plans(base, base_path, protocol, protocol_path, output, binary):
    if (output/"manifest.json").exists():
        raise ValueError("preserve earlier speed campaign; resume its existing plans")
    if digest(binary) != base["settings"]["binary_sha256"]:
        raise ValueError("binary differs from the frozen paired base")
    verify_native_build(binary)
    for name, expected in base["settings"]["source_sha256"].items():
        if digest(ROOT/name) != expected:
            raise ValueError("paired base method changed: "+name)
    selected = [c for c in base["cases"] if c["scenario"] in protocol["scenarios"]]
    counts = {name: sum(c["scenario"] == name for c in selected) for name in protocol["scenarios"]}
    if any(n != protocol["per_scenario"] for n in counts.values()) or len(selected) != protocol["paired_base_tasks"]:
        raise ValueError("speed protocol must include the complete paired scenario groups")
    output.mkdir(parents=True, exist_ok=True); plans = []
    for speed_index, speed in enumerate(protocol["desired_speed_mps"]):
        if not 0 < speed <= protocol["controls"]["max_speed_mps"]:
            raise ValueError("desired speed outside the tested vehicle limit")
        plan = copy.deepcopy(base)
        plan["parameters"]["desired_speed"] = speed
        plan["parameters"]["max_speed"] = protocol["controls"]["max_speed_mps"]
        plan["parameters"]["min_tracking_speed"] = min(base["parameters"]["min_tracking_speed"], speed*.5)
        plan["settings"].update(speed_protocol_sha256=digest(protocol_path), desired_speed_mps=speed,
            paired_base_sha256=digest(base_path), per_scenario=protocol["per_scenario"],
            scenarios=protocol["scenarios"], speed_parameters=plan["parameters"])
        plan["settings"]["source_sha256"][str(Path(__file__).relative_to(ROOT))] = digest(__file__)
        plan["cases"] = []
        for c in selected:
            case = copy.deepcopy(c); path = (base_path.parent/c["map_file"]).resolve()
            if digest(path) != c["map_sha256"]:
                raise ValueError("paired CDR map changed")
            case.update(id=c["id"]+f"__speed{speed_index}", paired_id=c["id"], map_file=str(path), desired_speed_mps=speed)
            plan["cases"].append(case)
        relative = Path(f"speed_{speed_index}")/"cases.json"
        (output/relative.parent).mkdir(parents=True, exist_ok=True)
        atomic_json(output/relative, plan)
        plans.append({"speed_index": speed_index, "desired_speed_mps": speed,
            "path": str(relative), "sha256": digest(output/relative), "planned_cases": len(selected)})
    manifest = {"study_id": protocol["study_id"], "protocol_sha256": digest(protocol_path), "protocol": protocol,
        "base_plan_path": str(base_path.resolve()), "base_plan_sha256": digest(base_path),
        "binary_path": str(binary.resolve()), "binary_sha256": digest(binary), "plans": plans, "runs": []}
    atomic_json(output/"manifest.json", manifest)
    return manifest


def run_plans(manifest, output, domain):
    if digest(manifest["binary_path"]) != manifest["binary_sha256"]:
        raise ValueError("speed binary changed")
    if digest(manifest["base_plan_path"]) != manifest["base_plan_sha256"]:
        raise ValueError("paired base plan changed")
    rows = []
    for entry in manifest["plans"]:
        path = output/entry["path"]
        if digest(path) != entry["sha256"]:
            raise ValueError("speed plan changed")
        plan = json.loads(path.read_text())
        for name, expected in plan["settings"]["source_sha256"].items():
            if digest(ROOT/name) != expected: raise ValueError("speed method source changed")
        for c in plan["cases"]:
            if digest(c["map_file"]) != c["map_sha256"]:
                raise ValueError("speed map changed")
        print(f"[speed study] {entry['desired_speed_mps']} m/s: {len(plan['cases'])} paired tasks", flush=True)
        folder = path.parent/"runs"
        with (path.parent/"native.log").open("a") as log:
            completed = subprocess.run([manifest["binary_path"], str(path), str(folder)],
                env={**os.environ, "ROS_DOMAIN_ID": str(domain), "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp"},
                stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError("native speed batch failed; raw log and checkpoints preserved")
        result = json.loads((folder/"results.json").read_text())
        expected = {c["id"]: c for c in plan["cases"]}
        if result["settings"] != plan["settings"] or len(result["runs"]) != len(expected) or {r["id"] for r in result["runs"]} != set(expected):
            raise ValueError("native speed results incomplete")
        for r in result["runs"]:
            c = expected[r["id"]]
            row = dict(r, paired_id=c["paired_id"], desired_speed_mps=entry["desired_speed_mps"])
            rows.append(row)
        print(f"[speed study] completed {len(rows)} tasks; last stage passed={result['passed']}/{len(expected)}", flush=True)
        manifest["runs"] = rows; atomic_json(output/"manifest.json", manifest)
    if len(rows) != manifest["protocol"]["planned_cases"]:
        raise ValueError("speed campaign does not match its planned count")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=ROOT/"log/supplemental_study_20261014/navigation/cases.json")
    parser.add_argument("--protocol", type=Path, default=ROOT/"scripts/sim/experiments/protocols/speed_response_v1.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--domain", type=int, default=92)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.domain <= 100: parser.error("use an isolated ROS domain")
    binary = ROOT/"install/terrain_nav/lib/terrain_nav/terrain_nav_benchmark"
    protocol = json.loads(args.protocol.read_text())
    if args.resume:
        manifest = json.loads((args.output/"manifest.json").read_text())
        if manifest["protocol_sha256"] != digest(args.protocol): raise ValueError("resume protocol changed")
    else:
        manifest = make_plans(json.loads(args.base.read_text()), args.base, protocol, args.protocol, args.output, binary)
    run_plans(manifest, args.output, args.domain)


if __name__ == "__main__":
    main()
