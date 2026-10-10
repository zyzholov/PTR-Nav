#!/usr/bin/env python3
"""Assert that FAST-LIO declares and receives every Isaac YAML parameter."""

import argparse
import json
import math
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]


def flatten_parameters(parameters, prefix=""):
    result = {}
    for name, value in parameters.items():
        key = f"{prefix}.{name}" if prefix else name
        if isinstance(value, dict):
            result.update(flatten_parameters(value, key))
        else:
            result[key] = value
    return result


def values_equal(actual, expected):
    if isinstance(expected, float):
        return (isinstance(actual, (int, float)) and not isinstance(actual, bool)
                and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9))
    if isinstance(expected, list):
        return (isinstance(actual, (list, tuple)) and len(actual) == len(expected)
                and all(values_equal(a, b) for a, b in zip(actual, expected)))
    return type(actual) is type(expected) and actual == expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "src/FAST_LIO/config/isaac_hunter_mid360_pointcloud.yaml")
    parser.add_argument("--enable-map-expansion", action="store_true")
    parser.add_argument("--dynamic-clear-enabled", choices=("true", "false"), default=None)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be positive and finite")
    with args.config.open() as source:
        expected = flatten_parameters(yaml.safe_load(source)["/**"]["ros__parameters"])
    expected["mapping.dynamic_expand_enabled"] = args.enable_map_expansion
    if args.dynamic_clear_enabled is not None:
        expected["mapping.dynamic_clear_enabled"] = args.dynamic_clear_enabled == "true"
    expected["use_sim_time"] = True

    import rclpy
    from rcl_interfaces.srv import GetParameters
    from rclpy.parameter import parameter_value_to_python

    rclpy.init()
    node = rclpy.create_node("hunter_mapping_parameter_probe")
    try:
        client = node.create_client(GetParameters, "/laser_mapping/get_parameters")
        if not client.wait_for_service(timeout_sec=args.timeout):
            print("FAIL: FAST-LIO parameter service unavailable", flush=True)
            return 1
        names = sorted(expected)
        future = client.call_async(GetParameters.Request(names=names))
        rclpy.spin_until_future_complete(node, future, timeout_sec=args.timeout)
        if not future.done() or future.exception() is not None:
            print("FAIL: FAST-LIO parameter query did not complete", flush=True)
            return 1
        values = future.result().values
        if len(values) != len(names):
            print("FAIL: incomplete FAST-LIO parameter response", flush=True)
            return 1
        actual = {name: parameter_value_to_python(value) for name, value in zip(names, values)}
        mismatches = {name: {"expected": expected[name], "actual": actual[name]}
                      for name in names if not values_equal(actual[name], expected[name])}
        print(json.dumps({"checked": len(names), "mismatches": mismatches,
                          "dynamic_policy": {name: value for name, value in actual.items()
                                             if name.startswith("mapping.dynamic_")}}), flush=True)
        print("FAIL" if mismatches else "PASS", flush=True)
        return 1 if mismatches else 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
