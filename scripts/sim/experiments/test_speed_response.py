"""Speed columns require full paired trials, not invented success heights."""

import copy
import json

import pytest

from pointcloud_batch import digest
from speed_response_evidence import task_tracking_value, validate_speed_campaign


def speed_fixture(tmp_path):
    protocol = {"scenarios": ["straight"], "desired_speed_mps": [.5, .8], "per_scenario": 2,
        "planned_cases": 4, "paired_base_tasks": 2}
    path = tmp_path/"protocol.json"; path.write_text(json.dumps(protocol))
    base = tmp_path/"base.json"; base.write_text("{}")
    binary = tmp_path/"native"; binary.write_bytes(b"fixture")
    rows = [{"id": f"{pair}_{speed}", "paired_id": pair, "scenario": "straight", "desired_speed_mps": speed}
        for pair in ("a", "b") for speed in (.5, .8)]
    record = {"protocol": protocol, "protocol_sha256": digest(path), "base_plan_path": str(base),
        "base_plan_sha256": digest(base), "binary_path": str(binary), "binary_sha256": digest(binary), "runs": rows}
    return record, protocol, path


def test_speed_denominators_require_all_pairs_and_all_cells(tmp_path):
    record, protocol, path = speed_fixture(tmp_path)
    validate_speed_campaign(record, protocol, path)
    incomplete = copy.deepcopy(record); incomplete["runs"].pop()
    with pytest.raises(ValueError): validate_speed_campaign(incomplete, protocol, path)
    wrong_pair = copy.deepcopy(record); wrong_pair["runs"][-1]["paired_id"] = "a"
    with pytest.raises(ValueError): validate_speed_campaign(wrong_pair, protocol, path)


def test_speed_duplicate_or_changed_base_is_not_reused(tmp_path):
    record, protocol, path = speed_fixture(tmp_path)
    duplicate = copy.deepcopy(record); duplicate["runs"][-1]["id"] = duplicate["runs"][0]["id"]
    with pytest.raises(ValueError): validate_speed_campaign(duplicate, protocol, path)
    (tmp_path/"base.json").write_text('{"changed":true}')
    with pytest.raises(ValueError): validate_speed_campaign(record, protocol, path)


def test_native_empty_tracking_zero_is_not_a_perfect_measurement():
    assert task_tracking_value({"status": "failed", "tracking_p95_m": 0, "planning": []}) is None
    assert task_tracking_value({"status": "failed", "tracking_p95_m": .1,
        "planning": [{"success": True, "path_points": 5}]}) == .1
    assert task_tracking_value({"status": "error"}) is None
