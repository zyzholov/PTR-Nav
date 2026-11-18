"""Contracts that keep question-led reporting separate from selective evidence."""

import copy
import json
from pathlib import Path

import pytest
import numpy as np

from systematic_study import complete_jobs, figure_names, grid_distance_errors, paired_source_series, require_hash, verify_analytic
from pointcloud_batch import digest


def test_physical_plan_rejects_partial_and_duplicate():
    plan = {"jobs": [{"index": 1}, {"index": 2}], "runs": [{"index": 1}, {"index": 2}]}
    complete_jobs(plan)
    with pytest.raises(ValueError):
        complete_jobs(dict(plan, runs=[{"index": 1}]))
    with pytest.raises(ValueError):
        complete_jobs(dict(plan, runs=[{"index": 1}, {"index": 1}]))


def test_backend_pairs_need_full_declared_design():
    factors = {"backends": ["cpu", "cupy"], "slope_rad": [.1], "direction_rad": [0.], "noise_repeats": 2}
    rows = [{"backend": b, "slope_rad": .1, "direction_rad": 0., "repeat": r,
             "geometry_id": "g0", "sensor_seed": r} for b in factors["backends"] for r in range(2)]
    assert len(verify_analytic({"runs": rows}, {"factors": factors})) == 2
    with pytest.raises(ValueError):
        verify_analytic({"runs": rows[:-1]}, {"factors": factors})
    changed = copy.deepcopy(rows)
    changed[-1]["sensor_seed"] = 20
    with pytest.raises(ValueError):
        verify_analytic({"runs": changed}, {"factors": factors})


def test_changed_evidence_is_not_reused(tmp_path):
    path = tmp_path / "record.json"
    path.write_text('{"runs": 2}')
    expected = digest(path)
    require_hash(path, expected)
    path.write_text('{"runs": 3}')
    with pytest.raises(ValueError):
        require_hash(path, expected)


def test_protocol_has_distinct_questions_and_honest_unrun_boundary():
    path = Path(__file__).with_name("protocols") / "systematic_study_v1.json"
    protocol = json.loads(path.read_text())
    experiments = protocol["experiments"]
    assert {e["id"] for e in experiments} == {"M1", "M2", "M3", "M4", "S1", "S2", "S3", "S4"}
    assert len({e["question"] for e in experiments}) == len(experiments)
    figures = [f for e in experiments for f in e["figures"]]
    assert len(set(figures)) == len(figures)
    boundary = next(e for e in experiments if e["id"] == "S4")
    assert boundary["status"].startswith("planned_") and not boundary["figures"]
    assert protocol["future_comparison"]["status"] == "design_only_not_executed"
    for e in experiments:
        assert all(e.get(k) for k in ("question", "factors", "truth", "statistical_unit", "metrics", "judgement"))


def test_v2_moves_speed_to_appendix_and_separates_timing_scopes():
    path = Path(__file__).with_name("protocols")/"systematic_study_v2.json"
    protocol = json.loads(path.read_text())
    names = figure_names(protocol)
    main = [n for e in protocol["experiments"] for n in e["figures"]]
    assert "C4_success_3d" not in names
    assert "C4_speed_rates" not in main and "C4_speed_rates" in protocol["appendix_figures"]
    assert "P1_phase_timing" in main and "P4_workload_3d" in main
    assert protocol["redesign"]["new_algorithm_runs"] == 0
    assert len(protocol["redesign"]["studies"]) == 6
    assert all(s["status"].startswith("planned") for s in protocol["redesign"]["studies"])
    invalid = copy.deepcopy(protocol); invalid["appendix_figures"].append("P1_phase_timing")
    with pytest.raises(ValueError): figure_names(invalid)


def test_difficulty_uses_common_sources_not_changing_cohorts():
    rows = [{"family": "indoor", "condition": c, "cluster_id": source, "status": "ok", "error": value}
            for c, source, value in (("a", "shared", 1.), ("b", "shared", 2.), ("b", "only_b", 100.))]
    result = paired_source_series(rows, "indoor", ["a", "b"], "error")
    assert result["source_groups"] == ["shared"]
    assert result["mean"].tolist() == [1., 2.]
    assert result["low"].tolist() == [1., 2.]


def test_grid_distance_uses_only_hidden_reference_not_implicit_border():
    ref = np.zeros((2, 3)); obs = np.zeros_like(ref, dtype=bool); obs[:, 0] = True
    mean = np.ones_like(ref); mean[:, 0] = 100.
    n, squared = grid_distance_errors(mean, ref, obs)
    assert n[:2].tolist() == [2, 2]
    assert squared[:2].tolist() == [2., 2.]
    assert n.sum() == 4
    assert grid_distance_errors(mean, ref, np.zeros_like(obs)) is None
    ref[0, 2] = np.nan
    assert grid_distance_errors(mean, ref, obs)[0].sum() == 3
