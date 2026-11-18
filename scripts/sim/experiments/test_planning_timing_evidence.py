"""Timing units, empty denominators and retry weighting must remain explicit."""

import copy

import numpy as np
import pytest

from planning_timing_evidence import audit_task, native_quantile, task_equal_ecdf


def fixture_row():
    calls = [dict(time_s=i*.2, elapsed_ms=v, success=i != 1, path_points=8, reason="test")
        for i, v in enumerate((2., 90., 30.))]
    return dict(id="a", scenario="emerge", status="failed", planning=calls, plan_count=3, replans=2,
        planner_p95_ms=30., planner_max_ms=90., control_p95_ms=.5, expected_stop=False, goal_error_m=4.)


def test_first_later_and_cumulative_time_are_separate():
    r = audit_task(fixture_row())
    assert r["first_call_ms"] == 2 and r["subsequent_case_p95_ms"] == pytest.approx(87.)
    assert r["planning_sum_ms"] == 122 and r["calls"] == 3
    assert r["returned_success_calls"] == 2 and r["failed_call_fraction"] == pytest.approx(1/3)


def test_no_subsequent_call_is_missing_not_zero_time():
    row = fixture_row(); row.update(planning=row["planning"][:1], plan_count=1, replans=1,
        planner_p95_ms=2., planner_max_ms=2.)
    assert audit_task(row)["subsequent_case_p95_ms"] is None
    assert native_quantile([], .95) is None


@pytest.mark.parametrize("field,value", [("plan_count", 4), ("replans", 3), ("planner_p95_ms", 87.)])
def test_trace_must_reproduce_saved_native_statistics(field, value):
    row = fixture_row(); row[field] = value
    with pytest.raises(ValueError): audit_task(row)


def test_invalid_or_nonchronological_trace_is_rejected():
    row = fixture_row(); row["planning"][-1]["time_s"] = -.1
    with pytest.raises(ValueError): audit_task(row)
    row = fixture_row(); row["planning"][1]["elapsed_ms"] = float("nan")
    with pytest.raises(ValueError): audit_task(row)


def test_retries_do_not_increase_a_tasks_ecdf_weight():
    calls = [{"task_id": "a", "elapsed_ms": 1.}, {"task_id": "b", "elapsed_ms": 10.}]
    x, y, n = task_equal_ecdf(calls)
    more = [copy.copy(calls[0]) for _ in range(100)]+[calls[1]]
    xx, yy, nn = task_equal_ecdf(more)
    assert n == nn == 2 and y[0] == pytest.approx(.5)
    assert yy[np.where(xx == 1.)[0][-1]] == pytest.approx(.5)
    assert yy[-1] == pytest.approx(1.)


def test_error_has_no_invented_wall_clock_timing():
    row = dict(id="bad", scenario="emerge", status="error")
    result = audit_task(row)
    assert result["first_call_ms"] is None and result["planning_sum_ms"] is None
