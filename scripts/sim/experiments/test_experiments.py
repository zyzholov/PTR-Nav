"""Regression tests for plotting labels, frames and large experiment plans."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from plot_style import configure_chinese, scenario_name, wilson_interval
from analyze_navigation_subject import path_in_truth_frame, read_trials
from terrain_risk_benchmark import classification, coordinates, oracle, params_from_config


def test_chinese_font_and_labels():
    assert configure_chinese()
    assert scenario_name("ramp") == "缓坡"
    assert scenario_name("crossing") == "动态横穿"


def test_small_success_sample_has_wide_interval():
    low, high = wilson_interval(2, 2)
    assert low < 0.35
    assert high == pytest.approx(1.0)
    low, high = wilson_interval(100, 100)
    assert low > 0.96
    assert wilson_interval(0, 0) == (None, None)


@pytest.mark.parametrize("total", [1, 2, 3, 100])
def test_wilson_boundary_intervals_allow_errorbar_plot(total):
    assert wilson_interval(0, total)[0] == 0.0
    assert wilson_interval(total, total)[1] == 1.0


def test_rigid_initial_frame_alignment():
    raw = {"map_start": [10, 20, 0, 0], "truth_start": [1, 2, 0],
           "truth_yaw_start": np.pi / 2}
    np.testing.assert_allclose(path_in_truth_frame([[10, 20], [11, 20]], raw),
                               [[1, 2], [1, 3]], atol=1e-8)


def test_missing_raw_failure_remains_in_denominator(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"runs": [{"index": 1, "scenario": "ramp",
                                              "passed": False, "run_dir": None}]}))
    _, trials = read_trials(manifest)
    assert len(trials) == 1
    assert not trials[0]["summary"]["reached"]


def test_classification_undefined_denominator_is_not_perfect_score():
    risk = np.zeros((4, 3, 3))
    metrics = classification(risk, np.zeros_like(risk, dtype=bool), np.ones((3, 3), dtype=bool), 0.58)
    assert metrics["false_safe_rate"] is None
    assert metrics["f1"] is None
    assert metrics["false_hazard_rate"] == 0


def test_exact_plane_oracle():
    params = params_from_config("cpu", 0.2, 36)
    x, y = coordinates(32, 0.2)
    risk, unsafe = oracle("ramp", x, y, params, 0.31)
    assert not unsafe.any()
    np.testing.assert_allclose(risk[:, 0, 0], risk[:, 20, 20])
    pitch = np.arctan(0.17 * np.cos(np.pi / 36 - 0.31))
    expected = (pitch - params.pitch_free_rad) / (params.max_pitch_rad - params.pitch_free_rad)
    assert risk[0, 0, 0] == pytest.approx(expected)


def test_400_job_plan_reuses_domain_pool_and_is_seeded(tmp_path):
    command = [sys.executable, str(HERE / "run_matrix.py"), "--case", "ramp:100",
               "--case", "crossing:100", "--case", "step:100", "--case", "detour:100",
               "--vary-goals", "--seed", "7", "--dry-run", "--output", str(tmp_path)]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    plan = json.loads(result.stdout)
    assert len(plan["jobs"]) == 400
    assert max(job["domain"] for job in plan["jobs"]) <= 67
    assert [job["scenario"] for job in plan["jobs"][:4]] == ["ramp", "crossing", "step", "detour"]
    again = json.loads(subprocess.run(command, check=True, text=True, capture_output=True).stdout)
    assert plan == again
    assert len({job["goal_distance_m"] for job in plan["jobs"]}) > 300


def test_resume_rejects_different_settings_without_overwrite(tmp_path):
    manifest = tmp_path / "manifest.json"
    original = '{"settings": {}, "jobs": [], "runs": []}'
    manifest.write_text(original)
    result = subprocess.run([sys.executable, str(HERE / "run_matrix.py"), "--case", "ramp:1",
                             "--resume", "--output", str(tmp_path)], text=True, capture_output=True)
    assert result.returncode != 0
    assert manifest.read_text() == original
