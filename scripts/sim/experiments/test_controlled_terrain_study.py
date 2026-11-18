"""Independent truth and masking contracts for the new controlled experiments."""

import json
from pathlib import Path

import numpy as np
import pytest

from controlled_terrain_study import footprint_truth, missing_mask, score_risk, specification, surface
from terrain_risk_benchmark import params_from_config


def test_frozen_protocol_has_full_factorial_and_distinct_seeds():
    protocol = json.loads((Path(__file__).with_name("protocols")/"supplemental_study_v1.json").read_text())
    p = protocol["terrain"]
    assert len(p["families"])*p["geometries_per_family"]*len(p["masks"])*len(p["noise_std_m"]) == p["planned_cases"]
    s = protocol["scaling"]
    assert len(s["sizes"])*len(s["yaw_bins"])*len(s["backends"])*s["measured_calls"] == s["planned_measured_calls"]
    assert len(protocol["navigation"]["scenarios"])*protocol["navigation"]["per_scenario"] == 280
    assert len({p["seed"], s["seed"], protocol["navigation"]["seed"]}) == 3


def test_mask_noise_is_paired_and_hole_does_not_create_observations():
    xy = np.array([[0., 0.], [.6, .6], [1., 0.], [-1., 0.]])
    assert missing_mask(xy, "hole16", 1).tolist() == [False, False, True, True]
    assert missing_mask(xy, "frontier50", 1).tolist() == [True, False, False, True]
    assert np.array_equal(missing_mask(xy, "random30", 20), missing_mask(xy, "random30", 20))
    with pytest.raises(ValueError): missing_mask(xy, "bad", 1)


def test_flat_reference_is_safe_and_plane_matches_analytic_direction():
    p = params_from_config("cpu", .2, 36)
    spec = specification("flat", 2)
    score, unsafe, pose, _ = footprint_truth(spec, np.array([[0., 0.]]), p)
    assert np.all(score == 0) and not unsafe.any()
    spec.update(family="slope_boundary", angle=0., slope_boundary=.5)
    _, _, pose, _ = footprint_truth(spec, np.array([[0., 0.]]), p)
    yaw = (np.arange(36)+.5)*2*np.pi/36
    assert np.allclose(pose[:, 0, 0], np.arctan(np.abs(np.tan(.5)*np.cos(yaw))))
    assert np.allclose(pose[:, 0, 1], np.arctan(np.abs(np.tan(.5)*np.sin(yaw))))


def test_step_truth_uses_actual_jump_at_threshold_not_only_residual():
    p = params_from_config("cpu", .2, 36)
    spec = specification("step_high", 3)
    spec.update(angle=0., offset=0., step_high=.20)
    _, unsafe, _, span = footprint_truth(spec, np.array([[0., 0.], [2., 0.]]), p)
    assert unsafe[:, 0].all() and not unsafe[:, 1].any()
    assert (span[:, 0] >= .20-1e-12).all()
    assert np.allclose(span[:, 1], 0)


def test_thin_column_collision_detected_even_between_footprint_samples():
    p = params_from_config("cpu", .2, 36)
    spec = specification("column", 4)
    spec.update(offset=.1, column_radius=.06)
    score, unsafe, _, _ = footprint_truth(spec, np.array([[0., 0.], [2., 0.]]), p)
    assert unsafe[:, 0].all() and np.all(score[:, 0] == 1)
    assert not unsafe[:, 1].any()


def test_risk_denominators_are_not_zero_filled_when_class_is_absent():
    r = score_risk(np.array([.2]), np.array([0.]), np.array([False]), .58)
    assert r["fnr"] is None and r["fpr"] == 0
    r = score_risk(np.array([.2]), np.array([1.]), np.array([True]), .58)
    assert r["fnr"] == 1 and r["fpr"] is None
