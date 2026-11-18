"""Reporting uses matched sources and cannot erase missing metric denominators."""

import copy
import json

import numpy as np
import pytest

from controlled_terrain_study import specification
from pointcloud_batch import digest
from supplemental_evidence import metric_summary, section_indices, validate_new_campaign


def test_noise_summary_is_source_equal_not_point_or_case_weighted():
    rows = [dict(family="ridge", mask="hole16", noise_std_m=.02, status="ok", cluster_id="a", error=1.) for _ in range(10)]
    rows.append(dict(family="ridge", mask="hole16", noise_std_m=.02, status="ok", cluster_id="b", error=9.))
    rows.append(dict(family="ridge", mask="hole16", noise_std_m=.04, status="ok", cluster_id="c", error=100.))
    s = metric_summary(rows, "ridge", "hole16", .02, "error")
    assert s["mean"] == 5. and s["clusters"] == 2 and s["cases"] == 11


def test_no_positive_reference_does_not_become_zero_fnr():
    rows = [dict(family="flat", mask="hole16", noise_std_m=.02, status="ok", cluster_id="flat_control", fnr=None)]
    s = metric_summary(rows, "flat", "hole16", .02, "fnr")
    assert s["mean"] is None and s["clusters"] == 0


@pytest.mark.parametrize("angle,axis", [(0., "x"), (np.pi/2, "y")])
def test_profile_crosses_geometry_boundary_without_prediction_interpolation(angle, axis):
    shape, origin, resolution = (50, 60), np.array([-6., -5.]), .2
    x, index, actual_axis, fixed = section_indices(shape, origin, resolution, angle)
    xx, yy = np.meshgrid(origin[0]+(np.arange(shape[1])+.5)*resolution,
        origin[1]+(np.arange(shape[0])+.5)*resolution)
    truth = np.where(xx*np.cos(angle)+yy*np.sin(angle) >= .3, .25, 0.)
    assert actual_axis == axis and abs(fixed) <= resolution/2+1e-10
    assert truth[index].shape == x.shape
    assert set(truth[index]) == {0., .25}


def miniature_campaign(tmp_path):
    p = {"families": ["flat"], "geometries_per_family": 1, "masks": ["hole16"],
        "noise_std_m": [.02], "planned_cases": 1, "seed": 1}
    scaling = {"sizes": [48], "yaw_bins": [18], "backends": ["cpu"],
        "warmup_calls": 0, "measured_calls": 2, "planned_measured_calls": 2}
    protocol = {"terrain": p, "scaling": scaling,
        "navigation": {"scenarios": ["straight"], "per_scenario": 1, "seed": 3, "mppi": False}}
    path = tmp_path/"protocol.json"; path.write_text(json.dumps(protocol))
    nav_settings = {"source_sha256": {}, "seed": 3, "mppi": False}
    data = {"supplemental_protocol": protocol,
        "new_terrain": {"settings": {"terrain_plan": p, "protocol_sha256": digest(path), "source_sha256": {}},
            "runs": [{"id": "flat_000_hole16_n0", "spec": specification("flat", 1), "mask": "hole16",
                "noise_std_m": .02, "status": "ok", "actual_device": "cuda", "actual_backend": "cupy"}]},
        "new_scaling": {"protocol_sha256": digest(path), "source_sha256": {},
            "runs": [{"size": 48, "yaw_bins": 18, "backend": "cpu", "repeat": i} for i in (1, 2)]},
        "new_navigation_plan": {"settings": nav_settings, "cases": [{"id": "n1", "scenario": "straight"}]},
        "new_navigation": {"settings": nav_settings, "runs": [{"id": "n1", "scenario": "straight"}]}}
    return data, {"supplemental_protocol": path}


def test_campaign_refuses_duplicates_partial_counts_and_gpu_fallback(tmp_path):
    data, paths = miniature_campaign(tmp_path)
    validate_new_campaign(data, paths)
    duplicate = copy.deepcopy(data); duplicate["new_terrain"]["runs"] *= 2
    with pytest.raises(ValueError): validate_new_campaign(duplicate, paths)
    partial = copy.deepcopy(data); partial["new_scaling"]["runs"].pop()
    with pytest.raises(ValueError): validate_new_campaign(partial, paths)
    fallback = copy.deepcopy(data); fallback["new_terrain"]["runs"][0]["actual_device"] = "cpu"
    with pytest.raises(ValueError): validate_new_campaign(fallback, paths)


def test_campaign_refuses_mislabeled_geometry_or_noise(tmp_path):
    data, paths = miniature_campaign(tmp_path)
    changed = copy.deepcopy(data); changed["new_terrain"]["runs"][0]["spec"]["offset"] += .1
    with pytest.raises(ValueError): validate_new_campaign(changed, paths)
    changed = copy.deepcopy(data); changed["new_terrain"]["runs"][0]["noise_std_m"] = .04
    with pytest.raises(ValueError): validate_new_campaign(changed, paths)
