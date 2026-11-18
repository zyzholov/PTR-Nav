from types import SimpleNamespace
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from terrain_system_study import evaluation_domains, domain_metrics, risk_metrics
from navigation_sandbox import refinement_surface


def test_artificial_and_natural_hidden_are_disjoint_and_complete():
    domains = evaluation_domains([1, 2, 3, np.nan], [True, True, False, False], [True, False, False, False])
    np.testing.assert_array_equal(domains["artificial"], [False, True, False, False])
    np.testing.assert_array_equal(domains["natural"], [False, False, True, False])
    assert not (domains["artificial"] & domains["natural"]).any()
    np.testing.assert_array_equal(domains["hidden"], domains["artificial"] | domains["natural"])
    with pytest.raises(ValueError, match="add observed"):
        evaluation_domains([1], [False], [True])


def test_empty_domain_is_not_zero_error_or_perfect_calibration():
    pred = SimpleNamespace(mean=np.zeros(2), latent_var=np.full(2, .02**2))
    result = domain_metrics(pred, np.zeros(2), np.zeros(2, bool), [.95])
    assert result["n"] == 0 and result["rmse_m"] is None and result["picp"] is None


def test_interval_width_is_vertical_not_xy_width():
    pred = SimpleNamespace(mean=np.zeros(2), latent_var=np.full(2, .02**2))
    result = domain_metrics(pred, np.array([0., 1.]), np.ones(2, bool), [.95])
    assert result["width_m"][0] == pytest.approx(.0783986)
    assert result["picp"] == [.5]


def test_risk_denominator_excludes_no_reference_and_keeps_unsafe_miss():
    result = risk_metrics(np.array([[.1, .9, .9]]), np.array([[1., 0., np.nan]]),
        np.array([[True, False, True]]), np.array([[True, True, False]]), .58)
    assert (result["n"], result["tp"], result["fn"], result["fp"], result["tn"]) == (2, 0, 1, 1, 0)
    assert result["fnr"] == 1
    empty = risk_metrics(np.array([[.1]]), np.array([[0.]]), np.array([[False]]), np.array([[True]]), .58)
    assert empty["fnr"] is None


def test_planning_maps_match_deployment_and_never_fill_unknown():
    measured = np.array([[1., 2., np.nan]])
    pred = SimpleNamespace(mean=np.full(3, 100.), latent_var=np.full(3, .0004),
                           accepted=np.array([True, False, True]))
    np.testing.assert_array_equal(refinement_surface(measured, pred), measured)
    result = refinement_surface(measured, pred, True)
    np.testing.assert_allclose(result[0, :2], [50.5, 2.])
    assert np.isnan(result[0, 2])


def test_source_group_weighting_does_not_treat_masks_as_independent():
    from report_terrain_system import cluster_values, summary
    rows=[dict(cluster_id="a",v=1.)]*20+[dict(cluster_id="b",v=3.)]
    np.testing.assert_array_equal(cluster_values(rows,lambda r:r["v"]),[1.,3.])
    assert summary(rows,lambda r:r["v"])["mean"]==2


def test_no_reference_cases_are_retained_but_not_given_zero_error():
    from report_terrain_system import summary
    result=summary([dict(cluster_id="a",v=None)],lambda r:r["v"])
    assert result==dict(n=0,mean=None,low=None,high=None)
