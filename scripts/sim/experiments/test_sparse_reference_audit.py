"""Scoring partitions must not treat absent truth as correct or safe predictions."""

import numpy as np
import pytest

from sparse_reference_audit import scoring_partition
from thesis_statistics import height_diagnostics


def test_hidden_reference_separates_removed_and_natural_missingness():
    before = np.array([[1, 1, 0], [1, 0, 0]], dtype=bool)
    observed = np.array([[1, 0, 0], [1, 0, 0]], dtype=bool)
    reference = np.array([[1, 1, 1], [0, 0, 0]], dtype=bool)
    c = scoring_partition(before, observed, reference)
    assert c["query_cells"] == 6
    assert c["hidden_reference_cells"] == 2
    assert c["removed_referenced_cells"] == 1
    assert c["naturally_unobserved_referenced_cells"] == 1
    assert c["observed_unreferenced_cells"] == 1
    assert c["unobserved_unreferenced_cells"] == 2
    assert c["unreferenced_cells"] == 3


def test_deleting_only_unreferenced_measurement_does_not_add_height_scores():
    c = scoring_partition([1, 1], [1, 0], [1, 0])
    assert c["removed_observed_cells"] == 1
    assert c["hidden_reference_cells"] == 0
    assert c["removed_referenced_cells"] == 0


@pytest.mark.parametrize("before,observed,reference", [
    ([0], [1], [1]), ([1], [1, 0], [1]), ([], [], [])])
def test_invalid_domains_rejected(before, observed, reference):
    with pytest.raises(ValueError):
        scoring_partition(before, observed, reference)


def test_nominal_is_not_empirical_coverage_and_width_is_in_meters():
    truth = np.array([.00, .01, -.01, .02, -.02, .03, -.03, .005, .06, -.07])
    d = height_diagnostics(np.zeros(10), np.full(10, .02**2), truth, np.zeros(10, bool), [.95])
    assert d["coverage_counts"].tolist() == [8]
    assert d["n"] == 10
    assert d["width_sums"][0]/10 == pytest.approx(.0784, rel=.0001)
    wider = height_diagnostics(np.zeros(10), np.full(10, .10**2), truth, np.zeros(10, bool), [.95])
    assert wider["coverage_counts"].tolist() == [10]
    assert wider["width_sums"][0]/10 == pytest.approx(.392, rel=.0001)
