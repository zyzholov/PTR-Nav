import numpy as np
import pytest

from thesis_statistics import (box_summary,equal_source_curve,height_diagnostics,
                               source_values,source_mean_interval,threshold_counts,zero_failure_required,zero_failure_upper)


def test_sources_not_repeated_points_define_box_samples():
    rows=[{"status":"ok","cluster_id":"a","mae":1.}]*100
    rows += [{"status":"ok","cluster_id":"b","mae":0.}]
    assert source_values(rows,"mae")=={"a":1.,"b":0.}
    assert box_summary(list(source_values(rows,"mae").values()))["median"]==.5


def test_float32_operating_threshold_matches_original_scorer():
    scores=np.array([.58,.65,1.,0.],np.float32);unsafe=np.array([1,0,1,0],bool)
    result=threshold_counts(scores,unsafe,np.ones(4,bool),[.58,.65,1.001])
    for i,t in enumerate((.58,.65,1.001)):
        hazard=scores>=t
        np.testing.assert_array_equal(result[i],[(hazard&unsafe).sum(),(~hazard&unsafe).sum(),
            (hazard&~unsafe).sum(),(~hazard&~unsafe).sum()])


def test_curve_equal_weights_source_groups_and_keeps_undefined_precision():
    small=np.array([[1,0,0,1],[0,1,0,1]])
    large=np.array([[0,100,100,0],[0,100,0,100]])
    curve=equal_source_curve([small,large])
    assert curve["recall"][0]==.5 and curve["fpr"][0]==.5
    assert np.isnan(curve["precision"][1])
    np.testing.assert_allclose(equal_source_curve([small,large*100])["recall"],curve["recall"])


def test_height_calibration_uses_only_hidden_reference_and_zero_variance_limit():
    d=height_diagnostics([0.,100.,1.],[1.,1.,0.],[0.,0.,0.],[False,True,False],[.95])
    assert d["n"]==2 and d["coverage_counts"][0]==1
    assert d["crps_sum_m"]==pytest.approx(1+(np.sqrt(2)-1)/np.sqrt(np.pi))
    with pytest.raises(ValueError,match="variance"):
        height_diagnostics([0.],[-1.],[0.],[False],[.95])


def test_zero_failure_bounds_do_not_claim_small_trials_prove_reliability():
    assert zero_failure_upper(2)==pytest.approx(.77639320225)
    assert zero_failure_required(.01)==299
    assert zero_failure_upper(298)>.01 and zero_failure_upper(299)<.01


def test_offline_point_occupancy_does_not_inherit_ros_pose_prior_contract():
    from terrain_risk_benchmark import params_from_config
    assert not params_from_config("cpu",.2,36).baseline_is_pose_risk


def test_interval_resamples_source_rows_not_individual_points():
    low,high=source_mean_interval([[.25,.75]],resamples=100)
    np.testing.assert_array_equal(low,[.25,.75]);np.testing.assert_array_equal(high,[.25,.75])
