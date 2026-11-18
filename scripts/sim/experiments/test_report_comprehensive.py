import pytest
import numpy as np

import report_comprehensive as report


@pytest.fixture(autouse=True)
def statistics_protocol(monkeypatch):
    monkeypatch.setattr(report,"P",{"statistics":{"seed":17,"bootstrap_resamples":4000}})


def test_sources_are_equal_weighted_not_repeated_points_or_submaps():
    rows=[dict(cluster_id="sequenceA",score=0.) for _ in range(100)]
    rows.append(dict(cluster_id="sequenceB",score=1.))
    result=report.summarize(rows,lambda r:r["score"])
    assert result["groups"]==2
    assert result["cases"]==101
    assert result["mean"]==.5
    assert report.summarize(rows,lambda r:r["score"])==result


def test_single_source_does_not_claim_population_confidence_interval():
    result=report.summarize([dict(cluster_id="one",score=.2)]*40,lambda r:r["score"])
    assert result["ci95"]==[None,None]
    assert "单组" in report.ci_text(result)


def test_no_reference_is_not_scored_as_zero_error():
    result=report.summarize([dict(cluster_id="empty",score=None)],lambda r:r["score"])
    assert result["groups"]==0 and result["mean"] is None
    assert report.ci_text(result)=="--"


def test_factorial_rejects_duplicate_even_when_total_count_matches():
    expected={("a",1),("b",1)}
    with pytest.raises(ValueError,match="factorial"):
        report.check_factorial([dict(source="a",mask=1)]*2,expected,("source","mask"))
    report.check_factorial([dict(source=s,mask=m) for s,m in expected],expected,("source","mask"))


def test_joint_selection_balances_sources_without_reading_scores():
    from joint_integration_study import selection
    rows=[dict(family=f,cluster_id=g,id=f+g+str(i),score=100-i)
        for f in report.FAMILIES for g in ("A","B") for i in range(3)]
    selected=selection(rows,3)
    assert len(selected)==15
    for f in report.FAMILIES:
        assert [r["id"] for r in selected if r["family"]==f]==[f+"A0",f+"B0",f+"A1"]
    assert selection(list(reversed(rows)),3)==selected


@pytest.fixture
def spatial_arrays():
    measured=np.array([[1.,2.,np.nan],[np.nan,3.,np.nan]],np.float32)
    observed=np.isfinite(measured)
    data=dict(measured=measured,observed=observed,occupied=np.array([[0,0,0],[0,0,1]],bool),
        reference=np.array([1.,2.,40.,4.,3.,np.nan]),mean=np.array([100.,200.,.4,.5,300.,.6]),
        variance=np.full(6,.0004),accepted=np.array([0,1,1,0,0,1],bool),supported=np.ones(6,bool),
        nearest=np.full(6,.1),xy=np.zeros((6,2)),origin=np.zeros(2),resolution=np.array(.2),
        full_observed=np.array([[1,1,1],[1,1,0]],bool))
    joint={k:v.copy() for k,v in data.items()}
    joint.update(local=np.ones(6,bool),filled=np.array([[0,0,1],[0,0,0]],bool))
    return data,joint


def test_spatial_output_keeps_every_measurement_even_if_gp_is_rejected(spatial_arrays):
    data,joint=spatial_arrays
    before=data["measured"].copy();result=report.spatial_layers(data,joint)
    observed=data["observed"]
    np.testing.assert_array_equal(result["completed"][observed],before[observed])
    np.testing.assert_array_equal(data["measured"],before)
    assert np.isfinite(result["completed"]).sum()==observed.sum()+1
    assert result["audit"]["old_plot_omitted_observed"]==2
    assert result["audit"]["observed_lost"]==0
    assert result["audit"]["observed_max_change_m"]==0
    assert result["status"][0,0]==2


def test_spatial_fill_uses_prediction_not_hidden_truth_or_rejected_candidates(spatial_arrays):
    data,joint=spatial_arrays
    result=report.spatial_layers(data,joint)
    assert result["completed"][0,2]==pytest.approx(.4)
    assert np.isnan(result["completed"][1,0])
    assert np.isnan(result["completed"][1,2])  # Occupied unknown, despite generic GP acceptance.
    data["reference"][2]=999.;joint["reference"][2]=999.
    np.testing.assert_array_equal(report.spatial_layers(data,joint)["completed"],result["completed"])


def test_spatial_generic_acceptance_is_not_strict_completion(spatial_arrays):
    data,joint=spatial_arrays
    data["variance"][2]=.08**2;joint["variance"][2]=.08**2
    joint["filled"][:]=False
    result=report.spatial_layers(data,joint)
    assert result["audit"]["hidden_gp_accepted"]==1
    assert result["audit"]["hidden_filled"]==0
    assert np.isnan(result["completed"][0,2])


def test_spatial_rejects_changed_joint_replay(spatial_arrays):
    data,joint=spatial_arrays;joint["mean"][0]+=1
    with pytest.raises(ValueError,match="replay changed mean"):
        report.spatial_layers(data,joint)


def test_spatial_rechecks_archived_completion_mask(spatial_arrays):
    data,joint=spatial_arrays;joint["filled"][0,0]=True
    with pytest.raises(ValueError,match="differs from archived"):
        report.spatial_layers(data,joint)


def test_3d_draws_every_finite_cell_without_bridging_unknown(monkeypatch):
    from mpl_toolkits.mplot3d import art3d

    captured={}
    def collection(vertices,**kwargs):
        captured["vertices"]=vertices
        return object()
    class Axis:
        def add_collection3d(self,artist):pass
        def auto_scale_xyz(self,*args):pass
    monkeypatch.setattr(art3d,"Poly3DCollection",collection)
    yy,xx=np.indices((2,3),dtype=float);height=np.array([[1,np.nan,3],[np.nan,5,np.nan]])
    report.grid_surface(Axis(),xx,yy,height,.2,"terrain",1,5)
    vertices=captured["vertices"]
    assert vertices.shape==(3,4,3)
    np.testing.assert_allclose(vertices[:,:,0].mean(axis=1),xx[np.isfinite(height)])
    np.testing.assert_allclose(vertices[:,:,1].mean(axis=1),yy[np.isfinite(height)])
    np.testing.assert_allclose(vertices[:,:,2],np.repeat(height[np.isfinite(height),None],4,axis=1))
    np.testing.assert_allclose(np.ptp(vertices[:,:,0],axis=1),.2)
    np.testing.assert_allclose(np.ptp(vertices[:,:,1],axis=1),.2)
