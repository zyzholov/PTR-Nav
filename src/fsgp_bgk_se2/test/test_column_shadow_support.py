"""A column's 2.5D shadow cannot erase independently enclosed road support."""
from dataclasses import replace
import numpy as np
import pytest
import torch
from fsgp_bgk_se2.shared_terrain import SharedTerrainParams,shared_terrain_prediction


def scene(device='cpu',problem=None):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.set_num_threads(1)
    x,y=np.meshgrid((np.arange(25)-12)*.2,(np.arange(25)-12)*.2)
    z=np.zeros(x.shape,np.float32);z[14,10]=.55
    seen=np.ones(z.shape,bool);seen[12,13:15]=False
    column=np.zeros(z.shape,bool);column[12,13]=True
    blocked=np.zeros_like(column)
    if problem=='one_sided':seen[:,14:]=False
    if problem=='rough':z+=np.where(np.indices(z.shape).sum(axis=0)%2,.025,-.025)
    if problem=='step':z[:,13:]=.3
    if problem=='blocked':blocked[12,13]=True
    points=np.c_[x[seen],y[seen],z[seen]]
    if problem=='pit':points=np.r_[points,[[x[12,13],y[12,13],-.4],[x[12,13]+.01,y[12,13],-.41]]]
    if problem=='far':seen[:]=False;seen[:,:9]=True;points=np.c_[x[seen],y[seen],z[seen]]
    model=SharedTerrainParams(max_distance_m=1.25,max_extrapolation_m=.75,predict_ground_under_columns=True,
        current_floor_column_seams=True,condition_measured_column_seams=True,condition_local_residual=True,
        condition_full_residual=True,residual_inducing_points=32)
    def run(enabled):
        return shared_terrain_prediction(np.where(seen,z,np.nan),np.full(z.shape,.0001),seen,x,y,points,0.,.2,
            blocked,device=device,params=replace(model,column_shadow_floor_reference=enabled),
            columns=column,max_step=.1,positive_veto_height=None)
    return run(False),run(True),column,blocked


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_enclosed_current_floor_restores_only_the_column_shadow_without_changing_the_posterior(device):
    before,after,column,blocked=scene(device)
    assert before.discontinuity_veto[12,13] and not before.valid[12,13]
    assert after.column_shadow_floor_supported[12,13] and after.valid[12,13]
    assert not after.discontinuity_veto[12,13]
    assert before.discontinuity_veto[12,14] and after.discontinuity_veto[12,14]
    assert not after.valid[12,14] # unobserved NON-column ground gets no exception
    assert column[12,13] and not blocked.any() # occupied volume never cleared
    np.testing.assert_array_equal(before.mean,after.mean)
    np.testing.assert_array_equal(before.variance,after.variance)
    torch.testing.assert_close(before.factor,after.factor,atol=0,rtol=0)
    assert before.training_points==after.training_points
    assert not (before.valid&~after.valid).any()


@pytest.mark.parametrize('problem',['one_sided','rough','step','blocked','pit','far'])
@pytest.mark.parametrize('device',['cpu','cuda'])
def test_no_exception_for_missing_reference_changed_terrain_or_negative_returns(problem,device):
    before,after,column,blocked=scene(device,problem)
    assert not after.valid[12,13]
    assert not after.column_shadow_floor_supported[12,13]
    if problem=='pit':assert after.contradicted[12,13]
    if problem=='blocked':assert blocked[12,13]


def test_column_reference_option_cannot_be_enabled_for_a_model_that_does_not_separate_columns():
    with pytest.raises(ValueError):SharedTerrainParams(column_shadow_floor_reference=True)
