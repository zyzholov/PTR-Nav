import numpy as np
import pytest
import torch
from fsgp_bgk_se2.shared_terrain import SharedTerrainParams,shared_terrain_prediction


def predict(device,detail,rough=True,pit=False):
    torch.set_num_threads(1)
    x,y=np.meshgrid((np.arange(35)-17)*.2,(np.arange(35)-17)*.2)
    h=.02*x+.01*y
    if rough:h=h+.06*((x>=.4)&(x<=.8))
    seen=np.ones(h.shape,bool);raw=np.c_[x.ravel(),y.ravel(),h.ravel()]
    if pit:raw=np.r_[raw,[[0.,0.,-.4],[.01,0.,-.41]]]
    q=shared_terrain_prediction(h,np.full(h.shape,.0001),seen,x,y,raw,0.,.2,np.zeros(h.shape,bool),
        device=device,params=SharedTerrainParams(max_detail_sources=detail),positive_veto_height=None)
    return q,h


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_real_surface_detail_adds_local_features_without_adding_observations(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    base,h=predict(device,0);detail,_=predict(device,64)
    assert 0<detail.detail_features<=64 and detail.training_points==base.training_points
    inside=np.zeros(h.shape,bool);inside[8:27,8:27]=True
    assert np.mean(abs(detail.mean[inside]-h[inside]))<.8*np.mean(abs(base.mean[inside]-h[inside]))
    assert detail.discrepancy>=.01
    assert detail.factor.shape[-1]==base.factor.shape[-1]+detail.detail_features
    pit,_=predict(device,64,pit=True)
    assert pit.contradicted[17,17] and not pit.valid[17,17]


def test_planar_grade_does_not_allocate_a_fine_bank_or_change_the_posterior():
    base,_=predict('cpu',0,False);fine,_=predict('cpu',64,False)
    assert fine.detail_features==0
    np.testing.assert_array_equal(base.mean,fine.mean)
    np.testing.assert_array_equal(base.variance,fine.variance)


def test_adaptive_fine_basis_cpu_cuda_match():
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    cpu,_=predict('cpu',64);gpu,_=predict('cuda',64)
    assert cpu.detail_features==gpu.detail_features
    np.testing.assert_array_equal(cpu.valid,gpu.valid)
    np.testing.assert_allclose(cpu.mean,gpu.mean,atol=2e-6)
    np.testing.assert_allclose(cpu.variance,gpu.variance,atol=2e-7)


def test_observation_halo_accepts_only_real_cells_outside_the_prediction_core():
    x,y=np.meshgrid((np.arange(45)-22)*.2,(np.arange(45)-22)*.2)
    h=.02*x+.01*y;seen=np.ones(h.shape,bool);seen[(x>3)&(y>1)]=False
    raw=np.c_[x[seen],y[seen],h[seen]]
    q=shared_terrain_prediction(np.where(seen,h,np.nan),np.full(h.shape,.0001),seen,x,y,raw,0.,.2,
        np.zeros(h.shape,bool),params=SharedTerrainParams(observation_halo_m=.8))
    outside=(abs(x)>3.00001)|(abs(y)>3.00001)
    assert q.valid[outside&seen].any()
    assert not q.valid[outside&~seen].any()
    assert not q.completed[outside].any()


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_footprint_halo_completes_supported_holes_without_altering_the_posterior_or_safety_gates(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.set_num_threads(1)
    x,y=np.meshgrid((np.arange(45)-22)*.2,(np.arange(45)-22)*.2)
    h=.02*x+.01*y;seen=np.ones(h.shape,bool)
    hole=(x>=3.19)&(x<=3.61)&(abs(y)<.41);seen[hole]=False
    raw=np.c_[x[seen],y[seen],h[seen]];empty=np.zeros(h.shape,bool)
    from dataclasses import replace
    p=SharedTerrainParams(observation_halo_m=.8)
    def run(params,extra=None,blocked=empty):
        points=raw if extra is None else np.r_[raw,extra]
        return shared_terrain_prediction(np.where(seen,h,np.nan),np.full(h.shape,.0001),seen,
            x,y,points,0.,.2,blocked,params=params,device=device,positive_veto_height=None)
    old=run(p);new=run(replace(p,complete_footprint_halo=True))
    assert not old.valid[hole].any() and new.valid[hole].all()
    np.testing.assert_allclose(old.mean,new.mean,atol=1e-7)
    np.testing.assert_allclose(old.variance,new.variance,atol=1e-8)
    assert old.training_points==new.training_points
    assert not new.valid[(abs(x)>3.80001)|(abs(y)>3.80001)].any()
    assert np.all(new.nearest[new.completed]<=p.max_distance_m+1e-6)
    assert np.all(new.variance[new.completed]<=p.max_std_m**2)
    r,c=np.argwhere(hole)[len(np.argwhere(hole))//2]
    pit=run(replace(p,complete_footprint_halo=True),[[x[r,c],y[r,c],h[r,c]-.4]]*2)
    assert pit.contradicted[r,c] and not pit.valid[r,c]
    blocked=empty.copy();blocked[r,c]=True
    assert not run(replace(p,complete_footprint_halo=True),blocked=blocked).valid[r,c]


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_real_floor_beside_columns_conditions_the_gp_without_training_on_the_columns(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.set_num_threads(1)
    from dataclasses import replace
    x,y=np.meshgrid((np.arange(35)-17)*.2,(np.arange(35)-17)*.2)
    h=.02*x-.01*y;seen=np.ones(h.shape,bool);blocked=np.zeros(h.shape,bool)
    blocked[8:28:4,12:24:4]=True
    raw=np.c_[x.ravel(),y.ravel(),h.ravel()]
    def run(enabled):
        return shared_terrain_prediction(h,np.full(h.shape,.0001),seen,x,y,raw,0.,.2,
            blocked,params=replace(SharedTerrainParams(),current_floor_column_seams=enabled),device=device)
    old=run(False);new=run(True)
    assert new.training_points>old.training_points
    assert not new.valid[blocked].any()
    assert not new.completed.any()  # conditioning never changes observed identity
    np.testing.assert_allclose(new.mean[new.valid],h[new.valid],atol=1e-5)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_occluded_measured_floor_near_a_wall_keeps_its_likelihood_but_not_the_wall(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from dataclasses import replace
    from scipy.ndimage import maximum_filter
    torch.set_num_threads(1)
    x,y=np.meshgrid((np.arange(35)-17)*.2,(np.arange(35)-17)*.2)
    h=.01*x-.02*y;seen=np.ones(h.shape,bool);blocked=np.zeros(h.shape,bool)
    blocked[12:24,20]=True
    seam=maximum_filter(blocked,size=3)&~blocked
    # Current rays cannot see this previously measured floor beside the wall.
    visible=~maximum_filter(blocked,size=3)
    raw=np.c_[x[visible],y[visible],h[visible]]
    raw=np.r_[raw,np.c_[x[blocked],y[blocked],np.full(blocked.sum(),1.2)]]
    p=SharedTerrainParams(current_floor_column_seams=True,condition_local_residual=True,condition_full_residual=True)
    def run(params,extra=None,heights=None):
        return shared_terrain_prediction(h if heights is None else heights,np.full(h.shape,.0001),seen,x,y,
            raw if extra is None else np.r_[raw,extra],0.,.2,blocked,params=params,device=device,positive_veto_height=None)
    old=run(p);new=run(replace(p,condition_measured_column_seams=True))
    assert new.training_points>old.training_points
    assert not new.valid[blocked].any() and not new.completed.any()
    assert np.all(new.local_variance[seam]<old.local_variance[seam])
    np.testing.assert_allclose(new.mean[new.valid],h[new.valid],atol=2e-5)
    r,c=np.argwhere(seam)[len(np.argwhere(seam))//2]
    pit=run(replace(p,condition_measured_column_seams=True),[[x[r,c],y[r,c],h[r,c]-.4]]*2)
    assert pit.contradicted[r,c] and not pit.valid[r,c]
    raised=h.copy();raised[r,c]+=.5
    top=run(replace(p,condition_measured_column_seams=True),heights=raised)
    assert not top.valid[r,c]  # an unoccupied raised top is not flattened into safe floor


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_positive_column_and_conditional_ground_are_separate_fields(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from dataclasses import replace
    torch.set_num_threads(1)
    x,y=np.meshgrid((np.arange(35)-17)*.2,(np.arange(35)-17)*.2)
    floor=.01*x-.02*y;columns=np.zeros(floor.shape,bool);columns[13:22,18]=True
    h=np.where(columns,1.2,floor);seen=np.ones(h.shape,bool);v=np.full(h.shape,.0001)
    raw=np.c_[x.ravel(),y.ravel(),h.ravel()]
    empty=np.zeros(h.shape,bool);original=columns.copy()
    p=SharedTerrainParams(condition_local_residual=True,condition_full_residual=True,
        condition_measured_column_seams=True)
    def run(params,heights=h,points=raw,blocked=empty):
        return shared_terrain_prediction(heights,v,seen,x,y,points,0.,.2,blocked,
            columns=columns,params=params,device=device,positive_veto_height=None)
    old=run(p);new=run(replace(p,predict_ground_under_columns=True))
    np.testing.assert_array_equal(columns,original)
    assert old.training_points==new.training_points  # no invented floor under the wall
    np.testing.assert_allclose(old.mean,new.mean,atol=1e-7)
    np.testing.assert_allclose(old.variance,new.variance,atol=1e-8)
    assert not old.valid[columns].any() and new.valid[columns].all()
    assert new.completed[columns].all()  # prediction, never a measured surface
    np.testing.assert_allclose(new.mean[columns],floor[columns],atol=1e-5)
    # Raising the column cannot raise the inferred road or change covariance.
    taller=h.copy();taller[columns]=3.
    high=run(replace(p,predict_ground_under_columns=True),heights=taller,
        points=np.c_[x.ravel(),y.ravel(),taller.ravel()])
    np.testing.assert_allclose(new.mean,high.mean,atol=1e-7)
    np.testing.assert_allclose(new.variance,high.variance,atol=1e-8)
    r,c=17,18
    pit=run(replace(p,predict_ground_under_columns=True),points=np.r_[raw,[[x[r,c],y[r,c],floor[r,c]-.4]]*2])
    assert pit.contradicted[r,c] and not pit.valid[r,c]
    veto=empty.copy();veto[r,c]=True
    assert not run(replace(p,predict_ground_under_columns=True),blocked=veto).valid[r,c]
