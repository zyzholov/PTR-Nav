from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
import torch

from fsgp_bgk_se2.shared_terrain import (SharedTerrainParams,shared_terrain_prediction,
                                       shared_geometric_probability)
from fsgp_bgk_se2.probabilistic_risk import ProbabilityParams


@pytest.mark.parametrize('reason',['missing_or_stale_scan','insufficient_measured_training_cells',
                                   'insufficient_current_floor_cells','posterior_ready'])
def test_shared_admission_diagnostics_preserve_original_evidence_and_posterior(reason):
    x,y=np.meshgrid((np.arange(12)-5.5)*.2,(np.arange(12)-5.5)*.2)
    z=.01*x-.02*y;seen=np.ones(z.shape,bool)
    if reason=='insufficient_measured_training_cells':seen[:]=False;seen[4,4:6]=True
    raw=np.c_[x[seen],y[seen],z[seen]]
    if reason=='insufficient_current_floor_cells':raw=raw[:2]
    age=1. if reason=='missing_or_stale_scan' else 0.
    args=(np.where(seen,z,np.nan),np.full(z.shape,.0001),seen,x,y,raw,age,.2,np.zeros(z.shape,bool))
    before=[a.copy() for a in args if isinstance(a,np.ndarray)]
    plain=shared_terrain_prediction(*args);diagnostic={'old':'must be cleared'}
    inspected=shared_terrain_prediction(*args,diagnostics=diagnostic)
    assert diagnostic['reason']==reason and 'old' not in diagnostic
    if plain is None:assert inspected is None
    else:
        for name in ('mean','variance','valid','completed'):
            np.testing.assert_array_equal(getattr(plain,name),getattr(inspected,name))
        assert torch.equal(plain.factor,inspected.factor)
    for a,b in zip(before,(a for a in args if isinstance(a,np.ndarray))):np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('bounded',[False,True])
def test_cpu_fit_gpu_geometry_keeps_covariance_risk_support_and_bounded_mean(bounded):
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from dataclasses import fields,replace
    from fsgp_bgk_se2.shared_terrain import prediction_on_device
    torch.set_num_threads(1)
    x,y=np.meshgrid((np.arange(20)-9.5)*.2,(np.arange(20)-9.5)*.2)
    z=.025*x-.018*y;seen=np.hypot(x,y)>.4;raw=np.c_[x[seen],y[seen],z[seen]]
    model=SharedTerrainParams(condition_local_residual=True,condition_full_residual=True,
        residual_inducing_points=64,residual_lengthscale_m=.35,bounded_queries=True,
        max_distance_m=1.25,max_extrapolation_m=.75)
    args=(np.where(seen,z,np.nan),np.full(z.shape,.0001),seen,
          x,y,raw,0.,.2,np.zeros_like(seen))
    cpu=shared_terrain_prediction(*args,device='cpu',params=model,positive_veto_height=None)
    assert cpu is not None
    if bounded:
        bias=torch.from_numpy(np.stack((.002*x,.003*y),axis=-1)).float()
        cpu=replace(cpu,bounded_mean_factor=bias,bounded_pending=~seen)
    before=cpu.factor.clone();stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):gpu=prediction_on_device(cpu,'cuda')
    assert gpu is not cpu and cpu.factor.device.type=='cpu' and gpu.factor.device.type=='cuda'
    torch.testing.assert_close(cpu.factor,before,atol=0,rtol=0)
    torch.testing.assert_close(gpu.factor.cpu(),before,atol=0,rtol=0)
    for field in fields(cpu):
        if field.name not in ('factor','bounded_mean_factor'):
            assert getattr(cpu,field.name) is getattr(gpu,field.name)
    if bounded:
        assert gpu.bounded_mean_factor.device.type=='cuda'
        torch.testing.assert_close(gpu.bounded_mean_factor.cpu(),cpu.bounded_mean_factor,atol=0,rtol=0)
    params=ProbabilityParams(yaw_bins=4,internal_steps=True,internal_step_model='height_jump',
        max_roughness=.125,max_step=.1,independent_model_floor=True)
    for a,b in zip(shared_geometric_probability(cpu,params,np.zeros_like(seen)),
                   shared_geometric_probability(gpu,params,np.zeros_like(seen))):
        np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
        np.testing.assert_allclose(a[2],b[2],atol=3e-6)
        np.testing.assert_array_equal(a[2]<=.05,b[2]<=.05)
    restored=prediction_on_device(gpu,'cpu')
    torch.testing.assert_close(restored.factor,cpu.factor,atol=0,rtol=0)
    if bounded:torch.testing.assert_close(restored.bounded_mean_factor,cpu.bounded_mean_factor,atol=0,rtol=0)


def test_cuda_posterior_waits_only_its_own_stream_and_preserves_cpu_covariance(monkeypatch):
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.set_num_threads(1)
    x,y=np.meshgrid((np.arange(20)-9.5)*.2,(np.arange(20)-9.5)*.2)
    z=.025*x-.018*y;seen=np.hypot(x,y)>.4;raw=np.c_[x[seen],y[seen],z[seen]]
    model=SharedTerrainParams(condition_local_residual=True,condition_full_residual=True,
        residual_inducing_points=64,residual_lengthscale_m=.35,bounded_queries=True,
        max_distance_m=1.25,max_extrapolation_m=.75)
    def run(device):
        return shared_terrain_prediction(np.where(seen,z,np.nan),np.full(z.shape,.0001),seen,
            x,y,raw,0.,.2,np.zeros_like(seen),device=device,params=model,positive_veto_height=None)
    cpu=run('cpu');assert cpu is not None
    stream=torch.cuda.Stream()
    def device_barrier(*args,**kwargs):
        raise AssertionError('local terrain posterior must not wait for unrelated CUDA streams')
    with monkeypatch.context() as patch:
        patch.setattr(torch.cuda,'synchronize',device_barrier)
        with torch.cuda.stream(stream):gpu=run('cuda')
    assert gpu is not None
    # Returned factors must already be ready outside the producer stream.
    for name in ('valid','completed','contradicted','discontinuity_veto'):
        np.testing.assert_array_equal(getattr(cpu,name),getattr(gpu,name))
    for name in ('mean','variance','local_variance'):
        np.testing.assert_allclose(getattr(cpu,name),getattr(gpu,name),atol=2e-6,rtol=2e-5)
    np.testing.assert_allclose(cpu.factor.numpy(),gpu.factor.cpu().numpy(),atol=2e-6,rtol=2e-5)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_sparse_noncollinear_measured_ground_has_a_proper_uncertain_posterior(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.set_num_threads(1)
    x,y=np.meshgrid((np.arange(20)-9.5)*.2,(np.arange(20)-9.5)*.2)
    z=.01*x-.02*y;seen=np.zeros(z.shape,bool);seen[9:12,8:12]=True
    raw=np.c_[x[seen],y[seen],z[seen]];diag={}
    result=shared_terrain_prediction(np.where(seen,z,np.nan),np.full(z.shape,.0001),seen,
        x,y,raw,0.,.2,np.zeros(z.shape,bool),device=device,diagnostics=diag)
    assert result is not None and result.training_points==12 and diag['reason']=='posterior_ready'
    assert result.valid[seen].all() and not result.valid[np.hypot(x,y)>1.5].any()
    np.testing.assert_allclose(result.mean[seen],z[seen],atol=2e-6)
    assert np.all(result.variance[seen]>0) and np.isfinite(result.factor.cpu().numpy()).all()
    # Repeated rays from a single measured cell cannot manufacture 3 sources.
    one=np.repeat(raw[:1],500,axis=0);diag={}
    assert shared_terrain_prediction(np.where(seen,z,np.nan),np.full(z.shape,.0001),seen,
        x,y,one,0.,.2,np.zeros(z.shape,bool),device=device,diagnostics=diag) is None
    assert diag['reason']=='insufficient_current_floor_cells' and diag['current_support_cells']==1
    line=np.zeros(z.shape,bool);line[10,3:17]=True;diag={}
    assert shared_terrain_prediction(np.where(line,z,np.nan),np.full(z.shape,.0001),line,
        x,y,np.c_[x[line],y[line],z[line]],0.,.2,np.zeros_like(line),device=device,diagnostics=diag) is None
    assert diag['reason']=='training_xy_not_two_dimensional'


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_removing_real_observations_cannot_reduce_the_fixed_model_joint_covariance(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.set_num_threads(1)
    x,y=np.meshgrid((np.arange(12)-5.5)*.2,(np.arange(12)-5.5)*.2);z=np.zeros(x.shape)
    sparse=np.zeros(z.shape,bool);sparse[4:7,4:8]=True
    dense=np.zeros(z.shape,bool);dense[3:9,3:9]=True
    model=SharedTerrainParams(condition_local_residual=True,condition_full_residual=True)
    def covariance(seen):
        r=shared_terrain_prediction(np.where(seen,z,np.nan),np.full(z.shape,.0001),seen,
            x,y,np.c_[x[seen],y[seen],z[seen]],0.,.2,np.zeros(z.shape,bool),device=device,params=model)
        assert r is not None
        f=r.factor.cpu().numpy().reshape(z.size,-1).astype(float)
        return f@f.T+np.diag(r.local_variance.ravel())
    difference=covariance(sparse)-covariance(dense)
    assert np.linalg.eigvalsh((difference+difference.T)/2).min()>-2e-7


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('contextual',[False,True])
@pytest.mark.parametrize('halo',[False,True])
def test_bounded_posterior_queries_preserve_the_same_likelihood_covariance_domain_and_risk(device,contextual,halo):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from fsgp_bgk_se2.shared_terrain import InitialPlaneContext
    torch.set_num_threads(1)
    x,y=np.meshgrid((np.arange(60)-29.5)*.2,(np.arange(50)-24.5)*.2)
    z=.03*x-.02*y;seen=np.hypot(x,y)>.6;initial=(abs(x)<.5)&(abs(y)<.5)
    raw=np.c_[x[seen],y[seen],z[seen]]
    context=InitialPlaneContext(np.zeros(2),np.array([.03,-.02,0.]),
        np.array([.005,.005,.015]),initial) if contextual else None
    model=SharedTerrainParams(max_distance_m=1.25,max_extrapolation_m=.75,
        condition_local_residual=True,condition_full_residual=True,residual_inducing_points=96,
        observation_halo_m=.8,complete_footprint_halo=halo)
    def run(bounded):
        return shared_terrain_prediction(np.where(seen,z,np.nan),np.full(z.shape,.0001),seen,
            x,y,raw,0.,.2,np.zeros_like(seen),device=device,params=replace(model,bounded_queries=bounded),
            initial_support=initial if contextual else None,initial_height=z if contextual else None,
            plane_context=context,positive_veto_height=None)
    full=run(False);tile=run(True)
    assert full.training_points==tile.training_points
    for name in ['valid','completed','candidate','assumed','contradicted','discontinuity_veto']:
        np.testing.assert_array_equal(getattr(full,name),getattr(tile,name))
    selected=np.isfinite(tile.mean)
    assert (~selected).sum()>selected.sum()/4 # substantial work actually avoided
    assert not tile.valid[~selected].any()
    for name in ['mean','variance','nearest','local_variance','relative_variance']:
        a,b=getattr(full,name),getattr(tile,name)
        if a is not None:np.testing.assert_allclose(a[selected],b[selected],atol=2e-6,rtol=2e-5)
    np.testing.assert_allclose(full.factor.cpu().numpy()[selected],tile.factor.cpu().numpy()[selected],atol=2e-6)
    assert not tile.factor.cpu().numpy()[~selected].any()
    p=ProbabilityParams(yaw_bins=4,internal_steps=True,internal_step_model='height_jump',
        max_roughness=.125,max_step=.1,independent_model_floor=True)
    for a,b in zip(shared_geometric_probability(full,p,np.zeros_like(seen)),
                   shared_geometric_probability(tile,p,np.zeros_like(seen))):
        np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
        np.testing.assert_allclose(a[2],b[2],atol=3e-6)
        np.testing.assert_array_equal(a[2]<=.05,b[2]<=.05)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_bounded_one_sided_extrapolation_keeps_uncertainty_distance_and_raw_pit_veto(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.set_num_threads(1)
    x,y=np.meshgrid((np.arange(30)-14.5)*.2,(np.arange(30)-14.5)*.2)
    z=.02*x+.01*y;seen=x>.7;empty=np.zeros(z.shape,bool)
    raw=np.c_[x[seen],y[seen],z[seen]]
    def run(extrapolation,extra=()):
        points=np.r_[raw,np.asarray(extra).reshape(-1,3)]
        return shared_terrain_prediction(np.where(seen,z,np.nan),np.full(z.shape,.0001),seen,
            x,y,points,0.,.2,empty,device=device,
            params=SharedTerrainParams(max_extrapolation_m=extrapolation))
    strict=run(0.);bounded=run(.75)
    assert not strict.valid[x<.7].any()
    extrap=bounded.valid&~seen
    assert extrap.any() and np.max(bounded.nearest[extrap])<=.75
    assert np.max(bounded.variance[extrap])<=.04**2
    np.testing.assert_allclose(bounded.mean[extrap],z[extrap],atol=1e-5)
    np.testing.assert_allclose(bounded.mean,strict.mean,atol=1e-7)
    np.testing.assert_allclose(bounded.variance,strict.variance,atol=1e-8)
    r,c=np.argwhere(extrap)[len(np.argwhere(extrap))//2]
    pit=run(.75,[[x[r,c],y[r,c],z[r,c]-.4],[x[r,c]+.01,y[r,c],z[r,c]-.41]])
    assert pit.contradicted[r,c] and not pit.valid[r,c]
    assert not bounded.valid[x<-.1].any()


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_one_sided_fixed_stance_retains_joint_covariance_and_never_becomes_training_data(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.set_num_threads(1)
    x,y=np.meshgrid((np.arange(40)-19.5)*.2,(np.arange(40)-19.5)*.2)
    z=.02*x+.01*y;seen=(x>1.)& (np.hypot(x,y)>1.1)
    from fsgp_bgk_se2.probabilistic_risk import _geometry
    params=ProbabilityParams(front=.42,back=.42,half_width=.313,margin=.1,yaw_bins=36,internal_steps=True)
    initial=np.zeros(z.shape,bool);hit=np.zeros(z.shape,bool)
    for dr,dc in _geometry(params,0)[0]:initial[20+dr,20+dc]=True
    points=np.c_[x[seen],y[seen],z[seen]]
    def run(prior):
        return shared_terrain_prediction(np.where(seen,z,np.nan),np.full(z.shape,.0001),seen,
            x,y,points,0.,.2,hit,device=device,
            params=SharedTerrainParams(max_distance_m=1.25,max_extrapolation_m=.75),
            initial_support=prior,initial_height=z)
    blank=run(None);fixed=run(initial)
    assert fixed.valid[initial].all() and fixed.assumed[initial].all()
    assert fixed.training_points==blank.training_points
    np.testing.assert_allclose(fixed.mean,blank.mean,atol=1e-7)
    np.testing.assert_allclose(fixed.variance,blank.variance,atol=1e-8)
    assert not fixed.valid[(x<0)&~initial].any()
    np.testing.assert_allclose(fixed.factor.cpu().numpy(),blank.factor.cpu().numpy(),atol=1e-7)
    r,c,upper=shared_geometric_probability(fixed,params,hit)[0]
    assert upper[(r==20)&(c==20)][0]<.05
    # Larger absolute uncertainty is kept; the pass comes from shared contrast
    # propagation, not from artificially shrinking per-cell variance.
    assert np.sqrt(fixed.variance[initial]).max()>.04


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('outside_observed',[False,True])
def test_outside_local_gp_cannot_veto_real_lower_terrain_or_unadmitted_unknown(device,outside_observed):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.set_num_threads(1)
    x,y=np.meshgrid((np.arange(50)-24.5)*.2,(np.arange(50)-24.5)*.2)
    z=np.where(x>3.5,-.6,0.).astype(np.float32);seen=np.ones(z.shape,bool)
    seen[x>3.5]=outside_observed
    raw=np.c_[x.ravel(),y.ravel(),z.ravel()]
    # Two genuine low-ground returns outside the fitted 6m region used to be
    # interpreted as a pit against the GP's unbounded extrapolation.
    raw=np.r_[raw,raw,[[x[25,25],y[25,25],-.4],[x[25,25]+.01,y[25,25],-.4]]]
    p=shared_terrain_prediction(np.where(seen,z,np.nan),np.full(z.shape,.0001),seen,x,y,raw,0.,.2,
        np.zeros(z.shape,bool),device=device,positive_veto_height=None)
    assert p is not None and not p.valid[x>3.5].any()
    assert not p.contradicted[x>3.5].any()
    # Actual below-measured-ground evidence remains an immediate veto.
    assert p.contradicted[25,25] and not p.valid[25,25]


def scene(device='cpu',extra=None):
    torch.set_num_threads(1)
    x,y=np.meshgrid((np.arange(30)-14.5)*.2,(np.arange(30)-14.5)*.2)
    z=.03*x-.02*y
    observed=np.hypot(x,y)>.7
    points=np.c_[x[observed],y[observed],z[observed]]
    if extra is not None:points=np.r_[points,extra]
    prediction=shared_terrain_prediction(np.where(observed,z,np.nan),np.full(z.shape,.0001),
        observed,x,y,points,0.,.2,np.zeros(z.shape,bool),device=device,
        params=SharedTerrainParams(radius_m=3.))
    return x,y,z,observed,prediction


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_selective_risk_refinement_preserves_the_exact_selected_proof_and_cannot_invent_support(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    _,_,_,_,prediction=scene(device)
    params=ProbabilityParams(yaw_bins=8,internal_steps=True,internal_step_model='height_jump')
    occupied=np.zeros(prediction.mean.shape,bool);occupied[12,18]=True
    all_queries=shared_geometric_probability(prediction,params,occupied)
    mask=np.zeros((8,*prediction.mean.shape),bool);mask[::2,::2,::3]=True
    selected=shared_geometric_probability(prediction,params,occupied,query_mask=mask)
    for k,((r,c,upper),(sr,sc,su)) in enumerate(zip(all_queries,selected)):
        keep=mask[k,r,c]
        np.testing.assert_array_equal(sr,r[keep]);np.testing.assert_array_equal(sc,c[keep])
        np.testing.assert_allclose(su,upper[keep],atol=2e-6)
    # No query is an instruction to retain the caller's existing valid bound,
    # not permission to return a zero-risk pose or to change ground support.
    blank=shared_geometric_probability(prediction,params,occupied,query_mask=np.zeros_like(mask))
    assert all(len(r)==len(c)==len(u)==0 for r,c,u in blank)
    with pytest.raises(ValueError,match='mask'):
        shared_geometric_probability(prediction,params,occupied,query_mask=mask[0])


@pytest.mark.parametrize('yaws',[7,36])
@pytest.mark.parametrize('steps',[False,True])
def test_factored_plane_cuda_matches_original_cuda_and_independent_tensor(yaws,steps):
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from fsgp_bgk_se2.shared_terrain_cuda import shared_probability_cuda
    x,y,z,_,p=scene('cuda')
    generator=torch.Generator(device='cuda').manual_seed(317)
    # Include a shared height-bias mode, local correlated noise and curvature;
    # this qualifies the propagation operator, not a new admission policy.
    factor=torch.cat((p.factor,torch.randn((*z.shape,5),device='cuda',generator=generator)*.004,
                      torch.full((*z.shape,1),.25,device='cuda')),dim=2)
    p=replace(p,mean=(z+.12*x*x+.04*y*y).astype(np.float32),factor=factor,discrepancy=.018)
    collision=np.zeros(z.shape,bool);collision[18,18]=True
    params=ProbabilityParams(yaw_bins=yaws,internal_steps=steps,front=.58,back=.48,half_width=.40)
    actual=shared_probability_cuda(p,params,collision)
    for reference in (shared_probability_cuda(p,params,collision,direct_span=False),
                      shared_probability_cuda(p,params,collision,factorized=False),
                      shared_geometric_probability(p,params,collision,fused=False)):
        for a,b in zip(actual,reference):
            np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
            # Float32 reductions use different orders in the factored
            # kernel and independent tensor reference (observed 2.09e-6).
            # Bound numeric error and separately require identical actual
            # navigation decisions at the 5% and 10% budgets.
            np.testing.assert_allclose(a[2],b[2],atol=3e-6)
            for threshold in (.05,.10):
                np.testing.assert_array_equal(a[2]<=threshold,b[2]<=threshold)


@pytest.mark.parametrize('plane_components',[0,3])
@pytest.mark.parametrize('tighten',[False,True])
def test_direct_span_bounds_keep_query_support_source_partition_and_local_variance(plane_components,tighten):
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from fsgp_bgk_se2.shared_terrain_cuda import shared_probability_cuda
    x,y,z,_,prediction=scene('cuda')
    random=np.random.default_rng(111)
    prediction=replace(prediction,mean=(z+.04*x*x-.02*y*y).astype(np.float32),
        local_variance=random.uniform(.00001,.0003,z.shape).astype(np.float32),
        plane_components=plane_components,discrepancy=.012)
    params=ProbabilityParams(yaw_bins=7,internal_steps=True,independent_model_floor=True)
    collision=np.zeros(z.shape,bool);collision[19,21]=True
    query=random.uniform(size=(7,*z.shape))>.3
    source=[prediction.mean.copy(),prediction.variance.copy(),prediction.valid.copy(),prediction.factor.clone()]
    actual=shared_probability_cuda(prediction,params,collision,query_mask=query,tighten_span=tighten)
    for reference in (shared_probability_cuda(prediction,params,collision,query_mask=query,tighten_span=tighten,direct_span=False),
                      shared_geometric_probability(prediction,params,collision,query_mask=query,tighten_span=tighten,fused=False)):
        for a,b in zip(actual,reference):
            np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
            np.testing.assert_allclose(a[2],b[2],atol=3e-6,rtol=2e-5)
            for threshold in (.05,.10):np.testing.assert_array_equal(a[2]<=threshold,b[2]<=threshold)
    for before,after in zip(source[:3],(prediction.mean,prediction.variance,prediction.valid)):
        np.testing.assert_array_equal(before,after)
    assert torch.equal(source[3],prediction.factor)


def test_large_covariance_falls_back_without_dropping_latent_components():
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from fsgp_bgk_se2.shared_terrain_cuda import shared_probability_cuda
    _,_,z,_,prediction=scene('cuda')
    factor=torch.cat((prediction.factor,torch.zeros((*z.shape,512),device='cuda')),dim=2)
    prediction=replace(prediction,factor=factor)
    params=ProbabilityParams(yaw_bins=2)
    occupied=np.zeros(z.shape,bool)
    actual=shared_probability_cuda(prediction,params,occupied)
    expected=shared_probability_cuda(prediction,params,occupied,direct_span=False)
    for a,b in zip(actual,expected):
        for aa,bb in zip(a,b):np.testing.assert_array_equal(aa,bb)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_level_lidar_enclosed_blind_annulus_uses_covariance_not_just_distance(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.set_num_threads(1)
    x,y=np.meshgrid((np.arange(40)-19.5)*.2,(np.arange(40)-19.5)*.2)
    z=.01*x;observed=np.hypot(x,y)>1.4
    raw=np.c_[x[observed],y[observed],z[observed]]
    initial=(abs(x)<.65)&(abs(y)<.55)
    empty=np.zeros(z.shape,bool);params=ProbabilityParams(yaw_bins=8,internal_steps=True)
    def run(distance,extra=None,one_sided=False):
        seen=observed&(x>0) if one_sided else observed
        points=np.c_[x[seen],y[seen],z[seen]]
        if extra is not None:points=np.r_[points,extra]
        return shared_terrain_prediction(np.where(seen,z,np.nan),np.full(z.shape,.0001),
            seen,x,y,points,0.,.2,empty,device=device,
            params=SharedTerrainParams(max_distance_m=distance),
            initial_support=initial,initial_height=z)
    short=run(.75);wide=run(1.25)
    def current(p):
        r,c,upper=shared_geometric_probability(p,params,empty)[0]
        return upper[(r==20)&(c==20)]
    assert len(current(short))==0
    assert len(current(wide))==1 and current(wide)[0]<.05
    np.testing.assert_allclose(short.mean,wide.mean,atol=1e-7)
    np.testing.assert_allclose(short.variance,wide.variance,atol=1e-8)
    assert wide.nearest[wide.completed].max()<=1.25
    assert np.max(wide.variance[wide.valid])<=.04**2
    assert not run(1.25,one_sided=True).valid[x<0].any()
    pit=np.array([[x[20,23],y[20,23],z[20,23]-.4],
                  [x[20,23]+.01,y[20,23],z[20,23]-.4]])
    contradicted=run(1.25,extra=pit)
    assert not contradicted.valid[20,23] and len(current(contradicted))==0


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_shared_posterior_fills_enclosed_blind_spot_without_changing_observation_identity(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    _,_,z,observed,p=scene(device)
    assert p is not None and p.valid[14:16,14:16].all()
    assert not p.completed[observed].any()
    np.testing.assert_allclose(p.mean[p.valid],z[p.valid],atol=2e-5)
    variance=p.factor.square().sum(dim=2).cpu().numpy()+p.discrepancy**2
    np.testing.assert_allclose(p.variance,variance,atol=1e-8)
    result=shared_geometric_probability(p,ProbabilityParams(yaw_bins=8,internal_steps=True),np.zeros(z.shape,bool))
    assert all(len(r)>0 and (upper<.05).all() for r,_,upper in result)


def test_raw_hidden_pit_is_a_veto_not_a_training_height():
    x,y,z,observed,p=scene()
    extra=np.array([[x[15,15],y[15,15],z[15,15]-.3],
                    [x[15,15]+.01,y[15,15]+.01,z[15,15]-.31]])
    *_,b=scene(extra=extra)
    assert p.completed[15,15] and not b.valid[15,15]
    assert b.training_points==p.training_points
    np.testing.assert_allclose(p.mean,b.mean,atol=1e-7)


def test_upper_structure_keeps_independent_occupancy_without_erasing_measured_ground():
    from fsgp_bgk_se2.shared_terrain import raw_measured_evidence
    from fsgp_bgk_se2.obstacle_columns import vertical_columns
    x,y,z,_,_=scene();seen=np.ones(z.shape,bool)
    floor=np.c_[x.ravel(),y.ravel(),z.ravel()]
    high=np.array([[x[15,15],y[15,15],z[15,15]+.95],
                   [x[15,15]+.01,y[15,15],z[15,15]+.96]])
    raw=np.r_[floor,high]
    veto,_=raw_measured_evidence(z,seen,x,y,raw,.2,.18,.75)
    assert not veto[15,15]
    # No clearance proof was supplied: the same high return is STILL a hard
    # occupied column. Removing a ground contradiction cannot authorize motion.
    occupied,_=vertical_columns(raw,z.shape,(x[0,0]-.1,y[0,0]-.1),.2,
                                min_hits=1,ground_elevation=z)
    assert occupied[15,15]
    for offset in (.35,-.4):
        bad=np.r_[floor,high+[0,0,offset-.95]]
        assert raw_measured_evidence(z,seen,x,y,bad,.2,.18,.75)[0][15,15]
    prediction=shared_terrain_prediction(z,np.full(z.shape,.0001),seen,x,y,raw,0.,.2,
        np.zeros(z.shape,bool),positive_veto_height=.75)
    assert prediction.valid[15,15] and not prediction.contradicted[15,15]


@pytest.mark.parametrize('height',[.20,.35,.65])
def test_additive_obstacle_does_not_destroy_real_floor_and_leaves_no_support_ghost(height):
    from fsgp_bgk_se2.shared_terrain import raw_measured_evidence
    from fsgp_bgk_se2.obstacle_columns import vertical_columns
    x,y,z,_,_=scene();seen=np.ones(z.shape,bool);empty=np.zeros(z.shape,bool)
    floor=np.c_[x.ravel(),y.ravel(),z.ravel()]
    hit=np.array([[x[15,15],y[15,15],z[15,15]+height]])
    raw=np.r_[floor,hit]
    occupied,_=vertical_columns(raw,z.shape,(x[0,0]-.1,y[0,0]-.1),.2,
        min_hits=1,min_height=.180001,ground_elevation=z)
    assert occupied[15,15]
    veto,_=raw_measured_evidence(z,seen,x,y,raw,.2,.18,None)
    assert not veto[15,15]
    # When the object has left, the current scan need not hit the covered floor
    # cell. Its prior actual measured height remains available, never fabricated.
    after=floor[np.hypot(floor[:,0]-x[15,15],floor[:,1]-y[15,15])>.7]
    prediction=shared_terrain_prediction(z,np.full(z.shape,.0001),seen,x,y,after,0.,.2,empty,
        positive_veto_height=None)
    assert prediction.valid[15,15] and not prediction.completed[15,15]
    pit=hit-[0,0,height+.4]
    assert raw_measured_evidence(z,seen,x,y,np.r_[after,pit],.2,.18,None)[0][15,15]
    prediction=shared_terrain_prediction(z,np.full(z.shape,.0001),seen,x,y,np.r_[after,pit,pit+[.01,0,0]],
        0.,.2,empty,positive_veto_height=None)
    assert not prediction.valid[15,15] and prediction.contradicted[15,15]


def test_shared_vertical_bias_cancels_in_slope_roughness_and_internal_step_contrasts():
    _,_,z,_,p=scene()
    params=ProbabilityParams(yaw_bins=8,internal_steps=True)
    baseline=shared_geometric_probability(p,params,np.zeros(z.shape,bool))
    factor=torch.cat((p.factor,torch.full((*z.shape,1),.20)),dim=2)
    common=replace(p,factor=factor,variance=p.variance+.20**2)
    other=shared_geometric_probability(common,params,np.zeros(z.shape,bool))
    for a,b in zip(baseline,other):
        np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
        np.testing.assert_allclose(a[2],b[2],atol=1e-7)


def test_one_sided_unknown_and_stale_returns_cannot_create_support():
    x,y,z,_,_=scene()
    observed=x>.7
    points=np.c_[x[observed],y[observed],z[observed]]
    def run(age):return shared_terrain_prediction(np.where(observed,z,np.nan),np.full(z.shape,.0001),
        observed,x,y,points,age,.2,np.zeros(z.shape,bool),params=SharedTerrainParams(radius_m=3.))
    p=run(0.)
    assert p is not None and not p.valid[x<.7].any()
    assert run(.3) is None and run(-.01) is None


def test_measured_step_and_collision_cannot_be_smoothed_into_safe_support():
    x,y,_,_,_=scene();z=np.where(x>0,.5,0.)
    points=np.c_[x.ravel(),y.ravel(),z.ravel()]
    collision=np.zeros(z.shape,bool);collision[15,15]=True
    p=shared_terrain_prediction(z,np.full(z.shape,.0001),np.ones(z.shape,bool),x,y,points,0.,.2,
        collision,params=SharedTerrainParams(radius_m=3.))
    assert p is not None and not p.valid[15,15]
    updates=shared_geometric_probability(p,ProbabilityParams(yaw_bins=8,internal_steps=True),collision)
    assert not any(np.any((r==15)&(c==15)&(upper<.05)) for r,c,upper in updates)


def test_shared_cpu_cuda_predictions_and_geometric_bounds_agree():
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    _,_,z,_,a=scene();*_,b=scene('cuda')
    np.testing.assert_array_equal(a.valid,b.valid)
    np.testing.assert_allclose(a.mean,b.mean,atol=1e-6)
    np.testing.assert_allclose(a.variance,b.variance,atol=1e-8)
    params=ProbabilityParams(yaw_bins=8,internal_steps=True)
    for u,v in zip(shared_geometric_probability(a,params,np.zeros(z.shape,bool)),
                   shared_geometric_probability(b,params,np.zeros(z.shape,bool))):
        np.testing.assert_array_equal(u[0],v[0]);np.testing.assert_array_equal(u[1],v[1])
        np.testing.assert_allclose(u[2],v[2],atol=2e-6)


def test_shared_factor_matches_independent_numpy_weight_space_posterior():
    from fsgp_bgk_se2.shared_terrain import _features
    x,y,z,observed,prediction=scene()
    params=SharedTerrainParams(radius_m=3.)
    xy=np.c_[x[observed],y[observed]]
    center=np.zeros(2);axis=np.arange(-3.,3.1,1.)
    kx,ky=np.meshgrid(axis,axis);knots=np.c_[kx.ravel(),ky.ravel()]
    def design(query):
        a,b=(query/3.).T
        trend=np.c_[np.ones(len(a)),a,b,a*a,a*b,b*b]
        radial=np.exp(-np.sum((query[:,None]-knots[None])**2,axis=2)/(2*.8**2))
        radial/=np.linalg.norm(radial,axis=1)[:,None]
        return np.c_[trend,radial]
    phi=design(xy);query=np.array([[.1,.1],[-.1,-.1],[.5,.3]])
    q=design(query)
    np.testing.assert_allclose(q,_features(torch.as_tensor(query),torch.as_tensor(center),
        torch.as_tensor(knots),params).numpy(),atol=1e-12)
    precision=phi.T@phi/.02**2+np.diag([0.]*6+[1/.04**2]*len(knots))
    covariance=np.linalg.inv(precision)
    mu=q@covariance@phi.T@z[observed]/.02**2
    expected=q@covariance@q.T
    indices=[(15,15),(14,14),(16,17)]
    factor=np.array([prediction.factor[r,c].numpy() for r,c in indices])
    actual_mean=np.array([prediction.mean[r,c] for r,c in indices])
    np.testing.assert_allclose(actual_mean,mu,atol=2e-6)
    np.testing.assert_allclose(factor@factor.T,expected,atol=2e-9)


def test_raised_low_variance_top_cannot_inflate_ground_uncertainty_or_become_safe():
    x,y,z,_,_=scene()
    observed=np.ones(z.shape,bool);z=z.copy();z[17,17]+=.45
    points=np.c_[x.ravel(),y.ravel(),z.ravel()]
    p=shared_terrain_prediction(z,np.full(z.shape,.0001),observed,x,y,points,0.,.2,
        np.zeros(z.shape,bool),params=SharedTerrainParams(radius_m=3.))
    assert p is not None and p.discrepancy==pytest.approx(.01)
    assert not p.valid[17,17]
    assert p.valid[14,14] and p.training_points<observed.sum()


def test_explicit_fixed_initial_assumption_does_not_expand_into_unobserved_frontier():
    x,y,z,_,_=scene();observed=np.hypot(x,y)>1.2
    points=np.c_[x[observed],y[observed],z[observed]]
    initial=(np.abs(x)<.6)&(np.abs(y)<.4)
    def run(mask=None):return shared_terrain_prediction(np.where(observed,z,np.nan),
        np.full(z.shape,.0001),observed,x,y,points,0.,.2,np.zeros(z.shape,bool),
        params=SharedTerrainParams(radius_m=3.),initial_support=mask,
        initial_height=z if mask is not None else None)
    plain=run();assumed=run(initial)
    assert not plain.valid[15,15] and assumed.valid[15,15]
    assert assumed.assumed[15,15] and not assumed.completed[15,15]
    assert not (assumed.valid&~plain.valid&~initial).any()
    np.testing.assert_allclose(plain.mean,assumed.mean)
    assert plain.training_points==assumed.training_points


def test_continuous_passable_slope_is_not_mistaken_for_an_obstacle_top():
    x,y,_,_,_=scene();z=np.tan(.30)*x
    p=shared_terrain_prediction(z,np.full(z.shape,.0001),np.ones(z.shape,bool),x,y,
        np.c_[x.ravel(),y.ravel(),z.ravel()],0.,.2,np.zeros(z.shape,bool))
    assert p is not None and p.valid[15,15] and p.training_points>700
    result=shared_geometric_probability(p,ProbabilityParams(yaw_bins=8,internal_steps=True),np.zeros(z.shape,bool))
    for r,c,upper in result:
        central=(r==15)&(c==15)
        assert central.any() and upper[central][0]<.05


def test_current_occlusion_keeps_original_measured_ground_but_does_not_invent_new_support():
    x,y,z,_,_=scene();current=np.hypot(x,y)>1.2
    raw=np.c_[x[current],y[current],z[current]]
    p=shared_terrain_prediction(z,np.full(z.shape,.0001),np.ones(z.shape,bool),x,y,raw,0.,.2,
        np.zeros(z.shape,bool))
    assert p.valid[15,15] and not p.completed.any()
    q=shared_terrain_prediction(np.where(current,z,np.nan),np.full(z.shape,.0001),current,x,y,raw,0.,.2,
        np.zeros(z.shape,bool))
    assert not q.valid[15,15]


def test_raw_pit_veto_survives_insufficient_gp_training():
    from fsgp_bgk_se2.shared_terrain import raw_measured_veto
    x,y,z,_,_=scene();observed=np.zeros(z.shape,bool);observed[15,15]=True
    raw=np.array([[x[15,15],y[15,15],z[15,15]-.4],
                  [x[15,15]+.01,y[15,15],z[15,15]-.4]])
    assert raw_measured_veto(z,observed,x,y,raw,.2,.18)[15,15]
    assert shared_terrain_prediction(z,np.full(z.shape,.0001),observed,x,y,raw,0.,.2,
        np.zeros(z.shape,bool)) is None


def test_voxel_filtered_ground_can_clear_only_after_three_distinct_scans():
    from fsgp_bgk_se2.shared_terrain import raw_measured_evidence
    from fsgp_bgk_se2.ground_evidence import GroundEvidenceHistory
    x,y,z,_,_=scene();observed=np.ones(z.shape,bool)
    empty=np.zeros(z.shape,bool);blocked=empty.copy();blocked[15,15]=True
    cache=GroundEvidenceHistory(1.)
    assert cache.veto_mask(x,y,blocked,empty,0.)[15,15]
    raw=np.array([[x[15,15],y[15,15],z[15,15]]])
    veto,clear=raw_measured_evidence(z,observed,x,y,raw,.2,.18)
    assert clear.sum()==1 and not veto.any()
    for stamp in (.1,.1,.2):
        assert cache.veto_mask(x,y,veto,clear,stamp)[15,15]
    assert not cache.veto_mask(x,y,veto,clear,.3)[15,15]
    veto,clear=raw_measured_evidence(z,observed,x,y,np.r_[raw,raw+[0,0,.4]],.2,.18)
    assert veto[15,15] and not clear[15,15]
    # A renewed obstacle wins even when ground is simultaneously visible.
    hits=np.r_[raw,raw+[0,0,.4],raw+[.01,0,.4]]
    veto,clear=raw_measured_evidence(z,observed,x,y,hits,.2,.18)
    assert cache.veto_mask(x,y,veto,clear,.4)[15,15]
    # Neither an empty scan nor an inferred (not measured) cell clears memory.
    for stamp,points,measured in ((.5,np.empty((0,3)),observed),
                                   (.6,raw,empty),(.7,raw,empty),(.8,raw,empty)):
        veto,clear=raw_measured_evidence(z,measured,x,y,points,.2,.18)
        assert not clear.any()
        assert cache.veto_mask(x,y,veto,clear,stamp)[15,15]


def test_measured_neighbours_keep_interpolation_supported_during_current_scan_occlusion():
    x,y,z,_,_=scene();observed=np.hypot(x,y)>.6
    current=np.hypot(x,y)>1.3
    raw=np.c_[x[current],y[current],z[current]]
    def run(points):return shared_terrain_prediction(np.where(observed,z,np.nan),
        np.full(z.shape,.0001),observed,x,y,points,0.,.2,np.zeros(z.shape,bool))
    p=run(raw)
    assert p.completed[15,15] and p.nearest[15,15]<.75
    # Keeping observed support is not permission to ignore new negative evidence.
    pit=np.array([[x[15,15],y[15,15],z[15,15]-.4],
                  [x[15,15]+.01,y[15,15],z[15,15]-.4]])
    q=run(np.r_[raw,pit])
    assert not q.valid[15,15] and q.contradicted[15,15]
    assert q.training_points==p.training_points


def test_fused_gpu_matches_tensor_reference_with_strided_factor_and_obstacles():
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    _,_,z,_,p=scene('cuda')
    # Same values, explicitly non-contiguous storage.
    f=p.factor.transpose(0,1).contiguous().transpose(0,1)
    p=replace(p,factor=f);collision=np.zeros(z.shape,bool);collision[18,18]=True
    params=ProbabilityParams(yaw_bins=8,internal_steps=True)
    for a,b in zip(shared_geometric_probability(p,params,collision),
                   shared_geometric_probability(p,params,collision,fused=False)):
        np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
        np.testing.assert_allclose(a[2],b[2],atol=2e-6)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_direct_span_tightens_curved_ground_without_changing_support_or_posterior(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    x,y,z,_,p=scene(device)
    p=replace(p,mean=(z+.24*x*x+.08*y*y).astype(np.float32),discrepancy=.018)
    before=p.mean.copy(),p.factor.clone(),p.valid.copy(),p.variance.copy()
    params=ProbabilityParams(yaw_bins=8,internal_steps=True)
    collision=np.zeros(z.shape,bool);collision[18,18]=True
    old=shared_geometric_probability(p,params,collision,tighten_span=False)
    new=shared_geometric_probability(p,params,collision)
    reference=shared_geometric_probability(p,params,collision,fused=False)
    reductions=[]
    for a,b,c in zip(old,new,reference):
        for row in (b,c):
            np.testing.assert_array_equal(a[0],row[0]);np.testing.assert_array_equal(a[1],row[1])
        np.testing.assert_allclose(b[2],c[2],atol=3e-6)
        assert np.all(b[2]<=a[2]+2e-7)
        reductions.extend(a[2]-b[2])
        assert not np.any((b[0]==18)&(b[1]==18))
    assert max(reductions)>.01
    np.testing.assert_array_equal(p.mean,before[0]);assert torch.equal(p.factor,before[1])
    np.testing.assert_array_equal(p.valid,before[2]);np.testing.assert_array_equal(p.variance,before[3])


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_direct_span_matches_independent_float64_gaussian_contrast_union(device):
    from scipy.special import ndtr
    from fsgp_bgk_se2.probabilistic_risk import _geometry
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    x,y,z,_,p=scene(device)
    p=replace(p,mean=(z+.24*x*x+.08*y*y).astype(np.float32),discrepancy=.018)
    params=ProbabilityParams(yaw_bins=8,internal_steps=True)
    k=2;offsets,weights,limits=_geometry(params,k);w=weights.astype(np.float64)
    r,c=15,15;rr=r+np.array(offsets)[:,0];cc=c+np.array(offsets)[:,1]
    h=p.mean[rr,cc].astype(np.float64);h-=h[offsets.index((0,0))]
    b=p.factor[rr,cc].cpu().numpy().astype(np.float64)
    xy=np.array(offsets)*params.resolution
    discrepancy=np.exp(-np.sum((xy[:,None]-xy[None])**2,axis=2)/(2*p.discrepancy_lengthscale_m**2))*p.discrepancy**2
    def tails(weight,limit):
        mean=weight@h
        sigma=np.sqrt(np.sum((weight@b)**2,axis=1)+np.einsum('ij,jk,ik->i',weight,discrepancy,weight))*np.sqrt(params.variance_scale)
        sigma+=np.abs(weight).sum(axis=1)*params.height_std_floor
        return np.where(abs(mean)>=limit,1,ndtr((-limit-mean)/sigma)+ndtr((mean-limit)/sigma))
    individual=tails(w,limits);count=len(offsets);i,j=np.triu_indices(count,1)
    pairs=tails(w[2+i]-w[2+j],params.max_step)
    expected=min(1,individual[:2].sum()+min(individual[2:2+count].sum(),pairs.sum())+
                 individual[2+count:].sum()+1e-7)
    actual=shared_geometric_probability(p,params,np.zeros(z.shape,bool))[k]
    at=(actual[0]==r)&(actual[1]==c)
    assert at.sum()==1
    assert actual[2][at][0]==pytest.approx(expected,abs=3e-6)


def test_full_span_is_the_same_physical_event_not_a_larger_limit():
    rng=np.random.default_rng(220)
    residual=rng.normal(size=(3000,39))*.06+.035
    span=np.ptp(residual,axis=1)>.18
    half=np.any(abs(residual)>.09,axis=1)
    i,j=np.triu_indices(39,1)
    pair=np.any(abs(residual[:,i]-residual[:,j])>.18,axis=1)
    assert np.array_equal(span,pair)
    assert np.all(~span|half)
    assert np.any(half&~span)


@pytest.mark.parametrize('height,accepted',[(.035,True),(.06,False),(.25,False),(-.25,False)])
def test_support_consistency_uses_likelihood_noise_without_admitting_step_contradictions(height,accepted):
    x,y,z,_,_=scene();z=z.copy();z[15,15]+=height
    observed=np.ones(z.shape,bool);variance=np.full(z.shape,.0001)
    points=np.c_[x.ravel(),y.ravel(),z.ravel()]
    p=shared_terrain_prediction(z,variance,observed,x,y,points,0.,.2,np.zeros(z.shape,bool))
    assert p is not None
    assert bool(p.valid[15,15])==accepted
    # The likelihood still uses 2 cm; neither that model noise nor the
    # geometric limits are lowered to make the outlier pass.
    assert p.discrepancy>=.01
    np.testing.assert_array_equal(variance,np.full(z.shape,.0001))
    if accepted:
        narrow=shared_terrain_prediction(z,variance,observed,x,y,points,0.,.2,np.zeros(z.shape,bool),
            params=SharedTerrainParams(measurement_std_m=.01))
        assert not narrow.valid[15,15]
@pytest.mark.parametrize('device',['cpu','cuda'])
def test_one_sided_ramp_observations_do_not_bow_through_fixed_initial_stance(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    d=np.load(Path(__file__).parent/'data/mid360_ramp_initial_measured.npz')
    p=shared_terrain_prediction(d['height'],d['variance'],d['observed'],d['x'],d['y'],
        d['points'],0.,.2,d['collision'],device=device,
        params=SharedTerrainParams(max_distance_m=1.25,max_extrapolation_m=.75),
        initial_support=d['initial'],initial_height=d['initial_height'],positive_veto_height=None)
    assert p is not None and p.valid[d['initial']].all()
    assert np.all(np.abs(p.mean[d['initial']]-d['initial_height'][d['initial']])<=.05)
    assert np.max(p.variance[d['initial']])<=.08**2
    assert not p.completed[d['initial']].any()
    assert not p.valid[(p.nearest>1.25)&~d['initial']&~d['observed']].any()
    params=ProbabilityParams(front=.42,back=.42,half_width=.313,margin=.1,yaw_bins=36,internal_steps=True)
    x,y,yaw=d['pose'];r=int((y-d['y'][0,0]+.1)/.2);c=int((x-d['x'][0,0]+.1)/.2)
    k=int((yaw%(2*np.pi))/(2*np.pi/36))
    rr,cc,upper=shared_geometric_probability(p,params,d['collision'])[k]
    at_start=upper[(rr==r)&(cc==c)]
    assert len(at_start)==1 and at_start[0]<.05


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_proper_curvature_prior_needs_affine_rank_without_requiring_identifiable_quadratic_trend(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.set_num_threads(1)
    x,y=np.meshgrid((np.arange(80)-39.5)*.1,(np.arange(80)-39.5)*.1)
    z=.02*x+.01*y;seen=(np.abs(y)<.051)&(np.abs(x)<2.8)
    xy=np.c_[x[seen],y[seen]]
    design=np.c_[np.ones(len(xy)),xy,xy[:,0]**2,xy[:,0]*xy[:,1],xy[:,1]**2]
    assert np.linalg.matrix_rank(design[:,:3])==3 and np.linalg.matrix_rank(design)<6
    raw=np.c_[xy,z[seen]]
    p=shared_terrain_prediction(np.where(seen,z,np.nan),np.full(z.shape,.0001),seen,x,y,raw,
        0.,.1,np.zeros(z.shape,bool),device=device)
    assert p is not None and p.valid[seen].all()
    np.testing.assert_allclose(p.mean[seen],z[seen],atol=1e-5)
    assert not p.valid[np.abs(y)>.051].any()  # No inference of a vehicle-wide observed corridor.

