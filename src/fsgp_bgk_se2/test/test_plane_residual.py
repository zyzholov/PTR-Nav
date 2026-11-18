from dataclasses import replace
import numpy as np
import pytest
import torch
from fsgp_bgk_se2.shared_terrain import (InitialPlaneContext,SharedTerrainParams,shared_terrain_prediction,
    shared_geometric_probability,relative_height_variance,plane_covariance_envelope)
from fsgp_bgk_se2.probabilistic_risk import ProbabilityParams


def predict(device='cpu',*,coeff=None,std=None,condition=True,pit=False,contextual=True,full=False):
    torch.set_num_threads(1)
    x,y=np.meshgrid((np.arange(40)-19.5)*.2,(np.arange(40)-19.5)*.2)
    h=np.zeros_like(x);seen=x>.8;prior=(abs(x)<=.7)&(abs(y)<=.5)
    raw=np.c_[x[seen],y[seen],h[seen]]
    if pit:raw=np.r_[raw,[[.1,.1,-.4],[.11,.11,-.41]]]
    context=InitialPlaneContext(np.zeros(2),np.zeros(3) if coeff is None else np.array(coeff),
        np.array([.005,.005,.015]) if std is None else np.array(std),prior)
    model=SharedTerrainParams(max_distance_m=1.25,max_extrapolation_m=.75,condition_local_residual=condition,
                             condition_full_residual=full)
    q=shared_terrain_prediction(np.where(seen,h,np.nan),np.full_like(h,.0001),seen,x,y,raw,0.,.2,
        np.zeros_like(seen),device=device,params=model,initial_support=prior,
        initial_height=np.zeros_like(h),plane_context=context if contextual else None,positive_veto_height=None)
    return q,prior,context


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('full',[False,True])
def test_measured_plane_drives_one_continuous_field_without_extra_training_data(device,full):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    q,prior,context=predict(device,full=full)
    assert q is not None and q.initial_plane_used and q.valid[prior].all()
    np.testing.assert_allclose(q.mean[prior],0,atol=1e-7)
    baseline,_,_=predict(device,contextual=False,full=full)
    assert q.training_points==baseline.training_points # no added assumption samples
    assert not (q.completed&prior).any()
    p=ProbabilityParams(yaw_bins=8,internal_steps=True,internal_step_model='height_jump')
    updates=shared_geometric_probability(q,p,np.zeros_like(prior))
    r,c,u=updates[0];value=u[(r==20)&(c==22)]
    assert len(value)==1 and value[0]<.05
    bad,_,_=predict(device,pit=True,full=full)
    assert bad.contradicted[20,20] and not bad.valid[20,20]


def test_plane_subtraction_sensitivity_matches_finite_difference():
    q,_,context=predict(condition=False)
    for i in range(3):
        plus=np.zeros(3);plus[i]=1e-4;minus=-plus
        a,_,_=predict(coeff=plus,condition=False);b,_,_=predict(coeff=minus,condition=False)
        finite=(a.mean-b.mean)/2e-4
        analytic=q.factor[:,:,-3+i].numpy()/(np.sqrt(6)*context.coefficient_std_bound[i])
        np.testing.assert_allclose(analytic,finite,atol=2e-4,rtol=2e-4)


def test_covariance_envelopes_dominate_arbitrary_source_and_cross_group_correlation():
    rng=np.random.default_rng(81)
    for _ in range(50):
        a=rng.normal(size=(3,12));sigma=rng.uniform(.001,.03,12)
        root=rng.normal(size=(12,7));root/=np.linalg.norm(root,axis=1)[:,None]
        mapping=a@(sigma[:,None]*root)
        envelope=3*np.diag((abs(a)@sigma)**2)
        assert np.linalg.eigvalsh(envelope-mapping@mapping.T).min()>-1e-12
        shaped=plane_covariance_envelope(a,sigma)
        assert np.linalg.eigvalsh(shaped-mapping@mapping.T).min()>-1e-12
        g=rng.normal(size=(3,7))
        total=(g+mapping)@(g+mapping).T
        assert np.linalg.eigvalsh(2*g@g.T+2*envelope-total).min()>-1e-12


def test_support_permission_is_not_a_low_risk_certificate():
    q,prior,_=predict(std=[1.,1.,.015])
    assert q.valid[prior].all() # original explicit fixed footprint only
    assert np.sqrt(q.variance[prior]).max()>.08 # no uncertainty was hidden
    p=ProbabilityParams(yaw_bins=8,internal_steps=True,internal_step_model='height_jump')
    r,c,u=shared_geometric_probability(q,p,np.zeros_like(prior))[0]
    assert u[(r==20)&(c==20)][0]>.9


def test_relative_support_retains_absolute_datum_uncertainty_and_self_cancels():
    rng=np.random.default_rng(37);a=torch.tensor(rng.normal(size=(8,5)),dtype=torch.float32)
    xy=np.c_[np.arange(8)*.2,np.zeros(8)];ids=np.array([1,1,1,3,3,5,5,7])
    local=torch.full((8,),.0001)
    base=relative_height_variance(a,local,.01,.2,xy,ids)
    extended=torch.cat((a,torch.full((8,1),10.)),dim=1)
    shifted=relative_height_variance(extended,local,.01,.2,xy,ids)
    np.testing.assert_array_equal(base.numpy(),shifted.numpy())
    assert torch.all(extended.square().sum(dim=1)>a.square().sum(dim=1)+99)
    assert torch.all(base[[1,3,5,7]]==0)


@pytest.mark.parametrize('full',[False,True])
def test_plane_residual_cpu_cuda_posterior_and_full_heading_bounds_agree(full):
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    cpu,prior,_=predict('cpu',full=full);gpu,_,_=predict('cuda',full=full)
    np.testing.assert_array_equal(cpu.valid,gpu.valid)
    np.testing.assert_allclose(cpu.mean,gpu.mean,atol=2e-6)
    np.testing.assert_allclose(cpu.variance,gpu.variance,atol=2e-7)
    p=ProbabilityParams(yaw_bins=8,internal_steps=True,internal_step_model='height_jump')
    for a,b in zip(shared_geometric_probability(cpu,p,np.zeros_like(prior)),
                   shared_geometric_probability(gpu,p,np.zeros_like(prior))):
        np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
        np.testing.assert_allclose(a[2],b[2],atol=3e-5)


def test_source_partition_tightens_only_the_certificate_not_the_latent_model():
    q,prior,_=predict(std=[.025,.015,.035],full=True)
    assert q.plane_components==3
    old=replace(q,plane_components=0)
    np.testing.assert_array_equal(q.mean,old.mean)
    np.testing.assert_array_equal(q.variance,old.variance)
    np.testing.assert_array_equal(q.factor.numpy(),old.factor.numpy())
    p=ProbabilityParams(yaw_bins=8,internal_steps=True,internal_step_model='height_jump')
    old_bounds=shared_geometric_probability(old,p,np.zeros_like(prior))
    bounds=shared_geometric_probability(q,p,np.zeros_like(prior))
    reduction=0.
    for a,b in zip(old_bounds,bounds):
        np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
        assert np.all(b[2]<=a[2]+3e-6)
        reduction=max(reduction,float(np.max(a[2]-b[2])))
    assert reduction>.01
    # A real, unacceptable jump remains unacceptable; the covariance split
    # cannot excuse a mean constraint violation or introduce new support.
    bad_height=q.mean.copy();bad_height[:,30:]+=.4
    bad=replace(q,mean=bad_height)
    for r,c,u in shared_geometric_probability(bad,p,np.zeros_like(prior)):
        near=(c>=29)&(c<=30)
        assert near.any() and np.all(u[near]>.99)


@pytest.mark.parametrize('factorized',[False,True])
def test_partitioned_cuda_including_pair_span_matches_scalar_cpu(factorized):
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from fsgp_bgk_se2.shared_terrain_cuda import shared_probability_cuda
    q,prior,_=predict('cuda',std=[.025,.015,.035],full=True)
    p=ProbabilityParams(yaw_bins=8,internal_steps=True,internal_step_model='height_jump')
    expected=shared_geometric_probability(q,p,np.zeros_like(prior),fused=False)
    actual=shared_probability_cuda(q,p,np.zeros_like(prior),factorized=factorized)
    for a,b in zip(expected,actual):
        np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
        np.testing.assert_allclose(a[2],b[2],atol=5e-5)


def test_per_contrast_triangle_bound_allows_arbitrary_cross_source_correlation():
    rng=np.random.default_rng(9201)
    # Independent latent coordinates are shared here across BOTH sources,
    # making their cross correlation arbitrary, including perfect alignment.
    for _ in range(100):
        gp=rng.normal(size=(9,7));initial=rng.normal(size=(9,7))
        a=rng.normal(size=9);u=a@gp;v=a@initial
        exact=np.linalg.norm(u+v)
        triangle=np.linalg.norm(u)+np.linalg.norm(v)
        young=np.sqrt(2*(np.dot(u,u)+np.dot(v,v)))
        assert exact<=triangle+1e-12 and triangle<=young+1e-12
    # The upper bound is attained by perfectly correlated aligned errors.
    u=np.array([.03,-.02,.01]);v=2*u
    assert np.isclose(np.linalg.norm(u+v),np.linalg.norm(u)+np.linalg.norm(v))
