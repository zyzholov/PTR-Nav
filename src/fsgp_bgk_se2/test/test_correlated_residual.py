"""A correlated residual is a proper GP prior, not a smaller uncertainty floor."""
from dataclasses import replace
import numpy as np
import pytest
import torch
from fsgp_bgk_se2.feature_sgp import spatial_sample_indices
from fsgp_bgk_se2.shared_terrain import (SharedTerrainParams,residual_fic_features,
    shared_terrain_prediction,shared_geometric_probability)
from fsgp_bgk_se2.probabilistic_risk import ProbabilityParams


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_fic_prior_and_posterior_match_independent_dense_kernel(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    axis=np.arange(7)*.2;x,y=np.meshgrid(axis,axis);xy=np.c_[x.ravel(),y.ravel()]
    tensor=torch.tensor(xy,dtype=torch.float64,device=device);sigma2=.0004;length=.35
    phi,query,d,qd=residual_fic_features(tensor,tensor,count=12,lengthscale=length,variance=sigma2)
    a,b,diagonal=phi.cpu().numpy(),query.cpu().numpy(),d.cpu().numpy()
    ids=spatial_sample_indices(np.c_[xy,np.zeros(len(xy))],12);u=xy[ids]
    kernel=lambda v,w:sigma2*np.exp(-np.sum((v[:,None]-w[None])**2,axis=2)/(2*length**2))
    cross=kernel(xy,u);gram=kernel(u,u)+np.eye(len(u))*sigma2*1e-6
    q=cross@np.linalg.solve(gram,cross.T)
    np.testing.assert_allclose(a@a.T,q,atol=1e-12)
    np.testing.assert_allclose(np.diag(a@a.T)+diagonal,sigma2,atol=1e-12)
    np.testing.assert_array_equal(a,b)
    assert np.linalg.eigvalsh(q+np.diag(diagonal)).min()>0
    # Independent dense conditioning vs the feature/diagonal Schur equations.
    n=len(xy);noise=.0001;target=np.sin(xy[:,0])*0.02
    full=q+np.diag(diagonal);expected=full@np.linalg.solve(full+np.eye(n)*noise,target)
    weight=1/(noise+diagonal);cov=np.linalg.inv(np.eye(a.shape[1])+a.T@(weight[:,None]*a))
    mean=a@(cov@(a.T@(weight*target)));alpha=diagonal/(noise+diagonal)
    mean=alpha*target+(1-alpha)*mean
    np.testing.assert_allclose(mean,expected,atol=1e-10)
    pcov=((1-alpha)[:,None]*a)@cov@((1-alpha)[:,None]*a).T+np.diag(diagonal*noise/(noise+diagonal))
    np.testing.assert_allclose(pcov,full-full@np.linalg.solve(full+np.eye(n)*noise,full),atol=1e-11)


def fit(device='cpu',extra=None):
    torch.set_num_threads(1)
    x,y=np.meshgrid((np.arange(25)-12)*.2,(np.arange(25)-12)*.2)
    h=.01*x+.015*y;seen=(np.abs(x)>.21)|(np.abs(y)>.21)
    raw=np.c_[x[seen],y[seen],h[seen]]
    if extra is not None:raw=np.r_[raw,extra]
    p=SharedTerrainParams(condition_local_residual=True,condition_full_residual=True,
        residual_inducing_points=64,complete_footprint_halo=True,max_extrapolation_m=.5)
    q=shared_terrain_prediction(np.where(seen,h,np.nan),np.full_like(h,.0001),seen,x,y,raw,0.,.2,
        np.zeros_like(seen),params=p,device=device,positive_veto_height=None,max_step=.1)
    return q,p,x,y,h,seen


def test_correlated_residual_conditions_only_real_sources_and_keeps_negative_evidence():
    q,p,x,y,h,seen=fit()
    assert q is not None and q.training_points==seen.sum()
    assert q.completed[12,12] and q.local_variance[12,12]>0 and q.variance[12,12]>0
    assert not (q.completed&seen).any()
    assert not fit(extra=[[0.,0.,-.4],[.01,.01,-.41]])[0].valid[12,12]
    with pytest.raises(ValueError):SharedTerrainParams(residual_inducing_points=64)
    # New prior keeps an explicit positive diagonal correction, not a
    # rank-deficient smooth surface with false certainty between inducing sites.
    assert np.max(q.local_variance[~seen])>1e-8


def test_correlated_full_model_cpu_gpu_and_real_step_collision_refusal():
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    a,_,x,y,h,seen=fit('cpu');b,*_=fit('cuda')
    np.testing.assert_allclose(a.mean,b.mean,atol=2e-6)
    np.testing.assert_allclose(a.variance,b.variance,atol=3e-7)
    np.testing.assert_allclose(a.local_variance,b.local_variance,atol=2e-7)
    np.testing.assert_array_equal(a.valid,b.valid)
    p=ProbabilityParams(yaw_bins=4,max_step=.1,max_roughness=.125,
        internal_steps=True,internal_step_model='height_jump')
    empty=np.zeros_like(seen)
    for u,v in zip(shared_geometric_probability(a,p,empty),shared_geometric_probability(b,p,empty)):
        np.testing.assert_array_equal(u[0],v[0]);np.testing.assert_array_equal(u[1],v[1])
        np.testing.assert_allclose(u[2],v[2],atol=6e-5)
    raised=a.mean.copy();raised[:,13:]+=.2
    for r,c,u in shared_geometric_probability(replace(a,mean=raised),p,empty):
        at=(r==12)&(c==12);assert at.any() and np.all(u[at]>.99)
    occupied=empty.copy();occupied[12,12]=True
    for r,c,_ in shared_geometric_probability(a,p,occupied):assert not ((r==12)&(c==12)).any()
