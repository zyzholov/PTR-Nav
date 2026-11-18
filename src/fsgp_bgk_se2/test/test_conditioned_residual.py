"""Joint Gaussian algebra and safety semantics, not empirical calibration."""
import math
import numpy as np
import pytest
import torch
from fsgp_bgk_se2.shared_terrain import SharedTerrainParams,shared_terrain_prediction,_features,shared_geometric_probability
from fsgp_bgk_se2.feature_sgp import spatial_sample_indices
from fsgp_bgk_se2.probabilistic_risk import ProbabilityParams,geometric_probability,_geometry


def scene(device='cpu',full=False):
    torch.set_num_threads(1)
    x,y=np.meshgrid((np.arange(15)-7)*.2,(np.arange(15)-7)*.2)
    h=.01*x+.08*(np.abs(x)<.11)+.02*np.sin(9*y)
    seen=np.ones(h.shape,bool);seen[7,7]=False
    params=SharedTerrainParams(radius_m=1.5,condition_local_residual=True,condition_full_residual=full)
    pred=shared_terrain_prediction(np.where(seen,h,np.nan),np.full(h.shape,.0001),seen,x,y,
        np.c_[x[seen],y[seen],h[seen]],0.,.2,np.zeros(h.shape,bool),params=params,device=device,
        positive_veto_height=None)
    return pred,params,x,y,h,seen


@pytest.mark.parametrize('full',[False,True])
def test_conditioned_local_residual_matches_independent_augmented_joint_precision(full):
    q,p,x,y,h,seen=scene(full=full);assert q is not None
    d2=float(q.local_variance[7,7]);assert d2>0
    xyz=np.c_[x[seen],y[seen],h[seen]];xyz=xyz[spatial_sample_indices(xyz,p.max_sources)]
    xy=torch.as_tensor(xyz[:,:2],dtype=torch.float64)
    reach=math.ceil(p.radius_m/p.basis_spacing_m)*p.basis_spacing_m
    kx,ky=np.meshgrid(np.arange(-reach,reach+1),np.arange(-reach,reach+1))
    knots=torch.as_tensor(np.c_[kx.ravel(),ky.ravel()],dtype=torch.float64)
    phi=_features(xy,torch.zeros(2,dtype=torch.float64),knots,p).numpy();n,m=phi.shape;v=.0001
    # Directly invert the joint coefficient + cell-residual precision. This
    # reference does not use the implementation's Schur/conditional formulas.
    precision=np.block([[phi.T@phi/v+np.diag([0.]*3+[1/p.residual_std_m**2]*(m-3)),phi.T/v],
                        [phi/v,np.eye(n)*(1/v+1/d2)]])
    response=xyz[:,2]-xyz[:,2].mean();rhs=np.r_[phi.T@response/v,response/v]
    cov=np.linalg.inv(precision);mu=cov@rhs
    indices=np.array([36,112,188]);query=np.c_[x.ravel()[indices],y.ravel()[indices]]
    design=np.zeros((len(indices),m+n));design[:,:m]=_features(
        torch.as_tensor(query,dtype=torch.float64),torch.zeros(2,dtype=torch.float64),knots,p).numpy()
    extra=np.zeros(len(indices))
    for j,point in enumerate(query):
        match=np.flatnonzero(np.max(abs(xyz[:,:2]-point),axis=1)<1e-8)
        if len(match):design[j,m+match[0]]=1
        else:extra[j]=d2
    expected=design@cov@design.T+np.diag(extra)
    b=q.factor.reshape(-1,q.factor.shape[-1]).numpy()[indices]
    actual=b@b.T+np.diag(q.local_variance.ravel()[indices])
    np.testing.assert_allclose(q.mean.ravel()[indices],design@mu+xyz[:,2].mean(),atol=2e-6)
    np.testing.assert_allclose(actual,expected,atol=2e-7)
    assert np.linalg.eigvalsh(actual).min()>0
    assert q.training_points==len(xyz) and not q.completed[seen].any()
    assert np.min(q.local_variance[seen])<d2 # observations condition local error; missing cells do not.
    if full:
        assert q.discrepancy==0 and d2>=p.discrepancy_std_m**2-1e-10
        np.testing.assert_allclose(q.variance.ravel()[indices],np.diag(actual),atol=2e-7)


def test_physical_step_limit_is_not_a_second_difference_or_twice_the_bump_height():
    h=np.zeros((31,31),np.float32);h[:,15]=.12;v=np.full_like(h,1e-12);seen=np.ones(h.shape,bool)
    physical=ProbabilityParams(yaw_bins=4,internal_steps=True,internal_step_model='height_jump',height_std_floor=0)
    off,w,_=_geometry(physical,0);n=len(off)
    assert np.all(np.abs(w[n+2:]).sum(axis=1)==2)
    result=geometric_probability(h,v,seen,physical)
    assert result['failure_probability_upper'][0][15,15]<1e-6
    h[:,15]=.25
    assert geometric_probability(h,v,seen,physical)['failure_probability_upper'][0][15,15]==1
    h[:]=np.tan(.65)*(np.arange(31)-15)[None,:]*.2
    assert geometric_probability(h,v,seen,physical)['failure_probability_upper'][0][15,15]==1
    h[:]=0;seen[15,15]=False
    assert np.isnan(geometric_probability(h,v,seen,physical)['failure_probability_upper'][0][15,15])


@pytest.mark.parametrize('step_model',['second_difference','height_jump'])
@pytest.mark.parametrize('full',[False,True])
def test_conditioned_residual_cpu_cuda_and_factored_kernels_agree(step_model,full):
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from fsgp_bgk_se2.shared_terrain_cuda import shared_probability_cuda
    cpu,*_=scene('cpu',full);gpu,*_=scene('cuda',full)
    np.testing.assert_allclose(cpu.mean,gpu.mean,atol=2e-6)
    np.testing.assert_allclose(cpu.variance,gpu.variance,atol=2e-7)
    np.testing.assert_allclose(cpu.local_variance,gpu.local_variance,atol=2e-7)
    np.testing.assert_array_equal(cpu.valid,gpu.valid)
    params=ProbabilityParams(yaw_bins=7,internal_steps=True,internal_step_model=step_model)
    empty=np.zeros(cpu.mean.shape,bool);reference=shared_geometric_probability(cpu,params,empty)
    for result in (shared_probability_cuda(gpu,params,empty),shared_probability_cuda(gpu,params,empty,factorized=False)):
        for a,b in zip(reference,result):
            np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
            np.testing.assert_allclose(a[2],b[2],atol=5e-6)
            for limit in (.05,.1):np.testing.assert_array_equal(a[2]<=limit,b[2]<=limit)


def test_full_residual_requires_conditioning_and_keeps_real_support_and_vetoes():
    with pytest.raises(ValueError):SharedTerrainParams(condition_full_residual=True)
    x,y=np.meshgrid((np.arange(25)-12)*.2,(np.arange(25)-12)*.2)
    h=.01*x;seen=np.ones(h.shape,bool);seen[12,12]=False
    raw=np.c_[x[seen],y[seen],h[seen]]
    params=SharedTerrainParams(condition_local_residual=True,condition_full_residual=True)
    def fit(raw,seen=seen):
        return shared_terrain_prediction(np.where(seen,h,np.nan),np.full(h.shape,.0001),seen,x,y,
            raw,0.,.2,np.zeros(h.shape,bool),params=params,positive_veto_height=None)
    q=fit(raw);assert q.discrepancy==0
    assert q.local_variance[12,12]>=.0001-1e-10
    assert q.completed[12,12] and not q.completed[seen].any()
    assert not fit(np.r_[raw,[[0.,0.,-.4],[.01,0.,-.41]]]).valid[12,12]
    assert fit(raw,np.zeros_like(seen)) is None
