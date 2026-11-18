from dataclasses import replace
import numpy as np
import pytest
import torch
from fsgp_bgk_se2.probabilistic_risk import ProbabilityParams,geometric_probability,_geometry
from fsgp_bgk_se2.startup_support import Stance,StartupSupport
from fsgp_bgk_se2.shared_terrain import SharedTerrainPrediction
from fsgp_bgk_se2.conditioned_probability import merge_shared_field,condition_initial_plane
from fsgp_bgk_se2.mixed_terrain import mixed_geometric_probability


def fixture(device):
    torch.set_num_threads(1)
    y,x=np.mgrid[-2:2.01:.2,-2:2.01:.2];h=np.zeros_like(x);v=np.full_like(h,.0001)
    seen=np.hypot(x,y)>.65;collision=np.zeros_like(seen)
    p=ProbabilityParams(yaw_bins=8,internal_steps=True,internal_step_model='height_jump')
    model=StartupSupport();stance=Stance(0,0,0,0,0,0,0)
    model.update(h,v,seen,collision,x,y,stance,0,p)
    prior,plane,var=model.update(h,v,seen,collision,x,y,replace(stance,stamp=2),2,p)
    factor=np.stack((np.full_like(h,.02),.003*x,.003*y),axis=2).astype(np.float32)
    prediction=SharedTerrainPrediction(np.where(prior,.15,h),np.sum(factor**2,axis=2)+.0001,
        np.ones_like(seen),~seen,np.zeros_like(h),torch.tensor(factor,device=device),.01,200,0,
        local_variance=np.full_like(h,.000001))
    return prediction,prior,plane,model,p,collision,(-2.1,-2.1),var,seen,x,y


def lookup(result,k,r,c):
    rr,cc,u=result[k];ids=np.flatnonzero((rr==r)&(cc==c))
    return None if not len(ids) else u[ids[0]]


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_mixed_sources_retain_gp_covariance_without_replacing_initial_heights(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    q,prior,plane,model,p,hit,low,var,seen,x,y=fixture(device)
    out=mixed_geometric_probability(q,prior,plane,model,p,hit,low)
    value=lookup(out,0,10,11)
    assert value is not None and value<.05
    h,v,support,joint=merge_shared_field(np.where(prior,plane,0),np.where(prior,var,.0001),seen|prior,prior,q)
    raw=geometric_probability(h,v,support,p,collision=hit)
    loose=condition_initial_plane(raw,h,v,support,hit,prior,x,y,model,p)
    assert value<=loose['failure_probability_upper'][0][10,11]+1e-6
    # Neither an actual discontinuity nor an unknown interior cell is erased.
    q.mean[:,12:]=.4
    assert lookup(mixed_geometric_probability(q,prior,plane,model,p,hit,low),0,10,11)==1
    missing=next((10+dr,11+dc) for dr,dc in _geometry(p,0)[0] if not prior[10+dr,11+dc])
    q.valid[missing]=False
    assert lookup(mixed_geometric_probability(q,prior,plane,model,p,hit,low),0,10,11) is None
    q.valid[:]=True;hit[10,11]=True
    assert lookup(mixed_geometric_probability(q,prior,plane,model,p,hit,low),0,10,11) is None


def test_mixed_source_cpu_cuda_all_query_bounds_agree():
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    outputs=[]
    for device in ('cpu','cuda'):
        q,prior,plane,model,p,hit,low,*_=fixture(device)
        outputs.append(mixed_geometric_probability(q,prior,plane,model,p,hit,low))
    for a,b in zip(*outputs):
        np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
        np.testing.assert_allclose(a[2],b[2],atol=3e-5)


def test_online_heading_refinement_is_a_subset_of_identical_full_bounds():
    q,prior,plane,model,p,hit,low,*_=fixture('cpu');p=replace(p,yaw_bins=12)
    full=mixed_geometric_probability(q,prior,plane,model,p,hit,low)
    subset=mixed_geometric_probability(q,prior,plane,model,p,hit,low,heading=.1)
    assert 0<sum(bool(len(r)) for r,_,_ in subset)<p.yaw_bins
    for a,b in zip(full,subset):
        if not len(b[0]):continue
        np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
        np.testing.assert_array_equal(a[2],b[2])
