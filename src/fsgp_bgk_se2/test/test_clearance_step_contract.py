"""Separate local step ability from the whole-footprint underbody span."""
from dataclasses import replace
import numpy as np
import pytest
import torch
from fsgp_bgk_se2.probabilistic_risk import ProbabilityParams,_geometry,geometric_probability
from fsgp_bgk_se2.shared_terrain import SharedTerrainPrediction,shared_geometric_probability


def parameters():
    return ProbabilityParams(yaw_bins=4,max_pitch=np.deg2rad(10),max_roll=np.deg2rad(10),
        max_step=.1,max_roughness=.125,height_std_floor=0.,internal_steps=True,
        internal_step_model='height_jump',exact_footprint=True)


def prediction(height,device='cpu'):
    shape=height.shape;zero=np.zeros(shape,np.float32)
    return SharedTerrainPrediction(mean=height,variance=zero,valid=np.ones(shape,bool),
        completed=np.zeros(shape,bool),nearest=zero,factor=torch.zeros((*shape,1),device=device),
        discrepancy=0.,training_points=100,fit_ms=0.,local_variance=zero)


def raster(pred,p,collision=None):
    shape=pred.mean.shape;result=np.full((p.yaw_bins,*shape),np.nan,np.float32)
    for k,(r,c,u) in enumerate(shared_geometric_probability(pred,p,np.zeros(shape,bool)if collision is None else collision)):
        result[k,r,c]=u
    return result


def test_constraints_have_distinct_limits_and_legacy_is_unchanged():
    p=parameters();offsets,_,limits=_geometry(p,0);n=len(offsets)
    np.testing.assert_allclose(limits[2:2+n],.125/2)
    np.testing.assert_allclose(limits[2+n:],.1)
    legacy=replace(p,max_roughness=None);_,w,old=_geometry(legacy,0)
    np.testing.assert_array_equal(w,_geometry(replace(p,max_roughness=.1),0)[1])
    np.testing.assert_array_equal(old,_geometry(replace(p,max_roughness=.1),0)[2])


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_smooth_local_relief_and_too_high_true_step_remain_different(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    axis=(np.arange(25)-12)*.2;x,y=np.meshgrid(axis,axis)
    relief=(.11*np.exp(-(x*x+y*y)/.035)).astype(np.float32)
    p=parameters();new=raster(prediction(relief,device),p);old=raster(prediction(relief,device),replace(p,max_roughness=None))
    assert np.all(new[:,12,12]<1e-5)
    assert np.all(old[:,12,12]>.99)
    step=np.where(x>=0,.11,0).astype(np.float32)
    unsafe=raster(prediction(step,device),p)
    assert np.all(unsafe[:,12,12]>.99) # 110mm discontinuity still exceeds 100mm
    obstacle=np.zeros_like(x,bool);obstacle[12,12]=True
    blocked=raster(prediction(relief,device),p,obstacle)
    # The actual runtime column/body guard remains separate; even this
    # conservative terrain occupancy cannot be erased by clearance changes.
    assert not np.isfinite(blocked[:,12,12]).any()


def test_cpu_gpu_shared_and_marginal_models_use_the_same_separate_thresholds():
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    import cupy as cp
    rng=np.random.default_rng(51);height=rng.normal(0,.008,(22,22)).astype(np.float32)
    p=replace(parameters(),height_std_floor=.005);pred=prediction(height)
    pred=replace(pred,factor=torch.tensor(rng.normal(0,.002,(22,22,5)),dtype=torch.float32),local_variance=np.full_like(height,.00004))
    cpu=raster(pred,p);gpu=raster(replace(pred,factor=pred.factor.cuda()),p)
    np.testing.assert_array_equal(np.isfinite(cpu),np.isfinite(gpu))
    np.testing.assert_allclose(cpu[np.isfinite(cpu)],gpu[np.isfinite(cpu)],rtol=2e-4,atol=8e-5)
    observed=np.ones_like(height,bool);variance=np.full_like(height,.0001)
    a=geometric_probability(height,variance,observed,p)
    b=geometric_probability(height,variance,observed,p,xp=cp)
    for k in range(p.yaw_bins):
        av=np.asarray(a['failure_probability_upper'][k]);bv=cp.asnumpy(b['failure_probability_upper'][k])
        np.testing.assert_array_equal(np.isfinite(av),np.isfinite(bv))
        np.testing.assert_allclose(av[np.isfinite(av)],bv[np.isfinite(av)],atol=2e-4)
