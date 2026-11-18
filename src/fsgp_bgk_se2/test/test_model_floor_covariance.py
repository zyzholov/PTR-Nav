"""Explicit nugget prior; the arbitrary-correlation mode remains available."""
from dataclasses import replace
import numpy as np
import pytest
import torch
from fsgp_bgk_se2.probabilistic_risk import ProbabilityParams
from fsgp_bgk_se2.shared_terrain import (SharedTerrainPrediction,integrate_independent_model_floor,
                                       shared_geometric_probability)


def prediction(factor,local=None,parts=0):
    shape=factor.shape[:2];z=np.zeros(shape,np.float32)
    return SharedTerrainPrediction(mean=z,variance=z.copy(),valid=np.ones(shape,bool),completed=np.zeros(shape,bool),
        nearest=z.copy(),factor=factor,discrepancy=0.,training_points=100,fit_ms=0.,local_variance=local,plane_components=parts)


@pytest.mark.parametrize('scale',[.5,1.,2.])
@pytest.mark.parametrize('parts',[0,7])
def test_original_nugget_sigma_is_preserved_and_unknown_source_cross_correlation_is_still_bounded(scale,parts):
    rng=np.random.default_rng(712);gp=rng.normal(size=(20,7))*.01;plane=rng.normal(size=(20,7))*.02
    factor=np.sqrt(2)*np.c_[gp,plane] if parts else gp
    q=prediction(torch.tensor(factor.reshape(4,5,-1),dtype=torch.float32),parts=parts)
    p=ProbabilityParams(height_std_floor=.005,variance_scale=scale,independent_model_floor=True)
    out,parameters=integrate_independent_model_floor(q,p)
    assert parameters.height_std_floor==0 and p.height_std_floor==.005
    assert out.mean is q.mean and out.factor is q.factor and out.valid is q.valid
    np.testing.assert_allclose(out.local_variance,.005**2/scale*(2 if parts else 1),rtol=1e-6)
    assert q.local_variance is None
    # Same latent coordinates may correlate both original sources arbitrarily.
    for _ in range(60):
        a=rng.normal(size=20);floor2=.005**2*np.dot(a,a)
        u=a@gp;v=a@plane
        actual=np.sqrt(scale*np.dot(u+v,u+v)+floor2) if parts else np.sqrt(scale*np.dot(u,u)+floor2)
        bound=np.sqrt(scale*np.dot(u,u)+floor2)+(np.sqrt(scale*np.dot(v,v))if parts else 0)
        assert actual<=bound+1e-12
    legacy=replace(p,independent_model_floor=False)
    assert integrate_independent_model_floor(q,legacy)==(q,legacy)


@pytest.mark.parametrize('parts',[0,3])
def test_independent_nugget_cpu_and_fused_cuda_use_the_same_prior_without_expanding_support(parts):
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    rng=np.random.default_rng(421);q=prediction(torch.tensor(rng.normal(0,.004,(22,22,8)).astype(np.float32)),
        local=np.full((22,22),.00004,np.float32),parts=parts)
    q.valid[11,11]=False;q.mean=.02*rng.normal(size=(22,22)).astype(np.float32)
    p=ProbabilityParams(yaw_bins=4,max_step=.1,max_roughness=.125,internal_steps=True,
        internal_step_model='height_jump',height_std_floor=.005,independent_model_floor=True)
    occupied=np.zeros_like(q.valid);occupied[13,13]=True
    cpu=shared_geometric_probability(q,p,occupied)
    gpu=replace(q,factor=q.factor.cuda())
    from fsgp_bgk_se2.shared_terrain_cuda import shared_probability_cuda
    for result in (shared_geometric_probability(gpu,p,occupied),shared_probability_cuda(gpu,p,occupied,factorized=False)):
        for a,b in zip(cpu,result):
            np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
            np.testing.assert_allclose(a[2],b[2],atol=4e-5)
            assert not ((b[0]==11)&(b[1]==11)).any()
            assert not ((b[0]==13)&(b[1]==13)).any()
    step=np.zeros_like(q.mean);step[:,12:]=.2
    raised=replace(q,mean=step,valid=np.ones_like(q.valid))
    for r,c,u in shared_geometric_probability(raised,p,np.zeros_like(occupied)):
        assert np.all(u[(r==12)&(c==12)]>.99)
