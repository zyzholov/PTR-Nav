"""Pi-periodic terrain proofs retain all requested SE(2) poses and events."""
from dataclasses import replace
import numpy as np
import pytest
import torch
from fsgp_bgk_se2.probabilistic_risk import ProbabilityParams
from fsgp_bgk_se2.shared_terrain import SharedTerrainPrediction,shared_geometric_probability
from fsgp_bgk_se2.shared_terrain_cuda import opposite_heading_count,shared_probability_cuda


@pytest.mark.parametrize('yaws,front,back,expected',[(36,.52,.52,18),(8,.42,.42,4),
    (7,.42,.42,7),(36,.521,.52,36),(1,.42,.42,1)])
def test_only_exact_front_back_symmetric_even_heading_geometry_is_reused(yaws,front,back,expected):
    p=ProbabilityParams(yaw_bins=yaws,front=front,back=back,internal_steps=True,exact_footprint=True)
    assert opposite_heading_count(p)==expected


@pytest.mark.parametrize('yaws,asymmetric',[(8,False),(7,False),(8,True)])
@pytest.mark.parametrize('bounded',[False,True])
def test_reused_proof_matches_full_headings_and_cpu_with_partial_masks_unknowns_and_columns(yaws,asymmetric,bounded):
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    pytest.importorskip('cupy');torch.set_num_threads(1)
    rng=np.random.default_rng(518)
    shape=(15,17);x,y=np.meshgrid((np.arange(shape[1])-8)*.2,(np.arange(shape[0])-7)*.2)
    z=(.025*x-.015*y+.035*np.sin(3*x)*np.cos(2*y)).astype(np.float32)
    factor=rng.normal(0,.004,(*shape,7)).astype(np.float32)
    factor[...,0]=.012;factor[...,1]=.003*x;factor[...,2]=.003*y
    local=np.full(shape,.000025,np.float32);valid=np.ones(shape,bool);valid[7,6]=False
    column=np.zeros(shape,bool);column[5,11]=True
    nuisance=torch.tensor(np.stack((.002*x,.003*y),axis=-1).astype(np.float32),device='cuda')if bounded else None
    p=SharedTerrainPrediction(z,(factor**2).sum(axis=-1)+local,valid,np.zeros(shape,bool),
        np.zeros(shape,np.float32),torch.tensor(factor,device='cuda'),.003,0,0.,
        local_variance=local,bounded_mean_factor=nuisance)
    physical=ProbabilityParams(yaw_bins=yaws,front=.42,back=.47 if asymmetric else .42,
        max_step=.10,max_roughness=.125,internal_steps=True,internal_step_model='height_jump',
        independent_model_floor=True,exact_footprint=True)
    query=rng.random((yaws,*shape))>.55
    # Exercise requests that ONLY the opposite heading makes.
    query[:yaws//2,8:10,8:10]=False;query[yaws//2:,8:10,8:10]=True
    before=query.copy();mean=p.mean.copy();cov=p.factor.clone();support=p.valid.copy()
    actual=shared_probability_cuda(p,physical,column,query_mask=query)
    full=shared_probability_cuda(p,physical,column,query_mask=query,heading_symmetry=False)
    cpu=replace(p,factor=p.factor.cpu(),bounded_mean_factor=None if nuisance is None else nuisance.cpu())
    reference=shared_geometric_probability(cpu,physical,column,query_mask=query)
    for k,(a,b,c)in enumerate(zip(actual,full,reference)):
        for other in (b,c):
            np.testing.assert_array_equal(a[0],other[0]);np.testing.assert_array_equal(a[1],other[1])
            np.testing.assert_allclose(a[2],other[2],rtol=2e-4,atol=2e-6)
            np.testing.assert_array_equal(a[2]<=.05,other[2]<=.05)
        assert query[k,a[0],a[1]].all()
    np.testing.assert_array_equal(query,before);np.testing.assert_array_equal(p.mean,mean)
    np.testing.assert_array_equal(p.valid,support);assert torch.equal(p.factor,cov)


def test_recorded_dynamic_escape_pose_in_second_heading_half_retains_its_full_covariance_proof():
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from test_refinement_navigation_frontier import case
    data,meta,p,physical,mask=case('cuda');target=tuple(meta['target'])
    assert target[0]>=physical.yaw_bins//2
    actual=shared_probability_cuda(p,physical,data['collision'],query_mask=mask)
    full=shared_probability_cuda(p,physical,data['collision'],query_mask=mask,heading_symmetry=False)
    for a,b in zip(actual,full):
        np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
        np.testing.assert_allclose(a[2],b[2],rtol=1e-4,atol=1e-6)
    assert len(actual[target[0]][2])==1 and actual[target[0]][2][0]<.0005
