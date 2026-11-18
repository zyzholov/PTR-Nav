from dataclasses import replace
import numpy as np
import pytest
from fsgp_bgk_se2.startup_support import StartupSupport,Stance
from fsgp_bgk_se2.probabilistic_risk import ProbabilityParams,geometric_probability,_geometry
from fsgp_bgk_se2.conditioned_probability import (condition_initial_plane,conditional_geometry,
    conditional_evaluation_geometry,conditional_union_upper,merge_shared_field)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_gp_extrapolation_cannot_replace_fixed_stance_without_measurement(device):
    from types import SimpleNamespace
    xp=np if device=='cpu' else pytest.importorskip('cupy')
    y,x=np.mgrid[-2:2.01:.2,-2:2.01:.2];h=np.zeros_like(x);v=np.full_like(h,.0001)
    measured=np.hypot(x,y)>.65;hit=np.zeros_like(measured)
    params=ProbabilityParams(yaw_bins=36,internal_steps=True,internal_step_model='height_jump')
    model=StartupSupport();stance=Stance(0,0,0,0,0,0,0)
    model.update(h,v,measured,hit,x,y,stance,0,params)
    prior,plane,var=model.update(h,v,measured,hit,x,y,replace(stance,stamp=2),2,params)
    prediction=SimpleNamespace(mean=np.where(prior,.15,h),variance=np.full_like(h,.0002),
        valid=np.ones_like(measured))
    source_h=np.where(prior,plane,h);source_v=np.where(prior,var,v)
    height,variance,support,joint=merge_shared_field(source_h,source_v,measured|prior,prior,prediction)
    np.testing.assert_array_equal(height[prior],plane[prior]);np.testing.assert_array_equal(variance[prior],var[prior])
    assert not joint[prior].any() and joint[measured].all()
    layers=geometric_probability(height,variance,support,params,collision=hit,xp=xp)
    out=condition_initial_plane(layers,height,variance,support,hit,prior,x,y,model,params,xp=xp)
    # A mixed footprint just outside the initial centre is bounded using its
    # actual two sources, not the inconsistent extrapolated initial heights.
    assert out['failure_probability_upper'][2][10,11]<.05
    assert out['probability_pose_event'][2][10,11]==1
    # No unknown cell, pit or collision receives an exception from the plane.
    support[10,14]=False;hit[10,11]=True
    layers=geometric_probability(height,variance,support,params,collision=hit,xp=xp)
    out=condition_initial_plane(layers,height,variance,support,hit,prior,x,y,model,params,xp=xp)
    assert out['failure_probability_upper'][2][10,11]==1


@pytest.mark.parametrize("device",["cpu","cuda"])
def test_shared_plane_keeps_unknown_collision_and_variance_semantics(device):
    xp=np if device=="cpu" else pytest.importorskip("cupy")
    y,x=np.mgrid[-2:2.01:.2,-2:2.01:.2]
    h=.02*x+.01*y;v=np.full_like(h,.0001);seen=np.hypot(x,y)>.65;hit=np.zeros_like(seen)
    p=ProbabilityParams(yaw_bins=8,internal_steps=True)
    stance=Stance(0,0,0,0,0,0,0);model=StartupSupport()
    model.update(h,v,seen,hit,x,y,stance,0,p)
    mask,mean,var=model.update(h,v,seen,hit,x,y,replace(stance,stamp=2),2,p)
    height=np.where(mask,mean,h);variance=np.where(mask,var,v);support=seen|mask
    raw=geometric_probability(height,variance,support,p,collision=hit,xp=xp)
    before=np.stack(raw["failure_probability_upper"]).copy()
    raw["probability_pose_event"]=[np.zeros_like(h) for _ in range(p.yaw_bins)]
    out=condition_initial_plane(raw,height,variance,support,hit,mask,x,y,model,p,xp=xp)
    # Only the initial heading is assumed supported. Rotation into a wider
    # unknown footprint must not be granted by this test fixture.
    assert out["probability_pose_event"][0][10,10]==1
    assert np.all(np.stack(out["failure_probability_upper"])<=before+1e-7,where=np.isfinite(before))
    assert out["failure_probability_upper"][0][10,10]<.05
    # A missing forward cell never receives support from the initial plane.
    support[10,14]=False;hit[10,11]=True
    raw=geometric_probability(height,variance,support,p,collision=hit,xp=xp)
    raw["probability_pose_event"]=[np.zeros_like(h) for _ in range(p.yaw_bins)]
    out=condition_initial_plane(raw,height,variance,support,hit,mask,x,y,model,p,xp=xp)
    assert all(a[10,10]==1 for a in out["failure_probability_upper"])
    assert all(a[10,10]==0 for a in out["probability_pose_event"])


@pytest.mark.parametrize("rough",[False,True])
@pytest.mark.parametrize('step_model',['second_difference','height_jump'])
def test_fused_conditional_matches_reference_over_all_poses(rough,step_model):
    import copy
    cp=pytest.importorskip("cupy")
    y,x=np.mgrid[-3:3.01:.2,-3:3.01:.2]
    h=.02*x+.01*y;v=np.full_like(h,.0001);seen=np.hypot(x,y)>.65;hit=np.zeros_like(seen)
    if rough:
        h+=.008*np.sin(3*x)*np.cos(2*y)
        v*=1+.8*np.sin(x+y)**2
    p=ProbabilityParams(yaw_bins=36,internal_steps=True,internal_step_model=step_model)
    stance=Stance(0,0,0,0,0,0,0);model=StartupSupport()
    model.update(h,v,seen,hit,x,y,stance,0,p)
    mask,mean,var=model.update(h,v,seen,hit,x,y,replace(stance,stamp=2),2,p)
    height=np.where(mask,mean,h);variance=np.where(mask,var,v);support=seen|mask
    support[8:11,12:16]=False;hit[17,17]=True
    raw=geometric_probability(height,variance,support,p,collision=hit)
    outputs=[condition_initial_plane(copy.deepcopy(raw),height,variance,support,hit,mask,x,y,model,p,xp=xp)
             for xp in (np,cp)]
    for key in ("failure_probability_upper","probability_pose_event","height_error_tail_upper"):
        np.testing.assert_allclose(outputs[0][key],outputs[1][key],atol=2e-6,rtol=2e-4,equal_nan=True)


def test_residual_pairs_are_the_physical_span_not_half_absolute_residuals():
    p=ProbabilityParams(yaw_bins=8,internal_steps=True)
    offsets,old,_=_geometry(p,0);_,features,limits=conditional_geometry(p,0)
    n=len(offsets);pairs=n*(n-1)//2
    h=np.zeros(n);h[offsets.index((0,0))]=.13
    residual=old[2:2+n]@h
    assert np.max(abs(residual))>p.max_step/2
    assert np.ptp(residual)<p.max_step
    np.testing.assert_allclose(np.max(abs(features[2:2+pairs]@h)),np.ptp(residual),atol=1e-7)
    assert np.all(limits[2:2+pairs]==np.float32(p.max_step))
    h[offsets.index((0,0))]=.4
    assert np.any(abs(features[2:2+pairs]@h)>limits[2:2+pairs])


def test_residual_pair_limits_retain_slope_and_step_constraints():
    p=ProbabilityParams(yaw_bins=8,internal_steps=True)
    offsets,old,limits=_geometry(p,0);_,features,new_limits=conditional_geometry(p,0)
    n=len(offsets)
    np.testing.assert_array_equal(features[:2],old[:2])
    np.testing.assert_array_equal(new_limits[:2],limits[:2])
    np.testing.assert_array_equal(features[2+n*(n-1)//2:],old[2+n:])
    np.testing.assert_array_equal(new_limits[2+n*(n-1)//2:],limits[2+n:])


def test_centered_residual_envelope_is_a_sufficient_span_condition():
    from scipy.special import ndtr
    from fsgp_bgk_se2.probabilistic_risk import _interval_upper
    p=ProbabilityParams(yaw_bins=8,internal_steps=True)
    offsets,features,limits=conditional_evaluation_geometry(p,0)
    n=len(offsets);rng=np.random.default_rng(19)
    heights=rng.normal(0,.02,(n,32));sigma=rng.uniform(.005,.04,(n,32))
    means=features@heights;stds=abs(features)@sigma
    upper=conditional_union_upper(means,stds,limits,n,np,ndtr)
    previous=np.minimum(1,_interval_upper(means[:-n],stds[:-n],limits[:-n,None],np,ndtr).sum(axis=0))
    assert np.all(upper<=previous+1e-12)
    increased=conditional_union_upper(means,1.5*stds,limits,n,np,ndtr)
    assert np.all(increased>=upper-1e-12)
    center=(means[-n:].max(axis=0)+means[-n:].min(axis=0))/2
    residual=center+rng.uniform(-p.max_step/2,p.max_step/2,(n,32))
    assert np.all(np.ptp(residual,axis=0)<=p.max_step)
    # Physical violations do not disappear under the sufficient alternative.
    bad=np.zeros((n,1));bad[offsets.index((0,0))]=.4
    assert conditional_union_upper(features@bad,abs(features)@np.full((n,1),.005),
                                   limits,n,np,ndtr)[0]==1


def test_external_step_contract_does_not_use_partial_conditional_proof():
    p=ProbabilityParams(yaw_bins=8,internal_steps=False)
    h=np.zeros((12,12));layers={"failure_probability_upper":[np.ones_like(h) for _ in range(8)]}
    out=condition_initial_plane(layers,h,h,h.astype(bool),h.astype(bool),h.astype(bool),h,h,None,p)
    assert np.all(np.stack(out["failure_probability_upper"])==1)
    assert not np.any(out["probability_pose_event"])


