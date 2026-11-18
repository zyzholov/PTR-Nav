"""Bounded mathematical qualification; not an experimental campaign."""
from dataclasses import replace
import numpy as np
import pytest

from fsgp_bgk_se2.probabilistic_risk import (ProbabilityParams,geometric_probability,
    _geometry,_interval_upper,_envelope_geometry,height_support_radius)


def grid():
    h=np.zeros((41,41),np.float32)
    return h,np.full_like(h,1e-8),np.ones_like(h,dtype=bool)


def test_exact_contact_stencil_matches_independent_convex_feasibility():
    from scipy.optimize import linprog
    for yaw_index in (3,8,13,20):
        p=ProbabilityParams(front=.48,back=.36,half_width=.32,exact_footprint=True)
        theta=(yaw_index+.5)*2*np.pi/p.yaw_bins;c,s=np.cos(theta),np.sin(theta)
        # Feasibility of eight polygon halfspaces; independent of the SAT
        # implementation and includes asymmetric front/back geometry.
        axes=np.array([[c,s],[-c,-s],[-s,c],[s,-c],[1,0],[-1,0],[0,1],[0,-1]])
        actual=set(_geometry(p,yaw_index)[0])
        for dr in range(-5,6):
            for dc in range(-5,6):
                x,y=dc*p.resolution,dr*p.resolution;h=p.resolution/2
                b=[p.front+p.margin,p.back+p.margin,p.half_width+p.margin,p.half_width+p.margin,
                   x+h,-x+h,y+h,-y+h]
                result=linprog([0.,0.],A_ub=axes,b_ub=b,bounds=[(None,None)]*2,method='highs')
                assert ((dr,dc) in actual)==result.success


def test_disjoint_corner_height_cannot_become_a_wheel_contact_but_real_step_remains():
    h,v,valid=grid();h[24,21]=.4
    p=ProbabilityParams(exact_footprint=True,internal_steps=True,internal_step_model='height_jump',height_std_floor=0)
    sat=geometric_probability(h,v,valid,p)
    old=geometric_probability(h,v,valid,replace(p,exact_footprint=False))
    assert old['failure_probability_upper'][3][20,20]==1
    assert sat['failure_probability_upper'][3][20,20]<.001
    h[23,21]=.4
    assert geometric_probability(h,v,valid,p)['failure_probability_upper'][3][20,20]==1


def test_exact_footprint_cpu_cuda_bounds_and_unknown_support_agree():
    cp=pytest.importorskip('cupy')
    h,v,valid=grid();h[:]=.15*np.sin(np.arange(41)[None,:]*.2)
    valid[22,21]=False
    p=ProbabilityParams(yaw_bins=8,exact_footprint=True,internal_steps=True,internal_step_model='height_jump')
    cpu=geometric_probability(h,v,valid,p)
    gpu=geometric_probability(h,v,valid,p,xp=cp)
    for name in cpu:
        np.testing.assert_allclose(np.asarray(cpu[name]),np.asarray(gpu[name]),atol=3e-5,equal_nan=True)


def test_unknown_variance_is_not_zero_uncertainty():
    h,v,valid=grid()
    v[:]=np.nan
    result=geometric_probability(h,v,valid,ProbabilityParams(yaw_bins=4))
    assert all(np.isnan(s).all() for s in result["failure_probability_upper"])
    assert all(not s.any() for s in result["probability_valid"])


def test_hole_and_boundary_remain_unsupported():
    h,v,valid=grid()
    valid[20,20]=False
    result=geometric_probability(h,v,valid,ProbabilityParams(yaw_bins=4))
    for p,s in zip(result["failure_probability_upper"],result["probability_valid"]):
        assert np.isnan(p[20,20]) and s[20,20]==0
        assert np.isnan(p[0,0]) and np.isfinite(p[10,10])


def test_slope_depends_on_heading():
    h,v,valid=grid()
    h[:]=np.tan(.50)*(np.arange(41)-20)[None,:]*.2
    p=geometric_probability(h,v,valid,ProbabilityParams(height_std_floor=0))
    assert p["failure_probability_upper"][0][20,20]<.01
    assert p["failure_probability_upper"][9][20,20]==1


def test_sharp_step_is_not_averaged_into_a_safe_plane():
    h,v,valid=grid()
    h[:,21:]=.25
    result=geometric_probability(h,v,valid,ProbabilityParams(yaw_bins=4,height_std_floor=0))
    assert all(p[20,20]==1 for p in result["failure_probability_upper"])


def test_increased_height_uncertainty_cannot_make_flat_ground_safer():
    h,v,valid=grid()
    params=ProbabilityParams(yaw_bins=4)
    low=geometric_probability(h,v,valid,params)
    high=geometric_probability(h,v+.01,valid,params)
    for a,b in zip(low["failure_probability_upper"],high["failure_probability_upper"]):
        assert b[20,20]>=a[20,20]


def test_constant_height_shift_does_not_change_probability():
    h,v,valid=grid()
    h[:]=.05*(np.arange(41)-20)[None,:]*.2
    params=ProbabilityParams(yaw_bins=4)
    a=geometric_probability(h,v,valid,params)
    b=geometric_probability(h+100,v,valid,params)
    np.testing.assert_allclose(np.asarray(a["failure_probability_upper"]),
                               np.asarray(b["failure_probability_upper"]),atol=1e-5,equal_nan=True)


def test_larger_uncertainty_scale_is_explicit():
    h,v,valid=grid()
    p=ProbabilityParams(yaw_bins=4)
    a=geometric_probability(h,v+.0001,valid,p)
    b=geometric_probability(h,v+.0001,valid,replace(p,variance_scale=4))
    assert b["failure_probability_upper"][0][20,20]>=a["failure_probability_upper"][0][20,20]


def test_confirmed_pillar_blocks_even_without_ground_support():
    h,v,valid=grid()
    valid[:]=False
    obstacle=np.zeros_like(valid)
    obstacle[20,21]=True
    result=geometric_probability(h,v,valid,ProbabilityParams(yaw_bins=4),collision=obstacle)
    assert all(p[20,20]==1 for p in result["failure_probability_upper"])
    assert all(s[20,20]==1 for s in result["probability_valid"])
    assert all(np.isnan(p[5,5]) for p in result["failure_probability_upper"])


def test_contrast_bound_covers_correlated_covariance_and_cancels_shared_offset():
    _,weights,_=_geometry(ProbabilityParams(),0)
    n=weights.shape[1]
    sigma=np.linspace(.008,.02,n)
    upper=(np.abs(weights)@sigma)**2
    for correlation in (np.eye(n),np.ones((n,n)),np.outer((-1.)**np.arange(n),(-1.)**np.arange(n))):
        covariance=sigma[:,None]*correlation*sigma[None,:]
        actual=np.einsum("ij,jk,ik->i",weights,covariance,weights)
        assert np.all(actual<=upper+1e-9)
    np.testing.assert_allclose(weights@np.ones(n),0,atol=2e-7)


def test_gaussian_tail_upper_is_only_used_for_mean_inside_limits():
    from scipy.special import ndtr
    means=np.array([0.,.5,.99,1.,1.5])
    actual=ndtr((-1-means)/.2)+ndtr((means-1)/.2)
    bound=_interval_upper(means,np.full_like(means,.4),1.,np,ndtr)
    assert np.all(bound>=actual)
    np.testing.assert_array_equal(bound[-2:],1)


def test_shared_height_envelope_is_not_an_independence_product():
    h,v,valid=grid();v[:]=1e-4
    p=ProbabilityParams()
    result=geometric_probability(h,v,valid,p)
    radius=height_support_radius(p)
    assert radius==6
    for k in range(p.yaw_bins):
        tail=result["height_error_tail_upper"][k][20,20]
        assert 0<tail<.001
        assert result["failure_probability_upper"][k][20,20]<=_envelope_geometry(p,k)[1]*tail+1e-7
        # Eleven shifted source windows for a straight two-metre path; use
        # the largest marginal tail, never an independent survival product.
    straight=result["height_error_tail_upper"][0][20,20]
    assert (2*radius+1)*(2*radius+11)*straight<.1


@pytest.mark.parametrize("internal",[False,True])
def test_height_envelope_suffices_for_all_geometry_contrasts(internal):
    from scipy.special import ndtri
    rng=np.random.default_rng(713)
    rows,cols=np.indices((41,41))
    h=(.02*np.sin(cols*.3)+.03*np.cos(rows*.2)).astype(np.float32)
    v=(.007+rng.uniform(0,.004,h.shape))**2
    p=ProbabilityParams(yaw_bins=8,internal_steps=internal)
    output=geometric_probability(h,v,np.ones_like(h,bool),p)
    sigma=np.sqrt(v+p.height_std_floor**2)
    for k in range(p.yaw_bins):
        tail=output["height_error_tail_upper"][k][20,20]
        assert 0<tail<1
        t=-ndtri(tail/2)
        offsets,weights,limits=_geometry(p,k)
        for _ in range(8):
            realization=h+rng.uniform(-.999,.999,h.shape)*t*sigma
            heights=np.array([realization[20+r,20+c] for r,c in offsets])
            features=weights@(heights-realization[20,20])
            assert np.all(np.abs(features[:2])<=limits[:2]+1e-6)
            assert np.ptp(features[2:2+len(offsets)])<=p.max_step+1e-6
            if internal:
                assert np.all(np.abs(features[2+len(offsets):])<=p.max_step+1e-6)
                continue
            for r,c in offsets:
                for dr,dc in ((0,1),(1,0)):
                    a,b,d,e=(realization[20+r+j*dr,20+c+j*dc] for j in (-1,0,1,2))
                    assert abs((d-b)-.5*((b-a)+(e-d)))<=p.max_step+1e-6


@pytest.mark.parametrize("internal",[False,True])
def test_absolute_height_radius_certifies_the_same_physics_without_uniform_variance(internal):
    rng=np.random.default_rng(719)
    rows,cols=np.indices((41,41))
    h=(.02*np.sin(cols*.3)+.03*np.cos(rows*.2)).astype(np.float32)
    p=ProbabilityParams(yaw_bins=8,internal_steps=internal)
    result=geometric_probability(h,np.full_like(h,.0001),np.ones_like(h,bool),p)
    for k in range(p.yaw_bins):
        radius=result["height_error_radius_m"][k][20,20]
        assert radius>0
        offsets,weights,limits=_geometry(p,k)
        for _ in range(8):
            realization=h+rng.uniform(-.999,.999,h.shape)*radius
            heights=np.array([realization[20+r,20+c] for r,c in offsets])
            features=weights@(heights-realization[20,20])
            assert np.all(np.abs(features[:2])<=limits[:2]+1e-6)
            assert np.ptp(features[2:2+len(offsets)])<=p.max_step+1e-6
            if internal:
                assert np.all(np.abs(features[2+len(offsets):])<=p.max_step+1e-6)
            else:
                for r,c in offsets:
                    for dr,dc in ((0,1),(1,0)):
                        a,b,d,e=(realization[20+r+j*dr,20+c+j*dc] for j in (-1,0,1,2))
                        assert abs((d-b)-.5*((b-a)+(e-d)))<=p.max_step+1e-6


def test_internal_stencil_cpu_cuda_match_and_detect_leading_edge_step():
    import pytest
    cp=pytest.importorskip("cupy")
    h,v,valid=grid();p=ProbabilityParams(internal_steps=True,yaw_bins=8)
    for column in (18,19,20,21,22):
        ground=h.copy();ground[:,column:]=.25
        cpu=geometric_probability(ground,v,valid,p)
        gpu=geometric_probability(ground,v,valid,p,xp=cp)
        for name in cpu:
            np.testing.assert_allclose(cpu[name],gpu[name],atol=2e-5,rtol=1e-4,equal_nan=True)
        assert cpu["failure_probability_upper"][0][20,20]>.95
    # The new model only requires the footprint itself, not unseen terrain
    # two cells beyond it. A known pillar still overrides missing support.
    valid[:]=False
    offsets,_,_=_geometry(p,0)
    for dr,dc in offsets:valid[20+dr,20+dc]=True
    assert np.isfinite(geometric_probability(h,v,valid,p)["failure_probability_upper"][0][20,20])


@pytest.mark.parametrize("internal",[False,True])
def test_exact_residual_span_proof_and_cuda_parity(internal):
    from scipy.special import ndtr
    from fsgp_bgk_se2.probabilistic_risk import _step_edges
    cp=pytest.importorskip("cupy")
    rows,cols=np.indices((25,25))
    h=(.04*np.sin(cols*.7)+.04*np.cos(rows*.6)).astype(np.float32)
    p=ProbabilityParams(yaw_bins=8,internal_steps=internal)
    valid=np.ones_like(h,bool)
    probabilities=[]
    for scale in (.4,.7,1.):
        v=np.full_like(h,.0001*scale)
        sigma=np.sqrt(v+p.height_std_floor**2)
        cpu=geometric_probability(h,v,valid,p)
        gpu=geometric_probability(h,v,valid,p,xp=cp)
        for name in cpu:
            np.testing.assert_allclose(cpu[name],gpu[name],atol=2e-5,rtol=1e-4,equal_nan=True)
        probabilities.append(np.stack(cpu["failure_probability_upper"]))
        edges,_,_,_=_step_edges(h,sigma,valid,p,np,ndtr)
        for k in range(p.yaw_bins):
            offsets,weights,limits=_geometry(p,k)
            n=len(offsets)
            z=np.array([h[12+r,12+c]-h[12,12] for r,c in offsets])
            std=np.array([sigma[12+r,12+c] for r,c in offsets])
            tails=_interval_upper(weights@z,np.abs(weights)@std,limits,np,ndtr)
            edge=0 if internal else sum(edges[12+r,12+c] for r,c in offsets)
            old=min(1,tails.sum()+edge,_envelope_geometry(p,k)[1]*cpu["height_error_tail_upper"][k][12,12])
            # Independent enumeration of all residual differences, not the
            # production pair cache or an independence assumption.
            residual=weights[2:2+n]
            pairs=np.array([a-b for i,a in enumerate(residual) for b in residual[:i]])
            alternative=min(1,_interval_upper(pairs@z,np.abs(pairs)@std,p.max_step,np,ndtr).sum()
                +tails[:2].sum()+tails[2+n:].sum()+edge)
            expected=min(old,max(.005,alternative)) if old>.005 else old
            np.testing.assert_allclose(cpu["failure_probability_upper"][k][12,12],expected,atol=2e-6)
            if scale==1.:
                assert expected < .8*old
    for a,b in zip(probabilities,probabilities[1:]):
        assert np.all(b[np.isfinite(a)]>=a[np.isfinite(a)]-1e-6)


def test_grouped_span_norm_is_the_same_correlated_contrast_not_a_looser_proxy():
    from fsgp_bgk_se2.probability_cuda import _span_groups
    rng=np.random.default_rng(20261021)
    p=ProbabilityParams(internal_steps=True)
    for k in range(p.yaw_bins):
        offsets,weights,_=_geometry(p,k)
        base,ends,ids,correction=_span_groups(p,k)
        sigma=rng.uniform(.005,.04,len(offsets))
        start=0
        assert len(base)<len(ids)/5
        for group,end in enumerate(ends):
            for pair in range(start,end):
                i,j=ids[pair]
                expected=np.abs(weights[2+i].astype(float)-weights[2+j])@sigma
                actual=base[group]@sigma+correction[pair]@sigma[[i,j]]
                np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=2e-8)
            start=end
        assert start==len(offsets)*(len(offsets)-1)//2
