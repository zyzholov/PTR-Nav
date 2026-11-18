import numpy as np
import pytest
from dataclasses import replace
from fsgp_bgk_se2.startup_support import StartupSupport,Stance
from fsgp_bgk_se2.probabilistic_risk import ProbabilityParams,geometric_probability


def scene():
    y,x=np.mgrid[-2:2.01:.2,-2:2.01:.2]
    h=.02*x+.01*y;v=np.full_like(h,.0001)
    observed=np.hypot(x,y)>.65
    return h,v,observed,np.zeros_like(observed),x,y


@pytest.mark.parametrize('yaw',[0.,.31,-.9])
def test_bootstrap_retires_only_after_full_departure_and_cannot_reactivate(yaw):
    y,x=np.mgrid[-4:4.01:.2,-4:4.01:.2];h=.02*x+.01*y;v=np.full_like(h,.0001)
    seen=np.hypot(x,y)>.65;hit=np.zeros_like(seen)
    p=ProbabilityParams(internal_steps=True,exact_footprint=True,margin=0.,front=.52,back=.52,half_width=.413)
    pose=Stance(0.,0.,yaw,0.,0.,0.,0.)
    model=StartupSupport(retire_after_departure=True,require_geometry=True)
    model.set_footprint(.68,.58,.50)
    model.update(h,v,seen,hit,x,y,pose,0.,p)
    initial=model.update(h,v,seen,hit,x,y,replace(pose,stamp=2.),2.,p)[0]
    anchor=replace(model.anchor);plane=model.plane.copy()
    assert initial.any()
    assert model.speed_limits(x,y)[np.hypot(x,y)<1.5].max()==.25
    # A displaced centre alone is insufficient; the trailing stencil overlaps.
    partial=replace(pose,x=.8*np.cos(yaw),y=.8*np.sin(yaw),stamp=3.)
    np.testing.assert_array_equal(initial,model.update(h,v,seen,hit,x,y,partial,3.,p)[0])
    assert not model.retired
    departed=replace(pose,x=2.*np.cos(yaw),y=2.*np.sin(yaw),stamp=4.)
    # Stale/missing odometry cannot certify even a geometric departure.
    assert model.update(h,v,seen,hit,x,y,None,3.2,p)[0].any()
    assert model.update(h,v,seen,hit,x,y,replace(departed,stamp=2.4),3.4,p)[0].any()
    mask,mean,variance=model.update(h,v,seen,hit,x,y,departed,4.,p)
    assert model.retired and not model.revoked
    assert not mask.any() and mean is None and variance is None
    assert np.all(model.speed_limits(x,y)==100.)
    assert model.anchor==anchor
    np.testing.assert_array_equal(model.plane,plane)
    model.arm_initial_goal()
    for t in (5.,6.):
        mask,mean,variance=model.update(h,v,seen,hit,x,y,replace(pose,stamp=t),t,p)
        assert not mask.any() and mean is None and variance is None
    # No supplied height, support or occupancy data was changed.
    np.testing.assert_array_equal(h,.02*x+.01*y)
    assert not hit.any()


def test_runtime_initial_support_requires_and_freezes_the_actual_protected_body():
    from fsgp_bgk_se2.probabilistic_risk import _geometry
    h,v,seen,hit,x,y=scene();seen=x>1.;p=ProbabilityParams(internal_steps=True)
    pose=Stance(.099,-.005,-.0001,0.,0.,0.,0.,.28)
    model=StartupSupport(allow_one_sided=True,require_goal=True,require_geometry=True)
    model.arm_initial_goal()
    assert not model.update(h,v,seen,hit,x,y,pose,0,p)[0].any()
    assert model.reason=='waiting for planner protected body geometry'
    # The values already include the planner's single 10cm protection margin.
    model.set_footprint(.68,.58,.50)
    model.update(h,v,seen,hit,x,y,pose,0,p)
    mask=model.update(h,v,seen,hit,x,y,replace(pose,stamp=2),2,p)[0]
    assert mask.any()
    row=int(np.floor((pose.y-y[0,0]+.1)/.2));col=int(np.floor((pose.x-x[0,0]+.1)/.2))
    for k in (0,35):
        assert all((mask|seen)[row+dr,col+dc] for dr,dc in _geometry(p,k)[0])
    np.testing.assert_array_equal(mask,model.update(h,v,seen,hit,x,y,replace(pose,x=1.,stamp=3),3,p)[0])
    model.set_footprint(.78,.58,.50)
    assert model.revoked and not model.update(h,v,seen,hit,x,y,replace(pose,stamp=4),4,p)[0].any()


def test_goal_latch_does_not_freeze_an_early_slam_anchor_or_follow_later_goals():
    h,v,seen,hit,x,y=scene();p=ProbabilityParams(internal_steps=True)
    model=StartupSupport(require_goal=True);pose=Stance(0,0,0,0,0,0,0)
    for stamp in (0,2,5):
        assert not model.update(h,v,seen,hit,x,y,replace(pose,stamp=stamp),stamp,p)[0].any()
    assert model.anchor is None
    model.arm_initial_goal()
    pose=replace(pose,x=-.15,stamp=6)
    assert not model.update(h,v,seen,hit,x,y,pose,6,p)[0].any()
    fixed=model.update(h,v,seen,hit,x,y,replace(pose,stamp=8),8,p)[0]
    assert fixed.any() and model.anchor.x==-.15
    model.arm_initial_goal()
    moved=model.update(h,v,seen,hit,x,y,replace(pose,x=1,stamp=9),9,p)[0]
    np.testing.assert_array_equal(fixed,moved)


def test_motion_before_goal_latch_cannot_create_a_new_initial_safe_bubble():
    h,v,seen,hit,x,y=scene();p=ProbabilityParams(internal_steps=True)
    model=StartupSupport(require_goal=True);model.observe_command(.2);model.arm_initial_goal()
    for stamp in (0,2,5):
        assert not model.update(h,v,seen,hit,x,y,Stance(0,0,0,0,0,0,stamp),stamp,p)[0].any()
    assert model.revoked and model.anchor is None


def test_initial_assumption_is_frozen_revocable_and_does_not_touch_observations():
    h,v,seen,hit,x,y=scene();p=ProbabilityParams(internal_steps=True)
    model=StartupSupport();pose=Stance(0,0,0,0,0,0,0)
    assert not model.update(h,v,seen,hit,x,y,pose,0,p)[0].any()
    mask,mean,var=model.update(h,v,seen,hit,x,y,replace(pose,stamp=2),2,p)
    assert mask.any() and not (mask&seen).any()
    assert np.max(np.abs(mean-h))<1e-8 and np.min(var)>=.0001
    query=np.c_[x[mask]-model.anchor.x,y[mask]-model.anchor.y,np.ones(mask.sum())]
    source_sigma=np.sqrt(model.measurement_sigma**2*p.variance_scale+p.height_std_floor**2)
    bound=np.abs(query@model.plane_weights)@source_sigma+p.height_std_floor
    assert np.all(np.sqrt(var[mask]*p.variance_scale+p.height_std_floor**2)>=bound-1e-12)
    frozen=mask.copy()
    moved=model.update(h,v,seen,hit,x,y,replace(pose,x=1,stamp=3),3,p)[0]
    np.testing.assert_array_equal(frozen,moved)
    # Once anchored, late/missing odometry cannot make a fixed prior flicker.
    np.testing.assert_array_equal(frozen,model.update(h,v,seen,hit,x,y,None,3.2,p)[0])
    np.testing.assert_array_equal(frozen,model.update(h,v,seen,hit,x,y,pose,3.3,p)[0])
    hit[10,10]=True
    assert not model.update(h,v,seen,hit,x,y,replace(pose,stamp=4),4,p)[0].any()
    assert model.revoked
    hit[:]=False
    assert not model.update(h,v,seen,hit,x,y,replace(pose,stamp=5),5,p)[0].any()


def test_unstable_or_unenclosed_or_nonplanar_stance_is_not_assumed_safe():
    h,v,seen,hit,x,y=scene();p=ProbabilityParams(internal_steps=True)
    for bad,mask,height in ((Stance(0,0,0,.4,0,0,0),seen,h),
                            (Stance(0,0,0,0,0,0,0,z=1.),seen,h),
                            (Stance(0,0,0,0,0,0,0),seen&(x>0),h),
                            (Stance(0,0,0,0,0,0,0),seen,h+.2*np.sin(8*x))):
        model=StartupSupport()
        model.update(height,v,mask,hit,x,y,bad,0,p)
        assert not model.update(height,v,mask,hit,x,y,replace(bad,stamp=2),2,p)[0].any()


def test_filled_initial_footprint_does_not_make_unseen_forward_ground_safe():
    h,v,seen,hit,x,y=scene();p=ProbabilityParams(internal_steps=True,yaw_bins=8)
    model=StartupSupport();pose=Stance(0,0,0,0,0,0,0)
    model.update(h,v,seen,hit,x,y,pose,0,p)
    initial,mean,var=model.update(h,v,seen,hit,x,y,replace(pose,stamp=2),2,p)
    seen &= x<.65
    out=geometric_probability(np.where(seen,h,mean),np.where(seen,v,var),seen|initial,p)
    assert np.isfinite(out["failure_probability_upper"][0][10,10])
    assert np.isnan(out["failure_probability_upper"][0][10,17])


def test_zero_heading_seam_keeps_both_initial_raster_footprints_supported():
    h,v,seen,hit,x,y=scene();p=ProbabilityParams(internal_steps=True,yaw_bins=36)
    model=StartupSupport();pose=Stance(0,0,-.001,0,0,0,0)
    model.update(h,v,seen,hit,x,y,pose,0,p)
    initial,mean,var=model.update(h,v,seen,hit,x,y,replace(pose,yaw=.001,stamp=2),2,p)
    out=geometric_probability(np.where(initial,mean,h),np.where(initial,var,v),seen|initial,p)
    assert all(np.isfinite(out["failure_probability_upper"][k][10,10]) for k in (0,35))
    later=model.update(h,v,seen,hit,x,y,replace(pose,x=1.,yaw=.5,stamp=3),3,p)[0]
    np.testing.assert_array_equal(initial,later)


def test_initial_footprint_is_not_shifted_to_a_nearest_cell_center():
    axis=(np.arange(31)-15+.5)*.2
    x,y=np.meshgrid(axis,axis)
    h=.02*x+.01*y;v=np.full_like(h,.0001)
    pose=Stance(-.186,-.006,.0006,0,0,0,0)
    seen=np.hypot(x-pose.x,y-pose.y)>.72
    hit=np.zeros_like(seen);p=ProbabilityParams(internal_steps=True)
    model=StartupSupport()
    model.update(h,v,seen,hit,x,y,pose,0,p)
    initial,mean,_=model.update(h,v,seen,hit,x,y,replace(pose,stamp=2),2,p)
    # This square intersects the true initial raster footprint. Snapping the
    # anchor from y=-.006 to y=-.1 wrongly leaves its entire row unsupported.
    row=np.argmin(abs(axis-.5));col=np.argmin(abs(axis+.1))
    assert initial[row,col]
    assert model.anchor.x==pose.x and model.anchor.y==pose.y
    np.testing.assert_allclose(mean,h,atol=1e-12)
    later=model.update(h,v,seen,hit,x,y,replace(pose,x=1,y=1,stamp=3),3,p)[0]
    np.testing.assert_array_equal(initial,later)


def test_stance_plane_rejects_wall_returns_without_clearing_obstacle_evidence():
    h,v,seen,hit,x,y=scene();p=ProbabilityParams(internal_steps=True)
    wall=(np.abs(y)>1.1)&(x<.4)
    h[wall]=.6
    original_h=h.copy();original_seen=seen.copy();original_hit=hit.copy()
    model=StartupSupport();pose=Stance(0,0,0,0,0,0,0,z=.3)
    model.update(h,v,seen,hit,x,y,pose,0,p)
    prior,mean,_=model.update(h,v,seen,hit,x,y,replace(pose,stamp=2),2,p)
    assert prior.any() and not (prior&wall).any()
    np.testing.assert_allclose(mean,.02*x+.01*y,atol=1e-12)
    np.testing.assert_array_equal(h,original_h)
    np.testing.assert_array_equal(seen,original_seen)
    np.testing.assert_array_equal(hit,original_hit)
    # A ground observation contradicting the frozen footprint is not discarded
    # by the evidence selector, even if its height is outside the contact band.
    seen[10,10]=True;h[10,10]=-.3
    assert not model.update(h,v,seen,hit,x,y,replace(pose,stamp=3),3,p)[0].any()
    assert model.revoked


def test_contact_height_filter_cannot_create_an_enclosure_or_ignore_stance_slope():
    h,v,seen,hit,x,y=scene();p=ProbabilityParams(internal_steps=True)
    pose=Stance(0,0,.7,.12,-.15,0,0,z=.3)
    u=np.cos(pose.yaw)*x+np.sin(pose.yaw)*y
    w=-np.sin(pose.yaw)*x+np.cos(pose.yaw)*y
    h=-np.tan(pose.pitch)*u+np.tan(pose.roll)*w
    model=StartupSupport();model.update(h,v,seen,hit,x,y,pose,0,p)
    assert model.update(h,v,seen,hit,x,y,replace(pose,stamp=2),2,p)[0].any()
    h[x<0]=.6
    model=StartupSupport();model.update(h,v,seen,hit,x,y,pose,0,p)
    assert not model.update(h,v,seen,hit,x,y,replace(pose,stamp=2),2,p)[0].any()
    assert model.reason=='surrounding observations do not enclose stance'


def test_explicit_one_sided_sim_stance_requires_attitude_height_and_area_and_stays_fixed():
    h,v,seen,hit,x,y=scene();p=ProbabilityParams(internal_steps=True)
    seen &= x>.7
    pose=Stance(0,0,0,0,0,0,0,z=.3)
    def fit(height,measured=seen,stance=pose):
        m=StartupSupport(allow_one_sided=True)
        m.update(height,v,measured,hit,x,y,stance,0,p)
        return m,m.update(height,v,measured,hit,x,y,replace(stance,stamp=2),2,p)
    model,(mask,mean,var)=fit(h)
    assert mask.any() and 'one-sided' in model.reason
    assert not (mask&seen).any()
    frozen=model.update(h,v,seen,hit,x,y,replace(pose,x=1,stamp=3),3,p)[0]
    np.testing.assert_array_equal(frozen,mask)
    assert not mask[x>1.].any()
    # A small line of observations, wrong clearance or an attitude-inconsistent
    # ramp cannot turn this explicit stance hypothesis into free-space filling.
    for heights,measured,stance in ((h,seen&(abs(y)<.2),pose),
            (h,seen,replace(pose,z=None)),(h,seen,replace(pose,z=.8)),
            (.18*x,seen,pose)):
        assert not fit(heights,measured,stance)[1][0].any()
    hit[10,10]=True
    assert not model.update(h,v,seen,hit,x,y,replace(pose,stamp=4),4,p)[0].any()
    assert model.revoked
