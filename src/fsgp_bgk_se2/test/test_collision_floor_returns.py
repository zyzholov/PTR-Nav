"""Measured shape refinement, never a replacement for terrain safety."""
from pathlib import Path
import numpy as np
import pytest
from fsgp_bgk_se2.obstacle_columns import (measured_floor_return_mask,
    measured_collision_floor_witness,collision_subcells,inspect_body_collision)


def scene():
    res=.2;origin=(-2.,-2.);shape=(20,20)
    x,y=np.meshgrid(np.arange(-.975,1.,.05),np.arange(-.975,1.,.05))
    points=np.c_[x.ravel(),y.ravel(),np.zeros(x.size)]
    wall=np.c_[np.arange(-.275,.3,.05),np.full(12,-.825),np.full(12,.4)]
    points=np.r_[points,wall,[[0.,-.85,.1],[.05,-.85,-.1]]]
    z=np.zeros(shape);v=np.full(shape,.0001);obs=np.ones(shape,bool);occ=np.zeros(shape,bool)
    occ[5,8:12]=True;z[occ]=.4
    return points,z,v,obs,occ,origin,res


def test_actual_low_surface_returns_do_not_enlarge_a_mixed_wall_but_all_hazards_survive():
    p,z,v,obs,occ,origin,res=scene();before=[a.copy() for a in (p,z,v,obs,occ)]
    mask=measured_floor_return_mask(p,z,v,obs,occ,origin,res,max_step_m=.1)
    assert mask.sum()>0
    assert not mask[p[:,2]!=0.].any() # wall, positive step and negative drop
    for a,b in zip(before,(p,z,v,obs,occ)):np.testing.assert_array_equal(a,b)
    floor,gradient=measured_collision_floor_witness(z,v,obs,occ,res,max_step_m=.1)
    bits,bounds=collision_subcells(p[~mask],occ,origin,res,measured_floor=floor,
        measured_floor_gradient=gradient,max_step_m=.1,factor=4,return_bounds=True)
    np.testing.assert_array_equal(bits>0,occ) # No occupied coarse cell was cleared.
    # Geometry builders consume the exact same retained original returns.
    cp=pytest.importorskip('cupy')
    if not cp.cuda.runtime.getDeviceCount():pytest.skip('CUDA unavailable')
    gpu=collision_subcells(p[~mask],occ,origin,res,measured_floor=floor,
        measured_floor_gradient=gradient,max_step_m=.1,factor=4,return_bounds=True,backend='cupy')
    np.testing.assert_array_equal(bits,gpu[0]);np.testing.assert_array_equal(bounds,gpu[1])


def test_missing_uncertain_references_or_one_dimensional_rays_cannot_remove_a_return():
    p,z,v,obs,occ,origin,res=scene()
    for height,var,observed in ((np.full_like(z,np.nan),v,obs),(z,np.ones_like(v),obs),(z,v,np.zeros_like(obs))):
        assert not measured_floor_return_mask(p,height,var,observed,occ,origin,res).any()
    # Many duplicate rays are one spatial witness, not six independent ones.
    line=np.r_[np.tile([[0.,-.79,0.]],(30,1)),[[0.,-.85,0.]]]
    assert not measured_floor_return_mask(line,np.zeros_like(z),v,obs,occ,origin,res).any()
    assert not measured_floor_return_mask(np.r_[p,[[np.nan,0,0]]],z,v,obs,occ,origin,res)[-1]
    with pytest.raises(ValueError):measured_floor_return_mask(p,z[:2],v,obs,occ,origin,res)


def test_actual_return_labels_are_equivariant_to_map_origin_and_height_datum():
    p,z,v,obs,occ,origin,res=scene()
    a=measured_floor_return_mask(p,z,v,obs,occ,origin,res)
    shift=np.array([100.123,-70.321,7.])
    b=measured_floor_return_mask(p+shift,z+shift[2],v,obs,occ,np.asarray(origin)+shift[:2],res)
    np.testing.assert_array_equal(a,b)


def test_recorded_complete_scan_removes_false_wall_extent_without_clearing_the_real_wall():
    data=np.load(Path(__file__).parent/'data/mixed_wall_floor_returns.npz')
    p,z,v,obs,occ=[data[k]for k in ('points','elevation','variance','observed','occupied')]
    origin=data['origin'];res=float(data['resolution']);pose=data['pose']
    mask=measured_floor_return_mask(p,z,v,obs,occ,origin,res,max_step_m=.1)
    assert mask.any() and not mask[p[:,2]>0].any()
    floor,gradient=measured_collision_floor_witness(z,v,obs,occ,res,max_step_m=.1)
    options=dict(measured_floor=floor,measured_floor_gradient=gradient,max_step_m=.1,factor=4,return_bounds=True)
    original=collision_subcells(p,occ,origin,res,**options)
    refined=collision_subcells(p[~mask],occ,origin,res,**options)
    check=lambda pair,at:inspect_body_collision(occ,pair[0],origin,res,*at,.521,.521,.46,factor=4,bounds=pair[1])
    assert check(original,pose)==(True,True)
    assert check(refined,pose)==(False,True)
    closer=pose.copy();closer[1]-=.2
    assert check(refined,closer)==(True,True)
    np.testing.assert_array_equal(refined[0]>0,occ)
