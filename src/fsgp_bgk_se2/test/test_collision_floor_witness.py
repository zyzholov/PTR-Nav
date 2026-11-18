"""Actual measured floor must not become an occupied subcell beside a pole."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pytest
from fsgp_bgk_se2.obstacle_columns import (
    measured_collision_floor_witness,collision_subcells,inspect_body_collision,collision_pose_masks)


def plane(sx=0.,sy=0.):
    y,x=np.mgrid[-5:6,-5:6]*.2
    z=sx*x+sy*y;v=np.full_like(z,.01**2);obs=np.ones_like(z,bool)
    occ=np.zeros_like(obs);occ[5,5]=True;obs[5,5]=False
    z[5,5]=np.nan
    return z,v,obs,occ


@pytest.mark.parametrize('slopes',[(0.,0.),(.5,-.3),(-.6,.25)])
def test_plane_classifies_raw_floor_at_its_actual_xy_and_keeps_the_column(slopes):
    args=plane(*slopes);copies=[x.copy() for x in args]
    floor,grad=measured_collision_floor_witness(*args,.2)
    assert np.isfinite(floor[5,5])
    np.testing.assert_allclose(grad[:,5,5],slopes,atol=1e-12)
    x,y=-.07,-.08
    points=np.array([[x,y,slopes[0]*x+slopes[1]*y],[.05,.08,1.]])
    old=collision_subcells(points,args[3],(-1.1,-1.1),.2,measured_floor=args[0],factor=4)
    new=collision_subcells(points,args[3],(-1.1,-1.1),.2,measured_floor=floor,
        measured_floor_gradient=grad,factor=4)
    assert new[5,5]>0 and new[5,5]!=old[5,5]
    assert (int(new[5,5])&int(old[5,5]))==new[5,5]
    for a,b in zip(args,copies):np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('height',[.18,.3,1.,-.3])
def test_steps_sides_tops_and_negative_returns_are_not_removed(height):
    args=plane();floor,grad=measured_collision_floor_witness(*args,.2)
    pts=np.array([[-.06,-.08,height],[.05,.08,1.]])
    before=collision_subcells(pts,args[3],(-1.1,-1.1),.2,factor=4)
    after=collision_subcells(pts,args[3],(-1.1,-1.1),.2,measured_floor=floor,
        measured_floor_gradient=grad,factor=4)
    np.testing.assert_array_equal(before,after)


@pytest.mark.parametrize('failure',['one_side','line','noise','high_variance','unknown'])
def test_ambiguous_neighbours_do_not_supply_a_reference(failure):
    z,v,obs,occ=plane()
    if failure=='one_side':obs[:6]=False
    elif failure=='line':obs[:]=False;obs[5,:]=True;obs[5,5]=False
    elif failure=='noise':z+=np.where(np.indices(z.shape).sum(axis=0)%2,.06,-.06)
    elif failure=='high_variance':v[:]=.1**2
    elif failure=='unknown':obs[:]=False
    floor,_=measured_collision_floor_witness(z,v,obs,occ,.2)
    assert not np.isfinite(floor[5,5])


def test_no_remaining_obstacle_witness_keeps_full_coarse_cell_fallback():
    args=plane();floor,grad=measured_collision_floor_witness(*args,.2)
    bits=collision_subcells(np.array([[0.,0.,0.]]),args[3],(-1.1,-1.1),.2,
        measured_floor=floor,measured_floor_gradient=grad,factor=4)
    assert bits[5,5]==65535


def test_real_recorded_floor_returns_no_longer_form_a_false_column_edge():
    path=Path(__file__).parent/'data/mixed_column_floor_witness_20261104.npz'
    meta=json.loads(path.with_suffix('.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['npz_sha256']
    d=np.load(path);floor,grad=measured_collision_floor_witness(
        d['elevation'],d['variance'],d['observed'],d['occupied'],float(d['resolution']))
    original=np.where(d['observed']&(d['variance']<=.02**2),d['elevation'],np.nan)
    before,oldbounds=collision_subcells(d['points'],d['occupied'],d['origin'],float(d['resolution']),
        measured_floor=original,factor=4,return_bounds=True)
    np.testing.assert_array_equal(before,d['old_bits']);np.testing.assert_array_equal(oldbounds,d['old_bounds'])
    after,bounds=collision_subcells(d['points'],d['occupied'],d['origin'],float(d['resolution']),
        measured_floor=floor,measured_floor_gradient=grad,factor=4,return_bounds=True)
    args=(d['origin'],float(d['resolution']),*d['pose'],.68,.58,.5)
    assert inspect_body_collision(d['occupied'],before,*args,factor=4,bounds=oldbounds)==(True,True)
    assert inspect_body_collision(d['occupied'],after,*args,factor=4,bounds=bounds)==(False,True)
    np.testing.assert_array_equal(before>0,after>0) # every original occupied column remains
    # A closer pose still cannot enter the retained real pole geometry.
    pose=d['pose'].copy();pose[1]+=.20
    assert inspect_body_collision(d['occupied'],after,d['origin'],float(d['resolution']),
        *pose,.68,.58,.5,factor=4,bounds=bounds)==(True,True)
    try:import cupy as cp
    except ImportError:return
    if cp.cuda.runtime.getDeviceCount():
        np.testing.assert_array_equal(
            collision_pose_masks(after,float(d['resolution']),.68,.58,.5,36,factor=4),
            collision_pose_masks(after,float(d['resolution']),.68,.58,.5,36,factor=4,backend='cupy'))


def test_rejected_stored_side_height_gets_an_independent_reference_before_fine_projection():
    z,v,obs,occ=plane();obs[5,5]=True;z[5,5]=.38
    original=[a.copy() for a in (z,v,obs,occ)]
    floor,gradient=measured_collision_floor_witness(z,v,obs,occ,.2,max_step_m=.1)
    assert abs(floor[5,5])<1e-10
    points=np.array([[-.06,-.08,0.],[.05,.08,.5]])
    bits=collision_subcells(points,occ,(-1.1,-1.1),.2,measured_floor=floor,
        measured_floor_gradient=gradient,max_step_m=.1,factor=4)
    assert 0<bits[5,5]<65535 # floor excluded, real column retained
    for a,b in zip((z,v,obs,occ),original):np.testing.assert_array_equal(a,b)
    # A one-sided reference cannot replace the stored height with a fabricated floor.
    obs[:6]=False;obs[5,5]=True
    floor,_=measured_collision_floor_witness(z,v,obs,occ,.2,max_step_m=.1)
    assert not np.isfinite(floor[5,5])


@pytest.mark.parametrize('event',[2,4])
def test_recorded_neos_projection_preserves_raw_obstacles_without_spurious_neighbour_cells(event):
    from fsgp_bgk_se2.obstacle_columns import vertical_columns
    path=Path(__file__).parent/'data'/f'neos_projection_event_{event}.npz'
    meta=json.loads(path.with_suffix('.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['npz_sha256']
    d=np.load(path);res=float(d['resolution']);origin=d['origin'];pose=d['pose']
    occupied,_=vertical_columns(d['points'],d['elevation'].shape,origin,res,
        min_height=.100001,min_hits=1,ground_elevation=d['ground'])
    cp=pytest.importorskip('cupy')
    gpu,_=vertical_columns(d['points'],d['elevation'].shape,origin,res,
        min_height=.100001,min_hits=1,ground_elevation=d['ground'],xp=cp)
    np.testing.assert_array_equal(occupied,cp.asnumpy(gpu))
    floor,grad=measured_collision_floor_witness(d['elevation'],d['variance'],d['observed'],occupied,res,max_step_m=.1)
    bits,bounds=collision_subcells(d['points'],occupied,origin,res,measured_floor=floor,
        measured_floor_gradient=grad,max_step_m=.1,factor=4,return_bounds=True)
    assert inspect_body_collision(d['old_mask'],d['old_bits'],origin,res,*pose,
        .521,.521,.460,factor=4,bounds=d['old_bounds'])==(True,True)
    assert inspect_body_collision(occupied,bits,origin,res,*pose,
        .521,.521,.460,factor=4,bounds=bounds)==(False,True)
    # A center on a CURRENT original non-floor return remains colliding.
    raw=d['points'];ij=np.floor((raw[:,:2].astype(float)-origin)/res).astype(int)
    inside=(ij[:,0]>=0)&(ij[:,1]>=0)&(ij[:,0]<occupied.shape[1])&(ij[:,1]<occupied.shape[0])
    raw,ij=raw[inside],ij[inside];ref=floor[ij[:,1],ij[:,0]]
    actual=occupied[ij[:,1],ij[:,0]]&np.isfinite(ref)&(raw[:,2]-ref>.1)&(raw[:,2]-ref<.65)
    assert actual.any();returns=raw[actual];point=returns[np.argmin(np.linalg.norm(returns[:,:2]-pose[:2],axis=1))]
    assert inspect_body_collision(occupied,bits,origin,res,float(point[0]),float(point[1]),float(pose[2]),
        .521,.521,.460,factor=4,bounds=bounds)==(True,True)
