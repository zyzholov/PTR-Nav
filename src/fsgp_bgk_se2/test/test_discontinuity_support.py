import numpy as np
import pytest
import torch
from fsgp_bgk_se2.shared_terrain import SharedTerrainParams,shared_terrain_prediction

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_missing_patch_cannot_smooth_away_an_observed_overlimit_step(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.set_num_threads(1)
    x,y=np.meshgrid((np.arange(41)-20)*.2,(np.arange(41)-20)*.2)
    h=.2*(x>.15);seen=np.hypot(x,y)>=.6;raw=np.c_[x[seen],y[seen],h[seen]]
    p=shared_terrain_prediction(np.where(seen,h,np.nan),np.full(h.shape,.0001),seen,
        x,y,raw,0.,.2,np.zeros(h.shape,bool),params=SharedTerrainParams(max_distance_m=1.25,max_extrapolation_m=.75),device=device)
    assert p is not None and p.discontinuity_veto[20,20] and not p.valid[20,20]
    assert not p.discontinuity_veto[seen].any()
    # A physically modest smooth grade is not a vertical step and must keep
    # its interpolation support. Do not obtain safety by rejecting every hole.
    h=.12*x;raw=np.c_[x[seen],y[seen],h[seen]]
    q=shared_terrain_prediction(np.where(seen,h,np.nan),np.full(h.shape,.0001),seen,
        x,y,raw,0.,.2,np.zeros(h.shape,bool),params=SharedTerrainParams(max_distance_m=1.25,max_extrapolation_m=.75),device=device)
    assert q.valid[20,20] and not q.discontinuity_veto.any()


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('registration_gap',[False,True])
def test_vertical_collision_columns_cannot_cast_false_terrain_shadows_across_flat_floor(device,registration_gap):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.set_num_threads(1)
    x,y=np.meshgrid((np.arange(41)-20)*.2,(np.arange(41)-20)*.2)
    seen=np.hypot(x,y)>=.6;wall=(abs(abs(y)-1.2)<.11);blocked=wall.copy()
    h=np.where(wall,.4+.2*np.sin(8*x),0.)
    if registration_gap:blocked[wall&(abs(x)<.11)]=False
    raw=np.c_[x[seen],y[seen],h[seen]]
    q=shared_terrain_prediction(np.where(seen,h,np.nan),np.full(h.shape,.0001),seen,
        x,y,raw,0.,.2,blocked,params=SharedTerrainParams(max_distance_m=1.25,max_extrapolation_m=.75),device=device)
    assert q is not None and q.valid[20,20] and not q.discontinuity_veto.any()
    assert not q.valid[blocked].any()  # Floor completion never clears wall occupancy.
