import numpy as np
import pytest
from fsgp_bgk_se2.overhead_clearance import measured_ceiling,clip_overhead
from fsgp_bgk_se2.obstacle_columns import vertical_columns
from fsgp_bgk_se2.shared_terrain import raw_measured_evidence


@pytest.mark.parametrize('radius',[.9,3.])
@pytest.mark.parametrize('seed',range(4))
def test_masked_gpu_envelope_is_exact_at_requested_targets_and_never_fabricates_other_heights(radius,seed):
    pytest.importorskip('cupy')
    from fsgp_bgk_se2.overhead_clearance_cuda import plane_envelope
    rng=np.random.default_rng(seed);y,x=np.mgrid[:28,:30]
    z=(.03*x*.2-.02*y*.2).astype(np.float32)
    floor=np.ones(z.shape,bool);witness=rng.random(z.shape)>.2
    columns=np.zeros(z.shape,bool);labels=np.zeros(z.shape,np.int32);owner=labels.copy()
    args=(z,floor,witness,witness,columns,labels,owner,.2,.65,.10,.015,radius,.04)
    complete=plane_envelope(*args)
    query=rng.random(z.shape)<.05;query[0,0]=query[-1,-1]=True
    selected=plane_envelope(*args,query_mask=query)
    np.testing.assert_array_equal(selected[query],complete[query])
    assert np.isposinf(selected[~query]).all()
    assert np.isposinf(plane_envelope(*args,query_mask=np.zeros_like(query))).all()


def test_masked_fitting_retains_original_normal_transfer_seeds_outside_the_query():
    pytest.importorskip('cupy')
    from fsgp_bgk_se2.overhead_clearance_cuda import plane_envelope
    z=np.zeros((24,24),np.float32);floor=np.ones(z.shape,bool)
    witness=floor.copy();witness[:,:10]=False
    labels=np.zeros(z.shape,np.int32);columns=np.zeros_like(floor)
    args=(z,floor,witness,witness,columns,labels,labels,.2,.65,.10,0.,.9,.04)
    complete=plane_envelope(*args);query=np.zeros_like(floor);query[12,10]=True
    assert np.isfinite(complete[12,10])
    masked=plane_envelope(*args,query_mask=query)
    assert masked[12,10]==complete[12,10]
    assert np.isposinf(masked[~query]).all()


def scene(slope=0.):
    y,x=np.mgrid[:20,:20]
    z=(slope*x*.2).astype(np.float32)
    points=np.c_[(x.ravel()+.5)*.2,(y.ravel()+.5)*.2,z.ravel()].astype(np.float32)
    return x*.2+.1,y*.2+.1,z,points


@pytest.mark.parametrize('backend',['cpu','cupy'])
def test_reused_multiscale_evidence_is_identical_to_independent_full_checks(backend):
    if backend=='cupy':pytest.importorskip('cupy')
    x,y=np.meshgrid((np.arange(36)+.5)*.2,(np.arange(36)+.5)*.2)
    z=(.06*np.sin(x)+.03*y).astype(np.float32)
    seen=np.ones(z.shape,bool);seen[10:13,19:22]=False
    current=(np.hypot(x-3.5,y-3.5)>1.4)&seen
    raw=np.c_[x[current],y[current],z[current]]
    raw=np.r_[raw,[[3.1,4.1,1.4],[3.3,4.1,-.8]]]
    options=dict(backend=backend,reference_points=raw[::2],reference_error_m=.015)
    local=measured_ceiling(z,seen,raw,(0,0),.2,**options)
    broad=measured_ceiling(z,seen,raw,(0,0),.2,reference_radius_m=3.,max_residual_m=.10,**options)
    expected=np.where(~np.isfinite(local)&seen,broad,local)
    actual=measured_ceiling(z,seen,raw,(0,0),.2,measured_fallback=True,**options)
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(seen[10:13,19:22],False)


@pytest.mark.parametrize('seed',range(4))
@pytest.mark.parametrize('radius',[.9,3.])
def test_fused_clearance_gpu_matches_cpu_with_occlusion_wall_returns_and_height_age(seed,radius):
    pytest.importorskip('cupy');rng=np.random.default_rng(seed)
    resolution=float(np.float32(.2));x,y=np.meshgrid(np.arange(104)*resolution,np.arange(104)*resolution)
    z=(.03*x-.02*y).astype(np.float32);seen=rng.uniform(size=z.shape)>.05
    z[45:48,:]+=.6
    ref=np.c_[x[seen],y[seen],z[seen]]
    keep=(np.hypot(ref[:,0]-10.2,ref[:,1]-10.2)>1.3)&(rng.uniform(size=len(ref))>.25)
    raw=ref[keep]
    upper=raw[(raw[:,1]>8.9)&(raw[:,1]<9.7)]+[0,0,.6]
    raw=np.r_[raw,upper,[[10.5,10.5,.8],[10.5,10.7,-.4]]]
    options=dict(reference_points=ref,reference_error_m=.05,reference_radius_m=radius,
                 max_residual_m=.10 if radius==3. else .04)
    cpu=measured_ceiling(z,seen,raw,(0,0),resolution,**options)
    gpu=measured_ceiling(z,seen,raw,(0,0),resolution,backend='cupy',**options)
    np.testing.assert_array_equal(np.isfinite(cpu),np.isfinite(gpu))
    good=np.isfinite(cpu)
    assert good.any()
    np.testing.assert_allclose(cpu[good],gpu[good],atol=2e-6,rtol=0.)


def test_enclosing_multiscale_reference_recovers_measured_blind_floor_but_not_unknown_support():
    x,y=np.meshgrid((np.arange(50)+.5)*.2,(np.arange(50)+.5)*.2)
    z=np.zeros(x.shape,np.float32);seen=np.ones(x.shape,bool)
    current=np.hypot(x-5.,y-5.)>1.4
    raw=np.c_[x[current],y[current],z[current]]
    local=measured_ceiling(z,seen,raw,(0,0),.2)
    broad=measured_ceiling(z,seen,raw,(0,0),.2,reference_radius_m=3.)
    assert not np.isfinite(local[25,30]) and np.isfinite(broad[25,30])
    combined=np.where(~np.isfinite(local)&seen,broad,local)
    high=np.array([[6.1,5.1,.95]])
    assert len(clip_overhead(high,combined,(0,0),.2))==0
    low=high-[0,0,.6]
    np.testing.assert_array_equal(clip_overhead(low,combined,(0,0),.2),low)
    seen[25,30]=False
    unknown=np.where(~np.isfinite(local)&seen,broad,local)
    assert not np.isfinite(unknown[25,30])


@pytest.mark.parametrize('backend',['cpu','cupy'])
def test_nonplanarity_is_added_to_the_height_bound_not_silently_ignored(backend):
    if backend=='cupy':pytest.importorskip('cupy')
    x,y=np.meshgrid((np.arange(50)+.5)*.2,(np.arange(50)+.5)*.2)
    z=(.06*np.sin(2*(x-5.1))).astype(np.float32);seen=np.ones(z.shape,bool)
    raw=np.c_[x.ravel(),y.ravel(),z.ravel()]
    tight=measured_ceiling(z,seen,raw,(0,0),.2,reference_radius_m=3.,backend=backend)
    broad=measured_ceiling(z,seen,raw,(0,0),.2,reference_radius_m=3.,max_residual_m=.10,backend=backend)
    assert not np.isfinite(tight[25,25]) and np.isfinite(broad[25,25])
    assert broad[25,25]>.80
    # A beam at 80cm may now hit the body above the raised local terrain;
    # admitting a less planar reference must keep that lower beam occupied.
    low=np.array([[5.1,5.1,.80]])
    np.testing.assert_array_equal(clip_overhead(low,broad,(0,0),.2),low)


def test_full_reference_scan_requires_matching_frame_and_nonfuture_source_stamp():
    from collections import deque
    from threading import Lock
    from types import SimpleNamespace
    from sensor_msgs_py.point_cloud2 import create_cloud_xyz32
    from std_msgs.msg import Header
    from se2_grid_msgs.msg import SE2Grid
    from fsgp_bgk_se2.fsgp_bgk_se2_node import FsgpBgkSe2Node as Node
    msg=SE2Grid();msg.info.header.frame_id='map';msg.info.header.stamp.sec=10
    def cloud(frame,sec,nsec):
        header=Header(frame_id=frame);header.stamp.sec=sec;header.stamp.nanosec=nsec
        return create_cloud_xyz32(header,[[1.,2.,3.]])
    node=SimpleNamespace(_input_lock=Lock(),_reference_cloud_history=deque(),
        _stamp_ns=Node._stamp_ns,_decode_points=Node._decode_points)
    fallback=np.array([[0.,0.,0.]])
    for candidate in (cloud('wrong',10,0),cloud('map',10,1),cloud('map',9,849999999)):
        node._reference_cloud_history=deque([candidate])
        points,age,full=Node._reference_points(node,msg,fallback,.1)
        assert not full and points is fallback and age==.1
    node._reference_cloud_history=deque([cloud('map',9,900000000)])
    points,age,full=Node._reference_points(node,msg,fallback,.1)
    assert full and age==pytest.approx(.1)
    np.testing.assert_array_equal(points,[[1.,2.,3.]])


def test_clearance_only_history_preserves_source_age_and_adds_conservative_height_reserve():
    from fsgp_bgk_se2.overhead_clearance import FloorReferenceHistory
    _,_,z,p=scene();seen=np.ones(z.shape,bool);history=FloorReferenceHistory()
    history.update(p,0.,'map',(0.,0.))
    ref,ages=history.update(np.empty((0,3)),1.,'map',(.2,0.))
    assert len(ref)==len(p) and (ages==1.).all()
    # Re-receiving the same old source cannot make it fresh.
    _,ages=history.update(p,1.1,'map',(.2,0.),source_age_s=1.1)
    np.testing.assert_allclose(ages,1.1)
    a=measured_ceiling(z,seen,p,(0,0),.2)
    b=measured_ceiling(z,seen,p,(0,0),.2,reference_points=ref,reference_error_m=.011)
    finite=np.isfinite(a)&np.isfinite(b)
    np.testing.assert_allclose(b[finite]-a[finite],.011,atol=1e-7)
    assert history.update(np.empty((0,3)),1.51,'map',(.2,0.))[0].size==0
    history.update(p,9.,'map',(.2,0.))
    assert history.update(np.empty((0,3)),9.1,'other',(.2,0.))[0].size==0
    history.update(p,10.,'map',(.2,0.))
    assert history.update(np.empty((0,3)),10.1,'map',(2.,0.))[0].size==0


def test_short_real_floor_history_bridges_occlusion_but_expires_and_never_removes_low_returns():
    from fsgp_bgk_se2.overhead_clearance import floor_witness_points
    from fsgp_bgk_se2.ground_evidence import GroundEvidenceHistory
    x,y,z,p=scene();seen=np.ones(z.shape,bool);history=GroundEvidenceHistory(1.5)
    source=floor_witness_points(z,seen,p,(0,0),.2)
    history.update(source,0.,'map',(0.,0.))
    current=p[p[:,0]>2.3]
    ref,ages=history.update(current,1.,'map',(.5,0.))
    old=measured_ceiling(z,seen,current,(0,0),.2)
    new=measured_ceiling(z,seen,current,(0,0),.2,reference_points=ref)
    assert not np.isfinite(old[10,9]) and np.isfinite(new[10,9])
    raw=np.r_[current,[[1.9,2.1,1.1],[1.9,2.1,.35],[1.9,2.1,-.3]]]
    clipped=clip_overhead(raw,new,(0,0),.2)
    assert len(clipped)==len(raw)-1
    veto,_=raw_measured_evidence(z,seen,x,y,clipped,.2,.18)
    assert veto[10,9]
    # Historical floor returns are not themselves an obstacle clear witness.
    assert not floor_witness_points(z,np.zeros_like(seen),ref,(0,0),.2).size
    ref,ages=history.update(current,1.6,'map',(.6,0.))
    expired=measured_ceiling(z,seen,current,(0,0),.2,reference_points=ref)
    assert not np.isfinite(expired[10,9]) and ages.max()<=1.5
    # Missing current floor, a frame change, and a changed measured height do
    # not retain a clearance certificate through the historical cache.
    assert not np.isfinite(measured_ceiling(z,seen,raw[-3:],(0,0),.2,reference_points=source)).any()
    history.update(source,2.,'map',(.7,0.))
    ref,_=history.update(current,2.1,'new_map',(.7,0.))
    assert not np.isfinite(measured_ceiling(z,seen,current,(0,0),.2,reference_points=ref)[10,9])
    changed=z.copy();changed[:, :11]=.3
    assert not np.isfinite(measured_ceiling(changed,seen,current,(0,0),.2,reference_points=source)[10,9])


def test_nearby_wall_top_cannot_poison_floor_plane_or_be_cleared_itself():
    x,y,z,p=scene();seen=np.ones(z.shape,bool)
    # A thin raised wall shares the floor's "observed" flag. Retain it as an
    # obstacle, but don't regress a ground plane through its top.
    z[7,:]=.6;raw=np.c_[x.ravel(),y.ravel(),z.ravel()]
    ceiling=measured_ceiling(z,seen,raw,(0,0),.2)
    assert np.isfinite(ceiling[10,9]) and not np.isfinite(ceiling[7,9])
    beam=np.array([[x[10,9],y[10,9],1.1]])
    wall=np.array([[x[7,9],y[7,9],.6]])
    out=clip_overhead(np.r_[beam,wall],ceiling,(0,0),.2)
    np.testing.assert_array_equal(out,wall)


def test_gradual_wall_minima_still_require_independent_vertical_column_exclusion():
    x,y,z,p=scene();seen=np.ones(z.shape,bool)
    z[5:8,:]=np.arange(3,0,-1)[:,None]*.12
    raw=np.c_[x.ravel(),y.ravel(),z.ravel()]
    upper=raw[(raw[:,1]>=1.)&(raw[:,1]<1.6)]+[0,0,.5]
    ceiling=measured_ceiling(z,seen,np.r_[raw,upper],(0,0),.2)
    assert np.isfinite(ceiling[10,9])
    assert not np.isfinite(ceiling[7,9])


@pytest.mark.parametrize('slope',[0.,.5,-.5])
@pytest.mark.parametrize('backend',['cpu','cuda'])
def test_overhead_is_not_a_column_but_low_bar_post_and_negative_evidence_remain(slope,backend):
    xp=np if backend=='cpu' else pytest.importorskip('cupy')
    x,y,z,p=scene(slope)
    extra=np.array([[x[10,9],y[10,9],z[10,9]+1.1],
                    [x[10,10],y[10,10],z[10,10]+.55],
                    [x[10,11],y[10,11],z[10,11]-.3]],np.float32)
    raw=np.r_[p,extra]
    ceiling=measured_ceiling(z,np.ones(z.shape,bool),raw,(0,0),.2,
                             backend='cpu' if backend=='cpu' else 'cupy')
    clipped=clip_overhead(xp.asarray(raw),ceiling,(0,0),.2,xp=xp)
    values=clipped if xp is np else xp.asnumpy(clipped)
    assert len(values)==len(raw)-1
    mask,_=vertical_columns(values,z.shape,(0,0),.2,min_hits=1,ground_elevation=z,xp=xp)
    mask=mask if xp is np else xp.asnumpy(mask)
    assert not mask[10,9] and mask[10,10]
    veto,_=raw_measured_evidence(z,np.ones(z.shape,bool),x,y,values,.2,.18)
    assert not veto[10,9] and veto[10,10] and veto[10,11]


def test_unknown_ground_absent_current_floor_steps_and_edges_never_qualify_overhead():
    x,y,z,p=scene()
    raw=np.array([[1.9,2.1,1.1]],np.float32)
    ceiling=measured_ceiling(z,np.ones(z.shape,bool),raw,(0,0),.2)
    assert not np.isfinite(ceiling).any()
    assert len(clip_overhead(raw,ceiling,(0,0),.2))==1
    observed=np.ones(z.shape,bool);observed[10,9]=False
    ceiling=measured_ceiling(z,observed,p,(0,0),.2)
    # Enclosed measured-plane reference is allowed for slicing only. It does
    # not change the observed mask or claim support in this missing cell.
    assert np.isfinite(ceiling[10,9]) and not np.isfinite(ceiling[0,0])
    assert not observed[10,9]
    z[10,9]=.8
    ceiling=measured_ceiling(z,np.ones(z.shape,bool),np.r_[p,raw],(0,0),.2)
    assert not np.isfinite(ceiling[10,9])


def test_vehicle_slope_envelope_and_headroom_are_not_world_z_or_sensor_origin_cutoffs():
    _,_,z,p=scene(.5)
    ceiling=measured_ceiling(z,np.ones(z.shape,bool),p,(0,0),.2)
    assert ceiling[10,9]==pytest.approx(z[10,9]+.65*np.sqrt(1.25)+.10,abs=1e-6)
    assert ceiling[10,15]>ceiling[10,5]


def test_slice_preserves_boundary_and_is_translation_invariant():
    _,_,z,p=scene(.25)
    observed=np.ones(z.shape,bool)
    ceiling=measured_ceiling(z,observed,p,(0,0),.2)
    boundary=ceiling[10,9]
    extra=np.array([[1.9,2.1,boundary-1e-4],
                    [1.9,2.1,boundary],
                    [1.9,2.1,boundary+1e-4]],np.float32)
    kept=clip_overhead(extra,ceiling,(0,0),.2)
    np.testing.assert_array_equal(kept,extra[:2])
    offset=np.array([10.,-4.,-3.],np.float32)
    shifted=measured_ceiling(z+offset[2],observed,p+offset,offset[:2],.2)
    np.testing.assert_array_equal(np.isfinite(ceiling),np.isfinite(shifted))
    valid=np.isfinite(ceiling)
    np.testing.assert_allclose(shifted[valid],ceiling[valid]+offset[2],atol=1e-6)
    np.testing.assert_allclose(clip_overhead(extra+offset,shifted,offset[:2],.2),
                               extra[:2]+offset,atol=1e-6)


def test_one_sided_floor_cannot_clear_unknown_region_or_add_support():
    _,_,z,p=scene()
    observed=np.ones(z.shape,bool);observed[:,10:]=False
    ceiling=measured_ceiling(z,observed,p,(0,0),.2)
    # The last genuinely measured column may borrow a nearby verified normal;
    # no unmeasured column receives an extrapolated clearance certificate.
    assert np.isfinite(ceiling[10,9])
    assert not np.isfinite(ceiling[:,10:]).any()
    assert not observed[:,10:].any()
    high=np.array([[2.1,2.1,1.1]],np.float32)
    np.testing.assert_array_equal(clip_overhead(high,ceiling,(0,0),.2),high)
    # A stale observed flag without a current/history return at the target
    # cannot use the direct-ground normal transfer.
    no_target=p[np.abs(p[:,0]-1.9)>.01]
    without=measured_ceiling(z,observed,no_target,(0,0),.2)
    assert not np.isfinite(without[:,9:]).any()


def test_measured_center_can_never_lower_the_required_vehicle_envelope():
    _,_,z,p=scene()
    # The center lacks a fresh corroborating floor return, but its original
    # measured height still must not be lowered by the surrounding plane fit.
    z[10,9]=.03
    p=np.delete(p,10*20+9,axis=0)
    ceiling=measured_ceiling(z,np.ones(z.shape,bool),p,(0,0),.2)
    assert ceiling[10,9]>=z[10,9]+.65+.10-1e-6


@pytest.mark.parametrize('height,margin',[(0.,.10),(-1.,.10),(np.nan,.10),
                                        (.65,0.),(.65,-.1),(.65,np.inf)])
def test_invalid_envelope_cannot_silently_remove_returns(height,margin):
    _,_,z,p=scene()
    with pytest.raises(ValueError,match='vehicle-height envelope'):
        measured_ceiling(z,np.ones(z.shape,bool),p,(0,0),.2,
                         vehicle_height_m=height,margin_m=margin)
