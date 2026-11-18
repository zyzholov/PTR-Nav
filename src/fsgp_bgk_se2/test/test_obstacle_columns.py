from types import SimpleNamespace

import numpy as np
import pytest

from fsgp_bgk_se2.obstacle_columns import preserve_collision_risk, vertical_columns, collision_subcells, collision_pose_masks


def test_subcell_rectangles_remove_only_unwitnessed_raster_extent_keep_padding_and_body():
    from fsgp_bgk_se2.obstacle_columns import inspect_body_collision
    occupied=np.zeros((16,16),bool);occupied[1,5]=True
    points=np.array([[1.,.35,1.]])
    bits,bounds=collision_subcells(points,occupied,(0,0),.2,factor=4,return_bounds=True)
    np.testing.assert_array_equal(bits,collision_subcells(points,occupied,(0,0),.2,factor=4))
    args=(occupied,bits,(0,0),.2,1.,.885,0.,.68,.58,.5)
    assert inspect_body_collision(*args,factor=4)==(True,True)
    assert inspect_body_collision(*args,factor=4,bounds=bounds)==(False,True)
    # Move the SAME full protected body into the original padded witness.
    args=(occupied,bits,(0,0),.2,1.,.85,0.,.68,.58,.5)
    assert inspect_body_collision(*args,factor=4,bounds=bounds)==(True,True)
    broken=bounds.astype(float);broken[:,1,5]=np.nan
    assert inspect_body_collision(occupied,bits,(0,0),.2,1.,.885,0.,.68,.58,.5,factor=4,bounds=broken)==(True,True)


@pytest.mark.parametrize('factor',[2,4])
def test_quantized_rectangles_outward_cover_every_original_padded_witness(factor):
    rng=np.random.default_rng(183);points=np.c_[rng.uniform(.21,.39,(120,2)),np.ones(120)]
    occupied=np.zeros((4,4),bool);occupied[1,1]=True;occupied[3,3]=True
    bits,bounds=collision_subcells(points,occupied,(0,0),.2,factor=factor,return_bounds=True)
    np.testing.assert_array_equal(bits,collision_subcells(points,occupied,(0,0),.2,factor=factor))
    assert bits[3,3]==(1<<(factor*factor))-1 and np.all(bounds[:,3,3]==0xF0F0)
    fine=.2/factor
    for sx in range(factor):
        for sy in range(factor):
            k=sy*factor+sx;ox=.2+sx*fine;oy=.2+sy*fine
            code=int(bounds[k,1,1]);rectangle=np.array([ox+(code&15)*fine/15,oy+((code>>8)&15)*fine/15,
                ox+((code>>4)&15)*fine/15,oy+((code>>12)&15)*fine/15])
            for x,y,_ in points:
                low=np.maximum([x-.02,y-.02],[ox,oy]);high=np.minimum([x+.02,y+.02],[ox+fine,oy+fine])
                if np.any(low>high):continue
                assert bits[1,1]&(1<<k)
                assert np.all(rectangle[:2]<=low+1e-12) and np.all(rectangle[2:]>=high-1e-12)


def test_positive_column_display_uses_raw_vertical_extent_without_inventing_unknown_floor():
    from fsgp_bgk_se2.obstacle_columns import occupied_column_extents
    occupied=np.zeros((3,3),bool);occupied[1,1]=True;occupied[2,2]=True
    points=np.array([[.23,.23,.2],[.25,.25,.7],[.05,.05,-1.],[.25,.25,np.nan]])
    low,high=occupied_column_extents(points,occupied,(0,0),.2)
    assert low[1,1]==np.float32(.2) and high[1,1]==np.float32(.7)
    assert np.isnan(low[2,2]) and np.isnan(high[2,2])
    assert np.isnan(low[~occupied]).all()
    # Occupancy is never cleared by absent measured vertical coordinates.
    assert occupied.sum()==2


@pytest.mark.parametrize('factor',[2,4])
def test_fine_pose_overlay_matches_independent_convex_polygon_feasibility(factor):
    from scipy.optimize import linprog
    bits=np.zeros((11,13),np.uint16);bits[3,4]=1<<(factor*factor-1);bits[6,8]=(1<<(factor*factor-1))|1;bits[1,10]=2
    res=.2;front,back,width=.683,.577,.497;bins=7
    masks=collision_pose_masks(bits,res,front,back,width,bins,factor=factor)
    squares=[]
    for row,col in zip(*np.where(bits)):
        for sy in range(factor):
            for sx in range(factor):
                if bits[row,col]&(1<<(sy*factor+sx)):
                    x=col*res+sx*res/factor;y=row*res+sy*res/factor
                    squares.append(np.array([[x,y],[x+res/factor,y],[x+res/factor,y+res/factor],[x,y+res/factor]]))
    def halfspaces(poly):
        edges=np.roll(poly,-1,axis=0)-poly
        normals=np.c_[edges[:,1],-edges[:,0]]
        return normals,np.sum(normals*poly,axis=1)
    for k in range(bins):
        yaw=(k+.5)*2*np.pi/bins;c,s=np.cos(yaw),np.sin(yaw)
        for row,col in ((2,3),(3,4),(4,5),(5,7),(6,8),(7,9),(1,11),(9,1)):
            body=np.array([[(col+.5)*res+c*u-s*v,(row+.5)*res+s*u+c*v]
                for u,v in ((front,width),(-back,width),(-back,-width),(front,-width))])
            a,b=halfspaces(body)
            expected=False
            for square in squares:
                aa,bb=halfspaces(square)
                if linprog([0.,0.],A_ub=np.vstack((a,aa)),b_ub=np.r_[b,bb],
                           bounds=[(None,None)]*2,method='highs').success:expected=True;break
            assert masks[k,row,col]==expected,(k,row,col)


@pytest.mark.parametrize('bins',[7,36])
def test_fine_collision_pose_gpu_cpu_parity_and_no_boundary_wrap(bins):
    pytest.importorskip('cupy')
    bits=np.zeros((13,15),np.uint8);bits[0,0]=15;bits[6,8]=2;bits[9,3]=9
    cpu=collision_pose_masks(bits,.2,.68,.58,.5,bins)
    gpu=collision_pose_masks(bits,.2,.68,.58,.5,bins,backend='cupy')
    np.testing.assert_array_equal(cpu,gpu)
    assert not cpu[:,-1,-1].any()
    assert not collision_pose_masks(np.zeros_like(bits),.2,.68,.58,.5,bins).any()


def test_five_centimeter_geometry_retains_padding_and_removes_only_raster_aliasing():
    from fsgp_bgk_se2.obstacle_columns import inspect_body_collision
    occupied=np.zeros((20,20),bool);occupied[12,10]=True
    origin=(-2.,-2.);points=np.array([[.12,.58,.6]])
    old=collision_subcells(points,occupied,origin,.2)
    fine=collision_subcells(points,occupied,origin,.2,factor=4)
    assert inspect_body_collision(occupied,old,origin,.2,0,0,0,.68,.58,.5)==(True,True)
    assert inspect_body_collision(occupied,fine,origin,.2,0,0,0,.68,.58,.5,factor=4)==(False,True)
    assert inspect_body_collision(occupied,fine,origin,.2,0,.08,0,.68,.58,.5,factor=4)==(True,True)
    assert inspect_body_collision(occupied,fine,origin,.2,0,0,np.pi/4,.68,.58,.5,factor=4)==(True,True)
    assert collision_subcells(None,occupied,origin,.2,factor=4)[12,10]==65535
    for bad in (0,65536,np.nan):
        invalid=fine.astype(float);invalid[12,10]=bad
        assert inspect_body_collision(occupied,invalid,origin,.2,0,0,0,.68,.58,.5,factor=4)==(True,True)


def test_five_centimeter_pose_overlay_matches_gpu_and_continuous_body_geometry():
    from fsgp_bgk_se2.obstacle_columns import inspect_body_collision
    bits=np.zeros((17,17),np.uint16);bits[8,8]=1<<15;bits[5,9]=1<<0
    res=.2;bins=12;shape=bits.shape
    cpu=collision_pose_masks(bits,res,.68,.58,.5,bins,factor=4)
    for k in range(bins):
        for row,col in ((8,8),(5,9),(8,5),(6,9),(10,10)):
            unsafe,known=inspect_body_collision(bits>0,bits,(0,0),res,(col+.5)*res,(row+.5)*res,
                (k+.5)*2*np.pi/bins,.68,.58,.5,factor=4)
            assert known and bool(cpu[k,row,col])==unsafe
    pytest.importorskip('cupy')
    gpu=collision_pose_masks(bits,res,.68,.58,.5,bins,factor=4,backend='cupy')
    np.testing.assert_array_equal(cpu,gpu)


def test_subcell_contract_retains_all_current_returns_and_coarse_fallback():
    occupied=np.zeros((3,4),bool);occupied[1,2]=True;occupied[2,0]=True
    bits=collision_subcells(np.array([[.51,.21,.6],[.59,.39,0.],[.1,.1,9.]]),occupied,(0,0),.2,xy_padding_m=0.)
    assert bits[1,2]==(2|8)  # upper AND lower return locations remain
    assert bits[2,0]==15  # no witness is never guessed empty
    assert np.all(bits[~occupied]==0)
    assert np.array_equal(occupied,bits>0)


def test_measured_floor_cannot_spread_a_pillar_over_unrelated_subcells():
    occupied=np.zeros((3,3),bool);occupied[1,1]=True
    floor=np.zeros((3,3))
    points=np.array([[.23,.23,.7],[.23,.23,0.],[.37,.23,0.],
                     [.23,.37,0.],[.37,.37,0.]])
    old=collision_subcells(points,occupied,(0,0),.2)
    refined=collision_subcells(points,occupied,(0,0),.2,measured_floor=floor)
    assert old[1,1]==15 and refined[1,1]==1
    np.testing.assert_array_equal(refined>0,occupied)
    # Missing ground, a missing obstacle witness and a raised measured top
    # cannot turn an existing occupied coarse column into free space.
    unknown=floor.copy();unknown[1,1]=np.nan
    assert collision_subcells(points,occupied,(0,0),.2,measured_floor=unknown)[1,1]==15
    assert collision_subcells(points[1:],occupied,(0,0),.2,measured_floor=floor)[1,1]==15
    top=floor.copy();top[1,1]=.7
    assert collision_subcells(points,occupied,(0,0),.2,measured_floor=top)[1,1]==15


def test_floor_refinement_retains_low_obstacles_negative_returns_and_xy_padding():
    occupied=np.zeros((3,3),bool);occupied[1,1]=True
    points=np.array([[.23,.23,.10],[.37,.37,-.10],[.299,.299,.6],[.37,.23,0.]])
    bits=collision_subcells(points,occupied,(0,0),.2,measured_floor=np.zeros((3,3)))
    assert bits[1,1]==15  # both low/negative witnesses and the padded edge stay
    with pytest.raises(ValueError,match='match'):
        collision_subcells(points,occupied,(0,0),.2,measured_floor=np.zeros((1,1)))


def test_subcell_padding_covers_boundary_and_never_reassigns_a_coarse_hazard():
    occupied=np.ones((1,2),bool)
    bits=collision_subcells(np.array([[.099,.099,.6],[.39,.1,.6]]),occupied,(0,0),.2)
    assert bits[0,0]==15
    assert bits[0,1]==(2|8)
    assert np.array_equal(collision_subcells(None,occupied,(0,0),.2),[[15,15]])


def test_subcell_projection_is_translation_invariant_and_filters_only_invalid_xyz():
    occupied=np.ones((2,2),bool);xyz=np.array([[.05,.05,.7],[.15,.15,.8],[.35,.35,.9],[np.nan,0,0.]])
    first=collision_subcells(xyz,occupied,(0,0),.2,xy_padding_m=0.)
    shift=np.array([-4.,6.,0.]);second=collision_subcells(xyz+shift,occupied,shift[:2],.2,xy_padding_m=0.)
    np.testing.assert_array_equal(first,second)
    assert first[0,0]==9 and first[1,1]==8


def ground_cloud(slope=0.0):
    yy, xx = np.mgrid[0:20, 0:20]
    return np.c_[xx.ravel() * .2 + .1, yy.ravel() * .2 + .1,
                 slope * xx.ravel() * .2].astype(np.float32)


@pytest.mark.parametrize("slope", [0.0, 0.25, -0.25])
def test_ground_not_vertical_obstacle(slope):
    mask, _ = vertical_columns(ground_cloud(slope), (20, 20), (0, 0), .2)
    assert not mask.any()


def test_thin_column_survives_smoothing_and_clears_on_next_ground_scan():
    ground = ground_cloud()
    pole = np.c_[np.full(14, 1.9), np.full(14, 2.1), np.linspace(.1, 1.4, 14)]
    occupied, _ = vertical_columns(np.r_[ground, pole], (20, 20), (0, 0), .2)
    assert occupied[10, 9]
    result = SimpleNamespace(baseline_risk=np.zeros((20, 20)),
                             final_risk_slices=[np.zeros((20, 20)) for _ in range(36)])
    preserve_collision_risk(result, occupied)
    assert result.baseline_risk[10, 9] == 1.0
    assert all(r[10, 9] == 1.0 for r in result.final_risk_slices)
    cleared, _ = vertical_columns(ground, (20, 20), (0, 0), .2)
    assert not cleared.any()


def test_gpu_cpu_agree():
    cp = pytest.importorskip("cupy")
    points = np.r_[ground_cloud(), [[1.9, 2.1, .5], [1.9, 2.1, .9]]]
    cpu, _ = vertical_columns(points, (20, 20), (0, 0), .2)
    gpu, _ = vertical_columns(points, (20, 20), (0, 0), .2, xp=cp)
    np.testing.assert_array_equal(cpu, cp.asnumpy(gpu))


@pytest.mark.parametrize("slope", [0.0, 0.5, -0.5])
def test_sparse_upper_returns_use_observed_ground_without_slope_false_alarm(slope):
    ground = ground_cloud(slope)
    elevation = ground[:, 2].reshape(20, 20)
    pole = np.array([[1.9, 2.1, slope * 1.8 + .6],
                     [2.1, 2.1, slope * 2.0 + .6]], np.float32)
    occupied, _ = vertical_columns(np.r_[ground, pole], (20, 20), (0, 0), .2,
                                   ground_elevation=elevation)
    assert occupied[10, 9] and occupied[10, 10]
    assert occupied.sum() == 2
    clear, _ = vertical_columns(ground, (20, 20), (0, 0), .2, ground_elevation=elevation)
    assert not clear.any()


def test_canopy_and_single_outlier_do_not_create_sparse_column():
    ground = ground_cloud()
    points = np.r_[ground, [[1.9, 2.1, 5.0], [2.1, 2.1, 5.2], [3.1, 3.1, .8]]]
    mask, _ = vertical_columns(points, (20, 20), (0, 0), .2,
                               ground_elevation=ground[:, 2].reshape(20, 20))
    assert not mask.any()


def test_trusted_ground_gpu_cpu_parity_and_edge_no_wrap():
    cp = pytest.importorskip("cupy")
    ground = ground_cloud(.25)
    elevation = ground[:, 2].reshape(20, 20)
    points = np.r_[ground, [[.1, .1, .8], [3.9, 3.9, 1.8]]]
    cpu, _ = vertical_columns(points, (20, 20), (0, 0), .2, ground_elevation=elevation)
    gpu, _ = vertical_columns(points, (20, 20), (0, 0), .2, ground_elevation=elevation, xp=cp)
    np.testing.assert_array_equal(cpu, cp.asnumpy(gpu))
    assert not cpu.any()


@pytest.mark.parametrize('with_ground', [False, True])
def test_fused_cuda_matches_cpu_for_sparse_signed_heights_edges_and_missing_ground(with_ground):
    cp = pytest.importorskip('cupy')
    rng = np.random.default_rng(1729)
    shape, origin, resolution = (19, 23), (-2.3, 1.1), .2
    points = rng.uniform([-2.7, .7, -2.], [2.7, 5.3, .4], (6000, 3)).astype(np.float32)
    points[:3] = [[np.nan, 0, 0], [0, np.inf, 0], [0, 2, np.nan]]
    trusted = rng.normal(-1.8, .05, shape).astype(np.float32) if with_ground else None
    if trusted is not None:
        trusted[rng.random(shape) < .4] = np.nan
    cpu, cpu_ground = vertical_columns(points, shape, origin, resolution, ground_elevation=trusted)
    gpu, gpu_ground = vertical_columns(points, shape, origin, resolution, ground_elevation=trusted, xp=cp)
    np.testing.assert_array_equal(cp.asnumpy(gpu), cpu)
    np.testing.assert_allclose(cp.asnumpy(gpu_ground), cpu_ground, rtol=0, atol=0, equal_nan=True)


def test_fused_cuda_handles_empty_and_extra_point_fields_like_cpu():
    cp = pytest.importorskip('cupy')
    ground = ground_cloud()
    xyz = np.r_[ground, [[1.9, 2.1, .5], [1.9, 2.1, .9]]].astype(np.float32)
    points = np.c_[xyz, np.ones(len(xyz), np.float32)]
    points[-1, 3] = np.nan
    for cloud in ([], points, points[:0]):
        cpu, cpu_ground = vertical_columns(cloud, (20, 20), (0, 0), .2)
        gpu, gpu_ground = vertical_columns(cloud, (20, 20), (0, 0), .2, xp=cp)
        np.testing.assert_array_equal(cp.asnumpy(gpu), cpu)
        np.testing.assert_allclose(cp.asnumpy(gpu_ground), cpu_ground, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize('backend', ['cpu', 'cuda'])
def test_low_variance_obstacle_top_cannot_be_its_own_ground_witness(backend):
    xp = np if backend == 'cpu' else pytest.importorskip('cupy')
    elevation = np.zeros((20, 20), np.float32)
    elevation[10, 9:11] = .8
    original = elevation.copy()
    points = np.array([[1.9, 2.1, .8], [2.1, 2.1, .8]], np.float32)
    occupied, ground = vertical_columns(points, elevation.shape, (0, 0), .2,
        ground_elevation=elevation, xp=xp)
    if backend == 'cuda':occupied, ground = xp.asnumpy(occupied), xp.asnumpy(ground)
    assert occupied[10, 9] and occupied[10, 10]
    assert occupied.sum() == 2
    assert ground[10, 9] == ground[10, 10] == 0
    np.testing.assert_array_equal(elevation, original)


@pytest.mark.parametrize('backend', ['cpu', 'cuda'])
def test_ground_witness_filter_preserves_diagonal_slopes_and_coarse_grids(backend):
    xp = np if backend == 'cpu' else pytest.importorskip('cupy')
    for resolution in (.2, 1.):
        y, x = np.mgrid[0:20, 0:20]
        elevation = (.5*x*resolution+.4*y*resolution).astype(np.float32)
        points = np.c_[(x.ravel()+.5)*resolution, (y.ravel()+.5)*resolution, elevation.ravel()]
        occupied, ground = vertical_columns(points, elevation.shape, (0, 0), resolution,
            ground_elevation=elevation, xp=xp)
        if backend == 'cuda':occupied, ground = xp.asnumpy(occupied), xp.asnumpy(ground)
        assert not occupied.any()
        np.testing.assert_array_equal(ground, elevation)


@pytest.mark.parametrize('backend', ['cpu', 'cuda'])
def test_lower_ground_hypothesis_cannot_remove_preexisting_obstacle_by_canopy_cutoff(backend):
    xp = np if backend == 'cpu' else pytest.importorskip('cupy')
    elevation = np.zeros((20, 20), np.float32)
    elevation[10, 9:11] = .8
    points = np.array([[1.9, 2.1, 2.0], [2.1, 2.1, 2.0]], np.float32)
    mask, ground = vertical_columns(points, elevation.shape, (0, 0), .2,
        ground_elevation=elevation, xp=xp)
    if backend == 'cuda':mask, ground = xp.asnumpy(mask), xp.asnumpy(ground)
    assert mask[10, 9] and mask[10, 10]
    assert ground[10, 9] == ground[10, 10] == 0


@pytest.mark.parametrize('backend', ['cpu', 'cuda'])
def test_single_return_policy_needs_observed_ground_and_does_not_block_high_canopy(backend):
    xp = np if backend == 'cpu' else pytest.importorskip('cupy')
    points = np.array([[1.9, 2.1, .6], [3.1, 3.1, 5.]], np.float32)
    mask, _ = vertical_columns(points, (20, 20), (0, 0), .2, min_hits=1,
        ground_elevation=np.zeros((20, 20), np.float32), xp=xp)
    if backend == 'cuda':mask = xp.asnumpy(mask)
    assert mask[10, 9] and not mask[15, 15]
    unsupported, _ = vertical_columns(points, (20, 20), (0, 0), .2, min_hits=1, xp=xp)
    assert not unsupported.any()


@pytest.mark.parametrize('origin',[(.1,-.3),(-5.400000095367432,-10.300000190734863),(-1000.1,999.7)])
def test_coarse_detection_and_fine_witnesses_use_the_same_half_open_cell_intervals(origin):
    cp=pytest.importorskip('cupy')
    resolution=float(np.float32(.2));shape=(20,20)
    # Float32 sensor coordinates immediately either side of Float64 map edges.
    xy=[]
    for row in (3,7,13):
        for col in (4,8,14):
            x=np.float32(origin[0]+col*resolution);y=np.float32(origin[1]+row*resolution)
            for target in (-np.inf,np.inf):
                xy.append((np.nextafter(x,np.float32(target)),np.nextafter(y,np.float32(target))))
    edges=[origin[i]+np.arange(21)*resolution for i in (0,1)]
    expected=np.zeros(shape,bool);points=[]
    for x,y in xy:
        # Independent interval membership, without rounding the map geometry.
        cols=np.flatnonzero((edges[0][:-1]<=float(x))&(float(x)<edges[0][1:]))
        rows=np.flatnonzero((edges[1][:-1]<=float(y))&(float(y)<edges[1][1:]))
        assert len(cols)==len(rows)==1
        expected[rows[0],cols[0]]=True;points.extend([(x,y,0.),(x,y,.5)])
    points=np.asarray(points,np.float32)
    cpu,_=vertical_columns(points,shape,origin,resolution,min_height=.1,min_hits=1,
        ground_elevation=np.zeros(shape,np.float32))
    gpu,_=vertical_columns(points,shape,origin,resolution,min_height=.1,min_hits=1,
        ground_elevation=np.zeros(shape,np.float32),xp=cp)
    np.testing.assert_array_equal(cpu,expected);np.testing.assert_array_equal(cp.asnumpy(gpu),expected)
    bits=collision_subcells(points,cpu,origin,resolution,factor=4,measured_floor=np.zeros(shape))
    np.testing.assert_array_equal(bits>0,expected)
