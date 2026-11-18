"""Exact discrete geometry parity, including float grid faces and packed bounds."""
import numpy as np
import pytest
from fsgp_bgk_se2.obstacle_columns import collision_subcells


@pytest.fixture(scope='module',autouse=True)
def cuda():
    cp=pytest.importorskip('cupy')
    if not cp.cuda.runtime.getDeviceCount():pytest.skip('CUDA unavailable')


@pytest.mark.parametrize('factor',[2,4])
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('floor_mode',['none','flat','slope','discontinuity'])
def test_gpu_bits_and_all_outward_bounds_equal_cpu_for_actual_boxes(factor,dtype,floor_mode):
    rng=np.random.default_rng(1823);shape=(17,23);res=.2;origin=(-2.4,-1.7)
    yy,xx=np.indices(shape);floor=(.03*xx-.02*yy).astype(dtype)
    gradient=np.stack((np.full(shape,.15),np.full(shape,-.10)))
    if floor_mode=='flat':floor[:]=-.4;gradient[:]=0
    if floor_mode=='discontinuity':floor[:,8:]+=1.;floor[3:6,4:7]=np.nan
    points=np.c_[rng.uniform(-3,3,3500),rng.uniform(-2,2,3500),rng.uniform(-.7,2.,3500)].astype(dtype)
    # Include exact physical and quantization faces, both coordinate signs.
    a=np.array([-.1,0.,.02,.05,.10,.15,.18,.20,.4],dtype=dtype)
    faces=np.c_[a+origin[0],np.full(len(a),origin[1]+.2),np.full(len(a),.4)]
    points=np.r_[points,faces,np.nextafter(faces,dtype(np.inf)),
                 np.nextafter(faces,dtype(-np.inf)),[[np.nan,0,0],[0,np.inf,0],[0,0,np.nan]]].astype(dtype)
    occupied=rng.random(shape)<.6
    for pad in (0.,.02,.1):
        options=dict(factor=factor,xy_padding_m=pad,return_bounds=True)
        if floor_mode!='none':options.update(measured_floor=floor,measured_floor_gradient=gradient)
        expected=collision_subcells(points,occupied,origin,res,**options)
        actual=collision_subcells(points,occupied,origin,res,backend='cupy',**options)
        np.testing.assert_array_equal(actual[0],expected[0])
        np.testing.assert_array_equal(actual[1],expected[1])


@pytest.mark.parametrize('factor',[2,4])
@pytest.mark.parametrize('points',[None,np.empty((0,3)),np.array([[0.,0.,.5],[0.,0.,.5]])])
def test_missing_rays_keep_full_square_fallback_and_point_order_never_changes_bits(factor,points):
    occupied=np.ones((3,4),bool)
    for bounds in (False,True):
        options=dict(factor=factor,return_bounds=bounds)
        a=collision_subcells(points,occupied,(-.2,-.2),.2,**options)
        b=collision_subcells(points,occupied,(-.2,-.2),.2,backend='cupy',**options)
        if bounds:
            np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
        else:np.testing.assert_array_equal(a,b)
    rng=np.random.default_rng(9);p=np.c_[rng.uniform(-.2,.5,500),rng.uniform(-.2,.3,500),np.ones(500)]
    a=collision_subcells(p,occupied,(-.2,-.2),.2,factor=factor,return_bounds=True,backend='cupy')
    b=collision_subcells(p[rng.permutation(len(p))],occupied,(-.2,-.2),.2,factor=factor,return_bounds=True,backend='cupy')
    np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])


def test_reversed_strided_floor_gradient_and_mask_and_empty_grid_keep_cpu_layout():
    rng=np.random.default_rng(79);occupied=(rng.random((8,12))>.3)[:,::-2]
    floor=np.zeros((8,12))[:,::-2];slope=np.zeros((2,8,12))[:,:,::-2]
    points=np.c_[rng.uniform(-.6,.6,500),rng.uniform(-.8,.8,500),rng.uniform(0.,1.,500)]
    for shape in (occupied.shape,(0,6)):
        mask=occupied[:shape[0]]
        options=dict(factor=4,return_bounds=True,measured_floor=floor[:shape[0]],measured_floor_gradient=slope[:,:shape[0]])
        a=collision_subcells(points,mask,(-.6,-.8),.2,**options)
        b=collision_subcells(points,mask,(-.6,-.8),.2,backend='cupy',**options)
        np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
