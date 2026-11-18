"""Collision overlay optimization must retain every original device predicate."""
import numpy as np
import pytest
from fsgp_bgk_se2.obstacle_columns import collision_pose_masks
from fsgp_bgk_se2 import collision_pose_cuda


@pytest.fixture(scope='module',autouse=True)
def cuda():
    cp=pytest.importorskip('cupy')
    if not cp.cuda.runtime.getDeviceCount():pytest.skip('CUDA unavailable')


@pytest.mark.parametrize('factor',[2,4])
@pytest.mark.parametrize('resolution',[.2,.20000000298023224])
@pytest.mark.parametrize('body',[(.521,.521,.460),(.68,.48,.50)])
def test_cached_device_stencil_equals_original_device_and_cpu(monkeypatch,factor,resolution,body):
    rng=np.random.default_rng(82391);bits=rng.integers(0,1<<(factor*factor),(17,23),dtype=np.uint32)
    bits[rng.random(bits.shape)<.9]=0
    # Include each subcell bit on map edges, where negative integer division
    # or an accidental wrapped coarse offset can hide a collision.
    for bit in range(factor*factor):bits[0,bit]=1<<bit
    bits[-1,-1]=(1<<(factor*factor))-1
    cached=collision_pose_masks(bits,resolution,*body,72,backend='cupy',factor=factor)
    with monkeypatch.context() as patch:
        patch.setattr(collision_pose_cuda,'cached_collision_poses',lambda *args:None)
        original=collision_pose_masks(bits,resolution,*body,72,backend='cupy',factor=factor)
    np.testing.assert_array_equal(cached,original)
    cpu=collision_pose_masks(bits,resolution,*body,72,factor=factor)
    np.testing.assert_array_equal(cached,cpu)
    # Device geometry is cached; each invocation must still consume new bits.
    changed=np.zeros_like(bits);changed[-1,0]=1
    again=collision_pose_masks(changed,resolution,*body,72,backend='cupy',factor=factor)
    np.testing.assert_array_equal(again,collision_pose_masks(changed,resolution,*body,72,factor=factor))


def test_geometry_cache_is_bounded_and_empty_or_strided_masks_keep_layout():
    assert collision_pose_cuda._stencil.cache_info().maxsize==4
    bits=np.zeros((10,14),np.uint32);bits[2,4]=3;bits=bits[::-2,::2]
    actual=collision_pose_masks(bits,.2,.521,.521,.460,36,backend='cupy',factor=4)
    np.testing.assert_array_equal(actual,collision_pose_masks(bits,.2,.521,.521,.460,36,factor=4))
    assert collision_pose_masks(np.zeros((0,3),np.uint32),.2,.521,.521,.460,36,backend='cupy',factor=4).shape==(36,0,3)
