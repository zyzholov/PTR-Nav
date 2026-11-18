"""Geometry-only caches must not retain observations or alter arithmetic."""
from dataclasses import replace
import math
import numpy as np
import pytest
from fsgp_bgk_se2.bgk_seb import BgkSebEstimator,BgkSebParams


@pytest.mark.parametrize('field,value',[
    ('resolution',.15),('yaw_bins',19),('footprint_front',.53),
    ('footprint_back',.43),('footprint_half_width',.38),('footprint_margin',.12),
    ('footprint_samples_x',7),('footprint_samples_y',7)])
def test_yaw_key_rebuilds_each_geometry_parameter_and_preserves_sample_order(field,value):
    e=BgkSebEstimator(BgkSebParams(resolution=.2,yaw_bins=12,prefer_gpu=False));first=e._yaw_geometry()
    assert e._yaw_geometry() is first
    e.params=replace(e.params,**{field:value});key,actual=e._yaw_geometry()
    assert key != first[0]
    expected=[]
    for k in range(max(1,int(e.params.yaw_bins))):
        yaw=(float(k)+.5)*2*math.pi/float(e.params.yaw_bins);c,s=math.cos(yaw),math.sin(yaw)
        samples=[]
        for x,y in e._footprint_offsets():
            dr,dc=e._offset_to_shift(yaw,x,y)
            samples.append((dr,dc,np.float32(e.params.resolution*(c*dc+s*dr)),
                            np.float32(e.params.resolution*(-s*dc+c*dr))))
        expected.append(tuple(samples))
    assert actual==tuple(expected)
    assert e._yaw_geometry_cache==(key,actual)


@pytest.mark.parametrize('field,value',[('resolution',.15),('bgk_radius_m',.7),('bgk_sigma_m',.4)])
def test_kernel_cache_key_preserves_original_weights(field,value):
    e=BgkSebEstimator(BgkSebParams(resolution=.2,yaw_bins=12,prefer_gpu=False));first=e._kernel_geometry()
    assert e._kernel_geometry() is first
    e.params=replace(e.params,**{field:value});key,actual=e._kernel_geometry()
    assert key!=first[0] and actual==tuple(e._kernel_offsets())


@pytest.mark.parametrize('gpu',[False,True])
def test_warm_cache_is_bit_identical_to_uncached_computation_on_new_observations(gpu):
    p=BgkSebParams(resolution=.2,prefer_gpu=gpu,gpu_backend='cupy',yaw_bins=12,store_diagnostic_slices=True)
    cached=BgkSebEstimator(p)
    if gpu and cached.backend_name!='cupy':pytest.skip('CUDA unavailable')
    rng=np.random.default_rng(510)
    def compare(seed):
        z=rng.normal(0.,.02,(30,28)).astype(np.float32)
        v=np.full_like(z,.001);seen=rng.random(z.shape)>.1
        baseline=[rng.uniform(0,.8,z.shape).astype(np.float32)for _ in range(p.yaw_bins)]
        # New estimator has no host or device cache and receives same inputs.
        fresh=BgkSebEstimator(replace(cached.params))
        a=cached.compute(z,v,seen,baseline);b=fresh.compute(z,v,seen,baseline)
        for name in ('bgk_mean','bgk_var','bgk_traversability','final_risk_slices',
                     'seb_risk_slices','seb_pitch_risk_slices','seb_roll_risk_slices',
                     'seb_step_risk_slices','seb_support_slices'):
            np.testing.assert_array_equal(getattr(a,name),getattr(b,name))
        return cached._yaw_geometry_cache,cached._cupy_yaw_geometry_cache
    old=compare(0);new=compare(1)
    assert old[0] is new[0]
    if gpu:assert old[1] is new[1]
    cached.params=replace(cached.params,resolution=.15);different=compare(2)
    assert different[0] is not new[0]
    if gpu:assert different[1] is not new[1]


def test_device_cached_uploads_are_ready_on_another_cuda_stream():
    e=BgkSebEstimator(BgkSebParams(resolution=.2,prefer_gpu=True,gpu_backend='cupy',yaw_bins=8))
    if e.backend_name!='cupy':pytest.skip('CUDA unavailable')
    cp=e._cupy;z=np.zeros((30,30),np.float32);v=np.full_like(z,.001);seen=np.ones(z.shape,bool)
    with cp.cuda.Stream(non_blocking=True):e.compute(z,v,seen,[z])
    arrays=e._cupy_yaw_geometry_cache
    with cp.cuda.Stream(non_blocking=True):a=e.compute(z,v,seen,[z])
    b=BgkSebEstimator(replace(e.params)).compute(z,v,seen,[z])
    assert e._cupy_yaw_geometry_cache is arrays
    np.testing.assert_array_equal(a.final_risk_slices,b.final_risk_slices)
