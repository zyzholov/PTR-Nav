"""Same current witness planes, without an intervening host transfer."""
import numpy as np
import pytest


@pytest.mark.parametrize('broad',[False,True])
@pytest.mark.parametrize('pattern',['dense','sparse','empty','discontinuous'])
def test_resident_two_radius_proof_matches_separate_gpu_passes_and_current_input_ownership(broad,pattern):
    cp=pytest.importorskip('cupy')
    if cp.cuda.runtime.getDeviceCount()<1:pytest.skip('CUDA unavailable')
    from fsgp_bgk_se2.overhead_clearance_cuda import plane_envelope,resident_plane_envelopes
    y,x=np.mgrid[:25,:29];z=(.02*x*.2-.015*y*.2).astype(np.float32)
    floor=np.ones(z.shape,bool);floor[:2,:3]=False
    witness=np.ones(z.shape,bool)if pattern=='dense'else (x%3==0)&(y%3==0)
    if pattern=='empty':witness[:]=False
    labels=np.zeros(z.shape,np.int32)
    if pattern=='discontinuous':
        z[:,15:]+=.3;labels[:,15:]=1
    columns=np.zeros(z.shape,bool);columns[10:12,17:19]=True;witness&=~columns
    direct=witness.copy();owner=labels.copy()
    args=(z,floor,witness,direct,columns,labels,owner,.2,.65,.1,.015)
    before=[v.copy()for v in args if isinstance(v,np.ndarray)]
    local=plane_envelope(*args,.9,.04)
    missing=~np.isfinite(local)&floor
    expected=(np.where(missing,plane_envelope(*args,3.,.10,query_mask=missing),local)
              if broad and missing.any()else local)
    actual=resident_plane_envelopes(*args,.9,.04,measured_fallback=broad)
    np.testing.assert_array_equal(actual,expected)
    for previous,current in zip(before,(v for v in args if isinstance(v,np.ndarray))):
        np.testing.assert_array_equal(previous,current)
    # An immutable geometry cache must not keep the last frame's heights.
    z+=.25
    next_=resident_plane_envelopes(*args,.9,.04,measured_fallback=broad)
    np.testing.assert_array_equal(np.isfinite(next_),np.isfinite(actual))
    np.testing.assert_allclose(next_[np.isfinite(next_)],actual[np.isfinite(actual)]+.25,atol=1e-6)
