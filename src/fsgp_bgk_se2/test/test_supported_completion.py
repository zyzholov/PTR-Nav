from types import SimpleNamespace
import numpy as np
import pytest
from fsgp_bgk_se2.supported_completion import supported_completion
from fsgp_bgk_se2.bgk_seb import BgkSebEstimator,BgkSebParams


def prediction(shape):
    n=np.prod(shape)
    return SimpleNamespace(mean=np.full(n,-.48),latent_var=np.full(n,.0004),
        nearest_distance=np.full(n,.1),accepted=np.ones(n,bool),local_conditioned=np.ones(n,bool))


def test_only_supported_small_uncertainty_interpolation_is_completed():
    shape=(8,8);z=np.full(shape,-.48);v=np.full(shape,.0004);valid=np.zeros(shape,bool)
    p=prediction(shape);p.local_conditioned[0]=False;p.nearest_distance[1]=1.
    p.latent_var[2]=.01;p.mean[3]=np.nan
    blocked=np.zeros(shape,bool);blocked.ravel()[4]=True
    out,var,mask,completed=supported_completion(z,v,valid,p,blocked=blocked)
    assert not completed.ravel()[:5].any() and completed.ravel()[5:].all()
    assert not mask.ravel()[:5].any() and np.isfinite(out).all()
    valid[:]=True;out,_,mask,completed=supported_completion(z,v,valid,p)
    assert mask.all() and not completed.any();np.testing.assert_equal(out,z)


def test_no_local_geometry_mask_cannot_release_unknown_cells():
    p=prediction((8,8));p.local_conditioned=None
    z=np.full((8,8),np.nan);valid=np.zeros((8,8),bool)
    _,_,out,completed=supported_completion(z,np.zeros_like(z),valid,p)
    assert not out.any() and not completed.any()


@pytest.mark.parametrize("backend",["cpu","cupy"])
def test_predicted_flat_hole_is_not_a_zero_height_observed_step(backend):
    if backend=="cupy":pytest.importorskip("cupy")
    shape=(32,32);observed=np.full(shape,-.48,np.float32);observed[15,15]=np.nan
    z=np.nan_to_num(observed,nan=-.48);valid=np.ones(shape,bool)
    result=BgkSebEstimator(BgkSebParams(resolution=.2,yaw_bins=8,gpu_backend=backend)).compute(
        z,np.full(shape,.0004,np.float32),valid,[np.zeros(shape,np.float32)],observed_elevation=observed)
    assert result.backend==backend
    assert max(r[15,15] for r in result.final_risk_slices)<.1
