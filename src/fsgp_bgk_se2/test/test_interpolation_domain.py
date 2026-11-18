"""Directed qualification of support geometry, not empirical safety calibration."""
import numpy as np
import pytest
import torch

from fsgp_bgk_se2.feature_sgp import FeatureSgpParams,FeatureSgpPredictor
from fsgp_bgk_se2.fresh_completion import fresh_local_prediction
from fsgp_bgk_se2.supported_completion import supported_completion


def surface(device,limit,*,one_sided=False,contradiction=False):
    torch.set_num_threads(1)
    x,y=np.meshgrid((np.arange(30)-14.5)*.2,(np.arange(30)-14.5)*.2)
    z=.03*x-.02*y
    observed=np.ones(x.shape,bool);observed[:,13:17]=False
    if one_sided:observed &= x<-.3
    height=np.where(observed,z,np.nan);variance=np.full(x.shape,.0001)
    points=np.c_[x[observed],y[observed],z[observed]]
    if contradiction:
        points=np.r_[points,[[x[15,15],y[15,15],z[15,15]-.25],
                             [x[15,15]+.01,y[15,15]+.01,z[15,15]-.26]]]
    predictor=FeatureSgpPredictor(FeatureSgpParams(device=device,training_steps=0,
        local_neighbors=64,local_nearest_m=limit))
    prediction=fresh_local_prediction(predictor,height,variance,observed,x,y,points,0.,.2,
        blocked=np.zeros(x.shape,bool))
    completed=supported_completion(height,variance,observed,prediction,max_distance_m=limit)
    return prediction,completed,z,observed


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_bounded_surrounded_interpolation_is_distinct_from_extrapolation(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    baseline,_,_,_=surface(device,.35)
    assert not baseline.accepted[15,15]
    prediction,(height,_,valid,used),truth,observed=surface(device,.6)
    assert prediction.accepted[15,15] and used[15,15] and valid[15,15]
    assert .35<prediction.nearest_distance[15,15]<=.6
    assert prediction.latent_var[15,15] <= .04**2
    assert height[15,15]==pytest.approx(truth[15,15],abs=2e-5)
    assert not used[observed].any()
    assert not surface(device,.6,one_sided=True)[0].accepted[15,15]
    assert not surface(device,.6,contradiction=True)[0].accepted[15,15]


def test_extended_interpolation_cpu_cuda_parity():
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    a,_,_,_=surface('cpu',.6);b,_,_,_=surface('cuda',.6)
    np.testing.assert_array_equal(a.accepted,b.accepted)
    np.testing.assert_allclose(a.mean,b.mean,atol=2e-5,equal_nan=True)
    np.testing.assert_allclose(a.latent_var,b.latent_var,atol=2e-6)


def test_uncertainty_and_explicit_distance_still_gate_extended_prediction():
    prediction,(height,variance,valid,_),_,observed=surface('cpu',.6)
    before=height.copy()
    prediction.latent_var[15,15]=.041**2
    _,_,_,used=supported_completion(height,variance,observed,prediction,max_distance_m=.6)
    assert not used[15,15]
    prediction.latent_var[15,15]=.001**2
    _,_,_,used=supported_completion(before,variance,observed,prediction,max_distance_m=.35)
    assert not used[15,15]
