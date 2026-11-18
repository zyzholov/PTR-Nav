"""Avoiding a device-to-host copy must preserve the complete FIC prior."""
import numpy as np
import pytest
import torch
from fsgp_bgk_se2.shared_terrain import residual_fic_features


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_host_training_coordinates_preserve_features_and_diagonal_correction(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.set_num_threads(1)
    rng=np.random.default_rng(292);xy=rng.uniform(-2,2,(83,2))[rng.permutation(83)]
    query=np.r_[xy[::3],rng.uniform(-2,2,(61,2))]
    training=torch.as_tensor(xy,dtype=torch.float64,device=device)
    test=torch.as_tensor(query,dtype=torch.float64,device=device)
    gate=torch.as_tensor(np.clip(np.linalg.norm(xy,axis=1)-.1,0,1),device=device)
    options=dict(count=31,lengthscale=.35,variance=.016,training_gate=gate)
    old=residual_fic_features(training,test,**options)
    new=residual_fic_features(training,test,training_xy=xy,**options)
    for a,b in zip(old,new):torch.testing.assert_close(a,b,rtol=0,atol=0)
    train_feature,_,diagonal,_=new
    torch.testing.assert_close(train_feature.square().sum(dim=1)+diagonal,.016*gate.square(),rtol=1e-12,atol=1e-12)
