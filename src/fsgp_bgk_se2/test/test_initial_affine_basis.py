"""Structural initial-stance prior, not synthetic ground observations."""
import numpy as np
import pytest
import torch
from fsgp_bgk_se2.probabilistic_risk import ProbabilityParams,_geometry
from fsgp_bgk_se2.shared_terrain import SharedTerrainParams,shared_terrain_prediction,shared_geometric_probability


def scene():
    x,y=np.meshgrid((np.arange(40)-19.5)*.2,(np.arange(40)-19.5)*.2)
    measured=x>1.
    h=.06*((x>2.35)&(x<2.75))+.045*((x>3.35)&(x<3.75))
    initial=np.zeros(h.shape,bool)
    physical=ProbabilityParams(internal_steps=True)
    for dr,dc in _geometry(physical,0)[0]:initial[20+dr,20+dc]=True
    return x,y,h,measured,initial,physical


def predict(device,flat=True,initial=True,pit=False):
    torch.set_num_threads(1)
    x,y,h,seen,prior,physical=scene()
    points=np.c_[x[seen],y[seen],h[seen]]
    if pit:points=np.r_[points,[[x[20,20],y[20,20],-.4],[x[20,20]+.01,y[20,20],-.41]]]
    q=shared_terrain_prediction(np.where(seen,h,np.nan),np.full(h.shape,.0001),seen,
        x,y,points,0.,.2,np.zeros(h.shape,bool),device=device,
        params=SharedTerrainParams(max_distance_m=1.25,max_extrapolation_m=.75,flat_initial_basis=flat),
        initial_support=prior if initial else None,initial_height=np.zeros_like(h) if initial else None,
        positive_veto_height=None)
    return q,(x,y,h,seen,prior,physical)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_fixed_flat_latent_field_uses_the_same_affine_basis_and_full_covariance(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    q,(x,y,h,seen,prior,p)=predict(device)
    old,_=predict(device,flat=False)
    assert q is not None and q.valid[prior].all()
    assert q.flat_initial_basis_used and not old.flat_initial_basis_used
    assert q.training_points==old.training_points and not (q.completed&prior).any()
    # The gated field is affine here in BOTH mean and shared latent error.
    # Flattening the mean after fitting would fail the covariance assertion.
    a=np.c_[x[prior],y[prior],np.ones(prior.sum())];residual=np.eye(len(a))-a@np.linalg.pinv(a)
    assert np.linalg.norm(residual@q.mean[prior])<2e-6
    factor=q.factor.detach().cpu().numpy()[prior]
    assert np.linalg.norm(residual@factor)<2e-6
    covariance=factor@factor.T
    assert np.linalg.eigvalsh(covariance.astype(float)).min()>-1e-7
    assert q.discrepancy>=.01 # Model error was not removed to make the test pass.
    assert not q.valid[(x<-.6)&~prior].any()
    rr,cc,upper=shared_geometric_probability(q,p,np.zeros(h.shape,bool))[0]
    assert upper[(rr==20)&(cc==20)][0]<.05


def test_initial_kernel_cpu_cuda_agree_and_raw_pit_evidence_still_revokes_support():
    cpu,(_,_,h,_,_,p)=predict('cpu')
    if torch.cuda.is_available():
        gpu,_=predict('cuda')
        np.testing.assert_array_equal(cpu.valid,gpu.valid)
        np.testing.assert_allclose(cpu.mean,gpu.mean,atol=2e-6)
        np.testing.assert_allclose(cpu.variance,gpu.variance,atol=2e-7)
        a=shared_geometric_probability(cpu,p,np.zeros(h.shape,bool))
        b=shared_geometric_probability(gpu,p,np.zeros(h.shape,bool))
        for old,new in zip(a,b):
            np.testing.assert_array_equal(old[0],new[0]);np.testing.assert_array_equal(old[1],new[1])
            np.testing.assert_allclose(old[2],new[2],atol=2e-5)
    pit,_=predict('cpu',pit=True)
    assert pit.contradicted[20,20] and not pit.valid[20,20]


def test_no_explicit_initial_hypothesis_leaves_the_default_model_unchanged():
    plain,_=predict('cpu',flat=False,initial=False)
    gated,_=predict('cpu',flat=True,initial=False)
    assert not gated.flat_initial_basis_used
    np.testing.assert_array_equal(plain.valid,gated.valid)
    np.testing.assert_array_equal(plain.mean,gated.mean)
    np.testing.assert_array_equal(plain.variance,gated.variance)
    np.testing.assert_array_equal(plain.factor.numpy(),gated.factor.numpy())
