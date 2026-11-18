import numpy as np
import pytest

torch = pytest.importorskip("torch")
gpytorch=pytest.importorskip("gpytorch")

from fsgp_bgk_se2.feature_sgp import FeatureSgpParams, FeatureSgpPredictor, spatial_sample_indices


def observations():
    x, y = np.meshgrid(np.linspace(-2, 2, 12), np.linspace(-2, 2, 12))
    z = 0.2 + 0.12 * x - 0.06 * y
    visible = ~((np.abs(x) < 0.6) & (np.abs(y) < 0.6))
    return np.column_stack((x[visible], y[visible], z[visible])).astype(np.float32)


def predictor(device="cpu"):
    return FeatureSgpPredictor(FeatureSgpParams(
        device=device, max_train_points=80, inducing_points=24, training_steps=2))


def test_unobserved_plane_and_physical_gradient():
    torch.set_num_threads(1)
    model = predictor().fit(observations())
    result = model.predict(np.array([[0, 0], [0.2, -0.1]], np.float32), gradients=True)
    np.testing.assert_allclose(result.mean, [0.2, 0.23], atol=0.005)
    np.testing.assert_allclose(result.gradient_xy, [[0.12, -0.06]] * 2, atol=0.005)
    assert result.supported.all()
    assert np.isfinite(result.latent_var).all()
    assert (result.observation_var > result.latent_var).all()


def test_far_extrapolation_is_not_accepted():
    result = predictor().fit(observations()).predict(np.array([[10, 10]], np.float32))
    assert not result.supported[0]
    assert not result.accepted[0]


@pytest.mark.parametrize("device",["cpu","cuda"])
def test_float32_rounding_cannot_move_a_query_inside_the_measured_hull(device):
    if device=="cuda" and not torch.cuda.is_available():pytest.skip("CUDA unavailable")
    torch.set_num_threads(1)
    x,y=np.meshgrid(np.linspace(99,101,6),np.linspace(99,101,6))
    xyz=np.c_[x.ravel(),y.ravel(),np.zeros(x.size)]
    model=FeatureSgpPredictor(FeatureSgpParams(device=device,max_train_points=32,
        inducing_points=8,training_steps=0,local_conditioning=False)).fit(xyz)
    query=np.array([[101.+2e-6,100.],[101.,100.]],dtype=np.float64)
    np.testing.assert_array_equal(query[0].astype(np.float32),query[1].astype(np.float32))
    prediction=model.predict(query)
    np.testing.assert_array_equal(prediction.supported,[False,True])
    assert not prediction.accepted[0]


def test_query_cannot_contain_hidden_height():
    model = predictor().fit(observations())
    with pytest.raises(ValueError):
        model.predict(np.array([[0, 0, 99]], np.float32))
    with pytest.raises(ValueError):
        model.fit(np.array([[0, 0, np.nan]], np.float32))


def test_repeated_query_does_not_change_training_observations():
    model = predictor().fit(observations())
    reference = model.reference_xy.clone()
    query = np.array([[0, 0]], np.float32)
    first = model.predict(query)
    second = model.predict(query)
    np.testing.assert_allclose(first.mean, second.mean)
    assert torch.equal(reference, model.reference_xy)


def test_cuda_matches_cpu_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    points = observations()
    query = np.array([[0, 0], [0.2, -0.1]], np.float32)
    cpu = predictor().fit(points).predict(query)
    cuda = predictor("cuda").fit(points).predict(query)
    assert cuda.device == "cuda"
    np.testing.assert_allclose(cuda.mean, cpu.mean, atol=2e-4, rtol=2e-3)
    np.testing.assert_allclose(cuda.latent_var, cpu.latent_var, atol=2e-5, rtol=0.05)


def test_spatial_budget_is_order_independent_and_keeps_all_quadrants():
    rng = np.random.default_rng(71)
    points = rng.normal(size=(4000, 3)).astype(np.float32)
    selected = points[spatial_sample_indices(points, 96)]
    shuffled = points[rng.permutation(len(points))]
    np.testing.assert_array_equal(selected, shuffled[spatial_sample_indices(shuffled, 96)])
    assert len(selected) == 96
    assert len(np.unique(selected[:, :2] > 0, axis=0)) == 4


def test_flat_and_degenerate_geometry_contract():
    model = predictor().fit(observations())
    result = model.predict(np.array([[0, 0], [10, 10]], np.float32))
    assert (result.latent_var >= model.params.noise_std_m ** 2).all()
    assert result.latent_var[1] > result.latent_var[0]
    with pytest.raises(ValueError, match="two-dimensional"):
        predictor().fit(np.c_[np.arange(12), np.zeros(12), np.zeros(12)])


def test_nonlinear_cuda_cpu_agreement():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    rng = np.random.default_rng(172)
    xy = rng.uniform(-2, 2, (300, 2)).astype(np.float32)
    xyz = np.c_[xy, .2 * np.sin(xy[:, 0]) * np.cos(xy[:, 1])]
    query = rng.uniform(-2.2, 2.2, (60, 2)).astype(np.float32)
    cpu = predictor().fit(xyz).predict(query)
    gpu = predictor("cuda").fit(xyz).predict(query)
    np.testing.assert_allclose(gpu.mean, cpu.mean, atol=1e-3, rtol=.01)
    np.testing.assert_allclose(gpu.latent_var, cpu.latent_var, atol=1e-4, rtol=.05)
    np.testing.assert_array_equal(gpu.accepted, cpu.accepted)


def test_wide_hole_keeps_global_model_instead_of_local_extrapolation():
    xy=np.stack(np.meshgrid(np.linspace(-3,3,25),np.linspace(-3,3,25)),axis=-1).reshape(-1,2)
    xyz=np.c_[xy,.2*np.sin(xy[:,0])];xyz=xyz[(np.abs(xy)>1).any(axis=1)]
    args=dict(device="cpu",max_train_points=80,inducing_points=24,training_steps=2)
    global_only=FeatureSgpPredictor(FeatureSgpParams(**args,local_conditioning=False)).fit(xyz)
    local=FeatureSgpPredictor(FeatureSgpParams(**args)).fit(xyz)
    query=np.array([[0.,0.]],np.float32)
    a,b=global_only.predict(query),local.predict(query)
    assert not b.local_conditioned[0]
    np.testing.assert_allclose(b.mean,a.mean,atol=1e-5)
    np.testing.assert_allclose(b.latent_var,a.latent_var,atol=1e-5)


def test_dense_local_surface_keeps_detail_lost_by_small_global_budget():
    xy=np.stack(np.meshgrid(np.linspace(-3,3,65),np.linspace(-3,3,65)),axis=-1).reshape(-1,2)
    height=lambda a:.15*np.sin(2*a[:,0])*np.cos(1.5*a[:,1])
    xyz=np.c_[xy,height(xy)];query=np.array([[.23,.37],[1.17,-.63],[-.53,1.13]],np.float32)
    args=dict(device="cpu",max_train_points=32,inducing_points=12,training_steps=2)
    a=FeatureSgpPredictor(FeatureSgpParams(**args,local_conditioning=False)).fit(xyz).predict(query)
    b=FeatureSgpPredictor(FeatureSgpParams(**args)).fit(xyz).predict(query)
    assert b.local_conditioned.all()
    assert np.mean(np.abs(b.mean-height(query)))<.01
    assert np.mean(np.abs(b.mean-height(query)))<np.mean(np.abs(a.mean-height(query)))


def test_incoherent_local_measurements_cannot_claim_precise_conditioning():
    rng=np.random.default_rng(1221);xy=rng.uniform(-2,2,(800,2));xyz=np.c_[xy,rng.normal(0,1,800)]
    model=predictor().fit(xyz)
    result=model.predict(np.array([[0,0],[.2,-.1]],np.float32))
    assert not result.local_conditioned.any()


@pytest.mark.parametrize("device",["cpu","cuda"])
def test_diagonal_extrapolation_is_not_a_supported_prediction(device):
    if device=="cuda" and not torch.cuda.is_available():pytest.skip("CUDA unavailable")
    xy=np.stack(np.meshgrid(np.linspace(-.8,.8,25),np.linspace(-.8,.8,25)),axis=-1).reshape(-1,2)
    xy=xy[xy.sum(axis=1)>.25]
    xyz=np.c_[xy,.3+.1*xy[:,0]-.2*xy[:,1]]
    result=predictor(device).fit(xyz).predict(np.array([[0,0],[.4,.4]],np.float32))
    assert result.nearest_distance[0]<.35
    assert not result.local_conditioned[0]
    assert not result.supported[0] and not result.accepted[0]
    assert result.supported[1] and result.accepted[1]


def test_global_plane_extrapolation_is_finite_but_not_accepted():
    xyz=observations();xyz=xyz[xyz[:,0]>=0]
    params=FeatureSgpParams(device="cpu",training_steps=2,max_train_points=80,inducing_points=24,local_conditioning=False)
    result=FeatureSgpPredictor(params).fit(xyz).predict(np.array([[-.01,0]],np.float32))
    assert np.isfinite(result.mean).all() and result.nearest_distance[0]<1.2
    assert not result.supported[0] and not result.accepted[0]


def test_observation_on_convex_hull_boundary_remains_supported():
    xyz=observations();q=xyz[[0,-1],:2]
    result=predictor().fit(xyz).predict(q)
    assert result.supported.all() and result.accepted.all()


@pytest.mark.parametrize("device",["cpu","cuda"])
def test_chunk_size_and_nondefault_cuda_stream_do_not_change_predictions(device):
    from contextlib import nullcontext
    if device=="cuda" and not torch.cuda.is_available():pytest.skip("CUDA unavailable")
    torch.set_num_threads(1)
    context=torch.cuda.stream(torch.cuda.Stream()) if device=="cuda" else nullcontext()
    with context:
        model=predictor(device).fit(observations())
        rng=np.random.default_rng(440)
        query=rng.uniform(-2.2,2.2,(101,2)).astype(np.float32)
        model.params.query_chunk=32
        a=model.predict(query)
        model.params.query_chunk=2048
        b=model.predict(query)
    np.testing.assert_allclose(a.mean,b.mean,atol=3e-5,rtol=1e-4)
    np.testing.assert_allclose(a.latent_var,b.latent_var,atol=3e-5,rtol=1e-3)
    np.testing.assert_array_equal(a.accepted,b.accepted)


@pytest.mark.parametrize("device",["cpu","cuda"])
def test_sparse_posterior_survives_equivalent_sum_operator(device):
    from unittest.mock import patch
    from linear_operator.operators import SumLinearOperator,DenseLinearOperator
    if device=="cuda" and not torch.cuda.is_available():pytest.skip("CUDA unavailable")
    torch.set_num_threads(1)
    params=FeatureSgpParams(device=device,max_train_points=80,inducing_points=24,
        training_steps=2,local_conditioning=False)
    xyz=observations();xyz[:,2]+=.2*np.sin(3*xyz[:,0])*np.cos(2*xyz[:,1])
    model=FeatureSgpPredictor(params).fit(xyz)
    query=np.array([[.2,-.1],[.1,.3]],np.float32)
    expected=model.predict(query)
    model.model.prediction_strategy=None
    kernel=model.model.covar_module
    covariance=kernel._get_covariance
    def equivalent_sum(a,b):
        value=covariance(a,b)
        if not torch.equal(a,b):
            return SumLinearOperator(value,DenseLinearOperator(torch.zeros_like(value.to_dense())))
        return value
    # Same numerical kernel, different operator representation, as reported
    # by the real worker. No variance or measurement changes are injected.
    with patch.object(kernel,"_get_covariance",side_effect=equivalent_sum):
        actual=model.predict(query)
    assert model.prediction_fallbacks==1 and model.generic_prediction
    np.testing.assert_allclose(actual.mean,expected.mean,atol=3e-5,rtol=1e-3)
    # Check exact Gaussian conditioning independently, not the optimized
    # library's approximate variance, which is a different numerical strategy.
    with torch.no_grad():
        encoded,nearest=model._encode_query(torch.as_tensor(query,device=model.device))
        train=model.model.train_inputs[0]
        matrix=kernel(train,train).to_dense()+torch.eye(len(train),device=model.device)*model.model.likelihood.noise
        cross=kernel(encoded,train).to_dense()
        prior=kernel(encoded,encoded,diag=True)
        design=torch.cat((torch.as_tensor(query,device=model.device)-model.xy_center,
                          torch.ones((len(query),1),device=model.device)),dim=1)
        extra=torch.einsum("ni,ij,nj->n",design,model.trend_covariance,design).clamp_min(0)
        extra+=(nearest*model.unresolved_slope*model.params.unresolved_scale).square()
        prior,cross,matrix=[v.cpu().numpy().astype(np.float64) for v in (prior,cross,matrix)]
        variance=prior-np.einsum("ij,ji->i",cross,np.linalg.solve(matrix,cross.T))
        reference=np.maximum(np.maximum(variance,1e-10)*float(model.target_scale.cpu())**2+
            extra.cpu().numpy(),model.params.noise_std_m**2)
    np.testing.assert_allclose(actual.latent_var,reference,atol=3e-5,rtol=1e-3)
