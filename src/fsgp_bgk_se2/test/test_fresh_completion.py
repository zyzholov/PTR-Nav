import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gpytorch")
from fsgp_bgk_se2.feature_sgp import FeatureSgpParams, FeatureSgpPredictor
from fsgp_bgk_se2.fresh_completion import fresh_local_prediction
from fsgp_bgk_se2.supported_completion import supported_completion


def scene():
    x, y = np.meshgrid((np.arange(30)-14.5)*.2, (np.arange(30)-14.5)*.2)
    z = .03*x-.02*y
    observed = np.ones(z.shape, bool)
    observed[:, 14:16] = False
    elevation = np.where(observed, z, np.nan)
    points = np.column_stack((x[observed], y[observed], z[observed]))
    return x, y, z, observed, elevation, points


def predict(device="cpu", age=.1, points_override=None, neighbors=20, **kwargs):
    torch.set_num_threads(1)
    x, y, z, observed, elevation, points = scene()
    model = FeatureSgpPredictor(FeatureSgpParams(device=device, training_steps=0, local_neighbors=neighbors))
    result = fresh_local_prediction(model, elevation, np.full(z.shape,.0001), observed,
        x, y, points if points_override is None else points_override, age, .2,
        blocked=np.zeros(z.shape, bool), **kwargs)
    return result


def test_current_interpolation_does_not_use_hidden_truth_or_change_measurements():
    x, y, z, observed, elevation, points = scene()
    p = predict()
    assert p.accepted[2:-2, 14:16].all()
    assert not p.accepted[observed].any()
    np.testing.assert_allclose(p.mean[p.accepted], z[p.accepted], atol=2e-5)
    assert np.min(p.latent_var[p.accepted]) > 0
    assert np.max(p.latent_var[p.accepted]) < .0004
    np.testing.assert_allclose(p.observation_var[p.accepted]-p.latent_var[p.accepted],.0004)
    height, var, valid, used = supported_completion(elevation, np.full(z.shape,.0001), observed,p)
    np.testing.assert_array_equal(height[observed], elevation[observed])
    np.testing.assert_array_equal(var[observed], np.full(z.shape,.0001)[observed])
    assert used[2:-2,14:16].all() and valid[observed].all()
    assert p.training_points == np.count_nonzero(observed)


@pytest.mark.parametrize("age", [-.01,.251,1.,np.inf])
def test_stale_or_future_cloud_never_relabels_history_as_current(age):
    assert not predict(age=age).accepted.any()


def test_hidden_raw_step_and_pit_are_veto_only_not_training():
    x,y,z,observed,elevation,points = scene()
    extra=np.array([[x[10,14],y[10,14],z[10,14]+.25],
                    [x[10,14]+.01,y[10,14]+.01,z[10,14]+.26],
                    [x[20,15],y[20,15],z[20,15]-.25],
                    [x[20,15]+.01,y[20,15]+.01,z[20,15]-.26]])
    a,b=predict(),predict(points_override=np.r_[points,extra])
    np.testing.assert_array_equal(a.mean,b.mean)
    assert a.accepted[10,14] and a.accepted[20,15]
    assert not b.accepted[10,14] and not b.accepted[20,15]
    assert a.training_points == b.training_points


def test_unseen_space_outside_measured_hull_is_not_filled():
    model=FeatureSgpPredictor(FeatureSgpParams(device="cpu",training_steps=0))
    x,y,z,observed,_,points=scene()
    points=points[points[:,0]>.15]
    result=model.condition_local(points,np.array([[0.,0.],[.5,0.]]))
    assert not result.accepted[0] and np.isnan(result.mean[0])
    assert result.accepted[1]


@pytest.mark.parametrize("device",["cpu","cuda"])
def test_dense_one_sided_neighbors_do_not_hide_measured_opposite_rim(device):
    from scipy.spatial import cKDTree
    from fsgp_bgk_se2.feature_sgp import surrounding_neighbors
    if device=="cuda" and not torch.cuda.is_available():pytest.skip("CUDA unavailable")
    x,y=np.meshgrid(np.r_[[-1.,-.9,-.8],np.arange(.2,1.01,.1)],np.arange(-.6,.61,.1))
    points=np.c_[x.ravel(),y.ravel(),(.03*x-.02*y).ravel()]
    query=np.array([[0.,0.]])
    tree=cKDTree(points[:,:2]);_,nearest=tree.query(query,k=20)
    assert (points[nearest[0],0]>0).all()
    distance,ids=surrounding_neighbors(tree,query,20,1.5,.35)
    assert (points[ids[0],0]<0).any()
    assert len(np.unique(ids[0]))==20 and np.max(distance)<=1.5
    model=FeatureSgpPredictor(FeatureSgpParams(device=device,training_steps=0))
    p=model.condition_local(points,query)
    assert p.accepted[0] and p.mean[0]==pytest.approx(0.,abs=2e-5)
    assert 0<p.latent_var[0]<.01
    # Removing the opposite observations must still reject extrapolation.
    assert not model.condition_local(points[points[:,0]>0],query).accepted[0]


@pytest.mark.parametrize("device",["cpu","cuda"])
def test_support_distance_uses_current_returns_not_snapped_cell_centers(device):
    if device=="cuda" and not torch.cuda.is_available():pytest.skip("CUDA unavailable")
    x,y,truth,observed,_,_=scene()
    observed[:,13:17]=False
    z=np.where(observed,truth,np.nan)
    px=x.copy();px[:,12]+=.09;px[:,17]-=.09
    points=np.c_[px[observed],y[observed],(.03*px-.02*y)[observed]]
    model=FeatureSgpPredictor(FeatureSgpParams(device=device,training_steps=0))
    def run(points):
        return fresh_local_prediction(model,z,np.full(z.shape,.0001),observed,x,y,points,0.,.2,
            blocked=np.zeros(z.shape,bool))
    p=run(points)
    # The nearest fresh cell center is 0.40m away, but the real return is 0.31m.
    assert p.accepted[15,15]
    assert p.nearest_distance[15,15]==pytest.approx(.31,abs=1e-6)
    assert p.mean[15,15]==pytest.approx(truth[15,15],abs=2e-5)
    assert 0<p.latent_var[15,15]<.0016
    # No new observation is created, and actual distances beyond 0.35m fail.
    assert not p.accepted[observed].any()
    far=np.c_[x[observed],y[observed],truth[observed]]
    assert not run(far).accepted[15,15]
    # Raw distance alone is not an enclosure or a permission to extrapolate.
    assert not run(points[points[:,0]>0]).accepted[15,15]


@pytest.mark.parametrize('neighbors', [20, 64])
def test_current_local_cuda_cpu_agreement(neighbors):
    if not torch.cuda.is_available():pytest.skip("CUDA unavailable")
    cpu,gpu=predict(neighbors=neighbors),predict("cuda",neighbors=neighbors)
    np.testing.assert_array_equal(cpu.accepted,gpu.accepted)
    np.testing.assert_allclose(cpu.mean,gpu.mean,atol=2e-5,equal_nan=True)
    np.testing.assert_allclose(cpu.latent_var,gpu.latent_var,atol=2e-6)


@pytest.mark.parametrize('neighbors', [20, 64])
def test_latent_variance_matches_universal_kriging_equation(neighbors):
    from scipy.spatial import cKDTree
    x,y,z,observed,elevation,points=scene()
    model=FeatureSgpPredictor(FeatureSgpParams(device="cpu",training_steps=0,local_neighbors=neighbors))
    query=np.array([[0.,0.]])
    result=model.condition_local(points,query)
    # Match the retained float32 measurements. Symmetric distance ties can
    # select a different neighborhood after input quantization/reordering.
    retained=model.local_xyz.cpu().numpy()
    distance,ids=cKDTree(retained[:,:2]).query(query[0],k=neighbors)
    uv=(retained[ids,:2].astype(np.float64)-query)/max(distance.max(),.1)
    xx,yy=uv.T
    design=np.c_[np.ones(neighbors),xx,yy,xx*xx,xx*yy,yy*yy]
    noise=.02**2
    kernel=noise*np.exp(-np.sum((uv[:,None]-uv[None])**2,axis=2)/(2*.7**2))
    cross=noise*np.exp(-np.sum(uv**2,axis=1)/(2*.7**2))
    inverse=np.linalg.inv(kernel+noise*np.eye(neighbors))
    contrast=np.array([1,0,0,0,0,0])-design.T@inverse@cross
    expected=noise-cross@inverse@cross+contrast@np.linalg.inv(design.T@inverse@design)@contrast
    assert result.latent_var[0]==pytest.approx(expected,rel=1e-4,abs=1e-8)
    assert result.observation_var[0]==pytest.approx(expected+noise,rel=1e-4,abs=1e-8)
    ids=result.conditioning_indices[0]
    uv=(retained[ids,:2].astype(np.float64)-query)/max(np.linalg.norm(retained[ids,:2]-query,axis=1).max(),.1)
    xx,yy=uv.T;design=np.c_[np.ones(neighbors),xx,yy,xx*xx,xx*yy,yy*yy]
    kernel=noise*np.exp(-np.sum((uv[:,None]-uv[None])**2,axis=2)/(2*.7**2))
    cross=noise*np.exp(-np.sum(uv**2,axis=1)/(2*.7**2))
    inverse=np.linalg.inv(kernel+noise*np.eye(neighbors))
    contrast=np.array([1,0,0,0,0,0])-design.T@inverse@cross
    expected_weights=inverse@cross+inverse@design@np.linalg.inv(design.T@inverse@design)@contrast
    np.testing.assert_allclose(result.conditioning_weights[0],expected_weights,atol=1e-5)
    assert result.conditioning_weights[0].sum()==pytest.approx(1.,abs=1e-5)
    assert result.mean[0]==pytest.approx(result.conditioning_weights[0]@retained[ids,2],abs=1e-5)


@pytest.mark.parametrize('neighbors', [20, 64])
def test_step_discontinuity_increases_latent_uncertainty_not_navigation_permission(neighbors):
    x,y,z,observed,elevation,points=scene()
    points[:,2]=np.where(points[:,0]>0,.5,0)
    model=FeatureSgpPredictor(FeatureSgpParams(device="cpu",training_steps=0,local_neighbors=neighbors))
    p=model.condition_local(points,np.array([[0.,0.]]))
    smooth=model.condition_local(scene()[-1],np.array([[0.,0.]]))
    assert not p.accepted[0] or p.latent_var[0]>5*smooth.latent_var[0]
