import numpy as np
import pytest
import torch

from fsgp_bgk_se2.ground_evidence import GroundEvidenceHistory,temporal_std_bound
from fsgp_bgk_se2.feature_sgp import FeatureSgpParams,FeatureSgpPredictor
from fsgp_bgk_se2.fresh_completion import fresh_local_prediction


@pytest.mark.parametrize('age',[-1,0,2.01,np.inf,np.nan])
def test_history_has_explicit_bounded_age(age):
    with pytest.raises(ValueError):GroundEvidenceHistory(age)


def test_raw_coordinates_ages_expiry_and_reset_are_not_fabricated():
    cache=GroundEvidenceHistory(2.)
    points=np.array([[.113,.217,.3],[.31,.42,.4]],np.float32)
    p,a=cache.update(points,1.,'map',(0,0),source_age_s=.1)
    np.testing.assert_array_equal(p,points)
    np.testing.assert_allclose(a,.1)
    p,a=cache.update(points[:0],1.5,'map',(0,0))
    np.testing.assert_array_equal(p,points);np.testing.assert_allclose(a,.6)
    p,a=cache.update(points[:0],3.01,'map',(0,0));assert len(p)==0
    cache.update(points,4.,'map',(0,0))
    assert not len(cache.update(points[:0],4.1,'other',(0,0))[0])
    cache.update(points,5.,'map',(0,0))
    assert not len(cache.update(points[:0],4.,'map',(0,0))[0])
    cache.update(points,5.,'map',(0,0))
    assert not len(cache.update(points[:0],5.1,'map',(2,0))[0])


def fixture(device):
    torch.set_num_threads(1)
    x,y=np.meshgrid((np.arange(30)-14.5)*.2,(np.arange(30)-14.5)*.2)
    z=.03*x-.02*y;observed=np.ones(z.shape,bool);observed[:,14:16]=False
    h=np.where(observed,z,np.nan);variance=np.full(z.shape,.0001)
    points=np.c_[x[observed],y[observed],z[observed]]
    model=FeatureSgpPredictor(FeatureSgpParams(device=device,training_steps=0,local_neighbors=64))
    cache=GroundEvidenceHistory(2.)
    def run(p,stamp,blocked=None):
        return fresh_local_prediction(model,h,variance,observed,x,y,p,0.,.2,
            blocked=np.zeros(z.shape,bool) if blocked is None else blocked,
            history=cache,stamp=stamp,frame='map',pose_xy=(0,0),process_std_mps=.01)
    return x,y,z,observed,points,cache,run


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_history_is_measured_only_ages_uncertainty_and_expires(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    x,y,z,observed,points,cache,run=fixture(device)
    first=run(points,1.)
    far=points[np.hypot(points[:,0],points[:,1])>1.2]
    second=run(far,1.3)
    i,j=15,15
    assert first.accepted[i,j] and second.accepted[i,j]
    assert second.conditioning_age[i,j]==pytest.approx(.3)
    assert np.sqrt(second.latent_var[i,j])>=np.sqrt(first.latent_var[i,j])+.003-1e-6
    assert second.mean[i,j]==pytest.approx(z[i,j],abs=2e-5)
    assert not second.accepted[observed].any()
    assert not run(far,3.1).accepted[i,j]
    assert all(observed[int(round((p[1]-y[0,0])/.2)),int(round((p[0]-x[0,0])/.2))]
               for p in cache.points)


def test_current_contradiction_vetoes_prediction_and_occupied_evidence_is_removed():
    x,y,z,observed,points,cache,run=fixture('cpu')
    run(points,1.)
    far=points[np.hypot(points[:,0],points[:,1])>1.2]
    extra=np.array([[x[15,15],y[15,15],z[15,15]-.25],
                    [x[15,15]+.01,y[15,15]+.01,z[15,15]-.26]])
    assert not run(np.r_[far,extra],1.2).accepted[15,15]
    blocked=np.ones(z.shape,bool)
    assert not run(far,1.3,blocked).accepted.any()
    assert len(cache.points)==0
    assert not run(far,1.4).accepted[15,15]


def test_temporal_conditioning_cpu_cuda_agreement():
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    results=[]
    for device in ('cpu','cuda'):
        _,_,_,_,points,_,run=fixture(device)
        run(points,1.)
        results.append(run(points[np.hypot(points[:,0],points[:,1])>1.2],1.3))
    a,b=results
    np.testing.assert_array_equal(a.accepted,b.accepted)
    np.testing.assert_allclose(a.mean,b.mean,atol=2e-5,equal_nan=True)
    np.testing.assert_allclose(a.latent_var,b.latent_var,atol=2e-6)
    np.testing.assert_array_equal(a.conditioning_age,b.conditioning_age)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_validated_raw_ground_survives_occlusion_without_inventing_a_measurement(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    x,y,z,observed,points,cache,run=fixture(device)
    run(points,1.)
    occluded=np.hypot(x,y)<.9
    now_observed=observed&~occluded
    raw=np.c_[x[now_observed],y[now_observed],z[now_observed]]
    predictor=FeatureSgpPredictor(FeatureSgpParams(device=device,training_steps=0,local_neighbors=64))
    h=np.where(now_observed,z,np.nan);v=np.where(now_observed,.0001,np.nan)
    def predict(stamp,extra=None):
        return fresh_local_prediction(predictor,h,v,now_observed,x,y,
            raw if extra is None else np.r_[raw,extra],0.,.2,
            blocked=np.zeros(z.shape,bool),history=cache,stamp=stamp,frame='map',pose_xy=(0,0))
    result=predict(1.3)
    assert result.accepted[15,14]
    assert result.conditioning_age[15,14]==pytest.approx(.3)
    assert result.mean[15,14]==pytest.approx(z[15,14],abs=2e-5)
    assert np.isnan(h[15,14]) and not now_observed[15,14]
    assert not result.accepted[now_observed].any()
    assert not predict(3.1).accepted[15,14]


def test_new_raw_pit_in_a_historical_ground_cell_revokes_that_training_source():
    x,y,z,observed,points,cache,run=fixture('cpu')
    run(points,1.)
    now=observed&(np.hypot(x,y)>.9)
    raw=np.c_[x[now],y[now],z[now]]
    predictor=FeatureSgpPredictor(FeatureSgpParams(device='cpu',training_steps=0,local_neighbors=64))
    def predict(stamp,extra):
        return fresh_local_prediction(predictor,np.where(now,z,np.nan),np.where(now,.0001,np.nan),
            now,x,y,np.r_[raw,extra],0.,.2,blocked=np.zeros(z.shape,bool),history=cache,
            stamp=stamp,frame='map',pose_xy=(0,0))
    i,j=15,13
    pit=np.array([[x[i,j],y[i,j],z[i,j]-.3],[x[i,j]+.01,y[i,j]+.01,z[i,j]-.31]])
    assert not predict(1.2,pit).accepted[i,j]
    assert not np.any(np.linalg.norm(cache.points[:,:2]-[x[i,j],y[i,j]],axis=1)<.02)
    assert not predict(1.3,np.empty((0,3))).accepted[i,j]


def test_raw_returns_from_never_observed_cells_cannot_enter_the_history():
    x,y,z,observed,points,cache,run=fixture('cpu')
    hidden=np.array([[x[15,15],y[15,15],z[15,15]],
                     [x[15,15]+.01,y[15,15]+.01,z[15,15]]])
    run(np.r_[points,hidden],1.)
    assert not np.any(np.linalg.norm(cache.points[:,:2]-[x[15,15],y[15,15]],axis=1)<.02)


def test_history_uses_the_exact_gp_conditioning_sources_without_a_second_search():
    from unittest.mock import patch
    from fsgp_bgk_se2.feature_sgp import surrounding_neighbors
    _,_,_,_,points,_,run=fixture('cpu')
    with patch('fsgp_bgk_se2.feature_sgp.surrounding_neighbors',wraps=surrounding_neighbors) as search:
        prediction=run(points,1.)
    assert prediction.accepted.any()
    assert search.call_count==1


def test_temporal_error_bound_accounts_for_negative_weights_and_actual_source_influence():
    weights=np.array([[.99,.01],[-.5,1.5]])
    ages=np.array([[0.,2.],[2.,2.]])
    np.testing.assert_allclose(temporal_std_bound(weights,ages,.01),[.0002,.04])
    # Adversarial fully correlated source errors attain the triangle bound.
    source_std=.01*ages
    covariance=np.sign(weights)[:,:,None]*np.sign(weights)[:,None,:]*source_std[:,:,None]*source_std[:,None,:]
    actual=np.sqrt(np.einsum('bi,bij,bj->b',weights,covariance,weights))
    np.testing.assert_allclose(actual,temporal_std_bound(weights,ages,.01))


@pytest.mark.parametrize('weights,ages',[( [[1.,-1.]], [[1.]] ),
    ( [[1.,np.nan]], [[0.,0.]] ),( [[1.,0.]], [[0.,-1.]] )])
def test_invalid_temporal_sources_are_rejected(weights,ages):
    with pytest.raises(ValueError):temporal_std_bound(weights,ages,.01)


def test_negative_ground_evidence_persists_when_gp_returns_before_prediction():
    x,y,z,observed,points,cache,run=fixture('cpu')
    run(points,1.)
    i,j=15,13
    now=observed&(np.hypot(x,y)>.35)
    raw=np.c_[x[now],y[now],z[now]]
    pit=np.array([[x[i,j],y[i,j],z[i,j]-.3],[x[i,j]+.01,y[i,j]+.01,z[i,j]-.31]])
    model=FeatureSgpPredictor(FeatureSgpParams(device='cpu',training_steps=0,local_neighbors=64))
    def predict(history,stamp,blocked,extra):
        return fresh_local_prediction(model,np.where(now,z,np.nan),np.where(now,.0001,np.nan),
            now,x,y,np.r_[raw,extra],0.,.2,blocked=blocked,history=history,
            stamp=stamp,frame='map',pose_xy=(0,0))
    first=predict(cache,1.2,np.ones(z.shape,bool),pit)
    assert not first.accepted.any() and cache.vetoes
    # The new measurement neighborhood could fill the hole absent a veto.
    assert predict(GroundEvidenceHistory(2.),1.3,np.zeros(z.shape,bool),np.empty((0,3))).accepted[i,j]
    assert not predict(cache,1.3,np.zeros(z.shape,bool),np.empty((0,3))).accepted[i,j]


def test_current_negative_evidence_survives_occlusion_and_needs_three_measured_clears():
    cache=GroundEvidenceHistory(2.)
    x,y=np.meshgrid(np.array([-.1,.1]),np.array([-.1,.1]))
    veto=np.zeros((2,2),bool);veto[1,1]=True
    empty=np.zeros_like(veto);clear=np.ones_like(veto)
    assert cache.veto_mask(x,y,veto,empty,1.)[1,1]
    assert cache.veto_mask(x,y,empty,empty,1.1)[1,1]
    assert cache.veto_mask(x,y,empty,clear,1.2)[1,1]
    assert cache.veto_mask(x,y,empty,clear,1.2)[1,1]
    assert cache.veto_mask(x,y,empty,clear,1.3)[1,1]
    assert not cache.veto_mask(x,y,empty,clear,1.4)[1,1]
    assert cache.veto_mask(x,y,veto,clear,1.5)[1,1]


def test_strong_current_floor_can_retire_old_contradiction_but_never_new_pit_or_duplicate_stamp():
    x,y=np.meshgrid([-.1,.1],[-.1,.1]);empty=np.zeros((2,2),bool)
    witness=~empty;veto=empty.copy();veto[1,1]=True
    cache=GroundEvidenceHistory(1.)
    assert cache.veto_mask(x,y,veto,empty,1.)[1,1]
    assert cache.veto_mask(x,y,empty,witness,1.,strong_ground_clear=witness)[1,1]
    assert cache.veto_mask(x,y,empty,empty,1.1,strong_ground_clear=witness)[1,1]
    assert cache.veto_mask(x,y,veto,witness,1.2,strong_ground_clear=witness)[1,1]
    assert not cache.veto_mask(x,y,empty,witness,1.3,strong_ground_clear=witness)[1,1]
    assert cache.veto_mask(x,y,veto,witness,1.4,strong_ground_clear=witness)[1,1]
    with pytest.raises(ValueError):cache.veto_mask(x,y,empty,witness,1.5,strong_ground_clear=np.ones((1,1)))


def dictionary_veto_reference(cache,x,y,new_veto,current_ground_clear,source_stamp):
    if not cache.vetoes and not new_veto.any():
        cache.clear_stamp=source_stamp
        return np.zeros(x.shape,bool)
    resolution=float(x[0,1]-x[0,0])
    xx=np.floor(x/resolution).astype(int);yy=np.floor(y/resolution).astype(int)
    lookup={(int(a),int(b)):(i,j) for i,j,a,b in
            zip(*np.indices(x.shape).reshape(2,-1),xx.ravel(),yy.ravel())}
    fresh=source_stamp!=cache.clear_stamp
    for key in list(cache.vetoes):
        if key not in lookup:del cache.vetoes[key];continue
        i,j=lookup[key]
        if fresh:cache.vetoes[key]=cache.vetoes[key]+1 if current_ground_clear[i,j] else 0
        if cache.vetoes[key]>=3:del cache.vetoes[key]
    for i,j in zip(*np.where(new_veto)):cache.vetoes[(int(xx[i,j]),int(yy[i,j]))]=0
    cache.clear_stamp=source_stamp
    mask=np.zeros(x.shape,bool)
    for key in cache.vetoes:mask[lookup[key]]=True
    return mask


@pytest.mark.parametrize('irregular',[False,True])
def test_fast_veto_index_is_identical_under_rolling_grid_repeated_scans_and_occlusion(irregular):
    actual=GroundEvidenceHistory(1.);reference=GroundEvidenceHistory(1.)
    rng=np.random.default_rng(1204)
    for i in range(36):
        x,y=np.meshgrid((np.arange(104)-51.5)*.2+(i//6)*.2,
                        (np.arange(76)-37.5)*.2-(i//9)*.4)
        if irregular:x[::2]+=.2
        veto=rng.random(x.shape)<.025;clear=rng.random(x.shape)<.7
        if i%4==0:clear[:]=False
        stamp=(i//2)*.1
        expected=dictionary_veto_reference(reference,x,y,veto,clear,stamp)
        result=actual.veto_mask(x,y,veto,clear,stamp)
        np.testing.assert_array_equal(result,expected)
        assert actual.vetoes==reference.vetoes
        assert actual.clear_stamp==reference.clear_stamp
