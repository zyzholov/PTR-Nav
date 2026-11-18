import itertools
import numpy as np
import pytest
import torch
from fsgp_bgk_se2.bounded_pending import condition

@pytest.mark.parametrize('sites',[[1],[1,3],[0,2,5]])
@pytest.mark.parametrize('device',['cpu','cuda'])
def test_every_posterior_vertex_has_exact_common_covariance_and_the_correct_mean(sites,device):
 if device=='cuda'and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
 rng=np.random.default_rng(12);mu=rng.standard_normal(7);f=rng.standard_normal((7,3))*.05
 d=rng.uniform(.0001,.0002,7);y=rng.standard_normal(len(sites));noise=np.full(len(sites),.0001);radius=np.full(len(sites),.02)
 before=[a.copy()for a in [mu,f,d]];F=torch.tensor(f,device=device,dtype=torch.float64)
 family=condition(mu,F,d,sites,y,noise,radius)
 prior=f@f.T+np.diag(d);cross=prior[:,sites];innovation=prior[np.ix_(sites,sites)]+np.diag(noise)
 gain=cross@np.linalg.inv(innovation);expected_cov=prior-gain@cross.T
 actual_cov=(family.factor@family.factor.T+torch.diag(family.local)).cpu().numpy()
 np.testing.assert_allclose(actual_cov,expected_cov,atol=1e-15,rtol=1e-11)
 assert np.linalg.eigvalsh(actual_cov).min()>0
 for signs in itertools.product([-1,1],repeat=len(sites)):
  b=radius*np.array(signs)
  expected=mu+gain@(y+b-mu[sites]);actual=(family.mean+family.bias@torch.tensor(signs,dtype=torch.float64,device=device)).cpu().numpy()
  np.testing.assert_allclose(actual,expected,atol=1e-13,rtol=1e-11)
 for a,b in zip([mu,f,d],before):np.testing.assert_array_equal(a,b)
 np.testing.assert_array_equal(F.cpu().numpy(),f)
 # Every linear geometric contrast attains the exact extrema predicted by
 # the L1 sensitivity, including negative weights and coupled sources.
 a=rng.standard_normal((5,7));center=a@family.mean.cpu().numpy();spread=np.abs(a@family.bias.cpu().numpy()).sum(axis=1)
 vertices=np.stack([a@(family.mean.cpu().numpy()+family.bias.cpu().numpy()@np.array(s))for s in itertools.product([-1,1],repeat=len(sites))])
 np.testing.assert_allclose(vertices.min(0),center-spread,atol=1e-12)
 np.testing.assert_allclose(vertices.max(0),center+spread,atol=1e-12)
 wider=condition(mu,F,d,sites,y,noise,2*radius)
 np.testing.assert_allclose(wider.factor.cpu(),family.factor.cpu(),atol=0,rtol=0)
 np.testing.assert_allclose(wider.mean.cpu(),family.mean.cpu(),atol=0,rtol=0)
 np.testing.assert_allclose(wider.bias.cpu(),2*family.bias.cpu(),atol=0,rtol=0)


def field(device='cpu',radius=.01):
 from fsgp_bgk_se2.shared_terrain import SharedTerrainPrediction
 rng=np.random.default_rng(24);shape=(13,13)
 f=torch.tensor(rng.normal(0,.006,(*shape,5)),dtype=torch.float32,device=device)
 bias=torch.tensor(rng.normal(0,radius,(*shape,2)),dtype=torch.float32,device=device)
 h=rng.normal(0,.002,shape).astype(np.float32);local=np.full(shape,1e-5,np.float32)
 return SharedTerrainPrediction(h,f.square().sum(-1).cpu().numpy()+local,np.ones(shape,bool),
  np.zeros(shape,bool),np.zeros(shape,np.float32),f,0.,20,0.,local_variance=local,
  bounded_mean_factor=bias)


def params():
 from fsgp_bgk_se2.probabilistic_risk import ProbabilityParams
 return ProbabilityParams(resolution=.2,yaw_bins=4,front=.4,back=.4,half_width=.25,margin=.1,
  max_pitch=.1745,max_roll=.1745,max_step=.1,max_roughness=.125,
  internal_steps=True,internal_step_model='height_jump',exact_footprint=True,
  independent_model_floor=True)


def dense_updates(prediction,**kw):
 from fsgp_bgk_se2.shared_terrain import shared_geometric_probability
 p=params();out=np.full((p.yaw_bins,*prediction.mean.shape),np.nan)
 for k,(r,c,u) in enumerate(shared_geometric_probability(prediction,p,np.zeros(prediction.mean.shape,bool),**kw)):
  out[k,r,c]=u
 return out


@pytest.mark.parametrize('factorized,direct',[(False,False),(True,False),(True,True)])
def test_robust_cuda_all_projection_paths_match_cpu(factorized,direct):
 if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
 from fsgp_bgk_se2.shared_terrain_cuda import shared_probability_cuda
 q=field('cuda');expected=dense_updates(q,fused=False)
 actual=np.full_like(expected,np.nan)
 for k,(r,c,u) in enumerate(shared_probability_cuda(q,params(),np.zeros(q.mean.shape,bool),
      factorized=factorized,direct_span=direct)):
  actual[k,r,c]=u
 np.testing.assert_allclose(actual,expected,atol=3e-6,rtol=5e-4,equal_nan=True)


def test_robust_pose_bound_encloses_every_global_family_vertex_and_is_monotone():
 from dataclasses import replace
 q=field();upper=dense_updates(q);bias=q.bounded_mean_factor
 for signs in itertools.product([-1,1],repeat=2):
  h=q.mean+(bias@torch.tensor(signs,dtype=torch.float32)).numpy()
  point=dense_updates(replace(q,mean=h,bounded_mean_factor=None))
  assert np.all((point<=upper+2e-6)|np.isnan(upper))
 wider=dense_updates(replace(q,bounded_mean_factor=2*bias))
 assert np.all((wider+1e-7>=upper)|np.isnan(upper))


def test_bounded_marginal_cpu_cuda_is_conservative_for_shared_field():
 if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
 import cupy as cp
 from fsgp_bgk_se2.probabilistic_risk import geometric_probability
 q=field('cuda');rho=q.bounded_mean_factor.abs().sum(-1).cpu().numpy()
 values=[]
 for xp in [np,cp]:
  layers=geometric_probability(q.mean,q.variance,q.valid,params(),xp=xp,bounded_radius=rho)
  values.append(np.stack(layers['failure_probability_upper']))
  assert np.all(np.stack(layers['height_error_tail_upper'])[np.isfinite(values[-1])]==1.)
 np.testing.assert_allclose(*values,atol=3e-6,equal_nan=True)
 full=dense_updates(q)
 assert np.all((full<=values[0]+2e-6)|np.isnan(full))


def pending_fixture():
 from dataclasses import replace
 q=field(radius=0.);shape=q.mean.shape
 q=replace(q,bounded_mean_factor=None,candidate=q.valid.copy(),
   admission_domain=q.valid.copy(),contradicted=np.zeros(shape,bool),discontinuity_veto=np.zeros(shape,bool))
 pending=np.zeros(shape,bool);pending[6,6]=True;q.valid[pending]=False
 old=q.mean.copy();new=old.copy();new[pending]+=.04;v=np.full(shape,.0001,np.float32)
 return q,dict(old=old,old_variance=v,pending_height=new,pending_variance=v,
  pending=pending,native_observed=np.ones(shape,bool),raw_veto=np.zeros(shape,bool),
  columns=np.zeros(shape,bool),max_step=.1,measured=~pending,measured_height=old,measured_variance=v)


@pytest.mark.parametrize('gate',['native_observed','raw_veto','columns','admission_domain',
 'contradicted','discontinuity_veto','large_innovation','budget','initial_plane','missing_variance'])
def test_pending_family_never_bypasses_unrelated_vetoes(gate):
 from dataclasses import replace
 from fsgp_bgk_se2.bounded_pending import condition_pending
 q,kw=pending_fixture()
 if gate=='native_observed':kw[gate][6,6]=False
 elif gate in ['raw_veto','columns']:kw[gate][6,6]=True
 elif gate in ['admission_domain','contradicted','discontinuity_veto']:
  getattr(q,gate)[6,6]=(gate!='admission_domain')
 elif gate=='large_innovation':kw['pending_height'][6,6]=kw['old'][6,6]+.101
 elif gate=='budget':kw['max_sites']=0
 elif gate=='initial_plane':q=replace(q,initial_plane_used=True)
 elif gate=='missing_variance':kw['pending_variance'][6,6]=np.nan
 assert condition_pending(q,**kw) is q


def test_pending_is_a_global_mean_family_not_a_measured_midpoint_and_is_not_reconditioned():
 from fsgp_bgk_se2.bounded_pending import condition_pending
 q,kw=pending_fixture();base=q.mean.copy();new=condition_pending(q,**kw)
 assert new is not q and new.bounded_pending[6,6] and new.bounded_mean_factor.shape[-1]==1
 assert not kw['measured'][6,6] and kw['pending'][6,6]
 np.testing.assert_array_equal(q.mean,base)
 assert condition_pending(new,**kw) is new
