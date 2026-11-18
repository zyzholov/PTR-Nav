"""Same measured posterior: compute-budget frontiers are not obstacles."""
from pathlib import Path
import json
import numpy as np
import pytest
import torch
from fsgp_bgk_se2.shared_terrain import SharedTerrainPrediction,shared_geometric_probability
from fsgp_bgk_se2.probabilistic_risk import ProbabilityParams
from fsgp_bgk_se2.refinement_region import refinement_query_mask


def case(device='cpu'):
    path=Path(__file__).parent/'data/refinement_navigation_frontier.npz'
    data=np.load(path);meta=json.loads(path.with_suffix('.json').read_text())
    p=SharedTerrainPrediction(data['mean'],data['variance'],data['valid'],data['completed'],data['nearest'],
        torch.tensor(data['factor'],device=device),meta['discrepancy'],0,0.,
        discrepancy_lengthscale_m=meta['discrepancy_lengthscale_m'],plane_components=meta['plane_components'],
        local_variance=data['local_variance']if 'local_variance'in data else None,
        bounded_mean_factor=torch.tensor(data['bounded_mean_factor'],device=device)if 'bounded_mean_factor'in data else None)
    mask=np.zeros_like(data['old_upper'],bool);mask[tuple(meta['target'])]=True
    return data,meta,p,ProbabilityParams(**meta['physics']),mask


def tightened(p,physical,occupied,mask,old):
    result=old.copy()
    for k,(r,c,upper)in enumerate(shared_geometric_probability(p,physical,occupied,query_mask=mask)):
        result[k,r,c]=np.minimum(result[k,r,c],upper)
    return result


def test_full_covariance_proof_recovers_a_recorded_measured_pose_without_changing_the_model():
    data,meta,p,physical,mask=case();target=tuple(meta['target'])
    before=p.factor.clone();height=p.mean.copy();variance=p.variance.copy();support=p.valid.copy()
    restricted=refinement_query_mask(mask,data['x'],data['y'],radius_m=2.5,
        center=data['center'],pose_stamp=0,map_stamp=0)
    assert not restricted[target] and data['old_upper'][target]>.05
    assert tightened(p,physical,data['collision'],restricted,data['old_upper'])[target]>.05
    full=refinement_query_mask(mask,data['x'],data['y'],radius_m=0.)
    result=tightened(p,physical,data['collision'],full,data['old_upper'])
    assert result[target]<.0005 # Same 5% admission limit, no covariance/height substitution.
    torch.testing.assert_close(p.factor,before,rtol=0,atol=0)
    for original,current in ((height,p.mean),(variance,p.variance),(support,p.valid)):
        np.testing.assert_array_equal(original,current)
    assert physical.independent_model_floor and physical.height_std_floor==pytest.approx(.005,abs=1e-9)


def test_more_proof_computation_cannot_fill_unknown_support_or_clear_current_occupancy():
    data,meta,p,physical,mask=case();target=tuple(meta['target']);_,r,c=target
    p.valid[:]=False
    result=tightened(p,physical,data['collision'],mask,data['old_upper'])
    assert result[target]==data['old_upper'][target]
    data,meta,p,physical,mask=case();occupied=data['collision'].copy();occupied[r,c]=True
    result=tightened(p,physical,occupied,mask,data['old_upper'])
    assert result[target]==data['old_upper'][target] and occupied[r,c]


def test_recorded_refinement_uses_the_same_covariance_and_decision_on_cpu_and_cuda():
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    pytest.importorskip('cupy')
    fields=[]
    for device in ('cpu','cuda'):
        data,meta,p,physical,mask=case(device)
        fields.append(tightened(p,physical,data['collision'],mask,data['old_upper']))
    np.testing.assert_allclose(fields[0][mask],fields[1][mask],rtol=1e-4,atol=1e-6)
    np.testing.assert_array_equal(fields[0][mask]<=.05,fields[1][mask]<=.05)
