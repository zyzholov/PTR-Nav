"""A longer control horizon needs measured conditioning, not free unknown cells."""
from dataclasses import replace
import numpy as np
import pytest
import torch
from fsgp_bgk_se2.shared_terrain import SharedTerrainParams, shared_terrain_prediction


@pytest.mark.parametrize('radius', [float('nan'), float('inf'), .99, 5.01])
def test_posterior_radius_is_bounded(radius):
    with pytest.raises(ValueError):
        SharedTerrainParams(radius_m=radius)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('spacing', [1., 1.5])
def test_larger_tile_conditions_real_far_ground_but_cannot_create_unsupported_ground(device, spacing):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.set_num_threads(1)
    x, y = np.meshgrid((np.arange(60)-29.5)*.2, (np.arange(40)-19.5)*.2)
    z = .015*x-.01*y
    # Far right is genuinely absent, not withheld data secretly used in fit.
    seen = x < 4.3
    raw = np.c_[x[seen], y[seen], z[seen]]
    blocked = np.zeros_like(seen)
    blocked[(x > 3.6) & (x < 4.0) & (abs(y) < .3)] = True
    params = SharedTerrainParams(radius_m=3., basis_spacing_m=spacing, bounded_queries=True,
        condition_local_residual=True, condition_full_residual=True,
        residual_inducing_points=96, max_distance_m=.35,
        max_extrapolation_m=0., observation_halo_m=.6, complete_footprint_halo=True)
    args = (np.where(seen, z, np.nan), np.full(z.shape, .0001), seen,
            x, y, raw, 0., .2, blocked)
    before = [a.copy() for a in args if isinstance(a, np.ndarray)]
    small = shared_terrain_prediction(*args, params=params, device=device)
    large = shared_terrain_prediction(*args, params=replace(params, radius_m=5.), device=device)
    far_measured = (x > 3.8) & (x < 4.2) & (abs(y) > .5) & (abs(y) < 2.)
    assert not small.valid[far_measured].any()
    assert large.valid[far_measured].all()
    assert not large.valid[x > 4.8].any()
    assert not large.valid[blocked].any()
    assert large.training_points > small.training_points
    assert np.all(large.variance[large.valid] > 0)
    np.testing.assert_allclose(large.mean[far_measured], z[far_measured], atol=2e-5)
    for a, b in zip(before, (a for a in args if isinstance(a, np.ndarray))):
        np.testing.assert_array_equal(a, b)
