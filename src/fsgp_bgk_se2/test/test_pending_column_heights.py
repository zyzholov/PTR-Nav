"""Pending side heights must not erase adjacent ground; pits remain unknown."""
import numpy as np
import pytest
from fsgp_bgk_se2.obstacle_columns import separate_pending_column_heights


def fixture(slope=0.,datum=0.):
    y,x=np.mgrid[-7:8,-7:8]*.2
    z=datum+slope*x;v=np.full_like(z,.01**2);obs=np.ones(z.shape,bool)
    col=np.zeros(z.shape,bool);col[7,7]=True
    pending=col.copy();pz=z.copy();pz[7,7]+=.43;pv=v.copy()
    obs[7,7]=False
    return [pending,pz,pv,col,z,v,obs]


def classify(args):
    return separate_pending_column_heights(*args,.2,max_step_m=.1,max_ground_slope_rad=np.deg2rad(10.))


@pytest.mark.parametrize('slope,datum',[(0.,0.),(np.tan(np.deg2rad(9.)),-.44),(0.,10000.)])
def test_independent_enclosed_floor_can_separate_current_positive_side(slope,datum):
    args=fixture(slope,datum);original=[a.copy() for a in args]
    pending,side=classify(args)
    assert side[7,7] and not pending[7,7]
    assert not args[6][7,7] and args[3][7,7] # no observation or occupancy manufactured
    for a,b in zip(args,original):np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('height',[-.4,-.1,0.,.08,.12,.15,float('nan')])
def test_negative_ambiguous_and_substep_pending_heights_keep_their_veto(height):
    args=fixture();args[1][7,7]=height
    pending,side=classify(args);assert pending[7,7] and not side.any()


@pytest.mark.parametrize('problem',['no_current_column','one_sided','line','noisy','missing_variance','rough','steep'])
def test_missing_independent_reference_or_bad_pending_measurement_is_not_bypassed(problem):
    a=fixture()
    if problem=='no_current_column':a[3][:]=False
    if problem=='one_sided':a[6][:,:8]=False
    if problem=='line':a[6][:]=False;a[6][7,:]=True
    if problem=='noisy':a[2][7,7]=.20**2 # lower bound no longer establishes a positive side
    if problem=='missing_variance':a[2][7,7]=np.nan
    if problem=='rough':a[4]+=np.where(np.indices(a[4].shape).sum(axis=0)%2,.08,-.08)
    if problem=='steep':a[4]+=np.indices(a[4].shape)[1]*.2*.5
    pending,side=classify(a);assert pending[7,7] and not side.any()


def test_candidate_cannot_supply_its_own_reference_and_true_ground_pending_is_retained():
    a=fixture();a[6][:]=False;a[6][7,7]=True;a[4][7,7]=0.
    pending,side=classify(a);assert pending[7,7] and not side.any()
    a=fixture();a[0][8,8]=True;a[1][8,8]=-.3
    pending,side=classify(a);assert pending[8,8] and not side[8,8] and side[7,7]


@pytest.mark.parametrize('std',[.0199,.0201,.03,.08])
def test_side_classification_propagates_noise_instead_of_using_floor_training_cutoff(std):
    a=fixture();a[2][7,7]=std**2
    pending,side=classify(a);assert side[7,7] and not pending[7,7]
    # Same measurement variance with inadequate height separation stays unknown.
    a[1][7,7]=.1+2.9*np.sqrt(std**2+.015**2)
    pending,side=classify(a);assert pending[7,7] and not side.any()


@pytest.mark.parametrize('name',['mid360_dense_pending_side','mid360_dense_pending_side_noisy'])
def test_actual_mid360_dense_frame_separates_only_the_documented_positive_side(name):
    from pathlib import Path
    d=np.load(Path(__file__).parent/'data'/f'{name}.npz')
    a=[d[n] for n in ('pending','pending_z','pending_var','columns','z','var','observed')]
    pending,side=classify(a)
    assert a[0][5,5] and side[5,5] and not pending[5,5]
    assert a[3][5,5] and not a[6][5,5]
    assert a[1][5,5]-(-.44048)>.3


def test_real_floor_in_an_occupied_mixed_square_is_a_source_not_a_second_exclusion():
    a=fixture();a[3][5:10,5:10]=True
    exclusions=np.zeros_like(a[3]);exclusions[7,7]=True
    assert classify(a)[0][7,7] # old occupancy-as-source-exclusion has no independent reference
    original=[v.copy() for v in a]
    pending,side=separate_pending_column_heights(*a,.2,max_step_m=.1,
        max_ground_slope_rad=np.deg2rad(10.),ground_exclusions=exclusions)
    assert side[7,7] and not pending[7,7]
    for x,y in zip(a,original):np.testing.assert_array_equal(x,y)
    assert a[3][5:10,5:10].all() # every occupied volume remains hard


def test_one_isolated_high_reference_does_not_erase_a_majority_real_floor_plane():
    a=fixture();a[4][6,6]=.5
    pending,side=classify(a);assert side[7,7] and not pending[7,7]
    # A bimodal rough surface cannot be certified by selecting half its samples.
    a=fixture();a[4]+=np.where(np.indices(a[4].shape).sum(axis=0)%2,.08,-.08)
    pending,side=classify(a);assert pending[7,7] and not side.any()
