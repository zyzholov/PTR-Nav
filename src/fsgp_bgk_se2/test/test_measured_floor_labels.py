"""Independent measured ground and positive occupied volume are different facts."""
import numpy as np
import pytest
from fsgp_bgk_se2.obstacle_columns import separate_measured_floor_labels


def fixture(slope_x=0.,slope_y=0.,datum=0.):
    y,x=np.mgrid[-7:8,-7:8]*.2
    z=datum+slope_x*x+slope_y*y
    var=np.full_like(z,.01**2);observed=np.ones_like(z,dtype=bool)
    columns=np.zeros_like(observed);columns[6:9,6:9]=True
    return columns,z,var,observed


@pytest.mark.parametrize('slopes',[(0.,0.),(.4,0.),(0.,-.35),(.35,-.25)])
def test_mixed_column_cells_keep_existing_flat_or_sloped_floor_without_changing_collision(slopes):
    args=fixture(*slopes);copies=[a.copy() for a in args]
    excluded,preserved=separate_measured_floor_labels(*args,.2)
    assert preserved[6:9,6:9].all()
    assert not excluded.any()
    for a,b in zip(args,copies):np.testing.assert_array_equal(a,b)
    # The original positive occupied volume remains true even with floor below it.
    assert args[0][7,7]


@pytest.mark.parametrize('height',[.14,.30,1.0,-.30])
def test_wall_sides_tops_steps_and_pits_are_not_relabeled_as_neighbouring_floor(height):
    col,z,var,obs=fixture();z[6:9,6:9]=height
    excluded,preserved=separate_measured_floor_labels(col,z,var,obs,.2)
    assert excluded[6:9,6:9].all()
    assert not preserved.any()
    np.testing.assert_array_equal(z[6:9,6:9],height) # never flatten the measurement


def test_unknown_noisy_or_one_dimensional_neighbours_cannot_supply_a_floor_witness():
    col,z,var,obs=fixture();obs[:]=False;obs[7,:]=True
    excluded,preserved=separate_measured_floor_labels(col,z,var,obs,.2)
    assert not preserved.any();np.testing.assert_array_equal(excluded,col)
    obs[:]=True;var[6:9,6:9]=.1**2
    excluded,preserved=separate_measured_floor_labels(col,z,var,obs,.2)
    assert not preserved.any();np.testing.assert_array_equal(excluded,col)
    var[:]=.01**2;obs[6:9,6:9]=False
    _,preserved=separate_measured_floor_labels(col,z,var,obs,.2)
    assert not preserved.any()


def test_vertical_datum_and_read_only_provenance_are_preserved():
    col,z,var,obs=fixture(.3,-.2)
    first=separate_measured_floor_labels(col,z,var,obs,.2)
    shifted=separate_measured_floor_labels(col,z+10000.,var,obs,.2)
    for a,b in zip(first,shifted):np.testing.assert_array_equal(a,b)
    for bad in [0.,float('nan')]:
        with pytest.raises(ValueError):separate_measured_floor_labels(col,z,var,obs,bad)


def test_nonplanar_neighbourhood_does_not_certify_a_column_height():
    col,z,var,obs=fixture()
    z+=np.where(np.indices(z.shape).sum(axis=0)%2,.12,-.12)
    excluded,preserved=separate_measured_floor_labels(col,z,var,obs,.2)
    assert not preserved.any();np.testing.assert_array_equal(excluded,col)


@pytest.mark.parametrize('slope', [0., np.tan(np.deg2rad(9.))])
def test_sparse_unlabelled_pole_sides_cannot_erase_existing_floor_next_to_a_column(slope):
    col,z,var,obs=fixture(slope_x=slope,datum=-.42)
    col[:]=False;col[7,7:10]=True
    # Sparse side heights, still mislabelled as ground by the frontend, used
    # to spoil the least-squares reference for the three genuine floor cells.
    # These are not new column labels or fabricated ground measurements.
    z[4,7]+=.615;z[5,6]+=.104;z[5,7]+=.20
    z[6,7]+=.744;z[6,9]+=.418;col[6,7]=col[6,9]=True
    original=[a.copy() for a in (col,z,var,obs)]
    excluded,preserved=separate_measured_floor_labels(col,z,var,obs,.2,max_step_m=.1,
        max_ground_slope_rad=np.deg2rad(10.))
    assert preserved[7,7:10].all()
    assert not excluded[7,7:10].any()
    assert not preserved[6,7] and not preserved[6,9]
    for a,b in zip((col,z,var,obs),original):np.testing.assert_array_equal(a,b)
    # The repair changes ground provenance only: the three occupied volumes
    # remain hard, and an unobserved floor cannot inherit this witness.
    assert col[7,7:10].all()
    obs[7,7:10]=False
    assert not separate_measured_floor_labels(col,z,var,obs,.2,max_step_m=.1,
        max_ground_slope_rad=np.deg2rad(10.))[1][7,7:10].any()


def test_a_minority_height_cluster_cannot_use_its_candidate_as_its_own_reference():
    col,z,var,obs=fixture();col[:]=False;col[7,7]=True
    z[7,7]=.25
    # Five adjacent measurements at that height are insufficient. The much
    # larger independent lower-floor population must not be silently ignored.
    for r,c in [(6,6),(6,7),(6,8),(7,6),(7,8)]:z[r,c]=.25
    excluded,preserved=separate_measured_floor_labels(col,z,var,obs,.2,max_step_m=.1,
        max_ground_slope_rad=np.deg2rad(10.))
    assert excluded[7,7] and not preserved.any()


@pytest.mark.parametrize('slope',[0.,float('nan'),np.pi/2])
def test_invalid_declared_ground_slope_is_rejected(slope):
    with pytest.raises(ValueError,match='geometry'):
        separate_measured_floor_labels(*fixture(),.2,max_ground_slope_rad=slope)


def test_preserved_floor_enters_the_same_shared_posterior_but_column_collision_stays_lethal():
    import torch
    from fsgp_bgk_se2.shared_terrain import SharedTerrainParams,shared_terrain_prediction,shared_geometric_probability
    from fsgp_bgk_se2.probabilistic_risk import ProbabilityParams
    torch.set_num_threads(1)
    col,z,var,obs=fixture();excluded,preserved=separate_measured_floor_labels(col,z,var,obs,.2)
    y,x=np.mgrid[-7:8,-7:8]*.2
    raw=np.r_[np.c_[x.ravel(),y.ravel(),z.ravel()],np.c_[x[col]+.04,y[col]+.04,z[col]+1.]]
    params=SharedTerrainParams(condition_local_residual=True,condition_full_residual=True,current_floor_column_seams=True,
        condition_measured_column_seams=True,predict_ground_under_columns=True)
    outputs=[]
    for device in (['cpu','cuda'] if torch.cuda.is_available() else ['cpu']):
        q=shared_terrain_prediction(z,var,obs&~excluded,x,y,raw,0.,.2,np.zeros_like(col),
            device=device,params=params,positive_veto_height=None,columns=excluded)
        assert q is not None and q.valid[preserved].all()
        np.testing.assert_allclose(q.mean[preserved],z[preserved],atol=1e-6)
        outputs.append(q)
        p=ProbabilityParams(yaw_bins=8,internal_steps=True,internal_step_model='height_jump')
        r,c,u=shared_geometric_probability(q,p,np.zeros_like(col))[0]
        assert len(u[(r==7)&(c==7)])==1 and u[(r==7)&(c==7)][0]<.05
        rows,cols,upper=shared_geometric_probability(q,p,col)[0]
        center=upper[(rows==7)&(cols==7)]
        assert not len(center) or center[0]>.05 # occupied pose cannot acquire a free certificate
    if len(outputs)>1:
        np.testing.assert_array_equal(outputs[0].valid,outputs[1].valid)
        np.testing.assert_allclose(outputs[0].mean,outputs[1].mean,atol=3e-5)
        np.testing.assert_allclose(outputs[0].variance,outputs[1].variance,atol=2e-6)
