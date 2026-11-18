import numpy as np
from audit_collision_geometry import analyze


def test_frozen_projection_audit_keeps_raw_data_and_low_obstacle():
    yy,xx=np.mgrid[:20,:20]
    z=np.zeros((20,20),np.float32)
    floor=np.c_[(xx.ravel()+.5)*.2,(yy.ravel()+.5)*.2,np.zeros(400)]
    raw=np.r_[floor,[[1.9,2.1,1.1],[2.3,2.1,.55]]].astype(np.float32)
    summary,arrays=analyze(z,np.full_like(z,.0001),np.ones_like(z,bool),raw,(0,0),.2,.65,.10)
    assert summary['raw_input_unchanged']
    assert summary['removed_high_points']==1
    assert summary['occupied_before']==2 and summary['occupied_after']==1
    assert not arrays['after'][10,9] and arrays['after'][10,11]
    np.testing.assert_array_equal(arrays['raw'],raw)


def test_frozen_audit_does_not_invent_a_ground_reference():
    z=np.zeros((20,20),np.float32)
    raw=np.array([[1.9,2.1,1.1]],np.float32)
    summary,arrays=analyze(z,np.full_like(z,.0001),np.zeros_like(z,bool),raw,(0,0),.2,.65,.10)
    assert summary['qualified_cells']==0 and summary['removed_high_points']==0
    np.testing.assert_array_equal(arrays['raw'],raw)
