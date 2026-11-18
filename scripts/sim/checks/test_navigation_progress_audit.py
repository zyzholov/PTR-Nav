import copy
import math
import pytest
from navigation_progress_audit import audit


def fixture(xy=None):
    xy=xy or [(0.,0.),(1.,0.),(2.,-.6),(3.,-.7),(4.,-.7),(5.,0.)]
    return dict(spec=dict(kind='dense',seed=41),goal_requests=[dict(receipt_sim_stamp_s=0.)],
        truth_columns=['stamp_s','x','y','yaw'],truth=[[2.*i,x,y,0.]for i,(x,y)in enumerate(xy)],
        truth_start=[0.,0.,0.],map_start=[0.,0.,0.,0.],
        path_history=[[.5,[[0.,0.],[5.,-.7]]]],
        summary=dict(reached=True,settled=True,evaluation_body_geometry=dict(front=.421,back=.421,half_width=.36)))


def test_direct_entry_passes_without_prescribing_a_particular_gap():
    record=fixture();result=audit(record)
    assert result['entrance_quality_passed']
    assert result['first_published_path_delay_s']==.5
    assert result['first_row_fully_cleared_s']==8.
    assert 4.<result['approach_distance_m']<5.
    assert result['proposed_gap_switches']==0
    record['truth']=[ [t,x,-y,yaw]for t,x,y,yaw in record['truth']]
    assert audit(record)['entrance_quality_passed'] # no preferred side in the assessor


def test_goal_arrival_does_not_hide_a_long_start_area_loop():
    record=fixture([(0.,0.),(1.,0.),(1.,-3.),(-.5,-3.),(-.5,0.),(0.,0.),(2.,-.6),(4.,-.7)])
    result=audit(record)
    assert result['navigation_arrived'] and not result['entrance_quality_passed']
    assert 'approach_distance_m' in result['failed_metrics']
    assert result['start_area_revisits']==1


def test_reference_curves_do_not_count_as_real_movement_and_missing_milestone_is_censored():
    record=fixture([(0.,0.)]*8)
    record['path_history']=[[float(i),[[0.,0.],[4.,(-1)**i]] ]for i in range(1,8)]
    result=audit(record)
    assert result['approach_distance_m']==0.
    assert result['first_row_fully_cleared_s'] is None
    assert not result['entrance_quality_passed']
    assert result['proposed_gap_switches']>0


def test_row_clearance_uses_the_oriented_whole_body_not_just_the_centroid():
    record=fixture();base=audit(record);face=base['first_row_face_world_x_m']
    record['truth']=[[0.,0.,0.,0.],[2.,face+.37,0.,math.pi/2],[4.,face+.7,0.,math.pi/2]]
    assert audit(record)['first_row_fully_cleared_s']==2.
    record['truth'][1][3]=0.
    assert audit(record)['first_row_fully_cleared_s']==4.


def test_timing_and_single_goal_provenance_are_checked():
    record=fixture();record['truth'][2][0]=-1.
    # Put the reversal inside the post-goal observation interval.
    record['truth'][2][0]=1.
    with pytest.raises(ValueError,match='nonmonotonic'):audit(record)
    record=fixture();record['goal_requests']*=2
    with pytest.raises(ValueError,match='one external'):audit(record)
