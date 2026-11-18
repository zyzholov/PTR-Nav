from types import SimpleNamespace as S
import pytest

from navigation_trial import motion_grid_sample,Trial


def message(data):
    return S(info=S(width=2,height=2,resolution=.2,
                    origin=S(position=S(x=-.4,y=.2))),
             header=S(frame_id='map',stamp=S(sec=12,nanosec=100)),data=data)


def test_sparse_motion_recording_keeps_original_geometry_stamp_and_cells():
    row=motion_grid_sample(message([0,100,0,100]))
    assert row['geometry']==[2,2,-.4,.2,.2]
    assert row['occupied']==[1,3]
    assert row['stamp_s']==pytest.approx(12.0000001,abs=1e-12)
    assert row['frame']=='map'


@pytest.mark.parametrize('data',[[0,100],[-1,100,0,0],[80,0,0,0]])
def test_processed_forecasts_unknowns_or_truncated_masks_cannot_be_replayed_as_raw(data):
    with pytest.raises(ValueError):motion_grid_sample(message(data))


def test_topology_recording_retains_unknown_distinct_from_observed_clear():
    msg=message([-1,100,0,-1]);row=motion_grid_sample(msg,allow_unknown=True)
    assert row['occupied']==[1]
    assert row['observed_free']==[2]
    assert row['geometry']==[2,2,-.4,.2,.2]
    assert msg.data==[-1,100,0,-1]
    with pytest.raises(ValueError):motion_grid_sample(message([0,50,-1,100]),allow_unknown=True)
    assert motion_grid_sample(message([0,75,-1,100]),allow_unknown=True)['historical_advisory']==[1]


def test_tracker_observations_are_recorded_without_replacing_collision_measurements():
    node=S(recording=True,motion_grids=[],motion_tracking_grids=[],motion_grid_overflow=False)
    Trial.on_motion_raw(node,message([0,100,0,0]))
    Trial.on_motion_tracking(node,message([0,100,0,100]))
    assert node.motion_grids[0]['occupied']==[1]
    assert node.motion_tracking_grids[0]['occupied']==[1,3]
    assert node.motion_grids[0]['stamp_s']==node.motion_tracking_grids[0]['stamp_s']
    node.recording=False
    Trial.on_motion_tracking(node,message([100,0,0,0]))
    assert len(node.motion_tracking_grids)==1
    node.recording=True;node.motion_tracking_grids=[None]*6000
    Trial.on_motion_tracking(node,message([100,0,0,0]))
    assert node.motion_grid_overflow
