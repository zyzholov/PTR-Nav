import math
import pytest
from clearance_metrics import moving_obstacle_observation


def fixture(alignment):
    tx,ty,a=alignment;c,s=math.cos(a),math.sin(a)
    poses=[(i*.05,5.,1.8-i*.05) for i in range(41)]
    grids=[]
    for t in (.125,.225,.325,.425,.525):
        x,y=tx+c*5-s*(1.8-t),ty+s*5+c*(1.8-t)
        # Occupied cell is near the actual moving body, outside lane-center ROI.
        grids.append(dict(stamp_s=t,frame='map',geometry=[10,10,x-.55,y-.55,.1],occupied=[55]))
    return grids,poses


@pytest.mark.parametrize('alignment',[(0.,0.,0.),(-.18,.03,.4),(10.,-7.,-1.2)])
def test_observation_follows_actual_source_time_and_fixed_map_alignment(alignment):
    grids,poses=fixture(alignment)
    r=moving_obstacle_observation(grids,poses,alignment,radius=.65,start_s=.1,end_s=.6)
    assert r['accepted'] and r['hit_frames']==5 and r['hit_span_s']==pytest.approx(.4)


def test_empty_stale_unrelated_single_frame_and_unmatched_observation_fail():
    alignment=(0.,0.,0.);grids,poses=fixture(alignment)
    check=lambda gs,ps=poses:moving_obstacle_observation(gs,ps,alignment,radius=.65,start_s=.1,end_s=.6)
    assert not check([])['accepted']
    assert not check(grids[:1])['accepted']
    assert not check(grids,poses[:2])['accepted']
    assert not check(grids,[poses[0],poses[-1]])['accepted']
    assert not check([dict(g,occupied=[]) for g in grids])['accepted']
    assert not check([dict(g,geometry=[10,10,20.,20.,.1]) for g in grids])['accepted']
    assert not check([grids[0]]*5)['accepted']
    assert not check([dict(g,frame='odom') for g in grids])['accepted']
    assert not check(grids,list(reversed(poses)))['accepted']


def test_inflated_halo_does_not_count_as_actual_measured_cylinder():
    grids,poses=fixture((0.,0.,0.))
    grids=[dict(g,geometry=[10,10,g['geometry'][2]+1.3,g['geometry'][3],.1]) for g in grids]
    r=moving_obstacle_observation(grids,poses,(0.,0.,0.),radius=.65,start_s=.1,end_s=.6)
    assert not r['accepted'] and r['hit_frames']==0
