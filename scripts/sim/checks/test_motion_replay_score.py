import numpy as np
import pytest

from score_motion_replay import circle_hit,interpolate_pose,mask_points,world_to_map


def test_assessment_transform_uses_robot_truth_to_odom_not_a_fixed_world_offset():
    point=world_to_map([12,4],[10,4,np.pi/2],[1,2,np.pi])
    assert point==pytest.approx([1,4])


def test_pose_interpolation_unwraps_heading_at_pi():
    pose=interpolate_pose([[1,0,0,3.1],[2,2,4,-3.1]],1.5)
    assert pose==pytest.approx([1,2,np.pi])


@pytest.mark.parametrize('rows,t',[
    ([[1,0,0,0],[2,0,0,0]],.9),
    ([[1,0,0,0],[1,0,0,0]],1),
    ([[1,0,0,np.nan],[2,0,0,0]],1.5)])
def test_unmatched_or_invalid_assessment_time_is_rejected(rows,t):
    with pytest.raises(ValueError):interpolate_pose(rows,t)


def test_grid_cells_are_not_used_as_world_coordinates():
    points=mask_points({'geometry':[2,3,-1,2,.2],'occupied':[0,5]},'occupied')
    np.testing.assert_allclose(points,[[-.9,2.1],[-.7,2.5]])
    assert circle_hit(points,[-.8,2.2],.2)
    assert not circle_hit(points,[0,0],.2)
    assert not circle_hit(np.empty((0,2)),[0,0],.2)
