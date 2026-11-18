"""Skipping an optional proof can only retain a more conservative bound."""
import numpy as np
import pytest
from fsgp_bgk_se2.refinement_region import refinement_query_mask


def test_priority_region_preserves_requested_masks_and_original_arrays():
    x,y=np.meshgrid(np.arange(-5,5.1,.2),np.arange(-5,5.1,.2))
    mask=np.ones((4,*x.shape),bool);mask[0,10,10]=False;before=mask.copy()
    actual=refinement_query_mask(mask,x,y,radius_m=3.,center=(.3,-.2),pose_stamp=10.1,map_stamp=10.)
    expected=((x-.3)**2+(y+.2)**2)<=9
    np.testing.assert_array_equal(actual,mask&expected[None]);np.testing.assert_array_equal(mask,before)
    assert not np.shares_memory(actual,mask)
    assert refinement_query_mask(mask,x,y,radius_m=0.) is mask


@pytest.mark.parametrize('center,pose_stamp,map_stamp',[
    (None,None,10.),((0.,0.),9.,10.),((0.,0.),11.,10.),((np.nan,0.),10.,10.),
    ((0.,0.),np.inf,10.),((0.,0.),10.,np.nan)])
def test_unknown_or_stale_pose_falls_back_to_full_proof_work_instead_of_guessing(center,pose_stamp,map_stamp):
    x,y=np.meshgrid(np.arange(6),np.arange(5));mask=np.ones((3,5,6),bool)
    assert refinement_query_mask(mask,x,y,radius_m=3.,center=center,pose_stamp=pose_stamp,map_stamp=map_stamp) is mask


@pytest.mark.parametrize('radius',[-1.,np.nan,np.inf])
def test_invalid_priority_configuration_is_not_silently_accepted(radius):
    x,y=np.meshgrid(np.arange(6),np.arange(5))
    with pytest.raises(ValueError):refinement_query_mask(np.ones((3,5,6),bool),x,y,radius_m=radius)
