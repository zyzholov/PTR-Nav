import numpy as np
from fsgp_bgk_se2.se2_grid_tools import repeat_or_trim_slices


def test_yaw_upsampling_preserves_all_four_angular_quadrants():
    source=[np.full((2,2),i,np.float32) for i in range(4)]
    result=repeat_or_trim_slices(source,8,np.zeros((2,2)))
    assert [r[0,0] for r in result]==[0,0,1,1,2,2,3,3]
    result[0][0,0]=99
    assert source[0][0,0]==0


def test_yaw_downsampling_uses_destination_centers_not_first_four_layers():
    source=[np.full((2,2),i,np.float32) for i in range(8)]
    result=repeat_or_trim_slices(source,4,np.zeros((2,2)))
    assert [r[0,0] for r in result]==[1,3,5,7]
