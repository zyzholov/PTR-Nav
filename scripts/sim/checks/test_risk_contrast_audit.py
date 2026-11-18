import math
from risk_contrast_audit import query_index


def test_archived_pose_z_is_not_yaw_and_offsets_follow_heading():
    assert query_index((1.,2.,-9.,math.pi/2),.61,(0.,0.),.2,36)==(13,5,9)
    assert query_index((1.,2.,9.,-math.pi/2),.61,(0.,0.),.2,36)==(6,5,27)
