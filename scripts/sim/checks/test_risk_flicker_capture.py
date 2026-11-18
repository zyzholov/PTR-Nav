from types import SimpleNamespace
import numpy as np
import pytest
from risk_flicker_capture import rotation,point_counts,Capture


def test_box_and_front_counts_do_not_call_underbody_ground_a_self_hit():
    p=np.array([[.2,0.,-.28],[.2,0.,-.1],[.7,0.,-.1],[.7,0.,-.28],[3.,0.,-.2]])
    assert point_counts(p)==dict(points=5,body_box=1,front_raised=1,near_floor=1)
    assert point_counts(np.empty((0,3)))==dict(points=0,body_box=0,front_raised=0,near_floor=0)


def test_world_to_body_rotation_roundtrip_retains_the_vertical_ground_gap():
    angle=np.deg2rad(25.)
    q=SimpleNamespace(x=0.,y=2*np.sin(angle/2),z=0.,w=2*np.cos(angle/2))
    r=rotation(q);p=np.array([[.2,0.,-.28],[.2,0.,-.1]])
    origin=np.array([4.,2.,.3]);world=p@r.T+origin
    np.testing.assert_allclose((world-origin)@r,p,atol=1e-12)
    assert point_counts((world-origin)@r)['body_box']==1
    with pytest.raises(ValueError):rotation(SimpleNamespace(x=0.,y=0.,z=0.,w=0.))


def test_diagnostic_waits_for_fresh_source_data_before_arming_the_initial_stance():
    stamp=lambda t:SimpleNamespace(sec=int(t),nanosec=round((t-int(t))*1e9))
    odom=lambda t:SimpleNamespace(header=SimpleNamespace(stamp=stamp(t)))
    node=SimpleNamespace(clock=stamp(50.),odom=odom(47.8),truth=odom(50.),
                         grid=SimpleNamespace(info=SimpleNamespace(header=SimpleNamespace(stamp=stamp(49.8)))))
    assert not Capture.inputs_fresh(node)
    node.odom=odom(49.95)
    assert Capture.inputs_fresh(node)
    node.clock=stamp(0.)
    assert not Capture.inputs_fresh(node)
