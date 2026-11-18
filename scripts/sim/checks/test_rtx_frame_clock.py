"""Acquisition clock and rigid-body sensor physics, no simulator required."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'src/isaac_sim_hunter_livo2/scripts'))
from rtx_frame_clock import RtxFrameClock,imu_point_acceleration


def test_actual_render_clock_handles_irregular_intervals_and_skipped_messages():
 c=RtxFrameClock()
 for t in [.015,.115,.245,.315]:c.render(t)
 assert c.source(245000000)==115000000
 assert c.source(315000000)==245000000
 assert c.source(315000000) is None
 assert c.source(115000000) is None
 assert c.source(400000000) is None


def test_no_warmup_invention_and_only_nanosecond_roundoff_is_accepted():
 c=RtxFrameClock();c.render(.015)
 assert c.source(15000000) is None
 c.render(.115);assert c.source(115000003) is None
 assert c.source(115000001)==15000000


def test_pause_and_restart_never_reuse_previous_epoch_acquisitions():
 c=RtxFrameClock();c.render(5);c.render(5.1);assert c.source(5100000000)==5000000000
 c.render(5.1);assert c.source(5100000000) is None
 c.render(.02);assert c.source(5100000000) is None
 assert c.source(20000000) is None
 c.render(.12);assert c.source(120000000)==20000000


@pytest.mark.parametrize('t',[float('nan'),float('inf'),-.1])
def test_invalid_source_clocks_are_rejected(t):
 with pytest.raises(ValueError):RtxFrameClock().render(t)


def test_external_imu_includes_tangential_and_centripetal_acceleration():
 root=np.array([.1,.2,.3]);w=np.array([0.,2.,0.]);alpha=np.array([0.,3.,0.]);lever=np.array([.18,0.,.205])
 expected=root+np.array([3*.205,0.,-3*.18])+np.array([-4*.18,0.,-4*.205])
 np.testing.assert_allclose(imu_point_acceleration(root,w,alpha,lever),expected,atol=1e-15)
 np.testing.assert_array_equal(imu_point_acceleration(root,w,alpha,[0,0,0]),root)


def test_point_acceleration_matches_an_independent_rigid_trajectory_derivative():
 from scipy.spatial.transform import Rotation
 t=.27;dt=1e-5;axis=np.array([.3,.4,.5]);axis/=np.linalg.norm(axis);lever=np.array([.18,0,.205])
 def position(t):return np.array([t*t,.2*t**3,-.4*t])+Rotation.from_rotvec(axis*(.3*t+t*t)).apply(lever)
 root=np.array([2.,1.2*t,0.]);w=axis*(.3+2*t);alpha=2*axis
 rw=Rotation.from_rotvec(axis*(.3*t+t*t)).apply(lever)
 numeric=(position(t+dt)-2*position(t)+position(t-dt))/dt**2
 np.testing.assert_allclose(imu_point_acceleration(root,w,alpha,rw),numeric,atol=2e-6)
