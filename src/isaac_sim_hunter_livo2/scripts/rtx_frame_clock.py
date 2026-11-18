"""Isaac 5.1 snapshot bridge clock, derived from actual render events.

The bridge stamps the current dispatch while the RTX XYZ snapshot refers to
the previous render pose. Retain that acquisition time, including irregular
render intervals and frame skipping. Never infer it from callback wall time,
fit an offset to ground truth, or alter any XYZ bytes.
"""
from collections import deque
import math


class RtxFrameClock:
    def __init__(self,capacity=64):
        self.renders=deque(maxlen=capacity);self.last_published=None

    def render(self,seconds):
        if not math.isfinite(seconds) or seconds<0:raise ValueError('invalid physical source clock')
        stamp=round(seconds*1e9)
        if self.renders and stamp<self.renders[-1]:
            self.renders.clear();self.last_published=None
        if not self.renders or stamp!=self.renders[-1]:self.renders.append(stamp)

    def source(self,dispatch_ns):
        stamps=list(self.renders)
        matches=[i for i,t in enumerate(stamps)if abs(t-dispatch_ns)<=2]
        if len(matches)!=1 or matches[0]==0:return None
        source=stamps[matches[0]-1]
        if self.last_published is not None and source<=self.last_published:return None
        self.last_published=source
        return source


def imu_point_acceleration(root_acceleration,angular_velocity,angular_acceleration,lever_world):
    """Rigid-body IMU origin acceleration, retaining tangential/centripetal terms."""
    import numpy as np
    return (np.asarray(root_acceleration)+np.cross(angular_acceleration,lever_world)+
            np.cross(angular_velocity,np.cross(angular_velocity,lever_world)))
