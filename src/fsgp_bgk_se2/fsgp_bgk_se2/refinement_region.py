"""Prioritize optional proof tightening; never change a height/risk source.

Outside this region the caller must RETAIN its already valid conservative
bound. Skipping a tightening is not permission to replace that bound by zero.
"""
import math
import numpy as np


def refinement_query_mask(mask,x,y,*,radius_m,center=None,pose_stamp=None,map_stamp=None):
    if not math.isfinite(radius_m) or radius_m<0:raise ValueError('invalid refinement radius')
    values=np.asarray(mask,bool)
    if values.ndim!=3 or x.shape!=y.shape or values.shape[1:]!=x.shape:
        raise ValueError('refinement mask and world grid differ')
    if not radius_m:return values
    if center is None or pose_stamp is None or map_stamp is None:return values
    if (len(center)!=2 or not np.isfinite([*center,pose_stamp,map_stamp]).all() or
            abs(pose_stamp-map_stamp)>.30):return values
    near=(x-center[0])**2+(y-center[1])**2<=radius_m**2
    return values&near[None]
