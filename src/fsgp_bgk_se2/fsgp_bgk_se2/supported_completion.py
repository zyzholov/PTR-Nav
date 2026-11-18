"""Bounded interpolation candidates, never relabeled as physical observations."""

import numpy as np


def supported_completion(elevation, variance, valid, prediction, *, max_std_m=.04,
                         max_distance_m=.35, blocked=None):
    shape=elevation.shape
    if variance.shape!=shape or valid.shape!=shape or max_std_m<=0 or max_distance_m<=0:
        raise ValueError("invalid completion geometry or gates")
    local=getattr(prediction,"local_conditioned",None)
    mask=np.zeros(shape,bool)
    if local is not None:
        mean=np.asarray(prediction.mean).reshape(shape)
        uncertainty=np.asarray(prediction.latent_var).reshape(shape)
        distance=np.asarray(prediction.nearest_distance).reshape(shape)
        mask=(~valid & np.asarray(local).reshape(shape) & np.asarray(prediction.accepted).reshape(shape)
            & np.isfinite(mean) & np.isfinite(uncertainty) & (uncertainty>=0)
            & (uncertainty<=max_std_m**2) & (distance<=max_distance_m))
        if blocked is not None:
            if blocked.shape!=shape:raise ValueError("blocked completion mask shape mismatch")
            mask &= ~blocked
        elevation=np.where(mask,mean,elevation).astype(elevation.dtype,copy=False)
        variance=np.where(mask,np.maximum(uncertainty,1e-10),variance).astype(variance.dtype,copy=False)
    return elevation,variance,valid|mask,mask
