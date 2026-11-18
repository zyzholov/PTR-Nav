"""Conditional geometric bounds retaining the shared initial-plane source.

The simulation-only initial plane is an explicit modeling assumption. Its
measurement errors remain shared, not independent per predicted cell. These
pose bounds use conservative union accounting; they are not mission guarantees.
"""
import numpy as np
from functools import lru_cache
from .probabilistic_risk import _geometry,_interval_upper


def merge_shared_field(height,variance,support,prior,prediction):
    """Preserve the original conditional stance source until real evidence.

    A GP extrapolation is not an observation contradicting or replacing the
    separately qualified fixed plane. Mixed footprints use the plane-source
    proof below (arbitrary cross correlation); the GP-only proof may only use
    its own selected cells. Collision/pit/pending vetoes are applied by ingress.
    """
    joint=prediction.valid&~prior
    return (np.where(joint,prediction.mean,height),np.where(joint,prediction.variance,variance),
            support|joint,joint)


@lru_cache(maxsize=256)
def conditional_geometry(params,yaw_index):
    offsets,weights,limits=_geometry(params,yaw_index)
    n=len(offsets)
    # max(r)-min(r)<=h is exactly |r_i-r_j|<=h for every pair.
    # The previous |r_i|<=h/2 condition was sufficient but needlessly stronger.
    i,j=np.triu_indices(n,1)
    pairs=weights[2+i]-weights[2+j]
    return offsets,np.vstack((weights[:2],pairs,weights[2+n:])),np.r_[
        limits[:2],np.full(len(i),params.roughness_limit,np.float32),limits[2+n:]]


@lru_cache(maxsize=256)
def conditional_evaluation_geometry(params,yaw_index):
    offsets,weights,limits=conditional_geometry(params,yaw_index)
    _,base,_=_geometry(params,yaw_index)
    n=len(offsets)
    return offsets,np.vstack((weights,base[2:2+n])),np.r_[
        limits,np.full(n,params.roughness_limit/2,np.float32)]


def conditional_union_upper(means,stds,limits,n,xp,ndtr):
    """Two sufficient proofs of the same residual-span event, not independence.

    The interval center is fixed from posterior means, never from sampled
    unknown errors. All residuals inside this interval imply span <= max_step.
    Pitch, roll and internal-step events remain present in both proofs.
    """
    pair_end=2+n*(n-1)//2
    residual_begin=means.shape[0]-n
    center=(xp.min(means[residual_begin:],axis=0)+xp.max(means[residual_begin:],axis=0))/2
    tails=_interval_upper(means[:residual_begin],stds[:residual_begin],limits[:residual_begin,None],xp,ndtr)
    residual_tails=_interval_upper(means[residual_begin:]-center,stds[residual_begin:],
        limits[residual_begin:,None],xp,ndtr)
    common=xp.sum(tails[:2],axis=0)+xp.sum(tails[pair_end:],axis=0)
    return xp.minimum(1,common+xp.minimum(xp.sum(tails[2:pair_end],axis=0),
                                         xp.sum(residual_tails,axis=0)))


def condition_initial_plane(layers,height,variance,support,collision,prior,x,y,model,params,xp=np):
    layers["probability_pose_event"]=[np.zeros_like(height,dtype=np.float32) for _ in range(params.yaw_bins)]
    # The conditional feature stencil contains internal steps only. External
    # edge constraints require their original full envelope, not a partial proof.
    if not params.internal_steps:return layers
    if xp is np:
        from scipy.special import ndtr
        host=np.asarray
    else:
        from cupyx.scipy.special import ndtr
        host=xp.asnumpy
    h=xp.asarray(height,dtype=xp.float32);var=xp.asarray(variance,dtype=xp.float32)
    observed=xp.asarray(support,dtype=bool);occupied=xp.asarray(collision,dtype=bool)
    initial=xp.asarray(prior,dtype=bool)
    dx=xp.asarray(x-model.anchor.x,dtype=xp.float32);dy=xp.asarray(y-model.anchor.y,dtype=xp.float32)
    coefficients=xp.asarray(model.plane_weights,dtype=xp.float32)
    original_sigma=xp.sqrt(xp.asarray(model.measurement_sigma,dtype=xp.float32)**2*params.variance_scale+
        params.height_std_floor**2)
    sigma=xp.sqrt(xp.maximum(var,0)*params.variance_scale+params.height_std_floor**2)
    if xp is not np:
        from .probability_cuda import fused_conditioned
        valid=observed&xp.isfinite(h)&xp.isfinite(var)&(var>=0)
        fused=fused_conditioned(layers,h,sigma,valid,occupied,initial,dx,dy,coefficients,original_sigma,params)
        if fused is not None:return fused
    rows,cols=h.shape
    for k in range(params.yaw_bins):
        layers["probability_pose_event"][k][:]=0
        offsets,weights,limits=conditional_evaluation_geometry(params,k)
        rr,cc=xp.indices(h.shape);affected=xp.zeros(h.shape,dtype=bool)
        for dr,dc in offsets:
            affected |= initial[xp.clip(rr+dr,0,rows-1),xp.clip(cc+dc,0,cols-1)]
        r,c=xp.where(affected)
        if not r.size:continue
        ro=xp.asarray([q[0] for q in offsets])[:,None]+r[None]
        co=xp.asarray([q[1] for q in offsets])[:,None]+c[None]
        inside=xp.all((ro>=0)&(ro<rows)&(co>=0)&(co<cols),axis=0)
        ro=xp.clip(ro,0,rows-1);co=xp.clip(co,0,cols-1)
        valid=inside&xp.all(observed[ro,co]&xp.isfinite(h[ro,co])&xp.isfinite(var[ro,co])&(var[ro,co]>=0),axis=0)
        valid &= ~xp.any(occupied[ro,co],axis=0)
        valid &= xp.isfinite(xp.asarray(layers["failure_probability_upper"][k])[r,c])
        own=initial[ro,co]
        features=xp.asarray(weights)
        heights=h[ro,co]-h[r,c][None]
        means=features@xp.nan_to_num(heights)
        design=xp.stack((dx[ro,co].T,dy[ro,co].T,xp.ones((r.size,len(offsets)),xp.float32)),axis=2)
        plane_projection=features[None]@(design*own.T[:,:,None])
        effective=plane_projection@coefficients
        shared_std=xp.sum(xp.abs(effective)*original_sigma[None,None],axis=2).T
        # Initial cells retain a separate model-floor error. Current measured
        # cells keep their own measurement moment; no covariance is invented.
        local_std=xp.where(own,params.height_std_floor,sigma[ro,co])
        bound_std=shared_std+xp.abs(features)@xp.nan_to_num(local_std)
        upper=conditional_union_upper(means,bound_std,xp.asarray(limits),len(offsets),xp,ndtr)
        rh,ch,good,values=host(r),host(c),host(valid),host(upper)
        rh,ch,values=rh[good],ch[good],values[good]
        layers["failure_probability_upper"][k][rh,ch]=np.minimum(
            layers["failure_probability_upper"][k][rh,ch],values)
        layers["probability_pose_event"][k][rh,ch]=1
    return layers
