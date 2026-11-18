"""Simulation-only posterior family for weak, conflicting native floor evidence.

Old and pending heights bound likelihood CENTRES. Neither their midpoint nor
this posterior is a ground measurement. For every member of this ONE global
family the covariance is common; B maps bounded nuisance coordinates into
mean changes. B must never be folded into Gaussian variance. No history is
conditioned twice: each call starts with the original pending-excluded GP.
"""
from dataclasses import dataclass, replace
import numpy as np
import torch


@dataclass
class ConditionalFamily:
    mean: torch.Tensor
    factor: torch.Tensor
    local: torch.Tensor
    bias: torch.Tensor


def condition(mean, factor, local, sites, centres, noise, radius):
    device=factor.device;dtype=torch.float64
    f=factor.to(dtype);mu=torch.as_tensor(mean,dtype=dtype,device=device)
    d=torch.as_tensor(local,dtype=dtype,device=device)
    ids=torch.as_tensor(sites,dtype=torch.long,device=device)
    y,r,rr=(torch.as_tensor(v,dtype=dtype,device=device) for v in (centres,noise,radius))
    if (f.ndim!=2 or mu.shape!=d.shape or mu.shape!=(len(f),) or ids.ndim!=1 or
        len(ids)!=len(torch.unique(ids)) or not len(ids) or ids.min()<0 or ids.max()>=len(f) or
        y.shape!=ids.shape or r.shape!=ids.shape or rr.shape!=ids.shape):
        raise ValueError('invalid bounded evidence geometry or duplicate source')
    if (not torch.isfinite(f).all() or not torch.isfinite(mu[ids]).all() or
        not torch.isfinite(d[ids]).all() or (d[ids]<0).any() or not torch.isfinite(y).all() or
        not torch.isfinite(r).all() or (r<=0).any() or not torch.isfinite(rr).all() or (rr<0).any()):
        raise ValueError('invalid bounded likelihood')
    source=f[ids];diag=d[ids]+r
    chol=torch.linalg.cholesky(source@source.T+torch.diag(diag))
    cross=f@source.T;cross[ids,torch.arange(len(ids),device=device)]+=d[ids]
    gain=torch.cholesky_solve(cross.T,chol).T
    u=source/diag.sqrt()[:,None]
    eigen,vectors=torch.linalg.eigh(u@u.T);root=(1+eigen.clamp_min(0)).sqrt()
    q=(vectors*(1/(root*(1+root)))[None])@vectors.T
    new_factor=f-((f@u.T)@q)@u
    alpha=d[ids]/diag;new_factor[ids]*=(1-alpha)[:,None]
    new_local=d.clone();new_local[ids]=d[ids]*r/diag
    return ConditionalFamily(mu+gain@(y-mu[ids]),new_factor,new_local,gain*rr[None])


def condition_pending(prediction, old, old_variance, pending_height, pending_variance,
                      pending, native_observed, raw_veto, columns, *, max_step,
                      measured, measured_height, measured_variance, max_sites=128):
    """Keep every unrelated veto; include ALL eligible sources or none.

    Supported only for the full-residual GP after the fixed initial stance
    retires. Weak evidence needs a previously observed native floor, finite
    bounded innovation, the original query/distance/uncertainty domain, and
    no collision, negative evidence or discontinuity. No new floor in a hole.
    """
    shape=prediction.mean.shape
    arrays=[np.asarray(a) for a in (old,old_variance,pending_height,pending_variance,
        pending,native_observed,raw_veto,columns,measured,measured_height,measured_variance)]
    if any(a.shape!=shape for a in arrays):raise ValueError('pending evidence must match the map')
    if (prediction.initial_plane_used or prediction.plane_components or prediction.discrepancy!=0 or
        prediction.local_variance is None or prediction.admission_domain is None or
        prediction.bounded_mean_factor is not None):return prediction
    old,ov,new,nv,mask,native,veto,column,seen,z,zv=arrays
    eligible=(mask.astype(bool)&native.astype(bool)&~veto.astype(bool)&~column.astype(bool)&
        prediction.admission_domain&~prediction.contradicted&~prediction.discontinuity_veto&
        np.isfinite(old)&np.isfinite(new)&np.isfinite(ov)&np.isfinite(nv)&(ov>=0)&(nv>=0)&
        np.isfinite(prediction.local_variance)&(np.abs(new-old)<=max_step))
    ids=np.flatnonzero(eligible)
    if not len(ids) or len(ids)>max_sites:return prediction
    family=condition(prediction.mean.ravel(),prediction.factor.reshape(old.size,-1),
        prediction.local_variance.ravel(),ids,(old.ravel()[ids]+new.ravel()[ids])/2,
        np.maximum(np.maximum(ov.ravel()[ids],nv.ravel()[ids]),.01**2),
        np.abs(new.ravel()[ids]-old.ravel()[ids])/2)
    factor=family.factor.to(torch.float32).reshape(*shape,-1)
    bias=family.bias.to(torch.float32).reshape(*shape,-1)
    mean=family.mean.cpu().numpy().reshape(shape).astype(np.float32)
    local=family.local.cpu().numpy().reshape(shape).astype(np.float32)
    variance=(factor.square().sum(dim=-1)).cpu().numpy()+local
    # Float32 casts and reductions are covered conservatively in the bound;
    # retain the same original source-consistency tolerance for ALL means.
    radius=bias.abs().sum(dim=-1).cpu().numpy()
    tolerance=np.minimum(.05,np.maximum(.03,2*np.sqrt(np.maximum(zv,.02**2))))
    valid=(prediction.valid|eligible)&np.isfinite(mean)&np.isfinite(variance)&(variance<=.04**2)
    valid &= ~seen.astype(bool)|((np.abs(mean-z)+radius)<=tolerance)
    return replace(prediction,mean=mean,variance=variance,local_variance=local,factor=factor,
        bounded_mean_factor=bias,bounded_pending=eligible,valid=valid,
        completed=valid&~seen.astype(bool),candidate=(prediction.candidate|eligible)&valid)
