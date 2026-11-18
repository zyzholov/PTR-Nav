"""Conditional fixed-plane / GP interface with explicit source covariance.

For a height contrast a'h, let a_I refer to the fixed initial plane and a_G
to selected GP cells. We retain GP cross covariance and the original plane
measurement coefficients. Unknown cross covariance BETWEEN these two sources
is bounded by sigma <= sigma_G + sum_j |(a_I'D A)_j| sigma_j.
No independent-source assumption, synthetic observations or added support.
"""
import math
import numpy as np
import torch
from .probabilistic_risk import _geometry


def mixed_geometric_probability(prediction,prior,initial_height,model,params,collision,map_min,*,heading=None):
    device=prediction.factor.device;dtype=torch.float32
    tensor=lambda a:torch.as_tensor(a,dtype=dtype,device=device)
    initial=torch.as_tensor(prior,dtype=torch.bool,device=device)
    valid=torch.as_tensor(prediction.valid,dtype=torch.bool,device=device)|initial
    occupied=torch.as_tensor(collision,dtype=torch.bool,device=device)
    height=torch.where(initial,tensor(initial_height),tensor(prediction.mean))
    rows,cols=height.shape;rr,cc=torch.where(valid);outputs=[]
    coefficients=tensor(model.plane_weights)
    source_sigma=(tensor(model.measurement_sigma).square()*params.variance_scale+params.height_std_floor**2).sqrt()
    local=None if prediction.local_variance is None else tensor(prediction.local_variance)
    factor=prediction.factor
    def tails(mean,sigma,limit):
        s=sigma.clamp_min(1e-12)*math.sqrt(2.)
        value=.5*(torch.special.erfc((limit+mean)/s)+torch.special.erfc((limit-mean)/s))
        return torch.where(mean.abs()>=limit,torch.ones_like(value),value)
    for k in range(params.yaw_bins):
        # Refine only locally relevant headings in the online path. Every
        # other heading keeps the existing valid, more conservative bound;
        # none is assigned free space or a reduced uncertainty by omission.
        if heading is not None and abs(math.atan2(math.sin((k+.5)*2*math.pi/params.yaw_bins-heading),
            math.cos((k+.5)*2*math.pi/params.yaw_bins-heading)))>1.2+math.pi/params.yaw_bins:
            empty=torch.empty(0,device=device);outputs.append((empty,empty,empty));continue
        offsets,weights,limits=_geometry(params,k);n=len(offsets)
        delta=torch.as_tensor(offsets,device=device);w=tensor(weights);limit=tensor(limits)
        r=rr[:,None]+delta[:,0];c=cc[:,None]+delta[:,1]
        inside=((r>=0)&(r<rows)&(c>=0)&(c<cols)).all(dim=1)
        r=r.clamp(0,rows-1);c=c.clamp(0,cols-1)
        own=initial[r,c]
        eligible=inside&valid[r,c].all(dim=1)&~occupied[r,c].any(dim=1)&own.any(dim=1)&~own.all(dim=1)
        r=r[eligible];c=c[eligible];sr=rr[eligible];sc=cc[eligible]
        pi,pj=torch.triu_indices(n,n,1,device=device)
        pair_l1=(w[2+pi]-w[2+pj]).abs().sum(dim=1)
        xy=delta.to(dtype)*params.resolution
        discrepancy=torch.exp(-(xy[:,None]-xy[None]).square().sum(dim=2)/(2*prediction.discrepancy_lengthscale_m**2))*prediction.discrepancy**2
        values=[];batch=128 if device.type=='cuda' else 32
        for begin in range(0,len(r),batch):
            ro,co=r[begin:begin+batch],c[begin:begin+batch]
            own=initial[ro,co];outside=(~own).to(dtype)
            h=height[ro,co];means=(h-h[:,offsets.index((0,0)),None])@w.T
            masked=w[None]*outside[:,None]
            projected=masked@factor[ro,co]
            covariance=projected@projected.transpose(1,2)+(masked@discrepancy)@masked.transpose(1,2)
            if local is not None:
                covariance+=(masked*local[ro,co][:,None])@masked.transpose(1,2)
            diagonal=covariance.diagonal(dim1=1,dim2=2).clamp_min(0)
            # Initial plane design uses original frozen world coordinates.
            x=(co.to(dtype)+.5)*params.resolution+map_min[0]-model.anchor.x
            y=(ro.to(dtype)+.5)*params.resolution+map_min[1]-model.anchor.y
            design=torch.stack((x,y,torch.ones_like(x)),dim=2)*own[:,:,None]
            effective=(w[None]@design)@coefficients
            shared_sigma=(effective.abs()*source_sigma[None,None]).sum(dim=2)
            std=(diagonal*params.variance_scale).sqrt()+shared_sigma+w.abs().sum(dim=1)[None]*params.height_std_floor
            pair_variance=(diagonal[:,2+pi]+diagonal[:,2+pj]-2*covariance[:,2+pi,2+pj]).clamp_min(0)
            pair_shared=((effective[:,2+pi]-effective[:,2+pj]).abs()*source_sigma[None,None]).sum(dim=2)
            pair_std=(pair_variance*params.variance_scale).sqrt()+pair_shared+pair_l1[None]*params.height_std_floor
            pair_mean=means[:,2+pi]-means[:,2+pj]
            span_pair=tails(pair_mean,pair_std,params.roughness_limit).sum(dim=1)
            residual=means[:,2:2+n];center=(residual.max(dim=1).values+residual.min(dim=1).values)/2
            span_center=tails(residual-center[:,None],std[:,2:2+n],params.roughness_limit/2).sum(dim=1)
            base=tails(means,std,limit[None])
            upper=base[:,:2].sum(dim=1)+base[:,2+n:].sum(dim=1)+torch.minimum(span_pair,span_center)
            values.append((upper+1e-7).clamp_max(1))
        outputs.append((sr,sc,torch.cat(values) if values else torch.empty(0,device=device)))
    counts=[len(r) for r,_,_ in outputs]
    packed=torch.cat([torch.stack((r,c,u),dim=1) for r,c,u in outputs]).cpu().numpy()
    out=[];offset=0
    for count in counts:
        part=packed[offset:offset+count];offset+=count
        out.append((part[:,0].astype(int),part[:,1].astype(int),part[:,2]))
    return out
