#!/usr/bin/env python3
"""Decompose a frozen shared-GP pose bound; no simulator truth or refitting changes."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import ndtr
import torch

from shared_certificate_qualification import frozen_prediction
from fsgp_bgk_se2.probabilistic_risk import _geometry


def query_index(pose, offset, minimum, resolution, yaw_bins):
    # Archived diagnostic poses contain x, y, Z, yaw, not x, y, yaw.
    x,y,_,yaw=pose
    row=math.floor((y+offset*math.sin(yaw)-minimum[1])/resolution)
    col=math.floor((x+offset*math.cos(yaw)-minimum[0])/resolution)
    k=int((yaw%(2*math.pi))*yaw_bins/(2*math.pi))
    return row,col,k


def decompose(prediction,params,occupied,row,col,k):
    offsets,weights,limits=_geometry(params,k)
    off=np.asarray(offsets);r=row+off[:,0];c=col+off[:,1]
    if (r.min()<0 or c.min()<0 or r.max()>=len(prediction.mean) or
            c.max()>=prediction.mean.shape[1]):return dict(eligible=False,reason='outside map')
    known=bool(prediction.valid[r,c].all());collision=bool(occupied[r,c].any())
    if not known or collision:return dict(eligible=False,known=known,collision=collision)
    height=prediction.mean[r,c]-prediction.mean[row,col]
    factor=prediction.factor.detach().cpu().numpy()[r,c].astype(float)
    distance=np.sum((off[:,None]-off[None])**2,axis=2)*params.resolution**2
    covariance=factor@factor.T+prediction.discrepancy**2*np.exp(
        -distance/(2*prediction.discrepancy_lengthscale_m**2))
    if prediction.local_variance is not None:covariance+=np.diag(prediction.local_variance[r,c])
    def tails(w,limit,shift=0.):
        mu=w@height-shift
        sigma=np.sqrt(np.maximum(0,np.einsum('ij,jk,ik->i',w,covariance,w))*params.variance_scale)
        sigma+=params.height_std_floor*np.abs(w).sum(axis=1)
        sigma=np.maximum(sigma,1e-12)
        return np.where(np.abs(mu)>=limit,1.,ndtr((-limit-mu)/sigma)+ndtr((mu-limit)/sigma))
    n=len(off);base=tails(weights,limits);residual=weights[2:2+n]
    i,j=np.triu_indices(n,1);means=residual@height
    half=float(base[2:2+n].sum());pairs=float(tails(residual[i]-residual[j],params.max_step).sum())
    center=float((means.min()+means.max())/2)
    centered=float(tails(residual,params.max_step/2,center).sum())
    pitch_roll=float(base[:2].sum());internal=float(base[2+n:].sum())
    return dict(eligible=True,yaw_index=k,footprint_cells=n,pitch_roll_upper=pitch_roll,
        residual_half_upper=half,residual_pair_upper=pairs,centered_interval_upper=centered,
        internal_constraint_upper=internal,internal_constraint_model=params.internal_step_model,
        residual_mean_range_m=[float(means.min()),float(means.max())],
        original_upper=min(1.,pitch_roll+min(half,pairs)+internal+1e-7),
        candidate_centered_upper=min(1.,pitch_roll+min(half,pairs,centered)+internal+1e-7))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--offsets',type=float,nargs='+',default=[0.,.2,.4,.6,1.])
    args=parser.parse_args();torch.set_num_threads(1)
    pred,p,occ,meta=frozen_prediction(args.snapshot,'cpu',replay_initial=True)
    records=[]
    for offset in args.offsets:
        r,c,k=query_index(meta['query_pose'],offset,meta['map_min'],p.resolution,p.yaw_bins)
        records.append(dict(offset_m=offset,grid_index=[r,c,k],**decompose(pred,p,occ,r,c,k)))
    result=dict(scope='conditional Gaussian engineering bound decomposition, not measured physical failure probability',
        provenance=meta,probes=records)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(records),flush=True)


if __name__=='__main__':main()
