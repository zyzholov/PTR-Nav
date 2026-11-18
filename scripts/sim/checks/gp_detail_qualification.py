#!/usr/bin/env python3
"""Refit two kernels to identical frozen real inputs; not navigation success."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from shared_certificate_qualification import frozen_prediction
from fsgp_bgk_se2.shared_terrain import shared_geometric_probability
from fsgp_bgk_se2.probabilistic_risk import _geometry


def grid_probability(prediction,physical,collision):
    start=time.perf_counter();updates=shared_geometric_probability(prediction,physical,collision)
    output=np.full((physical.yaw_bins,*prediction.mean.shape),np.nan,np.float32)
    for k,(r,c,p) in enumerate(updates):output[k,r,c]=p
    return output,1000*(time.perf_counter()-start)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    outputs={};records={};physical=None
    for name,detail,halo in (('core',0,0.),('detail',64,0.),('halo',0,.8),('detail_halo',64,.8)):
        pred,physical,collision,provenance=frozen_prediction(args.snapshot,'cuda',replay_initial=True,
            detail_override=detail,halo_override=halo,verify_posterior=name=='core')
        p,cold_ms=grid_probability(pred,physical,collision);times=[]
        for _ in range(3):_,elapsed=grid_probability(pred,physical,collision);times.append(elapsed)
        x,y,_,yaw=provenance['query_pose'];left,bottom=provenance['map_min'];probes=[]
        for offset in (0.,.6,1.,2.5):
            row=int(math.floor((y+offset*math.sin(yaw)-bottom)/physical.resolution))
            col=int(math.floor((x+offset*math.cos(yaw)-left)/physical.resolution))
            per_yaw=[]
            for k in (0,35):
                offsets=_geometry(physical,k)[0]
                missing=[(a,b) for a,b in offsets if not pred.valid[row+a,col+b]]
                value=float(p[k,row,col])
                per_yaw.append(dict(yaw_index=k,upper=value if math.isfinite(value) else None,
                                    unsupported_offsets=missing))
            probes.append(dict(offset_m=offset,poses=per_yaw))
        outputs[name]=dict(mean=pred.mean,variance=pred.variance,valid=pred.valid,probability=p)
        records[name]=dict(provenance=provenance,training_points=pred.training_points,
            detail_features=pred.detail_features,discrepancy_m=pred.discrepancy,fit_gpu_wall_ms=pred.fit_ms,
            cold_probability_ms=cold_ms,warm_probability_ms=times,probes=probes)
    cpu,_,_,_=frozen_prediction(args.snapshot,'cpu',replay_initial=True,detail_override=64,halo_override=.8,verify_posterior=False)
    cp,_=grid_probability(cpu,physical,collision);gpu=outputs['detail_halo']['probability'];common=np.isfinite(cp)&np.isfinite(gpu)
    parity=dict(support_equal=bool(np.array_equal(np.isfinite(cp),np.isfinite(gpu))),
        max_probability_delta=float(np.max(abs(cp[common]-gpu[common]))) if common.any() else None)
    passed=parity['support_equal'] and parity['max_probability_delta'] is not None and parity['max_probability_delta']<5e-5
    root=Path(__file__).resolve().parents[3]
    record=dict(scope='small implementation/model qualification on frozen measurements; neither physical arrival nor probability calibration',
        accepted=passed,models=records,cpu_cuda_parity=parity,
        hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in (
            'scripts/sim/checks/gp_detail_qualification.py','scripts/sim/checks/shared_certificate_qualification.py',
            'src/fsgp_bgk_se2/fsgp_bgk_se2/shared_terrain.py','src/fsgp_bgk_se2/fsgp_bgk_se2/shared_terrain_cuda.py')})
    (args.output/'qualification.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(args.output/'fields.npz',**{name+'_'+k:v for name,a in outputs.items() for k,v in a.items()})
    print(json.dumps(dict(accepted=passed,parity=parity,probes={k:v['probes'] for k,v in records.items()})),flush=True)
    return int(not passed)


if __name__=='__main__':raise SystemExit(main())
