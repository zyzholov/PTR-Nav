#!/usr/bin/env python3
"""Directed geometric/model-mismatch boundary audit, not field calibration.

Hidden analytic terrain is available only to the reference evaluator. The GP
receives masked measured cells and their current returns. Both refused and
accepted queries remain in the output, including confidently wrong completions.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from fsgp_bgk_se2.probabilistic_risk import ProbabilityParams
from fsgp_bgk_se2.shared_terrain import SharedTerrainParams,shared_terrain_prediction,shared_geometric_probability


def surface(kind,level,x,y):
    if kind=='slope':return np.tan(level)*x
    if kind=='ledge':return level*(x>.15)
    if kind=='rubble':return level*np.sin(2*np.pi*x/.8)*np.cos(2*np.pi*y/.9)
    if kind=='curvature':return .5*level*x*x
    raise ValueError(kind)


def reference(kind,level,yaw,params):
    # Dense physical contact rectangle; independently fit a plane. No producer
    # matrices, learned heights, prediction masks or probabilities enter labels.
    u,v=np.meshgrid(np.linspace(-params.back,params.front,45),np.linspace(-params.half_width,params.half_width,35))
    u=u.ravel();v=v.ravel();x=np.cos(yaw)*u-np.sin(yaw)*v;y=np.sin(yaw)*u+np.cos(yaw)*v
    z=surface(kind,level,x,y);a=np.c_[u,v,np.ones(len(u))]
    coeff=np.linalg.lstsq(a,z,rcond=None)[0];residual=z-a@coeff
    pitch=abs(math.atan(coeff[0]));roll=abs(math.atan(coeff[1]));span=float(np.ptp(residual))
    return dict(pitch_rad=pitch,roll_rad=roll,residual_span_m=span,
        geometric_unsafe=bool(pitch>params.max_pitch or roll>params.max_roll or span>params.max_step),
        scope='dense footprint slope/roll/residual proxy; not a wheel-climbing or tire-contact guarantee')


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--output',type=Path,required=True)
    a.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    a.add_argument('--detail-sources',type=int,default=0);a.add_argument('--observation-halo',type=float,default=0.)
    a.add_argument('--condition-local-residual',action='store_true')
    a.add_argument('--step-model',choices=('second_difference','height_jump'),default='second_difference')
    args=a.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    p=ProbabilityParams(internal_steps=True,internal_step_model=args.step_model)
    gp=SharedTerrainParams(max_distance_m=1.25,max_extrapolation_m=.75,
        max_detail_sources=args.detail_sources,observation_halo_m=args.observation_halo,
        condition_local_residual=args.condition_local_residual)
    x,y=np.meshgrid((np.arange(41)-20)*.2,(np.arange(41)-20)*.2)
    cases={'slope':[.25,.43,.54,.60],'ledge':[.06,.12,.20,.30],
           'rubble':[.02,.05,.10,.15],'curvature':[.08,.18,.35,.70]}
    rows=[]
    try:
        for kind,levels in cases.items():
            for level in levels:
                for hole in (0.,.6):
                    z=surface(kind,level,x,y).astype(np.float32)
                    observed=np.hypot(x,y)>=hole
                    variance=np.full(z.shape,.0001,np.float32)
                    raw=np.c_[x[observed],y[observed],z[observed]]
                    started=time.perf_counter()
                    prediction=shared_terrain_prediction(np.where(observed,z,np.nan),variance,observed,x,y,raw,
                        0.,.2,np.zeros(z.shape,bool),params=gp,device=args.device,positive_veto_height=None)
                    fit_ms=1000*(time.perf_counter()-started);prob=[]
                    if prediction is not None:
                        started=time.perf_counter();prob=shared_geometric_probability(prediction,p,np.zeros(z.shape,bool))
                        evaluation_ms=1000*(time.perf_counter()-started)
                    else:evaluation_ms=None
                    for k in (0,4,9):
                        yaw=(k+.5)*2*np.pi/p.yaw_bins;ref=reference(kind,level,yaw,p)
                        upper=None
                        if prob:
                            rr,cc,value=prob[k];v=value[(rr==20)&(cc==20)]
                            if len(v):upper=float(v[0])
                        row=dict(kind=kind,level=level,hidden_radius_m=hole,yaw_rad=yaw,
                            reported_upper=upper,accepted=upper is not None and upper<=.05,
                            reference=ref,fit_wall_ms=fit_ms,geometric_wall_ms=evaluation_ms,
                            unknown=upper is None)
                        row['unsafe_acceptance']=row['accepted'] and ref['geometric_unsafe']
                        row['safe_rejection']=not row['accepted'] and not ref['geometric_unsafe']
                        rows.append(row)
                    print(json.dumps(rows[-3:]),flush=True)
                    (args.output/'partial.json').write_text(json.dumps(rows,indent=2)+'\n')
        result=dict(scope='32 directed terrain/missingness inputs, 96 heading queries; not independent world count or probability calibration',
            geometry=asdict(p),model=asdict(gp),rows=rows,
            unsafe_acceptances=[r for r in rows if r['unsafe_acceptance']],
            safe_rejections=sum(r['safe_rejection'] for r in rows),unknown_queries=sum(r['unknown'] for r in rows),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            model_hashes={name:hashlib.sha256((Path(__file__).resolve().parents[3]/'src/fsgp_bgk_se2/fsgp_bgk_se2'/name).read_bytes()).hexdigest()
                for name in ('shared_terrain.py','shared_terrain_cuda.py','probabilistic_risk.py')},
            interpretation='deterministic dangerous acceptance is a model/stencil stress finding; a single true event does not disprove a conditional Gaussian probability bound')
        (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
        print('BOUNDARY '+json.dumps({k:len(v) if isinstance(v,list) else v for k,v in result.items() if k in ('unsafe_acceptances','safe_rejections','unknown_queries')}))
        return int(bool(result['unsafe_acceptances']))
    finally:
        if not (args.output/'summary.json').exists():
            (args.output/'interrupted.json').write_text(json.dumps({'completed_queries':len(rows),'partial':rows},indent=2)+'\n')


if __name__=='__main__':raise SystemExit(main())
