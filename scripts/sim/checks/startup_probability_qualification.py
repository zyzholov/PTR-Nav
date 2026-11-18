#!/usr/bin/env python3
"""Bounded conditional-plane CPU/CUDA parity/timing, not a navigation trial."""
import argparse
import copy
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from fsgp_bgk_se2.startup_support import Stance,StartupSupport
from fsgp_bgk_se2.probabilistic_risk import ProbabilityParams,geometric_probability
from fsgp_bgk_se2.conditioned_probability import condition_initial_plane


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[3]
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    p=ProbabilityParams(internal_steps=True)
    y,x=np.mgrid[-10.3:10.31:.2,-10.3:10.31:.2]
    h=(.02*x+.01*y).astype(np.float32);v=np.full_like(h,1e-4)
    seen=np.hypot(x,y)>.65;hit=np.zeros_like(seen)
    model=StartupSupport();pose=Stance(0,0,0,0,0,0,0)
    model.update(h,v,seen,hit,x,y,pose,0,p)
    prior,mean,var=model.update(h,v,seen,hit,x,y,replace(pose,stamp=2),2,p)
    assert prior.any(),model.reason
    height=np.where(prior,mean,h);variance=np.where(prior,var,v);support=seen|prior
    raw=geometric_probability(height,variance,support,p,collision=hit)
    outputs={};timings={}
    import cupy as cp
    for name,xp,runs in (("cpu",np,1),("cuda",cp,4)):
        timings[name]=[]
        for _ in range(runs):
            layers=copy.deepcopy(raw);start=time.perf_counter()
            outputs[name]=condition_initial_plane(layers,height,variance,support,hit,prior,x,y,model,p,xp=xp)
            if xp is cp:cp.cuda.Stream.null.synchronize()
            timings[name].append(1000*(time.perf_counter()-start))
    a,b=[np.stack(outputs[n]["failure_probability_upper"]) for n in ("cpu","cuda")]
    modes=[np.stack(outputs[n]["probability_pose_event"]) for n in ("cpu","cuda")]
    same=np.array_equal(np.isfinite(a),np.isfinite(b)) and np.array_equal(*modes)
    delta=float(np.nanmax(np.abs(a-b)))
    record=dict(scope="small_qualification_not_batch_experiment",shape=h.shape,parameters=asdict(p),
        conditional_cells=int(prior.sum()),source_count=len(model.measurement_sigma),
        max_probability_delta=delta,masks_equal=same,timing_ms=timings,
        gpu=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        hashes={n:digest(root/"src/fsgp_bgk_se2/fsgp_bgk_se2"/n) for n in
                ("startup_support.py","conditioned_probability.py","probability_cuda.py","probabilistic_risk.py")},
        runner_sha256=digest(Path(__file__)),passed=bool(same and delta<2e-5))
    np.savez_compressed(args.output/"arrays.npz",height=height,variance=variance,support=support,prior=prior,
        probability_cpu=a,probability_cuda=b,pose_events=modes[0])
    (args.output/"qualification.json").write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps(record,indent=2));return 0 if record["passed"] else 1


if __name__=="__main__":raise SystemExit(main())
