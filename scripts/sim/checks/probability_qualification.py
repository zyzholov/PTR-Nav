#!/usr/bin/env python3
"""Bounded qualification only: five analytic inputs, CPU/CUDA parity and timing.

This is not a comparative experiment, a calibrated failure test, or an Isaac
navigation result. Refuses to overwrite earlier evidence.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from fsgp_bgk_se2.probabilistic_risk import ProbabilityParams,geometric_probability,height_support_radius


ROOT=Path(__file__).resolve().parents[3]


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--internal-steps",action="store_true")
    parser.add_argument("--snapshot",type=Path,help="Replay one captured probability map; no startup conditional override")
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    params=ProbabilityParams(internal_steps=args.internal_steps)
    shape=(104,104); center=(52,52)
    h=np.zeros(shape,np.float32); v=np.full(shape,1e-4,np.float32)
    valid=np.ones(shape,bool); none=np.zeros(shape,bool)
    slope=np.broadcast_to(np.tan(.5)*(np.arange(104)-52)[None,:]*.2,shape).astype(np.float32)
    step=h.copy(); step[:,53:]=.25
    hole=valid.copy(); hole[50:55,50:55]=False
    column=none.copy(); column[52,53]=True
    scenes=[("flat",h,valid,none),("directional_slope",slope,valid,none),
            ("step",step,valid,none),("missing_support",h,hole,none),
            ("pillar_without_ground",h,none,column)]
    if args.snapshot:
        from rclpy.serialization import deserialize_message
        from se2_grid_msgs.msg import SE2Grid
        from fsgp_bgk_se2.se2_grid_tools import get_layer_slices,grid_buffer_to_world
        msg=deserialize_message(args.snapshot.read_bytes(),SE2Grid)
        def layer(name):return grid_buffer_to_world(get_layer_slices(msg,name)[0][0],msg)
        def scalar(name):return float(layer(name).flat[0])
        h=layer("probability_elevation");v=layer("probability_variance")
        valid=np.isfinite(h)&np.isfinite(v)&(v>=0);none=layer("collision_risk")>=.5
        params=ProbabilityParams(resolution=msg.info.length_x/h.shape[1],
            yaw_bins=len(get_layer_slices(msg,"failure_probability_upper")[0]),
            front=scalar("probability_front"),back=scalar("probability_back"),
            half_width=scalar("probability_half_width"),margin=0,
            max_pitch=scalar("probability_max_pitch"),max_roll=scalar("probability_max_roll"),
            max_step=scalar("probability_max_step"),height_std_floor=scalar("probability_height_std_floor"),
            variance_scale=scalar("probability_variance_scale"),
            internal_steps=scalar("probability_contract") in (3,4))
        shape=h.shape;center=tuple(n//2 for n in shape)
        scenes=[("recorded_map",h,valid,none)]
    source=ROOT/"src/fsgp_bgk_se2/fsgp_bgk_se2/probabilistic_risk.py"
    record={"scope":"small_qualification_not_batch_experiments","utc":datetime.now(timezone.utc).isoformat(),
            "source_sha256":digest(source),"runner_sha256":digest(Path(__file__)),
            "parameters":asdict(params),"contract":3 if params.internal_steps else 2,"height_support_radius_cells":height_support_radius(params),
            "cuda_source_sha256":digest(source.with_name("probability_cuda.py")),"shape":shape,"runs":[],"accepted":False}
    if args.snapshot:
        record.update(snapshot=str(args.snapshot),snapshot_sha256=digest(args.snapshot),
            scope="one_archived_map_ordinary_probability_stage_not_full_pipeline",conditional_startup_plane_applied=False)
    target=args.output/"qualification.json"
    try:
        import cupy as cp
        record["gpu"]={"cupy":cp.__version__,"device":cp.cuda.runtime.getDeviceProperties(0)["name"].decode()}
        cp.asarray([0],dtype=cp.float32); cp.cuda.Stream.null.synchronize()
        for name,height,support,occupied in scenes:
            start=time.perf_counter()
            cpu=geometric_probability(height,v,support,params,collision=occupied)
            cpu_ms=1000*(time.perf_counter()-start)
            start=time.perf_counter()
            gpu=geometric_probability(height,v,support,params,collision=occupied,xp=cp)
            cp.cuda.Stream.null.synchronize(); first_gpu_ms=1000*(time.perf_counter()-start)
            timings=[]
            for _ in range(2):
                cp.cuda.Stream.null.synchronize(); start=time.perf_counter()
                gpu=geometric_probability(height,v,support,params,collision=occupied,xp=cp)
                cp.cuda.Stream.null.synchronize(); timings.append(1000*(time.perf_counter()-start))
            a=np.stack(cpu["failure_probability_upper"]); b=np.stack(gpu["failure_probability_upper"])
            same_support=np.array_equal(np.isfinite(a),np.isfinite(b))
            delta=float(np.nanmax(np.abs(a-b)))
            ta=np.stack(cpu["height_error_tail_upper"]);tb=np.stack(gpu["height_error_tail_upper"])
            tail_same_support=np.array_equal(np.isfinite(ta),np.isfinite(tb))
            tail_delta=float(np.nanmax(np.abs(ta-tb))) if np.isfinite(ta).any() else 0.
            p0=float(b[(0,)+center]); p90=float(b[(9,)+center])
            semantic=(p0<.05 if name=="flat" else p0<.05 and p90>=.95 if name=="directional_slope" else
                      np.isnan(p0) if name=="missing_support" else
                      bool(np.isfinite(b).any()) if name=="recorded_map" else p0==1)
            filename=name+".npz"
            np.savez_compressed(args.output/filename,height=height,variance=v,observed=support,collision=occupied,
                                probability_cpu=a,probability_gpu=b,height_tail_cpu=ta,height_tail_gpu=tb)
            row=dict(scene=name,valid_fraction=float(np.isfinite(b).mean()),cpu_ms=cpu_ms,
                     first_gpu_ms=first_gpu_ms,warm_gpu_ms=timings,
                     max_cpu_gpu_delta=delta,same_support=same_support,semantic_pass=bool(semantic),
                     max_tail_cpu_gpu_delta=tail_delta,tail_same_support=tail_same_support,
                     center_probability=p0 if np.isfinite(p0) else None,
                     center_cross_heading_probability=p90 if np.isfinite(p90) else None,
                     archive=filename,archive_sha256=digest(args.output/filename))
            record["runs"].append(row)
            print(json.dumps(row),flush=True)
            target.write_text(json.dumps(record,indent=2,allow_nan=False)+"\n")
        record["numerical_pass"]=all(r["same_support"] and r["max_cpu_gpu_delta"]<2e-5 and
            r["tail_same_support"] and r["max_tail_cpu_gpu_delta"]<2e-5 and r["semantic_pass"] for r in record["runs"])
        record["probability_stage_125ms_pass"]=all(max(r["warm_gpu_ms"])<=125 for r in record["runs"])
        record["accepted"]=record["numerical_pass"] and record["probability_stage_125ms_pass"]
    except Exception as exc:
        record["error"]=repr(exc)
        print(record["error"],flush=True)
    finally:
        target.write_text(json.dumps(record,indent=2,allow_nan=False)+"\n")
    return 0 if record["accepted"] else 1


if __name__=="__main__": raise SystemExit(main())
