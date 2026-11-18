#!/usr/bin/env python3
"""GPU terrain -> real C++ planner/controller, accelerated bicycle closed loops.

Synthetic fully observed maps and current raw clouds, not Isaac physics/SLAM.
Map input uses ROS CDR serialization, preserving the actual SE2Grid contract.
"""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import yaml

from pointcloud_batch import ROOT, atomic_json, digest
from terrain_risk_benchmark import params_from_config
from paper_clouds import grid
from fsgp_bgk_se2.feature_sgp import FeatureSgpParams, FeatureSgpPredictor
from fsgp_bgk_se2.bgk_seb import BgkSebEstimator
from fsgp_bgk_se2.obstacle_columns import vertical_columns, preserve_collision_risk

SCENARIOS=("straight","turn","reverse","detour","corridor","ramp","cross_slope","waves","steep_ramp",
           "step","unknown","emerge","crossing","stale_map")


def verify_native_build(binary):
    compiled=json.loads(subprocess.check_output([str(binary),"--build-info"],text=True))
    for name,expected in compiled.items():
        if digest(ROOT/name)!=expected:raise ValueError("native binary is stale: "+name)
    return compiled


def scenario(name, seed):
    rng=np.random.default_rng(seed)
    case={"id":name+"_"+str(seed),"scenario":name,"seed":seed,"start":[0.,0.,0.],
        "goal":[float(rng.uniform(6,8)),float(rng.uniform(-.3,.3))],"obstacles":[],
        "terrain":{"kind":"flat","scale":0.},"speed_tau_s":float(rng.uniform(.12,.35)),
        "duration_s":40.,"expected_stop":False}
    box=lambda x,y,hx,hy,**kw:dict(x=float(x),y=float(y),hx=float(hx),hy=float(hy),**kw)
    if name=="turn": case["goal"][1]=float(rng.uniform(1.4,3))
    if name=="reverse": case["goal"]=[float(rng.uniform(-3,-1.6)),float(rng.uniform(-.45,.45))]
    if name=="detour": case["obstacles"]=[box(3.4,rng.uniform(-.15,.15),rng.uniform(.25,.55),rng.uniform(.3,.75))]
    if name=="corridor":
        gap=float(rng.uniform(1.2,2.6));case["corridor_width_m"]=gap;case["goal"][1]=0
        case["obstacles"]=[box(4,gap/2+.2,2.3,.2),box(4,-gap/2-.2,2.3,.2)]
    if name=="ramp": case["terrain"]={"kind":"ramp","scale":float(rng.uniform(.12,.34))}
    if name=="cross_slope": case["terrain"]={"kind":"cross_slope","scale":float(rng.uniform(.12,.35))}
    if name=="waves": case["terrain"]={"kind":"waves","scale":float(rng.uniform(.06,.18))}
    if name=="steep_ramp":
        case["terrain"]={"kind":"steep_ramp","scale":float(rng.uniform(.7,1.1)),"edge":3.5};case["expected_stop"]=True
    if name=="step":
        case["terrain"]={"kind":"step","scale":float(rng.uniform(.26,.5)),"edge":3.5};case["expected_stop"]=True
    if name=="unknown":
        case.update(unknown_band=True,unknown_edge=3.5,expected_stop=True)
    if name=="emerge":
        event=float(rng.uniform(1.1,2));case["event_s"]=event
        case["obstacles"]=[box(4.2,0,.45,.65,appear_s=event)]
    if name=="crossing":
        speed=float(rng.uniform(.5,1));case["event_s"]=1.0
        case["obstacles"]=[box(4.5,2.5,.4,.4,vy=-speed,appear_s=1.,leave_s=7.)]
    if name=="stale_map":case.update(stale_map=True,event_s=1.8,expected_stop=True)
    return case


def elevation(terrain, xy):
    kind,scale=terrain["kind"],terrain["scale"]
    if kind=="ramp": return scale*xy[:,0]
    if kind=="cross_slope": return scale*xy[:,1]
    if kind=="waves":return scale*np.sin(xy[:,0]*.65)*np.cos(xy[:,1]*.45)
    if kind=="steep_ramp":return scale*np.maximum(0,xy[:,0]-terrain["edge"])
    if kind=="step":return np.where(xy[:,0]>=terrain["edge"]-1e-9,scale,0)
    return np.zeros(len(xy))


def make_message(fields):
    from se2_grid_msgs.msg import SE2Grid
    from fsgp_bgk_se2.se2_grid_tools import matrix_to_multiarray_like, world_to_grid_buffer
    msg=SE2Grid();msg.info.header.frame_id="map";msg.info.pos_resolution=.2
    msg.info.yaw_resolution=2*np.pi/36;msg.info.length_x=msg.info.length_y=20.8
    msg.info.length_yaw=2*np.pi;msg.info.pose.position.x=5.2;msg.info.pose.orientation.w=1.
    for name,values in fields.items():
        msg.layers.append(name);msg.has_so2.append(len(values)>1)
        msg.data.extend(matrix_to_multiarray_like(world_to_grid_buffer(v,msg),None) for v in values)
    return msg


def refinement_surface(measured, prediction, enabled=False, noise_variance=.0004):
    if not enabled:
        return measured.copy()
    valid = np.isfinite(measured) & prediction.accepted.reshape(measured.shape)
    weight = noise_variance/(noise_variance+np.maximum(prediction.latent_var.reshape(measured.shape), 1e-6))
    return np.where(valid, (1-weight)*measured+weight*prediction.mean.reshape(measured.shape), measured)


def prepare(args):
    import torch
    import cupy as cp
    from rclpy.serialization import serialize_message
    torch.set_num_threads(1)
    output=args.output;output.mkdir(parents=True,exist_ok=True);(output/"maps").mkdir(exist_ok=True)
    path=output/"cases.json"
    if path.exists():raise ValueError("preserve earlier maps; use a new preparation output")
    params=params_from_config("cupy",.2,36);estimator=BgkSebEstimator(params)
    gp_params=FeatureSgpParams(device="cuda")
    refine_observed=bool(getattr(args,"gp_refine_observed",False))
    parameters=yaml.safe_load((ROOT/"src/planner/terrain_nav/config/hunter_gazebo.yaml").read_text())["terrain_nav_node"]["ros__parameters"]
    parameters.update(risk_query_mode="se2_yaw",se2_risk_lethal=True,use_sim_time=True,
        cmd_topic="/sandbox/cmd_vel",odom_topic="/sandbox/odom",fused_map_topic="/sandbox/map",
        live_cloud_topic="/sandbox/cloud",use_mppi=args.mppi)
    binary=args.binary or ROOT/"install/terrain_nav/lib/terrain_nav/terrain_nav_benchmark"
    compiled=verify_native_build(binary)
    settings={"seed":args.seed,"per_scenario":args.per_scenario,"scenarios":args.scenarios,"mppi":args.mppi,
        "map_refresh_hz":2.0,"live_cloud_hz":10.0,"plant_hz":50.0,
        "map_shape":[104,104],"resolution_m":.2,"yaw_bins":36,"gp":asdict(gp_params),"bgk":asdict(params),
        "gp_refine_observed":refine_observed,"unknown_completion":False,
        "risk_input":"accepted_observed_refinement" if refine_observed else "measured_only",
        "source_sha256":{name:digest(ROOT/name) for name in (
            "src/fsgp_bgk_se2/fsgp_bgk_se2/feature_sgp.py","src/fsgp_bgk_se2/fsgp_bgk_se2/bgk_seb.py",
            "src/fsgp_bgk_se2/fsgp_bgk_se2/supported_completion.py",
            "src/fsgp_bgk_se2/fsgp_bgk_se2/obstacle_columns.py","src/fsgp_bgk_se2/fsgp_bgk_se2/se2_grid_tools.py",
            "src/uneven_mapping/se2_grid/se2_grid_core/src/SE2Grid.cpp",
            "src/uneven_mapping/se2_grid/se2_grid_core/include/se2_grid_core/SE2Grid.hpp",
            "src/uneven_mapping/se2_grid/se2_grid_ros/src/SE2GridRosConverter.cpp",
            "scripts/sim/experiments/navigation_sandbox.py","src/planner/terrain_nav/config/hunter_gazebo.yaml",
            "src/planner/terrain_nav/src/terrain_nav_node.cpp","src/planner/terrain_nav/test/navigation_benchmark.cpp")},
        "binary_sha256":digest(binary),"scope":__doc__,"compiled_source_sha256":compiled}
    cases=[];xy=grid((104,104),[-5.2,-10.4],.2)
    for repeat in range(args.per_scenario):
        for name in args.scenarios:
            si=SCENARIOS.index(name)
            case=scenario(name,args.seed+repeat*100+si)
            height=elevation(case["terrain"],xy);valid=np.ones(len(xy),bool)
            if name=="unknown":valid &= xy[:,0]<case["unknown_edge"]-1e-9
            rng=np.random.default_rng(case["seed"]+100000)
            observed=np.c_[xy[valid],height[valid]+rng.normal(0,.008,valid.sum())].astype(np.float32)
            prediction=FeatureSgpPredictor(gp_params).fit(observed).predict(xy)
            raw=observed.copy();extras=[]
            for b in case["obstacles"]:
                if b.get("appear_s",0)>0:continue
                xx,yy=np.meshgrid(np.arange(b["x"]-b["hx"],b["x"]+b["hx"]+.02,.12),
                    np.arange(b["y"]-b["hy"],b["y"]+b["hy"]+.02,.12))
                ground=elevation(case["terrain"],np.c_[xx.ravel(),yy.ravel()])
                extras.extend([np.c_[xx.ravel(),yy.ravel(),ground+z] for z in (.45,.95)])
            if extras:raw=np.concatenate([raw,*extras]).astype(np.float32)
            measured=np.full(len(xy),np.nan,np.float32);measured[valid]=observed[:,2]
            measured=measured.reshape(104,104);valid=valid.reshape(104,104)
            refined=refinement_surface(measured,prediction,refine_observed)
            occupied,_=vertical_columns(raw,(104,104),[-5.2,-10.4],.2,ground_elevation=measured,xp=cp)
            occupied=cp.asnumpy(occupied)
            result=estimator.compute(refined,np.full((104,104),.0004,np.float32),valid,
                [occupied.astype(np.float32)],observed_elevation=measured)
            preserve_collision_risk(result,occupied)
            if prediction.device!="cuda" or result.backend!="cupy":raise RuntimeError("required GPU path fell back")
            message=make_message({"elevation":[refined],"valid":[valid.astype(np.float32)],
                "baseline_risk":[occupied.astype(np.float32)],"collision_risk":[occupied.astype(np.float32)],
                "risk":result.final_risk_slices})
            target=output/"maps"/(case["id"]+".cdr");target.write_bytes(serialize_message(message))
            case.update(map_file=str(target.relative_to(output)),map_sha256=digest(target),
                gp_fit_ms=prediction.fit_ms,gp_predict_ms=prediction.predict_ms,actual_device=prediction.device,
                actual_backend=result.backend)
            cases.append(case)
        if (repeat+1)%5==0:print(f"[sandbox maps] {len(cases)} generated",flush=True)
    atomic_json(path,{"settings":settings,"parameters":parameters,"cases":cases})
    print(f"[sandbox maps] frozen {len(cases)} cases: {path}",flush=True)


def run(args):
    path=args.output/"cases.json";plan=json.loads(path.read_text())
    binary=args.binary or ROOT/"install/terrain_nav/lib/terrain_nav/terrain_nav_benchmark"
    if digest(binary)!=plan["settings"]["binary_sha256"]:raise ValueError("binary changed; prepare a new frozen batch")
    for case in plan["cases"]:
        if digest(args.output/case["map_file"])!=case["map_sha256"]:raise ValueError("CDR input changed")
    result=subprocess.run([str(binary),str(path),str(args.output/"runs")],
        env={**os.environ,"ROS_DOMAIN_ID":str(args.domain),"RMW_IMPLEMENTATION":"rmw_fastrtps_cpp"})
    return result.returncode


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("phase",choices=("prepare","run"))
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--seed",type=int,default=103261013)
    parser.add_argument("--per-scenario",type=int,default=100);parser.add_argument("--domain",type=int,default=89)
    parser.add_argument("--mppi",action="store_true")
    parser.add_argument("--gp-refine-observed",action="store_true",
        help="opt-in variance-weighted refinement; default matches measured-only deployment")
    parser.add_argument("--binary",type=Path,help="explicit frozen native binary for strict earlier-revision resume")
    parser.add_argument("--scenarios",type=lambda s:s.split(','),default=list(SCENARIOS))
    args=parser.parse_args()
    if args.per_scenario<1 or not 1<=args.domain<=100:parser.error("positive repetitions and isolated domain required")
    if len(args.scenarios)!=len(set(args.scenarios)) or any(s not in SCENARIOS for s in args.scenarios):
        parser.error("unique supported scenario names required")
    if args.phase=="prepare":prepare(args);return 0
    return run(args)


if __name__=="__main__":raise SystemExit(main())
