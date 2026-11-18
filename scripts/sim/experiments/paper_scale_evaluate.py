#!/usr/bin/env python3
"""Our GP/BGK/SEB on every replacement cloud; independent footprint proxy.

Reports dimensionless risk errors separately from meter-valued height errors.
This is a local component study, not the paper's missing truth formula,
temporal BGK reproduction, automatic ground segmentation, or MINCO trials.
"""

import argparse
from collections import Counter
from dataclasses import asdict
import json
import importlib.util
from pathlib import Path
import sys
import time

import numpy as np
from scipy.ndimage import map_coordinates

from pointcloud_batch import ROOT, atomic_json, digest
from paper_clouds import grid, height_at, raster
from terrain_risk_benchmark import params_from_config
from fsgp_bgk_se2.feature_sgp import FeatureSgpParams, FeatureSgpPredictor
from fsgp_bgk_se2.bgk_seb import BgkSebEstimator
from fsgp_bgk_se2.obstacle_columns import vertical_columns, preserve_collision_risk
from fsgp_bgk_se2.supported_completion import supported_completion

CONDITIONS=("point15","hole16","frontier50")
STRESS_CONDITIONS=CONDITIONS+("point30","point60","hole08","hole32","frontier25")


def complete_generated_collision(case, item):
    """Include the actual exported canopy ellipsoids at bumper height on slopes."""
    field=case["truth_collision"].copy()
    if not item["source_kind"].startswith("epfl"):
        return field
    resolution=float(case["truth_resolution"]); origin=case["truth_origin"]
    for entity in item.get("entities",[]):
        if entity["type"]!="tree":
            continue
        center=np.asarray(entity["center"])
        lo=np.maximum(0,np.floor((center-.6-origin)/resolution).astype(int))
        hi=np.minimum(field.shape[::-1],np.ceil((center+.6-origin)/resolution).astype(int))
        if (hi<=lo).any():
            continue
        xy=grid((hi[1]-lo[1],hi[0]-lo[0]),origin+lo*resolution,resolution)
        radial=np.sum(((xy-center)/.6)**2,axis=1)
        half_height=.4*np.sqrt(np.maximum(0,1-radial))
        canopy_z=height_at(case["truth_height"],origin,resolution,center[None])[0]+entity["height"]
        ground=case["truth_height"][lo[1]:hi[1],lo[0]:hi[0]].ravel()
        occupied=(radial<=1)&(canopy_z-half_height<=ground+1.5)&(canopy_z+half_height>=ground+.32)
        field[lo[1]:hi[1],lo[0]:hi[0]] |= occupied.reshape(hi[1]-lo[1],hi[0]-lo[0])
    return field


def partial_points(xyz, kind, origin, seed):
    rng=np.random.default_rng(seed)
    if kind in ("point15","point30","point60"):
        return rng.random(len(xyz))>=int(kind[5:])/100
    center=np.asarray(origin)+10.4
    if kind in ("hole08","hole16","hole32"):
        return (np.abs(xyz[:,:2]-center)>=int(kind[4:])/20).any(axis=1)
    if kind in ("frontier25","frontier50"):
        return xyz[:,0]<=np.asarray(origin)[0]+20.8*(1-int(kind[8:])/100)
    raise ValueError("unknown missingness condition")


def footprint_reference(case, xy, params, seed, samples=128):
    heights=height_at(case["truth_height"],case["truth_origin"],float(case["truth_resolution"]),xy)
    shape=tuple(case["shape"])
    rr,cc=np.indices(shape)
    candidates=np.flatnonzero(np.isfinite(heights)&(rr.ravel()>=6)&(rr.ravel()<shape[0]-6)&
        (cc.ravel()>=6)&(cc.ravel()<shape[1]-6))
    if len(candidates)<8:
        raise ValueError("insufficient reference center locations")
    ids=np.sort(np.random.default_rng(seed).choice(candidates,min(samples,len(candidates)),replace=False))
    lx,ly=np.meshgrid(np.linspace(-params.footprint_back-params.footprint_margin,
        params.footprint_front+params.footprint_margin,13),np.linspace(-params.footprint_half_width-params.footprint_margin,
        params.footprint_half_width+params.footprint_margin,11))
    lx,ly=lx.ravel(),ly.ravel(); design=np.c_[lx,ly,np.ones_like(lx)]; inverse=np.linalg.pinv(design)
    scores=[]; hazards=[]; support=[]
    for yaw in (np.arange(params.yaw_bins)+.5)*2*np.pi/params.yaw_bins:
        dx=lx*np.cos(yaw)-ly*np.sin(yaw); dy=lx*np.sin(yaw)+ly*np.cos(yaw)
        queries=xy[ids,None,:]+np.stack((dx,dy),axis=1)[None]
        z=height_at(case["truth_height"],case["truth_origin"],float(case["truth_resolution"]),queries.reshape(-1,2)).reshape(len(ids),-1)
        supported=np.isfinite(z).all(axis=1)
        coefficients=z@inverse.T; residual=z-coefficients@design.T
        pitch=np.arctan(np.abs(coefficients[:,0])); roll=np.arctan(np.abs(coefficients[:,1])); span=np.ptp(residual,axis=1)
        score=np.maximum.reduce((np.clip((pitch-params.pitch_free_rad)/(params.max_pitch_rad-params.pitch_free_rad),0,1),
            np.clip((roll-params.roll_free_rad)/(params.max_roll_rad-params.roll_free_rad),0,1),
            np.clip((span-params.step_free_m)/(params.max_step_m-params.step_free_m),0,1)))
        indices=(queries-case["truth_origin"])/float(case["truth_resolution"])-.5
        occupancy=map_coordinates(case["truth_collision"].astype(np.float32),
            [indices[...,1].ravel(),indices[...,0].ravel()],order=0,mode="constant",cval=0).reshape(len(ids),-1).max(axis=1)>0
        score=np.where(occupancy,1,score)
        unsafe=occupancy|(pitch>=params.max_pitch_rad)|(roll>=params.max_roll_rad)|(span>=params.max_step_m)
        scores.append(score); hazards.append(unsafe); support.append(supported)
    return heights,ids,np.asarray(scores),np.asarray(hazards),np.asarray(support)


def height_metrics(prediction, reference, observed):
    valid=np.isfinite(reference); hidden=valid&~observed
    result={"reference_cells":int(valid.sum()),"hidden_cells":int(hidden.sum())}
    for label,mask in (("all_reference",valid),("hidden",hidden),("accepted_hidden",hidden&prediction.accepted)):
        error=prediction.mean[mask]-reference[mask]
        result[label+"_rmse_m"]=float(np.sqrt(np.mean(error**2))) if len(error) else None
        result[label+"_mae_m"]=float(np.mean(np.abs(error))) if len(error) else None
    if hidden.any():
        std=np.sqrt(prediction.latent_var[hidden]); error=np.abs(prediction.mean[hidden]-reference[hidden])
        result.update(hidden_picp95=float(np.mean(error<=1.96*std)),hidden_width95_m=float(np.mean(3.92*std)),
            hidden_candidate_coverage=float(prediction.accepted[hidden].mean()))
    else:
        result.update(hidden_picp95=None,hidden_width95_m=None,hidden_candidate_coverage=None)
    return result


def risk_metrics(predicted, reference, unsafe, supported):
    if not supported.any():
        return {"reference_states":0,"risk_mae":None,"risk_error_variance":None,"risk_rmse":None,
                "tp":0,"fn":0,"fp":0,"tn":0}
    error=predicted[supported]-reference[supported]
    hazard=predicted>=.58
    return {"reference_states":int(supported.sum()),"risk_mae":float(np.mean(np.abs(error))),
        "risk_rmse":float(np.sqrt(np.mean(error**2))),"risk_error_variance":float(np.var(error)),
        "tp":int((hazard&unsafe&supported).sum()),"fn":int((~hazard&unsafe&supported).sum()),
        "fp":int((hazard&~unsafe&supported).sum()),"tn":int((~hazard&~unsafe&supported).sum())}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,required=True); parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--conditions",default=",".join(CONDITIONS))
    parser.add_argument("--seed",type=int,default=51261013); parser.add_argument("--resume",action="store_true")
    parser.add_argument("--max-per-family",type=int,help="explicit small preflight only; not the full campaign")
    parser.add_argument("--core-source",type=Path,help="explicit frozen local source for paired development regression")
    parser.add_argument("--noise-std",type=float,default=.02,help="actual input Z noise; method noise model stays frozen")
    parser.add_argument("--complete-supported",action="store_true",help="bounded local interpolation; paired observed-only risk from identical observations")
    args=parser.parse_args()
    import torch
    import cupy as cp
    torch.set_num_threads(1)
    conditions=args.conditions.split(",")
    if any(c not in STRESS_CONDITIONS for c in conditions) or args.noise_std<0:
        parser.error("unsupported conditions")
    gp_class,gp_parameters,estimator_class=FeatureSgpPredictor,FeatureSgpParams,BgkSebEstimator
    core=args.core_source or ROOT/"src/fsgp_bgk_se2/fsgp_bgk_se2"
    if args.core_source:
        modules={}
        for name in ("feature_sgp","bgk_seb"):
            spec=importlib.util.spec_from_file_location("frozen_campaign_"+name,core/(name+".py"))
            module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
            modules[name]=module
        gp_class,gp_parameters=modules["feature_sgp"].FeatureSgpPredictor,modules["feature_sgp"].FeatureSgpParams
        estimator_class=modules["bgk_seb"].BgkSebEstimator
    catalog=json.loads(args.dataset.read_text()); datasets=catalog["datasets"]
    if args.max_per_family is not None:
        if args.max_per_family<1:
            parser.error("positive preflight count required")
        counts=Counter(); selected=[]
        for item in datasets:
            if counts[item["family"]]<args.max_per_family:
                selected.append(item);counts[item["family"]]+=1
        datasets=selected
    params=params_from_config("cupy",.2,36); gp_params=gp_parameters(device="cuda")
    if args.core_source:
        old_params=modules["bgk_seb"].BgkSebParams
        params=old_params(**{k:v for k,v in asdict(params).items() if k in old_params.__dataclass_fields__})
    estimator=estimator_class(params)
    settings={"dataset_sha256":digest(args.dataset),"script_sha256":digest(__file__),"conditions":conditions,"seed":args.seed,
        "reference_sampling_version":"finite_weight_v2",
        "reference_source_sha256":digest(Path(__file__).with_name("paper_clouds.py")),
        "selected_ids":[item["id"] for item in datasets],
        "sgp":asdict(gp_params),"bgk_seb":asdict(params),"input_noise_std_m":args.noise_std,
        "refinement":"accepted observed cells only, as the ROS gp_refine_observed mode",
        "core_sha256":{p:digest(core/p) for p in ("feature_sgp.py","bgk_seb.py","obstacle_columns.py")}}
    settings["complete_supported"]=args.complete_supported
    if args.complete_supported:settings["completion_sha256"]=digest(ROOT/"src/fsgp_bgk_se2/fsgp_bgk_se2/supported_completion.py")
    args.output.mkdir(parents=True,exist_ok=True); path=args.output/"results.json"
    record=json.loads(path.read_text()) if path.exists() and args.resume else {"date":"2026-10-13","settings":settings,
        "scope":__doc__,"planned_cases":len(datasets)*len(conditions),"source_cloud_count":len(datasets),"runs":[]}
    if path.exists() and not args.resume or record["settings"]!=settings:
        raise ValueError("new output or strict unchanged resume required")
    completed={r["id"] for r in record["runs"]}
    if args.resume:
        for di,item in enumerate(datasets):
            for ci,kind in enumerate(conditions):
                name=item["id"]+"__"+kind; checkpoint=args.output/(name+".json")
                if name not in completed and checkpoint.exists():
                    row=json.loads(checkpoint.read_text())
                    if row["dataset_id"]!=item["id"] or row["case_seed"]!=args.seed+di*10+ci:
                        raise ValueError("checkpoint does not match the frozen case plan")
                    if row["status"]=="ok" and not checkpoint.with_suffix(".npz").exists():
                        raise ValueError("successful checkpoint has no recorded prediction")
                    record["runs"].append(row);completed.add(name)
    for di,item in enumerate(datasets):
        source=args.dataset.parent/item["path"]
        if digest(source)!=item["sha256"]:
            raise ValueError("input archive hash mismatch")
        with np.load(source,allow_pickle=False) as arrays:
            case={key:arrays[key] for key in arrays.files if key!="full_xyz"}
        case["truth_collision"]=complete_generated_collision(case,item)
        xy=grid(tuple(case["shape"]),case["query_origin"],.2)
        try:
            reference,ids,oracle,unsafe,supported=footprint_reference(case,xy,params,args.seed+di)
        except ValueError as exc:
            for kind in conditions:
                name=item["id"]+"__"+kind
                if name not in completed:
                    record["runs"].append({"id":name,"dataset_id":item["id"],"family":item["family"],
                        "cluster_id":item["cluster_id"],"reference_kind":item["reference_kind"],"condition":kind,
                        "status":"not_evaluable","reason":"reference geometry: "+str(exc)})
            atomic_json(path,record)
            continue
        for ci,kind in enumerate(conditions):
            name=item["id"]+"__"+kind
            if name in completed:
                continue
            row={"id":name,"dataset_id":item["id"],"family":item["family"],"cluster_id":item["cluster_id"],
                 "reference_kind":item["reference_kind"],"condition":kind,"case_seed":args.seed+di*10+ci}
            try:
                keep=partial_points(case["xyz"],kind,case["query_origin"],row["case_seed"])
                raw=case["xyz"][keep].copy(); raw[:,2]+=np.random.default_rng(row["case_seed"]+100000).normal(0,args.noise_std,len(raw))
                observed=raw[case["is_ground"][keep]]
                predictor=gp_class(gp_params).fit(observed); prediction=predictor.predict(xy)
                elevation=raster(observed,tuple(case["shape"]),case["query_origin"],.2)
                valid=np.isfinite(elevation); variance=np.full_like(elevation,.0004)
                refine=valid&prediction.accepted.reshape(elevation.shape)
                weight=variance/(variance+prediction.latent_var.reshape(elevation.shape))
                refined=np.where(refine,(1-weight)*elevation+weight*prediction.mean.reshape(elevation.shape),elevation)
                cp.cuda.Stream.null.synchronize(); started=time.perf_counter()
                occupied,_=vertical_columns(raw,elevation.shape,case["query_origin"],.2,ground_elevation=elevation,xp=cp)
                occupied=cp.asnumpy(occupied)
                observed_valid=valid.copy();completed=np.zeros_like(valid)
                observed_refined=refined.copy();observed_variance=variance.copy()
                if args.complete_supported:
                    refined,variance,valid,completed=supported_completion(
                        refined,variance,valid,prediction,blocked=occupied)
                result=estimator.compute(refined,variance,valid,[occupied.astype(np.float32)],observed_elevation=elevation)
                preserve_collision_risk(result,occupied); cp.cuda.Stream.null.synchronize()
                risk_ms=(time.perf_counter()-started)*1000
                scores=np.stack(result.final_risk_slices).reshape(36,-1)[:,ids]
                row.update(status="ok",actual_device=prediction.device,actual_backend=result.backend,
                    input_points=len(raw),ground_input_points=len(observed),fit_ms=prediction.fit_ms,predict_ms=prediction.predict_ms,
                    risk_ms=risk_ms,completed_cells=int(completed.sum()),
                    completion_hidden_fraction=float(completed.sum()/max(1,(~observed_valid).sum())),
                    **height_metrics(prediction,reference,observed_valid.ravel()),**risk_metrics(scores,oracle,unsafe,supported))
                extra={}
                if args.complete_supported:
                    baseline=estimator.compute(observed_refined,observed_variance,observed_valid,
                        [occupied.astype(np.float32)],observed_elevation=elevation)
                    preserve_collision_risk(baseline,occupied)
                    baseline_scores=np.stack(baseline.final_risk_slices).reshape(36,-1)[:,ids]
                    row.update({"observed_only_"+k:v for k,v in risk_metrics(baseline_scores,oracle,unsafe,supported).items()})
                    hidden_reference=completed.ravel()&np.isfinite(reference)
                    error=prediction.mean[hidden_reference]-reference[hidden_reference]
                    row["completed_height_rmse_m"]=float(np.sqrt(np.mean(error**2))) if len(error) else None
                    extra["observed_only_risk"]=baseline_scores
                np.savez_compressed(args.output/(name+".npz"),predicted_risk=scores,reference_risk=oracle,unsafe=unsafe,supported=supported,
                    query_indices=ids,query_xy=xy[ids],height_mean=prediction.mean,latent_var=prediction.latent_var,
                    observed_cells=observed_valid,predicted_cells=completed,reference_height=reference.reshape(elevation.shape),**extra)
                snapshot=args.output/(item["family"]+"__"+kind+"__snapshot.npz")
                if not snapshot.exists():
                    np.savez_compressed(snapshot,case_id=name,dataset_id=item["id"],raw_xyz=raw,elevation=elevation,
                        mean=prediction.mean.reshape(elevation.shape),std=np.sqrt(prediction.latent_var).reshape(elevation.shape),
                        risk=result.final_risk_slices[0],occupied=occupied,query_origin=case["query_origin"],reference=reference.reshape(elevation.shape))
            except ValueError as exc:
                row.update(status="not_evaluable" if "eight" in str(exc) else "failed",reason=str(exc))
            except Exception as exc:
                row.update(status="failed",reason=f"{type(exc).__name__}: {exc}")
            atomic_json(args.output/(name+".json"),row);record["runs"].append(row)
        if (di+1)%20==0 or di+1==len(datasets):
            atomic_json(path,record); print(f"[paper scale] {di+1}/{len(datasets)} clouds; {dict(Counter(r['status'] for r in record['runs']))}",flush=True)
    atomic_json(path,record)
    print(f"[paper scale] finished {len(record['runs'])} cases; {path}",flush=True)
    return int(any(r["status"]=="failed" for r in record["runs"]))


if __name__=="__main__":
    raise SystemExit(main())
