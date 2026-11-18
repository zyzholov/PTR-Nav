"""Small, versioned correction checks: separate data-profile changes from method changes."""

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
import os
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import ConvexHull
import torch

from pointcloud_batch import ROOT, atomic_json, digest
from paper_clouds import grid, height_at, raster
from paper_scale_evaluate import partial_points, footprint_reference, complete_generated_collision, height_metrics, risk_metrics
from diagnose_terrain_prediction import BASE, OUTPUT, FAMILIES, errors
from probe_terrain_predictor import load_baseline
from terrain_risk_benchmark import params_from_config
from fsgp_bgk_se2.feature_sgp import FeatureSgpParams, FeatureSgpPredictor
from fsgp_bgk_se2.bgk_seb import BgkSebEstimator
from fsgp_bgk_se2.obstacle_columns import vertical_columns, preserve_collision_risk

CONDITIONS=("point15","hole16","frontier50")


def load_case(catalog_path,item):
    path=(catalog_path.parent/item["path"]).resolve()
    if digest(path)!=item["sha256"]:raise ValueError("source archive changed")
    with np.load(path) as a:case={k:a[k] for k in a.files if k!="full_xyz"}
    return case


def representation(case):
    xyz=case["xyz"][case["is_ground"]]
    xy=grid(tuple(case["shape"]),case["query_origin"],.2)
    ref=height_at(case["truth_height"],case["truth_origin"],float(case["truth_resolution"]),xy)
    native=height_at(case["truth_height"],case["truth_origin"],float(case["truth_resolution"]),xyz[:,:2])
    return {"point_to_source":errors(xyz[:,2],native,np.ones(len(xyz),bool)),
        "input_raster_to_reference":errors(raster(xyz,tuple(case["shape"]),case["query_origin"],.2).ravel(),ref,np.isfinite(ref))}


def source_audit():
    catalog_path=BASE/"catalog.json";catalog=json.loads(catalog_path.read_text());rows=[]
    for item in catalog["datasets"]:
        if not item["source_kind"].startswith("epfl"):continue
        case=load_case(catalog_path,item);z=case["truth_height"];xy=grid(tuple(case["shape"]),case["query_origin"],.2)
        ref=height_at(z,case["truth_origin"],.1,xy)
        sampled=RegularGridInterpolator((np.arange(0,500,2),np.arange(0,500,2)),z[::2,::2],bounds_error=False)
        linear=sampled(((xy-case["truth_origin"])/.1-.5)[:,::-1])
        slope=np.hypot(*np.gradient(z,.1))
        rows.append({"id":item["id"],"family":item["family"],"geometry_sha256":item["geometry_sha256"],
            "shader_type":item["shader_parameters"]["type"],"shader_H":item["shader_parameters"]["H"],
            "height_span_m":float(np.ptp(z)),"gradient_p95":float(np.quantile(slope,.95)),
            "complete_input_interpolation":errors(linear,ref,np.isfinite(ref))})
    atomic_json(OUTPUT/"all_generated_source_audit.json",{"scope":"representation stress audit; no cases excluded", "runs":rows})


def prepare_regression(generated_path,output,road_path):
    if output.exists():raise ValueError("preserve frozen regression plan")
    new=json.loads(generated_path.read_text());public_path=BASE/"catalog.json";public=json.loads(public_path.read_text())
    if len(new["datasets"])!=60 or any(d.get("generator_profile")!="metric_v2" for d in new["datasets"]):
        raise ValueError("expected 60 fixed-profile geometries")
    previous_geometry={d["geometry_sha256"] for d in public["datasets"] if d["source_kind"].startswith("epfl")}
    if previous_geometry & {d["geometry_sha256"] for d in new["datasets"]}:
        raise ValueError("validation geometry overlaps earlier campaign")
    audit=json.loads((OUTPUT/"source_and_branch_audit.json").read_text())
    used={r["cluster_id"] for r in audit["runs"]}
    if road_path is None:raise ValueError("explicit corrected road catalog required")
    road=json.loads(road_path.read_text());road_items={d["id"]:d for d in road["datasets"]}
    rows=[]
    for item in new["datasets"]:
        copy=dict(item);copy["path"]=os.path.relpath((generated_path.parent/item["path"]).resolve(),output.parent.resolve());rows.append(copy)
    for f,n in (("road",9),("indoor",11)):
        groups=defaultdict(list)
        for item in public["datasets"]:
            if item["family"]==f and item["cluster_id"] not in used:groups[item["cluster_id"]].append(item)
        order=[item for i in range(max(map(len,groups.values()))) for g in sorted(groups)
            for item in sorted(groups[g],key=lambda x:x["id"])[i:i+1]]
        if len(order)<n:raise ValueError("insufficient source-disjoint regression inputs")
        for item in order[:n]:
            source_path=public_path
            if f=="road":
                item=road_items[item["id"]];source_path=road_path
                if item.get("registration_pose_source")!="semantic_suma":raise ValueError("road correction requires explicit SuMa provenance")
            copy=dict(item);copy["path"]=os.path.relpath((source_path.parent/item["path"]).resolve(),output.parent.resolve());rows.append(copy)
    if len(rows)!=80:raise ValueError("expected 60 generated and 20 source-disjoint public inputs")
    atomic_json(output,{"datasets":rows,"scope":"targeted source-disjoint correction validation, not a formal comparison",
        "excluded_development_clusters":sorted(used),"source_sha256":{str(p):digest(p) for p in (generated_path,public_path,road_path)},
        "planned_computations":len(rows)*3*2})


def run(catalog_path,output,source_pair=False,protocol_path=None):
    import cupy as cp
    if output.exists():raise ValueError("preserve earlier correction results")
    output.mkdir(parents=True)
    baseline=load_baseline();torch.set_num_threads(1)
    catalog=json.loads(catalog_path.read_text());rows=[]
    params=params_from_config("cupy",.2,36);estimator=BgkSebEstimator(params)
    old_catalog_path=BASE/"catalog.json";old_catalog=json.loads(old_catalog_path.read_text())
    old_items={d["id"]:d for d in old_catalog["datasets"]}
    seeds={(r["dataset_id"],r["condition"]):r["case_seed"] for r in json.loads((BASE/"budget/results.json").read_text())["runs"]}
    settings={"catalog_sha256":digest(catalog_path),"baseline_sha256":digest(OUTPUT/"frozen_before/feature_sgp.py"),
        "fixed_sha256":digest(ROOT/"src/fsgp_bgk_se2/fsgp_bgk_se2/feature_sgp.py"),
        "generator_sha256":digest(ROOT/"scripts/sim/experiments/paper_clouds.py"),
        "runner_sha256":digest(Path(__file__)),"seed":261015977,"source_pair":source_pair,
        "protocol_sha256":digest(protocol_path or ROOT/"scripts/sim/experiments/protocols/terrain_correction_v2.json"),
        "scope":"data change uses frozen baseline on both profiles; method change uses identical fixed-profile observations",
        "gp":asdict(FeatureSgpParams(device="cuda",max_train_points=512,inducing_points=125))}
    for di,item in enumerate(catalog["datasets"]):
        current=load_case(catalog_path,item)
        variants=[("legacy_data",load_case(old_catalog_path,old_items[item["id"]]),baseline,old_items[item["id"]]),
                  ("fixed_data",current,baseline,item)] if source_pair else [("before",current,baseline,item),("fixed",current,None,item)]
        for variant,case,module,source_item in variants:
            xy=grid(tuple(case["shape"]),case["query_origin"],.2);shape=tuple(case["shape"])
            case["truth_collision"]=complete_generated_collision(case,source_item)
            reference,indices,oracle,unsafe,supported=footprint_reference(case,xy,params,settings["seed"]+di)
            rep=representation(case)
            for ci,condition in enumerate(CONDITIONS):
                seed=seeds[item["id"],condition] if source_pair else settings["seed"]+di*100+ci
                keep=partial_points(case["xyz"],condition,case["query_origin"],seed)
                raw=case["xyz"][keep].copy();raw[:,2]+=np.random.default_rng(seed+100000).normal(0,.02,len(raw))
                ground=raw[case["is_ground"][keep]];measured=raster(ground,shape,case["query_origin"],.2)
                observed=np.isfinite(measured);variance=np.full_like(measured,.0004)
                torch.manual_seed(seed)
                args=FeatureSgpParams(**settings["gp"]) if module is None else module.FeatureSgpParams(**settings["gp"])
                predictor=FeatureSgpPredictor(args) if module is None else module.FeatureSgpPredictor(args)
                prediction=predictor.fit(ground).predict(xy)
                origin=ground[:,:2].mean(axis=0,dtype=np.float64);hull=ConvexHull(ground[:,:2].astype(np.float64)-origin).equations
                outside=np.any((xy-origin)@hull[:,:2].T+hull[:,2]>1e-6,axis=1)
                hidden=np.isfinite(reference)&~observed.ravel()
                refined=np.where(observed&prediction.accepted.reshape(shape),measured+
                    variance/(variance+prediction.latent_var.reshape(shape))*(prediction.mean.reshape(shape)-measured),measured)
                occupied,_=vertical_columns(raw,shape,case["query_origin"],.2,ground_elevation=measured,xp=cp)
                occupied=cp.asnumpy(occupied)
                risk=estimator.compute(refined,variance,observed,[occupied.astype(np.float32)],observed_elevation=measured)
                preserve_collision_risk(risk,occupied)
                scores=np.stack(risk.final_risk_slices).reshape(36,-1)[:,indices]
                row={"id":item["id"],"family":item["family"],"cluster_id":item["cluster_id"],"variant":variant,"condition":condition,
                    "case_seed":seed,"actual_device":prediction.device,"actual_backend":risk.backend,
                    "fit_ms":prediction.fit_ms,"predict_ms":prediction.predict_ms,"source_sha256":source_item["sha256"],
                    "outside_hull_queries":int(outside.sum()),"outside_hull_accepted":int((outside&prediction.accepted).sum()),
                    "outside_hull_local":int((outside&prediction.local_conditioned).sum()),
                    "accepted_hidden":errors(prediction.mean,reference,hidden&prediction.accepted),
                    "supported_hidden":errors(prediction.mean,reference,hidden&prediction.supported),
                    "unsupported_hidden":errors(prediction.mean,reference,hidden&~prediction.supported),
                    "representation":rep,**height_metrics(prediction,reference,observed.ravel()),
                    **risk_metrics(scores,oracle,unsafe,supported)}
                if prediction.device!="cuda" or risk.backend!="cupy":raise ValueError("GPU fallback")
                archive=output/(item["id"]+"__"+variant+"__"+condition+".npz")
                np.savez_compressed(archive,mean=prediction.mean,variance=prediction.latent_var,accepted=prediction.accepted,
                    local=prediction.local_conditioned,supported=prediction.supported,observed=observed,reference=reference,
                    xy=xy,raw_xyz=raw,reference_risk=oracle,predicted_risk=scores,risk_supported=supported,unsafe=unsafe)
                row.update(archive=archive.name,archive_sha256=digest(archive));rows.append(row)
        atomic_json(output/"results.json",{"settings":settings,"runs":rows})
        print("[correction]",di+1,"/",len(catalog["datasets"]),item["id"],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("phase",choices=("source-audit","prepare","source-pair","regression"))
    parser.add_argument("--catalog",type=Path);parser.add_argument("--generated",type=Path)
    parser.add_argument("--road-catalog",type=Path)
    parser.add_argument("--protocol",type=Path)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if args.phase=="source-audit":source_audit()
    elif args.phase=="prepare":prepare_regression(args.generated,args.output,args.road_catalog)
    else:run(args.catalog,args.output,args.phase=="source-pair",args.protocol)


if __name__=="__main__":main()
