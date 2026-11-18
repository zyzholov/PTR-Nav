"""Controlled GP-to-risk connection checks, using fixed selections from campaign v4."""

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from comprehensive_study import BASE
from pointcloud_batch import ROOT,atomic_json,digest
from paper_scale_evaluate import partial_points
from terrain_system_study import domain_metrics,risk_metrics,evaluation_domains
from terrain_risk_benchmark import params_from_config
from navigation_sandbox import refinement_surface
from controlled_terrain_study import environment
from fsgp_bgk_se2.feature_sgp import FeatureSgpParams,FeatureSgpPredictor
from fsgp_bgk_se2.bgk_seb import BgkSebEstimator
from fsgp_bgk_se2.supported_completion import supported_completion
from fsgp_bgk_se2.obstacle_columns import preserve_collision_risk

PROTOCOL=ROOT/"scripts/sim/experiments/protocols/joint_integration_v1.json"


def selection(datasets,count):
    selected=[]
    for family in ("hilly","forest","ruin","road","indoor"):
        groups=defaultdict(list)
        for d in datasets:
            if d["family"]==family:groups[d["cluster_id"]].append(d)
        for values in groups.values():values.sort(key=lambda d:d["id"])
        ordered=[d for i in range(max(map(len,groups.values()))) for g in sorted(groups) for d in groups[g][i:i+1]]
        if len(ordered)<count:raise ValueError("not enough source-balanced inputs")
        selected.extend(ordered[:count])
    return selected


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--source",type=Path,default=BASE)
    parser.add_argument("--output",type=Path,default=BASE/"integration");parser.add_argument("--resume",action="store_true")
    args=parser.parse_args();p=json.loads(PROTOCOL.read_text());torch.set_num_threads(1)
    import cupy as cp
    source=json.loads((args.source/"public/results.json").read_text())
    catalog=json.loads((args.source/"catalog.json").read_text());items=selection(catalog["datasets"],p["inputs_per_family"])
    lookup={(r["dataset_id"],r["condition"]):r for r in source["runs"]}
    paths=[Path(__file__),PROTOCOL,*[ROOT/"src/fsgp_bgk_se2/fsgp_bgk_se2"/n for n in ("feature_sgp.py","bgk_seb.py","supported_completion.py","obstacle_columns.py")]]
    settings=dict(protocol_sha256=digest(PROTOCOL),source_results_sha256=digest(args.source/"public/results.json"),
        catalog_sha256=digest(args.source/"catalog.json"),source_sha256={str(path.relative_to(ROOT)):digest(path) for path in paths},
        selected_ids=[d["id"] for d in items],gp=asdict(FeatureSgpParams(device="cuda")),modes=p["modes"])
    folder=args.output;folder.mkdir(parents=True,exist_ok=True);target=folder/"results.json"
    if target.exists():
        record=json.loads(target.read_text())
        if not args.resume or record["settings"]!=settings:raise ValueError("strict unchanged resume required")
    else:
        if any(folder.iterdir()):raise ValueError("unidentified earlier output")
        record=dict(settings=settings,environment_start=environment(),runs=[]);atomic_json(target,record)
    rows={r["id"]:r for r in record["runs"]};estimator=BgkSebEstimator(params_from_config("cupy",.2,36))
    for index,item in enumerate(items):
        path=args.source/item["path"]
        if digest(path)!=item["sha256"]:raise ValueError("input changed")
        with np.load(path) as a:flags=a["is_ground"].copy();xyz=a["xyz"].copy();origin=a["query_origin"].copy()
        for condition in p["conditions"]:
            key=item["id"]+"__"+condition;checkpoint=folder/(key+".json")
            if checkpoint.exists() and key not in rows:
                if not args.resume:raise ValueError("checkpoint requires explicit resume")
                rows[key]=json.loads(checkpoint.read_text())
            if key in rows:
                if digest(folder/rows[key]["archive"])!=rows[key]["archive_sha256"]:raise ValueError("checkpoint changed")
                continue
            original=lookup[item["id"],condition];archive=args.source/"public"/original["archive"]
            if digest(archive)!=original["archive_sha256"]:raise ValueError("baseline changed")
            with np.load(archive) as a:data={k:a[k].copy() for k in a.files}
            keep=partial_points(xyz,condition,origin,original["case_seed"])
            if not np.array_equal(xyz[keep,:2],data["raw_xyz"][:,:2]):raise ValueError("observation identity mismatch")
            ground=data["raw_xyz"][flags[keep]]
            torch.manual_seed(original["case_seed"])
            prediction=FeatureSgpPredictor(FeatureSgpParams(device="cuda")).fit(ground).predict(data["xy"])
            delta=float(np.max(np.abs(prediction.mean-data["mean"])))
            if delta>1e-4:raise ValueError("replayed prediction differs from frozen input: "+str(delta))
            measured=data["measured"];valid=data["observed"];variance=np.full(measured.shape,.0004,np.float32)
            completed,compvar,compvalid,filled=supported_completion(measured,variance,valid,prediction,
                max_std_m=p["max_std_m"],max_distance_m=p["max_distance_m"],blocked=data["occupied"])
            modes={"measured_only":(measured,variance,valid,np.zeros_like(valid)),
                "refine_observed":(refinement_surface(measured,prediction,True),variance,valid,np.zeros_like(valid)),
                "bounded_completion":(completed,compvar,compvalid,filled)}
            scores=[];metrics={};filled_metrics={};elapsed={}
            for mode in p["modes"]:
                height,var,support,mask=modes[mode]
                cp.cuda.Stream.null.synchronize();started=time.perf_counter()
                result=estimator.compute(height,var,support,[data["occupied"].astype(np.float32)],observed_elevation=measured)
                preserve_collision_risk(result,data["occupied"]);cp.cuda.Stream.null.synchronize()
                elapsed[mode]=(time.perf_counter()-started)*1000
                if result.backend!="cupy":raise ValueError("GPU fallback")
                score=np.stack(result.final_risk_slices).reshape(36,-1)[:,data["query_indices"]]
                scores.append(score);metrics[mode]=risk_metrics(score,data["reference_risk"],data["unsafe"],data["risk_supported"],.58)
            scores=np.stack(scores)
            baseline_delta=float(np.max(np.abs(scores[0]-data["predicted_risk"])))
            if baseline_delta>1e-6:raise ValueError("measured-only risk replay differs from source experiment")
            old_missed=(scores[0]<.58)&data["unsafe"]&data["risk_supported"]
            newly_missed={mode:int(((scores[i]<.58)&data["unsafe"]&data["risk_supported"]&~old_missed).sum()) for i,mode in enumerate(p["modes"])}
            filled_metrics=domain_metrics(prediction,data["reference"],filled.ravel()&np.isfinite(data["reference"]),[.95])
            if (filled&valid).any() or (filled&data["occupied"]).any():raise ValueError("completion replaced observed/protected geometry")
            out=folder/(key+".npz")
            np.savez_compressed(out,risk=scores,reference_risk=data["reference_risk"],unsafe=data["unsafe"],supported=data["risk_supported"],
                filled=filled,observed=valid,occupied=data["occupied"],mean=prediction.mean,variance=prediction.latent_var,
                local=prediction.local_conditioned,nearest=prediction.nearest_distance,accepted=prediction.accepted,
                reference=data["reference"],xy=data["xy"],query_indices=data["query_indices"],measured=measured,
                origin=origin,resolution=.2)
            row=dict(id=key,dataset_id=item["id"],family=item["family"],cluster_id=item["cluster_id"],condition=condition,
                risk=metrics,newly_missed=newly_missed,filled_cells=int(filled.sum()),filled_height=filled_metrics,
                hidden_reference_cells=original["height"]["hidden"]["n"],fit_ms=prediction.fit_ms,predict_ms=prediction.predict_ms,
                replay_mean_max_delta_m=delta,replay_risk_max_delta=baseline_delta,
                risk_wall_ms=elapsed,archive=out.name,archive_sha256=digest(out))
            rows[key]=row;atomic_json(checkpoint,row)
        record["runs"]=list(rows.values());atomic_json(target,record)
        if (index+1)%10==0:print("[joint integration]",len(rows),"/",p["planned_fits"],flush=True)
    if len(rows)!=p["planned_fits"]:raise ValueError("incomplete joint experiment")
    record["environment_end"]=environment();atomic_json(target,record)


if __name__=="__main__":main()
