"""One frozen campaign entry: sources, terrain, robustness, runtime and navigation."""

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from pointcloud_batch import ROOT, atomic_json, digest
from paper_clouds import grid
from paper_scale_evaluate import partial_points
from paper_budget_campaign import require_qualified_sources
from terrain_system_study import source_cases, mechanism_cases, evaluate, risk_metrics, fingerprints as terrain_fingerprints
from controlled_terrain_study import specification, raw_cloud, footprint_truth, run_scaling, environment
from terrain_risk_benchmark import params_from_config
from fsgp_bgk_se2.feature_sgp import FeatureSgpParams, FeatureSgpPredictor
from fsgp_bgk_se2.bgk_seb import BgkSebEstimator
from fsgp_bgk_se2.obstacle_columns import vertical_columns, preserve_collision_risk

BASE = ROOT/"log/comprehensive_v4r1_20261015"
PROTOCOL = ROOT/"scripts/sim/experiments/protocols/comprehensive_study_v4r1.json"
HERE = Path(__file__).resolve().parent


def fingerprints():
    names = ("comprehensive_study.py", "navigation_sandbox.py", "planning_query_campaign.py", "paper_road_clouds.py")
    return {**terrain_fingerprints(), **{str((HERE/n).relative_to(ROOT)): digest(HERE/n) for n in names}}


def call(script, *arguments):
    subprocess.run([sys.executable, str(HERE/script), *map(str, arguments)], check=True)


def prepare(args, protocol):
    output = args.output
    target = output/"catalog.json"
    if target.exists():
        record = json.loads(target.read_text())
        if not args.resume or record["protocol_sha256"] != digest(args.protocol):
            raise ValueError("preserve existing catalog; identical explicit resume required")
        return
    output.mkdir(parents=True, exist_ok=True)
    generated = output/"generated/dataset_manifest.json"
    if not generated.exists():
        p=protocol["public"]
        call("paper_clouds.py", "--phase", "epfl", "--per-family", p["generated_per_family"],
             "--seed", p["generated_seed"], "--generator-profile", "metric_v2", "--output", output/"generated")
    original=ROOT/"log/paper_aligned_20261015/catalog.json"
    old=json.loads(original.read_text())["datasets"]
    road_ids=[d["id"] for d in old if d["family"]=="road"]
    road=output/"road_suma/dataset_manifest.json"
    if not road.exists():
        call("paper_clouds.py", "--phase", "road", "--per-family", len(road_ids),
             "--road-registration", "semantic_suma", "--road-export-ids", ",".join(road_ids), "--output", output/"road_suma")
    rows=[]
    for catalog,items,reuse in ((generated,json.loads(generated.read_text())["datasets"],False),
        (road,json.loads(road.read_text())["datasets"],True), (original,[d for d in old if d["family"]=="indoor"],True)):
        for item in items:
            row=dict(item,path=os.path.relpath((catalog.parent/item["path"]).resolve(),output.resolve()),reused_public_source=reuse)
            rows.append(row)
    require_qualified_sources(rows)
    counts=Counter(d["family"] for d in rows)
    expected=Counter({f:protocol["public"]["generated_per_family"] for f in ("hilly","forest","ruin")},
        road=protocol["public"]["road_inputs"],indoor=protocol["public"]["indoor_inputs"])
    if counts!=expected:raise ValueError("source counts differ from protocol")
    previous=set()
    for path in (original,ROOT/"log/terrain_system_v3_20261015/catalog.json",ROOT/"log/terrain_diagnosis_20261015/regression_v2_catalog.json"):
        previous.update(d["geometry_sha256"] for d in json.loads(path.read_text())["datasets"] if "geometry_sha256" in d)
    new=[d for d in rows if d["family"] in ("hilly","forest","ruin")]
    if len({d["geometry_sha256"] for d in new})!=len(new) or previous & {d["geometry_sha256"] for d in new}:
        raise ValueError("generated geometry is duplicated")
    groups={f:len({d["cluster_id"] for d in rows if d["family"]==f}) for f in counts}
    atomic_json(target,dict(study_id=protocol["study_id"],protocol_sha256=digest(args.protocol),datasets=rows,
        source_groups=groups,provenance={str(p):digest(p) for p in (original,generated,road)}))
    frozen=output/"frozen_sources"
    for name,value in fingerprints().items():
        destination=frozen/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,destination)
        if digest(destination)!=value:raise ValueError("source snapshot differs")
    shutil.copy2(args.protocol,output/"protocol.json")
    print("[v4 sources]",dict(counts),"groups",groups,flush=True)


def freeze_record(args, settings):
    folder=args.output/args.phase;folder.mkdir(parents=True,exist_ok=True)
    path=folder/"results.json"
    if path.exists():
        old=json.loads(path.read_text())
        if not args.resume:raise ValueError("explicit resume required")
        if old["settings"]!=settings:
            if not getattr(args,"migrate_input_refusals",False) or not migration_compatible(old["settings"],settings):
                raise ValueError("changed configuration; use a new batch")
            revision=args.output/"provenance/input_refusal_policy"
            revision.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,revision/(args.phase+"_before.json"))
            previous=args.output/"frozen_sources/scripts/sim/experiments/comprehensive_study.py"
            shutil.copy2(previous,revision/"previous_driver.py")
            prior=old["settings"]["source_sha256"]["scripts/sim/experiments/comprehensive_study.py"]
            old.setdefault("execution_history",[]).append(dict(settings=old["settings"],reason="record insufficient-observation refusals without changing any model or valid-input computation",
                completed_before_migration=len(old["runs"]),previous_driver_sha256=prior,utc=datetime.now(timezone.utc).isoformat()))
            for r in old["runs"]:r.setdefault("execution_driver_sha256",prior)
            old["checkpoint_driver_sha256"]=prior
            old["settings"]=settings
            shutil.copy2(Path(__file__),previous)
            atomic_json(path,old)
        return folder,old
    if any(folder.iterdir()):raise ValueError("unidentified earlier files")
    record=dict(settings=settings,environment_start=environment(),runs=[])
    atomic_json(path,record)
    return folder,record


def migration_compatible(previous,current):
    key="scripts/sim/experiments/comprehensive_study.py"
    previous=json.loads(json.dumps(previous))
    if previous["source_sha256"][key]!="f05402c4e04cf75f3e8ff12843ec0599cae29afeaa9a36811774846dabe8c667":return False
    previous["source_sha256"][key]=current["source_sha256"][key]
    return previous==current


def refused_input(case,condition,seed,noise,folder,key,reason):
    shape,origin=case["shape"],case["origin"]
    mask_origin=np.asarray(origin)+(np.asarray(shape[::-1])*.2-20.8)/2
    keep=partial_points(case["raw"],condition,mask_origin,seed)
    ground=case["raw"][keep&case["ground"]]
    target=folder/(key+".npz")
    np.savez_compressed(target,raw_xyz=case["raw"][keep],ground_xyz=ground,reference=case["reference"],
        reference_risk=case["reference_risk"],unsafe=case["unsafe"],risk_supported=case["risk_supported"],xy=case["xy"])
    item=case["item"]
    return dict(id=key,dataset_id=item["id"],family=item["family"],cluster_id=item["cluster_id"],condition=condition,
        case_seed=seed,noise_std_m=noise,status="input_refused",reason=reason,ground_input_points=len(ground),
        source_sha256=item["sha256"],actual_device=None,actual_backend=None,archive=target.name,archive_sha256=digest(target))


def warm(gp, estimator, seed):
    xy=grid((32,32),[-3.2,-3.2],.2)
    for i in range(3):
        torch.manual_seed(seed+i)
        FeatureSgpPredictor(gp).fit(np.c_[xy,.05*xy[:,0]]).predict(xy)
        estimator.compute(np.zeros((32,32),np.float32),np.full((32,32),.0004,np.float32),
            np.ones((32,32),bool),[np.zeros((32,32),np.float32)])


def terrain(args, protocol):
    torch.set_num_threads(1)
    gp=FeatureSgpParams(device="cuda")
    params=params_from_config("cupy",.2,36);estimator=BgkSebEstimator(params)
    section="public" if args.phase=="public" else "mechanism"
    p=protocol[section]
    settings=dict(study_id=protocol["study_id"],protocol_sha256=digest(args.protocol),source_sha256=fingerprints(),
        phase=args.phase,planned_cases=p["planned_cases"],gp=asdict(gp),bgk=asdict(params),
        catalog_sha256=digest(args.output/"catalog.json"),risk_input="measured_only",unknown_completion=False)
    folder,record=freeze_record(args,settings);rows={r["id"]:r for r in record["runs"]}
    warm(gp,estimator,protocol["seed"])
    cases=source_cases(args,protocol) if section=="public" else mechanism_cases(protocol)
    levels=[protocol["noise_std_m"]] if section=="public" else p["noise_levels_m"]
    for di,case in enumerate(cases):
        original=case["item"]
        for ci,condition in enumerate(p["conditions"]):
            # The entire noise triplet shares point deletion and standardized noise.
            seed=protocol["seed"]+di*100+ci
            for ni,noise in enumerate(levels):
                item=dict(original,id=original["id"]+f"__n{ni}")
                key=item["id"]+"__"+condition;checkpoint=folder/(key+".json")
                if checkpoint.exists() and key not in rows:
                    if not args.resume:raise ValueError("checkpoint requires explicit resume")
                    rows[key]=json.loads(checkpoint.read_text())
                    rows[key].setdefault("execution_driver_sha256",record.get("checkpoint_driver_sha256",digest(Path(__file__))))
                if key in rows:
                    row=rows[key]
                    if row["noise_std_m"]!=noise or digest(folder/row["archive"])!=row["archive_sha256"]:
                        raise ValueError("checkpoint changed")
                    continue
                selected=dict(case,item=item)
                try:
                    row=evaluate(selected,condition,seed,dict(protocol,noise_std_m=noise),gp,estimator,folder)
                except ValueError as exc:
                    if str(exc) not in ("at least eight finite observed points are required","ground observations do not span a two-dimensional surface"):raise
                    row=refused_input(selected,condition,seed,noise,folder,key,str(exc))
                row.update(dataset_id=original["id"],noise_std_m=noise,noise_level_index=ni,
                    geometry_sha256=original.get("geometry_sha256"),paired_key=original["id"]+"__"+condition,
                    execution_driver_sha256=digest(Path(__file__)))
                rows[key]=row;atomic_json(checkpoint,row)
        if (di+1)%5==0:
            record["runs"]=list(rows.values());atomic_json(folder/"results.json",record)
            print(f"[v4 {args.phase}] {len(rows)}/{p['planned_cases']}",flush=True)
    if len(rows)!=p["planned_cases"]:raise ValueError("incomplete factorial")
    record.update(runs=list(rows.values()),environment_end=environment())
    atomic_json(folder/"results.json",record)


def discretization(args, protocol):
    import cupy as cp
    p=protocol["discretization"]
    settings=dict(study_id=protocol["study_id"],protocol_sha256=digest(args.protocol),source_sha256=fingerprints(),
        planned_cases=p["planned_cases"],phase=args.phase,scope=p["scope"])
    folder,record=freeze_record(args,settings);rows={r["id"]:r for r in record["runs"]}
    qx,qy=np.meshgrid(np.arange(-2,2.01,.4),np.arange(-2,2.01,.4));centers=np.c_[qx.ravel(),qy.ravel()]
    yaws=(np.arange(72)+.5)*2*np.pi/72
    estimators={(res,yaw):BgkSebEstimator(params_from_config("cupy",res,yaw)) for res in p["resolution_m"] for yaw in p["yaw_bins"]}
    for fi,family in enumerate(p["families"]):
        for index in range(p["geometries_per_family"]):
            seed=protocol["seed"]+1000000+fi*10000+index;spec=specification(family,seed)
            reference,unsafe,_,_=footprint_truth(spec,centers,params_from_config("cupy",.2,72))
            for res in p["resolution_m"]:
                size=round(12.8/res);shape=(size,size);origin=np.full(2,-6.4-res/2)
                xy=grid(shape,origin,res);raw,ground_count=raw_cloud(spec,xy)
                measured=raw[:ground_count,2].reshape(shape);valid=np.ones(shape,bool)
                ij=np.rint((centers-origin)/res-.5).astype(int);ids=ij[:,1]*size+ij[:,0]
                if not np.allclose(xy[ids],centers,atol=1e-5):raise ValueError("physical query locations changed")
                occupied,_=vertical_columns(raw,shape,origin,res,ground_elevation=measured,xp=cp);occupied=cp.asnumpy(occupied)
                for yaw in p["yaw_bins"]:
                    key=f"{family}_{index:03d}_r{res}_y{yaw}"
                    checkpoint=folder/(key+".json")
                    if checkpoint.exists() and key not in rows:
                        if not args.resume:raise ValueError("explicit resume required")
                        rows[key]=json.loads(checkpoint.read_text())
                    if key in rows:
                        if digest(folder/rows[key]["archive"])!=rows[key]["archive_sha256"]:raise ValueError("changed archive")
                        continue
                    result=estimators[res,yaw].compute(measured,np.full(shape,.0004,np.float32),valid,[occupied.astype(np.float32)],observed_elevation=measured)
                    preserve_collision_risk(result,occupied)
                    if result.backend!="cupy":raise ValueError("GPU fallback")
                    coarse=np.floor(yaws*yaw/(2*np.pi)).astype(int)
                    scores=np.stack(result.final_risk_slices).reshape(yaw,-1)[coarse][:,ids]
                    target=folder/(key+".npz")
                    np.savez_compressed(target,predicted_risk=scores,reference_risk=reference,unsafe=unsafe,
                        supported=np.ones_like(unsafe),xy=centers,yaws=yaws,measured=measured,occupied=occupied,
                        origin=origin,resolution=res)
                    row=dict(id=key,family=family,cluster_id=f"{family}_{seed}",spec=spec,resolution_m=res,yaw_bins=yaw,
                        actual_backend=result.backend,status="ok",risk=risk_metrics(scores,reference,unsafe,np.ones_like(unsafe),.58),
                        archive=target.name,archive_sha256=digest(target))
                    rows[key]=row;atomic_json(checkpoint,row)
            record["runs"]=list(rows.values());atomic_json(folder/"results.json",record)
            if (index+1)%5==0:print(f"[v4 discretization] {len(rows)}/{p['planned_cases']}",flush=True)
    if len(rows)!=p["planned_cases"]:raise ValueError("incomplete discretization factorial")
    record["environment_end"]=environment();atomic_json(folder/"results.json",record)


def native(args, protocol):
    from navigation_sandbox import prepare as maps, run as closed_loop, SCENARIOS
    from planning_query_campaign import prepare as queries, run as query_run
    section=protocol[args.phase]
    binary=ROOT/"build/terrain_nav"/("terrain_nav_query_benchmark" if args.phase=="planning" else "terrain_nav_benchmark")
    values=SimpleNamespace(output=args.output/args.phase,seed=section["seed"],per_scenario=section["per_scenario"],
        repeats=section.get("repeats",1),domain=88,binary=binary,protocol=args.protocol,mppi=False,
        scenarios=section.get("scenarios",list(SCENARIOS)),gp_refine_observed=False)
    if not (values.output/"cases.json").exists():
        (queries if args.phase=="planning" else maps)(values)
    elif not args.resume:raise ValueError("native phase already prepared; explicit resume required")
    result=(query_run if args.phase=="planning" else closed_loop)(values)
    if result:raise RuntimeError("native test process failed")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    phases=("prepare","public","mechanism","discretization","scaling","planning","navigation")
    parser.add_argument("phase",choices=(*phases,"all"));parser.add_argument("--output",type=Path,default=BASE)
    parser.add_argument("--protocol",type=Path,default=PROTOCOL);parser.add_argument("--resume",action="store_true")
    parser.add_argument("--migrate-input-refusals",action="store_true",
        help="explicitly adopt the known old driver with identical model/data hashes, preserving its prefix and provenance")
    args=parser.parse_args();protocol=json.loads(args.protocol.read_text());torch.set_num_threads(1)
    os.environ.setdefault("OPENBLAS_NUM_THREADS","1");os.environ.setdefault("OMP_NUM_THREADS","1")
    selected=phases if args.phase=="all" else (args.phase,)
    for phase in selected:
        args.phase=phase;started=time.perf_counter()
        if phase=="prepare":prepare(args,protocol)
        elif phase in ("public","mechanism"):terrain(args,protocol)
        elif phase=="discretization":discretization(args,protocol)
        elif phase=="scaling":
            if args.resume and (args.output/phase/"results.json").exists():
                record=json.loads((args.output/phase/"results.json").read_text())
                if record["protocol_sha256"]!=digest(args.protocol) or len(record["runs"])!=protocol[phase]["planned_measured_calls"]:
                    raise ValueError("scaling resume mismatch")
            else:run_scaling(protocol,args.protocol,args.output/phase)
        else:native(args,protocol)
        atomic_json(args.output/(phase+"_completion.json"),dict(phase=phase,completed_utc=datetime.now(timezone.utc).isoformat(),
            wall_s=time.perf_counter()-started,protocol_sha256=digest(args.protocol)))


if __name__=="__main__":main()
