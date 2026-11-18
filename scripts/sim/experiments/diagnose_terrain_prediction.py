"""Attribute saved height error to source consistency, raster support and predictor branches."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree

from pointcloud_batch import ROOT, atomic_json, digest
from paper_clouds import grid, height_at, raster
from paper_scale_evaluate import partial_points
from fsgp_bgk_se2.feature_sgp import spatial_sample_indices

BASE = ROOT/"log/paper_aligned_20261015"
OUTPUT = ROOT/"log/terrain_diagnosis_20261015"
FAMILIES = ("hilly","forest","ruin","road","indoor")


def errors(predicted, reference, mask):
    selected = np.asarray(mask) & np.isfinite(reference) & np.isfinite(predicted)
    error = np.asarray(predicted)[selected]-np.asarray(reference)[selected]
    return {"n":len(error),"rmse_m":float(np.sqrt(np.mean(error**2))) if len(error) else None,
        "mae_m":float(np.mean(np.abs(error))) if len(error) else None,
        "bias_m":float(np.mean(error)) if len(error) else None,
        "p95_m":float(np.quantile(np.abs(error),.95)) if len(error) else None}


def diagnose(catalog_path=BASE/"catalog.json", output=OUTPUT, per_family=8):
    catalog=json.loads(catalog_path.read_text());record=json.loads((BASE/"budget/results.json").read_text())
    rows={(r["dataset_id"],r["condition"],r["requested_M"]):r for r in record["runs"]}
    selected=[]
    for family in FAMILIES:
        candidates=[d for d in catalog["datasets"] if d["family"]==family]
        # Group round-robin is source-based, never selected using prediction error.
        groups=defaultdict(list)
        for d in candidates:groups[d["cluster_id"]].append(d)
        order=[d for i in range(max(map(len,groups.values()))) for g in sorted(groups)
            for d in sorted(groups[g],key=lambda r:r["id"])[i:i+1]]
        selected.extend(order[:per_family])
    output.mkdir(parents=True,exist_ok=True);results=[]
    for item in selected:
        path=(catalog_path.parent/item["path"]).resolve()
        if digest(path)!=item["sha256"]:raise ValueError("source archive changed")
        with np.load(path) as a:case={k:a[k] for k in a.files if k!="full_xyz"}
        ground=case["xyz"][case["is_ground"]]
        ref_at_points=height_at(case["truth_height"],case["truth_origin"],float(case["truth_resolution"]),ground[:,:2])
        shape=tuple(case["shape"]);xy=grid(shape,case["query_origin"],.2)
        dense=height_at(case["truth_height"],case["truth_origin"],float(case["truth_resolution"]),xy)
        original=raster(ground,shape,case["query_origin"],.2).ravel()
        complete=LinearNDInterpolator(ground[spatial_sample_indices(ground,16384),:2],ground[spatial_sample_indices(ground,16384),2])(xy)
        base={"dataset_id":item["id"],"family":item["family"],"cluster_id":item["cluster_id"],"source_sha256":item["sha256"],
            "source_consistency":errors(ground[:,2],ref_at_points,np.ones(len(ground),bool)),
            "source_reference_fraction":float(np.isfinite(ref_at_points).mean()),
            "input_raster_center_difference":errors(original,dense,np.isfinite(original)),
            "full_input_linear_interpolation":errors(complete,dense,np.isfinite(dense))}
        for condition in ("point15","hole16","frontier50"):
            row=rows[item["id"],condition,125]
            with np.load(BASE/"budget"/row["reference"]) as a:
                reference=a["reference_height"].ravel();observed=a["observed"].ravel();raw=a["raw_xyz"]
            with np.load(BASE/"budget"/row["archive"]) as a:
                prediction=a["mean"];accepted=a["accepted"];local=a["local_conditioned"]
            if not np.allclose(reference,dense,equal_nan=True,atol=1e-6):raise ValueError("wrong reference/window")
            keep=partial_points(case["xyz"],condition,case["query_origin"],row["case_seed"])
            obs=raw[case["is_ground"][keep]]
            hidden=np.isfinite(reference)&~observed
            retained_hidden=hidden&np.isfinite(original)
            natural_hidden=hidden&~np.isfinite(original)
            nearest=cKDTree(obs[:,:2]).query(xy)[0]
            ids=spatial_sample_indices(obs,16384)
            linear=LinearNDInterpolator(obs[ids,:2],obs[ids,2])(xy)
            diagnostic={**base,"condition":condition,"case_seed":row["case_seed"],
                "saved_gp":errors(prediction,reference,hidden),"deleted_only_gp":errors(prediction,reference,retained_hidden),
                "naturally_missing_gp":errors(prediction,reference,natural_hidden),
                "accepted_gp":errors(prediction,reference,hidden&accepted),
                "local_branch_gp":errors(prediction,reference,hidden&local),
                "global_branch_gp":errors(prediction,reference,hidden&~local),
                "observed_linear_interpolation":errors(linear,reference,hidden),
                "distance_bins":[]}
            for lo,hi in ((0,.35),(.35,.8),(.8,1.2),(1.2,3),(3,np.inf)):
                mask=hidden&(nearest>=lo)&(nearest<hi)
                diagnostic["distance_bins"].append({"lo_m":lo,"hi_m":hi if np.isfinite(hi) else None,
                    "gp":errors(prediction,reference,mask),"linear":errors(linear,reference,mask)})
            results.append(diagnostic)
        print("[diagnose]",item["id"],flush=True)
    atomic_json(output/"source_and_branch_audit.json",{"scope":__doc__,"catalog_sha256":digest(catalog_path),
        "saved_results_sha256":digest(BASE/"budget/results.json"),"script_sha256":digest(Path(__file__)),
        "baseline_sha256":digest(OUTPUT/"frozen_before/feature_sgp.py"),"selection":"first source-round-robin inputs, diagnostic/development only",
        "runs":results})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--per-family",type=int,default=8)
    args=parser.parse_args();diagnose(per_family=args.per_family)


if __name__=="__main__":main()
