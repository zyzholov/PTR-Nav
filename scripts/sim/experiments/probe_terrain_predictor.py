"""Diagnostic parameter controls on fixed observations; not an external benchmark."""

import argparse
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

from pointcloud_batch import ROOT, atomic_json, digest
from paper_clouds import grid, height_at
from paper_scale_evaluate import partial_points, height_metrics
from diagnose_terrain_prediction import OUTPUT, BASE, FAMILIES


def load_baseline():
    path=OUTPUT/"frozen_before/feature_sgp.py"
    spec=importlib.util.spec_from_file_location("terrain_before_fix",path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def run(catalog_path,output,per_family):
    baseline=load_baseline();catalog=json.loads(catalog_path.read_text())
    source_rows=json.loads((BASE/"budget/results.json").read_text())["runs"]
    saved={(r["dataset_id"],r["condition"]):r for r in source_rows if r["requested_M"]==125}
    torch.set_num_threads(1);results=[]
    for family in FAMILIES:
        datasets=[d for d in catalog["datasets"] if d["family"]==family][:per_family]
        for item in datasets:
            with np.load(catalog_path.parent/item["path"]) as a:case={k:a[k] for k in a.files if k!="full_xyz"}
            xy=grid(tuple(case["shape"]),case["query_origin"],.2)
            reference=height_at(case["truth_height"],case["truth_origin"],float(case["truth_resolution"]),xy)
            for condition in ("point15","hole16","frontier50"):
                old=saved[item["id"],condition]
                keep=partial_points(case["xyz"],condition,case["query_origin"],old["case_seed"])
                with np.load(BASE/"budget"/old["reference"]) as a: raw=a["raw_xyz"]; observed=a["observed"].ravel()
                ground=raw[case["is_ground"][keep]]
                for name,changes in (("baseline",{}),("larger_interpolation_neighborhood",{"local_nearest_m":.9}),
                        ("longer_global_scale",{"lengthscale_m":1.5}),
                        ("both",{"local_nearest_m":.9,"lengthscale_m":1.5})):
                    args=dict(device="cuda",max_train_points=512,inducing_points=125,**changes)
                    torch.manual_seed(old["case_seed"])
                    model=baseline.FeatureSgpPredictor(baseline.FeatureSgpParams(**args)).fit(ground)
                    pred=model.predict(xy);hidden=np.isfinite(reference)&~observed
                    results.append({"id":item["id"],"family":family,"condition":condition,"variant":name,
                        "fit_ms":pred.fit_ms,"predict_ms":pred.predict_ms,"local_hidden_fraction":float(pred.local_conditioned[hidden].mean()) if hidden.any() else None,
                        **height_metrics(pred,reference,observed)})
            print("[probe]",item["id"],flush=True)
    atomic_json(output,{"scope":__doc__,"baseline_sha256":digest(OUTPUT/"frozen_before/feature_sgp.py"),
        "catalog_sha256":digest(catalog_path),"runs":results})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--catalog",type=Path,default=BASE/"catalog.json")
    parser.add_argument("--output",type=Path,default=OUTPUT/"parameter_probe.json");parser.add_argument("--per-family",type=int,default=4)
    args=parser.parse_args();run(args.catalog,args.output,args.per_family)


if __name__=="__main__":main()
