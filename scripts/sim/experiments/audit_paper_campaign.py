#!/usr/bin/env python3
"""Verify actual saved arrays, source hashes and GPU records without refitting."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from pointcloud_batch import atomic_json, digest
from plot_paper_scale import validate_campaign


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,required=True)
    parser.add_argument("--results",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--expanded",action="store_true",help="validate the frozen eight-condition expansion")
    parser.add_argument("--core-source",type=Path,help="verify recorded model source snapshot")
    parser.add_argument("--completion",action="store_true",help="also audit paired bounded-completion arrays")
    args=parser.parse_args()
    catalog=json.loads(args.dataset.read_text());record=json.loads(args.results.read_text())
    if args.expanded:
        from plot_expanded_campaign import validate_terrain
        validate_terrain(record,catalog)
    else:
        validate_campaign(record,catalog)
    if digest(args.dataset)!=record["settings"]["dataset_sha256"]:
        raise ValueError("input catalog hash mismatch")
    if args.core_source:
        for name,expected in record["settings"]["core_sha256"].items():
            if digest(args.core_source/name)!=expected:raise ValueError("frozen model source changed")
    items={d["id"]:d for d in catalog["datasets"]}
    for d in items.values():
        if digest(args.dataset.parent/d["path"])!=d["sha256"]:
            raise ValueError("source input hash mismatch: "+d["id"])
    completion_details=[]
    for row in record["runs"]:
        if row["family"]!=items[row["dataset_id"]]["family"] or row["cluster_id"]!=items[row["dataset_id"]]["cluster_id"]:
            raise ValueError("source grouping mismatch")
        path=args.results.parent/(row["id"]+".json")
        if path.exists() and json.loads(path.read_text())!=row:
            raise ValueError("per-run checkpoint differs from campaign record")
        if row["status"]!="ok":
            continue
        if not path.exists() or row["actual_device"]!="cuda" or row["actual_backend"]!="cupy":
            raise ValueError("successful run missing checkpoint or requested GPU backend")
        with np.load(path.with_suffix(".npz"),allow_pickle=False) as a:
            predicted,truth,unsafe,support=(a[k] for k in ("predicted_risk","reference_risk","unsafe","supported"))
            if predicted.shape!=truth.shape or predicted.shape!=support.shape or predicted.shape[0]!=36:
                raise ValueError("incorrect saved risk dimensions")
            if not np.isfinite(predicted).all() or not np.isfinite(a["height_mean"]).all() or not np.isfinite(a["latent_var"]).all():
                raise ValueError("nonfinite saved prediction")
            if (a["latent_var"]<0).any() or not np.isfinite(truth[support]).all():
                raise ValueError("invalid uncertainty or reference support")
            if row["reference_states"]!=int(support.sum()):
                raise ValueError("risk denominator mismatch")
            hazard=predicted>=.58
            counts={"tp":int((hazard&unsafe&support).sum()),"fn":int((~hazard&unsafe&support).sum()),
                    "fp":int((hazard&~unsafe&support).sum()),"tn":int((~hazard&~unsafe&support).sum())}
            if any(counts[k]!=row[k] for k in counts):
                raise ValueError("saved prediction and confusion counts disagree")
            if support.any():
                error=predicted[support]-truth[support]
                for key,value in (("risk_mae",np.abs(error).mean()),("risk_error_variance",error.var())):
                    if not np.isclose(value,row[key],atol=1e-7):
                        raise ValueError("saved prediction and reported risk error disagree")
            hidden=np.isfinite(a["reference_height"].ravel()) & ~a["observed_cells"].ravel()
            if hidden.any():
                error=a["height_mean"][hidden]-a["reference_height"].ravel()[hidden]
                for key,value in (("hidden_rmse_m",np.sqrt(np.mean(error**2))),
                    ("hidden_picp95",np.mean(np.abs(error)<=1.96*np.sqrt(a["latent_var"][hidden]))),
                    ("hidden_width95_m",np.mean(3.92*np.sqrt(a["latent_var"][hidden])))):
                    if not np.isclose(value,row[key],atol=1e-6):
                        raise ValueError("saved hidden height and uncertainty metrics disagree")
            if args.completion:
                observed_risk=a["observed_only_risk"]
                if observed_risk.shape!=truth.shape or not np.isfinite(observed_risk).all():
                    raise ValueError("invalid paired observed-only risk")
                oh=observed_risk>=.58
                oc={"tp":int((oh&unsafe&support).sum()),"fn":int((~oh&unsafe&support).sum()),
                    "fp":int((oh&~unsafe&support).sum()),"tn":int((~oh&~unsafe&support).sum())}
                if any(oc[k]!=row["observed_only_"+k] for k in oc):
                    raise ValueError("paired confusion count mismatch")
                if support.any() and not np.isclose(np.abs(observed_risk[support]-truth[support]).mean(),
                                                  row["observed_only_risk_mae"],atol=1e-7):
                    raise ValueError("paired risk MAE mismatch")
                filled=a["predicted_cells"].ravel().astype(bool)
                if (filled&a["observed_cells"].ravel()).any() or filled.sum()!=row["completed_cells"]:
                    raise ValueError("completion overwrote observations or count mismatch")
                scored=filled&np.isfinite(a["reference_height"].ravel())
                error=a["height_mean"][scored]-a["reference_height"].ravel()[scored]
                rmse=float(np.sqrt(np.mean(error**2))) if len(error) else None
                if rmse is not None and not np.isclose(rmse,row["completed_height_rmse_m"],atol=1e-6):
                    raise ValueError("filled-cell height error mismatch")
                completion_details.append({"id":row["id"],"family":row["family"],
                    "cluster_id":row["cluster_id"],"condition":row["condition"],"status":"ok",
                    "filled_scored_cells":int(scored.sum()),"hidden_reference_cells":int(hidden.sum()),
                    "filled_gt_coverage":float(scored.sum()/hidden.sum()) if hidden.any() else None,
                    "filled_outside_reference_cells":int((filled&~np.isfinite(a["reference_height"].ravel())).sum()),
                    "filled_rmse_m":rmse})
    generated=[d["geometry_sha256"] for d in items.values() if d["source_kind"].startswith("epfl")]
    if len(set(generated))!=len(generated):
        raise ValueError("duplicate generated terrain geometry")
    audit={"status":"passed","dataset_sha256":digest(args.dataset),"results_sha256":digest(args.results),
        "inputs_verified":len(items),"runs_verified":len(record["runs"]),
        "statuses":dict(Counter(r["status"] for r in record["runs"])),
        "unique_generated_heights":len(set(generated)),
        "successful_arrays_verified":sum(r["status"]=="ok" for r in record["runs"]),
        "scope":"input archives, per-run checkpoint, array finiteness, GPU backend, reference denominators and risk metrics; not external ground truth certification"}
    if args.completion:
        if not record["settings"].get("complete_supported"):
            raise ValueError("not a bounded-completion campaign")
        audit["completion_details"]=completion_details
        audit["scope"]+="; paired observed-only counts, non-overwrite, GT-scored completion coverage"
    args.output.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(args.output,audit)
    print(json.dumps({k:v for k,v in audit.items() if k!="completion_details"},ensure_ascii=False))


if __name__=="__main__":
    main()
