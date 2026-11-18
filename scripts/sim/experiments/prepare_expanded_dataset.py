#!/usr/bin/env python3
"""New generated holdout plus explicitly labelled unchanged public regression."""

import argparse
import copy
import json
import os
from pathlib import Path

from pointcloud_batch import atomic_json, digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated",type=Path,required=True);parser.add_argument("--public",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise ValueError("preserve previous experiment catalog")
    generated=json.loads(args.generated.read_text());prior=json.loads(args.public.read_text())
    previous_hashes={d["geometry_sha256"] for d in prior["datasets"] if d["source_kind"].startswith("epfl")}
    if any(d["geometry_sha256"] in previous_hashes for d in generated["datasets"]):
        raise ValueError("new generated holdout overlaps the old geometry")
    rows=[]
    for source,catalog,split in ((args.generated,generated,"new_generated_holdout"),(args.public,prior,"public_regression")):
        for item in catalog["datasets"]:
            if split=="public_regression" and item["family"] not in ("indoor","road"):continue
            d=copy.deepcopy(item);d["path"]=os.path.relpath((source.parent/d["path"]).resolve(),args.output.parent.resolve())
            d["split"]=split;rows.append(d)
    result={"date":"2026-10-13","schema_version":1,"datasets":rows,
        "source_catalogs":{"generated_sha256":digest(args.generated),"public_sha256":digest(args.public)},
        "scope":"1500 new generated geometries; 1000 unchanged public inputs are regression, not new real-world scenes"}
    atomic_json(args.output,result);print(f"[expanded] {len(rows)} explicitly split sources: {args.output}")


if __name__=="__main__":main()
