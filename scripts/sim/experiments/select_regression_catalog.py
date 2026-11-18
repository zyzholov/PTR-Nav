#!/usr/bin/env python3
"""Select source-balanced reused inputs before running completion regression."""

import argparse
from collections import defaultdict
import copy
import json
import os
from pathlib import Path
import numpy as np
from pointcloud_batch import atomic_json,digest


def select(items,count,seed):
    groups=defaultdict(list)
    for item in items:groups[item["cluster_id"]].append(item)
    rng=np.random.default_rng(seed);keys=sorted(groups);rng.shuffle(keys)
    for key in keys:rng.shuffle(groups[key])
    out=[]
    while len(out)<count and any(groups.values()):
        for key in keys:
            if groups[key]:out.append(groups[key].pop())
            if len(out)==count:break
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--per-family",type=int,default=100)
    p.add_argument("--seed",type=int,default=243261031);a=p.parse_args()
    if a.output.exists() or a.per_family<1:raise ValueError("new output and positive count required")
    catalog=json.loads(a.source.read_text());out=[]
    for i,family in enumerate(sorted({d["family"] for d in catalog["datasets"]})):
        for row in select([d for d in catalog["datasets"] if d["family"]==family],a.per_family,a.seed+i):
            d=copy.deepcopy(row)
            d["path"]=os.path.relpath((a.source.parent/d["path"]).resolve(),a.output.parent.resolve())
            d["split"]="completion_policy_regression_reused_source";out.append(d)
    atomic_json(a.output,{"datasets":out,"seed":a.seed,"source_sha256":digest(a.source),
        "scope":"source-balanced regression, not new independent terrain collection; no outcome-based selection"})
    print(f"[regression inputs] {len(out)} selected")


if __name__=="__main__":main()
