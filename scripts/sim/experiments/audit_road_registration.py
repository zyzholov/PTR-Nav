"""Cross-scan ground consistency using official pose alternatives, without ICP or Z adjustment."""

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from pointcloud_batch import ROOT, atomic_json
from paper_road_clouds import registration_poses, GROUND


def run():
    raw=ROOT/"data/public_clouds"
    selected=json.loads((raw/"paper_sources.json").read_text())["sources"]["kitti_selected"]["files"]
    rows=[]
    for sequence in (8,9,10):
        files=[r for r in selected if r["sequence"]==sequence][:4]
        for source in ("kitti_gt","semantic_suma"):
            poses,provenance=registration_poses(raw,sequence,source);clouds=[]
            anchor=np.linalg.inv(poses[files[0]["frame"]])
            for item in files:
                xyz=np.fromfile(item["points"],dtype="<f4").reshape(-1,4)[:,:3]
                semantic=np.fromfile(item["labels"],dtype="<u4")&0xffff
                xyz=xyz[np.isin(semantic,GROUND)]
                pose=anchor@poses[item["frame"]]
                points=xyz@pose[:3,:3].T+pose[:3,3]
                clouds.append(points[(np.abs(points[:,:2])<=20).all(axis=1)])
            tree=cKDTree(clouds[0][:,:2])
            for item,points in zip(files[1:],clouds[1:]):
                distance,ids=tree.query(points[:,:2]);keep=distance<=.1
                error=points[keep,2]-clouds[0][ids[keep],2]
                rows.append({"sequence":sequence,"source":source,"frame":item["frame"],"anchor_frame":files[0]["frame"],
                    "xy_tolerance_m":.1,"pairs":len(error),"mae_m":float(np.abs(error).mean()),
                    "median_abs_m":float(np.median(np.abs(error))),"p95_abs_m":float(np.quantile(np.abs(error),.95)),
                    "median_signed_m":float(np.median(error)),"provenance":provenance})
    atomic_json(ROOT/"log/terrain_diagnosis_20261015/road_registration_audit.json",
        {"scope":__doc__,"runs":rows,"note":"XY nearest matches are geometric consistency checks, not independent pose truth"})


if __name__=="__main__":run()
