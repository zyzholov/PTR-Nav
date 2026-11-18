"""Static KITTI submaps with semantic-assisted ground and explicit proxy truth."""

import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from scipy.spatial import cKDTree

from pointcloud_batch import atomic_json, digest
from paper_clouds import raster, write_case

GROUND = (40,44,48,49,60,72)
STRUCTURE = (10,11,13,15,16,18,20,30,31,32,50,51,52,70,71,80,81,99)


def lidar_poses(calibration, pose_file):
    values = {}
    for line in calibration.read_text().splitlines():
        key, _, value = line.partition(":")
        values[key] = np.fromstring(value,sep=" ")
    extrinsic=np.eye(4); extrinsic[:3]=values["Tr"].reshape(3,4)
    camera=np.loadtxt(pose_file).reshape(-1,3,4)
    poses=np.broadcast_to(np.eye(4),(len(camera),4,4)).copy(); poses[:,:3]=camera
    return np.linalg.inv(extrinsic)[None]@poses@extrinsic[None]


def registration_poses(raw, sequence, source="semantic_suma"):
    calibration=raw/"kitti_metadata/dataset/sequences"/f"{sequence:02d}/calib.txt"
    if source=="semantic_suma":
        archive=raw/"semantic_kitti_labels.zip"
        member=f"dataset/sequences/{sequence:02d}/poses.txt"
        with zipfile.ZipFile(archive) as stream:content=stream.read(member)
        poses=lidar_poses(calibration,io.BytesIO(content))
        provenance={"source":source,"archive":str(archive),"member":member,
            "poses_sha256":hashlib.sha256(content).hexdigest(),
            "source_documentation":"https://www.semantic-kitti.org/dataset.html"}
    elif source=="kitti_gt":
        path=raw/"kitti_metadata/dataset/poses"/f"{sequence:02d}.txt"
        poses=lidar_poses(calibration,path)
        provenance={"source":source,"path":str(path),"poses_sha256":digest(path)}
    else:raise ValueError("unknown registration pose source")
    provenance["calibration_sha256"]=digest(calibration)
    return poses,provenance


def build_road(args, raw, catalog):
    sources=json.loads((raw/"paper_sources.json").read_text())["sources"]["kitti_selected"]
    export_ids=getattr(args,"road_export_ids",None)
    export_ids=set(export_ids.split(",")) if export_ids else None
    available={r["id"] for r in sources["files"]}
    if (sources["status"]!="ok" or (export_ids is None and len(sources["files"])!=args.per_family)
            or (export_ids is not None and (not export_ids<=available or len(export_ids)!=args.per_family))):
        raise ValueError("complete the predetermined annotated scan download first")
    fingerprints={}
    for sequence in sorted({r["sequence"] for r in sources["files"]}):
        files=[r for r in sources["files"] if r["sequence"]==sequence]
        if export_ids is not None and not any(r["id"] in export_ids for r in files):continue
        poses,provenance=registration_poses(raw,sequence,getattr(args,"road_registration","semantic_suma"))
        coordinates=[]; labels=[]; centers=[]
        for item in files:
            point_path=Path(item["points"]); label_path=Path(item["labels"])
            if digest(point_path)!=item["points_sha256"] or digest(label_path)!=item["labels_sha256"]:
                raise ValueError("selected KITTI inputs changed")
            xyz=np.fromfile(point_path,dtype="<f4").reshape(-1,4)[:,:3]
            semantic=np.fromfile(label_path,dtype="<u4")&0xffff
            keep=np.isfinite(xyz).all(axis=1)&(np.linalg.norm(xyz,axis=1)>.5)&(semantic<252)
            transform=poses[item["frame"]]
            coordinates.append((xyz[keep]@transform[:3,:3].T+transform[:3,3]).astype(np.float32))
            labels.append(semantic[keep]); centers.append(transform[:2,3])
        centers=np.asarray(centers)
        for index,item in enumerate(files):
            if export_ids is not None and item["id"] not in export_ids:continue
            center=centers[index]
            neighbors=np.flatnonzero(np.linalg.norm(centers-center,axis=1)<=35)
            xyz=np.concatenate([coordinates[i] for i in neighbors]); semantic=np.concatenate([labels[i] for i in neighbors])
            take=(np.abs(xyz[:,:2]-center)<=25).all(axis=1)
            xyz,semantic=xyz[take],semantic[take]
            is_ground=np.isin(semantic,GROUND); ground=xyz[is_ground]
            if len(ground)<100:
                raise ValueError("insufficient annotated ground in a planned scan submap")
            origin=center-25; height=raster(ground,(250,250),origin,.2)
            collision=np.zeros((250,250),bool)
            distance, nearest=cKDTree(ground[:,:2]).query(xyz[:,:2])
            above=xyz[:,2]-ground[nearest,2]
            hazard=np.isin(semantic,STRUCTURE)&(distance<=.8)&(above>=.32)&(above<=1.5)
            ij=np.floor((xyz[hazard,:2]-origin)/.2).astype(int)
            inside=(ij>=0).all(axis=1)&(ij<250).all(axis=1)
            collision[ij[inside,1],ij[inside,0]]=True
            row={"id":item["id"],"family":"road","cluster_id":f"kitti_sequence_{sequence:02d}",
                "source_kind":"semantic_kitti_static_submap","reference_kind":"dense_cloud_proxy",
                "registration_pose_source":provenance["source"],"registration_poses_sha256":provenance["poses_sha256"],
                "frame":item["frame"],"source_extent_m":[50,50],"source_frames":[files[i]["id"] for i in neighbors],
                "frame_id":"first_lidar_sequence_frame_not_independently_gravity_aligned",
                "geometry_sha256":hashlib.sha256(xyz.tobytes()).hexdigest(),
                "reference_note":"pose-registered semantic ground proxy; moving semantic IDs >=252 excluded from both input and reference"}
            write_case(args.output,row,xyz,is_ground,height,origin,.2,collision,center,full_cloud=xyz)
            catalog["datasets"].append(row)
        fingerprints[f"{sequence:02d}"]=provenance
        atomic_json(args.output/"dataset_manifest.json",catalog)
        exported=sum(export_ids is None or item["id"] in export_ids for item in files)
        print(f"[clouds] annotated static road sequence {sequence:02d}: {exported} exported, {len(files)} registration inputs",flush=True)
    catalog["phases"]["road"]={"count":args.per_family,"source_groups":len(fingerprints),"registration":fingerprints,
        "script_sha256":digest(__file__),"input_sources_sha256":digest(raw/"paper_sources.json")}
