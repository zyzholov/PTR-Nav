#!/usr/bin/env python3
"""Pinned original SceneNet/EPFL plus spatially separated annotated KITTI scans."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import numpy as np
import requests
import yaml
from remotezip import RemoteZip, RemoteIOError

from fetch_public_clouds import extract_zip
from pointcloud_batch import atomic_json, digest

ROOT = Path(__file__).resolve().parents[3]
EPFL_COMMIT = "2fa980af462538a9301d52c1bd367c00adf31b80"
KITTI_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/"


def cache_archive(url, target, limit):
    if not target.exists():
        temporary = target.with_suffix(".part")
        for attempt in range(5):
            try:
                with requests.get(url, stream=True, timeout=(30, 60)) as response:
                    response.raise_for_status()
                    size = 0
                    with temporary.open("wb") as stream:
                        for block in response.iter_content(1024*1024):
                            size += len(block)
                            if size > limit:
                                raise ValueError("archive exceeds download budget")
                            stream.write(block)
                temporary.replace(target)
                break
            except requests.RequestException:
                temporary.unlink(missing_ok=True)
                if attempt == 4:
                    raise
                time.sleep(1)
    return {"url": url, "path": str(target), "bytes": target.stat().st_size, "sha256": digest(target)}


def select_frames(poses, quota, min_distance=12):
    selected, centers = [], []
    # Cover the trajectory before filling gaps; never use point-cloud scores.
    order = np.linspace(0, len(poses)-1, min(len(poses), quota*8), dtype=int)
    for index in dict.fromkeys(order):
        center = poses[index].reshape(3, 4)[:, 3]
        if not centers or min(np.linalg.norm(center - p) for p in centers) >= min_distance:
            selected.append(int(index)); centers.append(center)
        if len(selected) == quota:
            break
    return sorted(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/public_clouds")
    parser.add_argument("--road-count", type=int, default=500)
    args = parser.parse_args()
    root = args.output.resolve(); root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "paper_sources.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "date": "2026-10-13", "sources": {}, "scope": "replacement inputs, NOT the authors' 2500 data or truth maps"}
    epfl = root / "epfl_generator"
    if not epfl.exists():
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/droduit/procedural-terrain-generation.git", str(epfl)], check=True)
    commit = subprocess.check_output(["git", "-C", str(epfl), "rev-parse", "HEAD"], text=True).strip()
    if commit != EPFL_COMMIT:
        raise ValueError("EPFL source must use the pinned commit; do not silently change generator")
    manifest["sources"]["epfl"] = {"url": "https://github.com/droduit/procedural-terrain-generation",
        "commit": commit, "fragment_sha256": digest(epfl / "project/heightmap/heightmap_fshader.glsl"),
        "license_note": "No repository license file identified; kept local, not vendored/redistributed.", "status": "ok"}
    scenenet = root / "scenenet_cfe5ab85ddcc.zip"
    entry = cache_archive("https://bitbucket.org/robotvault/downloadscenenet/get/cfe5ab85ddcc.zip", scenenet, 2*1024**3)
    destination = root / "scenenet_original"
    if not destination.exists():
        extract_zip(scenenet, destination, 8*1024**3)
    entry.update(status="ok", obj_files=len(list(destination.rglob("*.obj"))),
        license="CC BY-NC 4.0; research-only per https://robotvault.bitbucket.io/")
    manifest["sources"]["scenenet"] = entry
    atomic_json(manifest_path, manifest)
    for name in ("data_odometry_poses", "data_odometry_calib"):
        archive = root / (name + ".zip")
        manifest["sources"][name] = cache_archive(KITTI_URL + archive.name, archive, 10*1024**2)
        extract_zip(archive, root / "kitti_metadata", 30*1024**2)
    labels_path = root / "semantic_kitti_labels.zip"
    label_entry = cache_archive("https://semantic-kitti.org/assets/data_odometry_labels.zip", labels_path, 250*1024**2)
    ontology_url = "https://raw.githubusercontent.com/PRBonn/semantic-kitti-api/master/config/semantic-kitti.yaml"
    ontology_file = root / "semantic_kitti_ontology.json"
    if not ontology_file.exists():
        response = requests.get(ontology_url, timeout=40); response.raise_for_status()
        values = yaml.safe_load(response.text)
        atomic_json(ontology_file, {"url": ontology_url, "labels": values["labels"],
                                   "content_sha256": hashlib.sha256(response.content).hexdigest()})
    atomic_json(manifest_path, manifest)
    road = root / "kitti_selected"; road.mkdir(exist_ok=True)
    candidates = []
    for sequence in range(11):
        poses = np.loadtxt(root / "kitti_metadata/dataset/poses" / f"{sequence:02d}.txt")
        candidates.append(select_frames(poses, max(100, args.road_count // 11 * 2)))
    plan = []
    for i in range(max(map(len, candidates))):
        for sequence, frames in enumerate(candidates):
            if i < len(frames) and len(plan) < args.road_count:
                plan.append((sequence, frames[i]))
    if len(plan) != args.road_count:
        raise ValueError("not enough spatially separated source locations; do not lower the separation silently")
    plan.sort()
    records = []
    old = manifest["sources"].get("kitti_selected", {})
    previous = {f["id"]: f for f in old.get("files", [])}
    with zipfile.ZipFile(labels_path) as labels, RemoteZip(KITTI_URL + "data_odometry_velodyne.zip", timeout=60) as cloud:
        members = set(cloud.namelist())
        for index, (sequence, frame) in enumerate(plan):
            name = f"kitti_{sequence:02d}_{frame:06d}"
            folder = road / f"{sequence:02d}"; folder.mkdir(exist_ok=True)
            point_path, label_path = folder / f"{frame:06d}.bin", folder / f"{frame:06d}.label"
            member = f"dataset/sequences/{sequence:02d}/velodyne/{frame:06d}.bin"
            if member not in members:
                raise ValueError(f"missing planned cloud: {member}")
            if name in previous:
                if digest(point_path) != previous[name]["points_sha256"] or digest(label_path) != previous[name]["labels_sha256"]:
                    raise ValueError(f"cached scan changed: {name}")
            else:
                for attempt in range(3):
                    try:
                        contents = cloud.read(member)
                        point_path.write_bytes(contents)
                        break
                    except (requests.RequestException, OSError, RemoteIOError) as exc:
                        if attempt == 2:
                            raise exc
                        time.sleep(1)
                label_path.write_bytes(labels.read(f"dataset/sequences/{sequence:02d}/labels/{frame:06d}.label"))
            rows = point_path.stat().st_size // 16
            if point_path.stat().st_size % 16 or label_path.stat().st_size != rows*4:
                raise ValueError(f"point/label alignment mismatch: {name}")
            records.append({"id": name, "sequence": sequence, "frame": frame, "point_count": rows,
                "points": str(point_path), "labels": str(label_path), "points_sha256": digest(point_path),
                "labels_sha256": digest(label_path), "archive_member": member})
            manifest["sources"]["kitti_selected"] = {"url": KITTI_URL + "data_odometry_velodyne.zip",
                "homepage": "https://registry.opendata.aws/kitti/", "label_archive": label_entry,
                "ontology_sha256": digest(ontology_file), "license": "CC BY-NC-SA 3.0; local non-commercial research",
                "selection": "500 predetermined pose locations; >=12m separation within each sequence, before reading any score",
                "status": "ok" if index+1 == len(plan) else "downloading", "source_groups": 11, "files": records}
            atomic_json(manifest_path, manifest)
            if (index+1) % 25 == 0:
                print(f"[sources] KITTI {index+1}/{len(plan)} annotated scans", flush=True)
    print(f"[sources] original generator, {entry['obj_files']} SceneNet OBJ resources, {len(records)} KITTI scans: {manifest_path}")


if __name__ == "__main__":
    main()
