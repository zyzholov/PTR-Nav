#!/usr/bin/env python3
"""Fetch bounded official examples, recording immutable URLs and file hashes."""

import argparse
import base64
from datetime import datetime
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

import yaml

from pointcloud_batch import atomic_json, digest

FSGP_REPO = "ZJU-FAST-Lab/FSGP_BGK"
FSGP_SCENES = ("desert", "forest", "hill", "mountain", "quick", "snow", "vocano")
RELLIS_ID = "1QikPnpmxneyCuwefr6m50fBOSB2ny4LC"


def get_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": "PTR-Nav-experiment-data/1.0"})
    with urllib.request.urlopen(request, timeout=40) as response:
        return json.load(response)


def bounded_download(url, output, limit=80 * 1024 * 1024):
    temporary = output.with_suffix(output.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "PTR-Nav-experiment-data/1.0"})
    size = 0
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as target:
            if int(response.headers.get("Content-Length", 0)) > limit:
                raise ValueError("download exceeds configured size limit")
            while block := response.read(1024 * 1024):
                size += len(block)
                if size > limit:
                    raise ValueError("download exceeds configured size limit")
                target.write(block)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def extract_zip(path, output, limit=250 * 1024 * 1024):
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if sum(item.file_size for item in entries) > limit:
            raise ValueError("example archive exceeds extraction budget")
        for item in entries:
            target = (output / item.filename).resolve()
            if not target.is_relative_to(output.resolve()) or (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("unsafe archive entry")
        for item in entries:
            if item.is_dir():
                continue
            target = output / item.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(item) as source, target.open("wb") as stream:
                shutil.copyfileobj(source, stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/public_clouds"))
    parser.add_argument("--source", choices=("fsgp", "rellis", "all"), default="all")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "fetch_manifest.json"
    record = json.loads(path.read_text()) if path.exists() else {
        "created_at": datetime.now().astimezone().isoformat(), "sources": []}
    if args.source in ("fsgp", "all"):
        old = next((s for s in record["sources"] if s["name"] == "fsgp_official_examples"), None)
        commit = old["commit"] if old else get_json(f"https://api.github.com/repos/{FSGP_REPO}/commits/main")["sha"]
        folder = output / "fsgp"
        folder.mkdir(exist_ok=True)
        source = {"name": "fsgp_official_examples", "commit": commit,
                  "homepage": f"https://github.com/{FSGP_REPO}",
                  "license_note": "Official README declares MIT; example data kept local, not redistributed.",
                  "scope": "seven repository examples, NOT the paper's 2500-scene benchmark", "files": []}
        source["status"] = "downloading"
        record["sources"] = [s for s in record["sources"] if s["name"] != source["name"]] + [source]
        atomic_json(path, record)
        for scene in FSGP_SCENES:
            url = f"https://raw.githubusercontent.com/{FSGP_REPO}/{commit}/src/simulation_env_ros2/PCD/{scene}.pcd"
            target = folder / f"{scene}.pcd"
            expected = next((f["sha256"] for f in old["files"] if f["scene"] == scene), None) if old else None
            if not target.exists() or expected is None:
                bounded_download(url, target)
            sha = digest(target)
            if expected and sha != expected:
                raise ValueError(f"download cache checksum mismatch: {target}")
            source["files"].append({"scene": scene, "path": str(target.relative_to(output)),
                                    "url": url, "sha256": sha, "bytes": target.stat().st_size})
            print(f"[download] {scene}: {target.stat().st_size} bytes", flush=True)
            atomic_json(path, record)
        source["status"] = "ok"
        record["sources"] = [s for s in record["sources"] if s["name"] != source["name"]] + [source]
        atomic_json(path, record)
    if args.source in ("rellis", "all"):
        source = {"name": "rellis_official_lidar_example",
                  "homepage": "https://github.com/unmannedlab/RELLIS-3D",
                  "url": f"https://drive.google.com/file/d/{RELLIS_ID}/view",
                  "license_note": "CC BY-NC-SA 3.0 per official repository; raw/derived data kept local for noncommercial research.",
                  "scope": "official annotated LiDAR examples, not the full five-sequence corpus"}
        try:
            import gdown
            archive = output / "Rellis_3D_lidar_example.zip"
            old = next((s for s in record["sources"] if s["name"] == source["name"]), None)
            if archive.exists() and old and old.get("sha256") and digest(archive) != old["sha256"]:
                raise ValueError("RELLIS archive cache checksum mismatch")
            if not archive.exists():
                result = gdown.download(id=RELLIS_ID, output=str(archive), quiet=False)
                if result is None:
                    raise RuntimeError("official Drive example unavailable; no access-control bypass attempted")
            if archive.stat().st_size > 80 * 1024 * 1024:
                raise ValueError("unexpected example archive size")
            extract_zip(archive, output / "rellis")
            ontology_file = output / "rellis_labels.json"
            if ontology_file.exists() and old and old.get("ontology_sha256") and digest(ontology_file) != old["ontology_sha256"]:
                raise ValueError("RELLIS ontology cache checksum mismatch")
            if not ontology_file.exists():
                repo = "unmannedlab/RELLIS-3D"
                commit = get_json(f"https://api.github.com/repos/{repo}/commits/main")["sha"]
                data = get_json(f"https://api.github.com/repos/{repo}/contents/benchmarks/"
                    f"SalsaNext/train/tasks/semantic/config/labels/rellis.yaml?ref={commit}")
                labels = yaml.safe_load(base64.b64decode(data["content"]))["labels"]
                atomic_json(ontology_file, {"commit": commit, "labels": labels,
                    "url": f"https://github.com/{repo}/blob/{commit}/" + data["path"]})
            source.update(status="ok", sha256=digest(archive), bytes=archive.stat().st_size,
                          ontology_sha256=digest(ontology_file),
                          files=[str(p.relative_to(output)) for p in (output / "rellis").rglob("*") if p.is_file()])
        except Exception as exc:
            source.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            print(f"[download] RELLIS example failed: {exc}", flush=True)
        record["sources"] = [s for s in record["sources"] if s["name"] != source["name"]] + [source]
        atomic_json(path, record)
    print(f"provenance: {path}")
    return int(any(s["status"] != "ok" for s in record["sources"]))


if __name__ == "__main__":
    raise SystemExit(main())
