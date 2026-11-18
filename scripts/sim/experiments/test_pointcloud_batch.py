import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pointcloud_batch import keep_cells, partial_input, point_cells, read_points, read_reference


def reference():
    x, y = np.meshgrid(np.arange(16) * 0.2, np.arange(16) * 0.2)
    return {"query_xy": np.column_stack((x.ravel(), y.ravel())), "height": x.ravel(),
            "shape": x.shape, "resolution": 0.2, "valid": np.ones(x.size, bool)}


@pytest.mark.parametrize("kind,severity", [("random", .6), ("block", 1.6), ("frontier", .5)])
def test_hidden_cells_are_removed_before_features(kind, severity):
    ref = reference()
    points = np.column_stack((ref["query_xy"], ref["height"]))
    keep = keep_cells(ref["query_xy"], kind, severity, 123)
    observed, mask = partial_input(points, ref, keep, 456, 0)
    ids, inside = point_cells(observed, ref)
    assert inside.all() and keep[ids].all()
    assert mask.any() and (~mask).any()
    assert not mask[~keep].any()


def test_reference_grid_requires_registered_structure(tmp_path):
    ref = reference()
    bad = ref["query_xy"].copy()
    bad[3, 0] += .1
    path = tmp_path / "bad.npz"
    np.savez(path, **{**ref, "query_xy": bad})
    with pytest.raises(ValueError, match="axis-aligned"):
        read_reference(path)


def test_structured_point_formats_and_nan_filtering(tmp_path):
    points = np.column_stack((reference()["query_xy"], reference()["height"]))
    path = tmp_path / "points.npy"
    np.save(path, np.concatenate((points, [[np.nan, 0, 0]])))
    assert len(read_points(path)) == len(points)
    o3d = pytest.importorskip("open3d")
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    assert o3d.io.write_point_cloud(str(tmp_path / "points.pcd"), cloud)
    np.testing.assert_allclose(read_points(tmp_path / "points.pcd"), points, atol=1e-6)


def test_batch_plan_is_repeatable_and_counts_worlds_not_noise(tmp_path):
    script = HERE / "pointcloud_batch.py"
    subprocess.run([sys.executable, str(script), "make-synthetic", "--output", str(tmp_path / "data"),
                    "--scenes-per-family", "1", "--size", "16"], check=True, capture_output=True)
    command = [sys.executable, str(script), "run", "--dataset", str(tmp_path / "data/dataset_manifest.json"),
               "--output", str(tmp_path / "results"), "--device", "cpu", "--dry-run"]
    first = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
    again = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
    assert first == again
    assert first["world_count"] == 5
    assert first["case_count"] == 60
    assert len({job["id"] for job in first["jobs"]}) == 60


def test_split_defaults_do_not_reuse_the_same_world(tmp_path):
    script = HERE / "pointcloud_batch.py"
    worlds = []
    for split in ("development", "calibration", "test"):
        output = tmp_path / split
        subprocess.run([sys.executable, str(script), "make-synthetic", "--output", str(output),
                        "--scenes-per-family", "1", "--size", "16", "--split", split],
                       check=True, capture_output=True)
        data = json.loads((output / "dataset_manifest.json").read_text())
        assert all(item["split"] == split for item in data["datasets"])
        worlds.append({item["geometry_seed"] for item in data["datasets"]})
    assert len(set.union(*worlds)) == 15


def test_failed_cases_remain_and_resume_does_not_rewrite(tmp_path):
    script = HERE / "pointcloud_batch.py"
    data = tmp_path / "data"
    subprocess.run([sys.executable, str(script), "make-synthetic", "--output", str(data),
                    "--scenes-per-family", "1", "--size", "16"], check=True, capture_output=True)
    np.save(data / "clouds/slope_000.npy", np.zeros((3, 3), dtype=np.float32))
    command = [sys.executable, str(script), "run", "--dataset", str(data / "dataset_manifest.json"),
               "--output", str(tmp_path / "results"), "--device", "cpu", "--repeats", "1",
               "--training-steps", "0"]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 1
    manifest = tmp_path / "results/batch_manifest.json"
    records = json.loads(manifest.read_text())
    assert len(records["runs"]) == 30
    assert sum(row["status"] == "failed" for row in records["runs"]) == 6
    original = manifest.read_bytes()
    resumed = subprocess.run(command + ["--resume"], capture_output=True, text=True)
    assert resumed.returncode == 1
    assert manifest.read_bytes() == original
    altered = subprocess.run(command + ["--resume", "--noise-std", "0.04"], capture_output=True, text=True)
    assert altered.returncode != 0
    assert manifest.read_bytes() == original
