import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_public_clouds import extract_zip
from pointcloud_batch import NoHeldOutReference, score
from public_cloud_study import (GROUND_NAMES, envelope, grid_reference, load_rellis,
                                rellis_identity, select_lower_points, semantic_targets)


def test_zip_rejects_path_traversal(tmp_path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("../escape.bin", b"bad")
    with pytest.raises(ValueError, match="unsafe"):
        extract_zip(archive, tmp_path / "output")
    assert not (tmp_path / "escape.bin").exists()


def test_kitti_labels_remain_aligned_after_invalid_point_removal(tmp_path):
    cloud = np.asarray([[1, 0, 0, .2], [np.nan, 0, 0, .1], [0, 0, 0, 0], [2, 0, 0, .1]], dtype="<f4")
    labels = np.asarray([3, 5, 31, (7 << 16) | 33], dtype="<u4")
    cloud.tofile(tmp_path / "points.bin")
    labels.tofile(tmp_path / "labels.label")
    points, kept_labels = load_rellis(tmp_path / "points.bin", tmp_path / "labels.label")
    np.testing.assert_array_equal(kept_labels, [3, 33])
    np.testing.assert_array_equal(points[:, 0], [1, 2])
    labels[:2].tofile(tmp_path / "labels.label")
    with pytest.raises(ValueError, match="lengths"):
        load_rellis(tmp_path / "points.bin", tmp_path / "labels.label")


def test_lower_envelope_extraction_cannot_use_hidden_cell_heights():
    ref = grid_reference((0, 0), 8, .2)
    points = np.column_stack((ref["query_xy"], np.zeros(64)))
    upper = points.copy()
    upper[:, 2] = 1
    cloud = np.concatenate((points, upper))
    full = select_lower_points(cloud, ref)
    observed = cloud[cloud[:, 0] < 0]
    partial = select_lower_points(observed, ref)
    np.testing.assert_array_equal(partial, full[full[:, 0] < 0])
    _, valid = envelope(partial, ref, min_hits=1)
    assert valid.sum() == 32


def test_semantic_proxy_excludes_canopy_void_and_water():
    ref = grid_reference((0, 0), 8, .2)
    ground = np.column_stack((ref["query_xy"], np.zeros(64)))
    points = np.concatenate((ground, [[0.3, 0.3, 1.0], [-0.3, -0.3, 2.5], [0.3, -0.3, .5]]))
    labels = np.concatenate((np.full(64, 3), [5, 4, 31]))
    ontology = {"labels": {"3": "grass", "5": "pole", "4": "tree", "31": "puddle"}}
    target, valid = semantic_targets(points, labels, ground, ref, ontology)
    assert target.sum() == 1
    assert "puddle" not in GROUND_NAMES and "bush" not in GROUND_NAMES
    assert valid.sum() == 63


def test_no_reference_is_not_a_zero_error_result():
    with pytest.raises(NoHeldOutReference):
        score(None, {"valid": np.ones(8, bool)}, np.ones(8, bool))


def test_same_frame_name_in_different_sequences_does_not_collide():
    one = rellis_identity(Path("00000/os1_cloud_node_kitti_bin/000000.bin"))
    two = rellis_identity(Path("00001/os1_cloud_node_kitti_bin/000000.bin"))
    assert one[0] != two[0] and one[1] != two[1]
    assert rellis_identity(Path("Rellis_3D_lidar_example/os1_cloud_node_kitti_bin/000000.bin"))[0] == "rellis_000000"


def test_public_protocol_and_source_group_count(tmp_path):
    here = Path(__file__).resolve().parent
    subprocess.run([sys.executable, str(here / "pointcloud_batch.py"), "make-synthetic",
                    "--output", str(tmp_path / "data"), "--size", "16", "--scenes-per-family", "1"],
                   check=True, capture_output=True)
    path = tmp_path / "data/dataset_manifest.json"
    catalog = json.loads(path.read_text())
    for item in catalog["datasets"]:
        item["cluster_id"] = "same_source"
    path.write_text(json.dumps(catalog))
    result = subprocess.run([sys.executable, str(here / "pointcloud_batch.py"), "run",
        "--dataset", str(path), "--output", str(tmp_path / "out"), "--protocol", "public15",
        "--repeats", "5", "--device", "cpu", "--dry-run"], check=True, capture_output=True, text=True)
    plan = json.loads(result.stdout)
    assert plan["world_count"] == 1 and plan["dataset_count"] == 5
    assert plan["case_count"] == 75
    assert {j["severity"] for j in plan["jobs"] if j["kind"] == "random"} == {.15}


def test_repeated_batches_cannot_reuse_the_same_mask_seed():
    from plot_public_clouds import verify_disjoint_mask_seeds
    row = {"dataset_id": "forest_p00", "kind": "random", "severity": .15, "mask_seed": 1}
    with pytest.raises(ValueError, match="overlapping"):
        verify_disjoint_mask_seeds([{"runs": [row]}, {"runs": [row]}])
    verify_disjoint_mask_seeds([{"runs": [row]}, {"runs": [{**row, "mask_seed": 1000}]}])


def test_generated_latex_statistics_reject_digit_command_names(tmp_path):
    from plot_public_clouds import write_numbers
    with pytest.raises(ValueError, match="ASCII letters"):
        write_numbers(tmp_path / "bad.tex", {"FitP95": "65.00"})
    write_numbers(tmp_path / "numbers.tex", {"FitPninetyfive": "65.00"})
    assert (tmp_path / "numbers.tex").read_text() == "\\newcommand{\\FitPninetyfive}{65.00}\n"
