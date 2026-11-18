import io
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parent))
from paper_road_clouds import lidar_poses, registration_poses


def fixture_data(tmp_path):
    folder=tmp_path/"kitti_metadata/dataset/sequences/08";folder.mkdir(parents=True)
    tr=np.array([[0.,-1,0,1],[1,0,0,2],[0,0,1,3],[0,0,0,1]])
    calibration=folder/"calib.txt"
    calibration.write_text("Tr: "+" ".join(map(str,tr[:3].ravel()))+"\n")
    target=np.eye(4);target[:3,3]=[4,5,.1]
    camera=tr@target@np.linalg.inv(tr)
    content=" ".join(map(str,camera[:3].ravel()))+"\n"
    return calibration,target,content


def test_camera_poses_are_conjugated_once_into_lidar_frame(tmp_path):
    calibration,target,content=fixture_data(tmp_path)
    np.testing.assert_allclose(lidar_poses(calibration,io.StringIO(content))[0],target,atol=1e-12)


def test_semantic_pose_archive_and_provenance_are_explicit(tmp_path):
    _,target,content=fixture_data(tmp_path)
    with zipfile.ZipFile(tmp_path/"semantic_kitti_labels.zip","w") as z:
        z.writestr("dataset/sequences/08/poses.txt",content)
    poses,provenance=registration_poses(tmp_path,8)
    np.testing.assert_allclose(poses[0],target,atol=1e-12)
    assert provenance["source"]=="semantic_suma" and len(provenance["poses_sha256"])==64
    assert provenance["member"]=="dataset/sequences/08/poses.txt"


def test_missing_semantic_poses_never_silently_falls_back_to_gps(tmp_path):
    fixture_data(tmp_path)
    with pytest.raises(FileNotFoundError):registration_poses(tmp_path,8)
    with pytest.raises(ValueError,match="unknown"):registration_poses(tmp_path,8,"unregistered")
