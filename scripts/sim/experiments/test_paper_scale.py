import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parent))
from fetch_paper_replacements import select_frames
from paper_clouds import grid, height_at, raster
from paper_road_clouds import lidar_poses
from paper_scale_evaluate import partial_points, risk_metrics, complete_generated_collision, footprint_reference
from terrain_risk_benchmark import params_from_config


def test_road_selection_is_spatially_separated_not_consecutive_frames():
    poses=np.zeros((300,12));poses[:,3]=np.arange(300)*2
    selected=select_frames(poses,20)
    assert len(selected)==20 and len(set(selected))==20
    coordinates=poses[selected,3]
    distances=np.abs(coordinates[:,None]-coordinates[None,:])+np.eye(20)*10000
    assert distances.min()>=12


def test_camera_to_lidar_conjugation_preserves_identity(tmp_path):
    calibration=tmp_path/"calib.txt";poses=tmp_path/"poses.txt"
    calibration.write_text("Tr: 0 -1 0 1 1 0 0 2 0 0 1 3\n")
    np.savetxt(poses,np.eye(4)[:3].reshape(1,12))
    np.testing.assert_allclose(lidar_poses(calibration,poses)[0],np.eye(4))


def test_metric_grid_centers_and_interpolation_keep_xy_contract():
    xy=grid((20,20),[-2,-2],.2);z=.1*xy[:,0]-.2*xy[:,1]+.3
    height=raster(np.c_[xy,z],(20,20),[-2,-2],.2)
    np.testing.assert_allclose(height_at(height,[-2,-2],.2,xy),z,atol=1e-6)


def test_three_conditions_remove_points_before_any_model_features():
    xy=grid((104,104),[-10.4,-10.4],.2);points=np.c_[xy,np.sin(xy[:,0])]
    point15=partial_points(points,"point15",[-10.4,-10.4],33)
    assert .83<point15.mean()<.87
    np.testing.assert_array_equal(point15,partial_points(points,"point15",[-10.4,-10.4],33))
    hole=partial_points(points,"hole16",[-10.4,-10.4],33)
    assert not hole[(np.abs(xy)<.8).all(axis=1)].any()
    frontier=partial_points(points,"frontier50",[-10.4,-10.4],33)
    assert (points[frontier,0]<=0).all()


def test_risk_statistics_are_dimensionless_and_empty_truth_is_not_zero_error():
    predicted=np.array([[.9,.1]]);truth=np.array([[1.,0.]]);unsafe=np.array([[True,False]])
    row=risk_metrics(predicted,truth,unsafe,np.ones((1,2),bool))
    assert row["risk_mae"]==pytest.approx(.1) and row["tp"]==1 and row["tn"]==1
    empty=risk_metrics(predicted,truth,unsafe,np.zeros((1,2),bool))
    assert empty["risk_mae"] is None and empty["reference_states"]==0


def test_canopy_reference_uses_exported_3d_shape_at_bumper_height():
    case={"truth_height":np.zeros((100,100)),"truth_collision":np.zeros((100,100),bool),
          "truth_origin":np.array([-5.,-5.]),"truth_resolution":.1}
    item={"source_kind":"epfl_shader_geometry","entities":[{"type":"tree","center":[0,0],"height":1.8}]}
    near=complete_generated_collision(case,item)
    assert near[49,49] and not near[10,10]
    item["entities"][0]["height"]=3
    assert not complete_generated_collision(case,item).any()


def test_dense_footprint_reference_recognizes_true_slope_and_uses_yaw_centers():
    xy=grid((240,240),[-12,-12],.1)
    case={"truth_height":(.17*xy[:,0]).reshape(240,240),"truth_collision":np.zeros((240,240),bool),
          "truth_origin":np.array([-12.,-12.]),"truth_resolution":.1,"shape":np.array([104,104])}
    query=grid((104,104),[-10.4,-10.4],.2);p=params_from_config("cpu",.2,36)
    _,ids,scores,unsafe,supported=footprint_reference(case,query,p,22,samples=12)
    assert supported.all() and not unsafe.any()
    yaw=.5*2*np.pi/36
    expected=max(np.clip((np.arctan(.17*np.cos(yaw))-p.pitch_free_rad)/(p.max_pitch_rad-p.pitch_free_rad),0,1),
        np.clip((np.arctan(.17*np.sin(yaw))-p.roll_free_rad)/(p.max_roll_rad-p.roll_free_rad),0,1))
    np.testing.assert_allclose(scores[0],expected,atol=1e-6)


def test_statistics_do_not_count_repeated_room_clouds_as_independent_worlds():
    from plot_paper_scale import clustered_summary
    rows=[{"status":"ok","cluster_id":"room_a","error":1.}]*100
    rows += [{"status":"ok","cluster_id":"room_b","error":0.}]
    rows += [{"status":"failed","cluster_id":"room_c","error":0.}]
    summary=clustered_summary(rows,"error",resamples=200)
    assert summary["mean"]==.5 and summary["clusters"]==2
    assert summary["cases"]==101


def test_confusion_ratios_pool_within_sequence_and_keep_empty_denominators():
    from plot_paper_scale import ratio_rows
    rows=[{"status":"ok","reference_states":100,"cluster_id":"sequence_a",
           "tp":10,"fn":0,"fp":90,"tn":0},
          {"status":"ok","reference_states":100,"cluster_id":"sequence_a",
           "tp":0,"fn":90,"fp":0,"tn":10},
          {"status":"ok","reference_states":100,"cluster_id":"sequence_b",
           "tp":0,"fn":0,"fp":0,"tn":100}]
    values=ratio_rows(rows)
    assert values[0]["recall"]==.1 and values[0]["false_hazard"]==.9
    assert values[1]["recall"] is None and values[1]["false_hazard"]==0


def test_final_plots_reject_partial_source_campaign():
    from plot_paper_scale import validate_campaign
    record={"settings":{"selected_ids":["one"],"conditions":["point15","hole16","frontier50"]},
        "planned_cases":3,"runs":[{"id":"one__"+c} for c in ("point15","hole16","frontier50")]}
    catalog={"datasets":[{"id":"one"},{"id":"two"}]}
    with pytest.raises(ValueError,match="partial source"):
        validate_campaign(record,catalog)
    validate_campaign(record,catalog,allow_preflight=True)


def test_final_plots_reject_duplicate_or_missing_runs():
    from plot_paper_scale import validate_campaign
    record={"settings":{"selected_ids":["one"],"conditions":["point15","hole16","frontier50"]},
        "planned_cases":3,"runs":[{"id":"one__point15"}]*3}
    with pytest.raises(ValueError,match="incomplete or duplicate"):
        validate_campaign(record,{"datasets":[{"id":"one"}]})
