"""Check authoring geometry against the actual mapper configuration files."""
import ast
import json
from pathlib import Path

import numpy as np
import yaml
import sys
import pytest


ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src/isaac_sim_hunter_livo2/scripts'))
from mid360_scan import Mid360ScanTable, SHOTS_PER_FRAME
from lidar_mount import (DEFAULT_PITCH_DEG, LIDAR_FRAME_TRANSLATE, LIVOX_BASE_TRANSLATE,
                         IMU_LINK_TRANSLATE, lidar_to_imu)


def test_isaac_mount_and_both_mapper_extrinsics_describe_one_rigid_transform():
    tree=ast.parse((ROOT/'src/isaac_sim_hunter_livo2/scripts/hunter_livo2_isaac_scene.py').read_text())
    names={'CAMERA_LINK_TRANSLATE','CAMERA_OPTICAL_OFFSET_X'}
    values={target.id:ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign)
            for target in node.targets if isinstance(target,ast.Name) and target.id in names}
    theta=np.deg2rad(DEFAULT_PITCH_DEG);c,s=np.cos(theta),np.sin(theta)
    r=np.array([[c,0,s],[0,1,0],[-s,0,c]])
    lidar=np.array(LIVOX_BASE_TRANSLATE)+r@LIDAR_FRAME_TRANSLATE
    imu=np.array(IMU_LINK_TRANSLATE);camera=np.array(values['CAMERA_LINK_TRANSLATE'])
    camera[0]+=values['CAMERA_OPTICAL_OFFSET_X']
    lio=yaml.safe_load((ROOT/'src/FAST_LIO/config/isaac_hunter_mid360_pointcloud.yaml').read_text())['/**']['ros__parameters']['mapping']
    livo=yaml.safe_load((ROOT/'src/FAST-LIVO2-ROS2/config/hunter_isaac_d455_sim.yaml').read_text())['/**']['ros__parameters']['extrin_calib']
    for config in (lio,livo):
        np.testing.assert_allclose(np.array(config['extrinsic_R']).reshape(3,3),r,atol=1e-9)
        np.testing.assert_allclose(config['extrinsic_T'],lidar-imu,atol=1e-9)
    optical=np.array([[0.,-1.,0.],[0.,0.,-1.],[1.,0.,0.]])
    np.testing.assert_allclose(np.array(livo['Rcl']).reshape(3,3),optical@r,atol=1e-9)
    np.testing.assert_allclose(livo['Pcl'],optical@(lidar-camera),atol=1e-9)
    # A MID360 mounted at +25deg does NOT illuminate the rear ground.
    # Preserve this physical asymmetry instead of asserting fake 360deg floor.
    down=np.deg2rad(-7.)
    assert np.sin(down-theta)<0 and np.sin(down+theta)>0
    np.testing.assert_allclose(theta,np.deg2rad(25.))
    for generated, stored in zip(lidar_to_imu(DEFAULT_PITCH_DEG), (lio['extrinsic_R'], lio['extrinsic_T'])):
        np.testing.assert_allclose(generated, stored, atol=1e-9)


def test_alternate_mount_changes_only_isaac_scene_pitch_and_lidar_to_imu(tmp_path):
    from pipeline_smoke import (ISAAC_MAPPING_CONFIG, parse_args, simulation_start_command,
                                write_mount_mapping_config)

    args = parse_args(['--lidar-mount-pitch', '5'])
    command = simulation_start_command(args)
    assert command[command.index('--lidar-mount-pitch')+1] == '5.0'
    assert write_mount_mapping_config(tmp_path, None) == ISAAC_MAPPING_CONFIG
    config = yaml.safe_load(write_mount_mapping_config(tmp_path, args.lidar_mount_pitch).read_text())
    baseline = yaml.safe_load(ISAAC_MAPPING_CONFIG.read_text())
    c, s = np.cos(np.deg2rad(5.)), np.sin(np.deg2rad(5.))
    mapping = config['/**']['ros__parameters']['mapping']
    np.testing.assert_allclose(mapping['extrinsic_R'], [c, 0, s, 0, 1, 0, -s, 0, c], atol=1e-9)
    np.testing.assert_allclose(mapping['extrinsic_T'], [-.012*c+.047*s, 0, .045+.012*s+.047*c], atol=1e-9)
    for key in ('extrinsic_R', 'extrinsic_T'):
        baseline['/**']['ros__parameters']['mapping'][key] = mapping[key]
    assert config == baseline


def test_default_scan_is_csv_solid_state_with_a_bounded_tick_and_asymmetric_ground_coverage():
    tree=ast.parse((ROOT/'src/isaac_sim_hunter_livo2/scripts/hunter_livo2_isaac_scene.py').read_text())
    defaults={t.id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
              for t in n.targets if isinstance(t,ast.Name) and t.id in
              ('DEFAULT_MAPPING_LIDAR_QUALITY','DEFAULT_VISUAL_LIDAR_QUALITY')}
    assert set(defaults.values())=={'mid360_csv'}
    profile=json.loads((ROOT/'src/isaac_sim_hunter_livo2/assets/lidar_configs/LivoxMid360CsvSolidState10hz.json').read_text())['profile']
    state=profile['emitterStates'][0]
    assert profile['scanType']=='solidState' and profile['numberOfEmitters']==20000
    assert len(state['azimuthDeg'])==len(state['elevationDeg'])==len(state['fireTimeNs'])==20000
    times=np.asarray(state['fireTimeNs']);assert times[0]==0 and np.all(np.diff(times)>0)
    assert times[-1]<1e9/profile['reportRateBaseHz']
    az=np.deg2rad(state['azimuthDeg']);el=np.deg2rad(state['elevationDeg']);theta=np.deg2rad(25)
    z=np.cos(theta)*np.sin(el)-np.sin(theta)*np.cos(el)*np.cos(az)
    # Data, not painted rings: downward ground intersections exist forward,
    # but the physical rear-ground blind region is preserved.
    assert np.count_nonzero((np.cos(az)>.5)&(z<0))>100
    assert np.count_nonzero((np.cos(az)<-.5)&(z<0))==0
    assert len(np.unique(np.round(state['elevationDeg'],3)))>1000


def test_rviz_live_cloud_keeps_one_actual_frame_not_overlapping_decay_frames():
    config=yaml.safe_load((ROOT/'src/fsgp_bgk_se2/rviz/hunter_terrain_navigation.rviz').read_text())
    display=next(d for d in config['Visualization Manager']['Displays'] if d['Name']=='Live LiDAR Returns (Range Color)')
    assert display['Decay Time']==0
    assert display['Topic']['Value']=='/terrain_display/lidar_returns'


def test_recorded_scan_advances_all_phases_without_exceeding_emission_rate():
    table=Mid360ScanTable()
    assert table.frame_count==40
    assert table.azimuth.shape==table.elevation.shape==(40,SHOTS_PER_FRAME)
    assert list(table.fire_time_ns[:3])==[0,5000,10000]
    assert table.fire_time_ns[-1]==99_995_000
    assert [table.phase_at(t) for t in (0,.099999,.1,.2,3.9,4.,4.1)]==[0,0,1,2,39,0,1]
    assert not np.array_equal(table.azimuth[0],table.azimuth[1])
    # Later phases must add genuine angular directions, not merely reorder a frame.
    cells=lambda i:set(zip(np.round(table.azimuth[i]*4).astype(int),np.round(table.elevation[i]*4).astype(int)))
    first=cells(0); union=set().union(*(cells(i) for i in range(40)))
    assert len(union)>5*len(first)
    for time in (-1,float('nan'),float('inf')):
        with pytest.raises(ValueError):table.phase_at(time)


def test_recorded_scan_rejects_incomplete_or_noncontiguous_source(tmp_path):
    source=tmp_path/'bad.csv'
    source.write_text('Time/s,Azimuth/deg,Zenith/deg\n1,0,90\n3,0,90\n')
    with pytest.raises(ValueError):Mid360ScanTable(source)


def test_rtx_bridge_receives_spherical_gmo_with_ros_cartesian_output():
    # Guard the installed bridge's implicit spherical->XYZ conversion. The
    # actual plane raycast qualification is rtx_lidar_geometry.py.
    source=(ROOT/'src/isaac_sim_hunter_livo2/scripts/mid360_scan.py').read_text()
    assert "'elementsCoordsType': 'SPHERICAL'" in source
