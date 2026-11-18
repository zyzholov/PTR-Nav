#!/usr/bin/env python3
"""Run reproducible Isaac navigation experiments and retain per-run evidence."""

import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src/isaac_sim_hunter_livo2/scripts'))
from navigation_courses import COMPOSITE_SCENARIOS, RAMP_SCENARIOS, CROSSING_SCENARIOS, course_distance

SCENARIOS = {"clearance", "clearance_guard", "detour", "emerge", "step", "mars",
             *COMPOSITE_SCENARIOS, *RAMP_SCENARIOS, *CROSSING_SCENARIOS}
ABLATIONS = {"full", "no_fastlio_dynamic_clear", "no_terrain_dynamic_clear", "no_live_layer", "cpu"}


def method_hashes():
    paths = (
        "scripts/sim/checks/navigation_trial.py",
        "scripts/sim/experiments/run_matrix.py",
        "scripts/sim/experiments/capability_qualification.py",
        "scripts/sim/checks/motion_grid_capture.py",
        "scripts/sim/checks/score_motion_replay.py",
        "src/planner/terrain_nav/test/motion_guard_replay.cpp",
        "src/planner/terrain_nav/test/dynamic_geometry_replay.cpp",
        "src/planner/terrain_nav/include/terrain_nav/motion_prediction.hpp",
        "src/planner/terrain_nav/include/terrain_nav/dynamic_occupancy.hpp",
        "src/planner/terrain_nav/include/terrain_nav/gaussian_motion.hpp",
        "src/planner/terrain_nav/include/terrain_nav/motion_risk.hpp",
        "src/planner/terrain_nav/include/terrain_nav/joint_motion_risk.hpp",
        "src/planner/terrain_nav/include/terrain_nav/joint_motion_kernel.hpp",
        "src/planner/terrain_nav/include/terrain_nav/motion_buffer_capacity.hpp",
        "src/planner/terrain_nav/include/terrain_nav/motion_resolution.hpp",
        "src/planner/terrain_nav/src/joint_motion_risk.cpp",
        "src/planner/terrain_nav/src/joint_motion_risk.cu",
        "src/planner/terrain_nav/test/test_gaussian_motion.cpp",
        "src/planner/terrain_nav/test/motion_screen_benchmark.cpp",
        "src/planner/terrain_nav/test/motion_observation_audit.cpp",
        "scripts/sim/checks/actuator_response_trial.py",
        "scripts/sim/checks/test_actuator_response_trial.py",
        "scripts/sim/checks/test_planning_motion_overrides.py",
        "log/neos_actuator_response_20261110/response.json",
        "log/neos_actuator_response_20261110/metadata.json",
        "log/neos_acceleration_envelope_20261110/estimate_observation_noise.py",
        "log/neos_acceleration_envelope_20261110/sim_observation_repeatability_v2.json",
        "src/planner/terrain_nav/test/data/qualified_center_before_identity.json",
        "scripts/sim/checks/clearance_metrics.py",
        "scripts/sim/checks/capture_terrain_sources.py",
        "scripts/sim/checks/capture_dynamic_rejection.py",
        "scripts/sim/checks/capture_terrain_runtime.py",
        "scripts/sim/checks/test_capture_terrain_runtime.py",
        "scripts/sim/checks/test_capture_dynamic_rejection.py",
        "scripts/sim/checks/posterior_horizon_qualification.py",
        "scripts/sim/checks/shared_certificate_qualification.py",
        "src/fsgp_bgk_se2/test/test_posterior_navigation_horizon.py",
        "src/fsgp_bgk_se2/test/test_bgk_geometry_cache.py",
        "scripts/sim/checks/test_moving_obstacle_observation.py",
        "scripts/sim/checks/test_drive_braking.py",
        "scripts/sim/checks/test_ackermann_execution.py",
        "scripts/sim/checks/terrain_ground_qualification.py",
        "scripts/sim/checks/pipeline_smoke.py",
        "scripts/sim/checks/map_rates.py",
        "scripts/sim/checks/rviz_snapshot.py",
        "scripts/sim/checks/visualization_probe.py",
        "scripts/sim/checks/test_visualization_fidelity.py",
        "scripts/sim/checks/capture_retention_rejection.py",
        "scripts/sim/checks/capture_braking_rejection.py",
        "scripts/sim/checks/capture_support_rejection.py",
        "scripts/sim/checks/navigation_progress_audit.py",
        "scripts/sim/checks/single_dynamic_audit.py",
        "scripts/sim/checks/process_ownership.py",
        "scripts/sim/checks/test_process_ownership.py",
        "scripts/sim/checks/test_single_dynamic_audit.py",
        "scripts/sim/checks/test_navigation_progress_audit.py",
        "scripts/sim/checks/serialized_header.py",
        "scripts/sim/checks/source_partition_qualification.py",
        "scripts/sim/checks/subcell_xy_qualification.py",
        "src/fsgp_bgk_se2/test/test_plane_residual.py",
        "scripts/sim/checks/test_scan_publication_order.py",
        "scripts/sim/checks/test_lidar_display.py",
        "src/planner/terrain_nav/include/terrain_nav/reverse_arc.hpp",
        "src/planner/terrain_nav/include/terrain_nav/body_kinematics.hpp",
        "src/planner/terrain_nav/include/terrain_nav/planning_budget.hpp",
        "scripts/tools/hunter_goal_probe.py",
        "scripts/tools/check_runtime_installation.py",
        "scripts/sim/checks/stale_map_trial.py",
        "scripts/sim/checks/map_fault_relay.py",
        "scripts/sim/checks/rates.sh",
        "scripts/sim/start_simulation.sh",
        "scripts/lib/runtime.sh",
        "scripts/lib/sim_navigation_profile.sh",
        "scripts/sim/config/neos_joint_sim.json",
        "scripts/sim/checks/test_sim_navigation_profile.py",
        "scripts/sim/config/fastdds_large_maps.xml",
        "scripts/sim/start_mapping.sh",
        "scripts/sim/start_terrain_analysis.sh",
        "scripts/sim/start_planning_control.sh",
        "src/fsgp_bgk_se2/fsgp_bgk_se2/bgk_seb.py",
        "src/fsgp_bgk_se2/fsgp_bgk_se2/feature_sgp.py",
        "src/fsgp_bgk_se2/fsgp_bgk_se2/prediction_worker.py",
        "src/fsgp_bgk_se2/fsgp_bgk_se2/obstacle_columns.py",
        "src/fsgp_bgk_se2/test/test_measured_floor_labels.py",
        "src/fsgp_bgk_se2/test/test_pending_column_heights.py",
        "src/fsgp_bgk_se2/test/test_column_shadow_support.py",
        "src/fsgp_bgk_se2/test/test_terrain_ingress.py",
        "src/fsgp_bgk_se2/test/test_collision_floor_witness.py",
        "src/fsgp_bgk_se2/test/test_collision_floor_returns.py",
        "src/fsgp_bgk_se2/test/data/mixed_wall_floor_returns.npz",
        "src/fsgp_bgk_se2/test/data/mixed_wall_floor_returns.json",
        "src/fsgp_bgk_se2/test/test_refinement_navigation_frontier.py",
        "src/fsgp_bgk_se2/test/test_shared_heading_symmetry.py",
        "src/fsgp_bgk_se2/test/test_resident_clearance.py",
        "src/fsgp_bgk_se2/test/data/refinement_navigation_frontier.npz",
        "src/fsgp_bgk_se2/test/data/refinement_navigation_frontier.json",
        "src/fsgp_bgk_se2/test/test_obstacle_columns.py",
        "src/fsgp_bgk_se2/test/test_subcell_cuda.py",
        "src/fsgp_bgk_se2/test/data/neos_projection_event_2.npz",
        "src/fsgp_bgk_se2/test/data/neos_projection_event_2.json",
        "src/fsgp_bgk_se2/test/data/neos_projection_event_4.npz",
        "src/fsgp_bgk_se2/test/data/neos_projection_event_4.json",
        "src/fsgp_bgk_se2/test/data/mid360_dense_pending_side.npz",
        "src/fsgp_bgk_se2/test/data/mid360_dense_pending_side.json",
        "src/fsgp_bgk_se2/test/data/mid360_dense_pending_side_noisy.npz",
        "src/fsgp_bgk_se2/test/data/mid360_dense_pending_side_noisy.json",
        "src/fsgp_bgk_se2/test/test_model_floor_covariance.py",
        "src/fsgp_bgk_se2/fsgp_bgk_se2/obstacle_columns_cuda.py",
        "src/fsgp_bgk_se2/fsgp_bgk_se2/overhead_clearance.py",
        "src/fsgp_bgk_se2/fsgp_bgk_se2/overhead_clearance_cuda.py",
        "src/fsgp_bgk_se2/test/test_overhead_clearance.py",
        "src/fsgp_bgk_se2/fsgp_bgk_se2/supported_completion.py",
        "src/fsgp_bgk_se2/fsgp_bgk_se2/fresh_completion.py",
        "src/fsgp_bgk_se2/fsgp_bgk_se2/ground_evidence.py",
        'src/fsgp_bgk_se2/fsgp_bgk_se2/shared_terrain.py',
        'src/fsgp_bgk_se2/fsgp_bgk_se2/bounded_pending.py',
        'src/fsgp_bgk_se2/test/test_bounded_pending.py',
        'src/fsgp_bgk_se2/fsgp_bgk_se2/shared_terrain_cuda.py',
        'src/fsgp_bgk_se2/test/test_shared_terrain.py',
        'src/fsgp_bgk_se2/test/test_shared_terrain_detail.py',
        'src/fsgp_bgk_se2/test/test_initial_affine_basis.py',
        'src/fsgp_bgk_se2/test/test_conditioned_residual.py',
        'scripts/sim/checks/gp_detail_qualification.py',
        'scripts/sim/checks/risk_flicker_capture.py',
        'scripts/sim/checks/risk_contrast_audit.py',
        'scripts/sim/checks/test_risk_contrast_audit.py',
        'scripts/sim/checks/test_risk_flicker_capture.py',
        'scripts/sim/checks/qualify_motion_observability.py',
        'scripts/sim/checks/summarize_risk_flicker.py',
        'scripts/sim/checks/terrain_boundary_audit.py',
        'src/fsgp_bgk_se2/test/data/mid360_ramp_initial_measured.npz',
        'src/fsgp_bgk_se2/test/data/mid360_ramp_initial_measured.json',
        'scripts/sim/checks/shared_certificate_qualification.py',
        "src/fsgp_bgk_se2/fsgp_bgk_se2/probabilistic_risk.py",
        "scripts/sim/checks/capture_ground_fault.py",
        "scripts/sim/checks/test_capture_ground_fault.py",
        "src/fsgp_bgk_se2/fsgp_bgk_se2/probability_cuda.py",
        "src/fsgp_bgk_se2/fsgp_bgk_se2/startup_support.py",
        "src/fsgp_bgk_se2/fsgp_bgk_se2/conditioned_probability.py",
        "src/fsgp_bgk_se2/fsgp_bgk_se2/mixed_terrain.py",
        "src/fsgp_bgk_se2/test/test_mixed_terrain.py",
        "src/fsgp_bgk_se2/test/test_plane_residual.py",
        "src/planner/terrain_nav/test/fixture_kinematic_reference.cpp",
        "src/fsgp_bgk_se2/fsgp_bgk_se2/display_geometry.py",
        "src/fsgp_bgk_se2/fsgp_bgk_se2/terrain_visuals.py",
        "src/fsgp_bgk_se2/rviz/hunter_terrain_navigation.rviz",
        "src/vehicle_visualizer/src/vehicle_visualizer_node.cpp",
        "src/visualization/ptr_navigation_rviz/src/status_panel.cpp",
        "src/visualization/ptr_navigation_rviz/include/ptr_navigation_rviz/status_panel.hpp",
        "install/ptr_navigation_rviz/lib/libptr_navigation_rviz.so",
        "src/uneven_mapping/se2_grid/se2_grid_converter/src/se2_grid_to_pointcloud_node.cpp",
        "src/uneven_mapping/se2_grid/se2_grid_converter/src/lidar_display_node.cpp",
        "src/uneven_mapping/se2_grid/se2_grid_converter/include/se2_grid_converter/surface_geometry.hpp",
        "src/uneven_mapping/se2_grid/se2_grid_converter/include/se2_grid_converter/display_colors.hpp",
        "install/se2_grid_converter/lib/se2_grid_converter/se2_grid_to_pointcloud_node",
        "install/se2_grid_converter/lib/se2_grid_converter/lidar_display_node",
        "src/fsgp_bgk_se2/fsgp_bgk_se2/navigation_display_node.py",
        "src/fsgp_bgk_se2/fsgp_bgk_se2/fsgp_bgk_se2_node.py",
        "src/fsgp_bgk_se2/fsgp_bgk_se2/se2_grid_tools.py",
        "src/fsgp_bgk_se2/test/test_output_map_ownership.py",
        "src/fsgp_bgk_se2/fsgp_bgk_se2/grid_cdr.py",
        "src/fsgp_bgk_se2/test/test_grid_cdr.py",
        "src/fsgp_bgk_se2/fsgp_bgk_se2/refinement_region.py",
        "src/fsgp_bgk_se2/test/test_refinement_region.py",
        "src/fsgp_bgk_se2/fsgp_bgk_se2/processing_loop.py",
        "src/fsgp_bgk_se2/fsgp_bgk_se2/bounded_egress.py",
        "src/fsgp_bgk_se2/test/test_bounded_egress.py",
        "src/fsgp_bgk_se2/test/test_processing_loop.py",
        "src/fsgp_bgk_se2/fsgp_bgk_se2/runtime_gc.py",
        "src/fsgp_bgk_se2/test/test_runtime_gc.py",
        "src/uneven_mapping/se2_grid/se2_grid_core/src/SE2Grid.cpp",
        "src/uneven_mapping/se2_grid/se2_grid_core/CMakeLists.txt",
        "src/uneven_mapping/se2_grid/se2_grid_ros/CMakeLists.txt",
        "install/se2_grid_core/lib/libse2_grid_core.a",
        "install/se2_grid_ros/lib/libse2_grid_ros.a",
        "src/uneven_mapping/terrain_analyzer/src/TerrainBank.cu",
        "src/uneven_mapping/terrain_analyzer/include/terrain_analyzer/ground_fusion.hpp",
        "src/uneven_mapping/terrain_analyzer/test/test_ground_fusion.cpp",
        "src/uneven_mapping/terrain_analyzer/test/test_ground_fusion_cuda.cu",
        "src/uneven_mapping/terrain_analyzer/test/ground_witness_cases.hpp",
        "src/uneven_mapping/terrain_analyzer/CMakeLists.txt",
        "src/uneven_mapping/terrain_analyzer/src/terrain_analyzer_gpu_node.cpp",
        "src/uneven_mapping/terrain_analyzer/src/terrain_analyzer_ros2_node.cpp",
        "src/uneven_mapping/terrain_analyzer/params/hunter_fastlio_gpu.yaml",
        "src/uneven_mapping/terrain_analyzer/params/hunter_fastlio_registered_live.yaml",
        "install/terrain_analyzer/lib/terrain_analyzer/terrain_analyzer_gpu",
        "src/uneven_mapping/se2_grid/se2_grid_core/include/se2_grid_core/SE2Grid.hpp",
        "src/uneven_mapping/se2_grid/se2_grid_ros/src/SE2GridRosConverter.cpp",
        "src/fsgp_bgk_se2/config/hunter_fastlio.yaml",
        "src/fsgp_bgk_se2/launch/terrain_analysis.launch.py",
        "src/fsgp_bgk_se2/launch/hunter_fsgp_bgk_nav.launch.py",
        "src/fsgp_bgk_se2/launch/fsgp_bgk_se2.launch.py",
        "src/FAST_LIO/config/isaac_hunter_mid360_pointcloud.yaml",
        "src/FAST_LIO/launch/mapping_hunter_sim.launch.py",
        "install/dynamic_map_filter/lib/libptr_dynamic_map.a",
        "install/dynamic_map_filter/lib/libptr_freedom.a",
        "src/FAST-LIVO2-ROS2/config/hunter_isaac_d455_sim.yaml",
        "scripts/sim/checks/test_sensor_mount.py",
        "scripts/sim/checks/lidar_mount_visibility.py",
        "src/FAST_LIO/src/laserMapping.cpp",
        "src/FAST_LIO/include/dynamic_voxel_ray.hpp",
        "install/fast_lio/lib/fast_lio/fastlio_mapping",
        "src/isaac_sim_hunter_livo2/scripts/hunter_livo2_isaac_scene.py",
        "src/isaac_sim_hunter_livo2/scripts/rtx_frame_clock.py",
        "scripts/sim/checks/test_rtx_frame_clock.py",
        "src/isaac_sim_hunter_livo2/scripts/mid360_scan.py",
        "src/isaac_sim_hunter_livo2/scripts/lidar_mount.py",
        "src/isaac_sim_hunter_livo2/scripts/wheel_control.py",
        "src/isaac_sim_hunter_livo2/scripts/vehicle_geometry.py",
        "config/vehicles/neos.json",
        "src/simulator/livox_mid360/scan_mode/mid360.csv",
        "scripts/sim/checks/mid360_scan_probe.py",
        "scripts/sim/checks/rtx_lidar_geometry.py",
        "scripts/sim/checks/replay_mid360_support.py",
        "src/isaac_sim_hunter_livo2/scripts/navigation_courses.py",
        "src/isaac_sim_hunter_livo2/scripts/boundary_courses.py",
        "scripts/sim/checks/boundary_trial.py",
        "src/FAST_LIO/src/IMU_Processing.hpp",
        "src/FAST_LIO/include/imu_initialization.hpp",
        "src/FAST_LIO/test/imu_initialization_test.cpp",
        "src/planner/terrain_nav/src/terrain_nav_node.cpp",
        "src/planner/terrain_nav/test/test_chance_navigation.cpp",
        "src/planner/terrain_nav/include/terrain_nav/chance_budget.hpp",
        "src/planner/terrain_nav/include/terrain_nav/probability_support.hpp",
        "src/planner/terrain_nav/include/terrain_nav/motion_guard.hpp",
        "src/planner/terrain_nav/include/terrain_nav/object_observation.hpp",
        "src/fsgp_bgk_se2/fsgp_bgk_se2/collision_pose_cuda.py",
        "src/fsgp_bgk_se2/test/test_collision_pose_cache.py",
        "src/fsgp_bgk_se2/test/test_fic_host_coordinates.py",
        "src/planner/terrain_nav/test/test_object_observation.cpp",
        "src/planner/terrain_nav/include/terrain_nav/temporal_geometry.hpp",
        "src/planner/terrain_nav/include/terrain_nav/mobility_recovery.hpp",
        "src/planner/terrain_nav/include/terrain_nav/nearest_sources.hpp",
        "src/planner/terrain_nav/test/benchmark_measured_columns.cpp",
        "src/planner/terrain_nav/test/test_mobility_recovery.cpp",
        "src/planner/terrain_nav/test/test_motion_guard.cpp",
        "src/planner/terrain_nav/test/data/single_crossing_identity_gap.json",
        "src/planner/terrain_nav/test/data/static_wall_endpoint_parallax.json",
        "src/planner/terrain_nav/test/data/retreat_velocity_quantization_gap.json",
        "scripts/sim/checks/boundary_cloud_evidence.py",
        "scripts/sim/checks/test_boundary_cloud_evidence.py",
        "src/planner/terrain_nav/test/data/closing_gate_translation_gap.hpp",
        "src/planner/terrain_nav/test/data/recorded_radius_collapse.hpp",
        "src/planner/terrain_nav/test/data/recorded_turn_identity_gap.json",
        "src/planner/terrain_nav/test/data/recorded_multi_identity_gap.json",
        "src/planner/terrain_nav/test/data/recorded_sparse_wall_identity.json",
        "src/planner/terrain_nav/CMakeLists.txt",
        "src/planner/terrain_nav/include/terrain_nav/route_memory.hpp",
        "src/planner/terrain_nav/test/test_route_memory.cpp",
        "src/planner/terrain_nav/test/route_memory_replay.cpp",
        "src/planner/terrain_nav/include/terrain_nav/dynamic_yield.hpp",
        "src/planner/terrain_nav/include/terrain_nav/measured_columns.hpp",
        "src/planner/terrain_nav/config/hunter_gazebo.yaml",
        "src/planner/terrain_nav/launch/fsgp_bgk_terrain_nav.launch.py",
        "install/terrain_nav/lib/terrain_nav/terrain_nav_node",
    )
    result={path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in paths if (ROOT / path).is_file()}
    for path in sorted((ROOT/'src/dynamic_map_filter').rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts:
            result[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted((ROOT/'scripts/sim/experiments/boundary_cases').glob('*.json')):
        result[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def parse_case(value):
    scenario, separator, count = value.partition(":")
    if scenario not in SCENARIOS or not separator or not count.isdecimal() or int(count) < 1:
        raise argparse.ArgumentTypeError("expected SCENARIO:COUNT, e.g. clearance:3")
    return scenario, int(count)


def stop_group(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def find_run_dir(text):
    match = re.search(r"\[smoke\] (?:PASS|FAIL); logs=(\S+)", text)
    if not match:
        match = re.search(r"\[smoke\] Domain=\d+; logs=(\S+)", text)
    return Path(match.group(1)) if match else None


def read_json_line(path):
    if not path or not path.exists():
        return {}
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and "scenario" in value:
                return value
    return {}


def parse_log_metrics(run_dir):
    metrics = {}
    if not run_dir:
        return metrics

    trial = read_json_line(run_dir / ("map_guard_trial.log" if (run_dir / "map_guard_trial.log").exists()
                                      else "navigation_trial.log"))
    for key in (
        "reached", "settled", "final_truth_speed_mps", "truth_error_m", "physical_obstacle_min_m", "front_stop_commands",
        "duration_s", "truth_path_length_m", "planned_path_length_m", "planned_path_points",
        "path_ratio_to_goal", "mean_curvature_1pm", "max_curvature_1pm",
        "avg_speed_mps", "max_speed_mps", "max_yaw_rate_radps",
        "tracking_error_mean_m", "tracking_error_p95_m", "tracking_error_max_m",
        "reverse_command_count", "max_abs_pitch_rad", "max_abs_roll_rad", "goal_mode",
        "risk_map_period_p95_s", "risk_map_age_p95_s", "ghost_laser_clear_delay_s",
        "ghost_laser_obstacle_clear_delay_s", "ghost_risk_clear_delay_s",
        "ghost_risk_known_clear_delay_s", "ghost_live_clear_delay_s",
        "ghost_laser_peak_before_vacate", "ghost_laser_obstacle_peak_before_vacate",
        "ghost_risk_peak_before_vacate", "static_laser_last", "live_peak_cells",
        "truth_lateral_max_m", "truth_pitch_max_rad", "truth_roll_max_rad",
        "goal_distance_m", "goal_lateral_m", "path_update_count",
    ):
        if key in trial:
            metrics[key] = trial[key]

    planning = (run_dir / "planning.log").read_text(errors="replace") if (run_dir / "planning.log").exists() else ""
    planner_ms = [float(value) for value in re.findall(r"planner: result=\w+ reason=.*? elapsed=([0-9.]+)ms", planning)]
    control_ms = [float(value) for value in re.findall(r"control timing: ([0-9.]+)ms", planning)]
    live_replans = len(re.findall(r"planner: result=\w+ reason=new live obstacle", planning))
    metrics["planner_count"] = len(planner_ms)
    metrics["planner_mean_ms"] = sum(planner_ms) / len(planner_ms) if planner_ms else None
    metrics["planner_p95_ms"] = percentile(planner_ms, 0.95)
    metrics["control_mean_ms"] = sum(control_ms) / len(control_ms) if control_ms else None
    metrics["control_p95_ms"] = percentile(control_ms, 0.95)
    metrics["live_replan_count"] = live_replans

    rates = (run_dir / "rates.log").read_text(errors="replace") if (run_dir / "rates.log").exists() else ""
    for topic, name in (("/livox/lidar", "lidar"), ("/imu/data_raw", "imu"), ("/livox/imu", "livox_imu")):
        section_match = re.search(
            r"\[hunter_isaac_rates\] Measuring " + re.escape(topic) +
            r".*?(?=\n\[hunter_isaac_rates\] Measuring |\Z)", rates, re.S)
        block = section_match.group(0) if section_match else ""
        for field, suffix in (("average wall rate", "wall_hz"), ("average stamp rate", "stamp_hz"),
                              ("point count min/mean/max", "points")):
            match = re.search(re.escape(field) + r": ([0-9.]+)(?:/([0-9.]+)/([0-9.]+))?", block)
            if match:
                if suffix == "points":
                    metrics[f"{name}_mean_points"] = float(match.group(2) or match.group(1))
                else:
                    metrics[f"{name}_{suffix}"] = float(match.group(1))

    map_rates = (run_dir / "map_rates.log").read_text(errors="replace") if (run_dir / "map_rates.log").exists() else ""
    for topic, name in (("terrain_analyzer/fused_map", "terrain"), ("fsgp_bgk/fused_map", "fsgp")):
        match = re.search(re.escape(topic) + r": samples=\d+ wall_rate=([0-9.]+)Hz fresh_stamps=\d+ fresh_rate=([0-9.]+)Hz", map_rates)
        if match:
            metrics[f"{name}_wall_hz"] = float(match.group(1))
            metrics[f"{name}_fresh_hz"] = float(match.group(2))
    terrain = (run_dir / "terrain.log").read_text(errors="replace") if (run_dir / "terrain.log").exists() else ""
    gp_times = re.findall(r"feature_sgp: device=(\w+) fit=([0-9.]+)ms predict=([0-9.]+)ms", terrain)
    if gp_times:
        metrics["gp_devices"] = sorted({device for device, _, _ in gp_times})
        metrics["gp_logged_samples"] = len(gp_times)
        metrics["gp_fit_logged_p95_ms"] = percentile([float(fit) for _, fit, _ in gp_times], 0.95)
        metrics["gp_predict_logged_p95_ms"] = percentile([float(pred) for _, _, pred in gp_times], 0.95)
    metrics["gp_failure_logs"] = terrain.count("feature_sgp failed; retaining")
    return metrics


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    index = min(len(values) - 1, max(0, int(round(fraction * (len(values) - 1)))))
    return round(values[index], 4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", type=parse_case, required=True,
                        help="SCENARIO:COUNT; repeat this option for multiple scenarios")
    parser.add_argument("--ablation", choices=sorted(ABLATIONS), default="full")
    parser.add_argument("--first-domain", type=int, default=60)
    parser.add_argument("--domain-count", type=int, default=8,
                        help="reuse a bounded domain pool after each isolated run shuts down")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--trial-timeout", type=float, default=45.0)
    parser.add_argument("--reverse-goal", action="store_true",
                        help="send the navigation trial goal behind the vehicle")
    parser.add_argument("--prediction-backend", choices=("none", "feature_sgp"), default="none")
    parser.add_argument("--gp-refine-observed", action="store_true")
    parser.add_argument("--gp-complete-supported",action="store_true")
    parser.add_argument('--probability-model', choices=('marginal', 'shared_gp'), default='marginal')
    parser.add_argument('--condition-full-residual',action='store_true')
    parser.add_argument('--startup-support', choices=('unknown', 'stable_stance'), default='unknown')
    parser.add_argument('--gp-support-distance', type=float, default=.35)
    parser.add_argument('--reverse-recovery', action='store_true')
    parser.add_argument('--dds-publication-mode',choices=('default','asynchronous','synchronous','auto'),default='default')
    parser.add_argument('--ramp-direction',choices=('uphill','downhill'),default='uphill')
    parser.add_argument('--ramp-rise',type=float,default=.45)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--vary-goals", action="store_true", help="seeded distance and lateral goal variations")
    parser.add_argument("--resume", action="store_true", help="continue missing jobs; retain failed runs")
    parser.add_argument("--dry-run", action="store_true", help="print job plan without launching Isaac")
    args = parser.parse_args()
    if args.gp_complete_supported and args.prediction_backend!="feature_sgp":
        parser.error("--gp-complete-supported requires feature_sgp")
    if args.probability_model == 'shared_gp' and not args.gp_complete_supported:
        parser.error('shared_gp requires explicit GP completion')
    if args.condition_full_residual and args.probability_model!='shared_gp':
        parser.error('--condition-full-residual requires shared_gp')
    if not 0 < args.gp_support_distance <= 1.25 or (args.gp_support_distance != .35 and not args.gp_complete_supported):
        parser.error('GP support distance must be in (0,1.25] with completion enabled')
    if args.gp_support_distance>.75 and args.probability_model!='shared_gp':
        parser.error('support beyond .75m requires --probability-model shared_gp')
    cases = [scenario for repeat in range(max(count for _, count in args.case))
             for scenario, count in args.case if repeat < count]
    if not math.isfinite(args.ramp_rise) or not .1<=args.ramp_rise<=1.2:
        parser.error('ramp rise must be finite and within [0.1,1.2] metres')
    if args.ramp_direction=='downhill' and any(case not in RAMP_SCENARIOS for case in cases):
        parser.error('downhill qualification requires ramp scenarios only')
    if (not cases or not 1 <= args.first_domain <= 100 or args.domain_count < 1
            or args.first_domain + args.domain_count - 1 > 100):
        parser.error("use a free ROS domain range inside 1..100")
    if args.reverse_goal and any(case in (*COMPOSITE_SCENARIOS,'ramp_crossing') for case in cases):
        parser.error('composite course qualification requires forward goals beyond the complete course')

    rng = random.Random(args.seed)
    jobs = []
    for index, scenario in enumerate(cases, 1):
        distance = 9.0 if scenario in ("emerge", "crossing", "clearance") else course_distance(scenario)
        lateral = 0.0
        if args.vary_goals and scenario not in ("mars", "clearance_guard", *COMPOSITE_SCENARIOS, 'ramp_crossing'):
            distance += rng.uniform(-0.3, 0.6)
            lateral = rng.uniform(-0.4, 0.4)
        jobs.append({"index": index, "scenario": scenario,
                     "domain": args.first_domain + (index - 1) % args.domain_count,
                     "goal_distance_m": round(distance, 4), "goal_lateral_m": round(lateral, 4)})
    settings = {key: getattr(args, key) for key in (
        "case", "ablation", "first_domain", "domain_count", "timeout", "trial_timeout",
        "reverse_goal", "seed", "vary_goals", "prediction_backend", "gp_refine_observed", "gp_complete_supported",
        'probability_model', 'startup_support', 'gp_support_distance', 'reverse_recovery', 'dds_publication_mode',
        'ramp_direction','ramp_rise','condition_full_residual')}
    settings = json.loads(json.dumps(settings))
    settings["method_hashes"] = method_hashes()
    if args.dry_run:
        print(json.dumps({"settings": settings, "jobs": jobs}, indent=2))
        return 0

    experiment_dir = args.output or (ROOT / "log" / ("experiment_matrix_" + time.strftime("%Y%m%d_%H%M%S")))
    experiment_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = experiment_dir / "manifest.json"
    manifest = {
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "study_type": "our_method_baseline",
        "method": "FAST-LIO2 + GPU terrain analyzer + FSGP-BGK/SEB + terrain_nav",
        "ablation": args.ablation,
        "cases": args.case,
        "settings": settings,
        "jobs": jobs,
        "runs": [],
    }
    if args.resume:
        if not manifest_path.exists():
            parser.error("--resume requires an existing manifest.json")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("settings") != settings or manifest.get("jobs") != jobs:
            parser.error("resume configuration must match the recorded experiment")
    elif manifest_path.exists():
        parser.error("output already contains a manifest; use --resume or a new directory")

    def checkpoint():
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(manifest_path)

    checkpoint()
    completed = {row["index"] for row in manifest["runs"]}

    for job in jobs:
        index, scenario, domain = job["index"], job["scenario"], job["domain"]
        if index in completed:
            continue
        if method_hashes() != settings["method_hashes"]:
            raise RuntimeError("method changed during the batch; start a new experiment directory")
        command = [sys.executable, "scripts/sim/checks/pipeline_smoke.py",
                   "--domain", str(domain), "--timeout", str(args.timeout),
                   "--trial-timeout", str(args.trial_timeout)]
        command.extend(["--prediction-backend", args.prediction_backend])
        command.extend(['--probability-model', args.probability_model, '--startup-support', args.startup_support,
                        '--gp-support-distance', str(args.gp_support_distance),
                        '--dds-publication-mode', args.dds_publication_mode])
        if args.reverse_recovery:
            command.append('--reverse-recovery')
        if args.condition_full_residual:command.append('--condition-full-residual')
        if args.gp_refine_observed:
            command.append("--gp-refine-observed")
        if args.gp_complete_supported:command.append("--gp-complete-supported")
        if scenario == "mars":
            command.append("--skip-goal")
        else:
            command.extend(["--trial-scenario", scenario])
        if scenario in RAMP_SCENARIOS:
            command.extend(['--ramp-direction',args.ramp_direction,'--ramp-rise',str(args.ramp_rise)])
        if scenario not in ("mars", "clearance_guard"):
            command.extend(["--trial-distance", str(job["goal_distance_m"]),
                            "--goal-lateral", str(job["goal_lateral_m"])])
        if args.reverse_goal:
            command.append("--reverse-goal")
        if args.ablation == "cpu":
            command.extend(["--terrain-backend", "cpu"])
        if args.ablation == "no_fastlio_dynamic_clear":
            command.extend(["--dynamic-clear-enabled", "false"])
        if args.ablation == "no_terrain_dynamic_clear":
            command.extend(["--terrain-dynamic-clear-enabled", "false"])
        if args.ablation == "no_live_layer":
            command.append("--disable-live-obstacles")

        output_file = experiment_dir / f"{index:02d}_{args.ablation}_{scenario}.log"
        print(f"[experiment] {index}/{len(cases)} ablation={args.ablation} scenario={scenario} domain={domain}", flush=True)
        started = time.monotonic()
        with output_file.open("w") as output:
            process = subprocess.Popen(command, cwd=ROOT, stdout=output,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                returncode = process.wait(timeout=args.timeout * 6 + args.trial_timeout + 180.0)
            except subprocess.TimeoutExpired:
                stop_group(process)
                returncode = -1
            except KeyboardInterrupt:
                stop_group(process)
                checkpoint()
                return 130
        text = output_file.read_text(errors="replace")
        run_dir = find_run_dir(text)
        row = {
            "index": index,
            "ablation": args.ablation,
            "scenario": scenario,
            "domain": domain,
            "passed": returncode == 0,
            "returncode": returncode,
            "pipeline_wall_duration_s": round(time.monotonic() - started, 2),
            "goal_seed": args.seed,
            "goal_distance_m": job["goal_distance_m"], "goal_lateral_m": job["goal_lateral_m"],
            "command": command,
            "experiment_log": str(output_file),
            "run_dir": str(run_dir) if run_dir else None,
        }
        row.update(parse_log_metrics(run_dir))
        manifest["runs"].append(row)
        checkpoint()
        print(f"[experiment] {'PASS' if row['passed'] else 'FAIL'} run_dir={run_dir}", flush=True)

    passed = sum(bool(row["passed"]) for row in manifest["runs"])
    print(f"[experiment] {passed}/{len(manifest['runs'])} passed; output={experiment_dir}", flush=True)
    return 0 if passed == len(manifest["runs"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
