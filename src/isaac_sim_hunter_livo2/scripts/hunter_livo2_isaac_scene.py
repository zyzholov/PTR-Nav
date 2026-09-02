#!/usr/bin/env python3
import argparse
import asyncio
import bisect
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from isaacsim import SimulationApp


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
GENERATED_DIR = REPO_ROOT / "build" / "isaac_sim_hunter_livo2"
DEFAULT_HUNTER_URDF = (
    REPO_ROOT
    / "src"
    / "simulator"
    / "fake_ugv"
    / "hunter_se"
    / "hunter_se_gazebo"
    / "xacro"
    / "hunter_se_gazebo.xacro"
)
MARS_GALE_ROOT = REPO_ROOT / "src" / "simulator" / "fake_ugv" / "maps" / "Mars_Gale"
MARS_GALE_MAP_ASSET = MARS_GALE_ROOT / "meshes" / "gale_crater_patch1.stl"
MARS_GALE_COLLISION_ASSET = MARS_GALE_ROOT / "meshes" / "gale_crater_collision_161.obj"
MARS_GALE_SAND_TEXTURE = MARS_GALE_ROOT / "materials" / "textures" / "sand_pebbles.png"
MARS_GALE_ROCK_TEXTURE = MARS_GALE_ROOT / "materials" / "textures" / "rock_texture_1.png"
MARS_SURFACE_ALBEDO_TEXTURE = PACKAGE_ROOT / "assets" / "materials" / "mars_surface_albedo.png"
MARS_SURFACE_NORMAL_TEXTURE = PACKAGE_ROOT / "assets" / "materials" / "mars_surface_normal.png"
MARS_SURFACE_ROUGHNESS_TEXTURE = PACKAGE_ROOT / "assets" / "materials" / "mars_surface_roughness.png"
CUSTOM_LIDAR_CONFIG_DIR = PACKAGE_ROOT / "assets" / "lidar_configs"
MARS_VISUAL_Z_OFFSET = 0.035
MARS_COLLISION_Z_OFFSET = 0.0
MARS_SURFACE_SKIN_Z_OFFSET = -0.020
MARS_FLAT_SPAWN_HINT = (14.0, 2.0)
MARS_ROUTE_WAYPOINTS = (
    (0.0, 0.0),
    (9.0, -1.7),
    (18.0, -5.0),
    (29.0, -7.2),
    (42.0, -5.4),
    (55.0, 0.5),
    (68.0, 8.5),
    (82.0, 18.0),
)
RUBICON_ROOT = REPO_ROOT / "src" / "simulator" / "fake_ugv" / "maps" / "rubicon"
RUBICON_MAP_ASSET = RUBICON_ROOT / "meshes" / "rubicon.dae"
RUBICON_DETAIL_ASSETS = {
    "rock01": RUBICON_ROOT / "meshes" / "collider_rock01.dae",
    "rock02": RUBICON_ROOT / "meshes" / "collider_rock02.dae",
    "rockpile": RUBICON_ROOT / "meshes" / "collider_rockpile.dae",
    "stump": RUBICON_ROOT / "meshes" / "collider_stump.dae",
}
THIRD_PARTY_ROOT = REPO_ROOT.parent / "third_party"
OFFROAD_NAV_ROOT = THIRD_PARTY_ROOT / "Offroad-Nav"
OFFROAD_NAV_ASSETS = {
    "easy": OFFROAD_NAV_ROOT / "assets" / "easy.usd",
    "medium": OFFROAD_NAV_ROOT / "assets" / "medium.usd",
    "hard": OFFROAD_NAV_ROOT / "assets" / "hard.usd",
    "barakuda": OFFROAD_NAV_ROOT / "assets" / "barakuda.usd",
}
OFFROAD_NAV_DEFAULT_ASSET = OFFROAD_NAV_ASSETS["hard"]
OFFROAD_NAV_DETAIL_ASSETS = {
    "rock": OFFROAD_NAV_ROOT / "terrain.generator" / "data" / "Rock_obj" / "Rock.usd",
    "rock_obj": OFFROAD_NAV_ROOT / "terrain.generator" / "data" / "Rock_obj" / "Rock.obj",
    "big_rock": OFFROAD_NAV_ROOT / "terrain.generator" / "data" / "Rock_obj" / "big_rock.usd",
    "douglas_fir": OFFROAD_NAV_ROOT / "terrain.generator" / "data" / "Douglas_Fir" / "Douglas_Fir.usd",
    "grass": OFFROAD_NAV_ROOT / "terrain.generator" / "data" / "Grass_Short_B" / "Grass_Short_B.usd",
}
DEFAULT_MAP_ASSET = None
DEFAULT_LIVOX_VISUAL = (
    REPO_ROOT / "src" / "simulator" / "livox_mid360" / "new" / "livox_mid40" / "meshes" / "mid360.dae"
)
LIDAR_QUALITY_CONFIGS = {
    "realtime": "OS1_REV6_32ch10hz512res",
    "balanced": "OS1_REV6_32ch10hz1024res",
    "dense": "OS1_REV6_32ch10hz2048res",
    "mapping_dense": "OS1_REV6_128ch10hz512res",
    "mid360": "LivoxMid360Surrogate10hz",
    "mid360_csv": "LivoxMid360CsvSolidState10hz",
}
DEFAULT_VISUAL_LIDAR_QUALITY = "balanced"
DEFAULT_MAPPING_LIDAR_QUALITY = "mapping_dense"
DEFAULT_WORLD_SIZE = 150.0
MILI_TECH_ROOT = REPO_ROOT / "src" / "simulator" / "fake_ugv" / "maps" / "mili_tech"
GRASS_TEXTURE = MILI_TECH_ROOT / "grass_plane" / "materials" / "textures" / "grass_dry.png"
DIGITAL_WALL_TEXTURE = MILI_TECH_ROOT / "digital_wall" / "materials" / "textures" / "digital_wall2.jpg"
HOUSE_ASSET = MILI_TECH_ROOT / "house_1" / "meshes" / "house_1.dae"
FIRE_STATION_ASSET = MILI_TECH_ROOT / "fire_station" / "meshes" / "fire_station.dae"
STOP_SIGN_ASSET = MILI_TECH_ROOT / "stop_sign" / "meshes" / "stop_sign.dae"
PINE_TREE_ASSET = MILI_TECH_ROOT / "pine_tree" / "meshes" / "pine_tree.dae"
OAK_TREE_ASSET = MILI_TECH_ROOT / "oak_tree" / "meshes" / "oak_tree.dae"
TANK1_ASSET = MILI_TECH_ROOT / "tank1" / "model.dae"
TANK2_ASSET = MILI_TECH_ROOT / "tank2" / "model.dae"
JET_ASSET = MILI_TECH_ROOT / "jet" / "model.dae"
SOLDIER_ASSET = MILI_TECH_ROOT / "soldier" / "model.dae"
OFFICER_ASSET = MILI_TECH_ROOT / "officer" / "model.dae"
FIRE_ASSET = MILI_TECH_ROOT / "fire" / "model.dae"
DEFAULT_LANDMARKS = (
    (HOUSE_ASSET, "/World/Landmarks/HouseNearLeft", (14.0, 9.5, 0.0), (0.0, 0.0, -25.0), (1.0, 1.0, 1.0)),
    (HOUSE_ASSET, "/World/Landmarks/HouseNearRight", (24.0, -10.5, 0.0), (0.0, 0.0, 155.0), (1.0, 1.0, 1.0)),
    (FIRE_STATION_ASSET, "/World/Landmarks/FireStationNorth", (42.0, 21.0, 0.0), (0.0, 0.0, 180.0), (1.0, 1.0, 1.0)),
    (FIRE_STATION_ASSET, "/World/Landmarks/FireStationSouth", (48.0, -23.0, 0.0), (0.0, 0.0, 180.0), (1.0, 1.0, 1.0)),
    (STOP_SIGN_ASSET, "/World/Landmarks/StopSignLeft", (6.0, 3.7, 0.0), (0.0, 0.0, -90.0), (1.0, 1.0, 1.0)),
    (STOP_SIGN_ASSET, "/World/Landmarks/StopSignRight", (28.0, -3.7, 0.0), (0.0, 0.0, 90.0), (1.0, 1.0, 1.0)),
    (JET_ASSET, "/World/Landmarks/JetNorth", (58.0, 26.0, 0.0), (0.0, 0.0, 180.0), (1.0, 1.0, 1.0)),
    (JET_ASSET, "/World/Landmarks/JetSouth", (61.0, -28.0, 0.0), (0.0, 0.0, 180.0), (1.0, 1.0, 1.0)),
    (OFFICER_ASSET, "/World/Landmarks/OfficerSouth", (39.0, -8.8, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    (HOUSE_ASSET, "/World/Landmarks/HouseFarNorth", (67.0, 14.0, 0.0), (0.0, 0.0, 120.0), (1.0, 1.0, 1.0)),
    (FIRE_STATION_ASSET, "/World/Landmarks/FireStationFarSouth", (72.0, -35.0, 0.0), (0.0, 0.0, 180.0), (1.0, 1.0, 1.0)),
)
TREE_LANDMARKS = (
    (PINE_TREE_ASSET, "/World/Landmarks/Trees/Pine01", (-18.0, -17.0, 0.0), (0.0, 0.0, 12.0), (1.2, 1.2, 1.2)),
    (PINE_TREE_ASSET, "/World/Landmarks/Trees/Pine02", (-12.0, 19.0, 0.0), (0.0, 0.0, 54.0), (1.1, 1.1, 1.1)),
    (PINE_TREE_ASSET, "/World/Landmarks/Trees/Pine03", (7.0, -18.0, 0.0), (0.0, 0.0, 93.0), (1.25, 1.25, 1.25)),
    (PINE_TREE_ASSET, "/World/Landmarks/Trees/Pine04", (16.0, 18.0, 0.0), (0.0, 0.0, 132.0), (1.3, 1.3, 1.3)),
    (PINE_TREE_ASSET, "/World/Landmarks/Trees/Pine05", (31.0, -19.0, 0.0), (0.0, 0.0, 175.0), (1.15, 1.15, 1.15)),
    (PINE_TREE_ASSET, "/World/Landmarks/Trees/Pine06", (39.0, 31.0, 0.0), (0.0, 0.0, 214.0), (1.35, 1.35, 1.35)),
    (PINE_TREE_ASSET, "/World/Landmarks/Trees/Pine07", (55.0, -36.0, 0.0), (0.0, 0.0, 253.0), (1.2, 1.2, 1.2)),
    (PINE_TREE_ASSET, "/World/Landmarks/Trees/Pine08", (-38.0, 25.0, 0.0), (0.0, 0.0, 292.0), (1.25, 1.25, 1.25)),
    (PINE_TREE_ASSET, "/World/Landmarks/Trees/Pine09", (-46.0, -31.0, 0.0), (0.0, 0.0, 331.0), (1.3, 1.3, 1.3)),
    (PINE_TREE_ASSET, "/World/Landmarks/Trees/Pine10", (66.0, 9.0, 0.0), (0.0, 0.0, 41.0), (1.1, 1.1, 1.1)),
    (OAK_TREE_ASSET, "/World/Landmarks/Trees/Oak01", (-8.0, 12.0, 0.0), (0.0, 0.0, 21.0), (1.1, 1.1, 1.1)),
    (OAK_TREE_ASSET, "/World/Landmarks/Trees/Oak02", (10.0, 17.0, 0.0), (0.0, 0.0, 68.0), (1.0, 1.0, 1.0)),
    (OAK_TREE_ASSET, "/World/Landmarks/Trees/Oak03", (21.0, -19.0, 0.0), (0.0, 0.0, 115.0), (1.1, 1.1, 1.1)),
    (OAK_TREE_ASSET, "/World/Landmarks/Trees/Oak04", (47.0, 14.0, 0.0), (0.0, 0.0, 162.0), (1.2, 1.2, 1.2)),
    (OAK_TREE_ASSET, "/World/Landmarks/Trees/Oak05", (63.0, -17.0, 0.0), (0.0, 0.0, 209.0), (1.05, 1.05, 1.05)),
    (OAK_TREE_ASSET, "/World/Landmarks/Trees/Oak06", (-56.0, 7.0, 0.0), (0.0, 0.0, 256.0), (1.15, 1.15, 1.15)),
    (OAK_TREE_ASSET, "/World/Landmarks/Trees/Oak07", (-26.0, -27.0, 0.0), (0.0, 0.0, 303.0), (1.0, 1.0, 1.0)),
    (OAK_TREE_ASSET, "/World/Landmarks/Trees/Oak08", (72.0, 33.0, 0.0), (0.0, 0.0, 350.0), (1.2, 1.2, 1.2)),
)


def normalize_boolean_value_args(argv, positive_to_negative):
    normalized = [argv[0]]
    skip_next = False
    truthy = {"true", "1", "yes", "on"}
    falsy = {"false", "0", "no", "off"}
    for index, item in enumerate(argv[1:], start=1):
        if skip_next:
            skip_next = False
            continue

        if "=" in item:
            name, value = item.split("=", 1)
            value_lower = value.lower()
            if name in positive_to_negative and value_lower in truthy | falsy:
                normalized.append(name if value_lower in truthy else positive_to_negative[name])
                continue

        if item in positive_to_negative and index + 1 < len(argv):
            value_lower = argv[index + 1].lower()
            if value_lower in truthy | falsy:
                normalized.append(item if value_lower in truthy else positive_to_negative[item])
                skip_next = True
                continue

        normalized.append(item)
    return normalized


def parse_args():
    normalized_argv = normalize_boolean_value_args(
        sys.argv,
        {
            "--camera-enabled": "--no-camera-enabled",
            "--lidar-full-scan": "--no-lidar-full-scan",
        },
    )
    parser = argparse.ArgumentParser(description="Hunter FAST-LIVO2 Isaac Sim sensor scene")
    parser.add_argument("--headless", action="store_true", help="Run Isaac Sim without UI")
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds to run; 0 means run until Ctrl-C")
    parser.add_argument("--renderer", default="RaytracedLighting")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=360)
    parser.add_argument("--physics-hz", type=float, default=200.0)
    parser.add_argument("--render-hz", type=float, default=10.0)
    parser.add_argument("--stationary-duration", type=float, default=45.0)
    parser.add_argument(
        "--sensor-profile",
        choices=("visual", "mapping"),
        default="visual",
        help="Use 'mapping' for stable LiDAR/IMU rates; keep 'visual' for maximum viewport quality.",
    )
    parser.add_argument(
        "--obstacle-density",
        choices=("none", "sparse", "medium", "dense"),
        default="sparse",
        help="Extra obstacle density for off-road scenes. Use dense for stress tests and sparse/medium for interactive GUI work.",
    )
    parser.add_argument(
        "--scene-profile",
        choices=("rich", "light", "rubicon", "mars_gale", "offroad", "offroad_nav"),
        default="rich",
        help=(
            "Scene complexity/profile. rich uses the project map, light is for debugging, "
            "rubicon loads the Rubicon mesh, mars_gale/offroad load the local Mars Gale crater, "
            "and offroad_nav loads the community Offroad-Nav USD terrain."
        ),
    )
    parser.add_argument("--world-size", type=float, default=DEFAULT_WORLD_SIZE, help="Meters covered by the generated project terrain")
    parser.add_argument(
        "--drive-mode",
        choices=("stationary", "circle", "cmd_vel"),
        default="stationary",
        help=(
            "Vehicle motion mode. 'stationary' keeps Hunter parked; 'circle' enables the repeatable SLAM test path; "
            "'cmd_vel' accepts geometry_msgs/Twist commands from ROS 2."
        ),
    )
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel", help="ROS 2 geometry_msgs/Twist topic used by --drive-mode cmd_vel")
    parser.add_argument(
        "--cmd-vel-backend",
        choices=("bridge", "rclpy"),
        default="rclpy",
        help="Use direct rclpy or Isaac ROS2 Bridge for /cmd_vel. rclpy is the default because it supports command timeout.",
    )
    parser.add_argument("--cmd-vel-max-speed", type=float, default=1.20, help="Speed clamp in m/s for --drive-mode cmd_vel")
    parser.add_argument("--cmd-vel-max-yaw-rate", type=float, default=1.20, help="Yaw-rate clamp in rad/s for --drive-mode cmd_vel")
    parser.add_argument("--cmd-vel-max-steer", type=float, default=0.55, help="Ackermann steering clamp in rad for --drive-mode cmd_vel")
    parser.add_argument("--cmd-vel-max-steer-rate", type=float, default=8.0, help="Steering slew-rate clamp in rad/s for --drive-mode cmd_vel")
    parser.add_argument("--cmd-vel-max-accel", type=float, default=1.80, help="Linear acceleration clamp in m/s^2 for --drive-mode cmd_vel")
    parser.add_argument("--cmd-vel-max-yaw-accel", type=float, default=1.80, help="Yaw-rate acceleration clamp in rad/s^2 for --drive-mode cmd_vel")
    parser.add_argument("--cmd-vel-linear-deadband", type=float, default=0.006, help="Treat tiny |linear.x| commands below this value as zero")
    parser.add_argument("--cmd-vel-angular-deadband", type=float, default=0.010, help="Treat tiny |angular.z| commands below this value as zero")
    parser.add_argument("--cmd-vel-min-turn-speed", type=float, default=0.02, help="Minimum |speed| before yaw-rate commands are applied")
    parser.add_argument("--cmd-vel-timeout", type=float, default=0.35, help="Stop the vehicle if no fresh /cmd_vel arrives for this many seconds")
    parser.add_argument(
        "--cmd-vel-angular-mode",
        choices=("steer", "yaw_rate"),
        default="steer",
        help="Interpret Twist.angular.z as front steering angle (Gazebo Hunter-compatible) or yaw rate.",
    )
    parser.add_argument(
        "--wheel-drive-mode",
        choices=("effort", "velocity"),
        default="velocity",
        help="Wheel joint drive backend. effort applies a Gazebo-like velocity PID torque; velocity uses Isaac PD velocity targets.",
    )
    parser.add_argument("--wheel-velocity-kp", type=float, default=14.0, help="Wheel speed error to torque gain for --wheel-drive-mode effort")
    parser.add_argument("--wheel-motor-effort", type=float, default=140.0, help="Wheel motor torque clamp in N*m")
    parser.add_argument("--wheel-idle-brake-kp", type=float, default=0.0, help="Extra wheel braking gain when speed and steering commands are zero")
    parser.add_argument("--wheel-idle-brake-effort", type=float, default=0.0, help="Wheel braking effort clamp in N*m when commands are zero")
    parser.add_argument("--wheel-velocity-damping", type=float, default=220.0, help="Isaac wheel velocity-drive damping for --wheel-drive-mode velocity")
    parser.add_argument("--steering-position-kp", type=float, default=2200.0, help="Steering joint position stiffness")
    parser.add_argument("--steering-position-kd", type=float, default=220.0, help="Steering joint position damping")
    parser.add_argument("--steering-max-effort", type=float, default=1800.0, help="Steering joint effort clamp in N*m")
    parser.add_argument("--steering-zero-snap", type=float, default=0.015, help="Snap tiny steering commands to zero radians")
    parser.add_argument("--steering-direct-zero-hold", dest="steering_direct_zero_hold", action="store_true", default=True,
                        help="Directly hold steering joints at zero for straight commands to prevent residual wheel angle")
    parser.add_argument("--no-steering-direct-zero-hold", dest="steering_direct_zero_hold", action="store_false",
                        help="Disable direct steering zero hold")
    parser.add_argument("--drive-speed-feedback-kp", type=float, default=0.0, help="Body-speed feedback gain for wheel targets in physics mode")
    parser.add_argument("--drive-yaw-feedback-kp", type=float, default=0.0, help="Body-yaw-rate feedback gain for wheel targets in physics mode")
    parser.add_argument("--physics-root-assist", dest="physics_root_assist", action="store_true", default=True,
                        help="Damp unrealistic chassis slip/yaw while still driving the Hunter through wheel joints")
    parser.add_argument("--no-physics-root-assist", dest="physics_root_assist", action="store_false",
                        help="Disable chassis slip/yaw damping and rely only on raw PhysX wheel contact")
    parser.add_argument("--physics-root-assist-mode", choices=("strict", "damped", "contact"), default="strict",
                        help="strict tracks commanded speed/yaw for controllable mapping; contact leaves forward motion to wheel contact; damped is intermediate")
    parser.add_argument("--physics-root-assist-accel", type=float, default=3.00,
                        help="Max chassis forward-speed correction in m/s^2")
    parser.add_argument("--physics-root-assist-yaw-accel", type=float, default=5.00,
                        help="Max chassis yaw-rate correction in rad/s^2")
    parser.add_argument("--physics-root-idle-brake-accel", type=float, default=3.00,
                        help="Max chassis braking correction in m/s^2 when speed command is zero")
    parser.add_argument("--physics-root-idle-stop-threshold", type=float, default=0.08,
                        help="Snap tiny body-frame forward speed to zero when there is no speed command")
    parser.add_argument("--physics-root-idle-yaw-brake", type=float, default=3.50,
                        help="Max chassis yaw braking correction in rad/s^2 when steering command is zero")
    parser.add_argument("--physics-root-lateral-damping", type=float, default=10.0,
                        help="First-order body-frame lateral slip damping rate in 1/s")
    parser.add_argument("--physics-root-roll-pitch-damping", type=float, default=3.0,
                        help="First-order body-frame roll/pitch angular damping rate in 1/s")
    parser.add_argument("--physics-straight-yaw-hold", dest="physics_straight_yaw_hold", action="store_true", default=True,
                        help="Force near-zero chassis yaw rate for straight steering commands")
    parser.add_argument("--no-physics-straight-yaw-hold", dest="physics_straight_yaw_hold", action="store_false",
                        help="Disable straight-line yaw-rate clamp")
    parser.add_argument("--physics-idle-pose-hold", dest="physics_idle_pose_hold", action="store_true", default=True,
                        help="Hold the Hunter root pose after the vehicle has settled and commands are zero")
    parser.add_argument("--no-physics-idle-pose-hold", dest="physics_idle_pose_hold", action="store_false",
                        help="Disable idle pose hold")
    parser.add_argument("--physics-idle-pose-hold-drift", type=float, default=0.012,
                        help="Meters of allowed idle root drift before the parked pose is restored")
    parser.add_argument("--physics-idle-pose-hold-speed", type=float, default=0.020,
                        help="Commanded speed threshold for parked pose hold")
    parser.add_argument("--physics-terrain-sink-guard", dest="physics_terrain_sink_guard", action="store_true", default=True,
                        help="Lift the Hunter back above Mars collision terrain if PhysX contact briefly sinks through the mesh")
    parser.add_argument("--no-physics-terrain-sink-guard", dest="physics_terrain_sink_guard", action="store_false",
                        help="Disable Mars terrain sink guard")
    parser.add_argument("--physics-terrain-min-base-clearance", type=float, default=0.235,
                        help="Minimum base_link height above Mars collision terrain before the sink guard corrects z")
    parser.add_argument("--physics-terrain-reset-base-clearance", type=float, default=0.305,
                        help="base_link height above Mars collision terrain after the sink guard corrects z")
    parser.add_argument(
        "--odom-topic",
        default="/isaac/ground_truth/odometry",
        help="ROS 2 nav_msgs/Odometry topic for measured Isaac Hunter motion. FAST-LIO keeps /Odometry.",
    )
    parser.add_argument("--odom-publish-hz", type=float, default=50.0, help="Measured odometry publish rate in Hz")
    parser.add_argument(
        "--motion-model",
        choices=("physics", "kinematic"),
        default="physics",
        help="physics drives the Hunter articulation through wheel/steering joints; kinematic directly writes the root pose for sensor debugging.",
    )
    parser.add_argument("--physics-settle-duration", type=float, default=1.5, help="Seconds to keep wheel targets at zero while the physics model settles")
    parser.add_argument("--physics-drive-log-period", type=float, default=5.0, help="Seconds between physics drive status logs")
    parser.add_argument("--realtime", action="store_true", help="Pace the loop close to wall-clock render rate")
    parser.add_argument("--imu-topic", default="/imu/data_raw", help="Primary external IMU topic")
    parser.add_argument(
        "--legacy-imu-topic",
        default="/livox/imu",
        help="Compatibility IMU topic for older Gazebo/FAST-LIO launch files. Use 'none' to disable.",
    )
    parser.add_argument(
        "--lidar-config",
        default=None,
        help="Explicit Isaac RTX LiDAR config. Overrides --lidar-quality when set.",
    )
    parser.add_argument(
        "--lidar-quality",
        choices=tuple(LIDAR_QUALITY_CONFIGS.keys()),
        default=None,
        help=(
            "RTX LiDAR preset: realtime=Ouster 32ch/512, balanced=Ouster 32ch/1024, "
            "dense=Ouster 32ch/2048, mapping_dense=Ouster 128ch/512 for live FAST-LIO mapping, "
            "mid360=local MID360 FOV/rate surrogate, "
            "mid360_csv=20k-ray solidState surrogate generated from Gazebo mid360.csv. "
            "Defaults to mapping_dense in mapping profile and balanced in visual profile."
        ),
    )
    parser.add_argument("--lidar-hz", type=float, default=10.0, help="Target LiDAR publication rate in simulation ticks")
    parser.add_argument(
        "--lidar-render-oversample",
        type=float,
        default=1.25,
        help="Mapping-mode RTX LiDAR render oversampling. 1.25 targets 12.5 render Hz for a 10 Hz LiDAR.",
    )
    parser.add_argument("--imu-hz", type=float, default=200.0, help="Target external IMU/update rate")
    parser.add_argument("--camera-hz", type=float, default=10.0, help="Target RGB/camera_info publication rate")
    parser.add_argument("--depth-hz", type=float, default=5.0, help="Target depth/depth point cloud publication rate")
    parser.add_argument(
        "--camera-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Publish D455 RGB/depth ROS topics. Defaults to off in mapping profile and on in visual profile.",
    )
    parser.add_argument(
        "--lidar-frame-skip",
        type=int,
        default=-1,
        help="Override LiDAR helper frameSkipCount; -1 derives it from --physics-hz and --lidar-hz.",
    )
    parser.add_argument(
        "--lidar-full-scan",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Publish only completed rotary scans. Defaults to off in mapping profile for throughput "
            "and on otherwise; use --lidar-full-scan for high-fidelity but slower mapping."
        ),
    )
    parser.add_argument("--show-lidar-debug", action="store_true")
    parser.add_argument("--robot-urdf", default=str(DEFAULT_HUNTER_URDF), help="Hunter SE URDF/xacro to import")
    parser.add_argument("--robot-z", type=float, default=0.33, help="Initial base height for the imported Hunter model")
    parser.add_argument("--spawn-x", type=float, default=None, help="Initial Hunter x position. Defaults to an auto-selected flat Mars point for Mars profiles.")
    parser.add_argument("--spawn-y", type=float, default=None, help="Initial Hunter y position. Defaults to an auto-selected flat Mars point for Mars profiles.")
    parser.add_argument("--spawn-yaw-deg", type=float, default=0.0, help="Initial Hunter yaw in degrees")
    parser.add_argument("--mars-auto-flat-spawn", dest="mars_auto_flat_spawn", action="store_true", default=True,
                        help="For Mars profiles, auto-select a locally flat spawn point unless --spawn-x/--spawn-y are given")
    parser.add_argument("--no-mars-auto-flat-spawn", dest="mars_auto_flat_spawn", action="store_false",
                        help="Do not auto-select a flat Mars spawn point")
    parser.add_argument(
        "--map-asset",
        default="auto" if DEFAULT_MAP_ASSET is None else str(DEFAULT_MAP_ASSET),
        help=(
            "Extra project mesh/USD asset for the Isaac scene, or 'none'. "
            "Aliases: auto, mars_gale, mars_collision, rubicon, offroad_nav, offroad_nav_easy/medium/hard/barakuda"
        ),
    )
    parser.add_argument("--no-landmarks", action="store_true", help="Skip project DAE landmark assets")
    parser.add_argument("--no-sensor-meshes", action="store_true", help="Skip decorative Livox/D455 sensor body meshes")
    parser.add_argument(
        "--sensor-visual-detail",
        choices=("cad", "procedural", "none"),
        default="cad",
        help="Sensor appearance detail. cad uses the local MID360 DAE when available; procedural uses lightweight Isaac geometry.",
    )
    parser.add_argument(
        "--livox-visual-yaw-deg",
        type=float,
        default=180.0,
        help="Yaw-only visual correction for the Livox MID360 body. This does not change the ROS LiDAR frame.",
    )
    parser.add_argument("--no-vehicle-contrast", action="store_true", help="Keep original imported robot materials instead of high-contrast Isaac materials")
    parser.add_argument("--visual-only", action="store_true", help="Build the Isaac scene for visual inspection without ROS sensor/clock graphs")
    parser.add_argument(
        "--viewer-camera",
        choices=("close", "overview", "top"),
        default="close",
        help="Initial Isaac UI camera. close starts focused on Hunter; overview/top show the full Mars terrain.",
    )
    parser.add_argument(
        "--mars-visual-detail",
        choices=("auto", "fast", "full"),
        default="auto",
        help=(
            "Mars visual terrain detail. auto uses full STL visual terrain in GUI and fast OBJ visual terrain in headless; "
            "fast uses the Gazebo collision OBJ as visible terrain; full uses the high-poly STL visual mesh."
        ),
    )
    parser.add_argument(
        "--mars-surface-skin",
        choices=("auto", "none", "local", "full"),
        default="auto",
        help="Extra textured Mars surface overlay. auto enables full visual skin in GUI and disables it for headless mapping.",
    )
    parser.add_argument("--show-start-pad", action="store_true", help="Show the artificial start pad/debug plate in off-road scenes")
    parser.add_argument("--show-route-surface", action="store_true", help="Show the artificial compacted route surface in off-road scenes")
    parser.add_argument("--show-route-guide", action="store_true", help="Show the artificial route center guide in the Mars scene")
    return parser.parse_known_args(normalized_argv[1:])


args, _unknown = parse_args()
if args.lidar_quality is None:
    args.lidar_quality = DEFAULT_MAPPING_LIDAR_QUALITY if args.sensor_profile == "mapping" else DEFAULT_VISUAL_LIDAR_QUALITY
args.lidar_quality_label = args.lidar_quality
if args.lidar_config is None:
    args.lidar_config = LIDAR_QUALITY_CONFIGS[args.lidar_quality]
else:
    args.lidar_quality_label = "custom"
if args.sensor_profile == "mapping":
    args.physics_hz = max(args.physics_hz, args.imu_hz, 200.0)
    mapping_render_hz = max(args.lidar_hz, 10.0) * max(args.lidar_render_oversample, 1.0)
    args.render_hz = max(args.render_hz, mapping_render_hz)
    args.camera_width = min(args.camera_width, 640)
    args.camera_height = min(args.camera_height, 360)
    if args.headless:
        if args.mars_visual_detail == "auto" or args.mars_visual_detail == "full":
            args.mars_visual_detail = "fast"
        if args.mars_surface_skin == "auto" or args.mars_surface_skin == "full":
            args.mars_surface_skin = "none"
        if args.sensor_visual_detail == "cad":
            args.sensor_visual_detail = "procedural"
if args.camera_enabled is None:
    args.camera_enabled = args.sensor_profile != "mapping"
if args.lidar_full_scan is None:
    args.lidar_full_scan = args.sensor_profile != "mapping"
if args.mars_visual_detail == "auto":
    args.mars_visual_detail = "fast" if args.headless else "full"
if args.mars_surface_skin == "auto":
    args.mars_surface_skin = "none" if args.headless else "full"
args.physics_hz = max(args.physics_hz, args.imu_hz, 1.0)
args.render_hz = max(args.render_hz, 1.0)
if args.lidar_frame_skip < 0:
    args.lidar_frame_skip = max(0, int(round(args.render_hz / max(args.lidar_hz, 1e-3))) - 1)
args.camera_frame_skip = max(0, int(round(args.render_hz / max(args.camera_hz, 1e-3))) - 1)
args.depth_frame_skip = max(0, int(round(args.render_hz / max(args.depth_hz, 1e-3))) - 1)
if args.drive_mode == "cmd_vel" and args.visual_only:
    print("[hunter_isaac_scene] warning: --drive-mode cmd_vel requires ROS graphs; disabling --visual-only", flush=True)
    args.visual_only = False
sys.argv = [sys.argv[0], *_unknown]

simulation_app = SimulationApp(
    {
        "headless": args.headless,
        "renderer": args.renderer,
        "width": max(320, args.camera_width),
        "height": max(240, args.camera_height),
    }
)

import carb
import omni
import omni.graph.core as og
import omni.kit.commands
import omni.replicator.core as rep
from isaacsim.core.api import SimulationContext
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.utils.types import ArticulationAction
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade


WORLD = "/World"
ROBOT = "/World/HunterSE"
BASE_LINK = f"{ROBOT}/base_link"
LIVOX_BASE = f"{BASE_LINK}/livox_base"
IMU_LINK = f"{BASE_LINK}/imu_link"
CAMERA_LINK = f"{BASE_LINK}/camera_link"
LIDAR_FRAME = "livox"
IMU_FRAME = "imu_link"
CAMERA_FRAME = "camera_color_optical_frame"
VIEW_CAMERA = f"{WORLD}/ViewerCamera"
LIVOX_BASE_TRANSLATE = (0.18, 0.0, 0.250)
LIDAR_FRAME_TRANSLATE = (-0.012, 0.0, 0.047)
IMU_LINK_TRANSLATE = (0.18, 0.0, 0.205)
CAMERA_LINK_TRANSLATE = (0.225, 0.0, 0.205)
CAMERA_OPTICAL_OFFSET_X = 0.02
WHEEL_BASE = 0.549185
TRACK_WIDTH = 0.50
WHEEL_RADIUS = 0.1349
LIVOX_MOUNT_PITCH_RAD = 0.436332313
STEERING_JOINTS = ("fr_steer_left_joint", "fr_steer_right_joint")
WHEEL_JOINT_SIGNS = {
    "fr_left_joint": 1.0,
    "fr_right_joint": 1.0,
    "re_left_joint": 1.0,
    "re_right_joint": 1.0,
}
HUNTER_PHYSICS_DOF_NAMES = (*STEERING_JOINTS, *WHEEL_JOINT_SIGNS.keys(), "steering_whee_joint")


def log(message):
    print(f"[hunter_isaac_scene] {message}", flush=True)


def create_preview_material(stage, path, color, roughness=0.75, metallic=0.0, texture_path=None):
    try:
        material = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))

        diffuse_input = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
        if texture_path and Path(texture_path).exists():
            st_reader = UsdShade.Shader.Define(stage, f"{path}/PrimvarReaderSt")
            st_reader.CreateIdAttr("UsdPrimvarReader_float2")
            st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            st_reader.CreateInput("fallback", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(0.5, 0.5))
            texture = UsdShade.Shader.Define(stage, f"{path}/DiffuseTexture")
            texture.CreateIdAttr("UsdUVTexture")
            texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(Path(texture_path).resolve())))
            texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_reader.ConnectableAPI(), "result")
            texture.CreateInput("fallback", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(color[0], color[1], color[2], 1.0))
            texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
            diffuse_input.ConnectToSource(texture.ConnectableAPI(), "rgb")
        else:
            diffuse_input.Set(Gf.Vec3f(*color))

        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return material
    except Exception as exc:
        log(f"warning: failed to create material {path}: {exc}")
        return None


def create_display_color_material(stage, path, fallback_color, roughness=0.88):
    try:
        material = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
        reader = UsdShade.Shader.Define(stage, f"{path}/DisplayColorReader")
        reader.CreateIdAttr("UsdPrimvarReader_float3")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
        reader.CreateInput("fallback", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*fallback_color))
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(reader.ConnectableAPI(), "result")
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return material
    except Exception as exc:
        log(f"warning: failed to create display-color material {path}: {exc}")
        return None


def create_mars_pbr_material(stage, path, fallback_color, albedo_path, normal_path, roughness_path):
    try:
        material = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

        st_reader = UsdShade.Shader.Define(stage, f"{path}/PrimvarReaderSt")
        st_reader.CreateIdAttr("UsdPrimvarReader_float2")
        st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        st_reader.CreateInput("fallback", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(0.5, 0.5))

        def texture_shader(name, file_path, source_color_space, fallback=None):
            if not file_path or not Path(file_path).exists():
                return None
            texture = UsdShade.Shader.Define(stage, f"{path}/{name}")
            texture.CreateIdAttr("UsdUVTexture")
            texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(Path(file_path).resolve())))
            texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_reader.ConnectableAPI(), "result")
            texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
            texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
            texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(source_color_space)
            if fallback is not None:
                texture.CreateInput("fallback", Sdf.ValueTypeNames.Float4).Set(fallback)
            return texture

        albedo = texture_shader(
            "AlbedoTexture",
            albedo_path,
            "sRGB",
            Gf.Vec4f(fallback_color[0], fallback_color[1], fallback_color[2], 1.0),
        )
        diffuse_input = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
        if albedo:
            diffuse_input.ConnectToSource(albedo.ConnectableAPI(), "rgb")
        else:
            diffuse_input.Set(Gf.Vec3f(*fallback_color))

        roughness = texture_shader("RoughnessTexture", roughness_path, "raw")
        roughness_input = shader.CreateInput("roughness", Sdf.ValueTypeNames.Float)
        if roughness:
            roughness_input.ConnectToSource(roughness.ConnectableAPI(), "r")
        else:
            roughness_input.Set(0.91)

        normal = texture_shader("NormalTexture", normal_path, "raw")
        if normal:
            shader.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(normal.ConnectableAPI(), "rgb")

        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return material
    except Exception as exc:
        log(f"warning: failed to create Mars PBR material {path}: {exc}")
        return create_preview_material(stage, path, fallback_color, roughness=0.91, texture_path=albedo_path)


def apply_physics_material(material, static_friction=1.0, dynamic_friction=0.9, restitution=0.02):
    if not material:
        return
    try:
        physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics_material.CreateStaticFrictionAttr().Set(float(static_friction))
        physics_material.CreateDynamicFrictionAttr().Set(float(dynamic_friction))
        physics_material.CreateRestitutionAttr().Set(float(restitution))
    except Exception as exc:
        prim_path = material.GetPrim().GetPath() if material else "<unknown>"
        log(f"warning: failed to apply physics material on {prim_path}: {exc}")


def make_invisible(prim_or_schema):
    prim = prim_or_schema.GetPrim() if hasattr(prim_or_schema, "GetPrim") else prim_or_schema
    try:
        UsdGeom.Imageable(prim).MakeInvisible()
    except Exception as exc:
        prim_path = prim.GetPath() if prim else "<unknown>"
        log(f"warning: failed to hide helper prim {prim_path}: {exc}")


def create_scene_materials(stage):
    stage.DefinePrim("/World/Materials", "Scope")
    materials = {
        "grass": create_preview_material(stage, "/World/Materials/Grass", (0.34, 0.43, 0.25), 0.92, texture_path=GRASS_TEXTURE),
        "asphalt": create_preview_material(stage, "/World/Materials/Asphalt", (0.08, 0.085, 0.085), 0.88),
        "lane": create_preview_material(stage, "/World/Materials/LanePaint", (0.92, 0.90, 0.78), 0.68),
        "concrete": create_preview_material(stage, "/World/Materials/Concrete", (0.47, 0.48, 0.45), 0.82),
        "soil": create_preview_material(stage, "/World/Materials/Soil", (0.29, 0.23, 0.18), 0.95),
        "sand": create_preview_material(stage, "/World/Materials/MarsSand", (0.66, 0.43, 0.25), 0.94, texture_path=MARS_GALE_SAND_TEXTURE),
        "mars_surface": create_mars_pbr_material(
            stage,
            "/World/Materials/MarsSurfacePBR",
            (0.58, 0.38, 0.23),
            MARS_SURFACE_ALBEDO_TEXTURE,
            MARS_SURFACE_NORMAL_TEXTURE,
            MARS_SURFACE_ROUGHNESS_TEXTURE,
        ),
        "route_dust": create_preview_material(stage, "/World/Materials/CompactedRouteDust", (0.50, 0.37, 0.25), 0.90, texture_path=MARS_GALE_SAND_TEXTURE),
        "mars_rock": create_preview_material(stage, "/World/Materials/MarsRock", (0.46, 0.38, 0.31), 0.88, texture_path=MARS_GALE_ROCK_TEXTURE),
        "dark_rock": create_preview_material(stage, "/World/Materials/DarkRock", (0.33, 0.29, 0.25), 0.90),
        "bark": create_preview_material(stage, "/World/Materials/Bark", (0.33, 0.22, 0.14), 0.86),
        "dusty_wall": create_preview_material(stage, "/World/Materials/DustyWall", (0.52, 0.44, 0.35), 0.84),
        "hunter_body": create_preview_material(stage, "/World/Materials/HunterBodyWhite", (0.86, 0.88, 0.86), 0.54),
        "hunter_wheel": create_preview_material(stage, "/World/Materials/HunterWheelBlack", (0.018, 0.018, 0.018), 0.72),
        "hunter_accent": create_preview_material(stage, "/World/Materials/HunterAccentBlue", (0.05, 0.42, 0.86), 0.36),
        "red": create_preview_material(stage, "/World/Materials/SignRed", (0.86, 0.05, 0.04), 0.55),
        "white": create_preview_material(stage, "/World/Materials/SignWhite", (0.96, 0.96, 0.90), 0.5),
        "metal": create_preview_material(stage, "/World/Materials/BrushedMetal", (0.55, 0.56, 0.53), 0.42, metallic=0.35),
        "digital_wall": create_preview_material(stage, "/World/Materials/DigitalWall", (0.08, 0.13, 0.16), 0.65, texture_path=DIGITAL_WALL_TEXTURE),
    }
    apply_physics_material(materials.get("sand"), static_friction=2.0, dynamic_friction=1.65, restitution=0.02)
    apply_physics_material(materials.get("route_dust"), static_friction=2.2, dynamic_friction=1.75, restitution=0.01)
    apply_physics_material(materials.get("mars_rock"), static_friction=1.4, dynamic_friction=1.1, restitution=0.03)
    apply_physics_material(materials.get("dark_rock"), static_friction=1.6, dynamic_friction=1.25, restitution=0.02)
    apply_physics_material(materials.get("bark"), static_friction=1.3, dynamic_friction=1.0, restitution=0.02)
    apply_physics_material(materials.get("dusty_wall"), static_friction=1.5, dynamic_friction=1.15, restitution=0.02)
    apply_physics_material(materials.get("soil"), static_friction=1.3, dynamic_friction=1.0, restitution=0.02)
    return materials


def bind_material(prim_or_schema, material):
    if not material:
        return
    prim = None
    try:
        prim = prim_or_schema.GetPrim() if hasattr(prim_or_schema, "GetPrim") else prim_or_schema
        UsdShade.MaterialBindingAPI.Apply(prim)
        binding_api = UsdShade.MaterialBindingAPI(prim)
        try:
            binding_api.Bind(material, getattr(UsdShade.Tokens, "strongerThanDescendants"))
        except (AttributeError, TypeError):
            binding_api.Bind(material)
    except Exception as exc:
        prim_path = prim.GetPath() if prim else "<unknown>"
        log(f"warning: failed to bind material on {prim_path}: {exc}")


def apply_collision(prim_or_schema, approximation=None, contact_offset=0.035, rest_offset=0.003):
    prim = prim_or_schema.GetPrim() if hasattr(prim_or_schema, "GetPrim") else prim_or_schema
    try:
        collision_api = UsdPhysics.CollisionAPI.Apply(prim)
        try:
            collision_api.CreateCollisionEnabledAttr().Set(True)
        except Exception:
            pass
        if approximation:
            mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mesh_collision.CreateApproximationAttr().Set(approximation)
        tune_contact_surface(prim, contact_offset=contact_offset, rest_offset=rest_offset)
    except Exception as exc:
        prim_path = prim.GetPath() if prim else "<unknown>"
        log(f"warning: failed to apply collision on {prim_path}: {exc}")


def define_box(stage, path, translate, scale, color, collision=False, material=None, rotate=(0.0, 0.0, 0.0)):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.XformCommonAPI(cube)
    xform.SetTranslate(Gf.Vec3d(*translate))
    xform.SetRotate(tuple(rotate), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    xform.SetScale(Gf.Vec3f(*scale))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    bind_material(cube, material)
    if collision:
        apply_collision(cube)
    return cube


def define_vertical_card(stage, path, translate, width, height, color, material=None):
    mesh = UsdGeom.Mesh.Define(stage, path)
    half_w = 0.5 * width
    half_h = 0.5 * height
    mesh.CreatePointsAttr([
        Gf.Vec3f(0.0, half_w, -half_h),
        Gf.Vec3f(0.0, -half_w, -half_h),
        Gf.Vec3f(0.0, -half_w, half_h),
        Gf.Vec3f(0.0, half_w, half_h),
    ])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    bind_material(mesh, material)
    xform = UsdGeom.XformCommonAPI(mesh)
    xform.SetTranslate(Gf.Vec3d(*translate))
    return mesh


def define_cylinder(stage, path, translate, radius, height, color, collision=False, material=None, rotate=(0.0, 0.0, 0.0)):
    cylinder = UsdGeom.Cylinder.Define(stage, path)
    cylinder.CreateRadiusAttr(radius)
    cylinder.CreateHeightAttr(height)
    cylinder.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    bind_material(cylinder, material)
    xform = UsdGeom.XformCommonAPI(cylinder)
    xform.SetTranslate(Gf.Vec3d(*translate))
    xform.SetRotate(tuple(rotate), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    if collision:
        contact_offset = max(0.045, min(0.075, float(radius) * 0.45))
        apply_collision(cylinder, contact_offset=contact_offset, rest_offset=0.003)
    return cylinder


def define_open_cylinder_shell(
    stage,
    path,
    translate,
    radius,
    height,
    color,
    material=None,
    rotate=(0.0, 0.0, 0.0),
    segments=40,
):
    mesh = UsdGeom.Mesh.Define(stage, path)
    half_h = 0.5 * height
    points = []
    normals = []
    uvs = []
    for iz, z in enumerate((-half_h, half_h)):
        for seg in range(segments):
            angle = 2.0 * math.pi * seg / segments
            nx = math.cos(angle)
            ny = math.sin(angle)
            points.append(Gf.Vec3f(radius * nx, radius * ny, z))
            normals.append(Gf.Vec3f(nx, ny, 0.0))
            uvs.append(Gf.Vec2f(seg / segments, float(iz)))

    face_counts = []
    face_indices = []
    for seg in range(segments):
        nxt = (seg + 1) % segments
        face_counts.append(4)
        face_indices.extend([seg, nxt, segments + nxt, segments + seg])

    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(face_counts)
    mesh.CreateFaceVertexIndicesAttr(face_indices)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateNormalsAttr(normals)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    try:
        primvar = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
            "st",
            Sdf.ValueTypeNames.TexCoord2fArray,
            UsdGeom.Tokens.vertex,
        )
        primvar.Set(uvs)
    except Exception as exc:
        log(f"warning: failed to create open cylinder UVs for {path}: {exc}")
    bind_material(mesh, material)
    xform = UsdGeom.XformCommonAPI(mesh)
    xform.SetTranslate(Gf.Vec3d(*translate))
    xform.SetRotate(tuple(rotate), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    return mesh


def define_sphere(
    stage,
    path,
    translate,
    radius,
    color,
    collision=False,
    material=None,
    rotate=(0.0, 0.0, 0.0),
    scale=(1.0, 1.0, 1.0),
):
    scale_values = tuple(float(component) for component in scale)
    if collision and (max(scale_values) - min(scale_values)) > 1e-4:
        deterministic_seed = sum(ord(char) for char in str(path)) % 997
        return define_irregular_rock(
            stage,
            path,
            translate,
            radius,
            color,
            collision,
            material,
            rotate=rotate,
            scale=scale_values,
            seed=deterministic_seed,
        )
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(radius)
    sphere.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    bind_material(sphere, material)
    xform = UsdGeom.XformCommonAPI(sphere)
    xform.SetTranslate(Gf.Vec3d(*translate))
    xform.SetRotate(tuple(rotate), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    xform.SetScale(Gf.Vec3f(*scale_values))
    if collision:
        apply_collision(sphere)
    return sphere


def define_irregular_rock(
    stage,
    path,
    translate,
    radius,
    color,
    collision=False,
    material=None,
    rotate=(0.0, 0.0, 0.0),
    scale=(1.0, 1.0, 1.0),
    seed=0,
):
    mesh = UsdGeom.Mesh.Define(stage, path)
    segments = 10
    latitudes = (-62.0, -26.0, 10.0, 48.0)
    bake_scale = tuple(float(component) for component in scale) if collision else (1.0, 1.0, 1.0)

    def rock_point(x, y, z):
        return Gf.Vec3f(x * bake_scale[0], y * bake_scale[1], z * bake_scale[2])

    points = [rock_point(0.0, 0.0, -0.72 * radius)]

    for lat_idx, lat_deg in enumerate(latitudes):
        lat = math.radians(lat_deg)
        ring_z = radius * math.sin(lat)
        ring_radius = radius * math.cos(lat)
        for seg in range(segments):
            angle = 2.0 * math.pi * seg / segments
            wobble = 1.0 + 0.16 * math.sin((seg + 1) * (seed + 0.73)) + 0.08 * math.cos((lat_idx + 2) * (seg + seed))
            points.append(rock_point(ring_radius * wobble * math.cos(angle), ring_radius * wobble * math.sin(angle), ring_z))

    top_index = len(points)
    points.append(rock_point(0.0, 0.0, 0.78 * radius))

    face_counts = []
    face_indices = []
    first_ring = 1
    for seg in range(segments):
        nxt = first_ring + (seg + 1) % segments
        face_counts.append(3)
        face_indices.extend([0, nxt, first_ring + seg])

    for lat_idx in range(len(latitudes) - 1):
        ring_a = 1 + lat_idx * segments
        ring_b = ring_a + segments
        for seg in range(segments):
            a0 = ring_a + seg
            a1 = ring_a + (seg + 1) % segments
            b0 = ring_b + seg
            b1 = ring_b + (seg + 1) % segments
            face_counts.append(4)
            face_indices.extend([a0, b0, b1, a1])

    last_ring = 1 + (len(latitudes) - 1) * segments
    for seg in range(segments):
        nxt = last_ring + (seg + 1) % segments
        face_counts.append(3)
        face_indices.extend([last_ring + seg, nxt, top_index])

    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(face_counts)
    mesh.CreateFaceVertexIndicesAttr(face_indices)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    bind_material(mesh, material)
    xform = UsdGeom.XformCommonAPI(mesh)
    xform.SetTranslate(Gf.Vec3d(*translate))
    xform.SetRotate(tuple(rotate), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    xform.SetScale(Gf.Vec3f(1.0, 1.0, 1.0) if collision else Gf.Vec3f(*scale))
    if collision:
        apply_collision(mesh, "convexHull", contact_offset=0.045, rest_offset=0.003)
    return mesh


def discover_ros_packages():
    packages = {}
    for package_xml in (REPO_ROOT / "src").rglob("package.xml"):
        try:
            root = ET.parse(package_xml).getroot()
            name = root.findtext("name")
        except ET.ParseError:
            continue
        if name:
            packages.setdefault(name.strip(), package_xml.parent)
    return packages


def resolve_ros_paths(text):
    packages = discover_ros_packages()

    def package_uri(match):
        package_name, rel_path = match.group(1), match.group(2)
        package_dir = packages.get(package_name)
        if package_dir is None:
            log(f"warning: cannot resolve ROS package URI for {package_name}")
            return match.group(0)
        return str((package_dir / rel_path).resolve())

    def find_expr(match):
        package_name = match.group(1).strip()
        package_dir = packages.get(package_name)
        if package_dir is None:
            log(f"warning: cannot resolve $(find {package_name})")
            return match.group(0)
        return str(package_dir.resolve())

    text = re.sub(r"package://([^/\s\"'<]+)/([^\"'<>\s]+)", package_uri, text)
    text = re.sub(r"\$\(find\s+([^)]+)\)", find_expr, text)
    return text


def resolve_urdf_source(path):
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"robot URDF/xacro not found: {source}")

    raw_text = source.read_text(errors="ignore")
    needs_xacro = "<xacro:" in raw_text or "${" in raw_text or "$(arg " in raw_text
    if needs_xacro:
        log(f"expanding xacro: {source}")
        setup_script = REPO_ROOT / "install" / "setup.bash"
        xacro_args = []
        if source.name == "hunter_se_gazebo.xacro":
            xacro_args.extend(("drive_mode:=hunter_joint", "enable_livox:=false", "enable_d455:=false"))
        xacro_arg_text = " ".join(shlex.quote(item) for item in xacro_args)
        command = (
            "unset PYTHONHOME PYTHONPATH; "
            "source /opt/ros/humble/setup.bash 2>/dev/null || true; "
            f"source {shlex.quote(str(setup_script))} 2>/dev/null || true; "
            f"xacro {shlex.quote(str(source))} {xacro_arg_text}"
        )
        clean_env = os.environ.copy()
        clean_env.pop("PYTHONHOME", None)
        clean_env.pop("PYTHONPATH", None)
        try:
            result = subprocess.run(["bash", "-lc", command], check=True, capture_output=True, text=True, env=clean_env)
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            stdout = (exc.stdout or "").strip()
            raise RuntimeError(
                f"xacro expansion failed for {source} (exit={exc.returncode}). "
                f"stderr={stderr[:2000]!r} stdout={stdout[:500]!r}"
            ) from exc
        text = result.stdout
    else:
        text = raw_text

    text = resolve_ros_paths(text)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    resolved = GENERATED_DIR / f"{source.stem}_resolved.urdf"
    if not resolved.exists() or resolved.read_text(errors="ignore") != text:
        resolved.write_text(text)
    return resolved


def ensure_offroad_nav_layout():
    if not OFFROAD_NAV_ROOT.exists():
        return

    source = OFFROAD_NAV_ROOT / "terrain.generator"
    target = OFFROAD_NAV_ROOT / "isaac" / "extsUser" / "terrain.generator"
    if not source.exists() or target.exists():
        return

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(Path("../../terrain.generator"))
        log(f"linked Offroad-Nav terrain.generator assets: {target}")
    except OSError as exc:
        log(f"warning: failed to link Offroad-Nav terrain.generator assets: {exc}")


def resolve_map_asset(asset_arg):
    if not asset_arg or asset_arg.lower() in {"none", "false", "off", "0"}:
        return None
    if asset_arg.lower() == "auto":
        if args.scene_profile == "rubicon":
            return RUBICON_MAP_ASSET.resolve()
        if args.scene_profile in {"mars_gale", "offroad"}:
            return MARS_GALE_MAP_ASSET.resolve()
        if args.scene_profile == "offroad_nav":
            ensure_offroad_nav_layout()
            default_offroad_nav = OFFROAD_NAV_ASSETS["hard"] if args.headless else OFFROAD_NAV_ASSETS["medium"]
            if default_offroad_nav.exists():
                log(f"using Offroad-Nav {'hard' if args.headless else 'medium'} asset for auto profile")
                return default_offroad_nav.resolve()
            log(f"warning: Offroad-Nav USD not found, falling back to Mars Gale: {default_offroad_nav}")
            return MARS_GALE_MAP_ASSET.resolve()
        return None
    aliases = {
        "mars_gale": MARS_GALE_MAP_ASSET,
        "mars": MARS_GALE_MAP_ASSET,
        "offroad": MARS_GALE_MAP_ASSET,
        "mars_collision": MARS_GALE_COLLISION_ASSET,
        "rubicon": RUBICON_MAP_ASSET,
        "offroad_nav": OFFROAD_NAV_ASSETS["hard"] if args.headless else OFFROAD_NAV_ASSETS["medium"],
        "offroad_nav_easy": OFFROAD_NAV_ASSETS["easy"],
        "offroad_nav_medium": OFFROAD_NAV_ASSETS["medium"],
        "offroad_nav_hard": OFFROAD_NAV_ASSETS["hard"],
        "offroad_nav_barakuda": OFFROAD_NAV_ASSETS["barakuda"],
    }
    if asset_arg.lower().startswith("offroad_nav"):
        ensure_offroad_nav_layout()
    asset = aliases.get(asset_arg.lower(), Path(asset_arg).expanduser())
    asset = asset.resolve()
    if not asset.exists():
        log(f"warning: map asset not found, skipping: {asset}")
        return None
    return asset


def run_async(coro):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


def usd_has_absolute_props_reference(usd_path):
    try:
        return b"@/props/" in Path(usd_path).read_bytes()
    except OSError:
        return False


def fix_absolute_props_references(usd_path):
    usd_path = Path(usd_path).resolve()
    if not usd_has_absolute_props_reference(usd_path):
        return

    try:
        layer = Sdf.Layer.FindOrOpen(str(usd_path))
        if layer is None:
            return

        def rewrite_reference(ref):
            if not ref.assetPath.startswith("/props/"):
                return ref, False
            fixed_asset = usd_path.parent / ref.assetPath.lstrip("/")
            return Sdf.Reference(str(fixed_asset.resolve()), ref.primPath, ref.layerOffset, ref.customData), True

        def rewrite_list_op(list_op):
            changed = False
            for attr in ("prependedItems", "appendedItems", "addedItems", "explicitItems"):
                items = list(getattr(list_op, attr))
                if not items:
                    continue
                rewritten = []
                for ref in items:
                    new_ref, did_change = rewrite_reference(ref)
                    changed = changed or did_change
                    rewritten.append(new_ref)
                setattr(list_op, attr, rewritten)
            return changed

        def walk_prim(spec):
            changed = rewrite_list_op(spec.referenceList)
            for child in spec.nameChildren:
                changed = walk_prim(child) or changed
            return changed

        changed = False
        for root in layer.rootPrims:
            changed = walk_prim(root) or changed
        if changed:
            layer.Save()
            log(f"fixed absolute /props references in USD cache: {usd_path}")
    except Exception as exc:
        log(f"warning: failed to fix /props references for {usd_path}: {exc}")


def convert_asset_to_usd(asset_path):
    asset_path = Path(asset_path).resolve()
    if asset_path.suffix.lower() in {".usd", ".usda", ".usdc"}:
        return asset_path

    import omni.kit.asset_converter

    cache_dir = GENERATED_DIR / "assets"
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        rel_stem = asset_path.relative_to(REPO_ROOT).with_suffix("")
    except ValueError:
        rel_stem = asset_path.with_suffix("")
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", "_".join(rel_stem.parts))
    output = cache_dir / f"{safe_stem}.usd"
    if output.exists() and output.stat().st_mtime >= asset_path.stat().st_mtime:
        fix_absolute_props_references(output)
        if not usd_has_absolute_props_reference(output):
            return output
        log(f"reconverting asset with embedded props: {asset_path}")

    log(f"converting project asset to USD: {asset_path}")
    context = omni.kit.asset_converter.AssetConverterContext()
    context.ignore_camera = True
    context.ignore_light = True
    context.single_mesh = True
    context.disabling_instancing = True
    context.export_preview_surface = True
    context.use_meter_as_world_unit = True
    context.convert_stage_up_z = True
    context.create_world_as_default_root_prim = False
    context.embed_textures = True
    task = omni.kit.asset_converter.get_instance().create_converter_task(
        str(asset_path),
        str(output),
        lambda current, total: log(f"asset conversion progress {current}/{total}") if current == total else None,
        context,
    )
    success = run_async(task.wait_until_finished())
    if not success:
        log(f"warning: asset conversion failed: {task.get_status()} {task.get_detailed_error()}")
        return None
    fix_absolute_props_references(output)
    return output


def add_reference(
    stage,
    path,
    asset_path,
    translate=(0.0, 0.0, 0.0),
    rotate=(0.0, 0.0, 0.0),
    scale=(1.0, 1.0, 1.0),
    reference_prim_path=None,
):
    prim = stage.DefinePrim(path, "Xform")
    xform = UsdGeom.XformCommonAPI(prim)
    xform.SetTranslate(Gf.Vec3d(*translate))
    xform.SetRotate(tuple(rotate), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    xform.SetScale(Gf.Vec3f(*scale))
    asset_prim = stage.DefinePrim(f"{path}/Asset", "Xform")
    if reference_prim_path:
        asset_prim.GetReferences().AddReference(str(asset_path), Sdf.Path(reference_prim_path))
    else:
        asset_prim.GetReferences().AddReference(str(asset_path))
    return prim


def for_each_mesh_under(stage, root_path, visitor):
    root_path = str(root_path)
    count = 0
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if prim_path != root_path and not prim_path.startswith(f"{root_path}/"):
            continue
        if prim.IsA(UsdGeom.Mesh):
            visitor(UsdGeom.Mesh(prim))
            count += 1
    return count


def bind_material_under(stage, root_path, material):
    if not material:
        return 0
    return for_each_mesh_under(stage, root_path, lambda mesh: bind_material(mesh, material))


def apply_collision_under(stage, root_path, approximation="meshSimplification", contact_offset=0.035, rest_offset=0.003):
    return for_each_mesh_under(
        stage,
        root_path,
        lambda mesh: apply_collision(
            mesh,
            approximation,
            contact_offset=contact_offset,
            rest_offset=rest_offset,
        ),
    )


def tune_collision_surfaces_under(stage, root_path, contact_offset=0.040, rest_offset=0.003):
    root_path = str(root_path)
    count = 0
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if prim_path != root_path and not prim_path.startswith(f"{root_path}/"):
            continue
        if not prim_has_api(prim, UsdPhysics.CollisionAPI):
            continue
        try:
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(True)
        except Exception:
            pass
        tune_contact_surface(prim, contact_offset=contact_offset, rest_offset=rest_offset)
        count += 1
    return count


def apply_visual_style_under(stage, root_path, display_color=None, double_sided=True):
    def style(mesh):
        if display_color is not None:
            mesh.CreateDisplayColorAttr([Gf.Vec3f(*display_color)])
        if double_sided:
            mesh.CreateDoubleSidedAttr(True)

    return for_each_mesh_under(stage, root_path, style)


def apply_xy_uvs_under(stage, root_path, uv_scale=34.0):
    def apply_uvs(mesh):
        points = mesh.GetPointsAttr().Get()
        if not points:
            return
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        span_x = max(1e-6, max_x - min_x)
        span_y = max(1e-6, max_y - min_y)
        uvs = [
            Gf.Vec2f(uv_scale * (float(point[0]) - min_x) / span_x, uv_scale * (float(point[1]) - min_y) / span_y)
            for point in points
        ]
        primvar = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
            "st",
            Sdf.ValueTypeNames.TexCoord2fArray,
            UsdGeom.Tokens.vertex,
        )
        primvar.Set(uvs)

    return for_each_mesh_under(stage, root_path, apply_uvs)


def reset_pose_ops(prim):
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    translate_op = xformable.AddTranslateOp()
    rotate_op = xformable.AddRotateXYZOp()
    return translate_op, rotate_op


def set_pose_ops(pose_ops, translate, rotate):
    translate_op, rotate_op = pose_ops
    translate_op.Set(Gf.Vec3d(*translate))
    rotate_op.Set(Gf.Vec3f(*rotate))


def terrain_height(x, y):
    if abs(x) < 5.0 and abs(y) < 4.0:
        return -0.035
    roll = 0.045 * math.sin(0.105 * x) * math.cos(0.085 * y)
    broad = 0.025 * math.sin(0.035 * (x + 1.7 * y))
    return -0.04 + roll + broad


_MARS_HEIGHT_GRID = None
_SCENE_SPAWN_XY = None


def is_mars_profile():
    return args.scene_profile in {"mars_gale", "offroad"}


def load_mars_height_grid():
    global _MARS_HEIGHT_GRID
    if _MARS_HEIGHT_GRID is not None:
        return _MARS_HEIGHT_GRID

    vertices = []
    try:
        with MARS_GALE_COLLISION_ASSET.open(errors="ignore") as obj_file:
            for line in obj_file:
                if not line.startswith("v "):
                    continue
                parts = line.split()
                if len(parts) < 4:
                    continue
                vertices.append((round(float(parts[1]), 6), round(float(parts[2]), 6), float(parts[3]) + MARS_COLLISION_Z_OFFSET))
    except OSError as exc:
        log(f"warning: failed to load Mars collision height grid: {exc}")
        _MARS_HEIGHT_GRID = ([-120.0, 120.0], [-90.0, 90.0], {})
        return _MARS_HEIGHT_GRID

    xs = sorted({v[0] for v in vertices})
    ys = sorted({v[1] for v in vertices})
    heights = {(x, y): z for x, y, z in vertices}
    _MARS_HEIGHT_GRID = (xs, ys, heights)
    log(f"loaded Mars Gale collision height grid ({len(xs)} x {len(ys)})")
    return _MARS_HEIGHT_GRID


def _bracket(sorted_values, value):
    if len(sorted_values) < 2:
        return sorted_values[0], sorted_values[0]
    if value <= sorted_values[0]:
        return sorted_values[0], sorted_values[1]
    if value >= sorted_values[-1]:
        return sorted_values[-2], sorted_values[-1]
    upper = bisect.bisect_right(sorted_values, value)
    return sorted_values[upper - 1], sorted_values[upper]


def mars_terrain_height(x, y):
    xs, ys, heights = load_mars_height_grid()
    x0, x1 = _bracket(xs, float(x))
    y0, y1 = _bracket(ys, float(y))
    z00 = heights.get((x0, y0), 0.0)
    z10 = heights.get((x1, y0), z00)
    z01 = heights.get((x0, y1), z00)
    z11 = heights.get((x1, y1), z10)
    tx = 0.0 if x1 == x0 else (float(x) - x0) / (x1 - x0)
    ty = 0.0 if y1 == y0 else (float(y) - y0) / (y1 - y0)
    z0 = z00 * (1.0 - tx) + z10 * tx
    z1 = z01 * (1.0 - tx) + z11 * tx
    return z0 * (1.0 - ty) + z1 * ty


def mars_terrain_slope_deg(x, y, eps=0.75):
    dzdx = (mars_terrain_height(x + eps, y) - mars_terrain_height(x - eps, y)) / (2.0 * eps)
    dzdy = (mars_terrain_height(x, y + eps) - mars_terrain_height(x, y - eps)) / (2.0 * eps)
    return math.degrees(math.atan(math.hypot(dzdx, dzdy)))


def mars_terrain_gradient(x, y, eps=0.75):
    dzdx = (mars_terrain_height(x + eps, y) - mars_terrain_height(x - eps, y)) / (2.0 * eps)
    dzdy = (mars_terrain_height(x, y + eps) - mars_terrain_height(x, y - eps)) / (2.0 * eps)
    return dzdx, dzdy


def mars_terrain_patch_roughness(x, y):
    heights = []
    for dx in (-0.60, 0.0, 0.60):
        for dy in (-0.42, 0.0, 0.42):
            heights.append(mars_terrain_height(x + dx, y + dy))
    return max(heights) - min(heights)


def auto_select_mars_spawn_xy():
    hint_x, hint_y = MARS_FLAT_SPAWN_HINT
    best = None
    for ix in range(-16, 17):
        x = hint_x + 0.50 * ix
        for iy in range(-12, 13):
            y = hint_y + 0.50 * iy
            if -5.2 < x < 5.8 and -4.2 < y < 4.2:
                continue
            slope = mars_terrain_slope_deg(x, y)
            roughness = mars_terrain_patch_roughness(x, y)
            score = 10.0 * slope + 55.0 * roughness + 0.14 * ((x - hint_x) ** 2 + (y - hint_y) ** 2)
            if best is None or score < best[0]:
                best = (score, x, y, slope, roughness)

    if best is None:
        return hint_x, hint_y

    _score, x, y, slope, roughness = best
    log(f"selected flat Mars spawn at ({x:.2f}, {y:.2f}) slope={slope:.2f}deg roughness={roughness:.3f}m")
    return x, y


def scene_spawn_xy():
    global _SCENE_SPAWN_XY
    if _SCENE_SPAWN_XY is not None:
        return _SCENE_SPAWN_XY

    if args.spawn_x is not None or args.spawn_y is not None:
        _SCENE_SPAWN_XY = (
            finite_or_zero(args.spawn_x if args.spawn_x is not None else 0.0),
            finite_or_zero(args.spawn_y if args.spawn_y is not None else 0.0),
        )
    elif is_mars_profile() and args.mars_auto_flat_spawn:
        _SCENE_SPAWN_XY = auto_select_mars_spawn_xy()
    else:
        _SCENE_SPAWN_XY = (0.0, 0.0)
    return _SCENE_SPAWN_XY


def scene_spawn_yaw_rad():
    return math.radians(finite_or_zero(args.spawn_yaw_deg))


def mars_robot_base_z(x, y):
    return mars_terrain_height(x, y) + max(0.18, float(args.robot_z))


def scene_robot_base_z(x, y):
    if is_mars_profile():
        return mars_robot_base_z(x, y)
    return float(args.robot_z)


def add_heightfield_terrain(stage, path, world_size, resolution, height_fn, material, display_color, uv_scale=8.0):
    half = 0.5 * world_size
    mesh = UsdGeom.Mesh.Define(stage, path)
    points = []
    uvs = []
    for iy in range(resolution + 1):
        y = -half + world_size * iy / resolution
        for ix in range(resolution + 1):
            x = -half + world_size * ix / resolution
            points.append(Gf.Vec3f(x, y, height_fn(x, y)))
            uvs.append(Gf.Vec2f(uv_scale * ix / resolution, uv_scale * iy / resolution))

    face_counts = []
    face_indices = []
    for iy in range(resolution):
        for ix in range(resolution):
            i0 = iy * (resolution + 1) + ix
            i1 = i0 + 1
            i2 = i0 + resolution + 1
            i3 = i2 + 1
            face_counts.extend([3, 3])
            face_indices.extend([i0, i2, i1, i1, i2, i3])

    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(face_counts)
    mesh.CreateFaceVertexIndicesAttr(face_indices)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*display_color)])
    try:
        primvar = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
            "st",
            Sdf.ValueTypeNames.TexCoord2fArray,
            UsdGeom.Tokens.vertex,
        )
        primvar.Set(uvs)
    except Exception as exc:
        log(f"warning: failed to create ground UVs: {exc}")
    bind_material(mesh, material)
    apply_collision(mesh, "meshSimplification")
    return mesh


def add_generated_terrain(stage, materials):
    world_size = max(40.0, float(args.world_size))
    resolution = 34 if args.scene_profile == "rich" else 10
    return add_heightfield_terrain(
        stage,
        "/World/Ground",
        world_size,
        resolution,
        terrain_height,
        materials.get("grass"),
        (0.34, 0.43, 0.25),
        uv_scale=8.0,
    )


def add_mars_generated_terrain(stage, materials):
    world_size = max(220.0, float(args.world_size))
    resolution = 72
    return add_heightfield_terrain(
        stage,
        "/World/MarsGale/ProceduralTerrain",
        world_size,
        resolution,
        mars_terrain_height,
        materials.get("sand"),
        (0.55, 0.38, 0.25),
        uv_scale=14.0,
    )


def mars_surface_micro_relief(x, y):
    return 0.0


def mars_surface_visual_height(x, y):
    return mars_terrain_height(x, y) + MARS_SURFACE_SKIN_Z_OFFSET + mars_surface_micro_relief(x, y)


def clamp(value, min_value=0.0, max_value=1.0):
    return max(min_value, min(max_value, value))


def mix_color(a, b, t):
    t = clamp(t)
    return tuple(a[idx] * (1.0 - t) + b[idx] * t for idx in range(3))


def mars_surface_display_color(x, y, z, dzdx, dzdy, min_z, max_z):
    span_z = max(1e-6, max_z - min_z)
    height_t = clamp((z - min_z) / span_z)
    slope = clamp(math.sqrt(dzdx * dzdx + dzdy * dzdy) * 1.9)
    wind = 0.5 + 0.5 * math.sin(0.075 * x + 0.115 * y + 0.7 * math.sin(0.025 * x))
    streak = 0.5 + 0.5 * math.sin(0.23 * x - 0.17 * y + 0.6 * math.sin(0.041 * y))
    fine = 0.5 + 0.5 * math.sin(1.7 * x + 2.4 * y)

    pale_sand = (0.78, 0.66, 0.50)
    ochre_sand = (0.62, 0.43, 0.27)
    rust = (0.50, 0.18, 0.10)
    dark_ridge = (0.25, 0.16, 0.12)

    color = mix_color(pale_sand, ochre_sand, 0.32 * height_t + 0.18 * (1.0 - wind))
    rust_weight = clamp(0.52 * slope + 0.28 * streak + 0.18 * height_t - 0.24)
    color = mix_color(color, rust, rust_weight)
    color = mix_color(color, dark_ridge, clamp((slope - 0.48) * 1.1))
    brightness = 0.92 + 0.13 * fine + 0.05 * wind
    return tuple(clamp(component * brightness, 0.06, 0.92) for component in color)


def add_mars_surface_skin(stage, materials):
    if args.mars_surface_skin == "none":
        return None

    xs, ys, heights = load_mars_height_grid()
    height_values = list(heights.values()) or [0.0]
    min_height = min(height_values)
    max_height = max(height_values)
    if args.mars_surface_skin == "local":
        min_x, max_x = -26.0, 90.0
        min_y, max_y = -44.0, 38.0
        step = 0.85
    else:
        min_x, max_x = xs[0], xs[-1]
        min_y, max_y = ys[0], ys[-1]
        step = 1.10

    nx = max(48, min(230, int((max_x - min_x) / step)))
    ny = max(42, min(180, int((max_y - min_y) / step)))
    mesh = UsdGeom.Mesh.Define(stage, "/World/MarsGale/SurfaceSkin")
    points = []
    normals = []
    uvs = []
    colors = []
    eps = 0.55
    uv_repeat = 0.055

    for iy in range(ny + 1):
        y = min_y + (max_y - min_y) * iy / ny
        for ix in range(nx + 1):
            x = min_x + (max_x - min_x) * ix / nx
            z = mars_surface_visual_height(x, y)
            points.append(Gf.Vec3f(x, y, z))
            dzdx = (mars_surface_visual_height(x + eps, y) - mars_surface_visual_height(x - eps, y)) / (2.0 * eps)
            dzdy = (mars_surface_visual_height(x, y + eps) - mars_surface_visual_height(x, y - eps)) / (2.0 * eps)
            length = max(1e-6, math.sqrt(dzdx * dzdx + dzdy * dzdy + 1.0))
            normals.append(Gf.Vec3f(-dzdx / length, -dzdy / length, 1.0 / length))
            uvs.append(Gf.Vec2f((x - min_x) * uv_repeat, (y - min_y) * uv_repeat))
            colors.append(Gf.Vec3f(*mars_surface_display_color(x, y, z, dzdx, dzdy, min_height, max_height)))

    face_counts = []
    face_indices = []
    row = nx + 1
    for iy in range(ny):
        for ix in range(nx):
            i0 = iy * row + ix
            i1 = i0 + 1
            i2 = i0 + row
            i3 = i2 + 1
            face_counts.append(4)
            face_indices.extend([i0, i2, i3, i1])

    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(face_counts)
    mesh.CreateFaceVertexIndicesAttr(face_indices)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateNormalsAttr(normals)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateDisplayColorAttr(colors)
    try:
        mesh.GetDisplayColorPrimvar().SetInterpolation(UsdGeom.Tokens.vertex)
    except Exception as exc:
        log(f"warning: failed to set Mars surface displayColor interpolation: {exc}")
    try:
        primvar = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
            "st",
            Sdf.ValueTypeNames.TexCoord2fArray,
            UsdGeom.Tokens.vertex,
        )
        primvar.Set(uvs)
    except Exception as exc:
        log(f"warning: failed to create Mars surface skin UVs: {exc}")
    bind_material(mesh, materials.get("mars_surface") or materials.get("sand"))
    log(f"created PBR Mars surface skin ({nx} x {ny} quads, mode={args.mars_surface_skin})")
    return mesh


def add_mars_pebble_field(stage, materials):
    if args.mars_surface_skin == "none":
        return

    stage.DefinePrim("/World/MarsGale/PebbleField", "Xform")
    count_by_density = {"none": 0, "sparse": 48, "medium": 92, "dense": 150}
    count = count_by_density.get(args.obstacle_density, 48)
    if count <= 0:
        return

    for idx in range(count):
        # Low-discrepancy deterministic scatter concentrated around the route corridor.
        t = (idx * 0.61803398875) % 1.0
        route_idx = min(len(MARS_ROUTE_WAYPOINTS) - 2, int(t * (len(MARS_ROUTE_WAYPOINTS) - 1)))
        local_t = t * (len(MARS_ROUTE_WAYPOINTS) - 1) - route_idx
        x0, y0 = MARS_ROUTE_WAYPOINTS[route_idx]
        x1, y1 = MARS_ROUTE_WAYPOINTS[route_idx + 1]
        cx = x0 + (x1 - x0) * local_t
        cy = y0 + (y1 - y0) * local_t
        tangent_x = x1 - x0
        tangent_y = y1 - y0
        tangent_len = max(1e-6, math.hypot(tangent_x, tangent_y))
        nx = -tangent_y / tangent_len
        ny = tangent_x / tangent_len
        side = -1.0 if idx % 2 == 0 else 1.0
        lateral = side * (2.2 + 0.37 * (idx % 11))
        along = 0.28 * ((idx * 17) % 9 - 4)
        x = cx + lateral * nx + along * tangent_x / tangent_len
        y = cy + lateral * ny + along * tangent_y / tangent_len
        radius = 0.035 + 0.014 * (idx % 5)
        scale = (1.0 + 0.32 * ((idx + 1) % 3), 0.74 + 0.16 * (idx % 4), 0.42 + 0.08 * (idx % 3))
        rotate = (3.0 * (idx % 5), -4.0 * (idx % 7), (idx * 47.0) % 360.0)
        material = materials.get("mars_rock") if idx % 3 else materials.get("dark_rock")
        define_irregular_rock(
            stage,
            f"/World/MarsGale/PebbleField/Pebble_{idx:03d}",
            (x, y, mars_surface_visual_height(x, y) + 0.012),
            radius,
            (0.40, 0.32, 0.24),
            False,
            material,
            rotate=rotate,
            scale=scale,
            seed=idx + 31,
        )
    log(f"created Mars visual pebble field ({count} small rocks)")


def resample_polyline(waypoints, spacing=1.4):
    samples = []
    for idx in range(len(waypoints) - 1):
        x0, y0 = waypoints[idx]
        x1, y1 = waypoints[idx + 1]
        length = math.hypot(x1 - x0, y1 - y0)
        steps = max(1, int(length / spacing))
        for step in range(steps):
            if idx > 0 and step == 0:
                continue
            t = step / steps
            samples.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
    samples.append(tuple(waypoints[-1]))
    return samples


def add_mars_route_surface(stage, materials):
    if not args.show_route_surface and not args.show_route_guide:
        return None

    stage.DefinePrim("/World/MarsGale/Route", "Xform")

    def make_strip(path, width, z_offset, material, display_color):
        centerline = resample_polyline(MARS_ROUTE_WAYPOINTS)
        if len(centerline) < 2:
            return None
        mesh = UsdGeom.Mesh.Define(stage, path)
        points = []
        uvs = []
        for idx, (x, y) in enumerate(centerline):
            if idx == 0:
                tx, ty = centerline[1][0] - x, centerline[1][1] - y
            elif idx == len(centerline) - 1:
                tx, ty = x - centerline[idx - 1][0], y - centerline[idx - 1][1]
            else:
                tx = centerline[idx + 1][0] - centerline[idx - 1][0]
                ty = centerline[idx + 1][1] - centerline[idx - 1][1]
            length = max(1e-6, math.hypot(tx, ty))
            nx, ny = -ty / length, tx / length
            for side, u in ((-1.0, 0.0), (1.0, 1.0)):
                px = x + side * 0.5 * width * nx
                py = y + side * 0.5 * width * ny
                points.append(Gf.Vec3f(px, py, mars_terrain_height(px, py) + z_offset))
                uvs.append(Gf.Vec2f(u, 0.12 * idx))

        face_counts = []
        face_indices = []
        for idx in range(len(centerline) - 1):
            i0 = 2 * idx
            i1 = i0 + 1
            i2 = i0 + 2
            i3 = i0 + 3
            face_counts.append(4)
            face_indices.extend([i0, i2, i3, i1])

        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr(face_counts)
        mesh.CreateFaceVertexIndicesAttr(face_indices)
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDoubleSidedAttr(True)
        mesh.CreateDisplayColorAttr([Gf.Vec3f(*display_color)])
        try:
            primvar = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
                "st",
                Sdf.ValueTypeNames.TexCoord2fArray,
                UsdGeom.Tokens.vertex,
            )
            primvar.Set(uvs)
        except Exception as exc:
            log(f"warning: failed to create Mars route UVs: {exc}")
        bind_material(mesh, material)
        return mesh

    route = None
    if args.show_route_surface:
        route = make_strip(
            "/World/MarsGale/Route/CompactedTrack",
            width=2.35,
            z_offset=0.018,
            material=materials.get("route_dust"),
            display_color=(0.54, 0.42, 0.30),
        )
    guide = None
    if args.show_route_guide:
        guide = make_strip(
            "/World/MarsGale/Route/CenterGuide",
            width=0.10,
            z_offset=0.024,
            material=materials.get("lane"),
            display_color=(0.90, 0.80, 0.48),
        )
    log("created optional terrain-following Mars route overlays")
    return route or guide


def add_mars_boulder_field(stage, materials):
    stage.DefinePrim("/World/MarsGale/Boulders", "Xform")
    boulders = (
        ((7.6, -2.9, 0.24), 0.72, (1.50, 0.86, 0.44), (0.0, 12.0, 18.0)),
        ((9.8, 2.8, 0.20), 0.54, (1.10, 0.70, 0.42), (6.0, -8.0, 63.0)),
        ((13.4, -5.4, 0.28), 0.82, (1.75, 1.05, 0.48), (2.0, 18.0, 108.0)),
        ((17.2, 3.6, 0.18), 0.60, (1.25, 0.78, 0.36), (-4.0, 10.0, 151.0)),
        ((22.0, -8.5, 0.42), 1.05, (1.85, 1.32, 0.56), (8.0, -12.0, 204.0)),
        ((28.5, 6.8, 0.36), 0.90, (1.70, 1.10, 0.52), (-9.0, 7.0, 258.0)),
        ((35.5, -12.0, 0.44), 1.12, (2.05, 1.18, 0.58), (4.0, 16.0, 314.0)),
        ((44.0, 8.0, 0.30), 0.78, (1.40, 0.92, 0.46), (-6.0, -11.0, 337.0)),
        ((-8.5, 8.5, 0.18), 0.55, (1.08, 0.72, 0.38), (0.0, 9.0, 41.0)),
        ((-15.0, -10.0, 0.30), 0.86, (1.55, 1.05, 0.48), (5.0, -15.0, 87.0)),
        ((-24.0, 13.0, 0.42), 1.10, (1.92, 1.30, 0.58), (-7.0, 12.0, 126.0)),
        ((-35.0, -16.0, 0.34), 0.95, (1.60, 1.12, 0.50), (9.0, 5.0, 173.0)),
    )
    for idx, (translate, radius, scale, rotate) in enumerate(boulders):
        x, y, z_offset = translate
        define_irregular_rock(
            stage,
            f"/World/MarsGale/Boulders/Rock_{idx:02d}",
            (x, y, mars_terrain_height(x, y) + z_offset),
            radius,
            (0.35, 0.29, 0.24),
            True,
            materials.get("mars_rock"),
            rotate=rotate,
            scale=scale,
            seed=idx + 71,
        )


def add_detail_asset(
    stage,
    path,
    asset_path,
    translate,
    rotate=(0.0, 0.0, 0.0),
    scale=(1.0, 1.0, 1.0),
    collision=False,
    material=None,
):
    asset_path = Path(asset_path)
    if not asset_path.exists():
        return None
    usd_path = convert_asset_to_usd(asset_path)
    if usd_path is None:
        return None
    prim = add_reference(stage, path, usd_path, translate=translate, rotate=rotate, scale=scale)
    if material:
        mesh_count = bind_material_under(stage, str(prim.GetPath()), material)
        log(f"applied override material under detail asset {prim.GetPath()} ({mesh_count} mesh prims)")
    if collision:
        mesh_count = apply_collision_under(stage, str(prim.GetPath()), contact_offset=0.050, rest_offset=0.003)
        log(f"applied collision under detail asset {prim.GetPath()} ({mesh_count} mesh prims)")
    return prim


def add_dense_obstacle_field(stage, materials, root_path="/World/DenseObstacles", include_vegetation=False):
    if args.obstacle_density == "none":
        log(f"obstacle density is none; skipping {root_path}")
        return

    stage.DefinePrim(root_path, "Xform")

    barrier_mat = materials.get("dusty_wall") or materials.get("concrete")
    dark_mat = materials.get("dark_rock")
    bark_mat = materials.get("bark")
    density_limits = {
        "sparse": {"barriers": 22, "trunks": 54, "posts": 20, "rear_posts": 34, "rocks": 8, "firs": 2},
        "medium": {"barriers": 40, "trunks": 90, "posts": 36, "rear_posts": 70, "rocks": 10, "firs": 3},
        "dense": {"barriers": 70, "trunks": 150, "posts": 60, "rear_posts": 120, "rocks": 12, "firs": 4},
    }
    limits = density_limits[args.obstacle_density]
    root_is_mars = root_path.startswith("/World/MarsGale")

    def terrain_xyz(translate):
        if not root_is_mars:
            return translate
        x, y, z_offset = translate
        return (x, y, mars_terrain_height(x, y) + z_offset)

    def route_spread_xy(idx, lateral_min, lateral_max, phase=0.0):
        t = (idx * 0.61803398875 + phase) % 1.0
        route_idx = min(len(MARS_ROUTE_WAYPOINTS) - 2, int(t * (len(MARS_ROUTE_WAYPOINTS) - 1)))
        local_t = t * (len(MARS_ROUTE_WAYPOINTS) - 1) - route_idx
        x0, y0 = MARS_ROUTE_WAYPOINTS[route_idx]
        x1, y1 = MARS_ROUTE_WAYPOINTS[route_idx + 1]
        cx = x0 + (x1 - x0) * local_t
        cy = y0 + (y1 - y0) * local_t
        tangent_x = x1 - x0
        tangent_y = y1 - y0
        tangent_len = max(1e-6, math.hypot(tangent_x, tangent_y))
        nx = -tangent_y / tangent_len
        ny = tangent_x / tangent_len
        side = -1.0 if idx % 2 == 0 else 1.0
        lateral_span = max(0.0, lateral_max - lateral_min)
        lateral = side * (lateral_min + lateral_span * (((idx * 37) % 23) / 22.0))
        along = (((idx * 19) % 13) - 6) * 0.55
        x = cx + lateral * nx + along * tangent_x / tangent_len
        y = cy + lateral * ny + along * tangent_y / tangent_len
        if -5.2 < x < 5.8 and -4.2 < y < 4.2:
            x += 7.5
            y += 4.0 if idx % 2 else -4.0
        return x, y

    def outer_spread_xy(idx):
        angle = 0.47 + idx * 2.399963
        radius = 18.0 + 2.8 * (idx % 15)
        x = 30.0 + radius * math.cos(angle)
        y = 1.5 + 0.72 * radius * math.sin(angle)
        if -5.2 < x < 5.8 and -4.2 < y < 4.2:
            x += 9.0
        return x, y

    def rear_spread_xy(idx):
        angle = math.pi - 0.72 + ((idx * 0.61803398875) % 1.44)
        radius = 7.5 + 2.7 * (idx % 14) + 0.38 * (idx // 14)
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        if x > -4.5:
            x -= 8.0 + 0.45 * (idx % 9)
        if -4.5 < y < 4.5 and x > -13.0:
            y += 5.5 if idx % 2 else -5.5
        return x, y

    stage.DefinePrim(f"{root_path}/UprightBarriers", "Xform")
    for idx in range(limits["barriers"]):
        x, y = route_spread_xy(idx + 17, 3.8, 16.5, phase=0.19)
        radius = 0.105 + 0.018 * (idx % 4)
        height = 1.20 + 0.18 * (idx % 5)
        define_cylinder(
            stage,
            f"{root_path}/UprightBarriers/BarrierPost_{idx:02d}",
            (x, y, mars_terrain_height(x, y) + 0.5 * height if root_is_mars else 0.5 * height),
            radius,
            height,
            (0.42, 0.34, 0.25),
            True,
            barrier_mat,
        )

    stage.DefinePrim(f"{root_path}/StandingTrunks", "Xform")
    for idx in range(limits["trunks"]):
        if idx % 3 == 0:
            x, y = outer_spread_xy(idx)
        else:
            x, y = route_spread_xy(idx, 5.0, 26.0, phase=0.07)
        radius = 0.11 + 0.025 * (idx % 4)
        height = 1.55 + 0.22 * (idx % 5)
        define_cylinder(
            stage,
            f"{root_path}/StandingTrunks/Trunk_{idx:02d}",
            (x, y, mars_terrain_height(x, y) + 0.5 * height if root_is_mars else 0.5 * height),
            radius,
            height,
            (0.25, 0.16, 0.10),
            True,
            bark_mat,
        )

    stage.DefinePrim(f"{root_path}/OuterPosts", "Xform")
    for idx in range(limits["posts"]):
        x, y = outer_spread_xy(idx + 31)
        radius = 0.145 + 0.018 * (idx % 3)
        height = 1.95 + 0.20 * (idx % 4)
        define_cylinder(
            stage,
            f"{root_path}/OuterPosts/StandingPost_{idx:02d}",
            (x, y, mars_terrain_height(x, y) + 0.5 * height if root_is_mars else 0.5 * height),
            radius,
            height,
            (0.22, 0.14, 0.09),
            True,
            bark_mat,
        )

    stage.DefinePrim(f"{root_path}/RearPosts", "Xform")
    for idx in range(limits["rear_posts"]):
        x, y = rear_spread_xy(idx)
        radius = 0.095 + 0.020 * (idx % 5)
        height = 1.20 + 0.16 * (idx % 6)
        material = barrier_mat if idx % 3 == 0 else bark_mat
        define_cylinder(
            stage,
            f"{root_path}/RearPosts/RearPost_{idx:02d}",
            (x, y, mars_terrain_height(x, y) + 0.5 * height if root_is_mars else 0.5 * height),
            radius,
            height,
            (0.32, 0.22, 0.14),
            True,
            material,
        )
    log(
        f"created upright obstacle field "
        f"({limits['barriers']} barrier posts, {limits['trunks']} standing trunks, "
        f"{limits['posts']} outer posts, {limits['rear_posts']} rear posts)"
    )

    stage.DefinePrim(f"{root_path}/DetailRocks", "Xform")
    if root_is_mars:
        rock_specs = (
            ("Rock_00", (9.4, 5.9, 0.44), 0.55, (1.35, 0.92, 0.55), (4.0, 11.0, 32.0)),
            ("Rock_01", (14.8, -10.6, 0.48), 0.70, (1.55, 1.05, 0.62), (0.0, -8.0, 91.0)),
            ("Rock_02", (25.0, 9.8, 0.54), 0.82, (1.80, 1.08, 0.68), (-5.0, 9.0, 148.0)),
            ("Rock_03", (-12.5, -9.8, 0.42), 0.58, (1.35, 0.95, 0.58), (6.0, 13.0, 236.0)),
            ("Rock_04", (36.0, 2.5, 0.58), 0.72, (1.62, 1.00, 0.65), (3.0, -12.0, 307.0)),
            ("Rock_05", (20.4, 12.5, 0.46), 0.52, (1.20, 0.82, 0.50), (-4.0, 7.0, 19.0)),
            ("Rock_06", (-19.5, 3.8, 0.50), 0.66, (1.45, 1.12, 0.60), (2.0, -10.0, 136.0)),
        )
        for idx, (name, translate, radius, scale, rotate) in enumerate(rock_specs[: limits["rocks"]]):
            define_irregular_rock(
                stage,
                f"{root_path}/DetailRocks/{name}",
                terrain_xyz(translate),
                radius,
                (0.42, 0.35, 0.29),
                True,
                dark_mat,
                rotate=rotate,
                scale=scale,
                seed=idx + 3,
            )
        log(f"created controlled procedural Mars detail rocks ({min(limits['rocks'], len(rock_specs))})")
    else:
        rock_specs = (
            ("RockAssetA", RUBICON_DETAIL_ASSETS["rock01"], (6.8, 5.8, 0.14), (0.0, 0.0, 32.0), (0.82, 0.82, 0.82)),
            ("RockAssetB", RUBICON_DETAIL_ASSETS["rock02"], (13.8, -10.5, 0.10), (0.0, 0.0, 91.0), (1.05, 1.05, 1.05)),
            ("RockAssetC", RUBICON_DETAIL_ASSETS["rockpile"], (25.0, 9.8, 0.12), (0.0, 0.0, 148.0), (0.92, 0.92, 0.92)),
            ("RockAssetD", RUBICON_DETAIL_ASSETS["rock01"], (-12.5, -9.8, 0.10), (0.0, 0.0, 236.0), (0.72, 0.72, 0.72)),
            ("RockAssetE", RUBICON_DETAIL_ASSETS["rock02"], (36.0, 2.5, 0.14), (0.0, 0.0, 307.0), (0.86, 0.86, 0.86)),
            ("StumpAssetA", RUBICON_DETAIL_ASSETS["stump"], (18.8, 2.0, 0.03), (0.0, 0.0, 19.0), (0.90, 0.90, 0.90)),
            ("StumpAssetB", RUBICON_DETAIL_ASSETS["stump"], (-18.5, 3.0, 0.03), (0.0, 0.0, 136.0), (0.78, 0.78, 0.78)),
        )
        for name, asset_path, translate, rotate, scale in rock_specs[: limits["rocks"]]:
            detail = add_detail_asset(
                stage,
                f"{root_path}/DetailRocks/{name}",
                asset_path,
                terrain_xyz(translate),
                rotate=rotate,
                scale=scale,
                collision=True,
                material=None,
            )
            if detail is None:
                define_sphere(
                    stage,
                    f"{root_path}/DetailRocks/{name}",
                    terrain_xyz(translate),
                    0.75,
                    (0.34, 0.29, 0.24),
                    True,
                    dark_mat,
                    rotate=rotate,
                    scale=(1.5, 1.0, 0.55),
                )

    if include_vegetation:
        stage.DefinePrim(f"{root_path}/Vegetation", "Xform")
        fir_specs = (
            ("FirA", (-18.0, 13.0, 0.0), (0.0, 0.0, 14.0), (0.85, 0.85, 0.85)),
            ("FirB", (-24.0, -15.0, 0.0), (0.0, 0.0, 88.0), (0.95, 0.95, 0.95)),
            ("FirC", (18.0, 16.5, 0.0), (0.0, 0.0, 167.0), (0.78, 0.78, 0.78)),
            ("FirD", (34.0, -15.5, 0.0), (0.0, 0.0, 241.0), (0.90, 0.90, 0.90)),
        )
        for name, translate, rotate, scale in fir_specs[: limits["firs"]]:
            add_detail_asset(
                stage,
                f"{root_path}/Vegetation/{name}",
                OFFROAD_NAV_DETAIL_ASSETS["douglas_fir"],
                terrain_xyz(translate),
                rotate=rotate,
                scale=scale,
                collision=True,
            )


def add_mars_gale_base(stage, materials):
    stage.DefinePrim("/World/MarsGale", "Xform")
    add_mars_surface_skin(stage, materials)
    if args.show_start_pad:
        pad_scale = (2.15, 1.55, 0.025)
        pad_z = mars_terrain_height(0.0, 0.0) + 0.5 * pad_scale[2] + 0.006
        define_box(
            stage,
            "/World/MarsGale/StartPad",
            (0.0, 0.0, pad_z),
            pad_scale,
            (0.44, 0.32, 0.22),
            True,
            materials.get("route_dust"),
        )
    else:
        log("Mars start pad disabled; Hunter starts directly on sampled terrain")
    add_mars_route_surface(stage, materials)
    add_mars_boulder_field(stage, materials)
    add_mars_pebble_field(stage, materials)
    add_dense_obstacle_field(stage, materials, "/World/MarsGale/DenseObstacles", include_vegetation=False)
    return stage.GetPrimAtPath("/World/MarsGale")


def add_offroad_nav_base(stage, materials):
    stage.DefinePrim("/World/OffroadNav", "Xform")
    if args.show_start_pad:
        define_box(
            stage,
            "/World/OffroadNav/StartPad",
            (0.0, 0.0, -0.020),
            (11.0, 8.0, 0.045),
            (0.24, 0.28, 0.18),
            True,
            materials.get("soil"),
        )
    else:
        log("Offroad-Nav start pad disabled; Hunter starts directly on terrain")
    if args.show_route_surface or args.show_route_guide:
        define_box(
            stage,
            "/World/OffroadNav/ForwardSightLine",
            (6.0, 0.0, 0.012),
            (9.5, 0.12, 0.006),
            (0.82, 0.74, 0.52),
            False,
            materials.get("lane"),
        )
    add_dense_obstacle_field(stage, materials, "/World/OffroadNav/DenseObstacles", include_vegetation=True)
    return stage.GetPrimAtPath("/World/OffroadNav")


def disable_known_bad_offroad_nav_instancers(stage):
    if args.scene_profile != "offroad_nav":
        return
    bad_instancers = (
        "/World/OffroadNav/Terrain/Asset/Vegetation_Parent/Grass_Instancer",
    )
    for prim_path in bad_instancers:
        prim = stage.OverridePrim(prim_path)
        prim.SetActive(False)
        log(f"disabled Offroad-Nav instancer for Isaac 5.1 compatibility: {prim_path}")


def add_project_roads(stage, materials):
    stage.DefinePrim("/World/Roads", "Xform")
    road_mat = materials.get("asphalt")
    lane_mat = materials.get("lane")
    define_box(stage, "/World/Roads/MainRoad", (8.0, 0.0, -0.014), (116.0, 6.0, 0.04), (0.08, 0.085, 0.085), True, road_mat)
    define_box(stage, "/World/Roads/CrossRoad", (31.0, 0.0, -0.013), (5.2, 82.0, 0.04), (0.08, 0.085, 0.085), True, road_mat)
    define_box(stage, "/World/Roads/NorthApron", (48.0, 21.0, -0.012), (38.0, 22.0, 0.035), (0.12, 0.125, 0.12), True, materials.get("concrete"))
    define_box(stage, "/World/Roads/SouthApron", (50.0, -23.0, -0.012), (42.0, 24.0, 0.035), (0.12, 0.125, 0.12), True, materials.get("concrete"))

    for idx, x in enumerate(range(-44, 62, 6)):
        define_box(
            stage,
            f"/World/Roads/LaneMark_{idx:02d}",
            (float(x), 0.0, 0.011),
            (2.3, 0.08, 0.006),
            (0.92, 0.90, 0.78),
            False,
            lane_mat,
        )
    define_box(stage, "/World/Roads/NorthEdgeLine", (8.0, 2.82, 0.011), (116.0, 0.06, 0.006), (0.92, 0.90, 0.78), False, lane_mat)
    define_box(stage, "/World/Roads/SouthEdgeLine", (8.0, -2.82, 0.011), (116.0, 0.06, 0.006), (0.92, 0.90, 0.78), False, lane_mat)
    for idx, y in enumerate(range(-34, 38, 6)):
        define_box(
            stage,
            f"/World/Roads/CrossLaneMark_{idx:02d}",
            (31.0, float(y), 0.012),
            (0.08, 2.3, 0.006),
            (0.92, 0.90, 0.78),
            False,
            lane_mat,
        )


def add_project_walls(stage, materials):
    stage.DefinePrim("/World/SceneWalls", "Xform")
    mat = materials.get("digital_wall")
    wall_color = (0.08, 0.13, 0.16)
    define_box(stage, "/World/SceneWalls/WallNorthA", (18.0, 31.0, 1.7), (28.0, 0.65, 3.4), wall_color, True, mat)
    define_box(stage, "/World/SceneWalls/WallNorthB", (56.0, 36.0, 1.7), (28.0, 0.65, 3.4), wall_color, True, mat)
    define_box(stage, "/World/SceneWalls/WallSouthA", (18.0, -31.0, 1.7), (28.0, 0.65, 3.4), wall_color, True, mat)
    define_box(stage, "/World/SceneWalls/WallSouthB", (56.0, -37.0, 1.7), (28.0, 0.65, 3.4), wall_color, True, mat)
    define_box(stage, "/World/SceneWalls/WallWest", (-26.0, 0.0, 1.7), (38.0, 0.65, 3.4), wall_color, True, mat, rotate=(0.0, 0.0, 90.0))
    define_box(stage, "/World/SceneWalls/WallEast", (75.0, 0.0, 1.7), (52.0, 0.65, 3.4), wall_color, True, mat, rotate=(0.0, 0.0, 90.0))


def same_asset(asset, candidate):
    try:
        return Path(asset).resolve() == Path(candidate).resolve()
    except OSError:
        return Path(asset).absolute() == Path(candidate).absolute()


def map_reference_options(asset):
    if same_asset(asset, MARS_GALE_MAP_ASSET):
        return {
            "path": "/World/MarsGale/CraterPatch",
            "translate": (0.0, 0.0, MARS_VISUAL_Z_OFFSET),
            "rotate": (0.0, 0.0, 0.0),
            "scale": (1.0, 1.0, 1.0),
            "material": "sand",
            "collision": False,
            "collision_approximation": "meshSimplification",
            "prim_path": None,
        }
    if same_asset(asset, MARS_GALE_COLLISION_ASSET):
        return {
            "path": "/World/MarsGale/CollisionPatch",
            "translate": (0.0, 0.0, MARS_COLLISION_Z_OFFSET),
            "rotate": (0.0, 0.0, 0.0),
            "scale": (1.0, 1.0, 1.0),
            "material": "sand",
            "collision": True,
            "collision_approximation": "none",
            "prim_path": None,
        }
    for name, offroad_asset in OFFROAD_NAV_ASSETS.items():
        if same_asset(asset, offroad_asset):
            return {
                "path": "/World/OffroadNav/Terrain",
                "translate": (0.0, 0.0, 0.0),
                "rotate": (0.0, 0.0, 0.0),
                "scale": (1.0, 1.0, 1.0),
                "material": None,
                "collision": False,
                "collision_approximation": "meshSimplification",
                "prim_path": "/Root" if name == "barakuda" else "/map/GeneratedForest",
            }
    return {
        "path": "/World/ProjectMap",
        "translate": (0.0, 0.0, 0.0),
        "rotate": (0.0, 0.0, 0.0),
        "scale": (1.0, 1.0, 1.0),
        "material": None,
        "collision": False,
        "collision_approximation": "meshSimplification",
        "prim_path": None,
    }


def add_mars_collision_asset(stage, materials):
    if not is_mars_profile() or not MARS_GALE_COLLISION_ASSET.exists():
        return None
    if stage.GetPrimAtPath("/World/MarsGale/CollisionPatch"):
        return stage.GetPrimAtPath("/World/MarsGale/CollisionPatch")
    usd_path = convert_asset_to_usd(MARS_GALE_COLLISION_ASSET)
    if usd_path is None:
        log("warning: Mars Gale collision OBJ conversion failed; visual terrain will not get the Gazebo collision mesh")
        return None
    collision_prim = add_reference(
        stage,
        "/World/MarsGale/CollisionPatch",
        usd_path,
        translate=(0.0, 0.0, MARS_COLLISION_Z_OFFSET),
    )
    mesh_count = bind_material_under(stage, str(collision_prim.GetPath()), materials.get("sand"))
    mesh_count = apply_collision_under(
        stage,
        str(collision_prim.GetPath()),
        approximation="none",
        contact_offset=0.030,
        rest_offset=0.001,
    )
    try:
        UsdGeom.Imageable(collision_prim).MakeInvisible()
    except Exception as exc:
        log(f"warning: failed to hide Mars collision patch: {exc}")
    log(f"loaded Mars Gale Gazebo collision OBJ ({mesh_count} mesh prims, high-friction sand material)")
    return collision_prim


def add_project_map(stage, materials):
    asset = resolve_map_asset(args.map_asset)
    use_mars_surface_skin_as_visual = (
        is_mars_profile()
        and args.mars_surface_skin != "none"
        and asset is not None
        and (same_asset(asset, MARS_GALE_MAP_ASSET) or same_asset(asset, MARS_GALE_COLLISION_ASSET))
    )
    if is_mars_profile() and asset is not None and same_asset(asset, MARS_GALE_MAP_ASSET) and args.mars_visual_detail == "fast":
        log("using Mars Gale Gazebo collision OBJ as fast visible terrain; use --mars-visual-detail full for the high-poly STL visual mesh")
        asset = MARS_GALE_COLLISION_ASSET.resolve()
        use_mars_surface_skin_as_visual = args.mars_surface_skin != "none"

    if args.scene_profile == "light":
        define_box(stage, "/World/Ground", (0.0, 0.0, -0.035), (45.0, 45.0, 0.03), (0.34, 0.42, 0.31), True, materials.get("grass"))
    elif args.scene_profile == "rubicon":
        define_box(stage, "/World/StartPad", (0.0, 0.0, -0.018), (22.0, 18.0, 0.045), (0.10, 0.10, 0.095), True, materials.get("asphalt"))
        define_box(stage, "/World/StartPad/CenterLine", (0.0, 0.0, 0.012), (18.0, 0.08, 0.006), (0.92, 0.90, 0.78), False, materials.get("lane"))
    elif args.scene_profile in {"mars_gale", "offroad"}:
        add_mars_gale_base(stage, materials)
        if use_mars_surface_skin_as_visual:
            log("using generated PBR Mars surface skin as visible terrain; loading raw Mars mesh only as hidden collision")
            add_mars_collision_asset(stage, materials)
            return stage.GetPrimAtPath("/World/MarsGale/SurfaceSkin") or stage.GetPrimAtPath("/World/MarsGale")
        if asset is None:
            log("map asset disabled; using procedural Mars Gale fallback terrain")
            return add_mars_generated_terrain(stage, materials)
    elif args.scene_profile == "offroad_nav":
        add_offroad_nav_base(stage, materials)
        if asset is None:
            log("map asset disabled; using procedural Mars-style fallback terrain")
            stage.DefinePrim("/World/MarsGale", "Xform")
            return add_mars_generated_terrain(stage, materials)
    else:
        add_generated_terrain(stage, materials)
        add_project_roads(stage, materials)
        add_project_walls(stage, materials)

    if asset is None:
        return None
    usd_path = convert_asset_to_usd(asset)
    if usd_path is None:
        return None
    options = map_reference_options(asset)
    log(f"loading project map asset: {asset}")
    map_prim = add_reference(
        stage,
        options["path"],
        usd_path,
        translate=options["translate"],
        rotate=options["rotate"],
        scale=options["scale"],
        reference_prim_path=options["prim_path"],
    )
    if options["material"]:
        mesh_count = bind_material_under(stage, str(map_prim.GetPath()), materials.get(options["material"]))
        log(f"applied {options['material']} material under {map_prim.GetPath()} ({mesh_count} mesh prims)")
    if is_mars_profile() and options["path"].startswith("/World/MarsGale"):
        uv_count = apply_xy_uvs_under(stage, str(map_prim.GetPath()), uv_scale=42.0)
        style_count = apply_visual_style_under(stage, str(map_prim.GetPath()), display_color=(0.66, 0.43, 0.25))
        log(f"applied Mars terrain viewport style under {map_prim.GetPath()} ({style_count} mesh prims, {uv_count} UV sets)")
    if options["collision"]:
        mesh_count = apply_collision_under(
            stage,
            str(map_prim.GetPath()),
            approximation=options["collision_approximation"],
            contact_offset=0.030 if is_mars_profile() else 0.040,
            rest_offset=0.001 if is_mars_profile() else 0.003,
        )
        log(f"applied mesh collision under {map_prim.GetPath()} ({mesh_count} mesh prims)")
    if same_asset(asset, MARS_GALE_MAP_ASSET):
        add_mars_collision_asset(stage, materials)
    disable_known_bad_offroad_nav_instancers(stage)
    return map_prim


def add_project_landmarks(stage, materials):
    if args.no_landmarks:
        return
    stage.DefinePrim("/World/Landmarks", "Xform")
    if args.scene_profile == "light":
        landmarks = DEFAULT_LANDMARKS[:5]
    elif args.scene_profile == "rubicon":
        landmarks = DEFAULT_LANDMARKS[:6] + TREE_LANDMARKS[:6]
    elif args.scene_profile in {"mars_gale", "offroad"}:
        landmarks = ()
    elif args.scene_profile == "offroad_nav":
        stage.DefinePrim("/World/Landmarks/Trees", "Xform")
        landmarks = TREE_LANDMARKS[:12]
    else:
        stage.DefinePrim("/World/Landmarks/Trees", "Xform")
        landmarks = DEFAULT_LANDMARKS + TREE_LANDMARKS

    usd_cache = {}
    for asset, prim_path, translate, rotate, scale in landmarks:
        if not asset.exists():
            log(f"warning: landmark asset missing, skipping: {asset}")
            continue
        usd_path = usd_cache.get(asset)
        if usd_path is None:
            usd_path = convert_asset_to_usd(asset)
            usd_cache[asset] = usd_path
        if usd_path is None:
            continue
        log(f"loading project landmark: {asset.name} -> {prim_path}")
        add_reference(stage, prim_path, usd_path, translate=translate, rotate=rotate, scale=scale)
    if args.scene_profile not in {"mars_gale", "offroad"}:
        add_front_feature_sign(stage, materials)


def add_front_feature_sign(stage, materials):
    stage.DefinePrim("/World/Landmarks/FrontSign", "Xform")
    define_cylinder(
        stage,
        "/World/Landmarks/FrontSign/Post",
        (5.4, -3.25, 0.48),
        0.035,
        0.84,
        (0.82, 0.82, 0.78),
        True,
        materials.get("metal"),
    )
    define_box(
        stage,
        "/World/Landmarks/FrontSign/Panel",
        (5.4, -3.25, 1.04),
        (0.04, 0.76, 0.54),
        (0.86, 0.05, 0.04),
        True,
        materials.get("red"),
    )
    define_box(
        stage,
        "/World/Landmarks/FrontSign/CenterPatch",
        (5.358, -3.25, 1.04),
        (0.045, 0.48, 0.14),
        (0.96, 0.96, 0.90),
        False,
        materials.get("white"),
    )


def tune_static_world_collisions(stage):
    tuning_specs = (
        ("/World/MarsGale/CollisionPatch", 0.028, 0.001),
        ("/World/MarsGale/Boulders", 0.050, 0.003),
        ("/World/MarsGale/DenseObstacles", 0.060, 0.003),
        ("/World/OffroadNav/DenseObstacles", 0.055, 0.003),
        ("/World/SceneWalls", 0.050, 0.003),
        ("/World/Roads", 0.026, 0.002),
        ("/World/Ground", 0.026, 0.002),
    )
    total = 0
    summaries = []
    for root_path, contact_offset, rest_offset in tuning_specs:
        if not stage.GetPrimAtPath(root_path):
            continue
        count = tune_collision_surfaces_under(
            stage,
            root_path,
            contact_offset=contact_offset,
            rest_offset=rest_offset,
        )
        if count:
            total += count
            summaries.append(f"{root_path}:{count}")
    if total:
        log(f"tuned static world collision contact offsets ({total} collision prims; {', '.join(summaries)})")


def configure_stage():
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    stage.DefinePrim(WORLD, "Xform")
    if not stage.GetPrimAtPath("/physicsScene"):
        scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene"))
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(9.81)
        try:
            physx_scene = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
            physx_scene.CreateSolverTypeAttr().Set("TGS")
            physx_scene.CreateEnableCCDAttr().Set(True)
            physx_scene.CreateEnableStabilizationAttr().Set(True)
            physx_scene.CreateEnableEnhancedDeterminismAttr().Set(True)
            physx_scene.CreateMinPositionIterationCountAttr().Set(4)
            physx_scene.CreateMaxPositionIterationCountAttr().Set(12)
            physx_scene.CreateMinVelocityIterationCountAttr().Set(1)
            physx_scene.CreateMaxVelocityIterationCountAttr().Set(4)
            physx_scene.CreateFrictionOffsetThresholdAttr().Set(0.04)
            physx_scene.CreateFrictionCorrelationDistanceAttr().Set(0.025)
        except Exception as exc:
            log(f"warning: failed to configure PhysX scene tuning: {exc}")

    materials = create_scene_materials(stage)
    add_project_map(stage, materials)
    add_project_landmarks(stage, materials)
    tune_static_world_collisions(stage)

    sky = UsdLux.DomeLight.Define(stage, "/World/SkyLight")
    if args.scene_profile in {"mars_gale", "offroad"}:
        sky.CreateIntensityAttr(330.0)
        sky.CreateColorAttr(Gf.Vec3f(0.72, 0.66, 0.58))
    else:
        sky.CreateIntensityAttr(420.0)
        sky.CreateColorAttr(Gf.Vec3f(0.74, 0.84, 1.0))

    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    if args.scene_profile in {"mars_gale", "offroad"}:
        sun.CreateIntensityAttr(2850.0)
        sun.CreateAngleAttr(0.28)
        sun.CreateColorAttr(Gf.Vec3f(1.0, 0.89, 0.74))
    else:
        sun.CreateIntensityAttr(2200.0)
        sun.CreateAngleAttr(0.55)
    sun_xform = UsdGeom.XformCommonAPI(sun)
    sun_xform.SetRotate((-42.0, 0.0, 28.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)

    fill = UsdLux.SphereLight.Define(stage, "/World/SoftFill")
    fill.CreateIntensityAttr(95.0 if args.scene_profile in {"mars_gale", "offroad"} else 120.0)
    fill.CreateRadiusAttr(7.0 if args.scene_profile in {"mars_gale", "offroad"} else 6.0)
    if args.scene_profile in {"mars_gale", "offroad"}:
        fill.CreateColorAttr(Gf.Vec3f(0.74, 0.66, 0.56))
    fill_xform = UsdGeom.XformCommonAPI(fill)
    fill_xform.SetTranslate(Gf.Vec3d(-8.0, -12.0, 8.0))
    return stage


def set_look_at(prim, eye, target, up=(0.0, 0.0, 1.0)):
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    transform_op = xformable.AddTransformOp()
    look_at = Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(*eye),
        Gf.Vec3d(*target),
        Gf.Vec3d(*up),
    ).GetInverse()
    transform_op.Set(look_at)


def create_viewer_camera(stage):
    log("creating Isaac GUI viewer camera focused on Hunter")
    spawn_x, spawn_y = scene_spawn_xy()
    if is_mars_profile():
        base_z = scene_robot_base_z(spawn_x, spawn_y)
        route_target = (32.0, -4.8, mars_terrain_height(32.0, -4.8) + 1.2)
        camera_specs = {
            "close": (VIEW_CAMERA, (spawn_x + 3.4, spawn_y - 10.5, base_z + 7.4), (spawn_x + 0.1, spawn_y, base_z + 0.65), 24.0),
            "overview": (f"{WORLD}/OverviewCamera", (28.0, -40.0, mars_terrain_height(28.0, -40.0) + 34.0), route_target, 20.0),
            "top": (f"{WORLD}/TopCamera", (34.0, -8.0, mars_terrain_height(34.0, -8.0) + 48.0), route_target, 24.0),
        }
    elif args.scene_profile == "offroad_nav":
        camera_specs = {
            "close": (VIEW_CAMERA, (4.8, -6.4, 2.8), (0.15, 0.0, 0.72), 26.0),
            "overview": (f"{WORLD}/OverviewCamera", (20.0, -26.0, 18.0), (9.0, 0.0, 1.0), 21.0),
            "top": (f"{WORLD}/TopCamera", (18.0, -4.0, 32.0), (9.0, 0.0, 0.8), 24.0),
        }
    else:
        camera_specs = {
            "close": (VIEW_CAMERA, (6.5, -7.4, 3.6), (0.6, 0.0, 0.55), 26.0),
            "overview": (f"{WORLD}/OverviewCamera", (18.0, -22.0, 13.0), (7.0, 0.0, 0.7), 21.0),
            "top": (f"{WORLD}/TopCamera", (8.0, 0.0, 28.0), (8.0, 0.0, 0.4), 24.0),
        }

    active_camera_path = None
    for name, (path, eye, target, focal_length) in camera_specs.items():
        camera = UsdGeom.Camera.Define(stage, path)
        set_look_at(camera.GetPrim(), eye, target)
        camera.GetProjectionAttr().Set("perspective")
        camera.GetFocalLengthAttr().Set(focal_length)
        camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.03, 1500.0))
        if name == args.viewer_camera:
            active_camera_path = camera.GetPath()
    return active_camera_path or stage.GetPrimAtPath(VIEW_CAMERA).GetPath()


def activate_viewer_camera(camera_path):
    if args.headless:
        return
    try:
        from omni.kit.viewport.utility import get_active_viewport

        viewport = get_active_viewport()
        if viewport:
            viewport.camera_path = camera_path
            log(f"active Isaac viewport camera: {camera_path}")
        else:
            log("warning: no active Isaac viewport found")
    except Exception as exc:
        log(f"warning: failed to activate viewer camera: {exc}")


def select_robot_in_stage():
    if args.headless:
        return
    try:
        omni.usd.get_context().get_selection().set_selected_prim_paths([ROBOT], True)
        log(f"selected Hunter in Isaac Stage: {ROBOT}")
    except Exception as exc:
        log(f"warning: failed to select Hunter in Isaac Stage: {exc}")


def log_hunter_spawn_clearance(spawn_x, spawn_y, robot_base_z):
    if not is_mars_profile():
        return
    collision_ground_z = mars_terrain_height(spawn_x, spawn_y)
    visual_ground_z = mars_surface_visual_height(spawn_x, spawn_y)
    approximate_lowest_tire_z = robot_base_z - 0.1535 - 0.0214 - WHEEL_RADIUS
    log(
        "Hunter spawn clearance: "
        f"base_z={robot_base_z:.3f}, collision_ground_z={collision_ground_z:.3f}, "
        f"visual_ground_z={visual_ground_z:.3f}, "
        f"approx_tire_clearance_collision={approximate_lowest_tire_z - collision_ground_z:+.3f}m, "
        f"approx_tire_clearance_visual={approximate_lowest_tire_z - visual_ground_z:+.3f}m"
    )


def add_hunter_auxiliary_colliders(stage):
    if not stage.GetPrimAtPath(BASE_LINK):
        return

    root_path = f"{BASE_LINK}/isaac_body_colliders"
    stage.DefinePrim(root_path, "Xform")
    chassis_mat = stage.GetPrimAtPath("/World/Materials/HunterChassisContact")
    material = UsdShade.Material(chassis_mat) if chassis_mat else None
    collider_specs = (
        ("body_hull", (0.045, 0.0, -0.058), (0.940, 0.600, 0.230)),
        ("front_bumper", (0.520, 0.0, -0.102), (0.120, 0.650, 0.150)),
        ("rear_bumper", (-0.420, 0.0, -0.108), (0.120, 0.590, 0.140)),
        ("side_guard_left", (0.040, 0.338, -0.112), (0.760, 0.050, 0.150)),
        ("side_guard_right", (0.040, -0.338, -0.112), (0.760, 0.050, 0.150)),
        ("front_corner_left", (0.460, 0.330, -0.105), (0.135, 0.080, 0.145)),
        ("front_corner_right", (0.460, -0.330, -0.105), (0.135, 0.080, 0.145)),
        ("rear_corner_left", (-0.360, 0.315, -0.110), (0.125, 0.080, 0.135)),
        ("rear_corner_right", (-0.360, -0.315, -0.110), (0.125, 0.080, 0.135)),
    )
    for name, translate, scale in collider_specs:
        collider = define_box(
            stage,
            f"{root_path}/{name}",
            translate,
            scale,
            (0.0, 0.0, 0.0),
            True,
            material,
        )
        make_invisible(collider)
        tune_contact_surface(collider.GetPrim(), contact_offset=0.045, rest_offset=0.002)
    log(f"added {len(collider_specs)} invisible Hunter body/bumper collision boxes under base_link")


def tune_hunter_mass_properties(stage):
    mass_specs = {
        "base_link": 24.73,
        "fr_steer_left_link": 3.149,
        "fr_left_link": 3.149,
        "fr_steer_right_link": 3.149,
        "fr_right_link": 3.149,
        "re_left_link": 3.149,
        "re_right_link": 3.149,
        "steering_wheel_link": 1.0,
    }
    count = 0
    for link_name, mass in mass_specs.items():
        prim = stage.GetPrimAtPath(f"{ROBOT}/{link_name}")
        if not prim:
            continue
        try:
            mass_api = UsdPhysics.MassAPI.Apply(prim)
            mass_api.CreateMassAttr().Set(float(mass))
            count += 1
        except Exception as exc:
            log(f"warning: failed to set Hunter mass for {link_name}: {exc}")
    if count:
        log(f"authored Hunter URDF mass properties on {count} links")


def import_hunter_robot(stage):
    log(f"importing Hunter SE URDF/STL model: {args.robot_urdf}")
    resolved_urdf = resolve_urdf_source(args.robot_urdf)
    status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    if not status:
        raise RuntimeError("failed to create Isaac URDF import config")
    import_config.merge_fixed_joints = False
    import_config.fix_base = False
    import_config.make_default_prim = False
    import_config.import_inertia_tensor = True
    import_config.create_physics_scene = True
    import_config.convex_decomp = False
    import_config.self_collision = False
    import_config.distance_scale = 1.0

    status, articulation_root = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=str(resolved_urdf),
        import_config=import_config,
        get_articulation_root=True,
    )
    if not status:
        raise RuntimeError(f"Isaac URDF import failed for {resolved_urdf}")

    import_root = "/" + articulation_root.strip("/").split("/")[0]
    if import_root != ROBOT:
        if stage.GetPrimAtPath(ROBOT):
            omni.kit.commands.execute("DeletePrims", paths=[ROBOT])
        omni.kit.commands.execute("MovePrimCommand", path_from=import_root, path_to=ROBOT, keep_world_transform=False)

    robot = stage.GetPrimAtPath(ROBOT)
    robot_pose_ops = reset_pose_ops(robot)
    spawn_x, spawn_y = scene_spawn_xy()
    robot_base_z = scene_robot_base_z(spawn_x, spawn_y)
    set_pose_ops(robot_pose_ops, (spawn_x, spawn_y, robot_base_z), (0.0, 0.0, float(args.spawn_yaw_deg)))
    log(f"placed Hunter base at ({spawn_x:.2f}, {spawn_y:.2f}, {robot_base_z:.3f}) using terrain-aligned spawn")
    log_hunter_spawn_clearance(spawn_x, spawn_y, robot_base_z)
    apply_hunter_visibility_style(stage)
    add_hunter_auxiliary_colliders(stage)
    tune_hunter_mass_properties(stage)
    configure_hunter_contact_materials(stage)
    configure_hunter_joint_drives(stage)
    return robot_pose_ops


def apply_hunter_visibility_style(stage):
    if args.no_vehicle_contrast:
        return

    body_mat = create_preview_material(stage, "/World/Materials/HunterBodyWhite", (0.86, 0.88, 0.86), 0.54)
    wheel_mat = create_preview_material(stage, "/World/Materials/HunterWheelBlack", (0.018, 0.018, 0.018), 0.72)
    accent_mat = create_preview_material(stage, "/World/Materials/HunterAccentBlue", (0.05, 0.42, 0.86), 0.36)
    apply_physics_material(wheel_mat, static_friction=2.35, dynamic_friction=1.85, restitution=0.01)
    mesh_count = 0
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if any(sensor_path in prim_path for sensor_path in ("/livox_base/", "/imu_link/", "/camera_link/")):
            continue
        in_robot_tree = prim_path == ROBOT or prim_path.startswith(f"{ROBOT}/")
        in_urdf_mesh_library = prim_path.startswith("/visuals/") or prim_path.startswith("/meshes/")
        if not in_robot_tree and not in_urdf_mesh_library:
            continue
        if not prim.IsA(UsdGeom.Mesh):
            continue
        lower_path = prim_path.lower()
        mesh = UsdGeom.Mesh(prim)
        if any(name in lower_path for name in ("wheel", "tire", "fr_left", "fr_right", "re_left", "re_right")):
            bind_material(mesh, wheel_mat)
        elif "front" in lower_path or "top" in lower_path or "cover" in lower_path:
            bind_material(mesh, accent_mat)
        else:
            bind_material(mesh, body_mat)
        mesh_count += 1
    if mesh_count == 0:
        type_counts = {}
        sample_paths = []
        for prim in stage.Traverse():
            prim_path = str(prim.GetPath())
            if not (
                prim_path == ROBOT
                or prim_path.startswith(f"{ROBOT}/")
                or prim_path.startswith("/visuals/")
                or prim_path.startswith("/meshes/")
            ):
                continue
            type_name = prim.GetTypeName() or "<untyped>"
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
            if len(sample_paths) < 16:
                sample_paths.append(f"{prim_path}:{type_name}")
        log(f"warning: no Hunter mesh prims found; robot prim types={type_counts}; sample={sample_paths}")
    log(f"applied high-contrast Hunter materials ({mesh_count} mesh prims)")


def prim_has_api(prim, api_schema):
    try:
        return prim.HasAPI(api_schema)
    except Exception:
        return False


def tune_contact_surface(prim, contact_offset=0.018, rest_offset=0.002):
    try:
        physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        physx_collision.CreateContactOffsetAttr().Set(float(contact_offset))
        physx_collision.CreateRestOffsetAttr().Set(float(rest_offset))
    except Exception as exc:
        log(f"warning: failed to tune PhysX contact surface on {prim.GetPath()}: {exc}")


def enable_rigid_body_ccd_under(stage, root_path):
    count = 0
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if not (prim_path == root_path or prim_path.startswith(f"{root_path}/")):
            continue
        if not prim_has_api(prim, UsdPhysics.RigidBodyAPI):
            continue
        try:
            physx_rigid = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            physx_rigid.CreateEnableCCDAttr().Set(True)
            count += 1
        except Exception as exc:
            log(f"warning: failed to enable CCD on {prim_path}: {exc}")
    return count


def _set_physx_rigid_float(physx_rigid, prim, method_name, attr_name, value):
    try:
        method = getattr(physx_rigid, method_name, None)
        if callable(method):
            method().Set(float(value))
            return True
        attr = prim.GetAttribute(attr_name)
        if not attr:
            attr = prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Float)
        attr.Set(float(value))
        return True
    except Exception:
        return False


def tune_rigid_body_damping_under(stage, root_path):
    count = 0
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if not (prim_path == root_path or prim_path.startswith(f"{root_path}/")):
            continue
        if not prim_has_api(prim, UsdPhysics.RigidBodyAPI):
            continue

        lower_path = prim_path.lower()
        if "base_link" in lower_path:
            linear_damping = 0.10
            angular_damping = 0.42
        elif any(name in lower_path for name in ("wheel", "fr_left", "fr_right", "re_left", "re_right")):
            linear_damping = 0.015
            angular_damping = 0.035
        else:
            linear_damping = 0.05
            angular_damping = 0.12

        try:
            physx_rigid = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            ok = True
            ok = _set_physx_rigid_float(
                physx_rigid, prim, "CreateLinearDampingAttr", "physxRigidBody:linearDamping", linear_damping
            ) and ok
            ok = _set_physx_rigid_float(
                physx_rigid, prim, "CreateAngularDampingAttr", "physxRigidBody:angularDamping", angular_damping
            ) and ok
            ok = _set_physx_rigid_float(
                physx_rigid, prim, "CreateMaxLinearVelocityAttr", "physxRigidBody:maxLinearVelocity", 3.0
            ) and ok
            ok = _set_physx_rigid_float(
                physx_rigid, prim, "CreateMaxAngularVelocityAttr", "physxRigidBody:maxAngularVelocity", 12.0
            ) and ok
            ok = _set_physx_rigid_float(
                physx_rigid, prim, "CreateMaxDepenetrationVelocityAttr", "physxRigidBody:maxDepenetrationVelocity", 1.25
            ) and ok
            if ok:
                count += 1
        except Exception as exc:
            log(f"warning: failed to tune rigid-body damping on {prim_path}: {exc}")
    return count


def configure_hunter_contact_materials(stage):
    wheel_mat = create_preview_material(stage, "/World/Materials/HunterWheelContact", (0.012, 0.012, 0.012), 0.84)
    chassis_mat = create_preview_material(stage, "/World/Materials/HunterChassisContact", (0.28, 0.29, 0.28), 0.78)
    apply_physics_material(wheel_mat, static_friction=3.0, dynamic_friction=2.45, restitution=0.01)
    apply_physics_material(chassis_mat, static_friction=1.0, dynamic_friction=0.82, restitution=0.02)

    wheel_count = 0
    chassis_count = 0
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if not (prim_path == ROBOT or prim_path.startswith(f"{ROBOT}/")):
            continue
        lower_path = prim_path.lower()
        is_collision = prim_has_api(prim, UsdPhysics.CollisionAPI) or "/collision" in lower_path or "/collisions" in lower_path
        if not is_collision:
            continue
        if any(name in lower_path for name in ("wheel", "tire", "fr_left", "fr_right", "re_left", "re_right")):
            bind_material(prim, wheel_mat)
            tune_contact_surface(prim, contact_offset=0.036, rest_offset=0.002)
            wheel_count += 1
        else:
            bind_material(prim, chassis_mat)
            tune_contact_surface(prim, contact_offset=0.026, rest_offset=0.002)
            chassis_count += 1
    ccd_count = enable_rigid_body_ccd_under(stage, ROBOT)
    damping_count = tune_rigid_body_damping_under(stage, ROBOT)
    log(
        f"applied Hunter contact materials "
        f"({wheel_count} wheel collision prims, {chassis_count} chassis collision prims, "
        f"CCD on {ccd_count} rigid links, damping on {damping_count} rigid links)"
    )


def get_drive(stage, joint_name):
    joint = stage.GetPrimAtPath(f"{ROBOT}/joints/{joint_name}")
    if not joint:
        return None
    drive = UsdPhysics.DriveAPI.Get(joint, "angular")
    if not drive:
        drive = UsdPhysics.DriveAPI.Apply(joint, "angular")
    return drive


def configure_hunter_joint_drives(stage):
    for joint_name in STEERING_JOINTS:
        drive = get_drive(stage, joint_name)
        if drive:
            drive.GetStiffnessAttr().Set(float(args.steering_position_kp))
            drive.GetDampingAttr().Set(float(args.steering_position_kd))
            drive.GetMaxForceAttr().Set(float(args.steering_max_effort))
    steering_wheel_drive = get_drive(stage, "steering_whee_joint")
    if steering_wheel_drive:
        steering_wheel_drive.GetStiffnessAttr().Set(6.0)
        steering_wheel_drive.GetDampingAttr().Set(1.0)
        steering_wheel_drive.GetMaxForceAttr().Set(25.0)
    for joint_name in WHEEL_JOINT_SIGNS:
        drive = get_drive(stage, joint_name)
        if drive:
            drive.GetStiffnessAttr().Set(0.0)
            drive.GetDampingAttr().Set(float(args.wheel_velocity_damping))
            drive.GetMaxForceAttr().Set(max(float(args.wheel_motor_effort), float(args.wheel_idle_brake_effort)))


def set_drive_target(drive, target_position=None, target_velocity=None):
    if not drive:
        return
    if target_position is not None:
        drive.GetTargetPositionAttr().Set(math.degrees(target_position))
    if target_velocity is not None:
        drive.GetTargetVelocityAttr().Set(math.degrees(target_velocity))


def define_xform(stage, path, translate=(0.0, 0.0, 0.0), rotate=(0.0, 0.0, 0.0)):
    prim = stage.DefinePrim(path, "Xform")
    xform = UsdGeom.XformCommonAPI(prim)
    xform.SetTranslate(Gf.Vec3d(*translate))
    xform.SetRotate(tuple(rotate), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    return prim


def create_sensor_materials(stage):
    return {
        "shell": create_preview_material(stage, "/World/Materials/SensorWarmWhite", (0.82, 0.80, 0.74), 0.46),
        "dark": create_preview_material(stage, "/World/Materials/SensorMatteBlack", (0.018, 0.018, 0.017), 0.68),
        "glass": create_preview_material(stage, "/World/Materials/SensorDarkGlass", (0.005, 0.010, 0.014), 0.18),
        "metal": create_preview_material(stage, "/World/Materials/SensorBrushedMetal", (0.56, 0.57, 0.55), 0.34, metallic=0.45),
        "board": create_preview_material(stage, "/World/Materials/ImuBoardGreen", (0.03, 0.26, 0.12), 0.58),
        "chip": create_preview_material(stage, "/World/Materials/ImuChipBlack", (0.010, 0.012, 0.012), 0.52),
        "realsense": create_preview_material(stage, "/World/Materials/RealsenseGraphite", (0.055, 0.060, 0.064), 0.44),
        "realsense_face": create_preview_material(stage, "/World/Materials/RealsenseFace", (0.010, 0.012, 0.014), 0.36),
        "accent": create_preview_material(stage, "/World/Materials/SensorAccentBlue", (0.05, 0.42, 0.86), 0.34),
    }


def add_mid360_procedural_visual(stage, materials, root_path=f"{LIVOX_BASE}/mid360_visual/procedural"):
    stage.DefinePrim(root_path, "Xform")
    shell = materials["shell"]
    dark = materials["dark"]
    glass = materials["glass"]
    metal = materials["metal"]
    accent = materials["accent"]

    define_box(
        stage,
        f"{root_path}/mount_plate",
        (0.0, 0.0, -0.008),
        (0.074, 0.060, 0.010),
        (0.50, 0.50, 0.48),
        False,
        metal,
    )
    define_cylinder(
        stage,
        f"{root_path}/lower_body",
        (0.0, 0.0, 0.020),
        0.033,
        0.038,
        (0.82, 0.80, 0.74),
        False,
        shell,
    )
    define_sphere(
        stage,
        f"{root_path}/upper_dome",
        (0.0, 0.0, 0.056),
        0.033,
        (0.78, 0.77, 0.72),
        False,
        shell,
        scale=(1.0, 1.0, 0.38),
    )
    define_box(
        stage,
        f"{root_path}/front_window",
        (0.034, 0.0, 0.043),
        (0.004, 0.044, 0.016),
        (0.004, 0.010, 0.014),
        False,
        glass,
    )
    define_sphere(
        stage,
        f"{root_path}/status_led",
        (0.036, -0.022, 0.025),
        0.004,
        (0.05, 0.42, 0.86),
        False,
        accent,
        scale=(0.35, 1.0, 1.0),
    )


def add_mid360_trim_visual(stage, materials, root_path):
    stage.DefinePrim(root_path, "Xform")
    define_box(
        stage,
        f"{root_path}/front_window",
        (0.036, 0.0, 0.043),
        (0.004, 0.046, 0.015),
        (0.004, 0.008, 0.012),
        False,
        materials["glass"],
    )
    define_sphere(
        stage,
        f"{root_path}/status_led",
        (0.038, -0.024, 0.026),
        0.004,
        (0.05, 0.42, 0.86),
        False,
        materials["accent"],
        scale=(0.35, 1.0, 1.0),
    )


def add_mid360_visual(stage, materials):
    visual_root = f"{LIVOX_BASE}/mid360_visual"
    define_xform(stage, visual_root, rotate=(0.0, 0.0, float(args.livox_visual_yaw_deg)))
    if args.sensor_visual_detail == "cad" and DEFAULT_LIVOX_VISUAL.exists():
        livox_usd = convert_asset_to_usd(DEFAULT_LIVOX_VISUAL)
        if livox_usd:
            cad = add_reference(stage, f"{visual_root}/cad_body", livox_usd)
            mesh_count = bind_material_under(stage, str(cad.GetPath()), materials["shell"])
            add_mid360_trim_visual(stage, materials, f"{visual_root}/cad_trim")
            log(f"loaded Livox MID360 CAD visual ({mesh_count} mesh prims)")
            return
        log("warning: Livox MID360 CAD conversion failed; using procedural visual")
    add_mid360_procedural_visual(stage, materials, f"{visual_root}/procedural_body")
    log("created procedural Livox MID360 visual")


def add_sensor_stack_mount_visual(stage, materials):
    root_path = f"{BASE_LINK}/sensor_stack_mount"
    stage.DefinePrim(root_path, "Xform")
    metal = materials["metal"]
    dark = materials["dark"]
    define_box(
        stage,
        f"{root_path}/lower_shelf",
        (0.205, 0.0, 0.184),
        (0.148, 0.162, 0.010),
        (0.50, 0.51, 0.49),
        False,
        metal,
    )
    define_box(
        stage,
        f"{root_path}/rear_mast",
        (0.150, 0.0, 0.222),
        (0.018, 0.042, 0.086),
        (0.038, 0.040, 0.040),
        False,
        dark,
    )
    define_box(
        stage,
        f"{root_path}/top_plate",
        (0.180, 0.0, 0.236),
        (0.096, 0.086, 0.010),
        (0.56, 0.57, 0.55),
        False,
        metal,
    )
    define_box(
        stage,
        f"{root_path}/camera_back_plate",
        (0.190, 0.0, 0.205),
        (0.010, 0.142, 0.042),
        (0.040, 0.042, 0.042),
        False,
        dark,
    )


def add_imu_visual(stage, materials):
    root_path = f"{IMU_LINK}/imu_visual"
    stage.DefinePrim(root_path, "Xform")
    define_box(
        stage,
        f"{root_path}/board",
        (0.0, 0.0, 0.0),
        (0.036, 0.030, 0.006),
        (0.03, 0.26, 0.12),
        False,
        materials["board"],
    )
    define_box(
        stage,
        f"{root_path}/chip",
        (0.002, 0.0, 0.0065),
        (0.017, 0.016, 0.006),
        (0.010, 0.012, 0.012),
        False,
        materials["chip"],
    )
    define_box(
        stage,
        f"{root_path}/connector",
        (-0.013, 0.0, 0.006),
        (0.006, 0.022, 0.006),
        (0.56, 0.57, 0.55),
        False,
        materials["metal"],
    )


def add_d455_visual(stage, materials):
    root_path = f"{CAMERA_LINK}/d455_visual"
    stage.DefinePrim(root_path, "Xform")
    body = materials["realsense"]
    face = materials["realsense_face"]
    glass = materials["glass"]
    metal = materials["metal"]

    define_box(
        stage,
        f"{root_path}/body",
        (0.0, 0.0, 0.0),
        (0.030, 0.126, 0.030),
        (0.055, 0.060, 0.064),
        False,
        body,
    )
    define_box(
        stage,
        f"{root_path}/front_face",
        (0.017, 0.0, 0.002),
        (0.004, 0.116, 0.022),
        (0.010, 0.012, 0.014),
        False,
        face,
    )
    for idx, y in enumerate((-0.040, 0.040)):
        define_cylinder(
            stage,
            f"{root_path}/stereo_lens_{idx}",
            (0.020, y, 0.004),
            0.010,
            0.005,
            (0.004, 0.006, 0.008),
            False,
            glass,
            rotate=(0.0, 90.0, 0.0),
        )
        define_cylinder(
            stage,
            f"{root_path}/lens_ring_{idx}",
            (0.019, y, 0.004),
            0.012,
            0.002,
            (0.56, 0.57, 0.55),
            False,
            metal,
            rotate=(0.0, 90.0, 0.0),
        )
    define_cylinder(
        stage,
        f"{root_path}/rgb_lens",
        (0.020, 0.000, 0.004),
        0.007,
        0.004,
        (0.006, 0.010, 0.014),
        False,
        glass,
        rotate=(0.0, 90.0, 0.0),
    )
    define_box(
        stage,
        f"{root_path}/ir_projector",
        (0.020, 0.021, 0.004),
        (0.004, 0.012, 0.008),
        (0.018, 0.018, 0.016),
        False,
        face,
    )
    define_box(
        stage,
        f"{root_path}/mount_foot",
        (-0.003, 0.0, -0.021),
        (0.020, 0.072, 0.010),
        (0.56, 0.57, 0.55),
        False,
        metal,
    )


def create_sensor_frames(stage):
    log("creating sensor frames on imported Hunter base_link")
    define_xform(stage, LIVOX_BASE, LIVOX_BASE_TRANSLATE, (0.0, math.degrees(LIVOX_MOUNT_PITCH_RAD), 0.0))
    define_xform(stage, f"{LIVOX_BASE}/{LIDAR_FRAME}", LIDAR_FRAME_TRANSLATE, (0.0, 0.0, 0.0))
    define_xform(stage, IMU_LINK, IMU_LINK_TRANSLATE, (0.0, 0.0, 0.0))
    define_xform(stage, CAMERA_LINK, CAMERA_LINK_TRANSLATE, (0.0, 0.0, 0.0))

    if args.no_sensor_meshes or args.sensor_visual_detail == "none":
        return

    sensor_materials = create_sensor_materials(stage)
    add_sensor_stack_mount_visual(stage, sensor_materials)
    add_mid360_visual(stage, sensor_materials)
    add_imu_visual(stage, sensor_materials)
    add_d455_visual(stage, sensor_materials)


def create_lidar():
    log(
        f"creating RTX LiDAR config={args.lidar_config} "
        f"(quality={args.lidar_quality_label}, full_scan={args.lidar_full_scan}, frame_skip={args.lidar_frame_skip})"
    )
    _result, sensor = omni.kit.commands.execute(
        "IsaacSensorCreateRtxLidar",
        path="/LivoxMid360",
        parent=f"{LIVOX_BASE}/{LIDAR_FRAME}",
        config=args.lidar_config,
        translation=(0.0, 0.0, 0.0),
        orientation=Gf.Quatd(1.0, 0.0, 0.0, 0.0),
        force_camera_prim=True,
    )
    if sensor is None or not sensor.IsValid():
        raise RuntimeError("Failed to create local RTX LiDAR prim")
    render_product = rep.create.render_product(
        sensor.GetPath(),
        resolution=(1, 1),
        name="LivoxRtxRenderProduct",
        render_vars=["GenericModelOutput", "RtxSensorMetadata"],
    )
    render_product_path = render_product.path

    keys = og.Controller.Keys
    og.Controller.edit(
        {"graph_path": "/ROS_Lidar", "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("PointCloudPublish", "isaacsim.ros2.bridge.ROS2RtxLidarHelper"),
            ],
            keys.SET_VALUES: [
                ("PointCloudPublish.inputs:renderProductPath", render_product_path),
                ("PointCloudPublish.inputs:topicName", "/livox/lidar"),
                ("PointCloudPublish.inputs:frameId", LIDAR_FRAME),
                ("PointCloudPublish.inputs:type", "point_cloud"),
                ("PointCloudPublish.inputs:frameSkipCount", max(0, args.lidar_frame_skip)),
                ("PointCloudPublish.inputs:fullScan", args.lidar_full_scan),
                ("PointCloudPublish.inputs:showDebugView", args.show_lidar_debug),
                ("PointCloudPublish.inputs:resetSimulationTimeOnStop", True),
                ("PointCloudPublish.inputs:useSystemTime", False),
            ],
            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "PointCloudPublish.inputs:execIn"),
            ],
        },
    )
    log(f"LiDAR render product: {render_product_path}")
    return render_product_path


def create_camera(stage):
    log("creating D455 camera")
    camera = UsdGeom.Camera.Define(stage, f"{CAMERA_LINK}/D455ColorCamera")
    set_look_at(
        camera.GetPrim(),
        (CAMERA_OPTICAL_OFFSET_X, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    camera.GetProjectionAttr().Set("perspective")
    camera.GetHorizontalApertureAttr().Set(20.955)
    camera.GetVerticalApertureAttr().Set(11.787)
    camera.GetFocalLengthAttr().Set(11.04)
    camera.GetFocusDistanceAttr().Set(5.0)
    camera.GetFStopAttr().Set(0.0)
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 50.0))

    render_product = rep.create.render_product(
        camera.GetPath(),
        (args.camera_width, args.camera_height),
        name="D455ColorRenderProduct",
    )
    render_product_path = render_product.path

    keys = og.Controller.Keys
    og.Controller.edit(
        {"graph_path": "/ROS_Camera", "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("RgbPublish", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ("InfoPublish", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
                ("DepthPublish", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ("DepthPclPublish", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ],
            keys.SET_VALUES: [
                ("RgbPublish.inputs:renderProductPath", render_product_path),
                ("RgbPublish.inputs:topicName", "/camera/camera/color/image_raw"),
                ("RgbPublish.inputs:frameId", CAMERA_FRAME),
                ("RgbPublish.inputs:type", "rgb"),
                ("RgbPublish.inputs:frameSkipCount", args.camera_frame_skip),
                ("RgbPublish.inputs:resetSimulationTimeOnStop", True),
                ("InfoPublish.inputs:renderProductPath", render_product_path),
                ("InfoPublish.inputs:topicName", "/camera/camera/color/camera_info"),
                ("InfoPublish.inputs:frameId", CAMERA_FRAME),
                ("InfoPublish.inputs:frameSkipCount", args.camera_frame_skip),
                ("InfoPublish.inputs:resetSimulationTimeOnStop", True),
                ("DepthPublish.inputs:renderProductPath", render_product_path),
                ("DepthPublish.inputs:topicName", "/camera/camera/depth/image_rect_raw"),
                ("DepthPublish.inputs:frameId", CAMERA_FRAME),
                ("DepthPublish.inputs:type", "depth"),
                ("DepthPublish.inputs:frameSkipCount", args.depth_frame_skip),
                ("DepthPublish.inputs:resetSimulationTimeOnStop", True),
                ("DepthPclPublish.inputs:renderProductPath", render_product_path),
                ("DepthPclPublish.inputs:topicName", "/camera/camera/depth/color/points"),
                ("DepthPclPublish.inputs:frameId", CAMERA_FRAME),
                ("DepthPclPublish.inputs:type", "depth_pcl"),
                ("DepthPclPublish.inputs:frameSkipCount", args.depth_frame_skip),
                ("DepthPclPublish.inputs:resetSimulationTimeOnStop", True),
            ],
            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "RgbPublish.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "InfoPublish.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "DepthPublish.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "DepthPclPublish.inputs:execIn"),
            ],
        },
    )
    log(f"D455 render product: {render_product_path}")
    return render_product_path


def create_clock_and_imu_graph():
    log("creating /clock and IMU graph")
    keys = og.Controller.Keys
    imu_topics = [("ImuPublish", args.imu_topic)]
    legacy_topic = args.legacy_imu_topic.strip()
    if legacy_topic and legacy_topic.lower() != "none" and legacy_topic != args.imu_topic:
        imu_topics.append(("LegacyImuPublish", legacy_topic))
    global IMU_PUBLISH_NODE_NAMES
    IMU_PUBLISH_NODE_NAMES = [node_name for node_name, _topic in imu_topics]

    create_nodes = [
        ("OnPhysicsStep", "isaacsim.core.nodes.OnPhysicsStep"),
        ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
        ("ClockPublish", "isaacsim.ros2.bridge.ROS2PublishClock"),
    ]
    set_values = [("ClockPublish.inputs:topicName", "/clock")]
    connections = [
        ("OnPhysicsStep.outputs:step", "ClockPublish.inputs:execIn"),
        ("ReadSimTime.outputs:simulationTime", "ClockPublish.inputs:timeStamp"),
    ]
    for node_name, topic in imu_topics:
        create_nodes.append((node_name, "isaacsim.ros2.bridge.ROS2PublishImu"))
        set_values.extend(
            [
                (f"{node_name}.inputs:topicName", topic),
                (f"{node_name}.inputs:frameId", IMU_FRAME),
                (f"{node_name}.inputs:publishOrientation", True),
                (f"{node_name}.inputs:publishAngularVelocity", True),
                (f"{node_name}.inputs:publishLinearAcceleration", True),
            ]
        )
        connections.extend(
            [
                ("OnPhysicsStep.outputs:step", f"{node_name}.inputs:execIn"),
                ("ReadSimTime.outputs:simulationTime", f"{node_name}.inputs:timeStamp"),
            ]
        )

    og.Controller.edit(
        {
            "graph_path": "/ROS_ClockImu",
            "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND,
        },
        {
            keys.CREATE_NODES: create_nodes,
            keys.SET_VALUES: set_values,
            keys.CONNECT: connections,
        },
    )
    log("external IMU topics: " + ", ".join(topic for _node_name, topic in imu_topics))


def create_cmd_vel_graph():
    log(f"creating ROS2 /cmd_vel subscriber graph on topic {args.cmd_vel_topic}")
    keys = og.Controller.Keys
    og.Controller.edit(
        {"graph_path": "/ROS_CmdVel", "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("SubscribeTwist", "isaacsim.ros2.bridge.ROS2SubscribeTwist"),
            ],
            keys.SET_VALUES: [
                ("SubscribeTwist.inputs:topicName", args.cmd_vel_topic),
                ("SubscribeTwist.inputs:queueSize", 1),
            ],
            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "SubscribeTwist.inputs:execIn"),
            ],
        },
    )


CMD_VEL_STATE = {
    "x": 0.0,
    "y": 0.0,
    "yaw": 0.0,
    "vx": 0.0,
    "vy": 0.0,
    "speed": 0.0,
    "steer": 0.0,
    "yaw_rate": 0.0,
    "cmd_linear_x": 0.0,
    "cmd_angular_z": 0.0,
    "last_cmd_wall_time": 0.0,
    "seen_cmd": False,
    "logged_nonzero_cmd": False,
    "last_publisher_report_wall_time": 0.0,
    "last_publisher_count": -1,
}
_CMD_VEL_WARNED = False
_CMD_VEL_DIRECT_WARNED = False
CMD_VEL_RCLPY = None
CMD_VEL_NODE = None
CMD_VEL_SUB = None
ODOM_PUBLISHER = None
ODOM_MSG_CLS = None
IMU_PUBLISH_NODE_NAMES = ["ImuPublish"]
PHYSICS_IMU_STATE = {
    "prev_position": None,
    "prev_linear_velocity": None,
    "last_drive_log_time": -1e9,
    "last_odom_publish_time": -1e9,
    "last_wheel_targets": {},
    "last_drive_feedback": (0.0, 0.0),
    "last_root_assist": (0.0, 0.0, 0.0),
    "idle_hold_pose": None,
    "warned_root_assist": False,
    "warned_steering_zero_hold": False,
    "warned_idle_pose_hold": False,
    "warned_sink_guard": False,
    "warned_imu": False,
    "warned_odom": False,
}


def read_vec3_attribute(path):
    value = og.Controller.attribute(path).get()
    return (float(value[0]), float(value[1]), float(value[2]))


def clamp_signed(value, limit):
    limit = abs(float(limit))
    return max(-limit, min(limit, float(value)))


def approach(current, target, max_delta):
    max_delta = abs(float(max_delta))
    if target > current + max_delta:
        return current + max_delta
    if target < current - max_delta:
        return current - max_delta
    return target


def finite_or_zero(value):
    value = float(value)
    return value if math.isfinite(value) else 0.0


def exp_damp(value, rate, dt):
    rate = max(0.0, float(rate))
    if rate <= 0.0:
        return finite_or_zero(value)
    return finite_or_zero(value) * math.exp(-rate * max(0.0, float(dt)))


def publish_imu_values(quat_ijkr, lin_acc, ang_vel):
    if args.visual_only:
        return
    for node_name in IMU_PUBLISH_NODE_NAMES:
        og.Controller.attribute(f"/ROS_ClockImu/{node_name}.inputs:orientation").set(tuple(float(v) for v in quat_ijkr))
        og.Controller.attribute(f"/ROS_ClockImu/{node_name}.inputs:linearAcceleration").set(tuple(float(v) for v in lin_acc))
        og.Controller.attribute(f"/ROS_ClockImu/{node_name}.inputs:angularVelocity").set(tuple(float(v) for v in ang_vel))


def normalize_quat_wxyz(quat):
    q = np.asarray(quat, dtype=float).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / norm


def quat_rotate_wxyz(quat, vector):
    q = normalize_quat_wxyz(quat)
    v = np.asarray(vector, dtype=float).reshape(3)
    q_vec = q[1:4]
    return v + 2.0 * np.cross(q_vec, np.cross(q_vec, v) + q[0] * v)


def quat_rotate_inverse_wxyz(quat, vector):
    q = normalize_quat_wxyz(quat)
    return quat_rotate_wxyz(np.array([q[0], -q[1], -q[2], -q[3]], dtype=float), vector)


def create_cmd_vel_subscriber():
    global CMD_VEL_RCLPY, CMD_VEL_NODE, CMD_VEL_SUB, ODOM_PUBLISHER, ODOM_MSG_CLS, _CMD_VEL_DIRECT_WARNED
    try:
        import rclpy
        from geometry_msgs.msg import Twist
        from nav_msgs.msg import Odometry

        if not rclpy.ok():
            rclpy.init(args=None)
        node_name = "hunter_isaac_cmd_vel_subscriber" if args.cmd_vel_backend == "rclpy" else "hunter_isaac_odometry_publisher"
        node = rclpy.create_node(node_name)

        CMD_VEL_RCLPY = rclpy
        CMD_VEL_NODE = node
        CMD_VEL_SUB = None
        ODOM_MSG_CLS = Odometry
        ODOM_PUBLISHER = node.create_publisher(Odometry, args.odom_topic, 10)
        log(f"publishing measured Hunter odometry on {args.odom_topic} at up to {args.odom_publish_hz:.1f} Hz")
        if args.cmd_vel_backend == "bridge":
            log(f"using Isaac ROS2 Bridge subscriber for {args.cmd_vel_topic}; direct rclpy is used only for odometry")
            return False

        def _cmd_vel_cb(msg):
            CMD_VEL_STATE["cmd_linear_x"] = finite_or_zero(msg.linear.x)
            CMD_VEL_STATE["cmd_angular_z"] = finite_or_zero(msg.angular.z)
            CMD_VEL_STATE["last_cmd_wall_time"] = time.monotonic()
            CMD_VEL_STATE["seen_cmd"] = True

        CMD_VEL_SUB = node.create_subscription(Twist, args.cmd_vel_topic, _cmd_vel_cb, 10)
        log(
            f"creating direct rclpy /cmd_vel subscriber on {args.cmd_vel_topic} "
            f"(timeout={args.cmd_vel_timeout:.2f}s)"
        )
        return True
    except Exception as exc:
        if not _CMD_VEL_DIRECT_WARNED:
            log(f"warning: direct rclpy support unavailable, falling back to OmniGraph cmd_vel subscriber: {exc}")
            _CMD_VEL_DIRECT_WARNED = True
        CMD_VEL_RCLPY = None
        CMD_VEL_NODE = None
        CMD_VEL_SUB = None
        ODOM_PUBLISHER = None
        ODOM_MSG_CLS = None
        return False


def shutdown_cmd_vel_subscriber():
    global CMD_VEL_NODE, CMD_VEL_SUB, CMD_VEL_RCLPY, ODOM_PUBLISHER, ODOM_MSG_CLS
    if CMD_VEL_NODE is not None:
        try:
            CMD_VEL_NODE.destroy_node()
        except Exception:
            pass
    if CMD_VEL_RCLPY is not None:
        try:
            if CMD_VEL_RCLPY.ok():
                CMD_VEL_RCLPY.shutdown()
        except Exception:
            pass
    CMD_VEL_NODE = None
    CMD_VEL_SUB = None
    CMD_VEL_RCLPY = None
    ODOM_PUBLISHER = None
    ODOM_MSG_CLS = None


def spin_cmd_vel_subscriber():
    if CMD_VEL_RCLPY is None or CMD_VEL_NODE is None or CMD_VEL_SUB is None:
        return
    try:
        CMD_VEL_RCLPY.spin_once(CMD_VEL_NODE, timeout_sec=0.0)
        monitor_cmd_vel_publishers()
    except Exception as exc:
        global _CMD_VEL_DIRECT_WARNED
        if not _CMD_VEL_DIRECT_WARNED:
            log(f"warning: failed while spinning direct cmd_vel subscriber: {exc}")
            _CMD_VEL_DIRECT_WARNED = True


def monitor_cmd_vel_publishers():
    if CMD_VEL_NODE is None:
        return
    now = time.monotonic()
    if now - float(CMD_VEL_STATE.get("last_publisher_report_wall_time", 0.0)) < 2.0:
        return
    CMD_VEL_STATE["last_publisher_report_wall_time"] = now
    try:
        infos = CMD_VEL_NODE.get_publishers_info_by_topic(args.cmd_vel_topic)
    except Exception:
        return

    count = len(infos)
    last_count = int(CMD_VEL_STATE.get("last_publisher_count", -1))
    if count == last_count and count <= 1:
        return
    CMD_VEL_STATE["last_publisher_count"] = count

    names = []
    for info in infos[:4]:
        node_name = getattr(info, "node_name", "") or "<unknown>"
        namespace = getattr(info, "node_namespace", "") or ""
        names.append(f"{namespace.rstrip('/')}/{node_name}" if namespace and namespace != "/" else node_name)
    suffix = f": {', '.join(names)}" if names else ""
    if count > 1:
        log(
            f"warning: {count} publishers are driving {args.cmd_vel_topic}{suffix}. "
            "Unexpected steering while using keyboard control usually means another controller is still publishing."
        )
    else:
        log(f"{args.cmd_vel_topic} publisher count: {count}{suffix}")


def read_cmd_vel_command():
    global _CMD_VEL_WARNED
    if CMD_VEL_NODE is not None and CMD_VEL_SUB is not None:
        spin_cmd_vel_subscriber()
        last_time = float(CMD_VEL_STATE["last_cmd_wall_time"])
        if not CMD_VEL_STATE["seen_cmd"] or time.monotonic() - last_time > max(0.02, args.cmd_vel_timeout):
            return 0.0, 0.0, False
        return float(CMD_VEL_STATE["cmd_linear_x"]), float(CMD_VEL_STATE["cmd_angular_z"]), True

    try:
        linear = read_vec3_attribute("/ROS_CmdVel/SubscribeTwist.outputs:linearVelocity")
        angular = read_vec3_attribute("/ROS_CmdVel/SubscribeTwist.outputs:angularVelocity")
    except Exception as exc:
        if not _CMD_VEL_WARNED:
            log(f"warning: failed to read cmd_vel subscriber outputs, keeping Hunter stopped: {exc}")
            _CMD_VEL_WARNED = True
        linear = (0.0, 0.0, 0.0)
        angular = (0.0, 0.0, 0.0)
    return linear[0], angular[2], True


def yaw_rate_from_speed_steer(speed, steer):
    if abs(speed) < abs(args.cmd_vel_min_turn_speed) or abs(steer) < 1e-5:
        return 0.0
    return float(speed) * math.tan(float(steer)) / max(WHEEL_BASE, 1e-6)


def steer_from_speed_yaw_rate(speed, yaw_rate):
    if abs(speed) < abs(args.cmd_vel_min_turn_speed) or abs(yaw_rate) < 1e-5:
        return 0.0
    return clamp_signed(math.atan((WHEEL_BASE * float(yaw_rate)) / float(speed)), args.cmd_vel_max_steer)


def cmd_vel_drive_command(dt):
    linear_x, angular_z, fresh = read_cmd_vel_command()
    dt = max(1e-4, float(dt))
    if fresh and not CMD_VEL_STATE["logged_nonzero_cmd"] and (abs(linear_x) > 1e-4 or abs(angular_z) > 1e-4):
        log(f"received nonzero cmd_vel: linear.x={linear_x:+.3f} angular.z={angular_z:+.3f}")
        CMD_VEL_STATE["logged_nonzero_cmd"] = True

    target_speed = clamp_signed(linear_x, args.cmd_vel_max_speed)
    if abs(target_speed) < max(0.0, float(args.cmd_vel_linear_deadband)):
        target_speed = 0.0
    speed = approach(CMD_VEL_STATE["speed"], target_speed, abs(args.cmd_vel_max_accel) * dt)

    if args.cmd_vel_angular_mode == "yaw_rate":
        target_yaw_rate = clamp_signed(angular_z, args.cmd_vel_max_yaw_rate)
        if abs(target_yaw_rate) < max(0.0, float(args.cmd_vel_angular_deadband)):
            target_yaw_rate = 0.0
        if abs(speed) < abs(args.cmd_vel_min_turn_speed):
            target_yaw_rate = 0.0
        else:
            ackermann_yaw_limit = abs(speed) * math.tan(abs(args.cmd_vel_max_steer)) / max(WHEEL_BASE, 1e-3)
            target_yaw_rate = clamp_signed(target_yaw_rate, min(abs(args.cmd_vel_max_yaw_rate), ackermann_yaw_limit))
        yaw_rate_cmd = approach(CMD_VEL_STATE["yaw_rate"], target_yaw_rate, abs(args.cmd_vel_max_yaw_accel) * dt)
        target_steer = steer_from_speed_yaw_rate(speed, yaw_rate_cmd)
    else:
        target_steer = clamp_signed(angular_z, args.cmd_vel_max_steer)
        if abs(target_steer) < max(float(args.steering_zero_snap), float(args.cmd_vel_angular_deadband)):
            target_steer = 0.0

    steer = approach(CMD_VEL_STATE["steer"], target_steer, abs(args.cmd_vel_max_steer_rate) * dt)
    yaw_rate = yaw_rate_from_speed_steer(speed, steer)
    CMD_VEL_STATE["speed"] = speed
    CMD_VEL_STATE["steer"] = steer
    CMD_VEL_STATE["yaw_rate"] = yaw_rate
    return speed, steer, yaw_rate


def circle_drive_command(t):
    if args.drive_mode == "stationary" or t < args.stationary_duration:
        return 0.0, 0.0, 0.0
    speed = 0.22
    yaw_rate = 0.10
    steer = steer_from_speed_yaw_rate(speed, yaw_rate)
    return speed, steer, yaw_rate_from_speed_steer(speed, steer)


def drive_command(t, dt):
    if args.drive_mode == "cmd_vel":
        return cmd_vel_drive_command(dt)
    return circle_drive_command(t)


def trajectory(t):
    if args.drive_mode == "stationary" or t < args.stationary_duration:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    s = t - args.stationary_duration
    radius = 2.2
    omega = 0.10
    x = radius * math.sin(omega * s)
    y = radius * (1.0 - math.cos(omega * s))
    yaw = omega * s
    ax = -radius * omega * omega * math.sin(omega * s)
    ay = radius * omega * omega * math.cos(omega * s)
    speed = radius * omega
    return x, y, yaw, ax, ay, omega, speed


def cmd_vel_trajectory(dt):
    dt = max(1e-4, float(dt))
    speed, _steer, yaw_rate = cmd_vel_drive_command(dt)
    yaw_mid = CMD_VEL_STATE["yaw"] + 0.5 * yaw_rate * dt
    vx = speed * math.cos(yaw_mid)
    vy = speed * math.sin(yaw_mid)
    ax_w = (vx - CMD_VEL_STATE["vx"]) / dt
    ay_w = (vy - CMD_VEL_STATE["vy"]) / dt
    CMD_VEL_STATE["x"] += vx * dt
    CMD_VEL_STATE["y"] += vy * dt
    CMD_VEL_STATE["yaw"] += yaw_rate * dt
    CMD_VEL_STATE["vx"] = vx
    CMD_VEL_STATE["vy"] = vy
    return CMD_VEL_STATE["x"], CMD_VEL_STATE["y"], CMD_VEL_STATE["yaw"], ax_w, ay_w, yaw_rate, speed


def ackermann_wheel_targets(speed, steer):
    speed = finite_or_zero(speed)
    steer = clamp_signed(steer, args.cmd_vel_max_steer)
    half_track = 0.5 * TRACK_WIDTH
    if abs(steer) < 1e-5:
        left_steer = 0.0
        right_steer = 0.0
    else:
        radius = abs(WHEEL_BASE / max(1e-6, abs(math.tan(steer))))
        inner = math.atan2(WHEEL_BASE, max(0.05, radius - half_track))
        outer = math.atan2(WHEEL_BASE, radius + half_track)
        if steer > 0.0:
            left_steer = inner
            right_steer = outer
        else:
            left_steer = -outer
            right_steer = -inner
    left_steer = clamp_signed(left_steer, args.cmd_vel_max_steer)
    right_steer = clamp_signed(right_steer, args.cmd_vel_max_steer)

    yaw_rate = yaw_rate_from_speed_steer(speed, steer)
    left_speed = speed - yaw_rate * half_track
    right_speed = speed + yaw_rate * half_track
    wheel_velocities = {
        "fr_left_joint": WHEEL_JOINT_SIGNS["fr_left_joint"] * left_speed / WHEEL_RADIUS,
        "fr_right_joint": WHEEL_JOINT_SIGNS["fr_right_joint"] * right_speed / WHEEL_RADIUS,
        "re_left_joint": WHEEL_JOINT_SIGNS["re_left_joint"] * left_speed / WHEEL_RADIUS,
        "re_right_joint": WHEEL_JOINT_SIGNS["re_right_joint"] * right_speed / WHEEL_RADIUS,
    }
    steering_positions = {
        "fr_steer_left_joint": left_steer,
        "fr_steer_right_joint": right_steer,
        "steering_whee_joint": 4.0 * 0.5 * (left_steer + right_steer),
    }
    return steering_positions, wheel_velocities


def wheel_targets_from_speed_yaw_rate(speed, yaw_rate):
    half_track = 0.5 * TRACK_WIDTH
    left_speed = speed - yaw_rate * half_track
    right_speed = speed + yaw_rate * half_track
    return {
        "fr_left_joint": WHEEL_JOINT_SIGNS["fr_left_joint"] * left_speed / WHEEL_RADIUS,
        "fr_right_joint": WHEEL_JOINT_SIGNS["fr_right_joint"] * right_speed / WHEEL_RADIUS,
        "re_left_joint": WHEEL_JOINT_SIGNS["re_left_joint"] * left_speed / WHEEL_RADIUS,
        "re_right_joint": WHEEL_JOINT_SIGNS["re_right_joint"] * right_speed / WHEEL_RADIUS,
    }


def physics_body_speed_yaw_rate(articulation):
    try:
        _position, quat_wxyz = articulation.get_world_pose()
        quat_wxyz = normalize_quat_wxyz(quat_wxyz)
        linear_velocity = articulation.get_linear_velocity()
        angular_velocity = articulation.get_angular_velocity()
        if linear_velocity is None:
            linear_velocity = np.zeros(3, dtype=float)
        if angular_velocity is None:
            angular_velocity = np.zeros(3, dtype=float)
        body_linear = quat_rotate_inverse_wxyz(quat_wxyz, np.asarray(linear_velocity, dtype=float).reshape(3))
        body_angular = quat_rotate_inverse_wxyz(quat_wxyz, np.asarray(angular_velocity, dtype=float).reshape(3))
        return finite_or_zero(body_linear[0]), finite_or_zero(body_angular[2])
    except Exception:
        return 0.0, 0.0


def apply_hunter_root_velocity_assist(articulation, speed, steer, dt):
    if articulation is None or not args.physics_root_assist:
        PHYSICS_IMU_STATE["last_root_assist"] = (0.0, 0.0, 0.0)
        return

    try:
        _position, quat_wxyz = articulation.get_world_pose()
        quat_wxyz = normalize_quat_wxyz(quat_wxyz)
        linear_velocity = articulation.get_linear_velocity()
        angular_velocity = articulation.get_angular_velocity()
        if linear_velocity is None:
            linear_velocity = np.zeros(3, dtype=float)
        if angular_velocity is None:
            angular_velocity = np.zeros(3, dtype=float)

        dt = max(1e-4, float(dt))
        target_speed = finite_or_zero(speed)
        target_steer = clamp_signed(steer, args.cmd_vel_max_steer)
        target_yaw_rate = yaw_rate_from_speed_steer(target_speed, target_steer)
        body_linear = quat_rotate_inverse_wxyz(quat_wxyz, np.asarray(linear_velocity, dtype=float).reshape(3))
        body_angular = quat_rotate_inverse_wxyz(quat_wxyz, np.asarray(angular_velocity, dtype=float).reshape(3))

        has_speed_cmd = abs(target_speed) >= 0.015
        has_steer_cmd = abs(target_steer) >= 0.015 and abs(target_yaw_rate) >= 0.005
        straight_yaw_hold = args.physics_straight_yaw_hold and abs(target_steer) < max(0.0, float(args.steering_zero_snap))
        mode = args.physics_root_assist_mode
        if mode == "strict":
            body_linear[0] = target_speed
            body_linear[1] = 0.0
            body_angular[0] = exp_damp(body_angular[0], args.physics_root_roll_pitch_damping, dt)
            body_angular[1] = exp_damp(body_angular[1], args.physics_root_roll_pitch_damping, dt)
            body_angular[2] = target_yaw_rate
        elif mode == "damped":
            speed_accel = args.physics_root_assist_accel if has_speed_cmd else args.physics_root_idle_brake_accel
            yaw_accel = args.physics_root_assist_yaw_accel if has_steer_cmd else args.physics_root_idle_yaw_brake

            body_linear[0] = approach(body_linear[0], target_speed, max(0.0, speed_accel) * dt)
            body_linear[1] = exp_damp(body_linear[1], args.physics_root_lateral_damping, dt)
            body_angular[0] = exp_damp(body_angular[0], args.physics_root_roll_pitch_damping, dt)
            body_angular[1] = exp_damp(body_angular[1], args.physics_root_roll_pitch_damping, dt)
            body_angular[2] = approach(body_angular[2], target_yaw_rate, max(0.0, yaw_accel) * dt)
        else:
            body_linear[1] = exp_damp(body_linear[1], args.physics_root_lateral_damping, dt)
            body_angular[0] = exp_damp(body_angular[0], args.physics_root_roll_pitch_damping, dt)
            body_angular[1] = exp_damp(body_angular[1], args.physics_root_roll_pitch_damping, dt)
            if not has_speed_cmd:
                body_linear[0] = approach(body_linear[0], 0.0, max(0.0, args.physics_root_idle_brake_accel) * dt)
            elif target_speed * body_linear[0] < -1e-4:
                body_linear[0] = approach(body_linear[0], 0.0, max(0.0, args.physics_root_idle_brake_accel) * dt)
            elif target_speed > 0.0 and body_linear[0] > target_speed:
                body_linear[0] = approach(body_linear[0], target_speed, max(0.0, args.physics_root_idle_brake_accel) * dt)
            elif target_speed < 0.0 and body_linear[0] < target_speed:
                body_linear[0] = approach(body_linear[0], target_speed, max(0.0, args.physics_root_idle_brake_accel) * dt)

            yaw_accel = args.physics_root_assist_yaw_accel if has_steer_cmd else args.physics_root_idle_yaw_brake
            body_angular[2] = approach(body_angular[2], target_yaw_rate, max(0.0, yaw_accel) * dt)

        if straight_yaw_hold:
            body_angular[2] = 0.0

        speed_limit = max(abs(args.cmd_vel_max_speed), abs(target_speed), 0.1) * 1.15
        yaw_limit = max(abs(args.cmd_vel_max_yaw_rate), abs(target_yaw_rate), 0.1) * 1.15
        body_linear[0] = clamp_signed(body_linear[0], speed_limit)
        body_linear[1] = clamp_signed(body_linear[1], speed_limit)
        body_angular[2] = clamp_signed(body_angular[2], yaw_limit)
        if not has_speed_cmd and abs(body_linear[0]) < max(0.0, float(args.physics_root_idle_stop_threshold)):
            body_linear[0] = 0.0
        if abs(body_linear[1]) < 0.008:
            body_linear[1] = 0.0
        if not has_steer_cmd and abs(body_angular[2]) < 0.008:
            body_angular[2] = 0.0

        articulation.set_linear_velocity(np.asarray(quat_rotate_wxyz(quat_wxyz, body_linear), dtype=float))
        articulation.set_angular_velocity(np.asarray(quat_rotate_wxyz(quat_wxyz, body_angular), dtype=float))
        PHYSICS_IMU_STATE["last_root_assist"] = (
            finite_or_zero(body_linear[0]),
            finite_or_zero(body_linear[1]),
            finite_or_zero(body_angular[2]),
        )
        PHYSICS_IMU_STATE["warned_root_assist"] = False
    except Exception as exc:
        if not PHYSICS_IMU_STATE["warned_root_assist"]:
            log(f"warning: failed to apply Hunter root velocity assist: {exc}")
            PHYSICS_IMU_STATE["warned_root_assist"] = True


def apply_hunter_idle_pose_hold(articulation, speed, steer, t):
    if articulation is None or not args.physics_idle_pose_hold:
        PHYSICS_IMU_STATE["idle_hold_pose"] = None
        return
    if t < max(0.0, float(args.physics_settle_duration)):
        PHYSICS_IMU_STATE["idle_hold_pose"] = None
        return

    idle_speed_limit = max(0.0, float(args.physics_idle_pose_hold_speed))
    idle_steer_limit = max(0.0, float(args.steering_zero_snap))
    if abs(finite_or_zero(speed)) > idle_speed_limit or abs(finite_or_zero(steer)) > idle_steer_limit:
        PHYSICS_IMU_STATE["idle_hold_pose"] = None
        return

    try:
        position, quat_wxyz = articulation.get_world_pose()
        position = np.asarray(position, dtype=float).reshape(3)
        quat_wxyz = normalize_quat_wxyz(quat_wxyz)
        linear_velocity = articulation.get_linear_velocity()
        angular_velocity = articulation.get_angular_velocity()
        if linear_velocity is None:
            linear_velocity = np.zeros(3, dtype=float)
        if angular_velocity is None:
            angular_velocity = np.zeros(3, dtype=float)
        body_linear = quat_rotate_inverse_wxyz(quat_wxyz, np.asarray(linear_velocity, dtype=float).reshape(3))
        body_angular = quat_rotate_inverse_wxyz(quat_wxyz, np.asarray(angular_velocity, dtype=float).reshape(3))

        if abs(body_linear[0]) > 0.055 or abs(body_linear[1]) > 0.055 or abs(body_angular[2]) > 0.060:
            PHYSICS_IMU_STATE["idle_hold_pose"] = None
            return

        hold_pose = PHYSICS_IMU_STATE.get("idle_hold_pose")
        if hold_pose is None:
            PHYSICS_IMU_STATE["idle_hold_pose"] = (position.copy(), quat_wxyz.copy())
            return

        hold_position, hold_quat = hold_pose
        hold_position = np.asarray(hold_position, dtype=float).reshape(3)
        hold_quat = normalize_quat_wxyz(hold_quat)
        drift_xy = float(np.linalg.norm(position[:2] - hold_position[:2]))
        drift_z = abs(float(position[2] - hold_position[2]))
        if drift_xy > max(0.0, float(args.physics_idle_pose_hold_drift)) or drift_z > 0.025:
            articulation.set_world_pose(position=hold_position, orientation=hold_quat)
        articulation.set_linear_velocity(np.zeros(3, dtype=float))
        articulation.set_angular_velocity(np.zeros(3, dtype=float))
        PHYSICS_IMU_STATE["warned_idle_pose_hold"] = False
    except Exception as exc:
        if not PHYSICS_IMU_STATE["warned_idle_pose_hold"]:
            log(f"warning: failed to hold parked Hunter pose: {exc}")
            PHYSICS_IMU_STATE["warned_idle_pose_hold"] = True


def apply_hunter_terrain_sink_guard(articulation):
    if articulation is None or not args.physics_terrain_sink_guard or not is_mars_profile():
        return
    try:
        position, quat_wxyz = articulation.get_world_pose()
        position = np.asarray(position, dtype=float).reshape(3)
        ground_z = mars_terrain_height(position[0], position[1])
        min_z = ground_z + max(0.0, float(args.physics_terrain_min_base_clearance))
        if position[2] >= min_z:
            return

        reset_z = ground_z + max(
            float(args.physics_terrain_reset_base_clearance),
            float(args.physics_terrain_min_base_clearance),
        )
        corrected_position = position.copy()
        corrected_position[2] = reset_z
        articulation.set_world_pose(position=corrected_position, orientation=normalize_quat_wxyz(quat_wxyz))

        linear_velocity = articulation.get_linear_velocity()
        angular_velocity = articulation.get_angular_velocity()
        linear_velocity = np.zeros(3, dtype=float) if linear_velocity is None else np.asarray(linear_velocity, dtype=float).reshape(3)
        angular_velocity = np.zeros(3, dtype=float) if angular_velocity is None else np.asarray(angular_velocity, dtype=float).reshape(3)
        if linear_velocity[2] < 0.0:
            linear_velocity[2] = 0.0
        articulation.set_linear_velocity(linear_velocity)
        articulation.set_angular_velocity(angular_velocity)
        log(
            "Hunter terrain sink guard corrected base_link z: "
            f"{position[2]:.3f} -> {reset_z:.3f} at x={position[0]:.2f}, y={position[1]:.2f}"
        )
    except Exception as exc:
        if not PHYSICS_IMU_STATE.get("warned_sink_guard", False):
            log(f"warning: failed to apply Hunter terrain sink guard: {exc}")
            PHYSICS_IMU_STATE["warned_sink_guard"] = True


def update_hunter_joint_targets(stage, speed, yaw_rate, steer=None):
    if steer is None:
        steer = steer_from_speed_yaw_rate(speed, yaw_rate)
    steering_positions, wheel_velocities = ackermann_wheel_targets(speed, steer)
    for joint_name, target_position in steering_positions.items():
        set_drive_target(get_drive(stage, joint_name), target_position=target_position)
    for joint_name, target_velocity in wheel_velocities.items():
        set_drive_target(get_drive(stage, joint_name), target_velocity=target_velocity)


def hunter_articulation_root_paths(stage):
    roots = []
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if not (prim_path == ROBOT or prim_path.startswith(f"{ROBOT}/")):
            continue
        if prim_has_api(prim, UsdPhysics.ArticulationRootAPI):
            roots.append(prim_path)
    for fallback in (BASE_LINK, ROBOT):
        if stage.GetPrimAtPath(fallback) and fallback not in roots:
            roots.append(fallback)
    return roots


def initialize_hunter_articulation(stage):
    roots = hunter_articulation_root_paths(stage)
    last_error = None
    for root_path in roots:
        try:
            articulation = SingleArticulation(root_path, name="hunter_se_articulation")
            articulation.initialize()
            if not articulation.handles_initialized or articulation.num_dof <= 0:
                last_error = f"no initialized physics handles/dofs at {root_path}"
                log(f"warning: {last_error}")
                continue

            dof_names = list(articulation.dof_names)
            missing = [name for name in HUNTER_PHYSICS_DOF_NAMES if name not in dof_names]
            try:
                articulation.set_solver_position_iteration_count(16)
                articulation.set_solver_velocity_iteration_count(4)
                articulation.set_stabilization_threshold(0.001)
                articulation.set_sleep_threshold(0.0)
            except Exception as exc:
                log(f"warning: failed to tune Hunter articulation solver settings: {exc}")

            kps = np.full(articulation.num_dof, 0.0, dtype=float)
            kds = np.full(articulation.num_dof, 0.05, dtype=float)
            max_efforts = np.full(articulation.num_dof, 40.0, dtype=float)
            for joint_name in STEERING_JOINTS:
                if joint_name in dof_names:
                    idx = articulation.get_dof_index(joint_name)
                    controller = articulation.get_articulation_controller()
                    controller.switch_dof_control_mode(idx, "position")
                    kps[idx] = float(args.steering_position_kp)
                    kds[idx] = float(args.steering_position_kd)
                    max_efforts[idx] = float(args.steering_max_effort)
            if "steering_whee_joint" in dof_names:
                idx = articulation.get_dof_index("steering_whee_joint")
                controller = articulation.get_articulation_controller()
                controller.switch_dof_control_mode(idx, "position")
                kps[idx] = 6.0
                kds[idx] = 1.0
                max_efforts[idx] = 25.0
            for joint_name in WHEEL_JOINT_SIGNS:
                if joint_name in dof_names:
                    idx = articulation.get_dof_index(joint_name)
                    controller = articulation.get_articulation_controller()
                    controller.switch_dof_control_mode(idx, args.wheel_drive_mode)
                    kps[idx] = 0.0
                    kds[idx] = 0.0 if args.wheel_drive_mode == "effort" else args.wheel_velocity_damping
                    max_efforts[idx] = max(float(args.wheel_motor_effort), float(args.wheel_idle_brake_effort))
            controller = articulation.get_articulation_controller()
            controller.set_gains(kps, kds)
            controller.set_max_efforts(max_efforts)
            try:
                articulation.set_joint_velocities(np.zeros(articulation.num_dof, dtype=float))
            except Exception as exc:
                log(f"warning: failed to zero initial Hunter joint velocities: {exc}")

            log(f"initialized Hunter physics articulation at {root_path}")
            log(
                "Hunter wheel drive: "
                f"mode={args.wheel_drive_mode}, kp={args.wheel_velocity_kp:.1f}, "
                f"max_effort={args.wheel_motor_effort:.1f}, idle_brake_kp={args.wheel_idle_brake_kp:.1f}, "
                f"idle_brake_effort={args.wheel_idle_brake_effort:.1f}, velocity_damping={args.wheel_velocity_damping:.1f}"
            )
            log(f"Hunter articulation DOFs: {', '.join(dof_names)}")
            if missing:
                log(f"warning: imported Hunter articulation missing expected DOFs: {', '.join(missing)}")
            return articulation
        except Exception as exc:
            last_error = str(exc)
            log(f"warning: failed to initialize Hunter articulation at {root_path}: {exc}")

    log(f"warning: falling back to kinematic motion model; Hunter articulation unavailable ({last_error})")
    args.motion_model = "kinematic"
    return None


def apply_hunter_articulation_targets(articulation, speed, steer):
    if articulation is None:
        return
    dof_names = list(articulation.dof_names)
    speed = finite_or_zero(speed)
    steer = 0.0 if abs(finite_or_zero(steer)) < max(0.0, float(args.steering_zero_snap)) else finite_or_zero(steer)
    steering_positions, wheel_velocities = ackermann_wheel_targets(speed, steer)
    target_yaw_rate = yaw_rate_from_speed_steer(speed, steer)
    if args.motion_model == "physics" and (args.drive_speed_feedback_kp > 0.0 or args.drive_yaw_feedback_kp > 0.0):
        actual_speed, actual_yaw_rate = physics_body_speed_yaw_rate(articulation)
        feedback_speed = finite_or_zero(speed) + max(0.0, args.drive_speed_feedback_kp) * (finite_or_zero(speed) - actual_speed)
        feedback_yaw_rate = target_yaw_rate + max(0.0, args.drive_yaw_feedback_kp) * (target_yaw_rate - actual_yaw_rate)
        speed_limit = max(abs(args.cmd_vel_max_speed), abs(speed), 0.1) * 1.35
        yaw_limit = max(abs(args.cmd_vel_max_yaw_rate), abs(target_yaw_rate), 0.1) * 1.35
        feedback_speed = clamp_signed(feedback_speed, speed_limit)
        feedback_yaw_rate = clamp_signed(feedback_yaw_rate, yaw_limit)
        if abs(speed) < 0.015 and abs(steer) < 0.015:
            feedback_speed = 0.0
            feedback_yaw_rate = 0.0
        wheel_velocities = wheel_targets_from_speed_yaw_rate(feedback_speed, feedback_yaw_rate)
        PHYSICS_IMU_STATE["last_drive_feedback"] = (feedback_speed, feedback_yaw_rate)
    else:
        PHYSICS_IMU_STATE["last_drive_feedback"] = (finite_or_zero(speed), target_yaw_rate)
    PHYSICS_IMU_STATE["last_wheel_targets"] = dict(wheel_velocities)

    steer_indices = []
    steer_targets = []
    for joint_name, target_position in steering_positions.items():
        if joint_name not in dof_names:
            continue
        steer_indices.append(articulation.get_dof_index(joint_name))
        steer_targets.append(float(target_position))
    if steer_indices:
        steer_indices_array = np.asarray(steer_indices, dtype=np.int32)
        steer_targets_array = np.asarray(steer_targets, dtype=float)
        if args.steering_direct_zero_hold and np.max(np.abs(steer_targets_array)) < max(0.0, float(args.steering_zero_snap)):
            try:
                articulation.set_joint_positions(np.zeros_like(steer_targets_array), joint_indices=steer_indices_array)
                articulation.set_joint_velocities(np.zeros_like(steer_targets_array), joint_indices=steer_indices_array)
                PHYSICS_IMU_STATE["warned_steering_zero_hold"] = False
            except Exception as exc:
                if not PHYSICS_IMU_STATE["warned_steering_zero_hold"]:
                    log(f"warning: failed to directly hold zero steering joints: {exc}")
                    PHYSICS_IMU_STATE["warned_steering_zero_hold"] = True
        articulation.apply_action(
            ArticulationAction(
                joint_positions=steer_targets_array,
                joint_velocities=np.zeros_like(steer_targets_array),
                joint_indices=steer_indices_array,
            )
        )

    wheel_indices = []
    wheel_targets = []
    for joint_name, target_velocity in wheel_velocities.items():
        if joint_name not in dof_names:
            continue
        wheel_indices.append(articulation.get_dof_index(joint_name))
        wheel_targets.append(float(target_velocity))
    if wheel_indices:
        wheel_indices_array = np.asarray(wheel_indices, dtype=np.int32)
        wheel_targets_array = np.asarray(wheel_targets, dtype=float)
        if args.wheel_drive_mode == "effort":
            try:
                current_wheel_velocities = articulation.get_joint_velocities(joint_indices=wheel_indices_array)
                current_wheel_velocities = np.asarray(current_wheel_velocities, dtype=float).reshape(-1)
                if current_wheel_velocities.shape[0] != wheel_targets_array.shape[0]:
                    current_wheel_velocities = np.zeros_like(wheel_targets_array)
            except Exception:
                current_wheel_velocities = np.zeros_like(wheel_targets_array)
            idle_brake = abs(speed) < 0.015 and abs(steer) < max(0.0, float(args.steering_zero_snap))
            effort_gain = float(args.wheel_idle_brake_kp) if idle_brake else float(args.wheel_velocity_kp)
            effort_limit = float(args.wheel_idle_brake_effort) if idle_brake else float(args.wheel_motor_effort)
            wheel_efforts = effort_gain * (wheel_targets_array - current_wheel_velocities)
            wheel_efforts = np.clip(wheel_efforts, -abs(effort_limit), abs(effort_limit))
            articulation.apply_action(
                ArticulationAction(
                    joint_efforts=wheel_efforts,
                    joint_indices=wheel_indices_array,
                )
            )
        else:
            articulation.apply_action(
                ArticulationAction(
                    joint_velocities=wheel_targets_array,
                    joint_indices=wheel_indices_array,
                )
            )


def update_physics_imu_from_articulation(articulation, dt):
    if articulation is None or args.visual_only:
        return
    try:
        position, quat_wxyz = articulation.get_world_pose()
        position = np.asarray(position, dtype=float).reshape(3)
        quat_wxyz = normalize_quat_wxyz(quat_wxyz)
        linear_velocity = articulation.get_linear_velocity()
        angular_velocity = articulation.get_angular_velocity()
        if linear_velocity is None:
            prev_position = PHYSICS_IMU_STATE["prev_position"]
            linear_velocity = np.zeros(3, dtype=float) if prev_position is None else (position - prev_position) / max(dt, 1e-4)
        if angular_velocity is None:
            angular_velocity = np.zeros(3, dtype=float)
        linear_velocity = np.asarray(linear_velocity, dtype=float).reshape(3)
        angular_velocity = np.asarray(angular_velocity, dtype=float).reshape(3)

        prev_linear_velocity = PHYSICS_IMU_STATE["prev_linear_velocity"]
        if prev_linear_velocity is None:
            linear_acc_world = np.zeros(3, dtype=float)
        else:
            linear_acc_world = (linear_velocity - prev_linear_velocity) / max(dt, 1e-4)
        lin_acc_body = quat_rotate_inverse_wxyz(quat_wxyz, linear_acc_world + np.array([0.0, 0.0, 9.81], dtype=float))
        ang_vel_body = quat_rotate_inverse_wxyz(quat_wxyz, angular_velocity)
        quat_ijkr = (quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0])
        publish_imu_values(quat_ijkr, lin_acc_body, ang_vel_body)
        PHYSICS_IMU_STATE["prev_position"] = position
        PHYSICS_IMU_STATE["prev_linear_velocity"] = linear_velocity
        PHYSICS_IMU_STATE["warned_imu"] = False
    except Exception as exc:
        if not PHYSICS_IMU_STATE["warned_imu"]:
            log(f"warning: failed to publish physics-derived IMU; keeping last IMU sample: {exc}")
            PHYSICS_IMU_STATE["warned_imu"] = True


def publish_odometry_from_articulation(articulation, t):
    if articulation is None or args.visual_only or ODOM_PUBLISHER is None or ODOM_MSG_CLS is None:
        return

    period = 1.0 / max(1.0, float(args.odom_publish_hz))
    if t - float(PHYSICS_IMU_STATE["last_odom_publish_time"]) < period:
        return
    PHYSICS_IMU_STATE["last_odom_publish_time"] = t

    try:
        position, quat_wxyz = articulation.get_world_pose()
        quat_wxyz = normalize_quat_wxyz(quat_wxyz)
        linear_velocity = articulation.get_linear_velocity()
        angular_velocity = articulation.get_angular_velocity()
        if linear_velocity is None:
            linear_velocity = np.zeros(3, dtype=float)
        if angular_velocity is None:
            angular_velocity = np.zeros(3, dtype=float)
        body_linear = quat_rotate_inverse_wxyz(quat_wxyz, np.asarray(linear_velocity, dtype=float).reshape(3))
        body_angular = quat_rotate_inverse_wxyz(quat_wxyz, np.asarray(angular_velocity, dtype=float).reshape(3))

        msg = ODOM_MSG_CLS()
        sec = int(max(0.0, float(t)))
        msg.header.stamp.sec = sec
        msg.header.stamp.nanosec = int((max(0.0, float(t)) - sec) * 1e9)
        msg.header.frame_id = "odom"
        msg.child_frame_id = "base_link"
        msg.pose.pose.position.x = float(position[0])
        msg.pose.pose.position.y = float(position[1])
        msg.pose.pose.position.z = float(position[2])
        msg.pose.pose.orientation.x = float(quat_wxyz[1])
        msg.pose.pose.orientation.y = float(quat_wxyz[2])
        msg.pose.pose.orientation.z = float(quat_wxyz[3])
        msg.pose.pose.orientation.w = float(quat_wxyz[0])
        msg.twist.twist.linear.x = float(body_linear[0])
        msg.twist.twist.linear.y = float(body_linear[1])
        msg.twist.twist.linear.z = float(body_linear[2])
        msg.twist.twist.angular.x = float(body_angular[0])
        msg.twist.twist.angular.y = float(body_angular[1])
        msg.twist.twist.angular.z = float(body_angular[2])
        ODOM_PUBLISHER.publish(msg)
        PHYSICS_IMU_STATE["warned_odom"] = False
    except Exception as exc:
        if not PHYSICS_IMU_STATE["warned_odom"]:
            log(f"warning: failed to publish Hunter odometry: {exc}")
            PHYSICS_IMU_STATE["warned_odom"] = True


def log_physics_drive_status(articulation, t, speed, steer):
    if articulation is None or args.drive_mode == "stationary":
        return
    if t - PHYSICS_IMU_STATE["last_drive_log_time"] < max(0.1, float(args.physics_drive_log_period)):
        return
    PHYSICS_IMU_STATE["last_drive_log_time"] = t
    try:
        position, _quat = articulation.get_world_pose()
        linear_velocity = articulation.get_linear_velocity()
        actual_speed = 0.0 if linear_velocity is None else float(np.linalg.norm(np.asarray(linear_velocity, dtype=float)[:2]))
        body_speed, body_yaw_rate = physics_body_speed_yaw_rate(articulation)
        feedback_speed, feedback_yaw_rate = PHYSICS_IMU_STATE.get(
            "last_drive_feedback",
            (speed, yaw_rate_from_speed_steer(speed, steer)),
        )
        assist_vx, assist_vy, assist_yaw_rate = PHYSICS_IMU_STATE.get("last_root_assist", (0.0, 0.0, 0.0))
        dof_names = list(articulation.dof_names)
        wheel_names = [name for name in ("fr_left_joint", "fr_right_joint", "re_left_joint", "re_right_joint") if name in dof_names]
        steer_names = [name for name in ("fr_steer_left_joint", "fr_steer_right_joint") if name in dof_names]
        wheel_summary = "wheel_vel=n/a"
        target_summary = "wheel_target=n/a"
        effort_summary = "wheel_effort=n/a"
        steer_summary = "steer_pos=n/a"
        if wheel_names:
            wheel_indices = np.asarray([articulation.get_dof_index(name) for name in wheel_names], dtype=np.int32)
            wheel_velocities = articulation.get_joint_velocities(joint_indices=wheel_indices)
            wheel_summary = "wheel_vel=[" + ", ".join(f"{name}:{float(value):+.2f}" for name, value in zip(wheel_names, wheel_velocities)) + "]"
            wheel_targets = PHYSICS_IMU_STATE.get("last_wheel_targets") or ackermann_wheel_targets(speed, steer)[1]
            target_summary = (
                "wheel_target=["
                + ", ".join(f"{name}:{float(wheel_targets.get(name, 0.0)):+.2f}" for name in wheel_names)
                + "]"
            )
            if args.wheel_drive_mode == "effort":
                try:
                    wheel_efforts = articulation.get_applied_joint_efforts(joint_indices=wheel_indices)
                    effort_summary = (
                        "wheel_effort=["
                        + ", ".join(f"{name}:{float(value):+.1f}" for name, value in zip(wheel_names, wheel_efforts))
                        + "]"
                    )
                except Exception:
                    effort_summary = "wheel_effort=unavailable"
        if steer_names:
            steer_indices = np.asarray([articulation.get_dof_index(name) for name in steer_names], dtype=np.int32)
            steer_positions = articulation.get_joint_positions(joint_indices=steer_indices)
            steer_summary = "steer_pos=[" + ", ".join(f"{name}:{float(value):+.2f}" for name, value in zip(steer_names, steer_positions)) + "]"
        terrain_summary = "terrain=n/a"
        if is_mars_profile():
            ground_z = mars_terrain_height(position[0], position[1])
            dzdx, dzdy = mars_terrain_gradient(position[0], position[1])
            slope_deg = math.degrees(math.atan(math.hypot(dzdx, dzdy)))
            terrain_summary = (
                f"terrain=[ground_z={ground_z:+.2f}, clearance={float(position[2] - ground_z):+.2f}, "
                f"slope={slope_deg:.1f}deg, dzdx={dzdx:+.2f}, dzdy={dzdy:+.2f}]"
            )
        log(
            "physics drive status: "
            f"cmd_speed={speed:+.3f} m/s, cmd_steer={steer:+.3f} rad, "
            f"body_speed={body_speed:+.3f} m/s, body_yaw_rate={body_yaw_rate:+.3f} rad/s, "
            f"feedback_speed={feedback_speed:+.3f} m/s, feedback_yaw_rate={feedback_yaw_rate:+.3f} rad/s, "
            f"assist_body_vx={assist_vx:+.3f} m/s, assist_body_vy={assist_vy:+.3f} m/s, "
            f"assist_yaw_rate={assist_yaw_rate:+.3f} rad/s, "
            f"actual_xy_speed={actual_speed:.3f} m/s, root=({position[0]:+.2f}, {position[1]:+.2f}, {position[2]:+.2f}), "
            f"{terrain_summary}, {steer_summary}, {target_summary}, {wheel_summary}, {effort_summary}"
        )
    except Exception as exc:
        log(f"warning: failed to read Hunter physics status: {exc}")


def update_hunter_and_imu(stage, robot_pose_ops, t, dt):
    if args.drive_mode == "cmd_vel":
        x, y, yaw, ax_w, ay_w, yaw_rate, speed = cmd_vel_trajectory(dt)
    else:
        x, y, yaw, ax_w, ay_w, yaw_rate, speed = trajectory(t)
    spawn_x, spawn_y = scene_spawn_xy()
    spawn_yaw = scene_spawn_yaw_rad()
    c0 = math.cos(spawn_yaw)
    s0 = math.sin(spawn_yaw)
    world_x = spawn_x + c0 * x - s0 * y
    world_y = spawn_y + s0 * x + c0 * y
    world_yaw = spawn_yaw + yaw
    set_pose_ops(robot_pose_ops, (world_x, world_y, scene_robot_base_z(world_x, world_y)), (0.0, 0.0, math.degrees(world_yaw)))
    update_hunter_joint_targets(stage, speed, yaw_rate)

    half_yaw = 0.5 * world_yaw
    quat_ijkr = (0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw))
    c = math.cos(world_yaw)
    s = math.sin(world_yaw)
    ax_b = c * ax_w + s * ay_w
    ay_b = -s * ax_w + c * ay_w
    lin_acc = (ax_b, ay_b, 9.81)
    ang_vel = (0.0, 0.0, yaw_rate)
    publish_imu_values(quat_ijkr, lin_acc, ang_vel)


def configure_visual_settings(settings):
    high_quality = args.renderer.lower() == "pathtracing"
    ambient_occlusion = True
    for key, value in (
        ("/rtx/ambientOcclusion/enabled", ambient_occlusion),
        ("/rtx/ambientOcclusion/intensity", 0.78 if is_mars_profile() else 0.36),
        ("/rtx/ambientOcclusion/radius", 1.8 if is_mars_profile() else 1.1),
        ("/rtx/directLighting/sampledLighting/enabled", high_quality),
        ("/rtx/reflections/enabled", high_quality),
        ("/app/viewport/grid/enabled", False),
    ):
        try:
            settings.set(key, value)
        except Exception as exc:
            log(f"warning: failed to set visual setting {key}: {exc}")


def configure_lidar_profile_folders(settings):
    if not CUSTOM_LIDAR_CONFIG_DIR.exists():
        return
    key = "/app/sensors/nv/lidar/profileBaseFolder"
    folder = str(CUSTOM_LIDAR_CONFIG_DIR.resolve())
    try:
        folders = settings.get(key) or []
        if isinstance(folders, str):
            folders = [folders]
        else:
            folders = list(folders)
        if folder not in folders:
            settings.set(key, [folder, *folders])
            log(f"registered custom RTX LiDAR profile folder: {folder}")
    except Exception as exc:
        log(f"warning: failed to register custom LiDAR profile folder {folder}: {exc}")


def main():
    log("enabling Isaac extensions")
    settings = carb.settings.get_settings()
    configure_visual_settings(settings)
    if not args.visual_only:
        settings.set("/exts/isaacsim.ros2.bridge/ros_distro", "humble")
        settings.set("/exts/isaacsim.ros2.bridge/publish_without_verification", True)
        enable_extension("isaacsim.ros2.bridge")
        enable_extension("isaacsim.sensors.rtx")
        enable_extension("isaacsim.sensors.camera")
        configure_lidar_profile_folders(settings)
    enable_extension("isaacsim.asset.importer.urdf")
    enable_extension("omni.kit.asset_converter")
    simulation_app.update()

    log("configuring stage")
    stage = configure_stage()
    simulation_app.update()
    viewer_camera_path = create_viewer_camera(stage)
    simulation_app.update()
    robot_pose_ops = import_hunter_robot(stage)
    simulation_app.update()
    create_sensor_frames(stage)
    simulation_app.update()
    if args.visual_only:
        log("visual-only mode: skipping ROS LiDAR, D455, /clock, and IMU graphs")
    else:
        create_lidar()
        simulation_app.update()
        if args.camera_enabled:
            create_camera(stage)
            simulation_app.update()
        else:
            log("D455 ROS camera topics disabled for this sensor profile")
        create_clock_and_imu_graph()
        simulation_app.update()
        if args.drive_mode == "cmd_vel":
            if not create_cmd_vel_subscriber():
                create_cmd_vel_graph()
                simulation_app.update()
    simulation_app.update()
    apply_hunter_visibility_style(stage)
    activate_viewer_camera(viewer_camera_path)
    select_robot_in_stage()
    simulation_app.update()

    log("initializing simulation context")
    sim_dt = 1.0 / max(args.physics_hz, 1.0)
    render_dt = 1.0 / max(args.render_hz, 1.0)
    render_every = max(1, int(round(args.physics_hz / max(args.render_hz, 1.0))))
    simulation_context = SimulationContext(physics_dt=sim_dt, rendering_dt=render_dt, stage_units_in_meters=1.0)
    simulation_context.initialize_physics()
    hunter_articulation = None
    if args.motion_model == "physics":
        hunter_articulation = initialize_hunter_articulation(stage)
        apply_hunter_articulation_targets(hunter_articulation, 0.0, 0.0)
    simulation_context.play()

    running = True

    def stop(_signum, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    if args.visual_only:
        log("visual-only scene is running; ROS sensor topics are disabled")
    else:
        imu_topics = [args.imu_topic]
        legacy_topic = args.legacy_imu_topic.strip()
        if legacy_topic and legacy_topic.lower() != "none" and legacy_topic != args.imu_topic:
            imu_topics.append(legacy_topic)
        camera_status = "with D455 camera topics" if args.camera_enabled else "without D455 camera topics"
        log("publishing /clock, /livox/lidar, " + ", ".join(imu_topics) + f" {camera_status}")
        log(
            "sensor timing: "
            f"physics={args.physics_hz:.1f}Hz, render={args.render_hz:.1f}Hz, "
            f"LiDAR target={args.lidar_hz:.1f}Hz frameSkip={args.lidar_frame_skip}, "
            f"fullScan={args.lidar_full_scan}, IMU target={args.imu_hz:.1f}Hz"
            + (
                f", camera frameSkip={args.camera_frame_skip}, depth frameSkip={args.depth_frame_skip}"
                if args.camera_enabled
                else ", camera disabled"
            )
        )
        if ODOM_PUBLISHER is not None:
            log(f"measured odometry topic: {args.odom_topic} at up to {args.odom_publish_hz:.1f}Hz")
    if args.drive_mode == "stationary":
        log("drive mode is stationary; Hunter will stay parked until a drive mode/controller is enabled")
    elif args.drive_mode == "cmd_vel":
        log(
            f"drive mode is cmd_vel; listening on {args.cmd_vel_topic} "
            f"(backend={args.cmd_vel_backend}, angular.z mode={args.cmd_vel_angular_mode}, "
            f"max speed={args.cmd_vel_max_speed:.2f} m/s, "
            f"max yaw rate={args.cmd_vel_max_yaw_rate:.2f} rad/s, "
            f"max steer={args.cmd_vel_max_steer:.2f} rad, max accel={args.cmd_vel_max_accel:.2f} m/s^2)"
        )
    else:
        log(f"drive mode is {args.drive_mode}; first {args.stationary_duration:.1f} seconds are stationary for IMU initialization")
    if args.motion_model == "physics":
        log("motion model is physics; Hunter root pose is not directly overwritten after spawn")
        log(f"physics settle duration is {max(0.0, args.physics_settle_duration):.2f}s with zero wheel targets")
        if args.physics_root_assist:
            log(
                f"physics root assist is enabled (mode={args.physics_root_assist_mode}, "
                f"accel={args.physics_root_assist_accel:.2f} m/s^2, "
                f"yaw_accel={args.physics_root_assist_yaw_accel:.2f} rad/s^2, "
                f"lateral_damping={args.physics_root_lateral_damping:.1f} 1/s)"
            )
        else:
            log("physics root assist is disabled; raw wheel contact controls chassis motion")
    else:
        log("motion model is kinematic; Hunter root pose is directly written each frame for sensor-pipeline debugging")

    frame = 0
    try:
        while simulation_app.is_running() and running:
            wall_frame_start = time.monotonic()
            t = frame * sim_dt
            if args.duration > 0.0 and t >= args.duration:
                break
            if args.motion_model == "physics":
                if t < max(0.0, args.physics_settle_duration):
                    speed, steer, _yaw_rate = 0.0, 0.0, 0.0
                else:
                    speed, steer, _yaw_rate = drive_command(t, sim_dt)
                apply_hunter_articulation_targets(hunter_articulation, speed, steer)
            else:
                update_hunter_and_imu(stage, robot_pose_ops, t, sim_dt)
                speed, steer = CMD_VEL_STATE["speed"], CMD_VEL_STATE["steer"]
            render_this_frame = (frame % render_every) == 0
            simulation_context.step(render=render_this_frame)
            if args.motion_model == "physics":
                apply_hunter_root_velocity_assist(hunter_articulation, speed, steer, sim_dt)
                apply_hunter_terrain_sink_guard(hunter_articulation)
                apply_hunter_idle_pose_hold(hunter_articulation, speed, steer, t)
                update_physics_imu_from_articulation(hunter_articulation, sim_dt)
                publish_odometry_from_articulation(hunter_articulation, t)
                log_physics_drive_status(hunter_articulation, t, speed, steer)
            if args.realtime:
                elapsed = time.monotonic() - wall_frame_start
                time.sleep(max(0.0, sim_dt - elapsed))
            frame += 1
    finally:
        simulation_context.stop()
        shutdown_cmd_vel_subscriber()
        simulation_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
