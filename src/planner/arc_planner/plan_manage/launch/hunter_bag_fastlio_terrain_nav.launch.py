from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    enable_fastlio = LaunchConfiguration("fastlio")
    enable_terrain = LaunchConfiguration("terrain")
    enable_planner = LaunchConfiguration("planner")
    enable_mpc = LaunchConfiguration("mpc")
    enable_rviz = LaunchConfiguration("rviz")
    enable_vehicle_model = LaunchConfiguration("vehicle_model")
    enable_grid_clouds = LaunchConfiguration("grid_clouds")
    enable_reference_guide = LaunchConfiguration("reference_guide")

    terrain_params = LaunchConfiguration("terrain_params")
    planning_params = LaunchConfiguration("planning_params")
    controller_params = LaunchConfiguration("controller_params")
    fastlio_config_path = LaunchConfiguration("fastlio_config_path")
    fastlio_config_file = LaunchConfiguration("fastlio_config_file")

    raw_bag_path = LaunchConfiguration("raw_bag_path")
    reference_bag_path = LaunchConfiguration("reference_bag_path")
    play_raw_bag = LaunchConfiguration("play_raw_bag")
    play_reference_bag = LaunchConfiguration("play_reference_bag")
    play_rate = LaunchConfiguration("play_rate")
    raw_bag_delay = LaunchConfiguration("raw_bag_delay")
    reference_bag_delay = LaunchConfiguration("reference_bag_delay")
    raw_start_offset = LaunchConfiguration("raw_start_offset")
    reference_start_offset = LaunchConfiguration("reference_start_offset")
    clock_hz = LaunchConfiguration("clock_hz")

    lid_topic = LaunchConfiguration("lid_topic")
    imu_topic = LaunchConfiguration("imu_topic")
    map_frame = LaunchConfiguration("map_frame")
    body_frame = LaunchConfiguration("body_frame")
    car_id = LaunchConfiguration("car_id")

    odom_topic = LaunchConfiguration("odom_topic")
    mpc_odom_topic = LaunchConfiguration("mpc_odom_topic")
    cloud_topic = LaunchConfiguration("cloud_topic")
    cmd_topic = LaunchConfiguration("cmd_topic")

    new_path_topic = LaunchConfiguration("new_path_topic")
    new_cloud_body_topic = LaunchConfiguration("new_cloud_body_topic")
    new_cloud_effected_topic = LaunchConfiguration("new_cloud_effected_topic")
    new_laser_map_topic = LaunchConfiguration("new_laser_map_topic")
    new_local_map_topic = LaunchConfiguration("new_local_map_topic")

    reference_bag_odom_topic = LaunchConfiguration("reference_bag_odom_topic")
    reference_odom_topic = LaunchConfiguration("reference_odom_topic")
    reference_lead_time = LaunchConfiguration("reference_lead_time")
    reference_window_duration = LaunchConfiguration("reference_window_duration")
    reference_max_points = LaunchConfiguration("reference_max_points")
    guide_path_topic = LaunchConfiguration("guide_path_topic")
    guide_marker_topic = LaunchConfiguration("guide_marker_topic")

    fused_map_topic = PythonExpression(["'/car_' + '", car_id, "' + '_terrain_analyzer/fused_map'"])
    sdf_map_topic = PythonExpression(["'/car_' + '", car_id, "' + '_terrain_analyzer/sdf_map'"])
    elevation_grid_cloud_topic = PythonExpression(["'/car_' + '", car_id, "' + '_terrain_analyzer/elevation_grid_cloud'"])
    sdf_grid_cloud_topic = PythonExpression(["'/car_' + '", car_id, "' + '_terrain_analyzer/sdf_grid_cloud'"])
    risk_grid_cloud_topic = PythonExpression(["'/car_' + '", car_id, "' + '_terrain_analyzer/risk_grid_cloud'"])

    raw_bag_player = ExecuteProcess(
        cmd=[
            "ros2",
            "bag",
            "play",
            raw_bag_path,
            "--rate",
            play_rate,
            "--clock",
            clock_hz,
            "--delay",
            raw_bag_delay,
            "--start-offset",
            raw_start_offset,
            "--topics",
            lid_topic,
            imu_topic,
        ],
        output="screen",
        condition=IfCondition(play_raw_bag),
    )

    reference_bag_player = Node(
        package="fast_lio_rebuild",
        executable="odom_bag_republisher_node",
        name="reference_odom_bag_republisher",
        output="screen",
        parameters=[{
            "bag_path": reference_bag_path,
            "input_topic": reference_bag_odom_topic,
            "output_topic": reference_odom_topic,
            "play_rate": ParameterValue(play_rate, value_type=float),
            "start_delay": ParameterValue(reference_bag_delay, value_type=float),
            "start_offset": ParameterValue(reference_start_offset, value_type=float),
        }],
        condition=IfCondition(play_reference_bag),
    )

    fast_lio_node = Node(
        package="fast_lio",
        executable="fastlio_mapping",
        name="fastlio_mapping_rebuild_for_terrain",
        output="screen",
        parameters=[
            PathJoinSubstitution([fastlio_config_path, fastlio_config_file]),
            {
                "use_sim_time": use_sim_time,
                "common.lid_topic": lid_topic,
                "common.imu_topic": imu_topic,
                "frame.map_frame": map_frame,
                "frame.body_frame": body_frame,
            },
        ],
        remappings=[
            ("/Odometry", odom_topic),
            ("/path", new_path_topic),
            ("/cloud_registered", cloud_topic),
            ("/cloud_registered_body", new_cloud_body_topic),
            ("/cloud_effected", new_cloud_effected_topic),
            ("/Laser_map", new_laser_map_topic),
            ("/cloud_local_map", new_local_map_topic),
        ],
        condition=IfCondition(enable_fastlio),
    )

    terrain_node = Node(
        package="terrain_analyzer",
        executable="terrain_analyzer",
        name=["car_", car_id, "_terrain_analyzer"],
        output="screen",
        parameters=[terrain_params, {"use_sim_time": use_sim_time}],
        remappings=[
            ("odom", odom_topic),
            ("cloud", cloud_topic),
            ("fused_map", ["/car_", car_id, "_terrain_analyzer/fused_map"]),
            ("sdf_map", ["/car_", car_id, "_terrain_analyzer/sdf_map"]),
            ("elevation_cloud", ["/car_", car_id, "_terrain_analyzer/elevation_cloud"]),
            ("sdf_cloud", ["/car_", car_id, "_terrain_analyzer/sdf_cloud"]),
            ("risk_cloud", ["/car_", car_id, "_terrain_analyzer/risk_cloud"]),
            ("obstacle_cloud", ["/car_", car_id, "_terrain_analyzer/obstacle_cloud"]),
            ("overhead_cloud", ["/car_", car_id, "_terrain_analyzer/overhead_cloud"]),
        ],
        condition=IfCondition(enable_terrain),
    )

    elevation_grid_converter = Node(
        package="se2_grid_converter",
        executable="se2_grid_to_pointcloud_node",
        name="bag_terrain_elevation_grid_converter",
        output="screen",
        parameters=[{
            "use_sim_time": use_sim_time,
            "input_topic": fused_map_topic,
            "output_topic": elevation_grid_cloud_topic,
            "layer_name": "elevation",
            "valid_layer_name": "valid",
            "valid_threshold": 0.5,
            "height_as_z": True,
            "z_offset": 0.02,
            "use_color_map": True,
            "min_height": -1.0,
            "max_height": 2.0,
        }],
        condition=IfCondition(enable_grid_clouds),
    )

    sdf_grid_converter = Node(
        package="se2_grid_converter",
        executable="se2_grid_to_pointcloud_node",
        name="bag_terrain_sdf_grid_converter",
        output="screen",
        parameters=[{
            "use_sim_time": use_sim_time,
            "input_topic": sdf_map_topic,
            "output_topic": sdf_grid_cloud_topic,
            "layer_name": "sdf_cpu",
            "valid_layer_name": "valid",
            "valid_threshold": 0.5,
            "invert_color_map": True,
            "height_as_z": False,
            "z_value": 0.04,
            "use_color_map": True,
            "min_height": 0.0,
            "max_height": 6.0,
            "colormap": "terrain",
        }],
        condition=IfCondition(enable_grid_clouds),
    )

    risk_grid_converter = Node(
        package="se2_grid_converter",
        executable="se2_grid_to_pointcloud_node",
        name="bag_terrain_risk_grid_converter",
        output="screen",
        parameters=[{
            "use_sim_time": use_sim_time,
            "input_topic": fused_map_topic,
            "output_topic": risk_grid_cloud_topic,
            "layer_name": "risk",
            "valid_layer_name": "valid",
            "valid_threshold": 0.5,
            "yaw_index": 0,
            "height_as_z": False,
            "z_value": 0.08,
            "use_color_map": True,
            "min_height": 0.0,
            "max_height": 1.0,
            "colormap": "risk_gray_red",
        }],
        condition=IfCondition(enable_grid_clouds),
    )

    reference_window_node = Node(
        package="fast_lio_rebuild",
        executable="reference_window_node",
        name="terrain_reference_window",
        output="screen",
        parameters=[{
            "use_sim_time": use_sim_time,
            "new_odom_topic": odom_topic,
            "reference_odom_topic": reference_odom_topic,
            "path_topic": guide_path_topic,
            "marker_topic": guide_marker_topic,
            "map_frame": map_frame,
            "reference_lead_time": ParameterValue(reference_lead_time, value_type=float),
            "window_duration": ParameterValue(reference_window_duration, value_type=float),
            "max_points": ParameterValue(reference_max_points, value_type=int),
            "line_width": 0.22,
            "point_scale": 0.32,
            "z_offset": 0.16,
            "pre_simplify_tolerance": 0.08,
            "smooth_iterations": 5,
            "smooth_spacing": 0.04,
            "min_smooth_points": 108,
            "max_smooth_points": 340,
            "post_smooth_passes": 6,
            "post_smooth_strength": 0.62,
        }],
        condition=IfCondition(enable_reference_guide),
    )

    planner_node = Node(
        package="plan_manage",
        executable="plan_node",
        name="planner_node",
        output="both",
        respawn=True,
        respawn_delay=1.0,
        parameters=[
            planning_params,
            {
                "use_sim_time": use_sim_time,
                "odometry_topic": odom_topic,
                "vehicle/car_id": ParameterValue(car_id, value_type=int),
            },
        ],
        condition=IfCondition(enable_planner),
    )

    mpc_node = Node(
        package="mpc_controller",
        executable="mpc_controller_node",
        name="hunter_bag_mpc_controller",
        output="screen",
        parameters=[controller_params, {"use_sim_time": use_sim_time}],
        remappings=[
            ("cmd_vel", cmd_topic),
            ("odom", mpc_odom_topic),
            ("traj", "/planner/trajectory"),
            ("dptraj", "/planner/dptrajectory"),
        ],
        condition=IfCondition(enable_mpc),
    )

    vehicle_model_node = Node(
        package="vehicle_visualizer",
        executable="vehicle_visualizer_node",
        name="vehicle_visualizer",
        output="screen",
        parameters=[{
            "use_sim_time": use_sim_time,
            "frame_id": "map",
            "publish_period_sec": 0.05,
            "marker_lifetime_sec": 0.30,
            "vehicle.length": 0.840,
            "vehicle.width": 0.626,
            "vehicle.height": 0.338,
            "vehicle.wheel_base": 0.430,
            "vehicle.track_width": 0.548,
            "mesh.resource": "package://fake_ugv/meshes/car.dae",
            "mesh.use_embedded_materials": True,
            "mesh.alpha": 0.96,
            "mesh.scale_x": 0.833,
            "mesh.scale_y": 0.816,
            "mesh.scale_z": 0.878,
            "mesh.offset_z": -0.13,
        }],
        remappings=[
            ("odom", odom_topic),
            ("vehicle_model", "/vehicle_model"),
        ],
        condition=IfCondition(enable_vehicle_model),
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz_hunter_bag_fastlio_terrain_nav",
        output="screen",
        arguments=[
            "-d",
            PathJoinSubstitution([
                FindPackageShare("plan_manage"),
                "launch",
                "bag_fastlio_terrain_nav.rviz",
            ]),
        ],
        parameters=[{"use_sim_time": use_sim_time}],
        condition=IfCondition(enable_rviz),
    )

    return LaunchDescription([
        SetEnvironmentVariable("RCUTILS_LOGGING_USE_STDOUT", "1"),
        SetEnvironmentVariable("RCUTILS_COLORIZED_OUTPUT", "1"),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("fastlio", default_value="true"),
        DeclareLaunchArgument("terrain", default_value="true"),
        DeclareLaunchArgument("planner", default_value="false"),
        DeclareLaunchArgument("mpc", default_value="false"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("vehicle_model", default_value="true"),
        DeclareLaunchArgument("grid_clouds", default_value="false"),
        DeclareLaunchArgument("reference_guide", default_value="true"),
        DeclareLaunchArgument("car_id", default_value="0"),
        DeclareLaunchArgument("lid_topic", default_value="/livox/lidar"),
        DeclareLaunchArgument("imu_topic", default_value="/imu/data_raw"),
        DeclareLaunchArgument("map_frame", default_value="map"),
        DeclareLaunchArgument("body_frame", default_value="base_link"),
        DeclareLaunchArgument("odom_topic", default_value="/fastlio_rebuild/Odometry"),
        DeclareLaunchArgument("mpc_odom_topic", default_value="/fastlio_rebuild/Odometry"),
        DeclareLaunchArgument("cloud_topic", default_value="/fastlio_rebuild/cloud_registered"),
        DeclareLaunchArgument("cmd_topic", default_value="/hunter_se/cmd_vel"),
        DeclareLaunchArgument("new_path_topic", default_value="/fastlio_rebuild/path"),
        DeclareLaunchArgument("new_cloud_body_topic", default_value="/fastlio_rebuild/cloud_registered_body"),
        DeclareLaunchArgument("new_cloud_effected_topic", default_value="/fastlio_rebuild/cloud_effected"),
        DeclareLaunchArgument("new_laser_map_topic", default_value="/fastlio_rebuild/Laser_map"),
        DeclareLaunchArgument("new_local_map_topic", default_value="/fastlio_rebuild/cloud_local_map"),
        DeclareLaunchArgument("reference_bag_odom_topic", default_value="/Odometry"),
        DeclareLaunchArgument("reference_odom_topic", default_value="/fastlio_reference/Odometry"),
        DeclareLaunchArgument("guide_path_topic", default_value="/fastlio_rebuild/reference_window_path"),
        DeclareLaunchArgument("guide_marker_topic", default_value="/fastlio_rebuild/reference_window_marker"),
        DeclareLaunchArgument("reference_lead_time", default_value="0.45"),
        DeclareLaunchArgument("reference_window_duration", default_value="3.0"),
        DeclareLaunchArgument("reference_max_points", default_value="80"),
        DeclareLaunchArgument(
            "raw_bag_path",
            default_value="/home/z/workspace_zyz/bags/rosbag2_2026_07_07-19_37_13",
        ),
        DeclareLaunchArgument("reference_bag_path", default_value="/home/z/workspace_zyz/bags/rosbag2_2026_07_07-21_53_43"),
        DeclareLaunchArgument("play_raw_bag", default_value="true"),
        DeclareLaunchArgument("play_reference_bag", default_value="true"),
        DeclareLaunchArgument("play_rate", default_value="1.0"),
        DeclareLaunchArgument("raw_bag_delay", default_value="0.60"),
        DeclareLaunchArgument("reference_bag_delay", default_value="0.01"),
        DeclareLaunchArgument("raw_start_offset", default_value="0.0"),
        DeclareLaunchArgument("reference_start_offset", default_value="0.0"),
        DeclareLaunchArgument("clock_hz", default_value="100.0"),
        DeclareLaunchArgument(
            "fastlio_config_path",
            default_value=PathJoinSubstitution([
                FindPackageShare("fast_lio_rebuild"),
                "config",
            ]),
        ),
        DeclareLaunchArgument("fastlio_config_file", default_value="neos_mid360_bag_rebuild.yaml"),
        DeclareLaunchArgument(
            "terrain_params",
            default_value=PathJoinSubstitution([
                FindPackageShare("terrain_analyzer"),
                "params",
                "hunter_fastlio_registered_live.yaml",
            ]),
        ),
        DeclareLaunchArgument(
            "planning_params",
            default_value=PathJoinSubstitution([
                FindPackageShare("plan_manage"),
                "config",
                "planning_hunter_gazebo.yaml",
            ]),
        ),
        DeclareLaunchArgument(
            "controller_params",
            default_value=PathJoinSubstitution([
                FindPackageShare("mpc_controller"),
                "params",
                "controller_hunter_gazebo.yaml",
            ]),
        ),
        raw_bag_player,
        reference_bag_player,
        fast_lio_node,
        terrain_node,
        elevation_grid_converter,
        sdf_grid_converter,
        risk_grid_converter,
        reference_window_node,
        planner_node,
        mpc_node,
        vehicle_model_node,
        rviz_node,
    ])
