from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    enable_terrain = LaunchConfiguration("terrain")
    enable_planner = LaunchConfiguration("planner")
    enable_mpc = LaunchConfiguration("mpc")
    enable_vehicle_model = LaunchConfiguration("vehicle_model")
    enable_grid_clouds = LaunchConfiguration("grid_clouds")
    enable_rviz = LaunchConfiguration("rviz")
    vehicle_driver = LaunchConfiguration("vehicle_driver")
    planner_backend = LaunchConfiguration("planner_backend")

    terrain_params = LaunchConfiguration("terrain_params")
    planning_params = LaunchConfiguration("planning_params")
    controller_params = LaunchConfiguration("controller_params")
    terrain_nav_params = LaunchConfiguration("terrain_nav_params")
    vehicle_params = LaunchConfiguration("vehicle_params")
    vehicle_ini_path = LaunchConfiguration("vehicle_ini_path")

    odom_topic = LaunchConfiguration("odom_topic")
    mpc_odom_topic = LaunchConfiguration("mpc_odom_topic")
    cloud_topic = LaunchConfiguration("cloud_topic")
    cmd_vel_topic = LaunchConfiguration("cmd_vel_topic")
    chassis_odom_topic = LaunchConfiguration("chassis_odom_topic")
    car_id = LaunchConfiguration("car_id")
    can_hub_addr = LaunchConfiguration("can_hub_addr")
    mode_can_lock = LaunchConfiguration("mode_can_lock")
    show_chassis_path = LaunchConfiguration("show_chassis_path")
    show_chassis_loc = LaunchConfiguration("show_chassis_loc")
    calc_chassis_speed = LaunchConfiguration("calc_chassis_speed")
    publish_chassis_tf = LaunchConfiguration("publish_chassis_tf")
    chassis_rate_x = LaunchConfiguration("chassis_rate_x")
    chassis_rate_y = LaunchConfiguration("chassis_rate_y")
    chassis_rate_z = LaunchConfiguration("chassis_rate_z")
    chassis_rate_az = LaunchConfiguration("chassis_rate_az")

    fused_map_topic = PythonExpression(["'/car_' + '", car_id, "' + '_terrain_analyzer/fused_map'"])
    sdf_map_topic = PythonExpression(["'/car_' + '", car_id, "' + '_terrain_analyzer/sdf_map'"])
    elevation_grid_cloud_topic = PythonExpression(["'/car_' + '", car_id, "' + '_terrain_analyzer/elevation_grid_cloud'"])
    sdf_grid_cloud_topic = PythonExpression(["'/car_' + '", car_id, "' + '_terrain_analyzer/sdf_grid_cloud'"])
    risk_grid_cloud_topic = PythonExpression(["'/car_' + '", car_id, "' + '_terrain_analyzer/risk_grid_cloud'"])
    use_hexman_driver = PythonExpression(["'", vehicle_driver, "' == 'hexman'"])

    return LaunchDescription([
        SetEnvironmentVariable("RCUTILS_LOGGING_USE_STDOUT", "1"),
        SetEnvironmentVariable("RCUTILS_COLORIZED_OUTPUT", "1"),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("terrain", default_value="false"),
        DeclareLaunchArgument("planner", default_value="true"),
        DeclareLaunchArgument("mpc", default_value="false"),
        DeclareLaunchArgument("vehicle_model", default_value="true"),
        DeclareLaunchArgument("grid_clouds", default_value="false"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("vehicle_driver", default_value="none"),
        DeclareLaunchArgument(
            "planner_backend",
            default_value="terrain_nav",
            description="Planning/control backend: terrain_nav uses the current terrain-aware planner/controller; arc keeps the legacy plan_manage + MPC stack.",
        ),
        DeclareLaunchArgument("car_id", default_value="0"),
        DeclareLaunchArgument("odom_topic", default_value="/Odometry"),
        DeclareLaunchArgument("mpc_odom_topic", default_value="/Odometry"),
        DeclareLaunchArgument("cloud_topic", default_value="/cloud_registered"),
        DeclareLaunchArgument("cmd_vel_topic", default_value="/cmd_vel"),
        DeclareLaunchArgument("chassis_odom_topic", default_value="/odom"),
        DeclareLaunchArgument("can_hub_addr", default_value="10.233.233.1"),
        DeclareLaunchArgument("mode_can_lock", default_value="true"),
        DeclareLaunchArgument("show_chassis_path", default_value="false"),
        DeclareLaunchArgument("show_chassis_loc", default_value="false"),
        DeclareLaunchArgument("calc_chassis_speed", default_value="false"),
        DeclareLaunchArgument("publish_chassis_tf", default_value="false"),
        DeclareLaunchArgument("chassis_rate_x", default_value="1.0"),
        DeclareLaunchArgument("chassis_rate_y", default_value="1.0"),
        DeclareLaunchArgument("chassis_rate_z", default_value="1.0"),
        DeclareLaunchArgument("chassis_rate_az", default_value="1.0"),
        DeclareLaunchArgument(
            "terrain_params",
            default_value=PathJoinSubstitution([
                FindPackageShare("terrain_analyzer"),
                "params",
                "real",
                "neos_registered_live.yaml",
            ]),
        ),
        DeclareLaunchArgument(
            "planning_params",
            default_value=PathJoinSubstitution([
                FindPackageShare("plan_manage"),
                "config",
                "real",
                "planning_neos_real.yaml",
            ]),
        ),
        DeclareLaunchArgument(
            "controller_params",
            default_value=PathJoinSubstitution([
                FindPackageShare("mpc_controller"),
                "params",
                "real",
                "controller_neos_real.yaml",
            ]),
        ),
        DeclareLaunchArgument(
            "terrain_nav_params",
            default_value=PathJoinSubstitution([
                FindPackageShare("terrain_nav"),
                "config",
                "neos_real.yaml",
            ]),
        ),
        DeclareLaunchArgument(
            "vehicle_params",
            default_value=PathJoinSubstitution([
                FindPackageShare("plan_manage"),
                "config",
                "real",
                "vehicle_neos_real.yaml",
            ]),
        ),
        DeclareLaunchArgument(
            "vehicle_ini_path",
            default_value=PathJoinSubstitution([
                FindPackageShare("xpkg_vehicle"),
                "ini",
                "device_id_list.ini",
            ]),
        ),
        Node(
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
        ),
        Node(
            package="se2_grid_converter",
            executable="se2_grid_to_pointcloud_node",
            name="real_terrain_elevation_grid_converter",
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
        ),
        Node(
            package="se2_grid_converter",
            executable="se2_grid_to_pointcloud_node",
            name="real_terrain_sdf_grid_converter",
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
        ),
        Node(
            package="se2_grid_converter",
            executable="se2_grid_to_pointcloud_node",
            name="real_terrain_risk_grid_converter",
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
        ),
        Node(
            package="plan_manage",
            executable="plan_node",
            name="planner_node",
            output="screen",
            parameters=[
                planning_params,
                {
                    "use_sim_time": use_sim_time,
                    "odometry_topic": odom_topic,
                    "vehicle/car_id": ParameterValue(car_id, value_type=int),
                },
            ],
            condition=IfCondition(PythonExpression(["'", enable_planner, "' == 'true' and '", planner_backend, "' == 'arc'"])),
        ),
        Node(
            package="terrain_nav",
            executable="terrain_nav_node",
            name="terrain_nav_node",
            output="screen",
            parameters=[
                terrain_nav_params,
                {
                    "use_sim_time": use_sim_time,
                    "odom_topic": odom_topic,
                    "fused_map_topic": fused_map_topic,
                    "cmd_topic": cmd_vel_topic,
                },
            ],
            condition=IfCondition(PythonExpression(["'", enable_planner, "' == 'true' and '", planner_backend, "' == 'terrain_nav'"])),
        ),
        Node(
            package="mpc_controller",
            executable="mpc_controller_node",
            name="neos_mpc_controller",
            output="screen",
            parameters=[controller_params, {"use_sim_time": use_sim_time}],
            remappings=[
                ("cmd_vel", cmd_vel_topic),
                ("odom", mpc_odom_topic),
                ("traj", "/planner/trajectory"),
                ("dptraj", "/planner/dptrajectory"),
            ],
            condition=IfCondition(PythonExpression(["'", enable_mpc, "' == 'true' and '", planner_backend, "' == 'arc'"])),
        ),
        Node(
            package="xpkg_comm",
            executable="xnode_comm",
            name="xnode_comm",
            output="screen",
            parameters=[{
                "can_hub_addr": can_hub_addr,
                "dev_list": False,
                "com_enable": True,
                "com_channel_common": False,
                "com_channel_xstd": True,
                "tcp_enable": False,
                "tcp_channel_common": False,
                "tcp_channel_xstd": False,
            }],
            condition=IfCondition(use_hexman_driver),
        ),
        Node(
            package="xpkg_vehicle",
            executable="xnode_vehicle",
            name="xnode_vehicle",
            output="screen",
            parameters=[{
                "ini_path": vehicle_ini_path,
                "show_path": ParameterValue(show_chassis_path, value_type=bool),
                "show_loc": ParameterValue(show_chassis_loc, value_type=bool),
                "calc_speed": ParameterValue(calc_chassis_speed, value_type=bool),
                "mode_can_lock": ParameterValue(mode_can_lock, value_type=bool),
                "pub_tf": ParameterValue(publish_chassis_tf, value_type=bool),
                "rate_x": ParameterValue(chassis_rate_x, value_type=float),
                "rate_y": ParameterValue(chassis_rate_y, value_type=float),
                "rate_z": ParameterValue(chassis_rate_z, value_type=float),
                "rate_az": ParameterValue(chassis_rate_az, value_type=float),
            }],
            remappings=[
                ("/cmd_vel", cmd_vel_topic),
                ("/odom", chassis_odom_topic),
            ],
            condition=IfCondition(use_hexman_driver),
        ),
        Node(
            package="vehicle_visualizer",
            executable="vehicle_visualizer_node",
            name="vehicle_visualizer",
            output="screen",
            parameters=[vehicle_params, {"use_sim_time": use_sim_time}],
            remappings=[
                ("odom", odom_topic),
                ("vehicle_model", "/vehicle_model"),
            ],
            condition=IfCondition(enable_vehicle_model),
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz_neos_planning_control",
            output="screen",
            arguments=[
                "-d",
                PathJoinSubstitution([
                    FindPackageShare("plan_manage"),
                    "launch",
                    "gazebo_fastlio_nav_clean.rviz",
                ]),
            ],
            parameters=[{"use_sim_time": use_sim_time}],
            condition=IfCondition(enable_rviz),
        ),
    ])
