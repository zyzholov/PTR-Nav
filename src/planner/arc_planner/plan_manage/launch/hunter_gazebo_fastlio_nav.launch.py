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
    enable_rviz = LaunchConfiguration("rviz")
    enable_vehicle_model = LaunchConfiguration("vehicle_model")
    enable_grid_clouds = LaunchConfiguration("grid_clouds")
    planner_backend = LaunchConfiguration("planner_backend")
    rviz_cfg = LaunchConfiguration("rviz_cfg")

    terrain_params = LaunchConfiguration("terrain_params")
    planning_params = LaunchConfiguration("planning_params")
    controller_params = LaunchConfiguration("controller_params")
    terrain_nav_params = LaunchConfiguration("terrain_nav_params")

    odom_topic = LaunchConfiguration("odom_topic")
    mpc_odom_topic = LaunchConfiguration("mpc_odom_topic")
    cloud_topic = LaunchConfiguration("cloud_topic")
    cmd_topic = LaunchConfiguration("cmd_topic")
    car_id = LaunchConfiguration("car_id")
    fused_map_topic = PythonExpression(["'/car_' + '", car_id, "' + '_terrain_analyzer/fused_map'"])
    sdf_map_topic = PythonExpression(["'/car_' + '", car_id, "' + '_terrain_analyzer/sdf_map'"])
    elevation_grid_cloud_topic = PythonExpression(["'/car_' + '", car_id, "' + '_terrain_analyzer/elevation_grid_cloud'"])
    sdf_grid_cloud_topic = PythonExpression(["'/car_' + '", car_id, "' + '_terrain_analyzer/sdf_grid_cloud'"])
    risk_grid_cloud_topic = PythonExpression(["'/car_' + '", car_id, "' + '_terrain_analyzer/risk_grid_cloud'"])

    return LaunchDescription([
        SetEnvironmentVariable("RCUTILS_LOGGING_USE_STDOUT", "1"),
        SetEnvironmentVariable("RCUTILS_COLORIZED_OUTPUT", "1"),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("terrain", default_value="true"),
        DeclareLaunchArgument("planner", default_value="true"),
        DeclareLaunchArgument("mpc", default_value="true"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("vehicle_model", default_value="true"),
        DeclareLaunchArgument("grid_clouds", default_value="false"),
        DeclareLaunchArgument(
            "rviz_cfg",
            default_value=PathJoinSubstitution([
                FindPackageShare("plan_manage"),
                "launch",
                "gazebo_fastlio_nav_clean.rviz",
            ]),
        ),
        DeclareLaunchArgument(
            "planner_backend",
            default_value="terrain_nav",
            description="Planning/control backend: terrain_nav uses the local SE2 state-lattice planner/tracker; arc keeps the existing Kino-A*/MPC stack.",
        ),
        DeclareLaunchArgument("car_id", default_value="0"),
        DeclareLaunchArgument("odom_topic", default_value="/Odometry"),
        DeclareLaunchArgument("mpc_odom_topic", default_value="/Odometry"),
        DeclareLaunchArgument("cloud_topic", default_value="/cloud_registered"),
        DeclareLaunchArgument("cmd_topic", default_value="/hunter_se/cmd_vel"),
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
        DeclareLaunchArgument(
            "terrain_nav_params",
            default_value=PathJoinSubstitution([
                FindPackageShare("terrain_nav"),
                "config",
                "hunter_gazebo.yaml",
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
            name="terrain_elevation_grid_converter",
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
            name="terrain_sdf_grid_converter",
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
            name="terrain_risk_grid_converter",
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
            output="both",
            respawn=True,
            respawn_delay=1.0,
            parameters=[
                planning_params,
                {
                    "use_sim_time": use_sim_time,
                    "odometry_topic": odom_topic,
                    "vehicle/car_id": ParameterValue(car_id, value_type=int),
                    "fsm/track_err_replan_enable": False,
                    "fsm/track_err_replan_dist": 2.60,
                    "fsm/track_err_replan_yaw": 1.25,
                    "fsm/track_err_replan_min_elapsed": 2.50,
                    "fsm/track_err_replan_hits": 5,
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
                    "cmd_topic": cmd_topic,
                },
            ],
            condition=IfCondition(PythonExpression(["'", enable_planner, "' == 'true' and '", planner_backend, "' == 'terrain_nav'"])),
        ),
        Node(
            package="mpc_controller",
            executable="mpc_controller_node",
            name="hunter_gazebo_mpc_controller",
            output="screen",
            parameters=[controller_params, {"use_sim_time": use_sim_time}],
            remappings=[
                ("cmd_vel", cmd_topic),
                ("odom", mpc_odom_topic),
                ("traj", "/planner/trajectory"),
                ("dptraj", "/planner/dptrajectory"),
            ],
            condition=IfCondition(PythonExpression(["'", enable_mpc, "' == 'true' and '", planner_backend, "' == 'arc'"])),
        ),
        Node(
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
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz_hunter_gazebo_fastlio_nav",
            output="screen",
            arguments=[
                "-d",
                rviz_cfg,
            ],
            parameters=[{"use_sim_time": use_sim_time}],
            condition=IfCondition(enable_rviz),
        ),
    ])
