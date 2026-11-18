from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def _equals(value, expected):
    return IfCondition(PythonExpression(["'", value, "' == '", expected, "'"]))


def generate_launch_description():
    numeric_threads=EnvironmentVariable('HUNTER_FSGP_CPU_THREADS',default_value=
        EnvironmentVariable('HUNTER_TERRAIN_CPU_THREADS',default_value='1'))
    use_sim_time = LaunchConfiguration("use_sim_time")
    terrain_backend = LaunchConfiguration("terrain_backend")
    terrain_cpu_params = LaunchConfiguration("terrain_cpu_params")
    terrain_gpu_params = LaunchConfiguration("terrain_gpu_params")
    fsgp_params = LaunchConfiguration("fsgp_params")
    terrain_nav_params = LaunchConfiguration("terrain_nav_params")
    fsgp_gpu_backend = LaunchConfiguration("fsgp_gpu_backend")
    fsgp_max_update_rate_hz = LaunchConfiguration("fsgp_max_update_rate_hz")
    odom_topic = LaunchConfiguration("odom_topic")
    cloud_topic = LaunchConfiguration("cloud_topic")
    cmd_topic = LaunchConfiguration("cmd_topic")
    input_fused_map_topic = LaunchConfiguration("input_fused_map_topic")
    input_sdf_map_topic = LaunchConfiguration("input_sdf_map_topic")
    output_fused_map_topic = LaunchConfiguration("output_fused_map_topic")
    enable_nav = LaunchConfiguration("nav")
    enable_rviz = LaunchConfiguration("rviz")
    enable_vehicle_model = LaunchConfiguration("vehicle_model")
    enable_grid_clouds = LaunchConfiguration("grid_clouds")
    rviz_cfg = LaunchConfiguration("rviz_cfg")

    common_terrain_remaps = [
        ("odom", odom_topic),
        ("cloud", cloud_topic),
        ("fused_map", input_fused_map_topic),
        ("sdf_map", input_sdf_map_topic),
        ("elevation_cloud", "/car_0_terrain_analyzer/baseline_elevation_cloud"),
        ("sdf_cloud", "/car_0_terrain_analyzer/baseline_sdf_cloud"),
        ("risk_cloud", "/car_0_terrain_analyzer/baseline_risk_cloud"),
        ("obstacle_cloud", "/car_0_terrain_analyzer/obstacle_cloud"),
        ("overhead_cloud", "/car_0_terrain_analyzer/overhead_cloud"),
    ]

    return LaunchDescription([
        SetEnvironmentVariable("RCUTILS_LOGGING_USE_STDOUT", "1"),
        SetEnvironmentVariable("RCUTILS_COLORIZED_OUTPUT", "1"),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument(
            "terrain_backend",
            default_value="gpu",
            description="Terrain analyzer implementation: gpu or cpu.",
        ),
        DeclareLaunchArgument("odom_topic", default_value="/Odometry"),
        DeclareLaunchArgument("cloud_topic", default_value="/cloud_registered"),
        DeclareLaunchArgument("cmd_topic", default_value="/cmd_vel"),
        DeclareLaunchArgument(
            "input_fused_map_topic",
            default_value="/car_0_terrain_analyzer/fused_map",
        ),
        DeclareLaunchArgument(
            "input_sdf_map_topic",
            default_value="/car_0_terrain_analyzer/sdf_map",
        ),
        DeclareLaunchArgument(
            "output_fused_map_topic",
            default_value="/car_0_fsgp_bgk/fused_map",
        ),
        DeclareLaunchArgument("fsgp_gpu_backend", default_value="auto"),
        DeclareLaunchArgument("fsgp_max_update_rate_hz", default_value="8.0"),
        DeclareLaunchArgument(
            "nav",
            default_value="true",
            description="Start terrain_nav and allow it to publish cmd_topic.",
        ),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("vehicle_model", default_value="true"),
        DeclareLaunchArgument("grid_clouds", default_value="true"),
        DeclareLaunchArgument(
            "terrain_cpu_params",
            default_value=PathJoinSubstitution([
                FindPackageShare("terrain_analyzer"),
                "params",
                "hunter_fastlio_registered_live.yaml",
            ]),
        ),
        DeclareLaunchArgument(
            "terrain_gpu_params",
            default_value=PathJoinSubstitution([
                FindPackageShare("terrain_analyzer"),
                "params",
                "hunter_fastlio_gpu.yaml",
            ]),
        ),
        DeclareLaunchArgument(
            "fsgp_params",
            default_value=PathJoinSubstitution([
                FindPackageShare("fsgp_bgk_se2"),
                "config",
                "hunter_fastlio.yaml",
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
        DeclareLaunchArgument(
            "rviz_cfg",
            default_value=PathJoinSubstitution([
                FindPackageShare("fsgp_bgk_se2"),
                "rviz",
                "hunter_terrain_navigation.rviz",
            ]),
        ),
        Node(
            package="terrain_analyzer",
            executable="terrain_analyzer",
            name="car_0_terrain_analyzer",
            output="screen",
            parameters=[terrain_cpu_params, {"use_sim_time": use_sim_time}],
            remappings=common_terrain_remaps,
            condition=_equals(terrain_backend, "cpu"),
        ),
        Node(
            package="terrain_analyzer",
            executable="terrain_analyzer_gpu",
            name="car_0_terrain_analyzer",
            output="screen",
            parameters=[terrain_gpu_params, {"use_sim_time": use_sim_time}],
            remappings=common_terrain_remaps,
            condition=_equals(terrain_backend, "gpu"),
        ),
        Node(
            package="fsgp_bgk_se2",
            executable="fsgp_bgk_se2_node",
            name="fsgp_bgk_se2_node",
            additional_env={"RMW_FASTRTPS_PUBLICATION_MODE": EnvironmentVariable(
                "RMW_FASTRTPS_PUBLICATION_MODE", default_value="ASYNCHRONOUS"),
                "OPENBLAS_NUM_THREADS": EnvironmentVariable("OPENBLAS_NUM_THREADS", default_value=numeric_threads),
                "OMP_NUM_THREADS": EnvironmentVariable("OMP_NUM_THREADS", default_value=numeric_threads),
                "MKL_NUM_THREADS": EnvironmentVariable("MKL_NUM_THREADS", default_value=numeric_threads)},
            output="screen",
            parameters=[
                fsgp_params,
                {
                    "use_sim_time": use_sim_time,
                    "input_fused_map_topic": input_fused_map_topic,
                    "output_fused_map_topic": output_fused_map_topic,
                    "odom_topic": odom_topic,
                    "gpu_backend": fsgp_gpu_backend,
                    "max_update_rate_hz": ParameterValue(
                        fsgp_max_update_rate_hz, value_type=float
                    ),
                    "include_diagnostic_layers": False,
                    "publish_debug_clouds": False,
                },
            ],
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
                    "fused_map_topic": output_fused_map_topic,
                    "cmd_topic": cmd_topic,
                    "risk_query_mode": "se2_yaw",
                    "se2_risk_layer": "risk",
                    "hard_risk_layer": "baseline_risk",
                    "se2_risk_lethal": True,
                    "chance_constrained": True,
                },
            ],
            condition=IfCondition(enable_nav),
        ),
        Node(
            package="fsgp_bgk_se2", executable="navigation_display", name="navigation_display",
            output="screen", condition=IfCondition(enable_grid_clouds),
            parameters=[{"use_sim_time": use_sim_time, "odom_topic": odom_topic,
                         "grid_topic": output_fused_map_topic, "cmd_topic": cmd_topic}],
        ),
        Node(
            package="se2_grid_converter",
            executable="se2_grid_to_pointcloud_node",
            name="fsgp_elevation_grid_converter",
            output="screen",
            parameters=[{
                "use_sim_time": use_sim_time,
                "input_topic": output_fused_map_topic,
                "output_topic": "/car_0_terrain_analyzer/elevation_cloud",
                "layer_name": "elevation",
                "valid_layer_name": "valid",
                "valid_threshold": 0.5,
                "height_as_z": True,
                "z_offset": 0.02,
                "height_layer_name": "elevation",
                "drape_on_height": False,
                "publish_surface": True,
                "surface_topic": "/car_0_terrain_analyzer/terrain_surface",
                "surface_max_step": 0.65,
                "surface_alpha": 0.92,
                "surface_risk_layer": "failure_probability_upper",
                "surface_body_collision_overlay": False,
                "body_pose_surface_topic": "/terrain_display/body_pose_risk_surface",
                "surface_probability_colors": True,
                "follow_odom_yaw": True,
                "odom_topic": odom_topic,
                "use_color_map": True,
                "colormap": "terrain_surface",
                "min_height": -1.0,
                "max_height": 2.0,
            }],
            condition=IfCondition(enable_grid_clouds),
        ),
        Node(
            package="se2_grid_converter",
            executable="se2_grid_to_pointcloud_node",
            name="fsgp_risk_grid_converter",
            output="screen",
            parameters=[{
                "use_sim_time": use_sim_time,
                "input_topic": output_fused_map_topic,
                "output_topic": "/car_0_terrain_analyzer/risk_cloud",
                "layer_name": "failure_probability_upper",
                "valid_layer_name": "probability_valid",
                "valid_threshold": 0.5,
                "follow_odom_yaw": True,
                "odom_topic": odom_topic,
                "height_layer_name": "elevation",
                "drape_on_height": True,
                "height_as_z": False,
                "z_value": 0.035,
                "z_offset": 0.035,
                "use_color_map": True,
                "min_height": 0.0,
                "max_height": 1.0,
                "colormap": "probability",
            }],
            condition=IfCondition(enable_grid_clouds),
        ),
        Node(
            package="se2_grid_converter",
            executable="se2_grid_to_pointcloud_node",
            name="fsgp_sdf_grid_converter",
            output="screen",
            parameters=[{
                "use_sim_time": use_sim_time,
                "input_topic": input_sdf_map_topic,
                "output_topic": "/car_0_terrain_analyzer/sdf_cloud",
                "layer_name": "sdf_cpu",
                "height_as_z": False,
                "z_value": 0.04,
                "use_color_map": True,
                "invert_color_map": True,
                "min_height": 0.0,
                "max_height": 6.0,
                "colormap": "terrain",
            }],
            condition=IfCondition(enable_grid_clouds),
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
                # Keep the latest model through short Isaac/ROS odometry gaps.
                "marker_lifetime_sec": 0.0,
                "vehicle.length": 0.840,
                "vehicle.width": 0.626,
                "vehicle.height": 0.338,
                "vehicle.wheel_base": 0.549,
                "vehicle.track_width": 0.548,
                "mesh.resource": "package://fake_ugv/meshes/car.dae",
                "mesh.use_embedded_materials": True,
                "mesh.alpha": 0.96,
                "mesh.scale_x": 0.833,
                "mesh.scale_y": 0.816,
                "mesh.scale_z": 0.878,
                "mesh.offset_z": -0.13,
            }],
            remappings=[("odom", odom_topic), ("vehicle_model", "/vehicle_model")],
            condition=IfCondition(enable_vehicle_model),
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz_hunter_fsgp_bgk_nav",
            output="screen",
            arguments=["-d", rviz_cfg],
            parameters=[{"use_sim_time": use_sim_time}],
            condition=IfCondition(enable_rviz),
        ),
    ])
