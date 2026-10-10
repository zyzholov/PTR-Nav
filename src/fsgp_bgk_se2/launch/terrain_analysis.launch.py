"""Start the terrain front-end and the FSGP-BGK/SEB enhancement only.

Planning/control is intentionally a separate launch boundary.  This makes it
possible to inspect the baseline and enhanced maps without a second node
publishing /cmd_vel.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def _equals(value, expected):
    return IfCondition(PythonExpression(["'", value, "' == '", expected, "'"]))


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    terrain_backend = LaunchConfiguration("terrain_backend")
    terrain_dynamic_clear_enabled = LaunchConfiguration("terrain_dynamic_clear_enabled")
    terrain_dynamic_clear_frames = LaunchConfiguration("terrain_dynamic_clear_frames")
    terrain_cpu_params = LaunchConfiguration("terrain_cpu_params")
    terrain_gpu_params = LaunchConfiguration("terrain_gpu_params")
    fsgp_params = LaunchConfiguration("fsgp_params")
    fsgp_gpu_backend = LaunchConfiguration("fsgp_gpu_backend")
    fsgp_max_update_rate_hz = LaunchConfiguration("fsgp_max_update_rate_hz")
    odom_topic = LaunchConfiguration("odom_topic")
    cloud_topic = LaunchConfiguration("cloud_topic")
    input_fused_map_topic = LaunchConfiguration("input_fused_map_topic")
    input_sdf_map_topic = LaunchConfiguration("input_sdf_map_topic")
    output_fused_map_topic = LaunchConfiguration("output_fused_map_topic")
    enable_rviz = LaunchConfiguration("rviz")
    enable_vehicle_model = LaunchConfiguration("vehicle_model")
    enable_grid_clouds = LaunchConfiguration("grid_clouds")
    rviz_cfg = LaunchConfiguration("rviz_cfg")

    terrain_remaps = [
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
        DeclareLaunchArgument("terrain_backend", default_value="gpu", choices=["gpu", "cpu"]),
        DeclareLaunchArgument("terrain_dynamic_clear_enabled", default_value="true", choices=["true", "false"]),
        DeclareLaunchArgument("terrain_dynamic_clear_frames", default_value="3"),
        DeclareLaunchArgument("enhance", default_value="true", choices=["true", "false"]),
        DeclareLaunchArgument("odom_topic", default_value="/Odometry"),
        DeclareLaunchArgument("cloud_topic", default_value="/cloud_registered"),
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
        DeclareLaunchArgument("fsgp_gpu_backend", default_value="auto", choices=["auto", "cpu", "cupy"]),
        DeclareLaunchArgument("fsgp_max_update_rate_hz", default_value="8.0"),
        DeclareLaunchArgument(
            "rviz",
            default_value="true",
            description="Launch the shared terrain/planning RViz window.",
        ),
        DeclareLaunchArgument(
            "vehicle_model",
            default_value="true",
            description="Show the vehicle pose in the shared RViz window.",
        ),
        DeclareLaunchArgument(
            "grid_clouds",
            default_value="true",
            description="Convert baseline/enhanced SE2Grid layers to RViz point clouds.",
        ),
        DeclareLaunchArgument(
            "rviz_cfg",
            default_value=PathJoinSubstitution([
                FindPackageShare("fsgp_bgk_se2"),
                "rviz",
                "hunter_terrain_navigation.rviz",
            ]),
            description="Shared RViz configuration for terrain and planning.",
        ),
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
        Node(
            package="terrain_analyzer",
            executable="terrain_analyzer",
            name="car_0_terrain_analyzer",
            output="screen",
            parameters=[terrain_cpu_params, {
                "use_sim_time": use_sim_time,
                "elevation.dynamic_clear_enabled": ParameterValue(
                    terrain_dynamic_clear_enabled, value_type=bool),
                "elevation.dynamic_clear_frames": ParameterValue(
                    terrain_dynamic_clear_frames, value_type=int),
            }],
            remappings=terrain_remaps,
            condition=_equals(terrain_backend, "cpu"),
        ),
        Node(
            package="terrain_analyzer",
            executable="terrain_analyzer_gpu",
            name="car_0_terrain_analyzer",
            output="screen",
            parameters=[terrain_gpu_params, {
                "use_sim_time": use_sim_time,
                "elevation.dynamic_clear_enabled": ParameterValue(
                    terrain_dynamic_clear_enabled, value_type=bool),
                "elevation.dynamic_clear_frames": ParameterValue(
                    terrain_dynamic_clear_frames, value_type=int),
            }],
            remappings=terrain_remaps,
            condition=_equals(terrain_backend, "gpu"),
        ),
        Node(
            package="fsgp_bgk_se2",
            executable="fsgp_bgk_se2_node",
            name="fsgp_bgk_se2_node",
            condition=IfCondition(LaunchConfiguration("enhance")),
            output="screen",
            parameters=[
                fsgp_params,
                {
                    "use_sim_time": use_sim_time,
                    "input_fused_map_topic": input_fused_map_topic,
                    "output_fused_map_topic": output_fused_map_topic,
                    "odom_topic": odom_topic,
                    "input_cloud_topic": cloud_topic,
                    "gpu_backend": fsgp_gpu_backend,
                    "max_update_rate_hz": ParameterValue(
                        fsgp_max_update_rate_hz, value_type=float
                    ),
                    "include_diagnostic_layers": False,
                    "publish_debug_clouds": False,
                },
            ],
        ),
        # The terrain analyzer and FSGP-BGK publish SE2Grid messages. These
        # lightweight converters provide the colored PointCloud2 overlays used
        # by RViz without coupling the planner to visualization code.
        Node(
            package="se2_grid_converter",
            executable="se2_grid_to_pointcloud_node",
            name="fsgp_elevation_grid_converter",
            output="screen",
            parameters=[{
                "use_sim_time": use_sim_time,
                "input_topic": output_fused_map_topic,
                "output_topic": "/car_0_terrain_analyzer/elevation_cloud",
                "layer_name": "bgk_mean",
                "valid_layer_name": "valid",
                "valid_threshold": 0.5,
                "height_as_z": True,
                "z_offset": 0.02,
                "height_layer_name": "bgk_mean",
                "drape_on_height": False,
                "publish_surface": True,
                "surface_topic": "/car_0_terrain_analyzer/terrain_surface",
                "surface_max_step": 0.65,
                "surface_alpha": 0.92,
                "surface_risk_layer": "risk",
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
                "layer_name": "risk",
                "valid_layer_name": "valid",
                "valid_threshold": 0.5,
                "follow_odom_yaw": True,
                "odom_topic": odom_topic,
                "height_layer_name": "bgk_mean",
                "drape_on_height": True,
                "height_as_z": False,
                "z_value": 0.035,
                "z_offset": 0.035,
                "use_color_map": True,
                "min_height": 0.0,
                "max_height": 1.0,
                "colormap": "risk_gray_red",
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
            name="rviz_hunter_terrain_nav",
            output="screen",
            arguments=["-d", rviz_cfg],
            parameters=[{"use_sim_time": use_sim_time}],
            condition=IfCondition(enable_rviz),
        ),
    ])
