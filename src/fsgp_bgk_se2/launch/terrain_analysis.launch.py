"""Start the terrain front-end and the FSGP-BGK/SEB enhancement only.

Planning/control is intentionally a separate launch boundary.  This makes it
possible to inspect the baseline and enhanced maps without a second node
publishing /cmd_vel.
"""

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
        DeclareLaunchArgument('obstacle_memory_topic', default_value='none'),
        DeclareLaunchArgument('obstacle_memory_max_age_sec', default_value='1.25'),
        DeclareLaunchArgument('obstacle_memory_required', default_value='false', choices=['true', 'false']),
        DeclareLaunchArgument('obstacle_memory_role', default_value='hard', choices=['hard', 'advisory']),
        DeclareLaunchArgument('overhead_filter_enabled', default_value='false', choices=['true', 'false']),
        DeclareLaunchArgument('vehicle_envelope_height_m', default_value='0.65'),
        DeclareLaunchArgument('overhead_clearance_margin_m', default_value='0.10'),
        DeclareLaunchArgument("cmd_topic", default_value="/cmd_vel"),
        DeclareLaunchArgument("navigation_display", default_value="true", choices=["true", "false"]),
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
        DeclareLaunchArgument('fsgp_processing_executor',default_value='thread_pool',
                              choices=['thread_pool','single_ingress']),
        DeclareLaunchArgument("fsgp_max_input_source_age_sec", default_value="0.0"),
        DeclareLaunchArgument("fsgp_max_output_source_age_sec", default_value="0.0"),
        DeclareLaunchArgument("fsgp_output_delivery_reserve_sec", default_value="0.0"),
        DeclareLaunchArgument("prediction_backend", default_value="feature_sgp", choices=["none", "feature_sgp"]),
        DeclareLaunchArgument("gp_device", default_value="auto", choices=["auto", "cpu", "cuda"]),
        DeclareLaunchArgument("gp_refine_observed", default_value="false", choices=["true", "false"]),
        DeclareLaunchArgument("gp_complete_supported", default_value="false", choices=["true", "false"]),
        DeclareLaunchArgument("probability_use_gp_completion", default_value=LaunchConfiguration("gp_complete_supported"), choices=["true", "false"]),
        DeclareLaunchArgument('probability_ground_history_s',default_value='0.0'),
        DeclareLaunchArgument('probability_gp_max_distance_m',default_value='0.35'),
        DeclareLaunchArgument('probability_gp_max_extrapolation_m',default_value='0.0'),
        DeclareLaunchArgument('startup_support_directional',default_value='false'),
        DeclareLaunchArgument('probability_model',default_value='marginal',choices=['marginal','shared_gp']),
        DeclareLaunchArgument('shared_gp_detail_sources',default_value='64'),
        DeclareLaunchArgument('shared_gp_posterior_radius_m',default_value='3.0'),
        DeclareLaunchArgument('shared_gp_basis_spacing_m',default_value='1.0'),
        DeclareLaunchArgument('shared_gp_refinement_radius_m',default_value='0.0'),
        DeclareLaunchArgument('shared_gp_condition_full_residual',default_value='false',choices=['true','false']),
        DeclareLaunchArgument('shared_gp_residual_inducing_points',default_value='0'),
        DeclareLaunchArgument('shared_gp_residual_lengthscale_m',default_value='0.35'),
        DeclareLaunchArgument('shared_gp_independent_model_floor',default_value='false',choices=['true','false']),
        DeclareLaunchArgument('shared_gp_bounded_pending',default_value='false',choices=['true','false']),
        DeclareLaunchArgument("startup_support_mode", default_value="unknown", choices=["unknown", "stable_stance"]),
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
                "elevation.preserve_measured_ground": ParameterValue(
                    PythonExpression(["'", LaunchConfiguration('probability_model'),
                                      "' == 'shared_gp' and '", use_sim_time, "' == 'true'"]),
                    value_type=bool),
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
            # Keep large reliable map transport off the terrain callback thread.
            additional_env={"RMW_FASTRTPS_PUBLICATION_MODE": EnvironmentVariable(
                "RMW_FASTRTPS_PUBLICATION_MODE", default_value="ASYNCHRONOUS"),
                "OPENBLAS_NUM_THREADS": EnvironmentVariable("OPENBLAS_NUM_THREADS", default_value=numeric_threads),
                "OMP_NUM_THREADS": EnvironmentVariable("OMP_NUM_THREADS", default_value=numeric_threads),
                "MKL_NUM_THREADS": EnvironmentVariable("MKL_NUM_THREADS", default_value=numeric_threads)},
            condition=IfCondition(LaunchConfiguration("enhance")),
            output="screen",
            parameters=[
                fsgp_params,
                {
                    'processing_executor':LaunchConfiguration('fsgp_processing_executor'),
                    "use_sim_time": use_sim_time,
                    "input_fused_map_topic": input_fused_map_topic,
                    "output_fused_map_topic": output_fused_map_topic,
                    "odom_topic": odom_topic,
                    "input_cloud_topic": cloud_topic,
                    'obstacle_memory_topic': ParameterValue(LaunchConfiguration('obstacle_memory_topic'), value_type=str),
                    'obstacle_memory_role': LaunchConfiguration('obstacle_memory_role'),
                    'overhead_filter_enabled': ParameterValue(LaunchConfiguration('overhead_filter_enabled'),value_type=bool),
                    'vehicle_envelope_height_m': ParameterValue(LaunchConfiguration('vehicle_envelope_height_m'),value_type=float),
                    'overhead_clearance_margin_m': ParameterValue(LaunchConfiguration('overhead_clearance_margin_m'),value_type=float),
                    'obstacle_memory_max_age_sec': ParameterValue(
                        LaunchConfiguration('obstacle_memory_max_age_sec'), value_type=float),
                    'obstacle_memory_required': ParameterValue(
                        LaunchConfiguration('obstacle_memory_required'), value_type=bool),
                    "gpu_backend": fsgp_gpu_backend,
                    "max_update_rate_hz": ParameterValue(
                        fsgp_max_update_rate_hz, value_type=float
                    ),
                    "max_input_source_age_sec": ParameterValue(
                        LaunchConfiguration("fsgp_max_input_source_age_sec"), value_type=float),
                    "max_output_source_age_sec": ParameterValue(
                        LaunchConfiguration("fsgp_max_output_source_age_sec"), value_type=float),
                    "output_delivery_reserve_sec": ParameterValue(
                        LaunchConfiguration("fsgp_output_delivery_reserve_sec"), value_type=float),
                    "include_diagnostic_layers": False,
                    "publish_debug_clouds": False,
                    "prediction_backend": LaunchConfiguration("prediction_backend"),
                    "gp_device": LaunchConfiguration("gp_device"),
                    "gp_refine_observed": ParameterValue(LaunchConfiguration("gp_refine_observed"), value_type=bool),
                    "gp_complete_supported": ParameterValue(LaunchConfiguration("gp_complete_supported"), value_type=bool),
                    "probability_use_gp_completion": ParameterValue(LaunchConfiguration("probability_use_gp_completion"), value_type=bool),
                    'probability_ground_history_s': ParameterValue(LaunchConfiguration('probability_ground_history_s'),value_type=float),
                    'probability_gp_max_distance_m': ParameterValue(LaunchConfiguration('probability_gp_max_distance_m'),value_type=float),
                    'probability_gp_max_extrapolation_m': ParameterValue(LaunchConfiguration('probability_gp_max_extrapolation_m'),value_type=float),
                    'startup_support_directional': ParameterValue(LaunchConfiguration('startup_support_directional'),value_type=bool),
                    'probability_model': LaunchConfiguration('probability_model'),
                    'shared_gp_detail_sources': ParameterValue(LaunchConfiguration('shared_gp_detail_sources'),value_type=int),
                    'shared_gp_refinement_radius_m': ParameterValue(LaunchConfiguration('shared_gp_refinement_radius_m'),value_type=float),
                    'shared_gp_posterior_radius_m': ParameterValue(LaunchConfiguration('shared_gp_posterior_radius_m'),value_type=float),
                    'shared_gp_basis_spacing_m': ParameterValue(LaunchConfiguration('shared_gp_basis_spacing_m'),value_type=float),
                    'shared_gp_condition_full_residual': ParameterValue(LaunchConfiguration('shared_gp_condition_full_residual'),value_type=bool),
                    'shared_gp_residual_inducing_points': ParameterValue(LaunchConfiguration('shared_gp_residual_inducing_points'),value_type=int),
                    'shared_gp_residual_lengthscale_m': ParameterValue(LaunchConfiguration('shared_gp_residual_lengthscale_m'),value_type=float),
                    'shared_gp_independent_model_floor': ParameterValue(LaunchConfiguration('shared_gp_independent_model_floor'),value_type=bool),
                    'shared_gp_bounded_pending': ParameterValue(LaunchConfiguration('shared_gp_bounded_pending'),value_type=bool),
                    "startup_support_mode": LaunchConfiguration("startup_support_mode"),
                },
            ],
        ),
        # The terrain analyzer and FSGP-BGK publish SE2Grid messages. These
        # lightweight converters provide the colored PointCloud2 overlays used
        # by RViz without coupling the planner to visualization code.
        Node(
            package="fsgp_bgk_se2", executable="navigation_display", name="navigation_display",
            output="screen", condition=IfCondition(LaunchConfiguration("navigation_display")),
            parameters=[{"use_sim_time":use_sim_time,"odom_topic":odom_topic,
                         "grid_topic":output_fused_map_topic,"cmd_topic":LaunchConfiguration("cmd_topic")}],
        ),
        Node(
            package="se2_grid_converter",executable="lidar_display_node",name="lidar_display",
            output="screen",condition=IfCondition(LaunchConfiguration("navigation_display")),
            parameters=[{"use_sim_time":use_sim_time,"odom_topic":odom_topic}],
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
                "height_layer_name": "probability_elevation",
                "drape_on_height": False,
                "publish_surface": True,
                "surface_topic": "/car_0_terrain_analyzer/terrain_surface",
                "height_surface_topic": "/terrain_display/elevation_surface",
                "surface_max_step": 0.25,
                "surface_alpha": 0.86,
                "surface_risk_layer": "failure_probability_upper",
                "surface_body_collision_overlay": False,
                # XY occupied-set boundary only; actual point clouds retain
                # measured vertical extent. Cell cages remain a node option.
                "occupied_display_style": "boundary",
                "body_pose_surface_topic": "/terrain_display/body_pose_risk_surface",
                "surface_probability_colors": True,
                "surface_use_navigation_height": True,
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
                "height_layer_name": "probability_elevation",
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
                "marker_lifetime_sec": 0.0,
                "vehicle.length": 0.840,
                "vehicle.width": 0.626,
                "vehicle.height": 0.338,
                "vehicle.wheel_base": 0.430,
                "vehicle.track_width": 0.548,
                "base_to_chassis_center.x": 0.0,
                "mesh.resource": "package://fake_ugv/meshes/car.dae",
                "mesh.use_embedded_materials": True,
                "mesh.alpha": 1.0,
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
