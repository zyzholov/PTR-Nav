from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    params = LaunchConfiguration("params")
    use_sim_time = LaunchConfiguration("use_sim_time")
    input_fused_map_topic = LaunchConfiguration("input_fused_map_topic")
    output_fused_map_topic = LaunchConfiguration("output_fused_map_topic")
    odom_topic = LaunchConfiguration("odom_topic")
    use_gpu = LaunchConfiguration("use_gpu")
    gpu_backend = LaunchConfiguration("gpu_backend")
    auto_gpu_min_work_items = LaunchConfiguration("auto_gpu_min_work_items")
    yaw_bins = LaunchConfiguration("yaw_bins")
    max_update_rate_hz = LaunchConfiguration("max_update_rate_hz")
    include_diagnostic_layers = LaunchConfiguration("include_diagnostic_layers")
    publish_debug_clouds = LaunchConfiguration("publish_debug_clouds")

    return LaunchDescription([
        DeclareLaunchArgument(
            "params",
            default_value=PathJoinSubstitution([
                FindPackageShare("fsgp_bgk_se2"),
                "config",
                "hunter_fastlio.yaml",
            ]),
        ),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument(
            "input_fused_map_topic",
            default_value="/car_0_terrain_analyzer/fused_map",
        ),
        DeclareLaunchArgument(
            "output_fused_map_topic",
            default_value="/car_0_fsgp_bgk/fused_map",
        ),
        DeclareLaunchArgument("odom_topic", default_value="/Odometry"),
        DeclareLaunchArgument("use_gpu", default_value="true"),
        DeclareLaunchArgument("gpu_backend", default_value="auto", choices=["auto", "cpu", "cupy"]),
        DeclareLaunchArgument("auto_gpu_min_work_items", default_value="100000"),
        DeclareLaunchArgument("yaw_bins", default_value="36"),
        DeclareLaunchArgument("max_update_rate_hz", default_value="8.0"),
        DeclareLaunchArgument("include_diagnostic_layers", default_value="false"),
        DeclareLaunchArgument("publish_debug_clouds", default_value="false"),
        Node(
            package="fsgp_bgk_se2",
            executable="fsgp_bgk_se2_node",
            name="fsgp_bgk_se2_node",
            output="screen",
            parameters=[
                params,
                {
                    "use_sim_time": use_sim_time,
                    "input_fused_map_topic": input_fused_map_topic,
                    "output_fused_map_topic": output_fused_map_topic,
                    "odom_topic": odom_topic,
                    "use_gpu": use_gpu,
                    "gpu_backend": gpu_backend,
                    "auto_gpu_min_work_items": ParameterValue(
                        auto_gpu_min_work_items, value_type=int
                    ),
                    "yaw_bins": yaw_bins,
                    "max_update_rate_hz": ParameterValue(
                        max_update_rate_hz, value_type=float
                    ),
                    "include_diagnostic_layers": include_diagnostic_layers,
                    "publish_debug_clouds": publish_debug_clouds,
                },
            ],
        ),
    ])
