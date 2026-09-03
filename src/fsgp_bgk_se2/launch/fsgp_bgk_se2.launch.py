from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    params = LaunchConfiguration("params")
    use_sim_time = LaunchConfiguration("use_sim_time")
    input_fused_map_topic = LaunchConfiguration("input_fused_map_topic")
    output_fused_map_topic = LaunchConfiguration("output_fused_map_topic")
    odom_topic = LaunchConfiguration("odom_topic")
    use_gpu = LaunchConfiguration("use_gpu")

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
                },
            ],
        ),
    ])
