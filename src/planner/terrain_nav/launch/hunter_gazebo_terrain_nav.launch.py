from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    params = LaunchConfiguration("params")
    odom_topic = LaunchConfiguration("odom_topic")
    fused_map_topic = LaunchConfiguration("fused_map_topic")
    cmd_topic = LaunchConfiguration("cmd_topic")
    risk_query_mode = LaunchConfiguration("risk_query_mode")
    se2_risk_layer = LaunchConfiguration("se2_risk_layer")
    hard_risk_layer = LaunchConfiguration("hard_risk_layer")

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("odom_topic", default_value="/Odometry"),
        DeclareLaunchArgument("fused_map_topic", default_value="/car_0_terrain_analyzer/fused_map"),
        DeclareLaunchArgument("cmd_topic", default_value="/hunter_se/cmd_vel"),
        DeclareLaunchArgument("risk_query_mode", default_value="flat_2d"),
        DeclareLaunchArgument("se2_risk_layer", default_value="risk"),
        DeclareLaunchArgument("hard_risk_layer", default_value="baseline_risk"),
        DeclareLaunchArgument(
            "params",
            default_value=PathJoinSubstitution([
                FindPackageShare("terrain_nav"),
                "config",
                "hunter_gazebo.yaml",
            ]),
        ),
        Node(
            package="terrain_nav",
            executable="terrain_nav_node",
            name="terrain_nav_node",
            output="screen",
            parameters=[
                params,
                {
                    "use_sim_time": use_sim_time,
                    "odom_topic": odom_topic,
                    "fused_map_topic": fused_map_topic,
                    "cmd_topic": cmd_topic,
                    "risk_query_mode": risk_query_mode,
                    "se2_risk_layer": se2_risk_layer,
                    "hard_risk_layer": hard_risk_layer,
                },
            ],
        ),
    ])
