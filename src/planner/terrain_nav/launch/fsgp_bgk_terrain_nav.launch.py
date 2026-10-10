"""Start only the planner/controller on the enhanced FSGP-BGK/SEB map."""

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
    se2_risk_lethal = LaunchConfiguration("se2_risk_lethal")
    live_obstacle_enable = LaunchConfiguration("live_obstacle_enable")

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("odom_topic", default_value="/Odometry"),
        DeclareLaunchArgument(
            "fused_map_topic",
            default_value="/car_0_fsgp_bgk/fused_map",
        ),
        DeclareLaunchArgument("cmd_topic", default_value="/cmd_vel"),
        DeclareLaunchArgument("risk_query_mode", default_value="se2_yaw"),
        DeclareLaunchArgument("se2_risk_layer", default_value="risk"),
        DeclareLaunchArgument("hard_risk_layer", default_value="baseline_risk"),
        DeclareLaunchArgument("se2_risk_lethal", default_value="false"),
        DeclareLaunchArgument("live_obstacle_enable", default_value="true"),
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
                    "se2_risk_lethal": se2_risk_lethal,
                    "live_obstacle_enable": live_obstacle_enable,
                },
            ],
        ),
    ])
