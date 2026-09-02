import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory("fast_livo")
    config_dir = os.path.join(package_share, "config")
    rviz_cfg = os.path.join(package_share, "rviz_cfg", "fast_livo2.rviz")

    params_file = LaunchConfiguration("params_file")
    camera_params_file = LaunchConfiguration("camera_params_file")
    use_sim_time = LaunchConfiguration("use_sim_time")
    use_rviz = LaunchConfiguration("rviz")

    return LaunchDescription([
        DeclareLaunchArgument(
            "params_file",
            default_value=os.path.join(config_dir, "hunter_mid360_d455_sim.yaml"),
        ),
        DeclareLaunchArgument(
            "camera_params_file",
            default_value=os.path.join(config_dir, "camera_d455_sim.yaml"),
        ),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("rviz", default_value="false"),
        Node(
            package="fast_livo",
            executable="fastlivo_mapping",
            name="laserMapping",
            parameters=[
                params_file,
                camera_params_file,
                {"use_sim_time": use_sim_time},
            ],
            output="screen",
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz_fast_livo2",
            arguments=["-d", rviz_cfg],
            output="screen",
            parameters=[{"use_sim_time": use_sim_time}],
            condition=IfCondition(use_rviz),
        ),
    ])
