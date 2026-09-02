import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    hunter_share = get_package_share_directory("hunter_se_gazebo")

    return LaunchDescription([
        DeclareLaunchArgument("drive_mode", default_value="hunter_joint"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(hunter_share, "launch", "hunter_se_empty_world.launch.py")
            ),
            launch_arguments={
                "drive_mode": LaunchConfiguration("drive_mode"),
            }.items(),
        )
    ])
