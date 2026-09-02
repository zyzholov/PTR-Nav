import os
import tempfile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import xacro


def _launch_setup(context, *args, **kwargs):
    pkg_share = get_package_share_directory("hunter_se_gazebo")
    xacro_file = os.path.join(pkg_share, "xacro", "hunter_se_gazebo.xacro")
    spawn_script = "/opt/ros/humble/lib/gazebo_ros/spawn_entity.py"
    drive_mode = LaunchConfiguration("drive_mode").perform(context)
    terrain_lidar_profile = LaunchConfiguration("terrain_lidar_profile").perform(context)
    enable_d455 = LaunchConfiguration("enable_d455").perform(context)

    robot_description = xacro.process_file(
        xacro_file,
        mappings={
            "drive_mode": drive_mode,
            "terrain_lidar_profile": terrain_lidar_profile,
            "enable_d455": enable_d455,
        },
    ).toxml()
    urdf_tmp = tempfile.NamedTemporaryFile(mode="w", suffix="_hunter_se.urdf", delete=False)
    urdf_tmp.write(robot_description)
    urdf_tmp.flush()
    urdf_tmp.close()

    return [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="hunter_se_robot_state_publisher",
            output="screen",
            parameters=[{"use_sim_time": True, "robot_description": robot_description}],
        ),
        ExecuteProcess(
            output="screen",
            cmd=[
                "/usr/bin/python3",
                spawn_script,
                "-entity", LaunchConfiguration("robot_name"),
                "-file", urdf_tmp.name,
                "-x", LaunchConfiguration("start_x"),
                "-y", LaunchConfiguration("start_y"),
                "-z", LaunchConfiguration("start_z"),
                "-Y", LaunchConfiguration("start_yaw"),
                "-timeout", "120",
            ],
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("start_x", default_value="0.0"),
        DeclareLaunchArgument("start_y", default_value="0.0"),
        DeclareLaunchArgument("start_z", default_value="0.30"),
        DeclareLaunchArgument("start_yaw", default_value="0.0"),
        DeclareLaunchArgument("robot_name", default_value="hunter_se"),
        DeclareLaunchArgument("drive_mode", default_value="hunter_joint"),
        DeclareLaunchArgument("terrain_lidar_profile", default_value="normal"),
        DeclareLaunchArgument("enable_d455", default_value="true"),
        OpaqueFunction(function=_launch_setup),
    ])
