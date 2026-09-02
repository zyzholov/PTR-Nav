from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("topic", default_value="/hunter_se/cmd_vel"),
        DeclareLaunchArgument("odom_topic", default_value="/hunter_se/wheel_odom"),
        Node(
            package="hunter_se_control",
            executable="keyboard_teleop_ros2.py",
            output="screen",
            arguments=[
                "--topic", LaunchConfiguration("topic"),
                "--odom-topic", LaunchConfiguration("odom_topic"),
            ],
        ),
    ])
