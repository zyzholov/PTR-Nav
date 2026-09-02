from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    rviz = LaunchConfiguration("rviz")
    rviz_cfg = LaunchConfiguration("rviz_cfg")
    chain_check = LaunchConfiguration("chain_check")

    return LaunchDescription([
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("chain_check", default_value="true"),
        DeclareLaunchArgument(
            "rviz_cfg",
            default_value=PathJoinSubstitution([
                FindPackageShare("fast_livo2_integration"),
                "rviz",
                "hunter_livo2_mapping.rviz",
            ]),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare("fast_livo2_integration"),
                    "launch",
                    "hunter_livo2_mapping.launch.py",
                ])
            ),
            launch_arguments={
                "use_sim_time": "true",
                "params_file": "hunter_isaac_d455_sim.yaml",
                "camera_params_file": "camera_d455_isaac.yaml",
                "lid_topic": "/livox/lidar",
                "imu_topic": "/livox/imu",
                "img_topic": "/camera/camera/color/image_raw",
                "odom_topic": "/Odometry",
                "cloud_topic": "/cloud_registered",
                "rviz": rviz,
                "rviz_cfg": rviz_cfg,
                "chain_check": chain_check,
            }.items(),
        )
    ])
