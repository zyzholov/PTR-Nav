import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    package_path = get_package_share_directory("fast_lio_rebuild")
    default_config_path = os.path.join(package_path, "config")
    default_rviz_config = os.path.join(package_path, "rviz", "fastlio_rebuild_compare.rviz")

    use_sim_time = LaunchConfiguration("use_sim_time")
    config_path = LaunchConfiguration("config_path")
    config_file = LaunchConfiguration("config_file")
    rviz = LaunchConfiguration("rviz")
    rviz_cfg = LaunchConfiguration("rviz_cfg")
    fastlio = LaunchConfiguration("fastlio")
    guide = LaunchConfiguration("guide")

    lid_topic = LaunchConfiguration("lid_topic")
    imu_topic = LaunchConfiguration("imu_topic")
    map_frame = LaunchConfiguration("map_frame")
    body_frame = LaunchConfiguration("body_frame")

    new_odom_topic = LaunchConfiguration("new_odom_topic")
    new_path_topic = LaunchConfiguration("new_path_topic")
    new_cloud_topic = LaunchConfiguration("new_cloud_topic")
    new_cloud_body_topic = LaunchConfiguration("new_cloud_body_topic")
    new_cloud_effected_topic = LaunchConfiguration("new_cloud_effected_topic")
    new_laser_map_topic = LaunchConfiguration("new_laser_map_topic")
    new_local_map_topic = LaunchConfiguration("new_local_map_topic")

    reference_bag_odom_topic = LaunchConfiguration("reference_bag_odom_topic")
    reference_odom_topic = LaunchConfiguration("reference_odom_topic")

    raw_bag_path = LaunchConfiguration("raw_bag_path")
    reference_bag_path = LaunchConfiguration("reference_bag_path")
    play_raw_bag = LaunchConfiguration("play_raw_bag")
    play_reference_bag = LaunchConfiguration("play_reference_bag")
    play_rate = LaunchConfiguration("play_rate")
    raw_bag_delay = LaunchConfiguration("raw_bag_delay")
    reference_bag_delay = LaunchConfiguration("reference_bag_delay")
    raw_start_offset = LaunchConfiguration("raw_start_offset")
    reference_start_offset = LaunchConfiguration("reference_start_offset")
    clock_hz = LaunchConfiguration("clock_hz")

    reference_lead_time = LaunchConfiguration("reference_lead_time")
    reference_window_duration = LaunchConfiguration("reference_window_duration")
    reference_max_points = LaunchConfiguration("reference_max_points")
    guide_path_topic = LaunchConfiguration("guide_path_topic")
    guide_marker_topic = LaunchConfiguration("guide_marker_topic")

    raw_bag_player = ExecuteProcess(
        cmd=[
            "ros2",
            "bag",
            "play",
            raw_bag_path,
            "--rate",
            play_rate,
            "--clock",
            clock_hz,
            "--delay",
            raw_bag_delay,
            "--start-offset",
            raw_start_offset,
            "--topics",
            lid_topic,
            imu_topic,
        ],
        output="screen",
        condition=IfCondition(play_raw_bag),
    )

    reference_bag_player = Node(
        package="fast_lio_rebuild",
        executable="odom_bag_republisher_node",
        name="reference_odom_bag_republisher",
        output="screen",
        parameters=[{
            "bag_path": reference_bag_path,
            "input_topic": reference_bag_odom_topic,
            "output_topic": reference_odom_topic,
            "play_rate": ParameterValue(play_rate, value_type=float),
            "start_delay": ParameterValue(reference_bag_delay, value_type=float),
            "start_offset": ParameterValue(reference_start_offset, value_type=float),
        }],
        condition=IfCondition(play_reference_bag),
    )

    fast_lio_node = Node(
        package="fast_lio",
        executable="fastlio_mapping",
        name="fastlio_mapping_rebuild",
        output="screen",
        parameters=[
            PathJoinSubstitution([config_path, config_file]),
            {
                "use_sim_time": use_sim_time,
                "common.lid_topic": lid_topic,
                "common.imu_topic": imu_topic,
                "frame.map_frame": map_frame,
                "frame.body_frame": body_frame,
            },
        ],
        remappings=[
            ("/Odometry", new_odom_topic),
            ("/path", new_path_topic),
            ("/cloud_registered", new_cloud_topic),
            ("/cloud_registered_body", new_cloud_body_topic),
            ("/cloud_effected", new_cloud_effected_topic),
            ("/Laser_map", new_laser_map_topic),
            ("/cloud_local_map", new_local_map_topic),
        ],
        condition=IfCondition(fastlio),
    )

    reference_window_node = Node(
        package="fast_lio_rebuild",
        executable="reference_window_node",
        name="fastlio_reference_window",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "new_odom_topic": new_odom_topic,
                "reference_odom_topic": reference_odom_topic,
                "path_topic": guide_path_topic,
                "marker_topic": guide_marker_topic,
                "map_frame": map_frame,
                "reference_lead_time": ParameterValue(reference_lead_time, value_type=float),
                "window_duration": ParameterValue(reference_window_duration, value_type=float),
                "max_points": ParameterValue(reference_max_points, value_type=int),
                "line_width": 0.22,
                "point_scale": 0.32,
                "z_offset": 0.16,
                "pre_simplify_tolerance": 0.08,
                "smooth_iterations": 5,
                "smooth_spacing": 0.04,
                "min_smooth_points": 108,
                "max_smooth_points": 340,
                "post_smooth_passes": 6,
                "post_smooth_strength": 0.62,
            }
        ],
        condition=IfCondition(guide),
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz_fastlio_rebuild_compare",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
        arguments=["-d", rviz_cfg],
        condition=IfCondition(rviz),
    )

    return LaunchDescription(
        [
            SetEnvironmentVariable("RCUTILS_LOGGING_USE_STDOUT", "1"),
            SetEnvironmentVariable("RCUTILS_COLORIZED_OUTPUT", "1"),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("fastlio", default_value="true"),
            DeclareLaunchArgument("guide", default_value="true"),
            DeclareLaunchArgument("rviz", default_value="true"),
            DeclareLaunchArgument("rviz_cfg", default_value=default_rviz_config),
            DeclareLaunchArgument("config_path", default_value=default_config_path),
            DeclareLaunchArgument("config_file", default_value="neos_mid360_bag_rebuild.yaml"),
            DeclareLaunchArgument("lid_topic", default_value="/livox/lidar"),
            DeclareLaunchArgument("imu_topic", default_value="/imu/data_raw"),
            DeclareLaunchArgument("map_frame", default_value="map"),
            DeclareLaunchArgument("body_frame", default_value="base_link_fastlio_rebuild"),
            DeclareLaunchArgument("new_odom_topic", default_value="/fastlio_rebuild/Odometry"),
            DeclareLaunchArgument("new_path_topic", default_value="/fastlio_rebuild/path"),
            DeclareLaunchArgument("new_cloud_topic", default_value="/fastlio_rebuild/cloud_registered"),
            DeclareLaunchArgument("new_cloud_body_topic", default_value="/fastlio_rebuild/cloud_registered_body"),
            DeclareLaunchArgument("new_cloud_effected_topic", default_value="/fastlio_rebuild/cloud_effected"),
            DeclareLaunchArgument("new_laser_map_topic", default_value="/fastlio_rebuild/Laser_map"),
            DeclareLaunchArgument("new_local_map_topic", default_value="/fastlio_rebuild/cloud_local_map"),
            DeclareLaunchArgument("reference_bag_odom_topic", default_value="/Odometry"),
            DeclareLaunchArgument("reference_odom_topic", default_value="/fastlio_reference/Odometry"),
            DeclareLaunchArgument(
                "raw_bag_path",
                default_value="/home/z/workspace_zyz/bags/rosbag2_2026_07_07-19_37_13",
            ),
            DeclareLaunchArgument("reference_bag_path", default_value="/home/z/workspace_zyz/bags/rosbag2_2026_07_07-21_53_43"),
            DeclareLaunchArgument("play_raw_bag", default_value="true"),
            DeclareLaunchArgument("play_reference_bag", default_value="true"),
            DeclareLaunchArgument("play_rate", default_value="1.0"),
            DeclareLaunchArgument("raw_bag_delay", default_value="0.60"),
            DeclareLaunchArgument("reference_bag_delay", default_value="0.01"),
            DeclareLaunchArgument("raw_start_offset", default_value="0.0"),
            DeclareLaunchArgument("reference_start_offset", default_value="0.0"),
            DeclareLaunchArgument("clock_hz", default_value="100.0"),
            DeclareLaunchArgument("reference_lead_time", default_value="0.45"),
            DeclareLaunchArgument("reference_window_duration", default_value="3.0"),
            DeclareLaunchArgument("reference_max_points", default_value="80"),
            DeclareLaunchArgument("guide_path_topic", default_value="/fastlio_rebuild/reference_window_path"),
            DeclareLaunchArgument("guide_marker_topic", default_value="/fastlio_rebuild/reference_window_marker"),
            raw_bag_player,
            reference_bag_player,
            fast_lio_node,
            reference_window_node,
            rviz_node,
        ]
    )
