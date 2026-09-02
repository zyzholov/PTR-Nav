from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    enable_livo2 = LaunchConfiguration("livo2")
    enable_rviz = LaunchConfiguration("rviz")
    enable_chain_check = LaunchConfiguration("chain_check")
    publish_map_tf = LaunchConfiguration("publish_map_to_camera_init_tf")

    config_path = LaunchConfiguration("config_path")
    params_file = LaunchConfiguration("params_file")
    camera_params_file = LaunchConfiguration("camera_params_file")
    rviz_cfg = LaunchConfiguration("rviz_cfg")

    lid_topic = LaunchConfiguration("lid_topic")
    imu_topic = LaunchConfiguration("imu_topic")
    img_topic = LaunchConfiguration("img_topic")
    odom_topic = LaunchConfiguration("odom_topic")
    cloud_topic = LaunchConfiguration("cloud_topic")

    fast_livo_node = Node(
        package="fast_livo",
        executable="fastlivo_mapping",
        name="laserMapping",
        output="screen",
        parameters=[
            PathJoinSubstitution([config_path, params_file]),
            PathJoinSubstitution([config_path, camera_params_file]),
            {
                "use_sim_time": use_sim_time,
                "common.lid_topic": lid_topic,
                "common.imu_topic": imu_topic,
                "common.img_topic": img_topic,
            },
        ],
        remappings=[
            ("/aft_mapped_to_init", odom_topic),
            ("/cloud_registered", cloud_topic),
        ],
        condition=IfCondition(enable_livo2),
    )

    map_to_camera_init_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="map_to_livo2_camera_init_tf",
        arguments=[
            "--x", "0", "--y", "0", "--z", "0",
            "--roll", "0", "--pitch", "0", "--yaw", "0",
            "--frame-id", "map",
            "--child-frame-id", "camera_init",
        ],
        condition=IfCondition(publish_map_tf),
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz_hunter_livo2_mapping",
        output="screen",
        arguments=["-d", rviz_cfg],
        parameters=[{"use_sim_time": use_sim_time}],
        condition=IfCondition(enable_rviz),
    )

    chain_check_node = Node(
        package="fast_livo2_integration",
        executable="livo2_chain_check.py",
        name="livo2_chain_check",
        output="screen",
        parameters=[{
            "use_sim_time": use_sim_time,
            "required_topics": [
                "/livox/lidar",
                "/livox/imu",
                "/camera/camera/color/image_raw",
                "/Odometry",
                "/cloud_registered",
            ],
            "optional_topics": [
                "/path",
                "/rgb_img",
                "/camera/camera/depth/image_rect_raw",
                "/car_0_terrain_analyzer/fused_map",
                "/car_0_terrain_analyzer/sdf_map",
            ],
            "tf_checks": [
                "map,camera_init",
                "map,aft_mapped",
            ],
            "report_period_sec": 2.0,
            "warn_after_sec": 8.0,
        }],
        condition=IfCondition(enable_chain_check),
    )

    return LaunchDescription([
        SetEnvironmentVariable("RCUTILS_LOGGING_USE_STDOUT", "1"),
        SetEnvironmentVariable("RCUTILS_COLORIZED_OUTPUT", "1"),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("livo2", default_value="true"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("chain_check", default_value="true"),
        DeclareLaunchArgument("publish_map_to_camera_init_tf", default_value="true"),
        DeclareLaunchArgument(
            "config_path",
            default_value=PathJoinSubstitution([
                FindPackageShare("fast_livo"),
                "config",
            ]),
        ),
        DeclareLaunchArgument("params_file", default_value="hunter_mid360_d455_sim.yaml"),
        DeclareLaunchArgument("camera_params_file", default_value="camera_d455_sim.yaml"),
        DeclareLaunchArgument("lid_topic", default_value="/livox/lidar"),
        DeclareLaunchArgument("imu_topic", default_value="/livox/imu"),
        DeclareLaunchArgument("img_topic", default_value="/camera/camera/color/image_raw"),
        DeclareLaunchArgument("odom_topic", default_value="/Odometry"),
        DeclareLaunchArgument("cloud_topic", default_value="/cloud_registered"),
        DeclareLaunchArgument(
            "rviz_cfg",
            default_value=PathJoinSubstitution([
                FindPackageShare("fast_livo2_integration"),
                "rviz",
                "hunter_livo2_mapping.rviz",
            ]),
        ),
        fast_livo_node,
        map_to_camera_init_tf,
        chain_check_node,
        rviz_node,
    ])
