import os.path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_path = get_package_share_directory("fast_lio")
    default_config_path = os.path.join(package_path, "config", "real")
    default_rviz_config_path = os.path.join(package_path, "rviz", "fastlio.rviz")

    use_sim_time = LaunchConfiguration("use_sim_time")
    config_path = LaunchConfiguration("config_path")
    config_file = LaunchConfiguration("config_file")
    enable_fastlio = LaunchConfiguration("fastlio")
    enable_terrain = LaunchConfiguration("terrain")
    enable_grid_clouds = LaunchConfiguration("grid_clouds")
    rviz_use = LaunchConfiguration("rviz")
    rviz_cfg = LaunchConfiguration("rviz_cfg")
    terrain_params = LaunchConfiguration("terrain_params")
    car_id = LaunchConfiguration("car_id")
    lid_topic = LaunchConfiguration("lid_topic")
    imu_topic = LaunchConfiguration("imu_topic")
    odom_topic = LaunchConfiguration("odom_topic")
    cloud_topic = LaunchConfiguration("cloud_topic")
    map_frame = LaunchConfiguration("map_frame")
    body_frame = LaunchConfiguration("body_frame")
    local_map_radius = LaunchConfiguration("local_map_radius")
    local_map_z_half = LaunchConfiguration("local_map_z_half")

    fused_map_topic = PythonExpression(["'/car_' + '", car_id, "' + '_terrain_analyzer/fused_map'"])
    sdf_map_topic = PythonExpression(["'/car_' + '", car_id, "' + '_terrain_analyzer/sdf_map'"])
    elevation_grid_cloud_topic = PythonExpression(["'/car_' + '", car_id, "' + '_terrain_analyzer/elevation_grid_cloud'"])
    risk_grid_cloud_topic = PythonExpression(["'/car_' + '", car_id, "' + '_terrain_analyzer/risk_grid_cloud'"])
    sdf_grid_cloud_topic = PythonExpression(["'/car_' + '", car_id, "' + '_terrain_analyzer/sdf_grid_cloud'"])

    fast_lio_node = Node(
        package="fast_lio",
        executable="fastlio_mapping",
        name="fastlio_mapping",
        parameters=[
            PathJoinSubstitution([config_path, config_file]),
            {
                "use_sim_time": use_sim_time,
                "common.lid_topic": lid_topic,
                "common.imu_topic": imu_topic,
                "frame.map_frame": map_frame,
                "frame.body_frame": body_frame,
                "publish.local_map_radius": ParameterValue(local_map_radius, value_type=float),
                "publish.local_map_z_half": ParameterValue(local_map_z_half, value_type=float),
            },
        ],
        output="screen",
        condition=IfCondition(enable_fastlio),
    )

    terrain_node = Node(
        package="terrain_analyzer",
        executable="terrain_analyzer",
        name=["car_", car_id, "_terrain_analyzer"],
        output="screen",
        parameters=[terrain_params, {"use_sim_time": use_sim_time}],
        remappings=[
            ("odom", odom_topic),
            ("cloud", cloud_topic),
            ("fused_map", ["/car_", car_id, "_terrain_analyzer/fused_map"]),
            ("sdf_map", ["/car_", car_id, "_terrain_analyzer/sdf_map"]),
            ("elevation_cloud", ["/car_", car_id, "_terrain_analyzer/elevation_cloud"]),
            ("sdf_cloud", ["/car_", car_id, "_terrain_analyzer/sdf_cloud"]),
            ("risk_cloud", ["/car_", car_id, "_terrain_analyzer/risk_cloud"]),
            ("obstacle_cloud", ["/car_", car_id, "_terrain_analyzer/obstacle_cloud"]),
            ("overhead_cloud", ["/car_", car_id, "_terrain_analyzer/overhead_cloud"]),
        ],
        condition=IfCondition(enable_terrain),
    )

    elevation_grid_converter = Node(
        package="se2_grid_converter",
        executable="se2_grid_to_pointcloud_node",
        name="real_terrain_elevation_grid_converter",
        output="screen",
        parameters=[{
            "use_sim_time": use_sim_time,
            "input_topic": fused_map_topic,
            "output_topic": elevation_grid_cloud_topic,
            "layer_name": "elevation",
            "valid_layer_name": "valid",
            "valid_threshold": 0.5,
            "height_as_z": True,
            "z_offset": 0.02,
            "use_color_map": True,
            "min_height": -1.0,
            "max_height": 2.0,
        }],
        condition=IfCondition(enable_grid_clouds),
    )

    risk_grid_converter = Node(
        package="se2_grid_converter",
        executable="se2_grid_to_pointcloud_node",
        name="real_terrain_risk_grid_converter",
        output="screen",
        parameters=[{
            "use_sim_time": use_sim_time,
            "input_topic": fused_map_topic,
            "output_topic": risk_grid_cloud_topic,
            "layer_name": "risk",
            "valid_layer_name": "valid",
            "valid_threshold": 0.5,
            "yaw_index": 0,
            "height_as_z": False,
            "z_value": 0.08,
            "use_color_map": True,
            "min_height": 0.0,
            "max_height": 1.0,
            "colormap": "risk_gray_red",
        }],
        condition=IfCondition(enable_grid_clouds),
    )

    sdf_grid_converter = Node(
        package="se2_grid_converter",
        executable="se2_grid_to_pointcloud_node",
        name="real_terrain_sdf_grid_converter",
        output="screen",
        parameters=[{
            "use_sim_time": use_sim_time,
            "input_topic": sdf_map_topic,
            "output_topic": sdf_grid_cloud_topic,
            "layer_name": "sdf_cpu",
            "valid_layer_name": "valid",
            "valid_threshold": 0.5,
            "invert_color_map": True,
            "height_as_z": False,
            "z_value": 0.04,
            "use_color_map": True,
            "min_height": 0.0,
            "max_height": 6.0,
            "colormap": "terrain",
        }],
        condition=IfCondition(enable_grid_clouds),
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz_neos_fastlio_mapping",
        parameters=[{"use_sim_time": use_sim_time}],
        arguments=["-d", rviz_cfg],
        condition=IfCondition(rviz_use),
    )

    return LaunchDescription([
        SetEnvironmentVariable("RCUTILS_LOGGING_USE_STDOUT", "1"),
        SetEnvironmentVariable("RCUTILS_COLORIZED_OUTPUT", "1"),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("fastlio", default_value="true"),
        DeclareLaunchArgument("terrain", default_value="true"),
        DeclareLaunchArgument("grid_clouds", default_value="false"),
        DeclareLaunchArgument("car_id", default_value="0"),
        DeclareLaunchArgument("config_path", default_value=default_config_path),
        DeclareLaunchArgument("config_file", default_value="neos_mid360.yaml"),
        DeclareLaunchArgument("lid_topic", default_value="/livox/lidar"),
        DeclareLaunchArgument("imu_topic", default_value="/imu_raw"),
        DeclareLaunchArgument("odom_topic", default_value="/Odometry"),
        DeclareLaunchArgument("cloud_topic", default_value="/cloud_registered"),
        DeclareLaunchArgument("map_frame", default_value="map"),
        DeclareLaunchArgument("body_frame", default_value="base_link"),
        DeclareLaunchArgument("local_map_radius", default_value="35.0"),
        DeclareLaunchArgument("local_map_z_half", default_value="5.0"),
        DeclareLaunchArgument(
            "terrain_params",
            default_value=PathJoinSubstitution([
                FindPackageShare("terrain_analyzer"),
                "params",
                "real",
                "neos_registered_live.yaml",
            ]),
        ),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("rviz_cfg", default_value=default_rviz_config_path),
        fast_lio_node,
        terrain_node,
        elevation_grid_converter,
        risk_grid_converter,
        sdf_grid_converter,
        rviz_node,
    ])
