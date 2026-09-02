from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    enable_rviz = LaunchConfiguration("rviz")
    enable_grid_clouds = LaunchConfiguration("grid_clouds")
    terrain_params = LaunchConfiguration("terrain_params")
    odom_topic = LaunchConfiguration("odom_topic")
    cloud_topic = LaunchConfiguration("cloud_topic")

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("grid_clouds", default_value="false"),
        DeclareLaunchArgument(
            "terrain_params",
            default_value=PathJoinSubstitution([
                FindPackageShare("terrain_analyzer"),
                "params",
                "hunter_fastlio_registered_live.yaml",
            ]),
        ),
        DeclareLaunchArgument("odom_topic", default_value="/Odometry"),
        DeclareLaunchArgument("cloud_topic", default_value="/cloud_registered"),
        Node(
            package="terrain_analyzer",
            executable="terrain_analyzer",
            name="car_0_terrain_analyzer",
            output="screen",
            parameters=[terrain_params, {"use_sim_time": use_sim_time}],
            remappings=[
                ("odom", odom_topic),
                ("cloud", cloud_topic),
                ("fused_map", "/car_0_terrain_analyzer/fused_map"),
                ("sdf_map", "/car_0_terrain_analyzer/sdf_map"),
                ("elevation_cloud", "/car_0_terrain_analyzer/elevation_cloud"),
                ("sdf_cloud", "/car_0_terrain_analyzer/sdf_cloud"),
                ("risk_cloud", "/car_0_terrain_analyzer/risk_cloud"),
                ("obstacle_cloud", "/car_0_terrain_analyzer/obstacle_cloud"),
            ],
        ),
        Node(
            package="se2_grid_converter",
            executable="se2_grid_to_pointcloud_node",
            name="terrain_elevation_grid_converter",
            output="screen",
            parameters=[{
                "use_sim_time": use_sim_time,
                "input_topic": "/car_0_terrain_analyzer/fused_map",
                "output_topic": "/car_0_terrain_analyzer/elevation_grid_cloud",
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
        ),
        Node(
            package="se2_grid_converter",
            executable="se2_grid_to_pointcloud_node",
            name="terrain_sdf_grid_converter",
            output="screen",
            parameters=[{
                "use_sim_time": use_sim_time,
                "input_topic": "/car_0_terrain_analyzer/sdf_map",
                "output_topic": "/car_0_terrain_analyzer/sdf_grid_cloud",
                "layer_name": "sdf_cpu",
                "valid_layer_name": "valid",
                "valid_threshold": 0.5,
                "invert_color_map": True,
                "height_as_z": False,
                "z_value": 0.04,
                "use_color_map": True,
                "min_height": 0.0,
                "max_height": 6.0,
            }],
            condition=IfCondition(enable_grid_clouds),
        ),
        Node(
            package="se2_grid_converter",
            executable="se2_grid_to_pointcloud_node",
            name="terrain_risk_grid_converter",
            output="screen",
            parameters=[{
                "use_sim_time": use_sim_time,
                "input_topic": "/car_0_terrain_analyzer/fused_map",
                "output_topic": "/car_0_terrain_analyzer/risk_grid_cloud",
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
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz_hunter_fastlio_terrain",
            output="screen",
            arguments=[
                "-d",
                PathJoinSubstitution([
                    FindPackageShare("plan_manage"),
                    "launch",
                    "gazebo_fastlio.rviz",
                ]),
            ],
            parameters=[{"use_sim_time": use_sim_time}],
            condition=IfCondition(enable_rviz),
        ),
    ])
