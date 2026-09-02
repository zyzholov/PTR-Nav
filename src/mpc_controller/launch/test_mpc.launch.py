#!/usr/bin/env python3
"""
test_mpc.launch.py - MPC控制器测试启动文件 (ROS2版本)

此启动文件启动MPC控制器和运动学模拟器进行测试。
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # 声明启动参数
    odom_topic_arg = DeclareLaunchArgument(
        'odom_topic',
        default_value='/car_0/odometry',
        description='Odometry topic name'
    )
    
    cmd_topic_arg = DeclareLaunchArgument(
        'cmd_topic',
        default_value='/car_0/cmd_vel',
        description='Command topic name'
    )
    
    car_id_arg = DeclareLaunchArgument(
        'car_id',
        default_value='0',
        description='Car ID'
    )
    
    init_x_arg = DeclareLaunchArgument(
        'init_x',
        default_value='0',
        description='Initial X position'
    )
    
    init_y_arg = DeclareLaunchArgument(
        'init_y',
        default_value='0',
        description='Initial Y position'
    )
    
    init_yaw_arg = DeclareLaunchArgument(
        'init_yaw',
        default_value='0',
        description='Initial yaw angle'
    )

    # 获取配置文件路径
    mpc_config = PathJoinSubstitution([
        FindPackageShare('mpc_controller'),
        'params',
        'controller_ros2.yaml'
    ])
    
    rviz_config = PathJoinSubstitution([
        FindPackageShare('mpc_controller'),
        'rviz',
        'test.rviz'
    ])

    # 创建运动学模拟器节点
    simulator_node = Node(
        package='fake_ugv',
        executable='kimatic_simulator_node',
        name=['car_', LaunchConfiguration('car_id'), '_ugv_kimatic_simulator'],
        output='screen',
        parameters=[
            {
                'max_height': 0.3,
                'init_x': LaunchConfiguration('init_x'),
                'init_y': LaunchConfiguration('init_y'),
                'init_z': 0.0,
                'init_yaw': LaunchConfiguration('init_yaw'),
            }
        ],
        remappings=[
            ('~command', LaunchConfiguration('cmd_topic')),
            ('~odometry', LaunchConfiguration('odom_topic')),
            ('~marker', [LaunchConfiguration('odom_topic'), '_marker']),
        ]
    )

    # 创建MPC控制器节点
    mpc_node = Node(
        package='mpc_controller',
        executable='mpc_controller_node',
        name='mpc_controller_node',
        output='screen',
        parameters=[mpc_config],
        remappings=[
            ('cmd', LaunchConfiguration('cmd_topic')),
            ('odom', LaunchConfiguration('odom_topic')),
            ('traj', '/planner/trajectory'),
            ('dptraj', '/planner/dptrajectory'),
        ]
    )

    # 创建RViz节点
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz',
        arguments=['-d', rviz_config],
        output='screen'
    )

    return LaunchDescription([
        odom_topic_arg,
        cmd_topic_arg,
        car_id_arg,
        init_x_arg,
        init_y_arg,
        init_yaw_arg,
        simulator_node,
        mpc_node,
        rviz_node,
    ])
