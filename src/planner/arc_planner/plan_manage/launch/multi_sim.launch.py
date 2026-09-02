import subprocess
import time
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _cleanup_previous_stack(_context):
    # Prevent duplicate stacks publishing the same topics (odom/map/traj), which causes jumping and flicker.
    patterns = [
        "kimatic_simulator_node",
        "car_0_ugv_kimatic_simulator",
        "fake_ugv/lib/fake_ugv/kimatic_simulator_node",
        "ugv_kinematic_model_node",
        "mpc_controller_node",
        "plan_node",
        "terrain_analyzer",
        "laser_sim_node",
        "terrain_loader",
    ]
    for pattern in patterns:
        subprocess.run(
            ["pkill", "-f", pattern],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    time.sleep(0.6)
    return []


def generate_launch_description():
    # launch arguments (defaults taken from original XML)
    cars_num = LaunchConfiguration('cars_num', default='1')
    car_id = LaunchConfiguration('car_id', default='0')
    car_width = LaunchConfiguration('car_width', default='0.5')
    car_length = LaunchConfiguration('car_length', default='0.5')

    init_x = LaunchConfiguration('init_x', default='3.05')
    init_y = LaunchConfiguration('init_y', default='0.5')
    init_z = LaunchConfiguration('init_z', default='0.0')
    init_yaw = LaunchConfiguration('init_yaw', default='0.0')

    target_x = LaunchConfiguration('target_x', default='32.0')
    target_y = LaunchConfiguration('target_y', default='0.0')
    target_yaw = LaunchConfiguration('target_yaw', default='0.0')

    odom_topic = LaunchConfiguration('odom_topic', default='/car_0/odometry')
    lidar_topic = LaunchConfiguration('lidar_topic', default='/car_0_laser_simulator/car_0_local_pointcloud')

    cmd_topic = LaunchConfiguration('cmd_topic', default='/car_0/cmd_vel')
    traj_topic = LaunchConfiguration('traj_topic', default='/car_0/minco_traj')

    max_vel = LaunchConfiguration('max_vel', default='2.0')
    max_acc = LaunchConfiguration('max_acc', default='0.5')
    max_cur = LaunchConfiguration('max_cur', default='0')
    rmw_implementation = LaunchConfiguration('rmw_implementation', default='rmw_fastrtps_cpp')
    fastdds_builtin_transports = LaunchConfiguration('fastdds_builtin_transports', default='UDPv4')
    enable_rviz = LaunchConfiguration('enable_rviz', default='true')
    auto_cleanup_old_nodes = LaunchConfiguration('auto_cleanup_old_nodes', default='true')

    ld = LaunchDescription()

    # Declare args (so they show up in --ros-args and are configurable)
    for name, default in [
        ('cars_num', '1'), ('car_id', '0'), ('car_width', '0.5'), ('car_length', '0.5'),
        ('init_x', '3.05'), ('init_y', '0.5'), ('init_z', '0.0'), ('init_yaw', '0.0'),
        ('target_x', '32.0'), ('target_y', '0.0'), ('target_yaw', '0.0'),
        ('odom_topic', '/car_0/odometry'), ('lidar_topic', '/car_0_laser_simulator/car_0_local_pointcloud'),
        ('cmd_topic', '/car_0/cmd_vel'), ('traj_topic', '/car_0/minco_traj'),
        ('max_vel', '2.0'), ('max_acc', '2.0'), ('max_cur', '0'),
        ('rmw_implementation', 'rmw_fastrtps_cpp'),
        ('fastdds_builtin_transports', 'UDPv4'),
        ('enable_rviz', 'true'),
        ('auto_cleanup_old_nodes', 'true')
    ]:
        ld.add_action(DeclareLaunchArgument(name, default_value=default))

    ld.add_action(SetEnvironmentVariable(name='RMW_IMPLEMENTATION', value=rmw_implementation))
    ld.add_action(SetEnvironmentVariable(name='FASTDDS_BUILTIN_TRANSPORTS', value=fastdds_builtin_transports))

    ld.add_action(OpaqueFunction(
        function=_cleanup_previous_stack,
        condition=IfCondition(auto_cleanup_old_nodes)
    ))

    ld.add_action(LogInfo(msg=['Launching adapted ROS2 launch (some nodes may be missing if not migrated):\\n',
                               'mpc_controller_node, kimatic_simulator_node, laser_sim_node (if available), plan_manage (if migrated)']))

    # rviz2 - with ROS2 config file (updated from ROS1 format)
    rviz_cfg = PathJoinSubstitution([FindPackageShare('plan_manage'), 'launch', 'multi_sim.rviz'])
    ld.add_action(Node(
        package='rviz2',
executable='rviz2',
        name='rviz',
        arguments=['-d', rviz_cfg],
        output='screen',
        condition=IfCondition(enable_rviz)
    ))
    
    # kimatic simulator (workspace package name is 'fake_ugv')
    ld.add_action(Node(
        package='fake_ugv',
        executable='kimatic_simulator_node',
        name=['car_', car_id, '_ugv_kimatic_simulator'],
        output='screen',
        parameters=[{
            'init_x': init_x,
            'init_y': init_y,
            'init_z': init_z,
            'init_yaw': init_yaw,
            'max_height': 0.05,
            # Headless runs usually don't publish /initialpose; keep simulator movable by default.
            'require_initialpose': False,
            'allow_runtime_reinit': True,
            'max_position_step': 0.10,
            'max_yaw_step': 0.15,
            'max_cmd_speed': 2.5,
            'min_cmd_speed': -1.0,
            'max_cmd_accel': 2.5,
            'max_cmd_steer': 0.48,
            'max_cmd_steer_rate': 0.8,
            'wheel_base': 0.60,
            'goal_stop_radius': 0.02,
            'goal_release_radius': 0.80,
            'goal_abort_radius': 8.0,
            'goal_no_progress_sec': 2.5,
            'goal_progress_eps': 0.03,
            'enable_goal_watchdog': False,
            'cmd_timeout_sec': 0.25,
            'map_bound_margin': 0.30,
            'att_filter_alpha': 0.20,
            'map_name': 'forest.pcd'
        }],
        remappings=[
            ('cmd_vel', cmd_topic), 
            ('odometry', odom_topic),
            ('marker', '/car_0/odometry_marker'),
            ('traj_has_height', '/car_0/traj_has_height'),
            ('initialpose', '/initialpose')  # Ensure we subscribe to global /initialpose from RViz
        ]
    ))

    # mpc controller
    ld.add_action(Node(
        package='mpc_controller',
        executable='mpc_controller_node',
        name=['car_', car_id, '_mpc_controller_node'],
        output='screen',
        parameters=[
            PathJoinSubstitution([FindPackageShare('mpc_controller'), 'params', 'controller.yaml']),
            {
                'mpc/dt': 0.1,
                'mpc/predict_steps': 10,
                'mpc/use_nmpc_solver': False,
                'mpc/max_speed': 2.50,
                'mpc/min_speed': -0.80,
                'mpc/reverse_speed_cap': 0.80,
                'mpc/max_accel': 2.00,
                'mpc/cmd_speed_scale': 1.0,
                'mpc/max_steer': 0.55,
                'mpc/max_dsteer': 0.80,
                'mpc/wheel_base': 0.70,
                'mpc/replan_cautious_sec': 0.6,
                'mpc/replan_cautious_speed_cap': 0.22,
                'mpc/turn_speed_steer_gain': 0.55,
                'mpc/turn_speed_min_cap': 0.12,
                'mpc/cmd_is_yaw_rate': False,
                'mpc/force_reverse_follow': False,
                'mpc/test_mpc': False,
                'mpc/matrix_q': [80.0, 80.0, 18.0],
                'mpc/matrix_r': [0.10, 0.02],
                'mpc/matrix_rd': [5.0, 10.0]
            }
        ],
        remappings=[
            ('cmd_vel', cmd_topic), 
            ('odom', odom_topic), 
            ('traj', '/planner/trajectory'),
            ('dptraj', '/planner/dptrajectory')
        ]
    ))

    # terrain loader - provides global terrain point cloud
    ld.add_action(Node(
        package='map_generator',
        executable='terrain_loader',
        name='terrain_loader',
        output='screen',
        parameters=[{
            'pcd_file': 'forest.pcd',
            'primary_topic': '/global_cloud',
            'compat_topic': 'terrain_cloud',
            'legacy_topic': '/map_generator/global_cloud'
        }]
    ))

    # laser simulator - generates local point cloud around the vehicle from global map
    ld.add_action(Node(
        package='laser_simulator',
        executable='laser_sim_node',
        name=['car_', car_id, '_laser_simulator'],
        output='screen',
        parameters=[{
            'car_id': 0,
            'map_topic': '/global_cloud',
              'sensing_horizon': 13.0,
              'sensing_rate': 16.0,
            'pc_resolution': 0.05,
                            'map_voxel_resolution': 0.03,
                        'use_resolution_filter': False,
              'hrz_laser_line_num': 480,
                            'vtc_laser_line_num': 48,
                            'vtc_laser_range_dgr': 46.0,
                        'obstacle_rel_z_thresh': 0.20,
            'odom_topic': odom_topic,
            'laser_pcd_topic': '/car_0_laser_simulator/car_0_local_pointcloud',
            'laser_range_topic': '/car_0_laser_simulator/car_0_laser_range'
        }]
    ))

    # terrain analyzer - analyzes terrain from point cloud and odometry
    ld.add_action(Node(
        package='terrain_analyzer',
        executable='terrain_analyzer',
        name=['car_', car_id, '_terrain_analyzer'],
        output='screen',
        parameters=[{
            'terrain_pcd_path': 'forest.pcd',
            'grid_size_x': 220,
            'grid_size_y': 220,
            'viz_grid_size_x': 120,
            'viz_grid_size_y': 120,
            'resolution': 0.2,
            'sensor_range': 35.0,
            'use_pcd_prior': True,
            'use_pcd_vertical_obstacles': True,
              'obstacle_min_height': 0.18,
            'obstacle_max_height': 1.80,
                            'obstacle_min_points': 6,
              'obstacle_count_decay': 0.86,
                              'vertical_obstacle_inflation_m': 0.45,
              'risk_hard_threshold': 0.90,
              'risk_vis_hard_threshold': 0.90,
                            'terrain_risk_to_hard_obstacle': False,
                            'terrain_hardness_scale': 1.30,
            'terrain_risk_scale': 0.08,
                        'tree_risk_radius_m': 1.50,
                        'tree_hard_keepout_radius_m': 0.60,
            'tree_risk_power': 2.0,
            'non_tree_risk_cap': 0.40,
            'sdf_obstacle_slope': 0.85,
            'sdf_hard_obstacle_slope': 1.05,
            'map_padding_m': 2.0,
            'ego_clear_radius': 0.32
        }],
        remappings=[
            ('odom', odom_topic),
            ('cloud', '/car_0_laser_simulator/car_0_local_pointcloud/map'),
            ('fused_map', ['/car_', car_id, '_terrain_analyzer/fused_map']),
            ('sdf_map', ['/car_', car_id, '_terrain_analyzer/sdf_map']),
            ('elevation_cloud', ['/car_', car_id, '_terrain_analyzer/elevation_cloud']),
            ('risk_cloud', ['/car_', car_id, '_terrain_analyzer/risk_cloud']),
            ('obstacle_cloud', ['/car_', car_id, '_terrain_analyzer/obstacle_cloud'])
        ]
    ))

    # plan_manage planning node
    ld.add_action(Node(
        package='plan_manage',
        executable='plan_node',
        name='planner_node',
        output='screen',
        parameters=[PathJoinSubstitution([FindPackageShare('plan_manage'), 'config', 'planning.yaml']),
                    {
                        'map_name': 'forest.pcd',
                        'odometry_topic': odom_topic,
                        # Risk-based collision: 0.28 blocks cells within ~0.7m of trees (tree_risk_radius=1.5m)
                        'risk_obstacle_threshold': 0.28,
                        'risk_free_threshold': 0.08,
                        'publish_rough_preview': False,
                        'publish_local_horizon': False,
                        'local_publish_horizon_sec': 10.0,
                                                    'traj_publish_min_sdf': 0.10,
                        'traj_publish_sdf_tolerance': 0.003,
                        'traj_publish_risk_hard': 0.90,
                          'traj_safety_ignore_start_sec': 0.03,
                          'traj_safety_ignore_start_dist': 0.03,
                        'traj_safety_check_horizon_sec': 6.0,
                        'traj_publish_allow_partial_on_gate': False,
                        'traj_publish_partial_min_safe_sec': 1.0,
                        'traj_publish_partial_backoff_sec': 0.35,
                                                    'safeMargin': 0.65,
                                                    'wheel_base': 0.60,
                                                    'phimax': 0.48,
                                                    'kmax': 0.88,
                                                'partial_frontier_goal_gap_thresh': 3.2,
                        'partial_frontier_max_length': 18.0,
                        'vmax': 2.50,
                        'lonAccmax': 2.50,
                        'latAccmax': 2.50,
                        'max_piece_num_per_seg': 8,
                        'max_piece_num_per_seg_partial': 5,
                        'rt/max_iterations_cap': 60,
                        'rt/aml_max_iter_cap': 4,
                        'front_end/allow_reverse': False,
                        'front_end/traj_back_penalty': 6.0,
                        'front_end/traj_gear_switch_penalty': 14.0,
                                                'front_end/traj_steer_penalty': 1.40,
                                                'front_end/traj_steer_change_penalty': 1.20,
                                                                                                'front_end/collision_check_num': 4,
                        'front_end/shot_max_dist': 20.0,
                        'front_end/shot_heading_gate': 1.05,
                        'front_end/shot_progress_step': 0.75,
                          'fsm/replan_retry_interval': 0.35,
                                                'fsm/replan_lead_time': 0.20,
                        'fsm/replan_retry_backoff_max': 2.20,
                                                    'fsm/replan_cooldown': 1.50,
                        'fsm/exec_min_dwell': 1.20,
                                                                                                'fsm/track_err_replan_dist': 0.60,
                                                                                                'fsm/track_err_replan_yaw': 0.70,
                                                                                                'fsm/track_err_replan_min_elapsed': 0.40,
                                                                                                'fsm/track_err_replan_hits': 2,
                                                                                                'fsm/track_err_replan_hysteresis': 0.10,
                                                                                                'fsm/replan_anchor_max_dist': 0.80,
                                                                                                'fsm/replan_anchor_max_yaw': 0.60,
                                                                                                'fsm/collision_eval_track_err_max': 1.00,
                                                'fsm/enable_end_of_traj_replan': True,
                                                'fsm/continuous_replan_enable': False,
                                                'fsm/safety_check_period_ms': 80,
                                                'fsm/preemptive_replan_time': 2.00,
                                                'fsm/traj_expired_replan_margin': 0.90,
                                                'fsm/end_replan_goal_dist': 1.20,
                          'fsm/collision_replan_hits': 2,
                                                      'fsm/collision_check_lookahead_sec': 2.0,
                                                      'fsm/collision_replan_min_elapsed': 0.35,
                                                                                                            'fsm/collision_replan_margin': 0.02,
                        'fsm/duplicate_goal_dist': 0.25,
                        'fsm/duplicate_goal_window': 0.35,
                        'fsm/active_goal_replace_dist': 0.90,
                        'fsm/active_goal_replace_yaw': 0.75,
                        'fsm/goal_projection_max_radius': 2.2,
                        'fsm/goal_projection_radius_step': 0.15,
                        'fsm/goal_projection_angle_step_deg': 10.0,
                        'fsm/goal_projection_min_sdf': 0.10,
                        'fsm/goal_projection_max_risk': 0.95
                    }],
        remappings=[
            ('odometry_topic', odom_topic)
        ]
    ))

    ld.add_action(LogInfo(msg=['System initialized. Planning node active.']))

    return ld
