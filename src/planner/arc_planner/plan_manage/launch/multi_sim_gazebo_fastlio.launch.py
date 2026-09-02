from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable, LogInfo
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    current_rmw = os.environ.get("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")

    enable_rviz = LaunchConfiguration("enable_rviz")
    use_sim_time = LaunchConfiguration("use_sim_time")
    rmw_implementation = LaunchConfiguration("rmw_implementation")

    ld = LaunchDescription()

    ld.add_action(DeclareLaunchArgument("enable_rviz", default_value="true"))
    ld.add_action(DeclareLaunchArgument("use_sim_time", default_value="true"))
    ld.add_action(DeclareLaunchArgument("rmw_implementation", default_value=current_rmw))

    ld.add_action(SetEnvironmentVariable(name="RMW_IMPLEMENTATION", value=rmw_implementation))

    ld.add_action(LogInfo(msg="Launching Gazebo + FAST_LIO + terrain analyzer + planner + MPC"))

    gazebo_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("fake_ugv"), "launch", "gazebo_spawn.launch.py"])
        ),
        launch_arguments={
            "x": "0.0",
            "y": "0.0",
            "z": "0.0",
            "yaw": "0.0",
            "enable_cmd_relay": "true",
            "cmd_topic": "/car_0/cmd_vel",
            "drive_topic": "/car_0/drive_cmd_vel",
            "command_relay_rate": "50.0",
            "command_relay_timeout": "0.30",
            "command_filter_alpha": "0.35",
        }.items(),
    )
    ld.add_action(gazebo_sim)

    ld.add_action(Node(
        package="fast_lio",
        executable="fastlio_mapping",
        name="fast_lio_mapping",
        output="screen",
        parameters=[
            PathJoinSubstitution([FindPackageShare("fast_lio"), "config", "gazebo_generic.yaml"]),
            {"use_sim_time": use_sim_time},
        ],
        remappings=[
            ("/Odometry", "/car_0/odometry"),
        ],
    ))

    ld.add_action(Node(
        package="terrain_analyzer",
        executable="terrain_analyzer",
        name="car_0_terrain_analyzer",
        output="screen",
        parameters=[{
            "use_sim_time": use_sim_time,
            "use_pcd_prior": False,
            "use_pcd_vertical_obstacles": True,
            "grid_size_x": 240,
            "grid_size_y": 240,
            "viz_grid_size_x": 100,
            "viz_grid_size_y": 100,
            "visual_publish_rate_hz": 4.0,
            "resolution": 0.2,
            "sensor_range": 45.0,
            "min_var": 0.001,
            "max_var": 1.0,
            "sdf_obstacle_slope": 0.60,
            "sdf_hard_obstacle_slope": 1.20,
            "sdf_obstacle_curvature": 0.55,
            "sdf_hard_obstacle_curvature": 1.35,
            "step_height_soft": 0.10,
            "step_height_hard": 0.35,
            "risk_weight_slope": 0.55,
            "risk_weight_curvature": 0.30,
            "risk_weight_step_height": 0.15,
            "risk_hard_threshold": 0.90,
            "tree_risk_radius_m": 1.55,
            "tree_hard_keepout_radius_m": 0.78,
            "tree_risk_power": 2.0,
            "obstacle_min_height": 0.20,
            "obstacle_max_height": 3.00,
            "obstacle_min_points": 4,
            "obstacle_count_decay": 0.80,
            "vertical_obstacle_inflation_m": 0.60,
            "ego_clear_radius": 0.26,
            "map_padding_m": 5.0,
        }],
        remappings=[
            ("odom", "/car_0/odometry"),
            ("cloud", "/cloud_registered"),
            ("fused_map", "/car_0_terrain_analyzer/fused_map"),
            ("sdf_map", "/car_0_terrain_analyzer/sdf_map"),
            ("elevation_cloud", "/car_0_terrain_analyzer/elevation_cloud"),
            ("risk_cloud", "/car_0_terrain_analyzer/risk_cloud"),
        ],
    ))

    ld.add_action(Node(
        package="plan_manage",
        executable="plan_node",
        name="planner_node",
        output="screen",
        respawn=True,
        respawn_delay=1.0,
        parameters=[
            PathJoinSubstitution([FindPackageShare("plan_manage"), "config", "planning_hunter_gazebo.yaml"]),
            {
                "use_sim_time": use_sim_time,
                "odometry_topic": "/car_0/odometry",
                "max_iterations": 34,
                "amlMaxIter": 2,
                "max_seach_time": 0.65,
                "pieceTime": 0.70,
                "safeMargin": 0.65,
                "riskWei": 1.80,
                "risk_safe_threshold": 0.12,
                "backend/jerk_weight": 0.22,
                "risk_obstacle_threshold": 0.24,
                "risk_free_threshold": 0.06,
                "traj_publish_risk_hard": 0.45,
                "traj_publish_max_start_gap": 0.30,
                "traj_publish_max_start_yaw_gap": 0.45,
                "partial_frontier_max_length": 10.0,
                "publish_optimized_path_visualization": False,
                "publish_local_horizon": False,
                "traj_publish_allow_partial_on_gate": False,
                "front_end/allow_reverse": False,
                "front_end/traj_steer_penalty": 2.5,
                "front_end/traj_steer_change_penalty": 2.2,
                "front_end/shot_max_dist": 20.0,
                "front_end/shot_heading_gate": 1.05,
                "front_end/shot_progress_step": 0.75,
                "fsm/replan_cooldown": 1.20,
                "fsm/exec_min_dwell": 1.00,
                "fsm/continuous_replan_enable": True,
                "fsm/continuous_replan_hz": 1.0,
                "fsm/replan_lead_time": 0.35,
                "fsm/replan_anchor_max_dist": 0.65,
                "fsm/use_predictive_replan_anchor": True,
                "fsm/safety_check_period_ms": 80,
                "fsm/preemptive_replan_time": 2.00,
                "fsm/duplicate_goal_window": 0.25,
                "fsm/goal_projection_max_radius": 3.0,
                "fsm/track_err_replan_enable": False,
                "fsm/track_err_replan_dist": 2.60,
                "fsm/track_err_replan_yaw": 1.25,
                "fsm/track_err_replan_min_elapsed": 2.50,
                "fsm/track_err_replan_hits": 5,
                "fsm/risk_replan_threshold": 0.50,
                "fsm/risk_replan_hits": 3,
                "fsm/start_rescue_enable": False,
                "fsm/start_rescue_trigger_risk": 0.24,
                "fsm/start_rescue_target_risk": 0.20,
                "fsm/start_rescue_max_radius": 0.90,
                "fsm/start_rescue_radius_step": 0.10,
                "fsm/start_rescue_angle_step_deg": 10.0,
                "fsm/collision_replan_hits": 2,
                "fsm/collision_check_lookahead_sec": 1.40,
                "fsm/collision_replan_min_elapsed": 0.35,
                "fsm/terrain_slope_replan_threshold": 0.50,
                "fsm/terrain_var_replan_threshold": 0.20,
                "fsm/terrain_replan_hits": 4,
                "fsm/risk_replan_min_elapsed": 0.80,
                "fsm/terrain_replan_min_elapsed": 1.00,
                "fsm/terrain_check_lookahead": 1.40,
                "fsm/min_exec_progress_for_hazard_replan": 0.18,
            },
        ],
    ))

    ld.add_action(Node(
        package="mpc_controller",
        executable="mpc_controller_node",
        name="car_0_mpc_controller_node",
        output="screen",
        parameters=[
            PathJoinSubstitution([FindPackageShare("mpc_controller"), "params", "controller_hunter_gazebo.yaml"]),
            {
                "use_sim_time": use_sim_time,
                "mpc/dt": 0.10,
                "mpc/predict_steps": 10,
                "mpc/use_nmpc_solver": False,
                "mpc/max_speed": 2.50,
                "mpc/min_speed": -0.80,
                "mpc/reverse_speed_cap": 0.50,
                "mpc/force_reverse_follow": False,
                "mpc/reverse_follow_speed_floor": 0.14,
                "mpc/reverse_hint_hold_sec": 2.0,
                "mpc/planner_update_max_jump_m": 0.65,
                "mpc/max_accel": 2.40,
                "mpc/max_steer": 0.5236,
                "mpc/max_dsteer": 2.80,
                "mpc/wheel_base": 0.549,
                "mpc/cmd_speed_scale": 1.00,
                "mpc/ref_projection_period_sec": 0.08,
                "mpc/control_visualization_period_sec": 0.50,
                "mpc/cmd_is_yaw_rate": False,
                "mpc/test_mpc": False,
                "mpc/enable_fallback_goal": False,
                "mpc/matrix_q": [80.0, 80.0, 18.0],
                "mpc/matrix_r": [0.10, 0.02],
                "mpc/matrix_rd": [5.0, 10.0],
            },
        ],
        remappings=[
            ("cmd_vel", "/car_0/cmd_vel"),
            ("odom", "/car_0/odometry"),
            ("traj", "/planner/trajectory"),
            ("dptraj", "/planner/dptrajectory"),
        ],
    ))

    ld.add_action(Node(
        package="rviz2",
        executable="rviz2",
        name="rviz_gazebo_fastlio",
        output="screen",
        arguments=["-d", PathJoinSubstitution([FindPackageShare("plan_manage"), "launch", "gazebo_fastlio.rviz"])],
        parameters=[{"use_sim_time": use_sim_time}],
        condition=IfCondition(enable_rviz),
    ))

    return ld
