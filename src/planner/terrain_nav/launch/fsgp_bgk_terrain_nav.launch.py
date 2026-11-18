"""Start only the planner/controller on the enhanced FSGP-BGK/SEB map."""

from launch import LaunchDescription
import math
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def optional_motion_parameters(values):
    """'auto' preserves the supplied YAML; explicit overrides stay numeric."""
    result={}
    for name,text in values.items():
        if text=='auto':continue
        if name=='live_motion_component_link_cells':
            if text not in ('1','2','3'):raise ValueError('invalid motion override: '+name)
            result[name]=int(text);continue
        if name in ('live_motion_track_known_slow_centers','unclassified_encounter_model_release',
                    'live_motion_gaussian_acceleration_envelope','live_motion_gaussian_constant_acceleration',
                    'live_motion_raw_surface_fit','dynamic_observed_only'):
            if text not in ('true','false'):raise ValueError('invalid motion override: '+name)
            result[name]=text=='true';continue
        value=float(text)
        bounds={'dynamic_yield_actuator_delay_sec':(0.,1.),
                'dynamic_yield_speed_dead_time_sec':(-1.,1.),
                'dynamic_yield_steering_dead_time_sec':(-1.,1.),
                'live_motion_gaussian_jerk_density':(0.,4.),
                'max_speed':(0.,20./3.6),
                'reverse_max_speed':(.1,.8),
                'unclassified_encounter_speed_mps':(0.,20./3.6)}
        valid=math.isfinite(value) and (bounds[name][0]<=value<=bounds[name][1]
            if name in bounds else 0<value<1)
        if (not valid or name=='max_speed' and value<=0 or
                name.endswith('_dead_time_sec') and value<0 and value!=-1):
            raise ValueError('invalid motion override: '+name)
        result[name]=value
    return result


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    params = LaunchConfiguration("params")
    odom_topic = LaunchConfiguration("odom_topic")
    fused_map_topic = LaunchConfiguration("fused_map_topic")
    cmd_topic = LaunchConfiguration("cmd_topic")
    risk_query_mode = LaunchConfiguration("risk_query_mode")
    se2_risk_layer = LaunchConfiguration("se2_risk_layer")
    hard_risk_layer = LaunchConfiguration("hard_risk_layer")
    se2_risk_lethal = LaunchConfiguration("se2_risk_lethal")
    live_obstacle_enable = LaunchConfiguration("live_obstacle_enable")

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("odom_topic", default_value="/Odometry"),
        DeclareLaunchArgument(
            "fused_map_topic",
            default_value="/car_0_fsgp_bgk/fused_map",
        ),
        DeclareLaunchArgument("cmd_topic", default_value="/cmd_vel"),
        DeclareLaunchArgument("risk_query_mode", default_value="se2_yaw"),
        DeclareLaunchArgument("se2_risk_layer", default_value="risk"),
        DeclareLaunchArgument("hard_risk_layer", default_value="collision_risk"),
        DeclareLaunchArgument("se2_risk_lethal", default_value="true"),
        DeclareLaunchArgument("chance_constrained", default_value="true"),
        DeclareLaunchArgument("chance_allow_startup_prior", default_value="false"),
        DeclareLaunchArgument("chance_allow_gp_completion", default_value=use_sim_time),
        DeclareLaunchArgument('chance_allow_joint_model',default_value='false'),
        DeclareLaunchArgument('chance_allow_independent_model_floor',default_value='false'),
        DeclareLaunchArgument('chance_allow_bounded_pending',default_value='false'),
        DeclareLaunchArgument("tracking_controller", default_value="auto",
                              choices=["auto", "hybrid", "sampling_mpc"]),
        DeclareLaunchArgument("live_obstacle_enable", default_value="true"),
        DeclareLaunchArgument("live_cloud_topic", default_value="/cloud_registered"),
        DeclareLaunchArgument("live_motion_prediction_model",default_value="reachable",
                              choices=["reachable","gaussian_joint_cuda","gaussian_joint_cpu"]),
        DeclareLaunchArgument("live_motion_sim_observation_calibration",default_value="false"),
        DeclareLaunchArgument('live_motion_gaussian_acceleration_envelope',default_value='auto',
                              choices=['auto','true','false']),
        DeclareLaunchArgument('live_motion_gaussian_constant_acceleration',default_value='auto',choices=['auto','true','false']),
        DeclareLaunchArgument('live_motion_raw_surface_fit',default_value='auto',choices=['auto','true','false']),
        DeclareLaunchArgument('live_motion_component_link_cells',default_value='auto',choices=['auto','1','2','3']),
        DeclareLaunchArgument('dynamic_observed_only',default_value='auto',choices=['auto','true','false']),
        DeclareLaunchArgument('live_motion_gaussian_jerk_density',default_value='auto'),
        DeclareLaunchArgument('max_speed',default_value='auto'),
        DeclareLaunchArgument('reverse_max_speed',default_value='auto'),
        DeclareLaunchArgument("dynamic_yield_actuator_delay_sec",default_value="auto"),
        DeclareLaunchArgument('dynamic_yield_speed_dead_time_sec',default_value='auto'),
        DeclareLaunchArgument('dynamic_yield_steering_dead_time_sec',default_value='auto'),
        DeclareLaunchArgument("live_motion_joint_risk_limit",default_value="auto"),
        DeclareLaunchArgument("unclassified_encounter_speed_mps",default_value="auto"),
        DeclareLaunchArgument("live_motion_track_known_slow_centers",default_value="auto",
                              choices=['auto','true','false']),
        DeclareLaunchArgument("unclassified_encounter_model_release",default_value="auto",
                              choices=['auto','true','false']),
        DeclareLaunchArgument("reverse_recovery_enable", default_value="false"),
        DeclareLaunchArgument("stuck_replan_attempts",default_value="0"),
        DeclareLaunchArgument(
            "params",
            default_value=PathJoinSubstitution([
                FindPackageShare("terrain_nav"),
                "config",
                "hunter_gazebo.yaml",
            ]),
        ),
        OpaqueFunction(function=lambda context: [Node(
            package="terrain_nav",
            executable="terrain_nav_node",
            name="terrain_nav_node",
            output="screen",
            parameters=[
                params,
                {
                    "use_sim_time": use_sim_time,
                    "odom_topic": odom_topic,
                    "fused_map_topic": fused_map_topic,
                    "cmd_topic": cmd_topic,
                    "risk_query_mode": risk_query_mode,
                    "se2_risk_layer": se2_risk_layer,
                    "hard_risk_layer": hard_risk_layer,
                    "se2_risk_lethal": se2_risk_lethal,
                    "chance_constrained": LaunchConfiguration("chance_constrained"),
                    "chance_allow_startup_prior": LaunchConfiguration("chance_allow_startup_prior"),
                    "chance_allow_gp_completion": LaunchConfiguration("chance_allow_gp_completion"),
                    'chance_allow_joint_model': LaunchConfiguration('chance_allow_joint_model'),
                    'chance_allow_independent_model_floor': LaunchConfiguration('chance_allow_independent_model_floor'),
                    'chance_allow_bounded_pending': LaunchConfiguration('chance_allow_bounded_pending'),
                    "tracking_controller": LaunchConfiguration("tracking_controller"),
                    "live_obstacle_enable": live_obstacle_enable,
                    "live_cloud_topic": LaunchConfiguration("live_cloud_topic"),
                    "live_motion_prediction_model": LaunchConfiguration("live_motion_prediction_model"),
                    "live_motion_sim_observation_calibration": LaunchConfiguration("live_motion_sim_observation_calibration"),
                    "reverse_recovery_enable": LaunchConfiguration("reverse_recovery_enable"),
                    "stuck_replan_attempts": LaunchConfiguration("stuck_replan_attempts"),
                },
                optional_motion_parameters({name:LaunchConfiguration(name).perform(context)
                    for name in ('dynamic_yield_actuator_delay_sec','dynamic_yield_speed_dead_time_sec',
                                 'dynamic_yield_steering_dead_time_sec','live_motion_joint_risk_limit',
                                 'unclassified_encounter_speed_mps','live_motion_track_known_slow_centers',
                                 'unclassified_encounter_model_release','live_motion_gaussian_acceleration_envelope',
                                 'live_motion_gaussian_constant_acceleration','live_motion_gaussian_jerk_density',
                                 'live_motion_raw_surface_fit','live_motion_component_link_cells',
                                 'dynamic_observed_only','max_speed','reverse_max_speed')}),
            ],
        )]),
    ])
