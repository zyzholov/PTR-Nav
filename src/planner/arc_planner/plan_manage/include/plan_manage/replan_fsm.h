#ifndef _REPLAN_FSM_H_
#define _REPLAN_FSM_H_

#include <Eigen/Eigen>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <iostream>
#include <limits>
#include <mutex>
#include <string>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/empty.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <vector>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <tf2/transform_datatypes.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>

#include<plan_manage/plan_manage.h>
#include <se2_grid_msgs/msg/se2_grid.hpp>
#include "se2_grid_core/SE2Grid.hpp"
#include "se2_grid_ros/se2_grid_ros.hpp"
class ReplanFSM
{
private:

    enum FSM_EXEC_STATE
    {
        INIT,
        WAIT_TARGET,
        REPLAN_TRAJ,
        EXEC_TRAJ,
        SEQUENTIAL_START
    };

    enum TARGET_TYPE
    {
        MANNUAL_TARGET = 1,
        PRESET_TARGET = 2,
        REFERENCE_PATH = 3
    };

    enum class ReplanReason
    {
        NONE = 0,
        TRAJ_EXPIRED = 1,
        HAZARD = 2,
        TRACK_MISMATCH = 3
    };


    std::shared_ptr<se2_grid::SE2Grid> se2_grid_ptr_;
    std::shared_ptr<se2_grid::SE2Grid> sdf_grid_ptr_;
    plan_manage::PlanManager planner_;
    std::shared_ptr<rclcpp::Node> node_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    // Async planning: runs process() in background thread so FSM stays responsive.
    std::thread async_plan_thread_;
    std::atomic<bool> async_plan_running_{false};
    std::atomic<bool> async_plan_done_{false};
    std::atomic<uint64_t> goal_seq_{0};
    std::atomic<uint64_t> async_plan_goal_seq_{0};
    std::atomic<bool> async_plan_success_{false};
    uint64_t executing_goal_seq_ = 0;
    rclcpp::Time async_plan_start_time_;
    std::mutex map_mutex_;
    void launchAsyncReplan();
    void finishAsyncReplan();

    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr parking_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr parking_sub_best_effort_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr parking_sub_legacy_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr parking_sub_legacy_best_effort_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<se2_grid_msgs::msg::SE2Grid>::SharedPtr sdf_grid_sub_;
    rclcpp::Subscription<se2_grid_msgs::msg::SE2Grid>::SharedPtr se2_grid_sub_;
    
    rclcpp::TimerBase::SharedPtr exec_timer_;
    rclcpp::TimerBase::SharedPtr safety_timer_;
    rclcpp::TimerBase::SharedPtr continuous_replan_timer_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr goal_marker_pub_;

    bool have_target_, collision_with_obs_;
    bool has_odom_ = false;
    bool in_swarm = true;
    Eigen::Vector4d init_state_;
    Eigen::Vector4d end_pt_;
    Eigen::Vector2d raw_goal_xy_ = Eigen::Vector2d::Zero();
    double raw_goal_yaw_ = 0.0;
    Eigen::Vector2d cur_pos_;
    double cur_yaw_, cur_vel_;
    int car_id_;
    double car_d_cr_;
    double start_world_time_;
    double target_x_, target_y_, target_yaw_;
    double safe_margin_;
    double time_budget_sec_ = 0.30;
    double replan_retry_interval_sec_ = 0.15;
    double replan_lead_time_sec_ = 0.40;
    double replan_retry_backoff_max_sec_ = 0.90;
    double replan_cooldown_sec_ = 1.50;
    double exec_min_dwell_sec_ = 1.50;
    double stall_dist_threshold_ = 0.35;
    double stall_goal_min_dist_ = 1.20;
    int collision_replan_hits_ = 3;
    int collision_hit_count_ = 0;
    double collision_check_lookahead_sec_ = 3.0;
    double collision_replan_margin_ = 0.05;
    double collision_time_to_hit_sec_ = std::numeric_limits<double>::infinity();
    double collision_detected_max_risk_ = -std::numeric_limits<double>::infinity();
    double collision_near_horizon_sec_ = 1.20;
    double collision_preemptive_cooldown_sec_ = 0.45;
    int plan_fail_count_ = 0;
    rclcpp::Time last_plan_try_time_;
    bool has_last_plan_try_time_ = false;
    rclcpp::Time last_replan_time_;
    bool has_last_replan_time_ = false;
    Eigen::Vector2d last_goal_xy_ = Eigen::Vector2d::Zero();
    double last_goal_yaw_ = 0.0;
    bool has_last_goal_ = false;
    rclcpp::Time last_goal_time_;
    double duplicate_goal_dist_ = 0.20;
    double duplicate_goal_window_sec_ = 1.50;
    double goal_projection_max_radius_ = 1.8;
    double goal_projection_radius_step_ = 0.15;
    double goal_projection_angle_step_deg_ = 10.0;
    double goal_projection_min_sdf_ = 0.03;
    double goal_projection_max_risk_ = 0.84;
    double active_goal_replace_dist_ = 0.80;
    double active_goal_replace_yaw_ = 0.80;
    double track_err_replan_dist_ = 0.55;
    double track_err_replan_yaw_ = 0.70;
    double track_err_replan_min_elapsed_ = 0.80;
    int track_err_replan_hits_ = 3;
    int track_mismatch_hit_count_ = 0;
    double track_err_replan_hysteresis_ = 0.10;
    bool track_err_replan_enable_ = false;
    double replan_anchor_max_dist_ = 0.75;
    double replan_anchor_max_yaw_ = 0.90;
    bool use_predictive_replan_anchor_ = false;
    bool snap_start_to_free_enable_ = false;
    bool start_rescue_enable_ = true;
    double start_rescue_trigger_risk_ = 0.24;
    double start_rescue_target_risk_ = 0.20;
    double start_rescue_max_radius_ = 0.90;
    double start_rescue_radius_step_ = 0.10;
    double start_rescue_angle_step_deg_ = 10.0;
    double collision_eval_track_err_max_ = 0.90;
    double risk_replan_threshold_ = 0.78;
    int risk_replan_hits_ = 2;
    int risk_hit_count_ = 0;
    bool continuous_replan_enable_ = false;
    double continuous_replan_hz_ = 5.0;
    double terrain_slope_replan_threshold_ = 0.62;
    double terrain_var_replan_threshold_ = 0.26;
    int terrain_replan_hits_ = 2;
    int terrain_hit_count_ = 0;
    double collision_replan_min_elapsed_sec_ = 0.90;
    double risk_replan_min_elapsed_sec_ = 0.90;
    double terrain_replan_min_elapsed_sec_ = 1.10;
    double terrain_check_lookahead_sec_ = 1.80;
    double min_exec_progress_for_hazard_replan_ = 0.18;
    int safety_check_period_ms_ = 80;
    bool enable_end_of_traj_replan_ = true;
    double traj_expired_replan_margin_sec_ = 0.35;
    double preemptive_replan_time_sec_ = 1.60;
    double end_replan_goal_dist_ = 1.80;
    ReplanReason last_replan_reason_ = ReplanReason::NONE;

    FSM_EXEC_STATE exec_state_;

    void changeFSMExecState(FSM_EXEC_STATE new_state, std::string pos_call);
    void publishGoalMarker(double x, double y, double yaw);
    void ParkingCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
    void OdomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void MapCallback(const se2_grid_msgs::msg::SE2Grid::SharedPtr msg);
    void SEMapCallback(const se2_grid_msgs::msg::SE2Grid::SharedPtr msg);
    void execFSMCallback();
    void checkCollisionCallback();
    void continuousReplanCallback();
    bool resolveMissionActiveGoal(
        double raw_goal_x, double raw_goal_y, double raw_goal_yaw,
        Eigen::Vector4d &active_goal, std::string &resolve_reason);

    rclcpp::Time traj_start_time_;

public:
    ReplanFSM()
    {
    }
    ~ReplanFSM()
    {
        if (async_plan_thread_.joinable())
            async_plan_thread_.join();
    }

    void init(std::shared_ptr<rclcpp::Node> node);
    std::string odom_topic_ = "map";
    
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

#endif
