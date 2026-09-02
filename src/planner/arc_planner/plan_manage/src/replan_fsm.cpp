#include <plan_manage/replan_fsm.h>
#include <chrono>
#include <array>
#include <cmath>
#include <limits>
#include <thread>
#include <atomic>

namespace {

double wrapToPi(double ang)
{
    while (ang > M_PI) ang -= 2.0 * M_PI;
    while (ang < -M_PI) ang += 2.0 * M_PI;
    return ang;
}

double sampleSdfAt(
    const std::shared_ptr<se2_grid::SE2Grid>& sdf_grid,
    double x, double y)
{
    if (!sdf_grid || !sdf_grid->exists("sdf_cpu"))
        return std::numeric_limits<double>::quiet_NaN();
    try {
        return sdf_grid->getValue("sdf_cpu", Eigen::Vector3d(x, y, 0.0));
    } catch (...) {
        return std::numeric_limits<double>::quiet_NaN();
    }
}

bool snapStateToFreeSdf(
    const std::shared_ptr<se2_grid::SE2Grid>& sdf_grid,
    Eigen::Vector4d& state,
    double min_sdf,
    double max_radius,
    double radius_step,
    double angle_step_deg,
    const rclcpp::Logger& logger,
    const char* tag)
{
    if (!sdf_grid || !sdf_grid->exists("sdf_cpu"))
        return false;

    const double x0 = state(0);
    const double y0 = state(1);
    const double sdf0 = sampleSdfAt(sdf_grid, x0, y0);
    if (std::isfinite(sdf0) && sdf0 > min_sdf)
        return false;

    Eigen::Vector2d min_range = Eigen::Vector2d::Constant(-std::numeric_limits<double>::infinity());
    Eigen::Vector2d max_range = Eigen::Vector2d::Constant(std::numeric_limits<double>::infinity());
    bool has_map_range = false;
    try {
        sdf_grid->getMapRange(min_range, max_range);
        has_map_range = true;
    } catch (...) {
        has_map_range = false;
    }

    const double r_step = std::max(0.05, radius_step);
    const double a_step = std::max(5.0, angle_step_deg) * M_PI / 180.0;
    double best_sdf = -std::numeric_limits<double>::infinity();
    double best_x = x0;
    double best_y = y0;
    bool found = false;

    for (double r = r_step; r <= max_radius + 1e-6; r += r_step)
    {
        for (double a = -M_PI; a < M_PI; a += a_step)
        {
            const double x = x0 + r * std::cos(a);
            const double y = y0 + r * std::sin(a);
            if (has_map_range &&
                (x < min_range.x() || x > max_range.x() || y < min_range.y() || y > max_range.y()))
            {
                continue;
            }

            const double sdf = sampleSdfAt(sdf_grid, x, y);
            if (!std::isfinite(sdf) || sdf <= min_sdf)
                continue;

            if (sdf > best_sdf)
            {
                best_sdf = sdf;
                best_x = x;
                best_y = y;
                found = true;
            }
        }
        if (found)
            break;
    }

    if (!found)
        return false;

    RCLCPP_WARN(logger,
                "Snap %s state from (%.2f, %.2f, sdf=%.3f) to (%.2f, %.2f, sdf=%.3f)",
                tag, x0, y0, sdf0, best_x, best_y, best_sdf);
    state(0) = best_x;
    state(1) = best_y;
    return true;
}

bool snapStateToFreeRisk(
    const std::shared_ptr<se2_grid::SE2Grid>& se2_grid,
    Eigen::Vector4d& state,
    double max_risk,
    double max_radius,
    double radius_step,
    double angle_step_deg,
    const rclcpp::Logger& logger,
    const char* tag)
{
    if (!se2_grid || !se2_grid->exists("risk"))
        return false;

    const double x0 = state(0);
    const double y0 = state(1);
    double risk0 = std::numeric_limits<double>::quiet_NaN();
    try {
        risk0 = se2_grid->getValue("risk", Eigen::Vector3d(x0, y0, 0.0));
    } catch (...) {}
    if (std::isfinite(risk0) && risk0 < max_risk)
        return false;

    Eigen::Vector2d min_range = Eigen::Vector2d::Constant(-std::numeric_limits<double>::infinity());
    Eigen::Vector2d max_range = Eigen::Vector2d::Constant(std::numeric_limits<double>::infinity());
    bool has_map_range = false;
    try {
        se2_grid->getMapRange(min_range, max_range);
        has_map_range = true;
    } catch (...) {
        has_map_range = false;
    }

    const double r_step = std::max(0.05, radius_step);
    const double a_step = std::max(5.0, angle_step_deg) * M_PI / 180.0;
    double best_risk = std::numeric_limits<double>::infinity();
    double best_x = x0;
    double best_y = y0;
    bool found = false;

    for (double r = r_step; r <= max_radius + 1e-6; r += r_step)
    {
        for (double a = -M_PI; a < M_PI; a += a_step)
        {
            const double x = x0 + r * std::cos(a);
            const double y = y0 + r * std::sin(a);
            if (has_map_range &&
                (x < min_range.x() || x > max_range.x() || y < min_range.y() || y > max_range.y()))
            {
                continue;
            }

            double risk = std::numeric_limits<double>::quiet_NaN();
            try {
                risk = se2_grid->getValue("risk", Eigen::Vector3d(x, y, 0.0));
            } catch (...) {}
            if (!std::isfinite(risk) || risk >= max_risk)
                continue;

            if (risk < best_risk)
            {
                best_risk = risk;
                best_x = x;
                best_y = y;
                found = true;
            }
        }
        if (found)
            break;
    }

    if (!found)
        return false;

    RCLCPP_WARN(logger,
                "Snap %s state from (%.2f, %.2f, risk=%.3f) to (%.2f, %.2f, risk=%.3f)",
                tag, x0, y0, risk0, best_x, best_y, best_risk);
    state(0) = best_x;
    state(1) = best_y;
    return true;
}

} // namespace

void ReplanFSM::init(std::shared_ptr<rclcpp::Node> node)
{
    node_ = node;
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(node_);
    exec_state_ = ReplanFSM::FSM_EXEC_STATE::INIT;
    have_target_ = false;
    collision_with_obs_ = false;
    has_odom_ = false;
    executing_goal_seq_ = 0;
    cur_pos_.setZero();
    cur_yaw_ = 0.0;
    cur_vel_ = 0.0;

    node_->declare_parameter("odometry_topic", odom_topic_);
    node_->declare_parameter("vehicle/car_d_cr", 1.015);
    node_->declare_parameter("vehicle/car_id", 0);
    node_->declare_parameter("fsm/target_x", 0.0);
    node_->declare_parameter("fsm/target_y", 0.0);
    node_->declare_parameter("fsm/target_yaw", 0.0);
    node_->declare_parameter("fsm/in_swarm", true);
    node_->declare_parameter("fsm/time_budget", time_budget_sec_);
    node_->declare_parameter("fsm/replan_retry_interval", 0.15);
    node_->declare_parameter("fsm/replan_lead_time", 0.40);
    node_->declare_parameter("fsm/replan_retry_backoff_max", replan_retry_backoff_max_sec_);
    node_->declare_parameter("fsm/replan_cooldown", replan_cooldown_sec_);
    node_->declare_parameter("fsm/exec_min_dwell", exec_min_dwell_sec_);
    node_->declare_parameter("fsm/collision_replan_hits", collision_replan_hits_);
    node_->declare_parameter("fsm/collision_check_lookahead_sec", collision_check_lookahead_sec_);
    node_->declare_parameter("fsm/collision_replan_margin", collision_replan_margin_);
    node_->declare_parameter("fsm/collision_replan_min_elapsed", collision_replan_min_elapsed_sec_);
    node_->declare_parameter("fsm/collision_near_horizon_sec", collision_near_horizon_sec_);
    node_->declare_parameter("fsm/collision_preemptive_cooldown", collision_preemptive_cooldown_sec_);
    node_->declare_parameter("fsm/enable_end_of_traj_replan", enable_end_of_traj_replan_);
    node_->declare_parameter("fsm/traj_expired_replan_margin", traj_expired_replan_margin_sec_);
    node_->declare_parameter("fsm/preemptive_replan_time", preemptive_replan_time_sec_);
    node_->declare_parameter("fsm/end_replan_goal_dist", end_replan_goal_dist_);
    node_->declare_parameter("fsm/track_err_replan_dist", track_err_replan_dist_);
    node_->declare_parameter("fsm/track_err_replan_yaw", track_err_replan_yaw_);
    node_->declare_parameter("fsm/track_err_replan_min_elapsed", track_err_replan_min_elapsed_);
    node_->declare_parameter("fsm/track_err_replan_hits", track_err_replan_hits_);
    node_->declare_parameter("fsm/track_err_replan_hysteresis", track_err_replan_hysteresis_);
    node_->declare_parameter("fsm/track_err_replan_enable", track_err_replan_enable_);
    node_->declare_parameter("fsm/replan_anchor_max_dist", replan_anchor_max_dist_);
    node_->declare_parameter("fsm/replan_anchor_max_yaw", replan_anchor_max_yaw_);
    node_->declare_parameter("fsm/use_predictive_replan_anchor", use_predictive_replan_anchor_);
    node_->declare_parameter("fsm/snap_start_to_free_enable", snap_start_to_free_enable_);
    node_->declare_parameter("fsm/start_rescue_enable", start_rescue_enable_);
    node_->declare_parameter("fsm/start_rescue_trigger_risk", start_rescue_trigger_risk_);
    node_->declare_parameter("fsm/start_rescue_target_risk", start_rescue_target_risk_);
    node_->declare_parameter("fsm/start_rescue_max_radius", start_rescue_max_radius_);
    node_->declare_parameter("fsm/start_rescue_radius_step", start_rescue_radius_step_);
    node_->declare_parameter("fsm/start_rescue_angle_step_deg", start_rescue_angle_step_deg_);
    node_->declare_parameter("fsm/collision_eval_track_err_max", collision_eval_track_err_max_);
    node_->declare_parameter("fsm/risk_replan_threshold", risk_replan_threshold_);
    node_->declare_parameter("fsm/risk_replan_hits", risk_replan_hits_);
    node_->declare_parameter("fsm/risk_replan_min_elapsed", risk_replan_min_elapsed_sec_);
    node_->declare_parameter("fsm/goal_projection_max_radius", goal_projection_max_radius_);
    node_->declare_parameter("fsm/goal_projection_radius_step", goal_projection_radius_step_);
    node_->declare_parameter("fsm/goal_projection_angle_step_deg", goal_projection_angle_step_deg_);
    node_->declare_parameter("fsm/goal_projection_min_sdf", goal_projection_min_sdf_);
    node_->declare_parameter("fsm/goal_projection_max_risk", goal_projection_max_risk_);
    node_->declare_parameter("fsm/duplicate_goal_dist", duplicate_goal_dist_);
    node_->declare_parameter("fsm/duplicate_goal_window", duplicate_goal_window_sec_);
    node_->declare_parameter("fsm/active_goal_replace_dist", active_goal_replace_dist_);
    node_->declare_parameter("fsm/active_goal_replace_yaw", active_goal_replace_yaw_);
    node_->declare_parameter("fsm/continuous_replan_enable", continuous_replan_enable_);
    node_->declare_parameter("fsm/continuous_replan_hz", continuous_replan_hz_);
    node_->declare_parameter("fsm/safety_check_period_ms", safety_check_period_ms_);
    safe_margin_ = node_->declare_parameter("safeMargin", 0.12);

    node_->get_parameter("odometry_topic", odom_topic_);
    node_->get_parameter("vehicle/car_d_cr", car_d_cr_);
    node_->get_parameter("vehicle/car_id", car_id_);
    node_->get_parameter("fsm/target_x", target_x_);
    node_->get_parameter("fsm/target_y", target_y_);
    node_->get_parameter("fsm/target_yaw", target_yaw_);
    node_->get_parameter("fsm/in_swarm", in_swarm);
    node_->get_parameter("fsm/time_budget", time_budget_sec_);
    node_->get_parameter("fsm/replan_retry_interval", replan_retry_interval_sec_);
    node_->get_parameter("fsm/replan_lead_time", replan_lead_time_sec_);
    node_->get_parameter("fsm/replan_retry_backoff_max", replan_retry_backoff_max_sec_);
    node_->get_parameter("fsm/replan_cooldown", replan_cooldown_sec_);
    node_->get_parameter("fsm/exec_min_dwell", exec_min_dwell_sec_);
    node_->get_parameter("fsm/collision_replan_hits", collision_replan_hits_);
    node_->get_parameter("fsm/collision_check_lookahead_sec", collision_check_lookahead_sec_);
    node_->get_parameter("fsm/collision_replan_margin", collision_replan_margin_);
    node_->get_parameter("fsm/collision_replan_min_elapsed", collision_replan_min_elapsed_sec_);
    node_->get_parameter("fsm/collision_near_horizon_sec", collision_near_horizon_sec_);
    node_->get_parameter("fsm/collision_preemptive_cooldown", collision_preemptive_cooldown_sec_);
    node_->get_parameter("fsm/enable_end_of_traj_replan", enable_end_of_traj_replan_);
    node_->get_parameter("fsm/traj_expired_replan_margin", traj_expired_replan_margin_sec_);
    node_->get_parameter("fsm/preemptive_replan_time", preemptive_replan_time_sec_);
    node_->get_parameter("fsm/end_replan_goal_dist", end_replan_goal_dist_);
    node_->get_parameter("fsm/track_err_replan_dist", track_err_replan_dist_);
    node_->get_parameter("fsm/track_err_replan_yaw", track_err_replan_yaw_);
    node_->get_parameter("fsm/track_err_replan_min_elapsed", track_err_replan_min_elapsed_);
    node_->get_parameter("fsm/track_err_replan_hits", track_err_replan_hits_);
    node_->get_parameter("fsm/track_err_replan_hysteresis", track_err_replan_hysteresis_);
    node_->get_parameter("fsm/track_err_replan_enable", track_err_replan_enable_);
    node_->get_parameter("fsm/replan_anchor_max_dist", replan_anchor_max_dist_);
    node_->get_parameter("fsm/replan_anchor_max_yaw", replan_anchor_max_yaw_);
    node_->get_parameter("fsm/use_predictive_replan_anchor", use_predictive_replan_anchor_);
    node_->get_parameter("fsm/snap_start_to_free_enable", snap_start_to_free_enable_);
    node_->get_parameter("fsm/start_rescue_enable", start_rescue_enable_);
    node_->get_parameter("fsm/start_rescue_trigger_risk", start_rescue_trigger_risk_);
    node_->get_parameter("fsm/start_rescue_target_risk", start_rescue_target_risk_);
    node_->get_parameter("fsm/start_rescue_max_radius", start_rescue_max_radius_);
    node_->get_parameter("fsm/start_rescue_radius_step", start_rescue_radius_step_);
    node_->get_parameter("fsm/start_rescue_angle_step_deg", start_rescue_angle_step_deg_);
    node_->get_parameter("fsm/collision_eval_track_err_max", collision_eval_track_err_max_);
    node_->get_parameter("fsm/risk_replan_threshold", risk_replan_threshold_);
    node_->get_parameter("fsm/risk_replan_hits", risk_replan_hits_);
    node_->get_parameter("fsm/risk_replan_min_elapsed", risk_replan_min_elapsed_sec_);
    node_->get_parameter("fsm/goal_projection_max_radius", goal_projection_max_radius_);
    node_->get_parameter("fsm/goal_projection_radius_step", goal_projection_radius_step_);
    node_->get_parameter("fsm/goal_projection_angle_step_deg", goal_projection_angle_step_deg_);
    node_->get_parameter("fsm/goal_projection_min_sdf", goal_projection_min_sdf_);
    node_->get_parameter("fsm/goal_projection_max_risk", goal_projection_max_risk_);
    node_->get_parameter("fsm/duplicate_goal_dist", duplicate_goal_dist_);
    node_->get_parameter("fsm/duplicate_goal_window", duplicate_goal_window_sec_);
    node_->get_parameter("fsm/active_goal_replace_dist", active_goal_replace_dist_);
    node_->get_parameter("fsm/active_goal_replace_yaw", active_goal_replace_yaw_);
    node_->get_parameter("fsm/continuous_replan_enable", continuous_replan_enable_);
    node_->get_parameter("fsm/continuous_replan_hz", continuous_replan_hz_);
    node_->get_parameter("fsm/safety_check_period_ms", safety_check_period_ms_);

    // End-of-trajectory replanning is the primary rolling-horizon mechanism.
    // Continuous replanning is optional because full Kino-A* + backend optimize is expensive.
    enable_end_of_traj_replan_ = true;
    continuous_replan_hz_ = std::clamp(continuous_replan_hz_, 0.2, 5.0);
    safety_check_period_ms_ = std::clamp(safety_check_period_ms_, 40, 250);

    time_budget_sec_ = std::clamp(time_budget_sec_, 0.05, 2.0);
    replan_retry_interval_sec_ = std::clamp(replan_retry_interval_sec_, 0.05, 2.0);
    replan_retry_backoff_max_sec_ = std::clamp(replan_retry_backoff_max_sec_, 0.10, 3.0);
    replan_cooldown_sec_ = std::clamp(replan_cooldown_sec_, 0.05, 3.0);
    exec_min_dwell_sec_ = std::clamp(exec_min_dwell_sec_, 0.0, 3.0);
    traj_expired_replan_margin_sec_ = std::max(0.0, traj_expired_replan_margin_sec_);
    preemptive_replan_time_sec_ = std::clamp(preemptive_replan_time_sec_, 0.6, 3.0);
    end_replan_goal_dist_ = std::max(0.5, end_replan_goal_dist_);
    track_err_replan_dist_ = std::max(0.30, track_err_replan_dist_);
    track_err_replan_yaw_ = std::max(0.35, track_err_replan_yaw_);
    track_err_replan_min_elapsed_ = std::max(0.10, track_err_replan_min_elapsed_);
    track_err_replan_hits_ = std::max(1, track_err_replan_hits_);
    track_err_replan_hysteresis_ = std::max(0.0, track_err_replan_hysteresis_);
    replan_anchor_max_dist_ = std::max(0.25, replan_anchor_max_dist_);
    replan_anchor_max_yaw_ = std::max(0.20, replan_anchor_max_yaw_);
    start_rescue_trigger_risk_ = std::clamp(start_rescue_trigger_risk_, 0.05, 0.95);
    start_rescue_target_risk_ = std::clamp(
        start_rescue_target_risk_, 0.02, std::max(0.02, start_rescue_trigger_risk_ - 0.01));
    start_rescue_max_radius_ = std::clamp(start_rescue_max_radius_, 0.20, 1.50);
    start_rescue_radius_step_ = std::clamp(start_rescue_radius_step_, 0.05, 0.30);
    start_rescue_angle_step_deg_ = std::clamp(start_rescue_angle_step_deg_, 5.0, 45.0);
    collision_eval_track_err_max_ = std::max(0.40, collision_eval_track_err_max_);
    collision_replan_hits_ = std::max(1, collision_replan_hits_);
    collision_check_lookahead_sec_ = std::clamp(collision_check_lookahead_sec_, 0.60, 8.0);
    collision_replan_min_elapsed_sec_ = std::clamp(collision_replan_min_elapsed_sec_, 0.05, 2.0);
    risk_replan_min_elapsed_sec_ = std::clamp(risk_replan_min_elapsed_sec_, 0.05, 2.0);
    collision_near_horizon_sec_ = std::clamp(
        collision_near_horizon_sec_, 0.40, std::max(0.40, collision_check_lookahead_sec_));
    collision_preemptive_cooldown_sec_ = std::clamp(
        collision_preemptive_cooldown_sec_, 0.05, std::max(0.05, replan_cooldown_sec_));
    risk_replan_threshold_ = std::clamp(risk_replan_threshold_, 0.05, 0.95);
    risk_replan_hits_ = std::max(1, risk_replan_hits_);

    se2_grid_ptr_.reset(new se2_grid::SE2Grid);
    // SDF/ESDF map is no longer published — pass nullptr to avoid hidden
    // try-catch exceptions from querying an empty grid shell thousands of
    // times per planning cycle (Gemini analysis: "performance assassin").
    // sdf_grid_ptr_.reset(new se2_grid::SE2Grid);
    planner_.init(node_, se2_grid_ptr_, nullptr);

    odom_sub_ = node_->create_subscription<nav_msgs::msg::Odometry>(
        odom_topic_, 100, std::bind(&ReplanFSM::OdomCallback, this, std::placeholders::_1));

    parking_sub_ = node_->create_subscription<geometry_msgs::msg::PoseStamped>(
        "/goal_pose", 1, std::bind(&ReplanFSM::ParkingCallback, this, std::placeholders::_1));

    auto goal_qos_best_effort = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();
    parking_sub_best_effort_ = node_->create_subscription<geometry_msgs::msg::PoseStamped>(
        "/goal_pose", goal_qos_best_effort,
        std::bind(&ReplanFSM::ParkingCallback, this, std::placeholders::_1));

    // Compatibility with RViz default 2D Nav Goal topic.
    parking_sub_legacy_ = node_->create_subscription<geometry_msgs::msg::PoseStamped>(
        "/move_base_simple/goal", 1, std::bind(&ReplanFSM::ParkingCallback, this, std::placeholders::_1));
    parking_sub_legacy_best_effort_ = node_->create_subscription<geometry_msgs::msg::PoseStamped>(
        "/move_base_simple/goal", goal_qos_best_effort,
        std::bind(&ReplanFSM::ParkingCallback, this, std::placeholders::_1));

    // SDF subscription REMOVED — topic is no longer published.
    // sdf_grid_sub_ = node_->create_subscription<...>("/.../sdf_map", ...);

    se2_grid_sub_ = node_->create_subscription<se2_grid_msgs::msg::SE2Grid>(
        "/car_" + std::to_string(car_id_) + "_terrain_analyzer/fused_map", rclcpp::QoS(1),
        std::bind(&ReplanFSM::SEMapCallback, this, std::placeholders::_1));

    goal_marker_pub_ = node_->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/visualization/goal_marker", rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local());

    exec_timer_ = node_->create_wall_timer(
        std::chrono::milliseconds(20), std::bind(&ReplanFSM::execFSMCallback, this));
    
    safety_timer_ = node_->create_wall_timer(
        std::chrono::milliseconds(safety_check_period_ms_), std::bind(&ReplanFSM::checkCollisionCallback, this));
    RCLCPP_INFO(node_->get_logger(), "Safety lookahead check period: %dms", safety_check_period_ms_);

    if (continuous_replan_enable_)
    {
        int continuous_period_ms = std::max(50, static_cast<int>(1000.0 / std::max(0.2, continuous_replan_hz_)));
        continuous_replan_timer_ = node_->create_wall_timer(
            std::chrono::milliseconds(continuous_period_ms),
            std::bind(&ReplanFSM::continuousReplanCallback, this));
        RCLCPP_INFO(node_->get_logger(), "Continuous replan enabled at %.1fHz (%dms period)",
                    continuous_replan_hz_, continuous_period_ms);
    }
}

void ReplanFSM::MapCallback(const se2_grid_msgs::msg::SE2Grid::SharedPtr msg)
{
    std::unique_lock<std::mutex> lock(map_mutex_, std::try_to_lock);
    if (!lock.owns_lock())
        return;
    se2_grid::SE2GridRosConverter::fromMessage(*msg, *sdf_grid_ptr_);
    sdf_grid_ptr_->convertToDefaultStartIndex();
    static int log_count = 0;
    if (log_count < 5)
    {
        RCLCPP_INFO(node_->get_logger(), "Received sdf_map, has sdf_cpu=%d", sdf_grid_ptr_->exists("sdf_cpu"));
        log_count++;
    }
}

void ReplanFSM::SEMapCallback(const se2_grid_msgs::msg::SE2Grid::SharedPtr msg)
{
    std::unique_lock<std::mutex> lock(map_mutex_, std::try_to_lock);
    if (!lock.owns_lock())
        return;
    se2_grid::SE2GridRosConverter::fromMessage(*msg, *se2_grid_ptr_);
    se2_grid_ptr_->convertToDefaultStartIndex();
    static int log_count = 0;
    if (log_count < 5)
    {
        RCLCPP_INFO(node_->get_logger(), "Received fused_map, has risk=%d zbx=%d zby=%d", 
                    se2_grid_ptr_->exists("risk"), se2_grid_ptr_->exists("zbx"), se2_grid_ptr_->exists("zby"));
        log_count++;
    }
}

void ReplanFSM::OdomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
    Eigen::Vector3d center_pos(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);
    Eigen::Vector3d pos2center(-car_d_cr_, 0, 0);

    Eigen::Quaterniond quaternion(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x, 
                                  msg->pose.pose.orientation.y, msg->pose.pose.orientation.z);
    Eigen::Vector3d pos = center_pos;
    cur_pos_ = pos.head(2);
    cur_vel_ = msg->twist.twist.linear.x;
    
    // Convert quaternion to yaw angle
    tf2::Quaternion q(msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, 
                      msg->pose.pose.orientation.z, msg->pose.pose.orientation.w);
    tf2::Matrix3x3 m(q);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);
    cur_yaw_ = yaw;
    has_odom_ = true;
    
    // Publish transform
    geometry_msgs::msg::TransformStamped transform;
    transform.header.stamp = msg->header.stamp;
    transform.header.frame_id = "map";
    transform.child_frame_id = "car_" + std::to_string(car_id_) + "_pos";
    
    transform.transform.translation.x = cur_pos_(0);
    transform.transform.translation.y = cur_pos_(1);
    transform.transform.translation.z = 0.0;
    
    tf2::Quaternion quat;
    quat.setRPY(0, 0, cur_yaw_);
    transform.transform.rotation = tf2::toMsg(quat);
    
    tf_broadcaster_->sendTransform(transform);
}

void ReplanFSM::publishGoalMarker(double x, double y, double yaw)
{
    visualization_msgs::msg::MarkerArray arr;
    auto now = node_->get_clock()->now();

    // Arrow showing goal position and heading
    visualization_msgs::msg::Marker arrow;
    arrow.header.frame_id = "map";
    arrow.header.stamp = now;
    arrow.ns = "goal_arrow";
    arrow.id = 0;
    arrow.type = visualization_msgs::msg::Marker::ARROW;
    arrow.action = visualization_msgs::msg::Marker::ADD;
    arrow.pose.position.x = x;
    arrow.pose.position.y = y;
    arrow.pose.position.z = 0.15;
    tf2::Quaternion q;
    q.setRPY(0, 0, yaw);
    arrow.pose.orientation = tf2::toMsg(q);
    arrow.scale.x = 0.8;
    arrow.scale.y = 0.18;
    arrow.scale.z = 0.18;
    arrow.color.r = 1.0;
    arrow.color.g = 0.2;
    arrow.color.b = 0.2;
    arrow.color.a = 0.95;
    arr.markers.push_back(arrow);

    // Text label with coordinates
    visualization_msgs::msg::Marker text;
    text.header.frame_id = "map";
    text.header.stamp = now;
    text.ns = "goal_text";
    text.id = 1;
    text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    text.action = visualization_msgs::msg::Marker::ADD;
    text.pose.position.x = x;
    text.pose.position.y = y;
    text.pose.position.z = 0.60;
    text.pose.orientation.w = 1.0;
    text.scale.z = 0.35;
    text.color.r = 1.0;
    text.color.g = 1.0;
    text.color.b = 1.0;
    text.color.a = 0.90;
    char buf[64];
    std::snprintf(buf, sizeof(buf), "Goal(%.1f, %.1f)", x, y);
    text.text = buf;
    arr.markers.push_back(text);

    goal_marker_pub_->publish(arr);
}

bool ReplanFSM::resolveMissionActiveGoal(
    double raw_goal_x, double raw_goal_y, double raw_goal_yaw,
    Eigen::Vector4d &active_goal, std::string &resolve_reason)
{
    active_goal << raw_goal_x, raw_goal_y, raw_goal_yaw, 0.50;
    resolve_reason.clear();

    if (!se2_grid_ptr_ || !se2_grid_ptr_->exists("risk"))
    {
        // Fallback: try SDF if available
        if (!sdf_grid_ptr_ || !sdf_grid_ptr_->exists("sdf_cpu"))
        {
            // No terrain map available yet — accept goal optimistically.
            // The planner safety gate will reject unsafe trajectories when data arrives.
            resolve_reason = "terrain map not ready, accept goal optimistically";
            return true;
        }
    }

    Eigen::Vector2d min_range = Eigen::Vector2d::Zero();
    Eigen::Vector2d max_range = Eigen::Vector2d::Zero();
    bool has_map_range = false;
    try
    {
        if (se2_grid_ptr_ && se2_grid_ptr_->getSizePos()(0) > 0)
            se2_grid_ptr_->getMapRange(min_range, max_range);
        else if (sdf_grid_ptr_ && sdf_grid_ptr_->getSizePos()(0) > 0)
            sdf_grid_ptr_->getMapRange(min_range, max_range);
        else {
            // No map data yet — accept goal optimistically
            resolve_reason = "terrain map range not ready, accept goal optimistically";
            return true;
        }
        has_map_range = true;
    }
    catch (...)
    {
        // No map range available — accept goal optimistically
        resolve_reason = "terrain map range unavailable, accept goal optimistically";
        return true;
    }

    const Eigen::Vector2d raw_goal(raw_goal_x, raw_goal_y);
    const Eigen::Vector2d cur = cur_pos_;
    const double raw_dist = (raw_goal - cur).norm();
    if (!std::isfinite(raw_dist) || raw_dist < 1.0e-3)
    {
        active_goal << raw_goal_x, raw_goal_y, raw_goal_yaw, 1.0e-2;
        resolve_reason = "goal already at current pose";
        return true;
    }

    auto inside_map = [&](const Eigen::Vector2d &p) -> bool {
        return p.x() >= min_range.x() && p.x() <= max_range.x() &&
               p.y() >= min_range.y() && p.y() <= max_range.y();
    };

    auto sample_sdf = [&](double x, double y) -> double {
        try {
            return sdf_grid_ptr_->getValue("sdf_cpu", Eigen::Vector3d(x, y, 0.0));
        } catch (...) {
            return std::numeric_limits<double>::quiet_NaN();
        }
    };
    auto sample_risk = [&](double x, double y) -> double {
        if (!se2_grid_ptr_ || !se2_grid_ptr_->exists("risk"))
            return std::numeric_limits<double>::quiet_NaN();
        try {
            return se2_grid_ptr_->getValue("risk", Eigen::Vector3d(x, y, 0.0));
        } catch (...) {
            return std::numeric_limits<double>::quiet_NaN();
        }
    };
    auto sample_valid = [&](double x, double y) -> double {
        if (!se2_grid_ptr_ || !se2_grid_ptr_->exists("valid"))
            return std::numeric_limits<double>::quiet_NaN();
        try {
            return se2_grid_ptr_->getValue("valid", Eigen::Vector3d(x, y, 0.0));
        } catch (...) {
            return std::numeric_limits<double>::quiet_NaN();
        }
    };
    auto is_traversable = [&](const Eigen::Vector2d &p, double *sdf_out, double *risk_out) -> bool {
        if (!inside_map(p))
            return false;
        const double sdf = sample_sdf(p.x(), p.y());
        const double risk = sample_risk(p.x(), p.y());
        const double valid = sample_valid(p.x(), p.y());
        if (sdf_out) *sdf_out = sdf;
        if (risk_out) *risk_out = risk;
        // Primary: risk-based check (always available from fused_map)
        if (std::isfinite(risk) && risk >= goal_projection_max_risk_)
            return false;
        if (std::isfinite(valid) && valid < 0.5)
            return false;
        // Secondary: SDF check (only when SDF map is available)
        if (std::isfinite(sdf) && sdf <= goal_projection_min_sdf_)
            return false;
        // If neither risk nor SDF is available, reject unobserved cells
        if (!std::isfinite(risk) && !std::isfinite(sdf))
            return false;
        return true;
    };

    if (is_traversable(raw_goal, nullptr, nullptr))
    {
        active_goal << raw_goal_x, raw_goal_y, raw_goal_yaw, 1.0e-2;
        resolve_reason = "raw goal is traversable inside terrain map";
        return true;
    }

    // If the raw goal is outside the map or in an untraversable cell, accept it
    // directly. Kino-A* will search toward it treating out-of-map cells as free
    // space. The search horizon naturally limits how far ahead we plan, and the
    // continuous replan loop will keep refreshing the trajectory as the map moves.
    active_goal << raw_goal_x, raw_goal_y, raw_goal_yaw, 0.50;

    if (!inside_map(raw_goal))
    {
        char buf[192];
        std::snprintf(buf, sizeof(buf),
                      "distant goal (%.1f,%.1f) %.1fm away, accepted for search",
                      raw_goal_x, raw_goal_y, raw_dist);
        resolve_reason = buf;
    }
    else
    {
        char buf[192];
        std::snprintf(buf, sizeof(buf),
                      "goal in map but untraversable, accepted for search (risk/sdf will guide around)");
        resolve_reason = buf;
    }
    return true;
}

void ReplanFSM::ParkingCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
    const double raw_goal_x = msg->pose.position.x;
    const double raw_goal_y = msg->pose.position.y;
    const double raw_goal_yaw = wrapToPi(2.0 * std::atan2(msg->pose.orientation.z, msg->pose.orientation.w));

    if (!std::isfinite(raw_goal_x) || !std::isfinite(raw_goal_y) || !std::isfinite(raw_goal_yaw))
    {
        RCLCPP_WARN(node_->get_logger(), "Ignore invalid goal pose: x=%.3f y=%.3f yaw=%.3f",
                    raw_goal_x, raw_goal_y, raw_goal_yaw);
        return;
    }

    const rclcpp::Time now_t = node_->now();
    // Only block duplicate goals when actively executing a trajectory.
    // In WAIT_TARGET, always accept the new goal (user explicitly requested it).
    const bool accepting_state =
        (exec_state_ == EXEC_TRAJ || exec_state_ == REPLAN_TRAJ || exec_state_ == SEQUENTIAL_START);
    if (has_last_goal_ && accepting_state)
    {
        const double dt = (now_t - last_goal_time_).seconds();
        const double dxy = (Eigen::Vector2d(raw_goal_x, raw_goal_y) - last_goal_xy_).norm();
        const double dyaw = std::fabs(wrapToPi(raw_goal_yaw - last_goal_yaw_));
        // Shorter window (1.0s) — 3.0s was too generous and blocked valid new goals
        if (dt >= 0.0 && dt < 1.0 &&
            dxy < duplicate_goal_dist_ && dyaw < 0.30)
        {
            RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 1200,
                                 "Ignore duplicate goal trigger (dt=%.2fs, dxy=%.2fm, dyaw=%.2frad)",
                                 dt, dxy, dyaw);
            return;
        }
    }

    Eigen::Vector4d active_goal;
    active_goal << raw_goal_x, raw_goal_y, raw_goal_yaw, 0.50;
    const double raw_goal_dist = (Eigen::Vector2d(raw_goal_x, raw_goal_y) - cur_pos_).norm();
    if (std::isfinite(raw_goal_dist) && raw_goal_dist < 0.30)
        active_goal(3) = 1.0e-2;
    const std::string resolve_reason =
        "accepted raw goal; terrain feasibility will be resolved inside the serialized planner thread";

    raw_goal_xy_ = Eigen::Vector2d(raw_goal_x, raw_goal_y);
    raw_goal_yaw_ = raw_goal_yaw;
    const uint64_t new_goal_seq = goal_seq_.fetch_add(1, std::memory_order_relaxed) + 1;
    init_state_.head(2) = cur_pos_;
    init_state_(2) = cur_yaw_;
    init_state_(3) = cur_vel_;
    end_pt_ = active_goal;
    executing_goal_seq_ = 0;
    last_goal_xy_ = raw_goal_xy_;
    last_goal_yaw_ = raw_goal_yaw_;
    last_goal_time_ = now_t;
    has_last_goal_ = true;
    have_target_ = true;
    last_replan_reason_ = ReplanReason::NONE;
    plan_fail_count_ = 0;
    has_last_plan_try_time_ = false;
    collision_hit_count_ = 0;
    collision_with_obs_ = false;
    collision_time_to_hit_sec_ = std::numeric_limits<double>::infinity();
    collision_detected_max_risk_ = -std::numeric_limits<double>::infinity();
    track_mismatch_hit_count_ = 0;
    // Keep the goal callback non-invasive: the background planner may be
    // inside process(), so planner_ is only touched from the serialized FSM
    // planning paths below.
    publishGoalMarker(raw_goal_x, raw_goal_y, raw_goal_yaw_);
    RCLCPP_INFO(node_->get_logger(),
                "Accept mission goal #%lu (%.2f, %.2f, yaw=%.2f), dist=%.2fm, odom=%d; active local goal=(%.2f, %.2f, yaw=%.2f), %s",
                static_cast<unsigned long>(new_goal_seq),
                raw_goal_x, raw_goal_y, raw_goal_yaw_, raw_goal_dist, has_odom_ ? 1 : 0,
                end_pt_(0), end_pt_(1), end_pt_(2), resolve_reason.c_str());
    if (async_plan_running_.load(std::memory_order_relaxed))
    {
        RCLCPP_INFO(node_->get_logger(),
                    "New goal accepted while async planning is running; stale async result will be ignored");
    }
    if (has_odom_)
        changeFSMExecState(SEQUENTIAL_START, "FSM");
    else
        RCLCPP_WARN(node_->get_logger(), "Goal accepted but odometry is not ready yet; planning will start after first odom");
}

void ReplanFSM::launchAsyncReplan()
{
    // A finished async result must be consumed by finishAsyncReplan() before
    // another background process() starts.  Otherwise the main FSM may publish
    // planner_ while a new async thread is already mutating planner_, which can
    // corrupt trajectory containers and abort the process.
    if (async_plan_done_.load(std::memory_order_acquire))
        return;

    if (async_plan_running_.exchange(true, std::memory_order_acq_rel))
        return;  // already planning in background

    if (async_plan_thread_.joinable())
        async_plan_thread_.join();

    if (async_plan_done_.load(std::memory_order_acquire))
    {
        async_plan_running_.store(false, std::memory_order_release);
        return;
    }

    const double now_sec = node_->now().seconds();
    const Eigen::Vector2d raw_goal = raw_goal_xy_;
    const double raw_goal_yaw = raw_goal_yaw_;
    const uint64_t plan_goal_seq = goal_seq_.load(std::memory_order_relaxed);

    Eigen::Vector4d start_state;
    start_state(0) = cur_pos_(0);
    start_state(1) = cur_pos_(1);
    start_state(2) = cur_yaw_;
    start_state(3) = cur_vel_;

    double anchor_lead_sec = 0.0;
    const double raw_goal_dist = (raw_goal - cur_pos_).norm();
    if (use_predictive_replan_anchor_ &&
        std::isfinite(raw_goal_dist) && raw_goal_dist > 1.50 &&
        std::abs(cur_vel_) > 0.25)
    {
        anchor_lead_sec = std::clamp(replan_lead_time_sec_, 0.10, 0.35);
        const double signed_shift = std::clamp(
            cur_vel_ * anchor_lead_sec,
            -replan_anchor_max_dist_, replan_anchor_max_dist_);
        start_state(0) += signed_shift * std::cos(cur_yaw_);
        start_state(1) += signed_shift * std::sin(cur_yaw_);
    }
    const double replan_start_time = now_sec + anchor_lead_sec;
    async_plan_goal_seq_.store(plan_goal_seq, std::memory_order_relaxed);

    async_plan_start_time_ = node_->now();
    async_plan_success_.store(false, std::memory_order_relaxed);
    async_plan_done_.store(false, std::memory_order_release);
    const bool is_vehicle_moving = (exec_state_ == EXEC_TRAJ) || (std::abs(cur_vel_) > 0.08);

    async_plan_thread_ = std::thread([this, start_state, raw_goal, raw_goal_yaw, replan_start_time, is_vehicle_moving, plan_goal_seq]() {
        auto finish_async = [this](bool success) {
            async_plan_success_.store(success, std::memory_order_release);
            async_plan_done_.store(true, std::memory_order_release);
            async_plan_running_.store(false, std::memory_order_release);
        };

        try {
            if (plan_goal_seq != goal_seq_.load(std::memory_order_relaxed))
            {
                finish_async(false);
                return;
            }

            std::unique_lock<std::mutex> map_lock(map_mutex_);

            Eigen::Vector4d active_goal;
            std::string reason;
            if (!resolveMissionActiveGoal(raw_goal.x(), raw_goal.y(), raw_goal_yaw,
                                          active_goal, reason))
            {
                finish_async(false);
                return;
            }

            if (se2_grid_ptr_)
            {
                if (!se2_grid_ptr_->exists("zbx"))
                    se2_grid_ptr_->add("zbx", true, 0.0);
                if (!se2_grid_ptr_->exists("zby"))
                    se2_grid_ptr_->add("zby", true, 0.0);
            }

            Eigen::Vector4d safe_start = start_state;
            double start_risk = std::numeric_limits<double>::quiet_NaN();
            if (se2_grid_ptr_ && se2_grid_ptr_->exists("risk"))
            {
                try {
                    start_risk = se2_grid_ptr_->getValue(
                        "risk", Eigen::Vector3d(start_state(0), start_state(1), 0.0));
                } catch (...) {}
            }
            const bool start_in_hard_risk =
                start_rescue_enable_ &&
                std::isfinite(start_risk) &&
                start_risk >= start_rescue_trigger_risk_;

            // Normal snapping remains disabled while moving to avoid creating a
            // trajectory offset.  The exception is start rescue: if the real car
            // is already inside the hard risk band, a bounded projection gives
            // Kino-A* a valid root instead of entering a repeated failure loop.
            if ((snap_start_to_free_enable_ && !is_vehicle_moving) || start_in_hard_risk)
            {
                const double target_risk = start_in_hard_risk ? start_rescue_target_risk_ : 0.85;
                const double max_radius = start_in_hard_risk ? start_rescue_max_radius_ : 1.6;
                const double radius_step = start_in_hard_risk ? start_rescue_radius_step_ : 0.10;
                const double angle_step = start_in_hard_risk ? start_rescue_angle_step_deg_ : 10.0;
                const bool snapped_risk = snapStateToFreeRisk(
                    se2_grid_ptr_, safe_start,
                    target_risk, max_radius, radius_step, angle_step,
                    node_->get_logger(), start_in_hard_risk ? "risk_rescue_start" : "async_start");
                if (!snapped_risk && !start_in_hard_risk)
                {
                    snapStateToFreeSdf(
                        sdf_grid_ptr_, safe_start,
                        std::max(0.08, safe_margin_ + 0.02),
                        1.6, 0.10, 10.0,
                        node_->get_logger(), "async_start");
                }
                if (start_in_hard_risk && !snapped_risk)
                {
                    RCLCPP_WARN(node_->get_logger(),
                                "Start rescue failed: start risk %.3f >= %.3f, no cell below %.3f within %.2fm",
                                start_risk, start_rescue_trigger_risk_,
                                start_rescue_target_risk_, start_rescue_max_radius_);
                }
            }

            if (plan_goal_seq != goal_seq_.load(std::memory_order_relaxed))
            {
                finish_async(false);
                return;
            }

            planner_.setTarget(active_goal(0), active_goal(1), active_goal(2));
            // Do NOT clear visualization here, it causes flickering during continuous planning.
            planner_.setInitStateAndInput(safe_start, replan_start_time);
            planner_.setParkingEnd(active_goal);

            // Always use full Kino-A* to guarantee smooth and collision-free paths.
            // The lightweight planner causes polynomial oscillations (wavy motion) 
            // and does not update the RViz visualization path.
            const bool success = planner_.process();
            finish_async(success &&
                (plan_goal_seq == goal_seq_.load(std::memory_order_relaxed)));
            return;
        }
        catch (const std::exception &e)
        {
            RCLCPP_ERROR(node_->get_logger(), "Async plan exception: %s", e.what());
            finish_async(false);
            return;
        }
        catch (...)
        {
            RCLCPP_ERROR(node_->get_logger(), "Async plan unknown exception");
            finish_async(false);
            return;
        }
    });
}

void ReplanFSM::finishAsyncReplan()
{
    if (!async_plan_done_.exchange(false, std::memory_order_acq_rel))
        return;

    if (async_plan_thread_.joinable())
        async_plan_thread_.join();

    const uint64_t result_seq = async_plan_goal_seq_.load(std::memory_order_relaxed);
    const uint64_t current_seq = goal_seq_.load(std::memory_order_relaxed);
    if (result_seq != current_seq)
    {
        RCLCPP_WARN(node_->get_logger(),
                    "Ignore stale async replan result for goal #%lu; current goal is #%lu",
                    static_cast<unsigned long>(result_seq),
                    static_cast<unsigned long>(current_seq));
        async_plan_success_.store(false, std::memory_order_relaxed);
        return;
    }

    if (async_plan_success_.load(std::memory_order_acquire))
    {
        executing_goal_seq_ = result_seq;
        last_plan_try_time_ = node_->now();
        has_last_plan_try_time_ = true;
        
        planner_.publishSimTraj2Controller();
        start_world_time_ = planner_.getTrajStartTime();
        if (!std::isfinite(start_world_time_) || start_world_time_ <= 0.0)
            start_world_time_ = node_->now().seconds() + time_budget_sec_;

        plan_fail_count_ = 0;
        collision_hit_count_ = 0;
        collision_with_obs_ = false;
        collision_time_to_hit_sec_ = std::numeric_limits<double>::infinity();
        collision_detected_max_risk_ = -std::numeric_limits<double>::infinity();
        track_mismatch_hit_count_ = 0;
        last_replan_reason_ = ReplanReason::NONE;
        traj_start_time_ = node_->now();

        RCLCPP_INFO_THROTTLE(node_->get_logger(), *node_->get_clock(), 1500,
                             "Async replan succeeded, trajectory published (%.2fs total)",
                             planner_.getDuration());
        if (exec_state_ == SEQUENTIAL_START)
            changeFSMExecState(EXEC_TRAJ, "FSM");
    }
    else
    {
        plan_fail_count_++;
        RCLCPP_WARN(node_->get_logger(),
                    "Async replan failed (%d consecutive failures)",
                    plan_fail_count_);
    }
}

void ReplanFSM::execFSMCallback()
{
    static int fsm_num = 0;
    fsm_num++;
    if (fsm_num == 50)
    {
      if (!have_target_)
      {
        RCLCPP_INFO_THROTTLE(
            node_->get_logger(), *node_->get_clock(), 2000,
            "wait for goal or trigger. odom=%d goal_seq=%lu exec_goal_seq=%lu",
            has_odom_ ? 1 : 0,
            static_cast<unsigned long>(goal_seq_.load(std::memory_order_relaxed)),
            static_cast<unsigned long>(executing_goal_seq_));
      }
      fsm_num = 0;
    }

    if (async_plan_done_)
    {
        finishAsyncReplan();
    }

    switch (exec_state_)
    {
        case INIT:
        {
            changeFSMExecState(WAIT_TARGET, "FSM");
            break;
        }

        case WAIT_TARGET:
        {
            if(!have_target_ /*|| !have_trigger_*/)
                goto force_return;

            if (!has_odom_)
            {
                RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 1000,
                                     "Have goal but odometry is not ready; hold WAIT_TARGET");
                goto force_return;
            }

            else
            {
                changeFSMExecState(SEQUENTIAL_START, "FSM");
            }
            break;
        }

        case SEQUENTIAL_START:
        {
            if (!has_odom_)
            {
                RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 1000,
                                     "Cannot start planning before odometry arrives");
                break;
            }

            if (async_plan_running_)
            {
                const double plan_elapsed = (node_->now() - async_plan_start_time_).seconds();
                if (plan_elapsed > 5.0)
                {
                    RCLCPP_ERROR_THROTTLE(node_->get_logger(), *node_->get_clock(), 1000,
                                          "Previous async planning still running after %.2fs; queueing new goal until planner exits",
                                          plan_elapsed);
                }
                else
                {
                    RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 1000,
                                         "Waiting for previous async planner thread before starting new goal");
                }
                break;
            }

            auto now_t = node_->now();
            const double fail_scale = 1.0 + 0.30 * static_cast<double>(std::min(plan_fail_count_, 8));
            const double retry_interval = std::min(
                replan_retry_backoff_max_sec_,
                std::max(0.05, replan_retry_interval_sec_ * fail_scale));
            if (has_last_plan_try_time_ && (now_t - last_plan_try_time_).seconds() < retry_interval)
            {
                break;
            }
            last_plan_try_time_ = now_t;
            has_last_plan_try_time_ = true;

            init_state_ << cur_pos_, cur_yaw_, cur_vel_;

            launchAsyncReplan();

            break;
        }

        case REPLAN_TRAJ:
        {
            // Async planning mode: launch background planning and return to
            // EXEC_TRAJ immediately. The result is published by finishAsyncReplan()
            // when the background thread completes.
            const double raw_goal_dist_pre = (cur_pos_ - raw_goal_xy_).norm();
            if(raw_goal_dist_pre < 0.25 && std::abs(cur_vel_) < 0.18)
            {
                RCLCPP_INFO(node_->get_logger(),
                            "Mission goal already at current pose before replan: dist=%.2f vel=%.2f",
                            raw_goal_dist_pre, cur_vel_);
                changeFSMExecState(WAIT_TARGET, "FSM");
                have_target_ = false;
                executing_goal_seq_ = 0;
                goto force_return;
            }
            launchAsyncReplan();
            changeFSMExecState(EXEC_TRAJ, "FSM");
            break;
        }

        case EXEC_TRAJ:
        {
            auto t_now = node_->now();

            // Timeout guard: if async planning hangs, reset after 5s.
            if (async_plan_running_)
            {
                double plan_elapsed = (t_now - async_plan_start_time_).seconds();
                if (plan_elapsed > 5.0)
                {
                    RCLCPP_ERROR_THROTTLE(node_->get_logger(), *node_->get_clock(), 1000,
                                          "Async planning still running after %.2fs; keeping planner serialized",
                                          plan_elapsed);
                }
            }

            const double active_goal_dist = (cur_pos_ - end_pt_.head(2)).norm();
            const double raw_goal_dist = (cur_pos_ - raw_goal_xy_).norm();
            const double exec_elapsed = t_now.seconds() - start_world_time_;
            double track_err = 0.0;
            double track_yaw_err = 0.0;
            Eigen::Vector4d pred_now;
            if (!async_plan_running_ && planner_.getPredictState(t_now.seconds(), pred_now))
            {
                track_err = (pred_now.head(2) - cur_pos_).norm();
                track_yaw_err = std::fabs(wrapToPi(pred_now(2) - cur_yaw_));
            }
            const bool dwell_ok = exec_elapsed > exec_min_dwell_sec_;
            const bool cooldown_ok = (!has_last_replan_time_) ||
                                     ((t_now - last_replan_time_).seconds() > replan_cooldown_sec_);
            const double fail_scale = 1.0 + 0.50 * static_cast<double>(std::min(plan_fail_count_, 8));
            const double retry_interval = std::min(
                replan_retry_backoff_max_sec_,
                std::max(0.20, replan_retry_interval_sec_ * fail_scale));
            const bool retry_ok = (!has_last_plan_try_time_) ||
                                  ((t_now - last_plan_try_time_).seconds() > retry_interval);
            const uint64_t current_goal_seq = goal_seq_.load(std::memory_order_relaxed);
            const bool executing_current_goal =
                executing_goal_seq_ != 0 && executing_goal_seq_ == current_goal_seq;

            const double track_err_entry = track_err_replan_dist_;
            const double track_yaw_entry = track_err_replan_yaw_;
            const bool track_mismatch_now =
                (track_err > track_err_entry) || (track_yaw_err > track_yaw_entry);
            const bool active_goal_reached =
                active_goal_dist < 0.60 && raw_goal_dist > std::max(0.65, end_replan_goal_dist_);

            if (track_err_replan_enable_ && exec_elapsed > track_err_replan_min_elapsed_)
            {
                if (track_mismatch_now)
                {
                    track_mismatch_hit_count_ = std::min(track_err_replan_hits_, track_mismatch_hit_count_ + 1);
                }
                else
                {
                    track_mismatch_hit_count_ = std::max(0, track_mismatch_hit_count_ - 1);
                }
            }
            else
            {
                track_mismatch_hit_count_ = 0;
            }

            // Track-mismatch replan: when the vehicle has drifted significantly
            // off the reference trajectory, force a replan anchored at the
            // current vehicle position.  This is the "disaster recovery" path
            // — MPC cannot self-correct large tracking errors.
            if (track_err_replan_enable_ &&
                dwell_ok && cooldown_ok && retry_ok &&
                track_mismatch_hit_count_ >= track_err_replan_hits_)
            {
                RCLCPP_WARN(node_->get_logger(),
                            "Track mismatch replan: err=%.2fm yaw=%.2frad hits=%d",
                            track_err, track_yaw_err, track_mismatch_hit_count_);
                last_replan_time_ = t_now;
                has_last_replan_time_ = true;
                last_replan_reason_ = ReplanReason::TRACK_MISMATCH;
                track_mismatch_hit_count_ = 0;
                launchAsyncReplan();
                break;
            }

            // Replan after trajectory expiry or nearing end (preemptive).
            const double traj_dur = planner_.getDuration();
            const double traj_remaining = traj_dur - exec_elapsed;
            const bool traj_nearing_end =
                (traj_dur > 0.05) &&
                (preemptive_replan_time_sec_ > 1e-3) &&
                (traj_remaining > -0.05) &&
                (traj_remaining < preemptive_replan_time_sec_);
            const bool mission_reached =
                executing_current_goal &&
                ((raw_goal_dist < 0.18) ||
                (raw_goal_dist < 0.28 &&
                 (std::abs(cur_vel_) < 0.18 || traj_remaining < -0.10)));
            // Suppress replanning when approaching the goal — let the MPC
            // decelerate and stop cleanly without a new trajectory disrupting.
            if(enable_end_of_traj_replan_ && dwell_ok && cooldown_ok && retry_ok &&                (active_goal_reached || (traj_nearing_end && !mission_reached)))
            {
                RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 1000,
                                     "Current trajectory nearing/sub-goal reached (remain=%.2fs, dur=%.2f, elapsed=%.2f, active_dist=%.2f, mission_dist=%.2f), trigger sliding replan",
                                     traj_remaining, traj_dur, exec_elapsed, active_goal_dist, raw_goal_dist);
                last_replan_time_ = t_now;
                has_last_replan_time_ = true;
                last_replan_reason_ = ReplanReason::TRAJ_EXPIRED;
                launchAsyncReplan();
                break;
            }
            const bool traj_expired = (traj_dur > 0.05) && (exec_elapsed > (traj_dur + traj_expired_replan_margin_sec_));
            if(enable_end_of_traj_replan_ && dwell_ok && cooldown_ok && retry_ok &&                traj_expired && !mission_reached)
            {
                     RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 1000,
                                                 "Current trajectory expired by %.2fs (dur=%.2f, elapsed=%.2f, active_dist=%.2f, mission_dist=%.2f), trigger replan",
                                                 exec_elapsed - traj_dur, traj_dur, exec_elapsed, active_goal_dist, raw_goal_dist);
                last_replan_time_ = t_now;
                has_last_replan_time_ = true;
                last_replan_reason_ = ReplanReason::TRAJ_EXPIRED;
                launchAsyncReplan();
                break;
            }

            const bool hazard_near =
                collision_with_obs_ &&
                std::isfinite(collision_time_to_hit_sec_) &&
                collision_time_to_hit_sec_ <= collision_near_horizon_sec_;
            const bool hazard_cooldown_ok =
                hazard_near || (!has_last_replan_time_) ||
                ((t_now - last_replan_time_).seconds() >
                 (collision_with_obs_ ? collision_preemptive_cooldown_sec_ : replan_cooldown_sec_));
            const bool hazard_retry_ok = hazard_near || retry_ok;
            const double hazard_min_elapsed =
                hazard_near ? std::min(0.12, collision_replan_min_elapsed_sec_)
                            : std::max(time_budget_sec_, collision_replan_min_elapsed_sec_);

            if(collision_with_obs_ && hazard_cooldown_ok && hazard_retry_ok &&
                    exec_elapsed > hazard_min_elapsed)
            {
                RCLCPP_WARN(node_->get_logger(),
                            "Hazard-triggered replan: time_to_hit=%.2fs max_risk=%.3f near=%d elapsed=%.2f",
                            collision_time_to_hit_sec_, collision_detected_max_risk_,
                            hazard_near ? 1 : 0, exec_elapsed);
                last_replan_time_ = t_now;
                has_last_replan_time_ = true;
                last_replan_reason_ = ReplanReason::HAZARD;
                collision_with_obs_ = false;
                collision_hit_count_ = 0;
                collision_time_to_hit_sec_ = std::numeric_limits<double>::infinity();
                collision_detected_max_risk_ = -std::numeric_limits<double>::infinity();
                launchAsyncReplan();
                break;
            }
            if(mission_reached)
            {
                RCLCPP_INFO(node_->get_logger(),
                            "Mission goal reached: dist=%.2f vel=%.2f remain=%.2f goal_seq=%lu exec_goal_seq=%lu",
                            raw_goal_dist, cur_vel_, traj_remaining,
                            static_cast<unsigned long>(current_goal_seq),
                            static_cast<unsigned long>(executing_goal_seq_));
                changeFSMExecState(WAIT_TARGET, "FSM");
                have_target_ = false;
                executing_goal_seq_ = 0;
                goto force_return;
            }

            // Periodic status log for debugging replan conditions
            {
                static int exec_status_count = 0;
                if (++exec_status_count % 150 == 1)
                {
                    const double traj_dur = planner_.getDuration();
                    const double traj_remaining = traj_dur - exec_elapsed;
                    RCLCPP_DEBUG(node_->get_logger(),
                                 "FSM EXEC elapsed=%.2f traj_dur=%.2f remain=%.2f dist_to_goal=%.2f vel=%.2f fails=%d collision=%d state=%d",
                                 exec_elapsed, traj_dur, traj_remaining, raw_goal_dist, cur_vel_,
                                 plan_fail_count_, collision_with_obs_ ? 1 : 0, static_cast<int>(exec_state_));
                }
            }

            break;
        }
    }
    force_return:;
}

void ReplanFSM::changeFSMExecState(FSM_EXEC_STATE new_state, std::string pos_call)
{
    static std::string state_str[8] = {"INIT", "WAIT_TARGET", "REPLAN_TRAJ", "EXEC_TRAJ", "SEQUENTIAL_START"};
    int pre_s = int(exec_state_);

    exec_state_ = new_state;
    std::cout << "[" + pos_call + "]: from " + state_str[pre_s] + " to " + state_str[int(new_state)] << std::endl;    
}

void ReplanFSM::checkCollisionCallback()
{
    if (exec_state_ != EXEC_TRAJ)
    {
        collision_with_obs_ = false;
        collision_hit_count_ = 0;
        collision_time_to_hit_sec_ = std::numeric_limits<double>::infinity();
        collision_detected_max_risk_ = -std::numeric_limits<double>::infinity();
        return;
    }

    // Skip collision lookahead during async planning — the trajectory is being
    // rebuilt in a background thread and optTraj may be in an inconsistent state.
    if (async_plan_running_)
        return;

    std::unique_lock<std::mutex> map_lock(map_mutex_, std::try_to_lock);
    if (!map_lock.owns_lock())
        return;

    Eigen::Vector4d state;
    double t_now = node_->now().seconds();
    if(t_now - start_world_time_ < 0.1){
        return;
    }
    const double rel_now = std::max(0.0, t_now - start_world_time_);
    const double total_dur = planner_.getDuration();
    if (total_dur <= 0.0)
    {
        collision_with_obs_ = false;
        collision_hit_count_ = 0;
        collision_time_to_hit_sec_ = std::numeric_limits<double>::infinity();
        collision_detected_max_risk_ = -std::numeric_limits<double>::infinity();
        return;
    }

    auto sample_pose_risk = [&](double x, double y, double yaw) -> double {
        if (!se2_grid_ptr_ || !se2_grid_ptr_->exists("risk"))
            return std::numeric_limits<double>::quiet_NaN();

        const double c = std::cos(yaw);
        const double s = std::sin(yaw);
        const Eigen::Vector2d base(x, y);
        const std::array<Eigen::Vector2d, 7> offsets = {
            Eigen::Vector2d(0.00, 0.00),
            Eigen::Vector2d(0.36, 0.00),
            Eigen::Vector2d(0.62, 0.00),
            Eigen::Vector2d(0.20, 0.36),
            Eigen::Vector2d(0.20, -0.36),
            Eigen::Vector2d(-0.28, 0.34),
            Eigen::Vector2d(-0.28, -0.34),
        };

        double max_risk = -std::numeric_limits<double>::infinity();
        for (const auto &off : offsets)
        {
            const Eigen::Vector2d p(
                base.x() + c * off.x() - s * off.y(),
                base.y() + s * off.x() + c * off.y());
            try {
                const double risk = se2_grid_ptr_->getValue("risk", Eigen::Vector3d(p.x(), p.y(), 0.0));
                if (std::isfinite(risk))
                    max_risk = std::max(max_risk, risk);
            } catch (...) {
            }
        }
        return max_risk;
    };

    Eigen::Vector4d pred_now;
    bool track_gap_large = false;
    if (planner_.getPredictState(t_now, pred_now))
    {
        const double track_gap = (pred_now.head(2) - cur_pos_).norm();
        if (track_gap > collision_eval_track_err_max_)
        {
            track_gap_large = true;
            RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 1200,
                                 "Track gap %.2f > %.2f; check ego footprint risk instead of skipping collision lookahead",
                                 track_gap, collision_eval_track_err_max_);
        }
    }

    const double lookahead = std::max(0.4, collision_check_lookahead_sec_);
    const double rel_end = std::min(total_dur, rel_now + lookahead);
    bool near_collision = false;
    double time_to_hit = std::numeric_limits<double>::infinity();
    double min_pred_sdf = std::numeric_limits<double>::infinity();
    double max_pred_risk = -std::numeric_limits<double>::infinity();
    const double collision_trigger_margin = std::max(0.0, safe_margin_ + collision_replan_margin_);
    const double collision_enter_threshold = std::max(0.0, collision_trigger_margin - 0.01);
    const bool has_risk = se2_grid_ptr_ && se2_grid_ptr_->exists("risk");
    const bool has_sdf = sdf_grid_ptr_ && sdf_grid_ptr_->exists("sdf_cpu");

    if (has_risk)
    {
        const double ego_lookahead = track_gap_large ? 1.40 : 0.80;
        const double ego_step = 0.20;
        for (double ds = 0.0; ds <= ego_lookahead + 1e-6; ds += ego_step)
        {
            const double x = cur_pos_.x() + ds * std::cos(cur_yaw_);
            const double y = cur_pos_.y() + ds * std::sin(cur_yaw_);
            const double risk = sample_pose_risk(x, y, cur_yaw_);
            if (std::isfinite(risk))
            {
                max_pred_risk = std::max(max_pred_risk, risk);
                if (risk >= risk_replan_threshold_)
                {
                    near_collision = true;
                    time_to_hit = 0.0;
                    break;
                }
            }
        }
    }

    const double traj_step = (rel_end - rel_now > 2.0) ? 0.15 : 0.10;
    for (double rel_t = rel_now; rel_t < rel_end; rel_t += traj_step)
    {
        if (near_collision)
            break;
        const double abs_t = start_world_time_ + rel_t;
        if(!planner_.getPredictState(abs_t, state))
            continue;

        const Eigen::Vector3d query(state(0), state(1), 0.0);

        // Primary: risk-based collision check
        if (has_risk)
        {
            double risk_val = sample_pose_risk(state(0), state(1), state(2));
            if (std::isfinite(risk_val))
            {
                max_pred_risk = std::max(max_pred_risk, risk_val);
                if (risk_val >= risk_replan_threshold_)
                {
                    near_collision = true;
                    time_to_hit = std::max(0.0, rel_t - rel_now);
                    break;
                }
            }
        }

        // Secondary: SDF-based collision check (fallback)
        if (!near_collision && has_sdf)
        {
            double sdf_val = std::numeric_limits<double>::quiet_NaN();
            try {
                sdf_val = sdf_grid_ptr_->getValue("sdf_cpu", query);
            } catch (...) {}
            if (std::isfinite(sdf_val))
            {
                min_pred_sdf = std::min(min_pred_sdf, sdf_val);
                if (sdf_val <= collision_enter_threshold)
                {
                    near_collision = true;
                    time_to_hit = std::max(0.0, rel_t - rel_now);
                    break;
                }
            }
        }
    }

    if (near_collision)
    {
        const int required_hits = std::max(1, std::min(collision_replan_hits_, risk_replan_hits_));
        collision_hit_count_ = std::min(required_hits, collision_hit_count_ + 1);
    }
    else
    {
        collision_hit_count_ = std::max(0, collision_hit_count_ - 1);
    }
    const int required_hits = std::max(1, std::min(collision_replan_hits_, risk_replan_hits_));
    collision_with_obs_ = (collision_hit_count_ >= required_hits);
    if (collision_with_obs_)
    {
        collision_time_to_hit_sec_ = time_to_hit;
        collision_detected_max_risk_ = max_pred_risk;
    }
    else
    {
        collision_time_to_hit_sec_ = std::numeric_limits<double>::infinity();
        collision_detected_max_risk_ = -std::numeric_limits<double>::infinity();
    }

    if (near_collision)
    {
        RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 1000,
                             "Predict collision ahead: t_hit=%.2fs lookahead=%.2fs max_risk=%.3f min_sdf=%.3f trigger_sdf=%.3f risk_thr=%.3f hits=%d/%d",
                             time_to_hit, lookahead, max_pred_risk, min_pred_sdf,
                             collision_enter_threshold, risk_replan_threshold_,
                             collision_hit_count_, required_hits);
    }

    return;
}

void ReplanFSM::continuousReplanCallback()
{
    if (!continuous_replan_enable_)
        return;
    if (exec_state_ != EXEC_TRAJ)
        return;
    // Don't trigger another replan while one is already running in background.
    if (async_plan_running_)
        return;

    auto t_now = node_->now();
    const double exec_elapsed = t_now.seconds() - start_world_time_;
    const double traj_dur = planner_.getDuration();
    const double traj_remaining = traj_dur - exec_elapsed;

    const bool traj_expired = (traj_dur > 0.05) && (exec_elapsed > traj_dur + 0.1);
    const bool hazard_active = collision_with_obs_;
    const bool hazard_near =
        hazard_active &&
        std::isfinite(collision_time_to_hit_sec_) &&
        collision_time_to_hit_sec_ <= collision_near_horizon_sec_;
    const bool dwell_ok = hazard_near || exec_elapsed > exec_min_dwell_sec_;
    const double cooldown =
        hazard_active ? collision_preemptive_cooldown_sec_ : replan_cooldown_sec_;
    const bool cooldown_ok = traj_expired || hazard_near || (!has_last_replan_time_) ||
                             ((t_now - last_replan_time_).seconds() > cooldown);
    const double fail_scale = 1.0 + 0.50 * static_cast<double>(std::min(plan_fail_count_, 8));
    const double retry_interval = std::min(
        replan_retry_backoff_max_sec_,
        std::max(0.10, replan_retry_interval_sec_ * fail_scale));
    const bool retry_ok = traj_expired || hazard_near || (!has_last_plan_try_time_) ||
                          ((t_now - last_plan_try_time_).seconds() > retry_interval);

    if (!dwell_ok || !cooldown_ok || !retry_ok)
    {
        RCLCPP_DEBUG_THROTTLE(node_->get_logger(), *node_->get_clock(), 3000,
                              "FSM continuous skip: dwell=%d cooldown=%d retry=%d elapsed=%.2f traj_dur=%.2f fails=%d",
                              dwell_ok ? 1 : 0, cooldown_ok ? 1 : 0, retry_ok ? 1 : 0,
                              exec_elapsed, traj_dur, plan_fail_count_);
        return;
    }

    const double raw_goal_dist = (cur_pos_ - raw_goal_xy_).norm();
    if (raw_goal_dist < 1.20 && !hazard_active && !traj_expired)
        return;

    const bool traj_nearing_end =
        (traj_dur > 0.05) &&
        (traj_remaining > -0.05) &&
        (traj_remaining < preemptive_replan_time_sec_) &&
        (raw_goal_dist > std::max(0.65, end_replan_goal_dist_));

    if (traj_expired || traj_nearing_end || hazard_active)
    {
        if (plan_fail_count_ < 10)
        {
            last_replan_time_ = t_now;
            has_last_replan_time_ = true;
            last_replan_reason_ = hazard_active ? ReplanReason::HAZARD : ReplanReason::TRAJ_EXPIRED;
            if (hazard_active)
            {
                collision_with_obs_ = false;
                collision_hit_count_ = 0;
                collision_time_to_hit_sec_ = std::numeric_limits<double>::infinity();
                collision_detected_max_risk_ = -std::numeric_limits<double>::infinity();
            }
            launchAsyncReplan();
        }
        else
        {
            plan_fail_count_ = 0;
            changeFSMExecState(SEQUENTIAL_START, "FSM");
        }
    }
}
