#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <queue>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#include "terrain_nav/reverse_arc.hpp"

#include <Eigen/Core>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/executors/multi_threaded_executor.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <se2_grid_core/SE2Grid.hpp>
#include <se2_grid_msgs/msg/se2_grid.hpp>
#include <se2_grid_ros/SE2GridRosConverter.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

namespace
{
constexpr double kInf = std::numeric_limits<double>::infinity();

double wrapAngle(double a)
{
  while (a > M_PI) a -= 2.0 * M_PI;
  while (a < -M_PI) a += 2.0 * M_PI;
  return a;
}

double clampAbs(double value, double limit)
{
  return std::clamp(value, -std::abs(limit), std::abs(limit));
}

struct Pose2
{
  double x = 0.0;
  double y = 0.0;
  double yaw = 0.0;
  double s = 0.0;
  int direction = 1;
};

struct OdomState
{
  double x = 0.0;
  double y = 0.0;
  double yaw = 0.0;
  double v = 0.0;
  double yaw_rate = 0.0;
  rclcpp::Time stamp;
  bool valid = false;
};

struct CostQuery
{
  bool known = false;
  bool lethal = false;
  double risk = 0.0;
  double clearance = kInf;
  double comfort_clearance = kInf;
  double cost = 0.0;
};

// Readers retain a complete generation without holding the search mutex.
struct ControlMapSnapshot
{
  se2_grid::SE2Grid grid;
  rclcpp::Time stamp;
  double min_x = 0.0, min_y = 0.0, resolution = 0.2;
  int width = 0, height = 0;
  std::vector<uint8_t> costs;
  std::vector<float> unknown_baseline, clearance, comfort_clearance;
};

struct LiveObstacleSnapshot
{
  double min_x = 0.0, min_y = 0.0, resolution = 0.2, stamp_sec = 0.0;
  int width = 0, height = 0;
  std::vector<uint8_t> hard;

  bool occupied(double x, double y) const
  {
    const int ix = static_cast<int>(std::floor((x - min_x) / resolution));
    const int iy = static_cast<int>(std::floor((y - min_y) / resolution));
    return ix >= 0 && ix < width && iy >= 0 && iy < height &&
      hard[static_cast<size_t>(iy * width + ix)] != 0;
  }
};
}  // namespace

class TerrainNavNode final : public rclcpp::Node
{
public:
  TerrainNavNode() : Node("terrain_nav_node"), rng_(7)
  {
    declareParameters();
    loadParameters();

    auto sensor_qos = rclcpp::SensorDataQoS();
    map_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    odom_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    live_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    control_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    rclcpp::SubscriptionOptions map_options, odom_options, live_options;
    map_options.callback_group = map_group_;
    odom_options.callback_group = odom_group_;
    live_options.callback_group = live_group_;
    map_sub_ = create_subscription<se2_grid_msgs::msg::SE2Grid>(
      fused_map_topic_, rclcpp::QoS(rclcpp::KeepLast(1)),
      std::bind(&TerrainNavNode::mapCallback, this, std::placeholders::_1), map_options);
    if (live_obstacle_enable_) {
      cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        live_cloud_topic_, sensor_qos,
        std::bind(&TerrainNavNode::cloudCallback, this, std::placeholders::_1), live_options);
    }
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      odom_topic_, rclcpp::QoS(rclcpp::KeepLast(5)),
      std::bind(&TerrainNavNode::odomCallback, this, std::placeholders::_1), odom_options);
    goal_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      "/goal_pose", rclcpp::QoS(rclcpp::KeepLast(1)),
      std::bind(&TerrainNavNode::goalCallback, this, std::placeholders::_1));
    legacy_goal_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      "/move_base_simple/goal", rclcpp::QoS(rclcpp::KeepLast(1)),
      std::bind(&TerrainNavNode::goalCallback, this, std::placeholders::_1));

    auto marker_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
    cmd_pub_ = create_publisher<geometry_msgs::msg::Twist>(cmd_topic_, rclcpp::QoS(rclcpp::KeepLast(1)).reliable());
    path_pub_ = create_publisher<nav_msgs::msg::Path>("/terrain_nav/path", rclcpp::QoS(rclcpp::KeepLast(1)).reliable());
    live_obstacle_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>(
      "/terrain_nav/live_obstacles", rclcpp::QoS(rclcpp::KeepLast(1)).reliable());
    ref_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("/reference_path", marker_qos);
    guidance_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("/visualization/terrain_guidance_path", marker_qos);
    goal_marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("/terrain_nav/active_goal", marker_qos);
    legacy_goal_marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("/visualization/goal_marker", marker_qos);

    nominal_v_.assign(mppi_steps_, desired_speed_);
    nominal_steer_.assign(mppi_steps_, 0.0);

    control_timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(control_period_)),
      std::bind(&TerrainNavNode::controlTimer, this), control_group_);

    planner_thread_ = std::thread(&TerrainNavNode::plannerLoop, this);

    RCLCPP_INFO(
      get_logger(),
      "terrain_nav ready: odom=%s fused_map=%s goal=/goal_pose cmd=%s wheel_base=%.3f max_steer=%.3f",
      odom_topic_.c_str(), fused_map_topic_.c_str(), cmd_topic_.c_str(), wheel_base_, max_steer_);
  }

  ~TerrainNavNode() override
  {
    {
      std::lock_guard<std::mutex> lock(planner_mutex_);
      planner_stop_ = true;
      planner_wakeup_ = true;
    }
    planner_cv_.notify_one();
    if (planner_thread_.joinable()) planner_thread_.join();
  }

private:
  void declareParameters()
  {
    declare_parameter("odom_topic", "/Odometry");
    declare_parameter("fused_map_topic", "/car_0_terrain_analyzer/fused_map");
    declare_parameter("cmd_topic", "/hunter_se/cmd_vel");
    declare_parameter("frame_id", "map");
    declare_parameter("live_obstacle_enable", false);
    declare_parameter("live_cloud_topic", "/cloud_registered");
    declare_parameter("live_obstacle_min_height", 0.38);
    declare_parameter("live_obstacle_neighbor_radius", 0.75);
    declare_parameter("live_obstacle_min_hits", 3);
    declare_parameter("live_obstacle_max_age_sec", 0.60);
    declare_parameter("live_obstacle_hold_sec", 1.20);
    declare_parameter("live_obstacle_inflation_radius", 0.45);

    declare_parameter("wheel_base", 0.549);
    declare_parameter("max_steer", 0.5236);
    declare_parameter("max_steer_rate", 2.6);
    declare_parameter("max_speed", 2.0);
    declare_parameter("min_speed", 0.0);
    declare_parameter("reverse_enable", true);
    declare_parameter("reverse_max_distance", 4.0);
    declare_parameter("reverse_max_speed", 0.55);
    declare_parameter("desired_speed", 1.0);
    declare_parameter("min_tracking_speed", 0.35);
    declare_parameter("goal_slowdown_distance", 0.90);
    declare_parameter("max_accel", 2.0);

    declare_parameter("footprint_front", 0.42);
    declare_parameter("footprint_back", 0.42);
    declare_parameter("footprint_half_width", 0.313);
    declare_parameter("footprint_margin", 0.05);

    declare_parameter("yaw_bins", 72);
    declare_parameter("primitive_length", 0.42);
    declare_parameter("integration_step", 0.10);
    declare_parameter("max_search_time_ms", 150.0);
    declare_parameter("max_iterations", 26000);
    declare_parameter("goal_tolerance", 0.42);
    declare_parameter("goal_reached_distance", 0.55);
    declare_parameter("goal_yaw_tolerance", 0.95);
    declare_parameter("local_goal_distance", 8.0);
    declare_parameter("local_goal_bearing_limit", 1.20);
    declare_parameter("direct_goal_bearing_limit", 1.45);
    declare_parameter("guide_search_enable", true);
    declare_parameter("guide_max_expansions", 9000);
    declare_parameter("guide_min_distance", 2.0);
    declare_parameter("guide_boundary_margin", 0.8);
    declare_parameter("guide_progress_weight", 2.2);
    declare_parameter("guide_goal_weight", 0.48);
    declare_parameter("guide_lateral_weight", 0.22);
    declare_parameter("guide_clearance_weight", 0.85);
    declare_parameter("guide_path_cost_weight", 0.055);
    declare_parameter("guide_follow_weight", 1.15);
    declare_parameter("guide_heading_weight", 0.22);
    declare_parameter("direct_curve_enable", true);
    declare_parameter("direct_curve_max_length_ratio", 1.55);
    declare_parameter("direct_curve_max_lateral_ratio", 0.55);
    declare_parameter("direct_curve_clearance_weight", 5.0);

    declare_parameter("risk_free_threshold", 0.08);
    declare_parameter("risk_hard_threshold", 0.58);
    declare_parameter("risk_query_mode", "flat_2d");
    declare_parameter("se2_risk_layer", "risk");
    declare_parameter("hard_risk_layer", "baseline_risk");
    declare_parameter("se2_risk_lethal", false);
    declare_parameter("unknown_cost", 60.0);
    declare_parameter("allow_unknown", true);
    declare_parameter("unknown_robot_bubble_radius", 0.0);
    declare_parameter("unknown_robot_bubble_max_baseline", 0.0);
    declare_parameter("max_odom_age_sec", 0.5);
    declare_parameter("max_map_age_sec", 1.0);
    declare_parameter("stuck_timeout_sec", 4.0);
    declare_parameter("stuck_progress_m", 0.12);
    declare_parameter("stuck_command_mps", 0.25);
    declare_parameter("risk_weight", 3.0);
    declare_parameter("clearance_distance", 0.85);
    declare_parameter("clearance_weight", 2.8);
    declare_parameter("hard_clearance_floor", -1.0);
    declare_parameter("comfort_clearance_distance", 1.35);
    declare_parameter("comfort_clearance_weight", 2.2);
    declare_parameter("comfort_clearance_risk_threshold", 0.18);
    declare_parameter("steer_weight", 0.22);
    declare_parameter("steer_change_weight", 0.20);
    declare_parameter("steer_reversal_weight", 0.0);
    declare_parameter("smoothing_passes", 3);
    declare_parameter("shortcut_smoothing_enable", true);
    declare_parameter("reference_resample_spacing", 0.22);
    declare_parameter("reference_smooth_iterations", 36);
    declare_parameter("reference_smooth_step", 0.32);
    declare_parameter("reference_anchor_weight", 0.16);
    declare_parameter("reference_smooth_weight", 0.82);
    declare_parameter("reference_curvature_weight", 0.18);
    declare_parameter("reference_max_curvature_ratio", 0.92);
    declare_parameter("heuristic_weight", 1.15);
    declare_parameter("progress_weight", 1.8);
    declare_parameter("path_lateral_weight", 0.45);
    declare_parameter("goal_heading_weight", 0.35);
    declare_parameter("backtrack_reject_distance", 0.28);
    declare_parameter("max_extra_path_ratio", 2.25);
    declare_parameter("accept_partial_progress", true);

    declare_parameter("replan_min_interval", 0.25);
    declare_parameter("replan_path_remaining", 2.0);
    declare_parameter("replan_track_error", 0.75);
    declare_parameter("replan_goal_shift", 0.35);
    declare_parameter("rolling_goal_update_distance", 2.8);
    declare_parameter("command_timeout_sec", 0.50);
    declare_parameter("command_filter_alpha", 0.75);
    declare_parameter("odom_speed_from_pose", true);
    declare_parameter("odom_speed_filter_alpha", 0.50);
    declare_parameter("odom_pose_prediction_time", 0.08);
    declare_parameter("stanley_gain", 1.25);
    declare_parameter("stanley_softening", 0.65);
    declare_parameter("pure_pursuit_weight", 0.45);
    declare_parameter("steer_output_sign", 1.0);

    declare_parameter("use_mppi", true);
    declare_parameter("control_period", 0.04);
    declare_parameter("mppi_dt", 0.10);
    declare_parameter("mppi_steps", 24);
    declare_parameter("mppi_batch_size", 320);
    declare_parameter("mppi_speed_std", 0.32);
    declare_parameter("mppi_steer_std", 0.30);
    declare_parameter("mppi_path_weight", 8.0);
    declare_parameter("mppi_heading_weight", 2.0);
    declare_parameter("mppi_goal_weight", 1.2);
    declare_parameter("mppi_risk_weight", 22.0);
    declare_parameter("mppi_clearance_weight", 4.0);
    declare_parameter("mppi_control_weight", 0.18);
    declare_parameter("mppi_rate_weight", 0.32);
    declare_parameter("mppi_progress_weight", 2.5);
    declare_parameter("mppi_speed_tracking_weight", 0.80);
    declare_parameter("mppi_collision_cost", 100000.0);
    declare_parameter("mppi_lookahead_base", 0.75);
    declare_parameter("mppi_lookahead_time", 0.75);
    declare_parameter("terrain_speed_enable", true);
    declare_parameter("terrain_speed_min_scale", 0.45);
    declare_parameter("terrain_risk_slowdown_weight", 0.55);
    declare_parameter("terrain_clearance_slowdown_weight", 0.28);
    declare_parameter("path_commit_enable", true);
    declare_parameter("path_commit_distance", 1.80);
    declare_parameter("path_commit_min_distance", 0.90);
    declare_parameter("path_commit_max_track_error", 0.55);
  }

  void loadParameters()
  {
    get_parameter("reverse_enable", reverse_enable_);
    get_parameter("reverse_max_distance", reverse_max_distance_);
    get_parameter("reverse_max_speed", reverse_max_speed_);
    reverse_max_speed_ = std::clamp(reverse_max_speed_, 0.1, 0.8);
    reverse_max_distance_ = std::clamp(reverse_max_distance_, 0.5, 6.0);
    get_parameter("odom_topic", odom_topic_);
    get_parameter("fused_map_topic", fused_map_topic_);
    get_parameter("cmd_topic", cmd_topic_);
    get_parameter("frame_id", frame_id_);
    get_parameter("live_obstacle_enable", live_obstacle_enable_);
    get_parameter("live_cloud_topic", live_cloud_topic_);
    get_parameter("live_obstacle_min_height", live_obstacle_min_height_);
    get_parameter("live_obstacle_neighbor_radius", live_obstacle_neighbor_radius_);
    get_parameter("live_obstacle_min_hits", live_obstacle_min_hits_);
    get_parameter("live_obstacle_max_age_sec", live_obstacle_max_age_sec_);
    get_parameter("live_obstacle_hold_sec", live_obstacle_hold_sec_);
    get_parameter("live_obstacle_inflation_radius", live_obstacle_inflation_radius_);

    get_parameter("wheel_base", wheel_base_);
    get_parameter("max_steer", max_steer_);
    get_parameter("max_steer_rate", max_steer_rate_);
    get_parameter("max_speed", max_speed_);
    get_parameter("min_speed", min_speed_);
    get_parameter("desired_speed", desired_speed_);
    get_parameter("min_tracking_speed", min_tracking_speed_);
    get_parameter("goal_slowdown_distance", goal_slowdown_distance_);
    get_parameter("max_accel", max_accel_);

    get_parameter("footprint_front", footprint_front_);
    get_parameter("footprint_back", footprint_back_);
    get_parameter("footprint_half_width", footprint_half_width_);
    get_parameter("footprint_margin", footprint_margin_);

    get_parameter("yaw_bins", yaw_bins_);
    get_parameter("primitive_length", primitive_length_);
    get_parameter("integration_step", integration_step_);
    get_parameter("max_search_time_ms", max_search_time_ms_);
    get_parameter("max_iterations", max_iterations_);
    get_parameter("goal_tolerance", goal_tolerance_);
    get_parameter("goal_reached_distance", goal_reached_distance_);
    get_parameter("goal_yaw_tolerance", goal_yaw_tolerance_);
    get_parameter("local_goal_distance", local_goal_distance_);
    get_parameter("local_goal_bearing_limit", local_goal_bearing_limit_);
    get_parameter("direct_goal_bearing_limit", direct_goal_bearing_limit_);
    get_parameter("guide_search_enable", guide_search_enable_);
    get_parameter("guide_max_expansions", guide_max_expansions_);
    get_parameter("guide_min_distance", guide_min_distance_);
    get_parameter("guide_boundary_margin", guide_boundary_margin_);
    get_parameter("guide_progress_weight", guide_progress_weight_);
    get_parameter("guide_goal_weight", guide_goal_weight_);
    get_parameter("guide_lateral_weight", guide_lateral_weight_);
    get_parameter("guide_clearance_weight", guide_clearance_weight_);
    get_parameter("guide_path_cost_weight", guide_path_cost_weight_);
    get_parameter("guide_follow_weight", guide_follow_weight_);
    get_parameter("guide_heading_weight", guide_heading_weight_);
    get_parameter("direct_curve_enable", direct_curve_enable_);
    get_parameter("direct_curve_max_length_ratio", direct_curve_max_length_ratio_);
    get_parameter("direct_curve_max_lateral_ratio", direct_curve_max_lateral_ratio_);
    get_parameter("direct_curve_clearance_weight", direct_curve_clearance_weight_);

    get_parameter("risk_free_threshold", risk_free_threshold_);
    get_parameter("risk_hard_threshold", risk_hard_threshold_);
    get_parameter("risk_query_mode", risk_query_mode_);
    get_parameter("se2_risk_layer", se2_risk_layer_);
    get_parameter("hard_risk_layer", hard_risk_layer_);
    get_parameter("se2_risk_lethal", se2_risk_lethal_);
    get_parameter("unknown_cost", unknown_cost_);
    get_parameter("allow_unknown", allow_unknown_);
    get_parameter("unknown_robot_bubble_radius", unknown_robot_bubble_radius_);
    get_parameter("unknown_robot_bubble_max_baseline", unknown_robot_bubble_max_baseline_);
    get_parameter("max_odom_age_sec", max_odom_age_sec_);
    get_parameter("max_map_age_sec", max_map_age_sec_);
    get_parameter("stuck_timeout_sec", stuck_timeout_sec_);
    get_parameter("stuck_progress_m", stuck_progress_m_);
    get_parameter("stuck_command_mps", stuck_command_mps_);
    get_parameter("risk_weight", risk_weight_);
    get_parameter("clearance_distance", clearance_distance_);
    get_parameter("clearance_weight", clearance_weight_);
    get_parameter("hard_clearance_floor", hard_clearance_floor_);
    get_parameter("comfort_clearance_distance", comfort_clearance_distance_);
    get_parameter("comfort_clearance_weight", comfort_clearance_weight_);
    get_parameter("comfort_clearance_risk_threshold", comfort_clearance_risk_threshold_);
    get_parameter("steer_weight", steer_weight_);
    get_parameter("steer_change_weight", steer_change_weight_);
    get_parameter("steer_reversal_weight", steer_reversal_weight_);
    get_parameter("smoothing_passes", smoothing_passes_);
    get_parameter("shortcut_smoothing_enable", shortcut_smoothing_enable_);
    get_parameter("reference_resample_spacing", reference_resample_spacing_);
    get_parameter("reference_smooth_iterations", reference_smooth_iterations_);
    get_parameter("reference_smooth_step", reference_smooth_step_);
    get_parameter("reference_anchor_weight", reference_anchor_weight_);
    get_parameter("reference_smooth_weight", reference_smooth_weight_);
    get_parameter("reference_curvature_weight", reference_curvature_weight_);
    get_parameter("reference_max_curvature_ratio", reference_max_curvature_ratio_);
    get_parameter("heuristic_weight", heuristic_weight_);
    get_parameter("progress_weight", progress_weight_);
    get_parameter("path_lateral_weight", path_lateral_weight_);
    get_parameter("goal_heading_weight", goal_heading_weight_);
    get_parameter("backtrack_reject_distance", backtrack_reject_distance_);
    get_parameter("max_extra_path_ratio", max_extra_path_ratio_);
    get_parameter("accept_partial_progress", accept_partial_progress_);

    get_parameter("replan_min_interval", replan_min_interval_);
    get_parameter("replan_path_remaining", replan_path_remaining_);
    get_parameter("replan_track_error", replan_track_error_);
    get_parameter("replan_goal_shift", replan_goal_shift_);
    get_parameter("rolling_goal_update_distance", rolling_goal_update_distance_);
    get_parameter("command_timeout_sec", command_timeout_sec_);
    get_parameter("command_filter_alpha", command_filter_alpha_);
    get_parameter("odom_speed_from_pose", odom_speed_from_pose_);
    get_parameter("odom_speed_filter_alpha", odom_speed_filter_alpha_);
    get_parameter("odom_pose_prediction_time", odom_pose_prediction_time_);
    get_parameter("stanley_gain", stanley_gain_);
    get_parameter("stanley_softening", stanley_softening_);
    get_parameter("pure_pursuit_weight", pure_pursuit_weight_);
    get_parameter("steer_output_sign", steer_output_sign_);

    get_parameter("use_mppi", use_mppi_);
    get_parameter("control_period", control_period_);
    get_parameter("mppi_dt", mppi_dt_);
    get_parameter("mppi_steps", mppi_steps_);
    get_parameter("mppi_batch_size", mppi_batch_size_);
    get_parameter("mppi_speed_std", mppi_speed_std_);
    get_parameter("mppi_steer_std", mppi_steer_std_);
    get_parameter("mppi_path_weight", mppi_path_weight_);
    get_parameter("mppi_heading_weight", mppi_heading_weight_);
    get_parameter("mppi_goal_weight", mppi_goal_weight_);
    get_parameter("mppi_risk_weight", mppi_risk_weight_);
    get_parameter("mppi_clearance_weight", mppi_clearance_weight_);
    get_parameter("mppi_control_weight", mppi_control_weight_);
    get_parameter("mppi_rate_weight", mppi_rate_weight_);
    get_parameter("mppi_progress_weight", mppi_progress_weight_);
    get_parameter("mppi_speed_tracking_weight", mppi_speed_tracking_weight_);
    get_parameter("mppi_collision_cost", mppi_collision_cost_);
    get_parameter("mppi_lookahead_base", mppi_lookahead_base_);
    get_parameter("mppi_lookahead_time", mppi_lookahead_time_);
    get_parameter("terrain_speed_enable", terrain_speed_enable_);
    get_parameter("terrain_speed_min_scale", terrain_speed_min_scale_);
    get_parameter("terrain_risk_slowdown_weight", terrain_risk_slowdown_weight_);
    get_parameter("terrain_clearance_slowdown_weight", terrain_clearance_slowdown_weight_);
    get_parameter("path_commit_enable", path_commit_enable_);
    get_parameter("path_commit_distance", path_commit_distance_);
    get_parameter("path_commit_min_distance", path_commit_min_distance_);
    get_parameter("path_commit_max_track_error", path_commit_max_track_error_);

    yaw_bins_ = std::max(16, yaw_bins_);
    live_obstacle_min_height_ = std::clamp(live_obstacle_min_height_, 0.15, 1.5);
    live_obstacle_neighbor_radius_ = std::clamp(live_obstacle_neighbor_radius_, 0.2, 2.0);
    live_obstacle_min_hits_ = std::clamp(live_obstacle_min_hits_, 1, 20);
    live_obstacle_max_age_sec_ = std::clamp(live_obstacle_max_age_sec_, 0.15, 2.0);
    live_obstacle_hold_sec_ = std::clamp(live_obstacle_hold_sec_, 0.2, 3.0);
    live_obstacle_inflation_radius_ = std::clamp(live_obstacle_inflation_radius_, 0.0, 1.0);
    mppi_steps_ = std::clamp(mppi_steps_, 8, 60);
    mppi_batch_size_ = std::clamp(mppi_batch_size_, 32, 1600);
    max_steer_ = std::clamp(max_steer_, 0.05, 1.2);
    desired_speed_ = std::clamp(desired_speed_, std::max(0.0, min_speed_), max_speed_);
    min_tracking_speed_ = std::clamp(min_tracking_speed_, std::max(0.0, min_speed_), desired_speed_);
    goal_slowdown_distance_ = std::max(0.20, goal_slowdown_distance_);
    goal_reached_distance_ = std::clamp(goal_reached_distance_, goal_tolerance_, std::max(0.60, 2.0 * goal_tolerance_));
    local_goal_bearing_limit_ = std::clamp(local_goal_bearing_limit_, 0.45, M_PI);
    direct_goal_bearing_limit_ = std::clamp(direct_goal_bearing_limit_, local_goal_bearing_limit_, M_PI);
    guide_max_expansions_ = std::clamp(guide_max_expansions_, 512, 40000);
    guide_min_distance_ = std::clamp(guide_min_distance_, 0.8, std::max(0.8, local_goal_distance_ * 0.75));
    guide_boundary_margin_ = std::clamp(guide_boundary_margin_, 0.2, 3.0);
    guide_progress_weight_ = std::clamp(guide_progress_weight_, 0.0, 8.0);
    guide_goal_weight_ = std::clamp(guide_goal_weight_, 0.0, 3.0);
    guide_lateral_weight_ = std::clamp(guide_lateral_weight_, 0.0, 3.0);
    guide_clearance_weight_ = std::clamp(guide_clearance_weight_, 0.0, 5.0);
    guide_path_cost_weight_ = std::clamp(guide_path_cost_weight_, 0.0, 0.50);
    guide_follow_weight_ = std::clamp(guide_follow_weight_, 0.0, 6.0);
    guide_heading_weight_ = std::clamp(guide_heading_weight_, 0.0, 3.0);
    if (risk_query_mode_ != "se2_yaw")
      risk_query_mode_ = "flat_2d";
    unknown_robot_bubble_radius_ = std::clamp(unknown_robot_bubble_radius_, 0.0, 1.25);
    unknown_robot_bubble_max_baseline_ = std::clamp(unknown_robot_bubble_max_baseline_, 0.0, risk_hard_threshold_);
    max_odom_age_sec_ = std::max(0.2, max_odom_age_sec_);
    max_map_age_sec_ = std::max(0.3, max_map_age_sec_);
    stuck_timeout_sec_ = std::clamp(stuck_timeout_sec_, 1.0, 15.0);
    stuck_progress_m_ = std::clamp(stuck_progress_m_, 0.03, 0.5);
    stuck_command_mps_ = std::clamp(stuck_command_mps_, 0.05, max_speed_);
    hard_clearance_floor_ = std::clamp(hard_clearance_floor_, -1.0, 1.5);
    direct_curve_max_length_ratio_ = std::clamp(direct_curve_max_length_ratio_, 1.05, 3.0);
    direct_curve_max_lateral_ratio_ = std::clamp(direct_curve_max_lateral_ratio_, 0.15, 1.5);
    direct_curve_clearance_weight_ = std::clamp(direct_curve_clearance_weight_, 0.0, 20.0);
    steer_reversal_weight_ = std::clamp(steer_reversal_weight_, 0.0, 8.0);
    progress_weight_ = std::clamp(progress_weight_, 0.0, 10.0);
    path_lateral_weight_ = std::clamp(path_lateral_weight_, 0.0, 5.0);
    goal_heading_weight_ = std::clamp(goal_heading_weight_, 0.0, 5.0);
    comfort_clearance_distance_ = std::clamp(comfort_clearance_distance_, clearance_distance_, clearance_distance_ + 3.0);
    comfort_clearance_weight_ = std::clamp(comfort_clearance_weight_, 0.0, 16.0);
    comfort_clearance_risk_threshold_ = std::clamp(comfort_clearance_risk_threshold_, 0.02, 0.95);
    smoothing_passes_ = std::clamp(smoothing_passes_, 1, 5);
    reference_resample_spacing_ = std::clamp(reference_resample_spacing_, 0.12, 0.45);
    reference_smooth_iterations_ = std::clamp(reference_smooth_iterations_, 0, 90);
    reference_smooth_step_ = std::clamp(reference_smooth_step_, 0.03, 0.65);
    reference_anchor_weight_ = std::clamp(reference_anchor_weight_, 0.0, 2.0);
    reference_smooth_weight_ = std::clamp(reference_smooth_weight_, 0.0, 3.0);
    reference_curvature_weight_ = std::clamp(reference_curvature_weight_, 0.0, 2.0);
    reference_max_curvature_ratio_ = std::clamp(reference_max_curvature_ratio_, 0.55, 1.0);
    backtrack_reject_distance_ = std::clamp(backtrack_reject_distance_, 0.0, 1.5);
    max_extra_path_ratio_ = std::clamp(max_extra_path_ratio_, 1.15, 5.0);
    rolling_goal_update_distance_ = std::clamp(rolling_goal_update_distance_, 1.2, std::max(1.2, local_goal_distance_ * 0.85));
    command_filter_alpha_ = std::clamp(command_filter_alpha_, 0.05, 1.0);
    odom_speed_filter_alpha_ = std::clamp(odom_speed_filter_alpha_, 0.05, 1.0);
    odom_pose_prediction_time_ = std::clamp(odom_pose_prediction_time_, 0.0, 0.25);
    stanley_gain_ = std::clamp(stanley_gain_, 0.0, 6.0);
    stanley_softening_ = std::clamp(stanley_softening_, 0.05, 3.0);
    pure_pursuit_weight_ = std::clamp(pure_pursuit_weight_, 0.0, 1.0);
    steer_output_sign_ = (steer_output_sign_ < 0.0) ? -1.0 : 1.0;
    terrain_speed_min_scale_ = std::clamp(terrain_speed_min_scale_, 0.15, 1.0);
    terrain_risk_slowdown_weight_ = std::clamp(terrain_risk_slowdown_weight_, 0.0, 1.0);
    terrain_clearance_slowdown_weight_ = std::clamp(terrain_clearance_slowdown_weight_, 0.0, 1.0);
    path_commit_distance_ = std::clamp(path_commit_distance_, 0.0, 4.0);
    path_commit_min_distance_ = std::clamp(path_commit_min_distance_, 0.0, path_commit_distance_);
    path_commit_max_track_error_ = std::clamp(path_commit_max_track_error_, 0.05, replan_track_error_);
  }

  void mapCallback(const se2_grid_msgs::msg::SE2Grid::SharedPtr msg)
  {
    se2_grid::SE2Grid grid;
    try {
      if (!se2_grid::SE2GridRosConverter::fromMessage(*msg, grid)) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "Failed to parse fused_map: invalid SE2Grid message layout");
        return;
      }
    } catch (const std::exception &e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "Failed to parse fused_map: %s", e.what());
      return;
    }

    std::lock_guard<std::mutex> lock(map_mutex_);
    map_ = std::move(grid);
    rebuildCostmapLocked();
    map_stamp_ = msg->info.header.stamp;
    auto snapshot = std::make_shared<ControlMapSnapshot>();
    snapshot->grid = map_;
    snapshot->stamp = map_stamp_;
    snapshot->min_x = min_x_;
    snapshot->min_y = min_y_;
    snapshot->resolution = resolution_;
    snapshot->width = width_;
    snapshot->height = height_;
    snapshot->costs = costs_;
    snapshot->unknown_baseline = unknown_baseline_;
    snapshot->clearance = clearance_;
    snapshot->comfort_clearance = comfort_clearance_;
    std::atomic_store(&control_map_snapshot_, std::shared_ptr<const ControlMapSnapshot>(snapshot));
    map_ready_ = true;
    map_dirty_ = true;
    map_update_seq_.fetch_add(1, std::memory_order_release);
    if (!logged_map_ready_.exchange(true)) {
      RCLCPP_INFO(
        get_logger(),
        "terrain_nav fused_map ready: cells=%dx%d map_yaw_slices=%d planner_yaw_bins=%d "
        "res=%.3f range=[%.2f %.2f]x[%.2f %.2f]",
        width_, height_, map_.getSizeYaw(), yaw_bins_, resolution_,
        min_x_, max_x_, min_y_, max_y_);
    }
  }

  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    if (msg->header.frame_id != frame_id_ || msg->point_step == 0) return;
    const auto has_float_field = [&](const char *name) {
      return std::any_of(msg->fields.begin(), msg->fields.end(), [&](const auto &field) {
        return field.name == name && field.datatype == sensor_msgs::msg::PointField::FLOAT32;
      });
    };
    if (!has_float_field("x") || !has_float_field("y") || !has_float_field("z")) return;

    const auto geometry = std::atomic_load(&control_map_snapshot_);
    if (!geometry || geometry->width <= 0 || geometry->height <= 0) return;
    const double min_x = geometry->min_x, min_y = geometry->min_y, resolution = geometry->resolution;
    const int width = geometry->width, height = geometry->height;

    const size_t cells = static_cast<size_t>(width * height);
    std::vector<float> min_z(cells, std::numeric_limits<float>::infinity());
    const auto cell_for = [&](float x, float y) {
      const int ix = static_cast<int>(std::floor((x - min_x) / resolution));
      const int iy = static_cast<int>(std::floor((y - min_y) / resolution));
      return ix >= 0 && ix < width && iy >= 0 && iy < height ? iy * width + ix : -1;
    };
    try {
      sensor_msgs::PointCloud2ConstIterator<float> x(*msg, "x"), y(*msg, "y"), z(*msg, "z");
      for (; x != x.end(); ++x, ++y, ++z) {
        if (!std::isfinite(*x) || !std::isfinite(*y) || !std::isfinite(*z)) continue;
        const int cell = cell_for(*x, *y);
        if (cell >= 0) {
          min_z[static_cast<size_t>(cell)] = std::min(min_z[static_cast<size_t>(cell)], *z);
        }
      }
    } catch (const std::exception &e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "live obstacle cloud rejected: %s", e.what());
      return;
    }

    std::vector<float> ground_z(cells, std::numeric_limits<float>::infinity());
    const int radius = std::max(1, static_cast<int>(std::ceil(live_obstacle_neighbor_radius_ / resolution)));
    for (int iy = 0; iy < height; ++iy) {
      for (int ix = 0; ix < width; ++ix) {
        const size_t cell = static_cast<size_t>(iy * width + ix);
        if (!std::isfinite(min_z[cell])) continue;
        float base = min_z[cell];
        for (int dy = -radius; dy <= radius; ++dy) {
          for (int dx = -radius; dx <= radius; ++dx) {
            if (dx * dx + dy * dy > radius * radius) continue;
            const int nx = ix + dx, ny = iy + dy;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height)
              base = std::min(base, min_z[static_cast<size_t>(ny * width + nx)]);
          }
        }
        ground_z[cell] = base;
      }
    }

    std::vector<uint8_t> tall_hits(cells, 0);
    sensor_msgs::PointCloud2ConstIterator<float> x(*msg, "x"), y(*msg, "y"), z(*msg, "z");
    for (; x != x.end(); ++x, ++y, ++z) {
      if (!std::isfinite(*x) || !std::isfinite(*y) || !std::isfinite(*z)) continue;
      const int cell = cell_for(*x, *y);
      if (cell >= 0 && *z - ground_z[static_cast<size_t>(cell)] >= live_obstacle_min_height_) {
        auto &hits = tall_hits[static_cast<size_t>(cell)];
        if (hits < 255) ++hits;
      }
    }
    const double stamp_sec = rclcpp::Time(msg->header.stamp).seconds();
    const double received_sec = now().seconds();
    if (stamp_sec > 0.0 && received_sec - stamp_sec > live_obstacle_max_age_sec_) return;
    std::vector<uint8_t> hard(cells, 0);
    {
      std::vector<double> expiry(cells, 0.0);
      std::vector<uint8_t> free_scans(cells, 0);
      if (live_expiry_.size() == cells && min_x == live_min_x_ && min_y == live_min_y_ &&
          resolution == live_resolution_ && width == live_width_ && height == live_height_) {
        expiry = live_expiry_;
        if (live_free_scans_.size() == cells) free_scans = live_free_scans_;
      } else {
        for (int iy = 0; iy < live_height_; ++iy) {
          for (int ix = 0; ix < live_width_; ++ix) {
            const size_t old_cell = static_cast<size_t>(iy * live_width_ + ix);
            if (old_cell >= live_expiry_.size() || live_expiry_[old_cell] <= received_sec) continue;
            const int cell = cell_for(
              static_cast<float>(live_min_x_ + (ix + 0.5) * live_resolution_),
              static_cast<float>(live_min_y_ + (iy + 0.5) * live_resolution_));
            if (cell >= 0) {
              expiry[static_cast<size_t>(cell)] = live_expiry_[old_cell];
              if (old_cell < live_free_scans_.size())
                free_scans[static_cast<size_t>(cell)] = live_free_scans_[old_cell];
            }
          }
        }
      }
      for (size_t i = 0; i < cells; ++i) {
        if (tall_hits[i] >= live_obstacle_min_hits_) {
          expiry[i] = received_sec + live_obstacle_hold_sec_;
          free_scans[i] = 0;
        } else if (tall_hits[i] == 0 && std::isfinite(min_z[i]) &&
                   std::isfinite(ground_z[i]) && min_z[i] - ground_z[i] < 0.15f) {
          free_scans[i] = std::min<int>(2, free_scans[i] + 1);
          if (free_scans[i] >= 2) expiry[i] = 0.0;
        }
        hard[i] = expiry[i] > received_sec ? 1 : 0;
      }
      const int inflation_cells = static_cast<int>(std::ceil(live_obstacle_inflation_radius_ / resolution));
      if (inflation_cells > 0) {
        std::vector<uint8_t> inflated = hard;
        for (int iy = 0; iy < height; ++iy) {
          for (int ix = 0; ix < width; ++ix) {
            if (!hard[static_cast<size_t>(iy * width + ix)]) continue;
            for (int dy = -inflation_cells; dy <= inflation_cells; ++dy) {
              for (int dx = -inflation_cells; dx <= inflation_cells; ++dx) {
                if (std::hypot(dx * resolution, dy * resolution) > live_obstacle_inflation_radius_) continue;
                const int nx = ix + dx, ny = iy + dy;
                if (nx >= 0 && nx < width && ny >= 0 && ny < height)
                  inflated[static_cast<size_t>(ny * width + nx)] = 1;
              }
            }
          }
        }
        hard.swap(inflated);
      }
      const bool changed = hard != live_hard_ || min_x != live_min_x_ || min_y != live_min_y_;
      live_hard_ = hard;
      live_expiry_ = std::move(expiry);
      live_free_scans_ = std::move(free_scans);
      live_min_x_ = min_x;
      live_min_y_ = min_y;
      live_resolution_ = resolution;
      live_width_ = width;
      live_height_ = height;
      auto snapshot = std::make_shared<LiveObstacleSnapshot>();
      snapshot->min_x = min_x;
      snapshot->min_y = min_y;
      snapshot->resolution = resolution;
      snapshot->width = width;
      snapshot->height = height;
      snapshot->stamp_sec = stamp_sec > 0.0 ? stamp_sec : received_sec;
      snapshot->hard = hard;
      std::atomic_store(&live_snapshot_, std::shared_ptr<const LiveObstacleSnapshot>(snapshot));
      if (changed) live_dirty_ = true;
    }

    nav_msgs::msg::OccupancyGrid display;
    display.header = msg->header;
    display.info.resolution = static_cast<float>(resolution);
    display.info.width = width;
    display.info.height = height;
    display.info.origin.position.x = min_x;
    display.info.origin.position.y = min_y;
    display.info.origin.orientation.w = 1.0;
    display.data.assign(cells, -1);
    int obstacle_cells = 0;
    double nearest_obstacle = kInf;
    double nearest_x = 0.0, nearest_y = 0.0;
    const double robot_x = unknown_robot_x_.load(), robot_y = unknown_robot_y_.load();
    for (size_t i = 0; i < cells; ++i) {
      if (!hard[i]) continue;
      display.data[i] = 100;
      ++obstacle_cells;
      const double wx = min_x + (static_cast<int>(i) % width + 0.5) * resolution;
      const double wy = min_y + (static_cast<int>(i) / width + 0.5) * resolution;
      const double distance = std::hypot(wx - robot_x, wy - robot_y);
      if (distance < nearest_obstacle) {
        nearest_obstacle = distance;
        nearest_x = wx;
        nearest_y = wy;
      }
    }
    live_obstacle_pub_->publish(display);
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 3000,
      "live obstacle cloud: %d occupied cells from %ux%u points; nearest=%.2fm at (%.2f, %.2f)",
      obstacle_cells, msg->width, msg->height, nearest_obstacle, nearest_x, nearest_y);
  }

  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    const double x = msg->pose.pose.position.x;
    const double y = msg->pose.pose.position.y;

    tf2::Quaternion q(
      msg->pose.pose.orientation.x,
      msg->pose.pose.orientation.y,
      msg->pose.pose.orientation.z,
      msg->pose.pose.orientation.w);
    double roll = 0.0, pitch = 0.0, yaw = 0.0;
    tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);

    rclcpp::Time stamp(msg->header.stamp);
    double stamp_sec = stamp.seconds();
    if (stamp_sec <= 0.0) {
      stamp_sec = this->now().seconds();
    }

    {
      std::lock_guard<std::mutex> lock(odom_mutex_);
      double measured_speed = std::isfinite(msg->twist.twist.linear.x) ? msg->twist.twist.linear.x : 0.0;
      bool have_pose_speed = false;
      double pose_speed = estimated_odom_v_;
      double pose_yaw_rate = estimated_odom_yaw_rate_;
      if (odom_speed_from_pose_ && have_prev_pose_for_speed_) {
        const double dt = stamp_sec - prev_pose_speed_stamp_sec_;
        if (dt > 0.005 && dt < 0.50) {
          const double dx = x - prev_pose_speed_x_;
          const double dy = y - prev_pose_speed_y_;
          pose_speed = (std::cos(yaw) * dx + std::sin(yaw) * dy) / dt;
          if (std::isfinite(pose_speed)) {
            pose_speed = clampAbs(pose_speed, std::max(0.50, max_speed_ * 1.50));
            have_pose_speed = true;
          }
          const double raw_yaw_rate = wrapAngle(yaw - prev_pose_speed_yaw_) / dt;
          if (std::isfinite(raw_yaw_rate)) {
            pose_yaw_rate = clampAbs(raw_yaw_rate, 3.0);
          }
        }
      }

      if (odom_speed_from_pose_ && (std::abs(measured_speed) < 0.03 || have_pose_speed)) {
        measured_speed = have_pose_speed ? pose_speed : estimated_odom_v_;
      }
      estimated_odom_v_ =
        odom_speed_filter_alpha_ * measured_speed + (1.0 - odom_speed_filter_alpha_) * estimated_odom_v_;
      estimated_odom_yaw_rate_ =
        odom_speed_filter_alpha_ * pose_yaw_rate + (1.0 - odom_speed_filter_alpha_) * estimated_odom_yaw_rate_;

      odom_.x = x;
      odom_.y = y;
      odom_.yaw = wrapAngle(yaw);
      odom_.v = estimated_odom_v_;
      odom_.yaw_rate = estimated_odom_yaw_rate_;
      odom_.stamp = msg->header.stamp;
      odom_.valid = true;
      unknown_robot_x_.store(x);
      unknown_robot_y_.store(y);
      unknown_robot_pose_ready_.store(true);

      prev_pose_speed_x_ = x;
      prev_pose_speed_y_ = y;
      prev_pose_speed_yaw_ = yaw;
      prev_pose_speed_stamp_sec_ = stamp_sec;
      have_prev_pose_for_speed_ = true;
    }
    if (!logged_odom_ready_.exchange(true)) {
      RCLCPP_INFO(
        get_logger(),
        "terrain_nav odometry ready: frame=%s child=%s pos=(%.2f, %.2f)",
        msg->header.frame_id.c_str(), msg->child_frame_id.c_str(),
        msg->pose.pose.position.x, msg->pose.pose.position.y);
    }
  }

  void goalCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    tf2::Quaternion q(
      msg->pose.orientation.x,
      msg->pose.orientation.y,
      msg->pose.orientation.z,
      msg->pose.orientation.w);
    double roll = 0.0, pitch = 0.0, yaw = 0.0;
    tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);

    Pose2 accepted_goal;
    bool new_goal = false;
    {
      std::lock_guard<std::mutex> lock(goal_mutex_);
      const double requested_yaw = wrapAngle(yaw);
      new_goal = !have_goal_ ||
        std::hypot(raw_goal_.x - msg->pose.position.x, raw_goal_.y - msg->pose.position.y) > 0.05 ||
        std::abs(wrapAngle(raw_goal_.yaw - requested_yaw)) > 0.05;
      if (new_goal) {
        raw_goal_.x = msg->pose.position.x;
        raw_goal_.y = msg->pose.position.y;
        raw_goal_.yaw = requested_yaw;
        raw_goal_.s = 0.0;
        ++goal_seq_;
      }
      accepted_goal = raw_goal_;
      have_goal_ = true;
    }
    if (!new_goal) return;
    requestPlan("new goal");

    OdomState odom_snapshot;
    {
      std::lock_guard<std::mutex> lock(odom_mutex_);
      odom_snapshot = odom_;
    }
    RCLCPP_INFO(
      get_logger(),
      "Received terrain_nav goal: %.2f %.2f %.2f | odom=%s map=%s",
      accepted_goal.x, accepted_goal.y, accepted_goal.yaw,
      odom_snapshot.valid ? "ready" : "waiting",
      map_ready_ ? "ready" : "waiting");
    publishActiveGoal(accepted_goal);
    // A failed new goal must not leave the vehicle following the previous goal.
    if (new_goal) {
      std::lock_guard<std::mutex> lock(path_mutex_);
      path_.clear();
      have_path_ = false;
    }
  }

  void rebuildCostmapLocked()
  {
    if (!map_.exists(se2_risk_layer_)) {
      width_ = height_ = 0;
      costs_.clear();
      hard_.clear();
      unknown_baseline_.clear();
      clearance_.clear();
      comfort_clearance_.clear();
      return;
    }

    Eigen::Vector2d min_range, max_range;
    map_.getMapRange(min_range, max_range);
    resolution_ = std::max(0.03, map_.getResolutionPos());
    min_x_ = min_range.x();
    min_y_ = min_range.y();
    max_x_ = max_range.x();
    max_y_ = max_range.y();
    width_ = std::max(1, static_cast<int>(std::ceil((max_x_ - min_x_) / resolution_)));
    height_ = std::max(1, static_cast<int>(std::ceil((max_y_ - min_y_) / resolution_)));
    costs_.assign(static_cast<size_t>(width_ * height_), 0);
    hard_.assign(static_cast<size_t>(width_ * height_), 0);
    unknown_baseline_.assign(static_cast<size_t>(width_ * height_), 1.0f);
    clearance_.assign(static_cast<size_t>(width_ * height_), static_cast<float>(clearance_distance_ + 2.0));
    comfort_clearance_.assign(
      static_cast<size_t>(width_ * height_), static_cast<float>(comfort_clearance_distance_ + 2.0));

    const bool has_valid = map_.exists("valid");
    int known_count = 0;
    int hard_count = 0;
    double max_risk = 0.0;
    for (int y = 0; y < height_; ++y) {
      for (int x = 0; x < width_; ++x) {
        const double wx = min_x_ + (x + 0.5) * resolution_;
        const double wy = min_y_ + (y + 0.5) * resolution_;
        const auto idx = static_cast<size_t>(y * width_ + x);

        double risk = 0.0;
        bool known = true;
        const std::string costmap_risk_layer =
          map_.exists(hard_risk_layer_) ? hard_risk_layer_ : se2_risk_layer_;
        try {
          risk = map_.getValue(
            costmap_risk_layer,
            Eigen::Vector3d(wx, wy, 0.0),
            se2_grid::InterpolationMethods::INTER_NEAREST);
        } catch (...) {
          known = false;
        }
        if (!std::isfinite(risk)) known = false;

        if (has_valid) {
          try {
            const double valid = map_.getValue("valid", Eigen::Vector3d(wx, wy, 0.0), se2_grid::InterpolationMethods::INTER_NEAREST);
            if (std::isfinite(valid) && valid < 0.5) known = false;
          } catch (...) {
            known = false;
          }
        }

        if (map_.exists("collision_risk")) {
          const double collision = map_.getValue("collision_risk", Eigen::Vector3d(wx, wy, 0.0),
                                                se2_grid::InterpolationMethods::INTER_NEAREST);
          if (std::isfinite(collision) && collision >= 0.5) {
            known = true;
            risk = 1.0;
          }
        }

        if (!known) {
          costs_[idx] = 255;
          hard_[idx] = 0;
          if (std::isfinite(risk)) unknown_baseline_[idx] = static_cast<float>(std::clamp(risk, 0.0, 1.0));
          continue;
        }

        risk = std::clamp(risk, 0.0, 1.0);
        ++known_count;
        max_risk = std::max(max_risk, risk);
        if (risk >= risk_hard_threshold_) {
          costs_[idx] = 254;
          hard_[idx] = 1;
          ++hard_count;
        } else {
          const double soft = std::max(0.0, risk - risk_free_threshold_) /
                              std::max(1e-3, risk_hard_threshold_ - risk_free_threshold_);
          costs_[idx] = static_cast<uint8_t>(std::clamp(soft * 220.0, 0.0, 220.0));
        }
      }
    }
    computeClearanceLocked();
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 3000,
      "terrain_nav cost grid: size=%dx%d known=%d hard=%d max_risk=%.2f hard_thr=%.2f clearance=%.2f comfort=%.2f risk_mode=%s",
      width_, height_, known_count, hard_count, max_risk, risk_hard_threshold_,
      clearance_distance_, comfort_clearance_distance_, risk_query_mode_.c_str());
  }

  void computeClearanceLocked()
  {
    auto fill_distance = [&](std::vector<float> &out, const auto &is_seed, double max_distance) {
      struct Item
      {
        int idx = 0;
        double d = 0.0;
        bool operator>(const Item &other) const { return d > other.d; }
      };

      std::priority_queue<Item, std::vector<Item>, std::greater<Item>> q;
      std::vector<double> dist(static_cast<size_t>(width_ * height_), kInf);
      for (int i = 0; i < width_ * height_; ++i) {
        if (is_seed(i)) {
          dist[static_cast<size_t>(i)] = 0.0;
          q.push({i, 0.0});
        }
      }

      if (q.empty()) {
        std::fill(out.begin(), out.end(), static_cast<float>(max_distance));
        return;
      }

      const std::array<int, 8> dx = {-1, 1, 0, 0, -1, -1, 1, 1};
      const std::array<int, 8> dy = {0, 0, -1, 1, -1, 1, -1, 1};
      while (!q.empty()) {
        const auto cur = q.top();
        q.pop();
        if (cur.d > dist[static_cast<size_t>(cur.idx)] + 1e-9) continue;
        const int cx = cur.idx % width_;
        const int cy = cur.idx / width_;
        for (int k = 0; k < 8; ++k) {
          const int nx = cx + dx[k];
          const int ny = cy + dy[k];
          if (nx < 0 || nx >= width_ || ny < 0 || ny >= height_) continue;
          const int ni = ny * width_ + nx;
          const double step = ((dx[k] == 0 || dy[k] == 0) ? 1.0 : std::sqrt(2.0)) * resolution_;
          const double nd = cur.d + step;
          if (nd + 1e-9 < dist[static_cast<size_t>(ni)]) {
            dist[static_cast<size_t>(ni)] = nd;
            q.push({ni, nd});
          }
        }
      }

      for (size_t i = 0; i < out.size(); ++i) {
        out[i] = static_cast<float>(std::min(dist[i], max_distance));
      }
    };

    fill_distance(
      clearance_,
      [&](int i) { return hard_[static_cast<size_t>(i)] != 0; },
      clearance_distance_ + 2.0);

    const uint8_t comfort_seed_cost = static_cast<uint8_t>(
      std::clamp(comfort_clearance_risk_threshold_ * 252.0, 1.0, 253.0));
    fill_distance(
      comfort_clearance_,
      [&](int i) {
        const auto idx = static_cast<size_t>(i);
        return hard_[idx] != 0 || (costs_[idx] != 255 && costs_[idx] >= comfort_seed_cost);
      },
      comfort_clearance_distance_ + 2.0);

  }

  bool worldToCostCell(double x, double y, int &ix, int &iy) const
  {
    if (x < min_x_ || x >= max_x_ || y < min_y_ || y >= max_y_) return false;
    ix = static_cast<int>(std::floor((x - min_x_) / resolution_));
    iy = static_cast<int>(std::floor((y - min_y_) / resolution_));
    return ix >= 0 && ix < width_ && iy >= 0 && iy < height_;
  }

  CostQuery queryCostLocked(
    double x, double y, double yaw = 0.0, bool allow_fresh_unknown = false,
    const ControlMapSnapshot *snapshot = nullptr) const
  {
    const auto &grid = snapshot ? snapshot->grid : map_;
    const auto &costs = snapshot ? snapshot->costs : costs_;
    const auto &unknown_baseline = snapshot ? snapshot->unknown_baseline : unknown_baseline_;
    const auto &clearance = snapshot ? snapshot->clearance : clearance_;
    const auto &comfort_clearance = snapshot ? snapshot->comfort_clearance : comfort_clearance_;
    const double min_x = snapshot ? snapshot->min_x : min_x_;
    const double min_y = snapshot ? snapshot->min_y : min_y_;
    const double resolution = snapshot ? snapshot->resolution : resolution_;
    const int width = snapshot ? snapshot->width : width_;
    const int height = snapshot ? snapshot->height : height_;
    const auto live = std::atomic_load(&live_snapshot_);
    CostQuery out;
    if (live && live->occupied(x, y)) {
      out.known = true;
      out.lethal = true;
      out.risk = 1.0;
      out.clearance = 0.0;
      out.comfort_clearance = 0.0;
      out.cost = mppi_collision_cost_;
      return out;
    }
    const int ix = static_cast<int>(std::floor((x - min_x) / resolution));
    const int iy = static_cast<int>(std::floor((y - min_y) / resolution));
    if (ix < 0 || iy < 0 || ix >= width || iy >= height) {
      out.known = false;
      out.lethal = !allow_unknown_;
      out.cost = allow_unknown_ ? unknown_cost_ : mppi_collision_cost_;
      return out;
    }

    const auto idx = static_cast<size_t>(iy * width + ix);
    const uint8_t c = costs[idx];
    if (c == 255) {
      out.known = false;
      const double dx = x - unknown_robot_x_.load();
      const double dy = y - unknown_robot_y_.load();
      const bool robot_bubble = !allow_unknown_ && unknown_robot_pose_ready_.load() &&
        unknown_robot_bubble_radius_ > 0.0 &&
        dx * dx + dy * dy <= unknown_robot_bubble_radius_ * unknown_robot_bubble_radius_ &&
        unknown_baseline[idx] <= unknown_robot_bubble_max_baseline_;
      const bool fresh_live_scan = allow_fresh_unknown && live &&
        (now().seconds() - live->stamp_sec <= live_obstacle_max_age_sec_);
      out.lethal = !allow_unknown_ && !robot_bubble && !fresh_live_scan;
      out.cost = allow_unknown_ ? unknown_cost_ : mppi_collision_cost_;
      if (robot_bubble) out.cost = unknown_cost_;
      out.clearance = clearance[idx];
      out.comfort_clearance = comfort_clearance.empty() ? out.clearance : comfort_clearance[idx];
      return out;
    }
    out.known = true;
    out.lethal = (c >= 254);
    out.risk = static_cast<double>(c) / 252.0;
    out.clearance = clearance[idx];
    out.comfort_clearance = comfort_clearance.empty() ? out.clearance : comfort_clearance[idx];
    out.cost = out.risk * risk_weight_;
    if (std::isfinite(out.clearance) && out.clearance < clearance_distance_) {
      const double margin = (clearance_distance_ - out.clearance) / std::max(1e-3, clearance_distance_);
      out.cost += clearance_weight_ * margin * margin;
    }
    if (risk_query_mode_ == "se2_yaw" &&
        grid.exists(se2_risk_layer_) &&
        grid.hasSO2(se2_risk_layer_))
    {
      try {
        const double se2_risk = grid.getValue(
          se2_risk_layer_,
          Eigen::Vector3d(x, y, yaw),
          se2_grid::InterpolationMethods::INTER_NEAREST);
        if (std::isfinite(se2_risk))
        {
          out.risk = std::max(out.risk, std::clamp(se2_risk, 0.0, 1.0));
          out.cost = std::max(out.cost, out.risk * risk_weight_);
          if (se2_risk_lethal_ && out.risk >= risk_hard_threshold_)
          {
            out.lethal = true;
            out.cost = mppi_collision_cost_;
          }
        }
      } catch (...) {
        // Keep the 2D fallback result when the enhanced yaw layer is absent
        // or the query is outside the map.
      }
    }
    if (out.lethal) out.cost = mppi_collision_cost_;
    return out;
  }

  bool liveObstacleAtLocked(double x, double y) const
  {
    const auto live = std::atomic_load(&live_snapshot_);
    return live && live->occupied(x, y);
  }


  bool livePathAheadUnsafe(const std::vector<Pose2> &path, size_t nearest_idx)
  {
    const auto live = std::atomic_load(&live_snapshot_);
    if (!live || path.empty()) return false;
    const double start_s = path[nearest_idx].s;
    const double end_s = std::min(path.back().s, start_s + std::max(3.0, replan_path_remaining_ + 1.2));
    for (double s = start_s; s <= end_s + 1e-6; s += 0.15) {
      const Pose2 p = samplePathAt(path, s);
      const double c = std::cos(p.yaw), sn = std::sin(p.yaw);
      for (const auto &off : footprintOffsets()) {
        if (live->occupied(p.x + c * off.x() - sn * off.y(),
                           p.y + sn * off.x() + c * off.y())) return true;
      }
    }
    return false;
  }

  std::array<Eigen::Vector2d, 25> footprintOffsets() const
  {
    const double f = footprint_front_ + footprint_margin_;
    const double b = footprint_back_ + footprint_margin_;
    const double w = footprint_half_width_ + footprint_margin_;
    const double f2 = 0.5 * f;
    const double b2 = 0.5 * b;
    const double w2 = 0.5 * w;
    return {
      Eigen::Vector2d(f, -w),
      Eigen::Vector2d(f, -w2),
      Eigen::Vector2d(f, 0.0),
      Eigen::Vector2d(f, w2),
      Eigen::Vector2d(f, w),
      Eigen::Vector2d(f2, -w),
      Eigen::Vector2d(f2, -w2),
      Eigen::Vector2d(f2, 0.0),
      Eigen::Vector2d(f2, w2),
      Eigen::Vector2d(f2, w),
      Eigen::Vector2d(0.0, -w),
      Eigen::Vector2d(0.0, -w2),
      Eigen::Vector2d(0.0, 0.0),
      Eigen::Vector2d(0.0, w2),
      Eigen::Vector2d(0.0, w),
      Eigen::Vector2d(-b2, -w),
      Eigen::Vector2d(-b2, -w2),
      Eigen::Vector2d(-b2, 0.0),
      Eigen::Vector2d(-b2, w2),
      Eigen::Vector2d(-b2, w),
      Eigen::Vector2d(-b, -w),
      Eigen::Vector2d(-b, -w2),
      Eigen::Vector2d(-b, 0.0),
      Eigen::Vector2d(-b, w2),
      Eigen::Vector2d(-b, w),
    };
  }

  bool footprintCostLocked(
    double x,
    double y,
    double yaw,
    double &cost,
    double *min_clearance = nullptr,
    double *min_comfort_clearance = nullptr,
    bool allow_fresh_unknown = false,
    const ControlMapSnapshot *snapshot = nullptr) const
  {
    cost = 0.0;
    double clr = kInf;
    double comfort_clr = kInf;
    const double c = std::cos(yaw);
    const double s = std::sin(yaw);
    for (const auto &off : footprintOffsets()) {
      const double px = x + c * off.x() - s * off.y();
      const double py = y + s * off.x() + c * off.y();
      const CostQuery q = queryCostLocked(px, py, yaw, allow_fresh_unknown, snapshot);
      if (q.lethal) return false;
      cost = std::max(cost, q.cost);
      clr = std::min(clr, q.clearance);
      comfort_clr = std::min(comfort_clr, q.comfort_clearance);
    }
    if (min_clearance) *min_clearance = clr;
    if (min_comfort_clearance) *min_comfort_clearance = comfort_clr;
    return true;
  }

  double hardClearanceFloor() const
  {
    if (hard_clearance_floor_ >= 0.0) return hard_clearance_floor_;
    return std::max(0.24, 0.30 * clearance_distance_);
  }

  bool directCorridorBlockedLocked(
    const Pose2 &start,
    const Pose2 &raw_goal,
    double max_probe,
    double &blocked_s) const
  {
    blocked_s = kInf;
    const double dx = raw_goal.x - start.x;
    const double dy = raw_goal.y - start.y;
    const double dist = std::hypot(dx, dy);
    if (dist < 1.0) return false;

    const double dir = std::atan2(dy, dx);
    const double probe = std::min({dist, std::max(1.0, max_probe), local_goal_distance_ + 1.4});
    int bad_count = 0;
    constexpr double kStep = 0.22;
    const double nx = -std::sin(dir);
    const double ny = std::cos(dir);
    const std::array<double, 5> lateral_checks = {
      0.0,
      footprint_half_width_ + 0.35,
      -(footprint_half_width_ + 0.35),
      footprint_half_width_ + 0.95,
      -(footprint_half_width_ + 0.95),
    };
    for (double s = std::max(1.20, footprint_front_ + 0.45); s <= probe + 1e-6; s += kStep) {
      int lateral_bad = 0;
      bool center_bad = false;
      for (double lateral : lateral_checks) {
        const double x = start.x + s * std::cos(dir) + lateral * nx;
        const double y = start.y + s * std::sin(dir) + lateral * ny;
        double fc = 0.0;
        double clearance = kInf;
        const bool footprint_ok = footprintCostLocked(x, y, dir, fc, &clearance);
        const bool hard_near = !footprint_ok ||
          (std::isfinite(clearance) && clearance < hardClearanceFloor());
        if (hard_near) {
          ++lateral_bad;
          if (std::abs(lateral) < 1e-6) center_bad = true;
        }
      }

      const bool corridor_blocked = center_bad || lateral_bad >= 5;
      if (corridor_blocked) {
        ++bad_count;
        if (bad_count >= 4) {
          blocked_s = std::max(0.0, s - 3.0 * kStep);
          return true;
        }
      } else {
        bad_count = 0;
      }
    }
    return false;
  }

  bool resolveGuideGoalLocked(
    const OdomState &odom,
    const Pose2 &raw_goal,
    Pose2 &active_goal,
    std::string &reason,
    std::vector<Pose2> &guide_path)
  {
    if (!guide_search_enable_ || !map_ready_ || width_ <= 0 || height_ <= 0) return false;
    const auto planning_start = std::chrono::steady_clock::now();

    int sx = 0, sy = 0;
    if (!worldToCostCell(odom.x, odom.y, sx, sy)) return false;

    const double raw_dx = raw_goal.x - odom.x;
    const double raw_dy = raw_goal.y - odom.y;
    const double raw_dist = std::hypot(raw_dx, raw_dy);
    if (raw_dist < std::max(1.2, guide_min_distance_)) return false;

    const double raw_dir = std::atan2(raw_dy, raw_dx);
    const double ux = std::cos(raw_dir);
    const double uy = std::sin(raw_dir);
    const double horizon = std::min(local_goal_distance_, raw_dist);
    const double search_radius = std::min(raw_dist + 0.75, local_goal_distance_ + 1.50);
    double blocked_s = kInf;
    const bool direct_corridor_blocked = directCorridorBlockedLocked(
      Pose2{odom.x, odom.y, raw_dir, 0.0},
      raw_goal,
      std::min(raw_dist, local_goal_distance_ + 1.2),
      blocked_s);
    const double min_candidate_dist = std::min(
      std::max(guide_min_distance_, 0.34 * horizon),
      std::max(1.0, horizon - 0.4));
    const double preferred_candidate_dist = std::max(min_candidate_dist, 0.68 * horizon);
    const double boundary_margin = std::max(guide_boundary_margin_, 2.0 * resolution_);

    const int total = width_ * height_;
    const int start_idx = sy * width_ + sx;

    struct GuideItem
    {
      int idx = 0;
      double f = 0.0;
      bool operator>(const GuideItem &other) const { return f > other.f; }
    };

    std::vector<double> dist(static_cast<size_t>(total), kInf);
    std::vector<int> parent(static_cast<size_t>(total), -1);
    std::vector<uint8_t> closed(static_cast<size_t>(total), 0);
    std::priority_queue<GuideItem, std::vector<GuideItem>, std::greater<GuideItem>> open;

    auto cellCenter = [&](int idx, double &wx, double &wy) {
      const int ix = idx % width_;
      const int iy = idx / width_;
      wx = min_x_ + (static_cast<double>(ix) + 0.5) * resolution_;
      wy = min_y_ + (static_cast<double>(iy) + 0.5) * resolution_;
    };

    auto nearBoundary = [&](double wx, double wy) {
      return wx <= min_x_ + boundary_margin || wx >= max_x_ - boundary_margin ||
             wy <= min_y_ + boundary_margin || wy >= max_y_ - boundary_margin;
    };

    auto guideHeuristic = [&](double wx, double wy) {
      const double vx = wx - odom.x;
      const double vy = wy - odom.y;
      const double progress = vx * ux + vy * uy;
      const double lateral = std::abs(-vx * uy + vy * ux);
      const double missing_progress = std::max(0.0, horizon - progress);
      const double progress_w = direct_corridor_blocked ? 0.20 : 0.30;
      const double lateral_w = direct_corridor_blocked ? 0.012 : 0.05;
      return progress_w * missing_progress + lateral_w * lateral;
    };

    auto considerCandidate = [&](int idx, double path_cost, int expansions, int &best_idx, double &best_score) {
      if (idx == start_idx) return;

      double wx = 0.0, wy = 0.0;
      cellCenter(idx, wx, wy);
      const double vx = wx - odom.x;
      const double vy = wy - odom.y;
      const double dist_from_robot = std::hypot(vx, vy);
      if (dist_from_robot < min_candidate_dist) return;

      const double progress = vx * ux + vy * uy;
      if (progress < (direct_corridor_blocked ? -0.55 * search_radius : -0.20)) return;

      const double lateral = std::abs(-vx * uy + vy * ux);
      const bool boundary_candidate = nearBoundary(wx, wy);
      const double lateral_cap = direct_corridor_blocked
        ? std::max(2.40, 0.92 * search_radius)
        : std::max(1.25, 0.46 * horizon);
      const double absolute_lateral_cap = direct_corridor_blocked
        ? std::max(3.00, 1.05 * search_radius)
        : std::max(1.80, 0.62 * horizon);
      if ((!boundary_candidate && lateral > lateral_cap) || lateral > absolute_lateral_cap) {
        return;
      }
      const bool far_enough = dist_from_robot >= preferred_candidate_dist;
      const bool bypass_side_step =
        direct_corridor_blocked && lateral >= std::max(0.90, 0.20 * horizon);
      const bool past_block_center =
        direct_corridor_blocked && progress >= blocked_s + 0.35;
      const bool meaningful_progress = direct_corridor_blocked
        ? (progress >= std::max(0.30, 0.08 * horizon) || bypass_side_step || past_block_center)
        : (progress >= std::max(0.75, 0.28 * horizon));
      if (!boundary_candidate && !far_enough && !meaningful_progress) return;
      if (direct_corridor_blocked &&
          progress > blocked_s - 0.55 &&
          progress < blocked_s + 0.75 &&
          lateral < std::max(0.85, 0.18 * horizon)) {
        return;
      }

      const double yaw = std::atan2(vy, vx);
      double fc = 0.0;
      double clearance = kInf;
      double comfort_clearance = kInf;
      if (!footprintCostLocked(wx, wy, yaw, fc, &clearance, &comfort_clearance)) return;

      const double raw_goal_dist = std::hypot(raw_goal.x - wx, raw_goal.y - wy);
      const double comfort_bonus = std::min(
        std::isfinite(comfort_clearance) ? comfort_clearance : comfort_clearance_distance_,
        comfort_clearance_distance_);
      const double hard_clearance_bonus = std::min(
        std::isfinite(clearance) ? clearance : clearance_distance_,
        clearance_distance_);
      double frontier_bonus = (boundary_candidate ? 0.75 : 0.0) +
        0.35 * std::min(dist_from_robot, horizon) / std::max(1e-3, horizon);
      if (direct_corridor_blocked) {
        frontier_bonus +=
          (boundary_candidate ? 1.35 : 0.0) +
          0.55 * std::min(dist_from_robot, search_radius) / std::max(1e-3, search_radius) +
          0.42 * std::min(lateral, search_radius) / std::max(1e-3, search_radius);
      }
      const double heading_gap = std::abs(wrapAngle(yaw - odom.yaw));
      const double lateral_excess = std::max(0.0, lateral - 0.24 * horizon);
      const double progress_weight = direct_corridor_blocked ? 0.72 * guide_progress_weight_ : guide_progress_weight_;
      const double lateral_weight = direct_corridor_blocked ? 0.18 * guide_lateral_weight_ : guide_lateral_weight_;
      const double heading_weight = direct_corridor_blocked ? 0.30 * guide_heading_weight_ : guide_heading_weight_;
      const double lateral_excess_weight = direct_corridor_blocked ? 0.05 : 0.38;

      const double score =
        progress_weight * progress -
        guide_goal_weight_ * raw_goal_dist -
        lateral_weight * lateral -
        guide_path_cost_weight_ * path_cost -
        heading_weight * heading_gap * heading_gap -
        lateral_excess_weight * lateral_excess * lateral_excess -
        0.85 * fc +
        guide_clearance_weight_ * (0.65 * comfort_bonus + 0.35 * hard_clearance_bonus) +
        frontier_bonus -
        0.0005 * static_cast<double>(expansions);

      if (score > best_score) {
        best_score = score;
        best_idx = idx;
      }
    };

    dist[static_cast<size_t>(start_idx)] = 0.0;
    double sx_world = 0.0, sy_world = 0.0;
    cellCenter(start_idx, sx_world, sy_world);
    open.push({start_idx, guideHeuristic(sx_world, sy_world)});

    int best_idx = -1;
    double best_score = -kInf;
    int expansions = 0;
    const std::array<int, 8> nx_step = {-1, 1, 0, 0, -1, -1, 1, 1};
    const std::array<int, 8> ny_step = {0, 0, -1, 1, -1, 1, -1, 1};

    while (!open.empty() && expansions < guide_max_expansions_) {
      if (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - planning_start).count()
          >= std::min(65.0, 0.60 * max_search_time_ms_)) break;
      const int cur_idx = open.top().idx;
      open.pop();
      if (closed[static_cast<size_t>(cur_idx)] != 0) continue;
      closed[static_cast<size_t>(cur_idx)] = 1;
      ++expansions;

      considerCandidate(cur_idx, dist[static_cast<size_t>(cur_idx)], expansions, best_idx, best_score);

      const int cx = cur_idx % width_;
      const int cy = cur_idx / width_;
      double cwx = 0.0, cwy = 0.0;
      cellCenter(cur_idx, cwx, cwy);

      for (size_t k = 0; k < nx_step.size(); ++k) {
        const int ix = cx + nx_step[k];
        const int iy = cy + ny_step[k];
        if (ix < 0 || ix >= width_ || iy < 0 || iy >= height_) continue;

        const int ni = iy * width_ + ix;
        if (closed[static_cast<size_t>(ni)] != 0) continue;

        double wx = 0.0, wy = 0.0;
        cellCenter(ni, wx, wy);
        const double vx = wx - odom.x;
        const double vy = wy - odom.y;
        if (std::hypot(vx, vy) > search_radius) continue;

        const double progress = vx * ux + vy * uy;
        if (progress < (direct_corridor_blocked ? -0.55 * search_radius : -1.00)) continue;

        const double step_yaw = std::atan2(wy - cwy, wx - cwx);
        const auto q = queryCostLocked(wx, wy, std::isfinite(step_yaw) ? step_yaw : odom.yaw);
        if (q.lethal) continue;

        const double step = ((nx_step[k] == 0 || ny_step[k] == 0) ? 1.0 : std::sqrt(2.0)) * resolution_;
        double soft_cost = q.cost;
        if (!q.known) {
          soft_cost += 2.0;
        }
        if (std::isfinite(q.comfort_clearance) && q.comfort_clearance < comfort_clearance_distance_) {
          const double margin =
            (comfort_clearance_distance_ - q.comfort_clearance) / std::max(1e-3, comfort_clearance_distance_);
          soft_cost += 4.2 * margin * margin;
        }
        if (std::isfinite(q.clearance) && q.clearance < clearance_distance_) {
          const double margin = (clearance_distance_ - q.clearance) / std::max(1e-3, clearance_distance_);
          soft_cost += 3.0 * margin * margin;
        }

        const double cur_progress = (cwx - odom.x) * ux + (cwy - odom.y) * uy;
        const double backtrack = std::max(0.0, cur_progress - progress);
        const double backtrack_weight = direct_corridor_blocked ? 0.08 : 0.45;
        const double nd =
          dist[static_cast<size_t>(cur_idx)] +
          step * (1.0 + 0.42 * soft_cost) +
          backtrack_weight * backtrack * backtrack;

        if (nd + 1e-9 >= dist[static_cast<size_t>(ni)]) continue;
        dist[static_cast<size_t>(ni)] = nd;
        parent[static_cast<size_t>(ni)] = cur_idx;
        open.push({ni, nd + guideHeuristic(wx, wy)});
      }
    }

    if (best_idx < 0) return false;

    double bx = 0.0, by = 0.0;
    cellCenter(best_idx, bx, by);
    int prev_idx = parent[static_cast<size_t>(best_idx)];
    double yaw = std::atan2(by - odom.y, bx - odom.x);
    if (prev_idx >= 0) {
      double px = 0.0, py = 0.0;
      cellCenter(prev_idx, px, py);
      if (std::hypot(bx - px, by - py) > 1e-4) {
        yaw = std::atan2(by - py, bx - px);
      }
    }

    active_goal = Pose2{bx, by, wrapAngle(yaw), 0.0};
    guide_path.clear();
    std::vector<int> rev_cells;
    for (int idx = best_idx; idx >= 0; idx = parent[static_cast<size_t>(idx)]) {
      rev_cells.push_back(idx);
      if (idx == start_idx) break;
    }
    std::reverse(rev_cells.begin(), rev_cells.end());

    guide_path.reserve(rev_cells.size());
    for (int idx : rev_cells) {
      double wx = 0.0, wy = 0.0;
      cellCenter(idx, wx, wy);
      guide_path.push_back(Pose2{wx, wy, yaw, 0.0});
    }
    if (guide_path.empty() || std::hypot(guide_path.front().x - odom.x, guide_path.front().y - odom.y) > 0.35) {
      guide_path.insert(guide_path.begin(), Pose2{odom.x, odom.y, odom.yaw, 0.0});
    } else {
      guide_path.front() = Pose2{odom.x, odom.y, odom.yaw, 0.0};
    }
    guide_path.back() = active_goal;
    updateYawFromTangent(guide_path);
    updatePathArcLength(guide_path);
    shortcutGuidePathLocked(guide_path);
    guide_path = resamplePathUniform(guide_path, std::max(0.24, 1.4 * reference_resample_spacing_));
    smoothGuidePathLocked(guide_path);
    updateYawFromTangent(guide_path);
    updatePathArcLength(guide_path);
    if (!guide_path.empty()) {
      active_goal = guide_path.back();
      active_goal.s = 0.0;
    }

    reason = "guide search subgoal";
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 1200,
      "terrain_nav guide goal: (%.2f, %.2f) score=%.2f expansions=%d raw_dist=%.2f mode=%s block_s=%.2f",
      active_goal.x, active_goal.y, best_score, expansions, raw_dist,
      direct_corridor_blocked ? "bypass" : "normal",
      std::isfinite(blocked_s) ? blocked_s : -1.0);
    return true;
  }

  bool resolveDirectLocalGoalLocked(
    const Pose2 &start,
    const Pose2 &raw_goal,
    Pose2 &active_goal,
    std::string &reason,
    std::vector<Pose2> &guide_path) const
  {
    guide_path.clear();

    const double dx = raw_goal.x - start.x;
    const double dy = raw_goal.y - start.y;
    const double dist = std::hypot(dx, dy);
    if (dist < 1e-3) {
      active_goal = raw_goal;
      reason = "goal at robot";
      return true;
    }

    const double raw_dir = std::atan2(dy, dx);
    const double bearing_error = wrapAngle(raw_dir - start.yaw);
    const double direct_bearing_gap = std::abs(bearing_error);
    const double local_dir = wrapAngle(
      start.yaw + std::clamp(bearing_error, -local_goal_bearing_limit_, local_goal_bearing_limit_));

    auto accept_endpoint = [&](double x, double y, double yaw, double &score) {
      int ix = 0;
      int iy = 0;
      if (!worldToCostCell(x, y, ix, iy)) return false;

      double fc = 0.0;
      double clearance = kInf;
      double comfort_clearance = kInf;
      if (!footprintCostLocked(x, y, yaw, fc, &clearance, &comfort_clearance)) return false;

      const double hard_bonus = std::min(
        std::isfinite(clearance) ? clearance : clearance_distance_,
        clearance_distance_);
      const double comfort_bonus = std::min(
        std::isfinite(comfort_clearance) ? comfort_clearance : comfort_clearance_distance_,
        comfort_clearance_distance_);
      score = fc - 0.18 * hard_bonus - 0.12 * comfort_bonus;
      return true;
    };

    double endpoint_score = 0.0;
    if (dist <= local_goal_distance_ + 0.05 &&
        direct_bearing_gap <= direct_goal_bearing_limit_ &&
        accept_endpoint(raw_goal.x, raw_goal.y, raw_dir, endpoint_score)) {
      active_goal = raw_goal;
      active_goal.yaw = (dist > 1.2) ? raw_dir : raw_goal.yaw;
      guide_path = {Pose2{start.x, start.y, raw_dir, 0.0}, active_goal};
      updatePathArcLength(guide_path);
      reason = "raw goal direct curve candidate";
      return true;
    }

    const double max_d = std::min(dist, local_goal_distance_);
    if (max_d < 0.65) return false;

    Pose2 best;
    double best_score = kInf;
    bool found = false;
    const std::array<double, 9> lateral_offsets = {0.0, 0.30, -0.30, 0.60, -0.60, 0.90, -0.90, 1.20, -1.20};
    const double min_d = std::max(1.20, std::min(max_d, guide_min_distance_));
    for (double d = max_d; d >= min_d; d -= 0.30) {
      const double cx = start.x + d * std::cos(local_dir);
      const double cy = start.y + d * std::sin(local_dir);
      const double lateral_cap = std::max(0.55, 0.18 * d);
      for (double lateral : lateral_offsets) {
        if (std::abs(lateral) > lateral_cap) continue;
        const double x = cx - lateral * std::sin(local_dir);
        const double y = cy + lateral * std::cos(local_dir);

        double score = 0.0;
        if (!accept_endpoint(x, y, local_dir, score)) continue;
        const double raw_goal_cost = 0.03 * std::hypot(raw_goal.x - x, raw_goal.y - y);
        const double lateral_cost = 0.90 * std::abs(lateral);
        const double heading_cost = 0.18 * direct_bearing_gap * direct_bearing_gap;
        const double progress_reward = 0.85 * d;
        score += raw_goal_cost + lateral_cost + heading_cost - progress_reward;

        if (score < best_score) {
          best_score = score;
          best = Pose2{x, y, local_dir, 0.0};
          found = true;
        }
      }

      if (found && d > max_d - 0.60) break;
    }

    if (!found) return false;

    active_goal = best;
    guide_path = {Pose2{start.x, start.y, local_dir, 0.0}, active_goal};
    updatePathArcLength(guide_path);
    reason = "rolling direct curve candidate";
    return true;
  }

  bool resolveActiveGoalLocked(
    const OdomState &odom,
    Pose2 &active_goal,
    std::string &reason,
    std::vector<Pose2> &guide_path)
  {
    std::lock_guard<std::mutex> glock(goal_mutex_);
    if (!have_goal_) return false;
    guide_path.clear();

    active_goal = raw_goal_;
    const double dx = raw_goal_.x - odom.x;
    const double dy = raw_goal_.y - odom.y;
    const double dist = std::hypot(dx, dy);
    if (dist < 1e-3) {
      reason = "goal at robot";
      return true;
    }
    const double raw_dir = std::atan2(dy, dx);
    const double bearing_error = wrapAngle(raw_dir - odom.yaw);
    const double local_dir = wrapAngle(
      odom.yaw + std::clamp(bearing_error, -local_goal_bearing_limit_, local_goal_bearing_limit_));
    const double direct_bearing_gap = std::abs(bearing_error);
    active_goal.yaw = (dist > 1.2) ? local_dir : raw_goal_.yaw;

    if (!map_ready_ || width_ <= 0 || height_ <= 0) {
      const double d = std::min(dist, local_goal_distance_);
      active_goal.x = odom.x + d * std::cos(local_dir);
      active_goal.y = odom.y + d * std::sin(local_dir);
      reason = "map not ready, use bounded raw direction";
      return true;
    }

    double dummy = 0.0;
    Pose2 raw_start{odom.x, odom.y, raw_dir, 0.0};
    Pose2 raw_target{raw_goal_.x, raw_goal_.y, raw_dir, 0.0};
    if (dist <= local_goal_distance_ &&
        direct_bearing_gap <= direct_goal_bearing_limit_ &&
        raw_goal_.x >= min_x_ && raw_goal_.x <= max_x_ &&
        raw_goal_.y >= min_y_ && raw_goal_.y <= max_y_ &&
        footprintCostLocked(raw_goal_.x, raw_goal_.y, raw_dir, dummy) &&
        straightSegmentSafeLocked(raw_start, raw_target, 0.18)) {
      active_goal = raw_goal_;
      active_goal.yaw = (dist > 1.2) ? raw_dir : raw_goal_.yaw;
      guide_path = {Pose2{odom.x, odom.y, raw_dir, 0.0}, active_goal};
      updatePathArcLength(guide_path);
      reason = "raw goal inside terrain window";
      return true;
    }

    Pose2 guide_goal;
    std::string guide_reason;
    if (resolveGuideGoalLocked(odom, raw_goal_, guide_goal, guide_reason, guide_path)) {
      active_goal = guide_goal;
      reason = guide_reason;
      return true;
    }

    const double max_d = std::min(dist, local_goal_distance_);
    Pose2 best;
    double best_score = kInf;
    bool found = false;
    for (double d = max_d; d >= 1.0; d -= 0.35) {
      const double cx = odom.x + d * std::cos(local_dir);
      const double cy = odom.y + d * std::sin(local_dir);
      for (double lateral : {0.0, 0.45, -0.45, 0.90, -0.90, 1.35, -1.35}) {
        const double x = cx - lateral * std::sin(local_dir);
        const double y = cy + lateral * std::cos(local_dir);
        double fc = 0.0;
        double clearance = 0.0;
        if (!footprintCostLocked(x, y, local_dir, fc, &clearance)) continue;
        const double lateral_cost = std::abs(lateral) * 0.35;
        const double progress_bonus = d * 0.15;
        const double raw_goal_cost = 0.04 * std::hypot(raw_goal_.x - x, raw_goal_.y - y);
        const double bearing_cost = 0.20 * std::abs(wrapAngle(local_dir - raw_dir));
        const double score = fc + lateral_cost + raw_goal_cost + bearing_cost - progress_bonus -
                             std::min(clearance, clearance_distance_) * 0.08;
        if (score < best_score) {
          best_score = score;
          best = Pose2{x, y, local_dir, 0.0};
          found = true;
        }
      }
      if (found && d > max_d - 0.8) break;
    }

    if (!found) {
      reason = "no local projection found";
      return false;
    }
    active_goal = best;
    guide_path = {Pose2{odom.x, odom.y, local_dir, 0.0}, active_goal};
    updatePathArcLength(guide_path);
    reason = "projected rolling subgoal";
    return true;
  }

  int yawToBin(double yaw) const
  {
    double a = wrapAngle(yaw);
    if (a < 0.0) a += 2.0 * M_PI;
    int bin = static_cast<int>(std::floor(a / (2.0 * M_PI) * yaw_bins_));
    return std::clamp(bin, 0, yaw_bins_ - 1);
  }

  double binToYaw(int bin) const
  {
    return wrapAngle((static_cast<double>(bin) + 0.5) * 2.0 * M_PI / static_cast<double>(yaw_bins_));
  }

  int64_t keyFor(int ix, int iy, int yaw_bin) const
  {
    return (static_cast<int64_t>(iy) * width_ + ix) * yaw_bins_ + yaw_bin;
  }

  double heuristic(double x, double y, double yaw, const Pose2 &goal) const
  {
    const double d = std::hypot(goal.x - x, goal.y - y);
    const double yaw_cost = (d < 2.0) ? 0.20 * std::abs(wrapAngle(goal.yaw - yaw)) : 0.0;
    return heuristic_weight_ * (d + yaw_cost);
  }

  struct SearchNode
  {
    double x = 0.0;
    double y = 0.0;
    double yaw = 0.0;
    double g = 0.0;
    double f = 0.0;
    double length = 0.0;
    double steer = 0.0;
    int ix = 0;
    int iy = 0;
    int yaw_bin = 0;
    int parent = -1;
  };

  struct PrimitiveResult
  {
    bool ok = false;
    Pose2 end;
    double terrain_cost = 0.0;
    double comfort_cost = 0.0;
    double min_clearance = kInf;
    double min_comfort_clearance = kInf;
  };

  PrimitiveResult propagatePrimitiveLocked(const SearchNode &node, double steer) const
  {
    PrimitiveResult result;
    double x = node.x;
    double y = node.y;
    double yaw = node.yaw;
    double terrain_sum = 0.0;
    double comfort_sum = 0.0;
    double min_clearance = kInf;
    double min_comfort_clearance = kInf;
    const int steps = std::max(2, static_cast<int>(std::ceil(primitive_length_ / std::max(0.03, integration_step_))));
    const double ds = primitive_length_ / static_cast<double>(steps);
    const double curvature = std::tan(steer) / std::max(1e-3, wheel_base_);

    for (int i = 0; i < steps; ++i) {
      x += ds * std::cos(yaw);
      y += ds * std::sin(yaw);
      yaw = wrapAngle(yaw + ds * curvature);

      double fc = 0.0;
      double clearance = 0.0;
      double comfort_clearance = 0.0;
      if (!footprintCostLocked(x, y, yaw, fc, &clearance, &comfort_clearance)) return result;
      terrain_sum += fc * ds;
      min_clearance = std::min(min_clearance, clearance);
      min_comfort_clearance = std::min(min_comfort_clearance, comfort_clearance);
      if (std::isfinite(comfort_clearance) && comfort_clearance < comfort_clearance_distance_) {
        const double margin =
          (comfort_clearance_distance_ - comfort_clearance) / std::max(1e-3, comfort_clearance_distance_);
        comfort_sum += comfort_clearance_weight_ * margin * margin * ds;
      }
    }

    int ix = 0, iy = 0;
    if (!worldToCostCell(x, y, ix, iy)) {
      if (!allow_unknown_) return result;
    }

    result.ok = true;
    result.end = Pose2{x, y, yaw, 0.0};
    result.terrain_cost = terrain_sum;
    result.comfort_cost = comfort_sum;
    result.min_clearance = min_clearance;
    result.min_comfort_clearance = min_comfort_clearance;
    return result;
  }

  std::vector<double> steerSamples() const
  {
    return {
      -max_steer_,
      -0.66 * max_steer_,
      -0.33 * max_steer_,
      0.0,
      0.33 * max_steer_,
      0.66 * max_steer_,
      max_steer_,
    };
  }

  bool shouldCommitPreviousPath(const std::string &reason) const
  {
    if (!path_commit_enable_) return false;
    return reason == "rolling local goal should advance" ||
           reason == "path almost consumed";
  }

  bool buildCommittedPrefixLocked(
    const OdomState &odom,
    const std::vector<Pose2> &old_path,
    std::vector<Pose2> &prefix,
    Pose2 &search_start) const
  {
    prefix.clear();
    if (!path_commit_enable_ || old_path.size() < 4 || path_commit_distance_ <= 1e-3) {
      return false;
    }

    double track_dist = 0.0;
    const size_t nearest = nearestPathIndex(old_path, odom.x, odom.y, &track_dist);
    if (track_dist > path_commit_max_track_error_) {
      return false;
    }
    const double nearest_heading_gap = std::abs(wrapAngle(old_path[nearest].yaw - odom.yaw));
    if (nearest_heading_gap > 0.42) {
      return false;
    }

    const double start_s = old_path[nearest].s;
    const double requested_end_s = std::min(old_path.back().s, start_s + path_commit_distance_);
    if (requested_end_s - start_s < path_commit_min_distance_) {
      return false;
    }

    constexpr double kStep = 0.25;
    double last_safe_s = start_s;
    for (double s = start_s; s <= requested_end_s + 1e-6; s += kStep) {
      Pose2 p = samplePathAt(old_path, std::min(s, requested_end_s));
      double fc = 0.0;
      double clearance = kInf;
      if (!footprintCostLocked(p.x, p.y, p.yaw, fc, &clearance)) {
        break;
      }
      prefix.push_back(p);
      last_safe_s = p.s;
    }

    if (last_safe_s - start_s < path_commit_min_distance_ || prefix.size() < 2) {
      prefix.clear();
      return false;
    }

    search_start = prefix.back();
    updateYawFromTangent(prefix);
    updatePathArcLength(prefix);
    return true;
  }

  std::vector<Pose2> mergeCommittedPrefix(
    std::vector<Pose2> prefix,
    const std::vector<Pose2> &tail) const
  {
    if (prefix.empty()) return tail;
    if (tail.empty()) return prefix;

    std::vector<Pose2> merged = std::move(prefix);
    size_t tail_begin = 0;
    if (std::hypot(tail.front().x - merged.back().x, tail.front().y - merged.back().y) < 0.30) {
      tail_begin = 1;
    }
    for (size_t i = tail_begin; i < tail.size(); ++i) {
      merged.push_back(tail[i]);
    }
    updateYawFromTangent(merged);
    updatePathArcLength(merged);
    return merged;
  }

  void requestPlan(const std::string &reason)
  {
    {
      std::lock_guard<std::mutex> lock(planner_mutex_);
      planner_reason_ = reason;
      planner_wakeup_ = true;
    }
    plan_requested_.store(true, std::memory_order_release);
    planner_cv_.notify_one();
  }

  bool goalSequenceCurrent(uint64_t sequence)
  {
    std::lock_guard<std::mutex> lock(goal_mutex_);
    return have_goal_ && goal_seq_ == sequence;
  }

  void plannerLoop()
  {
    while (true) {
      std::string reason;
      {
        std::unique_lock<std::mutex> lock(planner_mutex_);
        planner_cv_.wait(lock, [this]() { return planner_stop_ || planner_wakeup_; });
        if (planner_stop_) return;
        planner_wakeup_ = false;
        reason = planner_reason_;
      }

      if (!haveGoalThreadSafe() || !map_ready_) {
        plan_requested_.store(false, std::memory_order_release);
        continue;
      }

      planner_running_.store(true, std::memory_order_release);
      const auto start = std::chrono::steady_clock::now();
      const uint64_t map_seq_before = map_update_seq_.load(std::memory_order_acquire);
      plan_requested_.store(false, std::memory_order_release);
      const bool ok = planPath(reason);
      const double elapsed_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
      last_plan_time_sec_.store(this->now().seconds(), std::memory_order_release);
      last_plan_elapsed_ms_.store(elapsed_ms, std::memory_order_release);
      planner_running_.store(false, std::memory_order_release);

      // A map callback may have arrived while planPath held the map mutex. Do
      // not erase that newer invalidation when the old plan completes.
      if (map_update_seq_.load(std::memory_order_acquire) == map_seq_before) {
        map_dirty_.store(false, std::memory_order_release);
      }
      if (!ok) plan_requested_.store(true, std::memory_order_release);

      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "terrain_nav planner: result=%s reason=%s elapsed=%.1fms map_seq=%llu",
        ok ? "success" : "failed", reason.c_str(), elapsed_ms,
        static_cast<unsigned long long>(map_update_seq_.load(std::memory_order_acquire)));
    }
  }

  bool planPath(const std::string &replan_reason)
  {
    OdomState odom;
    {
      std::lock_guard<std::mutex> lock(odom_mutex_);
      odom = odom_;
    }
    if (!odom.valid) return false;

    Pose2 raw_goal_snapshot;
    uint64_t goal_seq_snapshot = 0;
    {
      std::lock_guard<std::mutex> lock(goal_mutex_);
      if (!have_goal_) return false;
      raw_goal_snapshot = raw_goal_;
      goal_seq_snapshot = goal_seq_;
    }

    std::vector<Pose2> old_path;
    bool old_have_path = false;
    {
      std::lock_guard<std::mutex> lock(path_mutex_);
      old_have_path = have_path_;
      old_path = path_;
    }

    std::lock_guard<std::mutex> map_lock(map_mutex_);
    if (!map_ready_ || width_ <= 0 || height_ <= 0) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "terrain_nav waits for fused_map");
      return false;
    }

    Pose2 search_start{odom.x, odom.y, odom.yaw, 0.0};
    std::vector<Pose2> committed_prefix;
    const bool committed =
      old_have_path &&
      shouldCommitPreviousPath(replan_reason) &&
      buildCommittedPrefixLocked(odom, old_path, committed_prefix, search_start);
    if (committed) {
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "terrain_nav committed %.2fm of previous path before replanning",
        committed_prefix.back().s);
    }

    int sx = 0, sy = 0;
    if (!worldToCostCell(search_start.x, search_start.y, sx, sy)) {
      RCLCPP_WARN(get_logger(), "Start pose is outside terrain map, skip plan");
      return false;
    }

    const double reverse_dx = raw_goal_snapshot.x - odom.x;
    const double reverse_dy = raw_goal_snapshot.y - odom.y;
    if (reverse_enable_ && std::hypot(reverse_dx, reverse_dy) <= reverse_max_distance_ &&
        reverse_dx * std::cos(odom.yaw) + reverse_dy * std::sin(odom.yaw) < -0.30) {
      const auto arc = terrain_nav::reverseArc(odom.x, odom.y, odom.yaw,
          raw_goal_snapshot.x, raw_goal_snapshot.y, reverse_max_distance_, maxReferenceCurvature());
      std::vector<Pose2> reverse_path;
      for (const auto &p : arc) reverse_path.push_back(Pose2{p[0], p[1], wrapAngle(p[2]), 0.0, -1});
      bool reverse_safe = false;
      if (reverse_path.size() >= 2) {
        std::vector<Pose2> diagnostic = reverse_path;
        updateYawFromTangent(diagnostic);
        updatePathArcLength(diagnostic);
        reverse_safe = true;
        for (double ds = 0.0; ds <= diagnostic.back().s + 1e-6; ds += 0.05) {
          const Pose2 sample = samplePathAt(diagnostic, std::min(ds, diagnostic.back().s));
          double fc = 0.0, clearance = kInf, comfort = kInf;
          if (!footprintCostLocked(sample.x, sample.y, sample.yaw, fc, &clearance, &comfort, true) ||
              (std::isfinite(clearance) && clearance < hardClearanceFloor())) {
            reverse_safe = false;
            break;
          }
        }
      }
      if (reverse_path.size() < 2 || !reverse_safe) {
        if (!reverse_path.empty()) {
          std::vector<Pose2> diagnostic = reverse_path;
          updateYawFromTangent(diagnostic);
          updatePathArcLength(diagnostic);
          for (double ds = 0.0; ds <= diagnostic.back().s + 1e-6; ds += 0.10) {
            const Pose2 sample = samplePathAt(diagnostic, std::min(ds, diagnostic.back().s));
            double fc = 0.0, clearance = kInf;
            if (!footprintCostLocked(sample.x, sample.y, sample.yaw, fc, &clearance, nullptr, true) ||
                (std::isfinite(clearance) && clearance < hardClearanceFloor())) {
              RCLCPP_WARN_THROTTLE(
                get_logger(), *get_clock(), 1000,
                "Reverse safety first failure: s=%.2f pos=(%.2f,%.2f) yaw=%.2f cost=%.2f clearance=%.2f floor=%.2f",
                ds, sample.x, sample.y, sample.yaw, fc, clearance, hardClearanceFloor());
              break;
            }
          }
        }
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "Reverse candidate rejected: arc_points=%zu max_curvature=%.3f safe=%s start=(%.2f,%.2f) goal=(%.2f,%.2f)",
          reverse_path.size(), maxReferenceCurvature(), reverse_safe ? "yes" : "no",
          odom.x, odom.y, raw_goal_snapshot.x, raw_goal_snapshot.y);
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                            "Close rear goal has no safe reverse arc; stopping instead of taking a forward detour");
        return false;
      }
      updateYawFromTangent(reverse_path);
      updatePathArcLength(reverse_path);
      if (!goalSequenceCurrent(goal_seq_snapshot)) return false;
      {
        std::lock_guard<std::mutex> lock(path_mutex_);
        path_ = std::move(reverse_path);
        active_goal_ = path_.back();
        have_path_ = true;
        ++path_seq_;
      }
      publishActiveGoal(raw_goal_snapshot, true);
      publishPath();
      RCLCPP_INFO(get_logger(), "terrain reverse arc success: close rear goal, distance=%.2fm",
                  std::hypot(reverse_dx, reverse_dy));
      return true;
    }

    std::vector<Pose2> continuous_path;
    Pose2 direct_goal;
    std::string direct_reason;
    std::vector<Pose2> direct_guide_path;
    double direct_blocked_s = kInf;
    const bool direct_corridor_blocked = directCorridorBlockedLocked(
      search_start,
      raw_goal_snapshot,
      std::min(local_goal_distance_ + 1.2, std::hypot(raw_goal_snapshot.x - search_start.x, raw_goal_snapshot.y - search_start.y)),
      direct_blocked_s);
    if (direct_corridor_blocked) {
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 1200,
        "terrain_nav skips direct rolling goal: blocked corridor at %.2fm", direct_blocked_s);
    }
    if (!direct_corridor_blocked &&
        resolveDirectLocalGoalLocked(search_start, raw_goal_snapshot, direct_goal, direct_reason, direct_guide_path) &&
        buildContinuousCurvePathLocked(search_start, direct_goal, continuous_path)) {
      if (committed) {
        continuous_path = mergeCommittedPrefix(std::move(committed_prefix), continuous_path);
        refinePathForTrackingLocked(continuous_path);
      }
      updatePathArcLength(continuous_path);
      if (!goalSequenceCurrent(goal_seq_snapshot)) return false;

      size_t planned_path_size = 0;
      {
        std::lock_guard<std::mutex> lock(path_mutex_);
        path_ = std::move(continuous_path);
        active_goal_ = direct_goal;
        have_path_ = path_.size() >= 2;
        planned_path_size = path_.size();
        path_seq_++;
        resetNominalControls();
      }

      publishActiveGoal(direct_goal, true);
      publishPath();
      consecutive_path_unsafe_ = 0;
      RCLCPP_INFO(
        get_logger(), "terrain direct continuous curve success: %zu pts, reason=%s",
        planned_path_size, direct_reason.c_str());
      return true;
    }

    Pose2 active_goal;
    std::string goal_reason;
    std::vector<Pose2> guide_path;
    if (!resolveActiveGoalLocked(odom, active_goal, goal_reason, guide_path)) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "Cannot resolve active terrain goal");
      return false;
    }
    publishActiveGoal(active_goal, true);

    if (buildContinuousCurvePathLocked(search_start, active_goal, continuous_path)) {
      if (committed) {
        continuous_path = mergeCommittedPrefix(std::move(committed_prefix), continuous_path);
        refinePathForTrackingLocked(continuous_path);
      }
      updatePathArcLength(continuous_path);
      if (!goalSequenceCurrent(goal_seq_snapshot)) return false;

      size_t planned_path_size = 0;
      {
        std::lock_guard<std::mutex> lock(path_mutex_);
        path_ = std::move(continuous_path);
        active_goal_ = active_goal;
        have_path_ = path_.size() >= 2;
        planned_path_size = path_.size();
        path_seq_++;
        resetNominalControls();
      }

      publishPath();
      consecutive_path_unsafe_ = 0;
      RCLCPP_INFO(
        get_logger(), "terrain continuous curve success: %zu pts, reason=%s",
        planned_path_size, goal_reason.c_str());
      return true;
    }

    const double goal_dx = active_goal.x - search_start.x;
    const double goal_dy = active_goal.y - search_start.y;
    const double goal_dist_from_start = std::hypot(goal_dx, goal_dy);
    const double goal_dir = (goal_dist_from_start > 1e-3)
      ? std::atan2(goal_dy, goal_dx)
      : search_start.yaw;
    const double ux = std::cos(goal_dir);
    const double uy = std::sin(goal_dir);
    const bool use_progress_guard = goal_dist_from_start > 1.2;
    const double max_tail_length = std::max(
      goal_dist_from_start + 2.6,
      goal_dist_from_start * max_extra_path_ratio_);
    auto progressAlongGoal = [&](double x, double y) {
      return (x - search_start.x) * ux + (y - search_start.y) * uy;
    };
    auto lateralToGoalRay = [&](double x, double y) {
      return -(x - search_start.x) * uy + (y - search_start.y) * ux;
    };

    const auto start_time = std::chrono::steady_clock::now();
    std::vector<SearchNode> nodes;
    nodes.reserve(4096);
    std::unordered_map<int64_t, double> best_g;
    best_g.reserve(20000);

    std::vector<Pose2> guide_tail;
    bool use_guide_tail = false;
    if (guide_path.size() >= 2) {
      guide_tail = makeGuideTailFromSearchStart(guide_path, search_start, active_goal);
      use_guide_tail = guide_tail.size() >= 2;
    }

    auto cmp = [](const std::pair<double, int> &a, const std::pair<double, int> &b) {
      return a.first > b.first;
    };
    std::priority_queue<std::pair<double, int>, std::vector<std::pair<double, int>>, decltype(cmp)> open(cmp);

    SearchNode start;
    start.x = search_start.x;
    start.y = search_start.y;
    start.yaw = search_start.yaw;
    start.ix = sx;
    start.iy = sy;
    start.yaw_bin = yawToBin(search_start.yaw);
    start.g = 0.0;
    start.f = heuristic(start.x, start.y, start.yaw, active_goal);
    start.length = 0.0;
    nodes.push_back(start);
    best_g[keyFor(start.ix, start.iy, start.yaw_bin)] = 0.0;
    open.push({start.f, 0});

    int goal_node = -1;
    int best_frontier = 0;
    double best_frontier_score = kInf;
    int iterations = 0;
    int reject_primitive = 0;
    int reject_outside = 0;
    int reject_progress = 0;
    int reject_duplicate = 0;
    const auto steers = steerSamples();

    while (!open.empty()) {
      const int idx = open.top().second;
      open.pop();
      const SearchNode cur = nodes[static_cast<size_t>(idx)];
      ++iterations;

      const double h = heuristic(cur.x, cur.y, cur.yaw, active_goal);
      if (cur.length >= 1.0) {
        double frontier_score = h;
        if (use_guide_tail) {
          double guide_yaw = cur.yaw;
          const double guide_dist = distanceToPath2D(guide_tail, cur.x, cur.y, &guide_yaw);
          frontier_score += 3.5 * guide_dist + 0.55 * std::abs(wrapAngle(cur.yaw - guide_yaw));
        }
        if (frontier_score < best_frontier_score) {
          best_frontier_score = frontier_score;
          best_frontier = idx;
        }
      }

      const double goal_dist = std::hypot(cur.x - active_goal.x, cur.y - active_goal.y);
      const double goal_yaw_gap = std::abs(wrapAngle(cur.yaw - active_goal.yaw));
      if (goal_dist <= goal_tolerance_ &&
          (goal_yaw_gap <= goal_yaw_tolerance_ || std::hypot(active_goal.x - search_start.x, active_goal.y - search_start.y) > 2.0)) {
        goal_node = idx;
        break;
      }

      const auto now = std::chrono::steady_clock::now();
      const double elapsed_ms = std::chrono::duration<double, std::milli>(now - start_time).count();
      if (iterations >= max_iterations_ || elapsed_ms >= max_search_time_ms_) break;

      for (double steer : steers) {
        const auto prim = propagatePrimitiveLocked(cur, steer);
        if (!prim.ok) {
          ++reject_primitive;
          continue;
        }
        int ix = 0, iy = 0;
        if (!worldToCostCell(prim.end.x, prim.end.y, ix, iy)) {
          ++reject_outside;
          continue;
        }
        const double cur_progress = progressAlongGoal(cur.x, cur.y);
        const double next_progress = progressAlongGoal(prim.end.x, prim.end.y);
        double guide_dist = std::abs(lateralToGoalRay(prim.end.x, prim.end.y));
        double reference_yaw = goal_dir;
        if (use_guide_tail) {
          double nearest_yaw = goal_dir;
          guide_dist = distanceToPath2D(guide_tail, prim.end.x, prim.end.y, &nearest_yaw);
          if (std::isfinite(nearest_yaw)) reference_yaw = nearest_yaw;
        }
        const double delta_progress = next_progress - cur_progress;
        const double next_length = cur.length + primitive_length_;
        if (use_progress_guard && next_length > max_tail_length) {
          ++reject_progress;
          continue;
        }
        const int yaw_bin = yawToBin(prim.end.yaw);
        const int64_t key = keyFor(ix, iy, yaw_bin);
        const double steer_norm = std::abs(steer) / std::max(1e-3, max_steer_);
        const double dsteer_norm = std::abs(steer - cur.steer) / std::max(1e-3, max_steer_);
        const bool reversing_steer =
          std::abs(steer) > 0.15 * max_steer_ &&
          std::abs(cur.steer) > 0.15 * max_steer_ &&
          steer * cur.steer < 0.0;
        const double lateral_norm = guide_dist / std::max(1.0, goal_dist_from_start);
        const double forward_deficit = std::max(0.0, 0.18 * primitive_length_ - delta_progress);
        const double backtrack = std::max(0.0, -delta_progress - backtrack_reject_distance_);
        const double heading_gap = std::abs(wrapAngle(prim.end.yaw - reference_yaw));
        const double lateral_tracking_weight = use_guide_tail ? guide_follow_weight_ : path_lateral_weight_;
        const double edge_cost = primitive_length_ *
          (1.0 + steer_weight_ * steer_norm * steer_norm + steer_change_weight_ * dsteer_norm) +
          prim.terrain_cost +
          prim.comfort_cost +
          steer_reversal_weight_ * (reversing_steer ? 1.0 : 0.0) * primitive_length_ +
          progress_weight_ * (forward_deficit + 2.0 * backtrack) +
          lateral_tracking_weight * lateral_norm * lateral_norm * primitive_length_ +
          goal_heading_weight_ * heading_gap * heading_gap * primitive_length_;
        const double ng = cur.g + edge_cost;
        auto it = best_g.find(key);
        if (it != best_g.end() && it->second <= ng) {
          ++reject_duplicate;
          continue;
        }

        SearchNode next;
        next.x = prim.end.x;
        next.y = prim.end.y;
        next.yaw = binToYaw(yaw_bin);
        next.g = ng;
        next.f = ng + heuristic(next.x, next.y, next.yaw, active_goal);
        next.length = next_length;
        next.steer = steer;
        next.ix = ix;
        next.iy = iy;
        next.yaw_bin = yaw_bin;
        next.parent = idx;
        const int next_idx = static_cast<int>(nodes.size());
        nodes.push_back(next);
        best_g[key] = ng;
        open.push({next.f, next_idx});
      }
    }

    int chosen = goal_node;
    bool partial = false;
    if (chosen < 0 && accept_partial_progress_) {
      const double progress = std::hypot(nodes[static_cast<size_t>(best_frontier)].x - search_start.x,
                                         nodes[static_cast<size_t>(best_frontier)].y - search_start.y);
      if (progress > 1.0) {
        chosen = best_frontier;
        partial = true;
      }
    }
    if (chosen < 0) {
      const auto &frontier = nodes[static_cast<size_t>(best_frontier)];
      const double frontier_dist = std::hypot(frontier.x - search_start.x, frontier.y - search_start.y);
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 800,
                           "terrain lattice failed: iter=%d nodes=%zu frontier=(%.2f, %.2f) dist=%.2f goal=(%.2f, %.2f) reject primitive/outside/progress/duplicate=%d/%d/%d/%d se2_lethal=%s",
                           iterations, nodes.size(), frontier.x, frontier.y, frontier_dist,
                           active_goal.x, active_goal.y,
                           reject_primitive, reject_outside, reject_progress, reject_duplicate,
                           se2_risk_lethal_ ? "true" : "false");
      return false;
    }

    std::vector<Pose2> path = extractPath(nodes, chosen);
    if (committed) {
      path = mergeCommittedPrefix(std::move(committed_prefix), path);
    }
    refinePathForTrackingLocked(path);
    updatePathArcLength(path);
    if (!latticePathReasonableLocked(path, search_start, active_goal, partial)) {
      const double direct = std::hypot(active_goal.x - search_start.x, active_goal.y - search_start.y);
      const double progress = direct > 1e-3
        ? ((path.back().x - search_start.x) * (active_goal.x - search_start.x) +
           (path.back().y - search_start.y) * (active_goal.y - search_start.y)) / direct
        : 0.0;
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 800,
        "terrain lattice rejected unreasonable %s path: pts=%zu length=%.2f progress=%.2f end=(%.2f, %.2f) goal=(%.2f, %.2f)",
        partial ? "partial" : "complete", path.size(), path.back().s, progress,
        path.back().x, path.back().y, active_goal.x, active_goal.y);
      return false;
    }

    size_t planned_path_size = 0;
    {
      std::lock_guard<std::mutex> lock(path_mutex_);
      if (!goalSequenceCurrent(goal_seq_snapshot)) return false;
      path_ = std::move(path);
      active_goal_ = active_goal;
      have_path_ = path_.size() >= 2;
      planned_path_size = path_.size();
      path_seq_++;
      resetNominalControls();
    }

    publishPath();
    consecutive_path_unsafe_ = 0;
    const double elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_time).count();
    RCLCPP_INFO(
      get_logger(), "terrain lattice %s: %zu pts, %.1f ms, iter=%d, reason=%s",
      partial ? "partial" : "success", planned_path_size, elapsed_ms, iterations, goal_reason.c_str());
    return true;
  }

  std::vector<Pose2> extractPath(const std::vector<SearchNode> &nodes, int chosen) const
  {
    std::vector<int> chain;
    int idx = chosen;
    while (idx >= 0) {
      chain.push_back(idx);
      const auto &n = nodes[static_cast<size_t>(idx)];
      idx = n.parent;
    }
    std::reverse(chain.begin(), chain.end());
    if (chain.empty()) return {};

    std::vector<Pose2> path;
    path.reserve(chain.size() * 5);
    const auto &start = nodes[static_cast<size_t>(chain.front())];
    path.push_back(Pose2{start.x, start.y, start.yaw, 0.0});

    const double sample_step = std::clamp(reference_resample_spacing_ * 0.65, 0.08, 0.18);
    for (size_t ci = 1; ci < chain.size(); ++ci) {
      const auto &parent = nodes[static_cast<size_t>(chain[ci - 1])];
      const auto &child = nodes[static_cast<size_t>(chain[ci])];
      double x = parent.x;
      double y = parent.y;
      double yaw = parent.yaw;
      const int steps = std::max(3, static_cast<int>(std::ceil(primitive_length_ / sample_step)));
      const double ds = primitive_length_ / static_cast<double>(steps);
      const double curvature = std::tan(child.steer) / std::max(1e-3, wheel_base_);
      for (int s = 1; s <= steps; ++s) {
        x += ds * std::cos(yaw);
        y += ds * std::sin(yaw);
        yaw = wrapAngle(yaw + ds * curvature);
        Pose2 p{x, y, yaw, 0.0};
        if (s == steps) {
          p.x = child.x;
          p.y = child.y;
          p.yaw = child.yaw;
        }
        if (std::hypot(p.x - path.back().x, p.y - path.back().y) > 0.03) {
          path.push_back(p);
        }
      }
    }

    updateYawFromTangent(path);
    updatePathArcLength(path);
    return path;
  }

  void updateYawFromTangent(std::vector<Pose2> &path) const
  {
    if (path.size() < 2) return;
    for (size_t i = 0; i + 1 < path.size(); ++i) {
      const double dx = path[i + 1].x - path[i].x;
      const double dy = path[i + 1].y - path[i].y;
      if (std::hypot(dx, dy) > 1e-4) {
        path[i].yaw = wrapAngle(std::atan2(dy, dx) + (path[i].direction < 0 ? M_PI : 0.0));
      }
    }
    path.back().yaw = path[path.size() - 2].yaw;
  }

  void updatePathArcLength(std::vector<Pose2> &path) const
  {
    if (path.empty()) return;
    path.front().s = 0.0;
    for (size_t i = 1; i < path.size(); ++i) {
      path[i].s = path[i - 1].s + std::hypot(path[i].x - path[i - 1].x, path[i].y - path[i - 1].y);
    }
  }

  double maxReferenceCurvature() const
  {
    return reference_max_curvature_ratio_ * std::tan(max_steer_) / std::max(1e-3, wheel_base_);
  }

  double pathMaxCurvature(std::vector<Pose2> path) const
  {
    if (path.size() < 3) return 0.0;
    updateYawFromTangent(path);
    double max_k = 0.0;
    for (size_t i = 1; i + 1 < path.size(); ++i) {
      const double dx0 = path[i].x - path[i - 1].x;
      const double dy0 = path[i].y - path[i - 1].y;
      const double dx1 = path[i + 1].x - path[i].x;
      const double dy1 = path[i + 1].y - path[i].y;
      const double l0 = std::hypot(dx0, dy0);
      const double l1 = std::hypot(dx1, dy1);
      if (l0 < 1e-4 || l1 < 1e-4) continue;
      const double yaw0 = std::atan2(dy0, dx0);
      const double yaw1 = std::atan2(dy1, dx1);
      const double ds = std::max(1e-3, 0.5 * (l0 + l1));
      max_k = std::max(max_k, std::abs(wrapAngle(yaw1 - yaw0)) / ds);
    }
    return max_k;
  }

  bool pathKinematicallyFeasible(std::vector<Pose2> path) const
  {
    if (path.size() < 3) return true;
    path = resamplePathUniform(path, std::clamp(reference_resample_spacing_, 0.12, 0.30));
    const double max_k = maxReferenceCurvature();
    return pathMaxCurvature(path) <= max_k + 1e-6;
  }

  bool latticePathReasonableLocked(
    std::vector<Pose2> path,
    const Pose2 &start,
    const Pose2 &goal,
    bool partial) const
  {
    if (path.size() < 2) return false;
    updateYawFromTangent(path);
    updatePathArcLength(path);

    const double chord = std::hypot(path.back().x - start.x, path.back().y - start.y);
    const double direct = std::hypot(goal.x - start.x, goal.y - start.y);
    if (chord < 0.45) return false;

    const double length = path.back().s;
    const double detour_limit = chord * (partial ? 1.80 : 2.05) + (partial ? 0.75 : 1.05);
    if (length > detour_limit) return false;

    if (direct > 0.8) {
      const double ux = (goal.x - start.x) / direct;
      const double uy = (goal.y - start.y) / direct;
      const double end_progress = (path.back().x - start.x) * ux + (path.back().y - start.y) * uy;
      if (partial && end_progress < std::max(0.90, 0.15 * std::min(direct, local_goal_distance_))) {
        return false;
      }

      double best_progress = 0.0;
      int regress_samples = 0;
      for (const auto &p : path) {
        const double progress = (p.x - start.x) * ux + (p.y - start.y) * uy;
        if (progress + 0.55 < best_progress) ++regress_samples;
        best_progress = std::max(best_progress, progress);
      }
      const int regress_limit = std::max(4, static_cast<int>(path.size() / (partial ? 8 : 6)));
      if (regress_samples > regress_limit) return false;
    }

    std::vector<Pose2> sampled = resamplePathUniform(path, std::max(0.18, reference_resample_spacing_));
    updateYawFromTangent(sampled);
    double total_abs_yaw = 0.0;
    double signed_yaw = 0.0;
    int reversals = 0;
    double last_turn = 0.0;
    for (size_t i = 1; i < sampled.size(); ++i) {
      const double dyaw = wrapAngle(sampled[i].yaw - sampled[i - 1].yaw);
      const double abs_dyaw = std::abs(dyaw);
      total_abs_yaw += abs_dyaw;
      signed_yaw += dyaw;
      if (abs_dyaw > 0.56) return false;
      if (abs_dyaw > 0.045 && std::abs(last_turn) > 0.045 && dyaw * last_turn < 0.0) {
        ++reversals;
      }
      if (abs_dyaw > 0.045) last_turn = dyaw;
    }

    const double turn_limit =
      partial ? std::min(3.80, 1.25 + 0.22 * length) : std::min(4.60, 1.55 + 0.25 * length);
    if (total_abs_yaw > turn_limit) return false;
    if (std::abs(signed_yaw) > 4.20) return false;
    if (partial && reversals > 3 && total_abs_yaw > 2.20) return false;
    return true;
  }

  static double mod2pi(double a)
  {
    a = std::fmod(a, 2.0 * M_PI);
    if (a < 0.0) a += 2.0 * M_PI;
    return a;
  }

  struct DubinsCandidate
  {
    std::array<char, 3> types = {'S', 'S', 'S'};
    std::array<double, 3> params = {0.0, 0.0, 0.0};
    double length = kInf;
  };

  static bool dubinsLSL(double alpha, double beta, double d, DubinsCandidate &out)
  {
    const double sa = std::sin(alpha);
    const double sb = std::sin(beta);
    const double ca = std::cos(alpha);
    const double cb = std::cos(beta);
    const double c_ab = std::cos(alpha - beta);
    const double tmp0 = d + sa - sb;
    const double p2 = 2.0 + d * d - 2.0 * c_ab + 2.0 * d * (sa - sb);
    if (p2 < 0.0) return false;
    const double tmp1 = std::atan2(cb - ca, tmp0);
    out.types = {'L', 'S', 'L'};
    out.params = {mod2pi(-alpha + tmp1), std::sqrt(p2), mod2pi(beta - tmp1)};
    out.length = out.params[0] + out.params[1] + out.params[2];
    return true;
  }

  static bool dubinsRSR(double alpha, double beta, double d, DubinsCandidate &out)
  {
    const double sa = std::sin(alpha);
    const double sb = std::sin(beta);
    const double ca = std::cos(alpha);
    const double cb = std::cos(beta);
    const double c_ab = std::cos(alpha - beta);
    const double tmp0 = d - sa + sb;
    const double p2 = 2.0 + d * d - 2.0 * c_ab + 2.0 * d * (-sa + sb);
    if (p2 < 0.0) return false;
    const double tmp1 = std::atan2(ca - cb, tmp0);
    out.types = {'R', 'S', 'R'};
    out.params = {mod2pi(alpha - tmp1), std::sqrt(p2), mod2pi(-beta + tmp1)};
    out.length = out.params[0] + out.params[1] + out.params[2];
    return true;
  }

  static bool dubinsLSR(double alpha, double beta, double d, DubinsCandidate &out)
  {
    const double sa = std::sin(alpha);
    const double sb = std::sin(beta);
    const double ca = std::cos(alpha);
    const double cb = std::cos(beta);
    const double c_ab = std::cos(alpha - beta);
    const double p2 = -2.0 + d * d + 2.0 * c_ab + 2.0 * d * (sa + sb);
    if (p2 < 0.0) return false;
    const double p = std::sqrt(p2);
    const double tmp2 = std::atan2(-ca - cb, d + sa + sb) - std::atan2(-2.0, p);
    out.types = {'L', 'S', 'R'};
    out.params = {mod2pi(-alpha + tmp2), p, mod2pi(-mod2pi(beta) + tmp2)};
    out.length = out.params[0] + out.params[1] + out.params[2];
    return true;
  }

  static bool dubinsRSL(double alpha, double beta, double d, DubinsCandidate &out)
  {
    const double sa = std::sin(alpha);
    const double sb = std::sin(beta);
    const double ca = std::cos(alpha);
    const double cb = std::cos(beta);
    const double c_ab = std::cos(alpha - beta);
    const double p2 = -2.0 + d * d + 2.0 * c_ab - 2.0 * d * (sa + sb);
    if (p2 < 0.0) return false;
    const double p = std::sqrt(p2);
    const double tmp2 = std::atan2(ca + cb, d - sa - sb) - std::atan2(2.0, p);
    out.types = {'R', 'S', 'L'};
    out.params = {mod2pi(alpha - tmp2), p, mod2pi(beta - tmp2)};
    out.length = out.params[0] + out.params[1] + out.params[2];
    return true;
  }

  static bool dubinsRLR(double alpha, double beta, double d, DubinsCandidate &out)
  {
    const double sa = std::sin(alpha);
    const double sb = std::sin(beta);
    const double ca = std::cos(alpha);
    const double cb = std::cos(beta);
    const double c_ab = std::cos(alpha - beta);
    const double tmp0 = (6.0 - d * d + 2.0 * c_ab + 2.0 * d * (sa - sb)) / 8.0;
    if (std::abs(tmp0) > 1.0) return false;
    const double p = mod2pi(2.0 * M_PI - std::acos(tmp0));
    const double t = mod2pi(alpha - std::atan2(ca - cb, d - sa + sb) + 0.5 * p);
    out.types = {'R', 'L', 'R'};
    out.params = {t, p, mod2pi(alpha - beta - t + p)};
    out.length = out.params[0] + out.params[1] + out.params[2];
    return true;
  }

  static bool dubinsLRL(double alpha, double beta, double d, DubinsCandidate &out)
  {
    const double sa = std::sin(alpha);
    const double sb = std::sin(beta);
    const double ca = std::cos(alpha);
    const double cb = std::cos(beta);
    const double c_ab = std::cos(alpha - beta);
    const double tmp0 = (6.0 - d * d + 2.0 * c_ab + 2.0 * d * (-sa + sb)) / 8.0;
    if (std::abs(tmp0) > 1.0) return false;
    const double p = mod2pi(2.0 * M_PI - std::acos(tmp0));
    const double t = mod2pi(-alpha - std::atan2(ca - cb, d + sa - sb) + 0.5 * p);
    out.types = {'L', 'R', 'L'};
    out.params = {t, p, mod2pi(beta - alpha - t + p)};
    out.length = out.params[0] + out.params[1] + out.params[2];
    return true;
  }

  bool sampleDubinsLocked(
    const Pose2 &start,
    const Pose2 &goal,
    const DubinsCandidate &candidate,
    double radius,
    std::vector<Pose2> &path) const
  {
    path.clear();
    path.push_back(start);

    double x = start.x;
    double y = start.y;
    double yaw = start.yaw;
    const double ds_nominal = std::clamp(reference_resample_spacing_ * 0.65, 0.08, 0.16);

    for (size_t seg = 0; seg < 3; ++seg) {
      const char type = candidate.types[seg];
      const double length = candidate.params[seg] * radius;
      if (length < 1e-5) continue;
      const int steps = std::max(2, static_cast<int>(std::ceil(length / ds_nominal)));
      const double ds = length / static_cast<double>(steps);
      for (int i = 0; i < steps; ++i) {
        if (type == 'S') {
          x += ds * std::cos(yaw);
          y += ds * std::sin(yaw);
        } else {
          const double sign = (type == 'L') ? 1.0 : -1.0;
          const double next_yaw = wrapAngle(yaw + sign * ds / radius);
          x += sign * radius * (std::sin(next_yaw) - std::sin(yaw));
          y += -sign * radius * (std::cos(next_yaw) - std::cos(yaw));
          yaw = next_yaw;
        }
        path.push_back(Pose2{x, y, yaw, 0.0});
      }
    }

    if (path.size() < 3) return false;
    path.back() = goal;
    updateYawFromTangent(path);
    updatePathArcLength(path);
    return true;
  }

  bool buildDubinsDirectPathLocked(
    const Pose2 &start,
    const Pose2 &goal,
    std::vector<Pose2> &best_path) const
  {
    best_path.clear();
    const double dx = goal.x - start.x;
    const double dy = goal.y - start.y;
    const double direct_dist = std::hypot(dx, dy);
    if (direct_dist < 0.65 || direct_dist > local_goal_distance_ + 0.8) return false;

    const double max_curvature = std::max(0.20, maxReferenceCurvature());
    const double radius = 1.0 / std::max(0.12, 0.62 * max_curvature);
    const double d = direct_dist / radius;
    if (d < 0.05) return false;

    const double theta = std::atan2(dy, dx);
    const double alpha = mod2pi(start.yaw - theta);
    const double beta = mod2pi(goal.yaw - theta);

    std::array<DubinsCandidate, 6> candidates;
    std::array<bool, 6> valid = {
      dubinsLSL(alpha, beta, d, candidates[0]),
      dubinsRSR(alpha, beta, d, candidates[1]),
      dubinsLSR(alpha, beta, d, candidates[2]),
      dubinsRSL(alpha, beta, d, candidates[3]),
      dubinsRLR(alpha, beta, d, candidates[4]),
      dubinsLRL(alpha, beta, d, candidates[5]),
    };

    double best_score = kInf;
    for (size_t i = 0; i < candidates.size(); ++i) {
      if (!valid[i]) continue;
      const double length = candidates[i].length * radius;
      if (length > direct_dist * direct_curve_max_length_ratio_ + 0.35) continue;
      if ((candidates[i].types[0] != candidates[i].types[2]) &&
          candidates[i].types[0] != 'S' && candidates[i].types[2] != 'S' &&
          candidates[i].params[1] < 0.35) {
        continue;
      }

      std::vector<Pose2> path;
      if (!sampleDubinsLocked(start, goal, candidates[i], radius, path)) continue;

      double score = kInf;
      const double goal_dir = std::atan2(dy, dx);
      if (!evaluateContinuousCurveLocked(path, start, goal, direct_dist, goal_dir, score)) continue;
      const double turn_sum = candidates[i].params[0] + candidates[i].params[2];
      score += 0.35 * length / std::max(1e-3, direct_dist) + 0.18 * turn_sum * turn_sum;
      if (score < best_score) {
        best_score = score;
        best_path = std::move(path);
      }
    }

    return best_path.size() >= 3;
  }

  bool evaluateContinuousCurveLocked(
    std::vector<Pose2> &path,
    const Pose2 &start,
    const Pose2 &goal,
    double direct_dist,
    double goal_dir,
    double &score) const
  {
    if (path.size() < 3) return false;
    updateYawFromTangent(path);
    updatePathArcLength(path);

    const double max_length = direct_dist * direct_curve_max_length_ratio_ + 0.40;
    if (path.back().s > max_length) return false;
    if (!pathKinematicallyFeasible(path)) return false;

    const double ux = std::cos(goal_dir);
    const double uy = std::sin(goal_dir);
    const double max_lateral = std::max(1.10, direct_curve_max_lateral_ratio_ * std::max(1.0, direct_dist));
    double prev_progress = 0.0;
    double max_lateral_seen = 0.0;
    double terrain_cost = 0.0;
    double clearance_cost = 0.0;
    double curvature_cost = 0.0;
    double last_curvature = 0.0;
    int curvature_reversals = 0;

    for (size_t i = 0; i < path.size(); ++i) {
      const auto &p = path[i];
      const double vx = p.x - start.x;
      const double vy = p.y - start.y;
      const double progress = vx * ux + vy * uy;
      const double lateral = std::abs(-vx * uy + vy * ux);
      if (progress < prev_progress - 0.28) return false;
      if (progress < -0.35) return false;
      if (lateral > max_lateral) return false;
      prev_progress = std::max(prev_progress, progress);
      max_lateral_seen = std::max(max_lateral_seen, lateral);

      double fc = 0.0;
      double clearance = kInf;
      double comfort_clearance = kInf;
      if (!footprintCostLocked(p.x, p.y, p.yaw, fc, &clearance, &comfort_clearance)) return false;
      terrain_cost += fc;
      if (std::isfinite(clearance) && clearance < hardClearanceFloor()) return false;
      if (std::isfinite(clearance) && clearance < clearance_distance_) {
        const double m = (clearance_distance_ - clearance) / std::max(1e-3, clearance_distance_);
        clearance_cost += 3.0 * m * m;
      }
      if (std::isfinite(comfort_clearance) && comfort_clearance < comfort_clearance_distance_) {
        const double m =
          (comfort_clearance_distance_ - comfort_clearance) / std::max(1e-3, comfort_clearance_distance_);
        clearance_cost += direct_curve_clearance_weight_ * m * m;
      }

      if (i > 0 && i + 1 < path.size()) {
        const double dx0 = path[i].x - path[i - 1].x;
        const double dy0 = path[i].y - path[i - 1].y;
        const double dx1 = path[i + 1].x - path[i].x;
        const double dy1 = path[i + 1].y - path[i].y;
        const double l0 = std::hypot(dx0, dy0);
        const double l1 = std::hypot(dx1, dy1);
        if (l0 > 1e-4 && l1 > 1e-4) {
          const double yaw0 = std::atan2(dy0, dx0);
          const double yaw1 = std::atan2(dy1, dx1);
          const double curvature = wrapAngle(yaw1 - yaw0) / std::max(1e-3, 0.5 * (l0 + l1));
          curvature_cost += curvature * curvature + 0.25 * std::pow(curvature - last_curvature, 2.0);
          if (std::abs(curvature) > 0.08 && std::abs(last_curvature) > 0.08 && curvature * last_curvature < 0.0) {
            ++curvature_reversals;
          }
          last_curvature = curvature;
        }
      }
    }

    if (curvature_reversals > 2) return false;

    const double goal_error = std::hypot(path.back().x - goal.x, path.back().y - goal.y);
    if (goal_error > std::max(0.35, goal_tolerance_ * 1.1)) return false;
    if (!pathFootprintSafeLocked(path, 0.14)) return false;

    const double length_ratio = path.back().s / std::max(1e-3, direct_dist);
    score =
      2.6 * length_ratio +
      0.30 * max_lateral_seen +
      0.020 * terrain_cost +
      0.018 * clearance_cost +
      0.28 * curvature_cost +
      0.80 * static_cast<double>(curvature_reversals) +
      0.80 * goal_error;
    return true;
  }

  bool buildFrenetDirectPathLocked(
    const Pose2 &start,
    const Pose2 &goal,
    std::vector<Pose2> &best_path) const
  {
    best_path.clear();

    const double dx = goal.x - start.x;
    const double dy = goal.y - start.y;
    const double direct_dist = std::hypot(dx, dy);
    if (direct_dist < 0.65 || direct_dist > local_goal_distance_ + 0.8) return false;

    const double goal_dir = std::atan2(dy, dx);
    const double heading_gap = wrapAngle(start.yaw - goal_dir);
    const std::array<double, 5> lateral_offsets = {0.0, 0.55, -0.55, 0.95, -0.95};
    const std::array<double, 3> start_slopes = {
      0.0,
      std::tan(std::clamp(0.35 * heading_gap, -0.38, 0.38)),
      std::tan(std::clamp(0.60 * heading_gap, -0.55, 0.55)),
    };
    const int samples = std::max(18, static_cast<int>(std::ceil(direct_dist / 0.14)));

    double best_score = kInf;
    const Eigen::Vector2d origin(start.x, start.y);
    const Eigen::Vector2d ex(std::cos(goal_dir), std::sin(goal_dir));
    const Eigen::Vector2d ey(-std::sin(goal_dir), std::cos(goal_dir));

    for (double m0 : start_slopes) {
      const double b0 = 0.0;
      const double b1 = m0 * direct_dist;
      const double b2 = 0.0;
      const double m1 = 0.0;
      Eigen::Matrix3d A;
      A << 1.0, 1.0, 1.0,
           3.0, 4.0, 5.0,
           6.0, 12.0, 20.0;
      Eigen::Vector3d rhs;
      rhs << -b1, m1 * direct_dist - b1, 0.0;
      const Eigen::Vector3d sol = A.fullPivLu().solve(rhs);
      const double b3 = sol.x();
      const double b4 = sol.y();
      const double b5 = sol.z();

      for (double offset : lateral_offsets) {
        if (std::abs(offset) > direct_curve_max_lateral_ratio_ * std::max(1.0, direct_dist)) continue;

        std::vector<Pose2> candidate;
        candidate.reserve(static_cast<size_t>(samples) + 1);
        for (int i = 0; i <= samples; ++i) {
          const double t = static_cast<double>(i) / static_cast<double>(samples);
          const double x_axis = direct_dist * t;
          const double bump = offset * 64.0 * t * t * t * std::pow(1.0 - t, 3.0);
          const double y_axis =
            b0 + b1 * t + b2 * t * t + b3 * t * t * t +
            b4 * t * t * t * t + b5 * t * t * t * t * t + bump;
          const Eigen::Vector2d p = origin + ex * x_axis + ey * y_axis;
          candidate.push_back(Pose2{p.x(), p.y(), 0.0, 0.0});
        }
        candidate.front() = start;
        candidate.back() = goal;
        updateYawFromTangent(candidate);
        updatePathArcLength(candidate);

        double score = kInf;
        if (!evaluateContinuousCurveLocked(candidate, start, goal, direct_dist, goal_dir, score)) continue;
        score += 0.80 * std::abs(offset) + 1.20 * std::abs(m0);
        if (score < best_score) {
          best_score = score;
          best_path = std::move(candidate);
        }
      }
    }

    if (best_path.size() < 3) return false;
    smoothReferenceLocked(best_path);
    updateYawFromTangent(best_path);
    updatePathArcLength(best_path);
    return pathFootprintSafeLocked(best_path, 0.14) && pathKinematicallyFeasible(best_path);
  }

  bool buildArcDirectPathLocked(
    const Pose2 &start,
    const Pose2 &goal,
    std::vector<Pose2> &best_path) const
  {
    best_path.clear();

    const double dx = goal.x - start.x;
    const double dy = goal.y - start.y;
    const double direct_dist = std::hypot(dx, dy);
    if (direct_dist < 0.45 || direct_dist > local_goal_distance_ + 0.8) return false;

    const double c = std::cos(start.yaw);
    const double s = std::sin(start.yaw);
    const double local_x = c * dx + s * dy;
    const double local_y = -s * dx + c * dy;
    if (local_x < 0.35) return false;

    const double denom = local_x * local_x + local_y * local_y;
    if (denom < 1e-6) return false;
    const double curvature = 2.0 * local_y / denom;
    const double max_curvature = maxReferenceCurvature();
    if (std::abs(curvature) > max_curvature + 1e-6) return false;

    const double heading_change = 2.0 * std::atan2(local_y, local_x);
    const double arc_length = (std::abs(curvature) > 1e-5)
      ? std::abs(heading_change / curvature)
      : direct_dist;
    if (arc_length > direct_dist * direct_curve_max_length_ratio_ + 0.40) return false;

    const int samples = std::max(8, static_cast<int>(std::ceil(arc_length / 0.14)));
    best_path.reserve(static_cast<size_t>(samples) + 1);
    for (int i = 0; i <= samples; ++i) {
      const double u = static_cast<double>(i) / static_cast<double>(samples);
      const double ds = arc_length * u;
      double lx = ds;
      double ly = 0.0;
      double yaw = start.yaw;
      if (std::abs(curvature) > 1e-5) {
        const double th = curvature * ds;
        lx = std::sin(th) / curvature;
        ly = (1.0 - std::cos(th)) / curvature;
        yaw = wrapAngle(start.yaw + th);
      }
      const double wx = start.x + c * lx - s * ly;
      const double wy = start.y + s * lx + c * ly;
      best_path.push_back(Pose2{wx, wy, yaw, 0.0});
    }
    best_path.front() = start;
    best_path.back().x = goal.x;
    best_path.back().y = goal.y;
    updateYawFromTangent(best_path);
    updatePathArcLength(best_path);

    double score = kInf;
    const double goal_dir = std::atan2(dy, dx);
    return evaluateContinuousCurveLocked(best_path, start, goal, direct_dist, goal_dir, score);
  }

  bool buildContinuousCurvePathLocked(
    const Pose2 &start,
    const Pose2 &goal,
    std::vector<Pose2> &best_path) const
  {
    best_path.clear();
    if (!direct_curve_enable_) return false;
    if (buildArcDirectPathLocked(start, goal, best_path)) return true;
    if (buildDubinsDirectPathLocked(start, goal, best_path)) return true;
    return buildFrenetDirectPathLocked(start, goal, best_path);
  }

  void smoothPathLocked(std::vector<Pose2> &path) const
  {
    if (path.size() < 4) return;
    for (int pass = 0; pass < smoothing_passes_; ++pass) {
      std::vector<Pose2> candidate;
      candidate.reserve(path.size() * 2);
      candidate.push_back(path.front());
      for (size_t i = 0; i + 1 < path.size(); ++i) {
        const auto &a = path[i];
        const auto &b = path[i + 1];
        Pose2 q;
        q.x = 0.75 * a.x + 0.25 * b.x;
        q.y = 0.75 * a.y + 0.25 * b.y;
        Pose2 r;
        r.x = 0.25 * a.x + 0.75 * b.x;
        r.y = 0.25 * a.y + 0.75 * b.y;
        candidate.push_back(q);
        candidate.push_back(r);
      }
      candidate.push_back(path.back());
      updateYawFromTangent(candidate);

      if (pathFootprintSafeLocked(candidate, 0.14) && pathKinematicallyFeasible(candidate)) {
        path.swap(candidate);
      }
    }
  }

  void shortcutGuidePathLocked(std::vector<Pose2> &path) const
  {
    if (path.size() < 5) return;

    std::vector<Pose2> src = path;
    updateYawFromTangent(src);
    updatePathArcLength(src);

    std::vector<Pose2> out;
    out.reserve(src.size());
    size_t i = 0;
    out.push_back(src.front());
    while (i + 1 < src.size()) {
      size_t best = i + 1;
      const size_t max_j = std::min(src.size() - 1, i + static_cast<size_t>(std::max(6.0, 3.2 / resolution_)));
      for (size_t j = max_j; j > i + 1; --j) {
        const double len = std::hypot(src[j].x - src[i].x, src[j].y - src[i].y);
        if (len > 3.4) continue;
        if (straightSegmentSafeLocked(src[i], src[j], 0.16)) {
          best = j;
          break;
        }
      }
      out.push_back(src[best]);
      i = best;
    }

    if (out.size() >= 2 && pathFootprintSafeLocked(out, 0.16)) {
      path = std::move(out);
      updateYawFromTangent(path);
      updatePathArcLength(path);
    }
  }

  void smoothGuidePathLocked(std::vector<Pose2> &path) const
  {
    if (path.size() < 5) return;

    const std::vector<Pose2> fallback = path;
    std::vector<Pose2> anchor = path;
    constexpr int kGuideIterations = 24;
    constexpr double kGuideStep = 0.20;
    constexpr double kAnchorWeight = 0.18;
    constexpr double kSmoothWeight = 0.95;
    constexpr double kCurvatureWeight = 0.18;

    for (int iter = 0; iter < kGuideIterations; ++iter) {
      std::vector<Pose2> candidate = path;
      for (size_t i = 1; i + 1 < path.size(); ++i) {
        const Eigen::Vector2d p(path[i].x, path[i].y);
        const Eigen::Vector2d a(anchor[i].x, anchor[i].y);
        const Eigen::Vector2d prev(path[i - 1].x, path[i - 1].y);
        const Eigen::Vector2d next(path[i + 1].x, path[i + 1].y);
        Eigen::Vector2d force = kAnchorWeight * (a - p) + kSmoothWeight * (prev + next - 2.0 * p);
        if (i > 1 && i + 2 < path.size()) {
          const Eigen::Vector2d p0(path[i - 2].x, path[i - 2].y);
          const Eigen::Vector2d p1(path[i - 1].x, path[i - 1].y);
          const Eigen::Vector2d p3(path[i + 1].x, path[i + 1].y);
          const Eigen::Vector2d p4(path[i + 2].x, path[i + 2].y);
          force -= kCurvatureWeight * (p0 - 4.0 * p1 + 6.0 * p - 4.0 * p3 + p4);
        }

        Eigen::Vector2d moved = p + kGuideStep * force;
        const Eigen::Vector2d shift = moved - a;
        const double max_shift = 0.55;
        if (shift.norm() > max_shift) {
          moved = a + shift * (max_shift / std::max(1e-6, shift.norm()));
        }
        candidate[i].x = moved.x();
        candidate[i].y = moved.y();
      }

      updateYawFromTangent(candidate);
      updatePathArcLength(candidate);
      if (pathFootprintSafeLocked(candidate, 0.16) && pathKinematicallyFeasible(candidate)) {
        path = std::move(candidate);
      }
    }

    if (!pathFootprintSafeLocked(path, 0.16) || !pathKinematicallyFeasible(path)) {
      path = fallback;
      updateYawFromTangent(path);
      updatePathArcLength(path);
    }
  }

  double distanceToPath2D(
    const std::vector<Pose2> &path,
    double x,
    double y,
    double *nearest_yaw = nullptr,
    double *nearest_s = nullptr) const
  {
    if (path.size() < 2) return kInf;

    double best_d2 = kInf;
    double best_yaw = path.front().yaw;
    double best_s = 0.0;
    for (size_t i = 0; i + 1 < path.size(); ++i) {
      const double ax = path[i].x;
      const double ay = path[i].y;
      const double bx = path[i + 1].x;
      const double by = path[i + 1].y;
      const double vx = bx - ax;
      const double vy = by - ay;
      const double len2 = vx * vx + vy * vy;
      if (len2 < 1e-8) continue;
      const double u = std::clamp(((x - ax) * vx + (y - ay) * vy) / len2, 0.0, 1.0);
      const double px = ax + u * vx;
      const double py = ay + u * vy;
      const double dx = x - px;
      const double dy = y - py;
      const double d2 = dx * dx + dy * dy;
      if (d2 < best_d2) {
        best_d2 = d2;
        best_yaw = std::atan2(vy, vx);
        best_s = path[i].s + u * std::sqrt(len2);
      }
    }

    if (nearest_yaw) *nearest_yaw = best_yaw;
    if (nearest_s) *nearest_s = best_s;
    return std::sqrt(best_d2);
  }

  std::vector<Pose2> makeGuideTailFromSearchStart(
    std::vector<Pose2> guide_path,
    const Pose2 &search_start,
    const Pose2 &active_goal) const
  {
    if (guide_path.size() < 2) {
      guide_path = {search_start, active_goal};
      updateYawFromTangent(guide_path);
      updatePathArcLength(guide_path);
      return guide_path;
    }

    updateYawFromTangent(guide_path);
    updatePathArcLength(guide_path);
    double nearest_dist = 0.0;
    const size_t nearest = nearestPathIndex(guide_path, search_start.x, search_start.y, &nearest_dist);
    std::vector<Pose2> tail;
    tail.reserve(guide_path.size() - nearest + 2);
    tail.push_back(search_start);
    for (size_t i = nearest + 1; i < guide_path.size(); ++i) {
      if (std::hypot(guide_path[i].x - tail.back().x, guide_path[i].y - tail.back().y) > 0.15) {
        tail.push_back(guide_path[i]);
      }
    }
    if (std::hypot(active_goal.x - tail.back().x, active_goal.y - tail.back().y) > 0.15) {
      tail.push_back(active_goal);
    } else {
      tail.back() = active_goal;
    }
    updateYawFromTangent(tail);
    updatePathArcLength(tail);
    return tail;
  }

  void refinePathForTrackingLocked(std::vector<Pose2> &path) const
  {
    if (path.size() < 4) return;

    const std::vector<Pose2> fallback = path;
    if (shortcut_smoothing_enable_) {
      shortcutPathLocked(path);
    }
    path = resamplePathUniform(path, reference_resample_spacing_);
    smoothReferenceLocked(path);
    smoothPathLocked(path);
    updateYawFromTangent(path);
    updatePathArcLength(path);

    if (!pathFootprintSafeLocked(path, 0.16) || !pathKinematicallyFeasible(path)) {
      path = fallback;
      updateYawFromTangent(path);
      updatePathArcLength(path);
    }
  }

  std::vector<Pose2> resamplePathUniform(const std::vector<Pose2> &path, double spacing) const
  {
    std::vector<Pose2> src = path;
    updateYawFromTangent(src);
    updatePathArcLength(src);
    if (src.size() < 2 || src.back().s < 1e-4) return src;

    std::vector<Pose2> out;
    const double ds = std::max(0.08, spacing);
    out.reserve(static_cast<size_t>(std::ceil(src.back().s / ds)) + 2);
    for (double s = 0.0; s < src.back().s; s += ds) {
      out.push_back(samplePathAt(src, s));
    }
    out.push_back(src.back());
    updateYawFromTangent(out);
    updatePathArcLength(out);
    return out;
  }

  double referenceSmoothQuality(const std::vector<Pose2> &path, const std::vector<Pose2> &anchor) const
  {
    if (path.size() != anchor.size() || path.empty()) return kInf;
    double cost = 0.0;
    for (size_t i = 0; i < path.size(); ++i) {
      const double ax = path[i].x - anchor[i].x;
      const double ay = path[i].y - anchor[i].y;
      cost += reference_anchor_weight_ * (ax * ax + ay * ay);
      if (i > 0 && i + 1 < path.size()) {
        const double ddx = path[i - 1].x - 2.0 * path[i].x + path[i + 1].x;
        const double ddy = path[i - 1].y - 2.0 * path[i].y + path[i + 1].y;
        cost += reference_smooth_weight_ * (ddx * ddx + ddy * ddy);
      }
      if (i > 1 && i + 2 < path.size()) {
        const double kx = path[i - 2].x - 4.0 * path[i - 1].x + 6.0 * path[i].x -
                          4.0 * path[i + 1].x + path[i + 2].x;
        const double ky = path[i - 2].y - 4.0 * path[i - 1].y + 6.0 * path[i].y -
                          4.0 * path[i + 1].y + path[i + 2].y;
        cost += reference_curvature_weight_ * (kx * kx + ky * ky);
      }
    }
    return cost;
  }

  void smoothReferenceLocked(std::vector<Pose2> &path) const
  {
    if (reference_smooth_iterations_ <= 0 || path.size() < 5) return;

    std::vector<Pose2> anchor = path;
    std::vector<Pose2> best = path;
    double best_quality = referenceSmoothQuality(best, anchor);

    for (int iter = 0; iter < reference_smooth_iterations_; ++iter) {
      std::vector<Pose2> candidate = best;
      for (size_t i = 1; i + 1 < best.size(); ++i) {
        const Eigen::Vector2d p(best[i].x, best[i].y);
        const Eigen::Vector2d a(anchor[i].x, anchor[i].y);
        const Eigen::Vector2d prev(best[i - 1].x, best[i - 1].y);
        const Eigen::Vector2d next(best[i + 1].x, best[i + 1].y);

        Eigen::Vector2d force =
          reference_anchor_weight_ * (a - p) +
          reference_smooth_weight_ * (prev + next - 2.0 * p);

        if (i > 1 && i + 2 < best.size()) {
          const Eigen::Vector2d p0(best[i - 2].x, best[i - 2].y);
          const Eigen::Vector2d p1(best[i - 1].x, best[i - 1].y);
          const Eigen::Vector2d p3(best[i + 1].x, best[i + 1].y);
          const Eigen::Vector2d p4(best[i + 2].x, best[i + 2].y);
          force -= reference_curvature_weight_ * (p0 - 4.0 * p1 + 6.0 * p - 4.0 * p3 + p4);
        }

        Eigen::Vector2d moved = p + reference_smooth_step_ * force;
        const double max_shift = 0.45;
        const Eigen::Vector2d from_anchor = moved - a;
        const double shift = from_anchor.norm();
        if (shift > max_shift) {
          moved = a + from_anchor * (max_shift / std::max(1e-6, shift));
        }

        candidate[i].x = moved.x();
        candidate[i].y = moved.y();
      }

      updateYawFromTangent(candidate);
      updatePathArcLength(candidate);
      if (!pathFootprintSafeLocked(candidate, 0.14) || !pathKinematicallyFeasible(candidate)) continue;

      const double quality = referenceSmoothQuality(candidate, anchor);
      if (quality <= best_quality + 1e-9) {
        best_quality = quality;
        best = std::move(candidate);
      }
    }

    path = std::move(best);
    updateYawFromTangent(path);
    updatePathArcLength(path);
  }

  bool straightSegmentSafeLocked(const Pose2 &a, const Pose2 &b, double step) const
  {
    const double dx = b.x - a.x;
    const double dy = b.y - a.y;
    const double len = std::hypot(dx, dy);
    if (len < 1e-4) return true;
    const double yaw = std::atan2(dy, dx);
    const double ds = std::clamp(step, 0.05, std::max(0.06, 0.45 * resolution_));
    const int steps = std::max(2, static_cast<int>(std::ceil(len / ds)));
    for (int i = 0; i <= steps; ++i) {
      const double u = static_cast<double>(i) / static_cast<double>(steps);
      const double x = a.x + u * dx;
      const double y = a.y + u * dy;
      double fc = 0.0;
      double clearance = kInf;
      double comfort_clearance = kInf;
      if (!footprintCostLocked(x, y, yaw, fc, &clearance, &comfort_clearance)) {
        return false;
      }
      if (std::isfinite(clearance) && clearance < hardClearanceFloor()) return false;
    }
    return true;
  }

  bool pathFootprintSafeLocked(const std::vector<Pose2> &path, double step) const
  {
    if (path.size() < 2) return false;
    std::vector<Pose2> tmp = path;
    updateYawFromTangent(tmp);
    updatePathArcLength(tmp);
    const double ds = std::clamp(step, 0.05, std::max(0.06, 0.45 * resolution_));
    for (double s = 0.0; s <= tmp.back().s + 1e-6; s += ds) {
      const Pose2 p = samplePathAt(tmp, std::min(s, tmp.back().s));
      double fc = 0.0;
      double clearance = kInf;
      double comfort_clearance = kInf;
      if (!footprintCostLocked(p.x, p.y, p.yaw, fc, &clearance, &comfort_clearance)) {
        return false;
      }
      if (std::isfinite(clearance) && clearance < hardClearanceFloor()) {
        return false;
      }
    }
    return true;
  }

  void shortcutPathLocked(std::vector<Pose2> &path) const
  {
    if (path.size() < 5) return;

    std::vector<Pose2> src = path;
    updateYawFromTangent(src);
    updatePathArcLength(src);

    std::vector<Pose2> out;
    out.reserve(src.size());
    size_t i = 0;
    out.push_back(src.front());
    while (i + 1 < src.size()) {
      size_t best = i + 1;
      const size_t max_j = std::min(src.size() - 1, i + static_cast<size_t>(10));
      for (size_t j = max_j; j > i + 1; --j) {
        const double len = std::hypot(src[j].x - src[i].x, src[j].y - src[i].y);
        if (len > 2.8) continue;
        const double straight_yaw = std::atan2(src[j].y - src[i].y, src[j].x - src[i].x);
        if (std::abs(wrapAngle(straight_yaw - src[i].yaw)) > 0.85) continue;
        if (std::abs(wrapAngle(src[j].yaw - straight_yaw)) > 0.85) continue;
        if (straightSegmentSafeLocked(src[i], src[j], 0.14)) {
          best = j;
          break;
        }
      }
      if (best <= i) break;
      out.push_back(src[best]);
      i = best;
    }

    if (out.size() >= 2 && pathFootprintSafeLocked(out, 0.14)) {
      path = std::move(out);
      updateYawFromTangent(path);
      updatePathArcLength(path);
    }
  }

  Pose2 samplePathAt(const std::vector<Pose2> &path, double s) const
  {
    if (path.empty()) return {};
    if (s <= 0.0) return path.front();
    if (s >= path.back().s) return path.back();
    auto it = std::lower_bound(
      path.begin(), path.end(), s,
      [](const Pose2 &p, double value) { return p.s < value; });
    if (it == path.begin()) return *it;
    const auto &b = *it;
    const auto &a = *(it - 1);
    const double u = std::clamp((s - a.s) / std::max(1e-6, b.s - a.s), 0.0, 1.0);
    Pose2 out;
    out.x = a.x + u * (b.x - a.x);
    out.y = a.y + u * (b.y - a.y);
    out.yaw = wrapAngle(a.yaw + u * wrapAngle(b.yaw - a.yaw));
    out.s = s;
    out.direction = a.direction;
    return out;
  }

  size_t nearestPathIndex(const std::vector<Pose2> &path, double x, double y, double *dist_out = nullptr) const
  {
    size_t best = 0;
    double best_d2 = kInf;
    for (size_t i = 0; i < path.size(); ++i) {
      const double dx = path[i].x - x;
      const double dy = path[i].y - y;
      const double d2 = dx * dx + dy * dy;
      if (d2 < best_d2) {
        best_d2 = d2;
        best = i;
      }
    }
    if (dist_out) *dist_out = std::sqrt(best_d2);
    return best;
  }

  bool needReplan(const OdomState &odom)
  {
    const double now_sec = this->now().seconds();
    if (!haveGoalThreadSafe()) return false;
    if (!map_ready_) return false;
    // The planner owns the map lock while searching. Keep the control loop
    // independent of that search and let it validate the new path next tick.
    if (planner_running_.load(std::memory_order_acquire)) return false;
    const double last_plan = last_plan_time_sec_.load(std::memory_order_acquire);
    const bool recently_planned = last_plan >= 0.0 && (now_sec - last_plan) < replan_min_interval_;

    std::vector<Pose2> path;
    Pose2 goal;
    bool have_path = false;
    {
      std::lock_guard<std::mutex> lock(path_mutex_);
      have_path = have_path_;
      path = path_;
      goal = active_goal_;
    }
    if (!have_path || path.size() < 2) {
      if (recently_planned) return false;
      last_replan_reason_ = "no valid path";
      return true;
    }

    double track_dist = 0.0;
    const size_t nearest = nearestPathIndex(path, odom.x, odom.y, &track_dist);
    if (track_dist > replan_track_error_) {
      last_replan_reason_ = "tracking error too large";
      return true;
    }
    const double remaining = std::max(0.0, path.back().s - path[nearest].s);
    if (remaining < (path.front().direction < 0 ? 0.20 : replan_path_remaining_)) {
      if (recently_planned) return false;
      last_replan_reason_ = "path almost consumed";
      return true;
    }

    if (map_dirty_) {
      if (pathAheadUnsafe(path, nearest)) {
        ++consecutive_path_unsafe_;
        if (consecutive_path_unsafe_ >= 2) {
          last_replan_reason_ = "updated terrain map invalidated path ahead";
          return true;
        }
      } else {
        consecutive_path_unsafe_ = 0;
      }
      map_dirty_ = false;
    }

    if (recently_planned) {
      return false;
    }
    if (plan_requested_.load(std::memory_order_acquire)) {
      last_replan_reason_ = "new goal or previous plan failed";
      return true;
    }

    Pose2 raw;
    {
      std::lock_guard<std::mutex> glock(goal_mutex_);
      raw = raw_goal_;
    }
    const double raw_to_active = std::hypot(raw.x - goal.x, raw.y - goal.y);
    const double robot_to_raw = std::hypot(raw.x - odom.x, raw.y - odom.y);
    const double robot_to_active = std::hypot(goal.x - odom.x, goal.y - odom.y);
    if (raw_to_active > replan_goal_shift_ &&
        robot_to_raw > local_goal_distance_ * 0.7 &&
        robot_to_active < rolling_goal_update_distance_) {
      last_replan_reason_ = "rolling local goal should advance";
      return true;
    }

    return false;
  }

  bool pathAheadUnsafe(const std::vector<Pose2> &path, size_t nearest_idx)
  {
    const auto snapshot = std::atomic_load(&control_map_snapshot_);
    if (!snapshot || path.empty()) return false;
    const double start_s = path[nearest_idx].s;
    const double end_s = std::min(path.back().s, start_s + std::max(1.5, replan_path_remaining_ + 1.2));
    int severe_samples = 0;
    int unknown_samples = 0;
    for (double s = start_s; s <= end_s + 1e-6; s += 0.25) {
      const Pose2 p = samplePathAt(path, s);
      double fc = 0.0;
      double clearance = kInf;
      if (!footprintCostLocked(p.x, p.y, p.yaw, fc, &clearance, nullptr, false, snapshot.get())) {
        return true;
      }
      if (std::isfinite(clearance) && clearance < hardClearanceFloor()) {
        return true;
      }
      if (fc >= 0.75 * unknown_cost_) {
        ++unknown_samples;
      }
      if (fc > risk_weight_ * 1.65 ||
          (std::isfinite(clearance) && clearance < std::max(0.25, 0.52 * clearance_distance_))) {
        ++severe_samples;
      }
      if (severe_samples >= 3 || unknown_samples >= 4) return true;
    }
    return false;
  }

  bool shouldDropPathAfterPlanFailure(
    const std::vector<Pose2> &path,
    const OdomState &odom,
    const std::string &reason) const
  {
    if (path.size() < 2) return true;
    double track_dist = 0.0;
    const size_t nearest = nearestPathIndex(path, odom.x, odom.y, &track_dist);
    const double heading_gap = std::abs(wrapAngle(path[nearest].yaw - odom.yaw));
    const double remaining = std::max(0.0, path.back().s - path[nearest].s);

    if (reason == "no valid path") {
      return true;
    }
    if (reason == "updated terrain map invalidated path ahead") return true;
    if (track_dist > std::max(0.85, 1.45 * replan_track_error_)) return true;
    if (heading_gap > 1.35) return true;
    if (remaining < std::max(0.35, 0.25 * replan_path_remaining_)) return true;
    return false;
  }

  bool haveGoalThreadSafe()
  {
    std::lock_guard<std::mutex> lock(goal_mutex_);
    return have_goal_;
  }

  bool rawGoalReachedThreadSafe(const OdomState &odom, double *dist_out = nullptr)
  {
    std::lock_guard<std::mutex> lock(goal_mutex_);
    if (!have_goal_) return false;
    const double dist = std::hypot(raw_goal_.x - odom.x, raw_goal_.y - odom.y);
    if (dist_out) *dist_out = dist;
    return dist <= goal_reached_distance_;
  }

  void clearGoalAndPath()
  {
    {
      std::lock_guard<std::mutex> lock(goal_mutex_);
      have_goal_ = false;
      plan_requested_ = false;
    }
    {
      std::lock_guard<std::mutex> lock(path_mutex_);
      path_.clear();
      have_path_ = false;
    }
    resetNominalControls();

    // Clear the latched goal and the old path immediately.  Otherwise RViz
    // can keep showing a completed command while the controller is already
    // idle, which is especially confusing during repeated goal tests.
    nav_msgs::msg::Path empty_path;
    empty_path.header.stamp = now();
    empty_path.header.frame_id = frame_id_;
    path_pub_->publish(empty_path);
    publishReferenceMarkers({});
    visualization_msgs::msg::MarkerArray clear_goal;
    visualization_msgs::msg::Marker clear;
    clear.action = visualization_msgs::msg::Marker::DELETEALL;
    clear_goal.markers.push_back(clear);
    goal_marker_pub_->publish(clear_goal);
    legacy_goal_marker_pub_->publish(clear_goal);
  }

  void predictOdomForControl(OdomState &odom) const
  {
    if (!odom.valid) return;

    double dt = odom_pose_prediction_time_;
    const double stamp_sec = odom.stamp.seconds();
    const double now_sec = this->now().seconds();
    if (stamp_sec > 0.0 && now_sec > stamp_sec) {
      dt += std::clamp(now_sec - stamp_sec, 0.0, 0.20);
    }
    dt = std::clamp(dt, 0.0, 0.25);
    if (dt <= 1e-4 || !std::isfinite(odom.v) || !std::isfinite(odom.yaw_rate)) return;

    const double mid_yaw = odom.yaw + 0.5 * odom.yaw_rate * dt;
    odom.x += odom.v * std::cos(mid_yaw) * dt;
    odom.y += odom.v * std::sin(mid_yaw) * dt;
    odom.yaw = wrapAngle(odom.yaw + odom.yaw_rate * dt);
  }

  bool stopIfStuck(const OdomState &odom)
  {
    uint64_t goal_seq = 0;
    bool have_goal = false;
    {
      std::lock_guard<std::mutex> lock(goal_mutex_);
      goal_seq = goal_seq_;
      have_goal = have_goal_;
    }
    if (!have_goal || goal_seq != stuck_goal_seq_) {
      stuck_goal_seq_ = goal_seq;
      stuck_since_sec_ = 0.0;
      stuck_latched_ = false;
    }
    if (!have_goal) return false;
    if (stuck_latched_) {
      publishImmediateStop();
      return true;
    }
    if (std::abs(last_cmd_v_) < stuck_command_mps_) {
      stuck_since_sec_ = 0.0;
      return false;
    }
    const double t = now().seconds();
    const double progress = std::hypot(odom.x - stuck_start_x_, odom.y - stuck_start_y_);
    if (stuck_since_sec_ <= 0.0 || progress >= stuck_progress_m_) {
      stuck_since_sec_ = t;
      stuck_start_x_ = odom.x;
      stuck_start_y_ = odom.y;
      return false;
    }
    if (t - stuck_since_sec_ < stuck_timeout_sec_) return false;
    stuck_latched_ = true;
    RCLCPP_ERROR(
      get_logger(),
      "terrain_nav stopped: commanded %.2f m/s but moved only %.2f m in %.1f s; send a new goal after checking the obstruction",
      last_cmd_v_, progress, stuck_timeout_sec_);
    publishImmediateStop();
    return true;
  }

  void controlTimer()
  {
    const auto control_start = std::chrono::steady_clock::now();
    if (live_obstacle_enable_) {
      const auto live = std::atomic_load(&live_snapshot_);
      if (live) {
        const double live_age = now().seconds() - live->stamp_sec;
        if (live_age > live_obstacle_max_age_sec_) {
          requestPlan("stale live cloud");
          RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 1000,
            "terrain_nav stops on stale live cloud: age=%.2fs",
            live_age);
          publishImmediateStop();
          return;
        }
      }
    }
    OdomState odom;
    {
      std::lock_guard<std::mutex> lock(odom_mutex_);
      odom = odom_;
    }
    predictOdomForControl(odom);
    if (!odom.valid) {
      if (haveGoalThreadSafe()) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "terrain_nav received a goal but is waiting for odometry on %s",
          odom_topic_.c_str());
      }
      publishStop();
      return;
    }

    if (haveGoalThreadSafe()) {
      const double now_sec = this->now().seconds();
      const double odom_age = now_sec - odom.stamp.seconds();
      const auto map = std::atomic_load(&control_map_snapshot_);
      const double map_age = map ? now_sec - map->stamp.seconds() : 0.0;
      if ((odom_age > max_odom_age_sec_ && odom_age >= 0.0) ||
          (map_ready_ && map_age > max_map_age_sec_ && map_age >= 0.0)) {
        requestPlan("stale odometry or map");
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "terrain_nav stops on stale input: odom_age=%.2fs map_age=%.2fs", odom_age, map_age);
        publishImmediateStop();
        return;
      }
    }

    double raw_goal_dist = 0.0;
    if (rawGoalReachedThreadSafe(odom, &raw_goal_dist)) {
      clearGoalAndPath();
      RCLCPP_INFO(
        get_logger(),
        "terrain_nav reached raw goal: dist=%.2f <= %.2f, clearing goal and stopping",
        raw_goal_dist, goal_reached_distance_);
      publishStop();
      return;
    }

    if (stopIfStuck(odom)) return;

    if (haveGoalThreadSafe() && !map_ready_) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "terrain_nav received a goal but is waiting for fused terrain map on %s",
        fused_map_topic_.c_str());
    }

    if (live_dirty_.exchange(false) && haveGoalThreadSafe()) {
      std::vector<Pose2> current_path;
      {
        std::lock_guard<std::mutex> lock(path_mutex_);
        if (have_path_) current_path = path_;
      }
      if (current_path.size() >= 2) {
        const size_t nearest = nearestPathIndex(current_path, odom.x, odom.y);
        if (livePathAheadUnsafe(current_path, nearest)) {
          {
            std::lock_guard<std::mutex> lock(path_mutex_);
            path_.clear();
            have_path_ = false;
          }
          requestPlan("new live obstacle");
          RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 800,
            "terrain_nav emergency stop: new live obstacle intersects the path ahead");
          publishImmediateStop();
          return;
        }
      }
    }

    if (needReplan(odom)) {
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 800,
        "terrain_nav replanning: %s",
        last_replan_reason_.c_str());
      const std::string replan_reason = last_replan_reason_;
      requestPlan(replan_reason);
      if (replan_reason == "updated terrain map invalidated path ahead") {
        // A path rejected by the new costmap is no longer safe to track. The
        // worker will replace it without blocking this fast safety cycle.
        publishImmediateStop();
        return;
      }
    }

    std::vector<Pose2> path;
    bool have_path = false;
    {
      std::lock_guard<std::mutex> lock(path_mutex_);
      have_path = have_path_;
      path = path_;
    }
    if (!have_path || path.size() < 2) {
      publishStop();
      return;
    }

    double track_dist = 0.0;
    const size_t nearest = nearestPathIndex(path, odom.x, odom.y, &track_dist);
    if (live_obstacle_enable_ && livePathAheadUnsafe(path, nearest)) {
      {
        std::lock_guard<std::mutex> lock(path_mutex_);
        path_.clear();
        have_path_ = false;
      }
      requestPlan("new live obstacle");
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 800,
        "terrain_nav emergency stop: live obstacle intersects newly planned path");
      publishImmediateStop();
      return;
    }
    // Planning is bounded but still consumes time. Track from the newest pose.
    {
      std::lock_guard<std::mutex> lock(odom_mutex_);
      odom = odom_;
    }
    if (now().seconds() - odom.stamp.seconds() > max_odom_age_sec_) {
      publishImmediateStop();
      return;
    }
    predictOdomForControl(odom);
    geometry_msgs::msg::Twist cmd = use_mppi_ && path.front().direction > 0
      ? computeMppiCommand(odom, path) : computePurePursuitCommand(odom, path);
    const double remaining = std::max(0.0, path.back().s - path[nearest].s);
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 1500,
      "terrain_nav tracking: err=%.2f rem=%.2f cmd_v=%.2f cmd_steer=%.2f odom_v=%.2f path_pts=%zu",
      track_dist, remaining, cmd.linear.x, cmd.angular.z, odom.v, path.size());
    limitAndPublish(cmd);
    const double control_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - control_start).count();
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 3000,
      "terrain_nav control timing: %.2fms period=%.1fms planner_last=%.1fms",
      control_ms, 1000.0 * control_period_, last_plan_elapsed_ms_.load(std::memory_order_acquire));
  }

  geometry_msgs::msg::Twist computePurePursuitCommand(const OdomState &odom, const std::vector<Pose2> &path)
  {
    double near_dist = 0.0;
    const size_t nearest = nearestPathIndex(path, odom.x, odom.y, &near_dist);
    const double remaining = std::max(0.0, path.back().s - path[nearest].s);
    const double lookahead = std::clamp(mppi_lookahead_base_ + std::abs(odom.v) * mppi_lookahead_time_, 0.55, 2.2);
    const Pose2 target = samplePathAt(path, path[nearest].s + lookahead);
    const double dx = target.x - odom.x;
    const double dy = target.y - odom.y;
    const double ld = std::max(0.25, std::hypot(dx, dy));
    const double alpha = wrapAngle(std::atan2(dy, dx) - odom.yaw);
    const double pure_steer = std::atan2(2.0 * wheel_base_ * std::sin(alpha), ld);

    const Pose2 nearest_pose = path[nearest];
    const double ex = odom.x - nearest_pose.x;
    const double ey = odom.y - nearest_pose.y;
    const double signed_lateral = -std::sin(nearest_pose.yaw) * ex + std::cos(nearest_pose.yaw) * ey;
    const double heading_error = wrapAngle(nearest_pose.yaw - odom.yaw);
    const int direction = path.front().direction;
    const double stanley_steer = direction * heading_error +
      std::atan2(-stanley_gain_ * signed_lateral, std::abs(odom.v) + stanley_softening_);
    const double steer =
      pure_pursuit_weight_ * pure_steer + (1.0 - pure_pursuit_weight_) * stanley_steer;

    geometry_msgs::msg::Twist cmd;
    const double terrain_scale = terrainSpeedScale(odom.x, odom.y, odom.yaw);
    const double steer_slowdown = std::clamp(1.0 - 0.22 * std::abs(steer) / max_steer_, 0.70, 1.0);
    const double lateral_slowdown = std::clamp(1.0 - 0.18 * std::max(0.0, std::abs(signed_lateral) - 0.20), 0.78, 1.0);
    cmd.linear.x = desired_speed_ * speedScaleForRemaining(remaining) * terrain_scale *
      steer_slowdown * lateral_slowdown;
    if (remaining > goal_slowdown_distance_) {
      cmd.linear.x = std::max(cmd.linear.x, min_tracking_speed_ * terrain_scale);
    }
    cmd.angular.z = clampAbs(steer, max_steer_);
    if (direction < 0) cmd.linear.x = -std::min(reverse_max_speed_, std::abs(cmd.linear.x));
    return cmd;
  }

  double speedScaleForRemaining(double remaining) const
  {
    if (remaining <= 0.15) return 0.0;
    if (remaining >= goal_slowdown_distance_) return 1.0;
    return std::clamp(remaining / goal_slowdown_distance_, 0.28, 1.0);
  }

  double terrainSpeedScaleLocked(
    double x, double y, double yaw, const ControlMapSnapshot *snapshot = nullptr) const
  {
    if (!terrain_speed_enable_ || !map_ready_) return 1.0;

    double max_risk = 0.0;
    double min_clearance = kInf;
    bool any_known = false;
    const double c = std::cos(yaw);
    const double s = std::sin(yaw);
    for (const auto &off : footprintOffsets()) {
      const double px = x + c * off.x() - s * off.y();
      const double py = y + s * off.x() + c * off.y();
      const auto q = queryCostLocked(px, py, yaw, false, snapshot);
      if (!q.known) continue;
      any_known = true;
      max_risk = std::max(max_risk, q.risk);
      min_clearance = std::min(min_clearance, q.clearance);
    }
    if (!any_known) return std::max(terrain_speed_min_scale_, 0.65);

    const double risk_scale = 1.0 - terrain_risk_slowdown_weight_ * std::clamp(max_risk, 0.0, 1.0);
    double clearance_scale = 1.0;
    if (std::isfinite(min_clearance) && min_clearance < clearance_distance_) {
      const double close = (clearance_distance_ - min_clearance) / std::max(1e-3, clearance_distance_);
      clearance_scale -= terrain_clearance_slowdown_weight_ * std::clamp(close, 0.0, 1.0);
    }
    return std::clamp(risk_scale * clearance_scale, terrain_speed_min_scale_, 1.0);
  }

  double terrainSpeedScale(double x, double y, double yaw) const
  {
    const auto snapshot = std::atomic_load(&control_map_snapshot_);
    if (!snapshot) return terrain_speed_min_scale_;
    return terrainSpeedScaleLocked(x, y, yaw, snapshot.get());
  }

  geometry_msgs::msg::Twist computeMppiCommand(const OdomState &odom, const std::vector<Pose2> &path)
  {
    if (nominal_v_.size() != static_cast<size_t>(mppi_steps_)) resetNominalControls();

    std::normal_distribution<double> speed_noise(0.0, mppi_speed_std_);
    std::normal_distribution<double> steer_noise(0.0, mppi_steer_std_);

    double nearest_dist = 0.0;
    const size_t nearest_idx = nearestPathIndex(path, odom.x, odom.y, &nearest_dist);
    const double current_s = path[nearest_idx].s;
    const double remaining = std::max(0.0, path.back().s - current_s);

    double best_cost = kInf;
    std::vector<double> best_v = nominal_v_;
    std::vector<double> best_steer = nominal_steer_;

    for (int rollout = 0; rollout < mppi_batch_size_; ++rollout) {
      std::vector<double> v_seq(static_cast<size_t>(mppi_steps_));
      std::vector<double> steer_seq(static_cast<size_t>(mppi_steps_));
      for (int k = 0; k < mppi_steps_; ++k) {
        if (rollout == 0) {
          v_seq[static_cast<size_t>(k)] = nominal_v_[static_cast<size_t>(k)];
          steer_seq[static_cast<size_t>(k)] = nominal_steer_[static_cast<size_t>(k)];
        } else if (rollout == 1) {
          v_seq[static_cast<size_t>(k)] = desired_speed_;
          steer_seq[static_cast<size_t>(k)] = 0.0;
        } else {
          const double noise_scale = 1.0 - 0.55 * static_cast<double>(k) / std::max(1, mppi_steps_ - 1);
          v_seq[static_cast<size_t>(k)] = std::clamp(
            nominal_v_[static_cast<size_t>(k)] + noise_scale * speed_noise(rng_), min_speed_, max_speed_);
          steer_seq[static_cast<size_t>(k)] = clampAbs(
            nominal_steer_[static_cast<size_t>(k)] + noise_scale * steer_noise(rng_), max_steer_);
        }
      }

      const double cost = evaluateRollout(odom, path, current_s, v_seq, steer_seq);
      if (cost < best_cost) {
        best_cost = cost;
        best_v.swap(v_seq);
        best_steer.swap(steer_seq);
      }
    }

    nominal_v_ = best_v;
    nominal_steer_ = best_steer;

    geometry_msgs::msg::Twist cmd;
    cmd.linear.x = nominal_v_.front();
    cmd.angular.z = nominal_steer_.front();
    const double terrain_scale = terrainSpeedScale(odom.x, odom.y, odom.yaw);
    cmd.linear.x = std::min(cmd.linear.x, desired_speed_ * terrain_scale);
    if (remaining > goal_slowdown_distance_ && nearest_dist < replan_track_error_) {
      cmd.linear.x = std::max(cmd.linear.x, min_tracking_speed_ * terrain_scale);
    }

    for (int k = 0; k + 1 < mppi_steps_; ++k) {
      nominal_v_[static_cast<size_t>(k)] = nominal_v_[static_cast<size_t>(k + 1)];
      nominal_steer_[static_cast<size_t>(k)] = nominal_steer_[static_cast<size_t>(k + 1)];
    }
    nominal_v_.back() = nominal_v_[nominal_v_.size() - 2];
    nominal_steer_.back() = nominal_steer_[nominal_steer_.size() - 2];

    return cmd;
  }

  double evaluateRollout(
    const OdomState &odom,
    const std::vector<Pose2> &path,
    double current_s,
    const std::vector<double> &v_seq,
    const std::vector<double> &steer_seq)
  {
    const auto snapshot = std::atomic_load(&control_map_snapshot_);
    if (!snapshot) return mppi_collision_cost_;
    double x = odom.x;
    double y = odom.y;
    double yaw = odom.yaw;
    double cost = 0.0;
    double prev_v = last_cmd_v_;
    double prev_steer = last_cmd_steer_;
    double traveled = 0.0;

    for (int k = 0; k < mppi_steps_; ++k) {
      const double v = v_seq[static_cast<size_t>(k)];
      const double steer = steer_seq[static_cast<size_t>(k)];
      x += v * std::cos(yaw) * mppi_dt_;
      y += v * std::sin(yaw) * mppi_dt_;
      yaw = wrapAngle(yaw + v * std::tan(steer) / std::max(1e-3, wheel_base_) * mppi_dt_);
      traveled += std::max(0.0, v) * mppi_dt_;
      const double remaining = std::max(0.0, path.back().s - current_s - traveled);
      const double target_speed = desired_speed_ * speedScaleForRemaining(remaining) *
        terrainSpeedScaleLocked(x, y, yaw, snapshot.get());

      const double target_s = std::min(path.back().s, current_s + traveled + mppi_lookahead_base_ + mppi_lookahead_time_ * v);
      const Pose2 target = samplePathAt(path, target_s);
      const double dx = target.x - x;
      const double dy = target.y - y;
      const double lateral = std::hypot(dx, dy);
      const double yaw_err = std::abs(wrapAngle(target.yaw - yaw));
      cost += mppi_path_weight_ * lateral * lateral;
      cost += mppi_heading_weight_ * yaw_err * yaw_err;

      double fc = 0.0;
      double clearance = 0.0;
      if (!footprintCostLocked(x, y, yaw, fc, &clearance, nullptr, false, snapshot.get())) {
        cost += mppi_collision_cost_;
        break;
      }
      cost += mppi_risk_weight_ * fc;
      if (std::isfinite(clearance) && clearance < clearance_distance_) {
        const double m = (clearance_distance_ - clearance) / std::max(1e-3, clearance_distance_);
        cost += mppi_clearance_weight_ * m * m;
      }

      cost += mppi_control_weight_ * (0.12 * v * v + steer * steer);
      cost += mppi_rate_weight_ * (std::pow(v - prev_v, 2.0) + 0.8 * std::pow(steer - prev_steer, 2.0));
      cost += mppi_speed_tracking_weight_ * std::pow(v - target_speed, 2.0);
      cost -= mppi_progress_weight_ * std::max(0.0, v) * mppi_dt_;
      prev_v = v;
      prev_steer = steer;
    }

    const double goal_dx = path.back().x - x;
    const double goal_dy = path.back().y - y;
    cost += mppi_goal_weight_ * std::hypot(goal_dx, goal_dy);
    return cost;
  }

  void limitAndPublish(geometry_msgs::msg::Twist cmd)
  {
    const double t = now().seconds();
    const double dt = (last_cmd_time_sec_ > 0.0) ? std::clamp(t - last_cmd_time_sec_, 0.005, 0.15) : control_period_;
    cmd.linear.x = std::clamp(cmd.linear.x, reverse_enable_ ? -reverse_max_speed_ : min_speed_, max_speed_);
    cmd.angular.z *= steer_output_sign_;
    cmd.angular.z = clampAbs(cmd.angular.z, max_steer_);
    cmd.linear.x = std::clamp(cmd.linear.x, last_cmd_v_ - max_accel_ * dt, last_cmd_v_ + max_accel_ * dt);
    cmd.angular.z = std::clamp(cmd.angular.z, last_cmd_steer_ - max_steer_rate_ * dt, last_cmd_steer_ + max_steer_rate_ * dt);

    const double alpha = command_filter_alpha_;
    cmd.linear.x = alpha * cmd.linear.x + (1.0 - alpha) * last_cmd_v_;
    cmd.angular.z = alpha * cmd.angular.z + (1.0 - alpha) * last_cmd_steer_;

    cmd_pub_->publish(cmd);
    last_cmd_v_ = cmd.linear.x;
    last_cmd_steer_ = cmd.angular.z;
    last_cmd_time_sec_ = t;
  }

  void publishStop()
  {
    geometry_msgs::msg::Twist cmd;
    limitAndPublish(cmd);
  }

  void publishImmediateStop()
  {
    cmd_pub_->publish(geometry_msgs::msg::Twist{});
    last_cmd_v_ = 0.0;
    last_cmd_steer_ = 0.0;
    last_cmd_time_sec_ = now().seconds();
  }

  void resetNominalControls()
  {
    nominal_v_.assign(static_cast<size_t>(mppi_steps_), desired_speed_);
    nominal_steer_.assign(static_cast<size_t>(mppi_steps_), last_cmd_steer_);
  }

  double visualizationTerrainZLocked(double x, double y, double fallback) const
  {
    // Path publication is called by plan() while map_mutex_ is held.
    if (!map_ready_) return fallback;
    for (const auto & layer : {std::string("bgk_mean"), std::string("elevation")}) {
      if (!map_.exists(layer)) continue;
      try {
        const double z = map_.getValue(
          layer, Eigen::Vector3d(x, y, 0.0), se2_grid::InterpolationMethods::INTER_LINEAR);
        if (std::isfinite(z)) return z;
      } catch (...) {
        // A visualization sample must never interrupt the control loop.
      }
    }
    return fallback;
  }

  void publishPath()
  {
    std::vector<Pose2> path;
    {
      std::lock_guard<std::mutex> lock(path_mutex_);
      path = path_;
    }
    if (path.empty()) {
      nav_msgs::msg::Path empty;
      empty.header.stamp = now();
      empty.header.frame_id = frame_id_;
      path_pub_->publish(empty);
      publishReferenceMarkers(path);
      return;
    }

    nav_msgs::msg::Path msg;
    msg.header.stamp = now();
    msg.header.frame_id = frame_id_;
    msg.poses.reserve(path.size());
    for (const auto &p : path) {
      geometry_msgs::msg::PoseStamped pose;
      pose.header = msg.header;
      pose.pose.position.x = p.x;
      pose.pose.position.y = p.y;
      pose.pose.position.z = visualizationTerrainZLocked(p.x, p.y, 0.0) + 0.12;
      tf2::Quaternion q;
      q.setRPY(0.0, 0.0, p.yaw);
      pose.pose.orientation = tf2::toMsg(q);
      msg.poses.push_back(pose);
    }
    path_pub_->publish(msg);
    publishReferenceMarkers(path);
  }

  void publishReferenceMarkers(const std::vector<Pose2> &path)
  {
    visualization_msgs::msg::MarkerArray arr;
    visualization_msgs::msg::Marker clear;
    clear.action = visualization_msgs::msg::Marker::DELETEALL;
    arr.markers.push_back(clear);

    auto make_line = [&](int id, double width, double r, double g, double b, double a) {
      visualization_msgs::msg::Marker m;
      m.header.frame_id = frame_id_;
      m.header.stamp = now();
      m.ns = "terrain_nav_reference";
      m.id = id;
      m.type = visualization_msgs::msg::Marker::LINE_STRIP;
      m.action = visualization_msgs::msg::Marker::ADD;
      m.pose.orientation.w = 1.0;
      m.scale.x = width;
      m.color.r = static_cast<float>(r);
      m.color.g = static_cast<float>(g);
      m.color.b = static_cast<float>(b);
      m.color.a = static_cast<float>(a);
      for (const auto &p : path) {
        geometry_msgs::msg::Point pt;
        pt.x = p.x;
        pt.y = p.y;
        pt.z = visualizationTerrainZLocked(p.x, p.y, 0.0) + 0.16;
        m.points.push_back(pt);
      }
      return m;
    };

    if (!path.empty()) {
      arr.markers.push_back(make_line(0, 0.14, 0.12, 0.92, 0.98, 0.55));
      arr.markers.push_back(make_line(1, 0.055, 0.95, 0.76, 0.22, 0.95));
    }
    ref_pub_->publish(arr);
    guidance_pub_->publish(arr);
  }

  void publishActiveGoal(const Pose2 &goal, bool map_locked = false)
  {
    double terrain_z = 0.0;
    if (map_locked) {
      terrain_z = visualizationTerrainZLocked(goal.x, goal.y, 0.0);
    } else {
      std::lock_guard<std::mutex> lock(map_mutex_);
      terrain_z = visualizationTerrainZLocked(goal.x, goal.y, 0.0);
    }

    visualization_msgs::msg::MarkerArray arr;
    visualization_msgs::msg::Marker clear;
    clear.action = visualization_msgs::msg::Marker::DELETEALL;
    arr.markers.push_back(clear);

    visualization_msgs::msg::Marker sphere;
    sphere.header.frame_id = frame_id_;
    sphere.header.stamp = now();
    sphere.ns = "terrain_nav_goal";
    sphere.id = 0;
    sphere.type = visualization_msgs::msg::Marker::SPHERE;
    sphere.action = visualization_msgs::msg::Marker::ADD;
    sphere.pose.position.x = goal.x;
    sphere.pose.position.y = goal.y;
    sphere.pose.position.z = terrain_z + 0.16;
    sphere.pose.orientation.w = 1.0;
    sphere.scale.x = 0.28;
    sphere.scale.y = 0.28;
    sphere.scale.z = 0.18;
    sphere.color.r = 1.0f;
    sphere.color.g = 0.72f;
    sphere.color.b = 0.18f;
    sphere.color.a = 0.78f;
    arr.markers.push_back(sphere);

    visualization_msgs::msg::Marker arrow;
    arrow.header = sphere.header;
    arrow.ns = "terrain_nav_goal";
    arrow.id = 1;
    arrow.type = visualization_msgs::msg::Marker::ARROW;
    arrow.action = visualization_msgs::msg::Marker::ADD;
    arrow.pose.position.x = goal.x;
    arrow.pose.position.y = goal.y;
    arrow.pose.position.z = terrain_z + 0.20;
    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, goal.yaw);
    arrow.pose.orientation = tf2::toMsg(q);
    arrow.scale.x = 0.70;
    arrow.scale.y = 0.08;
    arrow.scale.z = 0.08;
    arrow.color.r = 1.0f;
    arrow.color.g = 0.86f;
    arrow.color.b = 0.25f;
    arrow.color.a = 0.90f;
    arr.markers.push_back(arrow);

    goal_marker_pub_->publish(arr);
    legacy_goal_marker_pub_->publish(arr);
  }

  std::string odom_topic_;
  std::string fused_map_topic_;
  std::string cmd_topic_;
  std::string frame_id_;
  bool live_obstacle_enable_ = false;
  std::string live_cloud_topic_ = "/cloud_registered";
  double live_obstacle_min_height_ = 0.38;
  double live_obstacle_neighbor_radius_ = 0.75;
  int live_obstacle_min_hits_ = 3;
  double live_obstacle_max_age_sec_ = 0.60;
  double live_obstacle_hold_sec_ = 1.20;
  double live_obstacle_inflation_radius_ = 0.45;

  double wheel_base_ = 0.549;
  double max_steer_ = 0.5236;
  double max_steer_rate_ = 2.6;
  double max_speed_ = 2.0;
  double min_speed_ = 0.0;
  double desired_speed_ = 1.0;
  double min_tracking_speed_ = 0.35;
  double goal_slowdown_distance_ = 0.90;
  double max_accel_ = 2.0;

  double footprint_front_ = 0.42;
  double footprint_back_ = 0.42;
  double footprint_half_width_ = 0.313;
  double footprint_margin_ = 0.05;

  int yaw_bins_ = 72;
  double primitive_length_ = 0.42;
  double integration_step_ = 0.10;
  double max_search_time_ms_ = 150.0;
  int max_iterations_ = 26000;
  double goal_tolerance_ = 0.42;
  double goal_reached_distance_ = 0.55;
  double goal_yaw_tolerance_ = 0.95;
  double local_goal_distance_ = 8.0;
  double local_goal_bearing_limit_ = 1.20;
  double direct_goal_bearing_limit_ = 1.45;
  bool guide_search_enable_ = true;
  int guide_max_expansions_ = 9000;
  double guide_min_distance_ = 2.0;
  double guide_boundary_margin_ = 0.8;
  double guide_progress_weight_ = 2.2;
  double guide_goal_weight_ = 0.48;
  double guide_lateral_weight_ = 0.22;
  double guide_clearance_weight_ = 0.85;
  double guide_path_cost_weight_ = 0.055;
  double guide_follow_weight_ = 1.15;
  double guide_heading_weight_ = 0.22;
  bool direct_curve_enable_ = true;
  double direct_curve_max_length_ratio_ = 1.55;
  double direct_curve_max_lateral_ratio_ = 0.55;
  double direct_curve_clearance_weight_ = 5.0;

  double risk_free_threshold_ = 0.08;
  double risk_hard_threshold_ = 0.58;
  std::string risk_query_mode_ = "flat_2d";
  std::string se2_risk_layer_ = "risk";
  std::string hard_risk_layer_ = "baseline_risk";
  bool se2_risk_lethal_ = false;
  double unknown_cost_ = 60.0;
  bool allow_unknown_ = true;
  double unknown_robot_bubble_radius_ = 0.0;
  double unknown_robot_bubble_max_baseline_ = 0.0;
  double max_odom_age_sec_ = 0.5;
  double max_map_age_sec_ = 1.0;
  double stuck_timeout_sec_ = 4.0;
  double stuck_progress_m_ = 0.12;
  double stuck_command_mps_ = 0.25;
  double risk_weight_ = 3.0;
  double clearance_distance_ = 0.85;
  double clearance_weight_ = 2.8;
  double hard_clearance_floor_ = -1.0;
  double comfort_clearance_distance_ = 1.35;
  double comfort_clearance_weight_ = 2.2;
  double comfort_clearance_risk_threshold_ = 0.18;
  double steer_weight_ = 0.22;
  double steer_change_weight_ = 0.20;
  double steer_reversal_weight_ = 0.0;
  int smoothing_passes_ = 3;
  bool shortcut_smoothing_enable_ = true;
  double reference_resample_spacing_ = 0.22;
  int reference_smooth_iterations_ = 36;
  double reference_smooth_step_ = 0.32;
  double reference_anchor_weight_ = 0.16;
  double reference_smooth_weight_ = 0.82;
  double reference_curvature_weight_ = 0.18;
  double reference_max_curvature_ratio_ = 0.92;
  double heuristic_weight_ = 1.15;
  double progress_weight_ = 1.8;
  double path_lateral_weight_ = 0.45;
  double goal_heading_weight_ = 0.35;
  double backtrack_reject_distance_ = 0.28;
  double max_extra_path_ratio_ = 2.25;
  bool accept_partial_progress_ = true;

  double replan_min_interval_ = 0.25;
  double replan_path_remaining_ = 2.0;
  double replan_track_error_ = 0.75;
  double replan_goal_shift_ = 0.35;
  double rolling_goal_update_distance_ = 2.8;
  double command_timeout_sec_ = 0.50;
  double command_filter_alpha_ = 0.75;
  bool odom_speed_from_pose_ = true;
  double odom_speed_filter_alpha_ = 0.50;
  double odom_pose_prediction_time_ = 0.08;
  double stanley_gain_ = 1.25;
  double stanley_softening_ = 0.65;
  double pure_pursuit_weight_ = 0.45;
  double steer_output_sign_ = 1.0;

  bool use_mppi_ = true;
  double control_period_ = 0.04;
  double mppi_dt_ = 0.10;
  int mppi_steps_ = 24;
  int mppi_batch_size_ = 320;
  double mppi_speed_std_ = 0.32;
  double mppi_steer_std_ = 0.30;
  double mppi_path_weight_ = 8.0;
  double mppi_heading_weight_ = 2.0;
  double mppi_goal_weight_ = 1.2;
  double mppi_risk_weight_ = 22.0;
  double mppi_clearance_weight_ = 4.0;
  double mppi_control_weight_ = 0.18;
  double mppi_rate_weight_ = 0.32;
  double mppi_progress_weight_ = 2.5;
  double mppi_speed_tracking_weight_ = 0.80;
  double mppi_collision_cost_ = 100000.0;
  double mppi_lookahead_base_ = 0.75;
  double mppi_lookahead_time_ = 0.75;
  bool terrain_speed_enable_ = true;
  double terrain_speed_min_scale_ = 0.45;
  double terrain_risk_slowdown_weight_ = 0.55;
  double terrain_clearance_slowdown_weight_ = 0.28;
  bool path_commit_enable_ = true;
  double path_commit_distance_ = 1.80;
  double path_commit_min_distance_ = 0.90;
  double path_commit_max_track_error_ = 0.55;

  rclcpp::Subscription<se2_grid_msgs::msg::SE2Grid>::SharedPtr map_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr legacy_goal_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr live_obstacle_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr ref_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr guidance_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr goal_marker_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr legacy_goal_marker_pub_;
  rclcpp::TimerBase::SharedPtr control_timer_;
  rclcpp::CallbackGroup::SharedPtr map_group_;
  rclcpp::CallbackGroup::SharedPtr odom_group_;
  rclcpp::CallbackGroup::SharedPtr live_group_;
  rclcpp::CallbackGroup::SharedPtr control_group_;

  mutable std::mutex map_mutex_;
  se2_grid::SE2Grid map_;
  std::shared_ptr<const ControlMapSnapshot> control_map_snapshot_;
  std::shared_ptr<const LiveObstacleSnapshot> live_snapshot_;
  std::atomic_bool map_ready_{false};
  std::atomic_bool map_dirty_{false};
  std::atomic<uint64_t> map_update_seq_{0};
  std::atomic_int consecutive_path_unsafe_{0};
  std::atomic_bool logged_map_ready_{false};
  rclcpp::Time map_stamp_;
  double min_x_ = 0.0;
  double min_y_ = 0.0;
  double max_x_ = 0.0;
  double max_y_ = 0.0;
  double resolution_ = 0.2;
  int width_ = 0;
  int height_ = 0;
  std::vector<uint8_t> costs_;
  std::vector<uint8_t> hard_;
  std::vector<float> unknown_baseline_;
  std::vector<float> clearance_;
  std::vector<float> comfort_clearance_;
  std::vector<uint8_t> live_hard_;
  std::vector<double> live_expiry_;
  std::vector<uint8_t> live_free_scans_;
  std::atomic_bool live_dirty_{false};
  double live_min_x_ = 0.0;
  double live_min_y_ = 0.0;
  double live_resolution_ = 0.2;
  int live_width_ = 0;
  int live_height_ = 0;

  mutable std::mutex odom_mutex_;
  OdomState odom_;
  bool have_prev_pose_for_speed_ = false;
  double prev_pose_speed_x_ = 0.0;
  double prev_pose_speed_y_ = 0.0;
  double prev_pose_speed_yaw_ = 0.0;
  double prev_pose_speed_stamp_sec_ = 0.0;
  double estimated_odom_v_ = 0.0;
  double estimated_odom_yaw_rate_ = 0.0;
  std::atomic_bool logged_odom_ready_{false};
  std::atomic<double> unknown_robot_x_{0.0};
  std::atomic<double> unknown_robot_y_{0.0};
  std::atomic_bool unknown_robot_pose_ready_{false};

  mutable std::mutex goal_mutex_;
  Pose2 raw_goal_;
  bool have_goal_ = false;
  uint64_t goal_seq_ = 0;

  mutable std::mutex path_mutex_;
  std::vector<Pose2> path_;
  Pose2 active_goal_;
  bool have_path_ = false;
  bool reverse_enable_ = true;
  double reverse_max_distance_ = 4.0;
  double reverse_max_speed_ = 0.55;
  uint64_t path_seq_ = 0;

  std::atomic_bool plan_requested_{false};
  std::atomic_bool planner_running_{false};
  std::atomic<double> last_plan_time_sec_{-1.0};
  std::atomic<double> last_plan_elapsed_ms_{0.0};
  std::string last_replan_reason_ = "startup";
  std::mutex planner_mutex_;
  std::condition_variable planner_cv_;
  bool planner_stop_ = false;
  bool planner_wakeup_ = false;
  std::string planner_reason_ = "startup";
  std::thread planner_thread_;

  std::mt19937 rng_;
  std::vector<double> nominal_v_;
  std::vector<double> nominal_steer_;
  double last_cmd_v_ = 0.0;
  double last_cmd_steer_ = 0.0;
  double last_cmd_time_sec_ = 0.0;
  uint64_t stuck_goal_seq_ = 0;
  double stuck_since_sec_ = 0.0;
  double stuck_start_x_ = 0.0;
  double stuck_start_y_ = 0.0;
  bool stuck_latched_ = false;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 3);
  auto node = std::make_shared<TerrainNavNode>();
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
