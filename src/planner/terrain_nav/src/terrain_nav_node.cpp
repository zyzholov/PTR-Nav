#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <limits>
#include <map>
#include <iomanip>
#include <future>
#include <sstream>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <queue>
#include <random>
#include <string>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#include "terrain_nav/body_kinematics.hpp"
#include "terrain_nav/reverse_arc.hpp"
#include "terrain_nav/chance_budget.hpp"
#include "terrain_nav/probability_support.hpp"
#include "terrain_nav/motion_guard.hpp"
#include "terrain_nav/motion_resolution.hpp"
#include "terrain_nav/motion_prediction.hpp"
#include "terrain_nav/dynamic_occupancy.hpp"
#include "terrain_nav/joint_motion_risk.hpp"
#include "terrain_nav/mobility_recovery.hpp"
#include "terrain_nav/route_memory.hpp"
#include "terrain_nav/planning_budget.hpp"

using terrain_nav::PlanningBudget;
#include "terrain_nav/dynamic_yield.hpp"
#include "terrain_nav/temporal_geometry.hpp"
#include "terrain_nav/measured_columns.hpp"

#include <Eigen/Core>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
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

Pose2 advanceArc(Pose2 pose,double curvature,double distance,double offset=0.)
{
  return terrain_nav::advanceBodyArc(pose,curvature,distance,offset);
}

struct OdomState
{
  double x = 0.0;
  double y = 0.0;
  double yaw = 0.0;
  double v = 0.0;
  double yaw_rate = 0.0;
  rclcpp::Time stamp;
  bool valid = false;
  // Preserve the source state/clock separately from the control-time estimate.
  // Repeated tracking/safety queries must not integrate the same age twice.
  std::array<double,5> prediction_source{{NAN,NAN,NAN,NAN,NAN}};
  double prediction_sec=0.;
  bool command_prediction=false;
};

struct CostQuery
{
  bool known = false;
  bool lethal = false;
  double risk = 0.0;
  double clearance = kInf;
  double comfort_clearance = kInf;
  double cost = 0.0;
  double probability_upper = std::numeric_limits<double>::quiet_NaN();
  double height_tail_upper = std::numeric_limits<double>::quiet_NaN();
  int height_radius = 0, height_row = 0, height_col = 0, height_yaw = 0;
  uint64_t probability_event = 0;
};

// Readers retain a complete generation without holding the search mutex.
struct ControlMapSnapshot
{
  se2_grid::SE2Grid grid;
  rclcpp::Time stamp;
  double min_x = 0.0, min_y = 0.0, max_x = 0.0, max_y = 0.0, resolution = 0.2;
  int width = 0, height = 0;
  std::vector<uint8_t> costs, hard;
  bool historical_advisory_present = false;
  std::vector<float> unknown_baseline, clearance, comfort_clearance;
  std::vector<float> historical_clearance;
  std::vector<double> observed_ground_until;
  // Source-matched labels of measured moving bodies, not cleared map cells.
  std::map<uint64_t,std::map<size_t,uint16_t>> moving_subcells;
};

struct LiveObstacleSnapshot
{
  struct MapExclusions {const se2_grid::SE2Grid *grid=nullptr;std::map<size_t,uint16_t> bits;};
  std::shared_ptr<const MapExclusions> map_exclusions;
  std::map<uint64_t,std::vector<size_t>> moving_cells;
  std::map<uint64_t,std::vector<size_t>> held_moving_cells;
  size_t owned_history_exclusions=0;
  double min_x = 0.0, min_y = 0.0, resolution = 0.2, stamp_sec = 0.0;
  double inflation_radius = std::numeric_limits<double>::quiet_NaN();
  int width = 0, height = 0;
  std::vector<uint8_t> hard;
  std::vector<uint8_t> present_hard;
  std::vector<uint8_t> current_measured;
  std::vector<uint8_t> uninflated_present;
  std::vector<uint8_t> routing_clear;
  std::vector<uint8_t> forecast;
  std::vector<terrain_nav::ColumnRectangle> rectangles;
  std::vector<terrain_nav::MotionTrackSnapshot> tracks;
  std::vector<terrain_nav::MotionTrackSnapshot> provisional_tracks;
  std::vector<terrain_nav::MotionTrackSnapshot> resolved_tracks;
  std::vector<std::array<double,4>> compact_observed;

  std::array<double,4> rectangle(int x,int y) const
  {
    const double ox=min_x+x*resolution,oy=min_y+y*resolution;
    const std::array<double,4> whole{ox,oy,ox+resolution,oy+resolution};
    const size_t i=y*width+x,count=size_t(width)*height;
    if(rectangles.size()!=count || present_hard.size()!=count || !present_hard[i] ||
       forecast.size()!=count || forecast[i] || inflation_radius!=0. || !rectangles[i].valid())return whole;
    const auto &r=rectangles[i];
    const std::array<double,4> b{std::max(whole[0],double(r.x0)),std::max(whole[1],double(r.y0)),
      std::min(whole[2],double(r.x1)),std::min(whole[3],double(r.y1))};
    return b[0]<b[2]&&b[1]<b[3]?b:whole;
  }

  bool occupied(double x, double y) const
  {
    const int ix = static_cast<int>(std::floor((x - min_x) / resolution));
    const int iy = static_cast<int>(std::floor((y - min_y) / resolution));
    if(ix<0||ix>=width||iy<0||iy>=height||!hard[size_t(iy)*width+ix])return false;
    const auto b=rectangle(ix,iy);
    return x>=b[0]-1e-9&&x<=b[2]+1e-9&&y>=b[1]-1e-9&&y<=b[3]+1e-9;
  }
};

// One decision, one immutable input generation. Cache deterministic geometry
// queries only: probability ledgers and independent dynamic sampling are
// evaluated anew for EVERY policy, in their original order.
struct YieldFootprintCache
{
  struct Entry { bool safe=false;double cost=0.,clearance=kInf;CostQuery query; };
  const void *owner=nullptr;
  const ControlMapSnapshot *map=nullptr;
  const LiveObstacleSnapshot *live=nullptr;
  std::map<std::tuple<double,double,double,bool>,Entry> entries;
  std::shared_ptr<const LiveObstacleSnapshot> current_view,future_view;
  size_t hits=0,misses=0;
  void bind(const void *node,const ControlMapSnapshot &m,const LiveObstacleSnapshot &l) {
    if(owner!=node||map!=&m||live!=&l) {
      entries.clear();current_view.reset();future_view.reset();owner=node;map=&m;live=&l;
    }
  }
};

bool bodyOverlapsCell(double x,double y,double yaw,double front,double back,double width,
                      double cx,double cy,double resolution)
{
  const double c=std::cos(yaw),s=std::sin(yaw),half=.5*resolution;
  const double length=.5*(front+back),offset=.5*(front-back);
  const double dx=cx-x-offset*c,dy=cy-y-offset*s;
  const double projected=half*(std::abs(c)+std::abs(s));
  return std::abs(c*dx+s*dy)<=length+projected &&
    std::abs(-s*dx+c*dy)<=width+projected &&
    std::abs(dx)<=length*std::abs(c)+width*std::abs(s)+half &&
    std::abs(dy)<=length*std::abs(s)+width*std::abs(c)+half;
}

bool bodyOverlapsRectangle(double x,double y,double yaw,double front,double back,double width,
                           const std::array<double,4> &bounds)
{
  const double c=std::cos(yaw),s=std::sin(yaw),hx=(bounds[2]-bounds[0])*.5,hy=(bounds[3]-bounds[1])*.5;
  const double length=(front+back)*.5,offset=(front-back)*.5;
  const double dx=(bounds[0]+bounds[2])*.5-x-offset*c,dy=(bounds[1]+bounds[3])*.5-y-offset*s;
  return std::abs(c*dx+s*dy)<=length+hx*std::abs(c)+hy*std::abs(s)+1e-9 &&
    std::abs(-s*dx+c*dy)<=width+hx*std::abs(s)+hy*std::abs(c)+1e-9 &&
    std::abs(dx)<=length*std::abs(c)+width*std::abs(s)+hx+1e-9 &&
    std::abs(dy)<=length*std::abs(s)+width*std::abs(c)+hy+1e-9;
}
}  // namespace

class TerrainNavNode final : public rclcpp::Node
{
  struct RecoveryProgressOrigin;
public:
  explicit TerrainNavNode(const rclcpp::NodeOptions &options = rclcpp::NodeOptions(),
                          bool start_runtime = true)
  : Node("terrain_nav_node", options), rng_(7)
  {
    declareParameters();
    loadParameters();

    auto sensor_qos = rclcpp::SensorDataQoS();
    rclcpp::QoS live_cloud_qos=sensor_qos;
    if(live_cloud_reliable_)live_cloud_qos=rclcpp::QoS(1).reliable();
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
        live_cloud_topic_, live_cloud_qos,
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
    chance_status_pub_ = create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
      "/terrain_nav/chance_status",rclcpp::QoS(1).reliable());
    received_goal_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
      "/terrain_nav/received_goal",rclcpp::QoS(1).reliable().transient_local());
    path_pub_ = create_publisher<nav_msgs::msg::Path>("/terrain_nav/path", rclcpp::QoS(rclcpp::KeepLast(1)).reliable());
    topology_memory_pub_=create_publisher<nav_msgs::msg::OccupancyGrid>(
      "/terrain_nav/topology_memory",rclcpp::QoS(1).reliable());
    topology_route_pub_=create_publisher<nav_msgs::msg::Path>(
      "/terrain_nav/topology_route",rclcpp::QoS(1).reliable());
    live_obstacle_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>(
      "/terrain_nav/live_obstacles", rclcpp::QoS(rclcpp::KeepLast(1)).reliable());
    live_raw_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>(
      "/terrain_nav/live_obstacles_raw", rclcpp::QoS(2).reliable());
    motion_observations_pub_=create_publisher<nav_msgs::msg::OccupancyGrid>(
      "/terrain_nav/motion_observations",rclcpp::QoS(2).reliable());
    ref_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("/reference_path", marker_qos);
    guidance_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("/visualization/terrain_guidance_path", marker_qos);
    goal_marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("/terrain_nav/active_goal", marker_qos);
    local_goal_marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("/terrain_nav/local_goal", marker_qos);
    legacy_goal_marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("/visualization/goal_marker", marker_qos);

    nominal_v_.assign(mppi_steps_, desired_speed_);
    nominal_steer_.assign(mppi_steps_, 0.0);

    if (start_runtime) {
      background_guide_runtime_=true;
      control_timer_ = create_wall_timer(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(control_period_)),
        std::bind(&TerrainNavNode::controlTimer, this), control_group_);
      planner_thread_ = std::thread(&TerrainNavNode::plannerLoop, this);
    }

    RCLCPP_INFO(
      get_logger(),
      "terrain_nav ready: odom=%s fused_map=%s goal=/goal_pose cmd=%s wheel_base=%.3f max_steer=%.3f",
      odom_topic_.c_str(), fused_map_topic_.c_str(), cmd_topic_.c_str(), wheel_base_, max_steer_);
    RCLCPP_INFO(get_logger(),
      "terrain_nav risk contract: mode=%s state_limit=%.3f local_path_budget=%.3f braking_budget=%.3f GP_support=%s",
      chance_constrained_ ? "probability_upper_bound_v1_v6" : "legacy_heuristic",
      chance_state_limit_,chance_path_budget_,chance_braking_budget_,
      chance_allow_gp_completion_ ? "explicitly_allowed" : "disabled");
    RCLCPP_INFO(get_logger(),"terrain_nav tracking_controller=%s (sampling MPC uses CPU; reverse uses hybrid)",
      tracking_controller_.c_str());
    const auto envelope=collisionEnvelope(footprint_margin_);
    RCLCPP_INFO(get_logger(),
      "terrain_nav collision envelope: front=%.3f back=%.3f protected_width=%.3f margin=%.3f "
      "live_extra_inflation=%.3f hard_grid_reserve=%.3f vehicle_height=%.3f headroom=%.3f",
      envelope.front,envelope.back,2*envelope.half_width,footprint_margin_,
      live_obstacle_inflation_radius_,hardClearanceFloor(),vehicle_envelope_height_m_,overhead_clearance_margin_m_);
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
    if(background_guide_.valid())background_guide_.wait();
  }

private:
#ifdef TERRAIN_NAV_BENCHMARK
  friend struct TerrainNavBenchmark;
  // Deterministic callback interleaving in native tests; absent in production.
  std::function<void()> dynamic_before_commit_test_hook_;
  std::function<void()> dynamic_before_selection_proof_test_hook_;
#endif
  void declareParameters()
  {
    declare_parameter("odom_topic", "/Odometry");
    declare_parameter("fused_map_topic", "/car_0_terrain_analyzer/fused_map");
    declare_parameter("cmd_topic", "/hunter_se/cmd_vel");
    declare_parameter("frame_id", "map");
    declare_parameter("live_obstacle_enable", false);
    declare_parameter("live_cloud_reliable",false);
    declare_parameter("live_obstacle_resolution", 0.0);
    declare_parameter("live_cloud_topic", "/cloud_registered");
    declare_parameter("live_obstacle_min_height", 0.38);
    declare_parameter("live_obstacle_neighbor_radius", 0.75);
    declare_parameter("live_obstacle_min_hits", 3);
    declare_parameter("live_obstacle_max_age_sec", 0.60);
    declare_parameter("live_obstacle_hold_sec", 1.20);
    declare_parameter("live_obstacle_inflation_radius", 0.45);
    declare_parameter("live_motion_guard_enable",true);
    declare_parameter("live_motion_raw_surface_fit",false);
    declare_parameter("live_motion_component_link_cells",1);
    declare_parameter("live_motion_provisional_velocity_reserve_mps",1.0);
    declare_parameter("live_motion_min_resolution",0.0);
    declare_parameter("live_obstacle_ground_clear_enabled",false);
    declare_parameter("live_motion_horizon_sec",1.5);
    declare_parameter("dynamic_yield_actuator_delay_sec",.35);
    declare_parameter("dynamic_yield_speed_dead_time_sec",-1.);
    declare_parameter("dynamic_yield_steering_dead_time_sec",-1.);
    declare_parameter("dynamic_time_indexed_routing_enable",false);
    declare_parameter("dynamic_observed_only",false);
    declare_parameter("dynamic_yield_min_braking_decel",1.2);
    declare_parameter("dynamic_acceleration_escape_enable",false);
    declare_parameter("live_motion_acceleration_bound_mps2",0.);
    declare_parameter("live_motion_center_candidate_actions",false);
    declare_parameter("live_motion_identity_memory_sec",3.);
    declare_parameter("live_motion_turning_identity_reacquisition",false);
    declare_parameter("live_motion_prediction_maximum_speed_mps",4.5);
    declare_parameter("live_motion_prediction_model",std::string("reachable"));
    declare_parameter("live_motion_joint_samples",8192);
    declare_parameter("live_motion_joint_risk_limit",.02);
    declare_parameter("dynamic_risk_preference_weight",0.);
    declare_parameter("dynamic_anticipation_horizon_sec",3.);
    declare_parameter("live_motion_tracking_max_height_m",0.);
    declare_parameter("live_motion_gaussian_observation_std_m",.03);
    declare_parameter("live_motion_gaussian_process_density",.04);
    declare_parameter("live_motion_gaussian_constant_acceleration",false);
    declare_parameter("live_motion_gaussian_jerk_density",.04);
    declare_parameter("live_motion_gaussian_maximum_observations",3);
    declare_parameter("live_motion_gaussian_acceleration_envelope",false);
    declare_parameter("live_motion_track_known_slow_centers",false);
    declare_parameter("live_motion_sim_observation_calibration",false);
    declare_parameter("live_motion_sim_observation_std_m",.018);
    declare_parameter("unclassified_encounter_speed_mps",0.);
    declare_parameter("unclassified_encounter_model_release",false);
    declare_parameter("observation_speed_limit",.35);

    declare_parameter("wheel_base", 0.549);
    declare_parameter("rear_axle_offset", 0.0);
    declare_parameter("topology_unknown_reverse_weight",0.0);
    declare_parameter("topology_forward_seed_enable",false);
    declare_parameter("topology_async_guide_enable",false);
    declare_parameter("max_steer", 0.5236);
    declare_parameter("max_steer_rate", 2.6);
    declare_parameter("max_speed", 2.0);
    declare_parameter("min_speed", 0.0);
    declare_parameter("reverse_enable", true);
    declare_parameter("reverse_recovery_enable", false);
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
    declare_parameter("body_collision_front",0.0);
    declare_parameter("body_collision_back",0.0);
    declare_parameter("body_collision_half_width",0.0);
    declare_parameter("vehicle_envelope_height_m",.65);
    declare_parameter("overhead_clearance_margin_m",.10);

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
    declare_parameter("topology_guide_enable",false);
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
    declare_parameter("chance_constrained", false);
    declare_parameter("chance_state_limit", 0.05);
    declare_parameter("chance_allow_startup_prior", false);
    declare_parameter("chance_path_budget", 0.10);
    declare_parameter("chance_braking_budget", 0.05);
    declare_parameter("chance_allow_gp_completion", false);
    declare_parameter("chance_allow_joint_model",false);
    declare_parameter("chance_allow_independent_model_floor",false);
    declare_parameter("chance_allow_bounded_pending",false);
    declare_parameter("chance_max_pitch_rad", 0.55);
    declare_parameter("chance_max_roll_rad", 0.45);
    declare_parameter("chance_max_step_m", 0.18);
    declare_parameter("vehicle_ground_clearance_m",.135);
    declare_parameter("underbody_clearance_margin_m",.01);
    declare_parameter("se2_risk_layer", "risk");
    declare_parameter("hard_risk_layer", "baseline_risk");
    declare_parameter("se2_risk_lethal", false);
    declare_parameter("se2_hard_threshold", 0.999);
    declare_parameter("unknown_cost", 60.0);
    declare_parameter("allow_unknown", true);
    declare_parameter("unknown_robot_bubble_radius", 0.0);
    declare_parameter("unknown_robot_bubble_max_baseline", 0.0);
    declare_parameter("known_ground_memory_sec", 3.0);
    declare_parameter("max_odom_age_sec", 0.5);
    declare_parameter("max_map_age_sec", 1.0);
    declare_parameter("stuck_timeout_sec", 4.0);
    declare_parameter("stuck_progress_m", 0.12);
    declare_parameter("stuck_command_mps", 0.25);
    declare_parameter("stuck_replan_attempts",0);
    declare_parameter("risk_weight", 3.0);
    declare_parameter("clearance_distance", 0.85);
    declare_parameter("clearance_weight", 2.8);
    declare_parameter("hard_clearance_floor", -1.0);
    declare_parameter("historical_obstacle_cost_weight", 12.0);
    declare_parameter("historical_obstacle_influence_distance", 1.35);
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
    declare_parameter("reference_tracking_reserve_m",0.);
    declare_parameter("reference_tracking_reserve_cost",20.);
    declare_parameter("heuristic_weight", 1.15);
    declare_parameter("progress_weight", 1.8);
    declare_parameter("path_lateral_weight", 0.45);
    declare_parameter("goal_heading_weight", 0.35);
    declare_parameter("backtrack_reject_distance", 0.28);
    declare_parameter("max_extra_path_ratio", 2.25);
    declare_parameter("accept_partial_progress", true);
    declare_parameter("partial_search_time_ms", 0.0);

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

    declare_parameter("tracking_controller", "auto");
    declare_parameter("use_mppi", false);  // Deprecated alias for sampling MPC, not MPPI.
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
    get_parameter("reverse_recovery_enable", reverse_recovery_enable_);
    get_parameter("reverse_max_distance", reverse_max_distance_);
    get_parameter("reverse_max_speed", reverse_max_speed_);
    reverse_max_speed_ = std::clamp(reverse_max_speed_, 0.1, 0.8);
    reverse_max_distance_ = std::clamp(reverse_max_distance_, 0.5, 6.0);
    get_parameter("odom_topic", odom_topic_);
    get_parameter("fused_map_topic", fused_map_topic_);
    get_parameter("cmd_topic", cmd_topic_);
    get_parameter("frame_id", frame_id_);
    get_parameter("live_obstacle_enable", live_obstacle_enable_);
    get_parameter("live_cloud_reliable",live_cloud_reliable_);
    get_parameter("live_obstacle_resolution",live_obstacle_resolution_);
    get_parameter("live_cloud_topic", live_cloud_topic_);
    get_parameter("live_obstacle_min_height", live_obstacle_min_height_);
    get_parameter("live_obstacle_neighbor_radius", live_obstacle_neighbor_radius_);
    get_parameter("live_obstacle_min_hits", live_obstacle_min_hits_);
    get_parameter("live_obstacle_max_age_sec", live_obstacle_max_age_sec_);
    get_parameter("live_obstacle_hold_sec", live_obstacle_hold_sec_);
    get_parameter("live_obstacle_inflation_radius", live_obstacle_inflation_radius_);
    get_parameter("live_motion_guard_enable",live_motion_guard_enable_);
    get_parameter("dynamic_anticipation_horizon_sec",dynamic_anticipation_horizon_sec_);
    if(!std::isfinite(dynamic_anticipation_horizon_sec_)||dynamic_anticipation_horizon_sec_<3.||dynamic_anticipation_horizon_sec_>6.)
      throw std::invalid_argument("dynamic anticipation must be in [3,6]s");
    get_parameter("live_motion_tracking_max_height_m",motion_tracking_max_height_m_);
    if(!std::isfinite(motion_tracking_max_height_m_)||motion_tracking_max_height_m_<0||motion_tracking_max_height_m_>3.)
      throw std::invalid_argument("motion tracking height must be finite in [0,3]m");
    get_parameter("live_motion_provisional_velocity_reserve_mps",provisional_velocity_reserve_mps_);
    get_parameter("live_motion_min_resolution",live_motion_min_resolution_);
    get_parameter("live_motion_raw_surface_fit",live_motion_raw_surface_fit_);
    get_parameter("live_motion_component_link_cells",motion_guard_.component_link_cells);
    if(motion_guard_.component_link_cells<1||motion_guard_.component_link_cells>3)
      throw std::invalid_argument("motion association link must be 1..3 measured grid cells");
    if(live_motion_raw_surface_fit_&&(live_obstacle_resolution_<=0||live_motion_min_resolution_>live_obstacle_resolution_+1e-8))
      throw std::invalid_argument("raw motion surface fitting requires explicit native fine-grid geometry");
    get_parameter("live_obstacle_ground_clear_enabled",live_obstacle_ground_clear_enabled_);
    get_parameter("live_motion_horizon_sec",live_motion_horizon_sec_);
    get_parameter("dynamic_yield_actuator_delay_sec",dynamic_yield_delay_);
    get_parameter("dynamic_yield_speed_dead_time_sec",dynamic_speed_dead_time_);
    get_parameter("dynamic_yield_steering_dead_time_sec",dynamic_steering_dead_time_);
    get_parameter("dynamic_time_indexed_routing_enable",dynamic_time_indexed_routing_);
    get_parameter("dynamic_observed_only",dynamic_observed_only_);
    get_parameter("dynamic_yield_min_braking_decel",dynamic_yield_deceleration_);
    get_parameter("dynamic_acceleration_escape_enable",dynamic_acceleration_escape_enable_);
    get_parameter("live_motion_acceleration_bound_mps2",motion_guard_.acceleration_bound_mps2);
    get_parameter("live_motion_center_candidate_actions",center_candidate_actions_);
    get_parameter("live_motion_identity_memory_sec",motion_guard_.identity_memory_seconds);
    get_parameter("live_motion_turning_identity_reacquisition",motion_guard_.turning_identity_reacquisition);
    if(!std::isfinite(motion_guard_.identity_memory_seconds)||motion_guard_.identity_memory_seconds<.6||motion_guard_.identity_memory_seconds>6.)
      throw std::invalid_argument("motion identity memory must be finite within .6..6 seconds (not a prediction lifetime)");
    get_parameter("live_motion_prediction_maximum_speed_mps",motion_guard_.prediction_maximum_speed_mps);
    get_parameter("live_motion_prediction_model",motion_prediction_model_);
    get_parameter("live_motion_joint_samples",motion_joint_samples_);
    get_parameter("live_motion_joint_risk_limit",motion_joint_risk_limit_);
    get_parameter("live_motion_gaussian_observation_std_m",motion_guard_.gaussian_observation_std);
    get_parameter("live_motion_gaussian_process_density",motion_guard_.gaussian_process_density);
    get_parameter("live_motion_gaussian_constant_acceleration",motion_guard_.gaussian_constant_acceleration);
    get_parameter("live_motion_gaussian_jerk_density",motion_guard_.gaussian_jerk_density);
    get_parameter("live_motion_gaussian_maximum_observations",motion_guard_.gaussian_maximum_observations);
    get_parameter("live_motion_gaussian_acceleration_envelope",motion_guard_.gaussian_acceleration_envelope);
    if(motion_guard_.gaussian_constant_acceleration &&
       (motion_prediction_model_=="reachable"||motion_guard_.gaussian_acceleration_envelope))
      throw std::invalid_argument("CA posterior requires Gaussian joint risk and is separate from the bounded-mean CV stress model");
    if(!std::isfinite(motion_guard_.gaussian_jerk_density)||motion_guard_.gaussian_jerk_density<0||motion_guard_.gaussian_jerk_density>4.)
      throw std::invalid_argument("white jerk density must be within [0,4] m2/s5");
    get_parameter("live_motion_track_known_slow_centers",motion_guard_.track_known_slow_centers);
    get_parameter("live_motion_sim_observation_calibration",motion_guard_.sim_observation_calibration);
    get_parameter("live_motion_sim_observation_std_m",motion_guard_.sim_observation_std);
    if(motion_guard_.sim_observation_calibration &&
       (!get_parameter("use_sim_time").as_bool() || !std::isfinite(motion_guard_.sim_observation_std) ||
        motion_guard_.sim_observation_std<.05/std::sqrt(12.) || motion_guard_.sim_observation_std>motion_guard_.gaussian_observation_std))
      throw std::invalid_argument("simulation repeatability calibration requires sim time and a valid 5cm-domain noise assumption");
    if(motion_prediction_model_!="reachable" && motion_prediction_model_!="gaussian_joint_cuda" &&
       motion_prediction_model_!="gaussian_joint_cpu")throw std::invalid_argument("invalid motion prediction model");
    motion_guard_.fit_gaussian_motion=motion_prediction_model_!="reachable";
    if(motion_joint_samples_<1024 || motion_joint_samples_>65536 || !std::isfinite(motion_joint_risk_limit_) ||
       motion_joint_risk_limit_<=0 || motion_joint_risk_limit_>=1 ||
       (motion_prediction_model_!="reachable" && motion_joint_risk_limit_>get_parameter("chance_state_limit").as_double()) ||
       !std::isfinite(motion_guard_.gaussian_observation_std) || motion_guard_.gaussian_observation_std<=0 ||
       !std::isfinite(motion_guard_.gaussian_process_density) || motion_guard_.gaussian_process_density<0 ||
       motion_guard_.gaussian_maximum_observations<3 || motion_guard_.gaussian_maximum_observations>4)
      throw std::invalid_argument("invalid Gaussian model, samples or dynamic risk sub-budget (cannot exceed the combined state limit)");
    if(motion_prediction_model_=="gaussian_joint_cuda") {
      std::string error;if(!terrain_nav::jointMotionGpuAvailable(&error))throw std::runtime_error(error);
    }
    if(motion_guard_.fit_gaussian_motion)RCLCPP_WARN(get_logger(),
      "Conditional Gaussian motion qualification model enabled: %s; NOT calibrated real-human probabilities or MPPI",
      motion_prediction_model_.c_str());
    if(!std::isfinite(motion_guard_.acceleration_bound_mps2) || motion_guard_.acceleration_bound_mps2<0 ||
       motion_guard_.acceleration_bound_mps2>4.5)
      throw std::invalid_argument("motion acceleration bound must be finite within 0..4.5m/s2");
    if(!std::isfinite(motion_guard_.prediction_maximum_speed_mps) ||
       motion_guard_.prediction_maximum_speed_mps<=0 || motion_guard_.prediction_maximum_speed_mps>4.5)
      throw std::invalid_argument("prediction maximum speed must be within (0,4.5]; faster observed objects restore 4.5");
    get_parameter("unclassified_encounter_speed_mps",unclassified_encounter_speed_);
    get_parameter("unclassified_encounter_model_release",unclassified_encounter_model_release_);
    get_parameter("dynamic_risk_preference_weight",dynamic_risk_preference_weight_);
    if(!std::isfinite(dynamic_risk_preference_weight_)||dynamic_risk_preference_weight_<0||dynamic_risk_preference_weight_>10.)
      throw std::invalid_argument("dynamic risk preference weight must be finite within 0..10");
    get_parameter("observation_speed_limit",observation_speed_limit_);
    if(!std::isfinite(observation_speed_limit_) || observation_speed_limit_<=0)
      throw std::invalid_argument("observation_speed_limit must be finite and positive");
    observation_speed_limit_=std::clamp(observation_speed_limit_,.05,.35);
    if(!std::isfinite(dynamic_yield_delay_) || dynamic_yield_delay_<0 || dynamic_yield_delay_>1 ||
       !terrain_nav::validYieldDeadTimes(configuredYieldModel()) ||
       !std::isfinite(dynamic_yield_deceleration_) || dynamic_yield_deceleration_<=0 || dynamic_yield_deceleration_>3.6)
      throw std::invalid_argument("invalid dynamic yield plant bounds");

    get_parameter("wheel_base", wheel_base_);
    get_parameter("rear_axle_offset", rear_axle_offset_);
    get_parameter("topology_unknown_reverse_weight",topology_unknown_reverse_weight_);
    get_parameter("topology_forward_seed_enable",topology_forward_seed_enable_);
    get_parameter("topology_async_guide_enable",topology_async_guide_enable_);
    get_parameter("max_steer", max_steer_);
    get_parameter("max_steer_rate", max_steer_rate_);
    get_parameter("max_speed", max_speed_);
    if(!std::isfinite(unclassified_encounter_speed_) || unclassified_encounter_speed_<0 || unclassified_encounter_speed_>max_speed_)
      throw std::invalid_argument("invalid unclassified encounter speed");
    get_parameter("min_speed", min_speed_);
    get_parameter("desired_speed", desired_speed_);
    get_parameter("min_tracking_speed", min_tracking_speed_);
    get_parameter("goal_slowdown_distance", goal_slowdown_distance_);
    get_parameter("max_accel", max_accel_);

    get_parameter("footprint_front", footprint_front_);
    get_parameter("footprint_back", footprint_back_);
    get_parameter("footprint_half_width", footprint_half_width_);
    get_parameter("footprint_margin", footprint_margin_);
    get_parameter("body_collision_front",body_collision_front_);
    get_parameter("body_collision_back",body_collision_back_);
    get_parameter("body_collision_half_width",body_collision_half_width_);
    if(!std::isfinite(body_collision_front_) || !std::isfinite(body_collision_back_) ||
       !std::isfinite(body_collision_half_width_) ||
       std::min({body_collision_front_,body_collision_back_,body_collision_half_width_})<0)
      throw std::invalid_argument("invalid body collision bounds");
    get_parameter("vehicle_envelope_height_m",vehicle_envelope_height_m_);
    get_parameter("overhead_clearance_margin_m",overhead_clearance_margin_m_);
    if(!std::isfinite(vehicle_envelope_height_m_) || vehicle_envelope_height_m_<=0 ||
       !std::isfinite(overhead_clearance_margin_m_) || overhead_clearance_margin_m_<.05 ||
       !std::isfinite(footprint_front_) || !std::isfinite(footprint_back_) ||
       !std::isfinite(footprint_half_width_) || !std::isfinite(footprint_margin_) ||
       std::min({footprint_front_,footprint_back_,footprint_half_width_})<=0 || footprint_margin_<0)
      throw std::invalid_argument("invalid vehicle collision envelope or margin");

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
    get_parameter("topology_guide_enable",topology_guide_enable_);
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
    get_parameter("chance_constrained", chance_constrained_);
    if(dynamic_time_indexed_routing_ && (!chance_constrained_ || !live_obstacle_enable_ || !live_motion_guard_enable_))
      throw std::invalid_argument("time-indexed routing requires chance constraints, live occupancy and motion estimation");
    get_parameter("chance_state_limit", chance_state_limit_);
    get_parameter("chance_allow_startup_prior", chance_allow_startup_prior_);
    get_parameter("chance_path_budget", chance_path_budget_);
    get_parameter("chance_braking_budget", chance_braking_budget_);
    get_parameter("chance_allow_gp_completion", chance_allow_gp_completion_);
    get_parameter("chance_allow_joint_model",chance_allow_joint_model_);
    get_parameter("chance_allow_independent_model_floor",chance_allow_independent_model_floor_);
    get_parameter("chance_allow_bounded_pending",chance_allow_bounded_pending_);
    get_parameter("chance_max_pitch_rad", chance_max_pitch_rad_);
    get_parameter("chance_max_roll_rad", chance_max_roll_rad_);
    get_parameter("chance_max_step_m", chance_max_step_m_);
    get_parameter("vehicle_ground_clearance_m",vehicle_ground_clearance_m_);
    get_parameter("underbody_clearance_margin_m",underbody_clearance_margin_m_);
    get_parameter("se2_risk_layer", se2_risk_layer_);
    get_parameter("hard_risk_layer", hard_risk_layer_);
    get_parameter("se2_risk_lethal", se2_risk_lethal_);
    get_parameter("se2_hard_threshold", se2_hard_threshold_);
    get_parameter("unknown_cost", unknown_cost_);
    get_parameter("allow_unknown", allow_unknown_);
    get_parameter("unknown_robot_bubble_radius", unknown_robot_bubble_radius_);
    get_parameter("unknown_robot_bubble_max_baseline", unknown_robot_bubble_max_baseline_);
    get_parameter("known_ground_memory_sec", known_ground_memory_sec_);
    get_parameter("max_odom_age_sec", max_odom_age_sec_);
    get_parameter("max_map_age_sec", max_map_age_sec_);
    get_parameter("stuck_timeout_sec", stuck_timeout_sec_);
    get_parameter("stuck_progress_m", stuck_progress_m_);
    get_parameter("stuck_command_mps", stuck_command_mps_);
    get_parameter("stuck_replan_attempts",stuck_replan_attempts_);
    if(stuck_replan_attempts_<0||stuck_replan_attempts_>2)
      throw std::invalid_argument("stuck_replan_attempts must be within [0,2]");
    get_parameter("risk_weight", risk_weight_);
    get_parameter("clearance_distance", clearance_distance_);
    get_parameter("clearance_weight", clearance_weight_);
    get_parameter("hard_clearance_floor", hard_clearance_floor_);
    get_parameter("historical_obstacle_cost_weight", historical_obstacle_cost_weight_);
    get_parameter("historical_obstacle_influence_distance", historical_obstacle_influence_distance_);
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
    get_parameter("reference_tracking_reserve_m",reference_tracking_reserve_m_);
    get_parameter("reference_tracking_reserve_cost",reference_tracking_reserve_cost_);
    get_parameter("heuristic_weight", heuristic_weight_);
    get_parameter("progress_weight", progress_weight_);
    get_parameter("path_lateral_weight", path_lateral_weight_);
    get_parameter("goal_heading_weight", goal_heading_weight_);
    get_parameter("backtrack_reject_distance", backtrack_reject_distance_);
    get_parameter("max_extra_path_ratio", max_extra_path_ratio_);
    get_parameter("accept_partial_progress", accept_partial_progress_);
    get_parameter("partial_search_time_ms",partial_search_time_ms_);

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
    get_parameter("tracking_controller", tracking_controller_);
    if (tracking_controller_ == "auto") tracking_controller_ = use_mppi_ ? "sampling_mpc" : "hybrid";
    if (tracking_controller_ != "hybrid" && tracking_controller_ != "sampling_mpc")
      throw std::invalid_argument("tracking_controller must be auto, hybrid or sampling_mpc; MPPI is not implemented");
    if (use_mppi_) RCLCPP_WARN(get_logger(),
      "Deprecated use_mppi: random-shooting MPC is NOT path-integral MPPI; use tracking_controller");
    use_mppi_ = tracking_controller_ == "sampling_mpc";
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
    if(!std::isfinite(provisional_velocity_reserve_mps_) || provisional_velocity_reserve_mps_<0 ||
        provisional_velocity_reserve_mps_>4.5 ||
        !std::isfinite(live_obstacle_resolution_) || live_obstacle_resolution_<0 ||
        (live_obstacle_resolution_>0 && live_obstacle_resolution_<.025) ||
        !std::isfinite(live_motion_min_resolution_) || live_motion_min_resolution_<0 || live_motion_min_resolution_>1. ||
        !std::isfinite(partial_search_time_ms_) || partial_search_time_ms_<0)
      throw std::invalid_argument("invalid live resolution or partial search time");
    live_obstacle_min_hits_ = std::clamp(live_obstacle_min_hits_, 1, 20);
    live_obstacle_max_age_sec_ = std::clamp(live_obstacle_max_age_sec_, 0.15, 2.0);
    live_obstacle_hold_sec_ = std::clamp(live_obstacle_hold_sec_, 0.2, 3.0);
    live_obstacle_inflation_radius_ = std::clamp(live_obstacle_inflation_radius_, 0.0, 1.0);
    live_motion_horizon_sec_=std::clamp(live_motion_horizon_sec_,.2,3.);
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
    if (!(chance_state_limit_ > 0.0 && chance_state_limit_ < 1.0 &&
          chance_path_budget_ >= chance_state_limit_ && chance_path_budget_ < 1.0 &&
          chance_braking_budget_ > 0.0 && chance_braking_budget_ <= chance_path_budget_))
      throw std::invalid_argument("invalid chance state/path/braking budgets");
    if (!(std::isfinite(chance_max_pitch_rad_) && chance_max_pitch_rad_>0 && chance_max_pitch_rad_<M_PI/2 &&
          std::isfinite(chance_max_roll_rad_) && chance_max_roll_rad_>0 && chance_max_roll_rad_<M_PI/2 &&
          std::isfinite(chance_max_step_m_) && chance_max_step_m_>0))
      throw std::invalid_argument("invalid probability geometry limits");
    if(!std::isfinite(vehicle_ground_clearance_m_) || !std::isfinite(underbody_clearance_margin_m_) ||
       underbody_clearance_margin_m_<0 || vehicle_ground_clearance_m_<=underbody_clearance_margin_m_)
      throw std::invalid_argument("invalid underbody clearance model");
    if(!std::isfinite(rear_axle_offset_) || rear_axle_offset_<0 || rear_axle_offset_>wheel_base_)
      throw std::invalid_argument("rear_axle_offset must lie between base_link and front axle");
    if(!std::isfinite(topology_unknown_reverse_weight_) || topology_unknown_reverse_weight_<0. || topology_unknown_reverse_weight_>10.)
      throw std::invalid_argument("invalid finite unknown reverse routing weight");
    unknown_robot_bubble_radius_ = std::clamp(unknown_robot_bubble_radius_, 0.0, 1.25);
    unknown_robot_bubble_max_baseline_ = std::clamp(unknown_robot_bubble_max_baseline_, 0.0, risk_hard_threshold_);
    known_ground_memory_sec_=std::clamp(known_ground_memory_sec_,0.0,5.0);
    max_odom_age_sec_ = std::max(0.2, max_odom_age_sec_);
    max_map_age_sec_ = std::max(0.3, max_map_age_sec_);
    stuck_timeout_sec_ = std::clamp(stuck_timeout_sec_, 1.0, 15.0);
    stuck_progress_m_ = std::clamp(stuck_progress_m_, 0.03, 0.5);
    stuck_command_mps_ = std::clamp(stuck_command_mps_, 0.05, max_speed_);
    hard_clearance_floor_ = std::clamp(hard_clearance_floor_, -1.0, 1.5);
    historical_obstacle_cost_weight_=std::clamp(historical_obstacle_cost_weight_,0.,100.);
    historical_obstacle_influence_distance_=std::clamp(historical_obstacle_influence_distance_,.5,3.);
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
    if(!std::isfinite(reference_tracking_reserve_m_) || reference_tracking_reserve_m_<0. || reference_tracking_reserve_m_>.15 ||
       !std::isfinite(reference_tracking_reserve_cost_) || reference_tracking_reserve_cost_<0. || reference_tracking_reserve_cost_>100.)
      throw std::invalid_argument("invalid reference tracking reserve preference");
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
    const auto ingress_start=std::chrono::steady_clock::now();
    if (msg->info.header.frame_id != frame_id_) {
      map_ready_=false;
      input_epoch_.fetch_add(1,std::memory_order_release);
      spatial_epoch_.fetch_add(1,std::memory_order_release);
      std::atomic_store(&control_map_snapshot_,std::shared_ptr<const ControlMapSnapshot>{});
      std::atomic_store(&pending_search_map_,std::shared_ptr<const ControlMapSnapshot>{});
      RCLCPP_ERROR_THROTTLE(get_logger(),*get_clock(),2000,
        "Terrain frame mismatch: received %s, expected %s; navigation blocked",
        msg->info.header.frame_id.c_str(),frame_id_.c_str());
      return;
    }
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

    if(heightSliceMode(grid)<0) {
      std::atomic_store(&control_map_snapshot_,std::shared_ptr<const ControlMapSnapshot>{});
      std::atomic_store(&pending_search_map_,std::shared_ptr<const ControlMapSnapshot>{});
      map_ready_=false;
      input_epoch_.fetch_add(1,std::memory_order_release);
      spatial_epoch_.fetch_add(1,std::memory_order_release);
      RCLCPP_ERROR_THROTTLE(get_logger(),*get_clock(),2000,
        "Collision height contract missing/incompatible with vehicle/payload; navigation blocked");
      return;
    }
    if (chance_constrained_ && !probabilityContractValid(grid)) {
      // Replace the old snapshot as well: a fresh but incompatible map must
      // not leave the previous generation usable by the control thread.
      std::atomic_store(&control_map_snapshot_, std::shared_ptr<const ControlMapSnapshot>{});
      std::atomic_store(&pending_search_map_,std::shared_ptr<const ControlMapSnapshot>{});
      map_ready_ = false;
      input_epoch_.fetch_add(1,std::memory_order_release);
      spatial_epoch_.fetch_add(1,std::memory_order_release);
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000,
        "Probability contract missing/incompatible; navigation blocked (no heuristic fallback)");
      return;
    }
    auto snapshot = std::make_shared<ControlMapSnapshot>();
    snapshot->grid = std::move(grid);
    snapshot->stamp = msg->info.header.stamp;
    buildCostmap(*snapshot);
    if(dynamic_time_indexed_routing_ && motion_prediction_model_!="reachable") {
      std::shared_ptr<const LiveObstacleSnapshot> source;
      {std::lock_guard<std::mutex> lock(motion_sources_mutex_);
        for(const auto &entry:motion_sources_)
          if(entry.epoch==spatial_epoch_.load()&&std::abs(entry.live->stamp_sec-snapshot->stamp.seconds())<=5e-9)
            source=entry.live;
      }
      if(source)labelMovingSubcells(*snapshot,*source);
    }
    if(!chance_constrained_) {
      std::lock_guard<std::mutex> lock(map_mutex_);
      installSearchMapLocked(*snapshot);
      updateGroundEvidenceLocked(snapshot->stamp.seconds());
      snapshot->observed_ground_until=observed_ground_until_;
    }
    std::atomic_store(&control_map_snapshot_, std::shared_ptr<const ControlMapSnapshot>(snapshot));
    if(chance_constrained_) {
      // Ingress owns this complete generation. Never queue it behind a search;
      // the worker adopts the newest pending generation at its next entry.
      std::atomic_store(&pending_search_map_,std::shared_ptr<const ControlMapSnapshot>(snapshot));
      std::unique_lock<std::mutex> lock(map_mutex_,std::try_to_lock);
      if(lock.owns_lock())refreshSearchMapLocked();
      else deferred_search_maps_.fetch_add(1,std::memory_order_relaxed);
    }
    map_ready_ = true;
    map_dirty_ = true;
    map_update_seq_.fetch_add(1, std::memory_order_release);
    map_ingress_ms_.store(std::chrono::duration<double,std::milli>(
      std::chrono::steady_clock::now()-ingress_start).count());
    if (!logged_map_ready_.exchange(true)) {
      RCLCPP_INFO(
        get_logger(),
        "terrain_nav fused_map ready: cells=%dx%d map_yaw_slices=%d planner_yaw_bins=%d "
        "res=%.3f range=[%.2f %.2f]x[%.2f %.2f]",
        snapshot->width, snapshot->height, snapshot->grid.getSizeYaw(), yaw_bins_, snapshot->resolution,
        snapshot->min_x, snapshot->max_x, snapshot->min_y, snapshot->max_y);
    }
  }

  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    const auto processing_start=std::chrono::steady_clock::now();
    if (msg->header.frame_id != frame_id_ || msg->point_step == 0) return;
    const auto has_float_field = [&](const char *name) {
      return std::any_of(msg->fields.begin(), msg->fields.end(), [&](const auto &field) {
        return field.name == name && field.datatype == sensor_msgs::msg::PointField::FLOAT32;
      });
    };
    if (!has_float_field("x") || !has_float_field("y") || !has_float_field("z")) return;

    const auto geometry = std::atomic_load(&control_map_snapshot_);
    if (!geometry || geometry->width <= 0 || geometry->height <= 0) return;
    const int refinement=live_obstacle_resolution_>0 ? std::clamp(
      static_cast<int>(std::ceil(geometry->resolution/live_obstacle_resolution_-1e-6)),1,4) : 1;
    const double min_x=geometry->min_x,min_y=geometry->min_y,resolution=geometry->resolution/refinement;
    const int width=geometry->width*refinement,height=geometry->height*refinement;

    const size_t cells = static_cast<size_t>(width * height);
    const auto cell_for = [&](float x, float y) {
      const int ix = static_cast<int>(std::floor((x - min_x) / resolution));
      const int iy = static_cast<int>(std::floor((y - min_y) / resolution));
      return ix >= 0 && ix < width && iy >= 0 && iy < height ? iy * width + ix : -1;
    };
    std::vector<std::array<float,3>> points;
    points.reserve(static_cast<size_t>(msg->width)*msg->height);
    try {
      sensor_msgs::PointCloud2ConstIterator<float> x(*msg, "x"), y(*msg, "y"), z(*msg, "z");
      for (; x != x.end(); ++x, ++y, ++z) {
        if (!std::isfinite(*x) || !std::isfinite(*y) || !std::isfinite(*z)) continue;
        const int cell = cell_for(*x, *y);
        if (cell >= 0) points.push_back({*x,*y,*z});
      }
    } catch (const std::exception &e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "live obstacle cloud rejected: %s", e.what());
      return;
    }

    std::vector<float> measured,ceiling;
    const auto &grid=geometry->grid;
    const std::string source_layer=grid.exists("terrain_observed")?"terrain_observed":"ground_observed";
    const std::string height_layer=grid.exists("gp_elevation_observed")?"gp_elevation_observed":"elevation";
    const double ground_age=now().seconds()-geometry->stamp.seconds();
    if(grid.exists(source_layer) && grid.exists(height_layer) && ground_age>=-.05 && ground_age<=max_map_age_sec_) {
      measured.assign(cells,std::numeric_limits<float>::quiet_NaN());
      const double scan_delta=rclcpp::Time(msg->header.stamp).seconds()-geometry->stamp.seconds();
      if(heightSliceMode(grid)==1 && ground_age<=.30 && scan_delta>=-.05 && scan_delta<=.30)
        ceiling.assign(cells,std::numeric_limits<float>::infinity());
      // All fine cells within a parent cell use the same nearest-neighbour
      // terrain value. Query once, then replicate; collision returns stay fine.
      for(int iy=0;iy<geometry->height;++iy)for(int ix=0;ix<geometry->width;++ix) {
        const Eigen::Vector3d position(min_x+(ix+.5)*geometry->resolution,
                                      min_y+(iy+.5)*geometry->resolution,0.);
        if(!grid.isInMap(position))continue;
        const double observed=grid.getValue(source_layer,position,se2_grid::InterpolationMethods::INTER_NEAREST);
        const double z=grid.getValue(height_layer,position,se2_grid::InterpolationMethods::INTER_NEAREST);
        const float floor=std::isfinite(observed) && observed>=.5 && std::isfinite(z) ? z :
          std::numeric_limits<float>::quiet_NaN();
        const float upper=ceiling.empty()?std::numeric_limits<float>::infinity():grid.getValue(
          "collision_ceiling_elevation",position,se2_grid::InterpolationMethods::INTER_NEAREST);
        for(int dy=0;dy<refinement;++dy)for(int dx=0;dx<refinement;++dx) {
          const int i=(iy*refinement+dy)*width+ix*refinement+dx;
          measured[i]=floor;if(!ceiling.empty())ceiling[i]=upper;
        }
      }
    }
    const auto columns=terrain_nav::measuredColumns(points,width,height,min_x,min_y,resolution,
      live_obstacle_neighbor_radius_,chance_constrained_ ? std::min(live_obstacle_min_height_,chance_max_step_m_+1e-6) : live_obstacle_min_height_,
      live_obstacle_min_hits_,measured,ceiling);
    const auto &min_z=columns.minimum,&ground_z=columns.ground;
    const auto &tall_hits=columns.hits;
    const double stamp_sec = rclcpp::Time(msg->header.stamp).seconds();
    const double received_sec = now().seconds();
    if (stamp_sec > 0.0 && received_sec - stamp_sec > live_obstacle_max_age_sec_) return;
    std::vector<uint8_t> hard(cells, 0);
    const auto &raw=columns.occupied;
    // Upper returns of an ALREADY identified body can remain visible after
    // its lower surface leaves the tilted sensor's view. Keep these actual
    // measurements for association only; body collision/terrain/raw clearing
    // remain height-filtered and unchanged. Labels provide search regions,
    // never a fresh prediction or permission without new strict witnesses.
    const auto tracking_raw=motion_tracking_max_height_m_>0 ? terrain_nav::motionWitnessMask(
      raw,points,columns.ground,width,height,min_x,min_y,resolution,
      motion_guard_.trackingWitnessRegions(stamp_sec),motion_tracking_max_height_m_,live_obstacle_min_hits_) : raw;
    motion_upper_witness_cells_.store(std::count(tracking_raw.begin(),tracking_raw.end(),uint8_t(1))-
      std::count(raw.begin(),raw.end(),uint8_t(1)));
    const auto current_rectangles=terrain_nav::columnWitnessRectangles(points,raw,width,height,min_x,min_y,resolution);
    std::vector<uint8_t> forecast(cells,0);
    if(live_motion_guard_enable_) {
      // Collision remains fine. Aggregate complete squares only for velocity
      // association so scan-grid jitter cannot become a moving wall.
      const int factor=motionAggregationFactor(width,height,resolution);
      if(factor!=1&&motion_guard_.component_link_cells!=1) {
        RCLCPP_WARN_THROTTLE(get_logger(),*get_clock(),2000,
          "motion association link requires native collision grid geometry; withholding snapshot");
        return;
      }
      std::vector<std::array<double,2>> surface;
      if(live_motion_raw_surface_fit_) {
        if(factor!=1) {
          RCLCPP_WARN_THROTTLE(get_logger(),*get_clock(),2000,"raw motion surface geometry unavailable; withholding new live snapshot");
          return;
        }
        surface=terrain_nav::measuredMotionSurfacePoints(points,columns,tracking_raw,width,height,
          min_x,min_y,resolution,chance_constrained_?std::min(live_obstacle_min_height_,chance_max_step_m_+1e-6):live_obstacle_min_height_,
          motion_tracking_max_height_m_,ceiling);
      }
      if(factor==1)forecast=motion_guard_.update(tracking_raw,width,height,min_x,min_y,resolution,
          stamp_sec>0 ? stamp_sec : received_sec,live_motion_horizon_sec_,{},live_motion_raw_surface_fit_?&surface:nullptr);
      else {
        const auto grouped=terrain_nav::aggregateMotionComponents(tracking_raw,width,height,factor);
        const int cw=grouped.width,ch=grouped.height;
        const auto projected=motion_guard_.update(grouped.occupied,cw,ch,min_x,min_y,resolution*factor,
          stamp_sec>0 ? stamp_sec : received_sec,live_motion_horizon_sec_,grouped.components);
        for(int y=0;y<height;++y)for(int x=0;x<width;++x)
          forecast[y*width+x]=projected[(y/factor)*cw+x/factor];
      }
    }
    const auto action_tracks=motion_guard_.actionHypotheses(center_candidate_actions_&&dynamic_time_indexed_routing_);
    const auto moving_cells=terrain_nav::measuredMovingCells(action_tracks,raw,width,height,min_x,min_y,
      resolution,stamp_sec>0?stamp_sec:received_sec);
    std::vector<uint64_t> raw_owner(cells,0);
    for(const auto &[id,owned]:moving_cells)for(size_t cell:owned)raw_owner[cell]=id;
    {
      std::vector<double> expiry(cells, 0.0);
      std::vector<uint8_t> free_scans(cells, 0);
      std::vector<uint8_t> routing_free_scans(cells,0);
      // Preserve the lowest actual colliding return represented by a held
      // obstacle. A new clearance slice may disprove only an all-overhead
      // reserve, never a pillar/low bar or height-unknown occupancy.
      std::vector<float> occupied_minimum(cells,-std::numeric_limits<float>::infinity());
      if (live_expiry_.size() == cells && min_x == live_min_x_ && min_y == live_min_y_ &&
          resolution == live_resolution_ && width == live_width_ && height == live_height_) {
        expiry = live_expiry_;
        if (live_free_scans_.size() == cells) free_scans = live_free_scans_;
        if(live_routing_free_scans_.size()==cells)routing_free_scans=live_routing_free_scans_;
        if(live_occupied_minimum_.size()==cells)occupied_minimum=live_occupied_minimum_;
      } else {
        for (int iy = 0; iy < live_height_; ++iy) {
          for (int ix = 0; ix < live_width_; ++ix) {
            const size_t old_cell = static_cast<size_t>(iy * live_width_ + ix);
            if (old_cell >= live_expiry_.size()) continue;
            const int cell = cell_for(
              static_cast<float>(live_min_x_ + (ix + 0.5) * live_resolution_),
              static_cast<float>(live_min_y_ + (iy + 0.5) * live_resolution_));
            if (cell >= 0) {
              expiry[static_cast<size_t>(cell)] = live_expiry_[old_cell];
              if (old_cell < live_free_scans_.size())
                free_scans[static_cast<size_t>(cell)] = live_free_scans_[old_cell];
              if(old_cell<live_occupied_minimum_.size())occupied_minimum[cell]=live_occupied_minimum_[old_cell];
              if(old_cell<live_routing_free_scans_.size())routing_free_scans[cell]=live_routing_free_scans_[old_cell];
            }
          }
        }
      }
      for (size_t i = 0; i < cells; ++i) {
        routing_free_scans[i]=terrain_nav::routingClearStreak(routing_free_scans[i],
          live_routing_stamp_,stamp_sec,terrain_nav::routingFloorWitness(columns,measured,i));
        if (raw[i]) {
          const float lower=columns.occupied_minimum[i];
          occupied_minimum[i]=expiry[i]>received_sec ? std::min(occupied_minimum[i],lower) : lower;
          expiry[i] = received_sec + live_obstacle_hold_sec_;
          free_scans[i] = 0;
        } else if(!ceiling.empty() && std::isfinite(ceiling[i]) && std::isfinite(occupied_minimum[i]) &&
                  occupied_minimum[i]>ceiling[i]+.05F) {
          expiry[i]=0.;free_scans[i]=0;
        } else if (live_obstacle_ground_clear_enabled_ && tall_hits[i] == 0 && std::isfinite(min_z[i]) &&
                   std::isfinite(ground_z[i]) && min_z[i] - ground_z[i] < 0.15f) {
          free_scans[i] = std::min<int>(2, free_scans[i] + 1);
          if (free_scans[i] >= 2) expiry[i] = 0.0;
        }
        hard[i] = expiry[i] > received_sec ? 1 : 0;
      }
      // Retain EACH scan's witness rectangle for the same finite hold as its
      // occupancy. Unioning forever while a cell stays occupied would slowly
      // grow a static wall to its whole raster square again.
      if(!live_rectangle_frames_.empty() && live_rectangle_frames_.back().expires>received_sec+live_obstacle_hold_sec_+.05)
        live_rectangle_frames_.clear();
      while(!live_rectangle_frames_.empty() && live_rectangle_frames_.front().expires<=received_sec)
        live_rectangle_frames_.pop_front();
      LiveRectangleFrame witnesses;witnesses.expires=received_sec+live_obstacle_hold_sec_;
      for(size_t i=0;i<cells;++i)if(raw[i])witnesses.cells.push_back({
        min_x+(int(i)%width+.5)*resolution,min_y+(int(i)/width+.5)*resolution,resolution,current_rectangles[i],raw_owner[i]});
      live_rectangle_frames_.push_back(std::move(witnesses));
      std::vector<terrain_nav::ColumnRectangle> rectangles(cells);
      std::vector<uint8_t> missing(cells,0);
      std::vector<uint8_t> owner_seen(cells,0);std::vector<uint64_t> held_owner(cells,0);
      for(const auto &frame:live_rectangle_frames_)for(const auto &entry:frame.cells) {
        const int cell=cell_for(entry.x,entry.y);if(cell<0)continue;
        if(!owner_seen[cell]){held_owner[cell]=entry.owner;owner_seen[cell]=1;}
        else if(held_owner[cell]!=entry.owner)held_owner[cell]=0;
        const int ix=cell%width,iy=cell/width;
        if(!entry.rectangle.valid() || std::abs(entry.resolution-resolution)>1e-8 ||
           std::abs(entry.x-(min_x+(ix+.5)*resolution))>1e-6 ||
           std::abs(entry.y-(min_y+(iy+.5)*resolution))>1e-6) {missing[cell]=1;continue;}
        auto &a=rectangles[cell];const auto &b=entry.rectangle;
        if(!a.valid())a=b;
        else {a.x0=std::min(a.x0,b.x0);a.y0=std::min(a.y0,b.y0);a.x1=std::max(a.x1,b.x1);a.y1=std::max(a.y1,b.y1);}
      }
      for(size_t i=0;i<cells;++i)if(missing[i])rectangles[i]={};
      const auto uninflated_present=hard;
      hard=terrain_nav::inflateColumns(hard,width,height,resolution,live_obstacle_inflation_radius_);
      const auto present_hard=hard;
      for(size_t i=0;i<cells;++i)if(forecast[i])hard[i]=1;
      const bool changed = hard != live_hard_ || min_x != live_min_x_ || min_y != live_min_y_;
      live_hard_ = hard;
      live_expiry_ = std::move(expiry);
      live_free_scans_ = std::move(free_scans);
      live_routing_free_scans_=std::move(routing_free_scans);
      live_routing_stamp_=stamp_sec;
      live_occupied_minimum_=std::move(occupied_minimum);
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
      // Future positions are TIME constraints, not permanently occupied
      // geometry. The matching final-command shield below is mandatory when
      // this option is enabled; current/held measured occupancy is retained.
      snapshot->hard = dynamic_time_indexed_routing_ ? present_hard : hard;
      snapshot->present_hard = present_hard;
      snapshot->current_measured=raw;
      snapshot->uninflated_present = uninflated_present;
      snapshot->rectangles=std::move(rectangles);
      snapshot->forecast=forecast;
      snapshot->routing_clear.resize(cells,0);
      for(size_t i=0;i<cells;++i)snapshot->routing_clear[i]=live_routing_free_scans_[i]>=2 && !uninflated_present[i];
      snapshot->inflation_radius = live_obstacle_inflation_radius_;
      const bool fitted_actions=center_candidate_actions_ && dynamic_time_indexed_routing_;
      snapshot->tracks=dynamic_observed_only_?
        terrain_nav::currentMotionObservations(action_tracks,snapshot->stamp_sec):action_tracks;
      snapshot->moving_cells=moving_cells;
      for(size_t i=0;i<cells;++i)if(held_owner[i]&&uninflated_present[i]&&!missing[i])
        snapshot->held_moving_cells[held_owner[i]].push_back(i);
      snapshot->provisional_tracks=motion_guard_.brakeOnlyHypotheses(fitted_actions);
      motion_resolution_mailbox_.observe(snapshot->stamp_sec,received_sec,spatial_epoch_.load(),resolution,
        action_tracks,motion_guard_.resolvedSnapshot());
      snapshot->resolved_tracks=motion_resolution_mailbox_.snapshot(received_sec);
      snapshot->compact_observed=motion_guard_.compactObservedBounds();
      if(dynamic_time_indexed_routing_ && motion_prediction_model_!="reachable") {
        std::lock_guard<std::mutex> lock(motion_sources_mutex_);
        if(!motion_sources_.empty()&&(motion_sources_.back().epoch!=spatial_epoch_.load()||
            snapshot->stamp_sec<=motion_sources_.back().live->stamp_sec))motion_sources_.clear();
        motion_sources_.push_back({spatial_epoch_.load(),snapshot});
        while(motion_sources_.size()>8)motion_sources_.pop_front();
      }
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
    if(live_raw_pub_->get_subscription_count()>0) {
      auto diagnostic=display;diagnostic.data.resize(cells);
      for(size_t i=0;i<cells;++i)diagnostic.data[i]=raw[i]?100:0;
      live_raw_pub_->publish(diagnostic);
    }
    if(motion_observations_pub_->get_subscription_count()>0) {
      auto diagnostic=display;diagnostic.data.resize(cells);
      for(size_t i=0;i<cells;++i)diagnostic.data[i]=tracking_raw[i]?100:0;
      motion_observations_pub_->publish(diagnostic);
    }
    display.data.assign(cells, -1);
    int obstacle_cells = 0;
    double nearest_obstacle = kInf;
    double nearest_x = 0.0, nearest_y = 0.0;
    OdomState display_odom;
    {std::lock_guard<std::mutex> lock(odom_mutex_);display_odom=odom_;}
    const double robot_x = display_odom.x, robot_y = display_odom.y;
    for (size_t i = 0; i < cells; ++i) {
      if (!hard[i]) continue;
      display.data[i] = forecast[i] && !raw[i] ? 80 : 100;
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
      "live obstacle cloud: %d occupied cells from %ux%u points; nearest=%.2fm at (%.2f, %.2f), moving=%d processing=%.1fms source_age=%.3fs",
      obstacle_cells, msg->width, msg->height, nearest_obstacle, nearest_x, nearest_y,motion_guard_.moving_components,
      std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-processing_start).count(),
      now().seconds()-stamp_sec);
  }

  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    if(localization_fault_.load(std::memory_order_acquire))return;
    const double x = msg->pose.pose.position.x;
    const double y = msg->pose.pose.position.y;
    const double z = msg->pose.pose.position.z;
    const auto &orientation=msg->pose.pose.orientation;
    const double norm=orientation.x*orientation.x+orientation.y*orientation.y+
      orientation.z*orientation.z+orientation.w*orientation.w;
    const auto fault=[&](const char *reason) {
      localization_fault_.store(true,std::memory_order_release);
      input_epoch_.fetch_add(1,std::memory_order_release);
      spatial_epoch_.fetch_add(1,std::memory_order_release);
      {std::lock_guard<std::mutex> lock(odom_mutex_);odom_.valid=false;forward_trail_.clear();}
      // Serialize with command publication so an old control invocation cannot
      // publish a nonzero command after this stop. Do not mutate controller
      // state from the sensor callback.
      {std::lock_guard<std::mutex> lock(command_publish_mutex_);
        cmd_pub_->publish(geometry_msgs::msg::Twist{});
        issued_commands_.record({now().seconds(),0.,0.},std::max(2.,dynamic_yield_delay_+1.));}
      RCLCPP_ERROR(get_logger(),"Localization inconsistent (%s): motion inhibited; restart localization/navigation after correction",reason);
    };
    if(!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) ||
        !std::isfinite(norm) || std::abs(norm-1.)>.01 || msg->header.frame_id!=frame_id_) {
      fault("nonfinite pose, invalid rotation or wrong frame");return;
    }

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
      return; // no source time yet; wait for the clock without inventing it
    }

    bool impossible=false,old_sample=false;
    {
      std::lock_guard<std::mutex> lock(odom_mutex_);
      if(have_prev_pose_for_speed_) {
        const double dt=stamp_sec-prev_pose_speed_stamp_sec_;
        const double distance=std::sqrt(std::pow(x-prev_pose_speed_x_,2)+std::pow(y-prev_pose_speed_y_,2)+
          std::pow(z-prev_pose_speed_z_,2));
        const double turn=std::abs(wrapAngle(yaw-prev_pose_speed_yaw_));
        if(dt<0)old_sample=true;
        else if(dt==0) {old_sample=true;impossible=distance>1e-8 || turn>1e-8;}
        else {
          // Reject BEFORE the speed smoother/clamp. Clamping an impossible
          // pose velocity previously disguised runaway LIO as plausible speed.
          const double horizon=std::min(dt,.5);
          impossible=distance>.03+1.5*max_speed_*horizon ||
            turn>.03+std::max(3.,1.5*max_speed_*maxReferenceCurvature())*horizon;
        }
      }
    }
    if(impossible){fault("pose increment exceeds vehicle motion limits");return;}
    if(old_sample)return;

    {
      std::lock_guard<std::mutex> lock(odom_mutex_);
      if (have_prev_pose_for_speed_ && (stamp_sec < prev_pose_speed_stamp_sec_ ||
          stamp_sec-prev_pose_speed_stamp_sec_ > 1.0 ||
          std::hypot(x-prev_pose_speed_x_,y-prev_pose_speed_y_) > 1.0 ||
          msg->header.frame_id != trail_frame_)) forward_trail_.clear();
      trail_frame_=msg->header.frame_id;
      // Geometry only: a travelled corridor is never treated as free space.
      if (forward_trail_.empty()) forward_trail_.push_back(Pose2{x,y,wrapAngle(yaw),0});
      else {
        const auto &last=forward_trail_.back();
        const double dx=x-last.x,dy=y-last.y;
        if (std::hypot(dx,dy)>=.08 && dx*std::cos(yaw)+dy*std::sin(yaw)>.04) {
          forward_trail_.push_back(Pose2{x,y,wrapAngle(yaw),0});
          while (forward_trail_.size()>256) forward_trail_.pop_front();
        }
      }
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
      odom_.prediction_source.fill(NAN);odom_.prediction_sec=0.;odom_.command_prediction=false;
      current_robot_x_.store(x);current_robot_y_.store(y);
      if (!unknown_robot_pose_ready_.load()) {
        unknown_robot_x_.store(x);
        unknown_robot_y_.store(y);
        unknown_robot_pose_ready_.store(true);
      }

      prev_pose_speed_x_ = x;
      prev_pose_speed_y_ = y;
      prev_pose_speed_z_ = z;
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
        reverse_recovery_count_.store(0);
        reverse_recovery_total_.store(0);
        topology_recovery_search_round_.store(0);
        std::atomic_store(&reverse_progress_origin_,std::shared_ptr<const RecoveryProgressOrigin>{});
        reserve_recovery_count_.store(0);
        std::atomic_store(&recovery_route_memory_,std::shared_ptr<const std::vector<RecoveryRouteMemory>>{});
      }
      accepted_goal = raw_goal_;
      have_goal_ = true;
    }
    if (!new_goal) return;
    auto received=*msg;
    // A goal can be discovered before this participant's first /clock. Its
    // source already has a valid source stamp; do not replace it with zero.
    if(received.header.stamp.sec==0 && received.header.stamp.nanosec==0) {
      const auto time=now();
      if(time.nanoseconds()>0)received.header.stamp=time;
      else {
        std::lock_guard<std::mutex> lock(odom_mutex_);
        received.header.stamp=odom_.stamp;
      }
    }
    received_goal_pub_->publish(received);
    // Clear before waking the worker. Publishing the goal marker can wait for
    // the map lock; clearing afterwards could erase the newly computed path.
    {
      std::lock_guard<std::mutex> lock(path_mutex_);
      path_.clear();
      have_path_ = false;
      recovery_path_ = false;recovery_staged_gears_=false;recovery_legs_.clear();
      recovery_replan_only_ = false;
      recovery_resume_pending_ = false;
      recovery_exit_.clear();
    }
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
    publishActiveGoal(accepted_goal,false,true);
  }

  void rebuildCostmapLocked()
  {
    ControlMapSnapshot snapshot;
    snapshot.grid=map_;snapshot.stamp=map_stamp_;
    snapshot.observed_ground_until=observed_ground_until_;
    buildCostmap(snapshot);
    installSearchMapLocked(snapshot);
  }

  void installSearchMapLocked(const ControlMapSnapshot &snapshot)
  {
    map_=snapshot.grid;map_stamp_=snapshot.stamp;
    min_x_=snapshot.min_x;min_y_=snapshot.min_y;max_x_=snapshot.max_x;max_y_=snapshot.max_y;
    resolution_=snapshot.resolution;width_=snapshot.width;height_=snapshot.height;
    costs_=snapshot.costs;hard_=snapshot.hard;unknown_baseline_=snapshot.unknown_baseline;
    clearance_=snapshot.clearance;comfort_clearance_=snapshot.comfort_clearance;
    historical_clearance_=snapshot.historical_clearance;
    historical_advisory_present_=snapshot.historical_advisory_present;
    observed_ground_until_=snapshot.observed_ground_until;
  }

  void refreshSearchMapLocked()
  {
    const auto newest=std::atomic_exchange(&pending_search_map_,std::shared_ptr<const ControlMapSnapshot>{});
    if(newest)installSearchMapLocked(*newest);
  }

  void buildCostmap(ControlMapSnapshot &snapshot)
  {
    // The same builder serves ingress and the locked search/test wrapper.
    // Bind its mutable arrays to the destination, never to shared search state.
    auto &map_=snapshot.grid;
    auto &width_=snapshot.width;auto &height_=snapshot.height;
    auto &min_x_=snapshot.min_x;auto &max_x_=snapshot.max_x;
    auto &min_y_=snapshot.min_y;auto &max_y_=snapshot.max_y;
    auto &resolution_=snapshot.resolution;
    auto &costs_=snapshot.costs;auto &hard_=snapshot.hard;
    auto &unknown_baseline_=snapshot.unknown_baseline;
    auto &clearance_=snapshot.clearance;auto &comfort_clearance_=snapshot.comfort_clearance;
    if (!map_.exists(chance_constrained_ ? "collision_risk" : se2_risk_layer_)) {
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
          chance_constrained_ ? "collision_risk" :
          (map_.exists(hard_risk_layer_) ? hard_risk_layer_ : se2_risk_layer_);
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
            if (!std::isfinite(valid) || valid < 0.5) known = false;
          } catch (...) {
            known = false;
          }
        }

        if (chance_constrained_ && !known) {
          for (const auto &source : {"startup_prior_used","probability_gp_used"}) {
            if (map_.exists(source)) {
              const double supported=map_.getValue(source,Eigen::Vector3d(wx,wy,0),
                  se2_grid::InterpolationMethods::INTER_NEAREST);
              if (std::isfinite(supported) && supported>=.5) known=true;
            }
          }
          // Optimistic 2D search guidance only. Every full pose and submitted
          // path still needs the floating-point probability/support checks.
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
    computeClearance(snapshot);
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 3000,
      "terrain_nav cost grid: size=%dx%d known=%d hard=%d max_risk=%.2f hard_thr=%.2f clearance=%.2f comfort=%.2f risk_mode=%s",
      width_, height_, known_count, hard_count, max_risk, risk_hard_threshold_,
      clearance_distance_, comfort_clearance_distance_, risk_query_mode_.c_str());
  }

  uint64_t evidenceKey(double x,double y) const
  {
    const auto ix=static_cast<int32_t>(std::floor(x/resolution_));
    const auto iy=static_cast<int32_t>(std::floor(y/resolution_));
    return (static_cast<uint64_t>(static_cast<uint32_t>(ix))<<32)|static_cast<uint32_t>(iy);
  }

  void updateGroundEvidenceLocked(double stamp)
  {
    if (chance_constrained_) return;  // Legacy unknown bubbles are not probabilistic evidence.
    if (stamp<last_ground_evidence_sec_-.1 || std::abs(resolution_-ground_evidence_resolution_)>1e-6)
      ground_evidence_.clear();
    last_ground_evidence_sec_=stamp;ground_evidence_resolution_=resolution_;
    for (auto it=ground_evidence_.begin();it!=ground_evidence_.end();) {
      if (it->second<=stamp) it=ground_evidence_.erase(it);else ++it;
    }
    observed_ground_until_.assign(costs_.size(),0);
    if (known_ground_memory_sec_<=0 || !map_.exists("valid")) return;
    for (int iy=0;iy<height_;++iy) for(int ix=0;ix<width_;++ix) {
      const size_t cell=iy*width_+ix;
      const double x=min_x_+(ix+.5)*resolution_,y=min_y_+(iy+.5)*resolution_;
      const auto key=evidenceKey(x,y);
      if (costs_[cell]>=254 && costs_[cell]!=255) {ground_evidence_.erase(key);continue;}
      if (costs_[cell]<254 &&
          std::hypot(x-current_robot_x_.load(),y-current_robot_y_.load())<3.0) {
        bool supported=true;
        if (map_.exists("gp_observed")) {
          const double measured=map_.getValue("gp_observed",Eigen::Vector3d(x,y,0),
            se2_grid::InterpolationMethods::INTER_NEAREST);
          supported=std::isfinite(measured)&&measured>=.5;
        }
        if (map_.exists(se2_risk_layer_) && map_.hasSO2(se2_risk_layer_)) {
          for (int yaw=0;yaw<map_.getSizeYaw();++yaw) {
            const double risk=map_.getValue(se2_risk_layer_,Eigen::Vector3d(x,y,(yaw+.5)*map_.getResolutionYaw()),
                se2_grid::InterpolationMethods::INTER_NEAREST);
            if (!std::isfinite(risk) || risk>=se2_hard_threshold_) {supported=false;break;}
          }
        }
        if (supported) ground_evidence_[key]=stamp+known_ground_memory_sec_;
      }
      const auto found=ground_evidence_.find(key);
      if (found!=ground_evidence_.end()) {
        // The vehicle still covers this previously measured ground. Do not
        // expire it merely because the body occludes the sensor; new occupied
        // evidence and the live layer always retain precedence.
        if (costs_[cell]==255 && unknown_baseline_[cell]<risk_hard_threshold_ &&
            std::hypot(x-current_robot_x_.load(),y-current_robot_y_.load())<=.65)
          found->second=stamp+known_ground_memory_sec_;
        observed_ground_until_[cell]=found->second;
      }
    }
  }

  void computeClearance(ControlMapSnapshot &snapshot)
  {
    const auto &map_=snapshot.grid;
    const auto width_=snapshot.width,height_=snapshot.height;
    const auto resolution_=snapshot.resolution,min_x_=snapshot.min_x,min_y_=snapshot.min_y;
    const auto &costs_=snapshot.costs;const auto &hard_=snapshot.hard;
    auto &clearance_=snapshot.clearance;auto &comfort_clearance_=snapshot.comfort_clearance;
    auto &historical_clearance_=snapshot.historical_clearance;
    auto &historical_advisory_present_=snapshot.historical_advisory_present;
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
        // Consumers use min(distance,max_distance). Positive edge costs mean
        // expanding beyond that radius cannot change ANY returned value.
        if (cur.d >= max_distance) continue;
        const int cx = cur.idx % width_;
        const int cy = cur.idx / width_;
        for (int k = 0; k < 8; ++k) {
          const int nx = cx + dx[k];
          const int ny = cy + dy[k];
          if (nx < 0 || nx >= width_ || ny < 0 || ny >= height_) continue;
          const int ni = ny * width_ + nx;
          const double step = ((dx[k] == 0 || dy[k] == 0) ? 1.0 : std::sqrt(2.0)) * resolution_;
          const double nd = cur.d + step;
          if (nd > max_distance) continue;
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

    historical_advisory_present_=false;
    historical_clearance_.assign(costs_.size(),static_cast<float>(historical_obstacle_influence_distance_));
    if(map_.exists("historical_obstacle_advisory") && !map_.hasSO2("historical_obstacle_advisory")) {
      std::vector<uint8_t> seeds(costs_.size(),0);
      for(int iy=0;iy<height_;++iy)for(int ix=0;ix<width_;++ix) {
        const double value=map_.getValue("historical_obstacle_advisory",
            Eigen::Vector3d(min_x_+(ix+.5)*resolution_,min_y_+(iy+.5)*resolution_,0.),
            se2_grid::InterpolationMethods::INTER_NEAREST);
        seeds[iy*width_+ix]=std::isfinite(value)&&value>=.5;
      }
      historical_advisory_present_=std::any_of(seeds.begin(),seeds.end(),[](auto value){return value!=0;});
      fill_distance(historical_clearance_,[&](int i){return seeds[i]!=0;},historical_obstacle_influence_distance_);
    }

  }

  bool worldToCostCell(double x, double y, int &ix, int &iy) const
  {
    if (x < min_x_ || x >= max_x_ || y < min_y_ || y >= max_y_) return false;
    ix = static_cast<int>(std::floor((x - min_x_) / resolution_));
    iy = static_cast<int>(std::floor((y - min_y_) / resolution_));
    return ix >= 0 && ix < width_ && iy >= 0 && iy < height_;
  }

  int motionAggregationFactor(int width,int height,double resolution) const
  {
    if(!std::isfinite(live_motion_min_resolution_) || live_motion_min_resolution_<0 ||
       !std::isfinite(resolution) || resolution<=0 || width<=0 || height<=0)
      throw std::invalid_argument("invalid motion aggregation geometry");
    const double ratio=std::clamp(live_motion_min_resolution_/resolution,0.,4.);
    const int requested=std::clamp(static_cast<int>(std::ceil(ratio-1e-6)),1,4);
    for(int factor=requested;factor>1;--factor)if(width%factor==0 && height%factor==0)return factor;
    return 1;
  }

  int heightSliceMode(const se2_grid::SE2Grid &grid) const
  {
    if(!grid.exists("collision_ceiling_elevation"))return 0;
    if(grid.hasSO2("collision_ceiling_elevation"))return -1;
    const auto scalar=[&](const char *name) {
      if(!grid.exists(name) || grid.hasSO2(name))throw std::invalid_argument("height metadata missing");
      const auto &matrix=grid[name][0];
      if(!matrix.allFinite() || matrix.minCoeff()!=matrix.maxCoeff())
        throw std::invalid_argument("height metadata is not a finite scalar");
      return double(matrix(0,0));
    };
    try {
      if(scalar("collision_height_contract")!=1.)return -1;
      const double enabled=scalar("collision_height_filter_enabled");
      if(enabled==0.)return 0;
      if(enabled!=1. || scalar("collision_vehicle_height_m")+1e-6<vehicle_envelope_height_m_ ||
          scalar("collision_headroom_margin_m")+1e-6<overhead_clearance_margin_m_)return -1;
      return 1;
    } catch(...) {return -1;}
  }

  bool probabilityContractValid(const se2_grid::SE2Grid &grid) const
  {
    for (const auto &name : {"failure_probability_upper", "probability_valid"})
      if (!grid.exists(name) || !grid.hasSO2(name)) return false;
    for (const auto &name : {"probability_contract", "probability_front", "probability_back",
                             "probability_half_width", "probability_max_pitch", "probability_max_roll",
                             "probability_max_step", "probability_gp_enabled", "collision_risk"})
      if (!grid.exists(name) || grid.hasSO2(name)) return false;
    const auto position=grid.getPosition();
    const Eigen::Vector3d center(position.x(),position.y(),0);
    const auto scalar = [&](const char *name) {
      return grid.getValue(name, center, se2_grid::InterpolationMethods::INTER_NEAREST);
    };
    try {
      const double version = scalar("probability_contract");
      if(grid.exists("probability_floor_covariance_model") &&
         (grid.hasSO2("probability_floor_covariance_model") ||
          scalar("probability_floor_covariance_model")!=(version>=8. && version<=9.?1.:0.)))return false;
      if((version==8. || version==9.) && (!chance_allow_independent_model_floor_ || !get_parameter("use_sim_time").as_bool() ||
          !grid.exists("probability_floor_covariance_model") || grid.hasSO2("probability_floor_covariance_model") ||
          scalar("probability_floor_covariance_model")!=1.))return false;
      if(version==9.) {
        if(!chance_allow_bounded_pending_ || !get_parameter("use_sim_time").as_bool())return false;
        for(const char *name:{"probability_bounded_pending_model","probability_bounded_pending_sources",
              "bounded_pending_used","bounded_mean_radius_m","bounded_mean_lower","bounded_mean_upper"})
          if(!grid.exists(name) || grid.hasSO2(name))return false;
        const double sources=scalar("probability_bounded_pending_sources");
        if(scalar("probability_bounded_pending_model")!=1. || !std::isfinite(sources) ||
            sources<1. || sources>128. || sources!=std::floor(sources))return false;
      }
      const double front = scalar("probability_front"), back = scalar("probability_back");
      const double half_width = scalar("probability_half_width");
      const double pitch = scalar("probability_max_pitch"), roll = scalar("probability_max_roll");
      const double step = scalar("probability_max_step"), gp = scalar("probability_gp_enabled");
      const double footprint_geometry=grid.exists("probability_footprint_geometry")?
        scalar("probability_footprint_geometry"):1.;
      if(version==7. || version==8. || version==9.) {
        for(const char *name:{"probability_max_roughness","probability_ground_clearance","probability_underbody_margin"})
          if(!grid.exists(name) || grid.hasSO2(name) || !std::isfinite(scalar(name)))return false;
        const double rough=scalar("probability_max_roughness"),clear=scalar("probability_ground_clearance"),margin=scalar("probability_underbody_margin");
        if(rough<=0 || clear>vehicle_ground_clearance_m_+1e-6 || margin<underbody_clearance_margin_m_-1e-6 ||
           clear<=margin || rough>clear-margin+1e-6)return false;
      }
      if((footprint_geometry!=1. && footprint_geometry!=2.) ||
         (grid.exists("probability_footprint_geometry") && grid.hasSO2("probability_footprint_geometry")))return false;
      if(grid.exists("collision_subcell_contract") && scalar("collision_subcell_contract")>=2.) {
        if(scalar("collision_subcell_contract")==3. && (!grid.exists("terrain_failure_probability_upper") ||
            !grid.hasSO2("terrain_failure_probability_upper")))return false;
        const auto body=collisionEnvelope(footprint_margin_);
        for(const auto &entry:std::vector<std::pair<const char*,double>>{
            {"collision_body_front",body.front},{"collision_body_back",body.back},{"collision_body_half_width",body.half_width}}) {
          if(!grid.exists(entry.first) || grid.hasSO2(entry.first) ||
             !std::isfinite(scalar(entry.first)) || scalar(entry.first)+1e-6<entry.second)return false;
        }
      }
      if (grid.exists("probability_startup_prior_enabled")) {
        if (grid.hasSO2("probability_startup_prior_enabled")) return false;
        const double initial=scalar("probability_startup_prior_enabled");
        if (!std::isfinite(initial) || (initial!=0 && initial!=1) ||
            (initial==1 && (!chance_allow_startup_prior_ || !get_parameter("use_sim_time").as_bool()))) return false;
      } else if (version==3.0 || version==4.0 || version==5.0 || (version==6.0 || (version==7.0 || version==8.0 || version==9.0))) return false;
      if ((version==4.0 || version==5.0 || (version==6.0 || (version==7.0 || version==8.0 || version==9.0))) && (!grid.exists("probability_pose_event") || !grid.hasSO2("probability_pose_event"))) return false;
      if((version==5.0 || (version==6.0 || (version==7.0 || version==8.0 || version==9.0))) && (!chance_allow_joint_model_ || !get_parameter("use_sim_time").as_bool() || !grid.exists("probability_joint_model") ||
          grid.hasSO2("probability_joint_model") || scalar("probability_joint_model")!=1.0))return false;
      if((version==6.0 || (version==7.0 || version==8.0 || version==9.0)) && (!grid.exists("probability_step_model") || grid.hasSO2("probability_step_model") ||
          scalar("probability_step_model")!=1.0))return false;
      if (version==2.0 || version==3.0 || version==4.0 || version==5.0 || (version==6.0 || (version==7.0 || version==8.0 || version==9.0))) {
        if (!grid.exists("height_error_tail_upper") || !grid.hasSO2("height_error_tail_upper") ||
            !grid.exists("probability_height_radius_cells") || grid.hasSO2("probability_height_radius_cells")) return false;
        const double res=grid.getResolutionPos();
        if (!std::isfinite(front) || !std::isfinite(back) || !std::isfinite(half_width) ||
            std::min({front,back,half_width,res})<=0) return false;
        int needed=0;
        for (int k=0;k<grid.getSizeYaw();++k) {
          for(const auto &offset:terrain_nav::heightSupportOffsets(front,back,half_width,res,
              (k+.5)*grid.getResolutionYaw(),version!=2.,footprint_geometry==2.))
            needed=std::max(needed,std::max(std::abs(offset.first),std::abs(offset.second)));
        }
        const double radius=scalar("probability_height_radius_cells");
        if (!std::isfinite(radius) || radius!=std::floor(radius) || radius<needed || radius>32) return false;
      }
      return std::isfinite(version) && (version == 1.0 || version == 2.0 || version == 3.0 || version == 4.0 || version==5.0 || (version==6.0 || (version==7.0 || version==8.0 || version==9.0))) &&
        std::isfinite(gp) && (gp == 0.0 || (gp == 1.0 && chance_allow_gp_completion_)) &&
        std::isfinite(pitch) && pitch > 0 && pitch <= chance_max_pitch_rad_+1e-6 &&
        std::isfinite(roll) && roll > 0 && roll <= chance_max_roll_rad_+1e-6 &&
        std::isfinite(step) && step > 0 && step <= chance_max_step_m_+1e-6 &&
        std::isfinite(front) && front+1e-6 >= footprint_front_+footprint_margin_ &&
        std::isfinite(back) && back+1e-6 >= footprint_back_+footprint_margin_ &&
        std::isfinite(half_width) && half_width+1e-6 >= footprint_half_width_+footprint_margin_;
    } catch (...) { return false; }
  }

  bool addChancePose(double x, double y, double yaw, terrain_nav::ChanceBudget &budget,
                     double limit, const ControlMapSnapshot *snapshot = nullptr,
                     const LiveObstacleSnapshot *live_override = nullptr) const
  {
    if (!chance_constrained_) return true;
    const auto q = queryCostLocked(x, y, yaw, false, snapshot, true,true,live_override);
    return addChanceQuery(q,budget,limit,snapshot);
  }

  // Consume the exact query produced by the same immutable-map footprint
  // check in this call. This is not a cache across poses or generations.
  bool addChanceQuery(const CostQuery &q,terrain_nav::ChanceBudget &budget,
                      double limit,const ControlMapSnapshot *snapshot=nullptr) const
  {
    if (!chance_constrained_) return true;
    if (q.lethal || !q.known) return false;
    if (!q.height_radius) return budget.add(q.probability_event,q.probability_upper,limit);
    const auto &grid=snapshot ? snapshot->grid : map_;
    const auto size=grid.getSizePos();
    const auto center=grid.getPosition();
    Eigen::Array3i center_index;
    if(!grid.pos2Index(Eigen::Vector3d(center.x(),center.y(),0),center_index))return false;
    const auto scalar=[&](const char *name) {return grid.at(name,center_index);};
    const std::array<double,7> key={scalar("probability_front"),scalar("probability_back"),
      scalar("probability_half_width"),grid.getResolutionPos(),grid.getResolutionYaw(),scalar("probability_contract"),
      grid.exists("probability_footprint_geometry")?scalar("probability_footprint_geometry"):1.};
    thread_local std::array<double,7> cached_key{};
    thread_local std::vector<std::vector<std::pair<int,int>>> cached_offsets;
    if(cached_key!=key || cached_offsets.size()!=static_cast<size_t>(grid.getSizeYaw())) {
      cached_key=key;cached_offsets.clear();
      for(int k=0;k<grid.getSizeYaw();++k) cached_offsets.push_back(terrain_nav::heightSupportOffsets(
        key[0],key[1],key[2],key[3],(k+.5)*key[4],key[5]!=2.0,key[6]==2.));
    }
    std::vector<uint64_t> sources;sources.reserve(cached_offsets.at(q.height_yaw).size());
    std::vector<double> absolute_tails;
    const bool absolute=grid.exists("height_error_radius_m") && grid.hasSO2("height_error_radius_m") &&
      grid.exists("probability_variance") && !grid.hasSO2("probability_variance") &&
      grid.exists("probability_height_std_floor") && grid.exists("probability_variance_scale");
    double radius=0,floor=0,scale=0;
    if(absolute) {
      radius=grid.at("height_error_radius_m",Eigen::Array3i(q.height_row,q.height_col,q.height_yaw));
      floor=scalar("probability_height_std_floor");scale=scalar("probability_variance_scale");
      if(!std::isfinite(radius) || radius<=0 || !std::isfinite(floor) || floor<0 ||
         !std::isfinite(scale) || scale<=0) return false;
      absolute_tails.reserve(cached_offsets.at(q.height_yaw).size());
    }
    for(const auto &offset:cached_offsets.at(q.height_yaw)) {
      // Native circular rows/cols descend X/Y; world offsets ascend Y/X.
      const int row=((q.height_row-offset.second)%size[0]+size[0])%size[0];
      const int col=((q.height_col-offset.first)%size[1]+size[1])%size[1];
      const uint64_t event=static_cast<uint64_t>(row)*size[1]+col;
      sources.push_back(event);
      if(absolute) {
        const double variance=grid.at("probability_variance",Eigen::Array3i(row,col,0));
        if(!std::isfinite(variance) || variance<0) return false;
        const double sigma=std::sqrt(variance*scale+floor*floor);
        absolute_tails.push_back(std::erfc(radius/(std::sqrt(2.)*std::max(1e-12,sigma))));
      }
    }
    return budget.addAlternatives(q.probability_event|(uint64_t{1}<<63),
      q.probability_upper,sources,q.height_tail_upper,limit,absolute_tails);
  }

  bool chancePathSafe(const std::vector<Pose2> &path, const ControlMapSnapshot *snapshot,
                      double limit) const
  {
    if (!chance_constrained_) return true;
    if (path.empty()) return false;
    terrain_nav::ChanceBudget budget;
    if (!addChancePose(path.front().x,path.front().y,path.front().yaw,budget,limit,snapshot)) return false;
    const auto &grid = snapshot ? snapshot->grid : map_;
    for (size_t i=1;i<path.size();++i) {
      const auto &a=path[i-1], &b=path[i];
      const double angle=wrapAngle(b.yaw-a.yaw);
      const int steps=std::max(1,static_cast<int>(std::ceil(std::max(
        std::hypot(b.x-a.x,b.y-a.y)/(.25*grid.getResolutionPos()),
        std::abs(angle)/(.25*grid.getResolutionYaw())))));
      for (int j=1;j<=steps;++j) {
        const double u=static_cast<double>(j)/steps;
        if (!addChancePose(a.x+u*(b.x-a.x),a.y+u*(b.y-a.y),a.yaw+u*angle,budget,limit,snapshot)) return false;
      }
    }
    return true;
  }

  int collisionSubcellFactor(const se2_grid::SE2Grid &grid,const Eigen::Array3i &index) const
  {
    if(grid.exists("collision_subcell_factor") && !grid.hasSO2("collision_subcell_factor") &&
       grid.at("collision_subcell_factor",index)==4.)return 4;
    return 2;
  }

  int collisionSubcellBits(const se2_grid::SE2Grid &grid,const Eigen::Array3i &index) const
  {
    // Optional refinement of a coarse occupied square. Missing, malformed or
    // empty witnesses remain the WHOLE square, including legacy recordings.
    const int factor=collisionSubcellFactor(grid,index),full=(1<<(factor*factor))-1;
    for(const char *name:{"collision_subcell_bits","collision_subcell_factor","collision_subcell_contract"})
      if(!grid.exists(name) || grid.hasSO2(name))return full;
    const double contract=grid.at("collision_subcell_contract",index);
    if((grid.at("collision_subcell_factor",index)!=2. && grid.at("collision_subcell_factor",index)!=4.) ||
       (contract!=1. && contract!=2. && contract!=3.))return full;
    const double bits=grid.at("collision_subcell_bits",index);
    if(!std::isfinite(bits) || bits<1 || bits>full || bits!=std::floor(bits))return full;
    return static_cast<int>(bits);
  }

  int historicalObstacleSubcellBits(const se2_grid::SE2Grid &grid,const Eigen::Array3i &index) const
  {
    for(const char *name:{"historical_obstacle_subcell_bits","historical_obstacle_subcell_factor"})
      if(!grid.exists(name) || grid.hasSO2(name))return 15;
    if(grid.at("historical_obstacle_subcell_factor",index)!=2.)return 15;
    const double bits=grid.at("historical_obstacle_subcell_bits",index);
    if(!std::isfinite(bits) || bits<1 || bits>15 || bits!=std::floor(bits))return 15;
    return static_cast<int>(bits);
  }

  int collisionViewBits(const se2_grid::SE2Grid &grid,const Eigen::Array3i &index,
                        const LiveObstacleSnapshot *view) const
  {
    int bits=collisionSubcellBits(grid,index);
    if(view&&view->map_exclusions&&view->map_exclusions->grid==&grid) {
      const auto found=view->map_exclusions->bits.find(size_t(index[0])*grid.getSizePos()[1]+index[1]);
      if(found!=view->map_exclusions->bits.end())bits&=~found->second;
    }
    return bits;
  }

  void labelMovingSubcells(ControlMapSnapshot &map,const LiveObstacleSnapshot &source) const
  {
    if(std::abs(map.stamp.seconds()-source.stamp_sec)>5e-9||source.inflation_radius!=0.)return;
    const auto &grid=map.grid;const auto size=grid.getSizePos();
    for(const auto &[id,cells]:source.moving_cells)for(size_t cell:cells) {
      if(cell>=source.uninflated_present.size()||!source.uninflated_present[cell])continue;
      const double x=source.min_x+(cell%source.width)*source.resolution,
                   y=source.min_y+(cell/source.width)*source.resolution;
      const std::array<double,4> owner{x,y,x+source.resolution,y+source.resolution};
      // Usually one coarse cell; corners also cover grids with different
      // origins. Only a whole occupied subrectangle contained in the original
      // measured cell may be labeled, never a circle or its bounding box.
      std::set<size_t> visited;
      for(double px:{owner[0]+1e-8,owner[2]-1e-8})for(double py:{owner[1]+1e-8,owner[3]-1e-8}) {
        Eigen::Array3i index;if(!grid.pos2Index(Eigen::Vector3d(px,py,0.),index))continue;
        const size_t key=size_t(index[0])*size[1]+index[1];if(!visited.insert(key).second)continue;
        if(!grid.exists("collision_subcell_contract")||grid.at("collision_subcell_contract",index)!=3.||
           !std::isfinite(grid.at("collision_risk",index))||grid.at("collision_risk",index)<.5)continue;
        const int bits=collisionSubcellBits(grid,index),factor=collisionSubcellFactor(grid,index);
        for(int sy=0;sy<factor;++sy)for(int sx=0;sx<factor;++sx) {
          const int bit=1<<(sy*factor+sx);
          if((bits&bit)&&terrain_nav::rectangleInsideMeasuredCell(collisionSubcellRectangle(grid,index,sx,sy),owner))
            map.moving_subcells[id][key]|=bit;
        }
      }
    }
  }

  void bindYieldCollisionViews(const ControlMapSnapshot &map,const LiveObstacleSnapshot &live,
                              YieldFootprintCache &cache) const
  {
    cache.bind(this,map,live);if(cache.current_view&&cache.future_view)return;
    auto current=std::make_shared<LiveObstacleSnapshot>(live);current->hard=live.present_hard;
    auto future=std::make_shared<LiveObstacleSnapshot>(*current);
    if(dynamic_time_indexed_routing_&&motion_prediction_model_!="reachable"&&live.inflation_radius==0.) {
      std::map<uint64_t,size_t> ids;for(const auto &t:live.tracks)if(t.id)++ids[t.id];
      auto exclusions=std::make_shared<LiveObstacleSnapshot::MapExclusions>();exclusions->grid=&map.grid;
      for(const auto &t:live.tracks)if(ids[t.id]==1&&terrain_nav::currentMeasuredIdentity(t,live.stamp_sec,false)) {
        const auto found=map.moving_subcells.find(t.id);
        if(found!=map.moving_subcells.end())for(const auto &[cell,bits]:found->second)exclusions->bits[cell]|=bits;
        // Current returns remain physical occupancy. A source-owned historic
        // position is not a second stationary copy of the SAME freshly
        // observed body. No absent/unowned cell becomes ground support.
        // FUTURE static checks delegate the body to the same fresh all-object
        // time-indexed proof, including a measured near-zero velocity. A new
        // static object has no confirmed motion identity and cannot enter.
        const auto owned=live.held_moving_cells.find(t.id);
        if(owned!=live.held_moving_cells.end())for(size_t cell:owned->second) {
          if(cell>=future->hard.size())continue;
          future->hard[cell]=0;
          if(live.current_measured.size()==current->hard.size()&&!live.current_measured[cell]&&current->hard[cell]) {
            current->hard[cell]=0;++current->owned_history_exclusions;
          }
        }
      }
      current->map_exclusions=future->map_exclusions=exclusions;
    }
    cache.current_view=current;cache.future_view=future;
  }

  bool collisionAtPoint(const se2_grid::SE2Grid &grid,const Eigen::Array3i &index,double x,double y,
                        const LiveObstacleSnapshot *view=nullptr) const
  {
    const int bits=collisionViewBits(grid,index,view);
    if(!bits)return false;
    const int factor=collisionSubcellFactor(grid,index);
    if(bits==((1<<(factor*factor))-1) &&
       (!grid.exists("collision_subcell_xy_bounds_contract") ||
        grid.hasSO2("collision_subcell_xy_bounds_contract") ||
        grid.at("collision_subcell_xy_bounds_contract",index)!=1.))return true;
    Eigen::Vector3d center;if(!grid.index2Pos(index,center))return true;
    const double half=grid.getResolutionPos()/2.,fine=grid.getResolutionPos()/factor;
    const int sx=std::clamp(static_cast<int>(std::floor((x-center.x()+half)/fine)),0,factor-1);
    const int sy=std::clamp(static_cast<int>(std::floor((y-center.y()+half)/fine)),0,factor-1);
    if(!(bits&(1<<(sy*factor+sx))))return false;
    const auto b=collisionSubcellRectangle(grid,index,sx,sy);
    return x>=b[0]-1e-9&&x<=b[2]+1e-9&&y>=b[1]-1e-9&&y<=b[3]+1e-9;
  }

  std::array<double,4> collisionSubcellRectangle(const se2_grid::SE2Grid &grid,
      const Eigen::Array3i &index,int sx,int sy) const
  {
    Eigen::Vector3d center;grid.index2Pos(index,center);
    const int factor=collisionSubcellFactor(grid,index);
    const double fine=grid.getResolutionPos()/factor;
    const double ox=center.x()-grid.getResolutionPos()/2+sx*fine,
                 oy=center.y()-grid.getResolutionPos()/2+sy*fine;
    std::array<double,4> result={ox,oy,ox+fine,oy+fine};
    if(!grid.exists("collision_subcell_xy_bounds_contract") || grid.hasSO2("collision_subcell_xy_bounds_contract") ||
       grid.at("collision_subcell_xy_bounds_contract",index)!=1.)return result;
    static const auto names=[] {std::array<std::string,16> out;
      for(int i=0;i<16;++i)out[i]="collision_subcell_xy_bounds_"+std::to_string(i);return out;}();
    const auto &name=names[sy*factor+sx];
    if(!grid.exists(name)||grid.hasSO2(name))return result;
    const double value=grid.at(name,index);
    if(!std::isfinite(value)||value<0||value>65535||value!=std::floor(value))return result;
    const auto code=static_cast<unsigned>(value);
    const unsigned x0=code&15,x1=(code>>4)&15,y0=(code>>8)&15,y1=(code>>12)&15;
    if(x0>=x1||y0>=y1)return result;
    return {ox+x0*fine/15.,oy+y0*fine/15.,ox+x1*fine/15.,oy+y1*fine/15.};
  }

  bool collisionCellOverlapsBody(const se2_grid::SE2Grid &grid,const Eigen::Array3i &index,
      double x,double y,double yaw,double front,double back,double half_width,
      const LiveObstacleSnapshot *view=nullptr) const
  {
    Eigen::Vector3d center;if(!grid.index2Pos(index,center))return true;
    const int bits=collisionViewBits(grid,index,view),factor=collisionSubcellFactor(grid,index);
    const double c=std::cos(yaw),s=std::sin(yaw),a=(front+back)*.5;
    const double cx=x+(front-back)*.5*c,cy=y+(front-back)*.5*s;
    const double hx=std::abs(c)*a+std::abs(s)*half_width,hy=std::abs(s)*a+std::abs(c)*half_width;
    for(int sy=0;sy<factor;++sy)for(int sx=0;sx<factor;++sx) {
      if(!(bits&(1<<(sy*factor+sx))))continue;
      const auto b=collisionSubcellRectangle(grid,index,sx,sy);
      const double bx=(b[2]-b[0])*.5,by=(b[3]-b[1])*.5;
      const double dx=(b[0]+b[2])*.5-cx,dy=(b[1]+b[3])*.5-cy;
      if(std::abs(dx)<=hx+bx && std::abs(dy)<=hy+by &&
         std::abs(c*dx+s*dy)<=a+bx*std::abs(c)+by*std::abs(s) &&
         std::abs(-s*dx+c*dy)<=half_width+bx*std::abs(s)+by*std::abs(c))return true;
    }
    return false;
  }

  bool collisionBodyUnsafe(const se2_grid::SE2Grid &grid,double x,double y,double yaw,bool &known,
                           const LiveObstacleSnapshot *view=nullptr) const
  {
    known=true;const auto body=collisionEnvelope(footprint_margin_);
    const double res=grid.getResolutionPos(),radius=std::hypot(std::max(body.front,body.back),body.half_width)+res;
    const auto center=grid.getPosition();const auto length=grid.getLengthPos();
    const double left=center.x()-length[0]/2,bottom=center.y()-length[1]/2;
    for(int iy=int(std::floor((y-radius-bottom)/res));iy<=int(std::floor((y+radius-bottom)/res));++iy)
    for(int ix=int(std::floor((x-radius-left)/res));ix<=int(std::floor((x+radius-left)/res));++ix) {
      const double px=left+(ix+.5)*res,py=bottom+(iy+.5)*res;
      if(!bodyOverlapsCell(x,y,yaw,body.front,body.back,body.half_width,px,py,res))continue;
      Eigen::Array3i index;
      if(!grid.pos2Index(Eigen::Vector3d(px,py,yaw),index)){known=false;return true;}
      const double occupied=grid.at("collision_risk",index);
      if(!std::isfinite(occupied)){known=false;return true;}
      if(occupied>=.5 && collisionCellOverlapsBody(grid,index,x,y,yaw,body.front,body.back,body.half_width,view))return true;
    }
    return false;
  }

  CostQuery queryCostLocked(
    double x, double y, double yaw = 0.0, bool allow_fresh_unknown = false,
    const ControlMapSnapshot *snapshot = nullptr, bool include_yaw_risk = true,
    bool include_live = true, const LiveObstacleSnapshot *live_override = nullptr) const
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
    const auto live_owner = live_override || !include_live ? std::shared_ptr<const LiveObstacleSnapshot>{} : std::atomic_load(&live_snapshot_);
    const auto *live = live_override ? live_override : live_owner.get();
    CostQuery out;
    if (include_live && live && live->occupied(x, y)) {
      out.known = true;
      out.lethal = true;
      out.risk = 1.0;
      out.clearance = 0.0;
      out.comfort_clearance = 0.0;
      out.cost = mppi_collision_cost_;
      return out;
    }
    if (chance_constrained_) {
      // Occupancy remains a point-wise footprint check. Probability is a
      // whole-pose event and is evaluated only once at the vehicle center.
      Eigen::Array3i index;
      if (!std::isfinite(yaw) || !grid.pos2Index(Eigen::Vector3d(x, y, yaw),index)) {
        out.lethal = true;
        out.cost = mppi_collision_cost_;
        return out;
      }
      try {
        const double collision = grid.at("collision_risk",index);
        if (!std::isfinite(collision) || (collision >= 0.5 && collisionAtPoint(grid,index,x,y,live_override))) {
          out.known = std::isfinite(collision);
          if(out.known) {out.risk=1.;out.probability_upper=1.;}
          out.lethal = true;
          out.cost = mppi_collision_cost_;
          return out;
        }
        out.known = true;
        const int ix = static_cast<int>(std::floor((x-min_x)/resolution));
        const int iy = static_cast<int>(std::floor((y-min_y)/resolution));
        if (ix >= 0 && iy >= 0 && ix < width && iy < height) {
          const size_t idx = static_cast<size_t>(iy*width+ix);
          out.clearance = clearance[idx];
          out.comfort_clearance = comfort_clearance[idx];
        }
        if (include_yaw_risk) {
          const bool continuous=grid.exists("collision_subcell_contract") &&
            grid.at("collision_subcell_contract",index)==3.;
          if(continuous) {
            bool known=false;
            if(collisionBodyUnsafe(grid,x,y,yaw,known,live_override)) {
              out.known=known;out.lethal=true;out.cost=mppi_collision_cost_;
              if(known){out.risk=1.;out.probability_upper=1.;}
              return out;
            }
          }
          const double valid = grid.at("probability_valid",index);
          const double p = grid.at(continuous?"terrain_failure_probability_upper":"failure_probability_upper",index);
          if (!std::isfinite(valid) || valid < 0.5 || !std::isfinite(p) || p < 0 || p > 1) {
            out.known = false;
            out.lethal = true;
          } else {
            out.probability_upper = p;
            const auto size = grid.getSizePos();
            out.probability_event = (static_cast<uint64_t>(index[2])*size[0]+index[0])*size[1]+index[1];
            out.risk = p;
            // Conservative additive surrogate. This is not -log(survival)
            // and does not assume independent neighboring terrain errors.
            out.cost = risk_weight_*p/chance_state_limit_;
            out.lethal = p > chance_state_limit_;
            const double version=grid.at("probability_contract",index);
            bool pose_event=false;
            if (version==4.0 || version==5.0 || (version==6.0 || (version==7.0 || version==8.0 || version==9.0))) {
              const double mode=grid.at("probability_pose_event",index);
              if (!std::isfinite(mode) || (mode!=0 && mode!=1)) out.lethal=true;
              pose_event=mode==1;
              if(version==9. && !pose_event)out.lethal=true;
              if (pose_event) out.probability_event |= (uint64_t{1}<<63);
            }
            // A conditional pose proof does not invalidate the original
            // marginal-height envelope. Retain both complete certificates;
            // shared height events may be reused, distinct poses may not.
            const bool absolute_sources=grid.exists("height_error_radius_m") &&
              grid.hasSO2("height_error_radius_m") && grid.exists("probability_variance") &&
              !grid.hasSO2("probability_variance") && grid.exists("probability_height_std_floor") &&
              grid.exists("probability_variance_scale");
            if ((version==2.0 || version==3.0 || version==4.0 || version==5.0 || (version==6.0 || (version==7.0 || version==8.0))) && (!pose_event || absolute_sources)) {
              out.height_tail_upper=grid.at("height_error_tail_upper",index);
              const double radius=grid.at("probability_height_radius_cells",index);
              if(!std::isfinite(radius) || radius<1 || radius>32 || radius!=std::floor(radius)) {
                out.lethal=true;out.cost=mppi_collision_cost_;return out;
              }
              out.height_radius=static_cast<int>(radius);
              out.height_row=index[0]; out.height_col=index[1];out.height_yaw=index[2];
              if (!std::isfinite(out.height_tail_upper) || out.height_tail_upper<0 || out.height_tail_upper>1 ||
                  out.height_radius<1 || out.height_radius>32) out.lethal=true;
            }
          }
        }
        if (std::isfinite(out.clearance) && out.clearance < clearance_distance_) {
          const double margin = (clearance_distance_-out.clearance)/std::max(1e-3,clearance_distance_);
          out.cost += clearance_weight_*margin*margin;
        }
      } catch (...) {
        out.known = false;
        out.lethal = true;
      }
      // A bounded route preference is not occupancy or a failure probability.
      // Control/braking snapshots and every hard constraint remain unchanged.
      if(!snapshot && !out.lethal) {
        if(include_yaw_risk)out.cost+=recoveryRoutePenalty(x,y);
        out.cost+=historicalObstacleCost(x,y);
      }
      if (out.lethal) out.cost = mppi_collision_cost_;
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
      const auto &evidence=snapshot?snapshot->observed_ground_until:observed_ground_until_;
      const bool observed_shadow=evidence.size()==costs.size() && evidence[idx]>now().seconds() &&
        std::hypot(x-current_robot_x_.load(),y-current_robot_y_.load())<=1.0 &&
        unknown_baseline[idx]<risk_hard_threshold_;
      // A fresh point cloud is not evidence that every unknown terrain cell
      // has ground support. Wait for the terrain map outside the robot bubble.
      out.lethal = !allow_unknown_ && !robot_bubble && !observed_shadow;
      out.cost = allow_unknown_ ? unknown_cost_ : mppi_collision_cost_;
      if (robot_bubble) out.cost = unknown_cost_;
      if (observed_shadow) {out.known=true;out.risk=.65;out.cost=.65*risk_weight_;}
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
    if (include_yaw_risk && risk_query_mode_ == "se2_yaw" &&
        grid.exists(se2_risk_layer_) &&
        grid.hasSO2(se2_risk_layer_))
    {
      try {
        double se2_risk = grid.getValue(se2_risk_layer_,Eigen::Vector3d(x,y,yaw),
          se2_grid::InterpolationMethods::INTER_NEAREST);
        // A nearest grid center can miss a hazard at the vehicle pose by half
        // a cell. Use the surrounding cells conservatively, not an average.
        for (double dx:{-.5*resolution,.5*resolution}) for (double dy:{-.5*resolution,.5*resolution}) {
          try {
            const double neighbor=grid.getValue(se2_risk_layer_,Eigen::Vector3d(x+dx,y+dy,yaw),
              se2_grid::InterpolationMethods::INTER_NEAREST);
            if(std::isfinite(neighbor)) se2_risk=std::isfinite(se2_risk)?std::max(se2_risk,neighbor):neighbor;
          } catch (...) {}
        }
        if (std::isfinite(se2_risk))
        {
          out.risk = std::max(out.risk, std::clamp(se2_risk, 0.0, 1.0));
          out.cost = std::max(out.cost, out.risk * risk_weight_);
          if (se2_risk_lethal_ && se2_risk >= se2_hard_threshold_)
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

  double recoveryRoutePenalty(double x,double y) const
  {
    const auto memory=std::atomic_load(&recovery_route_memory_);
    if(!memory || memory->empty())return 0.;
    const double time=now().seconds();
    double penalty=0.;
    for(const auto &entry:*memory) {
      const double age=time-entry.stamp;
      if(entry.epoch!=input_epoch_.load() || age<0 || age>=15.)continue;
      const double distance2=(x-entry.x)*(x-entry.x)+(y-entry.y)*(y-entry.y);
      if(distance2>=.8*.8)continue;
      const double shape=1.-distance2/(.8*.8);
      penalty=std::max(penalty,15.*shape*shape*(1.-age/15.));
    }
    return penalty;
  }

  double historicalObstacleCost(double x,double y,const ControlMapSnapshot *snapshot=nullptr) const
  {
    const double min_x=snapshot?snapshot->min_x:min_x_,min_y=snapshot?snapshot->min_y:min_y_;
    const double res=snapshot?snapshot->resolution:resolution_;
    const int width=snapshot?snapshot->width:width_,height=snapshot?snapshot->height:height_;
    const auto &clearance=snapshot?snapshot->historical_clearance:historical_clearance_;
    if(res<=0 || width<=0 || height<=0 || clearance.size()!=static_cast<size_t>(width*height))return 0.;
    const int ix=static_cast<int>(std::floor((x-min_x)/res)),iy=static_cast<int>(std::floor((y-min_y)/res));
    if(ix<0 || iy<0 || ix>=width || iy>=height)return 0.;
    const double margin=std::max(0.,1.-clearance[iy*width+ix]/historical_obstacle_influence_distance_);
    return historical_obstacle_cost_weight_*margin*margin;
  }

  void rememberBlockedRoute(const OdomState &odom,uint64_t sequence)
  {
    std::lock_guard<std::mutex> lock(goal_mutex_);
    if(!have_goal_ || goal_seq_!=sequence)return;
    auto entries=std::make_shared<std::vector<RecoveryRouteMemory>>();
    const double time=now().seconds();const auto epoch=input_epoch_.load();
    const auto previous=std::atomic_load(&recovery_route_memory_);
    if(previous)
      for(const auto &entry:*previous)
        if(entry.epoch==epoch && time>=entry.stamp && time-entry.stamp<15.)entries->push_back(entry);
    const auto same=std::find_if(entries->begin(),entries->end(),[&](const auto &entry){
      return std::hypot(entry.x-odom.x,entry.y-odom.y)<.4;
    });
    // Failed retries at 3-4 Hz used to replace the immutable cost identity
    // faster than a 350ms global job could finish, invalidating its result
    // without a different branch/obstacle/goal. Coalesce this SAME finite
    // preference for one second; expiry and the actual safety maps are intact.
    if(previous && entries->size()==previous->size() && same!=entries->end() && time-same->stamp<1.)return;
    if(same==entries->end())entries->push_back({odom.x,odom.y,time,epoch});
    else same->stamp=time;
    if(entries->size()>4)entries->erase(entries->begin(),entries->end()-4);
    std::atomic_store(&recovery_route_memory_,std::shared_ptr<const std::vector<RecoveryRouteMemory>>(entries));
  }


  bool livePathAheadUnsafe(const std::vector<Pose2> &path, size_t nearest_idx)
  {
    const auto live = std::atomic_load(&live_snapshot_);
    if (!live || path.empty()) return false;
    const double start_s = path[nearest_idx].s;
    const double end_s = std::min(path.back().s, start_s + std::max(3.0, replan_path_remaining_ + 1.2));
    for (double s = start_s; s <= end_s + 1e-6; s += 0.15) {
      const Pose2 p = samplePathAt(path, s);
      std::array<double,4> rectangle{};int cell=-1;
      if(liveBodyOverlapCount(p.x,p.y,p.yaw,*live,1,&rectangle,&cell)) {
        const auto map=std::atomic_load(&control_map_snapshot_);
        OdomState root;{std::lock_guard<std::mutex> lock(odom_mutex_);root=odom_;}
        static rclcpp::Clock log_clock(RCL_STEADY_TIME);
        RCLCPP_WARN_THROTTLE(get_logger(),log_clock,500,
          "terrain ahead rejected: layer=live map_ns=%lld cloud_ns=%lld now_ns=%lld root=(%.9f,%.9f,%.9f) sample=(%.9f,%.9f,%.9f,%.9f) points=%zu end=(%.9f,%.9f) nearest=%zu rectangle=(%.9f,%.9f,%.9f,%.9f) present=%d forecast=%d",
          static_cast<long long>(map?map->stamp.nanoseconds():0),static_cast<long long>(std::llround(live->stamp_sec*1e9)),
          static_cast<long long>(now().nanoseconds()),root.x,root.y,root.yaw,p.x,p.y,p.yaw,s-start_s,
          path.size(),path.back().x,path.back().y,nearest_idx,rectangle[0],rectangle[1],rectangle[2],rectangle[3],
          int(cell>=0&&size_t(cell)<live->present_hard.size()&&live->present_hard[cell]),
          int(cell>=0&&size_t(cell)<live->forecast.size()&&live->forecast[cell]));
        return true;
      }
    }
    return false;
  }

  struct CollisionEnvelope {double front,back,half_width;};
  CollisionEnvelope collisionEnvelope(double margin) const
  {
    return {std::max(footprint_front_,body_collision_front_)+margin,
            std::max(footprint_back_,body_collision_back_)+margin,
            std::max(footprint_half_width_,body_collision_half_width_)+margin};
  }

  size_t liveBodyOverlapCount(double x,double y,double yaw,const LiveObstacleSnapshot &live,
                             size_t stop_after=std::numeric_limits<size_t>::max(),
                             std::array<double,4> *first_rectangle=nullptr,int *first_cell=nullptr) const
  {
    // This layer already carries inflation and the motion-association reserve;
    // the uninflated terrain layer retains the full separate footprint margin.
    const double inflation=std::isfinite(live.inflation_radius) ? live.inflation_radius : live_obstacle_inflation_radius_;
    const double margin=std::max(0.,footprint_margin_-inflation);
    const auto envelope=collisionEnvelope(margin);
    const double f=envelope.front,b=envelope.back,w=envelope.half_width;
    const double r=std::hypot(std::max(f,b),w)+live.resolution;
    const int x0=std::max(0,static_cast<int>(std::floor((x-r-live.min_x)/live.resolution)));
    const int x1=std::min(live.width-1,static_cast<int>(std::floor((x+r-live.min_x)/live.resolution)));
    const int y0=std::max(0,static_cast<int>(std::floor((y-r-live.min_y)/live.resolution)));
    const int y1=std::min(live.height-1,static_cast<int>(std::floor((y+r-live.min_y)/live.resolution)));
    size_t overlaps=0;
    for(int iy=y0;iy<=y1;++iy)for(int ix=x0;ix<=x1;++ix) {
      if(!live.hard[static_cast<size_t>(iy*live.width+ix)])continue;
      const auto rectangle=liveCellRectangle(live,ix,iy);
      if(bodyOverlapsRectangle(x,y,yaw,f,b,w,rectangle)) {
        if(overlaps==0) {
          if(first_rectangle)*first_rectangle=rectangle;
          if(first_cell)*first_cell=iy*live.width+ix;
        }
        if(++overlaps>=stop_after)return overlaps;
      }
    }
    return overlaps;
  }

  bool liveBodyUnsafe(double x,double y,double yaw,const LiveObstacleSnapshot &live) const
  {
    return liveBodyOverlapCount(x,y,yaw,live,1)>0;
  }

  std::array<double,4> liveCellRectangle(const LiveObstacleSnapshot &live,int x,int y) const
  {
    // Forecasts and extra inflation keep their FULL conservative geometry.
    // Old/missing/malformed witness generations likewise retain whole cells.
    return live.rectangle(x,y);
  }

  std::array<Eigen::Vector2d, 25> footprintOffsets() const
  {
    const auto envelope=collisionEnvelope(footprint_margin_);
    const double f=envelope.front,b=envelope.back,w=envelope.half_width;
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
    const ControlMapSnapshot *snapshot = nullptr,
    const LiveObstacleSnapshot *live_override = nullptr,
    CostQuery *center_query = nullptr) const
  {
    cost = 0.0;
    if(center_query)*center_query=CostQuery{};
    double clr = kInf;
    double comfort_clr = kInf;
    const double res=snapshot ? snapshot->resolution : resolution_;
    const double min_x=snapshot ? snapshot->min_x : min_x_;
    const double min_y=snapshot ? snapshot->min_y : min_y_;
    // Collision geometry is identical in legacy and probability modes.
    // Ground-contact support is separate from the raised body envelope.
    const auto envelope=collisionEnvelope(footprint_margin_);
    const double front=envelope.front,back=envelope.back,width=envelope.half_width;
    const double radius=std::hypot(std::max(front,back),width)+res;
    const int first_x=static_cast<int>(std::floor((x-radius-min_x)/res));
    const int last_x=static_cast<int>(std::floor((x+radius-min_x)/res));
    const int first_y=static_cast<int>(std::floor((y-radius-min_y)/res));
    const int last_y=static_cast<int>(std::floor((y+radius-min_y)/res));
    for (int iy=first_y;iy<=last_y;++iy) for (int ix=first_x;ix<=last_x;++ix) {
      const double px=min_x+(ix+.5)*res, py=min_y+(iy+.5)*res;
      if (!bodyOverlapsCell(x,y,yaw,front,back,width,px,py,res)) continue;
      if(chance_constrained_) {
        const auto &grid=snapshot?snapshot->grid:map_;
        Eigen::Array3i index;
        if(!grid.pos2Index(Eigen::Vector3d(px,py,yaw),index))return false;
        const double occupied=grid.at("collision_risk",index);
        if(std::isfinite(occupied) && occupied>=.5) {
          if(collisionCellOverlapsBody(grid,index,x,y,yaw,front,back,width,live_override))return false;
          // Same coarse hazard, refined geometry: do not use the occupied
          // coarse cell's zero distance or collision cost as a second veto.
          continue;
        }
      }
      // The terrain collision layer is not inflated: retain the body margin.
      const auto q=queryCostLocked(px,py,yaw,allow_fresh_unknown,snapshot,false,false,live_override);
      if(q.lethal) return false;
      cost=std::max(cost,q.cost); clr=std::min(clr,q.clearance);
      comfort_clr=std::min(comfort_clr,q.comfort_clearance);
    }
    const auto live_owner=live_override ? std::shared_ptr<const LiveObstacleSnapshot>{} : std::atomic_load(&live_snapshot_);
    const auto *live=live_override ? live_override : live_owner.get();
    if(live && liveBodyUnsafe(x,y,yaw,*live))return false;
    // SEB / pose probability already describes a whole vehicle. Applying it
    // again at each footprint cell would count the vehicle dimensions twice.
    const auto center=queryCostLocked(x,y,yaw,allow_fresh_unknown,snapshot,true,true,live_override);
    if(center_query)*center_query=center;
    if(center.lethal) return false;
    cost=std::max(cost,center.cost);
    if(min_clearance) *min_clearance=clr;
    if(min_comfort_clearance) *min_comfort_clearance=comfort_clr;
    return true;
  }

  double hardClearanceFloor() const
  {
    if (hard_clearance_floor_ >= 0.0) return hard_clearance_floor_;
    return std::max(0.24, 0.30 * clearance_distance_);
  }

  bool knownHazardAheadLocked(const Pose2 &start,double distance) const
  {
    const double c=std::cos(start.yaw),sn=std::sin(start.yaw);
    const double width=std::max(footprint_half_width_,body_collision_half_width_)+footprint_margin_;
    const double step=std::min(.10,.5*resolution_);
    for(double s=.25;s<=distance+1e-6;s+=.10) {
      const double x=start.x+s*c,y=start.y+s*sn;
      if(chance_constrained_ && map_.exists("collision_risk")) {
        bool known=false;
        if(collisionBodyUnsafe(map_,x,y,start.yaw,known) && known)return true;
      }
      const auto q=queryCostLocked(x,y,start.yaw);
      if(q.known && q.lethal) return true;
      for(double lateral=-width;lateral<=width+1e-6;lateral+=step) {
        const auto occupied=queryCostLocked(x-lateral*sn,y+lateral*c,start.yaw,false,nullptr,false);
        if(occupied.known && occupied.lethal) return true;
      }
    }
    return false;
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
        const bool route_near = !footprint_ok ||
          (std::isfinite(clearance) && clearance < hardClearanceFloor());
        // A historical proximity preference is NOT a blocked corridor. Using
        // 15% of its 2m influence as a boolean created an extra ~1.23m branch
        // exclusion around each pillar, even when the full body fitted.
        if (route_near) {
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

  static double unknownTerrainGuidanceCost(double comfort_weight)
  {
    return 8.+std::max(0.,comfort_weight);
  }

  static double terrainGuidanceCost(double upper,double mean_height_radius,double state_limit,
      double comfort_weight,double comfort)
  {
    const double weight=std::max(0.,comfort_weight);
    if(!std::isfinite(upper)||upper<0.||upper>1.||!std::isfinite(state_limit)||state_limit<=0.)
      return unknownTerrainGuidanceCost(weight);
    double risk=std::min(29.,6.*upper/state_limit);
    // A positive geometric height radius certifies only feasibility of the
    // SELECTED MEAN, not uncertain ground or an executable vehicle pose.
    // A loose statistical upper bound on such a mean must not make measured
    // flat ground worse guidance than a wholly unseen cell. Cap epistemic
    // preference at the unknown base cost, retaining the same comfort term.
    // Mean violations / absent witnesses keep the original high risk cost.
    // This changes ONLY optimistic routing; U, validity and every state/path/
    // braking admission test remain untouched.
    if(std::isfinite(mean_height_radius)&&mean_height_radius>1e-6)risk=std::min(risk,7.);
    return 1.+risk+std::clamp(comfort,0.,weight);
  }

  double topologyTerrainCostLocked(double x,double y,double yaw) const
  {
    // Absence of an execution certificate must not be CHEAPER than every
    // currently certified ground pose. Otherwise the optimistic global guide
    // steers into unseen margins instead of a known passable narrow channel.
    // This is a finite exploration preference only: unknown space still never
    // obtains a local footprint, state/path, or stopping certificate.
    const double unknown_cost=unknownTerrainGuidanceCost(comfort_clearance_weight_);
    Eigen::Array3i index;
    if(!map_.pos2Index(Eigen::Vector3d(x,y,yaw),index))return unknown_cost;
    const double valid=map_["probability_valid"][index[2]](index[0],index[1]);
    // RouteMemory already checks the continuous full body against its raw
    // fine cells. A rounded DISPLAY collision overlay must not add a second
    // fictitious wall as a very large stochastic terrain cost.
    const bool separate=map_.exists("collision_subcell_contract") &&
      map_.at("collision_subcell_contract",index)==3. && map_.exists("terrain_failure_probability_upper");
    const double p=map_[separate?"terrain_failure_probability_upper":"failure_probability_upper"][index[2]](index[0],index[1]);
    if(!std::isfinite(valid) || valid<.5 || !std::isfinite(p))return unknown_cost;
    if(p<0 || p>1)return unknown_cost;
    // An UPPER bound above budget is absence of an execution certificate,
    // not proof of an impassable surface. Keep it a finite exploration cost
    // so computational/observation frontiers cannot become fictitious walls.
    // Actual occupancy is hard; every executed prefix retains the risk limit.
    double comfort=0.;int ix=0,iy=0;
    if(worldToCostCell(x,y,ix,iy) && comfort_clearance_.size()==static_cast<size_t>(width_*height_)) {
      const double d=comfort_clearance_[static_cast<size_t>(iy*width_+ix)];
      if(std::isfinite(d) && d<comfort_clearance_distance_) {
        const double deficit=std::clamp((comfort_clearance_distance_-d)/std::max(1e-3,comfort_clearance_distance_),0.,1.);
        comfort=comfort_clearance_weight_*deficit*deficit;
      }
    }
    // The global guide used to omit the local planner's existing clearance
    // preference. Its otherwise certified prefixes repeatedly hugged a raw
    // column, where the next discretized observation consumed the remaining
    // margin. Reuse the SAME soft comfort cost, never another hard inflation.
    const double radius=map_.exists("height_error_radius_m")?map_.at("height_error_radius_m",index):
      std::numeric_limits<double>::quiet_NaN();
    return terrainGuidanceCost(p,radius,chance_state_limit_,comfort_clearance_weight_,comfort);
  }

  double topologyReverseObservationCostLocked(double x,double y,double yaw) const
  {
    if(!chance_constrained_ || topology_unknown_reverse_weight_<=0.)return 0.;
    Eigen::Array3i index;
    const double penalty=topology_unknown_reverse_weight_*(8.+std::max(0.,comfort_clearance_weight_));
    if(!map_.pos2Index(Eigen::Vector3d(x,y,yaw),index))return penalty;
    const char *field=map_.exists("terrain_failure_probability_upper")?
      "terrain_failure_probability_upper":"failure_probability_upper";
    const double valid=map_.at("probability_valid",index),p=map_.at(field,index);
    // A cheap directional ROUTING cue only. The raster is not a continuous
    // execution certificate, and predicted/unknown data never become measured.
    // A tilted LiDAR can observe ahead while advancing; backing into a still
    // unsupported side area cannot be assumed to reveal that ground.
    return std::isfinite(valid)&&valid>=.5&&std::isfinite(p)&&p>=0.&&p<=chance_state_limit_?0.:penalty;
  }

  bool topologyAdvisoryStartQualified(double x,double y,double yaw) const
  {
    const auto live=std::atomic_load(&live_snapshot_);
    if(live_obstacle_enable_ && (!live || now().seconds()-live->stamp_sec<-.05 ||
        now().seconds()-live->stamp_sec>live_obstacle_max_age_sec_ || liveBodyUnsafe(x,y,yaw,*live)))return false;
    double cost=0.;terrain_nav::ChanceBudget budget;
    return footprintCostLocked(x,y,yaw,cost) && addChancePose(x,y,yaw,budget,chance_state_limit_);
  }

  std::vector<Pose2> topologyContinuationReference(const OdomState &odom,const std::vector<Pose2> &path) const
  {
    if(path.size()<2 || path.front().direction<0)return {};
    double distance=0.;const size_t nearest=nearestPathIndex(path,odom.x,odom.y,&distance);
    if(distance>replan_track_error_ || std::abs(wrapAngle(odom.yaw-path[nearest].yaw))>.45 ||
        path.back().s-path[nearest].s<.5)return {};
    // A hazard in a hypothetical straight extension does not invalidate
    // this finite curve (e.g. a slalom entrance that must turn AFTER it).
    // Keep a FINITE preference for its still-clear portion. Search may pay
    // that cost to change sides; collision/certification never pays it away.
    std::vector<Pose2> result;
    for(size_t i=nearest;i<path.size() && path[i].s-path[nearest].s<=4.;++i) {
      double cost=0.;
      if(!footprintCostLocked(path[i].x,path[i].y,path[i].yaw,cost))break;
      result.push_back(path[i]);
    }
    return result;
  }

  static double continuationPenalty(const OdomState &odom,const std::vector<Pose2> &reference,
                                    double x,double y,double weight)
  {
    if(reference.size()<2 || weight<=0)return 0.;
    const double fade=std::clamp((6.-std::hypot(x-odom.x,y-odom.y))/2.,0.,1.);
    if(fade<=0)return 0.;
    double distance=kInf;
    for(size_t i=1;i<reference.size();++i) {
      const auto &a=reference[i-1],&b=reference[i];const double dx=b.x-a.x,dy=b.y-a.y;
      const double norm=dx*dx+dy*dy;if(norm<1e-12)continue;
      const double f=std::clamp(((x-a.x)*dx+(y-a.y)*dy)/norm,0.,1.);
      distance=std::min(distance,std::hypot(x-a.x-f*dx,y-a.y-f*dy));
    }
    return std::isfinite(distance)?weight*fade*std::min(2.,distance):0.;
  }

  double topologyContinuationPenalty(const OdomState &odom,const std::vector<Pose2> &reference,double x,double y) const
  {
    return continuationPenalty(odom,reference,x,y,guide_follow_weight_);
  }

  std::vector<std::vector<Pose2>> splitRouteGearLegs(const terrain_nav::RouteMemory::Result &route) const
  {
    if(route.path.size()<2 || route.path.size()!=route.yaws.size() || route.path.size()!=route.directions.size())return {};
    std::vector<std::vector<Pose2>> legs;
    double reverse_distance=0.;
    for(size_t i=0;i<route.path.size();++i) {
      const int gear=route.directions[i];
      if((gear!=1&&gear!=-1)||!std::isfinite(route.path[i][0])||!std::isfinite(route.path[i][1])||!std::isfinite(route.yaws[i]))return {};
      Pose2 p{route.path[i][0],route.path[i][1],route.yaws[i],0.,gear};
      if(!i){legs.push_back({p});continue;}
      auto previous=legs.back().back();
      const double ds=std::hypot(p.x-previous.x,p.y-previous.y);
      if(ds<1e-9)return {};
      reverse_distance=gear<0?reverse_distance+ds:0.;
      if(reverse_distance>reverse_max_distance_+1e-6)return {};
      if(gear!=previous.direction || previous.s+ds>std::min(4.,local_goal_distance_)) {
        if(legs.back().size()<2)return {};
        previous.direction=gear;previous.s=0.;legs.push_back({previous});
      }
      p.s=legs.back().back().s+ds;legs.back().push_back(p);
    }
    for(const auto &leg:legs)if(!bodyArcPathFeasible(leg))return {};
    return legs;
  }

  std::vector<Pose2> joinFixedGearLegLocked(const std::vector<Pose2> &leg,const OdomState &odom) const
  {
    if(leg.size()<2)return {};
    const auto &pivot=leg.front();
    if(std::hypot(odom.x-pivot.x,odom.y-pivot.y)<1e-6 &&
       std::abs(wrapAngle(odom.yaw-pivot.yaw))<1e-6)return leg;
    // Correct only a LOCAL connector. Rotating every queued leg to absorb a
    // small tracking error moved distant gates and even the final goal.
    // All original samples after the join and every later gear stay fixed.
    const bool reverse=leg.front().direction<0;const int gear=reverse?-1:1;
    Pose2 start{odom.x,odom.y,odom.yaw,0.,gear};
    const auto rear_start=terrain_nav::rearAxlePose(start,rear_axle_offset_);
    const auto live=std::atomic_load(&live_snapshot_);
    double best_score=kInf;std::vector<Pose2> best;
    for(size_t anchor=1;anchor<leg.size();++anchor) {
      if(PlanningBudget::expired())break;
      if(leg[anchor].s<.3 && anchor+1<leg.size())continue;
      if(leg[anchor].s>1.8)break;
      if(anchor+1<leg.size() && anchor%3)continue;
      const auto &target=leg[anchor];
      const auto rear_target=terrain_nav::rearAxlePose(target,rear_axle_offset_);
      const double dx=rear_target.x-rear_start.x,dy=rear_target.y-rear_start.y,distance=std::hypot(dx,dy);
      if(distance<.05 || gear*(dx*std::cos(start.yaw)+dy*std::sin(start.yaw))<=0.)continue;
      const double bearing=std::atan2(dy,dx),alpha=mod2pi(start.yaw+(reverse?M_PI:0.)-bearing);
      const double beta=mod2pi(target.yaw+(reverse?M_PI:0.)-bearing);
      for(double fraction:{.4,.7,1.}) {
        if(PlanningBudget::expired())break;
        const double radius=1./(fraction*maxReferenceCurvature()),d=distance/radius;
        std::array<DubinsCandidate,6> candidates;
        const std::array<bool,6> valid={dubinsLSL(alpha,beta,d,candidates[0]),dubinsRSR(alpha,beta,d,candidates[1]),
          dubinsLSR(alpha,beta,d,candidates[2]),dubinsRSL(alpha,beta,d,candidates[3]),
          dubinsRLR(alpha,beta,d,candidates[4]),dubinsLRL(alpha,beta,d,candidates[5])};
        for(size_t i=0;i<candidates.size();++i) {
          const double length=candidates[i].length*radius;
          if(!valid[i] || length>1.5*distance+.10)continue;
          const double score=length-target.s+.03*fraction;
          if(score>=best_score)continue;
          std::vector<Pose2> joined;
          if(!sampleDubinsLocked(start,target,candidates[i],radius,joined,reverse))continue;
          bool clear=true;
          for(const auto &p:joined) {
            bool known=false;
            if(collisionBodyUnsafe(map_,p.x,p.y,p.yaw,known) ||
               (live_obstacle_enable_ && live && liveBodyUnsafe(p.x,p.y,p.yaw,*live))){clear=false;break;}
          }
          if(!clear)continue;
          joined.insert(joined.end(),leg.begin()+anchor+1,leg.end());updatePathArcLength(joined);
          if((reverse && joined.back().s>reverse_max_distance_+1e-6) || !bodyArcPathFeasible(joined))continue;
          std::vector<Pose2> certified;
          if(!buildCertifiedGuidePrefixLocked(joined,certified,true))continue;
          best_score=score;best=std::move(joined);
        }
      }
    }
    if(!best.empty())return best;
    // This is a TRACKING REFERENCE, not an open-loop trajectory that must
    // start at exactly the measured stop pose. A tiny cusp tracking error can
    // make an exact short Dubins rejoin impossible, despite a safe reference.
    // Keep its world coordinates only inside a strict capture region, and
    // certify the actual pose, the intervening capture corridor and the
    // complete executable prefix in one account. Every actual command still
    // passes the independent measured-speed/steering braking envelope.
    if(std::hypot(odom.x-pivot.x,odom.y-pivot.y)>.05 ||
       std::abs(wrapAngle(odom.yaw-pivot.yaw))>.05 || !bodyArcPathFeasible(leg))return {};
    std::vector<Pose2> prefix;
    if(!buildCertifiedGuidePrefixLocked(leg,prefix,true))return {};
    std::vector<Pose2> capture;
    const int steps=std::max(1,int(std::ceil(std::max(std::hypot(odom.x-pivot.x,odom.y-pivot.y)/.01,
      std::abs(wrapAngle(odom.yaw-pivot.yaw))/.01))));
    for(int i=0;i<=steps;++i) {
      const double f=double(i)/steps;const Pose2 p{odom.x+f*(pivot.x-odom.x),odom.y+f*(pivot.y-odom.y),
        odom.yaw+f*wrapAngle(pivot.yaw-odom.yaw)};double cost=0.,clearance=kInf;
      if(!footprintCostLocked(p.x,p.y,p.yaw,cost,&clearance) ||
         (std::isfinite(clearance)&&clearance<hardClearanceFloor()))return {};
      capture.push_back(p);
    }
    capture.insert(capture.end(),prefix.begin(),prefix.end());
    if(!chancePathSafe(capture,nullptr,chance_path_budget_))return {};
    return leg;
  }

  // Called only at a confirmed stopped cusp, with the search map locked.
  // Future legs are geometric intentions, never cached safety certificates.
  bool activateNextStagedGearLocked(const OdomState &odom,uint64_t sequence,uint64_t epoch)
  {
    std::deque<std::vector<Pose2>> proposal;uint64_t path_sequence=0,spatial=0;int previous_gear=1;
    {
      std::lock_guard<std::mutex> guard(path_mutex_);
      if(!recovery_staged_gears_ || recovery_legs_.empty() || path_.empty())return false;
      proposal=recovery_legs_;path_sequence=path_seq_;spatial=recovery_legs_spatial_epoch_;previous_gear=path_.front().direction;
    }
    const auto live=std::atomic_load(&live_snapshot_);const double time=now().seconds();
    if(std::abs(odom.v)>.03 || spatial!=spatial_epoch_.load() || !live ||
       time-live->stamp_sec<-.05 || time-live->stamp_sec>live_obstacle_max_age_sec_ ||
       time-map_stamp_.seconds()<-.05 || time-map_stamp_.seconds()>max_map_age_sec_ ||
       time-odom.stamp.seconds()>max_odom_age_sec_) {
      RCLCPP_WARN_THROTTLE(get_logger(),*get_clock(),1000,
        "terrain staged handover rejected: freshness/epoch/stopped guard v=%.4f spatial=%llu/%llu map_age=%.3f live_age=%.3f odom_age=%.3f",
        odom.v,static_cast<unsigned long long>(spatial),static_cast<unsigned long long>(spatial_epoch_.load()),
        time-map_stamp_.seconds(),live?time-live->stamp_sec:kInf,time-odom.stamp.seconds());
      return false;
    }
    const auto pivot=proposal.front().front();
    if(std::hypot(pivot.x-odom.x,pivot.y-odom.y)>.15 || std::abs(wrapAngle(pivot.yaw-odom.yaw))>.3) {
      RCLCPP_WARN_THROTTLE(get_logger(),*get_clock(),1000,
        "terrain staged handover rejected: capture distance=%.5f heading=%.5f next_gear=%d",
        std::hypot(pivot.x-odom.x,pivot.y-odom.y),wrapAngle(pivot.yaw-odom.yaw),proposal.front().front().direction);
      return false;
    }
    const auto original_next=proposal.front();
    proposal.front()=joinFixedGearLegLocked(proposal.front(),odom);
    if(proposal.front().empty()) {
      const auto end=original_next.back();
      RCLCPP_WARN_THROTTLE(get_logger(),*get_clock(),1000,
        "terrain staged handover rejected: local join next_gear=%d length=%.5f actual=(%.9f,%.9f,%.9f) pivot=(%.9f,%.9f,%.9f) end=(%.9f,%.9f,%.9f)",
        original_next.front().direction,original_next.back().s,odom.x,odom.y,odom.yaw,
        pivot.x,pivot.y,pivot.yaw,end.x,end.y,end.yaw);
      return false;
    }
    std::vector<Pose2> certified;
    if(!buildCertifiedGuidePrefixLocked(proposal.front(),certified,true) || !bodyArcPathFeasible(certified)) {
      const auto &p=proposal.front()[std::min(certified.size(),proposal.front().size()-1)];
      const auto q=queryCostLocked(p.x,p.y,p.yaw);bool known=false;
      const bool collision=collisionBodyUnsafe(map_,p.x,p.y,p.yaw,known);
      RCLCPP_WARN_THROTTLE(get_logger(),*get_clock(),1500,
        "terrain staged continuation unavailable at (%.2f,%.2f,%.2f): support=%d upper=%.5g raw_body=%d live=%d; no current certified prefix, future intent is not permission",
        p.x,p.y,p.yaw,int(q.known),q.probability_upper,int(collision),int(liveBodyUnsafe(p.x,p.y,p.yaw,*live)));
      return false;
    }
    const bool reverse_transition=previous_gear>0&&certified.front().direction<0;
    const int next_gear=certified.front().direction;
    if(reverse_transition&&reverse_recovery_total_.load()>=16)return false;
    const bool full_leg=certified.back().s+1e-6>=proposal.front().back().s;
    if(full_leg)proposal.pop_front();
    else {
      // A future 4m leg is only an intention. Requiring all of it to be
      // observed before taking a safe first prefix stranded the car at the
      // reverse cusp. Keep its unexecuted SAME-GEAR tail, never skip the
      // unknown portion or advance to the following gear. Each prefix and
      // the measured stopping envelope still need current certificates.
      const auto &leg=proposal.front();std::vector<Pose2> tail{certified.back()};
      for(const auto &p:leg)if(p.s>certified.back().s+1e-6)tail.push_back(p);
      if(tail.size()<2)return false;
      updatePathArcLength(tail);proposal.front()=std::move(tail);
    }
    {
      std::lock_guard<std::mutex> guard(path_mutex_);
      if(!planSnapshotCurrent(sequence,epoch)||!recovery_staged_gears_||path_seq_!=path_sequence||spatial!=spatial_epoch_.load())return false;
      path_=std::move(certified);active_goal_=path_.back();have_path_=true;
      recovery_legs_=std::move(proposal);recovery_resume_pending_=false;recovery_geometry_epoch_=epoch;
      recovery_exit_active_=true;++path_seq_;
      if(reverse_transition) {
        ++reverse_recovery_total_;
        auto origin=std::make_shared<RecoveryProgressOrigin>(RecoveryProgressOrigin{odom.x,odom.y,sequence,epoch});
        recovery_progress_valid_epoch_.store(epoch);
        std::atomic_store(&reverse_progress_origin_,std::shared_ptr<const RecoveryProgressOrigin>(origin));
      }
    }
    resetNominalControls();publishPath();
    RCLCPP_INFO(get_logger(),"terrain staged gear switch: stopped, re-certified next prefix, gear=%d full_leg=%s",
      next_gear,full_leg?"true":"false");
    return true;
  }

  double certifiedTopologyProposalScoreLocked(const terrain_nav::RouteMemory::Result &candidate,const Pose2 &goal,
      bool staged_gears=false) const
  {
    if(candidate.path.size()!=candidate.yaws.size() || candidate.path.size()!=candidate.directions.size() ||
       candidate.directions.empty() || candidate.directions.front()!=1 ||
       std::any_of(candidate.directions.begin(),candidate.directions.end(),[&](int gear){return gear!=1&&(!staged_gears||gear!=-1);}))return kInf;
    if((!candidate.reached && !candidate.partial) || candidate.path.size()<2)return kInf;
    std::vector<Pose2> guide,certificate;double length=0;
    for(size_t i=0;i<candidate.path.size();++i) {
      if(candidate.directions[i]<0)break;
      if(i)length+=std::hypot(candidate.path[i][0]-candidate.path[i-1][0],candidate.path[i][1]-candidate.path[i-1][1]);
      if(length>local_goal_distance_)break;
      guide.push_back(Pose2{candidate.path[i][0],candidate.path[i][1],candidate.yaws[i],length});
    }
    if(!buildCertifiedGuidePrefixLocked(guide,certificate))return kInf;
    // Rank GLOBAL proposals by their complete frontier, after requiring
    // a currently certified local prefix. Ranking just the first 4m
    // preferred a straighter entrance into a dead end over a curved
    // proposal that already threaded several more rows of obstacles.
    double full_length=0.;
    for(size_t i=1;i<candidate.path.size();++i)full_length+=std::hypot(
      candidate.path[i][0]-candidate.path[i-1][0],candidate.path[i][1]-candidate.path[i-1][1]);
    return std::hypot(candidate.path.back()[0]-goal.x,candidate.path.back()[1]-goal.y)+
      .1*full_length+.05*std::hypot(certificate.back().x-goal.x,certificate.back().y-goal.y);
  }

  bool forwardSeedAdmittedLocked(const terrain_nav::RouteMemory::Result &route,
      const OdomState &odom,const Pose2 &goal,bool recheck_local_geometry=false) const
  {
    if(!route.reached || route.path.size()<2 || route.path.size()!=route.yaws.size() ||
       route.path.size()!=route.directions.size() ||
       std::any_of(route.directions.begin(),route.directions.end(),[](int gear){return gear!=1;}))return false;
    std::vector<Pose2> guide,certificate;double length=0.;
    const auto live=std::atomic_load(&live_snapshot_);
    for(size_t i=0;i<route.path.size();++i) {
      if(i)length+=std::hypot(route.path[i][0]-route.path[i-1][0],route.path[i][1]-route.path[i-1][1]);
      const Pose2 p{route.path[i][0],route.path[i][1],route.yaws[i],length};
      // A large UPPER bound is absence of a certificate, not a proof of an
      // impassable surface. The global objective already assigns finite
      // risk cost. Only the freshly certified prefix below can be executed;
      // its unknown/high-bound suffix remains an observation intention.
      if(length<=local_goal_distance_) {
        if(recheck_local_geometry) {
          bool known=false;
          if(collisionBodyUnsafe(map_,p.x,p.y,p.yaw,known) ||
             (live_obstacle_enable_ && live && liveBodyUnsafe(p.x,p.y,p.yaw,*live)))return false;
        }
        guide.push_back(p);
      }
    }
    if(!buildCertifiedGuidePrefixLocked(guide,certificate) ||
       certificate.back().s<std::min(.25,std::hypot(goal.x-odom.x,goal.y-odom.y)))return false;
    // Short certified prefixes are useful for observation-driven progress.
    // Only that prefix is executable; the unobserved suffix remains guidance.
    return true;
  }

  bool reuseForwardSeedLocked(const OdomState &odom,const Pose2 &goal,
                             terrain_nav::RouteMemory::Result &selected)
  {
    if(!topology_forward_seed_enable_ || !chance_constrained_ ||
       forward_seed_epoch_!=spatial_epoch_.load() ||
       forward_seed_preference_!=std::atomic_load(&recovery_route_memory_) ||
       std::hypot(goal.x-forward_seed_goal_.x,goal.y-forward_seed_goal_.y)>1e-6 ||
       std::abs(wrapAngle(goal.yaw-forward_seed_goal_.yaw))>1e-6 || forward_seed_.path.size()<2)return false;
    size_t nearest=0;double distance=kInf;
    for(size_t i=0;i<forward_seed_.path.size();++i) {
      const double d=std::hypot(odom.x-forward_seed_.path[i][0],odom.y-forward_seed_.path[i][1]);
      if(d<distance && std::abs(wrapAngle(odom.yaw-forward_seed_.yaws[i]))<=.3){distance=d;nearest=i;}
    }
    if(distance>.15 || nearest+1>=forward_seed_.path.size())return false;
    auto proposal=forward_seed_;
    proposal.path.erase(proposal.path.begin(),proposal.path.begin()+nearest);
    proposal.yaws.erase(proposal.yaws.begin(),proposal.yaws.begin()+nearest);
    proposal.directions.erase(proposal.directions.begin(),proposal.directions.begin()+nearest);
    // A changed preference snapshot forces a fresh search above. A freshly
    // scored route may still use a penalized branch; finite cost is not a ban.
    if(!forwardSeedAdmittedLocked(proposal,odom,goal,true))return false;
    proposal.expanded=0;proposal.preprocessing_ms=0.;proposal.reason="re-certified forward reference";
    selected=std::move(proposal);
    RCLCPP_INFO_THROTTLE(get_logger(),*get_clock(),1500,
      "terrain complete forward reference: current executable prefix re-certified; unknown suffix stays guidance");
    return true;
  }

  struct RecoveryRouteMemory {double x,y,stamp;uint64_t epoch;};
  struct BackgroundGuideResult {
    terrain_nav::RouteMemory::Result route;
    Pose2 goal;
    uint64_t spatial=0;
    double source_stamp=0.,elapsed_ms=0.;
    double objective=kInf;
    const char *candidate="none";
    bool geometry_reached=false,risk_reached=false;
    std::shared_ptr<const std::vector<RecoveryRouteMemory>> preference;
  };

  bool waitForCompleteGuideLocked(const OdomState &odom,const Pose2 &goal)
  {
    if(!background_guide_.valid())return false;
    const double time=now().seconds();const auto epoch=spatial_epoch_.load();
    if(global_wait_start_<0. || epoch!=global_wait_epoch_ ||
       std::hypot(goal.x-global_wait_goal_.x,goal.y-global_wait_goal_.y)>1e-6 ||
       std::abs(wrapAngle(goal.yaw-global_wait_goal_.yaw))>1e-6 ||
       std::hypot(odom.x-global_wait_origin_.x,odom.y-global_wait_origin_.y)>=.6) {
      global_wait_start_=time;global_wait_epoch_=epoch;global_wait_goal_=goal;
      global_wait_origin_={odom.x,odom.y,odom.yaw};
    }
    // ONE bounded pause at a stalled branch, not an unlimited wait that
    // prevents exploration if no complete route exists. Let the immutable
    // whole-goal job finish before moving to another unrelated local frontier.
    return time>=global_wait_start_ && time-global_wait_start_<=.8;
  }

  bool takeBackgroundManeuverLocked(const OdomState &odom,const Pose2 &goal,
      std::vector<Pose2> &prefix,std::deque<std::vector<Pose2>> &remaining,bool *forward_available=nullptr)
  {
    prefix.clear();remaining.clear();
    if(forward_available)*forward_available=false;
    if(std::abs(odom.v)<=.03 && background_guide_.valid() &&
       background_guide_.wait_for(std::chrono::milliseconds(0))==std::future_status::ready) {
      terrain_nav::RouteMemory::Result forward;
      if(takeBackgroundGuideLocked(odom,goal,forward)) {
        if(forward_available)*forward_available=true;
        return false; // hand a freshly certified forward guide back to planPath
      }
    }
    if(!background_maneuver_ || std::abs(odom.v)>.03)return false;
    auto proposal=std::move(*background_maneuver_);background_maneuver_.reset();
    const double age=now().seconds()-proposal.source_stamp;
    if(!proposal.route.reached || age<0. || age>2. || proposal.spatial!=spatial_epoch_.load() ||
       proposal.preference!=std::atomic_load(&recovery_route_memory_) ||
       std::hypot(goal.x-proposal.goal.x,goal.y-proposal.goal.y)>1e-6 ||
       std::abs(wrapAngle(goal.yaw-proposal.goal.yaw))>1e-6)return false;
    auto &route=proposal.route;
    if(route.path.size()<3 || route.yaws.size()!=route.path.size() ||
       route.directions.size()!=route.path.size() || route.directions.front()!=-1)return false;
    size_t nearest=0;double distance=kInf;
    for(size_t i=0;i<route.path.size() && route.directions[i]<0;++i) {
      const double d=std::hypot(odom.x-route.path[i][0],odom.y-route.path[i][1]);
      if(d<distance && std::abs(wrapAngle(odom.yaw-route.yaws[i]))<=.3){nearest=i;distance=d;}
    }
    if(distance>.15)return false;
    route.path.erase(route.path.begin(),route.path.begin()+nearest);
    route.yaws.erase(route.yaws.begin(),route.yaws.begin()+nearest);
    route.directions.erase(route.directions.begin(),route.directions.begin()+nearest);
    auto legs=splitRouteGearLegs(route);
    if(legs.size()<2 || legs.front().front().direction!=-1 ||
       std::none_of(legs.begin(),legs.end(),[](const auto &leg){return leg.front().direction>0;}))return false;
    legs.front()=joinFixedGearLegLocked(legs.front(),odom);
    if(legs.front().empty())return false;
    // Certify ONLY the currently executed prefix. The complete future gear
    // sequence is retained, but each stop/cusp must obtain a fresh certificate.
    if(!buildCertifiedGuidePrefixLocked(legs.front(),prefix))return false;
    if(prefix.back().s+1e-6<legs.front().back().s) {
      std::vector<Pose2> tail{prefix.back()};
      for(const auto &p:legs.front())if(p.s>prefix.back().s+1e-6)tail.push_back(p);
      if(tail.size()<2)return false;
      updatePathArcLength(tail);remaining.push_back(std::move(tail));
    }
    for(size_t i=1;i<legs.size();++i)remaining.push_back(std::move(legs[i]));
    RCLCPP_INFO(get_logger(),"terrain complete maneuver retained: first certified reverse prefix=%.2fm queued_legs=%zu source_age=%.3fs; future gears require stopped fresh certification",
      prefix.back().s,remaining.size(),age);
    return true;
  }

  static double routeGuidanceObjective(const terrain_nav::RouteMemory::Result &route,
      const std::function<double(double,double,double)> &cost)
  {
    if(!route.reached || route.path.size()<2 || route.yaws.size()!=route.path.size())return kInf;
    double total=0.,previous=cost(route.path[0][0],route.path[0][1],route.yaws[0]);
    if(!std::isfinite(previous)||previous<1.)return kInf;
    for(size_t i=1;i<route.path.size();++i) {
      const double current=cost(route.path[i][0],route.path[i][1],route.yaws[i]);
      const double ds=std::hypot(route.path[i][0]-route.path[i-1][0],route.path[i][1]-route.path[i-1][1]);
      if(!std::isfinite(current)||current<1.||!std::isfinite(ds)||ds<=0.)return kInf;
      total+=ds*(previous+current)*.5;previous=current;
    }
    return total;
  }

  static bool routeEntranceSupported(const terrain_nav::RouteMemory::Result &route,
      const se2_grid::SE2Grid &grid,double state_limit)
  {
    // Candidate scheduling only, NOT an execution certificate. A globally
    // complete forward curve with no supported first 25cm must not suppress
    // a reversible alternative. Full body/joint/braking proof stays native.
    if(!route.reached || route.path.size()<2 || route.path.size()!=route.yaws.size())return false;
    const char *field=grid.exists("terrain_failure_probability_upper")?
      "terrain_failure_probability_upper":"failure_probability_upper";
    double distance=0.;
    for(size_t i=0;i<route.path.size();++i) {
      if(i)distance+=std::hypot(route.path[i][0]-route.path[i-1][0],route.path[i][1]-route.path[i-1][1]);
      Eigen::Array3i index;
      if(!grid.pos2Index(Eigen::Vector3d(route.path[i][0],route.path[i][1],route.yaws[i]),index))return false;
      const double valid=grid.at("probability_valid",index),p=grid.at(field,index);
      if(!std::isfinite(valid)||valid<.5||!std::isfinite(p)||p<0.||p>state_limit)return false;
      if(distance>=.25)return true;
    }
    return false;
  }

  static double trackingReservePenalty(const std::vector<std::array<double,4>> &rectangles,
      const CollisionEnvelope &body,double reserve,double weight,double x,double y,double yaw)
  {
    if(reserve<=0. || weight<=0.)return 0.;
    for(const auto &b:rectangles)
      if(bodyOverlapsRectangle(x,y,yaw,body.front+reserve,body.back+reserve,body.half_width+reserve,b))return weight;
    return 0.;
  }

  bool takeBackgroundGuideLocked(const OdomState &odom,const Pose2 &goal,
      terrain_nav::RouteMemory::Result &selected)
  {
    if(!background_guide_.valid() || background_guide_.wait_for(std::chrono::milliseconds(0))!=std::future_status::ready)return false;
    BackgroundGuideResult proposal;
    try {proposal=background_guide_.get();}
    catch(const std::exception &e){RCLCPP_WARN(get_logger(),"background terrain guide failed: %s",e.what());return false;}
    if(!proposal.route.reached)RCLCPP_INFO_THROTTLE(get_logger(),*get_clock(),1500,
      "terrain background guide incomplete: reason=%s expanded=%d search=%.1fms; no execution permission",
      proposal.route.reason,proposal.route.expanded,proposal.elapsed_ms);
    const double age=now().seconds()-proposal.source_stamp;
    if(!proposal.route.reached || proposal.spatial!=spatial_epoch_.load() || age<0 || age>2. ||
       proposal.preference!=std::atomic_load(&recovery_route_memory_) ||
       std::hypot(goal.x-proposal.goal.x,goal.y-proposal.goal.y)>1e-6 ||
       std::abs(wrapAngle(goal.yaw-proposal.goal.yaw))>1e-6)return false;
    if(!proposal.route.directions.empty() && proposal.route.directions.front()<0) {
      // A reverse-first global maneuver has a DIFFERENT stopped executor.
      // Never splice it into the forward tracker or discard its later gears.
      background_maneuver_=std::move(proposal);return false;
    }
    auto previous=std::move(forward_seed_);const auto previous_goal=forward_seed_goal_;const auto previous_epoch=forward_seed_epoch_;
    const auto previous_preference=forward_seed_preference_;
    forward_seed_=std::move(proposal.route);forward_seed_goal_=goal;forward_seed_epoch_=proposal.spatial;
    forward_seed_preference_=proposal.preference;
    if(reuseForwardSeedLocked(odom,goal,selected)) {
      RCLCPP_INFO(get_logger(),"terrain background guide adopted after CURRENT prefix certification: search=%.1fms source_age=%.3fs candidate=%s objective=%.2f geometry/risk=%d/%d; guidance only",
        proposal.elapsed_ms,age,proposal.candidate,proposal.objective,int(proposal.geometry_reached),int(proposal.risk_reached));
      return true;
    }
    forward_seed_=std::move(previous);forward_seed_goal_=previous_goal;forward_seed_epoch_=previous_epoch;
    forward_seed_preference_=previous_preference;
    return false;
  }

  void queueBackgroundGuideLocked(const OdomState &odom,const Pose2 &goal,
      const std::vector<float> &lower,const terrain_nav::RouteOptions &source_options,
      std::vector<Pose2> continuity={},const terrain_nav::RouteMemory *current_geometry=nullptr)
  {
    if(!topology_async_guide_enable_ || !background_guide_runtime_ || background_guide_.valid())return;
    const double stamp=now().seconds();
    if(stamp-last_background_guide_request_<.5)return;
    last_background_guide_request_=stamp;
    // Deep immutable copies: this worker never holds map_mutex_, reads mutable
    // node state, updates occupancy/probabilities or publishes a command/path.
    auto grid=std::make_shared<const se2_grid::SE2Grid>(map_);
    auto memory=route_memory_;auto options=source_options;
    const double tracking_reserve=reference_tracking_reserve_m_,tracking_cost=reference_tracking_reserve_cost_;
    auto hard_rectangles=tracking_reserve>0.?memory.hardRectangles():std::vector<std::array<double,4>>{};
    std::optional<terrain_nav::RouteMemory> observation;
    if(current_geometry)observation=memory.currentGeometryGuidance(*current_geometry);
    const auto comfort=comfort_clearance_;const auto low=lower;
    const double minx=min_x_,miny=min_y_,res=resolution_,limit=chance_state_limit_;
    const int width=width_,height=height_;
    const double distance=comfort_clearance_distance_,weight=comfort_clearance_weight_;
    const auto penalties=std::atomic_load(&recovery_route_memory_);const auto temporal=input_epoch_.load();
    const auto spatial=spatial_epoch_.load();const double source_stamp=map_stamp_.seconds();
    const auto body=collisionEnvelope(footprint_margin_);const double curvature=maxReferenceCurvature();
    const double offset=rear_axle_offset_;
    const bool reversible=reverse_recovery_enable_&&reverse_enable_;
    const double reverse_limit=reverse_max_distance_;
    const double continuity_weight=guide_follow_weight_;
    background_guide_=std::async(std::launch::async,
      [memory=std::move(memory),observation=std::move(observation),hard_rectangles=std::move(hard_rectangles),
       tracking_reserve,tracking_cost,grid,options,comfort,low,minx,miny,res,width,height,limit,
       distance,weight,penalties,temporal,spatial,source_stamp,stamp,body,curvature,offset,reversible,reverse_limit,odom,goal,
       continuity=std::move(continuity),continuity_weight]() mutable {
        const auto began=std::chrono::steady_clock::now();
        const double unknown=unknownTerrainGuidanceCost(weight);
        const auto cell=[&](double x,double y) {
          const int c=int(std::floor((x-minx)/res)),r=int(std::floor((y-miny)/res));
          return c>=0&&r>=0&&c<width&&r<height?r*width+c:-1;
        };
        options.rear_axle_offset=offset;options.allow_partial=false;options.reverse_observation_cost={};
        options.heuristic_cost=[&](double x,double y){const int i=cell(x,y);return i<0?unknown:double(low[i]);};
        const char *field=grid->exists("terrain_failure_probability_upper")?"terrain_failure_probability_upper":"failure_probability_upper";
        const auto preference=[&](double x,double y) {
          double penalty=0.;
          if(penalties)for(const auto &p:*penalties) {
            const double age=stamp-p.stamp,d2=(x-p.x)*(x-p.x)+(y-p.y)*(y-p.y);
            if(p.epoch==temporal&&age>=0&&age<15.&&d2<.8*.8) {
              const double shape=1.-d2/(.8*.8);penalty=std::max(penalty,15.*shape*shape*(1.-age/15.));
            }
          }
          return penalty+continuationPenalty(odom,continuity,x,y,continuity_weight);
        };
        const auto cost=[&](double x,double y,double yaw) {
          Eigen::Array3i index;double value=unknown;
          if(grid->pos2Index(Eigen::Vector3d(x,y,yaw),index)) {
            const double v=grid->at("probability_valid",index),p=grid->at(field,index);
            if(std::isfinite(v)&&v>=.5&&std::isfinite(p)&&p>=0&&p<=1) {
              double extra=0.;const int i=cell(x,y);
              if(i>=0&&comfort.size()==low.size()&&std::isfinite(comfort[i])&&comfort[i]<distance) {
                const double deficit=std::clamp((distance-comfort[i])/std::max(1e-3,distance),0.,1.);
                extra=weight*deficit*deficit;
              }
              const double radius=grid->exists("height_error_radius_m")?grid->at("height_error_radius_m",index):
                std::numeric_limits<double>::quiet_NaN();
              value=terrainGuidanceCost(p,radius,limit,weight,extra);
            }
          }
          return value+preference(x,y);
        };
        BackgroundGuideResult out;out.goal=goal;out.spatial=spatial;out.source_stamp=source_stamp;
        out.preference=penalties;
        // Search diversity, one common objective and one total deadline.
        // Geometry is another candidate, never an execution certificate or
        // an authored reference. Keep risk/history/clearance in FINAL ranking.
        auto geometry_options=options;geometry_options.heuristic_cost={};geometry_options.heuristic_weight=1.5;
        auto geometry=memory.route({odom.x,odom.y},{goal.x,goal.y},odom.yaw,body.half_width+tracking_reserve,curvature,
          body.front+tracking_reserve,body.back+tracking_reserve,[&](double x,double y,double){return 1.+preference(x,y);},true,false,1,80.,geometry_options);
        const double remaining=350.-std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-began).count();
        auto risk=memory.route({odom.x,odom.y},{goal.x,goal.y},odom.yaw,body.half_width,curvature,
          body.front,body.back,cost,true,false,1,std::max(0.,remaining-(observation?120.:0.)),options);
        const auto objective=[&](double x,double y,double yaw) {
          return memory.guidanceCost({x,y},cost(x,y,yaw),options.advisory_cost)+
            trackingReservePenalty(hard_rectangles,body,tracking_reserve,tracking_cost,x,y,yaw);
        };
        const double geometric_score=routeGuidanceObjective(geometry,objective),risk_score=routeGuidanceObjective(risk,objective);
        out.geometry_reached=geometry.reached;out.risk_reached=risk.reached;
        if(geometric_score<risk_score) {
          out.route=std::move(geometry);out.objective=geometric_score;
          out.candidate=tracking_reserve>0.?"tracking_reserve_proposal":"geometry_proposal";
        } else {
          out.route=std::move(risk);out.objective=risk_score;out.candidate="risk_proposal";
        }
        // Strict retained geometry has first refusal. A failed strict search
        // is not a proof of physical impossibility: accumulated scan jitter
        // or changed scenery may disagree with the fresh local evidence.
        // Offer an observation route using CURRENT geometry plus a substantial
        // finite history prior. It cannot publish a path or clear a map cell.
        if(!out.route.reached && observation) {
          auto observed_options=options;observed_options.advisory_cost=12.;
          observed_options.heuristic_weight=2.5;
          const double left=350.-std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-began).count();
          auto proposal=observation->route({odom.x,odom.y},{goal.x,goal.y},odom.yaw,body.half_width,curvature,
            body.front,body.back,cost,true,false,1,std::max(0.,reversible?std::min(40.,left):left),observed_options);
          if(proposal.reached) {
            out.objective=routeGuidanceObjective(proposal,[&](double x,double y,double yaw){
              return observation->guidanceCost({x,y},cost(x,y,yaw),observed_options.advisory_cost);
            });
            out.route=std::move(proposal);out.candidate="current_evidence_observation";
          }
        }
        // Geometry completion and a usable current entrance are DIFFERENT
        // outcomes. In the recorded failure F reached the final goal, but its
        // first 10cm had no terrain certificate; no R candidate was even tried.
        if(observation && reversible && (!out.route.reached || !routeEntranceSupported(out.route,*grid,limit))) {
          const bool forward_complete=out.route.reached;
          auto observed_options=options;observed_options.advisory_cost=12.;
          observed_options.max_initial_reverse=reverse_limit;observed_options.heuristic_weight=4.;
          const double maneuver_time=350.-std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-began).count();
          auto maneuver=observation->route({odom.x,odom.y},{goal.x,goal.y},odom.yaw,body.half_width,curvature,
            body.front,body.back,cost,true,true,-1,std::max(0.,maneuver_time),observed_options);
          if(maneuver.reached && (!forward_complete || routeEntranceSupported(maneuver,*grid,limit))) {
            out.objective=routeGuidanceObjective(maneuver,[&](double x,double y,double yaw){
              return observation->guidanceCost({x,y},cost(x,y,yaw),observed_options.advisory_cost);
            });
            out.route=std::move(maneuver);
            out.candidate=forward_complete?"supported_reverse_entrance":"current_evidence_maneuver";
          }
        }
        out.elapsed_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-began).count();
        return out;
      });
  }

  bool completeForwardSeedLocked(const OdomState &odom,const Pose2 &goal,
                                terrain_nav::RouteMemory::Result &selected,
                                const terrain_nav::RouteOptions *weighted_options=nullptr)
  {
    if(!topology_forward_seed_enable_ || !chance_constrained_)return false;
    if(reuseForwardSeedLocked(odom,goal,selected))return true;
    // All optional seeds share ONE phase budget. Three independent 20 ms
    // attempts could exhaust the worker before weighted/reverse routing ran.
    PlanningBudget seed_budget(25.);
    const auto spatial=spatial_epoch_.load();
    const auto preference=std::atomic_load(&recovery_route_memory_);
    terrain_nav::RouteOptions options=weighted_options?*weighted_options:terrain_nav::RouteOptions{};
    options.advisory_cost=2.;
    options.rear_axle_offset=rear_axle_offset_;options.allow_partial=false;
    options.heuristic_weight=2.5;
    options.allow_advisory_start_overlap=topologyAdvisoryStartQualified(odom.x,odom.y,odom.yaw);
    // Use the same terrain/observation/clearance costs and heading-relaxed
    // heuristic as global routing. A purely geometric seed preferred unseen
    // space behind partial column returns, leading into the field's outer
    // dead ends. Comfort remains a finite COST, not another hard body margin.
    const std::vector<double> reserves=reference_tracking_reserve_m_>0.?
      std::vector<double>{reference_tracking_reserve_m_,0.}:std::vector<double>{0.};
    for(double reserve:reserves) {
      if(PlanningBudget::remaining(20.)<4.)break;
      auto route=route_memory_.route({odom.x,odom.y},{goal.x,goal.y},odom.yaw,
          std::max(footprint_half_width_,body_collision_half_width_)+footprint_margin_+reserve,
          maxReferenceCurvature(),body_collision_front_+footprint_margin_+reserve,
          body_collision_back_+footprint_margin_+reserve,
          [this](double x,double y,double yaw){return topologyTerrainCostLocked(x,y,yaw)+recoveryRoutePenalty(x,y);},true,false,1,
          PlanningBudget::remaining(reserve>0.?12.:22.),options);
      // Known unsafe terrain still calls for the risk-weighted alternatives.
      // Unseen future route cells remain optimistic guidance, never executable.
      if(!forwardSeedAdmittedLocked(route,odom,goal))continue;
      forward_seed_=route;forward_seed_epoch_=spatial;forward_seed_goal_=goal;
      forward_seed_preference_=preference;
      selected=std::move(route);
      RCLCPP_INFO(get_logger(),"terrain complete forward seed: current prefix certified tracking_reserve=%.2fm; future route is guidance only",reserve);
      return true;
    }
    return false;
  }

  bool resolveTopologyGoalLocked(const OdomState &odom,const Pose2 &raw_goal,
      Pose2 &active_goal,std::vector<Pose2> &guide_path,bool refresh_only=false,
      const std::vector<Pose2> &previous_path={},std::vector<std::vector<Pose2>> *staged_gears=nullptr)
  {
    topology_proposal_complete_=false;
    if(staged_gears)staged_gears->clear();
    if(!topology_guide_enable_ || !chance_constrained_ || !map_ready_)return false;
    std::optional<PlanningBudget> topology_phase;
    if(!refresh_only)topology_phase.emplace(std::min(40.,std::max(0.,PlanningBudget::remaining(max_search_time_ms_)-25.)));
    const bool stage_search=staged_gears&&reverse_recovery_enable_&&reverse_enable_&&reverse_recovery_total_.load()<16;
    const auto epoch=spatial_epoch_.load();
    if(route_memory_epoch_!=epoch){route_memory_.clear();route_memory_epoch_=epoch;}
    std::vector<terrain_nav::RouteMemory::Point> seen,occupied,confirmed_clear,advisory;
    std::vector<terrain_nav::RouteMemory::Square> occupied_squares,advisory_squares,clear_squares;
    const std::string observed_layer=map_.exists("terrain_observed")?"terrain_observed":"ground_observed";
    const auto centre=map_.getPosition();
    const auto scalar=[&](const char *name) {
      return map_.exists(name)?map_.getValue(name,Eigen::Vector3d(centre.x(),centre.y(),0.),
        se2_grid::InterpolationMethods::INTER_NEAREST):std::numeric_limits<double>::quiet_NaN();
    };
    const double history_age=scalar("historical_obstacle_source_age_s");
    const bool history_current=scalar("historical_obstacle_source_valid")==1. &&
      std::isfinite(history_age) && history_age>=-.05 && history_age<=1.25 &&
      map_.exists("historical_obstacle_advisory");
    for(int y=0;y<height_;++y)for(int x=0;x<width_;++x) {
      const size_t i=y*width_+x;
      const terrain_nav::RouteMemory::Point p{min_x_+(x+.5)*resolution_,min_y_+(y+.5)*resolution_};
      // Mapper history retains 3D wall returns through occlusion and deletes
      // dynamic returns using independent ray evidence. Use that geometry in
      // the GLOBAL GUIDE only. It does not alter local collision/probabilities.
      const double historical=map_.exists("historical_obstacle_advisory")?
        map_.getValue("historical_obstacle_advisory",Eigen::Vector3d(p[0],p[1],0.),
          se2_grid::InterpolationMethods::INTER_NEAREST):0.;
      if(std::isfinite(historical) && historical>=.5) {
        const double fine=resolution_/2.;
        Eigen::Array3i index;
        const int bits=map_.pos2Index(Eigen::Vector3d(p[0],p[1],0.),index)?
          historicalObstacleSubcellBits(map_,index):15;
        for(int sy=0;sy<2;++sy)for(int sx=0;sx<2;++sx)
          if(bits&(1<<(sy*2+sx)))advisory_squares.push_back({p[0]+(sx-.5)*fine,p[1]+(sy-.5)*fine,.5*fine});
      }
      if(hard_[i] && chance_constrained_) {
        Eigen::Array3i index;
        if(!map_.pos2Index(Eigen::Vector3d(p[0],p[1],0.),index))continue;
        const int bits=collisionSubcellBits(map_,index),factor=collisionSubcellFactor(map_,index);
        for(int sy=0;sy<factor;++sy)for(int sx=0;sx<factor;++sx)
          if(bits&(1<<(sy*factor+sx))) {
            const auto b=collisionSubcellRectangle(map_,index,sx,sy);
            occupied_squares.push_back({.5*(b[0]+b[2]),.5*(b[1]+b[3]),.5*(b[2]-b[0]),.5*(b[3]-b[1])});
          }
      } else if(hard_[i])occupied.push_back(p);
      else if(costs_[i]!=255) {
        // A prediction is useful for local chance-constrained execution, but
        // cannot masquerade as a measured global free-space observation.
        if(!map_.exists(observed_layer))continue;
        const double measured=map_.getValue(observed_layer,Eigen::Vector3d(p[0],p[1],0.),
            se2_grid::InterpolationMethods::INTER_NEAREST);
        if(!std::isfinite(measured) || measured<.5)continue;
        const double fine=resolution_/2.;
        for(int sy=0;sy<2;++sy)for(int sx=0;sx<2;++sx)
          seen.push_back({p[0]+(sx-.5)*fine,p[1]+(sy-.5)*fine});
      }
    }
    const auto live=std::atomic_load(&live_snapshot_);
    if(live && now().seconds()-live->stamp_sec>=-.05 && now().seconds()-live->stamp_sec<=live_obstacle_max_age_sec_ &&
       live->uninflated_present.size()==static_cast<size_t>(live->width*live->height)) {
      for(int y=0;y<live->height;++y)for(int x=0;x<live->width;++x)
        if(live->uninflated_present[y*live->width+x]) {
          const auto b=liveCellRectangle(*live,x,y);
          occupied_squares.push_back({.5*(b[0]+b[2]),.5*(b[1]+b[3]),.5*(b[2]-b[0]),.5*(b[3]-b[1])});
        }
      if(history_current && live->routing_clear.size()==live->uninflated_present.size())
        for(int y=0;y<live->height;++y)for(int x=0;x<live->width;++x)
          if(live->routing_clear[y*live->width+x])clear_squares.push_back(
            {live->min_x+(x+.5)*live->resolution,live->min_y+(y+.5)*live->resolution,.5*live->resolution});
    }
    route_memory_.observe(map_stamp_.seconds(),{min_x_,min_y_,max_x_,max_y_},seen,occupied,confirmed_clear,
      advisory,history_current,occupied_squares,advisory_squares,clear_squares);
    const auto memory_geometry=route_memory_.geometryStats();
    RCLCPP_INFO_THROTTLE(get_logger(),*get_clock(),5000,
      "terrain routing geometry: hard_cells=%zu full_cells=%zu source_rectangles=%zu bounded_raster_cells=%zu; historical union is not a bounding hull",
      memory_geometry.hard,memory_geometry.whole,memory_geometry.rectangles,memory_geometry.raster);
    if(refresh_only || PlanningBudget::expired())return false;
    // The older 2D guide already preferred the current bypass. The car-arc
    // guide did not: each rolling-window update could choose a different
    // side of the same column. Use a FINITE local reference cost, never an
    // obstacle exemption or an old probability certificate. A blocked old
    // corridor remains blocked and any other homotopy remains searchable.
    const auto continuity=topologyContinuationReference(odom,previous_path);
    const auto terrain_cost=[&](double x,double y,double yaw) {
      return topologyTerrainCostLocked(x,y,yaw)+topologyContinuationPenalty(odom,continuity,x,y)+recoveryRoutePenalty(x,y);
    };
    terrain_nav::RouteOptions routing;routing.rear_axle_offset=rear_axle_offset_;routing.heuristic_weight=2.5;routing.allow_partial=true;routing.advisory_cost=2.;
    routing.reverse_observation_cost=[this](double x,double y,double yaw){return topologyReverseObservationCostLocked(x,y,yaw);};
    // A constant 1/3 heuristic is far below the actual conditional terrain
    // and comfort costs (up to 30+). Under a short deadline this expanded
    // cheap dead-end frontiers instead of a car route around the next row.
    // Minimize over ALL heading slices, round down for stable cache keys,
    // and leave the finite continuity preference out of this lower estimate.
    const double unknown=unknownTerrainGuidanceCost(comfort_clearance_weight_);
    std::vector<float> lower_cost(width_*height_,unknown);
    const bool separated=map_.exists("terrain_failure_probability_upper") &&
      scalar("collision_subcell_contract")==3.;
    const auto &probability=map_[separated?"terrain_failure_probability_upper":"failure_probability_upper"];
    const auto &valid=map_["probability_valid"];
    for(int y=0;y<height_;++y)for(int x=0;x<width_;++x) {
      const size_t cell=y*width_+x;Eigen::Array3i index;
      if(!map_.pos2Index(Eigen::Vector3d(min_x_+(x+.5)*resolution_,min_y_+(y+.5)*resolution_,0.),index))continue;
      double comfort=0.;
      if(comfort_clearance_.size()==lower_cost.size()) {
        const double d=comfort_clearance_[cell];
        if(std::isfinite(d) && d<comfort_clearance_distance_) {
          const double deficit=std::clamp((comfort_clearance_distance_-d)/std::max(1e-3,comfort_clearance_distance_),0.,1.);
          comfort=comfort_clearance_weight_*deficit*deficit;
        }
      }
      double cost=unknown;
      for(int k=0;k<map_.getSizeYaw();++k) {
        const double v=valid[k](index[0],index[1]),p=probability[k](index[0],index[1]);
        if(std::isfinite(v) && v>=.5 && std::isfinite(p) && p>=0 && p<=1) {
          const double radius=map_.exists("height_error_radius_m")?map_.at("height_error_radius_m",Eigen::Array3i(index[0],index[1],k)):
            std::numeric_limits<double>::quiet_NaN();
          cost=std::min(cost,terrainGuidanceCost(p,radius,chance_state_limit_,comfort_clearance_weight_,comfort));
        }
      }
      lower_cost[cell]=std::max(1.,std::floor(cost*4.)/4.);
    }
    routing.heuristic_cost=[&](double x,double y) {
      int ix=0,iy=0;
      return worldToCostCell(x,y,ix,iy)?double(lower_cost[iy*width_+ix]):unknown;
    };
    routing.allow_advisory_start_overlap=topologyAdvisoryStartQualified(odom.x,odom.y,odom.yaw);
    // The ordinary executor consumes one FORWARD local prefix. Searching a
    // multi-gear route here and discarding its reverse suffix at the loop
    // below sent the car to a cusp it could not execute. Explicit reverse
    // goals and stopped, jointly certified recovery retain their own paths.
    terrain_nav::RouteMemory::Result route;
    const bool background=takeBackgroundGuideLocked(odom,raw_goal,route);
    // Construct only for a queued background job. This is a proposal model,
    // never the local collision map, controller snapshot or terrain posterior.
    std::optional<terrain_nav::RouteMemory> current_geometry;
    if(topology_async_guide_enable_ && background_guide_runtime_ && !background_guide_.valid() &&
       now().seconds()-last_background_guide_request_>=.5) {
      current_geometry.emplace(route_memory_.resolution());
      current_geometry->observe(map_stamp_.seconds(),{min_x_,min_y_,max_x_,max_y_},seen,occupied,
        confirmed_clear,advisory,false,occupied_squares,advisory_squares,{});
    }
    queueBackgroundGuideLocked(odom,raw_goal,lower_cost,routing,continuity,
      current_geometry?&*current_geometry:nullptr);
    if(background_maneuver_) {
      const auto &pending=*background_maneuver_;
      const double age=now().seconds()-pending.source_stamp;
      if(pending.spatial==spatial_epoch_.load() && age>=0. && age<=2. &&
         pending.preference==std::atomic_load(&recovery_route_memory_) &&
         std::hypot(raw_goal.x-pending.goal.x,raw_goal.y-pending.goal.y)<1e-6 &&
         std::abs(wrapAngle(raw_goal.yaw-pending.goal.yaw))<1e-6) {
        guide_handoff_=GuideHandoff::ReverseReady;return false;
      }
      background_maneuver_.reset();
    }
    const bool complete_seed=background || completeForwardSeedLocked(odom,raw_goal,route,&routing);
    if(!complete_seed)route=route_memory_.route({odom.x,odom.y},{raw_goal.x,raw_goal.y},odom.yaw,
        std::max(footprint_half_width_,body_collision_half_width_)+footprint_margin_,maxReferenceCurvature(),
        body_collision_front_+footprint_margin_,body_collision_back_+footprint_margin_,
        terrain_cost,false,stage_search,1,PlanningBudget::remaining(24.),routing);
    if(!complete_seed && !route.reached && !PlanningBudget::expired()) {
      // A coarse frontier is not evidence that the narrow route search is
      // finished. Previously ANY coarse partial result suppressed the finer
      // car primitives, sending the car around the outside of a pillar array.
      auto refined=route_memory_.route({odom.x,odom.y},{raw_goal.x,raw_goal.y},odom.yaw,
          std::max(footprint_half_width_,body_collision_half_width_)+footprint_margin_,maxReferenceCurvature(),
          body_collision_front_+footprint_margin_,body_collision_back_+footprint_margin_,
          terrain_cost,true,stage_search,1,
          PlanningBudget::remaining(route.partial?12.:24.),routing);
      const auto progress_score=[&](const terrain_nav::RouteMemory::Result &candidate) {
        return certifiedTopologyProposalScoreLocked(candidate,raw_goal,stage_search);
      };
      const double old_score=progress_score(route),new_score=progress_score(refined);
      if((!route.reached && !route.partial) ||
         (std::isfinite(new_score) && (refined.reached || new_score+.05<old_score)))route=std::move(refined);
    }
    publishTopologyDiagnostic(route);
    topology_proposal_complete_=route.reached;
    if(!route.reached && waitForCompleteGuideLocked(odom,raw_goal)) {
      guide_handoff_=GuideHandoff::Computing;
      RCLCPP_INFO_THROTTLE(get_logger(),*get_clock(),500,
        "terrain bounded global-route pause: partial frontier deferred while complete maneuver is computed; no local-goal branch switch");
      return false;
    }
    if((!route.reached && !route.partial) || route.path.size()<2) {
      RCLCPP_INFO_THROTTLE(get_logger(),*get_clock(),1500,
        "terrain topology unavailable: %s expanded=%d prepare=%.1fms heuristic_cache=%s; native certified local planning retained",
        route.reason,route.expanded,route.preprocessing_ms,
        route.quantized_heuristic_reused?"reused_quantized":route.heuristic_reused?"reused":"rebuilt");
      return false;
    }
    // A retained reference starts at its nearest existing curve sample. Do
    // not invent a lateral splice from the current pose into that car arc.
    std::vector<Pose2> prefix;prefix.push_back(Pose2{route.path.front()[0],route.path.front()[1],route.yaws.front(),0.});
    size_t selected=0;double travel=0;
    for(size_t i=1;i<route.path.size();++i) {
      // Look ahead through future gear changes, but execute only the first
      // forward leg here. A reverse leg needs its own current certificate and
      // the existing stop/gear-change control procedure.
      if(route.directions[i]<0)break;
      travel+=std::hypot(route.path[i][0]-route.path[i-1][0],route.path[i][1]-route.path[i-1][1]);
      if(travel>local_goal_distance_)break;
      const double yaw=route.yaws[i];
      const Pose2 candidate{route.path[i][0],route.path[i][1],yaw,travel};prefix.push_back(candidate);
      double cost=0;
      // Only select a currently supported endpoint. The optimistic route is
      // an informational guide, NOT an executable trajectory: continuous/
      // lattice paths still undergo full SE(2) footprint, kinematics, joint
      // chance budget and per-tick braking checks without any exemptions.
      if(travel>=.25 && footprintCostLocked(candidate.x,candidate.y,candidate.yaw,cost))selected=prefix.size()-1;
    }
    if(!selected) {
      RCLCPP_INFO_THROTTLE(get_logger(),*get_clock(),1500,
        "terrain topology has no supported local endpoint; unknown route is not executable");
      return false;
    }
    prefix.resize(selected+1);guide_path=std::move(prefix);active_goal=guide_path.back();
    if(stage_search) {
      const auto cusp=std::find(route.directions.begin(),route.directions.end(),-1);
      if(cusp!=route.directions.end() && cusp!=route.directions.begin()) {
        const size_t endpoint=std::distance(route.directions.begin(),cusp)-1;
        if(std::hypot(guide_path.back().x-route.path[endpoint][0],guide_path.back().y-route.path[endpoint][1])<1e-6) {
          auto legs=splitRouteGearLegs(route);
          if(legs.size()>1 && legs.front().front().direction==1 &&
             std::hypot(guide_path.back().x-legs.front().back().x,guide_path.back().y-legs.front().back().y)<1e-6)
            *staged_gears=std::move(legs);
        }
      }
    }
    RCLCPP_INFO_THROTTLE(get_logger(),*get_clock(),1200,
      "terrain topology route: memory=%zu expanded=%d mode=%s status=%s local=(%.2f, %.2f) final=(%.2f, %.2f) prepare=%.1fms heuristic_cache=%s; unknown routing never authorizes motion",
      route_memory_.size(),route.expanded,route.fine?"fine":"coarse",route.reached?"complete":"partial",active_goal.x,active_goal.y,raw_goal.x,raw_goal.y,
      route.preprocessing_ms,route.quantized_heuristic_reused?"reused_quantized":route.heuristic_reused?"reused":"rebuilt");
    return true;
  }

  // Optional observation of the actual global routing state. A diagnostic
  // route includes unknown terrain and must never be sent to a controller.
  void publishTopologyDiagnostic(const terrain_nav::RouteMemory::Result &route)
  {
    if(topology_memory_pub_->get_subscription_count()==0 && topology_route_pub_->get_subscription_count()==0)return;
    const double stamp=map_stamp_.seconds();
    if(stamp>=last_topology_diagnostic_stamp_ && stamp-last_topology_diagnostic_stamp_<1.)return;
    last_topology_diagnostic_stamp_=stamp;
    nav_msgs::msg::Path path;path.header.frame_id=frame_id_;path.header.stamp=map_stamp_;
    for(size_t i=0;i<route.path.size();++i) {
      geometry_msgs::msg::PoseStamped pose;pose.header=path.header;
      pose.pose.position.x=route.path[i][0];pose.pose.position.y=route.path[i][1];
      pose.pose.orientation.z=std::sin(route.yaws[i]/2);pose.pose.orientation.w=std::cos(route.yaws[i]/2);
      path.poses.push_back(pose);
    }
    topology_route_pub_->publish(path);
    if(topology_memory_pub_->get_subscription_count()==0)return;
    const auto evidence=route_memory_.evidence();if(evidence.empty())return;
    const double res=route_memory_.resolution();
    int x0=INT_MAX,y0=INT_MAX,x1=INT_MIN,y1=INT_MIN;
    for(const auto &cell:evidence) {
      const int x=int(std::floor(cell.position[0]/res)),y=int(std::floor(cell.position[1]/res));
      x0=std::min(x0,x);x1=std::max(x1,x);y0=std::min(y0,y);y1=std::max(y1,y);
    }
    const int width=x1-x0+1,height=y1-y0+1;
    if(width<=0 || height<=0 || width>1024 || height>1024)return;
    nav_msgs::msg::OccupancyGrid grid;grid.header=path.header;
    grid.info.resolution=res;grid.info.width=width;grid.info.height=height;
    grid.info.origin.position.x=x0*res;grid.info.origin.position.y=y0*res;grid.info.origin.orientation.w=1;
    grid.data.assign(width*height,-1);
    for(const auto &cell:evidence) {
      const int x=int(std::floor(cell.position[0]/res))-x0,y=int(std::floor(cell.position[1]/res))-y0;
      grid.data[y*width+x]=cell.occupied?100:cell.advisory?75:0;
    }
    topology_memory_pub_->publish(grid);
  }

  bool buildCertifiedGuidePrefixLocked(const std::vector<Pose2> &guide,std::vector<Pose2> &prefix,
      bool allow_short_complete=false) const
  {
    prefix.clear();
    if(!chance_constrained_ || guide.size()<2)return false;
    terrain_nav::ChanceBudget budget;double cost=0,clearance=kInf;
    if(!footprintCostLocked(guide.front().x,guide.front().y,guide.front().yaw,cost,&clearance) ||
       (std::isfinite(clearance) && clearance<hardClearanceFloor()) ||
       !addChancePose(guide.front().x,guide.front().y,guide.front().yaw,budget,chance_path_budget_))return false;
    prefix.push_back(guide.front());
    for(size_t i=1;i<guide.size();++i) {
      const auto &a=guide[i-1],&b=guide[i];
      const double length=std::hypot(b.x-a.x,b.y-a.y),turn=wrapAngle(b.yaw-a.yaw);
      if(prefix.back().s+length>std::min(4.,local_goal_distance_))break;
      const int steps=std::max(1,static_cast<int>(std::ceil(std::max(length/.025,
          std::abs(turn)/(.25*map_.getResolutionYaw())))));
      bool safe=true;
      for(int step=1;step<=steps;++step) {
        const double f=double(step)/steps,x=a.x+f*(b.x-a.x),y=a.y+f*(b.y-a.y),yaw=a.yaw+f*turn;
        if(!footprintCostLocked(x,y,yaw,cost,&clearance) ||
           (std::isfinite(clearance) && clearance<hardClearanceFloor()) ||
           !addChancePose(x,y,yaw,budget,chance_path_budget_)){
          safe=false;break;
        }
      }
      // Never jump over an unsupported section to a safe-looking endpoint.
      if(!safe)break;
      auto p=b;p.s=prefix.back().s+length;prefix.push_back(p);
    }
    // 25cm is a useful-observation-prefix requirement, not a prohibition on
    // finishing a fully certified 10cm remainder at a real gear cusp.
    const bool long_enough=prefix.back().s>=.25 || (allow_short_complete && prefix.back().s>.025 &&
      prefix.size()==guide.size());
    return long_enough && bodyArcPathFeasible(prefix) &&
      pathFootprintSafeLocked(prefix,.025) && chancePathSafe(prefix,nullptr,chance_path_budget_);
  }

  bool resolveGuideGoalLocked(
    const OdomState &odom,
    const Pose2 &raw_goal,
    Pose2 &active_goal,
    std::string &reason,
    std::vector<Pose2> &guide_path,
    const std::vector<Pose2> &previous_path = {})
  {
    if (!guide_search_enable_ || !map_ready_ || width_ <= 0 || height_ <= 0) return false;
    PlanningBudget guide_budget(std::min(20.,std::max(0.,PlanningBudget::remaining(max_search_time_ms_)-15.)));
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
    double preferred_lateral = 0.0;
    if (previous_path.size() >= 2 && previous_path.front().direction > 0) {
      double tracking_error = kInf;
      const size_t nearest = nearestPathIndex(previous_path, odom.x, odom.y, &tracking_error);
      if (tracking_error <= replan_track_error_) {
        const auto anchor = samplePathAt(previous_path,
          std::min(previous_path.back().s, previous_path[nearest].s + 2.0));
        preferred_lateral = -(anchor.x - odom.x) * uy + (anchor.y - odom.y) * ux;
      }
    }
    const auto continuityPenalty = [&](double vx, double vy) {
      const double lateral = -vx * uy + vy * ux;
      return std::abs(preferred_lateral) > 0.25
        ? guide_follow_weight_ * std::min(2.0,
            std::max(0.0, -std::copysign(1.0, preferred_lateral) * lateral))
        : 0.0;
    };
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

      double yaw = std::atan2(vy, vx);
      const int predecessor = parent[static_cast<size_t>(idx)];
      if (predecessor >= 0) {
        double px = 0.0, py = 0.0;
        cellCenter(predecessor, px, py);
        yaw = std::atan2(wy - py, wx - px);
      }
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
      // A soft preference, never a certificate: alternate sides remain
      // eligible when new observations invalidate the previous bypass.
      const double continuity_penalty = continuityPenalty(vx, vy);

      const double score =
        progress_weight * progress -
        guide_goal_weight_ * raw_goal_dist -
        lateral_weight * lateral -
        guide_path_cost_weight_ * path_cost -
        heading_weight * heading_gap * heading_gap -
        lateral_excess_weight * lateral_excess * lateral_excess -
        0.85 * fc +
        guide_clearance_weight_ * (0.65 * comfort_bonus + 0.35 * hard_clearance_bonus) +
        frontier_bonus - continuity_penalty -
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
      if (PlanningBudget::expired() || std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - planning_start).count()
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
          step * (1.0 + 0.42 * soft_cost + continuityPenalty(vx, vy)) +
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
    double final_cost = 0.0;
    if (!footprintCostLocked(active_goal.x, active_goal.y, active_goal.yaw, final_cost)) {
      guide_path.clear();
      return false;
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
    const Pose2 &raw_goal,
    Pose2 &active_goal,
    std::string &reason,
    std::vector<Pose2> &guide_path,
    const std::vector<Pose2> &previous_path = {},bool may_retain=false,
    std::vector<std::vector<Pose2>> *staged_gears=nullptr)
  {
    PlanningBudget goal_budget(std::min(55.,max_search_time_ms_));
    // The worker already captured this goal with its sequence number. Guide
    // search can take longer than a control tick; it must not hold goal_mutex_.
    guide_path.clear();

    active_goal = raw_goal;
    const double dx = raw_goal.x - odom.x;
    const double dy = raw_goal.y - odom.y;
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
    active_goal.yaw = (dist > 1.2) ? local_dir : raw_goal.yaw;

    if (!map_ready_ || width_ <= 0 || height_ <= 0) {
      const double d = std::min(dist, local_goal_distance_);
      active_goal.x = odom.x + d * std::cos(local_dir);
      active_goal.y = odom.y + d * std::sin(local_dir);
      reason = "map not ready, use bounded raw direction";
      return true;
    }

    double dummy = 0.0;
    Pose2 raw_start{odom.x, odom.y, raw_dir, 0.0};
    Pose2 raw_target{raw_goal.x, raw_goal.y, raw_dir, 0.0};
    if (dist <= local_goal_distance_ &&
        direct_bearing_gap <= direct_goal_bearing_limit_ &&
        raw_goal.x >= min_x_ && raw_goal.x <= max_x_ &&
        raw_goal.y >= min_y_ && raw_goal.y <= max_y_ &&
        footprintCostLocked(raw_goal.x, raw_goal.y, raw_dir, dummy) &&
        straightSegmentSafeLocked(raw_start, raw_target, 0.18)) {
      active_goal = raw_goal;
      active_goal.yaw = (dist > 1.2) ? raw_dir : raw_goal.yaw;
      guide_path = {Pose2{odom.x, odom.y, raw_dir, 0.0}, active_goal};
      updatePathArcLength(guide_path);
      reason = "raw goal inside terrain window";
      return true;
    }

    Pose2 guide_goal;
    std::string guide_reason;
    // Revalidate before replacing a still useful trajectory with another
    // optimistic homotopy. Map updates used to send the car left/right around
    // the same pole before it could align with the narrow opening. Actual
    // obstacles, missing support, changed goals and risk failure still revoke
    // retention; live/braking checks continue independently every control tick.
    const bool current_certificate=may_retain &&
      canRetainCertifiedForwardPathLocked(odom,previous_path,false);
    // A hypothetical tangent beyond the finite endpoint is a SEARCH hint,
    // never a collision witness on that finite trajectory. Search the next
    // turn early, while keeping the presently certified curve available.
    const bool continuation_search=current_certificate &&
      continuationSearchUsefulLocked(odom,previous_path);
    if(current_certificate && !continuation_search && !fresh_background_guide_ && !background_maneuver_) {
      active_goal=previous_path.back();guide_path=previous_path;
      reason="retain certified current path";return true;
    }
    const bool topology_resolved=resolveTopologyGoalLocked(odom,raw_goal,guide_goal,guide_path,false,
      may_retain?previous_path:std::vector<Pose2>{},staged_gears);
    if(current_certificate && continuation_search && background_guide_.valid() && !background_maneuver_ &&
       (!topology_resolved || !topology_proposal_complete_)) {
      // A failed/incomplete future search cannot revoke a current proof or
      // send the car to an unrelated 2-D side frontier. A complete new route
      // can still replace this curve; ordinary finite-prefix stopping and
      // current body/support/joint/braking guards remain in force.
      active_goal=previous_path.back();guide_path=previous_path;
      if(staged_gears)staged_gears->clear();
      guide_handoff_=GuideHandoff::None;
      reason="retain certified current path";
      RCLCPP_INFO_THROTTLE(get_logger(),*get_clock(),1500,
        "terrain continuation search: tangent hazard is beyond finite certified path; retain current curve while searching its next turn");
      return true;
    }
    if(may_retain && !background_maneuver_ && !topology_proposal_complete_) {
      std::vector<Pose2> safe_prefix;
      if(certifiedRemainingForwardPrefixLocked(odom,previous_path,safe_prefix) &&
         (!topology_resolved || std::hypot(raw_goal.x-safe_prefix.back().x,raw_goal.y-safe_prefix.back().y)<=
           std::hypot(raw_goal.x-guide_goal.x,raw_goal.y-guide_goal.y)+.2)) {
        active_goal=safe_prefix.back();guide_path=std::move(safe_prefix);
        guide_handoff_=GuideHandoff::None;
        if(staged_gears)staged_gears->clear();
        reason="retain certified current prefix";return true;
      }
    }
    if(topology_resolved) {
      active_goal=guide_goal;reason="persistent topology route (local certificate required)";return true;
    }
    if(guide_handoff_!=GuideHandoff::None){reason="waiting for complete global maneuver";return false;}
    if (resolveGuideGoalLocked(odom, raw_goal, guide_goal, guide_reason, guide_path, previous_path)) {
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
        const double raw_goal_cost = 0.04 * std::hypot(raw_goal.x - x, raw_goal.y - y);
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
    double probability_upper = 0.0;
    std::shared_ptr<const terrain_nav::ChanceBudget> probability_events;
  };

  struct PrimitiveResult
  {
    bool ok = false;
    Pose2 end;
    double terrain_cost = 0.0;
    double comfort_cost = 0.0;
    double min_clearance = kInf;
    double min_comfort_clearance = kInf;
    double probability_upper = 0.0;
    std::shared_ptr<const terrain_nav::ChanceBudget> probability_events;
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
    const int steps = primitiveIntegrationSteps();
    const double ds = primitive_length_ / static_cast<double>(steps);
    const double curvature = primitiveCurvature(steer);
    terrain_nav::ChanceBudget probability;

    if (node.probability_events) probability=*node.probability_events;
    if (!addChancePose(x,y,yaw,probability,chance_path_budget_)) return result;

    for (int i = 0; i < steps; ++i) {
      const auto next=advanceArc(Pose2{x,y,yaw,0.},curvature,ds);
      x=next.x;y=next.y;yaw=next.yaw;

      double fc = 0.0;
      double clearance = 0.0;
      double comfort_clearance = 0.0;
      if (!footprintCostLocked(x, y, yaw, fc, &clearance, &comfort_clearance)) return result;
      if(std::isfinite(clearance) && clearance<hardClearanceFloor())return result;
      if (!addChancePose(x,y,yaw,probability,chance_path_budget_)) return result;
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
    result.probability_upper = chance_constrained_ ? probability.upper() : 0.0;
    if (chance_constrained_) result.probability_events=std::make_shared<terrain_nav::ChanceBudget>(std::move(probability));
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
    if (!path_commit_enable_ || old_path.size() < 4 || old_path.front().direction<0 || path_commit_distance_ <= 1e-3) {
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
    if(!chance_constrained_) updateYawFromTangent(prefix);
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
    if(!chance_constrained_) updateYawFromTangent(merged);
    updatePathArcLength(merged);
    return merged;
  }

  void requestPlan(const std::string &reason)
  {
    if (stale_inputs_waiting_.load(std::memory_order_acquire) || localization_fault_.load(std::memory_order_acquire)) return;
    {
      std::lock_guard<std::mutex> lock(planner_mutex_);
      planner_reason_ = reason;
      planner_wakeup_ = true;
    }
    plan_requested_.store(true, std::memory_order_release);
    planner_cv_.notify_one();
  }

  bool planSnapshotCurrent(uint64_t sequence, uint64_t input_epoch)
  {
    std::lock_guard<std::mutex> lock(goal_mutex_);
    return have_goal_ && goal_seq_ == sequence &&
      !reserve_retreat_active_.load(std::memory_order_acquire) &&
      !localization_fault_.load(std::memory_order_acquire) &&
      !stale_inputs_waiting_.load(std::memory_order_acquire) &&
      input_epoch == input_epoch_.load(std::memory_order_acquire);
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

      executePlannerJob(reason);
    }
  }

  bool consumeRecoverySearchTurn(uint64_t sequence,uint64_t epoch)
  {
    const auto pending=std::atomic_exchange(&pending_recovery_search_,std::shared_ptr<const PendingRecoverySearch>{});
    return pending && pending->sequence==sequence && pending->epoch==epoch;
  }

  void deferRecoverySearch(uint64_t sequence,uint64_t epoch)
  {
    if(!reverse_recovery_enable_ || !planSnapshotCurrent(sequence,epoch))return;
    auto pending=std::make_shared<PendingRecoverySearch>(PendingRecoverySearch{sequence,epoch});
    std::atomic_store(&pending_recovery_search_,std::shared_ptr<const PendingRecoverySearch>(pending));
  }

  void dispatchReadyReverseHandoff(uint64_t sequence,uint64_t epoch,bool &ok,bool &tried_recovery)
  {
    if(guide_handoff_!=GuideHandoff::ReverseReady || tried_recovery || PlanningBudget::expired())return;
    // Retaining a finite stopping prefix is a valid safety result, but not
    // evidence of forward mission progress. It must not starve a completed
    // reverse-first whole-goal proposal forever. Recovery still checks actual
    // near-zero speed, a useful existing forward curve, current support/body,
    // live/map age, goal/epoch identity and the new leg's joint certificate.
    tried_recovery=true;
    if(planReverseRecovery(sequence,epoch))ok=true;
  }

  bool executePlannerJob(const std::string &reason)
  {
      if (!haveGoalThreadSafe() || !map_ready_ || stale_inputs_waiting_.load() || localization_fault_.load()) {
        plan_requested_.store(false, std::memory_order_release);
        return false;
      }

      planner_running_.store(true, std::memory_order_release);
      const auto start = std::chrono::steady_clock::now();
      PlanningBudget job_budget(max_search_time_ms_);
      uint64_t sequence=0;
      {std::lock_guard<std::mutex> guard(goal_mutex_);sequence=goal_seq_;}
      const auto epoch=input_epoch_.load();
      const bool recovery_first=consumeRecoverySearchTurn(sequence,epoch);
      const uint64_t map_seq_before = map_update_seq_.load(std::memory_order_acquire);
      plan_requested_.store(false, std::memory_order_release);
      bool ok=false,tried_forward=false,tried_recovery=false;
      if(recovery_first && !PlanningBudget::expired()) {
        tried_recovery=true;ok=planReverseRecovery(sequence,epoch);
      }
      if(!ok && !PlanningBudget::expired()) {tried_forward=true;ok=planPath(reason);}
      dispatchReadyReverseHandoff(sequence,epoch,ok,tried_recovery);
      if(!ok && guide_handoff_!=GuideHandoff::Computing && !tried_recovery && !PlanningBudget::expired()) {
        tried_recovery=true;ok=planReverseRecovery(sequence,epoch);
      }
      if(!ok && tried_forward && guide_handoff_!=GuideHandoff::Computing)deferRecoverySearch(sequence,epoch);
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

      RCLCPP_INFO(
        get_logger(),
        "terrain_nav planner: result=%s reason=%s elapsed=%.1fms map_seq=%llu first=%s phases=%d/%d handoff=%s",
        ok ? "success" : "failed", reason.c_str(), elapsed_ms,
        static_cast<unsigned long long>(map_update_seq_.load(std::memory_order_acquire)),
        recovery_first?"recovery":"forward",int(tried_forward),int(tried_recovery),
        guide_handoff_==GuideHandoff::Computing?"computing":guide_handoff_==GuideHandoff::ReverseReady?"reverse_ready":"none");
      return ok;
  }

  bool buildRecoveryManeuverLocked(const Pose2 &start,const Pose2 &goal,
                                  std::vector<Pose2> &retreat,
                                  std::vector<Pose2> *forward_exit=nullptr) const
  {
    PlanningBudget recovery_budget(max_search_time_ms_);
    retreat.clear();
    if(forward_exit)forward_exit->clear();
    const double dx=goal.x-start.x,dy=goal.y-start.y,distance=std::hypot(dx,dy);
    if(!reverse_enable_ || distance<1. || dx*std::cos(start.yaw)+dy*std::sin(start.yaw)<=0)
      return false;
    const double ux=dx/distance,uy=dy/distance,step=.025;
    double best_score=kInf;
    const auto deadline=std::chrono::steady_clock::now()+
      std::chrono::duration<double,std::milli>(PlanningBudget::remaining(max_search_time_ms_));
    for(double steering:steerSamples()) {
      if(std::chrono::steady_clock::now()>=deadline)break;
      std::vector<Pose2> reverse={Pose2{start.x,start.y,start.yaw,0.,-1}};
      double reverse_route_cost=0.;
      terrain_nav::ChanceBudget reverse_budget;
      if(!addChancePose(start.x,start.y,start.yaw,reverse_budget,chance_path_budget_))continue;
      for(int i=1;i<=32;++i) {
        auto p=advanceArc(reverse.back(),primitiveCurvature(steering),-step);
        p.direction=-1;p.s=i*step;
        double cost=0,clearance=kInf;
        if(!footprintCostLocked(p.x,p.y,p.yaw,cost,&clearance) ||
           (std::isfinite(clearance)&&clearance<hardClearanceFloor()) ||
           !addChancePose(p.x,p.y,p.yaw,reverse_budget,chance_path_budget_))break;
        reverse.push_back(p);
        reverse_route_cost+=cost*step;
        if(i<8 || i%4)continue;
        // A retreat alone is not progress. Certify a forward exit with the
        // SAME source-event account before offering this reverse-only leg.
        for(double exit_steering:steerSamples()) {
          if(std::chrono::steady_clock::now()>=deadline)break;
          auto exit=p;auto budget=reverse_budget;double exit_route_cost=0.;
          exit.direction=1;exit.s=0;
          std::vector<Pose2> exit_path={exit};
          for(int j=1;j<=64;++j) {
            exit=advanceArc(exit,primitiveCurvature(exit_steering),step);
            exit.s=j*step;
            double exit_cost=0,exit_clearance=kInf;
            if(!footprintCostLocked(exit.x,exit.y,exit.yaw,exit_cost,&exit_clearance) ||
               (std::isfinite(exit_clearance)&&exit_clearance<hardClearanceFloor()) ||
               !addChancePose(exit.x,exit.y,exit.yaw,budget,chance_path_budget_))break;
            exit_path.push_back(exit);
            exit_route_cost+=exit_cost*step;
            const double remaining=std::hypot(exit.x-goal.x,exit.y-goal.y);
            const double progress=(exit.x-start.x)*ux+(exit.y-start.y)*uy;
            if(progress<.25 || remaining>distance-.20)continue;
            const double score=remaining+.25*p.s+.10*std::abs(wrapAngle(exit.yaw-start.yaw))+
              .5*(reverse_route_cost+exit_route_cost);
            if(score<best_score) {
              best_score=score;retreat=reverse;
              if(forward_exit)*forward_exit=exit_path;
            }
          }
        }
      }
    }
    const bool safe=retreat.size()>=2 && bodyArcPathFeasible(retreat) &&
      chancePathSafe(retreat,nullptr,chance_path_budget_);
    if(!safe){retreat.clear();if(forward_exit)forward_exit->clear();}
    return safe;
  }

  bool planReverseRecovery(uint64_t expected_sequence=0,uint64_t expected_epoch=0)
  {
    PlanningBudget recovery_budget(max_search_time_ms_);
    if(PlanningBudget::expired())return false;
    {
      std::lock_guard<std::mutex> guard(path_mutex_);
      // The existing maneuver owns its stopped cusps and re-certification.
      // A newly completed background proposal is not a reason to consume
      // another recovery attempt or throw away its remaining gear intent.
      if(recovery_path_ && !path_.empty() && (have_path_ || recovery_resume_pending_))return false;
    }
    if (!reverse_recovery_enable_ || !reverse_enable_ || !live_obstacle_enable_ ||
        (topology_guide_enable_ && reverse_recovery_total_.load()>=16)) return false;
    OdomState odom;{std::lock_guard<std::mutex> lock(odom_mutex_);odom=odom_;}
    if (!odom.valid || std::abs(odom.v)>.08 || now().seconds()-odom.stamp.seconds()>max_odom_age_sec_) return false;
    const auto live=std::atomic_load(&live_snapshot_);
    if (!live || now().seconds()-live->stamp_sec>live_obstacle_max_age_sec_) return false;
    bool blocked=false;
    if(live->resolution<=0 || live->width<=0 || live->height<=0)return false;
    // Normal planning failed, or a complete reverse-first route awaits a
    // stopped handoff. A side obstacle can prevent turning even when the old
    // narrow forward-center strip is completely free. Useful forward curves
    // are independently protected below; a short stopping prefix is not one.
    constexpr double recovery_neighborhood=2.6;
    const int first_x=std::max(0,static_cast<int>(std::floor((odom.x-recovery_neighborhood-live->min_x)/live->resolution)));
    const int last_x=std::min(live->width-1,static_cast<int>(std::floor((odom.x+recovery_neighborhood-live->min_x)/live->resolution)));
    const int first_y=std::max(0,static_cast<int>(std::floor((odom.y-recovery_neighborhood-live->min_y)/live->resolution)));
    const int last_y=std::min(live->height-1,static_cast<int>(std::floor((odom.y+recovery_neighborhood-live->min_y)/live->resolution)));
    for(int iy=first_y;iy<=last_y&&!blocked;++iy)for(int ix=first_x;ix<=last_x;++ix) {
      const double x=live->min_x+(ix+.5)*live->resolution;
      const double y=live->min_y+(iy+.5)*live->resolution;
      if(std::hypot(x-odom.x,y-odom.y)<=recovery_neighborhood && live->occupied(x,y)) {blocked=true;break;}
    }
    if (!blocked) return false;
    uint64_t seq;Pose2 goal;
    {std::lock_guard<std::mutex> lock(goal_mutex_);if(!have_goal_)return false;seq=goal_seq_;goal=raw_goal_;}
    const auto epoch=input_epoch_.load();
    if(expected_sequence && (seq!=expected_sequence || epoch!=expected_epoch))return false;
    std::lock_guard<std::mutex> lock(map_mutex_);
    refreshSearchMapLocked();
    std::vector<Pose2> retreat,forward_exit;
    const Pose2 start{odom.x,odom.y,odom.yaw,0.,-1};
    double stance_cost=0.;terrain_nav::ChanceBudget stance;
    if(!footprintCostLocked(start.x,start.y,start.yaw,stance_cost) ||
       !addChancePose(start.x,start.y,start.yaw,stance,chance_state_limit_))return false;
    std::vector<Pose2> active;bool active_recovery=false;
    {std::lock_guard<std::mutex> guard(path_mutex_);
      if(have_path_)active=path_;active_recovery=recovery_path_;}
    if(!active_recovery && canRetainCertifiedForwardPathLocked(odom,active,false)) {
      // A timed-out search or a side obstacle inside the 2.6m neighborhood
      // is not a proof that forward motion is impossible. Never replace a
      // currently certified forward curve with an unsolicited reverse leg.
      RCLCPP_INFO_THROTTLE(get_logger(),*get_clock(),1500,
        "terrain recovery deferred: current forward curve still has a body/support/joint certificate");
      return false;
    }
    // Record the failed branch BEFORE selecting either kind of recovery.
    // Previously only the reverse-only fallback remembered it, so a paired
    // exit could deliberately return to the same dead end with no preference.
    std::deque<std::vector<Pose2>> global_legs;
    bool forward_available=false;
    const bool global_maneuver=takeBackgroundManeuverLocked(odom,goal,retreat,global_legs,&forward_available);
    if(forward_available)return false;
    // Two unplanned local retreats may be exhausted without invalidating a
    // newly completed whole-goal gear route. Its stopped current prefix was
    // just certified above; later gears still need fresh proof. The separate
    // unchanged per-goal total of 16 bounds BOTH kinds of reverse motion.
    if(!global_maneuver && reverse_recovery_count_.load()>=2)return false;
    rememberBlockedRoute(odom,seq);
    bool paired=global_maneuver;
    if(!paired) {
      // Reserve time for a bounded, independently certified retreat. A
      // coarse routing deadlock must not consume every recovery invocation.
      PlanningBudget pair_budget(std::max(0.,PlanningBudget::remaining()-25.));
      paired=topology_guide_enable_ ? buildTopologyRecoveryLocked(start,goal,retreat,forward_exit) :
        buildRecoveryManeuverLocked(start,goal,retreat,&forward_exit);
    }
    const bool replan_only=!paired;
    if(replan_only && !buildCertifiedDeadendRetreatLocked(start,retreat))return false;
    const double length=retreat.back().s;
    {std::lock_guard<std::mutex> guard(path_mutex_);
      if(!planSnapshotCurrent(seq,epoch)) return false;
      path_=std::move(retreat);active_goal_=path_.back();have_path_=true;recovery_path_=true;recovery_staged_gears_=false;recovery_legs_.clear();
      recovery_exit_=std::move(forward_exit);recovery_exit_active_=false;
      recovery_replan_only_=replan_only;
      recovery_geometry_epoch_=epoch;recovery_resume_pending_=false;
      if(global_maneuver) {
        recovery_staged_gears_=true;recovery_legs_=std::move(global_legs);
        recovery_legs_spatial_epoch_=spatial_epoch_.load();
      }
      ++path_seq_;if(!global_maneuver)++reverse_recovery_count_;++reverse_recovery_total_;
      auto origin=std::make_shared<RecoveryProgressOrigin>(RecoveryProgressOrigin{odom.x,odom.y,seq,epoch});
      recovery_progress_valid_epoch_.store(epoch);
      std::atomic_store(&reverse_progress_origin_,std::shared_ptr<const RecoveryProgressOrigin>(origin));}
    resetNominalControls();publishPath();
    RCLCPP_INFO(get_logger(),"terrain reverse recovery: verified %.2fm arc mode=%s, attempt=%d",
      length,global_maneuver?"global staged maneuver":replan_only?"bounded retreat then stop/replan":"paired forward exit",reverse_recovery_count_.load());
    return true;
  }

  bool buildCertifiedDeadendRetreatLocked(const Pose2 &start,std::vector<Pose2> &retreat) const
  {
    retreat.clear();
    if(!chance_constrained_ || !reverse_enable_ || !live_obstacle_enable_ || PlanningBudget::expired())return false;
    const auto live=std::atomic_load(&live_snapshot_);
    const double time=now().seconds();
    if(!live || time-live->stamp_sec<-.05 || time-live->stamp_sec>live_obstacle_max_age_sec_ ||
       time-map_stamp_.seconds()<-.05 || time-map_stamp_.seconds()>max_map_age_sec_)return false;
    // This is NOT a fabricated forward exit or mission progress. Certify a
    // finite reverse-only leg, stop there, then search from new observations.
    // Prefer straight backing; tighter arcs are alternatives only when their
    // entire protected body and current joint terrain account are admissible.
    const double limit=std::min(2.5,reverse_max_distance_);
    double longest=0.;
    for(double steer:{0.,-.33*max_steer_,.33*max_steer_,-.66*max_steer_,.66*max_steer_,-max_steer_,max_steer_}) {
      if(PlanningBudget::expired())break;
      std::vector<Pose2> path={Pose2{start.x,start.y,start.yaw,0.,-1}};
      terrain_nav::ChanceBudget budget;
      if(!addChancePose(start.x,start.y,start.yaw,budget,chance_path_budget_))return false;
      for(int i=1;i<=int(std::floor(limit/.025));++i) {
        if(PlanningBudget::expired())break;
        auto p=advanceArc(path.back(),primitiveCurvature(steer),-.025);p.direction=-1;p.s=i*.025;
        double cost=0.,clearance=0.;
        if(!footprintCostLocked(p.x,p.y,p.yaw,cost,&clearance) || clearance<hardClearanceFloor() ||
           !addChancePose(p.x,p.y,p.yaw,budget,chance_path_budget_))break;
        path.push_back(p);
      }
      // Tiny shuffles cannot escape a dead end and can oscillate forever.
      if(path.back().s>=.6 && path.back().s>longest+.05 && bodyArcPathFeasible(path)) {
        longest=path.back().s;retreat=std::move(path);
      }
      if(longest>=limit-1e-6)break;
    }
    return retreat.size()>1 && pathFootprintSafeLocked(retreat,.025) &&
      chancePathSafe(retreat,nullptr,chance_path_budget_);
  }

  bool buildTopologyRecoveryLocked(const Pose2 &start,const Pose2 &goal,
      std::vector<Pose2> &retreat,std::vector<Pose2> &forward_exit)
  {
    PlanningBudget recovery_budget(max_search_time_ms_);
    retreat.clear();forward_exit.clear();
    if(!topology_guide_enable_ || !chance_constrained_)return false;
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::duration<double,std::milli>(PlanningBudget::remaining(max_search_time_ms_));
    OdomState odom;odom.x=start.x;odom.y=start.y;odom.yaw=start.yaw;
    Pose2 local;std::vector<Pose2> unused;
    resolveTopologyGoalLocked(odom,goal,local,unused,true); // refresh evidence without an unused full forward search
    // Search the reverse-to-forward maneuver itself instead of guessing only
    // a retreat length/steer and hoping its cusp has a useful forward route.
    // This remains an optimistic guide: BOTH complete executed legs below
    // need the current full-body and shared joint-budget certificate.
    terrain_nav::RouteOptions recovery_routing;recovery_routing.rear_axle_offset=rear_axle_offset_;recovery_routing.heuristic_weight=4.;recovery_routing.advisory_cost=2.;
    recovery_routing.reverse_observation_cost=[this](double x,double y,double yaw){return topologyReverseObservationCostLocked(x,y,yaw);};
    // Match the forward rolling guide: a distant/unobserved final goal must
    // not veto an otherwise completely certified local reverse + exit pair.
    // A useful frontier is informational, never reported as mission arrival.
    recovery_routing.allow_partial=true;
    recovery_routing.allow_advisory_start_overlap=topologyAdvisoryStartQualified(start.x,start.y,start.yaw);
    recovery_routing.max_initial_reverse=reverse_max_distance_;
    // Later gears are informative global guidance. Only the first reverse
    // leg and its first forward exit are certified/executed by this recovery.
    recovery_routing.single_reverse_then_forward=false;
    const auto joint_route=[&]() -> bool {
    for(bool fine:{false,true}) {
      if(std::chrono::steady_clock::now()>=deadline)break;
      const auto route=route_memory_.route({start.x,start.y},{goal.x,goal.y},start.yaw,
          std::max(footprint_half_width_,body_collision_half_width_)+footprint_margin_,maxReferenceCurvature(),
          body_collision_front_+footprint_margin_,body_collision_back_+footprint_margin_,
          [this](double x,double y,double yaw){return topologyTerrainCostLocked(x,y,yaw)+recoveryRoutePenalty(x,y);},fine,true,-1,PlanningBudget::remaining(24.),recovery_routing);
      if((!route.reached && !route.partial) || route.path.size()<3 || route.directions.size()!=route.path.size() ||
         route.directions.front()!=-1)continue;
      size_t cusp=1;
      while(cusp<route.path.size() && route.directions[cusp]<0)++cusp;
      if(cusp>=route.path.size())continue;
      std::vector<Pose2> reverse_guide,exit_guide;double length=0;
      for(size_t i=0;i<cusp;++i) {
        if(i)length+=std::hypot(route.path[i][0]-route.path[i-1][0],route.path[i][1]-route.path[i-1][1]);
        reverse_guide.push_back(Pose2{route.path[i][0],route.path[i][1],route.yaws[i],length,-1});
      }
      if(length>reverse_max_distance_+1e-6)continue;
      length=0;
      for(size_t i=cusp-1;i<route.path.size();++i) {
        if(i>=cusp && route.directions[i]<0)break;
        if(i>=cusp)length+=std::hypot(route.path[i][0]-route.path[i-1][0],route.path[i][1]-route.path[i-1][1]);
        exit_guide.push_back(Pose2{route.path[i][0],route.path[i][1],route.yaws[i],length,1});
        if(length>=4.)break;
      }
      std::vector<Pose2> reverse,exit;
      if(!buildCertifiedGuidePrefixLocked(reverse_guide,reverse) ||
         reverse.back().s+1e-6<reverse_guide.back().s ||
         !buildCertifiedGuidePrefixLocked(exit_guide,exit) || exit.back().s<1.2 ||
         std::hypot(exit.back().x-start.x,exit.back().y-start.y)<.6)continue;
      auto joined=reverse;joined.insert(joined.end(),exit.begin(),exit.end());
      if(!chancePathSafe(joined,nullptr,chance_path_budget_))continue;
      retreat=std::move(reverse);forward_exit=std::move(exit);return true;
    }
    return false;
    };
    // A failed bidirectional search must not consume the whole budget on
    // every retry and starve an already working arc recovery. Alternate the
    // first strategy; each planning invocation keeps the same wall budget.
    const bool joint_first=(topology_recovery_search_round_.fetch_add(1)%2)!=0;
    if(joint_first && joint_route())return true;
    const double bearing=std::atan2(goal.y-start.y,goal.x-start.x);
    const double sign=wrapAngle(start.yaw-bearing)>=0?1.:-1.;
    for(double distance:terrain_nav::topologyRecoveryDistances(start.yaw-bearing,maxReferenceCurvature(),reverse_max_distance_))
    for(double steer:{sign*max_steer_,0.,-sign*max_steer_}) {
      if(std::chrono::steady_clock::now()>=deadline)return false;
      if(distance>reverse_max_distance_)continue;
      std::vector<Pose2> reverse={Pose2{start.x,start.y,start.yaw,0.,-1}};
      terrain_nav::ChanceBudget budget;bool safe=addChancePose(start.x,start.y,start.yaw,budget,chance_path_budget_);
      for(int i=1;safe && i<=int(std::lround(distance/.025));++i) {
        auto p=advanceArc(reverse.back(),primitiveCurvature(steer),-.025);p.direction=-1;p.s=i*.025;
        double cost=0,clearance=kInf;
        safe=footprintCostLocked(p.x,p.y,p.yaw,cost,&clearance) && clearance>=hardClearanceFloor() &&
          addChancePose(p.x,p.y,p.yaw,budget,chance_path_budget_);
        if(safe)reverse.push_back(p);
      }
      if(!safe || !bodyArcPathFeasible(reverse) || !pathFootprintSafeLocked(reverse,.025))continue;
      const auto &end=reverse.back();
      auto exit_routing=recovery_routing;
      // The cusp is a future state. Advisory relief is limited to the fixed
      // CURRENT stance, never expanded along a hypothetical retreat.
      exit_routing.allow_advisory_start_overlap=false;
      auto route=route_memory_.route({end.x,end.y},{goal.x,goal.y},end.yaw,
          std::max(footprint_half_width_,body_collision_half_width_)+footprint_margin_,maxReferenceCurvature(),
          body_collision_front_+footprint_margin_,body_collision_back_+footprint_margin_,
          [this](double x,double y,double yaw){return topologyTerrainCostLocked(x,y,yaw)+recoveryRoutePenalty(x,y);},false,false,0,PlanningBudget::remaining(24.),exit_routing);
      if(!route.reached && !route.partial && std::chrono::steady_clock::now()<deadline)
        route=route_memory_.route({end.x,end.y},{goal.x,goal.y},end.yaw,
            std::max(footprint_half_width_,body_collision_half_width_)+footprint_margin_,maxReferenceCurvature(),
            body_collision_front_+footprint_margin_,body_collision_back_+footprint_margin_,
            [this](double x,double y,double yaw){return topologyTerrainCostLocked(x,y,yaw)+recoveryRoutePenalty(x,y);},true,false,0,PlanningBudget::remaining(24.),exit_routing);
      if(!route.reached && !route.partial)continue;
      std::vector<Pose2> guide;double length=0;
      for(size_t i=0;i<route.path.size();++i) {
        if(i)length+=std::hypot(route.path[i][0]-route.path[i-1][0],route.path[i][1]-route.path[i-1][1]);
        guide.push_back(Pose2{route.path[i][0],route.path[i][1],route.yaws[i],length,1});
        if(length>=4.)break;
      }
      std::vector<Pose2> exit;
      if(!buildCertifiedGuidePrefixLocked(guide,exit) || exit.back().s<1.2 ||
         std::hypot(exit.back().x-start.x,exit.back().y-start.y)<.6)continue;
      auto joined=reverse;joined.insert(joined.end(),exit.begin(),exit.end());
      if(!chancePathSafe(joined,nullptr,chance_path_budget_))continue;
      retreat=std::move(reverse);forward_exit=std::move(exit);return true;
    }
    return !joint_first && std::chrono::steady_clock::now()<deadline && joint_route();
  }

  void completeTopologyRecovery(uint64_t sequence,uint64_t epoch,const OdomState &odom,
      const std::shared_ptr<const RecoveryProgressOrigin> &expected={})
  {
    std::lock_guard<std::mutex> guard(path_mutex_);
    const auto origin=std::atomic_load(&reverse_progress_origin_);
    if(topology_guide_enable_ && origin && (!expected || expected==origin) &&
       origin->sequence==sequence && (origin->epoch==epoch || recovery_progress_valid_epoch_.load()==epoch) &&
       std::hypot(odom.x-origin->x,odom.y-origin->y)>=.6) {
      reverse_recovery_count_.store(0);
      std::atomic_store(&reverse_progress_origin_,std::shared_ptr<const RecoveryProgressOrigin>{});
    }
  }

  bool pauseRecoveryForEvidence(uint64_t sequence,uint64_t epoch,uint64_t path_sequence)
  {
    {
      std::lock_guard<std::mutex> guard(path_mutex_);
      if(!chance_constrained_ || !planSnapshotCurrent(sequence,epoch) ||
         !recovery_path_ || path_.size()<2 || path_seq_!=path_sequence ||
         recovery_geometry_epoch_!=epoch)return false;
      if(recovery_staged_gears_ &&
         (recovery_legs_spatial_epoch_!=spatial_epoch_.load() ||
          (!recovery_legs_.empty() && path_.front().direction>0 &&
           recovery_legs_.front().front().direction<0 && reverse_recovery_total_.load()>=16)))return false;
      const double time=now().seconds();
      if(!recovery_resume_pending_)recovery_resume_saved_time_=time;
      const double age=time-recovery_resume_saved_time_;
      if(age<0. || age>5.)return false;
      // Geometry only. Neither this path nor a queued future gear is an
      // executable certificate while waiting; the deadline is not renewed.
      recovery_resume_pending_=true;have_path_=false;
    }
    publishImmediateStop();
    RCLCPP_INFO_THROTTLE(get_logger(),*get_clock(),1000,
      "terrain recovery waiting for fresh evidence: stopped intent retained, no new attempt or permission");
    return true;
  }

  bool planPath(const std::string &replan_reason)
  {
    guide_handoff_=GuideHandoff::None;
    fresh_background_guide_=false;
    PlanningBudget forward_budget(max_search_time_ms_);
    if(reserve_retreat_active_.load(std::memory_order_acquire))return false;
    const uint64_t input_epoch = input_epoch_.load(std::memory_order_acquire);
    OdomState odom;
    std::vector<Pose2> forward_trail;
    {
      std::lock_guard<std::mutex> lock(odom_mutex_);
      odom = odom_;
      forward_trail.assign(forward_trail_.begin(),forward_trail_.end());
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
    std::vector<Pose2> recovery_exit;
    bool old_have_path = false;
    bool old_recovery = false,executing_exit=false,replan_after_retreat=false,resume_recovery=false,staged_recovery=false;
    uint64_t old_path_sequence=0;
    {
      std::lock_guard<std::mutex> lock(path_mutex_);
      old_have_path = have_path_;
      old_path = path_;
      old_path_sequence=path_seq_;
      old_recovery=recovery_path_;executing_exit=recovery_exit_active_;
      replan_after_retreat=recovery_replan_only_;
      staged_recovery=recovery_staged_gears_;
      if(recovery_resume_pending_) {
        const double age=now().seconds()-recovery_resume_saved_time_;
        resume_recovery=recovery_path_ && recovery_geometry_epoch_==input_epoch && age>=0. && age<=5.;
        old_have_path=resume_recovery;
        if(!resume_recovery) {
          path_.clear();old_path.clear();have_path_=false;recovery_path_=false;recovery_staged_gears_=false;recovery_legs_.clear();
          recovery_resume_pending_=false;old_recovery=false;
        }
      }
      if(old_recovery)recovery_exit=recovery_exit_;
    }

    std::lock_guard<std::mutex> map_lock(map_mutex_);
    refreshSearchMapLocked();
    if(PlanningBudget::expired())return false;
    if (!map_ready_ || width_ <= 0 || height_ <= 0) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "terrain_nav waits for fused_map");
      return false;
    }

    if (chance_constrained_) {
      const auto current=queryCostLocked(odom.x,odom.y,odom.yaw);
      if (!current.known || current.lethal || !std::isfinite(current.probability_upper)) {
        RCLCPP_WARN_THROTTLE(get_logger(),*get_clock(),1500,
          "Probability current pose rejected before search: known=%s upper=%.6f limit=%.3f source_ns=%lld now_ns=%lld pose=(%.12f,%.12f,%.12f); observe/qualify ground support first",
          current.known ? "true" : "false",current.probability_upper,chance_state_limit_,
          static_cast<long long>(map_stamp_.nanoseconds()),static_cast<long long>(now().nanoseconds()),odom.x,odom.y,odom.yaw);
        return false;
      }
    }
    // Consume completed global work BEFORE old-path retention. Otherwise a
    // good result could sit unread until its source/pose had become stale.
    if(background_guide_.valid() && background_guide_.wait_for(std::chrono::milliseconds(0))==std::future_status::ready) {
      terrain_nav::RouteMemory::Result selected;
      fresh_background_guide_=takeBackgroundGuideLocked(odom,raw_goal_snapshot,selected);
    }

    const auto retain_better_current_prefix=[&](const std::vector<Pose2> &candidate) {
      if(!old_have_path || !preferCertifiedRemainingPathLocked(odom,raw_goal_snapshot,old_path,candidate,old_recovery))return false;
      std::lock_guard<std::mutex> lock(path_mutex_);
      if(!planSnapshotCurrent(goal_seq_snapshot,input_epoch) || !have_path_ || path_seq_!=old_path_sequence)return false;
      RCLCPP_INFO(get_logger(),"terrain retain current certified curve: replacement loses progress or switches the near approach; current geometry/support/joint budget revalidated");
      return true;
    };

    // A newly certified ordinary path can replace an invalidated recovery
    // exit. Its real forward progress must not be lost with the old path ID.
    // This is retry bookkeeping ONLY; it cannot authorize a motion or reset
    // the per-goal total. The current pose was checked immediately above.
    const auto forward_progress=std::atomic_load(&recovery_forward_progress_);
    if(forward_progress && forward_progress->distance>=.6)
      completeTopologyRecovery(goal_seq_snapshot,input_epoch,odom,forward_progress->origin);

    // Count actual certified forward-exit displacement even if a later
    // observation invalidates the far end of that exit. Waiting for a 4m
    // rolling leg to finish made two useful maneuvers permanently exhaust
    // recovery at a new location. A reverse-only displacement cannot reset
    // this guard; the unchanged per-goal total still bounds repeated retries.
    if(old_have_path && old_recovery && executing_exit && !old_path.empty() &&
       old_path.front().direction>0) {
      double track_distance=0.;const size_t nearest=nearestPathIndex(old_path,odom.x,odom.y,&track_distance);
      // Distance from the pre-retreat origin alone is insufficient: it can
      // already exceed .6m at the CUSP, before any forward motion happened.
      // Require actual forward-exit arc progress and path capture as well.
      if(old_path[nearest].s-old_path.front().s>=.6 &&
         track_distance<=std::min(.15,replan_track_error_) &&
         std::abs(wrapAngle(odom.yaw-old_path[nearest].yaw))<=.45)
        completeTopologyRecovery(goal_seq_snapshot,input_epoch,odom);
    }

    // Reserve control only executes the retreat. The planner owns the gear
    // change and re-certifies the proposed exit from the actual stopped pose.
    if(const auto pending=std::atomic_exchange(&reserve_exit_request_,
        std::shared_ptr<const ReserveExitRequest>{})) {
      if(pending->goal_seq==goal_seq_snapshot && pending->epoch==input_epoch &&
         std::abs(odom.v)<=.03 && pending->path.size()>=2 &&
         std::hypot(odom.x-pending->path.front().x,odom.y-pending->path.front().y)<=.15 &&
         std::abs(wrapAngle(odom.yaw-pending->path.front().yaw))<=.15) {
        auto exit=reanchorArc(pending->path,odom);
        if(bodyArcPathFeasible(exit) && pathFootprintSafeLocked(exit,.025) &&
           chancePathSafe(exit,nullptr,chance_path_budget_)) {
          {
            std::lock_guard<std::mutex> lock(path_mutex_);
            if(!planSnapshotCurrent(goal_seq_snapshot,input_epoch))return false;
            path_=std::move(exit);active_goal_=path_.back();have_path_=true;
            recovery_path_=true;recovery_staged_gears_=false;recovery_legs_.clear();recovery_exit_active_=true;recovery_replan_only_=false;recovery_exit_.clear();
            recovery_geometry_epoch_=input_epoch;recovery_resume_pending_=false;
            ++path_seq_;resetNominalControls();
          }
          publishPath();
          RCLCPP_INFO(get_logger(),"clearance reserve recovery: stopped; re-certified forward exit");
          return true;
        }
      }
      RCLCPP_WARN(get_logger(),"clearance reserve recovery: proposed forward exit invalidated; replanning");
    }

    if(old_have_path && old_recovery && old_path.size()>=2) {
      if(resume_recovery && std::abs(odom.v)>.03) {
        publishImmediateStop();return false; // fresh proof must not cancel an unfinished stop
      }
      const size_t nearest=nearestPathIndex(old_path,odom.x,odom.y);
      const bool exhausted=old_path.back().s-old_path[nearest].s<=.025;
      const double endpoint_distance=std::hypot(old_path.back().x-odom.x,old_path.back().y-odom.y);
      // The tracker stops at exhausted arc length. Capture its bounded lateral
      // error too; the forward exit is re-certified from the actual stopped
      // pose below. Larger errors discard the leg instead of retaining a
      // stopped path forever.
      const double capture_distance=std::min(.15,std::max(.025,replan_track_error_));
      const bool at_end=exhausted && endpoint_distance<=capture_distance;
      if(exhausted && !at_end && std::abs(odom.v)<=.03) {
        std::lock_guard<std::mutex> lock(path_mutex_);
        if(planSnapshotCurrent(goal_seq_snapshot,input_epoch)) {
          path_.clear();have_path_=false;recovery_path_=false;recovery_staged_gears_=false;recovery_legs_.clear();recovery_exit_.clear();
        }
        return false;
      }
      if(!at_end || std::abs(odom.v)>.03) {
        std::vector<Pose2> remaining(old_path.begin()+nearest,old_path.end());
        remaining.insert(remaining.begin(),Pose2{odom.x,odom.y,odom.yaw,0.,old_path.front().direction});
        if(pathFootprintSafeLocked(remaining,.025) && chancePathSafe(remaining,nullptr,chance_path_budget_)) {
          if(resume_recovery) {
            const auto live=std::atomic_load(&live_snapshot_);
            const double time=now().seconds();
            if(!live || time-live->stamp_sec<-.05 || time-live->stamp_sec>live_obstacle_max_age_sec_ ||
               time-map_stamp_.seconds()<-.05 || time-map_stamp_.seconds()>max_map_age_sec_ ||
               time-odom.stamp.seconds()>max_odom_age_sec_)return false;
            std::lock_guard<std::mutex> lock(path_mutex_);
            if(!planSnapshotCurrent(goal_seq_snapshot,input_epoch))return false;
            have_path_=true;recovery_resume_pending_=false;
            // No old certificate survives the interruption. Coordinate-only
            // progress identity is independent of temporal plan invalidation.
            RCLCPP_INFO(get_logger(),"terrain recovery resumed after fresh full-body/support/joint revalidation; no new attempt consumed");
          }
          return planSnapshotCurrent(goal_seq_snapshot,input_epoch);
        }
        // The planner and controller must agree about finite certificates.
        // A changed far suffix of a maneuver must not erase its safe current
        // leg and all subsequent gear intentions before control can repair it.
        if(old_have_path && repairInvalidPathSuffix(odom,old_path,old_path_sequence))return true;
        if(pauseRecoveryForEvidence(goal_seq_snapshot,input_epoch,old_path_sequence))return false;
        std::lock_guard<std::mutex> lock(path_mutex_);
        if(planSnapshotCurrent(goal_seq_snapshot,input_epoch) && path_seq_==old_path_sequence) {
          path_.clear();have_path_=false;recovery_path_=false;recovery_staged_gears_=false;recovery_legs_.clear();recovery_exit_.clear();
          recovery_resume_pending_=false;
        }
        return false;
      }
      if(staged_recovery) {
        bool more=false;{std::lock_guard<std::mutex> guard(path_mutex_);more=!recovery_legs_.empty();}
        if(more && activateNextStagedGearLocked(odom,goal_seq_snapshot,input_epoch))return true;
        if(more && pauseRecoveryForEvidence(goal_seq_snapshot,input_epoch,old_path_sequence))return false;
        {
          std::lock_guard<std::mutex> guard(path_mutex_);
          if(!planSnapshotCurrent(goal_seq_snapshot,input_epoch))return false;
          path_.clear();have_path_=false;recovery_path_=false;recovery_staged_gears_=false;recovery_legs_.clear();
          recovery_resume_pending_=false;
        }
        if(more)return false; // next leg failed current proof; do not execute it
        if(old_path.front().direction>0)completeTopologyRecovery(goal_seq_snapshot,input_epoch,odom);
        old_have_path=false;old_path.clear();old_recovery=false;
      }
      if(!staged_recovery && !executing_exit) {
        const auto origin=std::atomic_load(&reverse_progress_origin_);
        if(origin && origin->sequence==goal_seq_snapshot &&
           (origin->epoch==input_epoch || recovery_progress_valid_epoch_.load()==input_epoch)) {
          OdomState failed;failed.x=origin->x;failed.y=origin->y;
          // Start the unchanged finite lifetime at the actual route decision,
          // not only before a potentially long, slow reverse leg.
          rememberBlockedRoute(failed,goal_seq_snapshot);
        }
      }
      if(!staged_recovery && !executing_exit && replan_after_retreat) {
        std::lock_guard<std::mutex> lock(path_mutex_);
        if(!planSnapshotCurrent(goal_seq_snapshot,input_epoch))return false;
        path_.clear();have_path_=false;recovery_path_=false;recovery_staged_gears_=false;recovery_legs_.clear();recovery_replan_only_=false;recovery_exit_.clear();recovery_resume_pending_=false;
        // Do not reset retry counters or claim arrival after backing up.
        // Continue with a fresh forward search from this confirmed stop.
        RCLCPP_INFO(get_logger(),"terrain dead-end retreat: confirmed stop; retaining final goal and requesting a fresh route");
      }
      if(!staged_recovery && !executing_exit && !replan_after_retreat) {
        auto exit_path=reanchorArc(recovery_exit,odom);
        if(exit_path.size()<2 || !bodyArcPathFeasible(exit_path) ||
           !pathFootprintSafeLocked(exit_path,.025) || !chancePathSafe(exit_path,nullptr,chance_path_budget_)) {
          std::lock_guard<std::mutex> lock(path_mutex_);
          if(planSnapshotCurrent(goal_seq_snapshot,input_epoch)) {
            path_.clear();have_path_=false;recovery_path_=false;recovery_staged_gears_=false;recovery_legs_.clear();recovery_exit_.clear();
          }
          return false;
        }
        {
          std::lock_guard<std::mutex> lock(path_mutex_);
          if(!planSnapshotCurrent(goal_seq_snapshot,input_epoch))return false;
          path_=std::move(exit_path);active_goal_=path_.back();have_path_=true;
          recovery_exit_active_=true;recovery_resume_pending_=false;++path_seq_;resetNominalControls();
        }
        publishPath();
        RCLCPP_INFO(get_logger(),"terrain recovery: stopped at cusp; re-certified forward exit");
        return true;
      }
      {
        std::lock_guard<std::mutex> lock(path_mutex_);
        if(!planSnapshotCurrent(goal_seq_snapshot,input_epoch))return false;
        recovery_path_=false;recovery_staged_gears_=false;recovery_legs_.clear();recovery_exit_.clear();
      }
      if(!staged_recovery && !replan_after_retreat)completeTopologyRecovery(goal_seq_snapshot,input_epoch,odom);
      old_have_path=false;old_path.clear();
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
    const double reverse_distance=std::hypot(reverse_dx,reverse_dy);
    const double reverse_projection=reverse_dx*std::cos(odom.yaw)+reverse_dy*std::sin(odom.yaw);
    const bool rear_terminal=reverse_distance<=2*goal_reached_distance_ && reverse_projection<-1e-6;
    if (reverse_enable_ && reverse_distance <= reverse_max_distance_ &&
        (reverse_projection < -goal_reached_distance_ || rear_terminal)) {
      const Pose2 reverse_start{odom.x,odom.y,odom.yaw,0,-1};
      std::vector<Pose2> reverse_path;
      Pose2 guide;
      bool guided=reverseGuideTarget(reverse_start,raw_goal_snapshot,forward_trail,guide) &&
        (buildCertifiedReversePoseLocked(reverse_start,guide,reverse_path) ||
         buildCertifiedReverseArcLocked(reverse_start,guide,reverse_path));
      if (!guided && !buildCertifiedReverseArcLocked(reverse_start,raw_goal_snapshot,reverse_path)) {
        // Prefer direct reverse. If curvature/base offset make it impossible,
        // a finite, fully certified parking maneuver is different from an
        // unrestricted forward detour. Reuse stopped staged-gear handover.
        if(chance_constrained_ && reverse_recovery_enable_ && reverse_distance<=1.5 &&
            std::abs(odom.v)<=.03 && reverse_recovery_total_.load()<16) {
          std::vector<std::vector<Pose2>> legs;
          if(buildCertifiedTerminalManeuverLocked(reverse_start,raw_goal_snapshot,legs)) {
            const int first_gear=legs.front().front().direction;
            {
              std::lock_guard<std::mutex> lock(path_mutex_);
              if(!planSnapshotCurrent(goal_seq_snapshot,input_epoch))return false;
              path_=std::move(legs.front());active_goal_=path_.back();have_path_=true;
              recovery_path_=recovery_staged_gears_=true;recovery_exit_active_=false;
              recovery_replan_only_=false;recovery_exit_.clear();recovery_legs_.clear();
              recovery_legs_.push_back(std::move(legs.back()));
              recovery_legs_spatial_epoch_=spatial_epoch_.load();recovery_geometry_epoch_=input_epoch;
              recovery_resume_pending_=false;++path_seq_;
              if(first_gear<0)++reverse_recovery_total_;
            }
            resetNominalControls();publishActiveGoal(raw_goal_snapshot,true);publishPath();
            RCLCPP_INFO(get_logger(),"terrain terminal maneuver: full two-gear goal connector certified; stopped cusp required");
            return true;
          }
        }
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
          "Close rear goal has no certified reverse prefix or bounded terminal maneuver; stopping");
        return false;
      }
      // Reverse has the same rolling-horizon semantics as forward motion:
      // keep the raw goal, admit only the currently certified arc prefix, and
      // re-check its entire braking envelope on every control tick.
      const bool partial=std::hypot(reverse_path.back().x-raw_goal_snapshot.x,
                                    reverse_path.back().y-raw_goal_snapshot.y)>goal_tolerance_;
      {
        std::lock_guard<std::mutex> lock(path_mutex_);
        if (!planSnapshotCurrent(goal_seq_snapshot, input_epoch)) return false;
        path_ = std::move(reverse_path);
        recovery_path_ = false;recovery_staged_gears_=false;recovery_legs_.clear();
        active_goal_ = path_.back();
        have_path_ = true;
        ++path_seq_;
      }
      publishActiveGoal(raw_goal_snapshot, true);
      publishPath();
      RCLCPP_INFO(get_logger(), "terrain reverse arc success: distance=%.2fm partial=%s guide=%s",
                  std::hypot(reverse_dx, reverse_dy),partial?"true":"false",guided?"recorded corridor":"direct");
      return true;
    }

    const auto try_mission_progress_fan=[&]() {
      std::vector<Pose2> certified;
      if(!buildMissionProgressFanLocked(Pose2{odom.x,odom.y,odom.yaw,0.},raw_goal_snapshot,certified))
        return false;
      const auto endpoint=certified.back();
      {
        std::lock_guard<std::mutex> lock(path_mutex_);
        if(!planSnapshotCurrent(goal_seq_snapshot,input_epoch))return false;
        path_=std::move(certified);active_goal_=endpoint;have_path_=true;recovery_path_=false;recovery_staged_gears_=false;recovery_legs_.clear();
        ++path_seq_;observation_path_seq_=path_seq_;resetNominalControls();
      }
      publishActiveGoal(raw_goal_snapshot,true);publishPath();
      RCLCPP_INFO(get_logger(),"terrain certified mission-progress fan: %.2fm; failed rolling guide does not replace final goal",endpoint.s);
      return true;
    };

    std::vector<Pose2> continuous_path;
    const auto try_observation_prefix = [&]() {
      if (!chance_constrained_ || !chance_allow_startup_prior_ || !map_.exists("probability_pose_event"))
        return false;
      const Eigen::Vector3d start_position(search_start.x,search_start.y,search_start.yaw);
      const double conditional=map_.getValue("probability_pose_event",start_position,
          se2_grid::InterpolationMethods::INTER_NEAREST);
      const double prefix_limit=observationPrefixLimit(search_start,raw_goal_snapshot);
      // Observe while progressing only over a currently certified prefix.
      // No new ground is assumed by this maneuver; the fixed start prior is
      // not expanded, and the final braking envelope still gates every tick.
      const bool hazard=knownHazardAheadLocked(search_start,std::min(
        std::hypot(raw_goal_snapshot.x-search_start.x,raw_goal_snapshot.y-search_start.y),
        prefix_limit+footprint_front_+clearance_distance_));
      if (conditional==1 && prefix_limit>=.25 && !hazard) {
        std::vector<Pose2> prefix={search_start};
        for (double s=.05;s<=prefix_limit+1e-6;s+=.05) {
          Pose2 q{search_start.x+s*std::cos(search_start.yaw),
                  search_start.y+s*std::sin(search_start.yaw),search_start.yaw,s};
          double cost=0,clearance=kInf;
          prefix.push_back(q);
          if (!footprintCostLocked(q.x,q.y,q.yaw,cost,&clearance) ||
              (std::isfinite(clearance) && clearance<hardClearanceFloor()) ||
              !chancePathSafe(prefix,nullptr,chance_path_budget_)) {
            prefix.pop_back();break;
          }
        }
        if (prefix.back().s>=.25) {
          if (committed) prefix=mergeCommittedPrefix(committed_prefix,prefix);
          updatePathArcLength(prefix);
          if (!bodyArcPathFeasible(prefix) || !pathFootprintSafeLocked(prefix,.025) ||
              !chancePathSafe(prefix,nullptr,chance_path_budget_)) return false;
          {
            std::lock_guard<std::mutex> lock(path_mutex_);
            if (!planSnapshotCurrent(goal_seq_snapshot,input_epoch)) return false;
            path_=std::move(prefix);active_goal_=path_.back();have_path_=true;recovery_path_=false;recovery_staged_gears_=false;recovery_legs_.clear();++path_seq_;
            observation_path_seq_=path_seq_;
            resetNominalControls();
          }
          publishActiveGoal(active_goal_,true);publishPath();
          RCLCPP_INFO(get_logger(),"conditional support rolling prefix: %.2fm; raw goal retained",active_goal_.s);
          return true;
        }
      }
      return false;
    };
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
        continuous_path = mergeCommittedPrefix(committed_prefix, continuous_path);
        if (!chance_constrained_ && rear_axle_offset_==0.) refinePathForTrackingLocked(continuous_path);
      }
      updatePathArcLength(continuous_path);
      if ((!chance_constrained_ || (bodyArcPathFeasible(continuous_path) &&
           pathFootprintSafeLocked(continuous_path,.025))) &&
          chancePathSafe(continuous_path,nullptr,chance_path_budget_)) {
        size_t planned_path_size = 0;
        {
          std::lock_guard<std::mutex> lock(path_mutex_);
          if (!planSnapshotCurrent(goal_seq_snapshot, input_epoch)) return false;
          path_ = std::move(continuous_path);
          recovery_path_ = false;recovery_staged_gears_=false;recovery_legs_.clear();
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
    }

    // Missing distant support is an observation frontier, not an obstacle
    // requiring an arbitrary lateral detour. Keep aligned, certified progress
    // only when the complete local goal ray contains no known hazard.
    const double raw_distance=std::hypot(raw_goal_snapshot.x-search_start.x,raw_goal_snapshot.y-search_start.y);
    const double raw_bearing=std::atan2(raw_goal_snapshot.y-search_start.y,raw_goal_snapshot.x-search_start.x);
    if (direct_corridor_blocked && std::abs(wrapAngle(raw_bearing-search_start.yaw))<.15 &&
        !knownHazardAheadLocked(search_start,std::min(raw_distance,local_goal_distance_+1.2)) &&
        try_observation_prefix()) return true;

    Pose2 active_goal;
    std::string goal_reason;
    std::vector<Pose2> guide_path;
    std::vector<std::vector<Pose2>> proposed_gears;
    if (!resolveActiveGoalLocked(odom, raw_goal_snapshot, active_goal, goal_reason, guide_path, old_path,
                                old_have_path&&!old_recovery,&proposed_gears)) {
      if(guide_handoff_!=GuideHandoff::None) {
        // Keep the car at its currently certified stance rather than commit
        // a short unrelated frontier while a complete gear sequence is ready
        // or being computed. The ordinary controller provides the stop; a
        // reverse maneuver still requires measured near-zero speed.
        return retainFinitePathAtGuideHandoffLocked(odom,goal_seq_snapshot,input_epoch,old_path_sequence);
      }
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "Cannot resolve active terrain goal");
      return try_mission_progress_fan() || try_observation_prefix();
    }
    const double rolling_forward=(active_goal.x-odom.x)*std::cos(odom.yaw)+
      (active_goal.y-odom.y)*std::sin(odom.yaw);
    if((goal_reason=="retain certified current path" || (goal_reason=="guide search subgoal" && rolling_forward<0)) && old_have_path &&
       canRetainCertifiedForwardPathLocked(odom,old_path,old_recovery)) {
      std::lock_guard<std::mutex> lock(path_mutex_);
      if(!planSnapshotCurrent(goal_seq_snapshot,input_epoch) || !have_path_ || path_seq_!=old_path_sequence)
        return false;
      RCLCPP_INFO(get_logger(),"terrain retain current certified path: current body/support/joint risk revalidated; no new path published");
      return true;
    }
    publishActiveGoal(active_goal, true);

    // The global car-arc route is only an exploration proposal. Re-certify
    // its continuous initial segment using the FULL current vehicle envelope
    // and joint terrain budget. This also handles a U-turn whose initial arc
    // temporarily increases Euclidean goal distance, without weakening any
    // collision, probability or per-tick braking condition.
    if((goal_reason=="persistent topology route (local certificate required)" ||
        goal_reason=="retain certified current prefix") &&
       buildCertifiedGuidePrefixLocked(guide_path,continuous_path)) {
      if(retain_better_current_prefix(continuous_path))return true;
      const double length=continuous_path.back().s;
      {
        std::lock_guard<std::mutex> lock(path_mutex_);
        if(!planSnapshotCurrent(goal_seq_snapshot,input_epoch))return false;
        path_=std::move(continuous_path);active_goal_=path_.back();have_path_=true;
        recovery_path_=false;recovery_staged_gears_=false;recovery_legs_.clear();++path_seq_;
        recovery_staged_gears_=false;recovery_legs_.clear();
        if(proposed_gears.size()>1 && length+1e-6>=proposed_gears.front().back().s) {
          recovery_path_=true;recovery_staged_gears_=true;recovery_replan_only_=false;
          recovery_exit_active_=false;recovery_exit_.clear();recovery_resume_pending_=false;
          recovery_geometry_epoch_=input_epoch;recovery_legs_spatial_epoch_=spatial_epoch_.load();
          for(size_t i=1;i<proposed_gears.size();++i)recovery_legs_.push_back(std::move(proposed_gears[i]));
        }
        if(length<2.)observation_path_seq_=path_seq_;
        resetNominalControls();
      }
      publishActiveGoal(active_goal_,true);publishPath();consecutive_path_unsafe_=0;
      RCLCPP_INFO(get_logger(),"terrain topology certified prefix: %.2fm; final mission goal retained",length);
      return true;
    }

    if (buildContinuousCurvePathLocked(search_start, active_goal, continuous_path)) {
      if (committed) {
        continuous_path = mergeCommittedPrefix(committed_prefix, continuous_path);
        if (!chance_constrained_ && rear_axle_offset_==0.) refinePathForTrackingLocked(continuous_path);
      }
      updatePathArcLength(continuous_path);
      if ((!chance_constrained_ || (bodyArcPathFeasible(continuous_path) &&
           pathFootprintSafeLocked(continuous_path,.025))) &&
          chancePathSafe(continuous_path,nullptr,chance_path_budget_)) {
        size_t planned_path_size = 0;
        {
          std::lock_guard<std::mutex> lock(path_mutex_);
          if (!planSnapshotCurrent(goal_seq_snapshot, input_epoch)) return false;
          path_ = std::move(continuous_path);
          recovery_path_ = false;recovery_staged_gears_=false;recovery_legs_.clear();
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
    if (chance_constrained_) {
      const auto q=queryCostLocked(start.x,start.y,start.yaw);
      if (q.lethal || !std::isfinite(q.probability_upper)) {
        RCLCPP_WARN_THROTTLE(get_logger(),*get_clock(),1500,
          "Probability start rejected: known=%s upper=%.6f state_limit=%.3f (unknown is not free)",
          q.known ? "true" : "false",q.probability_upper,chance_state_limit_);
        return false;
      }
      start.probability_upper=q.probability_upper;
      auto events=std::make_shared<terrain_nav::ChanceBudget>();
      if (!addChancePose(start.x,start.y,start.yaw,*events,chance_path_budget_)) return false;
      start.probability_upper=events->upper(); start.probability_events=std::move(events);
    }
    const auto searchKey = [&](int ix,int iy,int yaw_bin,double upper) {
      const int bucket=chance_constrained_ ? std::clamp(
        static_cast<int>(15*upper/chance_path_budget_),0,15) : 0;
      return keyFor(ix,iy,yaw_bin)*16+bucket;
    };
    nodes.push_back(start);
    best_g[searchKey(start.ix, start.iy, start.yaw_bin,start.probability_upper)] = 0.0;
    open.push({start.f, 0});

    int goal_node = -1;
    int best_frontier = 0;
    double best_frontier_score = kInf;
    int iterations = 0;
    int reject_primitive = 0;
    int reject_outside = 0;
    int reject_progress = 0;
    int reject_duplicate = 0;
    bool early_partial=false;
    const auto steers = steerSamples();

    while (!open.empty()) {
      const int idx = open.top().second;
      open.pop();
      const SearchNode cur = nodes[static_cast<size_t>(idx)];
      ++iterations;

      const double h = heuristic(cur.x, cur.y, cur.yaw, active_goal);
      if (cur.length >= (chance_constrained_ ? .50 : 1.0)) {
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
      if(partialFrontierReady(search_start,active_goal,
          Pose2{nodes[best_frontier].x,nodes[best_frontier].y,nodes[best_frontier].yaw,nodes[best_frontier].length},
          elapsed_ms)) {early_partial=true;break;}
      if (iterations >= max_iterations_ || elapsed_ms >= max_search_time_ms_ || PlanningBudget::expired()) break;

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
        // The primitive extends the parent's source-event account. Overlapping
        // poses and boundary states do not create new independent terrain.
        const double next_probability=prim.probability_upper;
        if (chance_constrained_ && next_probability>chance_path_budget_) continue;
        const int64_t key = searchKey(ix,iy,yaw_bin,next_probability);
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
        next.yaw = prim.end.yaw;
        next.g = ng;
        next.f = ng + heuristic(next.x, next.y, next.yaw, active_goal);
        next.length = next_length;
        next.steer = steer;
        next.ix = ix;
        next.iy = iy;
        next.yaw_bin = yaw_bin;
        next.parent = idx;
        next.probability_upper = next_probability;
        next.probability_events = prim.probability_events;
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
      if (progress > (chance_constrained_ ? .45 : 1.0)) {
        chosen = best_frontier;
        partial = true;
      }
    }
    if (chosen < 0) {
      std::vector<Pose2> certified;
      if (buildCertifiedForwardPrefixLocked(Pose2{odom.x,odom.y,odom.yaw,0.},raw_goal_snapshot,certified)) {
        Pose2 endpoint=certified.back();
        {
          std::lock_guard<std::mutex> lock(path_mutex_);
          if (!planSnapshotCurrent(goal_seq_snapshot,input_epoch)) return false;
          path_=std::move(certified);active_goal_=endpoint;have_path_=true;recovery_path_=false;recovery_staged_gears_=false;recovery_legs_.clear();++path_seq_;
          observation_path_seq_=path_seq_;
          resetNominalControls();
        }
        publishActiveGoal(endpoint,true);publishPath();
        RCLCPP_INFO(get_logger(),"terrain certified forward prefix: %.2fm after search failure; raw goal retained",endpoint.s);
        return true;
      }
      const auto &frontier = nodes[static_cast<size_t>(best_frontier)];
      const double frontier_dist = std::hypot(frontier.x - search_start.x, frontier.y - search_start.y);
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 800,
                           "terrain lattice failed: iter=%d nodes=%zu frontier=(%.2f, %.2f) dist=%.2f goal=(%.2f, %.2f) reject primitive/outside/progress/duplicate=%d/%d/%d/%d se2_lethal=%s",
                           iterations, nodes.size(), frontier.x, frontier.y, frontier_dist,
                           active_goal.x, active_goal.y,
                           reject_primitive, reject_outside, reject_progress, reject_duplicate,
                           se2_risk_lethal_ ? "true" : "false");
      return try_mission_progress_fan() || try_observation_prefix();
    }

    std::vector<Pose2> path = extractPath(nodes, chosen);
    if(path.size()<2) return false;
    if (committed) {
      path = mergeCommittedPrefix(committed_prefix, path);
    }
    if(!chance_constrained_ && rear_axle_offset_==0.) refinePathForTrackingLocked(path);
    updatePathArcLength(path);
    if (chance_constrained_ && !bodyArcPathFeasible(path)) {
      RCLCPP_WARN_THROTTLE(get_logger(),*get_clock(),1000,"terrain lattice rejected inconsistent body-yaw/curvature reconstruction");
      return false;
    }
    if (!chancePathSafe(path,nullptr,chance_path_budget_)) {
      RCLCPP_WARN_THROTTLE(get_logger(),*get_clock(),1000,"terrain lattice rejected final path probability certificate");
      return false;
    }
    if(early_partial && !pathFootprintSafeLocked(path,.025))return false;
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
      return try_mission_progress_fan() || try_observation_prefix();
    }

    size_t planned_path_size = 0;
    {
      std::lock_guard<std::mutex> lock(path_mutex_);
      if (!planSnapshotCurrent(goal_seq_snapshot, input_epoch)) return false;
      path_ = std::move(path);
      recovery_path_ = false;recovery_staged_gears_=false;recovery_legs_.clear();
      active_goal_ = active_goal;
      have_path_ = path_.size() >= 2;
      planned_path_size = path_.size();
      path_seq_++;
      if (partial) observation_path_seq_=path_seq_;
      resetNominalControls();
    }

    publishPath();
    consecutive_path_unsafe_ = 0;
    if(early_partial)early_partial_commits_.fetch_add(1);
    const double elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_time).count();
    RCLCPP_INFO(
      get_logger(), "terrain lattice %s: %zu pts, %.1f ms, iter=%d, reason=%s early=%s",
      partial ? "partial" : "success", planned_path_size, elapsed_ms, iterations, goal_reason.c_str(),early_partial ? "true" : "false");
    return true;
  }

  bool continuationSearchUsefulLocked(const OdomState &odom,const std::vector<Pose2> &path)const
  {
    if(path.size()<2 || path.front().direction<0)return false;
    const size_t nearest=nearestPathIndex(path,odom.x,odom.y);
    return path.back().s-path[nearest].s<2.4 && knownHazardAheadLocked(path.back(),1.);
  }

  bool canRetainCertifiedForwardPathLocked(const OdomState &odom,const std::vector<Pose2> &path,bool recovery) const
  {
    if(!chance_constrained_ || recovery || path.size()<2 || path.front().direction<0)return false;
    double distance=0;const size_t nearest=nearestPathIndex(path,odom.x,odom.y,&distance);
    const auto reject=[&](const char *cause) {
      static rclcpp::Clock log_clock(RCL_STEADY_TIME);
      RCLCPP_INFO_THROTTLE(get_logger(),log_clock,500,
        "terrain retention rejected: cause=%s source_ns=%lld now_ns=%lld points=%zu end=(%.6f,%.6f) pose=(%.6f,%.6f,%.6f) nearest=%zu remaining=%.3f",
        cause,static_cast<long long>(map_stamp_.nanoseconds()),static_cast<long long>(now().nanoseconds()),
        path.size(),path.back().x,path.back().y,odom.x,odom.y,odom.yaw,nearest,path.back().s-path[nearest].s);
      return false;
    };
    if(distance>std::min(.15,replan_track_error_) ||
       std::abs(wrapAngle(odom.yaw-path[nearest].yaw))>.45 ||
       path.back().s-path[nearest].s<1.2)return reject("capture_or_remaining");
    std::vector<Pose2> remaining(path.begin()+nearest,path.end());
    remaining.insert(remaining.begin(),Pose2{odom.x,odom.y,odom.yaw,0.,1});
    // Revalidate against the CURRENT complete body, live columns, support
    // and joint risk account. No cached certificate or new timestamp is used.
    // The control loop independently checks dynamic/braking envelopes each
    // tick. An obsolete point-robot subgoal cannot replace this feasible car
    // trajectory merely because a new global search exhausted its budget.
    if(!bodyArcPathFeasible(path))return reject("kinematics");
    if(!pathFootprintSafeLocked(remaining,.025))return reject("current_body_or_support");
    if(!chancePathSafe(remaining,nullptr,chance_path_budget_))return reject("current_joint_budget");
    return true;
  }

  bool certifiedRemainingForwardPrefixLocked(const OdomState &odom,const std::vector<Pose2> &path,
      std::vector<Pose2> &prefix)const
  {
    prefix.clear();
    if(!chance_constrained_ || path.size()<2 || path.front().direction<0 || !bodyArcPathFeasible(path))return false;
    double distance=0.;const size_t nearest=nearestPathIndex(path,odom.x,odom.y,&distance);
    if(distance>std::min(.15,replan_track_error_) ||
       std::abs(wrapAngle(odom.yaw-path[nearest].yaw))>.3)return false;
    std::vector<Pose2> remaining(path.begin()+nearest,path.end());updatePathArcLength(remaining);
    // A far failed sample revokes only that suffix. The original turning
    // geometry stays fixed, and EVERY retained sample gets a CURRENT body,
    // live-obstacle, support and joint-budget check. Never bridge a gap.
    if(!buildCertifiedGuidePrefixLocked(remaining,prefix) || prefix.back().s<1.2 ||
       prefix.back().s+.05>=remaining.back().s)return false;
    return true;
  }

  bool preferCertifiedRemainingPathLocked(const OdomState &odom,const Pose2 &mission,
      const std::vector<Pose2> &previous,const std::vector<Pose2> &candidate,bool recovery) const
  {
    if(candidate.size()<2 || candidate.front().direction<0 || previous.size()<2 || recovery)return false;
    const size_t nearest=nearestPathIndex(previous,odom.x,odom.y);
    const double remaining=previous.back().s-previous[nearest].s;
    const bool loses_progress=candidate.back().s+.4<remaining &&
       std::hypot(candidate.back().x-mission.x,candidate.back().y-mission.y)>
         std::hypot(previous.back().x-mission.x,previous.back().y-mission.y)+.3;
    // A completed optimistic guide is not automatically a BETTER near-term
    // reference. Replacing a certified left approach by a longer right curve
    // made cold starts circle between gates while both branches were safe.
    // Compare the actual next vehicle-length of the two curves, anchored at
    // their current capture points. Same-approach extensions remain free;
    // divergent replacements wait only while the ENTIRE useful old curve
    // still passes CURRENT body/support/joint proof. The 1.2m retention floor
    // releases the choice before the old curve is consumed. This never
    // retains an invalid path, a stop-only prefix or a recovery gear leg.
    const size_t candidate_nearest=nearestPathIndex(candidate,odom.x,odom.y);
    const auto body=collisionEnvelope(footprint_margin_);
    const double horizon=std::min({std::max(.5,body.front+body.back),remaining,
      candidate.back().s-candidate[candidate_nearest].s});
    bool switches_approach=false;
    if(horizon>=.5) {
      const auto old_future=samplePathAt(previous,previous[nearest].s+horizon);
      const auto new_future=samplePathAt(candidate,candidate[candidate_nearest].s+horizon);
      switches_approach=std::hypot(old_future.x-new_future.x,old_future.y-new_future.y)>.15 ||
        std::abs(wrapAngle(old_future.yaw-new_future.yaw))>.3;
    }
    return (loses_progress || switches_approach) && canRetainCertifiedForwardPathLocked(odom,previous,false);
  }

  bool partialFrontierReady(const Pose2 &start,const Pose2 &goal,const Pose2 &frontier,double elapsed_ms) const
  {
    if(!chance_constrained_ || !accept_partial_progress_ || partial_search_time_ms_<=0 ||
        !std::isfinite(elapsed_ms) || elapsed_ms<partial_search_time_ms_)return false;
    const double distance=std::hypot(goal.x-start.x,goal.y-start.y);
    const double remaining=std::hypot(goal.x-frontier.x,goal.y-frontier.y);
    // Only stop improving an existing useful frontier. Reconstruction and all
    // original kinematic/probability checks still run before publication.
    return distance>2. && frontier.s>=.8 && distance-remaining>=.3 &&
      std::hypot(frontier.x-start.x,frontier.y-start.y)>=.75;
  }

  bool reverseGuideTarget(const Pose2 &start,const Pose2 &goal,
                          const std::vector<Pose2> &trail,Pose2 &target) const
  {
    if (trail.size()<3) return false;
    const size_t current=nearestPathIndex(trail,start.x,start.y);
    const size_t end=nearestPathIndex(trail,goal.x,goal.y);
    if (end>=current || std::hypot(trail[current].x-start.x,trail[current].y-start.y)>.30 ||
        std::hypot(trail[end].x-goal.x,trail[end].y-goal.y)>goal_tolerance_ ||
        std::abs(wrapAngle(trail[current].yaw-start.yaw))>.45) return false;
    double length=std::hypot(trail[current].x-start.x,trail[current].y-start.y);
    size_t index=current;
    while (index>end && length<.80) {
      length+=std::hypot(trail[index].x-trail[index-1].x,trail[index].y-trail[index-1].y);
      --index;
    }
    target=index==end?goal:trail[index];
    const double dx=target.x-start.x,dy=target.y-start.y;
    return dx*std::cos(start.yaw)+dy*std::sin(start.yaw)<-.30 &&
      std::hypot(target.x-goal.x,target.y-goal.y)<std::hypot(start.x-goal.x,start.y-goal.y)-.20;
  }

  bool buildCertifiedTerminalManeuverLocked(const Pose2 &start,const Pose2 &goal,
      std::vector<std::vector<Pose2>> &legs) const
  {
    legs.clear();
    if(!chance_constrained_ || !reverse_enable_ || !reverse_recovery_enable_ ||
        std::hypot(goal.x-start.x,goal.y-start.y)>1.5 || PlanningBudget::expired())return false;
    PlanningBudget terminal_budget(std::min(20.,PlanningBudget::remaining()));
    const auto candidates=terrain_nav::terminalGearManeuvers(start.x,start.y,start.yaw,
      goal.x,goal.y,goal_tolerance_,std::min(3.,reverse_max_distance_),maxReferenceCurvature(),.025,rear_axle_offset_);
    for(const auto &candidate:candidates) {
      if(PlanningBudget::expired())break;
      std::vector<std::vector<Pose2>> proposal(2);
      for(size_t leg=0;leg<2;++leg) {
        const auto &source=leg?candidate.second:candidate.first;
        for(const auto &p:source)proposal[leg].push_back(Pose2{p[0],p[1],wrapAngle(p[2]),0.,
          leg?-candidate.first_gear:candidate.first_gear});
        updatePathArcLength(proposal[leg]);
      }
      // One source-event account covers BOTH legs. Independent per-leg
      // admission would silently spend the path risk limit twice.
      terrain_nav::ChanceBudget account;bool safe=true;
      for(const auto &leg:proposal) {
        if(!bodyArcPathFeasible(leg)){safe=false;break;}
        for(const auto &p:leg) {
          double cost=0.,clearance=kInf;
          if(!footprintCostLocked(p.x,p.y,p.yaw,cost,&clearance) ||
             (std::isfinite(clearance)&&clearance<hardClearanceFloor()) ||
             !addChancePose(p.x,p.y,p.yaw,account,chance_path_budget_)){safe=false;break;}
        }
        if(!safe)break;
      }
      if(safe){legs=std::move(proposal);return true;}
    }
    return false;
  }

  bool buildCertifiedReverseArcLocked(const Pose2 &start,const Pose2 &goal,
                                     std::vector<Pose2> &prefix) const
  {
    prefix.clear();
    auto arc=terrain_nav::reverseArc(start.x,start.y,start.yaw,goal.x,goal.y,
                                           reverse_max_distance_,maxReferenceCurvature(),.05,rear_axle_offset_);
    auto candidates=terrain_nav::reverseGoalRegionArcs(start.x,start.y,start.yaw,goal.x,goal.y,
        goal_tolerance_,std::min(reverse_max_distance_,
          1.5*std::hypot(goal.x-start.x,goal.y-start.y)),maxReferenceCurvature(),.025,rear_axle_offset_);
    if(!arc.empty()) candidates.insert(candidates.begin(),std::move(arc));
    // Keep the analytic body yaw; chord tangents shift the initial pose into
    // a different heading bin and are not the arc's boundary condition.
    bool best_terminal=false;double best_score=kInf;
    for(const auto &candidate:candidates) {
      std::vector<Pose2> path,certified;
      for(const auto &p:candidate) path.push_back(Pose2{p[0],p[1],wrapAngle(p[2]),0,-1});
      updatePathArcLength(path);
      if(!certifyReversePrefixLocked(path,goal,certified)) continue;
      const double remaining=std::hypot(certified.back().x-goal.x,certified.back().y-goal.y);
      // Keep the exact connector when fully certified; the disk is a fallback.
      if(remaining<=1e-5) {prefix=std::move(certified);return true;}
      const bool terminal=remaining<=goal_tolerance_;
      const double score=terminal?certified.back().s:remaining+.05*certified.back().s;
      if((terminal && !best_terminal) || (terminal==best_terminal && score<best_score)) {
        best_terminal=terminal;best_score=score;prefix=std::move(certified);
      }
    }
    return prefix.size()>=2;
  }

  bool certifyReversePrefixLocked(const std::vector<Pose2> &path,const Pose2 &goal,
                                  std::vector<Pose2> &prefix) const
  {
    prefix.clear();
    if(path.size()<2) return false;
    const double length=path.back().s;
    const double spacing=std::min(.05,.25*resolution_);
    const int samples=std::max(1,static_cast<int>(std::ceil(length/spacing)));
    for (int i=0;i<=samples;++i) {
      auto sample=samplePathAt(path,length*i/samples);
      double cost=0,clearance=kInf;
      prefix.push_back(sample);
      if (!footprintCostLocked(sample.x,sample.y,sample.yaw,cost,&clearance,nullptr,true) ||
          (std::isfinite(clearance) && clearance<hardClearanceFloor()) ||
          !chancePathSafe(prefix,nullptr,chance_path_budget_)) {
        prefix.pop_back();break;
      }
    }
    const bool terminal=!prefix.empty() &&
      std::hypot(prefix.back().x-goal.x,prefix.back().y-goal.y)<=goal_tolerance_;
    if (prefix.size()<2 || (prefix.back().s<length-1e-6 && prefix.back().s<.25 && !terminal) ||
        !chancePathSafe(prefix,nullptr,chance_path_budget_)) {prefix.clear();return false;}
    return true;
  }

  bool buildCertifiedForwardPrefixLocked(
    const Pose2 &start,const Pose2 &goal,std::vector<Pose2> &prefix) const
  {
    prefix.clear();
    if (!chance_constrained_ || !accept_partial_progress_ || PlanningBudget::expired()) return false;
    const double dx=goal.x-start.x,dy=goal.y-start.y,distance=std::hypot(dx,dy);
    const double forward=dx*std::cos(start.yaw)+dy*std::sin(start.yaw);
    const double lateral=-dx*std::sin(start.yaw)+dy*std::cos(start.yaw);
    if (distance<goal_reached_distance_+.25 || forward<=0 ||
        std::abs(std::atan2(lateral,forward))>.65) return false;
    if(knownHazardAheadLocked(start,std::min(distance,1.2+footprint_front_+clearance_distance_)))
      return false;
    const auto connection=terrain_nav::connectBodyArc(forward,lateral,1,rear_axle_offset_);
    if(!connection.valid)return false;
    const double curvature=std::clamp(connection.curvature,
                                     -maxReferenceCurvature(),maxReferenceCurvature());
    const double ds=std::min(.25*resolution_,.25*map_.getResolutionYaw()/std::max(1e-6,std::abs(curvature)));
    const double limit=std::min(1.2,distance);
    for (double s=0;s<=limit+1e-6;s+=ds) {
      auto q=advanceArc(start,curvature,s);q.s=s;
      double cost=0,clearance=kInf;
      prefix.push_back(q);
      // A discretization fallback, not permission to relax support or budgets.
      if (!footprintCostLocked(q.x,q.y,q.yaw,cost,&clearance) ||
          (std::isfinite(clearance) && clearance<hardClearanceFloor()) ||
          !chancePathSafe(prefix,nullptr,chance_path_budget_)) {
        prefix.pop_back();break;
      }
    }
    if (prefix.size()<2 || prefix.back().s<.25 ||
        !pathKinematicallyFeasible(prefix) || !chancePathSafe(prefix,nullptr,chance_path_budget_)) {
      prefix.clear();return false;
    }
    return true;
  }

  bool buildMissionProgressFanLocked(const Pose2 &start,const Pose2 &goal,std::vector<Pose2> &best) const
  {
    best.clear();
    if(!chance_constrained_ || !accept_partial_progress_)return false;
    const double distance=std::hypot(goal.x-start.x,goal.y-start.y);
    const double bearing=std::atan2(goal.y-start.y,goal.x-start.x);
    if(distance<1.5 || std::cos(bearing-start.yaw)<=0)return false;
    const double initial_gap=std::abs(wrapAngle(bearing-start.yaw));
    double best_score=kInf;
    // A rolling guide can fall behind the car after its topology search fails.
    // Do not reject useful, certified turning solely for moving away from that
    // guide. This bounded last-resort fan measures progress to the unchanged
    // final mission goal; it never executes its optimistic unknown route.
    for(double steer:steerSamples()) {
      if(PlanningBudget::expired())break;
      std::vector<Pose2> path{start};terrain_nav::ChanceBudget account;
      if(!addChancePose(start.x,start.y,start.yaw,account,chance_path_budget_))continue;
      double integrated_cost=0;
      for(int i=1;i<=48;++i) {
        auto p=advanceArc(path.back(),primitiveCurvature(steer),.025);p.s=i*.025;
        double cost=0,clearance=kInf;
        if(!footprintCostLocked(p.x,p.y,p.yaw,cost,&clearance) ||
           (std::isfinite(clearance)&&clearance<hardClearanceFloor()) ||
           !addChancePose(p.x,p.y,p.yaw,account,chance_path_budget_))break;
        path.push_back(p);integrated_cost+=cost*.025;
        const double remaining=std::hypot(goal.x-p.x,goal.y-p.y);
        const double gap=std::abs(wrapAngle(std::atan2(goal.y-p.y,goal.x-p.x)-p.yaw));
        if(p.s<.8 || std::hypot(p.x-start.x,p.y-start.y)<.75 ||
           distance-remaining<.45 || (initial_gap>.35 && gap>initial_gap-.1))continue;
        const double score=remaining+.3*gap+.15*integrated_cost+.05*p.s;
        if(score<best_score && bodyArcPathFeasible(path) && pathFootprintSafeLocked(path,.025) &&
           chancePathSafe(path,nullptr,chance_path_budget_)) {best=path;best_score=score;}
      }
    }
    return best.size()>=2;
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

    for (size_t ci = 1; ci < chain.size(); ++ci) {
      const auto &parent = nodes[static_cast<size_t>(chain[ci - 1])];
      const auto &child = nodes[static_cast<size_t>(chain[ci])];
      double x = parent.x;
      double y = parent.y;
      double yaw = parent.yaw;
      const int steps = primitiveIntegrationSteps();
      const double ds = primitive_length_ / static_cast<double>(steps);
      const double curvature = primitiveCurvature(child.steer);
      for (int s = 1; s <= steps; ++s) {
        const auto next=advanceArc(Pose2{x,y,yaw,0.},curvature,ds);
        x=next.x;y=next.y;yaw=next.yaw;
        Pose2 p{x, y, yaw, 0.0};
        if (s == steps) {
          if(std::hypot(p.x-child.x,p.y-child.y)>1e-6 ||
              std::abs(wrapAngle(p.yaw-child.yaw))>1e-6) return {};
        }
        if (std::hypot(p.x - path.back().x, p.y - path.back().y) > 1e-8) {
          path.push_back(p);
        }
      }
    }

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

  double observationPrefixLimit(const Pose2 &start,const Pose2 &goal) const
  {
    const double dx=goal.x-start.x,dy=goal.y-start.y;
    const double angle=std::abs(wrapAngle(std::atan2(dy,dx)-start.yaw));
    if(angle<.15)return std::min(1.2,std::hypot(dx,dy));
    if(angle>.65)return 0;
    const double forward=std::cos(start.yaw)*dx+std::sin(start.yaw)*dy;
    const double lateral=std::abs(-std::sin(start.yaw)*dx+std::cos(start.yaw)*dy);
    // A circle tangent to the current heading reaches (x,y) with curvature
    // 2*y/(x*x+y*y). Keep 20% steering reserve after the observation prefix.
    const double required=std::sqrt(std::max(0.,2*lateral/(.8*maxReferenceCurvature())-lateral*lateral));
    return std::clamp(forward-required,0.,.4);
  }

  double maxReferenceCurvature() const
  {
    return reference_max_curvature_ratio_ * std::tan(max_steer_) / std::max(1e-3, wheel_base_);
  }

  Pose2 advanceArc(Pose2 pose,double curvature,double distance) const
  {
    return ::advanceArc(pose,curvature,distance,rear_axle_offset_);
  }

  double primitiveCurvature(double steering) const
  {
    return std::clamp(std::tan(steering)/std::max(1e-3,wheel_base_),
                      -maxReferenceCurvature(),maxReferenceCurvature());
  }

  int primitiveIntegrationSteps() const
  {
    double step=std::max(.01,integration_step_);
    if(chance_constrained_) step=std::min({step,.25*resolution_,
      .25*(2*M_PI/std::max(1,map_.getSizeYaw()))/std::max(1e-6,maxReferenceCurvature())});
    return std::max(2,static_cast<int>(std::ceil(primitive_length_/step)));
  }

  bool bodyArcPathFeasible(const std::vector<Pose2> &path) const
  {
    if(path.size()<2)return false;
    for(size_t i=1;i<path.size();++i) {
      const auto &a=path[i-1],&b=path[i];
      const auto ar=terrain_nav::rearAxlePose(a,rear_axle_offset_);
      const auto br=terrain_nav::rearAxlePose(b,rear_axle_offset_);
      const double dx=br.x-ar.x,dy=br.y-ar.y,ds=std::hypot(dx,dy);
      const double turn=wrapAngle(b.yaw-a.yaw);
      if(!std::isfinite(ds) || ds<1e-9 || !std::isfinite(turn) ||
          (a.direction!=1 && a.direction!=-1) || a.direction!=b.direction ||
          std::abs(2*std::sin(.5*turn))/ds>maxReferenceCurvature()+1e-6 ||
          std::abs(wrapAngle(std::atan2(dy,dx)-a.yaw-.5*turn-(a.direction<0?M_PI:0.)))>.03) return false;
    }
    return true;
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
    if(rear_axle_offset_>0.)return bodyArcPathFeasible(path);
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
    if (chord < 0.45) {
      const double terminal_error=std::hypot(path.back().x-goal.x,path.back().y-goal.y);
      // A complete terminal connector can be shorter than a useful rolling
      // frontier. Keep the frontier guard without rejecting a reachable goal.
      if(partial || terminal_error>goal_tolerance_+1e-6 ||
         chord<1e-3 || terminal_error>=direct-1e-3) return false;
    }

    const double length = path.back().s;
    const double detour_limit = chord * (partial ? 1.80 : 2.05) + (partial ? 0.75 : 1.05);
    if (length > detour_limit) return false;

    if (direct > 0.8) {
      const double ux = (goal.x - start.x) / direct;
      const double uy = (goal.y - start.y) / direct;
      const double end_progress = (path.back().x - start.x) * ux + (path.back().y - start.y) * uy;
      const double minimum_progress=chance_constrained_ ? .45 :
        std::max(0.90,0.15*std::min(direct,local_goal_distance_));
      if (partial && end_progress < minimum_progress) {
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
    std::vector<Pose2> &path, bool reverse = false) const
  {
    path.clear();
    path.push_back(Pose2{start.x,start.y,start.yaw,0,reverse?-1:1});

    const auto rear_start=terrain_nav::rearAxlePose(start,rear_axle_offset_);
    double x = rear_start.x;
    double y = rear_start.y;
    double yaw = wrapAngle(start.yaw+(reverse?M_PI:0.));
    const double ds_nominal = reverse ? std::min(.05,.25*resolution_) :
      std::clamp(reference_resample_spacing_ * 0.65, 0.08, 0.16);

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
        path.push_back(terrain_nav::baseLinkPose(Pose2{x,y,wrapAngle(yaw-(reverse?M_PI:0.)),0,reverse?-1:1},rear_axle_offset_));
      }
    }

    if (path.size() < 3) return false;
    // Do not hide a connector error by snapping its last point to the goal.
    if (std::hypot(path.back().x-goal.x,path.back().y-goal.y)>1e-5 ||
        std::abs(wrapAngle(path.back().yaw-goal.yaw))>1e-5) return false;
    if (!reverse && !chance_constrained_ && rear_axle_offset_==0.) updateYawFromTangent(path);
    updatePathArcLength(path);
    return true;
  }

  bool buildCertifiedReversePoseLocked(const Pose2 &start,const Pose2 &goal,
                                      std::vector<Pose2> &best) const
  {
    best.clear();
    const auto rear_start=terrain_nav::rearAxlePose(start,rear_axle_offset_);
    const auto rear_goal=terrain_nav::rearAxlePose(goal,rear_axle_offset_);
    const double dx=rear_goal.x-rear_start.x,dy=rear_goal.y-rear_start.y,distance=std::hypot(dx,dy);
    if (distance<.05 || distance>reverse_max_distance_ ||
        dx*std::cos(start.yaw)+dy*std::sin(start.yaw)>=0) return false;
    const double bearing=std::atan2(dy,dx);
    const double alpha=mod2pi(start.yaw+M_PI-bearing),beta=mod2pi(goal.yaw+M_PI-bearing);
    double best_score=kInf;
    std::vector<Pose2> best_prefix;
    double prefix_score=kInf;
    for (double fraction : {.4,.7,1.}) {
      const double radius=1./(fraction*maxReferenceCurvature()),d=distance/radius;
      std::array<DubinsCandidate,6> candidates;
      const std::array<bool,6> valid={dubinsLSL(alpha,beta,d,candidates[0]),
        dubinsRSR(alpha,beta,d,candidates[1]),dubinsLSR(alpha,beta,d,candidates[2]),
        dubinsRSL(alpha,beta,d,candidates[3]),dubinsRLR(alpha,beta,d,candidates[4]),
        dubinsLRL(alpha,beta,d,candidates[5])};
      for (size_t i=0;i<candidates.size();++i) {
        const auto &candidate=candidates[i];
        const double length=candidate.length*radius;
        if (!valid[i] || length>1.5*distance+.10) continue;
        std::vector<Pose2> path;
        if (!sampleDubinsLocked(start,goal,candidate,radius,path,true)) continue;
        std::vector<Pose2> prefix;
        if(!certifyReversePrefixLocked(path,goal,prefix)) continue;
        const double score=length+.15*fraction*(candidate.params[0]+candidate.params[2]);
        const double remaining=std::hypot(prefix.back().x-goal.x,prefix.back().y-goal.y);
        if(prefix.back().s<path.back().s-1e-6) {
          // Preserve the certified connector's heading evolution instead of
          // replacing it with a position-only arc after a budget truncation.
          const double partial_score=remaining+.2*std::abs(wrapAngle(prefix.back().yaw-goal.yaw));
          if(distance-remaining>=.20 && partial_score<prefix_score) {
            prefix_score=partial_score;best_prefix=std::move(prefix);
          }
        } else if (score<best_score) {best_score=score;best=std::move(prefix);}
      }
    }
    if(best.empty()) best=std::move(best_prefix);
    return best.size()>=3;
  }

  bool buildDubinsDirectPathLocked(
    const Pose2 &start,
    const Pose2 &goal,
    std::vector<Pose2> &best_path) const
  {
    best_path.clear();
    const auto rear_start=terrain_nav::rearAxlePose(start,rear_axle_offset_);
    const auto rear_goal=terrain_nav::rearAxlePose(goal,rear_axle_offset_);
    const double dx = rear_goal.x - rear_start.x;
    const double dy = rear_goal.y - rear_start.y;
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
    if (!chance_constrained_ && rear_axle_offset_==0.) updateYawFromTangent(path);
    updatePathArcLength(path);

    const double max_length = direct_dist * direct_curve_max_length_ratio_ + 0.40;
    if (path.back().s > max_length) return false;
    if (chance_constrained_ &&
        (std::abs(wrapAngle(path.front().yaw-start.yaw))>1e-6 || !bodyArcPathFeasible(path))) return false;
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

    const auto rear_start=terrain_nav::rearAxlePose(start,rear_axle_offset_);
    const auto rear_goal=terrain_nav::rearAxlePose(goal,rear_axle_offset_);
    const double dx = rear_goal.x - rear_start.x;
    const double dy = rear_goal.y - rear_start.y;
    const double direct_dist = std::hypot(dx, dy);
    if (direct_dist < 0.65 || direct_dist > local_goal_distance_ + 0.8) return false;

    const double goal_dir = std::atan2(dy, dx);
    const double heading_gap = wrapAngle(start.yaw - goal_dir);
    const double end_heading_gap = wrapAngle(goal.yaw - goal_dir);
    if (chance_constrained_ &&
        (std::abs(heading_gap)>=M_PI/2 || std::abs(end_heading_gap)>=M_PI/2)) return false;
    const std::array<double, 5> lateral_offsets = {0.0, 0.55, -0.55, 0.95, -0.95};
    const std::array<double, 3> start_slopes = {
      0.0,
      std::tan(std::clamp(0.35 * heading_gap, -0.38, 0.38)),
      std::tan(std::clamp(0.60 * heading_gap, -0.55, 0.55)),
    };
    const int samples = std::max(18, static_cast<int>(std::ceil(direct_dist / 0.14)));

    double best_score = kInf;
    const Eigen::Vector2d origin(rear_start.x, rear_start.y);
    const Eigen::Vector2d ex(std::cos(goal_dir), std::sin(goal_dir));
    const Eigen::Vector2d ey(-std::sin(goal_dir), std::cos(goal_dir));

    for (size_t slope=0;slope<(chance_constrained_?1u:start_slopes.size());++slope) {
      const double m0=chance_constrained_?std::tan(heading_gap):start_slopes[slope];
      const double b0 = 0.0;
      const double b1 = m0 * direct_dist;
      const double b2 = 0.0;
      const double m1 = chance_constrained_?std::tan(end_heading_gap):0.0;
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
          const double derivative = b1 + 2*b2*t + 3*b3*t*t + 4*b4*t*t*t + 5*b5*t*t*t*t +
            offset*192*t*t*(1-t)*(1-t)*(1-2*t);
          const double yaw=chance_constrained_?wrapAngle(goal_dir+std::atan2(derivative,direct_dist)):0.0;
          candidate.push_back(terrain_nav::baseLinkPose(Pose2{p.x(), p.y(), yaw, 0.0},rear_axle_offset_));
        }
        candidate.front() = start;
        candidate.back() = goal;
        if (!chance_constrained_ && rear_axle_offset_==0.) updateYawFromTangent(candidate);
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
    if (chance_constrained_) return bodyArcPathFeasible(best_path) && pathFootprintSafeLocked(best_path,.025);
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

    const auto connection=terrain_nav::connectBodyArc(local_x,local_y,1,rear_axle_offset_);
    if(!connection.valid)return false;
    const double curvature=connection.curvature;
    if(std::abs(curvature)>maxReferenceCurvature()+1e-6)return false;
    const double arc_length=connection.distance;
    if (arc_length > direct_dist * direct_curve_max_length_ratio_ + 0.40) return false;

    const int samples = std::max(8, static_cast<int>(std::ceil(arc_length / 0.14)));
    best_path.reserve(static_cast<size_t>(samples) + 1);
    for (int i = 0; i <= samples; ++i) {
      const double u = static_cast<double>(i) / static_cast<double>(samples);
      const double ds = arc_length * u;
      auto p=advanceArc(start,curvature,ds);best_path.push_back(p);
    }
    if(std::hypot(best_path.back().x-goal.x,best_path.back().y-goal.y)>1e-6)return false;
    if (!chance_constrained_ && rear_axle_offset_==0.) updateYawFromTangent(best_path);
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
    const double distance=std::hypot(goal.x-start.x,goal.y-start.y);
    const double direction=std::atan2(goal.y-start.y,goal.x-start.x);
    double best_score=kInf;
    if(buildArcDirectPathLocked(start,goal,best_path)) {
      const bool historical=std::any_of(best_path.begin(),best_path.end(),
        [&](const auto &p){return historicalObstacleCost(p.x,p.y)>1e-6;});
      if(!historical)return true;
      evaluateContinuousCurveLocked(best_path,start,goal,distance,direction,best_score);
    } else best_path.clear();
    // A first-feasible shortcut must not ignore the route objective when a
    // historical obstacle preference is present. All candidates stay certified.
    for(int family=0;family<2;++family) {
      std::vector<Pose2> candidate;
      const bool built=family==0 ? buildDubinsDirectPathLocked(start,goal,candidate) :
        buildFrenetDirectPathLocked(start,goal,candidate);
      double score=kInf;
      if(built && evaluateContinuousCurveLocked(candidate,start,goal,distance,direction,score) && score<best_score) {
        best_score=score;best_path=std::move(candidate);
      }
      if(built && !historical_advisory_present_)break;
    }
    return std::isfinite(best_score) && best_path.size()>=3;
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
    if(!chance_constrained_ && rear_axle_offset_==0.)updateYawFromTangent(tmp);
    updatePathArcLength(tmp);
    const double ds = chance_constrained_?std::clamp(step,.01,std::max(.01,.25*resolution_)):
      std::clamp(step, 0.05, std::max(0.06, 0.45 * resolution_));
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
    return chancePathSafe(tmp, nullptr, chance_path_budget_);
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
    if(rear_axle_offset_>0. && a.direction==b.direction) {
      const auto ar=terrain_nav::rearAxlePose(a,rear_axle_offset_);
      const auto br=terrain_nav::rearAxlePose(b,rear_axle_offset_);
      const double chord=std::hypot(br.x-ar.x,br.y-ar.y),turn=wrapAngle(b.yaw-a.yaw);
      if(chord>1e-9) {
        const double k=2.*std::sin(.5*turn)/(a.direction*chord);
        const double length=std::abs(k)>1e-8?std::abs(turn/k)*terrain_nav::bodyArcScale(k,rear_axle_offset_):chord;
        const auto end=advanceArc(a,k,a.direction*length);
        if(std::hypot(end.x-b.x,end.y-b.y)<1e-6) {
          auto out=advanceArc(a,k,a.direction*length*u);out.s=s;return out;
        }
      }
    }
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
    bool recovery=false;
    {
      std::lock_guard<std::mutex> lock(path_mutex_);
      have_path = have_path_;
      path = path_;
      goal = active_goal_;
      recovery=recovery_path_;
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
    if(recovery) {
      if(map_dirty_.load() && pathAheadUnsafe(path,nearest)) {
        last_replan_reason_="updated terrain map invalidated path ahead";
        return true;
      }
      map_dirty_=false;
      if(remaining<=.025 && std::abs(odom.v)<=.03 && !recently_planned) {
        last_replan_reason_="recovery leg complete";
        return true;
      }
      return false;
    }
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
    if (path.front().direction<0) return false;
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
    std::vector<Pose2> probability_path;
    int severe_samples = 0;
    int unknown_samples = 0;
    for (double s = start_s; s <= end_s + 1e-6; s += 0.25) {
      const Pose2 p = samplePathAt(path, s);
      if (chance_constrained_) probability_path.push_back(p);
      double fc = 0.0;
      double clearance = kInf;
      if (!footprintCostLocked(p.x, p.y, p.yaw, fc, &clearance, nullptr, false, snapshot.get())) {
        return true;
      }
      if (std::isfinite(clearance) && clearance < hardClearanceFloor()) {
        return true;
      }
      if(!chance_constrained_) {
        // Legacy scalar-cost heuristics are not probability certificates.
        // In probability mode the full body + its single margin was already
        // checked above. Adding a .338m clearance at EVERY body cell closed
        // valid narrow lanes and disagreed with the planner's own validator.
        if(fc>=.75*unknown_cost_)++unknown_samples;
        if(fc>risk_weight_*1.65 || (std::isfinite(clearance) &&
            clearance<std::max(.25,.52*clearance_distance_)))++severe_samples;
        if(severe_samples>=3 || unknown_samples>=4)return true;
      }
    }
    if (chance_constrained_) probability_path.push_back(samplePathAt(path,end_s));
    return !chancePathSafe(probability_path,snapshot.get(),chance_path_budget_);
  }

  bool certifyUnchangedPathPrefix(const OdomState &odom,const std::vector<Pose2> &path,
      const ControlMapSnapshot &map,const LiveObstacleSnapshot *live,std::vector<Pose2> &prefix)const
  {
    prefix.clear();
    if(!chance_constrained_ || path.size()<2 || std::abs(path.front().direction)!=1 ||
       std::any_of(path.begin(),path.end(),[&](const Pose2 &p){return p.direction!=path.front().direction;}) ||
       !bodyArcPathFeasible(path))return false;
    double distance=0.;const size_t nearest=nearestPathIndex(path,odom.x,odom.y,&distance);
    if(distance>.15 || std::abs(wrapAngle(odom.yaw-path[nearest].yaw))>.45)return false;
    terrain_nav::ChanceBudget budget;
    const auto safe=[&](const Pose2 &p) {
      double cost=0.,clearance=kInf;
      const bool geometry=footprintCostLocked(p.x,p.y,p.yaw,cost,&clearance,nullptr,false,&map,live) &&
        (!std::isfinite(clearance)||clearance>=hardClearanceFloor());
      const bool joint=geometry && addChancePose(p.x,p.y,p.yaw,budget,chance_path_budget_,&map,live);
      if(!joint) {
        // Preserve the identity of the OLD reference before its invalid
        // suffix is removed. A passive recorder can source-match this event
        // to the original map, scan and published path; no diagnostic input
        // changes the safety decision or grants a replacement certificate.
        static rclcpp::Clock diagnostic_clock(RCL_STEADY_TIME);
        RCLCPP_INFO_THROTTLE(get_logger(),diagnostic_clock,500,
          "terrain retention rejected: cause=prefix_certificate source_ns=%lld now_ns=%lld points=%zu end=(%.6f,%.6f) pose=(%.6f,%.6f,%.6f) nearest=%zu remaining=%.3f sample=(%.6f,%.6f,%.6f) geometry=%d joint=%d",
          static_cast<long long>(map.stamp.nanoseconds()),static_cast<long long>(now().nanoseconds()),
          path.size(),path.back().x,path.back().y,odom.x,odom.y,odom.yaw,nearest,
          path.back().s-path[nearest].s,p.x,p.y,p.yaw,geometry,joint);
      }
      return joint;
    };
    if(!safe(Pose2{odom.x,odom.y,odom.yaw}) || !safe(path[nearest]))return false;
    auto first=path[nearest];first.s=0.;prefix.push_back(first);
    const double end=std::min(path.back().s,path[nearest].s+4.);
    bool rejected=false;
    for(double s=path[nearest].s+.025;s<=end+1e-8;s+=.025) {
      auto p=samplePathAt(path,std::min(s,end));
      if(!safe(p)){rejected=true;break;}
      p.s=s-path[nearest].s;prefix.push_back(p);
    }
    // This is a newly certified FINITE path. It never jumps an unsupported
    // sample or moves the original curve away from its world coordinates.
    return rejected && prefix.size()>1 && prefix.back().s>=.25 &&
      prefix.back().s+.01<path.back().s-path[nearest].s && bodyArcPathFeasible(prefix);
  }

  bool retainFinitePathAtGuideHandoffLocked(const OdomState &odom,uint64_t goal_sequence,
      uint64_t epoch,uint64_t searched_path_sequence)
  {
    if(!planSnapshotCurrent(goal_sequence,epoch))return false;
    std::vector<Pose2> current;uint64_t sequence=0;bool recovering=false;
    {std::lock_guard<std::mutex> guard(path_mutex_);
      if(!have_path_)return false;current=path_;sequence=path_seq_;recovering=recovery_path_;}
    if(current.size()>=2) {
      double distance=0.;const size_t nearest=nearestPathIndex(current,odom.x,odom.y,&distance);
      std::vector<Pose2> remaining(current.begin()+nearest,current.end());
      remaining.insert(remaining.begin(),Pose2{odom.x,odom.y,odom.yaw,0.,current.front().direction});
      if(distance<=.15 && std::abs(wrapAngle(odom.yaw-current[nearest].yaw))<=.45 &&
         current.back().s-current[nearest].s>.15 && bodyArcPathFeasible(current) &&
         pathFootprintSafeLocked(remaining,.025) && chancePathSafe(remaining,nullptr,chance_path_budget_))return true;
      if(sequence==searched_path_sequence && repairInvalidPathSuffix(odom,current,sequence))return true;
    }
    // A failed OLD search is not the owner of a newer control certificate.
    // Even when the newer reference itself fails fresh proof, control stops
    // it independently; this obsolete worker must not erase its identity.
    std::lock_guard<std::mutex> guard(path_mutex_);
    if(planSnapshotCurrent(goal_sequence,epoch) && path_seq_==searched_path_sequence && !recovering && !recovery_path_) {
      path_.clear();have_path_=false;++path_seq_;
    }
    return false;
  }

  bool repairInvalidPathSuffix(const OdomState &odom,std::vector<Pose2> &path,uint64_t sequence)
  {
    const auto epoch=input_epoch_.load();
    bool recovering=false;
    {std::lock_guard<std::mutex> guard(path_mutex_);
      if(!have_path_ || path_seq_!=sequence)return false;recovering=recovery_path_;}
    const auto map=std::atomic_load(&control_map_snapshot_);
    const auto live=std::atomic_load(&live_snapshot_);
    std::vector<Pose2> prefix;
    if(!map || !certifyUnchangedPathPrefix(odom,path,*map,live.get(),prefix))return false;
    const double length=prefix.back().s;
    const double speed=std::max(std::abs(odom.v),std::abs(last_cmd_v_));
    const double latency=.35+.10+control_period_+std::max(0.,now().seconds()-map->stamp.seconds());
    const double deceleration=std::max(.1,std::min(max_accel_,dynamic_yield_deceleration_));
    const double stopping_distance=speed*latency+speed*speed/(2.*deceleration);
    // A near conflict still stops immediately. Keep a useful 25cm certified
    // advance BEYOND the momentum/reaction reserve, not just a few safe pixels.
    if(length<.25+stopping_distance)return false;
    // A recovery leg must actually reach its stopped cusp before the queue
    // can advance. Put its stopline 15cm inside the certified edge, rather
    // than stopping 15cm short of the leg endpoint forever.
    if(recovering) {
      const double stopline=length-.15;const auto endpoint=samplePathAt(prefix,stopline);
      prefix.erase(std::remove_if(prefix.begin(),prefix.end(),[&](const Pose2 &p){return p.s>=stopline;}),prefix.end());
      prefix.push_back(endpoint);updatePathArcLength(prefix);
    }
    const size_t nearest=nearestPathIndex(path,odom.x,odom.y);
    const double cut_s=path[nearest].s+prefix.back().s;
    std::vector<Pose2> tail{prefix.back()};
    for(const auto &p:path)if(p.s>cut_s+1e-6)tail.push_back(p);
    updatePathArcLength(tail);
    {
      std::lock_guard<std::mutex> guard(path_mutex_);
      if(!have_path_ || path_seq_!=sequence || recovery_path_!=recovering || input_epoch_.load()!=epoch)return false;
      if(recovering) {
        if(tail.size()<2)return false;
        if(!recovery_staged_gears_ && !recovery_exit_active_ && !recovery_replan_only_ && !recovery_exit_.empty())
          recovery_legs_.push_back(recovery_exit_);
        recovery_legs_.push_front(std::move(tail));
        recovery_staged_gears_=true;recovery_legs_spatial_epoch_=spatial_epoch_.load();
        recovery_resume_pending_=false;recovery_exit_.clear();
      }
      path_=prefix;active_goal_=path_.back();++path_seq_;safety_prefix_seq_=path_seq_;
    }
    path=std::move(prefix);
    publishPath(map.get());
    requestPlan("invalid suffix; current finite prefix certified");
    RCLCPP_INFO_THROTTLE(get_logger(),*get_clock(),800,
      "terrain suffix repaired: retained %.3fm of unchanged certified curve; speed bounded to stop before its end, global repair requested",length);
    return true;
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
    local_goal_marker_pub_->publish(clear_goal);
    legacy_goal_marker_pub_->publish(clear_goal);
  }

  void waitForFreshInputs()
  {
    fresh_inputs_since_ = std::chrono::steady_clock::time_point{};
    if (!stale_inputs_waiting_.exchange(true, std::memory_order_acq_rel)) {
      // Invalidate in-flight searches as well as the currently tracked path.
      // Replanning stale data would keep the map callback waiting for the worker.
      const auto previous_epoch=input_epoch_.fetch_add(1, std::memory_order_acq_rel);
      // A temporal pause changes plan freshness, not the map coordinate frame
      // or measured progress identity. Keep that identity stable so repeated
      // short pauses cannot permanently exhaust recovery. A separate frame/
      // contract invalidation creates a different epoch and cannot rebase it.
      auto progress_epoch=previous_epoch;
      recovery_progress_valid_epoch_.compare_exchange_strong(progress_epoch,previous_epoch+1);
      if(const auto prior=std::atomic_load(&recovery_route_memory_)) {
        auto updated=std::make_shared<std::vector<RecoveryRouteMemory>>(*prior);
        for(auto &entry:*updated)if(entry.epoch==previous_epoch)entry.epoch=previous_epoch+1;
        std::atomic_store(&recovery_route_memory_,std::shared_ptr<const std::vector<RecoveryRouteMemory>>(updated));
      }
      std::lock_guard<std::mutex> lock(path_mutex_);
      if(recovery_path_ && !path_.empty() && recovery_geometry_epoch_==previous_epoch &&
         (have_path_ || recovery_resume_pending_)) {
        // Save GEOMETRY only; have_path stays false until a fresh worker
        // certifies the complete remaining leg again. Frame/contract faults
        // have a different epoch and cannot use this temporal resume path.
        if(!recovery_resume_pending_)recovery_resume_saved_time_=now().seconds();
        recovery_resume_pending_=true;recovery_geometry_epoch_=previous_epoch+1;
      } else {path_.clear();recovery_resume_pending_=false;}
      have_path_ = false;
    }
    publishImmediateStop();
  }

  void predictOdomForControl(OdomState &odom,bool tracking_lead=true) const
  {
    if (!odom.valid) return;

    if(std::all_of(odom.prediction_source.begin(),odom.prediction_source.end(),
        [](double v){return std::isfinite(v);})) {
      const auto&s=odom.prediction_source;
      odom.x=s[0];odom.y=s[1];odom.yaw=s[2];odom.v=s[3];odom.yaw_rate=s[4];
    } else odom.prediction_source={{odom.x,odom.y,odom.yaw,odom.v,odom.yaw_rate}};
    odom.prediction_sec=0.;odom.command_prediction=false;

    // Tracking may use a lead pose. Safety trajectories start at the CURRENT
    // clock, not a pose silently advanced an extra 80ms into the future.
    // Keep the original source stamp for all freshness checks.
    double dt = tracking_lead?odom_pose_prediction_time_:0.;
    const double stamp_sec = odom.stamp.seconds();
    const double now_sec = this->now().seconds();
    if(stamp_sec>now_sec+.05) {odom.valid=false;return;}
    // Preserve the original source-age admission limit. An expired source
    // remains at its source state for the caller's existing stale-input gate.
    if(stamp_sec>0.&&now_sec-stamp_sec>max_odom_age_sec_)return;
    if (stamp_sec > 0.0 && now_sec > stamp_sec) {
      dt += now_sec-stamp_sec;
    }
    // A 200ms cap silently left otherwise-admitted 350ms odometry 150ms in
    // the past. Replay the WHOLE admitted age. Tracking adds only its explicit
    // lead; a safety root ends exactly at NOW, never at a relabelled old pose.
    dt = std::max(0.,dt);
    if (dt <= 1e-4 || !std::isfinite(odom.v) || !std::isfinite(odom.yaw_rate)) return;

    const auto issued=issuedCommandSnapshot();
    const double bicycle_rate=std::abs(odom.v)*std::tan(max_steer_)/wheel_base_;
    if(issued.empty()||std::abs(odom.yaw_rate)>bicycle_rate+.05) {
      // No publication evidence: retain the explicit measured-twist fallback.
      // Likewise preserve observed angular motion outside the bicycle model;
      // clamping a measured slip/push to maximum steering invents a new pose.
      odom=terrain_nav::integrateBodyVelocity(odom,odom.v,odom.yaw_rate,dt,rear_axle_offset_);
      odom.prediction_sec=dt;return;
    }
    if(issued.back().t>now_sec+.05) {odom.valid=false;return;} // Old clock epoch, never future input.
    auto model=configuredYieldModel();model.step=.005;
    const double pending_delay=terrain_nav::maximumYieldDelay(model);
    terrain_nav::YieldCommand active{0.,odom.v,0.};
    const auto next=std::upper_bound(issued.begin(),issued.end(),stamp_sec-pending_delay,
      [](double t,const auto&c){return t<c.t;});
    if(next!=issued.begin())active=*std::prev(next);
    const double steering=std::abs(odom.v)>.05?
      clampAbs(std::atan(odom.yaw_rate*wheel_base_/odom.v),max_steer_):clampAbs(active.steering,max_steer_);
    terrain_nav::YieldState root{0.,odom.x,odom.y,odom.yaw,odom.v,steering};
    // Every schedule entry through NOW was actually published, including
    // inter-frame acceleration, steering changes and emergency zero outputs.
    // The optional tracking lead holds the last known request; it sees no
    // future publication. Safety roots stop at the current clock.
    std::vector<terrain_nav::YieldCommand> schedule{{-pending_delay,active.speed,active.steering}};
    for(auto c:issued)if(c.t>=stamp_sec-pending_delay&&c.t<=now_sec+1e-8) {
      // Absolute publication timestamps were already admitted above. A
      // subtraction at the lower boundary can produce -0.040000000000006
      // after a -0.04 sentinel, falsely making a causal queue unordered.
      // Normalize only this admitted boundary; genuinely older/future or
      // out-of-order publications retain their original rejection checks.
      c.t=std::max(-pending_delay,c.t-stamp_sec);schedule.push_back(c);
    }
    const auto states=terrain_nav::rolloutQueuedYield(root,schedule,model,dt,stamp_sec,issued);
    if(states.empty()) {odom.valid=false;return;}
    const auto&end=states.back();odom.x=end.x;odom.y=end.y;odom.yaw=wrapAngle(end.yaw);
    odom.v=end.speed;odom.yaw_rate=end.speed*std::tan(end.steering)/wheel_base_;
    odom.prediction_sec=dt;odom.command_prediction=true;
  }

  terrain_nav::JointMotionRisk jointMovingRisk(const std::vector<terrain_nav::YieldState> &states,
      double origin,const LiveObstacleSnapshot &live,bool screen=false,double preference_clearance=0.,
      bool allow_confidence_refinement=true) const
  {
    const auto body=collisionEnvelope(footprint_margin_);
    const terrain_nav::MotionCheckGeometry geometry{body.front,body.back,body.half_width,
      max_steer_,wheel_base_,live_obstacle_inflation_radius_};
    const bool sampled=std::any_of(live.tracks.begin(),live.tracks.end(),[](const auto &t) {
      return t.circular && t.center_motion_fit && terrain_nav::validGaussianMotion(t.gaussian);
    });
    uint64_t check=sampled?++motion_joint_checks_:0;
    auto result=terrain_nav::jointGaussianMotionRisk(states,origin,live.tracks,geometry,
      screen?256:motion_joint_samples_,screen||!sampled?.05:terrain_nav::motionCheckAlpha(check),
      motion_joint_seed_^check,motion_prediction_model_=="gaussian_joint_cuda",preference_clearance,!screen);
    if(allow_confidence_refinement&&!screen&&sampled&&motion_prediction_model_=="gaussian_joint_cuda"&&
       terrain_nav::refineMotionConfidence(result,motion_joint_risk_limit_)) {
      // One bounded extra proof. It consumes a NEW lifetime confidence-error
      // allocation and an independent stream; do not relabel the first
      // estimate as a certificate or repeatedly sample until it happens to pass.
      const double initial_ms=result.elapsed_ms;
      check=++motion_joint_checks_;
      result=terrain_nav::jointGaussianMotionRisk(states,origin,live.tracks,geometry,65536,
        terrain_nav::motionCheckAlpha(check),motion_joint_seed_^check,true,preference_clearance,true);
      result.elapsed_ms+=initial_ms;
    }
    if(!screen) {
      motion_joint_last_upper_.store(result.valid?result.upper:1.);
      motion_joint_last_ms_.store(result.elapsed_ms);
      motion_joint_last_check_.store(check);
      motion_joint_last_samples_.store(result.samples);
      motion_joint_last_analytical_.store(result.analytical_objects);
      size_t draws=0;for(size_t n:result.sample_counts)draws+=n;
      motion_joint_last_draws_.store(draws);
      motion_joint_last_origin_s_.store(states.empty()?NAN:origin+states.front().t);
    }
    return result;
  }

  bool timedMovingUnsafe(const std::vector<terrain_nav::YieldState> &states,
                         double origin,const LiveObstacleSnapshot &live,
                         double velocity_reserve_mps=0.,double preference_clearance_m=0.,bool screen=false) const
  {
    bool ordered=true;
    for(size_t i=1;i<states.size();++i)if(states[i].t<=states[i-1].t)ordered=false;
    // possibleEarlyStops is a priority locus, NOT a time-ordered executable
    // trajectory. Retain its original monotone reachable-envelope check;
    // never reinterpret equal-time rest positions as a Gaussian policy proof.
    if(motion_prediction_model_!="reachable" && velocity_reserve_mps==0 && ordered) {
      const auto risk=jointMovingRisk(states,origin,live,screen,preference_clearance_m);
      // A cheap screening estimate may reject a candidate but NEVER grants
      // a command. Full independent confidence-bound checks are mandatory at
      // action selection and again on fresh inputs at final submission.
      return !risk.valid || (screen?risk.estimate:risk.upper)>motion_joint_risk_limit_;
    }
    const auto body=collisionEnvelope(footprint_margin_);
    const terrain_nav::MotionCheckGeometry geometry{body.front,body.back,body.half_width,
      max_steer_,wheel_base_,live_obstacle_inflation_radius_};
    return terrain_nav::timedMotionUnsafe(states,origin,live.tracks,geometry,
      velocity_reserve_mps,preference_clearance_m);
  }

  bool unconfirmedMotionUnsafe(const std::vector<terrain_nav::YieldState> &states,double origin,
                               const LiveObstacleSnapshot &live) const
  {
    const auto tracks=dynamic_observed_only_?
      terrain_nav::currentMotionObservations(live.provisional_tracks,live.stamp_sec):live.provisional_tracks;
    if(tracks.empty())return false;
    const auto body=collisionEnvelope(footprint_margin_);
    const terrain_nav::MotionCheckGeometry geometry{body.front,body.back,body.half_width,
      max_steer_,wheel_base_,live_obstacle_inflation_radius_};
    // Only a veto. Weak translation evidence gets its existing explicit
    // velocity reserve, never Gaussian qualification, identity or ownership.
    return terrain_nav::timedMotionUnsafe(states,origin,tracks,geometry,provisional_velocity_reserve_mps_);
  }

  bool earlierStoppingUnsafe(const std::vector<terrain_nav::YieldState> &states,double origin,
                            const LiveObstacleSnapshot &live) const
  {
    if(motion_prediction_model_=="reachable")
      return timedMovingUnsafe(terrain_nav::possibleEarlyStops(states),origin,live);
    const auto body=collisionEnvelope(footprint_margin_);
    const terrain_nav::MotionCheckGeometry g{body.front,body.back,body.half_width,
      max_steer_,wheel_base_,live_obstacle_inflation_radius_};
    const auto envelope=terrain_nav::earlyStopEnvelope(states,g);
    if(!envelope.valid)return true;
    return timedMovingUnsafe(envelope.stationary,origin,live,0.,envelope.expansion);
  }

  std::vector<terrain_nav::YieldCommand> issuedCommandSnapshot() const
  {
    std::lock_guard<std::mutex> lock(command_publish_mutex_);
    return issued_commands_.snapshot();
  }

  terrain_nav::YieldModel configuredYieldModel() const
  {
    terrain_nav::YieldModel model;model.delay=dynamic_yield_delay_;
    model.speed_dead_time=dynamic_speed_dead_time_;model.steering_dead_time=dynamic_steering_dead_time_;
    model.acceleration=std::max(.1,max_accel_);model.deceleration=dynamic_yield_deceleration_;
    model.wheelbase=wheel_base_;model.rear_axle_offset=rear_axle_offset_;model.steering_rate=max_steer_rate_;
    return model;
  }

  std::vector<terrain_nav::YieldState> issuedYieldRollout(terrain_nav::YieldState state,
      const std::vector<terrain_nav::YieldCommand>&commands,const terrain_nav::YieldModel&model,
      double end,double origin) const
  {
    return terrain_nav::rolloutQueuedYield(state,commands,model,end,origin,issuedCommandSnapshot());
  }

  bool timedYieldSafe(const std::vector<terrain_nav::YieldState> &states,double origin,
      const std::vector<terrain_nav::YieldCommand> &commands,
      const ControlMapSnapshot &map,const LiveObstacleSnapshot &live,
      double *path_upper=nullptr,double *braking_upper=nullptr,std::string *failure=nullptr,
      std::string *rejection_details=nullptr,YieldFootprintCache *cache=nullptr,
      bool allow_confidence_refinement=true) const
  {
    const terrain_nav::YieldState *evaluated=nullptr;
    double dynamic_upper=0.;
    terrain_nav::ChanceBudget path_budget,braking_budget;
    const auto reject=[&](const char *reason) {
      if(failure)*failure=reason;
      if(rejection_details) {
        std::ostringstream out;
        out<<"source_s="<<std::setprecision(15)<<map.stamp.seconds();
        if(evaluated)out<<" sample=("<<evaluated->x<<","<<evaluated->y<<","<<evaluated->yaw
          <<") t="<<evaluated->t<<" speed="<<evaluated->speed;
        out<<" path_upper="<<path_budget.upper()<<" brake_upper="<<braking_budget.upper()
          <<" conditional_dynamic_upper="<<dynamic_upper;
        *rejection_details=out.str();
      }
      return false;
    };
    if(rejection_details)rejection_details->clear();
    if(failure)*failure="certified";
    if(path_upper)*path_upper=std::numeric_limits<double>::quiet_NaN();
    if(braking_upper)*braking_upper=std::numeric_limits<double>::quiet_NaN();
    if(states.empty() || live.width<=0 || live.height<=0 || !std::isfinite(live.resolution) || live.resolution<=0 ||
       origin+states.front().t<live.stamp_sec-.05 || origin+states.front().t-live.stamp_sec>live_obstacle_max_age_sec_ ||
       origin+states.front().t<map.stamp.seconds()-.05 || origin+states.front().t-map.stamp.seconds()>max_map_age_sec_ ||
       live.present_hard.size()!=live.hard.size() ||
       live.hard.size()!=static_cast<size_t>(live.width*live.height) ||
       std::abs(states.back().speed)>.001)return reject("invalid/stale inputs or non-stopping rollout");
    if(motion_prediction_model_!="reachable") {
      if(unconfirmedMotionUnsafe(states,origin,live))return reject("unconfirmed motion reachability");
      const auto risk=jointMovingRisk(states,origin,live,false,0.,allow_confidence_refinement);
      dynamic_upper=risk.upper;
      if(!risk.valid || dynamic_upper>motion_joint_risk_limit_)return reject("conditional joint motion risk");
    } else if(timedMovingUnsafe(states,origin,live))return reject("moving obstacle reservation");
    if(std::all_of(commands.begin(),commands.end(),[](const auto &c){return c.speed==0;}) &&
       earlierStoppingUnsafe(states,origin,live))
      return reject("possible earlier stopping position");
    YieldFootprintCache local_cache;
    auto &views=cache?*cache:local_cache;bindYieldCollisionViews(map,live,views);
    const auto footprint=[&](const auto &state,bool future,double &cost,double &clearance,CostQuery &query) {
      if(!std::isfinite(state.x)||!std::isfinite(state.y)||!std::isfinite(state.yaw))return false;
      const auto key=std::make_tuple(state.x,state.y,state.yaw,future);
      if(cache) {
        const auto found=cache->entries.find(key);
        if(found!=cache->entries.end()) {
          ++cache->hits;cost=found->second.cost;clearance=found->second.clearance;
          query=found->second.query;return found->second.safe;
        }
        ++cache->misses;
      }
      const auto *view=future?views.future_view.get():views.current_view.get();
      const bool safe=footprintCostLocked(state.x,state.y,state.yaw,cost,&clearance,nullptr,false,&map,view,&query);
      if(cache&&cache->entries.size()<8192)cache->entries.emplace(key,YieldFootprintCache::Entry{safe,cost,clearance,query});
      return safe;
    };
    const auto braking_sample=[&](const auto &state) {
      const double request=terrain_nav::yieldCommandAt(commands,state.t).speed;
      // Account the selected policy's REAL slowing/reversal and final stop,
      // including the delay after a zero request. Forward cruising is not a
      // braking event; the complete action has its separate path ledger.
      return request==0 || request*state.speed<0 || std::abs(request)<std::abs(state.speed)-1e-8;
    };
    // If both ledgers visit exactly the SAME ordered events, sharing their
    // computation is exact. Keep both limits; a reversing rollout with any
    // excluded sample still uses the original independent ledgers.
    const bool identical_budgets=std::all_of(states.begin(),states.end(),braking_sample);
    for(const auto &state:states) {
      evaluated=&state;
      if(!std::isfinite(state.x) || !std::isfinite(state.y) || !std::isfinite(state.yaw) ||
         !std::isfinite(state.speed) || !std::isfinite(state.steering) || !std::isfinite(state.t))
        return reject("non-finite rollout");
      double cost=0,clearance=kInf;
      CostQuery query;
      if(!footprint(state,&state!=&states.front(),cost,clearance,query) ||
         (std::isfinite(clearance)&&clearance<hardClearanceFloor()))
        return reject("terrain/occupancy footprint or state probability");
      if(chance_constrained_ && dynamic_upper>0) {
        if(!query.known || !std::isfinite(query.probability_upper) || query.probability_upper+dynamic_upper>chance_state_limit_)
          return reject("combined terrain/dynamic state probability");
      }
      if(!addChanceQuery(query,path_budget,chance_path_budget_-dynamic_upper,&map))
        return reject("path probability budget");
      if((identical_budgets && path_budget.upper()+dynamic_upper>chance_braking_budget_) ||
         (!identical_budgets && braking_sample(state) &&
          !addChanceQuery(query,braking_budget,chance_braking_budget_-dynamic_upper,&map)))
        {
          if(identical_budgets)braking_budget=path_budget;
          return reject("braking probability budget");
        }
    }
    // A current immediate-stop policy is an ALTERNATIVE to the selected
    // maneuver, not a disjoint extra segment of that same path. Prove its
    // terrain/occupancy risk separately under the unchanged braking limit.
    // It may be dynamically unsafe (a crossing can hit a stopped car), so
    // this does not authorize choosing it or replace the action's ALL-object
    // time/body check above. Every executed continuation is revalidated.
    terrain_nav::YieldModel stopping_model;
    stopping_model.rear_axle_offset=rear_axle_offset_;stopping_model.wheelbase=wheel_base_;
    stopping_model.acceleration=std::max(.1,max_accel_);stopping_model.deceleration=dynamic_yield_deceleration_;
    stopping_model.steering_rate=max_steer_rate_;
    stopping_model.delay=dynamic_yield_delay_+.10+
      std::max(0.,origin+states.front().t-map.stamp.seconds());
    auto root=states.front();root.t=0.;
    // Pending acceleration may increase speed before this zero takes effect.
    const double stop_horizon=stopping_model.delay+
      (std::abs(root.speed)+stopping_model.acceleration*stopping_model.delay)/stopping_model.deceleration+.10;
    const auto stopping=issuedYieldRollout(root,{{0,0,0}},stopping_model,stop_horizon,origin+states.front().t);
    terrain_nav::ChanceBudget current_stop_budget;
    for(const auto &sample:stopping) {
      evaluated=&sample;
      double cost=0,clearance=kInf;
      CostQuery query;
      if(!footprint(sample,false,cost,clearance,query) ||
         (std::isfinite(clearance)&&clearance<hardClearanceFloor()))
        return reject("current stopping terrain/occupancy");
      if(!addChanceQuery(query,current_stop_budget,chance_braking_budget_,&map)) {
        braking_budget=current_stop_budget;
        return reject("current stopping probability budget");
      }
    }
    if(path_upper)*path_upper=std::min(1.,path_budget.upper()+dynamic_upper);
    if(braking_upper)*braking_upper=std::max(current_stop_budget.upper(),
      (identical_budgets?path_budget.upper():braking_budget.upper())+dynamic_upper);
    return true;
  }

  bool retireGeometricallySeparatedConflicts(const OdomState &odom,double time,
                                           const LiveObstacleSnapshot &live)
  {
    // Retiring interaction bookkeeping is NOT a motion certificate. Ground
    // and path admission remain mandatory in controlTimer/commit. Requiring a
    // terrain proof here lost positive separation evidence whenever GP output
    // paused, then latched the already-passed actor forever after it left view.
    if(dynamic_unresolved_.empty() || !odom.valid ||
       time-odom.stamp.seconds()<-.05 || time-odom.stamp.seconds()>max_odom_age_sec_ ||
       time-live.stamp_sec<-.05 || time-live.stamp_sec>live_obstacle_max_age_sec_)return false;
    // Legacy reachable mode keeps its original all-object geometric test.
    // Probability mode must retire relevance under the SAME conditional
    // stopping event used by its control shield, not an unrelated six-second
    // acceleration envelope. Missing identities themselves remain unresolved.
    if(motion_prediction_model_=="reachable")for(const auto &entry:dynamic_unresolved_)
      if(std::none_of(live.tracks.begin(),live.tracks.end(),[&](const auto &t){return t.motion_confirmed && t.id==entry.first;}))
        return false;
    const auto model=configuredYieldModel();
    terrain_nav::YieldState state;state.x=odom.x;state.y=odom.y;state.yaw=odom.yaw;state.speed=odom.v;
    state.steering=std::abs(odom.v)>.05?clampAbs(std::atan(odom.yaw_rate*wheel_base_/odom.v),max_steer_):
      clampAbs(last_cmd_steer_/steer_output_sign_,max_steer_);
    // ALL observed tracks, full body, uncertainty, and every earlier stopping
    // position. timedMovingUnsafe also rejects stale/invalid target evidence.
    // Include the previous command still in the actuator pipeline, as well
    // as measured motion. A new zero request does not cancel that latency.
    auto pending=state;pending.speed=last_cmd_v_;
    pending.steering=clampAbs(last_cmd_steer_/steer_output_sign_,max_steer_);
    if(motion_prediction_model_!="reachable") {
      std::array<std::vector<terrain_nav::YieldState>,3> stops;
      stops[0]=terrain_nav::finiteJointRollout(issuedYieldRollout(state,{{0,0,0}},model,6.,time));
      stops[2]=terrain_nav::finiteJointRollout(terrain_nav::rolloutYield(state,state,{{0,0,0}},model,6.));
      stops[1]=terrain_nav::finiteJointRollout(terrain_nav::rolloutYield(state,pending,{{0,0,0}},model,6.));
      bool retired=false;
      for(auto i=dynamic_unresolved_.begin();i!=dynamic_unresolved_.end();) {
        const auto found=std::find_if(live.tracks.begin(),live.tracks.end(),[&](const auto &t){return t.id==i->first;});
        const auto count=std::count_if(live.tracks.begin(),live.tracks.end(),[&](const auto &t){return t.id==i->first;});
        // Actual new source geometry is required. Coasts, ambiguity, expiry,
        // an absent ID and the mere passage of time never close its conflict.
        if(found==live.tracks.end()||count!=1||!terrain_nav::currentMeasuredIdentity(*found,live.stamp_sec,false)) {++i;continue;}
        // Bookkeeping is optional. Use the deterministic analytic Boole bound
        // of the SAME Gaussian event, without spending GPU sampling/proof
        // budget on relevance retirement. A loose bound simply keeps the ID.
        // The analytic event also encloses bounded signed mean acceleration.
        // This is positive CURRENT-source separation, not expiry or absence.
        bool separated=true;
        const auto body=collisionEnvelope(footprint_margin_);
        const terrain_nav::MotionCheckGeometry geometry{body.front,body.back,body.half_width,
          max_steer_,wheel_base_,live_obstacle_inflation_radius_};
        for(const auto &stop:stops) {
          const auto envelope=terrain_nav::earlyStopEnvelope(stop,geometry);
          if(!envelope.valid){separated=false;break;}
          const auto risk=terrain_nav::timedGaussianMotionRisk(envelope.stationary,time,{*found},geometry,envelope.expansion);
          if(!risk.valid||risk.upper>motion_joint_risk_limit_){separated=false;break;}
        }
        if(!separated) {++i;continue;}
        i=dynamic_unresolved_.erase(i);retired=true;
      }
      if(!retired)return false;
      ++dynamic_geometric_retirements_;
      // Bookkeeping only: the still-observed body remains in live occupancy
      // and ALL-object current/continuation/commit checks. Never cancel an
      // executing certified policy or authorize a command here.
      if(!dynamic_active_||dynamic_commands_.empty()) {
        dynamic_active_=false;dynamic_yield_state_="idle";
        dynamic_yield_reason_="fresh model stopping separation; motion still gated";
      }
      return true;
    }
    for(const auto &held:std::array<terrain_nav::YieldState,2>{state,pending}) {
      const auto stop=terrain_nav::rolloutYield(state,held,{{0,0,0}},model,6.);
      const auto body=collisionEnvelope(footprint_margin_);
      const terrain_nav::MotionCheckGeometry geometry{body.front,body.back,body.half_width,max_steer_,wheel_base_,live_obstacle_inflation_radius_};
      if(terrain_nav::timedMotionUnsafe(terrain_nav::possibleEarlyStops(stop),time,live.tracks,geometry))return false;
    }
    dynamic_unresolved_.clear();++dynamic_geometric_retirements_;
    // Resolving an identity is not a control handover. Keep an executing
    // finite policy until its own fresh continuation/commit checks decide;
    // cancelling it here inserted a stop in ordinary input recovery.
    if(!dynamic_active_ || dynamic_commands_.empty()) {
      dynamic_active_=false;
      dynamic_yield_state_="idle";dynamic_yield_reason_="fresh geometric stopping separation; motion still gated";
    }
    return true;
  }

  bool reserveRolloutSafe(const std::vector<terrain_nav::YieldState> &states,double origin,
      const std::vector<terrain_nav::YieldCommand> &commands,
      const ControlMapSnapshot &map,const LiveObstacleSnapshot &live,
      double *path_upper=nullptr,double *braking_upper=nullptr,std::string *failure=nullptr) const
  {
    const auto reject=[&](const char *reason){if(failure)*failure=reason;return false;};
    if(states.empty() || live.uninflated_present.size()!=live.hard.size() ||
       live.uninflated_present.empty())return reject("missing original occupancy");
    auto physical=live;
    physical.hard=physical.present_hard=live.uninflated_present;
    physical.inflation_radius=0.;
    const auto &end=states.back();double cost=0,clearance=kInf;
    if(!footprintCostLocked(end.x,end.y,end.yaw,cost,&clearance,nullptr,false,&map,&live) ||
       (std::isfinite(clearance)&&clearance<hardClearanceFloor()))return reject("terminal reserve not released");
    // Only recover the extra live-layer reserve. The uninflated collision grid,
    // full body margin, unknown support and probability budgets stay mandatory.
    const size_t initial=liveBodyOverlapCount(states.front().x,states.front().y,states.front().yaw,live);
    terrain_nav::ChanceBudget stopping_budget;
    for(const auto &state:states) {
      if(liveBodyOverlapCount(state.x,state.y,state.yaw,live,initial+1)>initial ||
         !addChancePose(state.x,state.y,state.yaw,stopping_budget,chance_braking_budget_,&map,&physical))
        return reject("extra reserve increased or full stopping budget exceeded");
    }
    if(!timedYieldSafe(states,origin,commands,map,physical,path_upper,braking_upper,failure))return false;
    if(braking_upper)*braking_upper=stopping_budget.upper();
    return true;
  }

  std::vector<Pose2> reanchorArc(const std::vector<Pose2> &path,const OdomState &odom) const
  {
    if(path.size()<2)return {};
    std::vector<Pose2> result{Pose2{odom.x,odom.y,odom.yaw,0.,path.front().direction}};
    for(size_t i=1;i<path.size();++i) {
      const auto &a=path[i-1],&b=path[i];
      const auto ar=terrain_nav::rearAxlePose(a,rear_axle_offset_);
      const auto br=terrain_nav::rearAxlePose(b,rear_axle_offset_);
      const double chord=std::hypot(br.x-ar.x,br.y-ar.y);
      const double ds=path[i].s-path[i-1].s;
      if(!std::isfinite(ds) || ds<=0 || chord<1e-9)return {};
      if(a.direction!=b.direction)return {};
      const double turn=wrapAngle(b.yaw-a.yaw);
      const double curvature=2*std::sin(.5*turn)/(a.direction*chord);
      const double length=std::abs(curvature)>1e-8?
        std::abs(turn/curvature)*terrain_nav::bodyArcScale(curvature,rear_axle_offset_):ds;
      auto p=advanceArc(result.back(),curvature,a.direction*length);p.s=result.back().s+length;
      result.push_back(p);
    }
    return result;
  }

  bool reserveForwardExit(const std::vector<terrain_nav::YieldState> &retreat,const Pose2 &goal,
      const ControlMapSnapshot &map,const LiveObstacleSnapshot &live,std::vector<Pose2> &exit_path,
      double *joint_upper=nullptr) const
  {
    exit_path.clear();
    if(joint_upper)*joint_upper=kInf;
    if(retreat.empty())return false;
    const auto &start=retreat.front(),&end=retreat.back();
    const double distance=std::hypot(goal.x-start.x,goal.y-start.y);
    if(distance<1. || std::abs(end.speed)>.001)return false;
    const double ux=(goal.x-start.x)/distance,uy=(goal.y-start.y)/distance;
    // One source-event account spans both legs. Only the retreat may release
    // pre-existing extra inflation; the whole forward exit uses normal guards.
    auto physical=live;physical.hard=physical.present_hard=live.uninflated_present;
    physical.inflation_radius=0.;
    terrain_nav::ChanceBudget reverse_budget;
    for(const auto &p:retreat)
      if(!addChancePose(p.x,p.y,p.yaw,reverse_budget,chance_path_budget_,&map,&physical))return false;
    double best=kInf;
    for(double steering:{0.,-max_steer_,max_steer_})for(double turn_length:{.5,1.,1.5}) {
      if(steering==0 && turn_length!=.5)continue;
      std::vector<Pose2> candidate{Pose2{end.x,end.y,end.yaw,0.,1}};
      auto budget=reverse_budget;
      double route_cost=0.;
      for(int i=1;i<=120;++i) {
        auto p=advanceArc(candidate.back(),i*.025<=turn_length+1e-9?primitiveCurvature(steering):0.,.025);p.s=i*.025;
        double cost=0,clearance=kInf;
        if(!footprintCostLocked(p.x,p.y,p.yaw,cost,&clearance,nullptr,false,&map,&live) ||
           (std::isfinite(clearance)&&clearance<hardClearanceFloor()) ||
           !addChancePose(p.x,p.y,p.yaw,budget,chance_path_budget_,&map,&live))break;
        candidate.push_back(p);
        route_cost+=historicalObstacleCost(p.x,p.y,&map)*.025;
        const double remaining=std::hypot(p.x-goal.x,p.y-goal.y);
        if((p.x-start.x)*ux+(p.y-start.y)*uy<.25 || remaining>distance-.20)continue;
        const double score=remaining+.1*std::abs(wrapAngle(p.yaw-start.yaw))+.5*route_cost;
        if(score<best) {best=score;exit_path=candidate;if(joint_upper)*joint_upper=budget.upper();}
      }
    }
    return exit_path.size()>=2;
  }

  bool reserveRecoveryControl(const OdomState &odom)
  {
    const double time=now().seconds();
    const auto live=std::atomic_load(&live_snapshot_);
    const auto map=std::atomic_load(&control_map_snapshot_);
    uint64_t seq=0;Pose2 goal;
    {std::lock_guard<std::mutex> lock(goal_mutex_);if(have_goal_){seq=goal_seq_;goal=raw_goal_;}}
    const auto epoch=input_epoch_.load();
    const bool fresh=!stale_inputs_waiting_.load() && live && map && odom.valid &&
      time>=live->stamp_sec-.05 && time-live->stamp_sec<=live_obstacle_max_age_sec_ &&
      time>=map->stamp.seconds()-.05 && time-map->stamp.seconds()<=max_map_age_sec_ &&
      time>=odom.stamp.seconds()-.05 && time-odom.stamp.seconds()<=max_odom_age_sec_;
    if(!reverse_recovery_enable_ || !reverse_enable_ || !chance_constrained_ || !live_obstacle_enable_ ||
       !seq || !fresh || (reserve_active_ && (seq!=reserve_goal_seq_ || epoch!=reserve_epoch_ || time<reserve_last_time_))) {
      const bool aborted=reserve_active_;reserve_active_=false;reserve_retreat_active_=false;
      reserve_state_="idle";
      if(aborted) {publishImmediateStop();publishChanceStatus(odom,kInf,false,"reserve recovery inputs invalidated");return true;}
      return false;
    }
    const auto model=configuredYieldModel();
    terrain_nav::YieldState state;
    state.x=odom.x;state.y=odom.y;state.yaw=odom.yaw;state.speed=odom.v;
    state.steering=std::abs(odom.v)>.05 ? clampAbs(std::atan(odom.yaw_rate*wheel_base_/odom.v),max_steer_) :
      clampAbs(last_cmd_steer_/steer_output_sign_,max_steer_);
    constexpr double horizon=6.5;
    if(reserve_active_) {
      const double stopped_after=reserve_commands_.back().t+terrain_nav::yieldSpeedDelay(model)+.30/model.deceleration;
      double current_cost=0,current_clearance=kInf;
      const bool released=std::abs(odom.v)<.03 &&
        footprintCostLocked(state.x,state.y,state.yaw,current_cost,&current_clearance,nullptr,false,map.get(),live.get()) &&
        (!std::isfinite(current_clearance)||current_clearance>=hardClearanceFloor());
      if(time-reserve_origin_>=stopped_after && released) {
        if(reserve_release_since_<0) {reserve_release_since_=time;reserve_release_stamp_=map->stamp.seconds();}
      } else reserve_release_since_=-1;
      const bool confirmed=reserve_release_since_>=0 && time-reserve_release_since_>=.2 &&
        map->stamp.seconds()>reserve_release_stamp_;
      if(confirmed || time-reserve_origin_>=horizon-1e-6) {
        reserve_active_=false;reserve_state_="completed";
        if(confirmed && reserve_forward_exit_.size()>=2) {
          auto request=std::make_shared<ReserveExitRequest>();
          request->path=reserve_forward_exit_;request->goal_seq=seq;request->epoch=epoch;
          std::atomic_store(&reserve_exit_request_,std::shared_ptr<const ReserveExitRequest>(request));
        }
        {
          std::lock_guard<std::mutex> lock(path_mutex_);
          path_.clear();have_path_=false;recovery_path_=false;recovery_staged_gears_=false;recovery_legs_.clear();recovery_exit_.clear();
          ++path_seq_;resetNominalControls();
        }
        reserve_retreat_active_=false;
        publishImmediateStop();requestPlan("clearance reserve recovery completed");
        if(!confirmed)reserve_state_="refused";
        publishChanceStatus(odom,reserve_braking_upper_,confirmed,
          confirmed?"reserve recovery completed; requesting fresh path":"reserve recovery ended without confirmed release");return true;
      }
      state.t=time-reserve_origin_;
      const auto states=issuedYieldRollout(state,reserve_commands_,model,horizon,reserve_origin_);
      if(!reserveRolloutSafe(states,reserve_origin_,reserve_commands_,*map,*live,&reserve_path_upper_,&reserve_braking_upper_)) {
        reserve_active_=false;reserve_retreat_active_=false;reserve_state_="refused";publishImmediateStop();
        publishChanceStatus(odom,kInf,false,"reserve recovery remaining rollout invalidated");return true;
      }
    } else {
      reserve_state_="idle";
      if(dynamic_active_ || !live->tracks.empty() || std::abs(odom.v)>.05 || reserve_recovery_count_.load()>=2 ||
         live->uninflated_present.size()!=live->hard.size() || live->uninflated_present.empty() ||
         !liveBodyUnsafe(state.x,state.y,state.yaw,*live))return false;
      if(time>=reserve_last_attempt_ && time-reserve_last_attempt_<.5)return false;
      reserve_last_attempt_=time;
      publishImmediateStop();
      reserve_commands_.clear();
      reserve_forward_exit_.clear();
      reserve_joint_upper_=kInf;
      const double speed=std::min(.30,reverse_max_speed_);
      for(double distance:{.12,.25,.4,.6,.9,1.2,1.5}) {
        for(double steering:{0.,-max_steer_,max_steer_}) {
          auto commands=terrain_nav::makeYieldCommands(state,model,distance,speed,steering,horizon);
          if(commands.empty())continue;
          const auto states=issuedYieldRollout(state,commands,model,horizon,time);
          if(reserveRolloutSafe(states,time,commands,*map,*live,&reserve_path_upper_,&reserve_braking_upper_) &&
             reserveForwardExit(states,goal,*map,*live,reserve_forward_exit_,&reserve_joint_upper_)) {
            reserve_commands_=std::move(commands);break;
          }
        }
        if(!reserve_commands_.empty())break;
      }
      if(reserve_commands_.empty())return false;
      reserve_initial_=state;reserve_origin_=time;reserve_goal_seq_=seq;reserve_epoch_=epoch;reserve_active_=true;
      reserve_retreat_active_=true;
      rememberBlockedRoute(odom,seq);
      reserve_release_since_=-1;
      ++reserve_recovery_count_;
      RCLCPP_INFO(get_logger(),"clearance reserve recovery: bounded retreat plus %.2fm forward exit, joint upper=%.6f, reserve attempt=%d/2 maneuver attempts=%d/2; full body margin retained",
        reserve_forward_exit_.back().s,reserve_joint_upper_,reserve_recovery_count_.load(),reverse_recovery_count_.load());
    }
    reserve_last_time_=time;
    const auto selected=terrain_nav::yieldCommandAt(reserve_commands_,time-reserve_origin_);
    geometry_msgs::msg::Twist cmd;cmd.linear.x=selected.speed;cmd.angular.z=selected.steering*steer_output_sign_;
    {std::lock_guard<std::mutex> lock(goal_mutex_);
      if(!have_goal_ || goal_seq_!=seq || input_epoch_.load()!=epoch) {
        reserve_active_=false;reserve_retreat_active_=false;publishImmediateStop();return true;
      }
      publishCommand(cmd);
    }
    reserve_state_=cmd.linear.x<0?"retreat":"hold";
    publishChanceStatus(odom,reserve_braking_upper_,true,"clearance reserve recovery certified");return true;
  }

  bool provisionalMotionBrake(const OdomState &odom)
  {
    provisional_brake_active_=false;provisional_warning_active_=false;
    if(!chance_constrained_ || !live_obstacle_enable_ || !haveGoalThreadSafe() ||
        dynamic_active_ || reserve_active_ || !odom.valid)return false;
    const auto live=std::atomic_load(&live_snapshot_);
    const auto map=std::atomic_load(&control_map_snapshot_);
    const double time=now().seconds();
    if(!live || !map || live->provisional_tracks.empty() ||
        time<live->stamp_sec-.05 || time-live->stamp_sec>live_obstacle_max_age_sec_ ||
        time<map->stamp.seconds()-.05 || time-map->stamp.seconds()>max_map_age_sec_ ||
        time<odom.stamp.seconds()-.05 || time-odom.stamp.seconds()>max_odom_age_sec_)return false;
    provisional_warning_active_=true;
    // Weak motion has a bounded confirmation opportunity, not a licence to
    // hold an uncertain prediction for the entire future braking horizon.
    // If the expected next observations fail to arrive, stop; never keep
    // nominal motion on a stale weak cue just because its geometry is remote.
    if(time-live->stamp_sec>std::min(.35,live_obstacle_max_age_sec_)) {
      provisional_brake_active_=true;publishImmediateStop();
      publishChanceStatus(odom,std::numeric_limits<double>::quiet_NaN(),false,
        "weak motion confirmation input deadline exceeded");return true;
    }
    const auto model=configuredYieldModel();
    terrain_nav::YieldState state;state.x=odom.x;state.y=odom.y;state.yaw=odom.yaw;state.speed=odom.v;
    state.steering=std::abs(odom.v)>.05 ? clampAbs(std::atan(odom.yaw_rate*wheel_base_/odom.v),max_steer_) :
      clampAbs(last_cmd_steer_/steer_output_sign_,max_steer_);
    constexpr double horizon=.20; // Two 10Hz observations for qualification.
    const auto held=issuedYieldRollout(state,{{0,state.speed,state.steering}},model,horizon,time);
    // Do not inflate a map or invent a confirmed track. Reuse exactly the
    // swept-body prediction test, restricted to the confirmation window.
    // A remote uncertain movement requests evidence/replanning, not an
    // unconditional stop in the path it might later cross. Normal command
    // terrain/current-occupancy/braking admission remains mandatory below;
    // qualified center hypotheses receive the FULL multi-second action proof.
    LiveObstacleSnapshot provisional;provisional.tracks=live->provisional_tracks;
    // One translated scan cannot estimate velocity precisely. Grow only this
    // brake-only candidate envelope over prediction time. At 0.20m / 10Hz,
    // even a half-cell displacement is 1m/s; the former 0.5m/s allowance was
    // too small for short sparse associations. This engineering reserve is
    // explicit and logged, not a calibrated velocity confidence interval.
    // It never paints the static map, confirms a track, or permits
    // a reverse/control maneuver. Confirmed-track safety remains independent.
    if(!timedMovingUnsafe(held,time,provisional,provisional_velocity_reserve_mps_)) {
      if(live->stamp_sec!=provisional_warning_stamp_) {
        provisional_warning_stamp_=live->stamp_sec;
        // A remote weak cue is not a route hazard. Replanning here changed
        // nominal steering while the object was still being qualified.
      }
      return false;
    }
    provisional_brake_active_=true;
    if(live->stamp_sec!=provisional_brake_stamp_) {++provisional_brake_count_;provisional_brake_stamp_=live->stamp_sec;}
    publishImmediateStop();
    publishChanceStatus(odom,std::numeric_limits<double>::quiet_NaN(),false,
        "provisional motion: braking before velocity confirmation");
    return true;
  }

  double encounterSpeed(double requested,double x,double y,double yaw,const LiveObstacleSnapshot &live) const
  {
    const double age=now().seconds()-live.stamp_sec;
    const bool modeled=unclassified_encounter_model_release_ && chance_constrained_ && dynamic_time_indexed_routing_ &&
      motion_prediction_model_!="reachable" && age>=0 && age<=std::min(.6,live_obstacle_max_age_sec_);
    const auto cues=modeled?terrain_nav::unmodeledCompactBounds(live.compact_observed,live.tracks,live.stamp_sec):
      live.compact_observed;
    return terrain_nav::compactEncounterSpeed(requested,x,y,yaw,cues,unclassified_encounter_speed_);
  }

  std::vector<terrain_nav::YieldCommand> nominalInteractionCommands(const OdomState &odom,
      const terrain_nav::YieldState &state,const terrain_nav::YieldModel &model,
      double horizon,const LiveObstacleSnapshot &live,bool terminal_only=false)
  {
    std::vector<Pose2> path;
    {std::lock_guard<std::mutex> lock(path_mutex_);
      if(!have_path_ || (recovery_path_ && !recovery_staged_gears_) || path_.size()<2)return {};
      path=path_;
    }
    const int direction=path.front().direction;
    if((direction<0 && !reverse_enable_) || (direction>0 && use_mppi_))return {};
    bool terminal=false;
    {std::lock_guard<std::mutex> lock(goal_mutex_);
      terminal=have_goal_ && std::hypot(path.back().x-raw_goal_.x,path.back().y-raw_goal_.y)<=goal_tolerance_;
    }
    const auto nearest=nearestPathIndex(path,odom.x,odom.y);
    const double remaining=path.back().s-path[nearest].s;
    if(terminal_only && (!terminal || remaining>1.2))return {};
    const double distance=std::min(1.2,remaining);
    if(distance<=(terminal?.02:.4))return {};
    auto requested=computePurePursuitCommand(odom,path);
    requested.linear.x=encounterSpeed(requested.linear.x,odom.x,odom.y,odom.yaw,live);
    if(!std::isfinite(requested.linear.x) || !std::isfinite(requested.angular.z) ||
       direction*requested.linear.x<=.05)return {};
    const double steering=clampAbs(requested.angular.z,max_steer_);
    if(direction<0)return terrain_nav::makeYieldCommands(state,model,distance,
      std::min(reverse_max_speed_,-requested.linear.x),steering,horizon);
    return terrain_nav::makeAdvanceCommands(state,model,distance,
      std::min(max_speed_,requested.linear.x),steering,horizon);
  }

  bool nominalDynamicHandoverSafe(const OdomState &odom,const terrain_nav::YieldState &state,
      const terrain_nav::YieldModel &model,double time,double horizon,
      const ControlMapSnapshot &map,const LiveObstacleSnapshot &live,bool stopping_checked=false)
  {
    if(live.tracks.empty())return false;
    const auto commands=nominalInteractionCommands(odom,state,model,horizon,live);
    if(commands.empty())return false;
    const auto rollout=issuedYieldRollout(state,commands,model,horizon,time);
    // Earlier stopping geometry belongs to the immediate braking rollout.
    // Enclosing the entire nominal drive in a stationary disk predicts
    // occupation to either side of the car where it never drives or brakes.
    if(!stopping_checked &&
       earlierStoppingUnsafe(issuedYieldRollout(state,{{0,0,0}},model,horizon,time),time,live))return false;
    return timedYieldSafe(rollout,time,commands,map,live);
  }

  std::vector<terrain_nav::ForwardYieldProposal> nearbyGoalCommands(const terrain_nav::YieldState &state,
      const Pose2 &goal,const terrain_nav::YieldModel &model,double horizon,double forward_limit) const
  {
    const double dx=goal.x-state.x,dy=goal.y-state.y;
    const double distance=std::hypot(dx,dy);
    if(!std::isfinite(distance) || distance<=.02 || distance>1.2)return {};
    const double bearing=wrapAngle(std::atan2(dy,dx)-state.yaw);
    const bool reverse=std::cos(bearing)<0.;
    if(reverse && !reverse_enable_)return {};
    // Short, low-speed arcs toward the mission goal can supply an explicit
    // stopped endpoint while the global replanner has no current terminal
    // connector. These are subject to the ordinary swept-body proof.
    const double speed=std::min({.6,reverse?reverse_max_speed_:forward_limit,
      terrain_nav::clearanceSpeedLimit(distance,0.,yieldSpeedDelay(model),model.deceleration)});
    if(speed<=.05)return {};
    const double aimed_steering=clampAbs(std::atan2(2.*wheel_base_*std::sin(bearing),distance),max_steer_);
    std::vector<terrain_nav::ForwardYieldProposal> candidates;
    for(double steering:{0.,-.5*max_steer_,.5*max_steer_,-max_steer_,max_steer_,aimed_steering}) {
      for(double length:{.75*distance,distance,1.15*distance}) {
        auto commands=reverse ? terrain_nav::makeYieldCommands(state,model,length,speed,steering,horizon) :
          terrain_nav::makeAdvanceCommands(state,model,length,speed,steering,horizon);
        if(!commands.empty())candidates.push_back({std::move(commands),length});
      }
    }
    return candidates;
  }

  bool commitDynamicCommand(const OdomState &odom,const geometry_msgs::msg::Twist &command,
      uint64_t sequence,uint64_t epoch,
      const std::shared_ptr<const ControlMapSnapshot> &map,
      const std::shared_ptr<const LiveObstacleSnapshot> &live,
      const LiveObstacleSnapshot *collision_view=nullptr)
  {
    const double time=now().seconds();
    const auto current_map=std::atomic_load(&control_map_snapshot_);
    const auto current_live=std::atomic_load(&live_snapshot_);
    // timedYieldSafe proves actual present occupancy plus a TIME-INDEXED
    // moving reservation. Its current state cannot be reinterpreted against
    // the spatial union of every future reservation (that union can cover a
    // currently empty crossing precisely while the car is escaping it).
    const bool live_geometry=live && live->width>0 && live->height>0 &&
      live->present_hard.size()==size_t(live->width)*live->height;
    YieldFootprintCache views;
    if(!collision_view&&map&&live_geometry) {
      bindYieldCollisionViews(*map,*live,views);collision_view=views.current_view.get();
    }
    const auto state=map && live_geometry?queryCostLocked(odom.x,odom.y,odom.yaw,false,map.get(),true,true,collision_view):CostQuery{};
    const auto recent=[&](double stamp,double maximum){return time-stamp>=-.05 && time-stamp<=maximum;};
    const bool valid=map && live_geometry && current_map==map && current_live==live && odom.valid &&
      recent(map->stamp.seconds(),max_map_age_sec_) && recent(live->stamp_sec,live_obstacle_max_age_sec_) &&
      recent(odom.stamp.seconds(),max_odom_age_sec_) && state.known && !state.lethal &&
      std::isfinite(state.probability_upper) && state.probability_upper>=0 && state.probability_upper<=chance_state_limit_ &&
      std::isfinite(dynamic_path_upper_) && dynamic_path_upper_>=0 && dynamic_path_upper_<=chance_path_budget_ &&
      std::isfinite(dynamic_braking_upper_) && dynamic_braking_upper_>=0 && dynamic_braking_upper_<=chance_braking_budget_;
    bool committed=false;
    if(valid) {
      std::lock_guard<std::mutex> lock(goal_mutex_);
      if(have_goal_ && goal_seq_==sequence && input_epoch_.load()==epoch && !localization_fault_.load()) {
        publishCommand(command);committed=true;
      }
    }
    if(!committed) {
      dynamic_active_=false;dynamic_yield_state_="refused";
      dynamic_yield_reason_="dynamic certificate changed or expired before command commit";
      publishImmediateStop();publishChanceStatus(odom,kInf,false,dynamic_yield_reason_.c_str());return false;
    }
    if(command.linear.x<-.03 && !dynamic_commands_.empty() && motion_prediction_model_!="reachable") {
      if(!last_retreat_ || last_retreat_->origin!=dynamic_origin_ ||
         last_retreat_->goal!=sequence || last_retreat_->epoch!=epoch)
        last_retreat_=RetreatSchedule{sequence,epoch,dynamic_origin_,dynamic_commands_};
    } else if(command.linear.x>=0.)last_retreat_.reset();
    // Report the SAME immutable inputs that certified this command. A map
    // callback may replace the latest pointer while diagnostics are built;
    // mixing its state bound with the previous path bound is not a proof.
    dynamic_yield_state_=command.linear.x>0 ? "advance" : command.linear.x<0 ? "retreat" :
      (std::abs(odom.v)>.03 ? "brake" : "hold");
    dynamic_yield_reason_="timed trajectory certified";
    publishChanceStatus(odom,dynamic_braking_upper_,true,dynamic_yield_reason_.c_str(),map.get(),collision_view);
    return true;
  }

  // Ordering only: remaining forward travel for an Ackermann body cannot
  // ignore its terminal heading. Reuse the planner's six Dubins families in
  // rear-axle coordinates and the ACTUAL steering limit. This open-plane
  // distance is not an obstacle/terrain certificate, a reverse distance, or
  // a claim that the unobserved remainder of the task is executable.
  double interactionRemainingDistance(const terrain_nav::YieldState &state,const Pose2 &goal) const
  {
    if(!std::isfinite(state.x)||!std::isfinite(state.y)||!std::isfinite(state.yaw)||
       !std::isfinite(goal.x)||!std::isfinite(goal.y)||!std::isfinite(goal.yaw)||
       !std::isfinite(max_steer_)||max_steer_<=0||max_steer_>=M_PI/2||
       !std::isfinite(wheel_base_)||wheel_base_<=0)return kInf;
    const auto start=terrain_nav::rearAxlePose(Pose2{state.x,state.y,state.yaw,0.},rear_axle_offset_);
    const auto target=terrain_nav::rearAxlePose(goal,rear_axle_offset_);
    const double dx=target.x-start.x,dy=target.y-start.y,distance=std::hypot(dx,dy);
    const double radius=wheel_base_/std::tan(max_steer_);
    const double bearing=std::atan2(dy,dx),alpha=mod2pi(start.yaw-bearing),beta=mod2pi(goal.yaw-bearing);
    std::array<DubinsCandidate,6> candidates;
    const std::array<bool,6> valid={dubinsLSL(alpha,beta,distance/radius,candidates[0]),
      dubinsRSR(alpha,beta,distance/radius,candidates[1]),dubinsLSR(alpha,beta,distance/radius,candidates[2]),
      dubinsRSL(alpha,beta,distance/radius,candidates[3]),dubinsRLR(alpha,beta,distance/radius,candidates[4]),
      dubinsLRL(alpha,beta,distance/radius,candidates[5])};
    double best=kInf;
    for(size_t i=0;i<candidates.size();++i)if(valid[i]&&std::isfinite(candidates[i].length))
      best=std::min(best,radius*candidates[i].length);
    return best;
  }

  double interactionPolicyUtility(const terrain_nav::YieldState &root,
      const std::vector<terrain_nav::YieldState> &states,const Pose2 &goal) const
  {
    if(states.empty())return -kInf;
    const double initial=interactionRemainingDistance(root,goal),terminal=interactionRemainingDistance(states.back(),goal);
    if(!std::isfinite(initial)||!std::isfinite(terminal))return -kInf;
    const double euclidean=std::hypot(goal.x-root.x,goal.y-root.y)-
      std::hypot(goal.x-states.back().x,goal.y-states.back().y);
    // Replace just the old 3*Euclidean-progress term. All existing travel,
    // reverse, elapsed-motion, early-progress and steering preferences stay.
    return terrain_nav::interactionUtility(root,states,goal.x,goal.y)+3.*(initial-terminal-euclidean);
  }

  bool dynamicConflictsRepresented(const LiveObstacleSnapshot &live) const
  {
    if(dynamic_observed_only_)return true; // Current observed tracks still receive the full action proof.
    for(const auto &[id,stamp]:dynamic_unresolved_) {
      const bool present=std::any_of(live.tracks.begin(),live.tracks.end(),[&](const auto &t) {
        return t.motion_confirmed&&t.id==id;
      });
      const bool resolved=std::any_of(live.resolved_tracks.begin(),live.resolved_tracks.end(),[&](const auto &t) {
        return terrain_nav::freshStationaryResolution(t,id,stamp,now().seconds())&&t.stamp<=live.stamp_sec+1e-8;
      });
      if(!present&&!resolved)return false;
    }
    return true;
  }

  bool dynamicYieldControl(const OdomState &odom,bool input_recovery_only=false,
                           unsigned replacement_retry=0)
  {
    auto control_odom=odom;
    YieldFootprintCache footprint_cache;
    if(!replacement_retry) {
      dynamic_evaluation_start_=std::chrono::steady_clock::now();
      dynamic_input_reselections_=0;dynamic_selection_input_refreshes_=0;
    }
    const bool previous_evaluation=dynamic_evaluating_;
    dynamic_evaluating_=true;
    struct EvaluationScope {bool& active;bool previous;~EvaluationScope(){active=previous;}}
      evaluation_scope{dynamic_evaluating_,previous_evaluation};
    double time=now().seconds();
    dynamic_control_callback_origin_s_=time;
    uint64_t seq=0;Pose2 goal;
    {std::lock_guard<std::mutex> lock(goal_mutex_);if(have_goal_){seq=goal_seq_;goal=raw_goal_;}}
    const auto epoch=input_epoch_.load();
    auto live=std::atomic_load(&live_snapshot_);
    auto map=std::atomic_load(&control_map_snapshot_);
    dynamic_track_count_=live ? live->tracks.size() : 0;
    if(!dynamic_active_)dynamic_path_upper_=dynamic_braking_upper_=std::numeric_limits<double>::quiet_NaN();
    // The nominal replanner's fresh-interval debounce is not a certificate
    // for waiting in a moving object's path. During that debounce only a
    // newly certified safety maneuver may run. Actual map,
    // cloud and odometry source-age limits remain identical.
    const bool fresh=(!stale_inputs_waiting_.load() || input_recovery_only) && live && map && control_odom.valid && time>=live->stamp_sec-.05 &&
      time-live->stamp_sec<=live_obstacle_max_age_sec_ && time>=map->stamp.seconds()-.05 &&
      time-map->stamp.seconds()<=max_map_age_sec_ && time>=control_odom.stamp.seconds()-.05 &&
      time-control_odom.stamp.seconds()<=max_odom_age_sec_;
    if(seq!=dynamic_conflict_goal_ || spatial_epoch_.load()!=dynamic_conflict_spatial_ || time<dynamic_conflict_time_) {
      dynamic_unresolved_.clear();dynamic_conflict_goal_=seq;dynamic_conflict_spatial_=spatial_epoch_.load();
      last_retreat_.reset();
    }
    dynamic_conflict_time_=time;
    if(dynamic_observed_only_&&fresh) {
      for(auto i=dynamic_unresolved_.begin();i!=dynamic_unresolved_.end();) {
        const bool measured=std::any_of(live->tracks.begin(),live->tracks.end(),[&](const auto &t) {
          return t.motion_confirmed&&t.id==i->first&&std::isfinite(t.stamp)&&
            std::abs(t.stamp-live->stamp_sec)<=5e-9;
        });
        if(measured)++i;
        else {i=dynamic_unresolved_.erase(i);++dynamic_unobserved_conflicts_skipped_;}
      }
    }
    if(seq && live && chance_constrained_ && live_obstacle_enable_)
      retireGeometricallySeparatedConflicts(control_odom,time,*live);
    if(!reverse_recovery_enable_ || !reverse_enable_ || !live_obstacle_enable_ || !chance_constrained_ ||
       !seq || !fresh || (dynamic_active_ && (seq!=dynamic_goal_seq_ || epoch!=dynamic_epoch_ || time<dynamic_last_time_))) {
      const bool aborted=dynamic_active_;dynamic_active_=false;
      dynamic_yield_state_="idle";dynamic_yield_reason_="disabled, changed goal/clock, or stale inputs";
      if(aborted) {publishImmediateStop();publishChanceStatus(control_odom,kInf,false,dynamic_yield_reason_.c_str());return true;}
      return false;
    }
    for(const auto &resolved:live->resolved_tracks) {
      const auto found=dynamic_unresolved_.find(resolved.id);
      if(found!=dynamic_unresolved_.end() &&
         terrain_nav::freshStationaryResolution(resolved,resolved.id,found->second,time))
        dynamic_unresolved_.erase(found);
    }
    bool missing_prediction=false;
    for(auto &entry:dynamic_unresolved_) {
      const auto found=std::find_if(live->tracks.begin(),live->tracks.end(),[&](const auto &t){return t.motion_confirmed && t.id==entry.first;});
      if(found==live->tracks.end())missing_prediction=true;
      else entry.second=std::max(entry.second,found->stamp);
    }
    if(missing_prediction) {
      dynamic_active_=false;dynamic_yield_state_="prediction_lost_hold";
      dynamic_yield_reason_="conflict prediction missing; no positive stationary resolution";
      publishImmediateStop();publishChanceStatus(control_odom,kInf,false,dynamic_yield_reason_.c_str());return true;
    }
    // Unavailable evidence is not a POSITIVE geometric conflict. A coasting
    // track can expire between cloud acquisition and this control tick. The
    // boolean timedMovingUnsafe() deliberately fails closed for such input;
    // enrolling its ID as a confirmed conflict would latch even a remote wall
    // forever. Stop this tick without creating/deleting conflict identities.
    // Existing unresolved targets still take the missing-prediction gate
    // above; no age expiry can clear them or authorize motion.
    const bool motion_evidence_usable=std::all_of(live->tracks.begin(),live->tracks.end(),[&](const auto &track) {
      return std::isfinite(track.stamp) && time-track.stamp>=-.05 && time-track.stamp<=.6 &&
        (track.motion_confirmed || (center_candidate_actions_ && track.center_motion_fit && track.id==0)) &&
        std::isfinite(track.resolution) && track.resolution>0 && !track.cells.empty() &&
        std::isfinite(track.vx) && std::isfinite(track.vy) &&
        std::isfinite(track.acceleration_bound) && track.acceleration_bound>=0 && track.acceleration_bound<=4.5 &&
        std::isfinite(track.prediction_maximum_speed) && track.prediction_maximum_speed>0 && track.prediction_maximum_speed<=4.5 &&
        std::all_of(track.cells.begin(),track.cells.end(),[](const auto &cell) {
          return std::isfinite(cell[0]) && std::isfinite(cell[1]);
        });
    });
    if(!motion_evidence_usable) {
      dynamic_active_=false;dynamic_yield_state_="motion_input_hold";
      dynamic_yield_reason_="motion evidence unavailable; await fresh snapshot without inventing conflict";
      publishImmediateStop();publishChanceStatus(control_odom,kInf,false,dynamic_yield_reason_.c_str());return true;
    }
    const auto model=configuredYieldModel();
    terrain_nav::YieldState state;
    state.x=control_odom.x;state.y=control_odom.y;state.yaw=control_odom.yaw;state.speed=control_odom.v;
    state.steering=std::abs(control_odom.v)>.05 ? clampAbs(std::atan(control_odom.yaw_rate*wheel_base_/control_odom.v),max_steer_) :
      clampAbs(last_cmd_steer_/steer_output_sign_,max_steer_);
    double forward_limit=encounterSpeed(max_speed_,state.x,state.y,state.yaw,*live);
    // Ordinary approach requests keep their advisory encounter cap. A finite
    // escape selected when immediate braking is threatened may use
    // the existing physical operating limit, with a NEW complete certificate
    // on every continuation. Clipping it back to approach speed can itself
    // place the car in the crossing. No datasheet/risk/source limit changes.
    if(dynamic_active_ && terrain_nav::yieldCommandAt(dynamic_commands_,
        std::max(0.,time-dynamic_origin_)).speed>
          (dynamic_escape_policy_?max_speed_:forward_limit)+1e-9)dynamic_active_=false;
    double horizon=dynamic_active_ ? std::max(3.,dynamic_horizon_sec_) :
      (motion_prediction_model_!="reachable"?dynamic_anticipation_horizon_sec_:
        terrain_nav::interactionHorizon(state,model,reverse_max_speed_,1.2));
    if(horizon<=0) {publishImmediateStop();return true;}
    dynamic_horizon_sec_=horizon;
    std::vector<terrain_nav::YieldCommand> warm_commands;
    bool warm_escape_policy=false;
    const double active_elapsed=std::max(0.,time-dynamic_origin_);
    const double scheduled_speed=terrain_nav::yieldCommandAt(dynamic_commands_,active_elapsed).speed;
    const bool remaining_requested_motion=dynamic_active_&&
      (std::abs(scheduled_speed)>.03||
       std::any_of(dynamic_commands_.begin(),dynamic_commands_.end(),[&](const auto &command) {
         return command.t>active_elapsed&&std::abs(command.speed)>.03;
       }));
    const bool completed_motion=dynamic_active_&&std::abs(control_odom.v)<.03&&!remaining_requested_motion;
    const bool scheduled_wait=std::abs(scheduled_speed)<=.03&&remaining_requested_motion;
    // Ordinary-route debounce is not a directional safety constraint. Every
    // avoidance command below receives a NEW full proof from CURRENT fresh
    // inputs, including the final commit. Banning forward proposals during
    // that pause can leave a crossing's only safe escape unevaluated.
    if(((!dynamic_active_ || dynamic_commands_.empty()) && !dynamic_unresolved_.empty())||
        (completed_motion&&time-dynamic_origin_<horizon-1e-6)) {
      const auto stop=issuedYieldRollout(state,{{0,0,0}},model,horizon,time);
      bool executable_nominal=false;
      {std::lock_guard<std::mutex> lock(path_mutex_);
        executable_nominal=!input_recovery_only && have_path_ &&
          (!recovery_path_ || recovery_staged_gears_) && path_.size()>=2;
      }
      if(timedYieldSafe(stop,time,{{0,0,0}},*map,*live) &&
          (!executable_nominal || nominalDynamicHandoverSafe(control_odom,state,model,time,horizon,*map,*live,true))) {
        // Every unresolved identity was found in the fresh track set above.
        // Normal control still rechecks current path, body, support and the
        // final command; no cached command or certificate is published here.
        // When ordinary control has no executable path (including its input
        // recovery pause), its intended action is STOP, not indefinitely
        // holding the old avoidance steering. A CURRENT full stopping proof
        // can end this interaction. This is positive separation while every
        // target is still observed, never release because a track expired.
        dynamic_active_=false;dynamic_unresolved_.clear();last_retreat_.reset();
        dynamic_yield_state_="idle";dynamic_yield_reason_=executable_nominal?
          "fresh nominal continuation and stop recertified":"fresh stopping separation; ordinary controller paused";
        return false;
      }
    }
    if(dynamic_active_ && live->tracks.empty()) {
      if(dynamic_observed_only_) {
        dynamic_active_=false;dynamic_yield_state_="idle";last_retreat_.reset();
        dynamic_yield_reason_="observation-limited scope: no current motion model; normal occupancy checks remain";
        return false;
      }
      dynamic_active_=false;dynamic_yield_state_="refused";dynamic_yield_reason_="motion prediction lost";
      requestPlan("dynamic forecast ended; revalidate current route");
      publishImmediateStop();publishChanceStatus(control_odom,std::numeric_limits<double>::quiet_NaN(),false,dynamic_yield_reason_.c_str());return true;
    }
    if(dynamic_active_ && time-dynamic_origin_>=horizon-1e-6) {
      dynamic_active_=false;publishImmediateStop();requestPlan("dynamic yield completed");
      // A completed old rollout is not a certificate for the handover tick.
      // Recheck the actual residual velocity, current ground and new tracks.
      // Otherwise the old branch advertised an accepted stop with NaN brake
      // probability, even though the trial correctly rejected that claim.
      dynamic_path_upper_=dynamic_braking_upper_=std::numeric_limits<double>::quiet_NaN();
      const std::vector<terrain_nav::YieldCommand> stop{{0,0,0}};
      const auto stopping=issuedYieldRollout(state,stop,model,horizon,time);
      std::string failure;
      const bool certified=timedYieldSafe(stopping,time,stop,*map,*live,
          &dynamic_path_upper_,&dynamic_braking_upper_,&failure);
      dynamic_yield_state_=certified?"hold":"refused";
      dynamic_yield_reason_=certified?"handover stop certified":"handover stop: "+failure;
      publishChanceStatus(control_odom,certified?dynamic_braking_upper_:std::numeric_limits<double>::quiet_NaN(),
          certified,dynamic_yield_reason_.c_str());return true;
    }
    if(dynamic_active_) {
      state.t=time-dynamic_origin_;
      // Revalidation alone keeps executing an avoidable constant arc until
      // it becomes infeasible. On fresh motion observations, periodically
      // reconsider the unissued suffix with the old finite action as a warm
      // proposal. It must receive a NEW full proof before any publication.
      const bool refine=!completed_motion&&!scheduled_wait&&motion_prediction_model_!="reachable"&&dynamic_last_optimization_time_>0.&&
        time-dynamic_last_optimization_time_>=.25&&live->stamp_sec>dynamic_last_optimization_source_&&
        terrain_nav::yieldCommandAt(dynamic_commands_,state.t).speed>=-.03;
      if(refine) {
        warm_commands=terrain_nav::remainingYieldCommands(dynamic_commands_,state.t);
        warm_escape_policy=dynamic_escape_policy_;
        ++dynamic_reoptimizations_;dynamic_active_=false;state.t=0.;
        horizon=dynamic_anticipation_horizon_sec_;
      }
      // A continuing action is proved below, AFTER the root and immutable
      // inputs are refreshed, before any command can be published. Repeating
      // that full sampling/terrain proof on the older root here only doubles
      // the critical path. A failed final proof still invokes the same one
      // bounded fresh-root reselection and otherwise stops. No certificate
      // or probability bound from the previous callback grants permission.
    }
    if(!dynamic_active_) {
      // A lost continuation proof can interrupt a retreat for one scan. Its
      // unissued suffix may compete again from the actual root, using its
      // original switch times and an entirely new all-actor proof.
      if(warm_commands.empty() && last_retreat_ && last_retreat_->goal==seq &&
         last_retreat_->epoch==epoch && time>=last_retreat_->origin && !live->tracks.empty()) {
        auto retreat=terrain_nav::remainingYieldCommands(last_retreat_->commands,time-last_retreat_->origin);
        if(!retreat.empty() && retreat.front().speed<-.03)warm_commands=std::move(retreat);
        else last_retreat_.reset();
      }
      const std::vector<terrain_nav::YieldCommand> brake{{0,0,0}};
      const auto braking=issuedYieldRollout(state,brake,model,horizon,time);
      bool unsafe_stop=earlierStoppingUnsafe(braking,time,*live);
      // A matched local actor is removed from the unclassified approach cap
      // by encounterSpeed(), so a modeled crossing can be passed early. An
      // unrelated valid track must not lift a cap imposed by a different,
      // unclassified nearby body. Only a threatened stop can justify trying
      // an escape above that cap, subject to the complete current proof.
      const auto has_current_model=[&](const LiveObstacleSnapshot &observed) {
        return std::any_of(observed.tracks.begin(),observed.tracks.end(),[&](const auto&t) {
          return terrain_nav::currentMeasuredIdentity(t,observed.stamp_sec,false)||
            (center_candidate_actions_&&dynamic_time_indexed_routing_&&
             terrain_nav::currentMeasuredCenterModel(t,observed.stamp_sec));
        });
      };
      bool escape_proposals=unsafe_stop&&motion_prediction_model_!="reachable"&&has_current_model(*live);
      const double proposal_speed=escape_proposals?max_speed_:forward_limit;
      dynamic_escape_policy_=false;
      dynamic_approach_limit_=forward_limit;dynamic_proposal_speed_limit_=proposal_speed;
      const std::vector<terrain_nav::YieldCommand> held{{0,state.speed,state.steering}};
      const auto holding=issuedYieldRollout(state,held,model,horizon,time);
      bool ordinary_paused=input_recovery_only;
      {std::lock_guard<std::mutex> lock(path_mutex_);ordinary_paused|=!have_path_ || path_.size()<2;}
      bool can_handover=live->tracks.empty();
      if(warm_commands.empty() && !can_handover && !unsafe_stop) {
        if(ordinary_paused)can_handover=timedYieldSafe(braking,time,brake,*map,*live);
        if(!can_handover) {
          // The same nominal action used to be checked twice when an earlier
          // branch failed. Test it once on this immutable root and sources.
          // Reachable mode still admits separation when the held motion and
          // stop are clear, including configurations with no nominal path.
          const bool reachable=motion_prediction_model_=="reachable";
          can_handover=reachable && !timedMovingUnsafe(holding,time,*live);
          if(!can_handover && !ordinary_paused &&
             nominalDynamicHandoverSafe(control_odom,state,model,time,horizon,*map,*live,true)) {
            can_handover=(!reachable && !timedMovingUnsafe(holding,time,*live)) ||
              timedYieldSafe(braking,time,brake,*map,*live);
          }
        }
      }
      if(warm_commands.empty() && can_handover) {
        // Fresh representations of every still-unresolved identity exist;
        // both current measured braking and held motion now avoid them.
        // No old/missing track can reach this release via expiry alone.
        if(!input_recovery_only)dynamic_unresolved_.clear();
        last_retreat_.reset();
        dynamic_yield_state_="idle";dynamic_yield_reason_="no predicted conflict";return false;
      }
      std::vector<std::pair<uint64_t,double>> witnessed_conflicts;
      const auto body=collisionEnvelope(footprint_margin_);
      const terrain_nav::MotionCheckGeometry conflict_geometry{body.front,body.back,body.half_width,
        max_steer_,wheel_base_,live_obstacle_inflation_radius_};
      const bool reachable=motion_prediction_model_=="reachable";
      const auto stop_envelope=reachable ? terrain_nav::EarlyStopEnvelope{} :
        terrain_nav::earlyStopEnvelope(braking,conflict_geometry);
      const auto stop_locus=reachable ? terrain_nav::possibleEarlyStops(braking) :
        std::vector<terrain_nav::YieldState>{};
      for(const auto &track:live->tracks) {
        LiveObstacleSnapshot one;one.tracks={track};
        if(!track.motion_confirmed||!track.id)continue;
        bool potential_conflict=false;
        if(!reachable) {
          const auto stop_risk=stop_envelope.valid?
            jointMovingRisk(stop_envelope.stationary,time,one,false,stop_envelope.expansion):
            terrain_nav::JointMotionRisk{};
          const auto known=[](const auto &risk){return risk.valid||risk.deterministic_conflict;};
          const bool stop_conflict=stop_risk.deterministic_conflict||
            (stop_risk.valid&&stop_risk.upper>motion_joint_risk_limit_);
          const auto held_risk=known(stop_risk)&&!stop_conflict?jointMovingRisk(holding,time,one):stop_risk;
          if(!known(stop_risk)||!known(held_risk)) {
            // A failed predictor/shape/time validation is not POSITIVE
            // conflict evidence. Still stop, without enrolling a remote ID
            // forever or clearing any previously unresolved identity.
            dynamic_active_=false;dynamic_yield_state_="motion_input_hold";
            dynamic_yield_reason_="conflict classification unavailable: "+
              (!known(stop_risk)?stop_risk.error:held_risk.error);
            publishImmediateStop();publishChanceStatus(control_odom,kInf,false,
              dynamic_yield_reason_.c_str());return true;
          }
          potential_conflict=stop_risk.deterministic_conflict||held_risk.deterministic_conflict||
            stop_risk.upper>motion_joint_risk_limit_||held_risk.upper>motion_joint_risk_limit_;
        } else {
          const auto stop=terrain_nav::assessTimedMotion(stop_locus,time,{track},conflict_geometry);
          const auto held=stop.valid&&!stop.unsafe?
            terrain_nav::assessTimedMotion(holding,time,{track},conflict_geometry):stop;
          if(!stop.valid||!held.valid) {
            dynamic_active_=false;dynamic_yield_state_="motion_input_hold";
            dynamic_yield_reason_="conflict classification unavailable: invalid reachable evidence";
            publishImmediateStop();publishChanceStatus(control_odom,kInf,false,dynamic_yield_reason_.c_str());return true;
          }
          potential_conflict=stop.unsafe||held.unsafe;
        }
        if(potential_conflict)witnessed_conflicts.emplace_back(track.id,track.stamp);
      }
      for(const auto &entry:witnessed_conflicts)dynamic_unresolved_[entry.first]=entry.second;
      dynamic_commands_.clear();
      std::string brake_failure="not certified",retreat_failure="not certified";
      dynamic_forward_attempts_=dynamic_retreat_attempts_=0;
      dynamic_proposals_generated_=dynamic_terminal_rejections_=dynamic_screen_survivors_=0;dynamic_screen_ms_=0.;
      dynamic_terminal_candidates_checked_=0;
      dynamic_long_generated_=dynamic_long_terminal_rejections_=dynamic_long_survivors_=0;
      dynamic_screen_min_estimate_=NAN;
      dynamic_screen_min_lower_=NAN;dynamic_screen_discard_error_=0.;
      dynamic_screen_coarse_discarded_=dynamic_screen_refined_candidates_=0;
      dynamic_forward_failure_="no dynamically admissible forward proposal";
      struct Proposal {
        std::vector<terrain_nav::YieldCommand> commands;
        std::vector<terrain_nav::YieldState> states;
        double utility=0;
        double screen_risk=1.; // Ordering only, never an execution certificate.
        int family=0; // 0 brake, 1 forward/side pass, -1 retreat
        double requested_distance=0.;
        bool warm=false;
        bool finish_only=false;
      };
      std::vector<Proposal> proposals;
      const double proposal_limit=motion_prediction_model_!="reachable"?6.:horizon;
      bindYieldCollisionViews(*map,*live,footprint_cache);
      // Evaluate the scene against ALL measured actors before ranking actions.
      // A threatened stop is not automatically preferable to safe progress.
      const auto at_goal=[&](const terrain_nav::YieldState &end) {
        return std::hypot(end.x-goal.x,end.y-goal.y)<=goal_reached_distance_;
      };
      const auto propose=[&](std::vector<terrain_nav::YieldCommand> commands,int family,double requested_distance=0.,
                            bool warm=false,bool finish_only=false) {
        if(!commands.empty())
          proposals.push_back({std::move(commands),{},0.,1.,family,requested_distance,warm,finish_only});
      };
      if(!warm_commands.empty()) {
        const double speed=warm_commands.front().speed;
        propose(warm_commands,speed>0?1:speed<0?-1:0,0.,true);
      }
      propose(brake,0);
      // Use the tracker's short terminal arc as another timed policy. The
      // ordinary escape library starts at 0.9 m forward and 0.4 m reverse,
      // which can skip a reachable goal between consecutive receding roots.
      if(motion_prediction_model_!="reachable" && !input_recovery_only) {
        auto terminal=nominalInteractionCommands(control_odom,state,model,proposal_limit,*live,true);
        if(!terminal.empty()) {
          const int gear=terminal.front().speed>0 ? 1 : -1;
          if(gear<0 || dynamic_acceleration_escape_enable_)propose(std::move(terminal),gear);
        }
        for(auto &candidate:nearbyGoalCommands(state,goal,model,proposal_limit,forward_limit)) {
          const int gear=candidate.commands.front().speed>0 ? 1 : -1;
          if(gear>0 && !dynamic_acceleration_escape_enable_)continue;
          propose(std::move(candidate.commands),gear,candidate.distance,false,true);
        }
        // A crossing may block the route now while leaving a safe place to
        // yield outside it. Try a bounded pause and onward motion as ONE
        // policy, so both waiting and departure see the same moving actors.
        if(!unsafe_stop&&warm_commands.empty()&&dynamic_acceleration_escape_enable_) {
          auto stopped=state;
          stopped.speed=0.;
          const auto advance=nominalInteractionCommands(control_odom,stopped,model,proposal_limit,*live);
          const double brake_time=terrain_nav::maximumYieldDelay(model)+std::abs(state.speed)/model.deceleration;
          for(double wait:{.2,.6,1.0})
            propose(terrain_nav::waitThenAdvance(advance,brake_time+wait,state.steering),1);
        }
      }
      // Task bearing orders actions; it is not a feasibility constraint.
      // After a lateral escape the goal may be behind the car. Removing all
      // forward turns then can eliminate the very maneuver needed to rejoin.
      if(dynamic_acceleration_escape_enable_) {
        for(auto &candidate:terrain_nav::forwardYieldProposals(state,model,proposal_speed,max_steer_,
             proposal_limit,motion_prediction_model_!="reachable"))
          propose(std::move(candidate.commands),1,candidate.distance);
        if(motion_prediction_model_!="reachable")
          for(auto &candidate:terrain_nav::turnReleaseYieldProposals(proposal_speed,max_steer_))
            propose(std::move(candidate.commands),1);
      }
      for(auto &candidate:terrain_nav::reverseYieldProposals(state,model,reverse_max_speed_,
           max_steer_,proposal_limit,motion_prediction_model_!="reachable"))
        propose(std::move(candidate.commands),-1,candidate.distance);
#ifdef TERRAIN_NAV_BENCHMARK
      if(dynamic_before_selection_proof_test_hook_)dynamic_before_selection_proof_test_hook_();
#endif
      // Generate command schedules first. Integrate their swept paths once,
      // from the same root and sources used for screening and candidate proofs.
      // Publication still uses a fresh independent proof and exact source check.
      double refresh_elapsed=0.;
      if(!reachable && !proposals.empty()) {
        const auto next_map=std::atomic_load(&control_map_snapshot_);
        const auto next_live=std::atomic_load(&live_snapshot_);
        OdomState next_odom;
        {std::lock_guard<std::mutex> lock(odom_mutex_);next_odom=odom_;}
        if(next_map&&next_live&&next_odom.valid) {
          predictOdomForControl(next_odom,false);
          const double root_time=next_odom.stamp.seconds()+next_odom.prediction_sec;
          if(root_time>=time && (next_map!=map||next_live!=live||root_time-time>.010)) {
            refresh_elapsed=root_time-time;
            map=next_map;live=next_live;control_odom=next_odom;time=root_time;
            state={0.,control_odom.x,control_odom.y,control_odom.yaw,control_odom.v,
              std::abs(control_odom.v)>.05 ? clampAbs(std::atan(control_odom.yaw_rate*wheel_base_/control_odom.v),max_steer_) :
                clampAbs(last_cmd_steer_/steer_output_sign_,max_steer_)};
            ++dynamic_selection_input_refreshes_;dynamic_track_count_=live->tracks.size();
            forward_limit=encounterSpeed(max_speed_,state.x,state.y,state.yaw,*live);
            dynamic_approach_limit_=forward_limit;
            const auto current_brake=issuedYieldRollout(state,brake,model,horizon,time);
            unsafe_stop=earlierStoppingUnsafe(current_brake,time,*live);
            escape_proposals=escape_proposals&&unsafe_stop&&has_current_model(*live);
            bindYieldCollisionViews(*map,*live,footprint_cache);
            if(!dynamicConflictsRepresented(*live)) {
              dynamic_yield_state_="prediction_lost_hold";
              dynamic_yield_reason_="conflict prediction missing in refreshed selection inputs";
              publishImmediateStop();publishChanceStatus(control_odom,kInf,false,dynamic_yield_reason_.c_str(),map.get(),footprint_cache.current_view.get());return true;
            }
          }
        }
      }
      const auto issued=issuedCommandSnapshot();
      proposals.erase(std::remove_if(proposals.begin(),proposals.end(),[&](auto &p) {
        if(p.warm && refresh_elapsed>0.) {
          p.commands=terrain_nav::remainingYieldCommands(p.commands,refresh_elapsed);
          if(p.commands.empty())return true;
          p.family=p.commands.front().speed>0?1:p.commands.front().speed<0?-1:0;
        }
        if(!escape_proposals&&!(p.warm&&warm_escape_policy)&&
            std::any_of(p.commands.begin(),p.commands.end(),[&](const auto &command) {
              return command.speed>forward_limit+1e-9;
            }))return true;
        if(!p.warm && !p.family && unsafe_stop)return true;
        p.states=terrain_nav::rolloutQueuedYield(state,p.commands,model,proposal_limit,time,issued);
        if(!reachable)p.states=terrain_nav::finiteJointRollout(std::move(p.states));
        else if(p.family>0)p.states=terrain_nav::finiteAdvanceRollout(std::move(p.states));
        if(p.states.size()<11||std::abs(p.states[p.states.size()-11].speed)>.001||
           (p.finish_only&&!at_goal(p.states.back()))||
           (reachable && timedMovingUnsafe(p.states,time,*live)))return true;
        p.utility=interactionPolicyUtility(state,p.states,goal);
        if(!std::isfinite(p.utility))return true;
        ++dynamic_proposals_generated_;
        if(p.requested_distance>3.000001)++dynamic_long_generated_;
        return false;
      }),proposals.end());
      if(motion_prediction_model_!="reachable" && !proposals.empty()) {
        // Apply this veto only AFTER adopting current inputs, for the same
        // reason stochastic screening must not discard on an older source.
        proposals.erase(std::remove_if(proposals.begin(),proposals.end(),[&](const auto&p) {
          return unconfirmedMotionUnsafe(p.states,time,*live);
        }),proposals.end());
        if(proposals.empty())dynamic_forward_failure_="unconfirmed motion reachability";
      }
      if(motion_prediction_model_!="reachable" && !proposals.empty()) {
        std::vector<std::vector<terrain_nav::YieldState>> candidates;candidates.reserve(proposals.size());
        for(const auto &p:proposals)candidates.push_back(p.states);
        const auto body=collisionEnvelope(footprint_margin_);
        const terrain_nav::MotionCheckGeometry geometry{body.front,body.back,body.half_width,max_steer_,wheel_base_,live_obstacle_inflation_radius_};
        const auto screen_seed=motion_joint_seed_^(++motion_joint_checks_);
        // A small reverse-only batch is cheaper in one pass. Large mixed
        // batches can reject obvious conflicts before the expensive pass.
        const auto screen=candidates.size()>32?
          terrain_nav::screenJointMotionProgressiveBatch(candidates,time,live->tracks,geometry,256,
            screen_seed,motion_prediction_model_=="gaussian_joint_cuda",motion_joint_risk_limit_):
          terrain_nav::screenJointMotionBatch(candidates,time,live->tracks,geometry,256,
            screen_seed,motion_prediction_model_=="gaussian_joint_cuda");
        dynamic_screen_ms_=screen.elapsed_ms;
        dynamic_screen_coarse_discarded_=screen.coarse_discarded;dynamic_screen_refined_candidates_=screen.refined_candidates;
        if(screen.valid&&!screen.estimates.empty())dynamic_screen_min_estimate_=*std::min_element(screen.estimates.begin(),screen.estimates.end());
        if(screen.valid&&!screen.lower_bounds.empty())dynamic_screen_min_lower_=*std::min_element(screen.lower_bounds.begin(),screen.lower_bounds.end());
        dynamic_screen_discard_error_=screen.discard_confidence_error;
        std::vector<Proposal> survivors;survivors.reserve(proposals.size());
        if(screen.valid&&screen.lower_bounds.size()==proposals.size()) {
          for(size_t i=0;i<proposals.size();++i)
            if(std::isfinite(screen.lower_bounds[i])&&
                (proposals[i].warm||screen.lower_bounds[i]<=motion_joint_risk_limit_)) {
              proposals[i].screen_risk=screen.estimates[i];
              // Prefer risk headroom instead of treating every sub-threshold
              // candidate as equally good. Screening cannot grant permission:
              // the chosen action still receives a fresh full CP-bound proof.
              proposals[i].utility=terrain_nav::interactionRiskPriority(proposals[i].utility,
                screen.estimates[i],motion_joint_risk_limit_,dynamic_risk_preference_weight_);
              if(std::isfinite(proposals[i].utility))survivors.push_back(std::move(proposals[i]));
            }
        } else dynamic_forward_failure_="batch motion screening: "+screen.error;
        proposals=std::move(survivors);
      }
      // Dynamic screening is advisory. Check terminal support lazily in the
      // proof order below, under the SAME search/source budget. Prechecking
      // every terminal spent the freshness margin before even the first full
      // proof. Every selected candidate still receives this exact predicate,
      // its whole-trajectory proof and the fresh-root final commit proof.
      dynamic_screen_survivors_=proposals.size();
      for(const auto &p:proposals)if(p.requested_distance>3.000001)++dynamic_long_survivors_;
      // Geometry/kinematics utility is an ordering, not permission to move.
      // Family fairness only orders proofs. Keep a certified fallback while
      // checking still-better policies; accepting the first fair-order success
      // made brake/retreat suppress lower-ranked but useful forward motion.
      const auto settles_at_goal=[&](const Proposal &proposal) {
        return at_goal(proposal.states.back());
      };
      // Completion is already defined by a position tolerance. A stopped
      // endpoint inside it must not lose to a longer forward Dubins arc.
      if(std::any_of(proposals.begin(),proposals.end(),settles_at_goal)) {
        const auto best_utility=std::max_element(proposals.begin(),proposals.end(),
          [](const auto &a,const auto &b){return a.utility<b.utility;})->utility;
        for(auto &proposal:proposals)if(settles_at_goal(proposal)) {
          const auto &end=proposal.states.back();
          proposal.utility=best_utility+1.+goal_reached_distance_-std::hypot(end.x-goal.x,end.y-goal.y);
        }
      }
      std::stable_sort(proposals.begin(),proposals.end(),[](const auto &a,const auto &b) {
        return a.utility>b.utility;
      });
      std::vector<int> families;std::vector<double> screen_risks;families.reserve(proposals.size());
      for(const auto &p:proposals){families.push_back(p.family);screen_risks.push_back(p.screen_risk);}
      auto order=motion_prediction_model_=="reachable"?terrain_nav::interactionBranchOrder(families):
        terrain_nav::interactionProofOrder(families,screen_risks,unsafe_stop);
      // Prove the incumbent FIRST. A fully certified forward action is
      // dispatched immediately; delaying it for utility refinement can make
      // every escape infeasible before publication. A brake/retreat remains
      // a fallback while a better forward action is still being checked.
      // Screening and early selection never replace the final current-source
      // validation below. Further improvement belongs to the receding update.
      for(size_t i=0;i<proposals.size();++i)if(proposals[i].warm) {
        order.erase(std::remove(order.begin(),order.end(),i),order.end());order.insert(order.begin(),i);break;
      }
      std::stable_partition(order.begin(),order.end(),[&](size_t i){return settles_at_goal(proposals[i]);});
      const auto selection_start=std::chrono::steady_clock::now();
      int checked=0;
      const double source_expiry=std::min({map->stamp.seconds()+max_map_age_sec_,
        live->stamp_sec+live_obstacle_max_age_sec_,control_odom.stamp.seconds()+max_odom_age_sec_});
      double proof_seconds=.005;
      dynamic_proof_deadline_stops_=0;
      std::vector<double> utility,path_bounds(proposals.size()),braking_bounds(proposals.size());
      utility.reserve(proposals.size());for(const auto &p:proposals)utility.push_back(p.utility);
      const auto best=terrain_nav::bestCertifiedInteraction(utility,order,[&](size_t index) {
        auto &proposal=proposals[index];
        ++dynamic_terminal_candidates_checked_;
        double cost=0,clearance=kInf;const auto &end=proposal.states.back();
        if(!footprintCostLocked(end.x,end.y,end.yaw,cost,&clearance,nullptr,
            false,map.get(),footprint_cache.future_view.get()) ||
            (std::isfinite(clearance)&&clearance<hardClearanceFloor())) {
          ++dynamic_terminal_rejections_;
          if(proposal.requested_distance>3.000001)++dynamic_long_terminal_rejections_;
          if(proposal.family>0)dynamic_forward_failure_="terminal terrain/occupancy footprint";
          else if(proposal.family<0)retreat_failure="terminal terrain/occupancy footprint";
          else brake_failure="terminal terrain/occupancy footprint";
          return false;
        }
        ++checked;
        if(proposal.family>0)++dynamic_forward_attempts_;
        if(proposal.family<0)++dynamic_retreat_attempts_;
        double path_upper=0,braking_upper=0;std::string failure,details;
        const auto proof_start=std::chrono::steady_clock::now();
        // A marginal proposal must not consume a new 65,536-draw proof while
        // other, better separated policies are waiting. The original full
        // sample count, lifetime confidence allocation and upper-bound test
        // still apply. A nonpassing bound remains a refusal; screening is
        // never substituted for it. Final current-input validation below
        // retains the ordinary independent proof/refinement policy.
        const bool proved=timedYieldSafe(proposal.states,time,proposal.commands,*map,*live,
                          &path_upper,&braking_upper,&failure,&details,&footprint_cache,false);
        proof_seconds=std::max(proof_seconds,std::chrono::duration<double>(
          std::chrono::steady_clock::now()-proof_start).count());
        if(!proved) {
          const auto diagnostic=failure+" ["+details+"]";
          if(proposal.family>0)dynamic_forward_failure_=diagnostic;
          else if(proposal.family<0)retreat_failure=diagnostic;
          else brake_failure=diagnostic;
          return false;
        }
        path_bounds[index]=path_upper;braking_bounds[index]=braking_upper;
        return true;
      },[&] {
        if(!terrain_nav::interactionProofFitsFreshness(now().seconds(),source_expiry,proof_seconds)) {
          ++dynamic_proof_deadline_stops_;return false;
        }
        return checked<12 && (!dynamic_terminal_candidates_checked_ ||
          std::chrono::steady_clock::now()-selection_start<=std::chrono::milliseconds(35));
      },[&](size_t index) {
        const auto &proposal=proposals[index];
        return settles_at_goal(proposal) || (proposal.warm && proposal.family<0) ||
          (proposal.family>0 && !proposal.warm);
      });
      if(best!=std::numeric_limits<size_t>::max()) {
        dynamic_commands_=std::move(proposals[best].commands);
        dynamic_escape_policy_=(escape_proposals||(proposals[best].warm&&warm_escape_policy))&&
          dynamic_commands_.front().speed>forward_limit+1e-9;
        dynamic_path_upper_=path_bounds[best];dynamic_braking_upper_=braking_bounds[best];
        horizon=proposals[best].states.back().t;
      }
      if(dynamic_commands_.empty()) {
        dynamic_yield_state_="refused";
        dynamic_yield_reason_="advance: "+dynamic_forward_failure_+"; brake: "+brake_failure+"; retreat: "+retreat_failure;
        RCLCPP_WARN_THROTTLE(get_logger(),*get_clock(),1000,"dynamic yield: no certified escape (tracks=%zu): %s",
          live->tracks.size(),dynamic_yield_reason_.c_str());
        publishImmediateStop();publishChanceStatus(control_odom,kInf,false,dynamic_yield_reason_.c_str(),map.get(),footprint_cache.current_view.get());return true;
      }
      dynamic_initial_=state;dynamic_origin_=time;dynamic_goal_seq_=seq;dynamic_epoch_=epoch;dynamic_active_=true;
      dynamic_last_optimization_time_=time;dynamic_last_optimization_source_=live->stamp_sec;
      dynamic_horizon_sec_=horizon;
      RCLCPP_INFO_THROTTLE(get_logger(),*get_clock(),1000,"dynamic yield selected: tracks=%zu horizon=%.1fs mode=%s",
        live->tracks.size(),horizon,dynamic_commands_.front().speed>0 ? "advance" : dynamic_commands_.front().speed<0 ? "retreat" : "brake");
      requestPlan("dynamic avoidance selected; refresh route");
    }
    dynamic_last_time_=time;
#ifdef TERRAIN_NAV_BENCHMARK
    if(dynamic_before_commit_test_hook_)dynamic_before_commit_test_hook_();
#endif
    auto commit_map=map;auto commit_live=live;auto command_odom=control_odom;
    // A normal 10Hz input replacement during candidate selection is not a
    // reason to inject a stop into a passing maneuver. Revalidate against
    // the replacement, bounded to two attempts; never reuse the old proof
    // for the new inputs. A failed/newly stale proof still stops below.
    for(int attempt=0;attempt<2;++attempt) {
      const auto next_map=std::atomic_load(&control_map_snapshot_);
      const auto next_live=std::atomic_load(&live_snapshot_);
      if(!next_map || !next_live)break;
      // Refresh the ROOT as well as its clock. Relabelling the old selection
      // position with a later time invents a slower car in a tight crossing.
      bool refreshed_odom=false;command_odom=control_odom;
      {std::lock_guard<std::mutex> lock(odom_mutex_);
        if(odom_.valid){command_odom=odom_;refreshed_odom=true;}
      }
      if(refreshed_odom)predictOdomForControl(command_odom,false);
      else command_odom=terrain_nav::integrateBodyVelocity(command_odom,command_odom.v,
        command_odom.yaw_rate,std::max(0.,now().seconds()-time),rear_axle_offset_);
      auto actual=state;actual.x=command_odom.x;actual.y=command_odom.y;actual.yaw=command_odom.yaw;
      actual.speed=command_odom.v;
      actual.steering=std::abs(command_odom.v)>.05?
        clampAbs(std::atan(command_odom.yaw_rate*wheel_base_/command_odom.v),max_steer_):state.steering;
      actual.t=std::max(0.,now().seconds()-dynamic_origin_);
      const auto remainder=issuedYieldRollout(actual,dynamic_commands_,model,std::max(horizon,actual.t+3.),dynamic_origin_);
      std::string failure,details;
      if(!dynamicConflictsRepresented(*next_live))failure="conflict prediction missing in replacement inputs";
      if(!failure.empty() || !timedYieldSafe(remainder,dynamic_origin_,dynamic_commands_,*next_map,*next_live,
          &dynamic_path_upper_,&dynamic_braking_upper_,&failure,&details,&footprint_cache)) {
        dynamic_active_=false;dynamic_yield_state_="refused";
        // A new map/root can invalidate the SELECTED action without ruling
        // out every alternative. Retry selection once from the current root
        // and current immutable inputs, before publishing anything. The old
        // certificate is discarded; all risk, support, freshness and final
        // commit checks still apply. A second failure stops, never recurses.
        if(!replacement_retry) {
          ++dynamic_input_reselections_;
          return dynamicYieldControl(command_odom,input_recovery_only,1);
        }
        dynamic_yield_reason_="new inputs rejected dynamic continuation: "+failure+" ["+details+"]";
        auto present=*next_live;present.hard=next_live->present_hard;
        publishImmediateStop();publishChanceStatus(command_odom,kInf,false,dynamic_yield_reason_.c_str(),next_map.get(),&present);return true;
      }
      commit_map=next_map;commit_live=next_live;
      if(std::atomic_load(&control_map_snapshot_)==commit_map && std::atomic_load(&live_snapshot_)==commit_live)break;
    }
    const auto selected=terrain_nav::yieldCommandAt(dynamic_commands_,now().seconds()-dynamic_origin_);
    geometry_msgs::msg::Twist cmd;cmd.linear.x=selected.speed;cmd.angular.z=selected.steering*steer_output_sign_;
    bindYieldCollisionViews(*commit_map,*commit_live,footprint_cache);
    commitDynamicCommand(command_odom,cmd,seq,epoch,commit_map,commit_live,footprint_cache.current_view.get());return true;
  }

  bool commandBrakingEnvelopeSafe(const OdomState &odom, const geometry_msgs::msg::Twist &cmd,
                                  double *probability_upper = nullptr) const
  {
    if (probability_upper) *probability_upper=std::numeric_limits<double>::quiet_NaN();
    const auto map = std::atomic_load(&control_map_snapshot_);
    if (!map) return false;
    const double origin=now().seconds();
    const auto live=dynamic_time_indexed_routing_?std::atomic_load(&live_snapshot_):
      std::shared_ptr<const LiveObstacleSnapshot>{};
    if(dynamic_time_indexed_routing_ && (!chance_constrained_ || !live_obstacle_enable_ || !live ||
        origin-live->stamp_sec<-.05 || origin-live->stamp_sec>live_obstacle_max_age_sec_ ||
        origin-map->stamp.seconds()<-.05 || origin-map->stamp.seconds()>max_map_age_sec_ ||
        origin-odom.stamp.seconds()<-.05 || origin-odom.stamp.seconds()>max_odom_age_sec_))return false;
    const double initial_speed = std::max(std::abs(odom.v), std::abs(cmd.linear.x));
    const double measured_direction = std::abs(odom.v)>1e-3 ? std::copysign(1.,odom.v) :
      (cmd.linear.x<0 ? -1. : 1.);
    const double requested_direction = std::abs(cmd.linear.x)>1e-3 ?
      std::copysign(1.,cmd.linear.x) : measured_direction;
    const double target_steering=clampAbs(cmd.angular.z/steer_output_sign_,max_steer_);
    const double initial_steering=std::abs(odom.v)>.05 ?
      clampAbs(std::atan(odom.yaw_rate*wheel_base_/odom.v),max_steer_) : target_steering;
    // Static side clearance is not a longitudinal stopping distance. Bound
    // the ACTUAL swept braking motion instead, with no smaller latency than
    // the former radial cap: actuator + one scan + original map source age.
    const double dt=.04, delay=.35+.10+std::max(0.,now().seconds()-map->stamp.seconds());
    const double deceleration=std::min(max_accel_,dynamic_yield_deceleration_);
    if(!std::isfinite(deceleration)||deceleration<=0)return false;
    const double horizon=delay+initial_speed/deceleration+.08;
    terrain_nav::ChanceBudget probability;
    std::vector<terrain_nav::YieldState> dynamic_branches;
    const auto record_rejection=[&](const char *reason,double x,double y,double yaw,double t) {
      try {
      const auto q=queryCostLocked(x,y,yaw,false,map.get());
      bool known=false;const bool body=collisionBodyUnsafe(map->grid,x,y,yaw,known);
      static rclcpp::Clock log_clock(RCL_STEADY_TIME);
      RCLCPP_WARN_THROTTLE(get_logger(),log_clock,500,
        "terrain braking rejected: source_ns=%lld now_ns=%lld root=(%.9f,%.9f,%.9f,%.9f,%.9f) cmd=(%.9f,%.9f) sample=(%.9f,%.9f,%.9f) t=%.3f cause=%s support=%d upper=%.9g body=%d",
        static_cast<long long>(map->stamp.nanoseconds()),static_cast<long long>(now().nanoseconds()),
        odom.x,odom.y,odom.yaw,odom.v,odom.yaw_rate,cmd.linear.x,cmd.angular.z,x,y,yaw,t,reason,int(q.known),q.probability_upper,int(body));
      } catch(...) { /* Diagnostic failure cannot change a safety rejection. */ }
    };
    // A gear-change request cannot reverse the existing momentum. Check the
    // measured and requested directions, sharing one conservative risk budget.
    const int branches=measured_direction==requested_direction ? 1 : 2;
    for(int branch=0;branch<branches;++branch) {
      const double direction=branch==0 ? measured_direction : requested_direction;
      double x=odom.x,y=odom.y,yaw=odom.yaw,speed=initial_speed,steering=initial_steering;
      std::vector<terrain_nav::YieldState> timed;
      for (double t=0; t<=horizon; t+=dt) {
        if(live)timed.push_back({t,x,y,yaw,direction*speed,steering});
        double cost=0, clearance=kInf;
        if (!footprintCostLocked(x,y,yaw,cost,&clearance,nullptr,false,map.get(),live.get()) ||
            (std::isfinite(clearance) && clearance<hardClearanceFloor())) {
          record_rejection("footprint",x,y,yaw,t);return false;
        }
        if (!addChancePose(x,y,yaw,probability,chance_braking_budget_,map.get(),live.get())) {
          if (probability_upper) *probability_upper=probability.upper();
          record_rejection("joint_budget",x,y,yaw,t);
          return false;
        }
        const double rate=direction*speed*std::tan(steering)/wheel_base_;
        const auto next=terrain_nav::integrateBodyVelocity(Pose2{x,y,yaw},direction*speed,rate,dt,rear_axle_offset_);
        x=next.x;y=next.y;yaw=next.yaw;
        steering+=std::clamp(target_steering-steering,-max_steer_rate_*dt,max_steer_rate_*dt);
        if (t>=delay) speed=std::max(0.0,speed-deceleration*dt);
      }
      if(live && (motion_prediction_model_=="reachable" || !live->tracks.empty())) {
        // The terminal stop must not become a target for a later crossing.
        // Directional reachable segments retain intermediate slowing/stops;
        // checking this stationary tail therefore also covers arrivals
        // between the stopping instant and the interaction horizon.
        if(timed.empty() || std::abs(timed.back().speed)>.001)return false;
        if(motion_prediction_model_=="reachable") {
          auto tail=timed.back();tail.t=std::max(horizon,6.);timed.push_back(tail);
          if(timedMovingUnsafe(timed,origin,*live) ||
             timedMovingUnsafe(terrain_nav::possibleEarlyStops(timed),origin,*live)) {
            record_rejection("time_indexed_motion",odom.x,odom.y,odom.yaw,0.);return false;
          }
        } else {
          // Gaussian positions do not retain all previous locations. A jump
          // from stopping to t=6 silently missed intervening crossings.
          // Sample the whole rolling tail and both possible gear branches.
          const double end=std::max(3.,timed.back().t);
          while(timed.back().t<end-1e-9) {
            auto tail=timed.back();tail.t=std::min(end,tail.t+dt);timed.push_back(tail);
          }
          dynamic_branches.insert(dynamic_branches.end(),timed.begin(),timed.end());
        }
      }
    }
    // The older direction/maximum-speed sweep above is retained. Additionally
    // replay the actual pending input sequence: a previous acceleration or
    // opposite steering request cannot vanish when odometry is nearly zero.
    // All branches share the original source-event ledger and risk limits.
    const auto issued=issuedCommandSnapshot();
    if(!issued.empty()) {
      auto model=configuredYieldModel();model.deceleration=deceleration;
      terrain_nav::YieldState root{0.,odom.x,odom.y,odom.yaw,odom.v,initial_steering};
      const double hold=.10+std::max(0.,origin-map->stamp.seconds());
      const double end=std::max(3.,hold+terrain_nav::maximumYieldDelay(model)+
        (std::abs(root.speed)+model.acceleration*(hold+terrain_nav::maximumYieldDelay(model)))/deceleration+.10);
      const auto queued=terrain_nav::rolloutQueuedYield(root,
        {{0.,cmd.linear.x,target_steering},{hold,0.,target_steering}},model,end,origin,issued);
      if(queued.empty()||std::abs(queued.back().speed)>.001)return false;
      for(const auto&s:queued) {
        double cost=0,clearance=kInf;CostQuery query;
        if(!footprintCostLocked(s.x,s.y,s.yaw,cost,&clearance,nullptr,false,map.get(),live.get(),&query)||
           (std::isfinite(clearance)&&clearance<hardClearanceFloor())||
           !addChanceQuery(query,probability,chance_braking_budget_,map.get())) {
          record_rejection("pending_command_pipeline",s.x,s.y,s.yaw,s.t);return false;
        }
      }
      if(live) {
        if(motion_prediction_model_=="reachable") {
          if(timedMovingUnsafe(queued,origin,*live)||
             timedMovingUnsafe(terrain_nav::possibleEarlyStops(queued),origin,*live))return false;
        } else dynamic_branches.insert(dynamic_branches.end(),queued.begin(),queued.end());
      }
    }
    double dynamic_upper=0.;
    if(live && motion_prediction_model_!="reachable" && !live->tracks.empty()) {
      std::stable_sort(dynamic_branches.begin(),dynamic_branches.end(),[](const auto &a,const auto &b){return a.t<b.t;});
      const auto risk=jointMovingRisk(dynamic_branches,origin,*live);
      dynamic_upper=risk.upper;
      if(!risk.valid || dynamic_upper>motion_joint_risk_limit_)return false;
      if(probability_upper)*probability_upper=std::min(1.,probability.upper()+dynamic_upper);
      if(probability.upper()+dynamic_upper>chance_braking_budget_)return false;
      for(const auto &s:dynamic_branches) {
        const auto q=queryCostLocked(s.x,s.y,s.yaw,false,map.get(),true,true,live.get());
        if(!q.known || !std::isfinite(q.probability_upper) || q.probability_upper+dynamic_upper>chance_state_limit_)return false;
      }
    }
    if (probability_upper) *probability_upper=std::min(1.,probability.upper()+dynamic_upper);
    return true;
  }

  void recordCertifiedRecoveryForwardProgress(const OdomState &odom,bool accepted)
  {
    // Called only by the control callback, BEFORE diagnostic rate limiting.
    // Count new source odometry samples, never commands, repeated timer ticks
    // or a reverse-only displacement. Worker reads an immutable summary.
    const auto origin=std::atomic_load(&reverse_progress_origin_);
    if(origin!=recovery_forward_origin_) {
      recovery_forward_origin_=origin;recovery_forward_distance_=0.;recovery_forward_sample_=false;
    }
    if(!origin) {
      std::atomic_store(&recovery_forward_progress_,std::shared_ptr<const RecoveryForwardProgress>{});return;
    }
    bool qualified=accepted && odom.valid && last_cmd_v_>.03 && odom.v>0;
    const auto map=std::atomic_load(&control_map_snapshot_);
    if(qualified) {
      const auto q=map?queryCostLocked(odom.x,odom.y,odom.yaw,false,map.get()):CostQuery{};
      qualified=map && now().seconds()-map->stamp.seconds()<=max_map_age_sec_ &&
        now().seconds()-odom.stamp.seconds()<=max_odom_age_sec_ && q.known && !q.lethal &&
        std::isfinite(q.probability_upper) && q.probability_upper<=chance_state_limit_;
    }
    if(!qualified) {
      recovery_forward_sample_=false;
      if(last_cmd_v_<-.03)recovery_forward_distance_=0.;
    } else {
      const double dt=recovery_forward_sample_?(odom.stamp-recovery_forward_last_.stamp).seconds():0.;
      // A 50 Hz controller may report the same 10 Hz source several times.
      // Its prediction is NOT another measured pose. Never move the source
      // baseline on a duplicate or older stamp.
      if(recovery_forward_sample_ && dt<=0)return;
      if(recovery_forward_sample_ && dt>0) {
        const double dx=odom.x-recovery_forward_last_.x,dy=odom.y-recovery_forward_last_.y;
        const double ds=std::hypot(dx,dy),heading=wrapAngle(odom.yaw-recovery_forward_last_.yaw);
        const bool plausible=dt<=.25 && ds<=max_speed_*dt+.03 &&
          std::abs(heading)<=maxReferenceCurvature()*ds+.08;
        if(plausible && dx*std::cos(odom.yaw)+dy*std::sin(odom.yaw)>0)
          recovery_forward_distance_+=ds;
      }
      recovery_forward_last_=odom;recovery_forward_sample_=true;
    }
    auto progress=std::make_shared<RecoveryForwardProgress>(RecoveryForwardProgress{origin,recovery_forward_distance_});
    std::atomic_store(&recovery_forward_progress_,std::shared_ptr<const RecoveryForwardProgress>(progress));
  }

  void publishChanceStatus(const OdomState &odom, double braking_upper, bool accepted,
                           const char *reason = nullptr,const ControlMapSnapshot *certificate_map=nullptr,
                           const LiveObstacleSnapshot *certificate_live=nullptr)
  {
    OdomState source_odom;
    {std::lock_guard<std::mutex> lock(odom_mutex_);source_odom=odom_;}
    recordCertifiedRecoveryForwardProgress(source_odom,accepted);
    const double time=now().seconds();
    const std::string message=reason ? reason : accepted ? "tracking" : "blocked/unsupported";
    if (accepted==last_chance_accepted_ && time-last_chance_status_sec_>=0 &&
        message==last_chance_reason_ && time-last_chance_status_sec_<0.2) return;
    last_chance_accepted_=accepted;
    last_chance_reason_=message;
    last_chance_status_sec_=time;
    const auto latest_map=certificate_map?std::shared_ptr<const ControlMapSnapshot>{}:std::atomic_load(&control_map_snapshot_);
    const auto *map=certificate_map?certificate_map:latest_map.get();
    const auto state=map ? queryCostLocked(odom.x,odom.y,odom.yaw,false,map,true,true,certificate_live) : CostQuery{};
    diagnostic_msgs::msg::DiagnosticArray report;
    report.header.stamp=now();
    diagnostic_msgs::msg::DiagnosticStatus status;
    status.name="terrain_nav/chance_constraints";
    status.hardware_id=chance_constrained_ ? "gaussian_height_geometry" : "legacy_heuristic";
    status.level=accepted ? diagnostic_msgs::msg::DiagnosticStatus::OK : diagnostic_msgs::msg::DiagnosticStatus::WARN;
    status.message=reason ? reason : accepted ? "tracking" : "blocked/unsupported";
    if (chance_constrained_ && status.message=="no admissible path") {
      if (!map || !state.known) status.message="start terrain support unknown";
      else if (state.lethal) status.message="start terrain constraint rejected";
    }
    const auto field=[&](const char *key,double value) {
      std::ostringstream encoded; encoded<<std::setprecision(17)<<value;
      diagnostic_msgs::msg::KeyValue item; item.key=key; item.value=encoded.str();
      status.values.push_back(item);
    };
    diagnostic_msgs::msg::KeyValue controller;
    controller.key="tracking_controller"; controller.value=tracking_controller_;
    status.values.push_back(controller);
    diagnostic_msgs::msg::KeyValue motion_model,motion_seed;
    motion_model.key="dynamic_prediction_model";motion_model.value=motion_prediction_model_;
    motion_seed.key="dynamic_sampling_base_seed";motion_seed.value=std::to_string(motion_joint_seed_);
    status.values.push_back(motion_model);status.values.push_back(motion_seed);
    diagnostic_msgs::msg::KeyValue dynamic_state,dynamic_reason;
    dynamic_state.key="dynamic_yield_state";dynamic_state.value=dynamic_yield_state_;
    dynamic_reason.key="dynamic_yield_reason";dynamic_reason.value=dynamic_yield_reason_;
    status.values.push_back(dynamic_state);status.values.push_back(dynamic_reason);
    field("dynamic_track_count",dynamic_track_count_);
    field("dynamic_observed_only",dynamic_observed_only_?1.:0.);
    field("dynamic_unobserved_conflicts_skipped",dynamic_unobserved_conflicts_skipped_);
    field("dynamic_center_candidate_actions",center_candidate_actions_?1.:0.);
    field("dynamic_identity_memory_sec",motion_guard_.identity_memory_seconds);
    field("dynamic_turning_identity_reacquisition",motion_guard_.turning_identity_reacquisition?1.:0.);
    field("dynamic_raw_surface_fit",live_motion_raw_surface_fit_?1.:0.);
    field("motion_component_link_cells",motion_guard_.component_link_cells);
    field("dynamic_unresolved_conflicts",dynamic_unresolved_.size());
    {
      diagnostic_msgs::msg::KeyValue ids;ids.key="dynamic_unresolved_ids";
      std::ostringstream encoded;encoded<<std::setprecision(17);
      for(const auto &entry:dynamic_unresolved_)encoded<<entry.first<<'@'<<entry.second<<';';
      ids.value=encoded.str();status.values.push_back(std::move(ids));
    }
    field("dynamic_geometric_retirements",dynamic_geometric_retirements_);
    field("dynamic_time_indexed_routing",dynamic_time_indexed_routing_?1.:0.);
    field("provisional_motion_brake",provisional_brake_active_?1.:0.);
    field("provisional_motion_warning",provisional_warning_active_?1.:0.);
    field("provisional_confirmation_horizon_s",.20);
    field("provisional_confirmation_input_deadline_s",std::min(.35,live_obstacle_max_age_sec_));
    field("provisional_motion_brake_scans",provisional_brake_count_);
    field("provisional_velocity_reserve_mps",provisional_velocity_reserve_mps_);
    field("dynamic_horizon_s",dynamic_horizon_sec_);
    field("dynamic_anticipation_horizon_s",dynamic_anticipation_horizon_sec_);
    field("dynamic_forward_candidates_checked",dynamic_forward_attempts_);
    field("dynamic_input_reselections",dynamic_input_reselections_);
    field("dynamic_selection_input_refreshes",dynamic_selection_input_refreshes_);
    field("motion_tracking_max_height_m",motion_tracking_max_height_m_);
    field("motion_upper_witness_cells",motion_upper_witness_cells_.load());
    field("dynamic_proposals_generated",dynamic_proposals_generated_);
    field("dynamic_terminal_rejections",dynamic_terminal_rejections_);
    field("dynamic_terminal_candidates_checked",dynamic_terminal_candidates_checked_);
    field("dynamic_screen_survivors",dynamic_screen_survivors_);
    field("dynamic_screen_ms",dynamic_screen_ms_);
    field("dynamic_screen_coarse_discarded",dynamic_screen_coarse_discarded_);
    field("dynamic_screen_refined_candidates",dynamic_screen_refined_candidates_);
    field("dynamic_reoptimizations",dynamic_reoptimizations_);
    field("dynamic_screen_min_estimate",dynamic_screen_min_estimate_);
    field("dynamic_screen_min_lower_bound",dynamic_screen_min_lower_);
    field("dynamic_screen_batch_discard_error",dynamic_screen_discard_error_);
    field("dynamic_proof_deadline_stops",dynamic_proof_deadline_stops_);
    field("dynamic_risk_preference_weight",dynamic_risk_preference_weight_);
    field("dynamic_long_generated",dynamic_long_generated_);
    field("dynamic_long_terminal_rejections",dynamic_long_terminal_rejections_);
    field("dynamic_long_screen_survivors",dynamic_long_survivors_);
    field("dynamic_retreat_candidates_checked",dynamic_retreat_attempts_);
    {
      const auto latest_live=certificate_live?std::shared_ptr<const LiveObstacleSnapshot>{}:
        std::atomic_load(&live_snapshot_);
      const auto *motion=certificate_live?certificate_live:latest_live.get();
      size_t unowned_index=0;
      std::vector<std::pair<const terrain_nav::MotionTrackSnapshot*,std::string>> records;
      if(motion) {
        for(const auto &track:motion->tracks) {
          auto key="motion_object_"+std::to_string(track.id);
          if(!track.id&&unowned_index++)key="motion_candidate_"+std::to_string(unowned_index-1);
          records.push_back({&track,std::move(key)});
        }
        for(size_t i=0;i<motion->provisional_tracks.size();++i)
          records.push_back({&motion->provisional_tracks[i],"motion_provisional_"+std::to_string(i)});
      }
      field("dynamic_unconfirmed_motion_count",motion?motion->provisional_tracks.size():0);
      field("dynamic_reverse_proposal_speed_mps",reverse_max_speed_);
      for(const auto &record:records) {
        const auto &track=*record.first;
        diagnostic_msgs::msg::KeyValue item;
        item.key=record.second;
        std::ostringstream description;description<<std::setprecision(17)
          <<"model="<<(track.circular?"measured_circle":"registered_cells")<<" identity="<<track.id
          <<" source_stamp_s="<<track.stamp
          <<" vx="<<track.vx<<" vy="<<track.vy<<" source_age_s="<<time-track.stamp
          <<" center_x="<<track.center_x<<" center_y="<<track.center_y
          <<" measured_extent_x_m="<<track.measured_extent_x<<" measured_extent_y_m="<<track.measured_extent_y
          <<" radius_bound_m="<<track.radius_bound<<" fit_rms_m="<<track.fit_rms
          <<" velocity_reserve_mps="<<track.velocity_error
          <<" velocity_averaging_lag_s="<<track.velocity_averaging_lag_s
          <<" longitudinal_acceleration_bound_mps2="<<track.acceleration_bound
          <<" prediction_profile_speed_mps="<<track.prediction_maximum_speed
          <<" cells="<<track.cells.size();
        description<<" confirmed_identity="<<track.motion_confirmed<<" causal_center_fit="<<track.center_motion_fit
          <<" retained_shape_radius="<<track.retained_radius<<" raw_surface_fit="<<track.raw_surface_fit
          <<" raw_surface_input="<<track.raw_surface_input<<" raw_surface_return_count="<<track.raw_surface_return_count;
        description<<" gaussian_valid="<<track.gaussian.valid<<" gaussian_observations="<<track.gaussian.observation_count
          <<" gaussian_vx="<<track.gaussian.mean[2]<<" gaussian_vy="<<track.gaussian.mean[3]
          <<" gaussian_mean_acceleration_bound="<<track.gaussian.longitudinal_acceleration_bound;
        description<<" gaussian_sigma_m="<<track.gaussian.observation_std;
        description<<" gaussian_source_stamp_s="<<track.gaussian.stamp;
        if(!track.center_observations.empty()) {
          description<<" measured_center_history=";
          for(size_t i=0;i<track.center_observations.size();++i) {
            const auto &c=track.center_observations[i];
            if(i)description<<';';
            description<<c.stamp_s<<','<<c.center[0]<<','<<c.center[1];
          }
        }
        description<<" gaussian_constant_acceleration="<<track.gaussian.constant_acceleration;
        if(track.gaussian.constant_acceleration) {
          description<<" gaussian_acceleration_mean=";
          for(int i=0;i<6;++i)description<<(i?",":"")<<track.gaussian.acceleration_mean[i];
          description<<" gaussian_acceleration_covariance=";
          for(int i=0;i<6;++i)for(int j=0;j<6;++j)description<<(i||j?",":"")<<track.gaussian.acceleration_covariance(i,j);
        }
        // Retain both accepted and rejected interaction inputs. Otherwise the
        // first invalidated continuation can be replayed, but the preceding
        // decision cannot: its posterior was never recorded. Registered-cell
        // fallback actors also need their original cells and source clock.
        // Diagnostics only; no recorded state or truth feeds control.
        if(dynamic_yield_state_=="refused" || dynamic_active_) {
          description<<" gaussian_process_density="<<track.gaussian.process_density<<" gaussian_mean=";
          for(int i=0;i<4;++i)description<<(i?",":"")<<track.gaussian.mean[i];
          description<<" gaussian_covariance=";
          for(int i=0;i<16;++i)description<<(i?",":"")<<track.gaussian.covariance(i/4,i%4);
          description<<" measured_cells=";
          for(size_t i=0;i<track.cells.size();++i)
            description<<(i?";":"")<<track.cells[i][0]<<","<<track.cells[i][1];
        }
        item.value=description.str();status.values.push_back(std::move(item));
      }
    }
    {diagnostic_msgs::msg::KeyValue item;item.key="dynamic_forward_rejection";
      item.value=dynamic_forward_failure_;status.values.push_back(std::move(item));}
    field("dynamic_evaluation_current",dynamic_evaluating_?1.:0.);
    field("dynamic_control_callback_origin_s",dynamic_control_callback_origin_s_);
    field("dynamic_yield_active",dynamic_active_?1.:0.);
    field("dynamic_policy_origin_s",dynamic_origin_);
    {diagnostic_msgs::msg::KeyValue item;item.key="dynamic_policy_commands";
      std::ostringstream out;out<<std::setprecision(17);
      for(size_t i=0;i<dynamic_commands_.size();++i) {
        const auto &c=dynamic_commands_[i];out<<(i?";":"")<<c.t<<","<<c.speed<<","<<c.steering;
      }
      item.value=out.str();status.values.push_back(std::move(item));}
    field("dynamic_evaluation_ms",!dynamic_evaluating_ ? 0. :
      std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-dynamic_evaluation_start_).count());
    field("dynamic_path_probability_upper",dynamic_path_upper_);
    field("dynamic_braking_probability_upper",dynamic_braking_upper_);
    field("dynamic_actuator_delay_s",dynamic_yield_delay_);
    field("dynamic_speed_dead_time_s",terrain_nav::yieldSpeedDelay(configuredYieldModel()));
    field("dynamic_steering_dead_time_s",terrain_nav::yieldSteeringDelay(configuredYieldModel()));
    field("dynamic_escape_policy",dynamic_escape_policy_?1.:0.);
    field("dynamic_approach_limit_mps",dynamic_approach_limit_);
    field("dynamic_proposal_speed_limit_mps",dynamic_proposal_speed_limit_);
    {const auto issued=issuedCommandSnapshot();
      field("issued_command_history_size",issued.size());
      field("issued_command_latest_stamp_s",issued.empty()?NAN:issued.back().t);
      diagnostic_msgs::msg::KeyValue item;item.key="issued_command_pipeline";
      std::ostringstream out;out<<std::setprecision(17);
      // Only the recent pending prefix is needed for a source-exact replay.
      const double cutoff=now().seconds()-std::max(1.,terrain_nav::maximumYieldDelay(configuredYieldModel())+.1);
      auto first=std::lower_bound(issued.begin(),issued.end(),cutoff,
        [](const auto&c,double t){return c.t<t;});
      if(first!=issued.begin())--first;
      for(auto it=first;it!=issued.end();++it)out<<(it==first?"":";")<<it->t<<","<<it->speed<<","<<it->steering;
      item.value=out.str();status.values.push_back(std::move(item));}

    field("dynamic_min_braking_decel",dynamic_yield_deceleration_);
    field("unclassified_encounter_speed_limit_mps",control_encounter_speed_limit_);
    field("unclassified_encounter_model_release",unclassified_encounter_model_release_?1.:0.);
    diagnostic_msgs::msg::KeyValue reserve_state;
    reserve_state.key="clearance_recovery_state";reserve_state.value=reserve_state_;
    status.values.push_back(reserve_state);
    field("clearance_recovery_path_upper",reserve_path_upper_);
    field("clearance_recovery_braking_upper",reserve_braking_upper_);
    field("clearance_recovery_joint_upper",reserve_joint_upper_);
    field("clearance_reserve_attempts",reserve_recovery_count_.load());
    field("recovery_route_penalty",recoveryRoutePenalty(odom.x,odom.y));
    field("recovery_maneuver_attempts",reverse_recovery_count_.load());
    field("recovery_maneuver_total",reverse_recovery_total_.load());
    {
      std::lock_guard<std::mutex> lock(path_mutex_);
      diagnostic_msgs::msg::KeyValue mode;mode.key="recovery_mode";
      mode.value=recovery_path_?(recovery_staged_gears_?"staged_gear_plan":recovery_replan_only_?"retreat_then_stop_replan":"paired_retreat_forward_exit"):"inactive";
      status.values.push_back(mode);
      field("queued_gear_legs",double(recovery_legs_.size()));
    }
    field("chance_constraints_enabled",chance_constrained_ ? 1.0 : 0.0);
    field("localization_fault",localization_fault_.load() ? 1.0 : 0.0);
    field("mobility_stuck",stuck_latched_ ? 1.0 : 0.0);
    field("mobility_replan_attempts",mobility_recovery_.attempts());
    field("mobility_stopping",mobility_recovery_.active()?1.:0.);
    field("command_accepted",accepted ? 1.0 : 0.0);
    field("command_speed_mps",last_cmd_v_);
    field("command_steering_rad",last_cmd_steer_/steer_output_sign_);
    const bool source=std::isfinite(odom.prediction_source[3]);
    field("measured_speed_mps",source?odom.prediction_source[3]:odom.v);
    field("control_state_speed_mps",odom.v);
    field("control_state_x_m",odom.x);field("control_state_y_m",odom.y);
    field("control_state_yaw_rad",odom.yaw);
    field("measured_yaw_rate_radps",source?odom.prediction_source[4]:odom.yaw_rate);
    field("control_state_yaw_rate_radps",odom.yaw_rate);
    field("control_prediction_sec",odom.prediction_sec);
    field("control_state_stamp_s",odom.stamp.seconds()+odom.prediction_sec);
    field("control_prediction_uses_issued_commands",odom.command_prediction?1.:0.);
    field("odometry_source_stamp_s",odom.stamp.seconds());
    field("dynamic_braking_policy_contract",2.);
    field("dynamic_prediction_model_joint",motion_prediction_model_!="reachable"?1.:0.);
    field("dynamic_joint_risk_limit",motion_joint_risk_limit_);
    field("dynamic_joint_probability_upper",motion_joint_last_upper_.load());
    field("dynamic_joint_assessment_ms",motion_joint_last_ms_.load());
    field("dynamic_joint_checks",double(motion_joint_checks_.load()));
    field("dynamic_joint_last_check",double(motion_joint_last_check_.load()));
    field("dynamic_joint_last_samples",double(motion_joint_last_samples_.load()));
    field("dynamic_joint_analytical_objects",double(motion_joint_last_analytical_.load()));
    field("dynamic_joint_actual_object_draws",double(motion_joint_last_draws_.load()));
    field("dynamic_joint_last_origin_s",motion_joint_last_origin_s_.load());
    field("dynamic_joint_sampling_alpha",terrain_nav::motionCheckAlpha(motion_joint_last_check_.load()));
    field("dynamic_gaussian_observation_std_m",motion_guard_.gaussian_observation_std);
    field("dynamic_gaussian_constant_acceleration",motion_guard_.gaussian_constant_acceleration?1.:0.);
    field("dynamic_gaussian_jerk_density",motion_guard_.gaussian_jerk_density);
    field("dynamic_gaussian_process_density_m2_s3",motion_guard_.gaussian_process_density);
    field("dynamic_gaussian_maximum_observations",motion_guard_.gaussian_maximum_observations);
    field("dynamic_gaussian_acceleration_stress",motion_guard_.gaussian_acceleration_envelope?1.:0.);
    field("dynamic_track_known_slow_centers",motion_guard_.track_known_slow_centers?1.:0.);
    field("dynamic_sim_observation_calibration",motion_guard_.sim_observation_calibration?1.:0.);
    field("sampling_feasible",sampling_feasible_ ? 1.0 : 0.0);
    field("state_probability_upper",state.probability_upper);
    field("state_support_known",state.known ? 1.0 : 0.0);
    field("braking_probability_upper",braking_upper);
    field("state_limit",chance_state_limit_);
    field("local_path_budget",chance_path_budget_);
    field("braking_budget",chance_braking_budget_);
    field("map_age_s",map ? time-map->stamp.seconds() : kInf);
    field("reported_map_source_stamp_s",map?map->stamp.seconds():std::numeric_limits<double>::quiet_NaN());
    const auto reported_live=certificate_live?std::shared_ptr<const LiveObstacleSnapshot>{}:std::atomic_load(&live_snapshot_);
    field("reported_live_source_stamp_s",certificate_live?certificate_live->stamp_sec:
      reported_live?reported_live->stamp_sec:std::numeric_limits<double>::quiet_NaN());
    field("certificate_source_stamp_s",certificate_map?certificate_map->stamp.seconds():std::numeric_limits<double>::quiet_NaN());
    field("certificate_live_stamp_s",certificate_live?certificate_live->stamp_sec:std::numeric_limits<double>::quiet_NaN());
    field("dynamic_time_indexed_occupancy_certificate",certificate_map && certificate_live?1.:0.);
    field("dynamic_owned_history_exclusions",certificate_live?certificate_live->owned_history_exclusions:0.);
    field("map_ingress_ms",map_ingress_ms_.load());
    field("deferred_search_maps",deferred_search_maps_.load());
    field("early_partial_commits",early_partial_commits_.load());
    const auto live_geometry=std::atomic_load(&live_snapshot_);
    field("live_grid_resolution_m",live_geometry ? live_geometry->resolution : kInf);
    field("live_cloud_reliable",live_cloud_reliable_?1.:0.);
    field("motion_grid_resolution_m",live_geometry && live_geometry->width>0 && live_geometry->height>0 &&
      std::isfinite(live_geometry->resolution) && live_geometry->resolution>0 ? motionAggregationFactor(live_geometry->width,
      live_geometry->height,live_geometry->resolution)*live_geometry->resolution : kInf);
    field("body_clearance_m",control_body_clearance_);
    // Legacy key above is retained for old recorders. This is an EDT/grid
    // quantity, not independently measured physical mesh separation.
    field("body_grid_clearance_m",control_body_clearance_);
    const auto envelope=collisionEnvelope(footprint_margin_);
    field("collision_front_m",envelope.front);
    field("collision_back_m",envelope.back);
    field("collision_protected_width_m",2*envelope.half_width);
    field("collision_margin_m",footprint_margin_);
    field("collision_live_extra_inflation_m",live_obstacle_inflation_radius_);
    field("collision_hard_grid_reserve_m",hardClearanceFloor());
    field("collision_vehicle_height_m",vehicle_envelope_height_m_);
    field("collision_headroom_margin_m",overhead_clearance_margin_m_);
    field("collision_height_slice_mode",map ? heightSliceMode(map->grid) : -1.);
    field("clearance_speed_limit_mps",control_clearance_speed_limit_);
    diagnostic_msgs::msg::KeyValue speed_gate;
    speed_gate.key="clearance_speed_limit_mode";speed_gate.value=(chance_constrained_&&terrain_speed_enable_)?
      "directional_swept_braking":"inactive";
    status.values.push_back(speed_gate);
    report.status.push_back(status);
    chance_status_pub_->publish(report);
  }

  bool mobilityRecoveryControl(const OdomState &odom)
  {
    if(stuck_replan_attempts_==0)return false;
    uint64_t sequence;bool goal;
    {std::lock_guard<std::mutex> lock(goal_mutex_);sequence=goal_seq_;goal=have_goal_;}
    mobility_recovery_.goal(sequence);
    if(!goal || (!mobility_recovery_.active()&&!mobility_recovery_.latched()))return false;
    const auto map=std::atomic_load(&control_map_snapshot_);
    const auto live=std::atomic_load(&live_snapshot_);const double time=now().seconds();
    const auto recent=[&](double stamp,double age){return time-stamp>=-.05&&time-stamp<=age;};
    const bool fresh=odom.valid&&map&&recent(odom.stamp.seconds(),max_odom_age_sec_)&&
      recent(map->stamp.seconds(),max_map_age_sec_)&&
      (!live_obstacle_enable_||(live&&recent(live->stamp_sec,live_obstacle_max_age_sec_)));
    double cost=0;
    const bool supported=fresh&&chance_constrained_&&footprintCostLocked(
      odom.x,odom.y,odom.yaw,cost,nullptr,nullptr,false,map.get());
    const auto action=mobility_recovery_.tick(time,odom.v,fresh,supported);
    publishImmediateStop();
    if(action==terrain_nav::MobilityRecovery::Action::Replan) {
      stuck_since_sec_=0.;
      if(!revokeStalledMotion(sequence))return true;
      rememberBlockedRoute(odom,sequence);
      requestPlan("traction stall: stopped; fresh certified route required");
      RCLCPP_WARN(get_logger(),"terrain mobility recovery: confirmed supported stop; requesting a NEW certified route, attempt=%d/%d; no automatic reverse command",
        mobility_recovery_.attempts(),stuck_replan_attempts_);
      publishChanceStatus(odom,kInf,false,"traction recovery: stopped; fresh replan requested");
    } else {
      stuck_latched_=action==terrain_nav::MobilityRecovery::Action::Latched;
      publishChanceStatus(odom,kInf,false,stuck_latched_?"no progress: recovery exhausted or stop unconfirmed":
        "traction recovery: braking and confirming supported stop");
    }
    return true;
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
    mobility_recovery_.goal(goal_seq);
    if (stuck_latched_) {
      publishImmediateStop();
      publishChanceStatus(odom,std::numeric_limits<double>::quiet_NaN(),false,"no progress: motion inhibited");
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
    if(chance_constrained_&&mobility_recovery_.stall(t,stuck_replan_attempts_)) {
      // Revoke the stalled maneuver and every in-flight commit. Replan only
      // after a fresh, supported physical stop; keep the external goal and
      // existing reverse-maneuver budgets. Temporal map updates cannot reset
      // this independent per-goal allowance.
      if(!revokeStalledMotion(goal_seq))return true;
      rememberBlockedRoute(odom,goal_seq);
      resetNominalControls();publishImmediateStop();
      RCLCPP_WARN(get_logger(),"terrain mobility recovery: commanded motion without %.2fm progress in %.1fs; stalled geometry revoked, attempt=%d/%d",
        stuck_progress_m_,stuck_timeout_sec_,mobility_recovery_.attempts(),stuck_replan_attempts_);
      publishChanceStatus(odom,kInf,false,"traction recovery: braking and confirming supported stop");
      return true;
    }
    stuck_latched_ = true;
    RCLCPP_ERROR(
      get_logger(),
      "terrain_nav stopped: commanded %.2f m/s but moved only %.2f m in %.1f s; send a new goal after checking the obstruction",
      last_cmd_v_, progress, stuck_timeout_sec_);
    publishImmediateStop();
    publishChanceStatus(odom,std::numeric_limits<double>::quiet_NaN(),false,"no progress: motion inhibited");
    return true;
  }

  bool revokeStalledMotion(uint64_t sequence)
  {
    std::scoped_lock<std::mutex,std::mutex> lock(goal_mutex_,path_mutex_);
    if(!have_goal_||goal_seq_!=sequence)return false;
    input_epoch_.fetch_add(1,std::memory_order_acq_rel);
    path_.clear();have_path_=false;recovery_path_=false;recovery_staged_gears_=false;
    recovery_legs_.clear();recovery_exit_.clear();recovery_resume_pending_=false;
    recovery_replan_only_=false;recovery_exit_active_=false;++path_seq_;
    return true;
  }

  void controlTimer()
  {
    const auto control_start = std::chrono::steady_clock::now();
    if(localization_fault_.load(std::memory_order_acquire)) {
      OdomState odom;{std::lock_guard<std::mutex> lock(odom_mutex_);odom=odom_;}
      publishImmediateStop();
      publishChanceStatus(odom,std::numeric_limits<double>::quiet_NaN(),false,"localization inconsistent: motion inhibited");
      return;
    }
    if (live_obstacle_enable_) {
      const auto live = std::atomic_load(&live_snapshot_);
      if (chance_constrained_ && !live) {
        RCLCPP_WARN_THROTTLE(get_logger(),*get_clock(),2000,
          "Probability navigation waits for a current live obstacle cloud");
        publishImmediateStop();
        return;
      }
      if (live) {
        const double live_age = now().seconds() - live->stamp_sec;
        if (live_age > live_obstacle_max_age_sec_) {
          RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 1000,
            "terrain_nav stops on stale live cloud: age=%.2fs",
            live_age);
          waitForFreshInputs();
          return;
        }
      }
    }
    OdomState odom;
    {
      std::lock_guard<std::mutex> lock(odom_mutex_);
      odom = odom_;
    }
    predictOdomForControl(odom,false);
    if(mobilityRecoveryControl(odom))return;
    if(provisionalMotionBrake(odom))return;
    if(reserveRecoveryControl(odom))return;
    if(dynamicYieldControl(odom,stale_inputs_waiting_.load(std::memory_order_acquire)))return;
    if (!odom.valid) {
      if (haveGoalThreadSafe()) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "terrain_nav received a goal but is waiting for odometry on %s",
          odom_topic_.c_str());
      }
      publishStop();
      publishChanceStatus(odom,std::numeric_limits<double>::quiet_NaN(),false,"waiting for odometry");
      return;
    }

    if (haveGoalThreadSafe()) {
      const double now_sec = this->now().seconds();
      const double odom_age = now_sec - odom.stamp.seconds();
      const auto map = std::atomic_load(&control_map_snapshot_);
      const double map_age = map ? now_sec - map->stamp.seconds() : 0.0;
      if ((odom_age > max_odom_age_sec_ && odom_age >= 0.0) ||
          (map_ready_ && map_age > max_map_age_sec_ && map_age >= 0.0)) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "terrain_nav stops on stale input: odom_age=%.2fs map_age=%.2fs", odom_age, map_age);
        waitForFreshInputs();
        return;
      }
    }

    if (stale_inputs_waiting_.load(std::memory_order_acquire)) {
      const auto fresh_now = std::chrono::steady_clock::now();
      if (fresh_inputs_since_ == std::chrono::steady_clock::time_point{}) {
        fresh_inputs_since_ = fresh_now;
      }
      // Require a stable fresh interval, not a single callback just below the
      // age limit. Otherwise intermittent delays repeatedly launch searches.
      if (std::chrono::duration<double>(fresh_now - fresh_inputs_since_).count() < 0.20) {
        publishImmediateStop();
        return;
      }
      stale_inputs_waiting_.store(false, std::memory_order_release);
      if (haveGoalThreadSafe()) {
        RCLCPP_INFO(get_logger(), "terrain_nav inputs fresh again: requesting a replacement path");
        requestPlan("fresh inputs restored");
      }
      publishImmediateStop();
      return;
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
      publishImmediateStop();
      return;
    }

    live_dirty_.exchange(false); // Current live geometry is checked below every control tick.
    bool map_path_invalidated=false;
    if (needReplan(odom)) {
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 800,
        "terrain_nav replanning: %s",
        last_replan_reason_.c_str());
      const std::string replan_reason = last_replan_reason_;
      requestPlan(replan_reason);
      if (replan_reason == "updated terrain map invalidated path ahead") {
        map_path_invalidated=true;
      }
    }

    std::vector<Pose2> path;
    bool have_path = false;
    bool recovery_path = false,observation_path=false,limited_prefix=false;
    uint64_t control_path_sequence=0;
    {
      std::lock_guard<std::mutex> lock(path_mutex_);
      have_path = have_path_;
      path = path_;
      recovery_path = recovery_path_;
      observation_path=path_seq_==observation_path_seq_;
      control_path_sequence=path_seq_;limited_prefix=path_seq_==safety_prefix_seq_;
    }
    if (!have_path || path.size() < 2) {
      publishStop();
      publishChanceStatus(odom,std::numeric_limits<double>::quiet_NaN(),false,
                          haveGoalThreadSafe() ? "no admissible path" : "idle");
      return;
    }

    double track_dist = 0.0;
    size_t nearest = nearestPathIndex(path, odom.x, odom.y, &track_dist);
    bool terminal_path=false;
    {
      std::lock_guard<std::mutex> lock(goal_mutex_);
      terminal_path=have_goal_ &&
        std::hypot(path.back().x-raw_goal_.x,path.back().y-raw_goal_.y)<=goal_tolerance_;
    }
    // A useful rolling prefix needs room to stop and extend. A certified
    // terminal connector must instead be executable up to the arrival disk.
    if (path.back().s-path[nearest].s<=((terminal_path || recovery_path)?.02:.15)) {
      publishImmediateStop();
      publishChanceStatus(odom,std::numeric_limits<double>::quiet_NaN(),false,"waiting for path extension");
      return;
    }
    if ((map_path_invalidated && pathAheadUnsafe(path,nearest)) ||
        (live_obstacle_enable_ && livePathAheadUnsafe(path, nearest))) {
      if(repairInvalidPathSuffix(odom,path,control_path_sequence)) {
        nearest=nearestPathIndex(path,odom.x,odom.y,&track_dist);
        limited_prefix=true;terminal_path=false;
      } else {
        {
          std::lock_guard<std::mutex> lock(path_mutex_);
          if(path_seq_==control_path_sequence){path_.clear();have_path_=false;}
        }
        // Failed proof/near hazard: stop immediately. A concurrent newer path
        // is rechecked next tick, never erased by this older failed repair.
        if (!planner_running_.load(std::memory_order_acquire) &&
            now().seconds() - last_plan_time_sec_.load(std::memory_order_acquire) >= replan_min_interval_)
          requestPlan("no certified prefix after path invalidation");
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 800,
          "terrain_nav emergency stop: no current certified prefix after path invalidation");
        publishImmediateStop();
        return;
      }
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
    auto tracking_odom=odom;
    predictOdomForControl(tracking_odom);
    predictOdomForControl(odom,false);
    const bool sampling = use_mppi_ && path.front().direction > 0;
    auto requested = sampling ? computeSamplingMpcCommand(odom,path) : computePurePursuitCommand(tracking_odom,path);
    if(observation_path)requested.linear.x=std::clamp(requested.linear.x,-observation_speed_limit_,observation_speed_limit_);
    if (chance_allow_startup_prior_) {
      const auto map=std::atomic_load(&control_map_snapshot_);
      if (map && map->grid.exists("startup_speed_limit")) {
        const Eigen::Vector3d position(odom.x,odom.y,odom.yaw);
        const double limit=map->grid.isInMap(position) ? map->grid.getValue("startup_speed_limit",position,
            se2_grid::InterpolationMethods::INTER_NEAREST) : 0.;
        if (!std::isfinite(limit) || limit<=0) requested.linear.x=0;
        else requested.linear.x=std::clamp(requested.linear.x,-limit,limit);
      }
    }
    if (sampling && !sampling_feasible_) {
      publishImmediateStop();
      publishChanceStatus(odom,std::numeric_limits<double>::quiet_NaN(),false);
      if (!planner_running_.load() && now().seconds()-last_plan_time_sec_.load()>=replan_min_interval_)
        requestPlan("sampling controller has no feasible rollout");
      return;
    }
    geometry_msgs::msg::Twist cmd = limitedCommand(requested);
    const auto encounter_live=std::atomic_load(&live_snapshot_);
    control_encounter_speed_limit_=kInf;
    if(encounter_live && now().seconds()-encounter_live->stamp_sec>=-.05 &&
       now().seconds()-encounter_live->stamp_sec<=live_obstacle_max_age_sec_) {
      cmd.linear.x=encounterSpeed(cmd.linear.x,odom.x,odom.y,odom.yaw,*encounter_live);
      control_encounter_speed_limit_=cmd.linear.x;
    }
    if(limited_prefix) {
      const auto map=std::atomic_load(&control_map_snapshot_);
      const double available=std::max(0.,path.back().s-path[nearest].s);
      const double latency=.35+.10+control_period_+(map?std::max(0.,now().seconds()-map->stamp.seconds()):max_map_age_sec_);
      const double limit=terrain_nav::clearanceSpeedLimit(available,recovery_path?0.:.15,latency,
        std::min(max_accel_,dynamic_yield_deceleration_));
      cmd.linear.x=std::clamp(cmd.linear.x,-limit,limit);
    }
    if(observation_path)cmd.linear.x=std::clamp(cmd.linear.x,-observation_speed_limit_,observation_speed_limit_);
    control_clearance_speed_limit_=kInf;
    control_body_clearance_=kInf;
    if(chance_constrained_ && terrain_speed_enable_) {
      const auto map=std::atomic_load(&control_map_snapshot_);
      double cost=0.;
      if(!map || !footprintCostLocked(odom.x,odom.y,odom.yaw,cost,&control_body_clearance_,nullptr,false,map.get())) {
        control_clearance_speed_limit_=0.;
      } else {
        // The nearest obstacle can be lateral or behind us. Treating this
        // scalar as forward free distance needlessly stalls parallel-wall
        // travel. The following full-body, source-age-aware stopping rollout
        // is the directional admission gate, including steering transitions,
        // real momentum during a gear change and one shared chance budget.
        control_clearance_speed_limit_=std::abs(cmd.linear.x);
      }
      cmd.linear.x=std::clamp(cmd.linear.x,-control_clearance_speed_limit_,control_clearance_speed_limit_);
    }
    // Safety deceleration can bypass smoothing, but never the stopping-distance
    // check against measured speed and the currently applied steering.
    double braking_probability_upper=std::numeric_limits<double>::quiet_NaN();
    bool braking_speed_limited=false;
    if (!commandBrakingEnvelopeSafe(odom,cmd,&braking_probability_upper)) {
      braking_speed_limited=true;
      bool safe_speed=false;
      for (int attempt=0;attempt<8;++attempt) {
        cmd.linear.x*=.7;
        if (commandBrakingEnvelopeSafe(odom,cmd,&braking_probability_upper)) {safe_speed=true;break;}
      }
      if (!safe_speed) {
        {
          std::lock_guard<std::mutex> lock(path_mutex_);
          path_.clear();have_path_=false;
        }
        // Stop every control tick, but do not relaunch the same rejected
        // search at 50 Hz. New obstacle events still bypass this cooldown.
        if (!planner_running_.load() && now().seconds()-last_plan_time_sec_.load()>=replan_min_interval_)
          requestPlan("command braking envelope intersects risk");
        publishImmediateStop();
        publishChanceStatus(odom,braking_probability_upper,false);
        return;
      }
    }
    if(chance_constrained_&&terrain_speed_enable_)
      control_clearance_speed_limit_=std::abs(cmd.linear.x);
    const double remaining = std::max(0.0, path.back().s - path[nearest].s);
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 1500,
      "terrain_nav tracking: err=%.2f rem=%.2f cmd_v=%.2f cmd_steer=%.2f odom_v=%.2f path_pts=%zu",
      track_dist, remaining, cmd.linear.x, cmd.angular.z, odom.v, path.size());
    publishCommand(cmd);
    publishChanceStatus(odom,braking_probability_upper,true,braking_speed_limited ? "risk-limited speed" : nullptr);
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
    const auto rear=terrain_nav::rearAxlePose(odom,rear_axle_offset_);
    const auto target_rear=terrain_nav::rearAxlePose(target,rear_axle_offset_);
    const double dx = target_rear.x - rear.x;
    const double dy = target_rear.y - rear.y;
    const double ld = std::max(0.25, std::hypot(dx, dy));
    const double alpha = wrapAngle(std::atan2(dy, dx) - odom.yaw);
    const double pure_steer = std::atan2(2.0 * wheel_base_ * std::sin(alpha), ld);

    const Pose2 nearest_pose = path[nearest];
    const auto nearest_rear=terrain_nav::rearAxlePose(nearest_pose,rear_axle_offset_);
    const double ex = rear.x - nearest_rear.x;
    const double ey = rear.y - nearest_rear.y;
    const double signed_lateral = -std::sin(nearest_pose.yaw) * ex + std::cos(nearest_pose.yaw) * ey;
    const double heading_error = wrapAngle(nearest_pose.yaw - odom.yaw);
    const int direction = path.front().direction;
    // Required local bicycle steering and preview feedback are different
    // objects. Averaging curvature over 45cm and blending it with the preview
    // steering under-steered even a perfectly tracked varying-curvature path.
    const size_t edge=std::min(nearest,path.size()-2);
    const Pose2 curvature_begin=path[edge],curvature_end=path[edge+1];
    const auto rear_begin=terrain_nav::rearAxlePose(curvature_begin,rear_axle_offset_);
    const auto rear_end=terrain_nav::rearAxlePose(curvature_end,rear_axle_offset_);
    const double rear_chord=std::hypot(rear_end.x-rear_begin.x,rear_end.y-rear_begin.y);
    const double reference_curvature=rear_chord>1e-4 ?
      2.*std::sin(.5*wrapAngle(curvature_end.yaw-curvature_begin.yaw))/(direction*rear_chord) : 0.;
    const double feedforward=std::atan(wheel_base_*reference_curvature);
    const double stanley_feedback = direction * heading_error +
      std::atan2(-stanley_gain_ * signed_lateral, std::abs(odom.v) + stanley_softening_);
    const double nominal_dx=target_rear.x-nearest_rear.x,nominal_dy=target_rear.y-nearest_rear.y;
    const double nominal_alpha=wrapAngle(std::atan2(nominal_dy,nominal_dx)-nearest_pose.yaw);
    const double nominal_preview=std::atan2(2.*wheel_base_*std::sin(nominal_alpha),
      std::max(.25,std::hypot(nominal_dx,nominal_dy)));
    // Subtract the preview induced by the reference itself. At zero error,
    // both feedback terms vanish for ANY feasible piecewise-curvature curve,
    // in either gear, rather than only for a single constant-radius circle.
    const double steer = feedforward + pure_pursuit_weight_*(pure_steer-nominal_preview) +
      (1.-pure_pursuit_weight_)*stanley_feedback;

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
    if (remaining <= 0.02) return 0.0;
    if (remaining >= goal_slowdown_distance_) return 1.0;
    return std::clamp(remaining / goal_slowdown_distance_, 0.28, 1.0);
  }

  double terrainSpeedScaleLocked(
    double x, double y, double yaw, const ControlMapSnapshot *snapshot = nullptr) const
  {
    if (!terrain_speed_enable_ || !map_ready_) return 1.0;
    if (chance_constrained_) {
      const auto q=queryCostLocked(x,y,yaw,false,snapshot);
      if (q.lethal || !std::isfinite(q.probability_upper)) return 0.0;
      // Advisory slowdown only; the stopping-horizon budget is the hard gate.
      return std::clamp(1.0-terrain_risk_slowdown_weight_*q.probability_upper/chance_state_limit_,
                        terrain_speed_min_scale_,1.0);
    }

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

  geometry_msgs::msg::Twist computeSamplingMpcCommand(const OdomState &odom, const std::vector<Pose2> &path)
  {
    sampling_feasible_=false;
    const auto map=std::atomic_load(&control_map_snapshot_);
    if(!map) return geometry_msgs::msg::Twist{};
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
    std::vector<double> v_seq(static_cast<size_t>(mppi_steps_));
    std::vector<double> steer_seq(static_cast<size_t>(mppi_steps_));

    for (int rollout = 0; rollout < mppi_batch_size_; ++rollout) {
      double previous_v=odom.v;
      double previous_steer=std::abs(odom.v)>.05 ?
        clampAbs(std::atan(odom.yaw_rate*wheel_base_/odom.v),max_steer_) : last_cmd_steer_/steer_output_sign_;
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
        // Evaluate only sequences obeying the same actuator rate bounds.
        auto &speed=v_seq[static_cast<size_t>(k)], &steer=steer_seq[static_cast<size_t>(k)];
        speed=std::clamp(speed,previous_v-max_accel_*mppi_dt_,previous_v+max_accel_*mppi_dt_);
        steer=std::clamp(steer,previous_steer-max_steer_rate_*mppi_dt_,previous_steer+max_steer_rate_*mppi_dt_);
        previous_v=speed; previous_steer=steer;
      }

      const double cost = evaluateRollout(odom, path, current_s, v_seq, steer_seq,map.get());
      if (cost < best_cost) {
        best_cost = cost;
        best_v.swap(v_seq);
        best_steer.swap(steer_seq);
      }
    }

    if(!std::isfinite(best_cost)) return geometry_msgs::msg::Twist{};
    sampling_feasible_=true;

    nominal_v_ = best_v;
    nominal_steer_ = best_steer;

    geometry_msgs::msg::Twist cmd;
    cmd.linear.x = nominal_v_.front();
    cmd.angular.z = nominal_steer_.front();
    const double terrain_scale = terrainSpeedScale(odom.x, odom.y, odom.yaw);
    cmd.linear.x = std::min(cmd.linear.x, desired_speed_ * terrain_scale);
    // Do not raise a validated sampled speed to a tracking-speed floor.

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
    const std::vector<double> &steer_seq,
    const ControlMapSnapshot *provided_map=nullptr)
  {
    const auto retained_map = provided_map ? std::shared_ptr<const ControlMapSnapshot>{} :
      std::atomic_load(&control_map_snapshot_);
    const auto *snapshot=provided_map ? provided_map : retained_map.get();
    if (!snapshot) return kInf;
    double x = odom.x;
    double y = odom.y;
    double yaw = odom.yaw;
    double cost = 0.0;
    double prev_v = odom.v;
    double prev_steer = std::abs(odom.v)>.05 ?
      clampAbs(std::atan(odom.yaw_rate*wheel_base_/odom.v),max_steer_) : last_cmd_steer_/steer_output_sign_;
    double traveled = 0.0;
    terrain_nav::ChanceBudget probability;
    if (!addChancePose(x,y,yaw,probability,chance_path_budget_,snapshot)) return kInf;

    for (int k = 0; k < mppi_steps_; ++k) {
      const double v = v_seq[static_cast<size_t>(k)];
      const double steer = steer_seq[static_cast<size_t>(k)];
      const double average_v=.5*(prev_v+v), average_steer=.5*(prev_steer+steer);
      const double yaw_rate=average_v*std::tan(average_steer)/wheel_base_;
      const int steps=std::max(1,static_cast<int>(std::ceil(std::max(
        std::abs(average_v)*mppi_dt_/(.25*snapshot->resolution),
        std::abs(yaw_rate)*mppi_dt_/(.25*snapshot->grid.getResolutionYaw())))));
      const double dt=mppi_dt_/steps;
      for(int i=0;i<steps;++i) {
        const auto next=terrain_nav::integrateBodyVelocity(Pose2{x,y,yaw},average_v,yaw_rate,dt,rear_axle_offset_);
        x=next.x;y=next.y;yaw=next.yaw;
        double sub_cost=0,sub_clearance=kInf;
        if(!footprintCostLocked(x,y,yaw,sub_cost,&sub_clearance,nullptr,false,snapshot) ||
           (std::isfinite(sub_clearance) && sub_clearance<hardClearanceFloor()) ||
           !addChancePose(x,y,yaw,probability,chance_path_budget_,snapshot)) return kInf;
      }
      traveled += std::max(0.0, v) * mppi_dt_;
      const double remaining = std::max(0.0, path.back().s - current_s - traveled);
      const double target_speed = desired_speed_ * speedScaleForRemaining(remaining) *
        terrainSpeedScaleLocked(x, y, yaw, snapshot);

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
      if (!footprintCostLocked(x, y, yaw, fc, &clearance, nullptr, false, snapshot)) return kInf;
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

  geometry_msgs::msg::Twist limitedCommand(geometry_msgs::msg::Twist cmd) const
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

    return cmd;
  }

  void publishCommand(const geometry_msgs::msg::Twist &cmd)
  {
    std::lock_guard<std::mutex> lock(command_publish_mutex_);
    const auto output=localization_fault_.load(std::memory_order_acquire)?geometry_msgs::msg::Twist{}:cmd;
    cmd_pub_->publish(output);
    last_cmd_v_ = output.linear.x;
    last_cmd_steer_ = output.angular.z;
    last_cmd_time_sec_ = now().seconds();
    issued_commands_.record({last_cmd_time_sec_,output.linear.x,output.angular.z/steer_output_sign_},
      std::max(2.,dynamic_yield_delay_+1.));
  }

  void limitAndPublish(geometry_msgs::msg::Twist cmd)
  {
    publishCommand(limitedCommand(cmd));
  }

  void publishStop()
  {
    if (chance_constrained_) {
      // A smoothed "stop" can still contain forward velocity; it has not
      // passed the final command envelope and must not bypass that gate.
      publishImmediateStop();
      return;
    }
    geometry_msgs::msg::Twist cmd;
    limitAndPublish(cmd);
  }

  void publishImmediateStop()
  {
    publishCommand(geometry_msgs::msg::Twist{});
  }

  void resetNominalControls()
  {
    nominal_v_.assign(static_cast<size_t>(mppi_steps_), desired_speed_);
    nominal_steer_.assign(static_cast<size_t>(mppi_steps_), last_cmd_steer_);
  }

  double visualizationTerrainZLocked(double x, double y, double fallback,const ControlMapSnapshot *snapshot=nullptr) const
  {
    // Path publication is called by plan() while map_mutex_ is held.
    if (!snapshot && !map_ready_) return fallback;
    const auto &grid=snapshot?snapshot->grid:map_;
    for (const auto & layer : {std::string("probability_elevation"),std::string("bgk_mean"),std::string("elevation")}) {
      if (!grid.exists(layer)) continue;
      try {
        const double z = grid.getValue(
          layer, Eigen::Vector3d(x, y, 0.0), se2_grid::InterpolationMethods::INTER_LINEAR);
        if (std::isfinite(z)) return z;
      } catch (...) {
        // A visualization sample must never interrupt the control loop.
      }
    }
    return fallback;
  }

  void publishPath(const ControlMapSnapshot *snapshot=nullptr)
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
      publishReferenceMarkers(path,snapshot);
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
      pose.pose.position.z = visualizationTerrainZLocked(p.x, p.y, 0.0,snapshot) + 0.12;
      tf2::Quaternion q;
      q.setRPY(0.0, 0.0, p.yaw);
      pose.pose.orientation = tf2::toMsg(q);
      msg.poses.push_back(pose);
    }
    path_pub_->publish(msg);
    publishReferenceMarkers(path,snapshot);
  }

  void publishReferenceMarkers(const std::vector<Pose2> &path,const ControlMapSnapshot *snapshot=nullptr)
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
        pt.z = visualizationTerrainZLocked(p.x, p.y, 0.0,snapshot) + 0.16;
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

  void publishActiveGoal(const Pose2 &goal, bool map_locked = false, bool mission = false)
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
    sphere.ns = mission?"mission_goal":"rolling_subgoal";
    sphere.id = 0;
    sphere.type = visualization_msgs::msg::Marker::SPHERE;
    sphere.action = visualization_msgs::msg::Marker::ADD;
    sphere.pose.position.x = goal.x;
    sphere.pose.position.y = goal.y;
    sphere.pose.position.z = terrain_z + 0.16;
    sphere.pose.orientation.w = 1.0;
    sphere.scale.x = sphere.scale.y = mission?.34:.18;
    sphere.scale.z = mission?.18:.10;
    sphere.color.r = mission?1.0f:.12f;
    sphere.color.g = mission?.72f:.78f;
    sphere.color.b = mission?.18f:1.0f;
    sphere.color.a = 0.78f;
    arr.markers.push_back(sphere);

    visualization_msgs::msg::Marker arrow;
    arrow.header = sphere.header;
    arrow.ns = sphere.ns;
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
    arrow.color=sphere.color;
    arrow.color.a = 0.90f;
    arr.markers.push_back(arrow);

    auto label=sphere;label.id=2;label.type=visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    label.pose.position.z+=.45;label.scale.z=.18;
    label.text=mission?"FINAL GOAL (fixed)":"LOCAL SUBGOAL";arr.markers.push_back(label);
    if(mission) {goal_marker_pub_->publish(arr);legacy_goal_marker_pub_->publish(arr);}
    else local_goal_marker_pub_->publish(arr);
  }

  std::string odom_topic_;
  std::string fused_map_topic_;
  std::string cmd_topic_;
  std::string frame_id_;
  bool live_obstacle_enable_ = false;
  bool live_cloud_reliable_=false;
  double live_obstacle_resolution_=0.;
  std::string live_cloud_topic_ = "/cloud_registered";
  double live_obstacle_min_height_ = 0.38;
  double live_obstacle_neighbor_radius_ = 0.75;
  int live_obstacle_min_hits_ = 3;
  double live_obstacle_max_age_sec_ = 0.60;
  double live_obstacle_hold_sec_ = 1.20;
  double live_obstacle_inflation_radius_ = 0.45;
  bool live_motion_guard_enable_=true;
  double motion_tracking_max_height_m_=0.;
  std::atomic<size_t> motion_upper_witness_cells_{0};
  double provisional_velocity_reserve_mps_=1.;
  double live_motion_min_resolution_=0.;
  bool live_motion_raw_surface_fit_=false;
  bool live_obstacle_ground_clear_enabled_=false;
  double live_motion_horizon_sec_=1.5;
  terrain_nav::GridMotionGuard motion_guard_;
  terrain_nav::MotionResolutionMailbox motion_resolution_mailbox_; // Producer owns it under live_mutex_.

  double wheel_base_ = 0.549;
  double rear_axle_offset_ = 0.;
  double topology_unknown_reverse_weight_=0.;
  bool topology_forward_seed_enable_=false;
  bool topology_async_guide_enable_=false,background_guide_runtime_=false;
  double last_background_guide_request_=-1.;
  std::future<BackgroundGuideResult> background_guide_;
  std::optional<BackgroundGuideResult> background_maneuver_;
  enum class GuideHandoff {None,Computing,ReverseReady};
  GuideHandoff guide_handoff_=GuideHandoff::None; // planner-thread-owned
  bool fresh_background_guide_=false,topology_proposal_complete_=false;
  double global_wait_start_=-1.;
  uint64_t global_wait_epoch_=0;
  Pose2 global_wait_origin_,global_wait_goal_;
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
  double body_collision_front_=0.0,body_collision_back_=0.0,body_collision_half_width_=0.0;
  double vehicle_envelope_height_m_=.65,overhead_clearance_margin_m_=.10;

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
  bool topology_guide_enable_=false;
  terrain_nav::RouteMemory route_memory_{.1};
  terrain_nav::RouteMemory::Result forward_seed_;
  Pose2 forward_seed_goal_;
  uint64_t forward_seed_epoch_=0;
  std::shared_ptr<const std::vector<RecoveryRouteMemory>> forward_seed_preference_;
  double last_topology_diagnostic_stamp_=-1.;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr topology_memory_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr topology_route_pub_;
  uint64_t route_memory_epoch_=0;
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
  bool chance_constrained_ = false;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr chance_status_pub_;
  double last_chance_status_sec_ = -1.0;
  bool last_chance_accepted_ = false;
  std::string last_chance_reason_;
  double chance_state_limit_ = 0.05;
  bool chance_allow_startup_prior_ = false;
  double chance_path_budget_ = 0.10;
  double chance_braking_budget_ = 0.05;
  double control_body_clearance_=kInf;
  double control_clearance_speed_limit_=kInf;
  struct RecoveryProgressOrigin {double x,y;uint64_t sequence,epoch;};
  std::shared_ptr<const RecoveryProgressOrigin> reverse_progress_origin_;
  std::atomic<uint64_t> recovery_progress_valid_epoch_{std::numeric_limits<uint64_t>::max()};
  struct RecoveryForwardProgress {std::shared_ptr<const RecoveryProgressOrigin> origin;double distance;};
  std::shared_ptr<const RecoveryForwardProgress> recovery_forward_progress_;
  // Control-thread-owned sample state; planner sees only the summary above.
  std::shared_ptr<const RecoveryProgressOrigin> recovery_forward_origin_;
  OdomState recovery_forward_last_;
  bool recovery_forward_sample_=false;
  double recovery_forward_distance_=0.;
  struct PendingRecoverySearch {uint64_t sequence,epoch;};
  std::shared_ptr<const PendingRecoverySearch> pending_recovery_search_;
  std::shared_ptr<const std::vector<RecoveryRouteMemory>> recovery_route_memory_;
  bool chance_allow_joint_model_=false;
  bool chance_allow_independent_model_floor_=false;
  bool chance_allow_bounded_pending_=false;
  bool chance_allow_gp_completion_ = false;
  double chance_max_pitch_rad_ = 0.55;
  double chance_max_roll_rad_ = 0.45;
  double chance_max_step_m_ = 0.18;
  double vehicle_ground_clearance_m_=.135,underbody_clearance_margin_m_=.01;
  std::string se2_risk_layer_ = "risk";
  std::string hard_risk_layer_ = "baseline_risk";
  bool se2_risk_lethal_ = false;
  double se2_hard_threshold_ = 0.999;
  double unknown_cost_ = 60.0;
  bool allow_unknown_ = true;
  double unknown_robot_bubble_radius_ = 0.0;
  double unknown_robot_bubble_max_baseline_ = 0.0;
  double known_ground_memory_sec_=3.0;
  double max_odom_age_sec_ = 0.5;
  double max_map_age_sec_ = 1.0;
  double stuck_timeout_sec_ = 4.0;
  double stuck_progress_m_ = 0.12;
  double stuck_command_mps_ = 0.25;
  int stuck_replan_attempts_=0;
  terrain_nav::MobilityRecovery mobility_recovery_;
  double risk_weight_ = 3.0;
  double clearance_distance_ = 0.85;
  double clearance_weight_ = 2.8;
  double hard_clearance_floor_ = -1.0;
  double historical_obstacle_cost_weight_=12.;
  double historical_obstacle_influence_distance_=1.35;
  std::vector<float> historical_clearance_;
  bool historical_advisory_present_=false;
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
  double reference_tracking_reserve_m_=0.,reference_tracking_reserve_cost_=20.;
  double heuristic_weight_ = 1.15;
  double progress_weight_ = 1.8;
  double path_lateral_weight_ = 0.45;
  double goal_heading_weight_ = 0.35;
  double backtrack_reject_distance_ = 0.28;
  double max_extra_path_ratio_ = 2.25;
  bool accept_partial_progress_ = true;
  double partial_search_time_ms_=0.;
  std::atomic<uint64_t> early_partial_commits_{0};

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

  std::string tracking_controller_ = "hybrid";
  bool use_mppi_ = false;
  bool sampling_feasible_ = false;
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
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr received_goal_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr live_obstacle_pub_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr live_raw_pub_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr motion_observations_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr ref_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr guidance_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr goal_marker_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr local_goal_marker_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr legacy_goal_marker_pub_;
  rclcpp::TimerBase::SharedPtr control_timer_;
  rclcpp::CallbackGroup::SharedPtr map_group_;
  rclcpp::CallbackGroup::SharedPtr odom_group_;
  rclcpp::CallbackGroup::SharedPtr live_group_;
  rclcpp::CallbackGroup::SharedPtr control_group_;

  mutable std::mutex map_mutex_;
  se2_grid::SE2Grid map_;
  std::shared_ptr<const ControlMapSnapshot> control_map_snapshot_;
  std::shared_ptr<const ControlMapSnapshot> pending_search_map_;
  std::atomic<double> map_ingress_ms_{0.};
  std::atomic<uint64_t> deferred_search_maps_{0};
  std::shared_ptr<const LiveObstacleSnapshot> live_snapshot_;
  struct MotionSource {uint64_t epoch;std::shared_ptr<const LiveObstacleSnapshot> live;};
  std::mutex motion_sources_mutex_;
  std::deque<MotionSource> motion_sources_;
  bool dynamic_active_=false;
  std::unordered_map<uint64_t,double> dynamic_unresolved_;
  uint64_t dynamic_conflict_goal_=0,dynamic_conflict_spatial_=0;
  double dynamic_conflict_time_=0;
  bool provisional_brake_active_=false;
  bool provisional_warning_active_=false;
  double provisional_warning_stamp_=-1;
  uint64_t provisional_brake_count_=0;
  double provisional_brake_stamp_=-1;
  bool reserve_active_=false;
  std::atomic_bool reserve_retreat_active_{false};
  uint64_t reserve_goal_seq_=0,reserve_epoch_=0;
  double reserve_origin_=0,reserve_last_time_=0,reserve_last_attempt_=-1e9;
  double reserve_release_since_=-1,reserve_release_stamp_=0;
  double reserve_path_upper_=std::numeric_limits<double>::quiet_NaN();
  double reserve_braking_upper_=std::numeric_limits<double>::quiet_NaN();
  double reserve_joint_upper_=std::numeric_limits<double>::quiet_NaN();
  struct ReserveExitRequest {std::vector<Pose2> path;uint64_t goal_seq=0,epoch=0;};
  std::vector<Pose2> reserve_forward_exit_;
  std::shared_ptr<const ReserveExitRequest> reserve_exit_request_;
  std::string reserve_state_="idle";
  terrain_nav::YieldState reserve_initial_;
  std::vector<terrain_nav::YieldCommand> reserve_commands_;
  uint64_t dynamic_goal_seq_=0,dynamic_epoch_=0;
  double dynamic_origin_=0,dynamic_last_time_=0,dynamic_horizon_sec_=0;
  double dynamic_last_optimization_time_=0.,dynamic_last_optimization_source_=0.;
  uint64_t dynamic_reoptimizations_=0;
  int dynamic_forward_attempts_=0,dynamic_retreat_attempts_=0;
  bool dynamic_escape_policy_=false;
  double dynamic_approach_limit_=0.,dynamic_proposal_speed_limit_=0.;
  unsigned dynamic_input_reselections_=0;
  unsigned dynamic_selection_input_refreshes_=0;
  int dynamic_proposals_generated_=0,dynamic_terminal_rejections_=0,dynamic_screen_survivors_=0;
  int dynamic_terminal_candidates_checked_=0;
  int dynamic_proof_deadline_stops_=0;
  double dynamic_screen_ms_=0.;
  size_t dynamic_screen_coarse_discarded_=0,dynamic_screen_refined_candidates_=0;
  double dynamic_screen_min_estimate_=NAN,dynamic_screen_min_lower_=NAN,dynamic_screen_discard_error_=0.;
  int dynamic_long_generated_=0,dynamic_long_terminal_rejections_=0,dynamic_long_survivors_=0;
  uint64_t dynamic_geometric_retirements_=0;
  std::string dynamic_forward_failure_="not evaluated";
  double dynamic_yield_delay_=.35,dynamic_yield_deceleration_=1.2;
  bool center_candidate_actions_=false;
  std::string motion_prediction_model_="reachable";
  int motion_joint_samples_=8192;
  double motion_joint_risk_limit_=.02;
  const uint64_t motion_joint_seed_=(uint64_t(std::random_device{}())<<32)^uint64_t(std::random_device{}());
  mutable std::atomic<uint64_t> motion_joint_checks_{0};
  mutable std::atomic<uint64_t> motion_joint_last_check_{0};
  mutable std::atomic<size_t> motion_joint_last_samples_{0};
  mutable std::atomic<size_t> motion_joint_last_analytical_{0},motion_joint_last_draws_{0};
  mutable std::atomic<double> motion_joint_last_upper_{NAN},motion_joint_last_ms_{0.};
  mutable std::atomic<double> motion_joint_last_origin_s_{NAN};
  bool dynamic_time_indexed_routing_=false;
  bool dynamic_observed_only_=false;
  double dynamic_speed_dead_time_=-1.,dynamic_steering_dead_time_=-1.;
  uint64_t dynamic_unobserved_conflicts_skipped_=0;
  bool dynamic_acceleration_escape_enable_=false;
  double unclassified_encounter_speed_=0.,control_encounter_speed_limit_=kInf;
  bool unclassified_encounter_model_release_=false;
  double dynamic_risk_preference_weight_=0.;
  double dynamic_anticipation_horizon_sec_=3.;
  double dynamic_path_upper_=std::numeric_limits<double>::quiet_NaN();
  double dynamic_braking_upper_=std::numeric_limits<double>::quiet_NaN();
  double observation_speed_limit_=.35;
  uint64_t observation_path_seq_=std::numeric_limits<uint64_t>::max();
  uint64_t safety_prefix_seq_=std::numeric_limits<uint64_t>::max(); // path_mutex_
  size_t dynamic_track_count_=0;
  std::string dynamic_yield_state_="idle",dynamic_yield_reason_="no prediction";
  terrain_nav::YieldState dynamic_initial_;
  std::vector<terrain_nav::YieldCommand> dynamic_commands_;
  struct RetreatSchedule {
    uint64_t goal=0,epoch=0;
    double origin=0.;
    std::vector<terrain_nav::YieldCommand> commands;
  };
  std::optional<RetreatSchedule> last_retreat_;
  std::chrono::steady_clock::time_point dynamic_evaluation_start_{};
  bool dynamic_evaluating_=false;
  double dynamic_control_callback_origin_s_=NAN;
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
  std::vector<double> observed_ground_until_;
  std::unordered_map<uint64_t,double> ground_evidence_;
  double last_ground_evidence_sec_=-1,ground_evidence_resolution_=0;
  std::vector<uint8_t> live_hard_;
  std::vector<double> live_expiry_;
  struct LiveRectangleEvidence {double x,y,resolution;terrain_nav::ColumnRectangle rectangle;uint64_t owner=0;};
  struct LiveRectangleFrame {double expires;std::vector<LiveRectangleEvidence> cells;};
  std::deque<LiveRectangleFrame> live_rectangle_frames_;
  std::vector<float> live_occupied_minimum_;
  std::vector<uint8_t> live_free_scans_;
  std::vector<uint8_t> live_routing_free_scans_;
  double live_routing_stamp_=-1.;
  std::atomic_bool live_dirty_{false};
  double live_min_x_ = 0.0;
  double live_min_y_ = 0.0;
  double live_resolution_ = 0.2;
  int live_width_ = 0;
  int live_height_ = 0;

  mutable std::mutex odom_mutex_;
  OdomState odom_;
  std::deque<Pose2> forward_trail_;
  std::string trail_frame_;
  bool have_prev_pose_for_speed_ = false;
  double prev_pose_speed_x_ = 0.0;
  double prev_pose_speed_y_ = 0.0;
  double prev_pose_speed_z_ = 0.0;
  double prev_pose_speed_yaw_ = 0.0;
  double prev_pose_speed_stamp_sec_ = 0.0;
  double estimated_odom_v_ = 0.0;
  double estimated_odom_yaw_rate_ = 0.0;
  std::atomic_bool logged_odom_ready_{false};
  std::atomic<double> unknown_robot_x_{0.0};
  std::atomic<double> unknown_robot_y_{0.0};
  std::atomic_bool unknown_robot_pose_ready_{false};
  std::atomic<double> current_robot_x_{0.0},current_robot_y_{0.0};
  std::atomic_int reverse_recovery_count_{0};
  std::atomic_int reverse_recovery_total_{0};
  std::atomic_uint64_t topology_recovery_search_round_{0};
  std::atomic_int reserve_recovery_count_{0};

  mutable std::mutex goal_mutex_;
  Pose2 raw_goal_;
  bool have_goal_ = false;
  uint64_t goal_seq_ = 0;

  mutable std::mutex path_mutex_;
  std::vector<Pose2> path_;
  Pose2 active_goal_;
  bool have_path_ = false;
  bool recovery_path_ = false;
  bool recovery_exit_active_ = false;
  bool recovery_replan_only_ = false;
  bool recovery_staged_gears_ = false;
  uint64_t recovery_legs_spatial_epoch_=0;
  std::deque<std::vector<Pose2>> recovery_legs_;
  bool recovery_resume_pending_ = false;
  uint64_t recovery_geometry_epoch_=0;
  double recovery_resume_saved_time_=0.;
  std::vector<Pose2> recovery_exit_;
  bool reverse_enable_ = true;
  bool reverse_recovery_enable_=false;
  double reverse_max_distance_ = 4.0;
  double reverse_max_speed_ = 0.55;
  uint64_t path_seq_ = 0;

  std::atomic_bool plan_requested_{false};
  std::atomic_bool planner_running_{false};
  std::atomic<uint64_t> spatial_epoch_{0};
  std::atomic_bool stale_inputs_waiting_{false};
  std::atomic_bool localization_fault_{false};
  mutable std::mutex command_publish_mutex_;
  terrain_nav::IssuedYieldHistory issued_commands_;
  std::atomic<uint64_t> input_epoch_{0};
  std::chrono::steady_clock::time_point fresh_inputs_since_{};
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

#ifndef TERRAIN_NAV_BENCHMARK
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
#endif
