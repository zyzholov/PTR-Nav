#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <se2_grid_msgs/msg/se2_grid.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/float32.hpp>
#include <cmath>
#include <vector>
#include <deque>
#include <algorithm>
#include <limits>
#include <cstdint>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <se2_grid_core/SE2Grid.hpp>
#include <se2_grid_ros/SE2GridRosConverter.hpp>

class TerrainAnalyzerNode : public rclcpp::Node
{
public:
  TerrainAnalyzerNode() : Node("terrain_analyzer")
  {
    RCLCPP_INFO(this->get_logger(), "Terrain Analyzer ROS2 Node Starting...");

    // Declare parameters
    declare_parameter("grid_size_x", 220);
    declare_parameter("grid_size_y", 220);
    declare_parameter("resolution", 0.2);
    declare_parameter("yaw_resolution", 0.1);
    declare_parameter("sensor_range", 50.0);
    declare_parameter("min_var", 0.001);
    declare_parameter("max_var", 1.0);
    declare_parameter("cache_size", 100);
    declare_parameter("publish_rate_hz", 10.0);
    declare_parameter("visual_publish_rate_hz", 10.0);
    declare_parameter("terrain_pcd_path", "RockyDesertMesh.pcd");
    declare_parameter("viz_grid_size_x", 100);
    declare_parameter("viz_grid_size_y", 100);
    declare_parameter("use_pcd_prior", true);
    declare_parameter("cloud_is_local_map", false);
    declare_parameter("local_map_sliding_window_fusion", false);
    declare_parameter("publish_on_cloud", false);
    declare_parameter("publish_internal_visual_clouds", true);
    declare_parameter("elevation_visual_fill_missing", true);
    declare_parameter("elevation_visual_fill_radius_m", 0.45);
    declare_parameter("sdf_visual_max_dist", 6.0);
    declare_parameter("sdf_visual_height_scale", 0.0);
    declare_parameter("sdf_visualize_obstacle_columns", true);
    declare_parameter("sdf_obstacle_column_step", 0.20);
    declare_parameter("sdf_obstacle_column_max_height", 4.0);
    declare_parameter("publish_overhead_cloud", true);
    declare_parameter("overhead_min_clearance_m", 0.20);
    declare_parameter("overhead_cloud_max_points", 80000);
    declare_parameter("sdf_use_terrain_hard_obstacles", false);
    declare_parameter("visualize_obstacle_tops", true);
    declare_parameter("use_pcd_vertical_obstacles", false);
    declare_parameter("min_points_for_traversability", 2);
    declare_parameter("sdf_obstacle_slope", 0.7);
    declare_parameter("sdf_hard_obstacle_slope", 1.1);
    declare_parameter("sdf_obstacle_curvature", 0.55);
    declare_parameter("sdf_hard_obstacle_curvature", 1.05);
    declare_parameter("step_height_soft", 0.10);
    declare_parameter("step_height_hard", 0.22);
    declare_parameter("traversable_slope_free", 0.35);
    declare_parameter("traversable_curvature_free", 0.20);
    declare_parameter("traversable_step_free", 0.05);
    declare_parameter("traversability_risk_gamma", 1.35);
    declare_parameter("risk_weight_slope", 0.55);
    declare_parameter("risk_weight_curvature", 0.30);
    declare_parameter("risk_weight_step_height", 0.15);
    declare_parameter("risk_hard_threshold", 0.92);
    declare_parameter("risk_vis_hard_threshold", 0.78);
    declare_parameter("risk_visual_deadband", 0.18);
    declare_parameter("risk_visual_gamma", 1.8);
    declare_parameter("terrain_risk_scale", 0.30);
    declare_parameter("terrain_risk_to_hard_obstacle", true);
    declare_parameter("terrain_hardness_scale", 1.0);
    declare_parameter("tree_risk_radius_m", 0.85);
    declare_parameter("tree_hard_keepout_radius_m", 0.70);
    declare_parameter("tree_risk_power", 3.0);
    declare_parameter("non_tree_risk_cap", 0.85);
    declare_parameter("obstacle_min_height", 0.18);
    declare_parameter("obstacle_max_height", 1.40);
    declare_parameter("obstacle_seed_max_height", 0.85);
    declare_parameter("obstacle_min_points", 4);
    declare_parameter("obstacle_hit_increment", 3);
    declare_parameter("obstacle_count_decay", 0.85);
    declare_parameter("obstacle_persistence_misses", 12);
    declare_parameter("obstacle_ground_search_radius_m", 0.35);
    declare_parameter("obstacle_footprint_filter_enable", true);
    declare_parameter("obstacle_footprint_filter_radius_m", 0.45);
    declare_parameter("obstacle_footprint_max_neighbors", 10);
    declare_parameter("obstacle_footprint_line_width_cells", 2);
    declare_parameter("vertical_obstacle_inflation_m", 0.75);
    declare_parameter("ego_clear_radius", 0.55);
    declare_parameter("ego_body_filter_enable", true);
    declare_parameter("ego_body_filter_front_m", 1.05);
    declare_parameter("ego_body_filter_back_m", 0.80);
    declare_parameter("ego_body_filter_half_width_m", 0.62);
    declare_parameter("ego_body_filter_z_min_m", -0.50);
    declare_parameter("ego_body_filter_z_max_m", 1.35);
    declare_parameter("map_padding_m", 2.0);

    // Get parameters
    grid_size_x_ = get_parameter("grid_size_x").as_int();
    grid_size_y_ = get_parameter("grid_size_y").as_int();
    resolution_ = get_parameter("resolution").as_double();
    yaw_resolution_ = get_parameter("yaw_resolution").as_double();
    sensor_range_ = get_parameter("sensor_range").as_double();
    min_var_ = get_parameter("min_var").as_double();
    max_var_ = get_parameter("max_var").as_double();
    cache_size_ = get_parameter("cache_size").as_int();
    publish_rate_hz_ = get_parameter("publish_rate_hz").as_double();
    visual_publish_rate_hz_ = get_parameter("visual_publish_rate_hz").as_double();
    viz_grid_size_x_ = get_parameter("viz_grid_size_x").as_int();
    viz_grid_size_y_ = get_parameter("viz_grid_size_y").as_int();
    use_pcd_prior_ = get_parameter("use_pcd_prior").as_bool();
    cloud_is_local_map_ = get_parameter("cloud_is_local_map").as_bool();
    local_map_sliding_window_fusion_ =
      get_parameter("local_map_sliding_window_fusion").as_bool();
    publish_on_cloud_ = get_parameter("publish_on_cloud").as_bool();
    publish_internal_visual_clouds_ = get_parameter("publish_internal_visual_clouds").as_bool();
    elevation_visual_fill_missing_ = get_parameter("elevation_visual_fill_missing").as_bool();
    elevation_visual_fill_radius_m_ = get_parameter("elevation_visual_fill_radius_m").as_double();
    sdf_visual_max_dist_ = get_parameter("sdf_visual_max_dist").as_double();
    sdf_visual_height_scale_ = get_parameter("sdf_visual_height_scale").as_double();
    sdf_visualize_obstacle_columns_ = get_parameter("sdf_visualize_obstacle_columns").as_bool();
    sdf_obstacle_column_step_ = get_parameter("sdf_obstacle_column_step").as_double();
    sdf_obstacle_column_max_height_ = get_parameter("sdf_obstacle_column_max_height").as_double();
    publish_overhead_cloud_ = get_parameter("publish_overhead_cloud").as_bool();
    overhead_min_clearance_m_ = get_parameter("overhead_min_clearance_m").as_double();
    overhead_cloud_max_points_ = get_parameter("overhead_cloud_max_points").as_int();
    sdf_use_terrain_hard_obstacles_ =
      get_parameter("sdf_use_terrain_hard_obstacles").as_bool();
    visualize_obstacle_tops_ = get_parameter("visualize_obstacle_tops").as_bool();
    use_pcd_vertical_obstacles_ = get_parameter("use_pcd_vertical_obstacles").as_bool();
    min_points_for_traversability_ = get_parameter("min_points_for_traversability").as_int();
    sdf_obstacle_slope_ = get_parameter("sdf_obstacle_slope").as_double();
    sdf_hard_obstacle_slope_ = get_parameter("sdf_hard_obstacle_slope").as_double();
    sdf_obstacle_curvature_ = get_parameter("sdf_obstacle_curvature").as_double();
    sdf_hard_obstacle_curvature_ = get_parameter("sdf_hard_obstacle_curvature").as_double();
    step_height_soft_ = get_parameter("step_height_soft").as_double();
    step_height_hard_ = get_parameter("step_height_hard").as_double();
    traversable_slope_free_ = get_parameter("traversable_slope_free").as_double();
    traversable_curvature_free_ = get_parameter("traversable_curvature_free").as_double();
    traversable_step_free_ = get_parameter("traversable_step_free").as_double();
    traversability_risk_gamma_ = get_parameter("traversability_risk_gamma").as_double();
    risk_weight_slope_ = get_parameter("risk_weight_slope").as_double();
    risk_weight_curvature_ = get_parameter("risk_weight_curvature").as_double();
    risk_weight_step_height_ = get_parameter("risk_weight_step_height").as_double();
    risk_hard_threshold_ = get_parameter("risk_hard_threshold").as_double();
    risk_vis_hard_threshold_ = get_parameter("risk_vis_hard_threshold").as_double();
    risk_visual_deadband_ = get_parameter("risk_visual_deadband").as_double();
    risk_visual_gamma_ = get_parameter("risk_visual_gamma").as_double();
    terrain_risk_scale_ = get_parameter("terrain_risk_scale").as_double();
    terrain_risk_to_hard_obstacle_ = get_parameter("terrain_risk_to_hard_obstacle").as_bool();
    terrain_hardness_scale_ = get_parameter("terrain_hardness_scale").as_double();
    tree_risk_radius_m_ = get_parameter("tree_risk_radius_m").as_double();
    tree_hard_keepout_radius_m_ = get_parameter("tree_hard_keepout_radius_m").as_double();
    tree_risk_power_ = get_parameter("tree_risk_power").as_double();
    non_tree_risk_cap_ = get_parameter("non_tree_risk_cap").as_double();
    obstacle_min_height_ = get_parameter("obstacle_min_height").as_double();
    obstacle_max_height_ = get_parameter("obstacle_max_height").as_double();
    obstacle_seed_max_height_ = get_parameter("obstacle_seed_max_height").as_double();
    obstacle_min_points_ = get_parameter("obstacle_min_points").as_int();
    obstacle_hit_increment_ = get_parameter("obstacle_hit_increment").as_int();
    obstacle_count_decay_ = get_parameter("obstacle_count_decay").as_double();
    obstacle_persistence_misses_ = get_parameter("obstacle_persistence_misses").as_int();
    obstacle_ground_search_radius_m_ = get_parameter("obstacle_ground_search_radius_m").as_double();
    obstacle_footprint_filter_enable_ = get_parameter("obstacle_footprint_filter_enable").as_bool();
    obstacle_footprint_filter_radius_m_ = get_parameter("obstacle_footprint_filter_radius_m").as_double();
    obstacle_footprint_max_neighbors_ = get_parameter("obstacle_footprint_max_neighbors").as_int();
    obstacle_footprint_line_width_cells_ = get_parameter("obstacle_footprint_line_width_cells").as_int();
    vertical_obstacle_inflation_m_ = get_parameter("vertical_obstacle_inflation_m").as_double();
    ego_clear_radius_ = get_parameter("ego_clear_radius").as_double();
    ego_body_filter_enable_ = get_parameter("ego_body_filter_enable").as_bool();
    ego_body_filter_front_m_ = get_parameter("ego_body_filter_front_m").as_double();
    ego_body_filter_back_m_ = get_parameter("ego_body_filter_back_m").as_double();
    ego_body_filter_half_width_m_ = get_parameter("ego_body_filter_half_width_m").as_double();
    ego_body_filter_z_min_m_ = get_parameter("ego_body_filter_z_min_m").as_double();
    ego_body_filter_z_max_m_ = get_parameter("ego_body_filter_z_max_m").as_double();
    map_padding_m_ = get_parameter("map_padding_m").as_double();
    std::string terrain_pcd_path = get_parameter("terrain_pcd_path").as_string();

    // Initialize global terrain storage with a fallback local-size map;
    // processTerrainCloud() will resize this to full PCD bounds.
    global_origin_x_ = -0.5f * static_cast<float>(grid_size_x_ * resolution_);
    global_origin_y_ = -0.5f * static_cast<float>(grid_size_y_ * resolution_);
    global_size_x_ = grid_size_x_;
    global_size_y_ = grid_size_y_;
    elevation_map_.assign(global_size_x_ * global_size_y_, 0.0f);
    top_elevation_map_.assign(global_size_x_ * global_size_y_, -std::numeric_limits<float>::infinity());
    collision_top_elevation_map_.assign(global_size_x_ * global_size_y_, -std::numeric_limits<float>::infinity());
    vertical_obstacle_map_.assign(global_size_x_ * global_size_y_, 0);
    obstacle_band_count_map_.assign(global_size_x_ * global_size_y_, 0);
    obstacle_miss_count_map_.assign(global_size_x_ * global_size_y_, 0);
    prior_vertical_obstacle_map_.assign(global_size_x_ * global_size_y_, 0);
    prior_obstacle_band_count_map_.assign(global_size_x_ * global_size_y_, 0);
    variance_map_.assign(global_size_x_ * global_size_y_, max_var_);
    point_count_map_.assign(global_size_x_ * global_size_y_, 0);
    // sdf_distance_map_ removed — planner uses terrain risk directly
    risk_map_cache_.assign(global_size_x_ * global_size_y_, 0.0f);

    // Initialize SE2 grid containers used by planner.
    configurePlannerGrids(grid_size_x_, grid_size_y_);

    if (!use_pcd_prior_)
    {
      RCLCPP_INFO(this->get_logger(),
                  "PCD prior disabled: terrain map starts empty and is updated only from live cloud.");
      map_center_x_ = global_origin_x_ + 0.5f * static_cast<float>(global_size_x_) * static_cast<float>(resolution_);
      map_center_y_ = global_origin_y_ + 0.5f * static_cast<float>(global_size_y_) * static_cast<float>(resolution_);
    }
    else
    {
      // Load terrain from PCD file
      pcl::PointCloud<pcl::PointXYZ>::Ptr terrain_cloud(new pcl::PointCloud<pcl::PointXYZ>);

      // Try multiple paths for the PCD file
      std::vector<std::string> pcd_paths = {
        terrain_pcd_path,
        "./install/share/fake_ugv/maps/" + terrain_pcd_path
      };

      try
      {
        const std::string fake_ugv_share = ament_index_cpp::get_package_share_directory("fake_ugv");
        pcd_paths.push_back(fake_ugv_share + "/maps/" + terrain_pcd_path);
      }
      catch (const std::exception &)
      {
        // Keep fallback search paths only if package index is unavailable.
      }

      bool terrain_loaded = false;
      for (const auto& path : pcd_paths)
      {
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(path, *terrain_cloud) >= 0)
        {
          RCLCPP_INFO(this->get_logger(), "Loaded terrain from: %s (%zu points)",
                      path.c_str(), terrain_cloud->size());
          terrain_loaded = true;
          break;
        }
      }

      if (!terrain_loaded)
      {
        RCLCPP_WARN(this->get_logger(),
          "Failed to load terrain PCD while use_pcd_prior=true. Using synthetic elevation data.");
        generateSyntheticTerrain();
      }
      else
      {
        // Process terrain PCD to elevation map
        processTerrainCloud(terrain_cloud);
      }
    }

    // Create subscriptions
    cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      "cloud", rclcpp::SensorDataQoS(),
      std::bind(&TerrainAnalyzerNode::cloudCallback, this, std::placeholders::_1));

    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      "odom", rclcpp::QoS(rclcpp::KeepLast(10)),
      std::bind(&TerrainAnalyzerNode::odomCallback, this, std::placeholders::_1));

    // Create publishers
    fused_map_pub_ = create_publisher<se2_grid_msgs::msg::SE2Grid>(
      "fused_map", rclcpp::QoS(rclcpp::KeepLast(1)));

    if (publish_internal_visual_clouds_)
    {
      auto elevation_qos = rclcpp::QoS(rclcpp::KeepLast(2));
      elevation_qos.best_effort().durability_volatile();
      elevation_cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
        "elevation_cloud", elevation_qos);
      // sdf_cloud_pub_ removed — planner uses terrain risk directly
      risk_cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
        "risk_cloud", elevation_qos);
      obstacle_cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
        "obstacle_cloud", elevation_qos);
      overhead_cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
        "overhead_cloud", elevation_qos);
    }

    normal_map_pub_ = create_publisher<visualization_msgs::msg::Marker>(
      "normal_map", rclcpp::QoS(rclcpp::KeepLast(5)));

    // sdf_map_pub_ removed — planner uses terrain risk directly

    RCLCPP_INFO(this->get_logger(),
      "Terrain Analyzer ROS2: Ready\n"
      "  Grid: %d×%d cells\n"
      "  Resolution: %.3f m/cell\n"
      "  Yaw resolution: %.3f rad\n"
      "  Sensor range: %.1f m\n"
      "  Live cloud mode: %s\n"
      "  Local-map sliding fusion: %s\n"
      "  Publish trigger: %s\n"
      "  Planner map rate: %.1f Hz\n"
      "  Visual cloud rate: %.1f Hz\n"
      "  Internal visual clouds: %s",
      grid_size_x_, grid_size_y_, resolution_, yaw_resolution_, sensor_range_,
      cloud_is_local_map_ ? "FAST_LIO local-map input" : "accumulate cloud into terrain grid",
      local_map_sliding_window_fusion_ ? "enabled" : "disabled",
      publish_on_cloud_ ? "on each cloud" : "periodic timer",
      publish_rate_hz_, visual_publish_rate_hz_,
      publish_internal_visual_clouds_ ? "enabled" : "disabled");

    if (cloud_is_local_map_ && local_map_sliding_window_fusion_)
    {
      RCLCPP_WARN(
        this->get_logger(),
        "local_map_sliding_window_fusion=true while cloud_is_local_map=true. "
        "If the input is FAST_LIO /cloud_local_map, this re-accumulates an already rolling map "
        "and can over-mark obstacles/risk.");
    }

    if (!publish_on_cloud_)
    {
      const double period_ms = 1000.0 / std::max(1.0, publish_rate_hz_);
      timer_ = create_wall_timer(
        std::chrono::milliseconds(static_cast<int>(std::round(period_ms))),
        std::bind(&TerrainAnalyzerNode::publishTerrain, this));
    }
  }

private:
  void processTerrainCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
  {
    if (!cloud || cloud->empty()) return;

    // Find bounds of the terrain
    float min_x = cloud->at(0).x, max_x = cloud->at(0).x;
    float min_y = cloud->at(0).y, max_y = cloud->at(0).y;
    float min_z = cloud->at(0).z, max_z = cloud->at(0).z;

    for (const auto& pt : cloud->points)
    {
      min_x = std::min(min_x, pt.x);
      max_x = std::max(max_x, pt.x);
      min_y = std::min(min_y, pt.y);
      max_y = std::max(max_y, pt.y);
      min_z = std::min(min_z, pt.z);
      max_z = std::max(max_z, pt.z);
    }

    RCLCPP_INFO(this->get_logger(),
      "Terrain bounds - X: [%.2f, %.2f], Y: [%.2f, %.2f], Z: [%.2f, %.2f]",
      min_x, max_x, min_y, max_y, min_z, max_z);

    // Allocate global grid from full terrain bounds so local windows can be
    // queried around the vehicle without running out-of-map.
    const float map_padding = std::max(static_cast<float>(resolution_), static_cast<float>(map_padding_m_));
    global_origin_x_ = min_x - map_padding;
    global_origin_y_ = min_y - map_padding;
    global_size_x_ = std::max(2, static_cast<int>(std::ceil((max_x - global_origin_x_) / static_cast<float>(resolution_))) + 1);
    global_size_y_ = std::max(2, static_cast<int>(std::ceil((max_y - global_origin_y_) / static_cast<float>(resolution_))) + 1);

    const int extra_cells = std::max(1, static_cast<int>(std::ceil(map_padding / static_cast<float>(resolution_))));
    global_size_x_ += extra_cells;
    global_size_y_ += extra_cells;

    elevation_map_.assign(global_size_x_ * global_size_y_, 0.0f);
    top_elevation_map_.assign(global_size_x_ * global_size_y_, -std::numeric_limits<float>::infinity());
    collision_top_elevation_map_.assign(global_size_x_ * global_size_y_, -std::numeric_limits<float>::infinity());
    vertical_obstacle_map_.assign(global_size_x_ * global_size_y_, 0);
    obstacle_band_count_map_.assign(global_size_x_ * global_size_y_, 0);
    obstacle_miss_count_map_.assign(global_size_x_ * global_size_y_, 0);
    prior_vertical_obstacle_map_.assign(global_size_x_ * global_size_y_, 0);
    prior_obstacle_band_count_map_.assign(global_size_x_ * global_size_y_, 0);
    variance_map_.assign(global_size_x_ * global_size_y_, static_cast<float>(max_var_));
    point_count_map_.assign(global_size_x_ * global_size_y_, 0);
    // sdf_distance_map_ removed — planner uses terrain risk directly
    risk_map_cache_.assign(global_size_x_ * global_size_y_, 0.0f);
    map_center_x_ = global_origin_x_ + 0.5f * static_cast<float>(global_size_x_) * static_cast<float>(resolution_);
    map_center_y_ = global_origin_y_ + 0.5f * static_cast<float>(global_size_y_) * static_cast<float>(resolution_);

    grid_size_x_ = global_size_x_;
    grid_size_y_ = global_size_y_;
    configurePlannerGrids(grid_size_x_, grid_size_y_);

    RCLCPP_INFO(this->get_logger(),
      "Global terrain grid: origin=(%.2f, %.2f), size=%dx%d, res=%.2f",
      global_origin_x_, global_origin_y_, global_size_x_, global_size_y_, resolution_);

    if (!use_pcd_prior_)
    {
      RCLCPP_INFO(this->get_logger(),
        "PCD prior disabled: terrain analysis will be updated by live lidar cloud only.");
      return;
    }

    for (const auto& pt : cloud->points)
    {
      int grid_x = 0, grid_y = 0;
      if (!worldToMapIndex(pt.x, pt.y, grid_x, grid_y))
        continue;

      int idx = grid_y * global_size_x_ + grid_x;

      // Keep maximum elevation for each grid cell (terrain surface)
      if (point_count_map_[idx] == 0)
      {
        elevation_map_[idx] = pt.z;
        top_elevation_map_[idx] = pt.z;
        variance_map_[idx] = min_var_;
      }
      else
      {
        elevation_map_[idx] = std::min(elevation_map_[idx], pt.z);
        top_elevation_map_[idx] = std::max(top_elevation_map_[idx], pt.z);
        variance_map_[idx] = min_var_;
      }
      point_count_map_[idx]++;
    }

    std::fill(vertical_obstacle_map_.begin(), vertical_obstacle_map_.end(), 0);
    std::fill(obstacle_band_count_map_.begin(), obstacle_band_count_map_.end(), 0);
    std::fill(obstacle_miss_count_map_.begin(), obstacle_miss_count_map_.end(), 0);
    std::fill(
      collision_top_elevation_map_.begin(), collision_top_elevation_map_.end(),
      -std::numeric_limits<float>::infinity());

    // Optional PCD prior for vertical obstacles. Disabled by default because
    // dense canopy points can over-close map cells and trigger persistent NO_PATH.
    if (use_pcd_vertical_obstacles_)
    {
      for (const auto& pt : cloud->points)
      {
        int grid_x = 0, grid_y = 0;
        if (!worldToMapIndex(pt.x, pt.y, grid_x, grid_y))
          continue;
        const int idx = grid_y * global_size_x_ + grid_x;
        if (idx < 0 || idx >= static_cast<int>(elevation_map_.size()) || point_count_map_[idx] <= 0)
          continue;
        const float cell_ground = elevation_map_[static_cast<size_t>(idx)];
        float reference_ground = cell_ground;
        int neighbor_support = 0;
        estimateObstacleReferenceGround(grid_x, grid_y, cell_ground, reference_ground, neighbor_support);
        if (isCollisionBandPoint(pt.z, cell_ground, reference_ground, neighbor_support))
          recordCollisionEvidence(
            idx, pt.z, static_cast<uint16_t>(std::max(1, obstacle_min_points_)), false);
      }

      const uint16_t min_pts = static_cast<uint16_t>(std::max(1, obstacle_min_points_));
      for (const auto& pt : cloud->points)
      {
        int grid_x = 0, grid_y = 0;
        if (!worldToMapIndex(pt.x, pt.y, grid_x, grid_y))
          continue;
        const int idx = grid_y * global_size_x_ + grid_x;
        if (idx < 0 || idx >= static_cast<int>(elevation_map_.size()) ||
            idx >= static_cast<int>(obstacle_band_count_map_.size()) ||
            obstacle_band_count_map_[static_cast<size_t>(idx)] < min_pts)
          continue;

        const float cell_ground = elevation_map_[static_cast<size_t>(idx)];
        float reference_ground = cell_ground;
        int neighbor_support = 0;
        estimateObstacleReferenceGround(grid_x, grid_y, cell_ground, reference_ground, neighbor_support);
        (void)neighbor_support;
        if (isWithinCollisionHeightBand(pt.z, reference_ground))
          updateCollisionTop(idx, pt.z);
      }

      for (size_t i = 0; i < vertical_obstacle_map_.size(); ++i)
      {
        vertical_obstacle_map_[i] = (obstacle_band_count_map_[i] >= min_pts) ? 1 : 0;
      }
      prior_vertical_obstacle_map_ = vertical_obstacle_map_;
      prior_obstacle_band_count_map_ = obstacle_band_count_map_;
    }
    else
    {
      prior_vertical_obstacle_map_.assign(vertical_obstacle_map_.size(), 0);
      prior_obstacle_band_count_map_.assign(obstacle_band_count_map_.size(), 0);
    }

    RCLCPP_INFO(this->get_logger(), "Processed terrain prior: %zu points into grid", cloud->size());
  }

  void generateSyntheticTerrain()
  {
    // Generate a simple rolling terrain for testing if PCD fails
    RCLCPP_WARN(this->get_logger(), "Generating synthetic terrain...");
    
    global_origin_x_ = -0.5f * static_cast<float>(grid_size_x_ * resolution_);
    global_origin_y_ = -0.5f * static_cast<float>(grid_size_y_ * resolution_);
    global_size_x_ = grid_size_x_;
    global_size_y_ = grid_size_y_;
    elevation_map_.assign(global_size_x_ * global_size_y_, 0.0f);
    top_elevation_map_.assign(global_size_x_ * global_size_y_, -std::numeric_limits<float>::infinity());
    collision_top_elevation_map_.assign(global_size_x_ * global_size_y_, -std::numeric_limits<float>::infinity());
    vertical_obstacle_map_.assign(global_size_x_ * global_size_y_, 0);
    obstacle_band_count_map_.assign(global_size_x_ * global_size_y_, 0);
    obstacle_miss_count_map_.assign(global_size_x_ * global_size_y_, 0);
    prior_vertical_obstacle_map_.assign(global_size_x_ * global_size_y_, 0);
    prior_obstacle_band_count_map_.assign(global_size_x_ * global_size_y_, 0);
    variance_map_.assign(global_size_x_ * global_size_y_, static_cast<float>(max_var_));
    point_count_map_.assign(global_size_x_ * global_size_y_, 0);
    // sdf_distance_map_ removed — planner uses terrain risk directly
    risk_map_cache_.assign(global_size_x_ * global_size_y_, 0.0f);
    map_center_x_ = global_origin_x_ + 0.5f * static_cast<float>(global_size_x_) * static_cast<float>(resolution_);
    map_center_y_ = global_origin_y_ + 0.5f * static_cast<float>(global_size_y_) * static_cast<float>(resolution_);
    configurePlannerGrids(grid_size_x_, grid_size_y_);

    for (int row = 0; row < grid_size_y_; ++row)
    {
      for (int col = 0; col < grid_size_x_; ++col)
      {
        int idx = row * global_size_x_ + col;
        float x = global_origin_x_ + (col + 0.5f) * static_cast<float>(resolution_);
        float y = global_origin_y_ + (row + 0.5f) * static_cast<float>(resolution_);
        
        // Simple wavy terrain
        elevation_map_[idx] = 0.5f * std::sin(x * 0.5f) * std::cos(y * 0.5f) + 0.2f;
        top_elevation_map_[idx] = elevation_map_[idx];
        variance_map_[idx] = min_var_;
        point_count_map_[idx] = 1;
      }
    }
  }

  void resetLiveLocalMap(float center_x, float center_y)
  {
    map_center_x_ = center_x;
    map_center_y_ = center_y;
    global_size_x_ = grid_size_x_;
    global_size_y_ = grid_size_y_;
    global_origin_x_ = center_x - 0.5f * static_cast<float>(global_size_x_) * static_cast<float>(resolution_);
    global_origin_y_ = center_y - 0.5f * static_cast<float>(global_size_y_) * static_cast<float>(resolution_);

    elevation_map_.assign(global_size_x_ * global_size_y_, 0.0f);
    top_elevation_map_.assign(global_size_x_ * global_size_y_, -std::numeric_limits<float>::infinity());
    collision_top_elevation_map_.assign(global_size_x_ * global_size_y_, -std::numeric_limits<float>::infinity());
    vertical_obstacle_map_.assign(global_size_x_ * global_size_y_, 0);
    obstacle_band_count_map_.assign(global_size_x_ * global_size_y_, 0);
    obstacle_miss_count_map_.assign(global_size_x_ * global_size_y_, 0);
    variance_map_.assign(global_size_x_ * global_size_y_, static_cast<float>(max_var_));
    point_count_map_.assign(global_size_x_ * global_size_y_, 0);
    // sdf_distance_map_ removed — planner uses terrain risk directly
    risk_map_cache_.assign(global_size_x_ * global_size_y_, 0.0f);

    if (prior_vertical_obstacle_map_.size() != elevation_map_.size())
      prior_vertical_obstacle_map_.assign(elevation_map_.size(), 0);
    if (prior_obstacle_band_count_map_.size() != elevation_map_.size())
      prior_obstacle_band_count_map_.assign(elevation_map_.size(), 0);
    hard_obstacle_map_.assign(elevation_map_.size(), 0);
    vertical_seed_map_cache_.assign(elevation_map_.size(), 0);
  }

  template <typename T>
  void shiftRollingLayer(
    const std::vector<T> &src,
    std::vector<T> &dst,
    const T &fill_value,
    int shift_x,
    int shift_y)
  {
    dst.assign(static_cast<size_t>(global_size_x_ * global_size_y_), fill_value);
    if (src.empty())
      return;

    for (int old_y = 0; old_y < global_size_y_; ++old_y)
    {
      const int new_y = old_y - shift_y;
      if (new_y < 0 || new_y >= global_size_y_)
        continue;

      for (int old_x = 0; old_x < global_size_x_; ++old_x)
      {
        const int new_x = old_x - shift_x;
        if (new_x < 0 || new_x >= global_size_x_)
          continue;

        const size_t old_idx = static_cast<size_t>(old_y * global_size_x_ + old_x);
        const size_t new_idx = static_cast<size_t>(new_y * global_size_x_ + new_x);
        if (old_idx < src.size() && new_idx < dst.size())
          dst[new_idx] = src[old_idx];
      }
    }
  }

  void rollLiveAccumulatedMap(float center_x, float center_y)
  {
    if (global_size_x_ <= 0 || global_size_y_ <= 0 || resolution_ <= 0.0)
      return;

    const float desired_origin_x =
      center_x - 0.5f * static_cast<float>(global_size_x_) * static_cast<float>(resolution_);
    const float desired_origin_y =
      center_y - 0.5f * static_cast<float>(global_size_y_) * static_cast<float>(resolution_);
    const int shift_x = static_cast<int>(
      std::round((desired_origin_x - global_origin_x_) / static_cast<float>(resolution_)));
    const int shift_y = static_cast<int>(
      std::round((desired_origin_y - global_origin_y_) / static_cast<float>(resolution_)));

    if (shift_x == 0 && shift_y == 0)
      return;

    std::vector<float> new_elevation;
    std::vector<float> new_top;
    std::vector<float> new_collision_top;
    std::vector<float> new_variance;
    std::vector<uint8_t> new_vertical;
    std::vector<uint16_t> new_obstacle_bands;
    std::vector<uint16_t> new_obstacle_misses;
    std::vector<uint8_t> new_prior_vertical;
    std::vector<uint16_t> new_prior_bands;
    std::vector<uint8_t> new_hard_obstacles;
    std::vector<uint8_t> new_vertical_seed_cache;
    std::vector<int> new_point_count;

    shiftRollingLayer(elevation_map_, new_elevation, 0.0f, shift_x, shift_y);
    shiftRollingLayer(
      top_elevation_map_, new_top, -std::numeric_limits<float>::infinity(), shift_x, shift_y);
    shiftRollingLayer(
      collision_top_elevation_map_, new_collision_top,
      -std::numeric_limits<float>::infinity(), shift_x, shift_y);
    shiftRollingLayer(
      variance_map_, new_variance, static_cast<float>(max_var_), shift_x, shift_y);
    shiftRollingLayer(vertical_obstacle_map_, new_vertical, static_cast<uint8_t>(0), shift_x, shift_y);
    shiftRollingLayer(obstacle_band_count_map_, new_obstacle_bands, static_cast<uint16_t>(0), shift_x, shift_y);
    shiftRollingLayer(obstacle_miss_count_map_, new_obstacle_misses, static_cast<uint16_t>(0), shift_x, shift_y);
    shiftRollingLayer(
      prior_vertical_obstacle_map_, new_prior_vertical, static_cast<uint8_t>(0), shift_x, shift_y);
    shiftRollingLayer(
      prior_obstacle_band_count_map_, new_prior_bands, static_cast<uint16_t>(0), shift_x, shift_y);
    shiftRollingLayer(hard_obstacle_map_, new_hard_obstacles, static_cast<uint8_t>(0), shift_x, shift_y);
    shiftRollingLayer(
      vertical_seed_map_cache_, new_vertical_seed_cache, static_cast<uint8_t>(0), shift_x, shift_y);
    shiftRollingLayer(point_count_map_, new_point_count, 0, shift_x, shift_y);

    elevation_map_.swap(new_elevation);
    top_elevation_map_.swap(new_top);
    collision_top_elevation_map_.swap(new_collision_top);
    variance_map_.swap(new_variance);
    vertical_obstacle_map_.swap(new_vertical);
    obstacle_band_count_map_.swap(new_obstacle_bands);
    obstacle_miss_count_map_.swap(new_obstacle_misses);
    prior_vertical_obstacle_map_.swap(new_prior_vertical);
    prior_obstacle_band_count_map_.swap(new_prior_bands);
    hard_obstacle_map_.swap(new_hard_obstacles);
    vertical_seed_map_cache_.swap(new_vertical_seed_cache);
    point_count_map_.swap(new_point_count);

    global_origin_x_ += static_cast<float>(shift_x) * static_cast<float>(resolution_);
    global_origin_y_ += static_cast<float>(shift_y) * static_cast<float>(resolution_);
    map_center_x_ = global_origin_x_ +
      0.5f * static_cast<float>(global_size_x_) * static_cast<float>(resolution_);
    map_center_y_ = global_origin_y_ +
      0.5f * static_cast<float>(global_size_y_) * static_cast<float>(resolution_);
  }

  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    // Process point cloud data and update elevation map
    if (msg->data.empty()) return;

    pcl::PointCloud<pcl::PointXYZ> cloud;
    try
    {
      pcl::fromROSMsg(*msg, cloud);
    }
    catch (const std::exception &e)
    {
      RCLCPP_WARN(this->get_logger(), "fromROSMsg failed: %s", e.what());
      return;
    }

    if (cloud.empty()) return;

    float center_x = has_odom_ ? static_cast<float>(latest_pose_.position.x) : 0.0f;
    float center_y = has_odom_ ? static_cast<float>(latest_pose_.position.y) : 0.0f;
    if (cloud_is_local_map_)
    {
      if (!has_odom_)
      {
        float min_x = std::numeric_limits<float>::infinity();
        float max_x = -std::numeric_limits<float>::infinity();
        float min_y = std::numeric_limits<float>::infinity();
        float max_y = -std::numeric_limits<float>::infinity();
        for (const auto &pt : cloud.points)
        {
          if (!std::isfinite(pt.x) || !std::isfinite(pt.y))
            continue;
          min_x = std::min(min_x, pt.x);
          max_x = std::max(max_x, pt.x);
          min_y = std::min(min_y, pt.y);
          max_y = std::max(max_y, pt.y);
        }
        if (std::isfinite(min_x) && std::isfinite(max_x) &&
            std::isfinite(min_y) && std::isfinite(max_y))
        {
          center_x = 0.5f * (min_x + max_x);
          center_y = 0.5f * (min_y + max_y);
        }
      }
    }

    const bool use_snapshot_reset =
      cloud_is_local_map_ && (!local_map_sliding_window_fusion_ || !has_odom_);
    if (use_snapshot_reset)
    {
      resetLiveLocalMap(center_x, center_y);
    }
    else
    {
      if (!use_pcd_prior_ && has_odom_)
        rollLiveAccumulatedMap(center_x, center_y);

      ensureObstacleEvidenceStorage();
    }

    const uint16_t min_pts = static_cast<uint16_t>(std::max(1, obstacle_min_points_));
    const bool filter_ego_body = has_odom_ && ego_body_filter_enable_;
    const double yaw = filter_ego_body ? yawFromPose(latest_pose_) : 0.0;
    const float cyaw = static_cast<float>(std::cos(yaw));
    const float syaw = static_cast<float>(std::sin(yaw));
    const float ego_z = has_odom_ ? static_cast<float>(latest_pose_.position.z) : 0.0f;

    int update_count = 0;
    int ego_filtered_count = 0;
    int overhead_added_count = 0;
    struct ObservedPoint
    {
      float x;
      float y;
      float z;
      int grid_x;
      int grid_y;
      int idx;
    };
    std::vector<ObservedPoint> observed_points;
    observed_points.reserve(cloud.size());
    std::vector<uint8_t> obstacle_hit_map(elevation_map_.size(), 0);
    std::vector<uint8_t> raw_obstacle_hit_map(elevation_map_.size(), 0);
    std::vector<float> raw_obstacle_seed_top_map(
      elevation_map_.size(), -std::numeric_limits<float>::infinity());

    // First pass: update ground/top surfaces for the whole cloud. Obstacle
    // evidence is evaluated after this pass so thin trunks can use nearby
    // ground cells from the same scan instead of being mistaken for ground.
    for (const auto &pt : cloud.points)
    {
      if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z))
        continue;

      const float dx = pt.x - center_x;
      const float dy = pt.y - center_y;
      if (filter_ego_body)
      {
        const float bx = cyaw * dx + syaw * dy;
        const float by = -syaw * dx + cyaw * dy;
        const float bz = pt.z - ego_z;
        if (bx <= static_cast<float>(ego_body_filter_front_m_) &&
            bx >= -static_cast<float>(ego_body_filter_back_m_) &&
            std::fabs(by) <= static_cast<float>(ego_body_filter_half_width_m_) &&
            bz >= static_cast<float>(ego_body_filter_z_min_m_) &&
            bz <= static_cast<float>(ego_body_filter_z_max_m_))
        {
          ++ego_filtered_count;
          continue;
        }
      }
      if (!cloud_is_local_map_ && sensor_range_ > 1e-3)
      {
        float dist = std::sqrt(dx * dx + dy * dy);
        if (dist > sensor_range_)
          continue;
      }

      int grid_x = 0, grid_y = 0;
      if (!worldToMapIndex(pt.x, pt.y, grid_x, grid_y))
        continue;

      int idx = grid_y * global_size_x_ + grid_x;
      if (idx < 0 || idx >= static_cast<int>(elevation_map_.size()))
        continue;

      const float ground_band =
        std::max(0.05f, static_cast<float>(std::min(step_height_soft_ * 0.5, 0.15)));
      if (point_count_map_[idx] == 0)
      {
        elevation_map_[idx] = pt.z;
        top_elevation_map_[idx] = pt.z;
        variance_map_[idx] = static_cast<float>(min_var_);
      }
      else
      {
        if (pt.z < elevation_map_[idx])
          elevation_map_[idx] = 0.85f * pt.z + 0.15f * elevation_map_[idx];
        else if (pt.z <= elevation_map_[idx] + ground_band)
          elevation_map_[idx] = 0.95f * elevation_map_[idx] + 0.05f * pt.z;
        top_elevation_map_[idx] = std::max(top_elevation_map_[idx], pt.z);
        variance_map_[idx] = std::max(static_cast<float>(min_var_), variance_map_[idx] * 0.95f);
      }
      point_count_map_[idx]++;
      observed_points.push_back(ObservedPoint{pt.x, pt.y, pt.z, grid_x, grid_y, idx});
      ++update_count;
    }

    // Second pass: keep overhead canopy out of SDF/risk and record only the
    // vehicle-collision height band as vertical-obstacle evidence.
    for (const auto &pt : observed_points)
    {
      if (pt.idx < 0 || pt.idx >= static_cast<int>(elevation_map_.size()))
        continue;

      const float cell_ground = elevation_map_[static_cast<size_t>(pt.idx)];
      float reference_ground = cell_ground;
      int neighbor_support = 0;
      estimateObstacleReferenceGround(
        pt.grid_x, pt.grid_y, cell_ground, reference_ground, neighbor_support);

      const float rel_h = pt.z - reference_ground;
      if (publish_overhead_cloud_ &&
          rel_h > static_cast<float>(obstacle_max_height_ + std::max(0.0, overhead_min_clearance_m_)))
      {
        pcl::PointXYZRGB overhead_pt;
        overhead_pt.x = pt.x;
        overhead_pt.y = pt.y;
        overhead_pt.z = pt.z;
        overhead_pt.r = 80;
        overhead_pt.g = 255;
        overhead_pt.b = 170;
        overhead_points_cache_.push_back(overhead_pt);
        ++overhead_added_count;
      }
      if (isCollisionBandPoint(pt.z, cell_ground, reference_ground, neighbor_support))
      {
        if (pt.idx >= 0 && pt.idx < static_cast<int>(raw_obstacle_hit_map.size()))
        {
          const size_t raw_idx = static_cast<size_t>(pt.idx);
          raw_obstacle_hit_map[raw_idx] = 1;
          raw_obstacle_seed_top_map[raw_idx] = std::max(raw_obstacle_seed_top_map[raw_idx], pt.z);
        }
      }
    }

    for (size_t idx = 0; idx < raw_obstacle_hit_map.size(); ++idx)
    {
      if (raw_obstacle_hit_map[idx] == 0)
        continue;
      if (!acceptObstacleSeedFootprint(static_cast<int>(idx), raw_obstacle_hit_map))
        continue;

      const float seed_top = raw_obstacle_seed_top_map[idx];
      if (!std::isfinite(seed_top))
        continue;

      recordCollisionEvidence(static_cast<int>(idx), seed_top, min_pts, !cloud_is_local_map_);
      if (idx < obstacle_hit_map.size())
        obstacle_hit_map[idx] = 1;
    }

    // Once a cell is seeded by low collision-band evidence, allow higher
    // collision-band returns in the same cell to shape the obstacle column.
    // Unseeded canopy-only cells stay out of SDF/risk.
    for (const auto &pt : observed_points)
    {
      if (pt.idx < 0 || pt.idx >= static_cast<int>(obstacle_band_count_map_.size()) ||
          obstacle_band_count_map_[static_cast<size_t>(pt.idx)] < min_pts)
        continue;

      const float cell_ground = elevation_map_[static_cast<size_t>(pt.idx)];
      float reference_ground = cell_ground;
      int neighbor_support = 0;
      estimateObstacleReferenceGround(
        pt.grid_x, pt.grid_y, cell_ground, reference_ground, neighbor_support);
      (void)neighbor_support;

      if (isWithinCollisionHeightBand(pt.z, reference_ground))
        updateCollisionTop(pt.idx, pt.z);
    }

    updateObstacleEvidenceLifecycle(obstacle_hit_map);
    refreshVerticalObstacleMapFromGeometry(false);

    int active_vertical_cells = 0;
    for (const auto occ : vertical_obstacle_map_)
    {
      if (occ != 0)
        ++active_vertical_cells;
    }

    last_update_time_ = now();
    RCLCPP_DEBUG(this->get_logger(), "Processed %d points, updated %d grid cells",
                 static_cast<int>(cloud.size()), update_count);
    RCLCPP_INFO_THROTTLE(
      this->get_logger(), *this->get_clock(), 2000,
      "Cloud ingest: pts=%zu, updated=%d, active_vertical_cells=%d",
      cloud.size(), update_count, active_vertical_cells);
    RCLCPP_DEBUG_THROTTLE(
      this->get_logger(), *this->get_clock(), 2000,
      "Ego body filter removed %d points, overhead cached +%d before terrain update",
      ego_filtered_count, overhead_added_count);

    const size_t max_overhead_points =
      static_cast<size_t>(std::max(1000, overhead_cloud_max_points_));
    while (overhead_points_cache_.size() > max_overhead_points)
      overhead_points_cache_.pop_front();

    if (publish_on_cloud_)
      publishTerrain();
  }

  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    // Store current pose for terrain analysis context
    latest_pose_ = msg->pose.pose;
    latest_stamp_ = msg->header.stamp;
    has_odom_ = true;
  }

  void publishTerrain()
  {
    if (!use_pcd_prior_ && has_odom_ &&
        !(cloud_is_local_map_ && !local_map_sliding_window_fusion_))
    {
      rollLiveAccumulatedMap(
        static_cast<float>(latest_pose_.position.x),
        static_cast<float>(latest_pose_.position.y));
    }

    // Keep planner grids in global map frame for stable world-coordinate planning.
    fused_map_grid_.setPosition(Eigen::Vector2d(map_center_x_, map_center_y_));

    auto &elevation_layer = fused_map_grid_["elevation"][0];
    auto &var_layer = fused_map_grid_["var"][0];

    const int yaw_size = fused_map_grid_.getSizeYaw();

    std::vector<uint8_t> obstacle_map(static_cast<size_t>(grid_size_x_ * grid_size_y_), 0);
    std::vector<uint8_t> vertical_seed_map(static_cast<size_t>(grid_size_x_ * grid_size_y_), 0);
    std::vector<float> terrain_risk_map(static_cast<size_t>(grid_size_x_ * grid_size_y_), 0.0f);
    const size_t cell_count = static_cast<size_t>(grid_size_x_ * grid_size_y_);
    if (risk_map_cache_.size() != cell_count)
      risk_map_cache_.assign(cell_count, 0.0f);
    // sdf_distance_map_ removed — planner uses terrain risk directly
    int hard_obstacle_cells = 0;
    int vertical_obstacle_cells = 0;

    for (int row = 0; row < grid_size_y_; ++row)
    {
      for (int col = 0; col < grid_size_x_; ++col)
      {
        float world_x = global_origin_x_ + (col + 0.5f) * static_cast<float>(resolution_);
        float world_y = global_origin_y_ + (row + 0.5f) * static_cast<float>(resolution_);
        float elev = 0.0f;
        float var = static_cast<float>(max_var_);
        const bool valid = sampleAtWorld(world_x, world_y, elev, var);
        const int se2_col = toSe2Col(col);
        const int se2_row = toSe2Row(row);

        // SE2Grid index 0 maps to the positive edge of the map. Raw vectors
        // use the usual increasing world-coordinate layout, so writes are mirrored.
        elevation_layer(se2_col, se2_row) = elev;
        var_layer(se2_col, se2_row) = var;
        // sdf_map_grid_ valid layer removed — planner uses terrain risk directly

        const int idx = row * grid_size_x_ + col;
        const bool vertical_evidence =
          idx >= 0 && idx < static_cast<int>(vertical_obstacle_map_.size()) &&
          vertical_obstacle_map_[static_cast<size_t>(idx)] != 0;
        const bool vertical_obstacle = vertical_evidence;
        if (vertical_obstacle)
          ++vertical_obstacle_cells;

        float dzdx = 0.0f;
        float risk = 0.0f;
        float valid_flag = valid ? 1.0f : 0.0f;
        float zbx = 0.0f;
        float zby = 0.0f;
        float slope = 0.0f;
        float curvature = 0.0f;
        float step_height = 0.0f;
        bool hard_obstacle = false;
        float terrain_risk = 0.0f;
        float cell_ground = elev;
        float cell_top = elev;
        float cell_span = 0.0f;
        float local_relief = 0.0f;
        float ground_gap = 0.0f;
        const bool geometry_vertical_obstacle = classifyGeometryVerticalObstacle(
          col, row, cell_ground, cell_top, cell_span, local_relief, ground_gap);
        (void)ground_gap;

        if (valid)
        {
          bool traversability_valid = computeTraversabilityMetrics(
            world_x, world_y, elev, risk, zbx, zby, slope, curvature, step_height, hard_obstacle);
          if (!traversability_valid)
          {
            risk = 0.0f;
            zbx = 0.0f;
            zby = 0.0f;
            hard_obstacle = false;
          }
          terrain_risk = risk;
          const bool terrain_hard_obstacle =
            hard_obstacle ||
            (terrain_risk_to_hard_obstacle_ &&
             terrain_risk >= static_cast<float>(risk_hard_threshold_));
          const bool vertical_seed = vertical_obstacle || geometry_vertical_obstacle;

          if (vertical_seed)
            risk = 1.0f;

          if (vertical_seed)
            vertical_seed_map[static_cast<size_t>(idx)] = 1;

          obstacle_map[static_cast<size_t>(idx)] =
            (vertical_seed || (sdf_use_terrain_hard_obstacles_ && terrain_hard_obstacle)) ? 1 : 0;
        }
        else
        {
          // In live-mapping mode, unknown cells should not be hard obstacles;
          // otherwise planner gets NO_PATH before enough map is observed.
          obstacle_map[static_cast<size_t>(idx)] = 0;
          terrain_risk = 0.0f;
        }

        terrain_risk_map[static_cast<size_t>(idx)] = std::clamp(terrain_risk, 0.0f, 1.0f);

        for (int yaw_idx = 0; yaw_idx < yaw_size; ++yaw_idx)
        {
          fused_map_grid_["risk"][yaw_idx](se2_col, se2_row) = risk;
          fused_map_grid_["valid"][yaw_idx](se2_col, se2_row) = valid_flag;
          fused_map_grid_["zbx"][yaw_idx](se2_col, se2_row) = zbx;
          fused_map_grid_["zby"][yaw_idx](se2_col, se2_row) = zby;
        }
      }
    }

    // Inflate vertical obstacles so planned paths keep a practical trunk clearance.
    const int inflate_cells = std::max(
      0, static_cast<int>(std::ceil(std::max(0.0, vertical_obstacle_inflation_m_) / std::max(1e-3, resolution_))));
    if (inflate_cells > 0)
    {
      std::vector<uint8_t> inflated = obstacle_map;
      const int r2 = inflate_cells * inflate_cells;
      for (int row = 0; row < grid_size_y_; ++row)
      {
        for (int col = 0; col < grid_size_x_; ++col)
        {
          const int idx = row * grid_size_x_ + col;
          if (vertical_seed_map[static_cast<size_t>(idx)] == 0)
            continue;
          for (int dy = -inflate_cells; dy <= inflate_cells; ++dy)
          {
            const int rr = row + dy;
            if (rr < 0 || rr >= grid_size_y_)
              continue;
            for (int dx = -inflate_cells; dx <= inflate_cells; ++dx)
            {
              if (dx * dx + dy * dy > r2)
                continue;
              const int cc = col + dx;
              if (cc < 0 || cc >= grid_size_x_)
                continue;
              inflated[static_cast<size_t>(rr * grid_size_x_ + cc)] = 1;
            }
          }
        }
      }
      obstacle_map.swap(inflated);
    }

    // Clear only the vehicle body core.  A wide circular clear zone erases
    // nearby tree/wall evidence exactly when the vehicle approaches it, making
    // SDF/risk look traversable and allowing the planner to drive into obstacles.
    if (has_odom_ && ego_clear_radius_ > 1e-3)
    {
      const float cx = static_cast<float>(latest_pose_.position.x);
      const float cy = static_cast<float>(latest_pose_.position.y);
      const double yaw = yawFromPose(latest_pose_);
      const float cyaw = static_cast<float>(std::cos(yaw));
      const float syaw = static_cast<float>(std::sin(yaw));
      const float clear_front = static_cast<float>(
        std::min(std::max(0.18, 0.32 * ego_body_filter_front_m_), ego_clear_radius_));
      const float clear_back = static_cast<float>(
        std::min(std::max(0.16, 0.35 * ego_body_filter_back_m_), ego_clear_radius_));
      const float clear_half_width = static_cast<float>(
        std::min(std::max(0.12, 0.42 * ego_body_filter_half_width_m_), ego_clear_radius_ * 0.65));
      const float preserve_inner_r2 = 0.12f * 0.12f;
      const uint16_t min_pts = static_cast<uint16_t>(std::max(1, obstacle_min_points_));
      for (int row = 0; row < grid_size_y_; ++row)
      {
        for (int col = 0; col < grid_size_x_; ++col)
        {
          const float wx = global_origin_x_ + (col + 0.5f) * static_cast<float>(resolution_);
          const float wy = global_origin_y_ + (row + 0.5f) * static_cast<float>(resolution_);
          const float dx = wx - cx;
          const float dy = wy - cy;
          const float bx = cyaw * dx + syaw * dy;
          const float by = -syaw * dx + cyaw * dy;
          if (bx <= clear_front && bx >= -clear_back && std::fabs(by) <= clear_half_width)
          {
            const size_t idx = static_cast<size_t>(row * grid_size_x_ + col);
            const bool obstacle_evidence =
              (idx < vertical_seed_map.size() && vertical_seed_map[idx] != 0) ||
              (idx < obstacle_map.size() && obstacle_map[idx] != 0) ||
              (idx < obstacle_band_count_map_.size() && obstacle_band_count_map_[idx] >= min_pts) ||
              (idx < collision_top_elevation_map_.size() && std::isfinite(collision_top_elevation_map_[idx]));
            if (obstacle_evidence && (dx * dx + dy * dy) > preserve_inner_r2)
              continue;

            obstacle_map[idx] = 0;
            vertical_seed_map[idx] = 0;
            terrain_risk_map[idx] = 0.0f;
            if (idx < vertical_obstacle_map_.size())
              vertical_obstacle_map_[idx] = 0;
            if (idx < obstacle_band_count_map_.size())
              obstacle_band_count_map_[idx] = 0;
            if (idx < obstacle_miss_count_map_.size())
              obstacle_miss_count_map_[idx] = 0;
            if (idx < collision_top_elevation_map_.size())
              collision_top_elevation_map_[idx] = -std::numeric_limits<float>::infinity();
            if (idx < top_elevation_map_.size() && idx < elevation_map_.size())
              top_elevation_map_[idx] = elevation_map_[idx];
          }
        }
      }
    }

    // Build tree-centered risk field from vertical obstacle seeds so risk region
    // is spatially tied to trunks instead of broad terrain gradients.
    std::vector<float> tree_dist_field(
      static_cast<size_t>(grid_size_x_ * grid_size_y_), std::numeric_limits<float>::infinity());
    for (int row = 0; row < grid_size_y_; ++row)
    {
      for (int col = 0; col < grid_size_x_; ++col)
      {
        const int idx = row * grid_size_x_ + col;
        if (vertical_seed_map[static_cast<size_t>(idx)] != 0)
          tree_dist_field[static_cast<size_t>(idx)] = 0.0f;
      }
    }

    const float w_diag_tree = std::sqrt(2.0f);
    for (int row = 0; row < grid_size_y_; ++row)
    {
      for (int col = 0; col < grid_size_x_; ++col)
      {
        const int idx = row * grid_size_x_ + col;
        float best = tree_dist_field[static_cast<size_t>(idx)];
        if (col > 0)
          best = std::min(best, tree_dist_field[static_cast<size_t>(idx - 1)] + 1.0f);
        if (row > 0)
          best = std::min(best, tree_dist_field[static_cast<size_t>(idx - grid_size_x_)] + 1.0f);
        if (col > 0 && row > 0)
          best = std::min(best, tree_dist_field[static_cast<size_t>(idx - grid_size_x_ - 1)] + w_diag_tree);
        if (col + 1 < grid_size_x_ && row > 0)
          best = std::min(best, tree_dist_field[static_cast<size_t>(idx - grid_size_x_ + 1)] + w_diag_tree);
        tree_dist_field[static_cast<size_t>(idx)] = best;
      }
    }
    for (int row = grid_size_y_ - 1; row >= 0; --row)
    {
      for (int col = grid_size_x_ - 1; col >= 0; --col)
      {
        const int idx = row * grid_size_x_ + col;
        float best = tree_dist_field[static_cast<size_t>(idx)];
        if (col + 1 < grid_size_x_)
          best = std::min(best, tree_dist_field[static_cast<size_t>(idx + 1)] + 1.0f);
        if (row + 1 < grid_size_y_)
          best = std::min(best, tree_dist_field[static_cast<size_t>(idx + grid_size_x_)] + 1.0f);
        if (col + 1 < grid_size_x_ && row + 1 < grid_size_y_)
          best = std::min(best, tree_dist_field[static_cast<size_t>(idx + grid_size_x_ + 1)] + w_diag_tree);
        if (col > 0 && row + 1 < grid_size_y_)
          best = std::min(best, tree_dist_field[static_cast<size_t>(idx + grid_size_x_ - 1)] + w_diag_tree);
        tree_dist_field[static_cast<size_t>(idx)] = best;
      }
    }

    const float terrain_risk_scale = std::clamp(static_cast<float>(terrain_risk_scale_), 0.0f, 1.0f);
    const float tree_risk_power = std::clamp(static_cast<float>(tree_risk_power_), 1.0f, 6.0f);
    const float tree_hard_keepout_radius = std::max(0.0f, static_cast<float>(tree_hard_keepout_radius_m_));
    const float non_tree_risk_cap = std::clamp(static_cast<float>(non_tree_risk_cap_), 0.05f, 0.95f);
    const float tree_risk_radius = std::max(
      static_cast<float>(resolution_),
      static_cast<float>(std::max(tree_risk_radius_m_, vertical_obstacle_inflation_m_ + 0.10)));
    int tree_keepout_cells = 0;
    for (int row = 0; row < grid_size_y_; ++row)
    {
      for (int col = 0; col < grid_size_x_; ++col)
      {
        const int idx = row * grid_size_x_ + col;
        const int se2_col = toSe2Col(col);
        const int se2_row = toSe2Row(row);
        const float valid_flag = fused_map_grid_["valid"][0](se2_col, se2_row);
        float tree_risk = 0.0f;
        const float tree_dist_m = tree_dist_field[static_cast<size_t>(idx)] * static_cast<float>(resolution_);
        if (std::isfinite(tree_dist_m) && tree_dist_m <= tree_risk_radius)
        {
          const float ratio = std::clamp(1.0f - tree_dist_m / tree_risk_radius, 0.0f, 1.0f);
          tree_risk = std::pow(ratio, tree_risk_power);
        }
        if (vertical_seed_map[static_cast<size_t>(idx)] != 0)
          tree_risk = 1.0f;

        if (valid_flag >= 0.5f && tree_hard_keepout_radius > 1e-3f &&
            std::isfinite(tree_dist_m) && tree_dist_m <= tree_hard_keepout_radius)
        {
          if (obstacle_map[static_cast<size_t>(idx)] == 0)
          {
            obstacle_map[static_cast<size_t>(idx)] = 1;
            ++tree_keepout_cells;
          }
        }

        const bool hard_obstacle = obstacle_map[static_cast<size_t>(idx)] != 0;
        float fused_risk = std::max(
          terrain_risk_scale * terrain_risk_map[static_cast<size_t>(idx)],
          tree_risk);
        if (valid_flag < 0.5f)
          fused_risk = 0.0f;

        // Keep high-risk concentrated near tree seeds/hard obstacles.
        if (!hard_obstacle && vertical_seed_map[static_cast<size_t>(idx)] == 0)
          fused_risk = std::min(fused_risk, non_tree_risk_cap);

        if (hard_obstacle)
          fused_risk = std::max(fused_risk, static_cast<float>(risk_hard_threshold_));
        risk_map_cache_[static_cast<size_t>(idx)] = std::clamp(fused_risk, 0.0f, 1.0f);

        for (int yaw_idx = 0; yaw_idx < yaw_size; ++yaw_idx)
        {
          fused_map_grid_["risk"][yaw_idx](se2_col, se2_row) = risk_map_cache_[static_cast<size_t>(idx)];
        }
      }
    }

    // Cache maps for risk visualization.
    vertical_seed_map_cache_ = vertical_seed_map;
    hard_obstacle_map_ = obstacle_map;

    hard_obstacle_cells = 0;
    for (const auto occ : obstacle_map)
    {
      if (occ != 0)
        ++hard_obstacle_cells;
    }

    RCLCPP_INFO_THROTTLE(
      this->get_logger(), *this->get_clock(), 2000,
      "Hard obstacles: %d/%d (%.1f%%), vertical=%d, keepout=%d(r=%.2fm), min_pts=%d, decay=%.2f, inflate=%.2fm, ego_clear=%.2fm",
      hard_obstacle_cells,
      grid_size_x_ * grid_size_y_,
      100.0 * static_cast<double>(hard_obstacle_cells) /
        std::max(1, grid_size_x_ * grid_size_y_),
      vertical_obstacle_cells,
      tree_keepout_cells,
      std::max(0.0, tree_hard_keepout_radius_m_),
      std::max(1, obstacle_min_points_),
      std::clamp(obstacle_count_decay_, 0.0, 1.0),
      std::max(0.0, vertical_obstacle_inflation_m_),
      std::max(0.0, ego_clear_radius_));

    // ESDF/SDF map generation removed — planner now uses terrain risk directly
    // from fused_map for obstacle avoidance, eliminating the chamfer distance
    // computation and sdf_map publishing.

    se2_grid_msgs::msg::SE2Grid fused_msg;
    se2_grid::SE2GridRosConverter::toMessage(fused_map_grid_, fused_msg);
    const auto terrain_stamp = has_odom_ ? latest_stamp_ : now();
    fused_msg.info.header.stamp = terrain_stamp;
    fused_msg.info.header.frame_id = "map";
    fused_map_pub_->publish(fused_msg);

    bool publish_visual_clouds = false;
    if (publish_internal_visual_clouds_)
    {
      const double now_sec = this->now().seconds();
      const double visual_period = 1.0 / std::max(0.1, visual_publish_rate_hz_);
      publish_visual_clouds =
        (last_visual_publish_sec_ < 0.0) ||
        ((now_sec - last_visual_publish_sec_) >= visual_period);
      if (publish_visual_clouds)
        last_visual_publish_sec_ = now_sec;
    }

    if (publish_visual_clouds)
    {
      publishElevationCloud(terrain_stamp);
      publishRiskCloud(terrain_stamp);
      publishObstacleCloud(terrain_stamp);
      publishOverheadCloud(terrain_stamp);
    }
  }

  void publishElevationCloud(const rclcpp::Time &stamp)
  {
    pcl::PointCloud<pcl::PointXYZRGB> pcl_cloud;
    pcl_cloud.is_dense = false;

    float center_x = has_odom_ ? static_cast<float>(latest_pose_.position.x) : 0.0f;
    float center_y = has_odom_ ? static_cast<float>(latest_pose_.position.y) : 0.0f;
    const int viz_x = std::max(2, std::min(viz_grid_size_x_, grid_size_x_));
    const int viz_y = std::max(2, std::min(viz_grid_size_y_, grid_size_y_));
    float grid_half_x = viz_x * resolution_ / 2.0f;
    float grid_half_y = viz_y * resolution_ / 2.0f;

    // First pass: count valid local points and min/max height for local contrast.
    size_t valid_count = 0;
    for (int row = 0; row < viz_y; ++row)
    {
      for (int col = 0; col < viz_x; ++col)
      {
        float world_x = center_x - grid_half_x + (col + 0.5f) * resolution_;
        float world_y = center_y - grid_half_y + (row + 0.5f) * resolution_;
        float elev = 0.0f;
        float var = static_cast<float>(max_var_);
        if (sampleVisualElevationFilledAtWorld(world_x, world_y, elev, var))
          ++valid_count;
      }
    }
    if (valid_count == 0)
      return;

    pcl_cloud.points.reserve(valid_count);
    // Compute local min/max in the vehicle-centered window to keep the
    // heatmap contrast focused on the nearby terrain like the ROS1 view.
    float min_elevation = std::numeric_limits<float>::max();
    float max_elevation = -std::numeric_limits<float>::max();
    for (int row = 0; row < viz_y; ++row)
    {
      for (int col = 0; col < viz_x; ++col)
      {
        float world_x = center_x - grid_half_x + (col + 0.5f) * resolution_;
        float world_y = center_y - grid_half_y + (row + 0.5f) * resolution_;
        float elev = 0.0f;
        float var = static_cast<float>(max_var_);
        if (sampleVisualElevationFilledAtWorld(world_x, world_y, elev, var))
        {
          min_elevation = std::min(min_elevation, elev);
          max_elevation = std::max(max_elevation, elev);
        }
      }
    }
    if (!std::isfinite(min_elevation) || !std::isfinite(max_elevation) || max_elevation <= min_elevation)
    {
      min_elevation = 0.0f;
      max_elevation = 1.0f;
    }
    float elevation_range = std::max(1e-3f, max_elevation - min_elevation);

    for (int row = 0; row < viz_y; ++row)
    {
      for (int col = 0; col < viz_x; ++col)
      {
        float world_x = center_x - grid_half_x + (col + 0.5f) * resolution_;
        float world_y = center_y - grid_half_y + (row + 0.5f) * resolution_;
        float elev = 0.0f;
        float var = static_cast<float>(max_var_);
        bool valid = sampleVisualElevationFilledAtWorld(world_x, world_y, elev, var);

        if (!valid)
          continue;

        float normalized_height = std::clamp((elev - min_elevation) / elevation_range, 0.0f, 1.0f);
        uint8_t r = 0, g = 0, b = 0;
        jetColor(normalized_height, r, g, b);

        pcl::PointXYZRGB pt;
        pt.x = world_x;
        pt.y = world_y;
        pt.z = elev;
        pt.r = r;
        pt.g = g;
        pt.b = b;
        pcl_cloud.points.push_back(pt);
      }
    }

    pcl_cloud.width = static_cast<uint32_t>(pcl_cloud.points.size());
    pcl_cloud.height = 1;

    sensor_msgs::msg::PointCloud2 ros_cloud;
    pcl::toROSMsg(pcl_cloud, ros_cloud);
    ros_cloud.header.stamp = stamp;
    ros_cloud.header.frame_id = "map";
    elevation_cloud_pub_->publish(ros_cloud);
  }

  // publishSdfCloud removed — planner uses terrain risk directly

  void publishRiskCloud(const rclcpp::Time &stamp)
  {
    pcl::PointCloud<pcl::PointXYZRGB> pcl_cloud;
    pcl_cloud.is_dense = false;
    float center_x = has_odom_ ? static_cast<float>(latest_pose_.position.x) : 0.0f;
    float center_y = has_odom_ ? static_cast<float>(latest_pose_.position.y) : 0.0f;
    const int viz_x = std::max(2, std::min(viz_grid_size_x_, grid_size_x_));
    const int viz_y = std::max(2, std::min(viz_grid_size_y_, grid_size_y_));
    float grid_half_x = viz_x * resolution_ / 2.0f;
    float grid_half_y = viz_y * resolution_ / 2.0f;
    pcl_cloud.points.reserve(static_cast<size_t>(viz_x * viz_y));
    size_t valid_points = 0;
    size_t seed_points = 0;

    for (int row = 0; row < viz_y; ++row)
    {
      for (int col = 0; col < viz_x; ++col)
      {
        float world_x = center_x - grid_half_x + (col + 0.5f) * resolution_;
        float world_y = center_y - grid_half_y + (row + 0.5f) * resolution_;

        float elev = 0.0f;
        float var = static_cast<float>(max_var_);
        const bool valid_sample = sampleAtWorld(world_x, world_y, elev, var);

        int gx = 0, gy = 0;
        if (!worldToMapIndex(world_x, world_y, gx, gy))
          continue;

        pcl::PointXYZRGB pt;
        pt.x = world_x;
        pt.y = world_y;

        if (!valid_sample)
        {
          // Keep traversability cloud visually continuous in RViz even where
          // elevation samples are temporarily unavailable.
          pt.z = has_odom_ ? static_cast<float>(latest_pose_.position.z + 0.02) : 0.02f;
          riskDisplayColor(0.0f, false, pt.r, pt.g, pt.b);
          pcl_cloud.points.push_back(pt);
          continue;
        }
        ++valid_points;

        float risk = 0.0f;
        if (sampleRiskAtWorld(world_x, world_y, risk))
          risk = std::clamp(risk, 0.0f, 1.0f);

        const int idx = gy * global_size_x_ + gx;
        const bool vertical_seed =
          (idx >= 0 && idx < static_cast<int>(vertical_seed_map_cache_.size()) &&
           vertical_seed_map_cache_[static_cast<size_t>(idx)] != 0);

        pt.x = world_x;
        pt.y = world_y;
        pt.z = elev + 0.03f;
        if (vertical_seed || risk >= static_cast<float>(risk_vis_hard_threshold_))
        {
          if (vertical_seed)
            ++seed_points;
          riskDisplayColor(1.0f, true, pt.r, pt.g, pt.b);
        }
        else
        {
          const float deadband = std::clamp(static_cast<float>(risk_visual_deadband_), 0.0f, 0.95f);
          if (risk <= deadband)
          {
            riskDisplayColor(0.0f, false, pt.r, pt.g, pt.b);
          }
          else
          {
            const float denom = std::max(1e-3f, static_cast<float>(risk_vis_hard_threshold_) - deadband);
            const float normalized = std::clamp((risk - deadband) / denom, 0.0f, 1.0f);
            const float gamma = std::clamp(static_cast<float>(risk_visual_gamma_), 0.4f, 4.0f);
            const float vis_risk = std::pow(normalized, gamma);
            riskDisplayColor(vis_risk, false, pt.r, pt.g, pt.b);
          }
        }
        pcl_cloud.points.push_back(pt);
      }
    }

    pcl_cloud.width = static_cast<uint32_t>(pcl_cloud.points.size());
    pcl_cloud.height = 1;

    sensor_msgs::msg::PointCloud2 ros_cloud;
    pcl::toROSMsg(pcl_cloud, ros_cloud);
    ros_cloud.header.stamp = stamp;
    ros_cloud.header.frame_id = "map";
    risk_cloud_pub_->publish(ros_cloud);
    RCLCPP_INFO_THROTTLE(
      this->get_logger(), *this->get_clock(), 2000,
      "Risk cloud: pts=%zu valid=%zu seed_red=%zu vis_hard=%.2f",
      pcl_cloud.points.size(), valid_points, seed_points,
      std::clamp(risk_vis_hard_threshold_, 0.0, 1.0));
  }

  bool worldToMapIndex(float world_x, float world_y, int &grid_x, int &grid_y) const
  {
    if (!std::isfinite(world_x) || !std::isfinite(world_y) || !std::isfinite(resolution_) || resolution_ <= 0.0)
      return false;

    grid_x = static_cast<int>((world_x - global_origin_x_) / static_cast<float>(resolution_));
    grid_y = static_cast<int>((world_y - global_origin_y_) / static_cast<float>(resolution_));

    return !(grid_x < 0 || grid_x >= global_size_x_ || grid_y < 0 || grid_y >= global_size_y_);
  }

  int toSe2Col(int grid_x) const
  {
    return grid_size_x_ - 1 - grid_x;
  }

  int toSe2Row(int grid_y) const
  {
    return grid_size_y_ - 1 - grid_y;
  }

  bool sampleAtWorld(float world_x, float world_y, float &elev, float &var) const
  {
    int gx = 0, gy = 0;
    if (!worldToMapIndex(world_x, world_y, gx, gy))
    {
      elev = 0.0f;
      var = static_cast<float>(max_var_);
      return false;
    }

    const int idx = gy * global_size_x_ + gx;
    if (idx < 0 || idx >= static_cast<int>(point_count_map_.size()) || point_count_map_[idx] <= 0)
    {
      elev = 0.0f;
      var = static_cast<float>(max_var_);
      return false;
    }

    elev = elevation_map_[idx];
    var = variance_map_[idx];
    return true;
  }

  // sampleSdfAtWorld removed — planner uses terrain risk directly

  bool sampleRiskAtWorld(float world_x, float world_y, float &risk) const
  {
    int gx = 0, gy = 0;
    if (!worldToMapIndex(world_x, world_y, gx, gy))
      return false;

    const int idx = gy * global_size_x_ + gx;
    if (idx < 0 || idx >= static_cast<int>(risk_map_cache_.size()))
      return false;

    risk = risk_map_cache_[static_cast<size_t>(idx)];
    return std::isfinite(risk);
  }

  bool sampleVisualElevationAtWorld(float world_x, float world_y, float &elev, float &var) const
  {
    if (!sampleAtWorld(world_x, world_y, elev, var))
      return false;

    if (!visualize_obstacle_tops_)
      return true;

    int gx = 0, gy = 0;
    if (!worldToMapIndex(world_x, world_y, gx, gy))
      return true;

    const int idx = gy * global_size_x_ + gx;
    if (idx < 0 || idx >= static_cast<int>(vertical_seed_map_cache_.size()) ||
        idx >= static_cast<int>(collision_top_elevation_map_.size()))
      return true;

    if (vertical_seed_map_cache_[static_cast<size_t>(idx)] != 0 &&
        std::isfinite(collision_top_elevation_map_[static_cast<size_t>(idx)]))
    {
      elev = std::max(elev, collision_top_elevation_map_[static_cast<size_t>(idx)]);
    }
    return true;
  }

  bool sampleVisualElevationFilledAtWorld(float world_x, float world_y, float &elev, float &var) const
  {
    if (sampleVisualElevationAtWorld(world_x, world_y, elev, var))
      return true;

    if (!elevation_visual_fill_missing_ || elevation_visual_fill_radius_m_ <= 1e-3)
      return false;

    int center_x = 0, center_y = 0;
    if (!worldToMapIndex(world_x, world_y, center_x, center_y))
      return false;

    const int radius_cells = std::max(
      1, static_cast<int>(std::ceil(elevation_visual_fill_radius_m_ / std::max(1e-3, resolution_))));
    const int radius_sq = radius_cells * radius_cells;
    int best_x = -1;
    int best_y = -1;
    int best_d2 = std::numeric_limits<int>::max();

    for (int dy = -radius_cells; dy <= radius_cells; ++dy)
    {
      const int gy = center_y + dy;
      if (gy < 0 || gy >= global_size_y_)
        continue;
      for (int dx = -radius_cells; dx <= radius_cells; ++dx)
      {
        const int d2 = dx * dx + dy * dy;
        if (d2 > radius_sq || d2 >= best_d2)
          continue;
        const int gx = center_x + dx;
        if (gx < 0 || gx >= global_size_x_)
          continue;
        const int idx = gy * global_size_x_ + gx;
        if (idx < 0 || idx >= static_cast<int>(point_count_map_.size()) || point_count_map_[idx] <= 0)
          continue;
        best_x = gx;
        best_y = gy;
        best_d2 = d2;
      }
    }

    if (best_x < 0 || best_y < 0)
      return false;

    const int best_idx = best_y * global_size_x_ + best_x;
    elev = elevation_map_[static_cast<size_t>(best_idx)];
    var = variance_map_[static_cast<size_t>(best_idx)];
    if (visualize_obstacle_tops_ &&
        best_idx >= 0 && best_idx < static_cast<int>(vertical_seed_map_cache_.size()) &&
        best_idx < static_cast<int>(collision_top_elevation_map_.size()) &&
        vertical_seed_map_cache_[static_cast<size_t>(best_idx)] != 0 &&
        std::isfinite(collision_top_elevation_map_[static_cast<size_t>(best_idx)]))
    {
      elev = std::max(elev, collision_top_elevation_map_[static_cast<size_t>(best_idx)]);
    }
    return true;
  }

  bool cellHasSupport(int grid_x, int grid_y, int min_points) const
  {
    if (grid_x < 0 || grid_x >= global_size_x_ || grid_y < 0 || grid_y >= global_size_y_)
      return false;

    const int idx = grid_y * global_size_x_ + grid_x;
    if (idx < 0 || idx >= static_cast<int>(point_count_map_.size()))
      return false;

    return point_count_map_[idx] >= std::max(1, min_points);
  }

  void ensureObstacleEvidenceStorage()
  {
    const size_t cell_count = elevation_map_.size();
    if (vertical_obstacle_map_.size() != cell_count)
      vertical_obstacle_map_.assign(cell_count, 0);
    if (obstacle_band_count_map_.size() != cell_count)
      obstacle_band_count_map_.assign(cell_count, 0);
    if (obstacle_miss_count_map_.size() != cell_count)
      obstacle_miss_count_map_.assign(cell_count, 0);
    if (collision_top_elevation_map_.size() != cell_count)
      collision_top_elevation_map_.assign(cell_count, -std::numeric_limits<float>::infinity());
  }

  bool estimateObstacleReferenceGround(
    int grid_x, int grid_y,
    float fallback_ground,
    float &reference_ground,
    int &neighbor_support) const
  {
    reference_ground = std::numeric_limits<float>::infinity();
    neighbor_support = 0;
    if (grid_x < 0 || grid_x >= global_size_x_ || grid_y < 0 || grid_y >= global_size_y_)
      return false;

    const int radius_cells = std::clamp(
      static_cast<int>(std::ceil(
        std::max(0.0, obstacle_ground_search_radius_m_) / std::max(1e-3, resolution_))),
      1, 4);
    const int radius_sq = radius_cells * radius_cells;
    std::vector<float> ground_candidates;
    ground_candidates.reserve(static_cast<size_t>((2 * radius_cells + 1) * (2 * radius_cells + 1)));

    for (int dy = -radius_cells; dy <= radius_cells; ++dy)
    {
      const int ny = grid_y + dy;
      if (ny < 0 || ny >= global_size_y_)
        continue;
      for (int dx = -radius_cells; dx <= radius_cells; ++dx)
      {
        if (dx == 0 && dy == 0)
          continue;
        if (dx * dx + dy * dy > radius_sq)
          continue;
        const int nx = grid_x + dx;
        if (nx < 0 || nx >= global_size_x_)
          continue;
        const int nidx = ny * global_size_x_ + nx;
        if (nidx < 0 || nidx >= static_cast<int>(point_count_map_.size()) ||
            point_count_map_[static_cast<size_t>(nidx)] <= 0 ||
            nidx >= static_cast<int>(elevation_map_.size()))
          continue;
        const float ground = elevation_map_[static_cast<size_t>(nidx)];
        if (!std::isfinite(ground))
          continue;
        ground_candidates.push_back(ground);
      }
    }

    neighbor_support = static_cast<int>(ground_candidates.size());
    if (!ground_candidates.empty())
    {
      std::sort(ground_candidates.begin(), ground_candidates.end());
      const size_t percentile_idx =
        ground_candidates.size() >= 5 ? ground_candidates.size() / 4 : 0;
      reference_ground = ground_candidates[percentile_idx];
      return true;
    }

    if (std::isfinite(fallback_ground))
    {
      reference_ground = fallback_ground;
      return true;
    }
    return false;
  }

  bool isCollisionBandPoint(
    float point_z,
    float cell_ground,
    float reference_ground,
    int neighbor_support) const
  {
    if (!std::isfinite(point_z) || !std::isfinite(cell_ground) || !std::isfinite(reference_ground))
      return false;
    if (neighbor_support <= 0)
      return false;

    const float min_h = static_cast<float>(std::max(0.01, obstacle_min_height_));
    const float max_h = static_cast<float>(std::max(obstacle_min_height_ + 0.05, obstacle_max_height_));
    const float rel_h = point_z - reference_ground;
    if (rel_h <= min_h || rel_h >= max_h)
      return false;
    const float seed_max_h = std::clamp(
      static_cast<float>(obstacle_seed_max_height_), min_h + 0.05f, max_h);
    if (rel_h > seed_max_h)
      return false;

    const float same_cell_span = point_z - cell_ground;
    const float base_gap = cell_ground - reference_ground;
    const float vertical_gate = std::max(0.16f, min_h * 0.80f);
    const bool has_vertical_structure =
      same_cell_span > vertical_gate ||
      (neighbor_support > 0 && base_gap > vertical_gate);

    return has_vertical_structure;
  }

  bool isWithinCollisionHeightBand(float point_z, float reference_ground) const
  {
    if (!std::isfinite(point_z) || !std::isfinite(reference_ground))
      return false;

    const float min_h = static_cast<float>(std::max(0.01, obstacle_min_height_));
    const float max_h = static_cast<float>(std::max(obstacle_min_height_ + 0.05, obstacle_max_height_));
    const float rel_h = point_z - reference_ground;
    return rel_h > min_h && rel_h < max_h;
  }

  bool acceptObstacleSeedFootprint(int idx, const std::vector<uint8_t> &raw_obstacle_hit_map) const
  {
    if (!obstacle_footprint_filter_enable_)
      return true;
    if (idx < 0 || idx >= static_cast<int>(raw_obstacle_hit_map.size()) ||
        raw_obstacle_hit_map[static_cast<size_t>(idx)] == 0 ||
        global_size_x_ <= 0 || global_size_y_ <= 0)
      return false;

    const int center_x = idx % global_size_x_;
    const int center_y = idx / global_size_x_;
    const int radius_cells = std::clamp(
      static_cast<int>(std::ceil(
        std::max(0.0, obstacle_footprint_filter_radius_m_) / std::max(1e-3, resolution_))),
      1, 4);
    const int radius_sq = radius_cells * radius_cells;
    int hit_count = 0;
    int min_x = center_x;
    int max_x = center_x;
    int min_y = center_y;
    int max_y = center_y;

    for (int dy = -radius_cells; dy <= radius_cells; ++dy)
    {
      const int gy = center_y + dy;
      if (gy < 0 || gy >= global_size_y_)
        continue;
      for (int dx = -radius_cells; dx <= radius_cells; ++dx)
      {
        if (dx * dx + dy * dy > radius_sq)
          continue;
        const int gx = center_x + dx;
        if (gx < 0 || gx >= global_size_x_)
          continue;
        const int nidx = gy * global_size_x_ + gx;
        if (nidx < 0 || nidx >= static_cast<int>(raw_obstacle_hit_map.size()) ||
            raw_obstacle_hit_map[static_cast<size_t>(nidx)] == 0)
          continue;

        ++hit_count;
        min_x = std::min(min_x, gx);
        max_x = std::max(max_x, gx);
        min_y = std::min(min_y, gy);
        max_y = std::max(max_y, gy);
      }
    }

    const int max_neighbors = std::max(1, obstacle_footprint_max_neighbors_);
    if (hit_count <= max_neighbors)
      return true;

    const int span_x = max_x - min_x + 1;
    const int span_y = max_y - min_y + 1;
    const int line_width = std::max(1, obstacle_footprint_line_width_cells_);
    const bool line_like = span_x <= line_width || span_y <= line_width;
    if (line_like)
      return true;

    // Dense 2-D blobs in the low seed band are usually low tree-crown skirts
    // or leaf returns. Trunks and walls remain compact or line-like locally.
    return false;
  }

  void recordCollisionEvidence(int idx, float point_z, uint16_t min_pts, bool set_immediate_vertical)
  {
    if (idx < 0 || idx >= static_cast<int>(obstacle_band_count_map_.size()))
      return;

    auto &cnt = obstacle_band_count_map_[static_cast<size_t>(idx)];
    const uint32_t increment = static_cast<uint32_t>(std::max(1, obstacle_hit_increment_));
    const uint32_t next_cnt = std::min<uint32_t>(
      static_cast<uint32_t>(std::numeric_limits<uint16_t>::max()),
      static_cast<uint32_t>(cnt) + increment);
    cnt = static_cast<uint16_t>(next_cnt);

    if (idx < static_cast<int>(collision_top_elevation_map_.size()))
    {
      auto &collision_top = collision_top_elevation_map_[static_cast<size_t>(idx)];
      if (!std::isfinite(collision_top) || point_z > collision_top)
        collision_top = point_z;
    }

    if (set_immediate_vertical && cnt >= min_pts &&
        idx < static_cast<int>(vertical_obstacle_map_.size()))
    {
      vertical_obstacle_map_[static_cast<size_t>(idx)] = 1;
    }
  }

  void updateObstacleEvidenceLifecycle(const std::vector<uint8_t> &obstacle_hit_map)
  {
    ensureObstacleEvidenceStorage();

    const uint16_t min_pts = static_cast<uint16_t>(std::max(1, obstacle_min_points_));
    const int persistence_misses = std::max(0, obstacle_persistence_misses_);
    const double decay = std::clamp(obstacle_count_decay_, 0.0, 1.0);
    const uint16_t max_miss_count = static_cast<uint16_t>(
      std::min(65535, std::max(1, persistence_misses + 20)));

    for (size_t i = 0; i < obstacle_band_count_map_.size(); ++i)
    {
      const bool hit = i < obstacle_hit_map.size() && obstacle_hit_map[i] != 0;
      const bool was_active = i < vertical_obstacle_map_.size() && vertical_obstacle_map_[i] != 0;
      const bool has_collision_top =
        i < collision_top_elevation_map_.size() &&
        std::isfinite(collision_top_elevation_map_[i]);

      uint16_t prior_cnt =
        (i < prior_obstacle_band_count_map_.size()) ? prior_obstacle_band_count_map_[i] : 0;
      uint32_t cnt = std::max<uint32_t>(obstacle_band_count_map_[i], prior_cnt);

      if (hit)
      {
        obstacle_miss_count_map_[i] = 0;
      }
      else
      {
        if (obstacle_miss_count_map_[i] < max_miss_count)
          ++obstacle_miss_count_map_[i];
        if (cnt > 0)
        {
          uint32_t decayed = static_cast<uint32_t>(std::floor(static_cast<double>(cnt) * decay));
          if (decay < 1.0 && decayed >= cnt)
            decayed = cnt - 1;
          cnt = decayed;
        }
      }

      obstacle_band_count_map_[i] = static_cast<uint16_t>(
        std::min<uint32_t>(cnt, static_cast<uint32_t>(std::numeric_limits<uint16_t>::max())));

      const bool active_by_confidence = has_collision_top && obstacle_band_count_map_[i] >= min_pts;
      const bool active_by_persistence =
        was_active && has_collision_top &&
        obstacle_miss_count_map_[i] <= static_cast<uint16_t>(persistence_misses);
      const bool active = active_by_confidence || active_by_persistence;
      vertical_obstacle_map_[i] = active ? 1 : 0;

      if (!active &&
          obstacle_band_count_map_[i] == 0 &&
          obstacle_miss_count_map_[i] > static_cast<uint16_t>(persistence_misses) &&
          i < collision_top_elevation_map_.size())
      {
        collision_top_elevation_map_[i] = -std::numeric_limits<float>::infinity();
      }
    }
  }

  void updateCollisionTop(int idx, float point_z)
  {
    if (idx < 0 || idx >= static_cast<int>(collision_top_elevation_map_.size()))
      return;

    auto &collision_top = collision_top_elevation_map_[static_cast<size_t>(idx)];
    if (!std::isfinite(collision_top) || point_z > collision_top)
      collision_top = point_z;
  }

  bool computeCellObstacleGeometry(
    int grid_x, int grid_y,
    float &cell_ground,
    float &cell_top,
    float &cell_span,
    float &local_relief) const
  {
    if (grid_x < 0 || grid_x >= global_size_x_ || grid_y < 0 || grid_y >= global_size_y_)
      return false;

    const int idx = grid_y * global_size_x_ + grid_x;
    if (idx < 0 || idx >= static_cast<int>(point_count_map_.size()) || point_count_map_[idx] <= 0)
      return false;
    if (idx >= static_cast<int>(collision_top_elevation_map_.size()) ||
        !std::isfinite(collision_top_elevation_map_[static_cast<size_t>(idx)]))
      return false;

    cell_ground = elevation_map_[idx];
    cell_top = collision_top_elevation_map_[static_cast<size_t>(idx)];
    cell_span = std::max(0.0f, cell_top - cell_ground);

    float neighbor_ground = cell_ground;
    int neighbor_support = 0;
    estimateObstacleReferenceGround(grid_x, grid_y, cell_ground, neighbor_ground, neighbor_support);

    local_relief = std::max(0.0f, cell_top - neighbor_ground);
    return true;
  }

  bool classifyGeometryVerticalObstacle(
    int grid_x, int grid_y,
    float &cell_ground,
    float &cell_top,
    float &cell_span,
    float &local_relief,
    float &ground_gap) const
  {
    if (!computeCellObstacleGeometry(grid_x, grid_y, cell_ground, cell_top, cell_span, local_relief))
      return false;

    const int idx = grid_y * global_size_x_ + grid_x;
    if (idx < 0 || idx >= static_cast<int>(point_count_map_.size()))
      return false;

    const int min_obstacle_support = std::max(1, obstacle_min_points_);
    if (idx >= static_cast<int>(obstacle_band_count_map_.size()) ||
        obstacle_band_count_map_[static_cast<size_t>(idx)] < min_obstacle_support)
      return false;

    float neighbor_ground = cell_ground;
    int neighbor_support = 0;
    estimateObstacleReferenceGround(grid_x, grid_y, cell_ground, neighbor_ground, neighbor_support);

    ground_gap = std::max(0.0f, cell_ground - neighbor_ground);
    local_relief = std::max(local_relief, std::max(0.0f, cell_top - neighbor_ground));

    const float obstacle_height = std::max(cell_span, local_relief);
    const bool height_above_min = obstacle_height > static_cast<float>(obstacle_min_height_);
    if (!height_above_min)
      return false;

    const float min_obstacle_h = static_cast<float>(obstacle_min_height_);
    const bool vertical_stack = cell_span > min_obstacle_h * 1.20f;
    const bool elevated_base =
      neighbor_support >= 3 &&
      ground_gap > static_cast<float>(std::max(0.12, obstacle_min_height_ * 0.75)) &&
      local_relief > min_obstacle_h * 1.60f;
    const bool pronounced_relief =
      local_relief > min_obstacle_h * (vertical_stack ? 1.00f : 1.60f);

    return pronounced_relief && (vertical_stack || elevated_base);
  }

  void refreshVerticalObstacleMapFromGeometry(bool overwrite_existing)
  {
    for (int gy = 0; gy < global_size_y_; ++gy)
    {
      for (int gx = 0; gx < global_size_x_; ++gx)
      {
        const int idx = gy * global_size_x_ + gx;
        if (idx < 0 || idx >= static_cast<int>(vertical_obstacle_map_.size()))
          continue;

        float cell_ground = 0.0f;
        float cell_top = 0.0f;
        float cell_span = 0.0f;
        float local_relief = 0.0f;
        float ground_gap = 0.0f;
        const bool geometry_obstacle = classifyGeometryVerticalObstacle(
          gx, gy, cell_ground, cell_top, cell_span, local_relief, ground_gap);
        (void)ground_gap;

        if (overwrite_existing)
          vertical_obstacle_map_[static_cast<size_t>(idx)] = geometry_obstacle ? 1 : 0;
        else if (geometry_obstacle)
          vertical_obstacle_map_[static_cast<size_t>(idx)] = 1;
      }
    }
  }

  bool computeTraversabilityMetrics(
    float world_x, float world_y,
    float &elev, float &risk,
    float &zbx, float &zby,
    float &slope, float &curvature,
    float &step_height,
    bool &hard_obstacle) const
  {
    float var = static_cast<float>(max_var_);
    float elev_l = 0.0f, elev_r = 0.0f, elev_d = 0.0f, elev_u = 0.0f;
    float var_l = var, var_r = var, var_d = var, var_u = var;
    const bool valid_c = sampleAtWorld(world_x, world_y, elev, var);
    const bool valid_l = sampleAtWorld(world_x - static_cast<float>(resolution_), world_y, elev_l, var_l);
    const bool valid_r = sampleAtWorld(world_x + static_cast<float>(resolution_), world_y, elev_r, var_r);
    const bool valid_d = sampleAtWorld(world_x, world_y - static_cast<float>(resolution_), elev_d, var_d);
    const bool valid_u = sampleAtWorld(world_x, world_y + static_cast<float>(resolution_), elev_u, var_u);
    if (!(valid_c && valid_l && valid_r && valid_d && valid_u))
    {
      risk = 0.0f;
      zbx = 0.0f;
      zby = 0.0f;
      slope = 0.0f;
      curvature = 0.0f;
      step_height = 0.0f;
      hard_obstacle = false;
      return false;
    }

    const int min_support = std::max(1, min_points_for_traversability_);
    int gx_c = 0, gy_c = 0;
    int gx_l = 0, gy_l = 0;
    int gx_r = 0, gy_r = 0;
    int gx_d = 0, gy_d = 0;
    int gx_u = 0, gy_u = 0;
    if (!(worldToMapIndex(world_x, world_y, gx_c, gy_c) &&
          worldToMapIndex(world_x - static_cast<float>(resolution_), world_y, gx_l, gy_l) &&
          worldToMapIndex(world_x + static_cast<float>(resolution_), world_y, gx_r, gy_r) &&
          worldToMapIndex(world_x, world_y - static_cast<float>(resolution_), gx_d, gy_d) &&
          worldToMapIndex(world_x, world_y + static_cast<float>(resolution_), gx_u, gy_u)))
    {
      risk = 0.0f;
      zbx = 0.0f;
      zby = 0.0f;
      slope = 0.0f;
      curvature = 0.0f;
      step_height = 0.0f;
      hard_obstacle = false;
      return false;
    }

    if (!(cellHasSupport(gx_c, gy_c, min_support) &&
          cellHasSupport(gx_l, gy_l, min_support) &&
          cellHasSupport(gx_r, gy_r, min_support) &&
          cellHasSupport(gx_d, gy_d, min_support) &&
          cellHasSupport(gx_u, gy_u, min_support)))
    {
      risk = 0.0f;
      zbx = 0.0f;
      zby = 0.0f;
      slope = 0.0f;
      curvature = 0.0f;
      step_height = 0.0f;
      hard_obstacle = false;
      return false;
    }

    const float res = std::max(1e-3f, static_cast<float>(resolution_));
    const float grad_x = (elev_r - elev_l) / (2.0f * res);
    const float grad_y = (elev_u - elev_d) / (2.0f * res);
    slope = std::sqrt(grad_x * grad_x + grad_y * grad_y);

    const float hess_xx = (elev_r - 2.0f * elev + elev_l) / (res * res);
    const float hess_yy = (elev_u - 2.0f * elev + elev_d) / (res * res);
    curvature = std::sqrt(hess_xx * hess_xx + hess_yy * hess_yy);

    step_height = std::max(
      std::max(std::fabs(elev - elev_l), std::fabs(elev - elev_r)),
      std::max(std::fabs(elev - elev_d), std::fabs(elev - elev_u)));

    const float norm = std::sqrt(1.0f + grad_x * grad_x + grad_y * grad_y);
    zbx = (norm > 1e-6f) ? (-grad_x / norm) : 0.0f;
    zby = (norm > 1e-6f) ? (-grad_y / norm) : 0.0f;

    const float slope_soft = std::max(1e-3f, static_cast<float>(sdf_obstacle_slope_));
    const float curv_soft = std::max(1e-3f, static_cast<float>(sdf_obstacle_curvature_));
    const float step_soft = std::max(1e-3f, static_cast<float>(step_height_soft_));
    const float slope_free = std::clamp(
      static_cast<float>(traversable_slope_free_), 0.0f, std::max(0.0f, slope_soft - 1e-3f));
    const float curv_free = std::clamp(
      static_cast<float>(traversable_curvature_free_), 0.0f, std::max(0.0f, curv_soft - 1e-3f));
    const float step_free = std::clamp(
      static_cast<float>(traversable_step_free_), 0.0f, std::max(0.0f, step_soft - 1e-3f));

    auto softScore = [](float value, float free_value, float soft_value) {
      const float denom = std::max(1e-3f, soft_value - free_value);
      return std::clamp((value - free_value) / denom, 0.0f, 1.0f);
    };
    const float gamma = std::clamp(static_cast<float>(traversability_risk_gamma_), 0.6f, 4.0f);
    const float slope_score = std::pow(softScore(slope, slope_free, slope_soft), gamma);
    const float curv_score = std::pow(softScore(curvature, curv_free, curv_soft), gamma);
    const float step_score = std::pow(softScore(step_height, step_free, step_soft), gamma);

    const float w_slope = std::max(0.0f, static_cast<float>(risk_weight_slope_));
    const float w_curv = std::max(0.0f, static_cast<float>(risk_weight_curvature_));
    const float w_step = std::max(0.0f, static_cast<float>(risk_weight_step_height_));
    const float w_sum = std::max(1e-3f, w_slope + w_curv + w_step);

    risk = std::clamp((w_slope * slope_score + w_curv * curv_score + w_step * step_score) / w_sum, 0.0f, 1.0f);

    const float hard_scale = std::max(1.0f, static_cast<float>(terrain_hardness_scale_));
    const float slope_hard = std::max(slope_soft, static_cast<float>(sdf_hard_obstacle_slope_)) * hard_scale;
    const float curv_hard = std::max(curv_soft, static_cast<float>(sdf_hard_obstacle_curvature_)) * hard_scale;
    const float step_hard = std::max(step_soft, static_cast<float>(step_height_hard_)) * hard_scale;
    hard_obstacle = (slope >= slope_hard) || (curvature >= curv_hard) || (step_height >= step_hard);
    return true;
  }

  void publishObstacleCloud(const rclcpp::Time &stamp)
  {
    pcl::PointCloud<pcl::PointXYZRGB> pcl_cloud;
    pcl_cloud.is_dense = false;

    float center_x = has_odom_ ? static_cast<float>(latest_pose_.position.x) : 0.0f;
    float center_y = has_odom_ ? static_cast<float>(latest_pose_.position.y) : 0.0f;
    const int viz_x = std::max(2, std::min(viz_grid_size_x_, grid_size_x_));
    const int viz_y = std::max(2, std::min(viz_grid_size_y_, grid_size_y_));
    float grid_half_x = viz_x * resolution_ / 2.0f;
    float grid_half_y = viz_y * resolution_ / 2.0f;

    pcl_cloud.points.reserve(static_cast<size_t>(viz_x * viz_y / 3));
    size_t seed_points = 0;
    size_t hard_points = 0;

    for (int row = 0; row < viz_y; ++row)
    {
      for (int col = 0; col < viz_x; ++col)
      {
        float world_x = center_x - grid_half_x + (col + 0.5f) * resolution_;
        float world_y = center_y - grid_half_y + (row + 0.5f) * resolution_;

        int gx = 0, gy = 0;
        if (!worldToMapIndex(world_x, world_y, gx, gy))
          continue;

        const int idx = gy * global_size_x_ + gx;
        const bool seed =
          (idx >= 0 && idx < static_cast<int>(vertical_seed_map_cache_.size()) &&
           vertical_seed_map_cache_[static_cast<size_t>(idx)] != 0);
        const bool hard =
          (idx >= 0 && idx < static_cast<int>(hard_obstacle_map_.size()) &&
           hard_obstacle_map_[static_cast<size_t>(idx)] != 0);
        if (!seed && !hard)
          continue;

        float elev = 0.0f;
        float var = static_cast<float>(max_var_);
        const bool valid = sampleAtWorld(world_x, world_y, elev, var);

        pcl::PointXYZRGB pt;
        pt.x = world_x;
        pt.y = world_y;
        pt.z = (valid ? elev : (has_odom_ ? static_cast<float>(latest_pose_.position.z) : 0.0f)) + 0.06f;

        if (seed)
        {
          ++seed_points;
          pt.r = 255;
          pt.g = 30;
          pt.b = 30;
        }
        else
        {
          ++hard_points;
          pt.r = 255;
          pt.g = 165;
          pt.b = 30;
        }

        pcl_cloud.points.push_back(pt);
      }
    }

    pcl_cloud.width = static_cast<uint32_t>(pcl_cloud.points.size());
    pcl_cloud.height = 1;

    sensor_msgs::msg::PointCloud2 ros_cloud;
    pcl::toROSMsg(pcl_cloud, ros_cloud);
    ros_cloud.header.stamp = stamp;
    ros_cloud.header.frame_id = "map";
    obstacle_cloud_pub_->publish(ros_cloud);

    RCLCPP_INFO_THROTTLE(
      this->get_logger(), *this->get_clock(), 2000,
      "Obstacle cloud: pts=%zu seed=%zu hard_only=%zu",
      pcl_cloud.points.size(), seed_points, hard_points);
  }

  void publishOverheadCloud(const rclcpp::Time &stamp)
  {
    if (!publish_overhead_cloud_ || overhead_cloud_pub_ == nullptr)
      return;

    pcl::PointCloud<pcl::PointXYZRGB> pcl_cloud;
    pcl_cloud.is_dense = false;

    const float center_x = has_odom_ ? static_cast<float>(latest_pose_.position.x) : map_center_x_;
    const float center_y = has_odom_ ? static_cast<float>(latest_pose_.position.y) : map_center_y_;
    const int viz_x = std::max(2, std::min(viz_grid_size_x_, grid_size_x_));
    const int viz_y = std::max(2, std::min(viz_grid_size_y_, grid_size_y_));
    const float grid_half_x = viz_x * resolution_ / 2.0f;
    const float grid_half_y = viz_y * resolution_ / 2.0f;
    const float overhead_clearance = static_cast<float>(
      std::max(obstacle_max_height_ + std::max(0.0, overhead_min_clearance_m_),
               obstacle_min_height_ + 0.20));

    pcl_cloud.points.reserve(
      std::min(overhead_points_cache_.size(), static_cast<size_t>(std::max(1000, overhead_cloud_max_points_))));
    const float min_x = center_x - grid_half_x;
    const float max_x = center_x + grid_half_x;
    const float min_y = center_y - grid_half_y;
    const float max_y = center_y + grid_half_y;
    for (const auto &cached_pt : overhead_points_cache_)
    {
      if (cached_pt.x < min_x || cached_pt.x > max_x ||
          cached_pt.y < min_y || cached_pt.y > max_y)
        continue;
      pcl_cloud.points.push_back(cached_pt);
    }

    // Fallback/top sample for cells whose canopy point was seen before the
    // ground estimate became available. This keeps high objects visible in the
    // air without feeding them back into the ground SDF/risk layers.
    for (int row = 0; row < viz_y; ++row)
    {
      for (int col = 0; col < viz_x; ++col)
      {
        const float world_x = center_x - grid_half_x + (col + 0.5f) * resolution_;
        const float world_y = center_y - grid_half_y + (row + 0.5f) * resolution_;
        int gx = 0, gy = 0;
        if (!worldToMapIndex(world_x, world_y, gx, gy))
          continue;

        const int idx = gy * global_size_x_ + gx;
        if (idx < 0 || idx >= static_cast<int>(top_elevation_map_.size()) ||
            idx >= static_cast<int>(elevation_map_.size()) ||
            idx >= static_cast<int>(point_count_map_.size()) ||
            point_count_map_[static_cast<size_t>(idx)] <= 0)
          continue;

        const float ground = elevation_map_[static_cast<size_t>(idx)];
        const float top = top_elevation_map_[static_cast<size_t>(idx)];
        if (!std::isfinite(ground) || !std::isfinite(top))
          continue;
        if (top <= ground + overhead_clearance)
          continue;

        pcl::PointXYZRGB pt;
        pt.x = world_x;
        pt.y = world_y;
        pt.z = top;
        pt.r = 80;
        pt.g = 255;
        pt.b = 170;
        pcl_cloud.points.push_back(pt);
      }
    }

    pcl_cloud.width = static_cast<uint32_t>(pcl_cloud.points.size());
    pcl_cloud.height = 1;

    sensor_msgs::msg::PointCloud2 ros_cloud;
    pcl::toROSMsg(pcl_cloud, ros_cloud);
    ros_cloud.header.stamp = stamp;
    ros_cloud.header.frame_id = "map";
    overhead_cloud_pub_->publish(ros_cloud);
  }

  void configurePlannerGrids(int size_x, int size_y)
  {
    const double length_x = size_x * resolution_;
    const double length_y = size_y * resolution_;
    fused_map_grid_ = se2_grid::SE2Grid();
    const double yaw_res = std::max(0.1, yaw_resolution_);
    fused_map_grid_.initGeometry(
      Eigen::Array2d(length_x, length_y), resolution_, yaw_res,
      Eigen::Vector2d(map_center_x_, map_center_y_));

    fused_map_grid_.add("elevation", false, 0.0);
    fused_map_grid_.add("var", false, static_cast<float>(max_var_));
    fused_map_grid_.add("risk", true, 0.0);
    fused_map_grid_.add("valid", true, 0.0);
    fused_map_grid_.add("zbx", true, 0.0);
    fused_map_grid_.add("zby", true, 0.0);
    // sdf_map_grid_ removed — planner uses terrain risk directly
  }

  void jetColor(float value, uint8_t &r, uint8_t &g, uint8_t &b) const
  {
    value = std::clamp(value, 0.0f, 1.0f);
    const float four = 4.0f * value;
    const float rf = std::clamp(std::min(four - 1.5f, -four + 4.5f), 0.0f, 1.0f);
    const float gf = std::clamp(std::min(four - 0.5f, -four + 3.5f), 0.0f, 1.0f);
    const float bf = std::clamp(std::min(four + 0.5f, -four + 2.5f), 0.0f, 1.0f);
    r = static_cast<uint8_t>(255.0f * rf);
    g = static_cast<uint8_t>(255.0f * gf);
    b = static_cast<uint8_t>(255.0f * bf);
  }

  void riskDisplayColor(float value, bool hard, uint8_t &r, uint8_t &g, uint8_t &b) const
  {
    if (hard)
    {
      r = 255;
      g = 78;
      b = 58;
      return;
    }

    const float t = std::clamp(value, 0.0f, 1.0f);
    const float s = std::pow(t, 0.45f);
    const float safe_r = 82.0f;
    const float safe_g = 186.0f;
    const float safe_b = 255.0f;
    const float halo_r = 255.0f;
    const float halo_g = 158.0f;
    const float halo_b = 78.0f;
    const float hot_r = 255.0f;
    const float hot_g = 34.0f;
    const float hot_b = 34.0f;

    if (s < 0.58f)
    {
      const float u = s / 0.58f;
      r = static_cast<uint8_t>(safe_r + u * (halo_r - safe_r));
      g = static_cast<uint8_t>(safe_g + u * (halo_g - safe_g));
      b = static_cast<uint8_t>(safe_b + u * (halo_b - safe_b));
    }
    else
    {
      const float u = (s - 0.58f) / 0.42f;
      r = static_cast<uint8_t>(halo_r + u * (hot_r - halo_r));
      g = static_cast<uint8_t>(halo_g + u * (hot_g - halo_g));
      b = static_cast<uint8_t>(halo_b + u * (hot_b - halo_b));
    }
  }

  // sdfColor removed — planner uses terrain risk directly

  double yawFromPose(const geometry_msgs::msg::Pose &pose) const
  {
    const auto &q = pose.orientation;
    const double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
    const double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
    return std::atan2(siny_cosp, cosy_cosp);
  }

  // Parameters
  int grid_size_x_, grid_size_y_;
  int viz_grid_size_x_, viz_grid_size_y_;
  bool use_pcd_prior_ = false;
  bool cloud_is_local_map_ = false;
  bool local_map_sliding_window_fusion_ = false;
  bool publish_on_cloud_ = false;
  bool use_pcd_vertical_obstacles_ = false;
  bool sdf_use_terrain_hard_obstacles_ = false;
  bool visualize_obstacle_tops_ = true;
  int min_points_for_traversability_ = 2;
  double resolution_;
  double yaw_resolution_ = 0.1;
  double sensor_range_;
  double min_var_, max_var_;
  double sdf_obstacle_slope_ = 0.7;
  double sdf_hard_obstacle_slope_ = 1.1;
  double sdf_obstacle_curvature_ = 0.55;
  double sdf_hard_obstacle_curvature_ = 1.05;
  double step_height_soft_ = 0.10;
  double step_height_hard_ = 0.22;
  double traversable_slope_free_ = 0.35;
  double traversable_curvature_free_ = 0.20;
  double traversable_step_free_ = 0.05;
  double traversability_risk_gamma_ = 1.35;
  double risk_weight_slope_ = 0.55;
  double risk_weight_curvature_ = 0.30;
  double risk_weight_step_height_ = 0.15;
  double risk_hard_threshold_ = 0.92;
  double risk_vis_hard_threshold_ = 0.78;
  double risk_visual_deadband_ = 0.18;
  double risk_visual_gamma_ = 1.8;
  double terrain_risk_scale_ = 0.30;
  bool terrain_risk_to_hard_obstacle_ = true;
  double terrain_hardness_scale_ = 1.0;
  double tree_risk_radius_m_ = 0.85;
  double tree_hard_keepout_radius_m_ = 0.0;
  double tree_risk_power_ = 3.0;
  double non_tree_risk_cap_ = 0.55;
  double obstacle_min_height_ = 0.18;
  double obstacle_max_height_ = 1.40;
  double obstacle_seed_max_height_ = 0.85;
  int obstacle_min_points_ = 4;
  int obstacle_hit_increment_ = 3;
  double obstacle_count_decay_ = 0.85;
  int obstacle_persistence_misses_ = 12;
  double obstacle_ground_search_radius_m_ = 0.35;
  bool obstacle_footprint_filter_enable_ = true;
  double obstacle_footprint_filter_radius_m_ = 0.45;
  int obstacle_footprint_max_neighbors_ = 10;
  int obstacle_footprint_line_width_cells_ = 2;
  double vertical_obstacle_inflation_m_ = 0.30;
  double ego_clear_radius_ = 0.55;
  bool ego_body_filter_enable_ = true;
  double ego_body_filter_front_m_ = 1.05;
  double ego_body_filter_back_m_ = 0.80;
  double ego_body_filter_half_width_m_ = 0.62;
  double ego_body_filter_z_min_m_ = -0.50;
  double ego_body_filter_z_max_m_ = 1.35;
  double map_padding_m_ = 2.0;
  double sdf_visual_max_dist_ = 6.0;
  double sdf_visual_height_scale_ = 0.0;
  bool sdf_visualize_obstacle_columns_ = true;
  double sdf_obstacle_column_step_ = 0.20;
  double sdf_obstacle_column_max_height_ = 4.0;
  bool publish_overhead_cloud_ = true;
  double overhead_min_clearance_m_ = 0.20;
  int overhead_cloud_max_points_ = 80000;
  int cache_size_;
  double publish_rate_hz_ = 10.0;
  double visual_publish_rate_hz_ = 10.0;
  bool publish_internal_visual_clouds_ = true;
  bool elevation_visual_fill_missing_ = true;
  double elevation_visual_fill_radius_m_ = 0.45;

  // Subscriptions and publishers
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Publisher<se2_grid_msgs::msg::SE2Grid>::SharedPtr fused_map_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr elevation_cloud_pub_;
  // sdf_cloud_pub_ removed — planner uses terrain risk directly
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr risk_cloud_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr obstacle_cloud_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr overhead_cloud_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr normal_map_pub_;
  // sdf_map_pub_ removed — planner uses terrain risk directly
  rclcpp::TimerBase::SharedPtr timer_;

  // State
  std::vector<float> elevation_map_;
  std::vector<float> top_elevation_map_;
  std::vector<float> collision_top_elevation_map_;
  std::vector<uint8_t> vertical_obstacle_map_;
  std::vector<uint16_t> obstacle_band_count_map_;
  std::vector<uint16_t> obstacle_miss_count_map_;
  std::vector<uint8_t> prior_vertical_obstacle_map_;
  std::vector<uint16_t> prior_obstacle_band_count_map_;
  std::vector<uint8_t> hard_obstacle_map_;
  std::vector<uint8_t> vertical_seed_map_cache_;
  // sdf_distance_map_ removed — planner uses terrain risk directly
  std::vector<float> risk_map_cache_;
  std::deque<pcl::PointXYZRGB> overhead_points_cache_;
  std::vector<float> variance_map_;
  std::vector<int> point_count_map_;
  float global_origin_x_ = 0.0f;
  float global_origin_y_ = 0.0f;
  int global_size_x_ = 0;
  int global_size_y_ = 0;
  float map_center_x_ = 0.0f;
  float map_center_y_ = 0.0f;
  geometry_msgs::msg::Pose latest_pose_;
  bool has_odom_ = false;
  rclcpp::Time latest_stamp_;
  rclcpp::Time last_update_time_;
  double last_visual_publish_sec_ = -1.0;

  se2_grid::SE2Grid fused_map_grid_;
  // sdf_map_grid_ removed — planner uses terrain risk directly
};

int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<TerrainAnalyzerNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
