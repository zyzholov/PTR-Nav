#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <se2_grid_msgs/msg/se2_grid.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <Eigen/Eigen>
#include <algorithm>
#include <cmath>
#include "se2_grid_core/SE2Grid.hpp"

class SE2GridToPointCloudConverter : public rclcpp::Node
{
public:
  SE2GridToPointCloudConverter()
    : Node("se2_grid_to_pointcloud_converter")
  {
    // Declare parameters
    this->declare_parameter("input_topic", std::string("/fused_map"));
    this->declare_parameter("output_topic", std::string("/elevation_cloud"));
    this->declare_parameter("layer_name", std::string("inpainted"));
    this->declare_parameter("valid_layer_name", std::string(""));
    this->declare_parameter("valid_threshold", 0.5);
    this->declare_parameter("yaw_index", 0);
    this->declare_parameter("height_as_z", true);
    this->declare_parameter("z_value", 0.0);
    this->declare_parameter("z_offset", 0.0);
    this->declare_parameter("use_color_map", true);
    this->declare_parameter("invert_color_map", false);
    this->declare_parameter("min_height", -5.0);
    this->declare_parameter("max_height", 5.0);
    this->declare_parameter("colormap", std::string("jet"));

    // Get parameters
    std::string input_topic = this->get_parameter("input_topic").as_string();
    std::string output_topic = this->get_parameter("output_topic").as_string();
    layer_name_ = this->get_parameter("layer_name").as_string();
    valid_layer_name_ = this->get_parameter("valid_layer_name").as_string();
    valid_threshold_ = this->get_parameter("valid_threshold").as_double();
    yaw_index_ = std::max(0, static_cast<int>(this->get_parameter("yaw_index").as_int()));
    height_as_z_ = this->get_parameter("height_as_z").as_bool();
    z_value_ = this->get_parameter("z_value").as_double();
    z_offset_ = this->get_parameter("z_offset").as_double();
    use_color_map_ = this->get_parameter("use_color_map").as_bool();
    invert_color_map_ = this->get_parameter("invert_color_map").as_bool();
    min_height_ = this->get_parameter("min_height").as_double();
    max_height_ = this->get_parameter("max_height").as_double();
    colormap_ = this->get_parameter("colormap").as_string();

    // Create publisher
    cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      output_topic, rclcpp::QoS(rclcpp::KeepLast(10)));

    // Create subscription
    grid_sub_ = this->create_subscription<se2_grid_msgs::msg::SE2Grid>(
      input_topic,
      rclcpp::QoS(rclcpp::KeepLast(5)),
      std::bind(&SE2GridToPointCloudConverter::gridCallback, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), 
      "SE2Grid to PointCloud2 converter initialized. Input: %s -> Output: %s (Layer: %s)",
      input_topic.c_str(), output_topic.c_str(), layer_name_.c_str());
  }

private:
  void gridCallback(const se2_grid_msgs::msg::SE2Grid::SharedPtr msg)
  {
    try {
      // Convert SE2Grid to PointCloud2
      auto cloud = convertSE2GridToPointCloud(msg);
      cloud_pub_->publish(cloud);
    } catch (const std::exception& e) {
      RCLCPP_ERROR(this->get_logger(), "Conversion error: %s", e.what());
    }
  }

  sensor_msgs::msg::PointCloud2 convertSE2GridToPointCloud(
    const se2_grid_msgs::msg::SE2Grid::SharedPtr& grid_msg)
  {
    sensor_msgs::msg::PointCloud2 cloud_msg;
    
    if (grid_msg->layers.empty() || grid_msg->data.empty()) {
      RCLCPP_WARN(this->get_logger(), "Received empty SE2Grid");
      return cloud_msg;
    }

    // Find the layer to extract (e.g., "inpainted" or "elevation")
    const int yaw_slices = std::max(
      1, static_cast<int>(std::ceil(grid_msg->info.length_yaw /
                                    std::max(1e-6, grid_msg->info.yaw_resolution))));

    auto resolveLayer = [&](const std::string &name, int &layer_idx, size_t &target_data_idx) -> bool {
      layer_idx = -1;
      target_data_idx = 0;
      size_t data_idx = 0;
      for (size_t i = 0; i < grid_msg->layers.size(); ++i) {
        if (grid_msg->layers[i] == name) {
          layer_idx = static_cast<int>(i);
          const bool has_so2 = (i < grid_msg->has_so2.size()) ? grid_msg->has_so2[i] : false;
          const size_t slice_count = has_so2 ? static_cast<size_t>(yaw_slices) : 1u;
          target_data_idx =
            data_idx + static_cast<size_t>(std::min<int>(yaw_index_, static_cast<int>(slice_count - 1)));
          return true;
        }
        const bool has_so2 = (i < grid_msg->has_so2.size()) ? grid_msg->has_so2[i] : false;
        data_idx += has_so2 ? static_cast<size_t>(yaw_slices) : 1u;
      }
      return false;
    };

    int layer_idx = -1;
    size_t target_data_idx = 0;
    resolveLayer(layer_name_, layer_idx, target_data_idx);

    // If not found, use first available layer/data block
    if (layer_idx < 0) {
      layer_idx = 0;
      target_data_idx = 0;
      RCLCPP_WARN(this->get_logger(), 
        "Layer '%s' not found, using first layer: %s",
        layer_name_.c_str(), grid_msg->layers[0].c_str());
    }

    // Extract layer data
    if (target_data_idx >= grid_msg->data.size()) {
      RCLCPP_WARN(this->get_logger(),
        "Layer '%s' resolved to data index %zu, but message only has %zu blocks",
        grid_msg->layers[layer_idx].c_str(), target_data_idx, grid_msg->data.size());
      return cloud_msg;
    }

    const auto& layer_data = grid_msg->data[target_data_idx];
    if (layer_data.data.empty()) {
      RCLCPP_WARN(this->get_logger(), "Layer data is empty");
      return cloud_msg;
    }

    // Get grid info
    const auto& info = grid_msg->info;
    double pos_resolution = info.pos_resolution;
    double length_x = info.length_x;
    double length_y = info.length_y;

    // Calculate grid dimensions (assuming square grid)
    int size_col = static_cast<int>(std::round(length_x / pos_resolution));
    int size_row = static_cast<int>(std::round(length_y / pos_resolution));

    // Create PCL point cloud with RGB
    pcl::PointCloud<pcl::PointXYZRGB> pcl_cloud;
    pcl_cloud.header.frame_id = info.header.frame_id;
    pcl_cloud.height = 1;
    pcl_cloud.is_dense = true;

    // Convert float array to points
    const auto& height_data = layer_data.data;
    size_t num_floats = height_data.size();
    const std::vector<float> *valid_data = nullptr;
    size_t valid_num_floats = 0;

    if (!valid_layer_name_.empty()) {
      int valid_layer_idx = -1;
      size_t valid_data_idx = 0;
      if (resolveLayer(valid_layer_name_, valid_layer_idx, valid_data_idx) &&
          valid_data_idx < grid_msg->data.size())
      {
        valid_data = &grid_msg->data[valid_data_idx].data;
        valid_num_floats = valid_data->size();
      }
      else
      {
        RCLCPP_WARN_THROTTLE(
          this->get_logger(), *this->get_clock(), 2000,
          "Valid layer '%s' not found in SE2Grid, disabling valid-mask filter",
          valid_layer_name_.c_str());
      }
    }

    if (num_floats != (size_t)size_row * size_col) {
      RCLCPP_WARN(this->get_logger(),
        "Data size mismatch: expected %d x %d = %d, got %zu",
        size_row, size_col, size_row * size_col, num_floats);
    }

    // Position offset (for grid centering based on pose)
    double center_x = info.pose.position.x;
    double center_y = info.pose.position.y;
    double start_x = center_x - length_x / 2.0 + grid_msg->start_index_col * pos_resolution;
    double start_y = center_y - length_y / 2.0 + grid_msg->start_index_row * pos_resolution;

    // Iterate through grid and create points
    for (int i = 0; i < size_row && (size_t)i * size_col < num_floats; ++i) {
      for (int j = 0; j < size_col && (size_t)(i * size_col + j) < num_floats; ++j) {
        float height = height_data[static_cast<size_t>(i * size_col + j)];
        if (valid_data != nullptr) {
          const size_t idx = static_cast<size_t>(i * size_col + j);
          if (idx >= valid_num_floats)
            continue;
          const float valid_value = (*valid_data)[idx];
          if (!std::isfinite(valid_value) || valid_value < static_cast<float>(valid_threshold_))
            continue;
        }
        
        // Skip invalid/NaN heights
        if (!std::isfinite(height)) {
          continue;
        }

        pcl::PointXYZRGB pt;
        pt.x = start_x + j * pos_resolution;
        pt.y = start_y + i * pos_resolution;
        pt.z = static_cast<float>(height_as_z_ ? (height + z_offset_) : z_value_);

        // Color based on height
        if (use_color_map_) {
          uint8_t r, g, b;
          const float color_value = invert_color_map_
            ? static_cast<float>(min_height_ + max_height_ - height)
            : height;
          if (colormap_ == "terrain") {
            terrainToRGB(color_value, min_height_, max_height_, r, g, b);
          } else if (colormap_ == "risk" || colormap_ == "risk_gray_red") {
            riskToRGB(color_value, min_height_, max_height_, r, g, b);
          } else {
            heightToRGB(color_value, min_height_, max_height_, r, g, b);
          }
          pt.r = r;
          pt.g = g;
          pt.b = b;
        } else {
          pt.r = 200;
          pt.g = 200;
          pt.b = 200;
        }
        pt.a = 255;

        pcl_cloud.push_back(pt);
      }
    }

    pcl_cloud.width = pcl_cloud.points.size();

    // Convert PCL cloud to ROS message
    pcl::toROSMsg(pcl_cloud, cloud_msg);
    cloud_msg.header.stamp = info.header.stamp;
    cloud_msg.header.frame_id = info.header.frame_id;

    RCLCPP_DEBUG(this->get_logger(),
      "Converted SE2Grid to PointCloud with %zu points from layer %d (%s)",
      pcl_cloud.points.size(), layer_idx, grid_msg->layers[layer_idx].c_str());

    return cloud_msg;
  }

  void heightToRGB(float height, float min_h, float max_h,
                   uint8_t& r, uint8_t& g, uint8_t& b)
  {
    // Normalize height to [0, 1]
    float normalized = (height - min_h) / (max_h - min_h);
    normalized = std::max(0.0f, std::min(1.0f, normalized));

    // Blue (0) -> Cyan -> Green -> Yellow -> Red (1)
    if (normalized < 0.25f) {
      // Blue to Cyan
      r = 0;
      g = (uint8_t)(255 * (normalized / 0.25f));
      b = 255;
    } else if (normalized < 0.5f) {
      // Cyan to Green
      r = 0;
      g = 255;
      b = (uint8_t)(255 * (1.0f - (normalized - 0.25f) / 0.25f));
    } else if (normalized < 0.75f) {
      // Green to Yellow
      r = (uint8_t)(255 * ((normalized - 0.5f) / 0.25f));
      g = 255;
      b = 0;
    } else {
      // Yellow to Red
      r = 255;
      g = (uint8_t)(255 * (1.0f - (normalized - 0.75f) / 0.25f));
      b = 0;
    }
  }

  // Terrain colormap: bright green (safe) -> dark olive -> brown -> magenta -> deep purple (obstacle)
  void terrainToRGB(float value, float min_v, float max_v,
                    uint8_t& r, uint8_t& g, uint8_t& b)
  {
    float t = (value - min_v) / (max_v - min_v);
    t = std::max(0.0f, std::min(1.0f, t));

    // 5-stop gradient:
    // 0.00: bright green  (80, 255, 100)
    // 0.25: dark olive     (60, 120,  40)
    // 0.50: dark brown     (90,  50,  30)
    // 0.75: magenta        (200,  50, 160)
    // 1.00: deep purple    (140,  30, 220)
    struct Stop { float t; float cr, cg, cb; };
    static const Stop stops[] = {
      {0.00f,  80, 255, 100},
      {0.25f,  60, 120,  40},
      {0.50f,  90,  50,  30},
      {0.75f, 200,  50, 160},
      {1.00f, 140,  30, 220},
    };
    const int n_stops = 5;

    // Find the two bounding stops
    int lo = 0;
    for (int i = 0; i < n_stops - 1; ++i) {
      if (t >= stops[i].t && t <= stops[i + 1].t) { lo = i; break; }
      if (i == n_stops - 2) lo = i;
    }
    const Stop& a = stops[lo];
    const Stop& bb = stops[lo + 1];
    float seg_t = (bb.t > a.t) ? (t - a.t) / (bb.t - a.t) : 0.0f;
    seg_t = std::max(0.0f, std::min(1.0f, seg_t));

    r = static_cast<uint8_t>(a.cr + seg_t * (bb.cr - a.cr));
    g = static_cast<uint8_t>(a.cg + seg_t * (bb.cg - a.cg));
    b = static_cast<uint8_t>(a.cb + seg_t * (bb.cb - a.cb));
  }

  void riskToRGB(float value, float min_v, float max_v,
                 uint8_t& r, uint8_t& g, uint8_t& b)
  {
    float t = (value - min_v) / std::max(1e-6f, max_v - min_v);
    t = std::max(0.0f, std::min(1.0f, t));

    if (t >= 0.82f) {
      r = 255;
      g = 78;
      b = 58;
      return;
    }

    const float deadband = 0.30f;
    const float normalized = std::max(0.0f, (t - deadband) / std::max(1e-6f, 0.82f - deadband));
    const float s = std::pow(normalized, 0.45f);
    const float safe_r = 82.0f;
    const float safe_g = 186.0f;
    const float safe_b = 255.0f;
    const float halo_r = 255.0f;
    const float halo_g = 158.0f;
    const float halo_b = 78.0f;
    const float hot_r = 255.0f;
    const float hot_g = 34.0f;
    const float hot_b = 34.0f;

    if (s < 0.58f) {
      const float u = s / 0.58f;
      r = static_cast<uint8_t>(safe_r + u * (halo_r - safe_r));
      g = static_cast<uint8_t>(safe_g + u * (halo_g - safe_g));
      b = static_cast<uint8_t>(safe_b + u * (halo_b - safe_b));
    } else {
      const float u = (s - 0.58f) / 0.42f;
      r = static_cast<uint8_t>(halo_r + u * (hot_r - halo_r));
      g = static_cast<uint8_t>(halo_g + u * (hot_g - halo_g));
      b = static_cast<uint8_t>(halo_b + u * (hot_b - halo_b));
    }
  }

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
  rclcpp::Subscription<se2_grid_msgs::msg::SE2Grid>::SharedPtr grid_sub_;
  std::string layer_name_;
  std::string valid_layer_name_;
  double valid_threshold_;
  int yaw_index_;
  bool height_as_z_;
  double z_value_;
  double z_offset_;
  bool use_color_map_;
  bool invert_color_map_;
  double min_height_;
  double max_height_;
  std::string colormap_;
};

int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<SE2GridToPointCloudConverter>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
