#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <cmath>
#include <unordered_set>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>

struct Point3f {
  float x, y, z;
};

class TerrainLoaderNode : public rclcpp::Node
{
public:
  TerrainLoaderNode() : Node("terrain_loader")
  {
    RCLCPP_INFO(this->get_logger(), "Terrain Loader Node Starting (Pure C++, No PCL/Boost)...");

    // Parameters
    declare_parameter("pcd_file", "RockyDesertMesh.pcd");
    declare_parameter("pub_rate", 10.0);
    declare_parameter("primary_topic", "/global_cloud");
    declare_parameter("compat_topic", "terrain_cloud");
    declare_parameter("legacy_topic", "/map_generator/global_cloud");

    std::string pcd_file = get_parameter("pcd_file").as_string();
    double pub_rate = get_parameter("pub_rate").as_double();
    std::string primary_topic = get_parameter("primary_topic").as_string();
    std::string compat_topic = get_parameter("compat_topic").as_string();
    std::string legacy_topic = get_parameter("legacy_topic").as_string();

    // Try to load PCD
    terrain_points_ = loadPCDFile(pcd_file);

    if (!terrain_points_.empty())
    {
      RCLCPP_INFO(this->get_logger(),
        "✓ Loaded terrain: %zu points from %s", terrain_points_.size(), pcd_file.c_str());
    }
    else
    {
      RCLCPP_WARN(this->get_logger(), 
        "✗ Failed to load PCD, using synthetic data");
      generateSyntheticTerrain();
    }

    // Publish with transient-local durability so late subscribers (e.g. laser sim)
    // immediately receive the static global map.
    auto qos_profile = rclcpp::QoS(1);
    qos_profile.reliable().transient_local();

    std::unordered_set<std::string> uniq_topics;
    auto add_topic = [&](const std::string &topic) {
      if (topic.empty())
        return;
      if (uniq_topics.insert(topic).second)
      {
        terrain_pubs_.push_back(create_publisher<sensor_msgs::msg::PointCloud2>(topic, qos_profile));
      }
    };
    add_topic(primary_topic);
    add_topic(compat_topic);
    add_topic(legacy_topic);

    // Timer
    timer_ = create_wall_timer(
      std::chrono::milliseconds(static_cast<int>(1000.0 / pub_rate)),
      std::bind(&TerrainLoaderNode::publishTerrain, this));

    RCLCPP_INFO(this->get_logger(),
      "Terrain Loader Ready: Publishing %zu points at %.1f Hz to %zu topic(s)",
      terrain_points_.size(), pub_rate, terrain_pubs_.size());
  }

private:
  std::vector<Point3f> loadPCDFile(const std::string& filename)
  {
    std::vector<Point3f> points;

    // Try multiple paths
    std::vector<std::string> paths = {
      filename,
      "./install/share/fake_ugv/maps/" + filename
    };
    
    // Add path from ament_index (ROS2 way to find packages)
    try {
      std::string fake_ugv_share = ament_index_cpp::get_package_share_directory("fake_ugv");
      paths.push_back(fake_ugv_share + "/maps/" + filename);
      RCLCPP_DEBUG(this->get_logger(), "Added ament path: %s", (fake_ugv_share + "/maps/" + filename).c_str());
    } catch (...) {
      // Package not found, will try other paths
    }

    // Use PCL to load PCD file (supports both ASCII and binary formats)
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    
    for (const auto& path : paths)
    {
      RCLCPP_DEBUG(this->get_logger(), "Trying to load PCD: %s", path.c_str());
      
      if (pcl::io::loadPCDFile<pcl::PointXYZ>(path, *cloud) == 0)
      {
        RCLCPP_INFO(this->get_logger(), "✓ Successfully loaded PCD: %s (%zu points)", 
                   path.c_str(), cloud->size());
        
        // Convert PCL points to our format
        points.resize(cloud->size());
        for (size_t i = 0; i < cloud->size(); ++i)
        {
          points[i].x = cloud->points[i].x;
          points[i].y = cloud->points[i].y;
          points[i].z = cloud->points[i].z;
        }
        return points;
      }
    }

    RCLCPP_WARN(this->get_logger(), "PCD file '%s' not found in any search paths", filename.c_str());
    return points;
  }

  void generateSyntheticTerrain()
  {
    RCLCPP_INFO(this->get_logger(), "Generating synthetic terrain...");

    int grid_size = 100;
    float range = 25.0f;

    for (int i = 0; i < grid_size; ++i)
    {
      for (int j = 0; j < grid_size; ++j)
      {
        float x = (i - grid_size / 2.0f) * (range / (grid_size / 2.0f));
        float y = (j - grid_size / 2.0f) * (range / (grid_size / 2.0f));
        float z = 0.5f * std::sin(x * 0.2f) * std::cos(y * 0.2f) + 0.2f;

        terrain_points_.push_back({x, y, z});
      }
    }

    RCLCPP_INFO(this->get_logger(), "Generated %zu synthetic points", terrain_points_.size());
  }

  void publishTerrain()
  {
    if (terrain_points_.empty())
      return;

    // Create PointCloud2 message manually
    auto msg = std::make_shared<sensor_msgs::msg::PointCloud2>();

    // Header
    msg->header.stamp = now();
    msg->header.frame_id = "map";
    msg->height = 1;
    msg->width = terrain_points_.size();
    msg->is_bigendian = false;
    msg->is_dense = true;

    // Fields (x, y, z as float32)
    sensor_msgs::msg::PointField x_field;
    x_field.name = "x";
    x_field.offset = 0;
    x_field.datatype = sensor_msgs::msg::PointField::FLOAT32;
    x_field.count = 1;

    sensor_msgs::msg::PointField y_field;
    y_field.name = "y";
    y_field.offset = 4;
    y_field.datatype = sensor_msgs::msg::PointField::FLOAT32;
    y_field.count = 1;

    sensor_msgs::msg::PointField z_field;
    z_field.name = "z";
    z_field.offset = 8;
    z_field.datatype = sensor_msgs::msg::PointField::FLOAT32;
    z_field.count = 1;

    msg->fields = {x_field, y_field, z_field};
    msg->point_step = 12;  // 3 * float32
    msg->row_step = msg->point_step * msg->width;

    // Data
    msg->data.resize(msg->row_step);
    std::memcpy(msg->data.data(), terrain_points_.data(),
                terrain_points_.size() * sizeof(Point3f));

    for (const auto &pub : terrain_pubs_)
    {
      if (pub)
        pub->publish(*msg);
    }
  }

  std::vector<Point3f> terrain_points_;
  std::vector<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr> terrain_pubs_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<TerrainLoaderNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
