#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <cmath>

class TerrainColoringNode : public rclcpp::Node {
public:
    TerrainColoringNode() : Node("terrain_coloring_node") {
        // Subscribe to terrain cloud
        cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/terrain_cloud", 10,
            std::bind(&TerrainColoringNode::cloudCallback, this, std::placeholders::_1));

        // Publish colored terrain cloud
        cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/terrain_cloud_colored", 10);

        RCLCPP_INFO(this->get_logger(), "Terrain Coloring Node initialized");
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;

    void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        // Convert ROS message to PCL
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*msg, *cloud);

        // Create colored version
        sensor_msgs::msg::PointCloud2 colored_msg = *msg;
        
        // Get min/max height for color mapping
        float min_z = 1e6, max_z = -1e6;
        for (const auto& pt : cloud->points) {
            if (pt.z < min_z) min_z = pt.z;
            if (pt.z > max_z) max_z = pt.z;
        }

        float z_range = max_z - min_z;
        if (z_range < 0.01) z_range = 0.01; // Avoid division by zero

        // Add RGB field if not present
        if (colored_msg.fields.size() == 3) { // x, y, z
            sensor_msgs::PointCloud2Modifier modifier(colored_msg);
            modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");
        }

        // Color points by height (rainbow: blue=low, red=high)
        sensor_msgs::PointCloud2Iterator<float> iter_r(colored_msg, "r");
        sensor_msgs::PointCloud2Iterator<float> iter_g(colored_msg, "g");
        sensor_msgs::PointCloud2Iterator<float> iter_b(colored_msg, "b");
        sensor_msgs::PointCloud2Iterator<float> iter_x(colored_msg, "x");
        sensor_msgs::PointCloud2Iterator<float> iter_y(colored_msg, "y");
        sensor_msgs::PointCloud2Iterator<float> iter_z(colored_msg, "z");

        for (size_t i = 0; i < cloud->points.size(); ++i) {
            // Normalize height to [0, 1]
            float normalized_z = (cloud->points[i].z - min_z) / z_range;
            
            // Rainbow color mapping
            float hue = normalized_z * 240.0; // 0° (red) to 240° (blue)
            float r, g, b;
            hsvToRgb(hue, 1.0f, 1.0f, r, g, b);

            *iter_r = r * 255.0f;
            *iter_g = g * 255.0f;
            *iter_b = b * 255.0f;
            *iter_x = cloud->points[i].x;
            *iter_y = cloud->points[i].y;
            *iter_z = cloud->points[i].z;

            ++iter_r; ++iter_g; ++iter_b;
            ++iter_x; ++iter_y; ++iter_z;
        }

        cloud_pub_->publish(colored_msg);
    }

    void hsvToRgb(float h, float s, float v, float& r, float& g, float& b) {
        float c = v * s;
        float x = c * (1.0f - fabs(fmod(h / 60.0f, 2.0f) - 1.0f));
        float m = v - c;

        if (h < 60.0f) {
            r = c; g = x; b = 0.0f;
        } else if (h < 120.0f) {
            r = x; g = c; b = 0.0f;
        } else if (h < 180.0f) {
            r = 0.0f; g = c; b = x;
        } else if (h < 240.0f) {
            r = 0.0f; g = x; b = c;
        } else if (h < 300.0f) {
            r = x; g = 0.0f; b = c;
        } else {
            r = c; g = 0.0f; b = x;
        }

        r += m; g += m; b += m;
    }
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<TerrainColoringNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
