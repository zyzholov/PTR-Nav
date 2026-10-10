#include <algorithm>
#include <array>
#include <cmath>
#include <string>

#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

namespace
{

struct Color
{
  float r = 1.0F;
  float g = 1.0F;
  float b = 1.0F;
  float a = 1.0F;
};

geometry_msgs::msg::Quaternion normalizedQuaternion(geometry_msgs::msg::Quaternion q)
{
  const double n = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
  if (n < 1.0e-9)
  {
    q.x = 0.0;
    q.y = 0.0;
    q.z = 0.0;
    q.w = 1.0;
    return q;
  }
  q.x /= n;
  q.y /= n;
  q.z /= n;
  q.w /= n;
  return q;
}

geometry_msgs::msg::Point rotateOffset(
  const geometry_msgs::msg::Quaternion & q_in, const double x, const double y, const double z)
{
  const auto q = normalizedQuaternion(q_in);
  const double vx = q.x;
  const double vy = q.y;
  const double vz = q.z;
  const double qw = q.w;

  const double uvx = vy * z - vz * y;
  const double uvy = vz * x - vx * z;
  const double uvz = vx * y - vy * x;

  const double uuvx = vy * uvz - vz * uvy;
  const double uuvy = vz * uvx - vx * uvz;
  const double uuvz = vx * uvy - vy * uvx;

  geometry_msgs::msg::Point out;
  out.x = x + 2.0 * (qw * uvx + uuvx);
  out.y = y + 2.0 * (qw * uvy + uuvy);
  out.z = z + 2.0 * (qw * uvz + uuvz);
  return out;
}

geometry_msgs::msg::Point translatedPoint(
  const geometry_msgs::msg::Pose & base, const double x, const double y, const double z)
{
  const auto offset = rotateOffset(base.orientation, x, y, z);
  geometry_msgs::msg::Point p;
  p.x = base.position.x + offset.x;
  p.y = base.position.y + offset.y;
  p.z = base.position.z + offset.z;
  return p;
}

}  // namespace

class VehicleVisualizerNode : public rclcpp::Node
{
public:
  VehicleVisualizerNode()
  : Node("vehicle_visualizer")
  {
    frame_id_ = declare_parameter<std::string>("frame_id", "map");
    publish_period_sec_ = declare_parameter<double>("publish_period_sec", 0.05);
    lifetime_sec_ = declare_parameter<double>("marker_lifetime_sec", 0.0);

    length_ = declare_parameter<double>("vehicle.length", 0.840);
    width_ = declare_parameter<double>("vehicle.width", 0.626);
    height_ = declare_parameter<double>("vehicle.height", 0.338);
    mesh_resource_ = declare_parameter<std::string>("mesh.resource", "package://neos_description/meshes/car.dae");
    mesh_use_embedded_materials_ = declare_parameter<bool>("mesh.use_embedded_materials", true);
    mesh_alpha_ = declare_parameter<double>("mesh.alpha", 0.96);
    mesh_native_length_ = declare_parameter<double>("mesh.native_length", 1.0086837);
    mesh_native_width_ = declare_parameter<double>("mesh.native_width", 0.7673552);
    mesh_native_height_ = declare_parameter<double>("mesh.native_height", 0.3848279);
    mesh_scale_x_ = declare_parameter<double>("mesh.scale_x", length_ / mesh_native_length_);
    mesh_scale_y_ = declare_parameter<double>("mesh.scale_y", width_ / mesh_native_width_);
    mesh_scale_z_ = declare_parameter<double>("mesh.scale_z", height_ / mesh_native_height_);
    mesh_offset_x_ = declare_parameter<double>("mesh.offset_x", 0.0);
    mesh_offset_y_ = declare_parameter<double>("mesh.offset_y", 0.0);
    mesh_offset_z_ = declare_parameter<double>("mesh.offset_z", 0.0);

    wheel_base_ = declare_parameter<double>("vehicle.wheel_base", 0.430);
    track_width_ = declare_parameter<double>("vehicle.track_width", 0.548);

    base_to_chassis_center_x_ = declare_parameter<double>("base_to_chassis_center.x", 0.0);
    base_to_chassis_center_y_ = declare_parameter<double>("base_to_chassis_center.y", 0.0);
    base_to_chassis_center_z_ = declare_parameter<double>("base_to_chassis_center.z", 0.0);
    wheel_center_z_ = declare_parameter<double>("wheel_center_z", -0.04);
    footprint_z_ = declare_parameter<double>("footprint_z", -0.16);
    heading_arrow_z_ = declare_parameter<double>("heading_arrow_z", 0.24);
    show_heading_arrow_ = declare_parameter<bool>("show_heading_arrow", true);
    show_lidar_hint_ = declare_parameter<bool>("show_lidar_hint", true);

    marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
      "vehicle_model", rclcpp::QoS(1).reliable().transient_local());
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      "odom", rclcpp::SensorDataQoS(),
      std::bind(&VehicleVisualizerNode::odomCallback, this, std::placeholders::_1));

    RCLCPP_INFO(
      get_logger(),
      "vehicle_visualizer active: odom->/vehicle_model, mesh=%s, scale=%.3f %.3f %.3f",
      mesh_resource_.c_str(), mesh_scale_x_, mesh_scale_y_, mesh_scale_z_);
  }

private:
  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    const auto now = get_clock()->now();
    if (publish_period_sec_ > 0.0 && last_publish_time_.nanoseconds() > 0)
    {
      const double elapsed = (now - last_publish_time_).seconds();
      if (elapsed < publish_period_sec_)
      {
        return;
      }
    }
    last_publish_time_ = now;

    geometry_msgs::msg::Pose base = msg->pose.pose;
    base.orientation = normalizedQuaternion(base.orientation);

    // Positions belong to the odometry frame, not a relabelled parameter frame.
    const std::string marker_frame = msg->header.frame_id;
    const rclcpp::Time stamp = msg->header.stamp.sec == 0 && msg->header.stamp.nanosec == 0
                                 ? now
                                 : rclcpp::Time(msg->header.stamp);

    visualization_msgs::msg::MarkerArray arr;
    arr.markers.reserve(10);

    addShadow(arr, marker_frame, stamp, base);
    addMesh(arr, marker_frame, stamp, base);

    addFootprint(arr, marker_frame, stamp, base);

    if (show_lidar_hint_)
    {
      addCylinder(
        arr, marker_frame, stamp, 40, "vehicle_lidar_hint", base, length_ * 0.18, 0.0,
        height_ * 0.58, 0.09, 0.04, Color{0.10F, 1.00F, 0.85F, 0.85F});
    }

    if (show_heading_arrow_)
    {
      addArrow(arr, marker_frame, stamp, base);
    }

    marker_pub_->publish(arr);
  }

  void applyCommon(
    visualization_msgs::msg::Marker & marker, const std::string & frame_id,
    const rclcpp::Time & stamp, const int id, const std::string & ns, const Color & color) const
  {
    marker.header.frame_id = frame_id;
    marker.header.stamp = stamp;
    marker.ns = ns;
    marker.id = id;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.color.r = color.r;
    marker.color.g = color.g;
    marker.color.b = color.b;
    marker.color.a = color.a;
    if (lifetime_sec_ > 0.0)
    {
      marker.lifetime = rclcpp::Duration::from_seconds(lifetime_sec_);
    }
  }

  void addMesh(
    visualization_msgs::msg::MarkerArray & arr, const std::string & frame_id,
    const rclcpp::Time & stamp, const geometry_msgs::msg::Pose & base) const
  {
    visualization_msgs::msg::Marker marker;
    applyCommon(
      marker, frame_id, stamp, 0, "vehicle_mesh",
      Color{0.92F, 0.96F, 1.00F, static_cast<float>(std::clamp(mesh_alpha_, 0.0, 1.0))});
    marker.type = visualization_msgs::msg::Marker::MESH_RESOURCE;
    marker.pose.position = translatedPoint(
      base, base_to_chassis_center_x_ + mesh_offset_x_, base_to_chassis_center_y_ + mesh_offset_y_,
      base_to_chassis_center_z_ + mesh_offset_z_);
    marker.pose.orientation = base.orientation;
    marker.scale.x = std::max(0.001, mesh_scale_x_);
    marker.scale.y = std::max(0.001, mesh_scale_y_);
    marker.scale.z = std::max(0.001, mesh_scale_z_);
    marker.mesh_resource = mesh_resource_;
    marker.mesh_use_embedded_materials = mesh_use_embedded_materials_;
    arr.markers.push_back(marker);
  }

  void addCylinder(
    visualization_msgs::msg::MarkerArray & arr, const std::string & frame_id,
    const rclcpp::Time & stamp, const int id, const std::string & ns,
    const geometry_msgs::msg::Pose & base, const double x, const double y, const double z,
    const double diameter, const double height, const Color & color) const
  {
    visualization_msgs::msg::Marker marker;
    applyCommon(marker, frame_id, stamp, id, ns, color);
    marker.type = visualization_msgs::msg::Marker::CYLINDER;
    marker.pose.position = translatedPoint(base, x, y, z);
    marker.pose.orientation = base.orientation;
    marker.scale.x = std::max(0.001, diameter);
    marker.scale.y = std::max(0.001, diameter);
    marker.scale.z = std::max(0.001, height);
    arr.markers.push_back(marker);
  }

  void addFootprint(
    visualization_msgs::msg::MarkerArray & arr, const std::string & frame_id,
    const rclcpp::Time & stamp, const geometry_msgs::msg::Pose & base) const
  {
    visualization_msgs::msg::Marker marker;
    applyCommon(marker, frame_id, stamp, 30, "vehicle_footprint", Color{1.0F, 0.78F, 0.08F, 0.95F});
    marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.035;

    const double hx = length_ * 0.5;
    const double hy = width_ * 0.5;
    const std::array<std::array<double, 2>, 5> corners = {
      {{{hx, hy}}, {{hx, -hy}}, {{-hx, -hy}}, {{-hx, hy}}, {{hx, hy}}}};
    for (const auto & c : corners)
    {
      marker.points.push_back(translatedPoint(base, c[0], c[1], footprint_z_));
    }
    arr.markers.push_back(marker);
  }

  void addShadow(
    visualization_msgs::msg::MarkerArray & arr, const std::string & frame_id,
    const rclcpp::Time & stamp, const geometry_msgs::msg::Pose & base) const
  {
    visualization_msgs::msg::Marker marker;
    applyCommon(marker, frame_id, stamp, 31, "vehicle_shadow", Color{0.0F, 0.0F, 0.0F, 0.28F});
    marker.type = visualization_msgs::msg::Marker::CYLINDER;
    marker.pose.position = translatedPoint(base, 0.0, 0.0, footprint_z_ - 0.015);
    marker.pose.orientation = base.orientation;
    marker.scale.x = length_ * 1.06;
    marker.scale.y = width_ * 1.06;
    marker.scale.z = 0.012;
    arr.markers.push_back(marker);
  }

  void addArrow(
    visualization_msgs::msg::MarkerArray & arr, const std::string & frame_id,
    const rclcpp::Time & stamp, const geometry_msgs::msg::Pose & base) const
  {
    visualization_msgs::msg::Marker marker;
    applyCommon(marker, frame_id, stamp, 50, "vehicle_heading", Color{1.0F, 0.18F, 0.08F, 0.95F});
    marker.type = visualization_msgs::msg::Marker::ARROW;
    marker.pose.position = translatedPoint(base, 0.0, 0.0, heading_arrow_z_);
    marker.pose.orientation = base.orientation;
    marker.scale.x = length_ * 0.72;
    marker.scale.y = 0.07;
    marker.scale.z = 0.07;
    arr.markers.push_back(marker);
  }

  std::string frame_id_;
  double publish_period_sec_ = 0.05;
  double lifetime_sec_ = 0.30;
  double length_ = 0.840;
  double width_ = 0.626;
  double height_ = 0.338;
  std::string mesh_resource_;
  bool mesh_use_embedded_materials_ = true;
  double mesh_alpha_ = 0.96;
  double mesh_native_length_ = 1.0086837;
  double mesh_native_width_ = 0.7673552;
  double mesh_native_height_ = 0.3848279;
  double mesh_scale_x_ = 0.833;
  double mesh_scale_y_ = 0.816;
  double mesh_scale_z_ = 0.878;
  double mesh_offset_x_ = 0.0;
  double mesh_offset_y_ = 0.0;
  double mesh_offset_z_ = 0.0;
  double wheel_base_ = 0.430;
  double track_width_ = 0.548;
  double base_to_chassis_center_x_ = 0.0;
  double base_to_chassis_center_y_ = 0.0;
  double base_to_chassis_center_z_ = 0.0;
  double wheel_center_z_ = -0.04;
  double footprint_z_ = -0.16;
  double heading_arrow_z_ = 0.24;
  bool show_heading_arrow_ = true;
  bool show_lidar_hint_ = true;

  rclcpp::Time last_publish_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<VehicleVisualizerNode>());
  rclcpp::shutdown();
  return 0;
}
