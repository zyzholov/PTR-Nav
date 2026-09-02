#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <nav_msgs/msg/odometry.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <se2_grid_core/SE2Grid.hpp>
#include <se2_grid_ros/SE2GridRosConverter.hpp>
#include <se2_grid_msgs/msg/se2_grid.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

using Eigen::Array2i;
using Eigen::Matrix3f;
using Eigen::Matrix4f;
using Eigen::MatrixXf;
using Eigen::Quaternionf;
using Eigen::Vector2d;
using Eigen::Vector2f;
using Eigen::Vector2i;
using Eigen::Vector3f;

void gpuInit(const Vector2i& size_pos_,
             int size_yaw_,
             int iter_num_,
             const Vector2f& length_pos_,
             float resolution_pos_,
             float resolution_yaw_,
             const Vector2f& position_,
             float min_var_,
             float max_var_,
             float hori_var_,
             float mahalanobis_threshold_,
             float ellipsoid_x_,
             float ellipsoid_y_,
             const Vector2f& ellipsoid_offset_,
             float normal_radius_,
             float max_phx_,
             float max_phy_,
             float max_curvature_,
             float weight_phi_,
             float weight_curvature_,
             float ignore_z_min_,
             float ignore_z_max_,
             float beam_sigma2_,
             const Vector3f& body2sensor_T_);

void computerMapSE2(const int& point_num_,
                    float* points_world,
                    const float& resolution_pos_,
                    const int& size_yaw_,
                    const Vector2i& size_pos_,
                    const Vector3f& body_T_,
                    const Matrix3f& body_R_,
                    const Matrix4f& body_Cov_ZR_,
                    MatrixXf& elevation_,
                    MatrixXf& var_,
                    MatrixXf& inpainted_,
                    MatrixXf& smooth_,
                    std::vector<MatrixXf>& zbx_,
                    std::vector<MatrixXf>& zby_,
                    std::vector<MatrixXf>& risk_,
                    MatrixXf& sdf_,
                    Array2i& start_idx_,
                    Vector2f& position_,
                    bool verbose_);

namespace {

template <typename GetterT, typename SetterT>
void fillEsdf1d(GetterT getter, SetterT setter, int start, int end, int size)
{
  std::vector<int> v(static_cast<size_t>(size), 0);
  std::vector<float> z(static_cast<size_t>(size + 1), 0.0f);

  int k = start;
  v[static_cast<size_t>(start)] = start;
  z[static_cast<size_t>(start)] = -std::numeric_limits<float>::max();
  z[static_cast<size_t>(start + 1)] = std::numeric_limits<float>::max();

  for (int q = start + 1; q <= end; ++q)
  {
    float s = 0.0f;
    do
    {
      --k;
      s = ((getter(q) + q * q) - (getter(v[static_cast<size_t>(k)]) +
           v[static_cast<size_t>(k)] * v[static_cast<size_t>(k)])) /
          (2.0f * q - 2.0f * v[static_cast<size_t>(k)]);
    } while (s <= z[static_cast<size_t>(k)]);

    ++k;
    v[static_cast<size_t>(k)] = q;
    z[static_cast<size_t>(k)] = s;
    z[static_cast<size_t>(k + 1)] = std::numeric_limits<float>::max();
  }

  k = start;
  for (int q = start; q <= end; ++q)
  {
    while (z[static_cast<size_t>(k + 1)] < q)
      ++k;
    const float val = (q - v[static_cast<size_t>(k)]) * (q - v[static_cast<size_t>(k)]) +
                      getter(v[static_cast<size_t>(k)]);
    setter(q, val);
  }
}

MatrixXf computeSignedDistanceField(const MatrixXf& source, float threshold, float resolution)
{
  const int rows = source.rows();
  const int cols = source.cols();
  const int size = std::max(rows, cols);

  Eigen::MatrixXi occ(rows, cols);
  occ.setZero();
  for (int r = 0; r < rows; ++r)
  {
    for (int c = 0; c < cols; ++c)
    {
      const float val = source(r, c);
      if (std::isfinite(val) && val > threshold)
        occ(r, c) = 1;
    }
  }

  MatrixXf tmp(rows, cols);
  MatrixXf dist(rows, cols);
  MatrixXf neg(rows, cols);
  Eigen::MatrixXi neg_occ(rows, cols);
  MatrixXf result(rows, cols);

  for (int x = 0; x < rows; ++x)
  {
    fillEsdf1d(
      [&](int y) {
        return occ(x, y) == 1 ? 0.0f : std::numeric_limits<float>::max();
      },
      [&](int y, float val) { tmp(x, y) = val; },
      0, cols - 1, size);
  }

  for (int y = 0; y < cols; ++y)
  {
    fillEsdf1d(
      [&](int x) { return tmp(x, y); },
      [&](int x, float val) { dist(x, y) = resolution * std::sqrt(val); },
      0, rows - 1, size);
  }

  for (int x = 0; x < rows; ++x)
  {
    for (int y = 0; y < cols; ++y)
      neg_occ(x, y) = occ(x, y) == 0 ? 1 : 0;
  }

  for (int x = 0; x < rows; ++x)
  {
    fillEsdf1d(
      [&](int y) {
        return neg_occ(x, y) == 1 ? 0.0f : std::numeric_limits<float>::max();
      },
      [&](int y, float val) { tmp(x, y) = val; },
      0, cols - 1, size);
  }

  for (int y = 0; y < cols; ++y)
  {
    fillEsdf1d(
      [&](int x) { return tmp(x, y); },
      [&](int x, float val) { neg(x, y) = resolution * std::sqrt(val); },
      0, rows - 1, size);
  }

  for (int x = 0; x < rows; ++x)
  {
    for (int y = 0; y < cols; ++y)
    {
      result(x, y) = dist(x, y);
      if (neg(x, y) > 0.0f)
        result(x, y) += (-neg(x, y) + resolution);
    }
  }

  return result;
}

}  // namespace

class TerrainAnalyzerGpuNode : public rclcpp::Node
{
public:
  TerrainAnalyzerGpuNode()
  : Node("terrain_analyzer_gpu")
  {
    declare_parameter("cache_size", 200);
    declare_parameter("verbose", true);
    declare_parameter("publish_fused_map", true);
    declare_parameter("sdf.threshold", 0.5);

    declare_parameter("sensor.ignore_z_max", 0.5);
    declare_parameter("sensor.ignore_z_min", -2.0);
    declare_parameter("sensor.sensor2body_t", std::vector<double>{0.0, 0.0, 0.0});
    declare_parameter("sensor.beam_sigma", 0.008);

    declare_parameter("elevation.length_pos_x", 10.0);
    declare_parameter("elevation.length_pos_y", 10.0);
    declare_parameter("elevation.resolution_pos", 0.1);
    declare_parameter("elevation.resolution_yaw", 0.2);
    declare_parameter("elevation.min_var", 1e-4);
    declare_parameter("elevation.max_var", 0.05);
    declare_parameter("elevation.min_var_hori", 1e-4);
    declare_parameter("elevation.mahalanobis_threshold", 2.5);
    declare_parameter("elevation.ellipsoid_x", 0.4);
    declare_parameter("elevation.ellipsoid_y", 0.3);
    declare_parameter("elevation.ellipsoid_offset", std::vector<double>{0.13, 0.0});
    declare_parameter("elevation.iter_num", 1);

    declare_parameter("gpu.normal_radius", 0.15);
    declare_parameter("gpu.max_phx", 0.786);
    declare_parameter("gpu.max_phy", 1.571);
    declare_parameter("gpu.max_curvature", 1000.0);
    declare_parameter("gpu.weight_phi", 0.2);
    declare_parameter("gpu.weight_curvature", 0.6);

    cache_size_ = get_parameter("cache_size").as_int();
    verbose_ = get_parameter("verbose").as_bool();
    publish_fused_map_ = get_parameter("publish_fused_map").as_bool();
    sdf_threshold_ = static_cast<float>(get_parameter("sdf.threshold").as_double());

    const auto sensor2body_t = get_parameter("sensor.sensor2body_t").as_double_array();
    const auto ellipsoid_offset = get_parameter("elevation.ellipsoid_offset").as_double_array();

    if (sensor2body_t.size() != 3 || ellipsoid_offset.size() != 2)
      throw std::runtime_error("terrain_analyzer_gpu parameter vector sizes are invalid.");

    Vector3f body2sensor_t;
    body2sensor_t << static_cast<float>(-sensor2body_t[0]),
                     static_cast<float>(-sensor2body_t[1]),
                     static_cast<float>(-sensor2body_t[2]);

    Vector2f length_pos;
    length_pos << static_cast<float>(get_parameter("elevation.length_pos_x").as_double()),
                  static_cast<float>(get_parameter("elevation.length_pos_y").as_double());

    const float resolution_pos = static_cast<float>(get_parameter("elevation.resolution_pos").as_double());
    const float resolution_yaw = static_cast<float>(get_parameter("elevation.resolution_yaw").as_double());
    const float min_var = static_cast<float>(get_parameter("elevation.min_var").as_double());
    const float max_var = static_cast<float>(get_parameter("elevation.max_var").as_double());
    const float min_var_hori = static_cast<float>(get_parameter("elevation.min_var_hori").as_double());
    const float mahalanobis_threshold =
      static_cast<float>(get_parameter("elevation.mahalanobis_threshold").as_double());
    const float ellipsoid_x = static_cast<float>(get_parameter("elevation.ellipsoid_x").as_double());
    const float ellipsoid_y = static_cast<float>(get_parameter("elevation.ellipsoid_y").as_double());
    const int iter_num = get_parameter("elevation.iter_num").as_int();
    const float normal_radius = static_cast<float>(get_parameter("gpu.normal_radius").as_double());
    const float max_phx = static_cast<float>(get_parameter("gpu.max_phx").as_double());
    const float max_phy = static_cast<float>(get_parameter("gpu.max_phy").as_double());
    const float max_curvature = static_cast<float>(get_parameter("gpu.max_curvature").as_double());
    const float weight_phi = static_cast<float>(get_parameter("gpu.weight_phi").as_double());
    const float weight_curvature = static_cast<float>(get_parameter("gpu.weight_curvature").as_double());
    const float ignore_z_min = static_cast<float>(get_parameter("sensor.ignore_z_min").as_double());
    const float ignore_z_max = static_cast<float>(get_parameter("sensor.ignore_z_max").as_double());
    const float beam_sigma = static_cast<float>(get_parameter("sensor.beam_sigma").as_double());

    Vector2f ellipsoid_offset_vec;
    ellipsoid_offset_vec << static_cast<float>(ellipsoid_offset[0]),
                            static_cast<float>(ellipsoid_offset[1]);

    fused_map_.initGeometry(length_pos.cast<double>(), resolution_pos, resolution_yaw, Vector2d::Zero());
    fused_map_.add("elevation", false, 0.0);
    fused_map_.add("var", false, 0.0);
    fused_map_.add("inpainted", false, 0.0);
    fused_map_.add("smooth", false, 0.0);
    fused_map_.add("normal_x", false, 0.0);
    fused_map_.add("normal_y", false, 0.0);
    fused_map_.add("risk", true, 0.0);
    fused_map_.add("zbx", true, 0.0);
    fused_map_.add("zby", true, 0.0);

    sdf_map_.initGeometry(length_pos.cast<double>(), resolution_pos, 8.0, Vector2d::Zero());
    sdf_map_.add("sdf", false, 0.0);
    sdf_map_.add("sdf_cpu", false, 0.0);

    const auto size_pos = fused_map_.getSizePos().matrix();
    const int size_yaw = fused_map_.getSizeYaw();

    gpuInit(size_pos,
            size_yaw,
            iter_num,
            length_pos,
            resolution_pos,
            resolution_yaw,
            fused_map_.getPosition().cast<float>(),
            min_var,
            max_var,
            min_var_hori,
            mahalanobis_threshold,
            ellipsoid_x,
            ellipsoid_y,
            ellipsoid_offset_vec,
            normal_radius,
            max_phx,
            max_phy,
            max_curvature,
            weight_phi,
            weight_curvature,
            ignore_z_min,
            ignore_z_max,
            beam_sigma * beam_sigma,
            body2sensor_t);

    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      "odom", rclcpp::QoS(rclcpp::KeepLast(50)),
      std::bind(&TerrainAnalyzerGpuNode::odomCallback, this, std::placeholders::_1));
    cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      "cloud", rclcpp::SensorDataQoS(),
      std::bind(&TerrainAnalyzerGpuNode::cloudCallback, this, std::placeholders::_1));

    fused_pub_ = create_publisher<se2_grid_msgs::msg::SE2Grid>("fused_map", rclcpp::QoS(5));
    sdf_pub_ = create_publisher<se2_grid_msgs::msg::SE2Grid>("sdf_map", rclcpp::QoS(5));

    RCLCPP_INFO(
      get_logger(),
      "terrain_analyzer_gpu ready: size=(%.1f, %.1f)m res=%.2fm yaw_res=%.2frad size_yaw=%d",
      length_pos.x(), length_pos.y(), resolution_pos, resolution_yaw, size_yaw);
  }

private:
  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    odom_cache_.push_back(msg);
    while (static_cast<int>(odom_cache_.size()) > cache_size_)
      odom_cache_.pop_front();
  }

  nav_msgs::msg::Odometry::SharedPtr getOdomBefore(const rclcpp::Time& stamp) const
  {
    for (auto it = odom_cache_.rbegin(); it != odom_cache_.rend(); ++it)
    {
      if (rclcpp::Time((*it)->header.stamp) <= stamp)
        return *it;
    }
    if (!odom_cache_.empty())
      return odom_cache_.back();
    return nullptr;
  }

  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    pcl::PointCloud<pcl::PointXYZ> cloud;
    try
    {
      pcl::fromROSMsg(*msg, cloud);
    }
    catch (const std::exception& e)
    {
      RCLCPP_WARN(get_logger(), "fromROSMsg failed: %s", e.what());
      return;
    }

    if (cloud.empty())
      return;

    const auto odom = getOdomBefore(rclcpp::Time(msg->header.stamp));
    if (!odom)
    {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "No odometry available for terrain update.");
      return;
    }

    const auto t0 = std::chrono::steady_clock::now();

    std::vector<float> points_world(static_cast<size_t>(cloud.size()) * 3u, 0.0f);
    for (size_t i = 0; i < cloud.size(); ++i)
    {
      points_world[i * 3u] = cloud.points[i].x;
      points_world[i * 3u + 1u] = cloud.points[i].y;
      points_world[i * 3u + 2u] = cloud.points[i].z;
    }

    const auto pose_cov = Eigen::Map<const Eigen::Matrix<double, 6, 6, Eigen::RowMajor>>(
      odom->pose.covariance.data());
    const Vector3f body_t(
      static_cast<float>(odom->pose.pose.position.x),
      static_cast<float>(odom->pose.pose.position.y),
      static_cast<float>(odom->pose.pose.position.z));
    const Matrix3f body_r = Quaternionf(
      static_cast<float>(odom->pose.pose.orientation.w),
      static_cast<float>(odom->pose.pose.orientation.x),
      static_cast<float>(odom->pose.pose.orientation.y),
      static_cast<float>(odom->pose.pose.orientation.z)).matrix();
    const Matrix4f body_cov_zr = pose_cov.bottomRightCorner(4, 4).cast<float>();

    auto& elevation = fused_map_["elevation"][0];
    auto& var = fused_map_["var"][0];
    auto& inpainted = fused_map_["inpainted"][0];
    auto& smooth = fused_map_["smooth"][0];
    auto& sdf = sdf_map_["sdf"][0];
    auto& zbx = fused_map_["zbx"];
    auto& zby = fused_map_["zby"];
    auto& risk = fused_map_["risk"];

    Vector2f position_map = fused_map_.getPosition().cast<float>();
    Array2i start_idx_map = fused_map_.getStartIndex();
    const Vector2i size_pos_map = fused_map_.getSizePos().matrix();
    const float resolution_pos = static_cast<float>(fused_map_.getResolutionPos());
    const int yaw_num = fused_map_.getSizeYaw();

    computerMapSE2(
      static_cast<int>(cloud.size()),
      points_world.data(),
      resolution_pos,
      yaw_num,
      size_pos_map,
      body_t,
      body_r,
      body_cov_zr,
      elevation,
      var,
      inpainted,
      smooth,
      zbx,
      zby,
      risk,
      sdf,
      start_idx_map,
      position_map,
      verbose_);

    fused_map_.setPosition(position_map.cast<double>());
    fused_map_.setStartIndex(start_idx_map);

    sdf_map_.setPosition(position_map.cast<double>());
    sdf_map_.setStartIndex(start_idx_map);
    sdf_map_.convertToDefaultStartIndex();
    sdf_map_["sdf_cpu"][0] = computeSignedDistanceField(sdf_map_["sdf"][0], sdf_threshold_, resolution_pos);

    publishMaps(msg->header.stamp);

    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "terrain_analyzer_gpu: pts=%zu proc=%.1fms yaw=%d map=(%d,%d)",
      cloud.size(), elapsed_ms, yaw_num, size_pos_map.x(), size_pos_map.y());
  }

  void publishMaps(const builtin_interfaces::msg::Time& stamp)
  {
    se2_grid_msgs::msg::SE2Grid fused_msg;
    se2_grid_msgs::msg::SE2Grid sdf_msg;

    if (publish_fused_map_)
    {
      se2_grid::SE2GridRosConverter::toMessage(fused_map_, fused_msg);
      fused_msg.info.header.stamp = stamp;
      fused_msg.info.header.frame_id = "map";
      fused_pub_->publish(fused_msg);
    }

    se2_grid::SE2GridRosConverter::toMessage(sdf_map_, sdf_msg);
    sdf_msg.info.header.stamp = stamp;
    sdf_msg.info.header.frame_id = "map";
    sdf_pub_->publish(sdf_msg);
  }

  int cache_size_ = 200;
  bool verbose_ = true;
  bool publish_fused_map_ = true;
  float sdf_threshold_ = 0.5f;

  std::deque<nav_msgs::msg::Odometry::SharedPtr> odom_cache_;
  se2_grid::SE2Grid fused_map_;
  se2_grid::SE2Grid sdf_map_;

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Publisher<se2_grid_msgs::msg::SE2Grid>::SharedPtr fused_pub_;
  rclcpp::Publisher<se2_grid_msgs::msg::SE2Grid>::SharedPtr sdf_pub_;
};

int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<TerrainAnalyzerGpuNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
