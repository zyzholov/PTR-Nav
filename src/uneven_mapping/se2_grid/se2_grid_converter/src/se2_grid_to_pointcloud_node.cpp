#include "se2_grid_converter/grid_projection.hpp"
#include "se2_grid_converter/surface_geometry.hpp"
#include "se2_grid_converter/display_colors.hpp"

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <se2_grid_msgs/msg/se2_grid.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace
{
struct LayerSelection
{
  int layer_index{-1};
  std::size_t data_index{0};
  bool has_so2{false};
};
}  // namespace

class SE2GridToPointCloudConverter : public rclcpp::Node
{
public:
  SE2GridToPointCloudConverter()
  : Node("se2_grid_to_pointcloud_converter")
  {
    declare_parameter("input_topic", std::string("/fused_map"));
    declare_parameter("output_topic", std::string("/elevation_cloud"));
    declare_parameter("layer_name", std::string("inpainted"));
    declare_parameter("valid_layer_name", std::string(""));
    declare_parameter("valid_threshold", 0.5);
    declare_parameter("yaw_index", 0);
    declare_parameter("follow_odom_yaw", false);
    declare_parameter("odom_topic", std::string("/Odometry"));
    declare_parameter("height_layer_name", std::string(""));
    declare_parameter("drape_on_height", false);
    declare_parameter("height_as_z", true);
    declare_parameter("z_value", 0.0);
    declare_parameter("z_offset", 0.0);
    declare_parameter("use_color_map", true);
    declare_parameter("invert_color_map", false);
    declare_parameter("min_height", -5.0);
    declare_parameter("max_height", 5.0);
    declare_parameter("colormap", std::string("terrain"));
    declare_parameter("publish_surface", false);
    declare_parameter("surface_topic", std::string(""));
    declare_parameter("height_surface_topic", std::string(""));
    declare_parameter("surface_max_step", 0.65);
    declare_parameter("surface_alpha", 0.92);
    declare_parameter("surface_risk_layer", std::string(""));
    declare_parameter("surface_probability_colors", false);
    declare_parameter("surface_use_navigation_height", false);
    declare_parameter("surface_body_collision_overlay",false);
    declare_parameter("occupied_display_style",std::string("cells"));
    declare_parameter("body_pose_surface_topic",std::string(""));

    input_topic_ = get_parameter("input_topic").as_string();
    output_topic_ = get_parameter("output_topic").as_string();
    layer_name_ = get_parameter("layer_name").as_string();
    valid_layer_name_ = get_parameter("valid_layer_name").as_string();
    valid_threshold_ = get_parameter("valid_threshold").as_double();
    yaw_index_ = std::max(0, static_cast<int>(get_parameter("yaw_index").as_int()));
    follow_odom_yaw_ = get_parameter("follow_odom_yaw").as_bool();
    odom_topic_ = get_parameter("odom_topic").as_string();
    height_layer_name_ = get_parameter("height_layer_name").as_string();
    drape_on_height_ = get_parameter("drape_on_height").as_bool();
    height_as_z_ = get_parameter("height_as_z").as_bool();
    z_value_ = get_parameter("z_value").as_double();
    z_offset_ = get_parameter("z_offset").as_double();
    use_color_map_ = get_parameter("use_color_map").as_bool();
    invert_color_map_ = get_parameter("invert_color_map").as_bool();
    min_height_ = get_parameter("min_height").as_double();
    max_height_ = get_parameter("max_height").as_double();
    colormap_ = get_parameter("colormap").as_string();
    publish_surface_ = get_parameter("publish_surface").as_bool();
    surface_topic_ = get_parameter("surface_topic").as_string();
    const auto height_topic=get_parameter("height_surface_topic").as_string();
    if(!height_topic.empty()) height_surface_pub_=create_publisher<visualization_msgs::msg::Marker>(height_topic,1);
    surface_max_step_ = get_parameter("surface_max_step").as_double();
    surface_alpha_ = get_parameter("surface_alpha").as_double();
    surface_risk_layer_ = get_parameter("surface_risk_layer").as_string();
    surface_body_collision_overlay_=get_parameter("surface_body_collision_overlay").as_bool();
    occupied_display_style_=get_parameter("occupied_display_style").as_string();
    if(occupied_display_style_!="cells"&&occupied_display_style_!="boundary")
      throw std::invalid_argument("occupied_display_style must be cells or boundary");
    const auto pose_topic=get_parameter("body_pose_surface_topic").as_string();
    if(!pose_topic.empty())body_pose_surface_pub_=create_publisher<visualization_msgs::msg::Marker>(pose_topic,1);
    surface_probability_colors_ = get_parameter("surface_probability_colors").as_bool();
    surface_use_navigation_height_=get_parameter("surface_use_navigation_height").as_bool();

    cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      output_topic_, rclcpp::QoS(rclcpp::KeepLast(1)).reliable());
    if (publish_surface_ && !surface_topic_.empty()) {
      surface_pub_ = create_publisher<visualization_msgs::msg::Marker>(
        surface_topic_, rclcpp::QoS(rclcpp::KeepLast(1)).reliable());
      occupied_pub_=create_publisher<visualization_msgs::msg::Marker>(surface_topic_+"/occupied_cells",1);
      if (surface_probability_colors_) {
        legend_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
          surface_topic_+"/legend", rclcpp::QoS(1).reliable().transient_local());
      }
    }
    grid_sub_ = create_subscription<se2_grid_msgs::msg::SE2Grid>(
      input_topic_, rclcpp::QoS(rclcpp::KeepLast(1)).reliable(),
      std::bind(&SE2GridToPointCloudConverter::gridCallback, this, std::placeholders::_1));

    if (follow_odom_yaw_) {
      odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        odom_topic_, rclcpp::QoS(rclcpp::KeepLast(5)).best_effort(),
        [this](const nav_msgs::msg::Odometry::SharedPtr msg) {
          const auto & q = msg->pose.pose.orientation;
          const double sin_yaw = 2.0 * (q.w * q.z + q.x * q.y);
          const double cos_yaw = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
          std::lock_guard<std::mutex> lock(odom_mutex_);
          odom_yaw_ = std::atan2(sin_yaw, cos_yaw);
          have_odom_yaw_ = std::isfinite(odom_yaw_);
        });
    }

    RCLCPP_INFO(
      get_logger(),
      "SE2Grid visualizer: %s -> %s, layer=%s, yaw=%s, layout-aware projection=enabled",
      input_topic_.c_str(), output_topic_.c_str(), layer_name_.c_str(),
      follow_odom_yaw_ ? "follow odom" : "fixed");
  }

private:
  bool resolveLayer(
    const se2_grid_msgs::msg::SE2Grid & msg,
    const std::string & name,
    LayerSelection & selection,
    std::string & error) const
  {
    if (msg.layers.size() != msg.has_so2.size()) {
      error = "layers and has_so2 have different lengths";
      return false;
    }
    const double yaw_resolution = std::max(1.0e-9, msg.info.yaw_resolution);
    const int yaw_count = std::max(
      1, static_cast<int>(std::ceil(std::max(msg.info.length_yaw, yaw_resolution) /
                                    yaw_resolution - 1.0e-6)));
    std::size_t data_index = 0;
    for (std::size_t i = 0; i < msg.layers.size(); ++i) {
      const std::size_t count = msg.has_so2[i] ? static_cast<std::size_t>(yaw_count) : 1u;
      if (msg.layers[i] == name) {
        const int requested_yaw = follow_odom_yaw_ ? currentYawIndex(msg) : yaw_index_;
        selection.layer_index = static_cast<int>(i);
        selection.data_index = data_index + static_cast<std::size_t>(
          msg.has_so2[i] ? std::clamp(requested_yaw, 0, yaw_count - 1) : 0);
        selection.has_so2 = msg.has_so2[i];
        if (selection.data_index >= msg.data.size()) {
          error = "layer resolves beyond message data blocks";
          return false;
        }
        return true;
      }
      data_index += count;
    }
    error = "configured layer '" + name + "' is absent";
    return false;
  }

  int currentYawIndex(const se2_grid_msgs::msg::SE2Grid & msg) const
  {
    std::lock_guard<std::mutex> lock(odom_mutex_);
    if (!have_odom_yaw_) {
      return yaw_index_;
    }
    const int yaw_count = std::max(
      1, static_cast<int>(std::ceil(std::max(msg.info.length_yaw, msg.info.yaw_resolution) /
                                    std::max(1.0e-9, msg.info.yaw_resolution) - 1.0e-6)));
    return se2_grid_converter::yawIndex(odom_yaw_, msg.info.yaw_resolution, yaw_count);
  }

  bool prepareLayout(
    const se2_grid_msgs::msg::SE2Grid & msg,
    const LayerSelection & selection,
    se2_grid_converter::MatrixLayout & layout,
    std::string & error) const
  {
    const auto & info = msg.info;
    if (!(info.pos_resolution > 0.0) || !(info.length_x > 0.0) || !(info.length_y > 0.0)) {
      error = "grid geometry has non-positive resolution or length";
      return false;
    }
    const std::size_t rows = static_cast<std::size_t>(std::ceil(info.length_x / info.pos_resolution));
    const std::size_t cols = static_cast<std::size_t>(std::ceil(info.length_y / info.pos_resolution));
    return se2_grid_converter::decodeLayout(
      msg.data[selection.data_index], rows, cols, layout, error);
  }

  void gridCallback(const se2_grid_msgs::msg::SE2Grid::SharedPtr msg)
  {
    auto cloud = convert(*msg);
    if (cloud.has_value()) {
      cloud_pub_->publish(*cloud);
    }
    if (surface_pub_ && surface_pub_->get_subscription_count() > 0) {
      auto surface = convertSurface(*msg);
      if (surface.has_value()) {
        surface_pub_->publish(*surface);
      } else {
        visualization_msgs::msg::Marker clear;
        clear.header = msg->info.header;
        clear.ns = "terrain_surface";
        clear.id = 0;
        clear.action = visualization_msgs::msg::Marker::DELETE;
        surface_pub_->publish(clear);
      }
    }
    if(body_pose_surface_pub_ && body_pose_surface_pub_->get_subscription_count()>0) {
      auto surface=convertSurface(*msg,false,true);
      if(surface)body_pose_surface_pub_->publish(*surface);
      else {
        visualization_msgs::msg::Marker clear;clear.header=msg->info.header;
        clear.ns="body_pose_risk_surface";clear.id=0;clear.action=visualization_msgs::msg::Marker::DELETE;
        body_pose_surface_pub_->publish(clear);
      }
    }
    if (legend_pub_ && legend_pub_->get_subscription_count() > 0) publishProbabilityLegend(*msg);
    if(height_surface_pub_ && height_surface_pub_->get_subscription_count()>0) {
      const auto begin=std::chrono::steady_clock::now();
      auto surface=convertSurface(*msg,true);
      if(surface) height_surface_pub_->publish(*surface);
      else {
        visualization_msgs::msg::Marker clear;clear.header=msg->info.header;
        clear.ns="measured_elevation";clear.id=0;clear.action=visualization_msgs::msg::Marker::DELETE;
        height_surface_pub_->publish(clear);
      }
      RCLCPP_INFO_THROTTLE(get_logger(),*get_clock(),5000,"measured elevation mesh: %zu triangles, %.2f ms build+publish",
        surface ? surface->points.size()/3 : 0,
        std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count());
    }
  }

  void publishProbabilityLegend(const se2_grid_msgs::msg::SE2Grid &msg)
  {
    visualization_msgs::msg::MarkerArray legend;
    const float probabilities[]={0,0.001F,0.01F,0.05F,1.0F};
    const char *labels[]={"0%","0.1%","1%","5%","100%","Unknown"};
    const double x=msg.info.pose.position.x-3.6;
    const double y=msg.info.pose.position.y+0.36*msg.info.length_y;
    double z=msg.info.pose.position.z+0.3;
    // Put the key at the terrain height near the window center, outside the
    // risk surface. It is a map annotation, not a fabricated terrain layer.
    LayerSelection h; se2_grid_converter::MatrixLayout layout; std::string error;
    if (resolveLayer(msg,layer_name_,h,error) && prepareLayout(msg,h,layout,error)) {
      float center=0;
      const auto row=(layout.rows/2+msg.start_index_row)%layout.rows;
      const auto col=(layout.cols/2+msg.start_index_col)%layout.cols;
      if (se2_grid_converter::readValue(msg.data[h.data_index],layout,row,col,center) &&
          std::isfinite(center)) z=center+0.3;
    }
    for (int i=0;i<6;++i) {
      visualization_msgs::msg::Marker swatch;
      swatch.header=msg.info.header; swatch.ns="probability_legend"; swatch.id=2*i;
      swatch.type=visualization_msgs::msg::Marker::CUBE;
      swatch.action=visualization_msgs::msg::Marker::ADD;
      swatch.pose.orientation.w=1;
      swatch.pose.position.x=x+1.2*i; swatch.pose.position.y=y; swatch.pose.position.z=z;
      swatch.scale.x=0.28; swatch.scale.y=0.28; swatch.scale.z=0.04;
      std::uint8_t r=87,g=94,b=105;
      if(i<5) probabilityColor(probabilities[i],r,g,b);
      swatch.color.r=r/255.0F; swatch.color.g=g/255.0F; swatch.color.b=b/255.0F; swatch.color.a=1;
      legend.markers.push_back(swatch);
      auto text=swatch;
      text.id=2*i+1; text.type=visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
      text.pose.position.x+=0.48; text.scale.z=0.30;
      text.color.r=0.86F; text.color.g=0.89F; text.color.b=0.92F;
      text.text=labels[i]; legend.markers.push_back(text);
    }
    auto title=legend.markers.back();
    title.id=12; title.pose.position.x=x+2.7; title.pose.position.y=y-0.65;
    title.text=surface_body_collision_overlay_?"Whole-vehicle pose bound (current heading)":
      "Terrain pose bound + raw obstacles | full body checked separately";
    title.scale.z=0.28; legend.markers.push_back(title);
    legend_pub_->publish(legend);
  }

  std::optional<visualization_msgs::msg::Marker> convertSurface(
    const se2_grid_msgs::msg::SE2Grid & msg,bool height_display=false,bool body_pose_display=false)
  {
    if (msg.layers.empty() || msg.data.empty()) {
      return std::nullopt;
    }

    LayerSelection value_selection;
    std::string error;
    const auto has=[&](const char *name){return std::find(msg.layers.begin(),msg.layers.end(),name)!=msg.layers.end();};
    const std::string height_layer=has("gp_elevation_observed") ? "gp_elevation_observed" : "elevation";
    if (!resolveLayer(msg, height_display || surface_use_navigation_height_ ? height_layer : layer_name_, value_selection, error)) {
      return std::nullopt;
    }
    se2_grid_converter::MatrixLayout value_layout;
    if (!prepareLayout(msg, value_selection, value_layout, error)) {
      return std::nullopt;
    }

    const auto & info = msg.info;
    const std::size_t rows = value_layout.rows;
    const std::size_t cols = value_layout.cols;
    if (rows < 2 || cols < 2) {
      return std::nullopt;
    }

    const std_msgs::msg::Float32MultiArray * valid_data = nullptr;
    se2_grid_converter::MatrixLayout valid_layout;
    const std::string mask_layer=height_display || surface_use_navigation_height_ ? (has("terrain_observed") ? "terrain_observed" :
      has("ground_observed") ? "ground_observed" : "valid") : valid_layer_name_;
    if (!mask_layer.empty()) {
      LayerSelection valid_selection;
      if (!resolveLayer(msg, mask_layer, valid_selection, error) ||
          !prepareLayout(msg, valid_selection, valid_layout, error)) {
        return std::nullopt;
      }
      valid_data = &msg.data[valid_selection.data_index];
    }

    const std_msgs::msg::Float32MultiArray *navigation_data=nullptr,*navigation_mask=nullptr,*occupied_data=nullptr;
    se2_grid_converter::MatrixLayout navigation_layout,navigation_mask_layout,occupied_layout;
    if(!height_display && surface_use_navigation_height_) {
      LayerSelection selection;
      if(resolveLayer(msg,"probability_elevation",selection,error) && prepareLayout(msg,selection,navigation_layout,error))
        navigation_data=&msg.data[selection.data_index];
      if(resolveLayer(msg,"probability_gp_used",selection,error) && prepareLayout(msg,selection,navigation_mask_layout,error))
        navigation_mask=&msg.data[selection.data_index];
    }
    if(!height_display) {
      LayerSelection selection;
      if(resolveLayer(msg,has("collision_current") ? "collision_current" : "collision_risk",selection,error) &&
          prepareLayout(msg,selection,occupied_layout,error))occupied_data=&msg.data[selection.data_index];
    }

    const std_msgs::msg::Float32MultiArray * risk_data = nullptr;
    se2_grid_converter::MatrixLayout risk_layout;
    const bool separate_body=!height_display && !body_pose_display && !surface_body_collision_overlay_ &&
      has("terrain_failure_probability_upper");
    if (!height_display && !surface_risk_layer_.empty()) {
      LayerSelection risk_selection;
      const std::string risk_layer=separate_body?"terrain_failure_probability_upper":surface_risk_layer_;
      if (!resolveLayer(msg, risk_layer, risk_selection, error) ||
          !prepareLayout(msg, risk_selection, risk_layout, error)) {
        return std::nullopt;
      }
      risk_data = &msg.data[risk_selection.data_index];
    }

    const auto & values = msg.data[value_selection.data_index];
    std::vector<float> heights(rows * cols, std::numeric_limits<float>::quiet_NaN());
    std::vector<float> risks(rows * cols, risk_data ? std::numeric_limits<float>::quiet_NaN() : 0.0F);
    for (std::size_t row = 0; row < rows; ++row) {
      for (std::size_t col = 0; col < cols; ++col) {
        float value = 0.0F;
        bool observed=se2_grid_converter::readValue(values,value_layout,row,col,value) && std::isfinite(value);
        if (valid_data != nullptr) {
          float support = 0.0F;
          if (!se2_grid_converter::readValue(*valid_data, valid_layout, row, col, support) ||
              !std::isfinite(support) || support < static_cast<float>(valid_threshold_)) {
            observed=false;
          }
        }
        float predicted=std::numeric_limits<float>::quiet_NaN(),qualified=0;
        if(navigation_data && navigation_mask) {
          se2_grid_converter::readValue(*navigation_data,navigation_layout,row,col,predicted);
          se2_grid_converter::readValue(*navigation_mask,navigation_mask_layout,row,col,qualified);
        }
        value=se2_grid_converter::displayElevation(value,observed,predicted,qualified>=.5F);
        if(!std::isfinite(value))continue;
        const float z = height_as_z_ ? value + static_cast<float>(z_offset_) :
          static_cast<float>(z_value_);
        heights[row * cols + col] = z;
        if (risk_data != nullptr) {
          float risk = 0.0F;
          if (se2_grid_converter::readValue(*risk_data, risk_layout, row, col, risk) &&
              std::isfinite(risk)) {
            risks[row * cols + col] = std::clamp(risk, 0.0F, 1.0F);
          }
        }
        // This DISPLAY separates obstacle extent from vehicle configuration
        // space. It never changes failure_probability_upper or planner data.
        if(separate_body && occupied_data) {
          float occupied=0;
          if(se2_grid_converter::readValue(*occupied_data,occupied_layout,row,col,occupied) &&
             std::isfinite(occupied) && occupied>=.5F)risks[row*cols+col]=1.F;
        }
      }
    }

    visualization_msgs::msg::Marker marker;
    marker.header = info.header;
    marker.ns = height_display ? "measured_elevation" : body_pose_display?"body_pose_risk_surface":"terrain_surface";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::TRIANGLE_LIST;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 1.0;
    marker.scale.y = 1.0;
    marker.scale.z = 1.0;
    marker.color.a = static_cast<float>(std::clamp(surface_alpha_, 0.0, 1.0));
    marker.lifetime = rclcpp::Duration::from_seconds(0.0);
    marker.points.reserve((rows - 1) * (cols - 1) * 6);
    marker.colors.reserve((rows - 1) * (cols - 1) * 6);

    float height_low=0,height_high=1;
    if(height_display) {
      std::vector<float> sorted;
      for(float z:heights) if(std::isfinite(z)) sorted.push_back(z-static_cast<float>(z_offset_));
      std::sort(sorted.begin(),sorted.end());
      const auto quantile=[&](double q) {
        if(sorted.empty()) return 0.F;
        const double i=q*(sorted.size()-1);const auto a=static_cast<size_t>(i);
        return static_cast<float>(sorted[a]+(sorted[std::min(a+1,sorted.size()-1)]-sorted[a])*(i-a));
      };
      height_low=quantile(.02);height_high=std::max(height_low+.2F,quantile(.98));
    }

    auto point = [&](std::size_t row, std::size_t col) {
        geometry_msgs::msg::Point p;
        p.x = se2_grid_converter::cellCoordinate(
          info.pose.position.x, info.length_x, info.pos_resolution, row,
          msg.start_index_row, rows);
        p.y = se2_grid_converter::cellCoordinate(
          info.pose.position.y, info.length_y, info.pos_resolution, col,
          msg.start_index_col, cols);
        p.z = heights[row * cols + col];
        return p;
      };
    auto colorFor = [&](float height, float risk, double shade) {
        if(height_display) {
          const float t=std::clamp((height-height_low)/(height_high-height_low),0.F,1.F);
          const float stops[3][3]={{72,103,184},{54,164,176},{168,220,166}};
          const int i=t>=.5F ? 1 : 0;const float u=2*t-i;
          std_msgs::msg::ColorRGBA color;
          color.r=(stops[i][0]*(1-u)+stops[i+1][0]*u)/255.F;
          color.g=(stops[i][1]*(1-u)+stops[i+1][1]*u)/255.F;
          color.b=(stops[i][2]*(1-u)+stops[i+1][2]*u)/255.F;color.a=1.F;
          return color;
        }
        std::uint8_t r = 0, g = 0, b = 0;
        color(height, r, g, b);
        std_msgs::msg::ColorRGBA c;
        c.r = static_cast<float>(std::clamp(static_cast<double>(r) * shade / 255.0, 0.0, 1.0));
        c.g = static_cast<float>(std::clamp(static_cast<double>(g) * shade / 255.0, 0.0, 1.0));
        c.b = static_cast<float>(std::clamp(static_cast<double>(b) * shade / 255.0, 0.0, 1.0));
        c.a = marker.color.a;
        if (surface_probability_colors_) {
          if (!std::isfinite(risk)) {
            c.r=0.34F; c.g=0.37F; c.b=0.41F;
          } else {
            probabilityColor(risk,r,g,b);
            c.r=r/255.0F*shade;
            c.g=g/255.0F*shade;
            c.b=b/255.0F*shade;
          }
          return c;
        }
        // Keep low-risk ground subdued; reserve amber/coral for real risk.
        const float blend = std::clamp((risk - 0.20F) / 0.62F, 0.0F, 1.0F);
        const float lethal = std::clamp((risk - 0.55F) / 0.27F, 0.0F, 1.0F);
        c.r = (1.0F - blend) * c.r + blend * 0.91F;
        c.g = (1.0F - blend) * c.g + blend * (0.69F - 0.43F * lethal);
        c.b = (1.0F - blend) * c.b + blend * (0.27F - 0.07F * lethal);
        return c;
      };
    auto appendTriangle = [&](const geometry_msgs::msg::Point & a,
                              const geometry_msgs::msg::Point & b,
                              const geometry_msgs::msg::Point & c, const std::array<float,3> &risk) {
        const double ux = b.x - a.x, uy = b.y - a.y, uz = b.z - a.z;
        const double vx = c.x - a.x, vy = c.y - a.y, vz = c.z - a.z;
        double nx = uy * vz - uz * vy;
        double ny = uz * vx - ux * vz;
        double nz = ux * vy - uy * vx;
        const double norm = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (norm < 1.0e-9) return;
        nx /= norm; ny /= norm; nz /= norm;
        if (nz < 0.0) { nx = -nx; ny = -ny; nz = -nz; }
        const double light = std::clamp(0.62 + 0.38 *
          (0.18 * nx - 0.34 * ny + 0.92 * nz), 0.42, 1.0);
        marker.points.push_back(a); marker.points.push_back(b); marker.points.push_back(c);
        if(height_display) {
          for(const auto &p:{a,b,c}) marker.colors.push_back(colorFor(p.z-static_cast<float>(z_offset_),0,1));
        } else {
          const std::array<geometry_msgs::msg::Point,3> vertices={a,b,c};
          for(size_t i=0;i<3;++i)marker.colors.push_back(colorFor(
            vertices[i].z-static_cast<float>(z_offset_),risk[i],light));
        }
      };

    const auto triangles = se2_grid_converter::surfaceTriangles(
      heights, rows, cols, msg.start_index_row, msg.start_index_col, surface_max_step_);
    for (const auto & t : triangles) {
      // Preserve each grid sample, rather than dilating one hazardous corner
      // over its entire triangle. Color interpolation is presentation only.
      const std::array<float,3> risk={risks[t[0]],risks[t[1]],risks[t[2]]};
      appendTriangle(point(t[0] / cols, t[0] % cols), point(t[1] / cols, t[1] % cols),
        point(t[2] / cols, t[2] % cols), risk);
    }
    if(!height_display && !body_pose_display && occupied_pub_ && occupied_pub_->get_subscription_count()) {
      const std_msgs::msg::Float32MultiArray *low_data=nullptr,*high_data=nullptr;
      se2_grid_converter::MatrixLayout low_layout,high_layout;
      LayerSelection selection;
      if(resolveLayer(msg,"collision_witness_z_min",selection,error) && prepareLayout(msg,selection,low_layout,error))
        low_data=&msg.data[selection.data_index];
      if(resolveLayer(msg,"collision_witness_z_max",selection,error) && prepareLayout(msg,selection,high_layout,error))
        high_data=&msg.data[selection.data_index];
      visualization_msgs::msg::Marker occupied;occupied.header=info.header;
      occupied.ns="measured_occupied_cells";occupied.id=0;occupied.type=visualization_msgs::msg::Marker::LINE_LIST;
      occupied.pose.orientation.w=1.;occupied.scale.x=.018;
      occupied.color.r=.90F;occupied.color.g=.32F;occupied.color.b=.38F;occupied.color.a=.95F;
      const bool boundary=occupied_display_style_=="boundary";
      std::vector<unsigned char> raw_occupied(rows*cols,0);
      if(boundary && occupied_data)for(size_t r=0;r<rows;++r)for(size_t c=0;c<cols;++c) {
        float value=0;
        raw_occupied[r*cols+c]=se2_grid_converter::readValue(*occupied_data,occupied_layout,r,c,value)&&
          std::isfinite(value)&&value>=.5F;
      }
      if(occupied_data)for(size_t r=0;r<rows;++r)for(size_t c=0;c<cols;++c) {
        float value=0;
        if(!se2_grid_converter::readValue(*occupied_data,occupied_layout,r,c,value) ||
            !std::isfinite(value) || value<.5F)continue;
        float low=NAN,high=NAN;
        if(low_data)se2_grid_converter::readValue(*low_data,low_layout,r,c,low);
        if(high_data)se2_grid_converter::readValue(*high_data,high_layout,r,c,high);
        auto center=point(r,c);const double half=.5*info.pos_resolution;
        // Positive occupancy and terrain support are independent. Draw its
        // actual measured Z extent even when the floor under it is unknown.
        // These visualization coordinates never enter the terrain GP.
        const bool extent=std::isfinite(low)&&std::isfinite(high)&&high>=low;
        if(!extent && !std::isfinite(center.z))continue;
        const double bottom=extent?low:center.z+.012;
        const double top=extent?std::max(double(high),bottom+.04):bottom;
        center.z=bottom;
        const auto exposed=(boundary ? se2_grid_converter::occupiedBoundarySides(raw_occupied,
          rows,cols,msg.start_index_row,msg.start_index_col,r,c) : std::array<bool,4>{true,true,true,true});
        for(int side=0;side<4;++side) {
          if(!exposed[side])continue;
          geometry_msgs::msg::Point a=center,b=center;
          constexpr int dx[]={-1,1,1,-1},dy[]={-1,-1,1,1};
          a.x+=half*dx[side];a.y+=half*dy[side];b.x+=half*dx[(side+1)%4];b.y+=half*dy[(side+1)%4];
          occupied.points.push_back(a);occupied.points.push_back(b);
          if(extent && !boundary) {
            a.z=top;b.z=top;occupied.points.push_back(a);occupied.points.push_back(b);
            b=a;b.z=bottom;occupied.points.push_back(a);occupied.points.push_back(b);
          }
        }
      }
      occupied.action=occupied.points.empty() ? visualization_msgs::msg::Marker::DELETE : visualization_msgs::msg::Marker::ADD;
      occupied_pub_->publish(occupied);
    }
    if (marker.points.empty()) marker.action = visualization_msgs::msg::Marker::DELETE;
    return marker;
  }

  std::optional<sensor_msgs::msg::PointCloud2> convert(
    const se2_grid_msgs::msg::SE2Grid & msg)
  {
    if (msg.layers.empty() || msg.data.empty()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Received empty SE2Grid");
      return std::nullopt;
    }

    LayerSelection value_selection;
    std::string error;
    if (!resolveLayer(msg, layer_name_, value_selection, error)) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000, "%s", error.c_str());
      return std::nullopt;
    }
    se2_grid_converter::MatrixLayout value_layout;
    if (!prepareLayout(msg, value_selection, value_layout, error)) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000, "%s", error.c_str());
      return std::nullopt;
    }

    const std_msgs::msg::Float32MultiArray * valid_data = nullptr;
    se2_grid_converter::MatrixLayout valid_layout;
    if (!valid_layer_name_.empty()) {
      LayerSelection valid_selection;
      if (!resolveLayer(msg, valid_layer_name_, valid_selection, error) ||
          !prepareLayout(msg, valid_selection, valid_layout, error)) {
        RCLCPP_ERROR_THROTTLE(
          get_logger(), *get_clock(), 2000, "Validity layer is unusable: %s", error.c_str());
        return std::nullopt;
      }
      valid_data = &msg.data[valid_selection.data_index];
    }

    const std_msgs::msg::Float32MultiArray * height_data = nullptr;
    se2_grid_converter::MatrixLayout height_layout;
    if (drape_on_height_ && !height_layer_name_.empty()) {
      LayerSelection height_selection;
      if (!resolveLayer(msg, height_layer_name_, height_selection, error) ||
          !prepareLayout(msg, height_selection, height_layout, error)) {
        RCLCPP_ERROR_THROTTLE(
          get_logger(), *get_clock(), 2000, "Drape height layer is unusable: %s", error.c_str());
        return std::nullopt;
      }
      height_data = &msg.data[height_selection.data_index];
    }

    const auto & info = msg.info;
    const std::size_t rows = value_layout.rows;
    const std::size_t cols = value_layout.cols;
    pcl::PointCloud<pcl::PointXYZRGB> pcl_cloud;
    pcl_cloud.header.frame_id = info.header.frame_id;
    pcl_cloud.height = 1;
    pcl_cloud.is_dense = false;
    pcl_cloud.points.reserve(rows * cols);

    const auto & values = msg.data[value_selection.data_index];
    for (std::size_t row = 0; row < rows; ++row) {
      for (std::size_t col = 0; col < cols; ++col) {
        float value = std::numeric_limits<float>::quiet_NaN();
        if (!se2_grid_converter::readValue(values, value_layout, row, col, value) ||
            !std::isfinite(value)) {
          continue;
        }
        if (valid_data != nullptr) {
          float valid = 0.0F;
          if (!se2_grid_converter::readValue(*valid_data, valid_layout, row, col, valid) ||
              !std::isfinite(valid) || valid < static_cast<float>(valid_threshold_)) {
            continue;
          }
        }

        pcl::PointXYZRGB point;
        point.x = static_cast<float>(se2_grid_converter::cellCoordinate(
          info.pose.position.x, info.length_x, info.pos_resolution, row,
          msg.start_index_row, rows));
        point.y = static_cast<float>(se2_grid_converter::cellCoordinate(
          info.pose.position.y, info.length_y, info.pos_resolution, col,
          msg.start_index_col, cols));
        point.z = static_cast<float>(height_as_z_ ? value + z_offset_ : z_value_);
        if (height_data != nullptr) {
          float height = 0.0F;
          // An explicitly draped product cannot fabricate a zero-height
          // plane for a pose whose terrain height is unavailable.
          if (!se2_grid_converter::readValue(*height_data, height_layout, row, col, height) ||
              !std::isfinite(height)) continue;
          point.z = height + static_cast<float>(z_offset_);
        }
        color(value, point.r, point.g, point.b);
        point.a = 255;
        pcl_cloud.push_back(point);
      }
    }

    pcl_cloud.width = static_cast<std::uint32_t>(pcl_cloud.points.size());
    sensor_msgs::msg::PointCloud2 cloud_msg;
    pcl::toROSMsg(pcl_cloud, cloud_msg);
    cloud_msg.header = info.header;
    return cloud_msg;
  }

  static void probabilityColor(float p, std::uint8_t &r, std::uint8_t &g, std::uint8_t &b)
  {
    // Fixed probability breakpoints, shared by surface and point overlays.
    // Five percent is the default state rejection threshold, not "50% risk".
    const auto rgb=se2_grid_converter::probabilityColor(p);
    r=static_cast<uint8_t>(255*rgb[0]);g=static_cast<uint8_t>(255*rgb[1]);b=static_cast<uint8_t>(255*rgb[2]);
  }

  void color(float value, std::uint8_t & r, std::uint8_t & g, std::uint8_t & b) const
  {
    if (colormap_=="probability") { probabilityColor(value,r,g,b); return; }
    if (!use_color_map_) {
      r = g = b = 190;
      return;
    }
    const float range = std::max(1.0e-6, max_height_ - min_height_);
    float t = (value - static_cast<float>(min_height_)) / static_cast<float>(range);
    if (invert_color_map_) {
      t = 1.0F - t;
    }
    t = std::clamp(t, 0.0F, 1.0F);
    if (colormap_ == "risk" || colormap_ == "risk_gray_red") {
      const float safe = 0.12F;
      const float lethal = 0.82F;
      if (t <= safe) {
        r = 50; g = 190; b = 145;
      } else if (t < 0.55F) {
        const float u = (t - safe) / (0.55F - safe);
        r = static_cast<std::uint8_t>(50.0F + 205.0F * u);
        g = static_cast<std::uint8_t>(190.0F - 45.0F * u);
        b = static_cast<std::uint8_t>(145.0F - 105.0F * u);
      } else if (t < lethal) {
        const float u = (t - 0.55F) / (lethal - 0.55F);
        r = 255;
        g = static_cast<std::uint8_t>(145.0F - 105.0F * u);
        b = static_cast<std::uint8_t>(40.0F - 20.0F * u);
      } else {
        r = 218; g = 48; b = 48;
      }
      return;
    }

    if (colormap_ == "terrain_surface" || colormap_ == "mars") {
      const float stops[][4] = {
        {0.0F, 83.0F, 109.0F, 120.0F},
        {0.30F, 117.0F, 147.0F, 142.0F},
        {0.62F, 172.0F, 172.0F, 151.0F},
        {1.0F, 221.0F, 216.0F, 196.0F},
      };
      int segment = 0;
      while (segment < 2 && t > stops[segment + 1][0]) {
        ++segment;
      }
      const float span = stops[segment + 1][0] - stops[segment][0];
      const float u = span > 0.0F ? (t - stops[segment][0]) / span : 0.0F;
      r = static_cast<std::uint8_t>(stops[segment][1] + u * (stops[segment + 1][1] - stops[segment][1]));
      g = static_cast<std::uint8_t>(stops[segment][2] + u * (stops[segment + 1][2] - stops[segment][2]));
      b = static_cast<std::uint8_t>(stops[segment][3] + u * (stops[segment + 1][3] - stops[segment][3]));
      return;
    }

    const float stops[][4] = {
      {0.0F, 70.0F, 94.0F, 108.0F},
      {0.35F, 105.0F, 137.0F, 133.0F},
      {0.68F, 191.0F, 157.0F, 102.0F},
      {1.0F, 238.0F, 211.0F, 151.0F},
    };
    int segment = 0;
    while (segment < 2 && t > stops[segment + 1][0]) {
      ++segment;
    }
    const float span = stops[segment + 1][0] - stops[segment][0];
    const float u = span > 0.0F ? (t - stops[segment][0]) / span : 0.0F;
    r = static_cast<std::uint8_t>(stops[segment][1] + u * (stops[segment + 1][1] - stops[segment][1]));
    g = static_cast<std::uint8_t>(stops[segment][2] + u * (stops[segment + 1][2] - stops[segment][2]));
    b = static_cast<std::uint8_t>(stops[segment][3] + u * (stops[segment + 1][3] - stops[segment][3]));
  }

  std::string input_topic_;
  std::string output_topic_;
  std::string layer_name_;
  std::string valid_layer_name_;
  std::string odom_topic_;
  std::string height_layer_name_;
  std::string colormap_;
  std::string surface_topic_;
  std::string surface_risk_layer_;
  bool surface_probability_colors_ = false;
  bool surface_use_navigation_height_=false;
  std::string occupied_display_style_="cells";
  bool surface_body_collision_overlay_=true;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr body_pose_surface_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr legend_pub_;
  double valid_threshold_{0.5};
  int yaw_index_{0};
  bool follow_odom_yaw_{false};
  bool drape_on_height_{false};
  bool height_as_z_{true};
  double z_value_{0.0};
  double z_offset_{0.0};
  bool use_color_map_{true};
  bool invert_color_map_{false};
  bool publish_surface_{false};
  double min_height_{-5.0};
  double max_height_{5.0};
  double surface_max_step_{0.65};
  double surface_alpha_{0.92};
  double odom_yaw_{0.0};
  bool have_odom_yaw_{false};
  mutable std::mutex odom_mutex_;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr surface_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr height_surface_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr occupied_pub_;
  rclcpp::Subscription<se2_grid_msgs::msg::SE2Grid>::SharedPtr grid_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SE2GridToPointCloudConverter>());
  rclcpp::shutdown();
  return 0;
}
