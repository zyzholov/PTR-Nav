#include "se2_grid_converter/display_colors.hpp"

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <chrono>
#include <cmath>
#include <memory>
#include <string>

// Presentation only: color actual returns, retaining their coordinates and
// source timestamp. No synthetic beams, extra points, or navigation outputs.
class LidarDisplay : public rclcpp::Node
{
public:
  LidarDisplay() : Node("lidar_display")
  {
    frame_=declare_parameter("frame_id",std::string("map"));
    const auto odom=declare_parameter("odom_topic",std::string("/Odometry"));
    returns_topic_=declare_parameter("returns_topic",std::string("/cloud_registered_visual"));
    map_pub_=create_publisher<sensor_msgs::msg::PointCloud2>("/terrain_display/lidar_map",1);
    returns_pub_=create_publisher<sensor_msgs::msg::PointCloud2>("/terrain_display/lidar_returns",1);
    odom_sub_=create_subscription<nav_msgs::msg::Odometry>(odom,rclcpp::QoS(1).reliable(),
      [this](nav_msgs::msg::Odometry::ConstSharedPtr msg){
        if(msg->header.frame_id==frame_ && std::isfinite(msg->pose.pose.position.x) &&
            std::isfinite(msg->pose.pose.position.y) && std::isfinite(msg->pose.pose.position.z))odom_=msg;
      });
    map_sub_=create_subscription<sensor_msgs::msg::PointCloud2>("/Laser_map",rclcpp::QoS(1),
      [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg){map_=msg;});
    demand_timer_=create_wall_timer(std::chrono::milliseconds(250),[this]{
      if(returns_pub_->get_subscription_count()) {
        if(!returns_sub_)returns_sub_=create_subscription<sensor_msgs::msg::PointCloud2>(returns_topic_,rclcpp::QoS(1).reliable(),
          [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg){colorize(msg,true);});
      } else returns_sub_.reset();
    });
    timer_=create_wall_timer(std::chrono::milliseconds(500),[this]{colorize(map_,false);});
    live_watchdog_=create_wall_timer(std::chrono::milliseconds(50),[this]{
      if(!live_visible_)return;
      const double age=now().seconds()-rclcpp::Time(last_live_header_.stamp).seconds();
      const double wall_age=std::chrono::duration<double>(
        std::chrono::steady_clock::now()-last_live_received_).count();
      if(age<=.3 && age>=-.05 && wall_age<=.6)return;
      // Clear a held display exactly once. Never restamp/replay old returns
      // to make the sensor appear live, and never change navigation clouds.
      sensor_msgs::msg::PointCloud2 empty;empty.header=last_live_header_;
      sensor_msgs::PointCloud2Modifier modifier(empty);
      modifier.setPointCloud2FieldsByString(2,"xyz","rgb");modifier.resize(0);
      returns_pub_->publish(empty);live_visible_=false;
    });
  }

private:
  void colorize(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &source,bool live)
  {
    const auto pub=live ? returns_pub_ : map_pub_;
    if(!pub->get_subscription_count() || !source || source->header.frame_id!=frame_ || !odom_)return;
    const double now_s=now().seconds(),pose_s=rclcpp::Time(odom_->header.stamp).seconds();
    const double source_s=rclcpp::Time(source->header.stamp).seconds();
    if(now_s-pose_s>.5 || now_s<pose_s-.05 || now_s<source_s-.05 ||
        (live && (now_s-source_s>.3 || std::abs(pose_s-source_s)>.25)))return;
    const auto start=std::chrono::steady_clock::now();
    const size_t count=static_cast<size_t>(source->width)*source->height;
    if(source->is_bigendian || source->point_step==0 || source->row_step!=source->width*source->point_step ||
        source->data.size()<count*source->point_step)return;
    for(const auto *name:{"x","y","z"}) {
      bool valid=false;
      for(const auto &field:source->fields)if(field.name==name && field.datatype==sensor_msgs::msg::PointField::FLOAT32 &&
          field.count==1 && field.offset+4<=source->point_step)valid=true;
      if(!valid)return;
    }
    sensor_msgs::msg::PointCloud2 out;out.header=source->header;
    sensor_msgs::PointCloud2Modifier modifier(out);
    modifier.setPointCloud2FieldsByString(2,"xyz","rgb");modifier.resize(count);
    sensor_msgs::PointCloud2ConstIterator<float> x(*source,"x"),y(*source,"y"),z(*source,"z");
    sensor_msgs::PointCloud2Iterator<float> ox(out,"x"),oy(out,"y"),oz(out,"z");
    sensor_msgs::PointCloud2Iterator<uint8_t> r(out,"r"),g(out,"g"),b(out,"b");
    const auto &p=odom_->pose.pose.position;
    out.is_dense=source->is_dense;
    for(size_t i=0;i<count;++i,++x,++y,++z,++ox,++oy,++oz,++r,++g,++b) {
      *ox=*x;*oy=*y;*oz=*z;
      const auto color=live ? se2_grid_converter::lidarReturnColor(std::hypot(*x-p.x,*y-p.y)) :
        se2_grid_converter::lidarMapColor(*z-p.z);
      *r=static_cast<uint8_t>(255*color[0]);*g=static_cast<uint8_t>(255*color[1]);*b=static_cast<uint8_t>(255*color[2]);
    }
    pub->publish(out);
    if(live) {
      live_visible_=true;last_live_header_=source->header;
      last_live_received_=std::chrono::steady_clock::now();
    }
    RCLCPP_INFO_THROTTLE(get_logger(),*get_clock(),5000,"LiDAR presentation: %s points=%zu color+publish=%.2fms; original XYZ/stamp, not reflectivity",
      live ? "live range" : "accumulated height",count,
      std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
  }
  std::string frame_,returns_topic_;
  nav_msgs::msg::Odometry::ConstSharedPtr odom_;
  sensor_msgs::msg::PointCloud2::ConstSharedPtr map_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_pub_,returns_pub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr map_sub_,returns_sub_;
  rclcpp::TimerBase::SharedPtr timer_,demand_timer_,live_watchdog_;
  std_msgs::msg::Header last_live_header_;
  std::chrono::steady_clock::time_point last_live_received_;
  bool live_visible_=false;
};

int main(int argc,char **argv)
{
  rclcpp::init(argc,argv);rclcpp::spin(std::make_shared<LidarDisplay>());rclcpp::shutdown();return 0;
}
