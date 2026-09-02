#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <string>

#include <geometry_msgs/msg/twist.hpp>
#include <rclcpp/rclcpp.hpp>

using namespace std::chrono_literals;

class CmdVelRelay : public rclcpp::Node
{
public:
  CmdVelRelay() : Node("cmd_vel_relay")
  {
    const std::string input_topic = declare_parameter<std::string>("input_topic", "/car_0/cmd_vel");
    const std::string output_topic = declare_parameter<std::string>("output_topic", "/car_0/drive_cmd_vel");
    const double publish_rate_hz = std::clamp(declare_parameter<double>("publish_rate_hz", 50.0), 5.0, 200.0);
    timeout_sec_ = std::clamp(declare_parameter<double>("timeout_sec", 0.30), 0.05, 2.0);
    filter_alpha_ = std::clamp(declare_parameter<double>("filter_alpha", 0.35), 0.0, 1.0);
    max_linear_cmd_ = std::max(0.05, declare_parameter<double>("max_linear_cmd", 2.8));
    max_angular_cmd_ = std::max(0.05, declare_parameter<double>("max_angular_cmd", 1.2));

    const auto qos = rclcpp::QoS(rclcpp::KeepLast(5)).reliable();
    pub_ = create_publisher<geometry_msgs::msg::Twist>(output_topic, qos);
    sub_ = create_subscription<geometry_msgs::msg::Twist>(
      input_topic, qos, [this](geometry_msgs::msg::Twist::SharedPtr msg) {
        last_cmd_ = *msg;
        last_cmd_time_ = now();
        have_cmd_ = true;
      });

    const auto period = std::chrono::duration<double>(1.0 / publish_rate_hz);
    timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      std::bind(&CmdVelRelay::onTimer, this));

    RCLCPP_INFO(
      get_logger(), "Relaying %s -> %s at %.1f Hz (timeout=%.2fs, alpha=%.2f)",
      input_topic.c_str(), output_topic.c_str(), publish_rate_hz, timeout_sec_, filter_alpha_);
  }

private:
  static double clampSymmetric(double value, double limit)
  {
    return std::clamp(value, -limit, limit);
  }

  void onTimer()
  {
    geometry_msgs::msg::Twist target;
    if (have_cmd_)
    {
      const double age = (now() - last_cmd_time_).seconds();
      if (age <= timeout_sec_)
      {
        target = last_cmd_;
      }
    }

    target.linear.x = clampSymmetric(target.linear.x, max_linear_cmd_);
    target.angular.z = clampSymmetric(target.angular.z, max_angular_cmd_);

    if (filter_alpha_ > 0.0 && filter_alpha_ < 1.0)
    {
      filtered_cmd_.linear.x += filter_alpha_ * (target.linear.x - filtered_cmd_.linear.x);
      filtered_cmd_.angular.z += filter_alpha_ * (target.angular.z - filtered_cmd_.angular.z);
    }
    else
    {
      filtered_cmd_ = target;
    }

    if (std::abs(filtered_cmd_.linear.x) < 1e-4)
    {
      filtered_cmd_.linear.x = 0.0;
    }
    if (std::abs(filtered_cmd_.angular.z) < 1e-4)
    {
      filtered_cmd_.angular.z = 0.0;
    }

    pub_->publish(filtered_cmd_);
  }

  double timeout_sec_ = 0.30;
  double filter_alpha_ = 0.35;
  double max_linear_cmd_ = 2.8;
  double max_angular_cmd_ = 1.2;
  bool have_cmd_ = false;
  rclcpp::Time last_cmd_time_;
  geometry_msgs::msg::Twist last_cmd_;
  geometry_msgs::msg::Twist filtered_cmd_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr pub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr sub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CmdVelRelay>());
  rclcpp::shutdown();
  return 0;
}
