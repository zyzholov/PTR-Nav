#include <algorithm>
#include <cmath>
#include <memory>
#include <mutex>
#include <string>

#include <gazebo/common/common.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo_ros/node.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <rclcpp/rclcpp.hpp>

namespace gazebo
{
class HunterSeVisualSteering : public ModelPlugin
{
public:
  HunterSeVisualSteering() = default;
  ~HunterSeVisualSteering() override = default;

  void Load(physics::ModelPtr model, sdf::ElementPtr sdf) override
  {
    model_ = model;
    world_ = model_->GetWorld();
    ros_node_ = gazebo_ros::Node::Get(sdf);

    wheel_base_ = getDouble(sdf, "wheel_base", 0.549185);
    max_visual_steer_ = std::max(0.01, getDouble(sdf, "max_visual_steer", 0.58));
    max_steer_rate_ = std::max(0.1, getDouble(sdf, "max_steer_rate", 10.0));
    cmd_timeout_ = std::max(0.05, getDouble(sdf, "cmd_timeout", 0.40));
    update_rate_ = std::max(1.0, getDouble(sdf, "update_rate", 100.0));
    steering_wheel_ratio_ = getDouble(sdf, "steering_wheel_ratio", 4.0);
    low_speed_reference_yaw_rate_ =
      std::max(0.1, getDouble(sdf, "low_speed_reference_yaw_rate", 1.20));

    steer_left_joint_ = findJoint(sdf, "left_steering_joint");
    steer_right_joint_ = findJoint(sdf, "right_steering_joint");
    steering_wheel_joint_ = findJoint(sdf, "steering_wheel_joint");

    if (!steer_left_joint_ || !steer_right_joint_)
    {
      RCLCPP_FATAL(
        ros_node_->get_logger(),
        "Hunter visual steering requires both front steering joints.");
      return;
    }

    cmd_sub_ = ros_node_->create_subscription<geometry_msgs::msg::Twist>(
      "cmd_vel", rclcpp::QoS(10),
      std::bind(&HunterSeVisualSteering::onCmd, this, std::placeholders::_1));

    last_sim_time_ = world_->SimTime();
    update_connection_ = event::Events::ConnectWorldUpdateBegin(
      std::bind(&HunterSeVisualSteering::onUpdate, this, std::placeholders::_1));

    RCLCPP_INFO(
      ros_node_->get_logger(),
      "Loaded HunterSeVisualSteering: wheel_base=%.3f max_visual_steer=%.2f max_steer_rate=%.2f",
      wheel_base_, max_visual_steer_, max_steer_rate_);
  }

private:
  static double clampAbs(double value, double limit)
  {
    return std::clamp(value, -limit, limit);
  }

  std::string getString(sdf::ElementPtr sdf, const std::string &name, const std::string &fallback)
  {
    return (sdf && sdf->HasElement(name)) ? sdf->Get<std::string>(name) : fallback;
  }

  double getDouble(sdf::ElementPtr sdf, const std::string &name, double fallback)
  {
    return (sdf && sdf->HasElement(name)) ? sdf->Get<double>(name) : fallback;
  }

  physics::JointPtr findJoint(sdf::ElementPtr sdf, const std::string &tag)
  {
    const auto joint_name = getString(sdf, tag, "");
    if (joint_name.empty())
      return nullptr;

    auto joint = model_->GetJoint(joint_name);
    if (!joint)
    {
      RCLCPP_WARN(
        ros_node_->get_logger(),
        "Joint '%s' configured by <%s> was not found.", joint_name.c_str(), tag.c_str());
    }
    return joint;
  }

  void onCmd(const geometry_msgs::msg::Twist::SharedPtr msg)
  {
    const double linear_x = std::isfinite(msg->linear.x) ? msg->linear.x : 0.0;
    const double yaw_rate = std::isfinite(msg->angular.z) ? msg->angular.z : 0.0;

    double target_steer = 0.0;
    if (std::abs(yaw_rate) < 1e-4)
    {
      target_steer = 0.0;
    }
    else if (std::abs(linear_x) > 0.05)
    {
      // In planar_visual mode cmd_vel.angular.z is a yaw-rate command. Convert
      // yaw-rate back to the equivalent Ackermann steering angle by curvature:
      //   yaw_rate = v * tan(delta) / L  ->  delta = atan(L * yaw_rate / v)
      // Do not use atan2(..., v): when v < 0 and yaw_rate == 0 it returns pi,
      // which makes straight reverse commands look like full steering lock.
      target_steer = std::atan(wheel_base_ * yaw_rate / linear_x);
    }
    else if (std::abs(yaw_rate) > 1e-4)
    {
      const double ratio = std::clamp(std::abs(yaw_rate) / low_speed_reference_yaw_rate_, 0.0, 1.0);
      target_steer = std::copysign(max_visual_steer_ * ratio, yaw_rate);
    }

    std::scoped_lock lock(cmd_mutex_);
    target_steer_ = clampAbs(target_steer, max_visual_steer_);
    last_cmd_stamp_ = ros_node_->get_clock()->now();
    has_cmd_ = true;
  }

  void setJointPosition(const physics::JointPtr &joint, double value)
  {
    if (!joint)
      return;

    try
    {
      joint->SetPosition(0, value);
    }
    catch (...)
    {
    }
  }

  void onUpdate(const common::UpdateInfo &info)
  {
    const auto now = info.simTime;
    const double dt = (now - last_sim_time_).Double();
    if (dt <= 0.0)
      return;

    last_sim_time_ = now;
    const double min_update_dt = 1.0 / update_rate_;
    update_accumulator_ += dt;
    if (update_accumulator_ + 1e-9 < min_update_dt)
      return;

    const double step_dt = update_accumulator_;
    update_accumulator_ = 0.0;

    double target_steer = 0.0;
    {
      std::scoped_lock lock(cmd_mutex_);
      target_steer = target_steer_;
      if (!has_cmd_ || (ros_node_->get_clock()->now() - last_cmd_stamp_).seconds() > cmd_timeout_)
        target_steer = 0.0;
    }

    const double ds_max = max_steer_rate_ * step_dt;
    current_steer_ += std::clamp(target_steer - current_steer_, -ds_max, ds_max);
    current_steer_ = clampAbs(current_steer_, max_visual_steer_);

    setJointPosition(steer_left_joint_, current_steer_);
    setJointPosition(steer_right_joint_, current_steer_);
    setJointPosition(steering_wheel_joint_, -current_steer_ * steering_wheel_ratio_);
  }

  physics::ModelPtr model_;
  physics::WorldPtr world_;
  gazebo_ros::Node::SharedPtr ros_node_;
  event::ConnectionPtr update_connection_;

  physics::JointPtr steer_left_joint_;
  physics::JointPtr steer_right_joint_;
  physics::JointPtr steering_wheel_joint_;

  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_sub_;
  std::mutex cmd_mutex_;
  rclcpp::Time last_cmd_stamp_{0, 0, RCL_ROS_TIME};
  bool has_cmd_ = false;
  double target_steer_ = 0.0;
  double current_steer_ = 0.0;

  double wheel_base_ = 0.549185;
  double max_visual_steer_ = 0.58;
  double max_steer_rate_ = 10.0;
  double cmd_timeout_ = 0.40;
  double update_rate_ = 100.0;
  double steering_wheel_ratio_ = 4.0;
  double low_speed_reference_yaw_rate_ = 1.20;

  common::Time last_sim_time_;
  double update_accumulator_ = 0.0;
};

GZ_REGISTER_MODEL_PLUGIN(HunterSeVisualSteering)
}  // namespace gazebo
