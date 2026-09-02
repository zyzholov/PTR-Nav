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
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_broadcaster.h>

namespace gazebo
{
class HunterSeKinematicDrive : public ModelPlugin
{
public:
  HunterSeKinematicDrive() = default;
  ~HunterSeKinematicDrive() override = default;

  void Load(physics::ModelPtr model, sdf::ElementPtr sdf) override
  {
    model_ = model;
    world_ = model_->GetWorld();
    ros_node_ = gazebo_ros::Node::Get(sdf);

    robot_base_frame_ = getString(sdf, "robot_base_frame", "base_link");
    odometry_frame_ = getString(sdf, "odometry_frame", "odom");

    wheel_base_ = getDouble(sdf, "wheel_base", 0.549185);
    max_steer_ = getDouble(sdf, "max_steer", 0.60);
    max_wheel_steer_ = getDouble(sdf, "max_wheel_steer", max_steer_);
    max_speed_ = getDouble(sdf, "max_speed", 2.50);
    max_accel_ = getDouble(sdf, "max_accel", 1.20);
    max_steer_rate_ = getDouble(sdf, "max_steer_rate", 1.20);
    cmd_timeout_ = getDouble(sdf, "cmd_timeout", 0.40);
    wheel_radius_ = getDouble(sdf, "wheel_radius", 0.1349);
    track_width_ = getDouble(sdf, "track_width", 0.50);
    steering_wheel_ratio_ = getDouble(sdf, "steering_wheel_ratio", 4.0);
    update_rate_ = std::max(1.0, getDouble(sdf, "update_rate", 100.0));
    publish_rate_ = std::max(1.0, getDouble(sdf, "publish_rate", 50.0));
    publish_odom_tf_ = getBool(sdf, "publish_odom_tf", false);

    steering_position_kp_ = getDouble(sdf, "steering_position_kp", 35.0);
    steering_position_kd_ = getDouble(sdf, "steering_position_kd", 3.0);
    steering_motor_effort_ = getDouble(sdf, "steering_motor_effort", 120.0);
    steering_wheel_position_kp_ = getDouble(sdf, "steering_wheel_position_kp", 4.0);
    wheel_velocity_kp_ = getDouble(sdf, "wheel_velocity_kp", 18.0);
    wheel_motor_effort_ = getDouble(sdf, "wheel_motor_effort", 32.0);
    front_wheel_drive_scale_ = getDouble(sdf, "front_wheel_drive_scale", 0.0);
    steering_deadband_ = std::max(0.0, getDouble(sdf, "steering_deadband", 0.01));
    idle_hold_enable_ = getBool(sdf, "idle_hold_enable", true);
    idle_hold_cmd_speed_threshold_ = std::max(0.0, getDouble(sdf, "idle_hold_cmd_speed_threshold", 0.025));
    idle_hold_cmd_steer_threshold_ = std::max(0.0, getDouble(sdf, "idle_hold_cmd_steer_threshold", 0.025));
    idle_hold_linear_speed_threshold_ = std::max(0.0, getDouble(sdf, "idle_hold_linear_speed_threshold", 0.055));
    idle_hold_yaw_rate_threshold_ = std::max(0.0, getDouble(sdf, "idle_hold_yaw_rate_threshold", 0.050));
    idle_hold_linear_damping_ = std::max(0.0, getDouble(sdf, "idle_hold_linear_damping", 55.0));
    idle_hold_yaw_damping_ = std::max(0.0, getDouble(sdf, "idle_hold_yaw_damping", 35.0));
    idle_hold_wheel_brake_kp_ = std::max(0.0, getDouble(sdf, "idle_hold_wheel_brake_kp", 28.0));
    idle_hold_wheel_brake_effort_ = std::max(0.0, getDouble(sdf, "idle_hold_wheel_brake_effort", 150.0));

    steer_left_joint_ = findJoint(sdf, "left_steering_joint");
    steer_right_joint_ = findJoint(sdf, "right_steering_joint");
    steering_wheel_joint_ = findJoint(sdf, "steering_wheel_joint");
    front_left_wheel_joint_ = findJoint(sdf, "front_left_joint");
    front_right_wheel_joint_ = findJoint(sdf, "front_right_joint");
    rear_left_wheel_joint_ = findJoint(sdf, "rear_left_joint");
    rear_right_wheel_joint_ = findJoint(sdf, "rear_right_joint");

    if (!steer_left_joint_ || !steer_right_joint_)
    {
      RCLCPP_FATAL(
        ros_node_->get_logger(),
        "Hunter joint drive requires both front steering joints.");
      return;
    }

    cmd_sub_ = ros_node_->create_subscription<geometry_msgs::msg::Twist>(
      "cmd_vel", rclcpp::QoS(rclcpp::KeepLast(1)).best_effort(),
      std::bind(&HunterSeKinematicDrive::onCmd, this, std::placeholders::_1));
    odom_pub_ = ros_node_->create_publisher<nav_msgs::msg::Odometry>(
      "odom", rclcpp::QoS(rclcpp::KeepLast(1)).reliable());
    if (publish_odom_tf_)
      tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(ros_node_);

    last_sim_time_ = world_->SimTime();
    last_odom_pub_time_ = last_sim_time_;

    update_connection_ = event::Events::ConnectWorldUpdateBegin(
      std::bind(&HunterSeKinematicDrive::onUpdate, this, std::placeholders::_1));

    RCLCPP_INFO(
      ros_node_->get_logger(),
      "Loaded HunterSeKinematicDrive as joint-space drive: wheel_base=%.3f track=%.3f max_speed=%.2f center_steer=%.2f wheel_steer=%.2f steer_rate=%.2f idle_hold=%s",
      wheel_base_, track_width_, max_speed_, max_steer_, max_wheel_steer_, max_steer_rate_,
      idle_hold_enable_ ? "on" : "off");
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

  bool getBool(sdf::ElementPtr sdf, const std::string &name, bool fallback)
  {
    return (sdf && sdf->HasElement(name)) ? sdf->Get<bool>(name) : fallback;
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
    std::scoped_lock lock(cmd_mutex_);
    target_speed_ = clampAbs(std::isfinite(msg->linear.x) ? msg->linear.x : 0.0, max_speed_);
    target_steer_ = clampAbs(std::isfinite(msg->angular.z) ? msg->angular.z : 0.0, max_steer_);
    if (std::abs(target_steer_) < steering_deadband_)
      target_steer_ = 0.0;
    last_cmd_sim_time_ = world_ ? world_->SimTime() : common::Time();
    has_cmd_ = true;
  }

  static double normalizeAngle(double value)
  {
    while (value > M_PI)
      value -= 2.0 * M_PI;
    while (value < -M_PI)
      value += 2.0 * M_PI;
    return value;
  }

  void setMotorTorque(const physics::JointPtr &joint, double torque)
  {
    if (!joint)
      return;

    joint->SetForce(0, torque);
  }

  void setSteeringTarget(const physics::JointPtr &joint, double target, double effort_scale = 1.0)
  {
    if (!joint)
      return;

    const double current = finiteOrZero(joint->Position(0));
    const double current_velocity = finiteOrZero(joint->GetVelocity(0));
    const double error = normalizeAngle(target - current);
    const double torque = std::clamp(
      steering_position_kp_ * error - steering_position_kd_ * current_velocity,
      -std::abs(steering_motor_effort_) * effort_scale,
      std::abs(steering_motor_effort_) * effort_scale);
    setMotorTorque(joint, torque);
  }

  void setWheelTarget(const physics::JointPtr &joint, double angular_velocity)
  {
    if (!joint)
      return;

    const double current_velocity = finiteOrZero(joint->GetVelocity(0));
    const double torque = std::clamp(
      wheel_velocity_kp_ * (angular_velocity - current_velocity),
      -std::abs(wheel_motor_effort_),
      std::abs(wheel_motor_effort_));
    setMotorTorque(joint, torque);
  }

  void applyWheelBrake(const physics::JointPtr &joint)
  {
    if (!joint)
      return;

    const double velocity = finiteOrZero(joint->GetVelocity(0));
    const double torque = std::clamp(
      -idle_hold_wheel_brake_kp_ * velocity,
      -idle_hold_wheel_brake_effort_,
      idle_hold_wheel_brake_effort_);
    setMotorTorque(joint, torque);
  }

  void applyIdleHold(double dt)
  {
    if (!idle_hold_enable_ || !model_)
      return;

    applyWheelBrake(front_left_wheel_joint_);
    applyWheelBrake(front_right_wheel_joint_);
    applyWheelBrake(rear_left_wheel_joint_);
    applyWheelBrake(rear_right_wheel_joint_);

    auto base_link = model_->GetLink(robot_base_frame_);
    if (!base_link)
      base_link = model_->GetLink();
    if (!base_link)
      return;

    const auto linear_vel = base_link->WorldLinearVel();
    const auto angular_vel = base_link->WorldAngularVel();
    const double planar_speed = std::hypot(linear_vel.X(), linear_vel.Y());
    const double yaw_rate = std::abs(angular_vel.Z());

    if (planar_speed <= idle_hold_linear_speed_threshold_ &&
        yaw_rate <= idle_hold_yaw_rate_threshold_)
    {
      current_speed_ = 0.0;
      return;
    }

    // Apply gentle physical damping instead of SetLinearVel/SetAngularVel. Direct
    // velocity snapping fights the ODE contact solver on triangle terrain and
    // produces the visible idle jitter.
    base_link->AddForce(ignition::math::Vector3d(
      -idle_hold_linear_damping_ * linear_vel.X(),
      -idle_hold_linear_damping_ * linear_vel.Y(),
      0.0));
    base_link->AddTorque(ignition::math::Vector3d(
      0.0,
      0.0,
      -idle_hold_yaw_damping_ * angular_vel.Z()));
  }

  double steeringForWheel(double center_steer, double lateral_offset) const
  {
    if (std::abs(center_steer) < 1e-5 || std::abs(wheel_base_) < 1e-6)
      return center_steer;

    const double curvature = std::tan(center_steer) / wheel_base_;
    const double denominator = 1.0 - curvature * lateral_offset;
    if (std::abs(denominator) < 1e-3)
      return clampAbs(center_steer, max_wheel_steer_);

    return clampAbs(std::atan2(wheel_base_ * curvature, denominator), max_wheel_steer_);
  }

  double finiteOrZero(double value) const
  {
    return std::isfinite(value) ? value : 0.0;
  }

  void publishOdometry(
    const ignition::math::Pose3d &pose,
    double forward_speed,
    double yaw_rate)
  {
    auto stamp = ros_node_->get_clock()->now();

    nav_msgs::msg::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = odometry_frame_;
    odom.child_frame_id = robot_base_frame_;
    odom.pose.pose.position.x = pose.Pos().X();
    odom.pose.pose.position.y = pose.Pos().Y();
    odom.pose.pose.position.z = pose.Pos().Z();
    odom.pose.pose.orientation.x = pose.Rot().X();
    odom.pose.pose.orientation.y = pose.Rot().Y();
    odom.pose.pose.orientation.z = pose.Rot().Z();
    odom.pose.pose.orientation.w = pose.Rot().W();
    // nav_msgs/Odometry twist is expressed in child_frame_id. Publishing world
    // X/Y velocity here made the planner think forward speed changed with yaw,
    // which corrupts replanning initial conditions during turns.
    odom.twist.twist.linear.x = forward_speed;
    odom.twist.twist.linear.y = 0.0;
    odom.twist.twist.linear.z = 0.0;
    odom.twist.twist.angular.z = yaw_rate;
    odom_pub_->publish(odom);

    if (publish_odom_tf_ && tf_broadcaster_)
    {
      geometry_msgs::msg::TransformStamped tf_msg;
      tf_msg.header = odom.header;
      tf_msg.child_frame_id = robot_base_frame_;
      tf_msg.transform.translation.x = odom.pose.pose.position.x;
      tf_msg.transform.translation.y = odom.pose.pose.position.y;
      tf_msg.transform.translation.z = odom.pose.pose.position.z;
      tf_msg.transform.rotation = odom.pose.pose.orientation;
      tf_broadcaster_->sendTransform(tf_msg);
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

    double target_speed = 0.0;
    double target_steer = 0.0;
    {
      std::scoped_lock lock(cmd_mutex_);
      target_speed = target_speed_;
      target_steer = target_steer_;
      if (!has_cmd_ || (now - last_cmd_sim_time_).Double() > cmd_timeout_)
      {
        target_speed = 0.0;
        target_steer = 0.0;
      }
    }

    const double dv_max = std::max(0.0, max_accel_) * step_dt;
    const double ds_max = std::max(0.0, max_steer_rate_) * step_dt;
    current_speed_ += std::clamp(target_speed - current_speed_, -dv_max, dv_max);
    current_steer_ += std::clamp(target_steer - current_steer_, -ds_max, ds_max);
    current_speed_ = clampAbs(current_speed_, max_speed_);
    current_steer_ = clampAbs(current_steer_, max_steer_);
    const bool idle_command =
      std::abs(target_speed) <= idle_hold_cmd_speed_threshold_ &&
      std::abs(target_steer) <= idle_hold_cmd_steer_threshold_;
    if (idle_command)
      current_speed_ = 0.0;

    const double half_track = std::max(0.05, track_width_ * 0.5);
    const double left_steer = steeringForWheel(current_steer_, half_track);
    const double right_steer = steeringForWheel(current_steer_, -half_track);
    setSteeringTarget(steer_left_joint_, left_steer);
    setSteeringTarget(steer_right_joint_, right_steer);
    setSteeringTarget(steering_wheel_joint_, -current_steer_ * steering_wheel_ratio_, 0.04);

    const double desired_yaw_rate = (std::abs(wheel_base_) > 1e-6)
      ? current_speed_ * std::tan(current_steer_) / wheel_base_
      : 0.0;
    const double left_speed = current_speed_ - desired_yaw_rate * half_track;
    const double right_speed = current_speed_ + desired_yaw_rate * half_track;
    if (wheel_radius_ > 1e-4)
    {
      if (!idle_command)
      {
        if (front_wheel_drive_scale_ > 1e-6)
          setWheelTarget(front_left_wheel_joint_, front_wheel_drive_scale_ * left_speed / wheel_radius_);
        setWheelTarget(rear_left_wheel_joint_, left_speed / wheel_radius_);
        if (front_wheel_drive_scale_ > 1e-6)
          setWheelTarget(front_right_wheel_joint_, front_wheel_drive_scale_ * right_speed / wheel_radius_);
        setWheelTarget(rear_right_wheel_joint_, right_speed / wheel_radius_);
      }
    }

    if (idle_command)
      applyIdleHold(step_dt);

    if ((now - last_odom_pub_time_).Double() + 1e-9 >= 1.0 / publish_rate_)
    {
      const auto pose = model_->WorldPose();
      const auto linear_vel = model_->WorldLinearVel();
      const auto angular_vel = model_->WorldAngularVel();
      const double yaw = pose.Rot().Yaw();
      const double forward_speed = finiteOrZero(
        std::cos(yaw) * linear_vel.X() + std::sin(yaw) * linear_vel.Y());
      publishOdometry(pose, forward_speed, finiteOrZero(angular_vel.Z()));
      last_odom_pub_time_ = now;
    }
  }

  physics::ModelPtr model_;
  physics::WorldPtr world_;
  gazebo_ros::Node::SharedPtr ros_node_;
  event::ConnectionPtr update_connection_;

  physics::JointPtr steer_left_joint_;
  physics::JointPtr steer_right_joint_;
  physics::JointPtr steering_wheel_joint_;
  physics::JointPtr front_left_wheel_joint_;
  physics::JointPtr front_right_wheel_joint_;
  physics::JointPtr rear_left_wheel_joint_;
  physics::JointPtr rear_right_wheel_joint_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  std::mutex cmd_mutex_;
  common::Time last_cmd_sim_time_;
  bool has_cmd_ = false;
  double target_speed_ = 0.0;
  double target_steer_ = 0.0;
  double current_speed_ = 0.0;
  double current_steer_ = 0.0;
  double wheel_base_ = 0.549185;
  double max_steer_ = 0.60;
  double max_wheel_steer_ = 0.60;
  double max_speed_ = 2.50;
  double max_accel_ = 1.20;
  double max_steer_rate_ = 1.20;
  double cmd_timeout_ = 0.40;
  double wheel_radius_ = 0.1349;
  double track_width_ = 0.50;
  double steering_wheel_ratio_ = 4.0;
  double update_rate_ = 100.0;
  double publish_rate_ = 50.0;
  bool publish_odom_tf_ = false;

  double steering_position_kp_ = 8.0;
  double steering_position_kd_ = 0.35;
  double steering_motor_effort_ = 160.0;
  double steering_wheel_position_kp_ = 6.0;
  double wheel_velocity_kp_ = 18.0;
  double wheel_motor_effort_ = 35.0;
  double front_wheel_drive_scale_ = 0.0;
  double steering_deadband_ = 0.01;
  bool idle_hold_enable_ = true;
  double idle_hold_cmd_speed_threshold_ = 0.025;
  double idle_hold_cmd_steer_threshold_ = 0.025;
  double idle_hold_linear_speed_threshold_ = 0.055;
  double idle_hold_yaw_rate_threshold_ = 0.050;
  double idle_hold_linear_damping_ = 55.0;
  double idle_hold_yaw_damping_ = 35.0;
  double idle_hold_wheel_brake_kp_ = 28.0;
  double idle_hold_wheel_brake_effort_ = 150.0;

  std::string robot_base_frame_ = "base_link";
  std::string odometry_frame_ = "odom";

  common::Time last_sim_time_;
  common::Time last_odom_pub_time_;
  double update_accumulator_ = 0.0;
};

GZ_REGISTER_MODEL_PLUGIN(HunterSeKinematicDrive)
}  // namespace gazebo
