/*
  MIT License

  Copyright (c) 2021 Hongkai Ye (kyle_yeh@163.com, hkye@zju.edu.cn)

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
*/
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>

std::shared_ptr<rclcpp::Node> g_node;
std::shared_ptr<rclcpp::Publisher<nav_msgs::msg::Odometry>> _odom_pub;
std::shared_ptr<rclcpp::Subscription<nav_msgs::msg::Odometry>> _odom_sub;
nav_msgs::msg::Odometry _odom;

void rcvOdometryCallbck(const nav_msgs::msg::Odometry::SharedPtr laser_odom)
{
  _odom = *laser_odom;
  _odom.header.stamp = g_node->now();
  _odom_pub->publish(_odom);
}

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  g_node = std::make_shared<rclcpp::Node>("transmit_odom");

  _odom_sub = g_node->create_subscription<nav_msgs::msg::Odometry>(
      "/drone0/odom", 1, rcvOdometryCallbck);
  _odom_pub = g_node->create_publisher<nav_msgs::msg::Odometry>("/my_odometry", 1);

  rclcpp::spin(g_node);
  rclcpp::shutdown();
  return 0;
}
