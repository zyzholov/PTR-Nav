#pragma once
#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <plan_manage/replan_fsm.h>

int main(int argc, char **argv)
{

  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("planner_node");

  ReplanFSM rebo_replan;

  rebo_replan.init(node);

  rclcpp::spin(node);

  return 0;
}

