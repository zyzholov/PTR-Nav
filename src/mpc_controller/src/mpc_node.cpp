#include "mpc/mpc.h"
#include <rclcpp/rclcpp.hpp>

int main( int argc, char * argv[] )
{ 
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("mpc_node");

  auto mpc_tracker = std::make_shared<MPC>();
  mpc_tracker->init(node);

  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 3);
  executor.add_node(node);
  executor.spin();

  return 0;
}
