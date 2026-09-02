cmake_minimum_required(VERSION 3.8)
project(vehicle_visualizer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

add_executable(vehicle_visualizer_node
  src/vehicle_visualizer_node.cpp
)

ament_target_dependencies(vehicle_visualizer_node
  rclcpp
  nav_msgs
  geometry_msgs
  visualization_msgs
)

install(TARGETS vehicle_visualizer_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
