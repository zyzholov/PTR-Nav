cmake_minimum_required(VERSION 3.5)
project(laser_simulator)
SET(CMAKE_BUILD_TYPE Release) 
# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -use_fast_math")
set(CMAKE_CXX_FLAGS "-std=c++17 -g")

find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(rclcpp REQUIRED)

include_directories(
  SYSTEM
  ${Eigen_INCLUDE_DIRS}
)

add_executable(
  laser_sim_node
  src/laser_sim_node.cpp
)

ament_target_dependencies(laser_sim_node
  rclcpp
  std_msgs
  pcl_ros
  sensor_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
)

target_link_libraries(laser_sim_node
)

add_executable(
  transmit_odom_node
  src/transmit_odom.cpp
)

ament_target_dependencies(transmit_odom_node
  rclcpp
  std_msgs
  geometry_msgs
  nav_msgs
)

target_link_libraries(transmit_odom_node
)

# Install executables
install(TARGETS laser_sim_node transmit_odom_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
