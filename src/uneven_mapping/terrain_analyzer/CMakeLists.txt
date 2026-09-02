cmake_minimum_required(VERSION 3.5.1)

project(terrain_analyzer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(ament_cmake REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)
find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(se2_grid_core REQUIRED)
find_package(se2_grid_ros REQUIRED)
find_package(se2_grid_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(ament_index_cpp REQUIRED)

# Build ROS2 terrain analyzer node (executable)
add_executable(terrain_analyzer 
  src/terrain_analyzer_ros2_node.cpp
)

target_include_directories(terrain_analyzer PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

ament_target_dependencies(terrain_analyzer
  rclcpp
  sensor_msgs
  nav_msgs
  se2_grid_msgs
  geometry_msgs
  visualization_msgs
  std_msgs
  Eigen3
  PCL
  pcl_conversions
  ament_index_cpp
  se2_grid_core
  se2_grid_ros
)

# Install executable
install(
  TARGETS terrain_analyzer
  DESTINATION lib/${PROJECT_NAME}
)

# Install parameter files
install(
  DIRECTORY params/
  DESTINATION share/${PROJECT_NAME}/params
)

# Install header files for library users
install(
  DIRECTORY include/${PROJECT_NAME}/
  DESTINATION include/${PROJECT_NAME}
  FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
)

ament_package()
