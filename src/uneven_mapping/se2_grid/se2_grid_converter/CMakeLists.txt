cmake_minimum_required(VERSION 3.5)
project(se2_grid_converter)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(se2_grid_msgs REQUIRED)
find_package(se2_grid_core REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)

include_directories(
  include
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
)

# Main executable: SE2Grid to PointCloud2 converter
add_executable(se2_grid_to_pointcloud_node 
  src/se2_grid_to_pointcloud_node.cpp
)

ament_target_dependencies(se2_grid_to_pointcloud_node
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  se2_grid_msgs
  se2_grid_core
  Eigen3
  PCL
  pcl_conversions
)

target_link_libraries(se2_grid_to_pointcloud_node
  ${PCL_LIBRARIES}
  Eigen3::Eigen
)

# Install executable
install(TARGETS se2_grid_to_pointcloud_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()
