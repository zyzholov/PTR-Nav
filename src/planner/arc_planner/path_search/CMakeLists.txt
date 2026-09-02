cmake_minimum_required(VERSION 3.5)
project(path_search)

set(CMAKE_VERBOSE_MAKEFILE "true")
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tools REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(se2_grid_core REQUIRED)
find_package(se2_grid_ros REQUIRED)
find_package(se2_grid_msgs REQUIRED)
find_package(rclcpp REQUIRED)

# Create library with source files
# ROS2 migration of kinodynamic A* and A* implementations complete
add_library(${PROJECT_NAME}
  src/kino_astar.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# Add compiler flags to handle ROS1->ROS2 migration issues in tools/gridmap.hpp
target_compile_definitions(${PROJECT_NAME} PUBLIC
  GRIDMAP_ROS2_COMPAT=1
)

ament_target_dependencies(${PROJECT_NAME} PUBLIC
  Eigen3
  PCL
  visualization_msgs
  std_msgs
  se2_grid_core
  se2_grid_ros
  se2_grid_msgs
  rclcpp
  pcl_conversions
  tools
)

# Export targets
install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}Targets
)

install(
  EXPORT ${PROJECT_NAME}Targets
  FILE ${PROJECT_NAME}Targets.cmake
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION share/${PROJECT_NAME}/cmake
)

ament_export_targets(${PROJECT_NAME}Targets)
ament_export_libraries(${PROJECT_NAME})
ament_export_dependencies(
  Eigen3
  PCL
  visualization_msgs
  std_msgs
  se2_grid_core
  se2_grid_ros
  se2_grid_msgs
  rclcpp
)

# Mark cpp header files for installation
install(
  DIRECTORY include/path_searching/
  DESTINATION include/path_searching
  FILES_MATCHING 
  PATTERN "*.hpp"
  PATTERN "*.h"
)

ament_package()
