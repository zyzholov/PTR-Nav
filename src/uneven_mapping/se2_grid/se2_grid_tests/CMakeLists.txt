cmake_minimum_required(VERSION 3.5.1)
project(se2_grid_tests)

set(CMAKE_CXX_STANDARD 17)
add_compile_options(-Wall -Wextra -Wpedantic)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

## Find catkin macros and libraries
find_package(ament_cmake REQUIRED)
find_package(se2_grid_core REQUIRED)
find_package(se2_grid_msgs REQUIRED)
find_package(se2_grid_ros REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rclcpp REQUIRED)

# Create INTERFACE library - header-only for now
# TODO: Full ROS2 migration of test nodes (message header issues)
add_library(${PROJECT_NAME} INTERFACE)

target_include_directories(${PROJECT_NAME} INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(${PROJECT_NAME} INTERFACE
  rclcpp
  se2_grid_core
  se2_grid_msgs
  se2_grid_ros
  geometry_msgs
  sensor_msgs
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

# Export package
ament_export_targets(${PROJECT_NAME}Targets)
ament_export_dependencies(
  rclcpp
  se2_grid_core
  se2_grid_msgs
  se2_grid_ros
  geometry_msgs
  sensor_msgs
)

# Mark other files for installation
install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
