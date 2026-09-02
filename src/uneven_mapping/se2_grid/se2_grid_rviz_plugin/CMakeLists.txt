cmake_minimum_required(VERSION 3.5.1)
project(se2_grid_rviz_plugin)

set(CMAKE_CXX_STANDARD 17)
add_compile_options(-Wall -Wextra -Wpedantic)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(ament_cmake REQUIRED)
find_package(se2_grid_msgs REQUIRED)
find_package(se2_grid_ros REQUIRED)

# Create header-only INTERFACE library
# Note: Full ROS1→ROS2 rviz plugin migration requires rviz2 API
# TODO: Implement rviz2 plugin API for se2_grid visualization
add_library(${PROJECT_NAME} INTERFACE)

target_include_directories(${PROJECT_NAME} INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(${PROJECT_NAME} INTERFACE
  se2_grid_msgs
  se2_grid_ros
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
  se2_grid_msgs
  se2_grid_ros
)

# Mark cpp header files for installation
install(
  DIRECTORY include/${PROJECT_NAME}/
  DESTINATION include/${PROJECT_NAME}
  FILES_MATCHING PATTERN "*.hpp"
  PATTERN "*.h"
)

ament_package()

