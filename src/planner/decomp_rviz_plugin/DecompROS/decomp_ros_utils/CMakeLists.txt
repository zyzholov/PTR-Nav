cmake_minimum_required(VERSION 3.16)
project(decomp_ros_utils)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(Eigen3 REQUIRED)

# Header-only library (visualization code is ROS1-specific and not migrated)
# TODO: Migrate visualization plugins to ROS2 rviz2 API
add_library(decomp_ros_utils INTERFACE)

target_include_directories(decomp_ros_utils INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(decomp_ros_utils INTERFACE Eigen3::Eigen)

ament_target_dependencies(decomp_ros_utils INTERFACE
  rclcpp
  Eigen3
)

install(
  TARGETS decomp_ros_utils
  EXPORT decomp_ros_utilsTargets
)

install(
  EXPORT decomp_ros_utilsTargets
  FILE decomp_ros_utilsTargets.cmake
  NAMESPACE decomp_ros_utils::
  DESTINATION share/${PROJECT_NAME}/cmake
)

install(DIRECTORY include/ DESTINATION include)

ament_export_targets(decomp_ros_utilsTargets)
ament_export_include_directories(include)
ament_export_dependencies(rclcpp Eigen3)

ament_package()

