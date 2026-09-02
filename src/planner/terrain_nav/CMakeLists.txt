cmake_minimum_required(VERSION 3.8)
project(terrain_nav)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(se2_grid_core REQUIRED)
find_package(se2_grid_ros REQUIRED)
find_package(se2_grid_msgs REQUIRED)

add_executable(terrain_nav_node src/terrain_nav_node.cpp)

target_include_directories(terrain_nav_node PRIVATE
  ${EIGEN3_INCLUDE_DIRS}
)

ament_target_dependencies(terrain_nav_node
  Eigen3
  rclcpp
  std_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
  tf2
  tf2_geometry_msgs
  se2_grid_core
  se2_grid_ros
  se2_grid_msgs
)

install(TARGETS terrain_nav_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

ament_package()
