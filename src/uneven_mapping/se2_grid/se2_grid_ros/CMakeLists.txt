cmake_minimum_required(VERSION 3.16)
project(se2_grid_ros)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(se2_grid_core REQUIRED)
find_package(se2_grid_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(tf2 REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(rosidl_runtime_cpp REQUIRED)
find_package(rosidl_default_runtime REQUIRED)

add_library(${PROJECT_NAME} 
  src/SE2GridRosConverter.cpp
  src/SE2GridMsgHelpers.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# All dependencies through ament
ament_target_dependencies(${PROJECT_NAME} PUBLIC
  rclcpp
  se2_grid_core
  se2_grid_msgs
  sensor_msgs
  nav_msgs
  std_msgs
  geometry_msgs
  cv_bridge
  tf2
  visualization_msgs
  rosidl_runtime_cpp
  Eigen3
)

# Export the INTERFACE library target
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
ament_export_libraries(${PROJECT_NAME})
ament_export_dependencies(
  rclcpp
  se2_grid_core
  se2_grid_msgs
  sensor_msgs
  nav_msgs
  std_msgs
  geometry_msgs
  cv_bridge
  tf2
  visualization_msgs
  rosidl_runtime_cpp
  Eigen3
)

ament_package()

#############
## Install ##
#############

# Mark executables and/or libraries for installation
install(
  TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Mark cpp header files for installation
install(
  DIRECTORY include/${PROJECT_NAME}/
  DESTINATION include/${PROJECT_NAME}
  FILES_MATCHING PATTERN "*.hpp"
)
