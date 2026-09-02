cmake_minimum_required(VERSION 3.5)
project(fake_ugv)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)

include_directories(
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
)

add_executable(kimatic_simulator_node src/poscmd_2_odom.cpp)
add_executable(cmd_vel_relay_node src/cmd_vel_relay.cpp)

ament_target_dependencies(kimatic_simulator_node
  rclcpp
  nav_msgs
  std_msgs
  geometry_msgs
  tf2
  visualization_msgs
  PCL
  Eigen3
  pcl_conversions
)

ament_target_dependencies(cmd_vel_relay_node
  rclcpp
  geometry_msgs
)

target_link_libraries(kimatic_simulator_node
  ${PCL_LIBRARIES}
)

install(TARGETS kimatic_simulator_node cmd_vel_relay_node
  DESTINATION lib/${PROJECT_NAME})

install(PROGRAMS
  scripts/keyboard_ctrl.py
  scripts/startup_speed_hold.py
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY maps/
  DESTINATION share/${PROJECT_NAME}/maps)

install(DIRECTORY meshes/
  DESTINATION share/${PROJECT_NAME}/meshes)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch)

install(DIRECTORY urdf/
  DESTINATION share/${PROJECT_NAME}/urdf)

install(DIRECTORY worlds/
  DESTINATION share/${PROJECT_NAME}/worlds)

ament_package()
