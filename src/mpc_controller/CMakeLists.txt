cmake_minimum_required(VERSION 3.5)
project(mpc_controller)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CASADI REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/SinglePoly.msg"
  "msg/PolyTraj.msg"
  "msg/SE2Traj.msg"
  "msg/SinglePolyAC.msg"
  "msg/PolyTrajAC.msg"
  "msg/DPtrajContainer.msg"
  "msg/DesiredState.msg"
  DEPENDENCIES std_msgs geometry_msgs builtin_interfaces
)

include_directories( 
  SYSTEM 
  include 
  ${Eigen3_INCLUDE_DIRS} 
  ${PROJECT_SOURCE_DIR}/include
  ${CASADI_INCLUDE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}/rosidl_generator_cpp
)

add_library(${PROJECT_NAME}_lib
  src/mpc.cpp
)

# Get typesupport target for messages
rosidl_get_typesupport_target(cpp_typesupport_target "${PROJECT_NAME}" "rosidl_typesupport_cpp")

# Add dependency on generated messages
add_dependencies(${PROJECT_NAME}_lib ${PROJECT_NAME}__cpp)

# Link generated message target
target_link_libraries(${PROJECT_NAME}_lib "${cpp_typesupport_target}")

ament_target_dependencies(${PROJECT_NAME}_lib
  rclcpp
  std_msgs
  nav_msgs
  geometry_msgs
  visualization_msgs
  Eigen3
)

target_link_libraries(${PROJECT_NAME}_lib
  ${CASADI_LIBRARIES}
)

add_executable(${PROJECT_NAME}_node
  src/mpc_node.cpp
)
ament_target_dependencies(${PROJECT_NAME}_node
  rclcpp
  std_msgs
  nav_msgs
  geometry_msgs
  visualization_msgs
)
target_link_libraries(${PROJECT_NAME}_node
  ${PROJECT_NAME}_lib
  "${cpp_typesupport_target}"
)

install(TARGETS ${PROJECT_NAME}_node
  DESTINATION lib/${PROJECT_NAME})

# Install the shared library so the runtime loader can find libmpc_controller_lib.so
install(TARGETS ${PROJECT_NAME}_lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION lib
)

# Install parameter files
install(DIRECTORY params/
  DESTINATION share/${PROJECT_NAME}/params)

# Install launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch)

# Install RViz config
install(DIRECTORY rviz/
  DESTINATION share/${PROJECT_NAME}/rviz)

ament_package()