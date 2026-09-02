cmake_minimum_required(VERSION 3.5)
project(plan_manage)

set(CMAKE_VERBOSE_MAKEFILE "true")
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)

# Find all required packages
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tools REQUIRED)
find_package(path_search REQUIRED)
find_package(decomp_ros_utils REQUIRED)
find_package(mpc_controller REQUIRED)
find_package(se2_grid_core REQUIRED)
find_package(se2_grid_ros REQUIRED)
find_package(se2_grid_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(pcl_conversions REQUIRED)

include_directories(
    include
    ${EIGEN3_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${se2_grid_core_INCLUDE_DIRS}
)

# Define list of dependencies
set(dependencies
    Eigen3
    PCL
    rclcpp
    visualization_msgs
    std_msgs
    geometry_msgs
    sensor_msgs
    nav_msgs
    tf2
    tf2_ros
    tf2_geometry_msgs
    tools
    path_search
    decomp_ros_utils
    mpc_controller
    se2_grid_core
    se2_grid_ros
    se2_grid_msgs
)

# Create plan_manage library from source files
add_library(plan_manage_lib STATIC
    src/plan_manage.cpp
    src/replan_fsm.cpp
)

target_include_directories(plan_manage_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(plan_manage_lib PUBLIC ${dependencies})
target_link_libraries(plan_manage_lib PUBLIC ${PCL_LIBRARIES})

# Create INTERFACE library for compatibility
add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(${PROJECT_NAME} INTERFACE ${dependencies})

# Build plan_node executable (path planning node)
add_executable(plan_node src/plan_node.cpp)
target_link_libraries(plan_node PRIVATE plan_manage_lib)
ament_target_dependencies(plan_node PUBLIC ${dependencies})

add_executable(stable_cloud_republisher src/stable_cloud_republisher.cpp)
ament_target_dependencies(stable_cloud_republisher PUBLIC rclcpp sensor_msgs)

# Build terrain_coloring_node executable (terrain visualization with color)
add_executable(terrain_coloring_node src/terrain_coloring_node.cpp)
ament_target_dependencies(terrain_coloring_node PUBLIC 
    rclcpp
    sensor_msgs
    pcl_conversions
)
target_link_libraries(terrain_coloring_node PUBLIC ${PCL_LIBRARIES})

# Install executables
install(TARGETS terrain_coloring_node plan_node stable_cloud_republisher
  DESTINATION lib/${PROJECT_NAME}
)

# Install libraries
# install(TARGETS plan_manage_lib ${PROJECT_NAME}
#   LIBRARY DESTINATION lib
#   ARCHIVE DESTINATION lib
# )

# Install header files
install(
  DIRECTORY include/${PROJECT_NAME}/
  DESTINATION include/${PROJECT_NAME}
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

# Install launch files
install(
  DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# Install config files
install(
  DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

# Export dependencies
ament_export_dependencies(${dependencies})

ament_package()
