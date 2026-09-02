cmake_minimum_required(VERSION 3.5)
project(map_generator)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(ament_index_cpp REQUIRED)
set(Boost_NO_BOOST_CMAKE ON)
find_package(Boost REQUIRED COMPONENTS system filesystem iostreams)

include_directories(
# include
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
)

add_executable (random_forest src/random_forest_sensing.cpp )
ament_target_dependencies(random_forest std_msgs geometry_msgs nav_msgs sensor_msgs pcl_conversions rclcpp)
target_link_libraries(random_forest
  ${PCL_LIBRARIES})

# Terrain loader executable
add_executable (terrain_loader src/terrain_loader_node.cpp)
ament_target_dependencies(terrain_loader std_msgs sensor_msgs pcl_conversions rclcpp ament_index_cpp PCL)
target_link_libraries(terrain_loader
  ${PCL_LIBRARIES}
  Boost::system
  Boost::filesystem
  Boost::iostreams)

# Install executables with proper names for ROS2
install(TARGETS random_forest terrain_loader
  DESTINATION lib/${PROJECT_NAME})

# Install Python scripts
install(PROGRAMS
  src/terrain_publisher.py
  DESTINATION lib/${PROJECT_NAME})

ament_package()