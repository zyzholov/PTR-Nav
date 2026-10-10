cmake_minimum_required(VERSION 3.5.1)

project(terrain_analyzer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(TERRAIN_ANALYZER_BUILD_CUDA "Build the native CUDA terrain analyzer" ON)

find_package(ament_cmake REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)
find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(se2_grid_core REQUIRED)
find_package(se2_grid_ros REQUIRED)
find_package(se2_grid_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(ament_index_cpp REQUIRED)

# Build ROS2 terrain analyzer node (executable)
add_executable(terrain_analyzer 
  src/terrain_analyzer_ros2_node.cpp
)

target_include_directories(terrain_analyzer PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

ament_target_dependencies(terrain_analyzer
  rclcpp
  sensor_msgs
  nav_msgs
  se2_grid_msgs
  geometry_msgs
  visualization_msgs
  std_msgs
  Eigen3
  PCL
  pcl_conversions
  ament_index_cpp
  se2_grid_core
  se2_grid_ros
)

set(TERRAIN_ANALYZER_TARGETS terrain_analyzer)

if(TERRAIN_ANALYZER_BUILD_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    # CMake otherwise leaves CUDA_ARCHITECTURES empty and emits toolchain-dependent
    # code. Prefer the installed GPU on modern CMake, while keeping an override for
    # cross-compilation (for example: -DCMAKE_CUDA_ARCHITECTURES=86).
    if(NOT CMAKE_CUDA_ARCHITECTURES)
      if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
        set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures" FORCE)
      else()
        set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "CUDA architectures" FORCE)
      endif()
    endif()

    add_executable(terrain_analyzer_gpu
      src/terrain_analyzer_gpu_node.cpp
      src/TerrainBank.cu
    )
    set_target_properties(terrain_analyzer_gpu PROPERTIES
      CUDA_STANDARD 14
      CUDA_STANDARD_REQUIRED ON
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )
    target_include_directories(terrain_analyzer_gpu PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${OpenCV_INCLUDE_DIRS}
    )
    target_compile_options(terrain_analyzer_gpu PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
    target_link_libraries(terrain_analyzer_gpu
      CUDA::cudart
      ${OpenCV_LIBS}
    )
    ament_target_dependencies(terrain_analyzer_gpu
      rclcpp
      sensor_msgs
      nav_msgs
      se2_grid_msgs
      Eigen3
      PCL
      pcl_conversions
      cv_bridge
      se2_grid_core
      se2_grid_ros
    )
    list(APPEND TERRAIN_ANALYZER_TARGETS terrain_analyzer_gpu)
    message(STATUS "terrain_analyzer: native CUDA backend enabled")
  else()
    message(WARNING "terrain_analyzer: CUDA compiler not found; building CPU backend only")
  endif()
endif()

install(
  TARGETS ${TERRAIN_ANALYZER_TARGETS}
  DESTINATION lib/${PROJECT_NAME}
)

# Install parameter files
install(
  DIRECTORY params/
  DESTINATION share/${PROJECT_NAME}/params
)

# Install header files for library users
install(
  DIRECTORY include/${PROJECT_NAME}/
  DESTINATION include/${PROJECT_NAME}
  FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
)

ament_package()
