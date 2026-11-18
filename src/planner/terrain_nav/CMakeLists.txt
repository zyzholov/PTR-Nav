cmake_minimum_required(VERSION 3.8)
project(terrain_nav)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(se2_grid_core REQUIRED)
find_package(se2_grid_ros REQUIRED)
find_package(se2_grid_msgs REQUIRED)

add_library(joint_motion_risk STATIC src/joint_motion_risk.cpp)
target_include_directories(joint_motion_risk PUBLIC ${EIGEN3_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} include)
ament_target_dependencies(joint_motion_risk Eigen3 se2_grid_core)
option(TERRAIN_NAV_MOTION_CUDA "Build CUDA joint dynamic-motion collision assessment" ON)
if(TERRAIN_NAV_MOTION_CUDA)
  include(CheckLanguage)
  find_program(TERRAIN_NAV_NVCC nvcc HINTS /usr/local/cuda/bin)
  if(TERRAIN_NAV_NVCC AND NOT CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER "${TERRAIN_NAV_NVCC}")
  endif()
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    target_sources(joint_motion_risk PRIVATE src/joint_motion_risk.cu)
    target_compile_definitions(joint_motion_risk PRIVATE TERRAIN_NAV_MOTION_CUDA)
    # Include forward-compatible PTX; the installed driver JITs for RTX5060.
    set_target_properties(joint_motion_risk PROPERTIES CUDA_STANDARD 17 CUDA_ARCHITECTURES 86)
  endif()
endif()

add_executable(terrain_nav_node src/terrain_nav_node.cpp)
target_link_libraries(terrain_nav_node joint_motion_risk)

target_include_directories(terrain_nav_node PRIVATE
  ${EIGEN3_INCLUDE_DIRS}
  include
)

ament_target_dependencies(terrain_nav_node
  Eigen3
  rclcpp
  std_msgs
  geometry_msgs
  nav_msgs
  diagnostic_msgs
  sensor_msgs
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

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_chance_budget test/test_chance_budget.cpp)
  target_include_directories(test_chance_budget PRIVATE include)
  ament_add_gtest(test_chance_navigation test/test_chance_navigation.cpp)
  target_include_directories(test_chance_navigation PRIVATE ${EIGEN3_INCLUDE_DIRS} include)
  ament_target_dependencies(test_chance_navigation Eigen3 rclcpp std_msgs geometry_msgs nav_msgs diagnostic_msgs
    sensor_msgs visualization_msgs tf2 tf2_geometry_msgs se2_grid_core se2_grid_ros se2_grid_msgs)
  target_link_libraries(test_chance_navigation joint_motion_risk)
  ament_add_gtest(test_reverse_arc test/test_reverse_arc.cpp)
  target_include_directories(test_reverse_arc PRIVATE include)
  ament_add_gtest(test_motion_guard test/test_motion_guard.cpp)
  target_include_directories(test_motion_guard PRIVATE ${EIGEN3_INCLUDE_DIRS} include)
  ament_target_dependencies(test_motion_guard Eigen3 se2_grid_core)
  ament_add_gtest(test_object_observation test/test_object_observation.cpp)
  target_include_directories(test_object_observation PRIVATE ${EIGEN3_INCLUDE_DIRS} include)
  ament_target_dependencies(test_object_observation Eigen3 se2_grid_core)
  ament_add_gtest(test_gaussian_motion test/test_gaussian_motion.cpp)
  target_include_directories(test_gaussian_motion PRIVATE ${EIGEN3_INCLUDE_DIRS} include)
  ament_target_dependencies(test_gaussian_motion Eigen3 se2_grid_core)
  target_link_libraries(test_gaussian_motion joint_motion_risk)
  ament_add_gtest(test_mobility_recovery test/test_mobility_recovery.cpp)
  target_include_directories(test_mobility_recovery PRIVATE include)
  ament_add_gtest(test_route_memory test/test_route_memory.cpp)
  target_include_directories(test_route_memory PRIVATE include)
  ament_add_gtest(test_measured_columns test/test_measured_columns.cpp)
  target_include_directories(test_measured_columns PRIVATE include)
  add_custom_target(motion_qualification DEPENDS terrain_nav_node test_chance_budget test_chance_navigation
    test_reverse_arc test_motion_guard test_object_observation test_gaussian_motion test_mobility_recovery
    test_route_memory test_measured_columns)
  find_package(nlohmann_json REQUIRED)
  add_executable(actuator_model_audit test/actuator_model_audit.cpp)
  target_include_directories(actuator_model_audit PRIVATE include)
  target_link_libraries(actuator_model_audit nlohmann_json::nlohmann_json)
  target_link_libraries(test_motion_guard nlohmann_json::nlohmann_json)
  target_link_libraries(test_gaussian_motion nlohmann_json::nlohmann_json)
  target_compile_definitions(test_gaussian_motion PRIVATE
    GAUSSIAN_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/data")
  target_compile_definitions(test_motion_guard PRIVATE
    MOTION_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/data")
  add_executable(audit_frozen_map test/audit_frozen_map.cpp)
  target_link_libraries(audit_frozen_map joint_motion_risk)
  target_include_directories(audit_frozen_map PRIVATE ${EIGEN3_INCLUDE_DIRS} include)
  target_link_libraries(audit_frozen_map nlohmann_json::nlohmann_json)
  ament_target_dependencies(audit_frozen_map Eigen3 rclcpp std_msgs geometry_msgs nav_msgs diagnostic_msgs
    sensor_msgs visualization_msgs tf2 tf2_geometry_msgs se2_grid_core se2_grid_ros se2_grid_msgs)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS include/terrain_nav/motion_guard.hpp)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/include/terrain_nav/motion_guard.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/terrain_nav_node.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/navigation_benchmark.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/navigation_query_benchmark.cpp")
  file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/include/terrain_nav/motion_guard.hpp" MOTION_GUARD_SHA)
  add_executable(motion_guard_replay test/motion_guard_replay.cpp)
  target_include_directories(motion_guard_replay PRIVATE ${EIGEN3_INCLUDE_DIRS} include)
  ament_target_dependencies(motion_guard_replay Eigen3 se2_grid_core)
  target_link_libraries(motion_guard_replay nlohmann_json::nlohmann_json)
  target_compile_definitions(motion_guard_replay PRIVATE MOTION_GUARD_SHA="${MOTION_GUARD_SHA}")
  add_executable(dynamic_geometry_replay test/dynamic_geometry_replay.cpp)
  target_include_directories(dynamic_geometry_replay PRIVATE ${EIGEN3_INCLUDE_DIRS} include)
  ament_target_dependencies(dynamic_geometry_replay Eigen3 se2_grid_core)
  target_link_libraries(dynamic_geometry_replay nlohmann_json::nlohmann_json)
  target_link_libraries(dynamic_geometry_replay joint_motion_risk)
  if(TERRAIN_NAV_MOTION_CUDA AND CMAKE_CUDA_COMPILER)
    add_executable(motion_screen_benchmark test/motion_screen_benchmark.cpp)
    target_include_directories(motion_screen_benchmark PRIVATE ${EIGEN3_INCLUDE_DIRS} include)
    ament_target_dependencies(motion_screen_benchmark Eigen3 se2_grid_core)
    target_link_libraries(motion_screen_benchmark joint_motion_risk nlohmann_json::nlohmann_json)
  endif()
  add_executable(motion_observation_audit test/motion_observation_audit.cpp)
  target_include_directories(motion_observation_audit PRIVATE ${EIGEN3_INCLUDE_DIRS} include)
  ament_target_dependencies(motion_observation_audit Eigen3 se2_grid_core)
  target_link_libraries(motion_observation_audit nlohmann_json::nlohmann_json)
  add_executable(replay_topology_snapshot test/replay_topology_snapshot.cpp)
  target_link_libraries(replay_topology_snapshot joint_motion_risk)
  target_include_directories(replay_topology_snapshot PRIVATE ${EIGEN3_INCLUDE_DIRS} include)
  target_link_libraries(replay_topology_snapshot nlohmann_json::nlohmann_json)
  ament_target_dependencies(replay_topology_snapshot Eigen3 rclcpp std_msgs geometry_msgs nav_msgs diagnostic_msgs
    sensor_msgs visualization_msgs tf2 tf2_geometry_msgs se2_grid_core se2_grid_ros se2_grid_msgs)
  add_executable(replay_reverse_snapshot test/replay_reverse_snapshot.cpp)
  target_link_libraries(replay_reverse_snapshot joint_motion_risk)
  target_include_directories(replay_reverse_snapshot PRIVATE ${EIGEN3_INCLUDE_DIRS} include)
  target_link_libraries(replay_reverse_snapshot nlohmann_json::nlohmann_json)
  ament_target_dependencies(replay_reverse_snapshot Eigen3 rclcpp std_msgs geometry_msgs nav_msgs diagnostic_msgs
    sensor_msgs visualization_msgs tf2 tf2_geometry_msgs se2_grid_core se2_grid_ros se2_grid_msgs)
  find_path(SE2GRID_HEADER_ROOT se2_grid_core/SE2Grid.hpp HINTS ${se2_grid_core_INCLUDE_DIRS} REQUIRED)
  set(GRID_HEADER "${SE2GRID_HEADER_ROOT}/se2_grid_core/SE2Grid.hpp")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    src/terrain_nav_node.cpp test/navigation_benchmark.cpp "${GRID_HEADER}")
  file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/src/terrain_nav_node.cpp" BENCH_NODE_SHA)
  file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/test/navigation_benchmark.cpp" BENCH_HARNESS_SHA)
  file(SHA256 "${GRID_HEADER}" BENCH_GRID_SHA)
  add_executable(terrain_nav_benchmark test/navigation_benchmark.cpp)
  target_compile_definitions(terrain_nav_benchmark PRIVATE BENCH_NODE_SHA="${BENCH_NODE_SHA}"
    BENCH_HARNESS_SHA="${BENCH_HARNESS_SHA}" BENCH_GRID_SHA="${BENCH_GRID_SHA}")
  target_include_directories(terrain_nav_benchmark PRIVATE ${EIGEN3_INCLUDE_DIRS} include)
  target_link_libraries(terrain_nav_benchmark nlohmann_json::nlohmann_json)
  target_link_libraries(terrain_nav_benchmark joint_motion_risk)
  ament_target_dependencies(terrain_nav_benchmark Eigen3 rclcpp std_msgs geometry_msgs nav_msgs diagnostic_msgs sensor_msgs
    visualization_msgs tf2 tf2_geometry_msgs se2_grid_core se2_grid_ros se2_grid_msgs)
  install(TARGETS terrain_nav_benchmark DESTINATION lib/${PROJECT_NAME})
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS test/navigation_query_benchmark.cpp)
  file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/test/navigation_query_benchmark.cpp" QUERY_HARNESS_SHA)
  add_executable(terrain_nav_query_benchmark test/navigation_query_benchmark.cpp)
  target_compile_definitions(terrain_nav_query_benchmark PRIVATE BENCH_NODE_SHA="${BENCH_NODE_SHA}"
    BENCH_HARNESS_SHA="${QUERY_HARNESS_SHA}" BENCH_GRID_SHA="${BENCH_GRID_SHA}")
  target_include_directories(terrain_nav_query_benchmark PRIVATE ${EIGEN3_INCLUDE_DIRS} include)
  target_link_libraries(terrain_nav_query_benchmark nlohmann_json::nlohmann_json)
  target_link_libraries(terrain_nav_query_benchmark joint_motion_risk)
  ament_target_dependencies(terrain_nav_query_benchmark Eigen3 rclcpp std_msgs geometry_msgs nav_msgs diagnostic_msgs sensor_msgs
    visualization_msgs tf2 tf2_geometry_msgs se2_grid_core se2_grid_ros se2_grid_msgs)
  install(TARGETS terrain_nav_query_benchmark DESTINATION lib/${PROJECT_NAME})
endif()

ament_package()
