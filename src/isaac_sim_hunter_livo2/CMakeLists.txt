cmake_minimum_required(VERSION 3.8)
project(isaac_sim_hunter_livo2)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch scripts assets
  DESTINATION share/${PROJECT_NAME}
  USE_SOURCE_PERMISSIONS
)

install(
  DIRECTORY docs
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
