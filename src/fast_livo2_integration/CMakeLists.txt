cmake_minimum_required(VERSION 3.8)
project(fast_livo2_integration)

find_package(ament_cmake REQUIRED)

install(DIRECTORY config launch rviz
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/livo2_chain_check.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
