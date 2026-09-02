cmake_minimum_required(VERSION 3.5)
project(catkin_simple)

find_package(ament_cmake REQUIRED)

ament_package(
  CONFIG_EXTRAS "catkin_simple-extras.cmake"
)
