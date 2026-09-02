cmake_minimum_required(VERSION 3.10)

# Set the project name
project(grad_test)

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)

# Include Eigen3 directories
include_directories(${EIGEN3_INCLUDE_DIR})

# Add the executables
add_executable(grad_test latGrad_test.cpp)

# Specify the C++ standard
set_target_properties(grad_test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

install(TARGETS grad_test
  RUNTIME DESTINATION lib/${PROJECT_NAME})

ament_package()