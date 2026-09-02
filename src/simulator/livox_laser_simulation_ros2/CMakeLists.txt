cmake_minimum_required(VERSION 3.5)
project(ros2_livox_simulation)
add_compile_options(-std=c++17)
# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(gazebo_dev REQUIRED)
find_package(gazebo_ros REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(livox_ros_driver2 REQUIRED)
find_package(Protobuf REQUIRED)
find_package(Boost REQUIRED COMPONENTS chrono)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
include_directories(
  include
)
include_directories(${GAZEBO_INCLUDE_DIRS})
include_directories(/usr/include/gazebo-11/gazebo)
link_directories(${GAZEBO_LIBRARY_DIRS})



install(DIRECTORY include/
        DESTINATION include
)



add_library(ros2_livox SHARED src/livox_points_plugin.cpp src/livox_ode_multiray_shape.cpp)
target_link_libraries(ros2_livox
  ${GAZEBO_LIBRARIES}
  RayPlugin
  GpuRayPlugin
  protobuf::libprotobuf
  Boost::chrono
)
ament_target_dependencies(ros2_livox rclcpp std_msgs sensor_msgs geometry_msgs gazebo_dev gazebo_ros tf2_ros livox_ros_driver2 )
target_include_directories(ros2_livox PRIVATE include)

#install(TARGETS ros2_livox DESTINATION lib/${PROJECT_NAME})

install(TARGETS ros2_livox
    LIBRARY DESTINATION lib)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_libraries(
  ros2_livox
)

set(CSV_FILES
    scan_mode/mid40.csv
    scan_mode/mid70.csv
    scan_mode/mid360.csv
    scan_mode/tele.csv
    scan_mode/horizon.csv
    scan_mode/HAP.csv
    scan_mode/avia.csv
)
foreach(CSV_FILE ${CSV_FILES})
    configure_file(${CSV_FILE} ${CMAKE_CURRENT_BINARY_DIR}/${CSV_FILE} COPYONLY)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${CSV_FILE}
        DESTINATION share/${PROJECT_NAME}/scan_mode
    )
endforeach()

set(URDF_FILES
    urdf/mid40.xacro
    urdf/mid70.xacro
    urdf/mid360.xacro
    urdf/tele.xacro
    urdf/horizon.xacro
    urdf/HAP.xacro
    urdf/avia.xacro
)
foreach(URDF_FILES ${URDF_FILES})
    configure_file(${URDF_FILES} ${CMAKE_CURRENT_BINARY_DIR}/${URDF_FILES} COPYONLY)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${URDF_FILES}
        DESTINATION share/${PROJECT_NAME}/urdf
    )
endforeach()

set(EXTERNAL_LIVOX_NEW_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../livox_mid360/new")
set(EXTERNAL_LIVOX_MODEL_DIR "${EXTERNAL_LIVOX_NEW_DIR}/livox_mid40")

set(EXTERNAL_LIVOX_MESH_FILES
    "${EXTERNAL_LIVOX_MODEL_DIR}/meshes/mid360.dae"
    "${EXTERNAL_LIVOX_MODEL_DIR}/meshes/mid360_rules.dae"
)
foreach(EXTERNAL_LIVOX_MESH_FILE ${EXTERNAL_LIVOX_MESH_FILES})
    if(EXISTS "${EXTERNAL_LIVOX_MESH_FILE}")
        install(FILES "${EXTERNAL_LIVOX_MESH_FILE}"
            DESTINATION share/${PROJECT_NAME}/meshes
        )
    else()
        message(WARNING "External Livox asset not found at ${EXTERNAL_LIVOX_MESH_FILE}")
    endif()
endforeach()

if(EXISTS "${EXTERNAL_LIVOX_MODEL_DIR}/scan_mode/mid360.csv")
    install(FILES "${EXTERNAL_LIVOX_MODEL_DIR}/scan_mode/mid360.csv"
        DESTINATION share/${PROJECT_NAME}/scan_mode
    )
endif()

set(EXTERNAL_LIVOX_SOURCE_ASSETS
    "${EXTERNAL_LIVOX_NEW_DIR}/livox_mid360.xacro"
    "${EXTERNAL_LIVOX_NEW_DIR}/mid360_IMU_platform.xacro"
    "${EXTERNAL_LIVOX_NEW_DIR}/mid360_IMU_platform.launch"
)
foreach(EXTERNAL_LIVOX_SOURCE_ASSET ${EXTERNAL_LIVOX_SOURCE_ASSETS})
    if(EXISTS "${EXTERNAL_LIVOX_SOURCE_ASSET}")
        install(FILES "${EXTERNAL_LIVOX_SOURCE_ASSET}"
            DESTINATION share/${PROJECT_NAME}/source_assets
        )
    endif()
endforeach()

ament_package()
