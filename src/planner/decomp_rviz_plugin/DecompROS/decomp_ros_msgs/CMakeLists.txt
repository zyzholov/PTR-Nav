cmake_minimum_required(VERSION 3.5)
project(decomp_ros_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)

set(msg_files
  "msg/Ellipsoid.msg"
  "msg/EllipsoidArray.msg"
  "msg/Polyhedron.msg"
  "msg/PolyhedronArray.msg"
)

if(msg_files)
  rosidl_generate_interfaces(${PROJECT_NAME}
    ${msg_files}
    DEPENDENCIES geometry_msgs std_msgs
  )
  ament_export_dependencies(rosidl_default_runtime geometry_msgs std_msgs)
endif()

ament_package()
