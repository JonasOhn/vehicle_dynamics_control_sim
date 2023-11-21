cmake_minimum_required(VERSION 3.8)
project(sim_backend)

set (CMAKE_CXX_STANDARD 20)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/SysInput.msg"
  "msg/VehicleState.msg"
  "msg/RefPath.msg"
  "msg/Point2D.msg"
)

ament_export_dependencies(rosidl_default_runtime)

include_directories(include)

add_executable(dynamics_simulator 
  src/dynamics_sim_node.cpp
  src/dynamic_system.cpp)

add_executable(vehicle_frame_publisher
  src/vehicle_frame_publisher.cpp)

ament_target_dependencies(vehicle_frame_publisher
  geometry_msgs
  rclcpp
  tf2
  tf2_ros)

ament_target_dependencies(dynamics_simulator
  rclcpp 
  geometry_msgs
  builtin_interfaces
  tf2
  tf2_ros
  tf2_geometry_msgs)

# install such that ros2 run can find exe
install(TARGETS
  dynamics_simulator
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS
  vehicle_frame_publisher
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
)

rosidl_get_typesupport_target(cpp_typesupport_target
  ${PROJECT_NAME} rosidl_typesupport_cpp)

target_link_libraries(dynamics_simulator "${cpp_typesupport_target}")

install(
  DIRECTORY include/
  DESTINATION include
)

ament_export_include_directories(include)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
