cmake_minimum_required(VERSION 3.8)
project(sim_visualization)

set (CMAKE_CXX_STANDARD 20)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sim_backend REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)
find_package(pcl_conversions REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(rcutils REQUIRED)
find_package(std_msgs REQUIRED)
find_package(PCL REQUIRED)
find_package(visualization_msgs REQUIRED)


ament_export_include_directories(include)

include_directories( 
  include 
  ${PCL_INCLUDE_DIRS}
  ${sim_backend_INCLUDE_DIRS}
)

add_executable(sim_visualization_node 
  src/sim_visualization_node.cpp
)

ament_target_dependencies(sim_visualization_node
  rclcpp
  sim_backend
  sensor_msgs
  pcl_conversions
  rcl_interfaces
  rcutils
  tf2_geometry_msgs
  std_msgs
  visualization_msgs
)

install(TARGETS
  sim_visualization_node  
  DESTINATION lib/${PROJECT_NAME}
)

target_link_libraries(sim_visualization_node "${Boost_SYSTEM_LIBRARY}")

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
