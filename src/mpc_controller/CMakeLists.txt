cmake_minimum_required(VERSION 3.8)
project(mpc_controller)

set (CMAKE_CXX_STANDARD 20)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sim_backend REQUIRED)
find_package(Eigen3 3.3 REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(rcutils REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/MpcStateTrajectory.msg"
  "msg/MpcInputTrajectory.msg"
  "msg/MpcSolverState.msg"
  "msg/MpcKappaTrajectory.msg"
  "msg/MpcAlgStateTrajectory.msg"
)

ament_export_dependencies(rosidl_default_runtime)

# c-compiler flags for debugging
set(CMAKE_C_FLAGS_DEBUG "-O0 -ggdb")

set(CMAKE_C_FLAGS "-fPIC -Wno-unused-parameter ")
#-fno-diagnostics-show-line-numbers -g

# Set paths
set(acados_include "${PROJECT_SOURCE_DIR}/../../acados/include")
set(acados_lib "${PROJECT_SOURCE_DIR}/../../acados/lib")
set(full_model_build ${PROJECT_SOURCE_DIR}/scripts/c_generated_solver_mpc)
set(qp_model_build ${PROJECT_SOURCE_DIR}/scripts/c_generated_solver_mpc_qp_initializer)

# object target names
set(MODEL_OBJ model_veh_dynamics_ode)
set(OCP_OBJ ocp_veh_dynamics_ode)
set(QP_MODEL_OBJ model_veh_kinematics_ode_init)
set(QP_OCP_OBJ ocp_veh_kinematics_ode_init)

include_directories(include
	${full_model_build}
  ${qp_model_build}
	${acados_include}
	${acados_include}/blasfeo/include
	${acados_include}/hpipm/include
	${acados_include}/acados
	${acados_include}/qpOASES_e
)

# ========================= NLP ======================
# NLP model
file(GLOB MODEL_SRC_FILES ${full_model_build}/veh_dynamics_ode_model/*.c)
set(MODEL_SRC
  ${MODEL_SRC_FILES}
)
add_library(${MODEL_OBJ} OBJECT ${MODEL_SRC} )

# NLP optimal control problem - mostly CasADi exports
file(GLOB OCP_COST_SRC_FILES ${full_model_build}/veh_dynamics_ode_cost/*.c)
file(GLOB OCP_CONSTRAINTS_SRC_FILES ${full_model_build}/veh_dynamics_ode_constraints/*.c)
set(OCP_SRC
  ${OCP_COST_SRC_FILES}
  ${OCP_CONSTRAINTS_SRC_FILES}
  ${full_model_build}/acados_solver_veh_dynamics_ode.c
)
add_library(${OCP_OBJ} OBJECT ${OCP_SRC})

# ========================= QP ======================
# QP model
file(GLOB QP_MODEL_SRC_FILES ${qp_model_build}/veh_kinematics_ode_init_model/*.c)
set(QP_MODEL_SRC
  ${QP_MODEL_SRC_FILES}
)
add_library(${QP_MODEL_OBJ} OBJECT ${QP_MODEL_SRC} )

# QP problem - mostly CasADi exports
file(GLOB OCP_CONSTRAINTS_SRC_FILES ${full_model_build}/veh_dynamics_ode_constraints/*.c)
set(QP_OCP_SRC
  ${qp_model_build}/acados_solver_veh_kinematics_ode_init.c
)
add_library(${QP_OCP_OBJ} OBJECT ${QP_OCP_SRC})


# ================ Link Directories ==================
link_directories(include 
  ${acados_lib} 
  ${acados_include} 
  ${full_model_build}
  ${qp_model_build}
)

link_libraries(acados hpipm blasfeo m )

#-------------------
add_executable(mpc_controller_node
  src/mpc_controller_node.cpp
  src/mpc_controller_class.cpp
  src/mpc_geometry.cpp
  $<TARGET_OBJECTS:${MODEL_OBJ}> 
  $<TARGET_OBJECTS:${OCP_OBJ}>
  $<TARGET_OBJECTS:${QP_MODEL_OBJ}>
  $<TARGET_OBJECTS:${QP_OCP_OBJ}>
)

ament_target_dependencies(mpc_controller_node
  rclcpp
  sim_backend
  builtin_interfaces
  rcl_interfaces
  rcutils
  visualization_msgs
  sensor_msgs
)

install(TARGETS 
  mpc_controller_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
)

rosidl_get_typesupport_target(cpp_typesupport_target
  ${PROJECT_NAME} rosidl_typesupport_cpp)

target_link_libraries(mpc_controller_node 
  Eigen3::Eigen
  "${cpp_typesupport_target}"
  "${Boost_SYSTEM_LIBRARY}"
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

install(
  DIRECTORY include/
  DESTINATION include
)

ament_export_include_directories(include)

#-------------------

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
