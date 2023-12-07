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
find_package(Boost REQUIRED COMPONENTS system)
find_package(pcl_conversions REQUIRED)
find_package(PCL COMPONENTS REQUIRED)
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

# object target names
set(MODEL_OBJ model_veh_dynamics_ode)
set(OCP_OBJ ocp_veh_dynamics_ode)
set(SIM_OBJ sim_veh_dynamics_ode)

include_directories(
	include
	${full_model_build}
	${acados_include}
	${acados_include}/blasfeo/include
	${acados_include}/hpipm/include
	${acados_include}/acados
	${acados_include}/qpOASES_e
  ${PCL_INCLUDE_DIRS}
)

# model
file(GLOB MODEL_SRC_FILES ${full_model_build}/veh_dynamics_ode_model/*.c)
# set(MODEL_SRC
#   ${full_model_build}/veh_dynamics_ode_model/veh_dynamics_ode_impl_dae_fun.c
#   ${full_model_build}/veh_dynamics_ode_model/veh_dynamics_ode_impl_dae_fun_jac_x_xdot_z.c
#   ${full_model_build}/veh_dynamics_ode_model/veh_dynamics_ode_impl_dae_jac_x_xdot_u_z.c
#   ${full_model_build}/veh_dynamics_ode_model/veh_dynamics_ode_impl_dae_hess.c
# )
set(MODEL_SRC
  ${MODEL_SRC_FILES}
)
add_library(${MODEL_OBJ} OBJECT ${MODEL_SRC} )

# optimal control problem - mostly CasADi exports
file(GLOB OCP_COST_SRC_FILES ${full_model_build}/veh_dynamics_ode_cost/*.c)
file(GLOB OCP_CONSTRAINTS_SRC_FILES ${full_model_build}/veh_dynamics_ode_constraints/*.c)
# set(OCP_SRC
#   ${full_model_build}/veh_dynamics_ode_constraints/veh_dynamics_ode_constr_h_fun_jac_uxt_zt.c
#   ${full_model_build}/veh_dynamics_ode_constraints/veh_dynamics_ode_constr_h_fun.c
#   ${full_model_build}/veh_dynamics_ode_constraints/veh_dynamics_ode_constr_h_fun_jac_uxt_zt_hess.c
#   ${full_model_build}/veh_dynamics_ode_constraints/veh_dynamics_ode_constr_h_0_fun_jac_uxt_zt.c
#   ${full_model_build}/veh_dynamics_ode_constraints/veh_dynamics_ode_constr_h_0_fun.c
#   ${full_model_build}/veh_dynamics_ode_constraints/veh_dynamics_ode_constr_h_0_fun_jac_uxt_zt_hess.c
#   ${full_model_build}/veh_dynamics_ode_constraints/veh_dynamics_ode_constr_h_e_fun_jac_uxt_zt.c
#   ${full_model_build}/veh_dynamics_ode_constraints/veh_dynamics_ode_constr_h_e_fun.c
#   ${full_model_build}/veh_dynamics_ode_constraints/veh_dynamics_ode_constr_h_e_fun_jac_uxt_zt_hess.c
#   ${full_model_build}/veh_dynamics_ode_cost/veh_dynamics_ode_cost_ext_cost_0_fun.c
#   ${full_model_build}/veh_dynamics_ode_cost/veh_dynamics_ode_cost_ext_cost_0_fun_jac.c
#   ${full_model_build}/veh_dynamics_ode_cost/veh_dynamics_ode_cost_ext_cost_0_fun_jac_hess.c
#   ${full_model_build}/veh_dynamics_ode_cost/veh_dynamics_ode_cost_ext_cost_fun.c
#   ${full_model_build}/veh_dynamics_ode_cost/veh_dynamics_ode_cost_ext_cost_fun_jac.c
#   ${full_model_build}/veh_dynamics_ode_cost/veh_dynamics_ode_cost_ext_cost_fun_jac_hess.c
#   ${full_model_build}/acados_solver_veh_dynamics_ode.c)
set(OCP_SRC
  ${OCP_COST_SRC_FILES}
  ${OCP_CONSTRAINTS_SRC_FILES}
  ${full_model_build}/acados_solver_veh_dynamics_ode.c
)
add_library(${OCP_OBJ} OBJECT ${OCP_SRC})

# for sim solver
set(SIM_SRC 
  ${full_model_build}/acados_sim_solver_veh_dynamics_ode.c
)
add_library(${SIM_OBJ} OBJECT ${SIM_SRC})

link_directories(${acados_lib} ${acados_include} ${full_model_build})

link_libraries(acados hpipm blasfeo m )

#-------------------
add_executable(mpc_controller_node
  src/mpc_controller_node.cpp
  $<TARGET_OBJECTS:${MODEL_OBJ}> 
  $<TARGET_OBJECTS:${OCP_OBJ}>
  # $<TARGET_OBJECTS:${SIM_OBJ}>
)

ament_target_dependencies(mpc_controller_node
  rclcpp
  sim_backend
  builtin_interfaces
  rcl_interfaces
  rcutils
  pcl_conversions
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
