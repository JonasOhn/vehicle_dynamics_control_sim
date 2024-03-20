/*
 * Simple Vehicle Dynamics Simulator Project
 *
 * Copyright (c) 2023-2024 Authors:
 *   - Jonas Ohnemus <johnemus@ethz.ch>
 *
 * All rights reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */

#ifndef MPC_CONTROLLER_H
#define MPC_CONTROLLER_H

// standard
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
// acados
#include "acados/utils/math.h"
#include "acados/utils/print.h"
#include "acados_c/external_function_interface.h"
#include "acados_c/ocp_nlp_interface.h"
#include "acados_solver_veh_dynamics_ode.h"

// blasfeo
#include "blasfeo/include/blasfeo_d_aux_ext_dep.h"

// geometry class include
#include "mpc_controller/mpc_geometry.hpp"

#define NX VEH_DYNAMICS_ODE_NX
#define NZ VEH_DYNAMICS_ODE_NZ
#define NU VEH_DYNAMICS_ODE_NU
#define NP VEH_DYNAMICS_ODE_NP
#define NBX0 VEH_DYNAMICS_ODE_NBX0

typedef struct {
  int sqp_max_iter;
  int rti_phase;
  bool warm_start_first_qp;
} solver_parameters;

typedef struct {
  double kkt_norm_inf;
  double solve_time;
  int sqp_iterations;
  int nlp_solver_status;
  int qp_solver_status;
} solver_output;

typedef struct {
  int N_horizon_mpc;
  double T_final_mpc;
  double dt_mpc;
} horizon_parameters;

typedef struct {
  double l_f; // m
  double l_r; // m
  double m;   // kg
  double C_d; // 0.5 * rho * CdA
  double C_r; // -
} model_parameters;

typedef struct {
  double q_n;
  double q_sd;
  double q_mu;
  double q_ax;
  double q_dels;
  double r_dax;
  double r_ddels;
} cost_parameters;

class MpcController {

private:
  // Parameterization
  solver_parameters solver_params_;
  model_parameters model_params_;
  horizon_parameters horizon_params_;
  solver_output solver_out_;
  cost_parameters cost_params_;

  MpcGeometry mpc_geometry_obj_;

  double dt_ctrl_callback_;

  int i_ = 0, j_ = 0, k_ = 0;

  // Current State for MPC
  // [s:0, n:1, mu:2, vx:3, axm:4, dels:5]
  double x_[6] = {0.0};

  // NLP Prediction trajectories
  double x_traj_[VEH_DYNAMICS_ODE_N + 1][NX];
  double u_traj_[VEH_DYNAMICS_ODE_N][NU];

  double curv_ref_mpc_[VEH_DYNAMICS_ODE_N] = {0.0};
  double s_ref_mpc_[VEH_DYNAMICS_ODE_N] = {0.0};

  // Initialization container vectors
  double x_stage_to_fill_[NX];
  double u_stage_to_fill_[NU];
  // NLP: Solver variables
  ocp_nlp_config *nlp_config_;
  ocp_nlp_dims *nlp_dims_;
  ocp_nlp_in *nlp_in_;
  ocp_nlp_out *nlp_out_;
  ocp_nlp_solver *nlp_solver_;
  void *nlp_opts_;
  veh_dynamics_ode_solver_capsule *acados_ocp_capsule_;

public:
  MpcController();

  void reset_prediction_trajectories();

  int8_t init_solver();

  int8_t init_mpc_horizon();

  int8_t solve_mpc();

  int8_t get_input(double (&u)[2]);

  int8_t get_solver_statistics(int &nlp_solv_status, int &qp_solv_status,
                               int &sqp_iter, double &kkt_norm,
                               double &solv_time);

  int8_t get_predictions(
      std::vector<double> &s_predict, std::vector<double> &n_predict,
      std::vector<double> &mu_predict, std::vector<double> &vx_predict,
      std::vector<double> &vy_predict, std::vector<double> &dpsi_predict,
      std::vector<double> &axm_predict, std::vector<double> &dels_predict,
      std::vector<double> &daxm_predict, std::vector<double> &ddels_predict,
      std::vector<double> &s_traj_mpc, std::vector<double> &kappa_traj_mpc,
      std::vector<double> &s_ref_spline, std::vector<double> &kappa_ref_spline,
      std::vector<double> &s_ref_mpc, std::vector<double> &kappa_ref_mpc);

  int get_number_of_spline_evaluations();

  int8_t
  get_visual_predictions(std::vector<std::vector<double>> &xy_spline_points,
                         std::vector<std::vector<double>> &xy_predict_points);

  int8_t set_reference_path(std::vector<std::vector<double>> &path);

  int8_t set_state(double x_c, double y_c, double psi, double vx_local,
                   double dels, double ax, double &s, double &n, double &mu);

  int8_t set_initial_state();

  int8_t init_mpc_parameters();

  int8_t set_model_parameters(double l_f, // m
                              double l_r, // m
                              double m,   // kg
                              double C_d, // 0.5 * rho * CdA
                              double C_r);

  int8_t set_cost_parameters(double q_sd, double q_n, double q_mu, double q_ax,
                             double q_dels, double r_dax, double r_ddels);

  int8_t set_horizon_parameters(int N, double T_f);

  int8_t set_solver_parameters(int sqp_max_iter, int rti_phase,
                               bool warm_start_first_qp);

  int8_t update_solver_statistics();

  void set_dt_control_feedback(double dt_ctrl_callback);

  double get_mass();
};

#endif
