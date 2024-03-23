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

#include "sim_backend/dynamic_system.hpp"
#include <boost/numeric/odeint.hpp>

using namespace boost::numeric::odeint;

DynamicSystem::DynamicSystem() {
  // Constructor for some standard parameter system
  T_mot_fl_set_ = 0.0;
  T_mot_fr_set_ = 0.0;
  T_mot_rl_set_ = 0.0;
  T_mot_rr_set_ = 0.0;
  delta_s_set_ = 0.0;
  l_ = 0.0;
}

void DynamicSystem::update_parameters(parameters param_struct) {
  params_ = param_struct;
}

void DynamicSystem::update_inputs(double Tm_fl_set, double Tm_fr_set,
                                  double Tm_rl_set, double Tm_rr_set,
                                  double delta_s_set){
  T_mot_fl_set_ = Tm_fl_set;
  T_mot_fr_set_ = Tm_fr_set;
  T_mot_rl_set_ = Tm_rl_set;
  T_mot_rr_set_ = Tm_rr_set;
  delta_s_set_ = delta_s_set;
}

void DynamicSystem::get_inputs(double &Tm_fl_set, double &Tm_fr_set,
                              double &Tm_rl_set, double &Tm_rr_set,
                              double &delta_s_set) {
  Tm_fl_set = T_mot_fl_set_;
  Tm_fr_set = T_mot_fr_set_;
  Tm_rl_set = T_mot_rl_set_;
  Tm_rr_set = T_mot_rr_set_;
  delta_s_set = delta_s_set_;
}

void DynamicSystem::operator()(const state_type &x, state_type &dxdt,
                               const double /* t */) {
  // Differential Equations for vehicle

  // kinematic states describing position and orientation of the vehicle
  double x_C_I = x[0];
  double y_C_I = x[1];
  double psi = x[2];
  
  double vx_C_V = x[3];
  double vy_C_V = x[4];
  double dpsi = x[5];

  double delta_s_act = x[6];

  double omega_wheel_fl = x[7];
  double omega_wheel_fr = x[8];
  double omega_wheel_rl = x[9];
  double omega_wheel_rr = x[10];

  double T_mot_fl_act = x[11];
  dxdt[11] = x[12];
  double dT_mot_fl_act = x[12];

  double T_mot_fr_act = x[13];
  dxdt[13] = x[14];
  double dT_mot_fr_act = x[14];
  
  double T_mot_rl_act = x[15];
  dxdt[15] = x[16];
  double dT_mot_rl_act = x[16];
  
  double T_mot_rr_act = x[17];
  dxdt[17] = x[18];
  double dT_mot_rr_act = x[18];
  
  double Fx_fl_act = x[19];
  double Fx_fr_act = x[20];
  double Fx_rl_act = x[21];
  double Fx_rr_act = x[22];

  double Fy_fl_act = x[23];
  double Fy_fr_act = x[24];
  double Fy_rl_act = x[25];
  double Fy_rr_act = x[26];

  // =========== SATURATION ================

  delta_s_act = fmax(fmin(delta_s_act, params_.delta_s_max), params_.delta_s_min);

  T_mot_fl_act = fmax(fmin(T_mot_fl_act, params_.T_mot_max), params_.T_mot_min);
  T_mot_fr_act = fmax(fmin(T_mot_fr_act, params_.T_mot_max), params_.T_mot_min);
  T_mot_rl_act = fmax(fmin(T_mot_rl_act, params_.T_mot_max), params_.T_mot_min);
  T_mot_rr_act = fmax(fmin(T_mot_rr_act, params_.T_mot_max), params_.T_mot_min);

  T_mot_fl_set_ = fmax(fmin(T_mot_fl_set_, params_.T_mot_max), params_.T_mot_min);
  T_mot_fr_set_ = fmax(fmin(T_mot_fr_set_, params_.T_mot_max), params_.T_mot_min);
  T_mot_rl_set_ = fmax(fmin(T_mot_rl_set_, params_.T_mot_max), params_.T_mot_min);
  T_mot_rr_set_ = fmax(fmin(T_mot_rr_set_, params_.T_mot_max), params_.T_mot_min);

  // ====================================================================
  // Aero Resistance only in x-direction of vehicle frame
  double F_x_aero = - tanh(vx_C_V / 1e-6) * (params_.C_d * pow(vx_C_V, 2));
  double F_x_rollres = tanh(vx_C_V / 1e-6) * (params_.C_r * params_.m * params_.g);

  // dxdt[0]: dX_C/dt in INERTIAL FRAME
  double dx_C_I = vx_C_V * cos(psi) - vy_C_V * sin(psi);
  dxdt[0] = dx_C_I;

  // dxdt[1]: dY_C/dt in INERTIAL FRAME
  double dy_C_I = vx_C_V * sin(psi) + vy_C_V * cos(psi);
  dxdt[1] = dy_C_I;
  
  // dxdt[2]: dpsi/dt
  dxdt[2] = dpsi;

  // ax_C_V
  double ax_C_V = ((Fx_fl_act + Fx_fr_act) * cos(delta_s_act)
          + (Fx_rl_act + Fx_rr_act) 
          - (Fy_fl_act + Fy_fr_act) * sin(delta_s_act) 
          + F_x_aero - F_x_rollres) / params_.m 
          + vy_C_V * dpsi;
  dxdt[3] = ax_C_V;

  // ay_C_V
  double ay_C_V = (Fy_rl_act + Fy_rr_act 
          + (Fy_fl_act + Fy_fr_act) * cos(delta_s_act)
          + (Fx_fl_act + Fx_fr_act) * sin(delta_s_act)) / params_.m 
          - vx_C_V * dpsi;
  dxdt[4] = ay_C_V;

  // ddpsi
  double ddpsi = (
            Fx_fr_act * params_.l_f * sin(delta_s_act)
          + Fx_fr_act * params_.wb_f / 2.0 * cos(delta_s_act)
          + Fy_fr_act * params_.l_f * cos(delta_s_act)
          - Fy_fr_act * params_.wb_f / 2.0 * sin(delta_s_act)
          + Fx_fl_act * params_.l_f * sin(delta_s_act)
          - Fx_fl_act * params_.wb_f / 2.0 * cos(delta_s_act)
          + Fy_fl_act * params_.l_f * cos(delta_s_act)
          + Fy_fl_act * params_.wb_f / 2.0 * sin(delta_s_act)
          - Fy_rl_act * params_.l_r
          - Fx_rl_act * params_.wb_r / 2.0
          - Fy_rr_act * params_.l_r
          + Fx_rr_act * params_.wb_r / 2.0) /
            params_.Iz;
  dxdt[5] = ddpsi;

  // ddel_steer_act defined by input (transient)
  dxdt[6] = (delta_s_act - delta_s_set_) / params_.tau_steer;

  // =====================================================================

  // ======== MOTORS ================
  // d²Tmot_fl/dt²
  dxdt[12] = (T_mot_fl_set_ - 2 * params_.D_mot * params_.tau_mot * dT_mot_fl_act - T_mot_fl_act) / pow(params_.tau_mot, 2.0);

  // d²Tmot_fr/dt²
  dxdt[14] = (T_mot_fr_set_ - 2 * params_.D_mot * params_.tau_mot * dT_mot_fr_act - T_mot_fr_act) / pow(params_.tau_mot, 2.0);

  // d²Tmot_rl/dt²
  dxdt[16] = (T_mot_rl_set_ - 2 * params_.D_mot * params_.tau_mot * dT_mot_rl_act - T_mot_rl_act) / pow(params_.tau_mot, 2.0);

  // d²Tmot_rr/dt²
  dxdt[18] = (T_mot_rr_set_ - 2 * params_.D_mot * params_.tau_mot * dT_mot_rr_act - T_mot_rr_act) / pow(params_.tau_mot, 2.0);

  // ========== WHEELS ==============
  // (rolling resistance modeled as acting on body)
  // domega_wheel_fl
  dxdt[7] = (T_mot_fl_act * params_.iG - Fx_fl_act * params_.r_wheel) / params_.Iwheel;

  // domega_wheel_fr
  dxdt[8] = (T_mot_fr_act * params_.iG - Fx_fr_act * params_.r_wheel) / params_.Iwheel;

  // domega_wheel_rl
  dxdt[9] = (T_mot_rl_act * params_.iG - Fx_rl_act * params_.r_wheel) / params_.Iwheel;

  // domega_wheel_rr
  dxdt[10] = (T_mot_rr_act * params_.iG - Fx_rr_act * params_.r_wheel) / params_.Iwheel;

  // ========== SLIP ================

  // sideslip angles
  double vy_f = vy_C_V + dpsi * params_.l_f;
  double vy_r = vy_C_V - dpsi * params_.l_r;
  double vx_fl = vx_C_V - params_.wb_f / 2.0 * dpsi;
  double vx_fr = vx_C_V + params_.wb_f / 2.0 * dpsi;
  double vx_rl = vx_C_V - params_.wb_r / 2.0 * dpsi;
  double vx_rr = vx_C_V + params_.wb_r / 2.0 * dpsi;

  double alpha_fl = - delta_s_act + atan2(vy_f, vx_fl);
  double alpha_fr = - delta_s_act + atan2(vy_f, vx_fr);
  double alpha_rl = atan2(vy_r, vx_rl);
  double alpha_rr = atan2(vy_r, vx_rr);

  // longitudinal slip calculation
  double vx_tire_fl = vx_fl * cos(delta_s_act) + vy_f * sin(delta_s_act);
  double vx_tire_fr = vx_fr * cos(delta_s_act) + vy_f * sin(delta_s_act);
  double vx_tire_rl = vx_rl;
  double vx_tire_rr = vx_rr;

  double vn_fl = fmax(params_.r_wheel * fabs(omega_wheel_fl), fabs(vx_tire_fl));
  double vn_fr = fmax(params_.r_wheel * fabs(omega_wheel_fr), fabs(vx_tire_fr));
  double vn_rl = fmax(params_.r_wheel * fabs(omega_wheel_rl), fabs(vx_tire_rl));
  double vn_rr = fmax(params_.r_wheel * fabs(omega_wheel_rr), fabs(vx_tire_rr));

  double sx_fl = (params_.r_wheel * omega_wheel_fl - vx_tire_fl) / vn_fl;
  double sx_fr = (params_.r_wheel * omega_wheel_fr - vx_tire_fr) / vn_fr;
  double sx_rl = (params_.r_wheel * omega_wheel_rl - vx_tire_rl) / vn_rl;
  double sx_rr = (params_.r_wheel * omega_wheel_rr - vx_tire_rr) / vn_rr;

  // ========= TIRE LOADS ==========

  double wb_avg = (params_.wb_f + params_.wb_r) / 2.0;

  // Vertical tire loads (static)
  double Fz_fl_stat = 0.5 * params_.m * params_.g * params_.l_r / l_;
  double Fz_fr_stat = 0.5 * params_.m * params_.g * params_.l_r / l_;
  double Fz_rl_stat = 0.5 * params_.m * params_.g * params_.l_f / l_;
  double Fz_rr_stat = 0.5 * params_.m * params_.g * params_.l_f / l_;

  // vertical tire loads (dynamic)
  double delta_Fz_lon = 0.5 * params_.m * ax_C_V * params_.h_cg / l_;
  double delta_Fz_lat = 0.5 * params_.m * ay_C_V * params_.h_cg / wb_avg;

  double Fz_aero = 0.25 * params_.C_l * pow(vx_C_V, 2);

  double Fz_fl = Fz_aero + Fz_fl_stat - delta_Fz_lat - delta_Fz_lon;
  double Fz_fr = Fz_aero + Fz_fr_stat + delta_Fz_lat - delta_Fz_lon;
  double Fz_rl = Fz_aero + Fz_rl_stat - delta_Fz_lat + delta_Fz_lon;
  double Fz_rr = Fz_aero + Fz_rr_stat + delta_Fz_lat + delta_Fz_lon;

  // ============ PACEJKA ================

  // Lateral tire forces
  double Fy_fl = Fz_fl * params_.D_tire_lat * sin(params_.C_tire_lat * atan(params_.B_tire_lat * alpha_fl));
  double Fy_fr = Fz_fr * params_.D_tire_lat * sin(params_.C_tire_lat * atan(params_.B_tire_lat * alpha_fr));
  double Fy_rl = Fz_rl * params_.D_tire_lat * sin(params_.C_tire_lat * atan(params_.B_tire_lat * alpha_rl));
  double Fy_rr = Fz_rr * params_.D_tire_lat * sin(params_.C_tire_lat * atan(params_.B_tire_lat * alpha_rr));

  // Longitudinal tire forces
  double Fx_fl = Fz_fl * params_.D_tire_lon * sin(params_.C_tire_lon * atan(params_.B_tire_lon * sx_fl));
  double Fx_fr = Fz_fr * params_.D_tire_lon * sin(params_.C_tire_lon * atan(params_.B_tire_lon * sx_fr));
  double Fx_rl = Fz_rl * params_.D_tire_lon * sin(params_.C_tire_lon * atan(params_.B_tire_lon * sx_rl));
  double Fx_rr = Fz_rr * params_.D_tire_lon * sin(params_.C_tire_lon * atan(params_.B_tire_lon * sx_rr));

  // ======== Tire Force transients as linear first order system ======

  // d(Fx_fl) / dt
  dxdt[19] = (Fx_fl - Fx_fl_act) / params_.tau_tire_x;

  // d(Fx_fr) / dt
  dxdt[20] = (Fx_fr - Fx_fr_act) / params_.tau_tire_x;

  // d(Fx_rl) / dt
  dxdt[21] = (Fx_rl - Fx_rl_act) / params_.tau_tire_x;

  // d(Fx_rr) / dt
  dxdt[22] = (Fx_rr - Fx_rr_act) / params_.tau_tire_x;


  // d(Fy_fl) / dt
  dxdt[23] = (Fy_fl - Fy_fl_act) / params_.tau_tire_y;

  // d(Fy_fr) / dt
  dxdt[24] = (Fy_fr - Fy_fr_act) / params_.tau_tire_y;

  // d(Fy_rl) / dt
  dxdt[25] = (Fy_rl - Fy_rl_act) / params_.tau_tire_y;

  // d(Fy_rr) / dt
  dxdt[26] = (Fy_rr - Fy_rr_act) / params_.tau_tire_y;
}
