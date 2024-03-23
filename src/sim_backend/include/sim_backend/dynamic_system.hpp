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

#ifndef DYNAMIC_SYSTEM_H
#define DYNAMIC_SYSTEM_H

#include <boost/numeric/odeint.hpp>
#include <iostream>

using namespace std;
using namespace boost::numeric::odeint;

typedef struct {
  // overall geometry
  double l_f; // m
  double l_r; // m
  double wb_f; // m
  double wb_r; // m
  double h_cg; // m
  double r_wheel; // m

  // mass and inertia
  double m;   // kg
  double Iz;  // kg m m
  double g;   // m s-2
  double Iwheel; // kg m m

  // Pacejka longitudinal
  double D_tire_lon;
  double C_tire_lon;
  double B_tire_lon;
  double tau_tire_x;

  // Pacejka lateral
  double D_tire_lat;
  double C_tire_lat;
  double B_tire_lat;
  double tau_tire_y;

  // resistance
  double C_d; // 0.5 * rho * CdA
  double C_r; // -
  double C_l;

  // motors
  double tau_mot;
  double D_mot;
  double iG;
  double tau_steer;

  // saturation
  double delta_s_max;
  double delta_s_min;
  double T_mot_max;
  double T_mot_min;
} parameters;

typedef std::vector<double> state_type;

/* The rhs of dxdt = f(x) defined as a class */
class DynamicSystem {

private:
  // indep. params
  parameters params_;
  // dep. params
  double l_;
  // inputs
  double T_mot_fl_set_;
  double T_mot_fr_set_;
  double T_mot_rl_set_;
  double T_mot_rr_set_;
  double delta_s_set_;

public:
  DynamicSystem();

  void update_inputs(double Tm_fl_set, double Tm_fr_set,
                      double Tm_rl_set, double Tm_rr_set,
                      double delta_s_set);

  void get_inputs(double &Tm_fl_set, double &Tm_fr_set,
                  double &Tm_rl_set, double &Tm_rr_set,
                  double &delta_s_set);

  void update_parameters(parameters param_struct);

  void operator()(const state_type &x, state_type &dxdt, const double /* t */);
};

#endif
