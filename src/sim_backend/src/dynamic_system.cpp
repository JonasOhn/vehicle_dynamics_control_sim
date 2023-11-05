#include <iostream>
#include <boost/numeric/odeint.hpp>
#include "dynamic_system.hpp"

using namespace std;
using namespace boost::numeric::odeint;

DynamicSystem::DynamicSystem(){
    // Constructor for some standard parameter system
    params_.l_f = 0.8; // m
    params_.l_r = 0.7; // m
    l_ = params_.l_f + params_.l_r;
    params_.m = 220; // kg
    params_.Iz = 100; // kg m m
    params_.g = 9.81; // m s-2
    params_.D_tire = -5;
    params_.C_tire = 1.2;
    params_.B_tire = 9.5;
    params_.C_d = 0.5 * 1.225 * 1.85; // 0.5 * rho * CdA
    params_.C_r = 0.5; // -
    params_.T_mot = 0.1;
    params_.D_mot = 1.0;
    Fx_f_ = 0.0;
    Fx_r_ = 0.0;
    delta_s_ = 0.0;
}

void DynamicSystem::update_inputs(double fx_f, double fx_r, double delta_steer){
    Fx_f_ = fx_f;
    Fx_r_ = fx_r;
    delta_s_ = delta_steer;
}

void DynamicSystem::operator() ( const state_type &x , state_type &dxdt , const double /* t */ )
{
    // Differential Equations for x = [X_C, Y_C, psi, dX_cdt, dY_cdt, dpsidt]
    // from Lagrange Equations in inertial frame

    // Resistance only in x-direction of vehicle frame
    double F_resist = tanh(x[3]/1e-6) * (params_.C_d * pow(x[3], 2) + params_.C_r * params_.m * params_.g);

    // sideslip angles
    double alpha_f =
        - delta_s_ 
        + atan2(x[4] + x[5] * params_.l_f,
                x[3]);
    double alpha_r =
        + atan2(x[4] - x[5] * params_.l_r,
                x[3]);

    // Vertical tire loads (static)
    double Fz_f = params_.m * params_.g * params_.l_r/l_;
    double Fz_r = params_.m * params_.g * params_.l_f/l_;

    // Lateral tire loads (Pacejka model)
    double Fy_f = Fz_f * params_.D_tire * sin(params_.C_tire * atan(params_.B_tire * alpha_f));
    double Fy_r = Fz_r * params_.D_tire * sin(params_.C_tire * atan(params_.B_tire * alpha_r));

    // Differential Equations Vehcile Dynamics

    // x[0]: X_C in INERTIAL FRAME
    dxdt[0] = x[3] * cos(x[2]) - x[4] * sin(x[2]);
    // x[1]: Y_C in INERTIAL FRAME
    dxdt[1] = x[3] * sin(x[2]) + x[4] * cos(x[2]);
    // x[2]: psi
    dxdt[2] = x[5];

    // x[3]: dX_C / dt in VEHICLE FRAME
    dxdt[3] = (x[6] * cos(delta_s_)
        + x[8]
        - Fy_f * sin(delta_s_)
        - F_resist
        ) / params_.m
        + x[4] * (x[6] * params_.l_f * sin(delta_s_)
        - Fy_r * params_.l_r
        + Fy_f * params_.l_f * cos(delta_s_)) / params_.Iz;
    // x[4]: dY_C / dt in VEHICLE FRAME
    dxdt[4] = (Fy_r
        + Fy_f * cos(delta_s_)
        + x[6] * sin(delta_s_)
        ) / params_.m
        - x[3] * (x[6] * params_.l_f * sin(delta_s_)
        - Fy_r * params_.l_r
        + Fy_f * params_.l_f * cos(delta_s_)) / params_.Iz;
    // x[5]: dpsi / dt
    dxdt[5] = (x[6] * params_.l_f * sin(delta_s_)
        - Fy_r * params_.l_r
        + Fy_f * params_.l_f * cos(delta_s_)) / params_.Iz;

    /* Motor Dynamics as linear second order system */

    // Fx_f
    dxdt[6] = x[7];
    // dFx_f / dt
    dxdt[7] = 1/params_.T_mot * (Fx_f_ - 2 * params_.D_mot * params_.T_mot * x[7] - x[6]);
    // Fx_r
    dxdt[8] = x[9];
    // dFx_r / dt
    dxdt[9] = 1/params_.T_mot * (Fx_r_ - 2 * params_.D_mot * params_.T_mot * x[9] - x[8]);
}