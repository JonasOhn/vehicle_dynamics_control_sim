#include <boost/numeric/odeint.hpp>
#include "sim_backend/dynamic_system.hpp"

using namespace boost::numeric::odeint;

DynamicSystem::DynamicSystem(){
    // Constructor for some standard parameter system
    Fx_f_ = 0.0;
    Fx_r_ = 0.0;
    delta_s_ = 0.0;
    l_ = 0.0;
}

void DynamicSystem::update_parameters(parameters param_struct){
    params_.l_f = param_struct.l_f; // m
    params_.l_r = param_struct.l_r; // m
    l_ = param_struct.l_f + param_struct.l_r;
    params_.m = param_struct.m; // kg
    params_.Iz = param_struct.Iz; // kg m m
    params_.g = param_struct.g; // m s-2
    params_.D_tire = param_struct.D_tire;
    params_.C_tire = param_struct.C_tire;
    params_.B_tire = param_struct.B_tire;
    params_.C_d = param_struct.C_d; // 0.5 * rho * CdA
    params_.C_r = param_struct.C_r; // -
    params_.T_mot = param_struct.T_mot;
    params_.D_mot = param_struct.D_mot;
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