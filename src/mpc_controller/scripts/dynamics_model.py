from acados_template import AcadosModel
from casadi import SX, vertcat, sin, cos, atan, tanh, atan2
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt


def export_vehicle_ode_model(testing : bool = False,
                             mpc_horizon_parameters : dict = {},
                             model_cost_parameters : dict = {}) -> AcadosModel:

    # model name
    model_name = 'veh_dynamics_ode'

    # ============== Symbolics in CasADi ===========
    # Symbolic Parameters
    m = SX.sym("m")
    g = SX.sym("g")
    l_f = SX.sym("l_f")
    l_r = SX.sym("l_r")
    Iz = SX.sym("Iz")
    B_tire = SX.sym("B_tire")
    C_tire = SX.sym("C_tire")
    D_tire = SX.sym("D_tire")
    C_d = SX.sym("C_d")
    C_r = SX.sym("C_r")
    kappa_ref = SX.sym("kappa_ref")
    p = vertcat(m, g, l_f, l_r, Iz, B_tire, C_tire, D_tire, C_d, C_r, kappa_ref)

    # Symbolic States
    s       = SX.sym('s')
    n      = SX.sym('n')
    mu   = SX.sym('mu')
    vx      = SX.sym('vx')
    vy      = SX.sym('vy')
    dpsi  = SX.sym('dpsi')
    x = vertcat(s, n, mu, vx, vy, dpsi)

    # Symbolic Inputs
    ax_m = SX.sym('ax_m')
    del_s = SX.sym('del_s')
    u = vertcat(ax_m, del_s)

    # Symbolic State Derivative f(x,u)
    s_dot       = SX.sym('s_dot')
    n_dot      = SX.sym('n_dot')
    mu_dot   = SX.sym('mu_dot')
    vx_dot      = SX.sym('vx_dot')
    vy_dot      = SX.sym('vy_dot')
    dpsi_dot  = SX.sym('dpsi_dot')
    xdot = vertcat(s_dot, n_dot, mu_dot, vx_dot, vy_dot, dpsi_dot)

    # =============== dynamics ========================
    # numerical approximation factor
    eps = 1e-3

    # Slip Angles
    alpha_f = - del_s + (vy + l_f * dpsi) / (vx + eps)
    alpha_r = (vy - l_r * dpsi) / (vx + eps)

    # lateral forces
    Fz_f = m * g * l_r / (l_r + l_f)
    Fz_r = m * g * l_f / (l_r + l_f)
    Fy_f = Fz_f * D_tire * C_tire * B_tire * alpha_f
    Fy_r = Fz_r * D_tire * C_tire * B_tire * alpha_r

    # derivative of state w.r.t time
    s_dot_expl_dyn = (vx * cos(mu) - vy * sin(mu)) / ((1 - n * kappa_ref))
    n_dot_expl_dyn =  vx * sin(mu) + vy * cos(mu)
    mu_dot_expl_dyn = dpsi - kappa_ref * (vx * cos(mu) - vy * sin(mu)) / ((1 - n * kappa_ref))
    vx_dot_expl_dyn = 1/m * (m * ax_m / 2.0 * (1 + cos(del_s)) - Fy_f * sin(del_s) - (C_r + C_d * vx**2)) + vy * dpsi
    vy_dot_expl_dyn = 1/m * (Fy_r + m * ax_m / 2.0 * sin(del_s) + Fy_f * cos(del_s)) - vx * dpsi
    dpsi_dot_expl_dyn = 1/Iz * (l_f * (m * ax_m / 2.0 * sin(del_s) + Fy_f * cos(del_s)) - Fy_r * l_r)

    # Explicit expression
    f_expl_time = vertcat(s_dot_expl_dyn,
                          n_dot_expl_dyn, 
                          mu_dot_expl_dyn, 
                          vx_dot_expl_dyn,
                          vy_dot_expl_dyn,
                          dpsi_dot_expl_dyn)

    # Implicit expression
    f_impl_time = vertcat(xdot) - f_expl_time # = 0

    """ STAGE Cost (model-based, slack is defined on the solver) """
    # Progress Rate Cost
    # cost_sd = - model_cost_parameters['q_sd'] * s_dot_expl_dyn
    cost_sd = model_cost_parameters['q_sd'] * (vx - 10.0)**2
    cost_n = model_cost_parameters['q_n'] * n**2
    cost_mu =  model_cost_parameters['q_mu'] * mu**2
    cost_vy =  model_cost_parameters['q_vy'] * vy**2
    cost_dpsi =  model_cost_parameters['q_dpsi'] * dpsi**2
    cost_dels = model_cost_parameters['r_dels'] * del_s**2
    cost_ax = model_cost_parameters['r_ax'] * ax_m**2

    # Stage Cost
    stage_cost = cost_sd + cost_n + cost_mu + cost_vy + cost_dpsi + cost_dels + cost_ax


    # ============= ACADOS ===============
    # Acados Model Creation from CasADi symbolic expressions
    model = AcadosModel()

    model.f_impl_expr = f_impl_time
    model.x = x
    model.xdot = xdot
    model.u = u
    model.p = p
    model.name = model_name
    model.cost_expr_ext_cost = stage_cost

    return model


def kappa_test_gt(s):
    return 0.5 * np.sin(s) - 0.2 * np.cos(2 * s)
    # return 1.2 * s


if __name__ == '__main__':
    export_vehicle_ode_model(testing=True)
