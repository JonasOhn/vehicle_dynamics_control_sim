from acados_template import AcadosModel
from casadi import MX, vertcat, sin, cos, atan, tanh, atan2
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
    m = MX.sym("m")
    g = MX.sym("g")
    l_f = MX.sym("l_f")
    l_r = MX.sym("l_r")
    Iz = MX.sym("Iz")
    B_tire = MX.sym("B_tire")
    C_tire = MX.sym("C_tire")
    D_tire = MX.sym("D_tire")
    C_d = MX.sym("C_d")
    C_r = MX.sym("C_r")
    kappa_ref = MX.sym("kappa_ref", mpc_horizon_parameters['n_s'], 1)
    p = vertcat(m, g, l_f, l_r, Iz, B_tire, C_tire, D_tire, C_d, C_r, kappa_ref)

    # Symbolic States
    s       = MX.sym('s')
    n      = MX.sym('n')
    mu   = MX.sym('mu')
    vx      = MX.sym('vx')
    vy      = MX.sym('vy')
    dpsi  = MX.sym('dpsi')
    x = vertcat(s, n, mu, vx, vy, dpsi)

    # Symbolic Inputs
    Fx_m = MX.sym('Fx_m')
    del_s = MX.sym('del_s')
    u = vertcat(Fx_m, del_s)

    # Algebraic states
    kappa_ref_algebraic = MX.sym('kappa_ref_algebraic')
    z = vertcat(kappa_ref_algebraic)

    # Symbolic State Derivative f(x,u)
    s_dot       = MX.sym('s_dot')
    n_dot      = MX.sym('n_dot')
    mu_dot   = MX.sym('mu_dot')
    vx_dot      = MX.sym('vx_dot')
    vy_dot      = MX.sym('vy_dot')
    dpsi_dot  = MX.sym('dpsi_dot')
    xdot = vertcat(s_dot, n_dot, mu_dot, vx_dot, vy_dot, dpsi_dot)

    # Symbolic Spline Interpolation for kappa(s)
    s_sym_lut = MX.sym('s_sym_lut')
    ref_path_s = np.linspace(0, mpc_horizon_parameters['s_max'], mpc_horizon_parameters['n_s'])
    interpolant_s2k = ca.interpolant("interpol_spline_kappa", "bspline", [ref_path_s])
    interp_exp = interpolant_s2k(s_sym_lut, kappa_ref)
    interp_fun = ca.Function('interp_fun', [s_sym_lut, kappa_ref], [interp_exp])
    kappa_bspline = interp_fun(s, kappa_ref)

    # ====== If the spline fit should be tested =====
    if testing:
        kappa_fit = kappa_test_gt(ref_path_s)
        plt.plot(ref_path_s, kappa_fit)
        s_test = np.linspace(0.1, mpc_horizon_parameters['s_max'] - 0.1, 10)
        kappa_test = np.zeros_like(s_test)
        for i in range(s_test.shape[0]):
             kappa_test[i] = interp_fun(s_test[i], kappa_fit)
        plt.scatter(s_test, kappa_test)
        plt.grid()
        plt.show()

    # =============== dynamics ========================
    # numerical approximation factor
    eps = 1e-1

    # Slip Angles
    alpha_f = - del_s + atan2((vy + dpsi * l_f), ca.fmax(vx, eps))
    alpha_r = atan2((vy - dpsi * l_r), ca.fmax(vx, eps))
    # lateral forces
    Fz_f = m * g * l_r / (l_r + l_f)
    Fz_r = m * g * l_f / (l_r + l_f)
    # Fy_f = Fz_f * D_tire * sin(C_tire * atan(B_tire * alpha_f))
    # Fy_r = Fz_r * D_tire * sin(C_tire * atan(B_tire * alpha_r))
    Fy_f = Fz_f * D_tire * C_tire * B_tire * alpha_f
    Fy_r = Fz_r * D_tire * C_tire * B_tire * alpha_r

    # derivative of state w.r.t time
    s_dot_expl_dyn = (vx * cos(mu) - vy * sin(mu)) / ((1 - n * kappa_ref_algebraic))
    n_dot_expl_dyn =  vx * sin(mu) + vy * cos(mu)
    mu_dot_expl_dyn = dpsi - kappa_ref_algebraic * (vx * cos(mu) - vy * sin(mu)) / ((1 - n * kappa_ref_algebraic))
    # vx_dot_expl_dyn = 1/m * (Fx_m / 2.0 * (1 + cos(del_s)) - Fy_f * sin(del_s) - (C_r + C_d * vx**2)) + vy * dpsi
    vx_dot_expl_dyn = 1/m * (Fx_m / 2.0 * (1 + cos(del_s)) - Fy_f * sin(del_s) - (C_r)) + vy * dpsi
    vy_dot_expl_dyn = 1/m * (Fy_r + Fx_m / 2.0 * sin(del_s) + Fy_f * cos(del_s)) - vx * dpsi
    dpsi_dot_expl_dyn = 1/Iz * (l_f * (Fx_m / 2.0 * sin(del_s) + Fy_f * cos(del_s)) - Fy_r * l_r)
    kappa_ref_algebraic_expl_dyn = kappa_bspline

    # Explicit expression
    f_expl_time = vertcat(s_dot_expl_dyn,
                          n_dot_expl_dyn, 
                          mu_dot_expl_dyn, 
                          vx_dot_expl_dyn,
                          vy_dot_expl_dyn,
                          dpsi_dot_expl_dyn,
                          kappa_ref_algebraic_expl_dyn)

    # Implicit expression
    f_impl_time = vertcat(xdot, z) - f_expl_time # = 0

    """ STAGE Cost (model-based, slack is defined on the solver) """
    # Progress Rate Cost
    cost_sd = - model_cost_parameters['q_sd'] * s_dot_expl_dyn

    # Stage Cost
    stage_cost = cost_sd

    # ============= ACADOS ===============
    # Acados Model Creation from CasADi symbolic expressions
    model = AcadosModel()

    model.f_impl_expr = f_impl_time
    model.x = x
    model.z = z
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
