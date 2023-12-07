from acados_template import AcadosModel
from casadi import MX, vertcat, sin, cos, atan, tanh, atan2
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt


def export_vehicle_ode_model(testing : bool = False,
                             mpc_horizon_parameters : dict = {},
                             model_cost_parameters : dict = {},
                             model_constraint_parameters : dict = {}) -> AcadosModel:

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
    blending_factor = MX.sym("blending_factor")
    kappa_ref = MX.sym("kappa_ref", mpc_horizon_parameters['n_s'], 1)
    p = vertcat(m, g, l_f, l_r, Iz, B_tire, C_tire, D_tire, C_d, C_r, blending_factor, kappa_ref)

    # Symbolic States
    s       = MX.sym('s')
    n      = MX.sym('n')
    mu   = MX.sym('mu')
    vx      = MX.sym('vx')
    vy      = MX.sym('vy')
    dpsi  = MX.sym('dpsi')
    Fx_m = MX.sym('Fx_m')
    del_s = MX.sym('del_s')
    x = vertcat(s, n, mu, vx, vy, dpsi, Fx_m, del_s)

    # Symbolic Inputs
    Fx_m_dot_u = MX.sym('Fx_m_dot_u')
    del_s_dot_u = MX.sym('del_s_dot_u')
    u = vertcat(Fx_m_dot_u, del_s_dot_u)

    # Algebraic states
    kappa_ref_algebraic = MX.sym('kappa_ref_algebraic')
    alpha_f_algebraic = MX.sym('alpha_f_algebraic')
    alpha_r_algebraic = MX.sym('alpha_r_algebraic')
    z = vertcat(kappa_ref_algebraic, alpha_f_algebraic, alpha_r_algebraic)

    # Symbolic State Derivative f(x,u)
    s_dot       = MX.sym('s_dot')
    n_dot      = MX.sym('n_dot')
    mu_dot   = MX.sym('mu_dot')
    vx_dot      = MX.sym('vx_dot')
    vy_dot      = MX.sym('vy_dot')
    dpsi_dot  = MX.sym('dpsi_dot')
    Fx_m_dot = MX.sym('Fx_m_dot')
    del_s_dot = MX.sym('del_s_dot')
    xdot = vertcat(s_dot, n_dot, mu_dot, vx_dot, vy_dot, dpsi_dot, Fx_m_dot, del_s_dot)

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
    # alpha_f = - del_s + atan2((vy + dpsi * l_f), ca.sign(vx) * ca.fmax(ca.fabs(vx), eps))
    # alpha_r = atan2((vy - dpsi * l_r), ca.sign(vx) * ca.fmax(ca.fabs(vx), eps))
    alpha_f = - del_s + atan2((vy + dpsi * l_f), ca.fmax(vx, eps))
    alpha_r = atan2((vy - dpsi * l_r), ca.fmax(vx, eps))
    # lateral forces
    Fz_f = m * g * l_r / (l_r + l_f)
    Fz_r = m * g * l_f / (l_r + l_f)
    Fy_f = Fz_f * D_tire * sin(C_tire * atan(B_tire * alpha_f))
    Fy_r = Fz_r * D_tire * sin(C_tire * atan(B_tire * alpha_r))

    # derivative of state w.r.t time
    s_dot_expl_dyn = (vx * cos(mu) - vy * sin(mu)) / ((1 - n * kappa_bspline))
    n_dot_expl_dyn =  vx * sin(mu) + vy * cos(mu)
    mu_dot_expl_dyn = dpsi - kappa_bspline * s_dot_expl_dyn
    # vx_dot_expl_dyn = 1/m * (Fx_m / 2.0 * (1 + cos(del_s)) - Fy_f * sin(del_s) - tanh(vx / eps) * (C_r + C_d * vx**2)) + vy * dpsi
    vx_dot_expl_dyn = 1/m * (Fx_m / 2.0 * (1 + cos(del_s)) - Fy_f * sin(del_s) - (C_r + C_d * vx**2)) + vy * dpsi
    vy_dot_expl_dyn = 1/m * (Fy_r + Fx_m / 2.0 * sin(del_s) + Fy_f * cos(del_s)) - vx * dpsi
    dpsi_dot_expl_dyn = 1/Iz * (l_f * (Fx_m / 2.0 * sin(del_s) + Fy_f * cos(del_s)) - Fy_r * l_r)
    Fx_m_dot_expl_dyn = Fx_m_dot_u
    del_s_dot_expl_dyn = del_s_dot_u

    # Kinematic Formulation
    vx_dot_expl_kin = 1/m * (Fx_m - (C_r + C_d * vx**2)) #+ vy * dpsi
    # vx_dot_expl_kin = 1/m * (Fx_m) # + vy * dpsi
    vy_dot_expl_kin = (del_s_dot_u * vx + del_s * 1/m * (Fx_m))*(l_r / (l_f+l_r))
    dpsi_dot_expl_kin = (del_s_dot_u * vx + del_s * 1/m * (Fx_m))*(1 / (l_f+l_r))

    s_dot_expl = s_dot_expl_dyn
    n_dot_expl = n_dot_expl_dyn
    mu_dot_expl = mu_dot_expl_dyn
    vx_dot_expl = blending_factor * vx_dot_expl_dyn + (1 - blending_factor) * vx_dot_expl_kin
    vy_dot_expl = blending_factor * vy_dot_expl_dyn + (1 - blending_factor) * vy_dot_expl_kin
    dpsi_dot_expl = blending_factor * dpsi_dot_expl_dyn + (1 - blending_factor) * dpsi_dot_expl_kin
    Fx_m_dot_expl = Fx_m_dot_expl_dyn
    del_s_dot_expl = del_s_dot_expl_dyn

    # Explicit expression
    f_expl_time = vertcat(s_dot_expl,
                          n_dot_expl, 
                          mu_dot_expl, 
                          vx_dot_expl,
                          vy_dot_expl,
                          dpsi_dot_expl,
                          Fx_m_dot_expl,
                          del_s_dot_expl,
                          kappa_bspline,
                          alpha_f,
                          alpha_r)

    # Implicit expression
    f_impl_time = vertcat(xdot, z) - f_expl_time # = 0

    """ STAGE Cost (model-based, slack is defined on the solver) """
    # Lateral deviation from path cost
    cost_n = model_cost_parameters['q_n'] * n**2

    # Stage cost on heading difference
    cost_mu = model_cost_parameters['q_mu'] * mu**2

    # Steering angle cost
    cost_dels = model_cost_parameters['q_del'] * del_s**2

    # Progress Rate Cost
    cost_sd = - model_cost_parameters['q_sd'] * s_dot_expl

    # Body Slip angle Regularization
    beta_kin = ca.atan(ca.tan(del_s) * l_r / (l_r + l_f))
    beta_dyn = ca.atan(vy / ca.fmax(vx, eps))
    cost_bsa = model_cost_parameters['q_beta'] * (beta_dyn - beta_kin)**2

    # Input Cost
    R_mat = np.diag([model_cost_parameters['r_dFx'],
                     model_cost_parameters['r_ddel_s']])
    cost_u = ca.transpose(u) @ R_mat @ u

    # Stage Cost
    stage_cost = cost_sd + cost_n + cost_dels + cost_u + cost_bsa + cost_mu

    """ TERMINAL Cost (model-based, slack is defined on the solver) """
    # Lateral deviation from path cost
    cost_n_e = model_cost_parameters['q_n_e'] * n**2

    # Heading Difference cost
    cost_mu_e = model_cost_parameters['q_mu_e'] * mu**2

    # yaw rate cost
    cost_r_e = model_cost_parameters['q_r_e'] * dpsi**2

    # Stage Cost
    terminal_cost = cost_n_e + cost_mu_e + cost_r_e

    """" STAGE Constraints """
    e_max_front = D_tire * model_constraint_parameters['e_y_front']
    e_max_rear = D_tire * model_constraint_parameters['e_y_rear']

    bounds_front = model_constraint_parameters['L_F']  * sin(mu)
    bounds_rear = model_constraint_parameters['L_R'] * sin(mu)
    bounds_width = model_constraint_parameters['W'] / 2.0 * cos(mu)

    tire_ellipse_con_f = (Fx_m / 2.0 * model_constraint_parameters['e_x_front'] / (Fz_f * e_max_front))**2 +\
                         (Fy_f / (Fz_f * e_max_front))**2 - 1.0
    
    tire_ellipse_con_r = (Fx_m / 2.0 * model_constraint_parameters['e_x_rear'] / (Fz_r * e_max_rear))**2 +\
                         (Fy_r / (Fz_r * e_max_rear))**2 - 1.0
    
    track_con_fl = n + bounds_front + bounds_width - model_constraint_parameters['n_max']
    track_con_rl = n - bounds_rear + bounds_width - model_constraint_parameters['n_max']
    track_con_fr = - n - bounds_front + bounds_width + model_constraint_parameters['n_min']
    track_con_rr = - n + bounds_rear + bounds_width + model_constraint_parameters['n_min']

    kappa_n_con = kappa_bspline * n - 1

    vx_con = vx - model_constraint_parameters['vx_max']

    stage_constraint_nonlinear = vertcat(tire_ellipse_con_f,
                                         tire_ellipse_con_r,
                                         track_con_fl,
                                         track_con_rl,
                                         track_con_fr,
                                         track_con_rr,
                                         kappa_n_con,
                                         vx_con)

    """" TERMINAL Constraints """
    tire_ellipse_con_f_e = tire_ellipse_con_f
    tire_ellipse_con_r_e = tire_ellipse_con_r
    track_con_fl_e = track_con_fl
    track_con_rl_e = track_con_rl
    track_con_fr_e = track_con_fr
    track_con_rr_e = track_con_rr
    kappa_n_con_e = kappa_n_con
    vx_con_e = vx - model_constraint_parameters['vx_max_e']

    terminal_constraint_nonlinear = vertcat(tire_ellipse_con_f_e,
                                            tire_ellipse_con_r_e,
                                            track_con_fl_e,
                                            track_con_rl_e,
                                            track_con_fr_e,
                                            track_con_rr_e,
                                            kappa_n_con_e,
                                            vx_con_e)

    # ============= ACADOS ===============
    # Acados Model Creation from CasADi symbolic expressions
    model = AcadosModel()

    model.f_impl_expr = f_impl_time
    model.f_expl_expr = f_expl_time
    model.x = x
    model.z = z
    model.xdot = xdot
    model.u = u
    model.p = p
    model.name = model_name
    model.cost_expr_ext_cost = stage_cost
    model.cost_expr_ext_cost_e = terminal_cost
    model.con_h_expr = stage_constraint_nonlinear
    model.con_h_expr_e = terminal_constraint_nonlinear
    model.con_h_expr_0 = stage_constraint_nonlinear

    return model


def kappa_test_gt(s):
    return 0.5 * np.sin(s) - 0.2 * np.cos(2 * s)
    # return 1.2 * s


if __name__ == '__main__':
    export_vehicle_ode_model(testing=True)
