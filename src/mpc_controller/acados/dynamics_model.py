from acados_template import AcadosModel
from casadi import SX, MX, Function, vertcat, sin, cos, atan, interpolant
import numpy as np

def export_vehicle_ode_model() -> AcadosModel:

    model_name = 'veh_dynamics_ode'

    N_refpath = 50

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
    kappa_ref = MX.sym("kappa_ref", N_refpath, 1)
    s_ref = MX.sym("s_ref", N_refpath, 1)
    p = vertcat(m, g, l_f, l_r, Iz, B_tire, C_tire, D_tire, C_d, C_r, kappa_ref, s_ref)

    # States
    s      = SX.sym('s')
    n      = SX.sym('n')
    mu   = SX.sym('mu')
    vx      = SX.sym('vx')
    vy      = SX.sym('vy')
    dpsi  = SX.sym('dpsi')

    x = vertcat(s, n, mu, vx, vy, dpsi)

    # Interpolate function from s to kappa(s)
    ref_path_s = np.linspace(0, 1, N_refpath)

    interpol_path_x = interpolant("interpol_spline_x", "bspline", [ref_path_s])
    interp_exp = interpol_path_x(s, kappa_ref)
    kapparef_s = Function('interp_fun', [s, kappa_ref], [interp_exp])

    # # test interp_fun

    # x_test = 0.5
    # p_test = np.linspace(0, 1, N_refpath)
    # y_test = interp_fun(x_test, p_test)
    # print(y_test)

    # p_test = np.linspace(0, 2, N_refpath)
    # y_test = interp_fun(x_test, p_test)
    # print(y_test)

    # # generate C code
    # interp_fun.generate() 


    # Inputs
    Fx_f = SX.sym('Fx_f')
    Fx_r = SX.sym('Fx_r')
    del_s = SX.sym('del_s')
    u = vertcat(Fx_f, Fx_r, del_s)

    # xdot
    s_dot      = SX.sym('s_dot')
    n_dot      = SX.sym('n_dot')
    mu_dot   = SX.sym('mu_dot')
    vx_dot      = SX.sym('vx_dot')
    vy_dot      = SX.sym('vy_dot')
    dpsi_dot  = SX.sym('dpsi_dot')

    xdot = vertcat(s_dot, n_dot, mu_dot, vx_dot, vy_dot, dpsi_dot)

    # dynamics
    alpha_f = - del_s + atan((vy + dpsi * l_f)/(vx))
    alpha_r = atan((vy - dpsi * l_r)/(vx))
    Fy_f = m * g * l_r / (l_r + l_f) * D_tire * sin(C_tire * atan(B_tire * alpha_f))
    Fy_r = m * g * l_f / (l_r + l_f) * D_tire * sin(C_tire * atan(B_tire * alpha_r))

    f_expl = vertcat((vx * cos(mu) - vy * sin(mu)) / (1 - n * kapparef_s),
                     vx * sin(mu) + vy * cos(mu),
                     dpsi - kapparef_s * (vx * cos(mu) - vy * sin(mu)) / (1 - n * kapparef_s),
                     1/m * (Fx_r + Fx_f * cos(del_s) - Fy_f * sin(del_s) - C_r - C_d * vx**2) + vy * dpsi,
                     1/m * (Fy_r + Fx_f * sin(del_s) + Fy_f * cos(del_s)) - vx * dpsi,
                     1/Iz * (l_f * cos(del_s) * (Fx_f * sin(del_s) + Fy_f * cos(del_s)) - Fy_r * l_r)
                     )

    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.p = p
    model.name = model_name

    return model

