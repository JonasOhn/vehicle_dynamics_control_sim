from acados_template import AcadosModel
from casadi import MX, vertcat, sin, cos, atan, tanh, atan2
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt


def export_vehicle_ode_init_model() -> AcadosModel:

    # model name
    model_name = 'veh_kinematics_ode_init'

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
    vx_const = MX.sym("vx_sym")
    kappa_const = MX.sym("kappa_ref_const")
    p = vertcat(m, 
                g, 
                l_f, 
                l_r, 
                Iz, 
                B_tire, C_tire, D_tire,
                vx_const,
                kappa_const)

    # Symbolic States
    s = MX.sym('s')
    n = MX.sym('n')
    mu = MX.sym("mu")
    vy = MX.sym("vy")
    dpsi = MX.sym("dpsi")
    x = vertcat(s, n, mu, vy, dpsi)

    # Symbolic Inputs
    del_s = MX.sym('del_s')
    u = vertcat(del_s)

    # Symbolic State Derivative f(x,u)
    s_dot = MX.sym('s_dot')
    n_dot = MX.sym('n_dot')
    mu_dot = MX.sym('mu_dot')
    vy_dot = MX.sym('vy_dot')
    dpsi_dot = MX.sym('dpsi_dot')
    xdot = vertcat(s_dot, n_dot, mu_dot, vy_dot, dpsi_dot)

    # =============== differential equations ========================
    # normal tire loads
    Fz_f = m * g * l_r / (l_r + l_f)
    Fz_r = m * g * l_f / (l_r + l_f)
    # lateral tire stiffnesses
    c_alpha_f = Fz_f * B_tire * C_tire * D_tire
    c_alpha_r = Fz_r * B_tire * C_tire * D_tire
    # Linearized side slip angles
    alpha_f = - del_s + (vy + l_f * dpsi) / vx_const
    alpha_r = (vy - l_r * dpsi) / vx_const
    # Linearized lateral forces
    Fy_f = c_alpha_f * alpha_f
    Fy_r = c_alpha_r * alpha_r

    # derivative of state w.r.t time
    s_dot_expl = vx_const
    n_dot_expl = vx_const * mu + vy
    mu_dot_expl = dpsi - kappa_const * vx_const
    vy_dot_expl = - vx_const * dpsi + (Fy_f + Fy_r) / m
    dpsi_dot_expl = (l_f * Fy_f - l_r * Fy_r) / Iz

    # Explicit expression
    f_expl_time = vertcat(s_dot_expl,
                          n_dot_expl,
                          mu_dot_expl,
                          vy_dot_expl,
                          dpsi_dot_expl)

    # Implicit expression
    f_impl_time = vertcat(xdot) - f_expl_time # = 0

    # ============= ACADOS ===============
    # Acados Model Creation from CasADi symbolic expressions
    model = AcadosModel()

    model.f_impl_expr = f_impl_time
    model.f_expl_expr = f_expl_time
    model.x = x
    model.xdot = xdot
    model.u = u
    model.p = p
    model.name = model_name

    return model

if __name__ == '__main__':
    export_vehicle_ode_init_model()
