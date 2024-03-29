"""
 * Simple Vehicle Dynamics Simulator Project
 *
 * Copyright (c) 2023-2024 Authors:
 *   - Jonas Ohnemus <johnemus@ethz.ch>
 *
 * All rights reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
"""

from acados_template import AcadosModel
from casadi import SX, vertcat, sin, cos, atan, tanh, atan2, tan
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt


def export_vehicle_ode_model() -> AcadosModel:

    # model name
    model_name = "veh_dynamics_ode"

    # ============== Symbolics in CasADi ===========
    # Symbolic Parameters
    m = SX.sym("m")
    l_f = SX.sym("l_f")
    l_r = SX.sym("l_r")
    C_d = SX.sym("C_d")
    C_r = SX.sym("C_r")
    kappa_ref = SX.sym("kappa_ref")
    q_n = SX.sym("q_n")
    q_sd = SX.sym("q_sd")
    q_mu = SX.sym("q_mu")
    q_dels = SX.sym("q_dels")
    q_ax = SX.sym("q_ax")
    r_dax = SX.sym("r_dax")
    r_ddels = SX.sym("r_ddels")
    p = vertcat(
        m, l_f, l_r, C_d, C_r, kappa_ref, q_n, q_sd, q_mu, q_ax, q_dels, r_dax, r_ddels
    )

    # Symbolic States
    s = SX.sym("s")
    n = SX.sym("n")
    mu = SX.sym("mu")
    vx = SX.sym("vx")
    ax_m = SX.sym("ax_m")
    del_s = SX.sym("del_s")
    x = vertcat(s, n, mu, vx, ax_m, del_s)

    # Symbolic Inputs
    dax_m = SX.sym("dax_m")
    ddel_s = SX.sym("ddel_s")
    u = vertcat(dax_m, ddel_s)

    # Symbolic State Derivative f(x,u)
    s_dot = SX.sym("s_dot")
    n_dot = SX.sym("n_dot")
    mu_dot = SX.sym("mu_dot")
    vx_dot = SX.sym("vx_dot")
    ax_dot = SX.sym("ax_dot")
    dels_dot = SX.sym("dels_dot")
    xdot = vertcat(s_dot, n_dot, mu_dot, vx_dot, ax_dot, dels_dot)

    # =============== dynamics ========================

    vy = tan(del_s) * vx
    dpsi = vy / l_r

    # lateral forces
    Fx = m * ax_m - C_r - C_d * vx**2

    # derivative of state w.r.t time
    s_dot_expl_dyn = (vx * cos(mu) - vy * sin(mu)) / ((1 - n * kappa_ref))
    n_dot_expl_dyn = vx * sin(mu) + vy * cos(mu)
    mu_dot_expl_dyn = dpsi - kappa_ref * (vx * cos(mu) - vy * sin(mu)) / (
        (1 - n * kappa_ref)
    )
    vx_dot_expl_dyn = 1 / m * Fx
    ax_m_dot_expl_dyn = dax_m
    del_s_dot_expl_dyn = ddel_s

    # Explicit expression
    f_expl_time = vertcat(
        s_dot_expl_dyn,
        n_dot_expl_dyn,
        mu_dot_expl_dyn,
        vx_dot_expl_dyn,
        ax_m_dot_expl_dyn,
        del_s_dot_expl_dyn,
    )

    # Implicit expression
    f_impl_time = vertcat(xdot) - f_expl_time  # = 0

    """ STAGE Cost (model-based, slack is defined on the solver) """
    # Progress Rate Cost
    cost_sd = -q_sd * s_dot_expl_dyn
    cost_n = q_n * n**2
    cost_mu = q_mu * mu**2
    cost_dels = q_dels * del_s**2
    cost_ax = q_ax * ax_m**2
    cost_dax_m = r_dax * dax_m**2
    cost_ddel_s = r_ddels * ddel_s**2

    # Stage Cost
    stage_cost = (
        cost_sd + cost_n + cost_mu + cost_dels + cost_ax + cost_dax_m + cost_ddel_s
    )

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
    model.cost_expr_ext_cost = stage_cost

    return model


if __name__ == "__main__":
    export_vehicle_ode_model(testing=True)
