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

import numpy as np
from setup_kinematic_bicycle_curv import load_mpc_yaml_params, setup_nlp_ocp_and_sim
from plot_sim_ocp import plot_nlp_dynamics

# constant curvature horizon for simulation
CONST_CURV = 0.1


def main(use_stepped_sim: bool = False):
    model_params, horizon_params, cost_params, _, _ = load_mpc_yaml_params()

    """ =========== INITIAL STATE FOR SIMULATION ============ """
    # x0 =        [s,   n,   mu,  vx, ax, dels]
    x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    """ =========== GET SOLVER AND INTEGRATOR ============ """
    ocp_solver, integrator = setup_nlp_ocp_and_sim(x0, simulate_ocp=True)

    """ =========== GET DIMS ============ """
    nx = ocp_solver.acados_ocp.dims.nx
    nu = ocp_solver.acados_ocp.dims.nu

    """ =========== GET CONSTRAINTS ========= """
    idx_b_x = ocp_solver.acados_ocp.constraints.idxbx
    lb_x = ocp_solver.acados_ocp.constraints.lbx
    ub_x = ocp_solver.acados_ocp.constraints.ubx
    idx_b_u = ocp_solver.acados_ocp.constraints.idxbu
    lb_u = ocp_solver.acados_ocp.constraints.lbu
    ub_u = ocp_solver.acados_ocp.constraints.ubu

    """ =========== SET SIMULATION PARAMS ============ """
    Nsim = 1000
    stepping_start_idx = 50
    simX = np.ndarray((Nsim + 1, nx))
    simU = np.ndarray((Nsim, nu))

    # get s-state sim vector because it is reset
    # to 0 at the beginning of each prediction horizon
    s_values = np.zeros((Nsim + 1, 1))
    s_values[0, 0] = x0[0]

    # set initial simulation state
    simX[0, :] = x0

    # init prediction trajectories
    X_predict = np.zeros((horizon_params["N_horizon"], nx))
    U_predict = np.zeros((horizon_params["N_horizon"] - 1, nu))

    # parameterize the problem
    m = model_params["m"]  # mass
    l_f = model_params["l_f"]
    l_r = model_params["l_r"]
    C_d = model_params["C_d"]  # effective drag coefficient
    C_r = model_params["C_r"]  # const. rolling resistance
    kappa_ref = CONST_CURV  # reference curvature
    q_n = cost_params["q_n"]
    q_sd = cost_params["q_sd"]
    q_mu = cost_params["q_mu"]
    q_dels = cost_params["q_dels"]
    q_ax = cost_params["q_ax"]
    r_dax = cost_params["r_dax"]
    r_ddels = cost_params["r_ddels"]

    paramvec = np.array(
        (
            m,
            l_f,
            l_r,
            C_d,
            C_r,
            kappa_ref,
            q_n,
            q_sd,
            q_mu,
            q_ax,
            q_dels,
            r_dax,
            r_ddels,
        )
    )
    for j in range(horizon_params["N_horizon"]):
        ocp_solver.set(j, "p", paramvec)
    integrator.set("p", paramvec)

    # Vectors that contain the solve times at each simulation step
    t_ocp = np.zeros((Nsim))

    # closed loop sim (may be stepped)
    for i in range(Nsim):

        """Solve OCP"""
        init_state = np.copy(simX[i, :])
        init_state[0] = 0.0
        simU[i, :] = ocp_solver.solve_for_x0(
            x0_bar=init_state, fail_on_nonzero_status=False
        )

        # get elapsed solve time
        t_ocp[i] = ocp_solver.get_stats("time_tot")

        # simulate system one step
        simX[i + 1, :] = integrator.simulate(x=simX[i, :], u=simU[i, :])
        s_values[i + 1] = s_values[i] + simX[i + 1, 0]
        simX[i + 1, 0] = 0.0

        print(
            f"{100 * (i+1)/Nsim:.2f} % of sim solved, solve time: {1000*t_ocp[i]:.3f} [ms]"
        )

        if use_stepped_sim and i > stepping_start_idx:

            for j in range(horizon_params["N_horizon"] - 1):
                X_predict[j, :] = ocp_solver.get(j, "x")
                U_predict[j, :] = ocp_solver.get(j, "u")

            X_predict[horizon_params["N_horizon"] - 1, :] = ocp_solver.get(
                horizon_params["N_horizon"] - 1, "x"
            )

            t_predict = (
                horizon_params["T_final"] / horizon_params["N_horizon"]
            ) * i + np.linspace(
                0, horizon_params["T_final"], horizon_params["N_horizon"]
            )

            plot_nlp_dynamics(
                np.linspace(
                    0,
                    (horizon_params["T_final"] / horizon_params["N_horizon"]) * i,
                    i + 1,
                ),
                idx_b_x,
                lb_x,
                ub_x,
                idx_b_u,
                lb_u,
                ub_u,
                simU[:i, :],
                simX[: i + 1, :],
                U_predict,
                X_predict,
                t_predict,
            )

            input("Press Enter to continue...")

    # scale to milliseconds
    t_ocp *= 1000
    print(
        f"Computation time (nlp-ocp) in ms: min {np.min(t_ocp):.3f} median {np.median(t_ocp):.3f} max {np.max(t_ocp):.3f}"
    )

    # plot s values
    simX[:, 0] = np.squeeze(s_values)

    plot_nlp_dynamics(
        np.linspace(
            0,
            (horizon_params["T_final"] / horizon_params["N_horizon"]) * Nsim,
            Nsim + 1,
        ),
        idx_b_x,
        lb_x,
        ub_x,
        idx_b_u,
        lb_u,
        ub_u,
        simU,
        simX,
    )

    ocp_solver = None


"""
SQP Solver Status:
    0 --> ACADOS_SUCCESS,
    1 --> ACADOS_FAILURE,
    2 --> ACADOS_MAXITER,
    3 --> ACADOS_MINSTEP,
    4 --> ACADOS_QP_FAILURE,
    5 --> ACADOS_READY

HPIPM Solver Status:
    0 --> SUCCESS, // found solution satisfying accuracy tolerance
    1 --> MAX_ITER, // maximum iteration number reached
    2 --> MIN_STEP, // minimum step length reached
    3 --> NAN_SOL, // NaN in solution detected
    4 --> INCONS_EQ, // unconsistent equality constraints

"""


if __name__ == "__main__":
    main(use_stepped_sim=False)
