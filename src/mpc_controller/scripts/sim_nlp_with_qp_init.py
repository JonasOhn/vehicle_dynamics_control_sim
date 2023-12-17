import numpy as np
from setup_nlp_ocp_and_sim import load_mpc_yaml_params, setup_nlp_ocp_and_sim
from setup_qp_init import setup_ocp_init
from plot_sim_ocp import plot_nlp_dynamics

CONST_CURV = 0.2


def main(simulate_ocp:bool=True):
    model_params, horizon_params, _, _, _ = load_mpc_yaml_params()

    """ =========== INITIAL STATE FOR SIMULATION ============ """
    # x =         [s,   n,   mu,  vx,  vy,  dpsi]
    x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # x_initializer = [s,   n,   mu,  vy,  dpsi]
    x0_initializer = np.array([x0[0], x0[1], x0[2], x0[4], x0[5]])
    vx_const_0 = 0.1
    kappa_at_stages = np.zeros(horizon_params["N_horizon"])
    u_axm_0 = 0.0

    """ =========== GET SOLVER AND INTEGRATOR ============ """
    ocp_init_solver = setup_ocp_init(x0_initializer)
    ocp_solver, integrator = setup_nlp_ocp_and_sim(x0, simulate_ocp=simulate_ocp)

    if simulate_ocp:
        """ =========== GET DIMS ============ """
        nx = ocp_solver.acados_ocp.dims.nx
        nu = ocp_solver.acados_ocp.dims.nu
        nx_initializer = ocp_init_solver.acados_ocp.dims.nx
        nu_initializer = ocp_init_solver.acados_ocp.dims.nu

        """ =========== SET SIMULATION PARAMS ============ """
        Nsim = 200
        simX = np.ndarray((Nsim+1, nx))
        simU = np.ndarray((Nsim, nu))
        stagesX_initializer = np.ndarray((horizon_params["N_horizon"], nx_initializer))
        stagesU_initializer = np.ndarray((horizon_params["N_horizon"] - 1, nu_initializer))
        s_values = np.zeros((Nsim+1, 1))

        # set initial simulation state
        simX[0,:] = x0
        stagesX_initializer[0,:] = x0_initializer

        # parameterize the problem
        m = model_params['m'] # mass
        g = model_params['g'] # gravity
        l_f = model_params['l_f']
        l_r = model_params['l_r']
        Iz = model_params['Iz'] # yaw moment of inertia
        B_tire = model_params['B_tire'] # pacejka
        C_tire = model_params['C_tire'] # pacejka
        D_tire = model_params['D_tire'] # pacejka
        C_d = model_params['C_d'] # effective drag coefficient
        C_r = model_params['C_r'] # const. rolling resistance
        kappa_ref = CONST_CURV * np.ones(horizon_params['n_s']) # reference curvature along s

        paramvec = np.array((m, g, l_f, l_r, Iz, 
                            B_tire, C_tire, D_tire, C_d, C_r))
        paramvec = np.concatenate((paramvec, kappa_ref))

        for j in range(horizon_params["N_horizon"]):
            ocp_solver.set(j, 'p', paramvec)
        
        # Vectors that contain the solve times at each simulation step
        t_ocp = np.zeros((Nsim))
        t_qp = np.zeros((Nsim))

        # closed loop
        for i in range(Nsim):
            """  Solve QP for initialization """
            # set vx and kappa parameters
            # reset state and input to zero
            for j in range(horizon_params["N_horizon"]):
                ocp_init_solver.set_params_sparse(j, np.array((8,)), np.array((vx_const_0,)))
                ocp_init_solver.set_params_sparse(j, np.array((9,)), np.array((kappa_at_stages[j],)))
                ocp_init_solver.set(j, "x", np.zeros(nx_initializer))
                ocp_init_solver.set(j, "u", np.zeros(nu_initializer))

            # solve QP given previously computed x0 (same x0 as for QCP solver)
            ocp_init_solver.solve_for_x0(x0_bar=x0_initializer)
            
            # retrieve state and input trajectories from QP
            for j in range(horizon_params["N_horizon"]-1):
                stagesU_initializer[j, 0] = ocp_init_solver.get(j, "u")
                stagesX_initializer[j, :] = ocp_init_solver.get(j, "x")
            stagesX_initializer[horizon_params["N_horizon"]-1, :] = ocp_init_solver.get(horizon_params["N_horizon"]-1, "x")

            """  Initialize OCP """
            for j in range(horizon_params["N_horizon"]-1):
                init_state_stage_j = np.array((stagesX_initializer[j, 0], # s
                                               stagesX_initializer[j, 1], # n
                                               stagesX_initializer[j, 2], # mu
                                               vx_const_0, # vx
                                               stagesX_initializer[j, 3], # vy
                                               stagesX_initializer[j, 4], # dpsi
                                               ))
                init_u_stage_j = np.array((u_axm_0, # axm
                                           stagesU_initializer[j, 0], # del_s
                                           ))
                ocp_solver.set(j, "x", init_state_stage_j)
                ocp_solver.set(j, "u", init_u_stage_j)

            init_state_stage_j = np.array((stagesX_initializer[horizon_params["N_horizon"]-1, 0], # s
                                            stagesX_initializer[horizon_params["N_horizon"]-1, 1], # n
                                            stagesX_initializer[horizon_params["N_horizon"]-1, 2], # mu
                                            vx_const_0, # vx
                                            stagesX_initializer[horizon_params["N_horizon"]-1, 3], # vy
                                            stagesX_initializer[horizon_params["N_horizon"]-1, 4], # dpsi
                                            ))
            ocp_solver.set(horizon_params["N_horizon"]-1, "x", init_state_stage_j)

            """  Solve OCP """
            init_state = np.copy(simX[i, :])
            init_state[0] = 0.0
            simU[i,:] = ocp_solver.solve_for_x0(x0_bar = init_state, fail_on_nonzero_status=False)

            # get elapsed solve time
            t_qp[i] = ocp_init_solver.get_stats('time_tot')
            t_ocp[i] = ocp_solver.get_stats('time_tot')

            # simulate system one step
            simX[i+1, :] = integrator.simulate(x=simX[i, :], u=simU[i,:])
            s_values[i+1] = s_values[i] + simX[i+1, 0]
            simX[i+1, 0] = 0.0

            """ Get simulation state for QP initializer """
            # parameter calculation
            vx_const_0 = simX[i+1, 3]
            for j in range(horizon_params["N_horizon"]):
                kappa_at_stages[j] = ocp_solver.get(j, "z")
            # initial state for QP
            x0_initializer = np.array((simX[i+1, 0],
                                       simX[i+1, 1],
                                       simX[i+1, 2],
                                       simX[i+1, 4],
                                       simX[i+1, 5]))
            # initial input for QP
            u_axm_0 = simU[i, 0]

            print("vx0_initializer = ", vx_const_0)

        # scale to milliseconds
        t_ocp *= 1000
        t_qp *= 1000
        t_total = t_ocp + t_qp
        print(f'Computation time (total) in ms: min {np.min(t_total):.3f} median {np.median(t_total):.3f} max {np.max(t_total):.3f}')
        print(f'Computation time (qp) in ms: min {np.min(t_qp):.3f} median {np.median(t_qp):.3f} max {np.max(t_qp):.3f}')
        print(f'Computation time (nlp-ocp) in ms: min {np.min(t_ocp):.3f} median {np.median(t_ocp):.3f} max {np.max(t_ocp):.3f}')

        # plot results
        idx_b_x = ocp_solver.acados_ocp.constraints.idxbx
        lb_x = ocp_solver.acados_ocp.constraints.lbx
        ub_x = ocp_solver.acados_ocp.constraints.ubx
        idx_b_u = ocp_solver.acados_ocp.constraints.idxbu
        lb_u = ocp_solver.acados_ocp.constraints.lbu
        ub_u = ocp_solver.acados_ocp.constraints.ubu

        # plot s values
        simX[:, 0] = np.squeeze(s_values)
        
        plot_nlp_dynamics(np.linspace(0, (horizon_params['T_final']/horizon_params['N_horizon'])*Nsim, Nsim+1),
                    idx_b_x, lb_x, ub_x,
                    idx_b_u, lb_u, ub_u, 
                    simU, simX)

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


if __name__ == '__main__':
    main(simulate_ocp=True)