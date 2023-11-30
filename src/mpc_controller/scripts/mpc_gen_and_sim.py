from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from acados_template.acados_ocp_solver import ocp_get_default_cmake_builder
from dynamics_model import export_vehicle_ode_model
import numpy as np
from plot_dynamics_sim import plot_dynamics
from os.path import dirname, join, abspath

MPC_T_FINAL = 1.0
MPC_N_HORIZON = 40


def setup_ocp_and_sim(x0, RTI:bool=False, simulate_ocp:bool=True):

    """ ========== OCP Setup ============ """

    # Paths
    ACADOS_PATH = join(dirname(abspath(__file__)), "../../../acados")
    if simulate_ocp:
        codegen_export_dir = '/home/jonas/AMZ/vehicle_dynamics_control_sim/src/mpc_controller/scripts/c_generated_solver_mpc_sim'
        ocp_solver_json_path = '/home/jonas/AMZ/vehicle_dynamics_control_sim/src/mpc_controller/scripts/acados_ocp_sim.json'
    else:
        codegen_export_dir = '/home/jonas/AMZ/vehicle_dynamics_control_sim/src/mpc_controller/scripts/c_generated_solver_mpc'
        ocp_solver_json_path = '/home/jonas/AMZ/vehicle_dynamics_control_sim/src/mpc_controller/scripts/acados_ocp.json'

    # Set up optimal control problem
    ocp = AcadosOcp()
    # Set code export directory
    ocp.code_export_directory = codegen_export_dir
    # set header paths
    ocp.acados_include_path  = f'{ACADOS_PATH}/include'
    ocp.acados_lib_path      = f'{ACADOS_PATH}/lib'

    """ ========== MODEL ============ """
    mpc_horizon_parameters = {}
    # final time for prediction horizon
    mpc_horizon_parameters['T_final'] = MPC_T_FINAL
    # number of steps along horizon
    mpc_horizon_parameters['N_horizon'] = MPC_N_HORIZON
    # number of samples along reference path
    mpc_horizon_parameters['n_s'] = 40
    # maximum distance along reference path
    mpc_horizon_parameters['s_max'] = 20

    # external cost is defined inside the model
    model_cost_parameters = {}
    # cost weight input: dFx
    model_cost_parameters['r_dFx'] = 1e-6
    # cost weight input: ddel_s
    model_cost_parameters['r_del_s'] = 10.0

    # cost weight: beta deviaiton from kinematic
    model_cost_parameters['q_beta'] = 300.0
    # cost weight: progress rate
    model_cost_parameters['q_sd'] = 1.0
    # cost weight: steering angle
    model_cost_parameters['q_del'] = 5.0
    # cost weight: deviation from ref path lateral
    model_cost_parameters['q_n'] = 5.0

    # terminal cost weight: yaw rate
    model_cost_parameters['q_r_e'] = 1000.0
    # terminal cost weight: heading difference
    model_cost_parameters['q_mu_e'] = 1.0
    # terminal cost weight: deviation from ref path lateral
    model_cost_parameters['q_n_e'] = 100.0

    # nonlinear constraints also defined inside the model
    model_constraint_parameters = {}
    # tire ellipse mult. factor for Fx
    model_constraint_parameters['e_x_front'] = 1.2
    model_constraint_parameters['e_x_rear'] = 1.2
    # tire ellipse: e_y together with D_tire defines max. e
    model_constraint_parameters['e_y_front'] = 1.2
    model_constraint_parameters['e_y_rear'] = 1.2
    # track boundary: max. lateral deviation from ref. path
    #TODO: make this parameter for online track boundaries
    model_constraint_parameters['n_max'] = 3.0
    model_constraint_parameters['n_min'] = -3.0
    # slacked track boundary constraint: vehicle geometry (padded rectangle)
    
    l_f = 0.8 # length cog to front axle
    l_r = 0.7 # length cog to rear axle
    
    model_constraint_parameters['L_F'] = 1.2 * l_f
    model_constraint_parameters['L_R'] = 1.2 * l_r
    model_constraint_parameters['W'] = 2.0
    # stages: slacked velocity constraint
    model_constraint_parameters['vx_max'] = 10.0
    # stages: slacked velocity constraint
    model_constraint_parameters['vx_max_e'] = 6.0


    # Get AcadosModel form other python file
    model = export_vehicle_ode_model(mpc_horizon_parameters=mpc_horizon_parameters,
                                     model_cost_parameters=model_cost_parameters,
                                     model_constraint_parameters=model_constraint_parameters)
    ocp.model = model

    """ ========== DIMENSIONS ============ """
    # Dimensions
    # x: state
    # u: input
    # y: optimization variable vector for cost (output)
    # y: optimization variable vector for cost terminal (output)
    # N: prediction horizon number of stages

    nx = model.x.size()[0]
    nu = model.u.size()[0]
    ny = nx + nu
    ny_e = nx
    ocp.dims.N = mpc_horizon_parameters['N_horizon']

    """ ========= INTERNAL STAGE CONSTRAINTS ======== """
    # Terminal Nonlinear Constraints: lower bounds are -Inf but acados doesn't seem to support it
    ocp.constraints.lh = -1e15 * np.ones(8)
    # Terminal Nonlinear Constraints: upper bounds
    ocp.constraints.uh = np.zeros(8)

    # Terminal Nonlinear Constraints: lower bounds are -Inf but acados doesn't seem to support it
    ocp.constraints.lh_0 = -1e15 * np.ones(8)
    # Terminal Nonlinear Constraints: upper bounds
    ocp.constraints.uh_0 = np.zeros(8)

    # ---
    # State: [s, n, mu, vx, vy, dpsi, Fx, del_s]
    # State Constraints: lower bounds
    lb_x = [-100.0,
            -2.0,
            -0.1,
            -5.0,
            -5.0,
            -3000.0,
            -1.0]
    # State Constraints: upper bounds
    ub_x = [100.0,
            3.0,
            100.0,
            5.0,
            5.0,
            3000.0,
            1.0]
    # State Constraints: indices of lb and ub in State vector
    idx_b_x = [1, 2, 3, 4, 5, 6, 7]

    ocp.constraints.lbx = np.array(lb_x)
    ocp.constraints.ubx = np.array(ub_x)
    ocp.constraints.idxbx = np.array(idx_b_x)

    # ---

    # Input: [dFx, ddel_s]
    # Input Constraints: lower bounds
    lb_u = [-1e3,
            -2.0]
    # Input Constraints: upper bounds
    ub_u = [+1e3,
            +2.0]
    # Input Constraints: indices of lb and ub in input vector
    idx_b_u = [0, 1]

    ocp.constraints.lbu = np.array(lb_u)
    ocp.constraints.ubu = np.array(ub_u)
    ocp.constraints.idxbu = np.array(idx_b_u)

    """ ========= INTERNAL TERMINAL CONSTRAINTS ======== """
    # Terminal Nonlinear Constraints: lower bounds are -Inf but acados doesn't seem to support it
    ocp.constraints.lh_e = -1e15 * np.ones(8)
    # Terminal Nonlinear Constraints: upper bounds
    ocp.constraints.uh_e = np.zeros(8)

    """ ========= INITIAL STATE CONSTRAINT =========== """
    ocp.constraints.x0 = x0

    """ ======== STAGE SLACK COST ========== """
    # (model cost inside ocp.model) --> cost type external
    ocp.cost.cost_type = 'EXTERNAL'

    # Slack Weights
    S_n_lin = 100.0
    S_n_quad = 300.0
    S_vx_lin = 0.1
    S_vx_quad = 0.1
    S_te_lin = 5000.0
    S_te_quad = 2000.0
    # Quadratic Slack Cost weights
    ocp.cost.Zu = np.diag(np.array([S_te_quad,
                                    S_te_quad,
                                    S_n_quad,
                                    S_n_quad,
                                    S_n_quad,
                                    S_n_quad,
                                    S_vx_quad]))
    ocp.cost.Zl = np.diag(np.zeros(7))
    # Linear Slack Cost Weights
    ocp.cost.zu = np.array([S_te_lin,
                            S_te_lin,
                            S_n_lin,
                            S_n_lin,
                            S_n_lin,
                            S_n_lin,
                            S_vx_lin])
    ocp.cost.zl = np.zeros(7)
    ocp.constraints.idxsh = np.array([0, 1, 2, 3, 4, 5, 7])

    ocp.cost.Zu_0 = np.copy(ocp.cost.Zu)
    ocp.cost.Zl_0 = np.copy(ocp.cost.Zl)
    ocp.cost.zu_0 = np.copy(ocp.cost.zu)
    ocp.cost.zl_0 = np.copy(ocp.cost.zl)
    ocp.constraints.idxsh_0 = np.copy(ocp.constraints.idxsh)

    """ ======== TERMINAL SLACK COST ========== """
    # Slack Weights
    S_n_lin_e = 100.0
    S_n_quad_e = 300.0
    S_vx_lin_e = 500.0
    S_vx_quad_e = 1.0
    S_te_lin_e = 5000.0
    S_te_quad_e = 2000.0
    # Quadratic Slack Cost weights
    ocp.cost.Zu_e = np.diag(np.array([S_te_quad_e,
                                    S_te_quad_e,
                                    S_n_quad_e,
                                    S_n_quad_e,
                                    S_n_quad_e,
                                    S_n_quad_e,
                                    S_vx_quad_e]))
    ocp.cost.Zl_e = np.diag(np.zeros(7))
    # Linear Slack Cost Weights
    ocp.cost.zu_e = np.array([S_te_lin_e,
                            S_te_lin_e,
                            S_n_lin_e,
                            S_n_lin_e,
                            S_n_lin_e,
                            S_n_lin_e,
                            S_vx_lin_e])
    ocp.cost.zl_e = np.zeros(7)

    # Indices of slacks in terminal nonlinear constraint
    ocp.constraints.idxsh_e = np.array([0, 1, 2, 3, 4, 5, 7])

    """ ============ SOLVER OPTIONS ================== """
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = 'EXACT' #'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'IRK'
    if simulate_ocp:
        ocp.solver_options.sim_method_newton_iter = 20
    ocp.solver_options.nlp_solver_max_iter = 100
    ocp.solver_options.qp_solver_iter_max = 50
    ocp.solver_options.tol = 1e-2
    ocp.solver_options.qp_tol = 1e-3

    if RTI:
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    else:
        ocp.solver_options.nlp_solver_type = 'SQP'

    # Set prediction horizon
    ocp.solver_options.tf = MPC_T_FINAL

    """ ============ INITIAL PARAMETER VALUES ================== """
    m = 220 # mass
    g = 9.81 # gravity
    Iz = 100 # yaw moment of inertia
    B_tire = 9.5 # pacejka
    C_tire = 1.4 # pacejka
    D_tire = -1.0 # pacejka
    C_d = 1.133 # effective drag coefficient
    C_r = 0.5 # const. rolling resistance
    blending_factor = 0.0 # blending between kinematic and dynamic model
    kappa_ref = 0.1 * np.ones(mpc_horizon_parameters['n_s']) # reference curvature along s

    paramvec = np.array((m, g, l_f, l_r, Iz, 
                         B_tire, C_tire, D_tire, C_d, C_r, blending_factor))
    paramvec = np.concatenate((paramvec, kappa_ref))
    ocp.parameter_values = paramvec

    """ ====== CREATE OCP AND SIM SOLVERS =========== """

    cmake_builder = ocp_get_default_cmake_builder()
    # cmake_builder = None
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file = ocp_solver_json_path,
                                        cmake_builder=cmake_builder)

    # create an integrator with the same settings as used in the OCP solver.
    if simulate_ocp:
        acados_integrator = AcadosSimSolver(ocp, json_file = ocp_solver_json_path)
    else:
        acados_integrator = None

    return acados_ocp_solver, acados_integrator


def main(use_RTI:bool=False, simulate_ocp:bool=True):
    """ =========== INITIAL STATE FOR SIMULATION ============ """
    # x =         [s,   n,   mu,  vx,  vy,  dpsi, Fx_m, del_s]
    x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    """ =========== GET SOLVER AND INTEGRATOR ============ """
    ocp_solver, integrator = setup_ocp_and_sim(x0, use_RTI, simulate_ocp=simulate_ocp)

    if simulate_ocp:
        """ =========== GET DIMS ============ """
        nx = ocp_solver.acados_ocp.dims.nx
        nu = ocp_solver.acados_ocp.dims.nu

        """ =========== GET PARAMS ============ """
        param_vec = ocp_solver.acados_ocp.parameter_values

        """ =========== SET SIMULATION PARAMS ============ """
        Nsim = 500
        simX = np.ndarray((Nsim+1, nx))
        simU = np.ndarray((Nsim, nu))
        simBlendingFactor = np.ndarray((Nsim, 1))

        simX[0,:] = x0

        if use_RTI:
            t_preparation = np.zeros((Nsim))
            t_feedback = np.zeros((Nsim))

        else:
            t = np.zeros((Nsim))

        # do some initial iterations to start with a good initial guess
        num_iter_initial = 30
        for _ in range(num_iter_initial):
            ocp_solver.solve_for_x0(x0_bar = x0)

        # closed loop
        for i in range(Nsim):
            if use_RTI:

                # preparation phase
                ocp_solver.options_set('rti_phase', 1)
                status = ocp_solver.solve()
                t_preparation[i] = ocp_solver.get_stats('time_tot')

                # Set initial State
                init_state = np.copy(simX[i, :])
                init_state[0] = 0.0
                ocp_solver.set(0, "lbx", init_state)
                ocp_solver.set(0, "ubx", init_state)

                # feedback phase
                ocp_solver.options_set('rti_phase', 2)
                status = ocp_solver.solve()
                t_feedback[i] = ocp_solver.get_stats('time_tot')

                simU[i, :] = ocp_solver.get(0, "u")

            else:
                # solve ocp and get next control input
                init_state = np.copy(simX[i, :])
                init_state[0] = 0.0
                simU[i,:] = ocp_solver.solve_for_x0(x0_bar = init_state)

                t[i] = ocp_solver.get_stats('time_tot')

            # simulate system
            simX[i+1, :] = integrator.simulate(x=simX[i, :], u=simU[i,:])

            # States
            vx = simX[i+1, 3]
            vy = simX[i+1, 4]
            dpsi = simX[i+1, 5]

            # Inputs
            Fx_f = simU[i, 0]
            del_s = simU[i, 1]

            # Parameters
            m = param_vec[0]
            g = param_vec[1]
            l_f = param_vec[2]
            l_r = param_vec[3]
            B_tire = param_vec[5]
            C_tire = param_vec[6]
            D_tire = param_vec[7]

            # Slip Angles
            alpha_f = - del_s + np.arctan2((vy + dpsi * l_f), vx)
            alpha_r = np.arctan2((vy - dpsi * l_r), vx)

            # Lateral forces
            Fz_f = m * g * l_r / (l_r + l_f)
            Fz_r = m * g * l_f / (l_r + l_f)
            Fy_f = Fz_f * D_tire * np.sin(C_tire * np.arctan(B_tire * alpha_f))
            Fy_r = Fz_r * D_tire * np.sin(C_tire * np.arctan(B_tire * alpha_r))

            ay = 1/m * (Fy_r + Fx_f * np.sin(del_s) + Fy_f * np.cos(del_s)) - vx * dpsi

            slope_ay_blend = 5.0
            ay_kin2dyn = 1.0

            # Sigmoid for blending between kinematic and dynamic model
            blending_factor = 1 / (1 + np.exp(- slope_ay_blend * (np.abs(ay) - ay_kin2dyn)))
            if vx > 0.5:
                blending_factor = 1.0
            else:
                blending_factor = 0.0
            simBlendingFactor[i] = blending_factor

            param_vec[10] = blending_factor

            for j in range(MPC_N_HORIZON):
                ocp_solver.set(j, 'p', param_vec)

        # evaluate timings
        if use_RTI:
            # scale to milliseconds
            t_preparation *= 1000
            t_feedback *= 1000
            print(f'Computation time in preparation phase in ms: \
                    min {np.min(t_preparation):.3f} median {np.median(t_preparation):.3f} max {np.max(t_preparation):.3f}')
            print(f'Computation time in feedback phase in ms:    \
                    min {np.min(t_feedback):.3f} median {np.median(t_feedback):.3f} max {np.max(t_feedback):.3f}')
        else:
            # scale to milliseconds
            t *= 1000
            print(f'Computation time in ms: min {np.min(t):.3f} median {np.median(t):.3f} max {np.max(t):.3f}')

        # plot results
        idx_b_x = ocp_solver.acados_ocp.constraints.idxbx
        lb_x = ocp_solver.acados_ocp.constraints.lbx
        ub_x = ocp_solver.acados_ocp.constraints.ubx
        idx_b_u = ocp_solver.acados_ocp.constraints.idxbu
        lb_u = ocp_solver.acados_ocp.constraints.lbu
        ub_u = ocp_solver.acados_ocp.constraints.ubu
        plot_dynamics(np.linspace(0, (MPC_T_FINAL/MPC_N_HORIZON)*Nsim, Nsim+1),
                    idx_b_x, lb_x, ub_x,
                    idx_b_u, lb_u, ub_u, 
                    simU, simX,
                    simBlendingFactor,
                    plot_constraints=False)

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
    main(use_RTI=True, simulate_ocp=True)
    main(use_RTI=True, simulate_ocp=False)