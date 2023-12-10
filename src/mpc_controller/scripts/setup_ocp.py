import matplotlib.pyplot as plt
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from acados_template.acados_ocp_solver import ocp_get_default_cmake_builder
from dynamics_model import export_vehicle_ode_model
import numpy as np
from plot_dynamics_sim import plot_dynamics
from os.path import dirname, join, abspath
import yaml
import pprint
import shutil
import os
import random

def load_mpc_yaml_params():
    yaml_path = join(dirname(abspath(__file__)), "../config/mpc_controller.yaml")
    with open(yaml_path, 'r') as file:
        yaml_params = yaml.safe_load(file)
    yaml_params = yaml_params['/mpc_controller_node']['ros__parameters']
    model_params = yaml_params['model']
    horizon_params = yaml_params['horizon']
    cost_params = yaml_params['cost']
    constraints_params = yaml_params['constraints']
    solver_options_params = yaml_params['solver_options']
    return model_params, horizon_params, cost_params, constraints_params, solver_options_params

def setup_ocp_and_sim(x0, RTI:bool=False, simulate_ocp:bool=False):
    
    """ Load .yaml file parameters """
    model_params, horizon_params, cost_params, constraints_params, solver_options_params = load_mpc_yaml_params()
    pprint.pprint(model_params)
    pprint.pprint(horizon_params)
    pprint.pprint(cost_params)
    pprint.pprint(constraints_params)
    pprint.pprint(solver_options_params)

    """ ========== OCP Setup ============ """

    # Paths
    ACADOS_PATH = join(dirname(abspath(__file__)), "../../../acados")
    codegen_export_dir = join(dirname(abspath(__file__)), "c_generated_solver_mpc")
    print("Trying to remove codegen folder...")
    try:
        shutil.rmtree(codegen_export_dir)
        print("Codegen folder removed.")
    except FileNotFoundError:
        print("Codegen folder not found and thus not removed.")
    print("Codegen directory: ", codegen_export_dir)
    ocp_solver_json_path = join(dirname(abspath(__file__)), "acados_ocp.json")
    print("Trying to remove codegen .json...")
    try:
        os.remove(ocp_solver_json_path)
        print("Codegen .json removed.")
    except FileNotFoundError:
        print("Codegen .json not found and thus not removed.")
    print("Solver .json directory: ", ocp_solver_json_path)

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
    mpc_horizon_parameters['T_final'] = horizon_params['T_final']
    # number of steps along horizon
    mpc_horizon_parameters['N_horizon'] = horizon_params['N_horizon']
    # number of samples along reference path
    mpc_horizon_parameters['n_s'] = horizon_params['n_s']
    # maximum distance along reference path
    mpc_horizon_parameters['s_max'] = horizon_params['s_max']

    # external cost is defined inside the model
    model_cost_parameters = {}
    # cost weight: progress rate
    model_cost_parameters['q_sd'] = cost_params['q_sd']

    # Get AcadosModel form other python file
    model = export_vehicle_ode_model(mpc_horizon_parameters=mpc_horizon_parameters,
                                     model_cost_parameters=model_cost_parameters)
    ocp.model = model


    """ ========== DIMENSIONS ============ """
    # Dimensions
    # x: state
    # u: input
    # N: prediction horizon number of stages

    ocp.dims.N = mpc_horizon_parameters['N_horizon']


    """ ========= CONSTRAINT: INITIAL STATE =========== """
    ocp.constraints.x0 = x0


    """ ========= CONSTRAINTS: STAGE STATE ======== """
    # ---
    # State: [s, n, mu, vx, vy, dpsi]
    # State Constraints: lower bounds
    ocp.constraints.lbx = np.array((constraints_params['soft']['lb_n'],
                                    constraints_params['soft']['lb_mu'],
                                    constraints_params['soft']['lb_vx'],
                                    constraints_params['soft']['lb_vy'],
                                    constraints_params['soft']['lb_dpsi']))
    # State Constraints: upper bounds
    ocp.constraints.ubx = np.array((constraints_params['soft']['ub_n'],
                                    constraints_params['soft']['ub_mu'],
                                    constraints_params['soft']['ub_vx'],
                                    constraints_params['soft']['ub_vy'],
                                    constraints_params['soft']['ub_dpsi']))
    # State Constraints: indices of lb and ub in State vector
    ocp.constraints.idxbx = np.array((1, 2, 3, 4, 5))

    """ ========= CONSTRAINTS: STAGE INPUT ======== """
    # ---
    # Input: [Fx_m, del_s]
    # Input Constraints: lower bounds
    ocp.constraints.lbu = np.array((constraints_params['hard']['lb_Fxm'],
                                    constraints_params['hard']['lb_dels']))
    # Input Constraints: upper bounds
    ocp.constraints.ubu = np.array((constraints_params['hard']['ub_Fxm'],
                                    constraints_params['hard']['ub_dels']))
    # Input Constraints: indices of lb and ub in input vector
    ocp.constraints.idxbu = np.array((0, 1))


    """ ========= CONSTRAINTS: TERMINAL STATE ======== """
    # ---
    # Terminal State: [s, n, mu, vx, vy, dpsi]
    # Terminal State Constraints: lower bounds
    ocp.constraints.lbx_e = np.array((constraints_params['soft']['lb_n'],
                                      constraints_params['soft']['lb_mu'],
                                      constraints_params['soft']['lb_vx'],
                                      constraints_params['soft']['lb_vy'],
                                      constraints_params['soft']['lb_dpsi']))
    # Terminal State Constraints: upper bounds
    ocp.constraints.ubx_e = np.array((constraints_params['soft']['ub_n'],
                                      constraints_params['soft']['ub_mu'],
                                      constraints_params['soft']['ub_vx'],
                                      constraints_params['soft']['ub_vy'],
                                      constraints_params['soft']['ub_dpsi']))
    # Terminal State Constraints: indices of lb and ub in State vector
    ocp.constraints.idxbx_e = np.array((1, 2, 3, 4, 5))
    
    """ ========= COST =========== """
    # (model cost inside ocp.model) --> cost type external
    ocp.cost.cost_type = 'EXTERNAL'

    """ ========== STAGE SLACK COST =========== """
    # Slack Weights
    S_n_lin = cost_params['slack_penalties']['linear']['S_n_lin']
    S_n_quad = cost_params['slack_penalties']['quadratic']['S_n_quad']
    S_mu_lin = cost_params['slack_penalties']['linear']['S_mu_lin']
    S_mu_quad = cost_params['slack_penalties']['quadratic']['S_mu_quad']
    S_vx_lin = cost_params['slack_penalties']['linear']['S_vx_lin']
    S_vx_quad = cost_params['slack_penalties']['quadratic']['S_vx_quad']
    S_vy_lin = cost_params['slack_penalties']['linear']['S_vy_lin']
    S_vy_quad = cost_params['slack_penalties']['quadratic']['S_vy_quad']
    S_dpsi_lin = cost_params['slack_penalties']['linear']['S_dpsi_lin']
    S_dpsi_quad = cost_params['slack_penalties']['quadratic']['S_dpsi_quad']

    # Quadratic Slack Cost weights
    ocp.cost.Zu = np.diag(np.array((S_n_quad,
                                    S_mu_quad,
                                    S_vx_quad,
                                    S_vy_quad,
                                    S_dpsi_quad)))
    ocp.cost.Zl = np.diag(np.array((S_n_quad,
                                    S_mu_quad,
                                    S_vx_quad,
                                    S_vy_quad,
                                    S_dpsi_quad)))
    # Linear Slack Cost Weights
    ocp.cost.zu = np.array((S_n_lin,
                            S_mu_lin,
                            S_vx_lin,
                            S_vy_lin,
                            S_dpsi_lin))
    ocp.cost.zl = np.array((S_n_lin,
                            S_mu_lin,
                            S_vx_lin,
                            S_vy_lin,
                            S_dpsi_lin))

    # Indices of slack variables in stage linear constraints
    ocp.constraints.idxsbx = np.array((0, 1, 2, 3, 4))

    """ ======== TERMINAL SLACK COST ========== """
    # Slack Weights
    S_n_lin_e = cost_params['slack_penalties']['linear']['S_n_lin_e']
    S_n_quad_e = cost_params['slack_penalties']['quadratic']['S_n_quad_e']
    S_mu_lin_e = cost_params['slack_penalties']['linear']['S_mu_lin_e']
    S_mu_quad_e = cost_params['slack_penalties']['quadratic']['S_mu_quad_e']
    S_vx_lin_e = cost_params['slack_penalties']['linear']['S_vx_lin_e']
    S_vx_quad_e = cost_params['slack_penalties']['quadratic']['S_vx_quad_e']
    S_vy_lin_e = cost_params['slack_penalties']['linear']['S_vy_lin_e']
    S_vy_quad_e = cost_params['slack_penalties']['quadratic']['S_vy_quad_e']
    S_dpsi_lin_e = cost_params['slack_penalties']['linear']['S_dpsi_lin_e']
    S_dpsi_quad_e = cost_params['slack_penalties']['quadratic']['S_dpsi_quad_e']

    # Quadratic Slack Cost weights
    ocp.cost.Zu_e = np.diag(np.array((S_n_quad_e,
                                      S_mu_quad_e,
                                      S_vx_quad_e,
                                      S_vy_quad_e,
                                      S_dpsi_quad_e)))
    ocp.cost.Zl_e = np.diag(np.array((S_n_quad_e,
                                      S_mu_quad_e,
                                      S_vx_quad_e,
                                      S_vy_quad_e,
                                      S_dpsi_quad_e)))
    # Linear Slack Cost Weights
    ocp.cost.zu_e = np.array((S_n_lin_e,
                              S_mu_lin_e,
                              S_vx_lin_e,
                              S_vy_lin_e,
                              S_dpsi_lin_e))
    ocp.cost.zl_e = np.array((S_n_lin_e,
                              S_mu_lin_e,
                              S_vx_lin_e,
                              S_vy_lin_e,
                              S_dpsi_lin_e))

    # Indices of slack variables in stage linear constraints
    ocp.constraints.idxsbx_e = np.array((0, 1, 2, 3, 4))


    """ ============ SOLVER OPTIONS ================== """
    ocp.solver_options.qp_solver = solver_options_params['qp_solver']
    ocp.solver_options.hessian_approx = solver_options_params['hessian_approx']
    ocp.solver_options.integrator_type = solver_options_params['integrator_type']
    ocp.solver_options.sim_method_newton_iter = solver_options_params['sim_method_newton_iter']
    ocp.solver_options.nlp_solver_max_iter = solver_options_params['nlp_solver_max_iter']
    ocp.solver_options.qp_solver_iter_max = solver_options_params['qp_solver_iter_max']
    ocp.solver_options.tol = solver_options_params['tol']
    ocp.solver_options.qp_tol = solver_options_params['qp_tol']
    ocp.solver_options.alpha_min = solver_options_params['alpha_min']
    ocp.solver_options.qp_solver_warm_start = solver_options_params['qp_solver_warm_start']
    ocp.solver_options.regularize_method = solver_options_params['regularize_method']
    ocp.solver_options.reg_epsilon = solver_options_params['reg_epsilon']

    if RTI:
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    else:
        ocp.solver_options.nlp_solver_type = 'SQP'

    # Set prediction horizon
    ocp.solver_options.tf = horizon_params['T_final']

    """ ============ INITIAL PARAMETER VALUES ================== """
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
    kappa_ref = model_params['kappa_ref'] * np.ones(horizon_params['n_s']) # reference curvature along s
    kappa_ref = 0.2 * np.ones(horizon_params['n_s']) # reference curvature along s

    paramvec = np.array((m, g, l_f, l_r, Iz, 
                         B_tire, C_tire, D_tire, C_d, C_r))
    paramvec = np.concatenate((paramvec, kappa_ref))
    ocp.parameter_values = paramvec

    """ ====== CREATE OCP AND SIM SOLVERS =========== """
    if not simulate_ocp:
        cmake_builder = ocp_get_default_cmake_builder()
    else:
        cmake_builder = None
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file = ocp_solver_json_path,
                                        cmake_builder=cmake_builder)

    # create an integrator with the same settings as used in the OCP solver.
    if simulate_ocp:
        acados_integrator = AcadosSimSolver(ocp, json_file = ocp_solver_json_path,
                                            cmake_builder=cmake_builder)
    else:
        acados_integrator = None

    return acados_ocp_solver, acados_integrator


def main(use_RTI:bool=False, simulate_ocp:bool=True):
    _, horizon_params, _, _, _ = load_mpc_yaml_params()

    """ =========== INITIAL STATE FOR SIMULATION ============ """
    # x =         [s,   n,   mu,  vx,  vy,  dpsi]
    x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    """ =========== GET SOLVER AND INTEGRATOR ============ """
    ocp_solver, integrator = setup_ocp_and_sim(x0, use_RTI, simulate_ocp=simulate_ocp)

    if simulate_ocp:
        """ =========== GET DIMS ============ """
        nx = ocp_solver.acados_ocp.dims.nx
        nu = ocp_solver.acados_ocp.dims.nu

        """ =========== SET SIMULATION PARAMS ============ """
        Nsim = 500
        simX = np.ndarray((Nsim+1, nx))
        simU = np.ndarray((Nsim, nu))
        s_values = np.zeros((Nsim+1, 1))

        simX[0,:] = x0

        if use_RTI:
            t_preparation = np.zeros((Nsim))
            t_feedback = np.zeros((Nsim))

        else:
            t = np.zeros((Nsim))

        # do some initial iterations to start with a good initial guess
        num_iter_initial = 10
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
            s_values[i+1] = s_values[i] + simX[i+1, 0]
            simX[i+1, 0] = 0.0

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
        simX[:, 0] = np.squeeze(s_values)
        plot_dynamics(np.linspace(0, (horizon_params['T_final']/horizon_params['N_horizon'])*Nsim, Nsim+1),
                    idx_b_x, lb_x, ub_x,
                    idx_b_u, lb_u, ub_u, 
                    simU, simX,
                    plot_constraints=True)

        print(ocp_solver)

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