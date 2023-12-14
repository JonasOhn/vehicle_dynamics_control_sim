import matplotlib.pyplot as plt
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from acados_template.acados_ocp_solver import ocp_get_default_cmake_builder
from dynamics_model import export_vehicle_ode_model
from dynamics_model_initializer import export_vehicle_ode_init_model
import numpy as np
from plot_dynamics_sim import plot_dynamics
from os.path import dirname, join, abspath
import yaml
import pprint
import shutil
import os
import scipy.linalg
import random

def load_mpc_initializer_yaml_params():
    yaml_path = join(dirname(abspath(__file__)), "../config/mpc_initializer.yaml")
    with open(yaml_path, 'r') as file:
        yaml_params = yaml.safe_load(file)
    model_params = yaml_params['model']
    horizon_params = yaml_params['horizon']
    cost_params = yaml_params['cost']
    solver_options_params = yaml_params['solver_options']
    return model_params, horizon_params, cost_params, solver_options_params

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


def setup_ocp_init(x0):
    
    """ Load .yaml file parameters """
    model_params, horizon_params, cost_params, solver_options_params = load_mpc_initializer_yaml_params()
    pprint.pprint(model_params)
    pprint.pprint(horizon_params)
    pprint.pprint(cost_params)
    pprint.pprint(solver_options_params)

    """ ========== OCP Setup ============ """

    # Paths
    ACADOS_PATH = join(dirname(abspath(__file__)), "../../../acados")
    codegen_export_dir = join(dirname(abspath(__file__)), "c_generated_solver_mpc_qp_initializer")
    print("Trying to remove codegen folder...")
    try:
        shutil.rmtree(codegen_export_dir)
        print("Codegen folder removed.")
    except FileNotFoundError:
        print("Codegen folder not found and thus not removed.")
    print("Codegen directory: ", codegen_export_dir)
    ocp_solver_json_path = join(dirname(abspath(__file__)), "acados_ocp_qp_init.json")
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

    # Get AcadosModel form other python file
    model = export_vehicle_ode_init_model()
    ocp.model = model


    """ ========== DIMENSIONS ============ """
    # Dimensions
    # x: state
    # u: input
    # N: prediction horizon number of stages

    ocp.dims.N = mpc_horizon_parameters['N_horizon']

    """ ========= CONSTRAINT: INITIAL STATE =========== """
    ocp.constraints.x0 = x0

    
    """ ========= COST =========== """
    # (model cost inside ocp.model) --> cost type external
    ocp.cost.cost_type = 'LINEAR_LS'
    nx = ocp.model.x.size()[0]
    nu = ocp.model.u.size()[0]
    ny = nx + nu
    ny_e = nx

    """ ========== STAGE COST =========== """
    Q = np.diag(np.array((cost_params['q_s'],
                          cost_params['q_n'],
                          cost_params['q_mu'],
                          cost_params['q_vy'],
                          cost_params['q_dpsi'])))
    R = cost_params['r_dels'] * np.eye(1)
    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e = Q

    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx,:nx] = np.eye(nx)

    Vu = np.zeros((ny, nu))
    Vu[nx:nx+nu, :] = np.eye(nu)
    ocp.cost.Vu = Vu

    ocp.cost.Vx_e = np.eye(ny_e)

    ocp.cost.yref = np.zeros(ny)
    ocp.cost.yref_e = np.zeros(ny_e)

    """ ============ SOLVER OPTIONS ================== """
    ocp.solver_options.qp_solver = solver_options_params['qp_solver']
    ocp.solver_options.hessian_approx = solver_options_params['hessian_approx']
    ocp.solver_options.integrator_type = solver_options_params['integrator_type']
    ocp.solver_options.nlp_solver_max_iter = solver_options_params['nlp_solver_max_iter']
    ocp.solver_options.qp_solver_iter_max = solver_options_params['qp_solver_iter_max']
    ocp.solver_options.tol = solver_options_params['tol']
    ocp.solver_options.qp_tol = solver_options_params['qp_tol']
    ocp.solver_options.qp_solver_warm_start = solver_options_params['qp_solver_warm_start']
    ocp.solver_options.hpipm_mode = solver_options_params['hpipm_mode']
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
    vx_const = model_params['vx_const']
    kappa_const = model_params['kappa_const']

    paramvec = np.array((m, g, l_f, l_r, Iz, 
                         B_tire, C_tire, D_tire, 
                         vx_const, kappa_const))
    ocp.parameter_values = paramvec

    """ ====== CREATE OCP AND SIM SOLVERS =========== """
    cmake_builder = ocp_get_default_cmake_builder()
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file = ocp_solver_json_path,
                                        cmake_builder=cmake_builder)

    return acados_ocp_solver


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
    ocp.solver_options.alpha_reduction = solver_options_params['alpha_reduction']
    ocp.solver_options.hpipm_mode = solver_options_params['hpipm_mode']
    ocp.solver_options.cost_discretization = solver_options_params['cost_discretization']

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

    # x_initializer = [s,   n,   mu,  vy,  dpsi]
    x0_initializer = np.array([x0[0], x0[1], x0[2], x0[4], x0[5]])
    vx_const_0 = 0.1
    kappa_at_stages = np.zeros(horizon_params["N_horizon"])
    u_Fxm_0 = 0.0

    """ =========== GET SOLVER AND INTEGRATOR ============ """
    ocp_init_solver = setup_ocp_init(x0_initializer)
    ocp_solver, integrator = setup_ocp_and_sim(x0, use_RTI, simulate_ocp=simulate_ocp)

    if simulate_ocp:
        """ =========== GET DIMS ============ """
        nx = ocp_solver.acados_ocp.dims.nx
        nu = ocp_solver.acados_ocp.dims.nu
        nx_initializer = ocp_init_solver.acados_ocp.dims.nx
        nu_initializer = ocp_init_solver.acados_ocp.dims.nu

        """ =========== SET SIMULATION PARAMS ============ """
        Nsim = 100
        simX = np.ndarray((Nsim+1, nx))
        simU = np.ndarray((Nsim, nu))
        stagesX_initializer = np.ndarray((horizon_params["N_horizon"], nx_initializer))
        stagesU_initializer = np.ndarray((horizon_params["N_horizon"] - 1, nu_initializer))
        s_values = np.zeros((Nsim+1, 1))

        simX[0,:] = x0
        stagesX_initializer[0,:] = x0_initializer
        
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
                init_u_stage_j = np.array((u_Fxm_0, # Fxm
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
            simU[i,:] = ocp_solver.solve_for_x0(x0_bar = init_state)

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
            u_Fxm_0 = simU[i, 0]

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
        simX[:, 0] = np.squeeze(s_values)
        plot_dynamics(np.linspace(0, (horizon_params['T_final']/horizon_params['N_horizon'])*Nsim, Nsim+1),
                    idx_b_x, lb_x, ub_x,
                    idx_b_u, lb_u, ub_u, 
                    simU, simX,
                    plot_constraints=True)

        # print(ocp_solver)

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
    main(use_RTI=False, simulate_ocp=True)