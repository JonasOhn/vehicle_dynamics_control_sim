from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from acados_template.acados_ocp_solver import ocp_get_default_cmake_builder
from dynamics_model import export_vehicle_ode_model
import numpy as np
from plot_dynamics_sim import plot_dynamics
from os.path import dirname, join, abspath
import yaml
import pprint

def load_mpc_yaml_params():
    yaml_path = join(dirname(abspath(__file__)), "../config/mpc_controller.yaml")
    with open(yaml_path, 'r') as file:
        yaml_params = yaml.safe_load(file)
    yaml_params = yaml_params['/mpc_controller']['ros__parameters']
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
    print("Codegen directory: ", codegen_export_dir)
    ocp_solver_json_path = join(dirname(abspath(__file__)), "acados_ocp.json")
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
    # cost weight input: dFx
    model_cost_parameters['r_dFx'] = cost_params['r_dFx']
    # cost weight input: ddel_s
    model_cost_parameters['r_del_s'] = cost_params['r_del_s']

    # cost weight: beta deviaiton from kinematic
    model_cost_parameters['q_beta'] = cost_params['q_beta']
    # cost weight: progress rate
    model_cost_parameters['q_sd'] = cost_params['q_sd']
    # cost weight: steering angle
    model_cost_parameters['q_del'] = cost_params['q_del']
    # cost weight: deviation from ref path lateral
    model_cost_parameters['q_n'] = cost_params['q_n']

    # terminal cost weight: yaw rate
    model_cost_parameters['q_r_e'] = cost_params['q_r_e']
    # terminal cost weight: heading difference
    model_cost_parameters['q_mu_e'] = cost_params['q_mu_e']
    # terminal cost weight: deviation from ref path lateral
    model_cost_parameters['q_n_e'] = cost_params['q_n_e']

    # nonlinear constraints also defined inside the model
    model_constraint_parameters = {}
    # tire ellipse mult. factor for Fx
    model_constraint_parameters['e_x_front'] = constraints_params['soft']['e_x_front']
    model_constraint_parameters['e_x_rear'] = constraints_params['soft']['e_x_rear']
    # tire ellipse: e_y together with D_tire defines max. e
    model_constraint_parameters['e_y_front'] = constraints_params['soft']['e_y_front']
    model_constraint_parameters['e_y_rear'] = constraints_params['soft']['e_y_rear']
    # track boundary: max. lateral deviation from ref. path
    #TODO: make this parameter for online track boundaries
    model_constraint_parameters['n_max'] = constraints_params['soft']['n_max']
    model_constraint_parameters['n_min'] = constraints_params['soft']['n_min']
    # slacked track boundary constraint: vehicle geometry (padded rectangle)    
    model_constraint_parameters['L_F'] = constraints_params['soft']['L_F']
    model_constraint_parameters['L_R'] = constraints_params['soft']['L_R']
    model_constraint_parameters['W'] = constraints_params['soft']['W']
    # stages: slacked velocity constraint
    model_constraint_parameters['vx_max'] = constraints_params['soft']['vx_max']
    # stages: slacked velocity constraint
    model_constraint_parameters['vx_max_e'] = constraints_params['soft']['vx_max_e']


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

    ocp.dims.N = mpc_horizon_parameters['N_horizon']

    """ ========= INTERNAL STAGE CONSTRAINTS ======== """
    # Terminal Nonlinear Constraints: lower bounds are -Inf but acados doesn't seem to support it
    ocp.constraints.lh = constraints_params['hard']['lb_h'] * np.ones(8)
    # Terminal Nonlinear Constraints: upper bounds
    ocp.constraints.uh = constraints_params['hard']['ub_h'] * np.ones(8)

    # Terminal Nonlinear Constraints: lower bounds are -Inf but acados doesn't seem to support it
    ocp.constraints.lh_0 = constraints_params['hard']['lb_h'] * np.ones(8)
    # Terminal Nonlinear Constraints: upper bounds
    ocp.constraints.uh_0 = constraints_params['hard']['ub_h'] * np.ones(8)

    # ---
    # State: [s, n, mu, vx, vy, dpsi, Fx, del_s]
    # State Constraints: lower bounds
    lb_x = [constraints_params['hard']['lb_x_n'],
            constraints_params['hard']['lb_x_mu'],
            constraints_params['hard']['lb_x_vx'],
            constraints_params['hard']['lb_x_vy'],
            constraints_params['hard']['lb_x_dpsi'],
            constraints_params['hard']['lb_x_Fx'],
            constraints_params['hard']['lb_x_dels']]
    # State Constraints: upper bounds
    ub_x = [constraints_params['hard']['ub_x_n'],
            constraints_params['hard']['ub_x_mu'],
            constraints_params['hard']['ub_x_vx'],
            constraints_params['hard']['ub_x_vy'],
            constraints_params['hard']['ub_x_dpsi'],
            constraints_params['hard']['ub_x_Fx'],
            constraints_params['hard']['ub_x_dels']]
    # State Constraints: indices of lb and ub in State vector
    idx_b_x = [1, 2, 3, 4, 5, 6, 7]

    ocp.constraints.lbx = np.array(lb_x)
    ocp.constraints.ubx = np.array(ub_x)
    ocp.constraints.idxbx = np.array(idx_b_x)

    # ---

    # Input: [dFx, ddel_s]
    # Input Constraints: lower bounds
    lb_u = [constraints_params['hard']['lb_u_dFx'],
            constraints_params['hard']['lb_u_ddels']]
    # Input Constraints: upper bounds
    ub_u = [constraints_params['hard']['ub_u_dFx'],
            constraints_params['hard']['ub_u_ddels']]
    # Input Constraints: indices of lb and ub in input vector
    idx_b_u = [0, 1]

    ocp.constraints.lbu = np.array(lb_u)
    ocp.constraints.ubu = np.array(ub_u)
    ocp.constraints.idxbu = np.array(idx_b_u)

    """ ========= INTERNAL TERMINAL CONSTRAINTS ======== """
    # Terminal Nonlinear Constraints: lower bounds are -Inf but acados doesn't seem to support it
    ocp.constraints.lh_e = constraints_params['hard']['lb_h'] * np.ones(8)
    # Terminal Nonlinear Constraints: upper bounds
    ocp.constraints.uh_e = constraints_params['hard']['ub_h'] * np.ones(8)

    """ ========= INITIAL STATE CONSTRAINT =========== """
    ocp.constraints.x0 = x0

    """ ======== STAGE SLACK COST ========== """
    # (model cost inside ocp.model) --> cost type external
    ocp.cost.cost_type = 'EXTERNAL'

    # Slack Weights
    S_n_lin = cost_params['slack_penalties']['linear']['S_n_lin']
    S_n_quad = cost_params['slack_penalties']['quadratic']['S_n_quad']
    S_vx_lin = cost_params['slack_penalties']['linear']['S_vx_lin']
    S_vx_quad = cost_params['slack_penalties']['quadratic']['S_vx_quad']
    S_te_lin = cost_params['slack_penalties']['linear']['S_te_lin']
    S_te_quad = cost_params['slack_penalties']['quadratic']['S_te_quad']
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
    S_n_lin_e = cost_params['slack_penalties']['linear']['S_n_lin_e']
    S_n_quad_e = cost_params['slack_penalties']['quadratic']['S_n_quad_e']
    S_vx_lin_e = cost_params['slack_penalties']['linear']['S_vx_lin_e']
    S_vx_quad_e = cost_params['slack_penalties']['quadratic']['S_vx_quad_e']
    S_te_lin_e = cost_params['slack_penalties']['linear']['S_te_lin_e']
    S_te_quad_e = cost_params['slack_penalties']['quadratic']['S_te_quad_e']
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
    ocp.solver_options.qp_solver = solver_options_params['qp_solver']
    ocp.solver_options.hessian_approx = solver_options_params['hessian_approx']
    ocp.solver_options.integrator_type = solver_options_params['integrator_type']
    ocp.solver_options.sim_method_newton_iter = solver_options_params['sim_method_newton_iter']
    ocp.solver_options.nlp_solver_max_iter = solver_options_params['nlp_solver_max_iter']
    ocp.solver_options.qp_solver_iter_max = solver_options_params['qp_solver_iter_max']
    ocp.solver_options.tol = solver_options_params['tol']
    ocp.solver_options.qp_tol = solver_options_params['qp_tol']

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
    blending_factor = model_params['blending_factor'] # blending between kinematic and dynamic model
    kappa_ref = model_params['kappa_ref'] * np.ones(horizon_params['n_s']) # reference curvature along s

    paramvec = np.array((m, g, l_f, l_r, Iz, 
                         B_tire, C_tire, D_tire, C_d, C_r, blending_factor))
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
        s_values = np.ndarray((Nsim+1, 1))
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
                # print(ocp_solver.get(0, 'x'))
                # print(ocp_solver.get(horizon_params['N_horizon'], 'x'))
                # print(ocp_solver.get_cost(), (horizon_params['T_final']/horizon_params['N_horizon'])*i)

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

            # States
            vx = simX[i+1, 3]
            
            if vx > 0.5:
                blending_factor = 1.0
            else:
                blending_factor = 0.0
            simBlendingFactor[i] = blending_factor

            param_vec[10] = blending_factor

            for j in range(horizon_params['N_horizon']):
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
        simX[:, 0] = np.squeeze(s_values)
        plot_dynamics(np.linspace(0, (horizon_params['T_final']/horizon_params['N_horizon'])*Nsim, Nsim+1),
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
    main(use_RTI=True, simulate_ocp=False)