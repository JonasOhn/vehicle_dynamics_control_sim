import matplotlib.pyplot as plt
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from acados_template.acados_ocp_solver import ocp_get_default_cmake_builder
from dynamics_model_initializer import export_vehicle_ode_init_model
import numpy as np
from os.path import dirname, join, abspath
import yaml
import pprint
import shutil
import os
import scipy.linalg


def load_mpc_initializer_yaml_params():
    yaml_path = join(dirname(abspath(__file__)), "../config/mpc_initializer.yaml")
    with open(yaml_path, 'r') as file:
        yaml_params = yaml.safe_load(file)
    model_params = yaml_params['model']
    horizon_params = yaml_params['horizon']
    cost_params = yaml_params['cost']
    solver_options_params = yaml_params['solver_options']
    return model_params, horizon_params, cost_params, solver_options_params

def setup_ocp_init(x0, simulate_ocp:bool = False):
    
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

if __name__ == '__main__':
    # x =         [s,   n,   mu,  vy,  dpsi]
    x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    setup_ocp_init(x0)