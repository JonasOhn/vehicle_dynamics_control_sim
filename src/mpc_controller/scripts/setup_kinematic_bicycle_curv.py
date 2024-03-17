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

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from acados_template.acados_ocp_solver import ocp_get_default_cmake_builder
from model_kinematic_bicycle_curv import export_vehicle_ode_model
import numpy as np
from os.path import dirname, join, abspath
import yaml
import pprint
import shutil
import os
import scipy.linalg


def load_mpc_yaml_params():
    yaml_path = join(dirname(abspath(__file__)), "../config/mpc_controller.yaml")
    with open(yaml_path, "r") as file:
        yaml_params = yaml.safe_load(file)
    yaml_params = yaml_params["/mpc_controller_node"]["ros__parameters"]
    model_params = yaml_params["model"]
    horizon_params = yaml_params["horizon"]
    cost_params = yaml_params["cost"]
    constraints_params = yaml_params["constraints"]
    solver_options_params = yaml_params["solver_options"]
    return (
        model_params,
        horizon_params,
        cost_params,
        constraints_params,
        solver_options_params,
    )


def setup_nlp_ocp_and_sim(x0, simulate_ocp: bool = False):
    """Load .yaml file parameters"""
    (
        model_params,
        horizon_params,
        cost_params,
        constraints_params,
        solver_options_params,
    ) = load_mpc_yaml_params()
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
    ocp.acados_include_path = f"{ACADOS_PATH}/include"
    ocp.acados_lib_path = f"{ACADOS_PATH}/lib"

    """ ========== MODEL ============ """
    mpc_horizon_parameters = {}
    # final time for prediction horizon
    mpc_horizon_parameters["T_final"] = horizon_params["T_final"]
    # number of steps along horizon
    mpc_horizon_parameters["N_horizon"] = horizon_params["N_horizon"]

    # external cost is defined inside the model
    model_cost_parameters = {}
    # cost weight: progress rate
    model_cost_parameters["q_sd"] = cost_params["q_sd"]
    model_cost_parameters["q_n"] = cost_params["q_n"]
    model_cost_parameters["q_mu"] = cost_params["q_mu"]
    model_cost_parameters["q_ax"] = cost_params["q_ax"]
    model_cost_parameters["q_dels"] = cost_params["q_dels"]
    model_cost_parameters["r_dax"] = cost_params["r_dax"]
    model_cost_parameters["r_ddels"] = cost_params["r_ddels"]

    # Get AcadosModel form other python file
    model = export_vehicle_ode_model()
    ocp.model = model

    """ ========== DIMENSIONS ============ """
    # Dimensions
    # x: state
    # u: input
    # N: prediction horizon number of stages
    ocp.dims.N = mpc_horizon_parameters["N_horizon"]

    """ ========= CONSTRAINT: INITIAL STATE =========== """
    ocp.constraints.x0 = x0

    """ ========= CONSTRAINTS: STAGE STATE ======== """
    # ---
    # State: [s, n, mu, vx, ax, dels]
    # State Constraints: lower bounds
    ocp.constraints.lbx = np.array(
        (
            constraints_params["hard"]["lb_s"],
            constraints_params["soft"]["lb_n"],
            constraints_params["soft"]["lb_mu"],
            constraints_params["hard"]["lb_vx"],
            constraints_params["hard"]["lb_ax"],
            constraints_params["hard"]["lb_dels"],
        )
    )
    # State Constraints: upper bounds
    ocp.constraints.ubx = np.array(
        (
            constraints_params["hard"]["ub_s"],
            constraints_params["soft"]["ub_n"],
            constraints_params["soft"]["ub_mu"],
            constraints_params["hard"]["ub_vx"],
            constraints_params["hard"]["ub_ax"],
            constraints_params["hard"]["ub_dels"],
        )
    )
    # State Constraints: indices of lb and ub in State vector
    ocp.constraints.idxbx = np.array((0, 1, 2, 3, 4, 5))

    """ ========= CONSTRAINTS: STAGE INPUT ======== """
    # ---
    # Input: [dax_m, ddel_s]
    # Input Constraints: lower bounds
    ocp.constraints.lbu = np.array(
        (constraints_params["hard"]["lb_dax"], constraints_params["hard"]["lb_ddels"])
    )
    # Input Constraints: upper bounds
    ocp.constraints.ubu = np.array(
        (constraints_params["hard"]["ub_dax"], constraints_params["hard"]["ub_ddels"])
    )
    # Input Constraints: indices of lb and ub in input vector
    ocp.constraints.idxbu = np.array((0, 1))

    """ ========= CONSTRAINTS: TERMINAL STATE ======== """
    # ---
    # Terminal State: [s, n, mu, vx, ax, dels]
    # Terminal State Constraints: lower bounds
    ocp.constraints.lbx_e = np.array(
        (
            constraints_params["hard"]["lb_s"],
            constraints_params["soft"]["lb_n"],
            constraints_params["soft"]["lb_mu"],
            constraints_params["hard"]["lb_vx"],
            constraints_params["hard"]["lb_ax"],
            constraints_params["hard"]["lb_dels"],
        )
    )
    # Terminal State Constraints: upper bounds
    ocp.constraints.ubx_e = np.array(
        (
            constraints_params["hard"]["ub_s"],
            constraints_params["soft"]["ub_n"],
            constraints_params["soft"]["ub_mu"],
            constraints_params["hard"]["ub_vx"],
            constraints_params["hard"]["ub_ax"],
            constraints_params["hard"]["ub_dels"],
        )
    )
    # Terminal State Constraints: indices of lb and ub in State vector
    ocp.constraints.idxbx_e = np.array((0, 1, 2, 3, 4, 5))

    """ ========= COST =========== """
    # (model cost inside ocp.model) --> cost type external
    ocp.cost.cost_type = "EXTERNAL"

    """ ========== STAGE SLACK COST =========== """
    # Slack Weights
    S_n_lin = cost_params["slack_penalties"]["linear"]["S_n_lin"]
    S_n_quad = cost_params["slack_penalties"]["quadratic"]["S_n_quad"]
    S_mu_lin = cost_params["slack_penalties"]["linear"]["S_mu_lin"]
    S_mu_quad = cost_params["slack_penalties"]["quadratic"]["S_mu_quad"]

    # Quadratic Slack Cost weights
    ocp.cost.Zu = np.diag(
        np.array(
            (
                S_n_quad,
                S_mu_quad,
            )
        )
    )
    ocp.cost.Zl = np.diag(
        np.array(
            (
                S_n_quad,
                S_mu_quad,
            )
        )
    )
    # Linear Slack Cost Weights
    ocp.cost.zu = np.array(
        (
            S_n_lin,
            S_mu_lin,
        )
    )
    ocp.cost.zl = np.array(
        (
            S_n_lin,
            S_mu_lin,
        )
    )

    # Indices of slack variables in stage linear constraints
    ocp.constraints.idxsbx = np.array(
        (
            1,
            2,
        )
    )

    """ ======== TERMINAL SLACK COST ========== """
    # Slack Weights
    S_n_lin_e = cost_params["slack_penalties"]["linear"]["S_n_lin_e"]
    S_n_quad_e = cost_params["slack_penalties"]["quadratic"]["S_n_quad_e"]
    S_mu_lin_e = cost_params["slack_penalties"]["linear"]["S_mu_lin_e"]
    S_mu_quad_e = cost_params["slack_penalties"]["quadratic"]["S_mu_quad_e"]

    # Quadratic Slack Cost weights
    ocp.cost.Zu_e = np.diag(
        np.array(
            (
                S_n_quad_e,
                S_mu_quad_e,
            )
        )
    )
    ocp.cost.Zl_e = np.diag(
        np.array(
            (
                S_n_quad_e,
                S_mu_quad_e,
            )
        )
    )
    # Linear Slack Cost Weights
    ocp.cost.zu_e = np.array(
        (
            S_n_lin_e,
            S_mu_lin_e,
        )
    )
    ocp.cost.zl_e = np.array(
        (
            S_n_lin_e,
            S_mu_lin_e,
        )
    )

    # Indices of slack variables in stage linear constraints
    ocp.constraints.idxsbx_e = np.array(
        (
            1,
            2,
        )
    )

    """ ============ SOLVER OPTIONS ================== """
    ocp.solver_options.qp_solver = solver_options_params["qp_solver"]
    ocp.solver_options.hessian_approx = solver_options_params["hessian_approx"]
    ocp.solver_options.integrator_type = solver_options_params["integrator_type"]
    ocp.solver_options.sim_method_newton_iter = solver_options_params[
        "sim_method_newton_iter"
    ]
    ocp.solver_options.nlp_solver_max_iter = solver_options_params[
        "nlp_solver_max_iter"
    ]
    ocp.solver_options.qp_solver_iter_max = solver_options_params["qp_solver_iter_max"]
    ocp.solver_options.tol = solver_options_params["tol"]
    ocp.solver_options.qp_tol = solver_options_params["qp_tol"]
    ocp.solver_options.alpha_min = solver_options_params["alpha_min"]
    ocp.solver_options.qp_solver_warm_start = solver_options_params[
        "qp_solver_warm_start"
    ]
    ocp.solver_options.regularize_method = solver_options_params["regularize_method"]
    ocp.solver_options.reg_epsilon = solver_options_params["reg_epsilon"]
    ocp.solver_options.alpha_reduction = solver_options_params["alpha_reduction"]
    ocp.solver_options.hpipm_mode = solver_options_params["hpipm_mode"]
    ocp.solver_options.cost_discretization = solver_options_params[
        "cost_discretization"
    ]
    ocp.solver_options.globalization = solver_options_params["globalization"]
    ocp.solver_options.levenberg_marquardt = solver_options_params[
        "levenberg_marquardt"
    ]
    ocp.solver_options.nlp_solver_step_length = solver_options_params[
        "nlp_solver_step_length"
    ]
    ocp.solver_options.nlp_solver_type = "SQP"

    # Set prediction horizon
    ocp.solver_options.tf = horizon_params["T_final"]

    """ ============ INITIAL PARAMETER VALUES ================== """
    m = model_params["m"]  # mass
    l_f = model_params["l_f"]
    l_r = model_params["l_r"]
    C_d = model_params["C_d"]  # effective drag coefficient
    C_r = model_params["C_r"]  # const. rolling resistance
    kappa_ref = model_params["kappa_ref"]  # reference curvature
    q_n = cost_params["q_n"]
    q_sd = cost_params["q_sd"]
    q_mu = cost_params["q_mu"]
    q_ax = cost_params["q_ax"]
    q_dels = cost_params["q_dels"]
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
    ocp.parameter_values = paramvec

    """ ====== CREATE OCP AND SIM SOLVERS =========== """
    if not simulate_ocp:
        cmake_builder = ocp_get_default_cmake_builder()
    else:
        cmake_builder = None
    acados_ocp_solver = AcadosOcpSolver(
        ocp, json_file=ocp_solver_json_path, cmake_builder=cmake_builder
    )

    # create an integrator with the same settings as used in the OCP solver.
    if simulate_ocp:
        acados_integrator = AcadosSimSolver(
            ocp, json_file=ocp_solver_json_path, cmake_builder=cmake_builder
        )
    else:
        acados_integrator = None

    return acados_ocp_solver, acados_integrator


if __name__ == "__main__":
    # x =         [s,   n,   mu,  vx, ax, dels]
    x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    setup_nlp_ocp_and_sim(x0, simulate_ocp=True)
