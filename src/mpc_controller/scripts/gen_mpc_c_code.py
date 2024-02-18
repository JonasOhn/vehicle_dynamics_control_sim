import numpy as np
from setup_nlp_ocp_and_sim import setup_nlp_ocp_and_sim

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


# x =         [s,   n,   mu,  vx,  vy,  dpsi]
x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
setup_nlp_ocp_and_sim(x0, simulate_ocp=False)
