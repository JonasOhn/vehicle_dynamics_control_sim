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
from setup_kinematic_bicycle_curv import setup_nlp_ocp_and_sim

"""
==== Solver Status Messages ====

ACADOS
    SQP Solver Status:
        0 --> ACADOS_SUCCESS,
        1 --> ACADOS_FAILURE,
        2 --> ACADOS_MAXITER,
        3 --> ACADOS_MINSTEP,
        4 --> ACADOS_QP_FAILURE,
        5 --> ACADOS_READY

        QP-Solver HPIPM
    HPIPM Solver Status:
        0 --> SUCCESS, // found solution satisfying accuracy tolerance
        1 --> MAX_ITER, // maximum iteration number reached
        2 --> MIN_STEP, // minimum step length reached
        3 --> NAN_SOL, // NaN in solution detected
        4 --> INCONS_EQ, // unconsistent equality constraints
"""

# x =         [s,   n,   mu,  vx, ax, dels]
x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
setup_nlp_ocp_and_sim(x0, simulate_ocp=False)
