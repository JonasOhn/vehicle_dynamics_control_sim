import numpy as np
from setup_nlp_ocp_and_sim import setup_nlp_ocp_and_sim
from setup_qp_init import setup_ocp_init


# x =         [s,   n,   mu,  vx,  vy,  dpsi]
x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
setup_nlp_ocp_and_sim(x0, simulate_ocp=False)

# x =         [s,   n,   mu,  vy,  dpsi]
x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
setup_ocp_init(x0)