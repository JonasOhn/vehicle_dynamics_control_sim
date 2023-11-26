import matplotlib.pyplot as plt
import numpy as np

ay_kin2dyn = 3.0
slope_kin2dyn_switch = 100.0

vy_dot_expl_dyn = np.linspace(-10.0, 10.0, 1000)

blending_factor = np.minimum(np.maximum(slope_kin2dyn_switch * np.abs(vy_dot_expl_dyn) - slope_kin2dyn_switch * ay_kin2dyn, 0.0), 1.0)

plt.plot(vy_dot_expl_dyn, blending_factor)
plt.show()