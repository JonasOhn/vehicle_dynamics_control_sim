import matplotlib.pyplot as plt
import numpy as np

D_tire = -1.0
C_tire = 1.2
B_tire = 9.0

# Parameters used in controllers??
# D_tire = -3.3663
# C_tire = 0.9767
# B_tire = 60.3197

max_abs_slip_angle = np.pi/4
slip_angles = np.linspace(-max_abs_slip_angle, max_abs_slip_angle, num=1000)
mu_norm = - D_tire * np.sin(C_tire * np.arctan(B_tire * slip_angles))

fig, ax = plt.subplots()

ax.plot(slip_angles, mu_norm)
plt.grid('minor')
plt.show()