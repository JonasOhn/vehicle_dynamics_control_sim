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

import matplotlib.pyplot as plt
import numpy as np

D_tire = -1.0
C_tire = 1.5
B_tire = 10.0

# Parameters used in controllers??
# D_tire = -3.3663
# C_tire = 0.9767
# B_tire = 60.3197

max_abs_slip_angle = np.pi / 4
slip_angles = np.linspace(-max_abs_slip_angle, max_abs_slip_angle, num=1000)
mu_norm = - D_tire * np.sin(C_tire * np.arctan(B_tire * slip_angles))

fig, ax = plt.subplots()

slip_max = np.tan(np.pi/(2*C_tire))/B_tire

ax.plot(slip_angles, mu_norm)
ax.vlines(slip_max, 0, -D_tire, 'red')
print("Max. slip: ", slip_max)
plt.grid("minor")
plt.show()
