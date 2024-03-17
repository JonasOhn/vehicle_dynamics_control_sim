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

x1 = 0.0
x2 = 10.0

y1 = 0.0
y2 = 10.0

# define the length of the orthogonal line segment
length = 2.0

# calculate the direction vector of the original line segment
dx = x2 - x1
dy = y2 - y1

# calculate the normalized direction vector
norm = np.sqrt(dx**2 + dy**2)
dx_norm = dx / norm
dy_norm = dy / norm

# calculate the coordinates of the orthogonal line segment
x3 = x2 + length * dy_norm
y3 = y2 - length * dx_norm
x4 = x2 - length * dy_norm
y4 = y2 + length * dx_norm

plt.plot([x1, x2], [y1, y2], "ro-")
plt.plot([x3, x4], [y3, y4], "bo-")

plt.axis("equal")

plt.show()
