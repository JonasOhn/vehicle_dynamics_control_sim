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
import matplotlib.patches
import numpy as np

L_F = 1.8
L_R = 2.5
W = 2.1
n = -4
n_max = 10.0
n_min = -10.0
mu_rad = -0.5
mu_deg = np.rad2deg(mu_rad)

bounds_front = L_F * np.sin(mu_rad)
bounds_rear = L_R * np.sin(mu_rad)
bounds_width = W / 2.0 * np.cos(mu_rad)

track_con_fl = n + bounds_front + bounds_width
track_con_rl = n - bounds_rear + bounds_width
track_con_fr = n + bounds_front - bounds_width
track_con_rr = n - bounds_rear - bounds_width

constraints_n = (track_con_fl, track_con_fr, track_con_rl, track_con_rr)

x_anchor = -L_R * np.cos(mu_rad) + W / 2 * np.sin(mu_rad)
print(x_anchor)
y_anchor = n - L_R * np.sin(mu_rad) - W / 2 * np.cos(mu_rad)
pat = matplotlib.patches.Rectangle(
    xy=(x_anchor, y_anchor),
    width=L_R + L_F,
    height=W,
    angle=mu_deg,
    rotation_point="xy",
    edgecolor="k",
)

xmin_hlines = -10.0
xmax_hlines = 10.0
plt.figure()
plt.hlines(np.array([n_max]), xmin_hlines, xmax_hlines, "r", linestyles="dashed")
plt.text(xmin_hlines, n_max, "n_min for XL", ha="right", va="center")
plt.hlines(np.array([n_min]), xmin_hlines, xmax_hlines, "r", linestyles="dashed")
plt.text(xmin_hlines, n_min, "n_min for XR", ha="right", va="center")

plt.hlines(np.array([0]), xmin_hlines, xmax_hlines, "k", linestyles="dashed")
plt.hlines(np.array(constraints_n), xmin_hlines, xmax_hlines, "g", linestyles="dotted")
plt.text(xmin_hlines, constraints_n[0], "FL", ha="right", va="center")
plt.text(xmin_hlines, constraints_n[1], "FR", ha="right", va="center")
plt.text(xmin_hlines, constraints_n[2], "RL", ha="right", va="center")
plt.text(xmin_hlines, constraints_n[3], "RR", ha="right", va="center")
ax = plt.gca()
ax.add_patch(pat)
ax.axis("equal")

plt.scatter(0, n, s=10, c="k")
plt.grid("both")
plt.show()
