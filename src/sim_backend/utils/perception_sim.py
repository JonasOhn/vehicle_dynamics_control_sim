import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from matplotlib import collections  as mc
import matplotlib.patches as mpatches

heading_angle = np.deg2rad(390.0)

gamma = np.deg2rad(30)

x = 0.0
y = 0.0

dx = np.cos(heading_angle)
dy = np.sin(heading_angle)

x_head = x + dx
y_head = y + dy

x_vals = np.linspace(-5, 5, 50)

a1_neg = np.sin(heading_angle - gamma)
a1_pos = np.sin(heading_angle + gamma)
a2_neg = - np.cos(heading_angle - gamma)
a2_pos = - np.cos(heading_angle + gamma)

b_neg = - a1_neg * x - a2_neg * y
b_pos = a1_pos * x + a2_pos * y

y_vals_1 = (b_neg - a1_neg * x_vals)/a2_neg
y_vals_2 = (b_pos - a1_pos * x_vals)/a2_pos

# a_1_neg * path_pos.point_2d[0] + a_2_neg * path_pos.point_2d[1] <= b_neg
# a_1_pos * path_pos.point_2d[0] + a_2_pos * path_pos.point_2d[1] >= b_pos

# double a_1_neg = sin(psi - gamma_)
# double a_2_neg = - cos(psi - gamma_)
# double b_neg = - a_1_neg * x_c - a_2_neg * y_c

# double a_1_pos = sin(psi + gamma_)
# double a_2_pos = - cos(psi + gamma_)
# double b_pos = a_1_pos * x_c + a_2_pos * y_c

plt.arrow(x, y, dx, dy, head_width=0.05, head_length=0.1, fc='black', ec='black')
plt.plot(x_vals, y_vals_1, 'r')
plt.plot(x_vals, y_vals_2, 'm')

ax = plt.gca()

plot_limit = 2
    
ax.fill([x_vals[-1], x_vals[0], x_vals[0] - a1_neg, x_vals[-1] - a1_neg], [y_vals_1[-1], y_vals_1[0], y_vals_1[0] - a2_neg, y_vals_1[-1] - a2_neg], 'r', alpha=0.3)
ax.fill([x_vals[-1], x_vals[0], x_vals[0] + a1_pos, x_vals[-1] + a1_pos], [y_vals_2[-1], y_vals_2[0], y_vals_2[0] + a2_pos, y_vals_2[-1] + a2_pos], 'm', alpha=0.3)

ax.set_aspect('equal', 'box')
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.show()