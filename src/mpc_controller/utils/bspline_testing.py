import matplotlib.pyplot as plt
import numpy as np
from matplotlib import collections  as mc

# control_points = np.array([
#     [-5, -5],
#     [-4, -3],
#     [-2, 0],
#     [0, 1],
#     [2, 4],
#     [2, 5],
#     [3, 8],
#     [4, 10],
#     [5, 8],
#     [5, 6],
# ])

control_points = []
for angle in np.linspace(0, np.pi, 20):
    control_points.append([100 * np.cos(angle), 100 * np.sin(angle)])
control_points = np.array(control_points)

control_points = np.insert(control_points, 0, control_points[0], axis=0)
control_points = np.insert(control_points, -1, control_points[-1], axis=0)

# control_points = np.insert(control_points, 0, control_points[-1], axis=0)
# control_points = np.insert(control_points, -1, control_points[0], axis=0)

# control_points = np.insert(control_points, 0, 2 * control_points[0] - control_points[1], axis=0)
# control_points = np.insert(control_points, -1, 2 * control_points[-1] - control_points[-2], axis=0)

print(control_points)

BSpline_mat = 1/6 * np.array([
    [-1, 3, -3, 1],
    [3, -6, 0, 4],
    [-3, 3, 3, 1],
    [1, 0, 0, 0]
])

print('num_points:', control_points.shape[0])
n_s = control_points.shape[0] - 3
print('n_s:', n_s)

t_step = 0.5
t = np.arange(0, n_s, t_step)
spline_points = []
dspline_points = []
ddspline_points = []

for idx_t in range(len(t)):
    j = int(t[idx_t])
    t_frac = t[idx_t] - j
    print('j: ', j, ', t_frac: ', t_frac)
    tvec = np.array([[t_frac**3], [t_frac**2], [t_frac], [1.0]])
    dtvec = np.array([[3 * t_frac**2], [2 * t_frac], [1.0], [0.0]])
    ddtvec = np.array([[6 * t_frac], [2.0], [0.0], [0.0]])
    spline_points.append(np.squeeze(control_points[j:j+4].T @ BSpline_mat @ tvec))
    dspline_points.append(np.squeeze(control_points[j:j+4].T @ BSpline_mat @ dtvec))
    ddspline_points.append(np.squeeze(control_points[j:j+4].T @ BSpline_mat @ ddtvec))

spline_points = np.array(spline_points)
dspline_points = np.array(dspline_points)
ddspline_points = np.array(ddspline_points)
print('spline_points: ', spline_points)

s_distances = []
for idx_t in range(len(t)):
    if idx_t > 0:
        s_distances.append(np.sqrt((spline_points[idx_t, 0] - spline_points[idx_t-1, 0])**2 + 
                                    (spline_points[idx_t, 1] - spline_points[idx_t-1, 1])**2))

curvatures = []
for i in range(spline_points.shape[0]):
    x_d = dspline_points[i, 0]
    y_d = dspline_points[i, 1]
    print('x_d, y_d, abs(r_d): ', x_d, y_d, np.sqrt(x_d**2 + y_d**2))
    x_dd = ddspline_points[i, 0]
    y_dd = ddspline_points[i, 1]
    print('x_dd, y_dd: ', x_dd, y_dd)
    curv = (x_d * y_dd - y_d * x_dd)
    curv = curv / ((x_d**2 + y_d**2)**(3/2))
    print('curv: ', curv)
    curvatures.append(curv)


curvatures = np.array(curvatures)
print('curvatures: ', curvatures)
abs_curvatures = np.abs(curvatures)
print('abs_curvatures: ', abs_curvatures)

fd_splines = []
for i in range(spline_points.shape[0]):
    fd_splines.append([(spline_points[i, 0], spline_points[i, 1]), 
                       (spline_points[i, 0] + dspline_points[i, 0], 
                        spline_points[i, 1] + dspline_points[i, 1])])

r_max = 200
radii = []
for i in range(curvatures.shape[0]):
    tang_norm = np.sqrt(dspline_points[i, 0]**2 + dspline_points[i, 1]**2)
    r = 1 / curvatures[i] if curvatures[i] else r_max
    dc_x = np.maximum(np.minimum(r, r_max), -r_max) * (- dspline_points[i, 1]) / tang_norm
    dc_y = np.maximum(np.minimum(r, r_max), -r_max) * (dspline_points[i, 0]) / tang_norm
    radii.append([(spline_points[i, 0], spline_points[i, 1]), 
                  (spline_points[i, 0] + dc_x, 
                  spline_points[i, 1] + dc_y)])

lc = mc.LineCollection(fd_splines, colors='k', linewidths=1, label='tangent vectors')
lc_radii = mc.LineCollection(radii, colors='g', linewidths=1, label='radii')

plt.figure(1)
plt.scatter(control_points[:, 0], control_points[:, 1], label='control points')
plt.scatter(control_points[1, 0], control_points[1, 1], c='m', s=100, marker='o', label='first given control point')
plt.plot(spline_points[:, 0], spline_points[:, 1], 'r+', label='spline points')
plt.scatter(spline_points[:, 0], spline_points[:, 1], s=100*abs_curvatures, label='curvature')
ax = plt.gca()
ax.add_collection(lc)
ax.add_collection(lc_radii)
ax.legend()
ax.set_xlabel("x - position")
ax.set_ylabel("y - position")
ax.axis('equal')
plt.grid()

plt.figure(2)
plt.subplot(2,1,1)
plt.bar(np.array(range(len(s_distances))) + 1, np.array(s_distances), label="ds")
ax = plt.gca()
ax.legend()
ax.set_xlabel("idx")
ax.set_ylabel("ds")

plt.subplot(2,1,2)
plt.bar(np.array(range(len(curvatures))), np.array(curvatures), label="curv.")
ax = plt.gca()
ax.legend()
ax.set_xlabel("idx")
ax.set_ylabel("curvature")

plt.tight_layout()
plt.show()