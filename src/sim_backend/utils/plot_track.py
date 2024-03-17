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

import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from matplotlib import collections as mc


left_blue_cones_positions = []
right_yellow_cones_positions = []
orange_cones_positions = []
with open("src/sim_backend/tracks/FSG.csv") as csv_file:
    csv_reader = csv.DictReader(csv_file, delimiter=",")
    for row in csv_reader:
        if row["tag"] == "blue":
            left_blue_cones_positions.append([float(row["x"]), float(row["y"])])
        elif row["tag"] == "yellow":
            right_yellow_cones_positions.append([float(row["x"]), float(row["y"])])
        else:
            orange_cones_positions.append([float(row["x"]), float(row["y"])])

left_blue_cones_positions = np.array(left_blue_cones_positions)
right_yellow_cones_positions = np.array(right_yellow_cones_positions)
orange_cones_positions = np.array(orange_cones_positions)

plt.scatter(
    left_blue_cones_positions[:, 0],
    left_blue_cones_positions[:, 1],
    s=3.5,
    c="b",
    marker="^",
)
plt.scatter(
    left_blue_cones_positions[0, 0],
    left_blue_cones_positions[0, 1],
    s=10,
    c="r",
    marker="^",
)
plt.scatter(
    right_yellow_cones_positions[:, 0],
    right_yellow_cones_positions[:, 1],
    s=3.5,
    c="y",
    marker="^",
)
plt.scatter(
    right_yellow_cones_positions[0, 0],
    right_yellow_cones_positions[0, 1],
    s=10,
    c="r",
    marker="^",
)
plt.scatter(
    orange_cones_positions[:, 0],
    orange_cones_positions[:, 1],
    s=5,
    c="orange",
    marker="^",
)
plt.gca().axis("equal")

n_spline_evals = 200

""" BLUE """
x_blue = left_blue_cones_positions[:, 0]
y_blue = left_blue_cones_positions[:, 1]

tck_blue, u_blue = interpolate.splprep([x_blue, y_blue], s=0, k=3)
xs_blue, ys_blue = interpolate.splev(np.linspace(0, 1, n_spline_evals), tck_blue)

""" YELLOW """
x_yellow = right_yellow_cones_positions[:, 0]
y_yellow = right_yellow_cones_positions[:, 1]

tck_yellow, u_yellow = interpolate.splprep([x_yellow, y_yellow], s=0, k=3)
xs_yellow, ys_yellow = interpolate.splev(np.linspace(0, 1, n_spline_evals), tck_yellow)


count = len(xs_yellow)
x_midline = np.empty((count,))
y_midline = np.empty((count,))
lines = []

blue_points = np.transpose(np.vstack((xs_blue, ys_blue)))

for i in range(count):
    current_point = np.array([xs_yellow[i], ys_yellow[i]])
    distances = np.linalg.norm(blue_points - current_point, axis=1)
    idx_min = np.argmin(distances)
    x_midline[i] = (xs_blue[idx_min] + xs_yellow[i]) / 2.0
    y_midline[i] = (ys_blue[idx_min] + ys_yellow[i]) / 2.0
    lines.append([(xs_blue[idx_min], ys_blue[idx_min]), (xs_yellow[i], ys_yellow[i])])

tck_mid, u_mid = interpolate.splprep([x_midline, y_midline], s=0, k=3)
xs_midline, ys_midline = interpolate.splev(np.linspace(0, 1, n_spline_evals), tck_mid)

lc = mc.LineCollection(lines, colors="k", linewidths=0.5)
plt.gca().add_collection(lc)

plt.scatter(xs_blue, ys_blue, s=6, c="b", marker="o")
plt.scatter(xs_yellow, ys_yellow, s=6, c="y", marker="o")
plt.scatter(x_midline, y_midline, s=6, c="k", marker="o")
plt.plot(xs_midline, ys_midline, linewidth=0.5, marker="o", markersize=1)
plt.scatter(x_midline[0], y_midline[0], s=10, c="r", marker="o")
plt.scatter(x_midline[-1], y_midline[-1], s=10, c="r", marker="o")
plt.grid()
plt.show()


with open("src/sim_backend/tracks/FSG_middle_path.csv", "w", newline="") as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=",")
    for i in range(len(xs_midline)):
        csv_writer.writerow([xs_midline[i], ys_midline[i]])
