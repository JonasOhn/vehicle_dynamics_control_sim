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
with open("src/sim_backend/tracks/FSG.csv") as csv_file:
    csv_reader = csv.DictReader(csv_file, delimiter=",")
    for row in csv_reader:
        if row["tag"] == "blue":
            left_blue_cones_positions.append([float(row["x"]), float(row["y"])])
        elif row["tag"] == "yellow":
            right_yellow_cones_positions.append([float(row["x"]), float(row["y"])])
        else:
            pass

left_blue_cones_positions = np.array(left_blue_cones_positions)
right_yellow_cones_positions = np.array(right_yellow_cones_positions)

with open("src/sim_backend/tracks/FSG_blue_cones.csv", "w", newline="") as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=",")
    for i in range(left_blue_cones_positions.shape[0]):
        csv_writer.writerow(
            [left_blue_cones_positions[i][0], left_blue_cones_positions[i][1]]
        )

with open("src/sim_backend/tracks/FSG_yellow_cones.csv", "w", newline="") as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=",")
    for i in range(right_yellow_cones_positions.shape[0]):
        csv_writer.writerow(
            [right_yellow_cones_positions[i][0], right_yellow_cones_positions[i][1]]
        )
