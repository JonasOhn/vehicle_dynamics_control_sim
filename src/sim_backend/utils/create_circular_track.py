import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from matplotlib import collections  as mc

midline_positions = []
left_blue_cones_positions = []
right_yellow_cones_positions = []

num_points = 100
alpha_min = 0.0
alpha_max = 0.999 * 2 * np.pi
radius = 50.0
trackwidth = 6.0
angles = np.linspace(alpha_min, alpha_max, num_points)

for angle in angles:
    midline_positions.append([radius * np.sin(angle), radius - radius * np.cos(angle)])
    left_blue_cones_positions.append([(radius - trackwidth / 2.0)* np.cos(angle), radius - (radius - trackwidth / 2.0) * np.sin(angle)])
    right_yellow_cones_positions.append([(radius + trackwidth / 2.0)* np.cos(angle), radius - (radius + trackwidth / 2.0) * np.sin(angle)])

left_blue_cones_positions = np.array(left_blue_cones_positions)
right_yellow_cones_positions = np.array(right_yellow_cones_positions)
midline_positions = np.array(midline_positions)

with open('src/sim_backend/tracks/circular_track_blue_cones.csv', 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',')
    for i in range(left_blue_cones_positions.shape[0]):
        csv_writer.writerow([left_blue_cones_positions[i][0], left_blue_cones_positions[i][1]])

with open('src/sim_backend/tracks/circular_track_yellow_cones.csv', 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',')
    for i in range(right_yellow_cones_positions.shape[0]):
        csv_writer.writerow([right_yellow_cones_positions[i][0], right_yellow_cones_positions[i][1]])

with open('src/sim_backend/tracks/circular_track_middle_path.csv', 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',')
    for i in range(midline_positions.shape[0]):
        csv_writer.writerow([midline_positions[i][0], midline_positions[i][1]])