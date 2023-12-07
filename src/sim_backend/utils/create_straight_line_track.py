import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from matplotlib import collections  as mc

midline_positions = []
left_blue_cones_positions = []
right_yellow_cones_positions = []

ds = 1.0
s_min = -10.0
s_max = 1000.0
trackwidth = 6.0

for s in np.arange(s_min, s_max, ds):
    midline_positions.append([s, 0.0])
    left_blue_cones_positions.append([s, trackwidth / 2.0])
    right_yellow_cones_positions.append([s, - trackwidth / 2.0])

left_blue_cones_positions = np.array(left_blue_cones_positions)
right_yellow_cones_positions = np.array(right_yellow_cones_positions)
midline_positions = np.array(midline_positions)

with open('src/sim_backend/tracks/straight_track_blue_cones.csv', 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',')
    for i in range(left_blue_cones_positions.shape[0]):
        csv_writer.writerow([left_blue_cones_positions[i][0], left_blue_cones_positions[i][1]])

with open('src/sim_backend/tracks/straight_track_yellow_cones.csv', 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',')
    for i in range(right_yellow_cones_positions.shape[0]):
        csv_writer.writerow([right_yellow_cones_positions[i][0], right_yellow_cones_positions[i][1]])

with open('src/sim_backend/tracks/straight_track_middle_path.csv', 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',')
    for i in range(midline_positions.shape[0]):
        csv_writer.writerow([midline_positions[i][0], midline_positions[i][1]])