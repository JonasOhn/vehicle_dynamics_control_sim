import rclpy
from rclpy.node import Node
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from sim_backend.msg import VehicleState
import numpy as np
import csv

class PosePlotter(Node):

    def __init__(self):
        super().__init__('plotting_node')
        self.state_subscription = self.create_subscription(
            VehicleState,
            'vehicle_state',
            self.state_callback,
            10)
        self.state_subscription  # prevent unused variable warning

        self.load_track()

        self.timer = self.create_timer(0.01, self.plot_callback)
        
        plt.ion()

        self.fig = plt.figure()
        self.x_pos = 0.0
        self.y_pos = 0.0
        self.psi = 0.0
        self.dx_pos = 0.0
        self.dy_pos = 0.0
        self.vx_scaled = 0.0
        self.vy_scaled = 0.0

        self.callback_count = 0
        self.past_positions = np.zeros((1, 2))

    def plot_callback(self):
        # Clear current axis
        plt.cla()
        # Get axis from fig
        ax = self.fig.axes[0]

        dx = np.cos(self.psi)
        dy = np.sin(self.psi)
        if self.callback_count < 500:
            self.past_positions = np.append(self.past_positions, np.array([[self.x_pos, self.y_pos]]), axis=0)
        else:
            self.past_positions[0:-1, :] = self.past_positions[1:, :]
            self.past_positions[-1, 0] = self.x_pos
            self.past_positions[-1, 1] = self.y_pos

        plt.plot(self.past_positions[:, 0], self.past_positions[:, 1])
        plt.plot(self.x_pos - dx, self.y_pos - dy, self.x_pos + dx, self.y_pos + dy, marker = 'o')
        plt.plot(self.x_pos, self.y_pos, 'r*')
        plt.arrow(self.x_pos, self.y_pos, self.vx_scaled, self.vy_scaled, width = 0.1, head_width=0.3)

        plt.plot(self.midline_positions[:, 0], self.midline_positions[:, 1])
        plt.scatter(self.blue_cones_positions[:, 0], self.blue_cones_positions[:, 1], s=3.5, c='b', marker="^")
        plt.scatter(self.yellow_cones_positions[:, 0], self.yellow_cones_positions[:, 1], s=3.5, c='y', marker="^")
        plt.scatter(self.orange_cones_positions[:, 0], self.orange_cones_positions[:, 1], s=5, c='orange', marker="^")

        ax.set_xlim(left=self.x_pos-30, right=self.x_pos+30)
        ax.set_ylim(bottom=self.y_pos-30, top=self.y_pos+30)    
        
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def load_track(self):
        left_blue_cones_positions = []
        right_yellow_cones_positions = []
        orange_cones_positions = []
        with open('src/sim_backend/tracks/FSG.csv') as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=',')
            for row in csv_reader:
                if row['tag'] == 'blue':
                    left_blue_cones_positions.append([float(row['x']), float(row['y'])])
                elif row['tag'] == 'yellow':
                    right_yellow_cones_positions.append([float(row['x']), float(row['y'])])
                else:
                    orange_cones_positions.append([float(row['x']), float(row['y'])])
        self.blue_cones_positions = np.array(left_blue_cones_positions)
        self.yellow_cones_positions = np.array(right_yellow_cones_positions)
        self.orange_cones_positions = np.array(orange_cones_positions)

        midline_positions = []
        with open('src/sim_backend/tracks/FSG_middle_path.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                midline_positions.append([float(row[0]), float(row[1])])
        self.midline_positions = np.array(midline_positions)
        

    def state_callback(self, msg):
        self.vx_scaled = msg.dx_c * 0.1
        self.vy_scaled = msg.dy_c * 0.1
        self.x_pos = msg.x_c
        self.y_pos = msg.y_c
        self.psi = msg.psi

def euler_from_quat(quaternion):
    """
    Converts quaternion (w in last place) to euler roll, pitch, yaw
    quaternion = [x, y, z, w]
    Bellow should be replaced when porting for ROS 2 Python tf_conversions is done.
    """
    x = quaternion.x
    y = quaternion.y
    z = quaternion.z
    w = quaternion.w

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

def main(args=None):
    rclpy.init(args=args)

    pose_plotter = PosePlotter()

    rclpy.spin(pose_plotter)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    pose_plotter.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()