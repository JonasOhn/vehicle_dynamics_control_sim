import rclpy
from rclpy.node import Node
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from geometry_msgs.msg import Pose
import numpy as np

class PosePlotter(Node):

    def __init__(self):
        super().__init__('plotting_node')
        self.subscription = self.create_subscription(
            Pose,
            'vehicle_pose',
            self.plot_pose_callback,
            10)
        self.subscription  # prevent unused variable warning
        
        plt.ion()

        self.fig = plt.figure()
        self.x_pos = 0.0
        self.y_pos = 0.0
        self.psi = 0.0

    def plot_pose_callback(self, msg):
        # Clear current axis
        plt.cla()
        # Get axis from fig
        ax = self.fig.axes[0]

        # Get new car pose in 2D from msg
        self.x_pos = msg.position.x
        self.y_pos = msg.position.y
        _, _, self.psi = euler_from_quat(msg.orientation)
        dx = np.cos(self.psi) * 2
        dy = np.sin(self.psi) * 24

        ax.arrow(self.x_pos, self.y_pos, dx, dy, width=0.5, head_width=0.6)

        ax.set_xlim(left=-50, right=50)
        ax.set_ylim(bottom=-50, top=50)

        print(self.psi*180.0/np.pi)
    
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        self.get_logger().info("Plotting")

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