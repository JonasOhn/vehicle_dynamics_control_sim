import rclpy
import numpy as np
from rclpy.node import Node

from geometry_msgs.msg import Pose


class PlotPublisherTest(Node):

    def __init__(self):
        super().__init__('plot_publisher_test')
        self.publisher_ = self.create_publisher(Pose, 'vehicle_pose', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = Pose()
        angle = self.i/10
        msg.position.x = 2*np.cos(angle)
        msg.position.y = 2*np.sin(angle)

        q = quat_from_euler(0.0, 0.0, angle + np.pi/2)
        msg.orientation.x = q[0]
        msg.orientation.y = q[1]
        msg.orientation.z = q[2]
        msg.orientation.w = q[3]

        print(msg.orientation)
        
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing Position: i="%d"' % self.i)
        self.get_logger().info('Quaternion: q_w="%.3f"' % msg.orientation.w)
        self.i += 1
 
def quat_from_euler(roll, pitch, yaw):
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return [qx, qy, qz, qw]

def main(args=None):
    rclpy.init(args=args)

    publisher = PlotPublisherTest()

    rclpy.spin(publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()