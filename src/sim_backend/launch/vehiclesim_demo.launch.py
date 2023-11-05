from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='sim_backend',
            executable='dynamics_sim_node',
            name='dynamics_simulator'
        ),
        Node(
            package='sim_backend',
            executable='vehicle_frame_publisher',
            name='vehicle_frame_publisher',
            parameters=[
                {'tf_name': 'vehicle_frame'}
            ]
        ),
    ])