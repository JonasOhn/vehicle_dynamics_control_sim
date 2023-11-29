#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='sim_frontend',
            executable='plotting_node',
            name='plotting_node',
            output='log',
        ),
        Node(
            package='sim_backend',
            executable='dynamics_simulator',
            name='dynamics_simulator',
            parameters=[os.path.join(
                get_package_share_directory('sim_backend'),
                'config', 'vehicle_params.yaml')],
            output='screen',
        ),
    ])