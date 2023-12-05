#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='mpc_controller',
            executable='mpc_controller_node',
            name='mpc_controller_node',
            parameters=[os.path.join(
                get_package_share_directory('mpc_controller'),
                'config', 'mpc_controller.yaml')],
            output='log',
        ),
    ])