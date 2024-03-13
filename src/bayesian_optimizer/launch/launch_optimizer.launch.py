#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='bayesian_optimizer',
            executable='bayesian_optimizer_node',
            name='bayesian_optimizer_node',
            parameters=[os.path.join(
                get_package_share_directory('bayesian_optimizer'),
                'config', 'optimizer_params.yaml')],
            output='log',
            arguments=['--ros-args', '--log-level', 'info'],
        ),
    ])