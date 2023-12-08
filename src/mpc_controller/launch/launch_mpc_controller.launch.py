#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "log_level",
            default_value=["debug"],
            description="Logging level",
        ),
        Node(
            package='mpc_controller',
            executable='mpc_controller_node',
            name='mpc_controller_node',
            parameters=[os.path.join(
                get_package_share_directory('mpc_controller'),
                'config', 'mpc_controller.yaml')],
            output='log',
            arguments=['--ros-args', '--log-level', 'debug'],
        ),
    ])