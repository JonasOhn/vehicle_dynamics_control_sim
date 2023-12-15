#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch_xml.launch_description_sources import XMLLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    ld = LaunchDescription()

    sim_time_node = Node(
            package='sim_backend',
            executable='sim_time_node',
            name='sim_time_node',
            parameters=[os.path.join(
                get_package_share_directory('sim_backend'),
                'config', 'time_params.yaml')],
            output='screen',
        )
    
    ld.add_action(sim_time_node)
    
    return ld