#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch_xml.launch_description_sources import XMLLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    ld = LaunchDescription()
    
    visualization = IncludeLaunchDescription(
            XMLLaunchDescriptionSource(
                os.path.join(get_package_share_directory('foxglove_bridge'),
                             'launch/foxglove_bridge_launch.xml')
            )
        )
    
    sim_backend_node = Node(
            package='sim_backend',
            executable='dynamics_simulator',
            name='dynamics_simulator',
            parameters=[os.path.join(
                get_package_share_directory('sim_backend'),
                'config', 'vehicle_params.yaml')],
            output='log',
        )
    
    controller_node = Node(
            package='pure_pursuit_controller',
            executable='pp_controller',
            name='pp_controller',
            parameters=[os.path.join(
                get_package_share_directory('pure_pursuit_controller'),
                'config', 'pp_controller.yaml')],
            output='log',
        )
    
    tf_pub_node = Node(
            package='sim_backend',
            executable='vehicle_frame_publisher',
            name='vehicle_frame_publisher',
            parameters=[
                {'tf_name': 'vehicle_frame'}
            ]
        )
    
    ld.add_action(visualization)
    ld.add_action(sim_backend_node)
    ld.add_action(controller_node)
    ld.add_action(tf_pub_node)
    
    return ld