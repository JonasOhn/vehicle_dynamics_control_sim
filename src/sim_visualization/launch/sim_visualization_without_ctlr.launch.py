"""
 * Simple Vehicle Dynamics Simulator Project
 *
 * Copyright (c) 2023-2024 Authors:
 *   - Jonas Ohnemus <johnemus@ethz.ch>
 *
 * All rights reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
"""

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
            os.path.join(
                get_package_share_directory("foxglove_bridge"),
                "launch/foxglove_bridge_launch.xml",
            )
        )
    )

    sim_time_node = Node(
        package="sim_backend",
        executable="sim_time_node",
        name="sim_time_node",
        parameters=[
            os.path.join(
                get_package_share_directory("sim_backend"), "config", "time_params.yaml"
            )
        ],
        output="screen",
    )

    sim_backend_node = Node(
        package="sim_backend",
        executable="dynamics_simulator",
        name="dynamics_simulator",
        parameters=[
            os.path.join(
                get_package_share_directory("sim_backend"),
                "config",
                "vehicle_params.yaml",
            )
        ],
        output="log",
        arguments=["--ros-args", "--log-level", "debug"],
    )

    tf_pub_node = Node(
        package="sim_backend",
        executable="vehicle_frame_publisher",
        name="vehicle_frame_publisher",
        parameters=[
            os.path.join(
                get_package_share_directory("sim_backend"),
                "config",
                "transform_params.yaml",
            )
        ],
        output="screen",
        arguments=["--ros-args", "--log-level", "info"],
    )

    sim_visualization = Node(
        package="sim_visualization",
        executable="sim_visualization_node",
        name="sim_visualization",
        output="screen",
    )

    ld.add_action(visualization)
    ld.add_action(sim_time_node)
    ld.add_action(sim_backend_node)
    ld.add_action(tf_pub_node)
    ld.add_action(sim_visualization)

    return ld
