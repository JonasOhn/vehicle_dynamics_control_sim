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
    )

    controller_node = Node(
        package="mpc_controller",
        executable="mpc_controller",
        name="mpc_controller",
        parameters=[
            os.path.join(
                get_package_share_directory("mpc_controller"),
                "config",
                "mpc_controller.yaml",
            )
        ],
        output="log",
    )

    tf_pub_node = Node(
        package="sim_backend",
        executable="vehicle_frame_publisher",
        name="vehicle_frame_publisher",
        parameters=[{"tf_name": "vehicle_frame"}],
    )

    ld.add_action(visualization)
    ld.add_action(sim_backend_node)
    ld.add_action(controller_node)
    ld.add_action(tf_pub_node)

    return ld
