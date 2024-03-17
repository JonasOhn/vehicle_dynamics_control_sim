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
from launch.actions import DeclareLaunchArgument
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="mpc_controller",
                executable="mpc_controller_node",
                name="mpc_controller_node",
                parameters=[
                    os.path.join(
                        get_package_share_directory("mpc_controller"),
                        "config",
                        "mpc_controller.yaml",
                    )
                ],
                output="log",
                arguments=["--ros-args", "--log-level", "info"],
            ),
        ]
    )
