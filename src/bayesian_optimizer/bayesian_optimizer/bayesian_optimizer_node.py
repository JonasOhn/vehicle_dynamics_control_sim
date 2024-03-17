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

import rclpy
from rclpy.node import Node
import numpy as np
from mpc_controller.msg import MpcCostParameters
from std_msgs.msg import Int32, Float64, Bool, Empty
from bayesian_optimizer.bo import BayesianOptimizer
import yaml
from bayesian_optimizer.gp import GaussianProcess
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import typing
import time

RANDOM_SAMPLE = False


class BayesianOptimizerNode(Node):
    def __init__(self):
        super().__init__("bayesian_optimizer")

        self.fig = plt.figure(1)
        plt.ion()
        plt.show()

        self.results_csv_filepath = "/home/jonas/AMZ/vehicle_dynamics_control_sim/src/bayesian_optimizer/results/results.csv"

        # Publishers
        self.mpc_cost_parameters_publisher_ = self.create_publisher(
            MpcCostParameters, "mpc_cost_parameters", 10
        )
        self.reset_mpc_publisher_ = self.create_publisher(Empty, "reset_controller", 10)
        self.start_mpc_publisher_ = self.create_publisher(Empty, "start_controller", 10)
        self.stop_mpc_publisher_ = self.create_publisher(Empty, "stop_controller", 10)
        self.reset_sim_publisher_ = self.create_publisher(Empty, "reset_sim", 10)

        self.plot_timer_period = 0.1
        self.plot_bo_timer = self.create_timer(
            self.plot_timer_period, self.plot_bo_results
        )
        self.plot_angle = 0.0

        # Subscribers
        self.last_lap_time_subscription = self.create_subscription(
            Float64, "last_lap_time", self.last_lap_time_callback, 1
        )
        self.last_lap_time_subscription

        self.current_lap_time_subscription = self.create_subscription(
            Float64, "lap_time", self.current_lap_time_callback, 1
        )
        self.current_lap_time_subscription

        self.num_cones_hit_subscription = self.create_subscription(
            Int32, "num_cones_hit", self.num_cones_hit_callback, 1
        )
        self.num_cones_hit_subscription

        self.lap_ongoing = False

        # period after which simulation and controller are reset
        self.max_lap_time = 40.0

        # define bounds on the decision variables
        self.lb_q_sd = 0.01
        self.ub_q_sd = 2.0

        self.lb_q_n = 0.01
        self.ub_q_n = 2.0

        self.lb_q_mu = 0.01
        self.ub_q_mu = 2.0

        self.num_acqfct_samples = 500
        self.Q = np.zeros((self.num_acqfct_samples, 3))

        # time penalty per cone in seconds
        self.penalty_per_cone = 2.0

        self.num_cones_hit = 0

        gp_noise_covariance = 0.1
        gp_lengthscale = 1.0
        gp_output_variance = 1.0
        self.gp_mean = 30.0
        beta = 1.0

        gp = GaussianProcess(
            noise_covariance=gp_noise_covariance,
            lengthscale=gp_lengthscale,
            output_variance=gp_output_variance,
        )

        self.bayesian_optimizer = BayesianOptimizer(gp=gp, beta=beta)

        mpc_config_yaml_path = "/home/jonas/AMZ/vehicle_dynamics_control_sim/src/mpc_controller/config/mpc_controller.yaml"
        with open(mpc_config_yaml_path, "r") as file:
            yaml_params = yaml.safe_load(file)

        yaml_params = yaml_params["/mpc_controller_node"]["ros__parameters"]["cost"]

        q_sd = yaml_params["q_sd"]
        q_n = yaml_params["q_n"]
        q_mu = yaml_params["q_mu"]
        q_dels = yaml_params["q_dels"]
        q_ax = yaml_params["q_ax"]
        r_dax = yaml_params["r_dax"]
        r_ddels = yaml_params["r_ddels"]

        self.current_parameters = np.array(
            [q_sd, q_n, q_mu, q_dels, q_ax, r_dax, r_ddels]
        )

        self.send_current_controller_params()

        self.load_data()

        self.start_first_lap()

        self.resample_acqfct_optimizer_inputs()

    def resample_acqfct_optimizer_inputs(self):

        if RANDOM_SAMPLE:
            # q_sd:
            self.Q[:, 0] = np.random.uniform(
                low=self.lb_q_sd, high=self.ub_q_sd, size=self.num_acqfct_samples
            )
            # q_n:
            self.Q[:, 1] = np.random.uniform(
                low=self.lb_q_n, high=self.ub_q_n, size=self.num_acqfct_samples
            )
            # q_mu
            self.Q[:, 2] = np.random.uniform(
                low=self.lb_q_mu, high=self.ub_q_mu, size=self.num_acqfct_samples
            )
        else:
            q_sd = np.linspace(
                start=self.lb_q_sd,
                stop=self.ub_q_sd,
                num=int(np.cbrt(self.num_acqfct_samples)),
            )
            q_n = np.linspace(
                start=self.lb_q_n,
                stop=self.ub_q_n,
                num=int(np.cbrt(self.num_acqfct_samples)),
            )
            q_mu = np.linspace(
                start=self.lb_q_mu,
                stop=self.ub_q_mu,
                num=int(np.cbrt(self.num_acqfct_samples)),
            )

            # Generate meshgrid
            X, Y, Z = np.meshgrid(q_sd, q_n, q_mu, indexing="ij")

            # Reshape meshgrid to form a matrix where each column represents one parameter dimension
            self.Q = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T

    def current_lap_time_callback(self, msg):
        if msg.data > self.max_lap_time and self.lap_ongoing:
            self.get_logger().info("Stopping lap because controller took too long.")

            self.stop_lap()

            cost = self.max_lap_time
            x_data = np.reshape(self.current_parameters[0:3], (1, 3))
            cost = np.reshape(np.array((cost)), (1, 1))

            self.bayesian_optimizer.add_data(x_data, cost - self.gp_mean)
            self.get_logger().info(
                "Added data q_sd = {0}, q_n = {1}, q_mu = {2} to Bayesian optimizer with cost = {3}.".format(
                    self.current_parameters[0],
                    self.current_parameters[1],
                    self.current_parameters[2],
                    cost - self.gp_mean,
                )
            )

            self.get_logger().info("Optimizing acquisition function.")
            self.choose_new_parameters()

            self.get_logger().info("Sending new parameters to controller.")
            self.send_current_controller_params()

            self.get_logger().info("Starting new lap.")
            self.start_lap()

        else:
            pass

    def choose_new_parameters(self):
        self.get_logger().info(
            "Choosing new parameters by optimizing acquisition function."
        )
        param_opt, _ = self.bayesian_optimizer.aquisition_function(self.Q)
        self.current_parameters[0:3] = np.squeeze(param_opt)
        self.get_logger().info(
            "Got new optimized q_sd = {0}, q_n = {1}, q_mu = {2}.".format(
                self.current_parameters[0],
                self.current_parameters[1],
                self.current_parameters[2],
            )
        )
        self.resample_acqfct_optimizer_inputs()

    def write_data_to_csv(self):
        # save current parameters as well as cost value to .csv file
        self.get_logger().info("Writing current parameters and cost to .csv.")
        X, y = self.bayesian_optimizer.get_data()
        data = np.column_stack((X, y))
        np.savetxt(self.results_csv_filepath, data, delimiter=",")
        self.get_logger().info("Written .csv file.")

    def num_cones_hit_callback(self, msg):
        self.num_cones_hit = msg.data

    def start_first_lap(self):
        self.get_logger().info("Starting first lap.")
        start_ctrl_msg = Empty()
        self.start_mpc_publisher_.publish(start_ctrl_msg)
        self.lap_ongoing = True

    def start_lap(self):
        self.get_logger().info("Sending Start Lap message.")
        start_ctrl_msg = Empty()
        self.start_mpc_publisher_.publish(start_ctrl_msg)
        self.lap_ongoing = True

    def last_lap_time_callback(self, msg):

        self.get_logger().info("Lap completion detected.")

        if self.lap_ongoing:
            # get cost from number of cones hit and lap time
            lap_time = msg.data
            num_cones_hit = self.num_cones_hit
            cost = self.calculate_cost(lap_time=lap_time, num_cones_hit=num_cones_hit)

            # Stop Lap (resets sim and controller)
            self.stop_lap()

            # add new data to BayesOpt
            x_data = np.reshape(self.current_parameters[0:3], (1, 3))
            cost = np.reshape(np.array((cost)), (1, 1))
            self.bayesian_optimizer.add_data(x_data, cost - self.gp_mean)
            self.get_logger().info(
                "Added data q_sd = {0}, q_n = {1}, q_mu = {2} to Bayesian optimizer with cost = {3}.".format(
                    self.current_parameters[0],
                    self.current_parameters[1],
                    self.current_parameters[2],
                    cost - self.gp_mean,
                )
            )

            self.write_data_to_csv()

            # optimize acquisition function to get new parameters
            self.choose_new_parameters()

            # send new parameters to controller
            self.send_current_controller_params()

            # start lap again
            self.start_lap()

    def stop_lap(self):
        self.lap_ongoing = False
        self.get_logger().info("Stopping lap.")
        stop_controller_msg = Empty()
        self.stop_mpc_publisher_.publish(stop_controller_msg)
        reset_sim_msg = Empty()
        self.reset_sim_publisher_.publish(reset_sim_msg)
        reset_ctrl_msg = Empty()
        self.reset_mpc_publisher_.publish(reset_ctrl_msg)

    def load_data(self):
        # try to load data from .csv file, if it exists. else, start with empty data and print to console
        try:
            data = np.loadtxt(self.results_csv_filepath, delimiter=",")
        except:
            self.get_logger().info(
                "No data found in .csv file. Starting with empty data."
            )
            return
        self.bayesian_optimizer.add_data(data[:, 0:3], data[:, 3])

        self.choose_new_parameters()

        self.send_current_controller_params()

    def send_current_controller_params(self):
        self.get_logger().info("Callback called for parameter update.")
        msg = MpcCostParameters()

        q_sd = self.current_parameters[0]
        q_n = self.current_parameters[1]
        q_mu = self.current_parameters[2]
        q_dels = self.current_parameters[3]
        q_ax = self.current_parameters[4]
        r_dax = self.current_parameters[5]
        r_ddels = self.current_parameters[6]

        msg.q_sd = q_sd
        msg.q_n = q_n
        msg.q_mu = q_mu
        msg.q_dels = q_dels
        msg.q_ax = q_ax
        msg.r_dax = r_dax
        msg.r_ddels = r_ddels

        self.mpc_cost_parameters_publisher_.publish(msg)
        self.get_logger().info(
            """Publishing updated parameters.
                                q_sd = {0}, q_n = {1}, q_mu = {2},
                                q_dels = {3}, q_ax = {4}, r_dax = {5},
                                r_ddels = {6}""".format(
                msg.q_sd,
                msg.q_n,
                msg.q_mu,
                msg.q_dels,
                msg.q_ax,
                msg.r_dax,
                msg.r_ddels,
            )
        )

    def calculate_cost(self, lap_time, num_cones_hit):
        return lap_time + num_cones_hit * self.penalty_per_cone


def main(args=None):
    rclpy.init(args=args)
    bayesian_opt_node = BayesianOptimizerNode()
    rclpy.spin(bayesian_opt_node)
    bayesian_opt_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
