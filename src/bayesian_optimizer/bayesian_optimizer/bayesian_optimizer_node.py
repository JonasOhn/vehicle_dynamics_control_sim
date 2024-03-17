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
from std_msgs.msg import Int32, Float64, Empty
from bayesian_optimizer.bo import BayesianOptimizer
import yaml
from bayesian_optimizer.gp import GaussianProcess
import time

RANDOM_SAMPLE = True


class BayesianOptimizerNode(Node):
    def __init__(self):
        super().__init__("bayesian_optimizer")

        # --- Declare parameters
        self.declare_parameter("lb_q_sd", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("ub_q_sd", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("lb_q_n", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("ub_q_n", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("lb_q_mu", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("ub_q_mu", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("n_samples_acquisition", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("n_samples_initial", rclpy.Parameter.Type.INTEGER)
        self.declare_parameter("gp_noise_covariance", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("gp_lengthscale", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("gp_output_variance", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("gp_mean", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("bo_beta", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("penalty_per_cone", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("max_lap_time", rclpy.Parameter.Type.DOUBLE)
        self.declare_parameter("results_csv_filepath", rclpy.Parameter.Type.STRING)

        self.lb_q_sd = self.get_parameter("lb_q_sd").get_parameter_value().double_value
        self.ub_q_sd = self.get_parameter("ub_q_sd").get_parameter_value().double_value
        self.lb_q_n = self.get_parameter("lb_q_n").get_parameter_value().double_value
        self.ub_q_n = self.get_parameter("ub_q_n").get_parameter_value().double_value
        self.lb_q_mu = self.get_parameter("lb_q_mu").get_parameter_value().double_value
        self.ub_q_mu = self.get_parameter("ub_q_mu").get_parameter_value().double_value
        self.num_acqfct_samples = self.get_parameter("n_samples_acquisition").get_parameter_value().integer_value
        gp_noise_covariance = self.get_parameter("gp_noise_covariance").get_parameter_value().double_value
        gp_lengthscale = self.get_parameter("gp_lengthscale").get_parameter_value().double_value
        gp_output_variance = self.get_parameter("gp_output_variance").get_parameter_value().double_value
        self.gp_mean = self.get_parameter("gp_mean").get_parameter_value().double_value
        bo_beta = self.get_parameter("bo_beta").get_parameter_value().double_value
        self.penalty_per_cone = self.get_parameter("penalty_per_cone").get_parameter_value().double_value
        self.max_lap_time = self.get_parameter("max_lap_time").get_parameter_value().double_value
        self.n_samples_initial = self.get_parameter("n_samples_initial").get_parameter_value().integer_value

        self.results_csv_filepath = self.get_parameter("results_csv_filepath").get_parameter_value().string_value

        # --- Publishers
        self.mpc_cost_parameters_publisher_ = self.create_publisher(
            MpcCostParameters, "mpc_cost_parameters", 1
        )
        self.reset_mpc_publisher_ = self.create_publisher(Empty, "reset_controller", 1)
        self.start_mpc_publisher_ = self.create_publisher(Empty, "start_controller", 1)
        self.stop_mpc_publisher_ = self.create_publisher(Empty, "stop_controller", 1)
        self.reset_sim_publisher_ = self.create_publisher(Empty, "reset_sim", 1)

        # --- Subscribers
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

        # --- Initialize variables
        
        # flag to determine if a lap is currently ongoing
        self.lap_ongoing = False

        # counter for the number of initial data points collected    
        self.n_collected_initial_data = 0
        
        # initialize Q matrix for acquisition function optimization
        self.Q = np.zeros((self.num_acqfct_samples, 3))
        self.resample_acqfct_optimizer_inputs()

        # member variable to store the number of cones hit
        self.num_cones_hit = 0

        # initialize Gaussian Process and Bayesian Optimizer
        gp = GaussianProcess(
            noise_covariance=gp_noise_covariance,
            lengthscale=gp_lengthscale,
            output_variance=gp_output_variance,
        )
        self.bayesian_optimizer = BayesianOptimizer(gp=gp, beta=bo_beta)

        # --- Load the initial MPC parameters from the config file
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

        self.acquiring_initial_data = False

        if not self.load_data():
            self.acquiring_initial_data = True
            self.get_logger().info("Acquiring initial data.")
        else:
            self.acquiring_initial_data = False
            self.get_logger().info("Not acquiring initial data.")

        self.start_lap()

    def last_lap_time_callback(self, msg):
        """
            Callback for the last lap time topic, indicating the completion of a lap.

            Args:
                msg: last lap time in seconds

            Returns:
                None
        """

        if self.lap_ongoing:
            self.get_logger().info("Lap completion detected.")

            self.perform_optimization_step(lap_time=msg.data)
        else:
            pass

    def current_lap_time_callback(self, msg):
        """
            Callback for the current lap time topic, indicating the current lap time.

            Args:
                msg: current lap time in seconds

            Returns:
                None
        """

        if msg.data > self.max_lap_time and self.lap_ongoing:
            self.get_logger().info("Stopping lap because controller took too long.")

            self.perform_optimization_step(lap_time=msg.data)

        else:
            pass

    def perform_optimization_step(self, lap_time):
        # Stop Lap (resets sim and controller)
        self.stop_lap()

        # compute cost and add data to BayesOpt (also writes data to .csv file)
        self.get_logger().info("Adding data to Bayesian optimizer.")
        self.compute_and_add_data(lap_time=lap_time)

        # get new parameters
        self.get_logger().info("Optimizing acquisition function.")
        self.choose_new_parameters()

        # send new parameters to controller
        self.get_logger().info("Sending new parameters to controller.")
        self.send_current_controller_params()

        # start lap again
        self.get_logger().info("Starting new lap.")
        self.start_lap()

        # resample inputs for acquisition function optimizer
        self.resample_acqfct_optimizer_inputs()

    def compute_and_add_data(self, lap_time):
        """
            Compute cost from lap time and number of cones hit and add to Bayesian optimizer.

            Args:
                lap_time: lap time in seconds

            Returns:
                None
        """
        # get cost from number of cones hit and lap time
        cost = self.calculate_cost(lap_time)

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

        # write data to .csv file from BayesOpt
        self.write_data_to_csv()

    def choose_new_parameters(self):
        """
            Choose new parameters for the controller, either by 
            optimizing the acquisition function or by choosing random parameters.

            Args:
                None

            Returns:
                None
        """
        # get new parameters
        if self.acquiring_initial_data:
            if self.n_collected_initial_data < self.n_samples_initial:
                self.get_logger().info("Collecting initial data, iteration {0} out of {1}.".format(self.n_collected_initial_data, self.n_samples_initial))
                self.choose_random_parameters()
                self.n_collected_initial_data += 1
            else:
                self.get_logger().info("Initial data collection finished.")
                self.acquiring_initial_data = False
                self.get_logger().info("Optimizing acquisition function.")
                self.choose_optimized_parameters()
        else:
            self.get_logger().info("Optimizing acquisition function.")
            self.choose_optimized_parameters()

    def resample_acqfct_optimizer_inputs(self):
        """
            Resample the inputs for the acquisition function optimizer.

            Args:
                None

            Returns:
                None
        """

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

    def choose_random_parameters(self):
        """
            Choose uniformly random parameters for the controller.

            Args:
                None

            Returns:
                None
        """
        self.get_logger().info("Choosing random parameters.")
        self.current_parameters[0] = np.random.uniform(
            low=self.lb_q_sd, high=self.ub_q_sd
        )
        self.current_parameters[1] = np.random.uniform(
            low=self.lb_q_n, high=self.ub_q_n
        )
        self.current_parameters[2] = np.random.uniform(
            low=self.lb_q_mu, high=self.ub_q_mu
        )
        self.get_logger().info(
            "Got new random q_sd = {0}, q_n = {1}, q_mu = {2}.".format(
                self.current_parameters[0],
                self.current_parameters[1],
                self.current_parameters[2],
            )
        )

    def choose_optimized_parameters(self):
        """
            Choose new parameters by optimizing the acquisition function.

            Args:
                None

            Returns:
                None
        """
        self.get_logger().info(
            "Choosing new parameters by optimizing acquisition function."
        )
        param_opt, _, _ = self.bayesian_optimizer.aquisition_function(self.Q)
        self.current_parameters[0:3] = np.squeeze(param_opt)
        self.get_logger().info(
            "Got new optimized q_sd = {0}, q_n = {1}, q_mu = {2}.".format(
                self.current_parameters[0],
                self.current_parameters[1],
                self.current_parameters[2],
            )
        )

    def write_data_to_csv(self):
        """
            Write current parameters and cost to .csv file.

            Args:
                None

            Returns:
                None
        """
        # save current parameters as well as cost value to .csv file
        self.get_logger().info("Writing current parameters and cost to .csv.")

        X, y = self.bayesian_optimizer.get_data()

        self.get_logger().info("X: {0}".format(X))
        self.get_logger().info("y: {0}".format(y))

        data = np.column_stack((X, y))

        np.savetxt(self.results_csv_filepath, data, delimiter=",")
        
        self.get_logger().info("Written .csv file.")

    def num_cones_hit_callback(self, msg):
        """
            Callback for the number of cones hit.

            Args:
                msg: number of cones hit

            Returns:
                None
        """
        self.num_cones_hit = msg.data

    def start_lap(self):
        """
            Start a new lap by sending a start message to the controller.
        """
        self.get_logger().info("Sending Start Lap message.")
        start_ctrl_msg = Empty()
        self.start_mpc_publisher_.publish(start_ctrl_msg)
        self.lap_ongoing = True

        # punlish ster message two more times to make sure the controller is started
        time.sleep(0.1)
        self.get_logger().info("Sending Start Lap message a second time.")
        self.start_mpc_publisher_.publish(start_ctrl_msg)
        time.sleep(0.1)
        self.get_logger().info("Sending Start Lap message a third time.")
        self.start_mpc_publisher_.publish(start_ctrl_msg)

    def stop_lap(self):
        """
            Stop the current lap by sending a stop message to the controller.
        """
        self.lap_ongoing = False
        self.get_logger().info("Stopping lap.")
        stop_controller_msg = Empty()
        self.stop_mpc_publisher_.publish(stop_controller_msg)
        reset_sim_msg = Empty()
        self.reset_sim_publisher_.publish(reset_sim_msg)
        reset_ctrl_msg = Empty()
        self.reset_mpc_publisher_.publish(reset_ctrl_msg)

    def load_data(self):
        """
            Load data from .csv file.
        """
        self.get_logger().info("Loading data from .csv file.")
        # try to load data from .csv file, if it exists. else, start with empty data and print to console
        try:
            data = np.loadtxt(self.results_csv_filepath, delimiter=",")
        except:
            self.get_logger().info(
                "No data found in .csv file. Starting with empty data."
            )
            return False
        self.bayesian_optimizer.add_data(data[:, 0:3], data[:, 3])
        # get new parameters
        self.get_logger().info("Optimizing acquisition function.")
        self.choose_new_parameters()

        self.get_logger().info("Loaded data from .csv file.")
        return True

    def send_current_controller_params(self):
        """
            Takes the current parameter array and sends them to the MPC controller.

            Args:
                None

            Returns:
                None
        """
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

    def calculate_cost(self, lap_time_nom):
        """
            Calculate cost from lap time and number of cones hit

            Args:
                lap_time: lap time in seconds

            Returns:
                cost: cost value in seconds
        """
        if lap_time_nom < self.max_lap_time:
            lap_time = lap_time_nom
        else:
            lap_time = self.max_lap_time

        return lap_time + self.num_cones_hit * self.penalty_per_cone


def main(args=None):
    rclpy.init(args=args)
    bayesian_opt_node = BayesianOptimizerNode()
    rclpy.spin(bayesian_opt_node)
    bayesian_opt_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
