import rclpy
from rclpy.node import Node
from rcl_interfaces.srv import GetParameters
from rclpy.parameter import Parameter
import numpy as np
from mpc_controller.msg import MpcCostParameters
from std_msgs.msg import Int32, Float64, Bool, Empty
from bayesian_optimizer.bo import BayesianOptimizer
import csv
import yaml
from itertools import chain
from bayesian_optimizer.gp import GaussianProcess

class BayesianOptimizerNode(Node):
    def __init__(self):
        super().__init__('bayesian_optimizer', allow_undeclared_parameters=True)

        self.results_csv_filepath = "/home/jonas/AMZ/vehicle_dynamics_control_sim/src/bayesian_optimizer/results/results.csv"

        # Publishers
        self.mpc_cost_parameters_publisher_ = self.create_publisher(MpcCostParameters, "mpc_cost_parameters", 10)
        self.reset_mpc_publisher_ = self.create_publisher(Empty, "reset_controller", 10)
        self.start_mpc_publisher_ = self.create_publisher(Empty, "start_controller", 10)
        self.stop_mpc_publisher_ = self.create_publisher(Empty, "stop_controller", 10)
        self.reset_sim_publisher_ = self.create_publisher(Empty, "reset_sim", 10)

        # Subscribers
        self.last_lap_time_subscription = self.create_subscription(
            Float64,
            "last_lap_time",
            self.last_lap_time_callback,
            1
        )
        self.last_lap_time_subscription

        self.current_lap_time_subscription = self.create_subscription(
            Float64,
            "lap_time",
            self.current_lap_time_callback,
            1
        )
        self.current_lap_time_subscription

        self.num_cones_hit_subscription = self.create_subscription(
            Int32,
            "num_cones_hit",
            self.num_cones_hit_callback,
            1
        )

        # period after which simulation and controller are reset
        self.stop_sim_period = 80.0

        # define bounds on the decision variables
        self.lb_q_sd = 0.01
        self.ub_q_sd = 2.0

        self.lb_q_n = 0.01
        self.ub_q_n = 2.0

        self.lb_q_mu = 0.01
        self.ub_q_mu = 2.0

        self.num_acqfct_samples_perdim = 100
        self.Q = np.zeros((self.num_acqfct_samples_perdim, 3))
        self.resample_acqfct_optimizer_inputs()
  
        # time penalty per cone in seconds
        self.penalty_per_cone = 2.0

        self.num_cones_hit = 0

        gp_noise_covariance = 0.0001
        gp_lengthscale = 5.0
        gp_output_variance = 1.0

        gp = GaussianProcess(noise_covariance=gp_noise_covariance,
                             lengthscale=gp_lengthscale,
                             output_variance=gp_output_variance)
        
        self.bayesian_optimizer = BayesianOptimizer(gp=gp)


        mpc_config_yaml_path = "/home/jonas/AMZ/vehicle_dynamics_control_sim/src/mpc_controller/config/mpc_controller.yaml"
        with open(mpc_config_yaml_path, 'r') as file:
            yaml_params = yaml.safe_load(file)

        yaml_params = yaml_params['/mpc_controller_node']['ros__parameters']['cost']

        q_sd = yaml_params['q_sd']
        q_n = yaml_params['q_n']
        q_mu = yaml_params['q_mu']
        r_dels = yaml_params['r_dels']
        r_ax = yaml_params['r_ax']

        self.current_parameters = np.array([q_sd,
                                            q_n,
                                            q_mu,
                                            r_dels,
                                            r_ax])

        self.send_current_controller_params()

        self.start_first_lap()

    def resample_acqfct_optimizer_inputs(self):
        # q_sd:
        self.Q[:, 0] = np.random.uniform(low=self.lb_q_sd, high=self.ub_q_sd, size=self.num_acqfct_samples_perdim)
        # q_n:
        self.Q[:, 1] = np.random.uniform(low=self.lb_q_n, high=self.ub_q_n, size=self.num_acqfct_samples_perdim)
        # q_mu
        self.Q[:, 2] = np.random.uniform(low=self.lb_q_mu, high=self.ub_q_mu, size=self.num_acqfct_samples_perdim)

    def current_lap_time_callback(self, msg):
        if msg.data > self.stop_sim_period:
            self.get_logger().info("Stopping lap because controller took too long.")
            self.stop_lap()
            cost = 1.2 * self.stop_sim_period
            x_data = np.reshape(self.current_parameters[0:3], (1, 3))
            cost = np.reshape(np.array((cost)), (1,1))
            self.bayesian_optimizer.add_data(x_data, cost)
            self.choose_new_parameters()
            self.send_current_controller_params()
            self.start_lap()
        else:
            pass

    def choose_new_parameters(self):
        self.get_logger().info("Choosing new parameters by optimizing acquisition function.")
        param_opt, _ = self.bayesian_optimizer.aquisition_function(self.Q)
        self.current_parameters[0:3] = np.squeeze(param_opt)
        self.resample_acqfct_optimizer_inputs()

    def num_cones_hit_callback(self, msg):
        self.num_cones_hit = msg.data

    def start_first_lap(self):
        self.get_logger().info("Starting first lap.")
        start_ctrl_msg = Empty()
        self.start_mpc_publisher_.publish(start_ctrl_msg)

    def start_lap(self):
        self.get_logger().info("Sending Start Lap message.")
        start_ctrl_msg = Empty()
        self.start_mpc_publisher_.publish(start_ctrl_msg)
    
        # save current parameters as well as cost value to .csv file
        self.get_logger().info("Writing parameters and cost to .csv.")
        with open(self.results_csv_filepath, 'w') as csvfile:
            wr = csv.writer(csvfile, quoting=csv.QUOTE_ALL, dialect="excel")
            wr.writerow(chain.from_iterable(self.bayesian_optimizer.get_data()))
        self.get_logger().info("Written .csv file.")
    
    def last_lap_time_callback(self, msg):
        self.get_logger().info("Lap completion detected.")
        # get cost from number of cones hit and lap time
        lap_time = msg.data
        num_cones_hit = self.num_cones_hit
        cost = self.calculate_cost(lap_time=lap_time, num_cones_hit=num_cones_hit)
        
        # Stop Lap (resets sim and controller)
        self.stop_lap()

        # add new data to BayesOpt
        x_data = np.reshape(self.current_parameters[0:3], (1, 3))
        cost = np.reshape(np.array((cost)), (1,1))
        self.bayesian_optimizer.add_data(x_data, cost)
        self.get_logger().info("Added data to Bayesian optimizer.")
        
        # optimize acquisition function to get new parameters
        self.choose_new_parameters()

        # send new parameters to controller
        self.send_current_controller_params()

        # start lap again
        self.start_lap()


    def stop_lap(self):
        self.get_logger().info("Stopping lap.")
        stop_controller_msg = Empty()
        self.stop_mpc_publisher_.publish(stop_controller_msg)
        reset_sim_msg = Empty()
        self.reset_sim_publisher_.publish(reset_sim_msg)
        reset_ctrl_msg = Empty()
        self.reset_mpc_publisher_.publish(reset_ctrl_msg)

    def send_current_controller_params(self):
        self.get_logger().info("Callback called for parameter update.")
        msg = MpcCostParameters()

        q_sd = self.current_parameters[0]
        q_n = self.current_parameters[1]
        q_mu = self.current_parameters[2]
        r_dels = self.current_parameters[3]
        r_ax = self.current_parameters[4]

        msg.q_sd = q_sd
        msg.q_n = q_n
        msg.q_mu = q_mu
        msg.r_dels = r_dels
        msg.r_ax = r_ax

        self.mpc_cost_parameters_publisher_.publish(msg)
        self.get_logger().info("Publishing updated parameters. q_sd = {0}, q_n = {1}, q_mu = {2}, r_dels = {3}, r_ax = {4}".format(msg.q_sd, msg.q_n, msg.q_mu, msg.r_dels, msg.r_ax))

    def calculate_cost(self, lap_time, num_cones_hit):
        return lap_time + num_cones_hit * self.penalty_per_cone
        

def main(args=None):
    rclpy.init(args=args)
    bayesian_opt_node = BayesianOptimizerNode()
    rclpy.spin(bayesian_opt_node)
    bayesian_opt_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()