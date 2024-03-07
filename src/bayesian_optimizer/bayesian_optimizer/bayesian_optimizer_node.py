import rclpy
from rclpy.node import Node
from rcl_interfaces.srv import GetParameters, ListParameters
from rclpy.parameter import Parameter

class BayesianOptimizer(Node):
    def __init__(self):
        super().__init__('bayesian_optimizer')

        # Initialize the parameter client
        self.parameter_client = self.create_client(
            GetParameters, '/mpc_controller_node/get_parameters')

        # Wait for the service to become available
        while not self.parameter_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Parameter service not available, waiting again...')

        # Now, you can change the parameter
        self.change_parameter()

    def change_parameter(self):
        # Prepare the parameter request
        req = GetParameters.Request()
        req.names = ['cost.q_sd']  # Change 'cost.q_sd' to the parameter you want to change

        # Call the parameter service
        future = self.parameter_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            # Extract the parameter value
            parameter_value = future.result().values[0]

            # Modify the parameter value as needed
            modified_parameter_value = 0.5  # Change this to the new value you want

            # Set the new parameter value
            self.set_parameters([Parameter('cost.q_sd', Parameter.Type.DOUBLE, modified_parameter_value)])

            self.get_logger().info('Parameter "cost.q_sd" changed to {}'.format(modified_parameter_value))
        else:
            self.get_logger().error('Failed to get parameter value')

def main(args=None):
    rclpy.init(args=args)
    bayesian_opt = BayesianOptimizer()
    rclpy.spin(bayesian_opt)
    bayesian_opt.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

if __name__ == '__main__':
    main()