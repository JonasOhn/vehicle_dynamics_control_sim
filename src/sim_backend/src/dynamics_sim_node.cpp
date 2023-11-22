#include <boost/numeric/odeint.hpp>
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <cmath>

#include <tf2/LinearMath/Quaternion.h>
#include "rclcpp/rclcpp.hpp"
#include "sim_backend/msg/vehicle_state.hpp"
#include "sim_backend/msg/sys_input.hpp"
#include "sim_backend/msg/ref_path.hpp"
#include "sim_backend/msg/point2_d.hpp"
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include "sim_backend/dynamic_system.hpp"
#include "sim_backend/sim_geometry.hpp"
#include <fstream>

using namespace std::chrono_literals;
using namespace boost::numeric::odeint;


class DynamicsSimulator : public rclcpp::Node
{
    public:
        DynamicsSimulator()
        : Node("dynamics_simulator",
                rclcpp::NodeOptions()
                    .allow_undeclared_parameters(true)
                    .automatically_declare_parameters_from_overrides(true))
        {
            if (!(this->get_csv_ref_track())){
            RCLCPP_ERROR_STREAM(this->get_logger(), "Something went wrong reading CSV ref points file!");
            }
            print_global_refpoints();

            x_[0] = 0.0;
            x_[1] = 0.0;
            x_[2] = 0.0;
            x_[3] = 0.0;
            x_[4] = 0.0;
            x_[5] = 0.0;
            x_[6] = 0.0;
            x_[7] = 0.0;
            x_[8] = 0.0;
            x_[9] = 0.0;

            this->gamma_ = this->get_parameter("gamma").as_double();
            this->r_perception_ = this->get_parameter("r_perception").as_double();

            initial_idx_refloop_ = 0;

            sys_ = DynamicSystem();
            parameters param_struct;
            param_struct.l_f = this->get_parameter("l_f").as_double();
            param_struct.l_r = this->get_parameter("l_r").as_double();
            param_struct.m = this->get_parameter("m").as_double();
            param_struct.Iz = this->get_parameter("Iz").as_double();
            param_struct.g = this->get_parameter("g").as_double();
            param_struct.D_tire = this->get_parameter("D_tire").as_double();
            param_struct.C_tire = this->get_parameter("C_tire").as_double();
            param_struct.B_tire = this->get_parameter("B_tire").as_double();
            param_struct.C_d = this->get_parameter("C_d").as_double();
            param_struct.C_r = this->get_parameter("C_r").as_double();
            param_struct.T_mot = this->get_parameter("T_mot").as_double();
            param_struct.D_mot = this->get_parameter("D_mot").as_double();
            sys_.update_parameters(param_struct);

            dt_seconds_ = dt_.count() / 1e3;
            t_ = 0.0;
            start_time_ns_ = (double)(this->now().nanoseconds());  // [ns]

            state_publisher_ = this->create_publisher<sim_backend::msg::VehicleState>("vehicle_state", 10);
            ref_path_publisher_ = this->create_publisher<sim_backend::msg::RefPath>("reference_path", 10);

            solve_timer_ = this->create_wall_timer(
                this->dt_, std::bind(&DynamicsSimulator::solve_step, this));
            ref_path_timer_ = this->create_wall_timer(
                this->dt_, std::bind(&DynamicsSimulator::ref_path_callback, this));

            input_subscription_ = this->create_subscription<sim_backend::msg::SysInput>(
                "vehicle_input", 10, std::bind(&DynamicsSimulator::update_input, this, std::placeholders::_1));

            RCLCPP_INFO_STREAM(this->get_logger(), "Node " << this->get_name() << " initialized.");
        }

    private:
        void update_input(const sim_backend::msg::SysInput & msg)
        {
            RCLCPP_INFO_STREAM(this->get_logger(), 
                "Received Input Fx_f: " << msg.fx_f << ""
                << ", Fx_r: " << msg.fx_r << ", delta_s: " << msg.del_s);
            this->sys_.update_inputs(msg.fx_f, msg.fx_r, msg.del_s);
        }

        int get_csv_ref_track(){
            std::ifstream  data("src/sim_backend/tracks/FSG_middle_path.csv");
            std::string line;
            while(std::getline(data, line))
            {
                std::stringstream lineStream(line);
                std::string cell;
                std::vector<double> parsedRow;
                while(std::getline(lineStream, cell, ','))
                {
                    parsedRow.push_back(std::stod(cell));
                }
                ref_points_global_.push_back(parsedRow);
            }
            return 1;
        }

        void print_global_refpoints()
        {
            RCLCPP_INFO_STREAM(this->get_logger(), "=== \nGot 2D Points (reference) as an array:");
            for (size_t i=0; i<ref_points_global_.size(); i++)
            {
                RCLCPP_INFO_STREAM(this->get_logger(), "x: " << ref_points_global_[i][0] << ", y: " << ref_points_global_[i][1]);
            }
            RCLCPP_INFO_STREAM(this->get_logger(), "===");
        }

        void solve_step()
        {
            /* Get start time of solve step */
            double t0 = (double_t)(this->now().nanoseconds())/1e6;  // [ms]

            //RCLCPP_INFO_STREAM(this->get_logger(), "Step began at time t_ = " << t_ << " s.");
            //RCLCPP_INFO_STREAM(this->get_logger(), "Start relative clock time: " << (t0 - start_time_ns_/1e6)/1e3 << " s.");
            
            double adaptive_dt = 1e-4;
            size_t steps = integrate_adaptive(stepper_, sys_, x_, t_, t_ + dt_seconds_, adaptive_dt);
            t_ = t_ + dt_seconds_;
            //RCLCPP_INFO_STREAM(this->get_logger(), "Solver produced a valid result integrating " << steps << " step(s) forward.");

            auto state_msg = sim_backend::msg::VehicleState();
            /* x = [xc_I, yc_I, psi, dxc_V, dyc_V, dpsi, fx_f, dfx_f, fx_r, dfx_r] */
            state_msg.x_c = x_[0];
            state_msg.y_c = x_[1];
            state_msg.psi = x_[2];
            state_msg.dx_c = x_[3] * cos(x_[2]) - x_[4] * sin(x_[2]); // rotated into global frame
            state_msg.dy_c = x_[3] * sin(x_[2]) + x_[4] * cos(x_[2]);   
            state_msg.dpsi = x_[5];
            state_msg.fx_f_act = x_[6];
            state_msg.dfx_f_act = x_[7];
            state_msg.fx_r_act = x_[8];
            state_msg.dfx_r_act = x_[9];

            state_publisher_->publish(state_msg);
            
            /* Get end time of solve step */
            double t1 = (double_t)(this->now().nanoseconds()) / 1e6;  // [ms]
            RCLCPP_INFO_STREAM(this->get_logger(), "Time needed for step: " << t1-t0 << " ms. \nSolver did " << steps << " step(s).");

            if (t1 - t0 > 0.99 * dt_.count()){
                RCLCPP_ERROR_STREAM(this->get_logger(), "Needed too long for solver step!");
            }
            // RCLCPP_INFO_STREAM(this->get_logger(), "Step ended at time t_ = " << t_ << " s.");
            // RCLCPP_INFO_STREAM(this->get_logger(), "End relative clock time: " << (t1 - start_time_ns_/1e6)/1e3 << " s.");
        }

        void ref_path_callback()
        {
            this->gamma_ = this->get_parameter("gamma").as_double();
            this->r_perception_ = this->get_parameter("r_perception").as_double();

            double x_c = x_[0];
            double y_c = x_[1];
            double psi = x_[2];

            double a_1_neg = - sin(psi - gamma_);
            double a_2_neg = cos(psi - gamma_);
            double b_neg = a_1_neg * x_c + a_2_neg * y_c;

            double a_1_pos = sin(psi + gamma_);
            double a_2_pos = - cos(psi + gamma_);
            double b_pos = a_1_pos * x_c + a_2_pos * y_c;

            auto ref_path_message = sim_backend::msg::RefPath();
            auto path_pos = sim_backend::msg::Point2D();

            path_pos.point_2d[0] = x_c;
            path_pos.point_2d[1] = y_c;

            ref_path_message.ref_path.push_back(path_pos);

            size_t idx = 0;
            bool first_visited = false;

            for (size_t i = initial_idx_refloop_; i < (ref_points_global_.size() + initial_idx_refloop_); i++)
            {
                idx = i % ref_points_global_.size();
                path_pos.point_2d[0] = ref_points_global_[idx][0];
                path_pos.point_2d[1] = ref_points_global_[idx][1];

                // Check if global candidate point lies in the cone defined by the two "perception" halfspaces
                // And if global point lies inside a "perception circle"
                if((a_1_neg * path_pos.point_2d[0] + a_2_neg * path_pos.point_2d[1] >= b_neg) && 
                    (a_1_pos * path_pos.point_2d[0] + a_2_pos * path_pos.point_2d[1] >= b_pos) &&
                    (sqrt(pow(path_pos.point_2d[0] - x_c, 2) + pow(path_pos.point_2d[1] - y_c, 2)) <= this->r_perception_)){
                    ref_path_message.ref_path.push_back(path_pos);
                    if (!first_visited){
                        first_visited = true;
                        initial_idx_refloop_ = idx;
                    }
                    // RCLCPP_INFO_STREAM(this->get_logger(), "Adding point to local ref path, x: " << path_pos.point_2d[0] << ", y: " << path_pos.point_2d[1]);
                }
            }
            ref_path_publisher_->publish(ref_path_message);
        }

        // Timer for solving the ODE
        rclcpp::TimerBase::SharedPtr solve_timer_;
        rclcpp::TimerBase::SharedPtr ref_path_timer_;

        // Publisher for vehicle state
        rclcpp::Publisher<sim_backend::msg::VehicleState>::SharedPtr state_publisher_;
        rclcpp::Publisher<sim_backend::msg::RefPath>::SharedPtr ref_path_publisher_;

        // Subscriber to update input signals
        rclcpp::Subscription<sim_backend::msg::SysInput>::SharedPtr input_subscription_;

        // Cycle time of Simulation node
        std::chrono::milliseconds dt_{std::chrono::milliseconds(5)};
        double dt_seconds_;

        // Maximum step time for ODE solver
        double max_dt_ = 1e-3;

        // Time scalar that gets updated each time step
        double t_;
        double start_time_ns_;

        // "Perception Cone angle"
        double gamma_;
        double r_perception_;
        size_t initial_idx_refloop_;

        // State vector of dynamic system
        state_type x_{state_type(10)};

        // Init System
        DynamicSystem sys_;

        // Global reference path
        std::vector<std::vector<double>> ref_points_global_;
        // Local reference path
        std::vector<std::vector<double>> ref_points_local_;

        // Create uncontrolled ODE stepper
        runge_kutta_dopri5< state_type > stepper_uncontrolled_ = runge_kutta_dopri5< state_type >();
        // Create Default Error checker
        default_error_checker<double, 
            range_algebra, 
            default_operations
            > error_checker = default_error_checker<double, range_algebra, default_operations>(1e-2, 1e-2);
        // Create Default Step Adjuster
        default_step_adjuster<double, double> step_adjuster = default_step_adjuster<double, double>(static_cast<double>(max_dt_));

        // finally create controlled stepper for error-aware stepping in the ODE solve process
        controlled_runge_kutta<
            runge_kutta_dopri5< state_type >, 
            default_error_checker<double, range_algebra, default_operations>,
            default_step_adjuster<double, double>,
            initially_resizer
            > stepper_ = controlled_runge_kutta<runge_kutta_dopri5< state_type >, 
                                                default_error_checker<double, range_algebra, default_operations>,
                                                default_step_adjuster<double, double>,
                                                initially_resizer>(error_checker,
                                                                   step_adjuster,
                                                                   stepper_uncontrolled_);
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DynamicsSimulator>());
  rclcpp::shutdown();
  return 0;
}