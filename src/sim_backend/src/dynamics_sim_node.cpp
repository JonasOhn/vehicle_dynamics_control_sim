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
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include "dynamic_system.hpp"

using namespace std::chrono_literals;
using namespace boost::numeric::odeint;


class DynamicsSimulator : public rclcpp::Node
{
    public:
        DynamicsSimulator()
        : Node("dynamics_simulator")
        {
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
            dt_seconds_ = dt_.count() / 1e3;
            t_ = 0.0;
            sys_ = DynamicSystem();
            start_time_ns_ = (double)(this->now().nanoseconds());  // [ns]

            state_publisher_ = this->create_publisher<sim_backend::msg::VehicleState>("vehicle_state", 10);

            input_subscription_ = this->create_subscription<sim_backend::msg::SysInput>(
                "vehicle_input", 10, std::bind(&DynamicsSimulator::update_input, this, std::placeholders::_1));
            solve_timer_ = this->create_wall_timer(
                this->dt_, std::bind(&DynamicsSimulator::solve_step, this));
        }

    private:
        void update_input(const sim_backend::msg::SysInput & msg)
        {
            RCLCPP_INFO_STREAM(this->get_logger(), 
                "Received Input Fx_f: " << msg.fx_f << ""
                << ", Fx_r: " << msg.fx_r << ", delta_s: " << msg.del_s);
            this->sys_.update_inputs(msg.fx_f, msg.fx_r, msg.del_s);
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

        rclcpp::TimerBase::SharedPtr solve_timer_;
        rclcpp::Publisher<sim_backend::msg::VehicleState>::SharedPtr state_publisher_;
        rclcpp::Subscription<sim_backend::msg::SysInput>::SharedPtr input_subscription_;

        /* Cycle time of Simulation node */
        std::chrono::milliseconds dt_{std::chrono::milliseconds(5)};
        double dt_seconds_;

        double t_;
        double start_time_ns_;
        state_type x_{state_type(10)};

        DynamicSystem sys_;

        double max_dt_ = 1e-3;

        runge_kutta_dopri5< state_type > stepper_uncontrolled_ = runge_kutta_dopri5< state_type >();
        default_error_checker<double, 
            range_algebra, 
            default_operations
            > error_checker = default_error_checker<double, range_algebra, default_operations>(1e-2, 1e-2);
        default_step_adjuster<double, double> step_adjuster = default_step_adjuster<double, double>(static_cast<double>(max_dt_));

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