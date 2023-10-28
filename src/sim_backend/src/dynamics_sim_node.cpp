#include <boost/numeric/odeint.hpp>
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <cmath>

#include <tf2/LinearMath/Quaternion.h>
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "sim_backend/msg/sys_input.hpp"
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

using namespace std::chrono_literals;
using std::placeholders::_1;
using namespace boost::numeric::odeint;


class DynamicsSimulator : public rclcpp::Node
{
    typedef std::vector<double> state_type;
    public:
        DynamicsSimulator()
        : Node("dynamics_simulator")
        {
            pose_publisher_ = this->create_publisher<geometry_msgs::msg::Pose>("vehicle_pose", 10);
            twist_publisher_ = this->create_publisher<geometry_msgs::msg::Twist>("vehicle_twist", 10);
            input_subscription_ = this->create_subscription<sim_backend::msg::SysInput>(
                "vehicle_input", 10, std::bind(&DynamicsSimulator::update_input, this, _1));
            solve_timer_ = this->create_wall_timer(
                this->dt_, std::bind(&DynamicsSimulator::solve_step, this)
            );
            this->t_ = 0.0;
            publish_timer_ = this->create_wall_timer(
                this->publish_dt_, std::bind(&DynamicsSimulator::publish_state, this)
            );
        }

    private:
        void update_input(const sim_backend::msg::SysInput & msg)
        {
            RCLCPP_INFO_STREAM(this->get_logger(), 
                "Received Input Fx_f: " << msg.fx_f << ""
                << ", Fx_r: " << msg.fx_r << ", delta_s: " << msg.del_s);
            this->Fx_f = msg.fx_f;
            this->Fx_r = msg.fx_r;
            this->delta_s = msg.del_s;
        }

        void publish_state()
        {
            auto pose_msg = geometry_msgs::msg::Pose();
            tf2::Quaternion q;
            q.setRPY(0, 0, this->x[2]);
            pose_msg.position.x = this->x[0];
            pose_msg.position.y = this->x[1];
            geometry_msgs::msg::Quaternion msg_quat = tf2::toMsg(q);
            pose_msg.orientation = msg_quat;

            auto twist_msg = geometry_msgs::msg::Twist();
            twist_msg.linear.x = this->x[3];
            twist_msg.linear.y = this->x[4];
            twist_msg.angular.z = this->x[5];

            RCLCPP_INFO(this->get_logger(), "Publishing Pose and Twist");
            pose_publisher_->publish(pose_msg);
            twist_publisher_->publish(twist_msg);
        }

        void solve_step()
        {
            this->rk4.do_step(
                [this]
                (const DynamicsSimulator::state_type & x , 
                DynamicsSimulator::state_type & dxdt , 
                const double t){
                    /*
                        Formulation of the vehicle dynamics
                        - dynamic bicycle model 
                        - lateral Pacejka
                        - instant driving forces
                        - formulated in vehicle body frame

                        Inputs:
                        - Steering angle
                        - Force from motor torque on front axle
                        - Force from motor torque on rear axle

                        State:
                        - [x, y, psi, v_x, v_y, psi_dot]
                    */

                    // params
                    double l_f = 0.8; // m
                    double l_r = 0.7; // m
                    double l = l_f + l_r;
                    double m = 220; // kg
                    double Iz = 100; // kg m m
                    double g = 9.81; // m s-2
                    double D_tire = -5;
                    double C_tire = 1.2;
                    double B_tire = 9.5;
                    double C_d = 0.5 * 1.225 * 1.85; // 0.5 * rho * CdA
                    double C_r = 0.5; // -

                    // inputs
                    double F_drive_f = this->Fx_f;
                    double F_drive_r = this->Fx_r;
                    double delta_steer = this->delta_s;

                    // sideslip angles
                    double alpha_f = delta_steer - atan2((l_f * x[5] + x[4]), x[3]);
                    double alpha_r = - atan2((-l_r * x[5] + x[4]), x[3]);

                    // Resistance only in x-direction of vehicle frame
                    double F_resist = tanh(x[3]/1e-6) * (C_d * pow(x[3], 2) + C_r * m * g);

                    // Vertical tire loads
                    double Fz_f = m * g * l_r/l;
                    double Fz_r = m * g * l_f/l;

                    // Lateral tire loads (Pacejka model)
                    double Fy_f = Fz_f * D_tire * sin(C_tire * atan(B_tire * alpha_f));
                    double Fy_r = Fz_r * D_tire * sin(C_tire * atan(B_tire * alpha_r));

                    // Differential Equations
                    dxdt[0] = x[3];
                    dxdt[1] = x[4];
                    dxdt[2] = x[5];
                    dxdt[3] = (F_drive_f * cos(delta_steer)
                        + F_drive_r
                        - Fy_f * sin(delta_steer)
                        - F_resist) / m
                        + x[5] * x[4];
                    dxdt[4] = (Fy_r 
                        + Fy_f * cos(delta_steer)
                        + F_drive_f * sin(delta_steer)) / m
                        - x[5] * x[3];
                    dxdt[5] = (F_drive_f * sin(delta_steer)
                        - Fy_r * l_r
                        + Fy_f * cos(delta_steer)) / Iz;
                } , this->x , this->t_ , this->dt_f_ );
        }

        rclcpp::TimerBase::SharedPtr solve_timer_;
        rclcpp::TimerBase::SharedPtr publish_timer_;
        rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr pose_publisher_;
        rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr twist_publisher_;
        rclcpp::Subscription<sim_backend::msg::SysInput>::SharedPtr input_subscription_;
        std::chrono::microseconds dt_{std::chrono::microseconds(10)};
        std::chrono::milliseconds publish_dt_{std::chrono::milliseconds(10)};
        const double dt_f_ = dt_.count() / 1e6;
        DynamicsSimulator::state_type x{DynamicsSimulator::state_type(6)};
        double t_ = 0.0;
        double Fx_f = 0.0;
        double Fx_r = 0.0;
        double delta_s = 0.0;

        runge_kutta4< DynamicsSimulator::state_type > rk4;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DynamicsSimulator>());
  rclcpp::shutdown();
  return 0;
}