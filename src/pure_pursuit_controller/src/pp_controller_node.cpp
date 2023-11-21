#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <cmath>

#include "rclcpp/rclcpp.hpp"
#include "sim_backend/msg/sys_input.hpp"
#include "sim_backend/msg/vehicle_state.hpp"
#include "sim_backend/msg/ref_path.hpp"
#include "sim_backend/sim_geometry.hpp"

using namespace std::chrono_literals;


class PPController : public rclcpp::Node
{
  public:
    PPController()
    : Node("pp_controller",
            rclcpp::NodeOptions()
                .allow_undeclared_parameters(true)
                .automatically_declare_parameters_from_overrides(true))
    {
        // Init pose to zero
        current_pose_.x = 0.0;
        current_pose_.y = 0.0;
        current_pose_.psi = 0.0;

        // Init current velocity
        v_act_ = 0.0;

        // Init Velocity Error
        e_v_ = 0.0;

        // Init Velocity Error Integral
        e_v_integral_ = 0.0;

        // Calculate the inverse node rate in seconds
        dt_seconds_ = dt_.count() / 1e3;

        // Init State Subscriber
        state_subscriber_ = this->create_subscription<sim_backend::msg::VehicleState>(
            "vehicle_state", 1, std::bind(&PPController::state_update, this, std::placeholders::_1));
        
        // Init reference path subscriber
        ref_path_subscriber_ = this->create_subscription<sim_backend::msg::RefPath>(
            "reference_path", 1, std::bind(&PPController::ref_path_update, this, std::placeholders::_1));

        // Init Control Command Publisher and corresponding Timer with respecitve callback
        control_cmd_publisher_ = this->create_publisher<sim_backend::msg::SysInput>("vehicle_input", 10);
        control_cmd_timer_ = this->create_wall_timer(this->dt_, std::bind(&PPController::control_callback, this));
    }

  private:

    void control_callback()
    {
        this->L_wb_ = this->get_parameter("L_wb").as_double();
        this->l_d_ = this->get_parameter("l_d").as_double();
        this->K_I_v_ = this->get_parameter("k_i_v").as_double();
        this->K_P_v_ = this->get_parameter("k_p_v").as_double();
        this->K_D_v_ = this->get_parameter("k_d_v").as_double();
        this->v_ref_ = this->get_parameter("v_ref").as_double();

        auto veh_input_msg = sim_backend::msg::SysInput();

        double e_v_prev = e_v_;
        double fx = 0.0;
        
        if (ref_points_.size() >= 2){
            this->e_v_ = this->v_ref_ - this->v_act_;
            this->e_v_integral_ += this->e_v_ * this->dt_seconds_;

            // PID velocity control
            fx = this->K_P_v_ * this->e_v_ + 
                 this->K_I_v_ * this->e_v_integral_ + 
                 this->K_D_v_ * (this->e_v_ - e_v_prev) / this->dt_seconds_;

            // Split force in half for front and rear tire
            veh_input_msg.fx_r = fx / 2.0;
            veh_input_msg.fx_f = fx / 2.0;

            // Calculate Steering 
            veh_input_msg.del_s = get_target_steering();
        } else {
            this->e_v_ = 0.0 - this->v_act_;
            this->e_v_integral_ += this->e_v_ * this->dt_seconds_;

            // PID velocity control
            fx = this->K_P_v_ * this->e_v_ + 
                 this->K_I_v_ * this->e_v_integral_ + 
                 this->K_D_v_ * (this->e_v_ - e_v_prev) / this->dt_seconds_;
            veh_input_msg.fx_r = fx / 2.0;
            veh_input_msg.fx_f = fx / 2.0;
            veh_input_msg.del_s = 0.0;
        }
        
        // Publish Control Command message
        control_cmd_publisher_->publish(veh_input_msg);
    }

    void state_update(const sim_backend::msg::VehicleState & state_msg)
    {
        current_pose_.x = state_msg.x_c;
        current_pose_.y = state_msg.y_c;
        current_pose_.psi = state_msg.psi;
        this->v_act_ = sqrt(pow(state_msg.dx_c, 2) + pow(state_msg.dy_c, 2));
        // v_act_ = state_msg.dx_c * cos(current_pose_.psi) + state_msg.dy_c * sin(current_pose_.psi);
    }

    void ref_path_update(const sim_backend::msg::RefPath & refpath_msg){
        this->ref_points_.clear();
        auto path_pos = sim_backend::msg::Point2D();
        std::vector<double> ref_point = {0.0, 0.0};
        for (size_t i=0; i<refpath_msg.ref_path.size(); i++){
            path_pos = refpath_msg.ref_path[i];
            ref_point[0] = path_pos.point_2d[0];
            ref_point[1] = path_pos.point_2d[1];
            this->ref_points_.push_back(ref_point);
            // RCLCPP_INFO_STREAM(this->get_logger(), "Adding: x=" << ref_point[0] << ", y=" << ref_point[1]);
        }
    }

    double get_target_steering(){

        double distance_to_ref_point = 0.0;
        size_t lookahead_idx_target_1 = 0;
        size_t lookahead_idx_target_2 = 0;

        // Target point for Pure Pursuit in inertial frame
        sim_geometry::Point2D target_point_I;
        // Target point for Pure Pursuit in vehicle frame
        sim_geometry::Point2D target_point_V;

        /*
            Since the first point of the reference path (published by the simulator)
            is always the car's current position, it is trivially the nearest point
            of the reference path to the vehicle. 
            We then iterate through the reference path to find the point closest to 
            the lookahead distance circle defined by the euclidean norm.

            Target Point 1: last point on path that is inside the lookahead circle
            Target Point 2: first point on path that is outside the lookahead circle
        */

        // Find target point (xp_l, yp_l) up the path after (xp_c, yp_c) one lookahead distance from (x_c, y_c)
        for (size_t i=0; i<ref_points_.size(); i++)
        {
            // get delta of lookahead distance to car-point distance
            distance_to_ref_point = sqrt(pow(current_pose_.x - ref_points_[i][0], 2) 
                                         + pow(current_pose_.y - ref_points_[i][1], 2));

            RCLCPP_INFO_STREAM(this->get_logger(), "Distance: " << distance_to_ref_point << ", Lookahead: " << this->l_d_);

            // For each point inside the circle, set it as a possible target point 1
            if (distance_to_ref_point < this->l_d_)
            {
                lookahead_idx_target_1 = i;
            // As soon as there is a point outside the lookahead circle, set it to be target point 2 and break out of loop
            }else{
                lookahead_idx_target_2 = i;
                RCLCPP_INFO_STREAM(this->get_logger(), "Breaking out of loop with idx_1=" 
                    << lookahead_idx_target_1 << " and idx_2=" << lookahead_idx_target_2);
                break;
            }
        }

        double x_c = ref_points_[0][0];
        double y_c = ref_points_[0][1];
        double x_1 = ref_points_[lookahead_idx_target_1][0];
        double y_1 = ref_points_[lookahead_idx_target_1][1];
        double x_2 = ref_points_[lookahead_idx_target_2][0];
        double y_2 = ref_points_[lookahead_idx_target_2][1];

        // Define Quadratic equation coefficients: a*t^2 + b*t + c = 0
        double a_quad_eq = pow(x_1 - x_2, 2) + pow(y_1 - y_2, 2);
        double b_quad_eq = 2 * (x_1 * x_2 - x_1 * x_c + x_2 * x_c - pow(x_2, 2) +
                                y_1 * y_2 - y_1 * y_c + y_2 * y_c - pow(y_2, 2));
        double c_quad_eq = pow(x_2 - x_c, 2) + pow(y_2 - y_c, 2) - pow(this->l_d_, 2);

        // Two possible solutions to quadratic equation
        double t_line_1 = (- b_quad_eq + sqrt(pow(b_quad_eq, 2) - 4 * a_quad_eq * c_quad_eq)) / (2 * a_quad_eq);
        double t_line_2 = (- b_quad_eq - sqrt(pow(b_quad_eq, 2) - 4 * a_quad_eq * c_quad_eq)) / (2 * a_quad_eq);

        double t = 0.0;

        if (t_line_1 <= 1.0 && t_line_1 >= 0.0)
        {
            t = t_line_1;
        }
        else if (t_line_2 <= 1.0 && t_line_2 >= 0.0)
        {
            t = t_line_2;
        }else{
            RCLCPP_ERROR_STREAM(this->get_logger(), "Could not find line-circle intersection for PP controller!");
        }

        // In case no point outside lookahead circle found
        if ( lookahead_idx_target_2 == 0 ){
            RCLCPP_ERROR_STREAM(this->get_logger(), "No target point outside lookahead distance found!");
            // Use target 1
            t = 1.0;
        }

        // Calculate target point as convex combination of targets 1 and 2
        target_point_I.x = t * x_1 + (1.0 - t) * x_2;
        target_point_I.y = t * y_1 + (1.0 - t) * y_2;

        RCLCPP_INFO_STREAM(this->get_logger(), "Target Point: x=" << target_point_I.x << ", y=" << target_point_I.y);
        
        // Get position delta vector from car to target in inertial frame
        double delta_x_I = target_point_I.x - current_pose_.x;
        double delta_y_I = target_point_I.y - current_pose_.y;

        // Transform goal point into local vehicle frame
        target_point_V.x = delta_x_I * cos(current_pose_.psi) + delta_y_I * sin(current_pose_.psi);
        target_point_V.y = - delta_x_I * sin(current_pose_.psi) + delta_y_I * cos(current_pose_.psi);

        // Get angle alpha from vehicle heading to (x_c, y_c)->(xp_l, yp_l)
        double alpha = atan2(target_point_V.y, target_point_V.x);

        return atan(2 * L_wb_ * sin(alpha) / l_d_);
    }
    
    // Subscriber to state message published at faster frequency
    rclcpp::Subscription<sim_backend::msg::VehicleState>::SharedPtr state_subscriber_;
    rclcpp::Subscription<sim_backend::msg::RefPath>::SharedPtr ref_path_subscriber_;

    // Timer for control command publishing
    rclcpp::TimerBase::SharedPtr control_cmd_timer_;

    // Control command publisher
    rclcpp::Publisher<sim_backend::msg::SysInput>::SharedPtr control_cmd_publisher_;

    // Current state as member variable of the controller
    sim_geometry::Pose2D current_pose_;

    // Reference Path ahead of the vehicle
    std::vector<std::vector<double>> ref_points_;

    // Step Time for controller publisher
    std::chrono::milliseconds dt_{std::chrono::milliseconds(5)};
    double dt_seconds_;
    
    // reference velocity (const. for now)
    double v_ref_;
    double v_act_;
    double e_v_;
    double e_v_integral_;

    // PID velocity controller
    double K_I_v_;
    double K_P_v_;
    double K_D_v_;

    // lookahead distance
    double l_d_;

    // vehicle wheel base
    double L_wb_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PPController>());
  rclcpp::shutdown();
  return 0;
}