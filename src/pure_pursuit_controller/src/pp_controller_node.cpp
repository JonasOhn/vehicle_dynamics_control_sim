#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <cmath>

#include "rclcpp/rclcpp.hpp"
#include "sim_backend/msg/sys_input.hpp"
#include "sim_backend/msg/vehicle_state.hpp"
#include <fstream>

using namespace std::chrono_literals;

struct Pose2D {
    double x;
    double y;
    double psi;
};

struct Point2D {
    double x;
    double y;
};


class PPController : public rclcpp::Node
{
  public:
    PPController()
    : Node("pp_controller")
    {
        if (!(this->get_csv_ref_track())){
            RCLCPP_ERROR_STREAM(this->get_logger(), "Something went wrong reading CSV ref points file!");
        }
        print_refpoints();

        L_wb_ = 1.5;

        current_pose_.x = 0.0;
        current_pose_.y = 0.0;
        current_pose_.psi = 0.0;
        l_d_ = 3.0;
        v_act_ = 0.0;
        v_ref_ = 8.0;
        e_v_ = 0.0;
        e_v_integral_ = 0.0;
        dt_seconds_ = dt_.count() / 1e3;

        K_I_v_ = 50;
        K_P_v_ = 400;
        K_D_v_ = 150;

        state_subscriber_ = this->create_subscription<sim_backend::msg::VehicleState>(
            "vehicle_state", 1, std::bind(&PPController::state_update, this, std::placeholders::_1));
        control_cmd_publisher_ = this->create_publisher<sim_backend::msg::SysInput>("vehicle_input", 10);
        control_cmd_timer_ = this->create_wall_timer(this->dt_, std::bind(&PPController::control_callback, this));
        
    }

  private:

    void control_callback()
    {
        auto veh_input_msg = sim_backend::msg::SysInput();

        double fx = 0.0;
        double e_v_prev = e_v_;
        e_v_ = v_ref_ - v_act_;
        e_v_integral_ += e_v_ * dt_seconds_;

        // PID velocity control
        fx = K_P_v_ * e_v_ + 
             K_I_v_ * e_v_integral_ + 
             K_D_v_ * (e_v_ - e_v_prev) / dt_seconds_;
        
        veh_input_msg.fx_r = fx / 2.0;
        veh_input_msg.fx_f = fx / 2.0;

        // Calculate Steering 
        veh_input_msg.del_s = get_target_steering();

        control_cmd_publisher_->publish(veh_input_msg);
    }

    void state_update(const sim_backend::msg::VehicleState & state_msg)
    {
        current_pose_.x = state_msg.x_c;
        current_pose_.y = state_msg.y_c;
        current_pose_.psi = state_msg.psi;
        v_act_ = sqrt(pow(state_msg.dx_c, 2) + pow(state_msg.dy_c, 2));
        // v_act_ = state_msg.dx_c * cos(current_pose_.psi) + state_msg.dy_c * sin(current_pose_.psi);
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
            Point2D point;
            point.x = parsedRow[0];
            point.y = parsedRow[1];
            ref_points_.push_back(point);
        }
        return 1;
    }

    void print_refpoints()
    {
        RCLCPP_INFO_STREAM(this->get_logger(), "=== \nGot 2D Points (reference) as an array:");
        for (size_t i=0; i<ref_points_.size(); i++)
        {
            RCLCPP_INFO_STREAM(this->get_logger(), "x: " << ref_points_[i].x << ", y: " << ref_points_[i].y);
        }
        RCLCPP_INFO_STREAM(this->get_logger(), "===");
    }

    double get_target_steering(){
        double distance = 0.0;
        double shortest_distance = 100000.0;
        size_t shortest_idx_nearest_point = 0;
        size_t shortest_idx_target_point = 0;
        size_t i=0;
        Point2D target_point_I;
        Point2D target_point_V;
        // Find point (xp_c, yp_c) closest to the vehicle
        //TODO: Currently brute-force, better search possible
        for(i=shortest_idx_nearest_point; i<ref_points_.size(); i++)
        {
            distance = sqrt(pow(current_pose_.x - ref_points_[i].x, 2) + pow(current_pose_.y - ref_points_[i].y, 2));
            if (distance < shortest_distance)
            {
                shortest_distance = distance;
                shortest_idx_nearest_point = i;
            }
        }

        RCLCPP_INFO_STREAM(this->get_logger(), "Nearest Point: idx=" << shortest_idx_nearest_point);

        i = shortest_idx_nearest_point;
        // get delta between l_d_ and current ref point on track
        shortest_distance = 100000.0;
        // Find target point (xp_l, yp_l) up the path after (xp_c, yp_c) one lookahead distance from (x_c, y_c)
        do{
            // begin at start of array again
            if(i >= ref_points_.size())
            {
                RCLCPP_INFO_STREAM(this->get_logger(), "i=" << i << " but array size= " << ref_points_.size());
                i = 0;
            }
            // get delta of lookahead distance to car-point distance
            distance = fabs(l_d_ - sqrt(pow(current_pose_.x - ref_points_[i].x, 2) + pow(current_pose_.y - ref_points_[i].y, 2)));
            RCLCPP_INFO_STREAM(this->get_logger(), "Distance: " << distance << "Shortest Distance: " << shortest_distance);
            if (distance < shortest_distance)
            {
                shortest_distance = distance;
                shortest_idx_target_point = i;
            }else{
                RCLCPP_INFO_STREAM(this->get_logger(), "Breaking out of loop with idx=" << shortest_idx_target_point);
                break;
            }
            i++;
        }while (i!=shortest_idx_nearest_point);
        
        target_point_I = ref_points_[shortest_idx_target_point];

        RCLCPP_INFO_STREAM(this->get_logger(), "Target Point: idx=" << shortest_idx_target_point << ", x="
            << target_point_I.x << ", y=" << target_point_I.y);
        
        double delta_x_I = target_point_I.x - current_pose_.x;
        double delta_y_I = target_point_I.y - current_pose_.y;
        // Transform goal point into local vehicle frame
        target_point_V.x = delta_x_I * cos(current_pose_.psi) + delta_y_I * sin(current_pose_.psi);
        target_point_V.y = - delta_x_I * sin(current_pose_.psi) + delta_y_I * cos(current_pose_.psi);
        // Get angle alpha from vehicle heading to (x_c, y_c)->(xp_l, yp_l)
        double alpha = atan2(target_point_V.y, target_point_V.x);
        return atan(2 * L_wb_ * sin(alpha) / l_d_);
    }
    rclcpp::Subscription<sim_backend::msg::VehicleState>::SharedPtr state_subscriber_;
    rclcpp::TimerBase::SharedPtr control_cmd_timer_;
    rclcpp::Publisher<sim_backend::msg::SysInput>::SharedPtr control_cmd_publisher_;
    Pose2D current_pose_;
    std::vector<Point2D> ref_points_;
    std::chrono::milliseconds dt_{std::chrono::milliseconds(15)};
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
    // Lateral controller  
    double l_d_;    // lookahead distance

    //
    double L_wb_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PPController>());
  rclcpp::shutdown();
  return 0;
}