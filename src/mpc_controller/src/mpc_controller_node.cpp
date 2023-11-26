#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <cmath>

#include "rclcpp/rclcpp.hpp"
#include "sim_backend/msg/sys_input.hpp"
#include "sim_backend/msg/vehicle_state.hpp"
#include "acados_solvers_library/acados_solver_library.hpp"
#include <fstream>

using namespace std::chrono_literals;


class MPCController : public rclcpp::Node
{
  public:
    MPCController()
    : Node("mpc_controller",
            rclcpp::NodeOptions()
                .allow_undeclared_parameters(true)
                .automatically_declare_parameters_from_overrides(true))
    {
        if ((this->get_csv_ref_track())){
            RCLCPP_ERROR_STREAM(this->get_logger(), "Something went wrong reading CSV ref points file!");
        }
        print_refpoints();

        dt_seconds_ = dt_.count() / 1e3;

        l_f_ = this->get_parameter("l_f").as_double();
        l_r_ = this->get_parameter("l_r").as_double();
        m_ = this->get_parameter("m").as_double();
        Iz_ = this->get_parameter("Iz").as_double();
        g_ = this->get_parameter("g").as_double();
        D_tire_ = this->get_parameter("D_tire").as_double();
        C_tire_ = this->get_parameter("C_tire").as_double();
        B_tire_ = this->get_parameter("B_tire").as_double();
        C_d_ = this->get_parameter("C_d").as_double();
        C_r_ = this->get_parameter("C_r").as_double();

        state_subscriber_ = this->create_subscription<sim_backend::msg::VehicleState>(
            "vehicle_state", 1, std::bind(&MPCController::cartesian_state_update, this, std::placeholders::_1));

        control_cmd_publisher_ = this->create_publisher<sim_backend::msg::SysInput>("vehicle_input", 10);
        control_cmd_timer_ = this->create_wall_timer(this->dt_, std::bind(&MPCController::control_callback, this));
    }

  private:

    void control_callback()
    {
        auto veh_input_msg = sim_backend::msg::SysInput();

        double u[] = {0.0, 0.0, 0.0};
        
        veh_input_msg.fx_r = u[0];
        veh_input_msg.fx_f = u[1];
        veh_input_msg.del_s = u[2];

        control_cmd_publisher_->publish(veh_input_msg);
    }

    void cartesian_state_update(const sim_backend::msg::VehicleState & state_msg)
    {
        x_[0] = state_msg.x_c;
        x_[1] = state_msg.y_c;
        x_[2] = state_msg.psi;
        x_[3] = state_msg.dx_c;
        x_[4] = state_msg.dy_c;
        x_[5] = state_msg.dpsi;
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
            ref_points_.push_back(parsedRow);
        }
        return 0;
    }

    void print_refpoints()
    {
        RCLCPP_INFO_STREAM(this->get_logger(), "=== \nGot 2D Points (reference) as an array:");
        for (size_t i=0; i<ref_points_.size(); i++)
        {
            RCLCPP_INFO_STREAM(this->get_logger(), "x: " << ref_points_[i][0] << ", y: " << ref_points_[i][1]);
        }
        RCLCPP_INFO_STREAM(this->get_logger(), "===");
    }


    rclcpp::Subscription<sim_backend::msg::VehicleState>::SharedPtr state_subscriber_;
    rclcpp::TimerBase::SharedPtr control_cmd_timer_;
    rclcpp::Publisher<sim_backend::msg::SysInput>::SharedPtr control_cmd_publisher_;
    std::vector<std::vector<double>> ref_points_;
    std::chrono::milliseconds dt_{std::chrono::milliseconds(50)};
    double dt_seconds_;
    double x_[6] = {0.0};

    AcadosSolver acados_solver_;

    double l_f_;
    double l_r_;
    double m_;
    double Iz_;
    double g_;
    double D_tire_;
    double C_tire_;
    double B_tire_;
    double C_d_;
    double C_r_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MPCController>());
  rclcpp::shutdown();
  return 0;
}