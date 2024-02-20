#include <boost/numeric/odeint.hpp>
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <cmath>
#include <fstream>
#include <tf2/LinearMath/Quaternion.h>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/empty.hpp"
#include "sim_backend/msg/vehicle_state.hpp"
#include "sim_backend/msg/sys_input.hpp"
#include "sim_backend/msg/point2_d.hpp"
#include "sim_backend/msg/point2_d_array.hpp"
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include "tf2/LinearMath/Quaternion.h"

#include "sim_backend/dynamic_system.hpp"

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
            track_fpath_midline_ = this->get_parameter("track_filepath_midline").as_string();
            track_fpath_leftbound_ = this->get_parameter("track_filepath_leftbound").as_string();
            track_fpath_rightbound_ = this->get_parameter("track_filepath_rightbound").as_string();

            if (!(this->get_csv_ref_track()))
            {
                RCLCPP_ERROR_STREAM(this->get_logger(), "Something went wrong reading CSV ref points file!");
                rclcpp::shutdown();
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
            this->r_perception_max_ = this->get_parameter("r_perception_max").as_double();
            this->r_perception_min_ = this->get_parameter("r_perception_min").as_double();

            this->initial_idx_refloop_ = 0;

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
            track_publisher_ = this->create_publisher<sim_backend::msg::Point2DArray>("track_points2d", 10);
            trackbounds_left_publisher_ = this->create_publisher<sim_backend::msg::Point2DArray>("trackbounds_left_points2d", 10);
            trackbounds_right_publisher_ = this->create_publisher<sim_backend::msg::Point2DArray>("trackbounds_right_points2d", 10);
            velocity_vector_publisher_ = this->create_publisher<sim_backend::msg::Point2DArray>("car_cog_velocity_vector", 10);
            ref_path_publisher_marker_array_ = this->create_publisher<sim_backend::msg::Point2DArray>("reference_path_points2d", 10);

            solve_timer_ = rclcpp::create_timer(this, this->get_clock(), this->dt_, std::bind(&DynamicsSimulator::solve_step, this));
            trackpub_timer_ = rclcpp::create_timer(this, this->get_clock(), this->dt_trackpub_, std::bind(&DynamicsSimulator::track_callback, this));

            input_subscription_ = this->create_subscription<sim_backend::msg::SysInput>(
                "vehicle_input", 10, std::bind(&DynamicsSimulator::update_input, this, std::placeholders::_1));

            reset_subscription_ = this->create_subscription<std_msgs::msg::Empty>(
                "reset_sim", 10, std::bind(&DynamicsSimulator::reset_simulator, this, std::placeholders::_1));

            RCLCPP_INFO_STREAM(this->get_logger(), "Node " << this->get_name() << " initialized.");
        }

    private:

        void reset_simulator(const std_msgs::msg::Empty & msg)
        {
            RCLCPP_DEBUG_STREAM(this->get_logger(), "Resetting Sim.");
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
            start_time_ns_ = (double)(this->now().nanoseconds());  // [ns]
            this->initial_idx_refloop_ = 0;
            RCLCPP_DEBUG_STREAM(this->get_logger(), "Sim Reset.");
        }

        void update_input(const sim_backend::msg::SysInput & msg)
        {
            RCLCPP_DEBUG_STREAM(this->get_logger(), 
                "Received Input Fx_f: " << msg.fx_f << ""
                << ", Fx_r: " << msg.fx_r << ", delta_s: " << msg.del_s);
            this->sys_.update_inputs(msg.fx_f, msg.fx_r, msg.del_s);
        }

        int get_csv_ref_track(){
            std::ifstream  data_middle(track_fpath_midline_);
            std::string line;
            while(std::getline(data_middle, line))
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

            // left cones are blue
            std::ifstream data_blue(track_fpath_leftbound_);
            while(std::getline(data_blue, line))
            {
                std::stringstream lineStream(line);
                std::string cell;
                std::vector<double> parsedRow;
                while(std::getline(lineStream, cell, ','))
                {
                    parsedRow.push_back(std::stod(cell));
                }
                trackbounds_left_.push_back(parsedRow);
            }

            // right cones are yellow
            std::ifstream data_yellow(track_fpath_rightbound_);
            while(std::getline(data_yellow, line))
            {
                std::stringstream lineStream(line);
                std::string cell;
                std::vector<double> parsedRow;
                while(std::getline(lineStream, cell, ','))
                {
                    parsedRow.push_back(std::stod(cell));
                }
                trackbounds_right_.push_back(parsedRow);
            }
            return 1;
        }

        void print_global_refpoints()
        {
            RCLCPP_DEBUG_STREAM(this->get_logger(), "=== \nGot 2D Points (reference) as an array:");
            for (size_t i=0; i<ref_points_global_.size(); i++)
            {
                RCLCPP_DEBUG_STREAM(this->get_logger(), "x: " << ref_points_global_[i][0] << ", y: " << ref_points_global_[i][1]);
            }
            RCLCPP_DEBUG_STREAM(this->get_logger(), "===");
        }

        void solve_step()
        {
            RCLCPP_DEBUG_STREAM(this->get_logger(), "Solve step started.");
            /* Get start time of solve step */
            double t0 = (double_t)(this->now().nanoseconds())/1e6;  // [ms]

            RCLCPP_DEBUG_STREAM(this->get_logger(), "Step began at time t_ = " << t_ << " s.");
            RCLCPP_DEBUG_STREAM(this->get_logger(), "Start relative clock time: " << (t0 - start_time_ns_/1e6)/1e3 << " s.");
            
            double adaptive_dt = 1e-4;
            size_t steps = integrate_adaptive(stepper_, sys_, x_, t_, t_ + dt_seconds_, adaptive_dt);
            t_ = t_ + dt_seconds_;
            RCLCPP_DEBUG_STREAM(this->get_logger(), "Solver produced a valid result integrating " << steps << " step(s) forward.");

            auto state_msg = sim_backend::msg::VehicleState();
            /* x = [xc_I, yc_I, psi, dxc_V, dyc_V, dpsi, fx_f, dfx_f, fx_r, dfx_r] */
            state_msg.x_c = x_[0];
            state_msg.y_c = x_[1];
            state_msg.psi = x_[2];
            state_msg.dx_c = x_[3] * cos(x_[2]) - x_[4] * sin(x_[2]); // rotated into global frame
            state_msg.dy_c = x_[3] * sin(x_[2]) + x_[4] * cos(x_[2]);
            state_msg.dx_c_v = x_[3];
            state_msg.dy_c_v = x_[4];  
            state_msg.dpsi = x_[5];
            state_msg.fx_f_act = x_[6];
            state_msg.dfx_f_act = x_[7];
            state_msg.fx_r_act = x_[8];
            state_msg.dfx_r_act = x_[9];

            double fx_f = 0.0;
            double fx_r = 0.0;
            double del_s = 0.0;

            this->sys_.get_inputs(&fx_f, &fx_r, &del_s);

            state_msg.del_s_ref = del_s;
            state_msg.fx_f_ref = fx_f;
            state_msg.fx_r_ref = fx_r;

            state_publisher_->publish(state_msg);

            this->ref_path_callback();
            
            /* Get end time of solve step */
            double t1 = (double_t)(this->now().nanoseconds()) / 1e6;  // [ms]
            RCLCPP_DEBUG_STREAM(this->get_logger(), "Time needed for step: " << t1-t0 << " ms. \nSolver did " << steps << " step(s).");

            if (t1 - t0 > 0.99 * dt_.count()){
                RCLCPP_ERROR_STREAM(this->get_logger(), "Needed too long for solver step! Took: " << t1-t0 << " ms, but dt is " << dt_.count() << " ms!");
            }
            RCLCPP_DEBUG_STREAM(this->get_logger(), "Step ended at time t_ = " << t_ << " s.");
            RCLCPP_DEBUG_STREAM(this->get_logger(), "End relative clock time: " << (t1 - start_time_ns_/1e6)/1e3 << " s.");


            /*
            *   Publish the velocity vector in the vehicle frame as a marker arrow
            */
            auto velvec_msg = visualization_msgs::msg::Marker();
            velvec_msg.header.frame_id = "vehicle_frame";
            velvec_msg.header.stamp = this->now();
            // set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
            velvec_msg.type = 0;
            velvec_msg.id = 0;

            // Set the scale of the marker
            velvec_msg.scale.x = sqrt(pow(x_[3], 2.0) + pow(x_[4], 2.0));
            velvec_msg.scale.y = 0.1;
            velvec_msg.scale.z = 0.1;

            // Set the color
            velvec_msg.color.r = 0.0;
            velvec_msg.color.g = 1.0;
            velvec_msg.color.b = 0.0;
            velvec_msg.color.a = 1.0;

            // Set the pose of the marker
            tf2::Quaternion q;
            q.setRPY(0, 0, atan2(x_[4], x_[3]));
            velvec_msg.pose.position.x = 0;
            velvec_msg.pose.position.y = 0;
            velvec_msg.pose.position.z = 0;
            velvec_msg.pose.orientation.x = q.x();
            velvec_msg.pose.orientation.y = q.y();
            velvec_msg.pose.orientation.z = q.z();
            velvec_msg.pose.orientation.w = q.w();
            velocity_vector_publisher_->publish(velvec_msg);

            RCLCPP_DEBUG_STREAM(this->get_logger(), "Solve step end.");
        }

        void track_callback()
        {
            // This callback publishes the left cones, the right cones, 
            // and the middle points for the whole track (for visualization)

            RCLCPP_DEBUG_STREAM(this->get_logger(), "Track callback started.");
            
            // ======= Track publisher ==========
            auto track_msg = visualization_msgs::msg::MarkerArray();
            auto marker_i = visualization_msgs::msg::Marker();
            double r_val_point = 0;
            double g_val_point = 0;
            double b_val_point = 0;

            for (size_t i=0; i<ref_points_global_.size(); i++) 
            {                
                marker_i.header.frame_id = "world";
                marker_i.header.stamp = this->now();
                // set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
                marker_i.type = 2;
                marker_i.id = (int)i;

                // Set the scale of the marker
                marker_i.scale.x = 0.1;
                marker_i.scale.y = 0.1;
                marker_i.scale.z = 0.1;

                // Set the color
                marker_i.color.r = r_val_point;
                marker_i.color.g = g_val_point;
                marker_i.color.b = b_val_point;
                marker_i.color.a = 1.0;

                // Set the pose of the marker
                marker_i.pose.position.x = ref_points_global_[i][0];
                marker_i.pose.position.y = ref_points_global_[i][1];
                marker_i.pose.position.z = 0;
                marker_i.pose.orientation.x = 0;
                marker_i.pose.orientation.y = 0;
                marker_i.pose.orientation.z = 0;
                marker_i.pose.orientation.w = 1;

                // append to marker array
                track_msg.markers.push_back(marker_i);
            }
            track_publisher_->publish(track_msg);

            
            // ======= Left/Blue Boundary publisher ==========
            
            auto left_bound_msg = visualization_msgs::msg::MarkerArray();

            r_val_point = 0;
            g_val_point = 0;
            b_val_point = 1.0;
            
            for (size_t i=0; i<trackbounds_left_.size(); i++)
            {
                marker_i.header.frame_id = "world";
                marker_i.header.stamp = this->now();
                // set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
                marker_i.type = 3;
                marker_i.id = (int)i;

                // Set the scale of the marker
                marker_i.scale.x = 0.25;
                marker_i.scale.y = 0.25;
                marker_i.scale.z = 0.3;

                // Set the color
                marker_i.color.r = r_val_point;
                marker_i.color.g = g_val_point;
                marker_i.color.b = b_val_point;
                marker_i.color.a = 1.0;

                // Set the pose of the marker
                marker_i.pose.position.x = trackbounds_left_[i][0];
                marker_i.pose.position.y = trackbounds_left_[i][1];
                marker_i.pose.position.z = 0;
                marker_i.pose.orientation.x = 0;
                marker_i.pose.orientation.y = 0;
                marker_i.pose.orientation.z = 0;
                marker_i.pose.orientation.w = 1;

                // append to marker array
                left_bound_msg.markers.push_back(marker_i);
            }
            trackbounds_left_publisher_->publish(left_bound_msg);


            // ======= Right/Yellow Boundary publisher ==========

            auto right_bound_msg = visualization_msgs::msg::MarkerArray();

            r_val_point = 0.9;
            g_val_point = 0.9;
            b_val_point = 0;
            
            for (size_t i=0; i<trackbounds_right_.size(); i++)
            {
                marker_i.header.frame_id = "world";
                marker_i.header.stamp = this->now();
                // set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
                marker_i.type = 3;
                marker_i.id = (int)i;

                // Set the scale of the marker
                marker_i.scale.x = 0.25;
                marker_i.scale.y = 0.25;
                marker_i.scale.z = 0.3;

                // Set the color
                marker_i.color.r = r_val_point;
                marker_i.color.g = g_val_point;
                marker_i.color.b = b_val_point;
                marker_i.color.a = 1.0;

                // Set the pose of the marker
                marker_i.pose.position.x = trackbounds_right_[i][0];
                marker_i.pose.position.y = trackbounds_right_[i][1];
                marker_i.pose.position.z = 0;
                marker_i.pose.orientation.x = 0;
                marker_i.pose.orientation.y = 0;
                marker_i.pose.orientation.z = 0;
                marker_i.pose.orientation.w = 1;

                // append to marker array
                right_bound_msg.markers.push_back(marker_i);
            }
            trackbounds_right_publisher_->publish(right_bound_msg);

            RCLCPP_DEBUG_STREAM(this->get_logger(), "Track callback ended.");
        }

        void ref_path_callback()
        {
            // This callback publishes the middle points ahead of the vehicle

            RCLCPP_DEBUG_STREAM(this->get_logger(), "Ref path callback started.");
            
            this->gamma_ = this->get_parameter("gamma").as_double();
            this->r_perception_max_ = this->get_parameter("r_perception_max").as_double();
            this->r_perception_min_ = this->get_parameter("r_perception_min").as_double();

            double x_c = x_[0];
            double y_c = x_[1];
            double psi = x_[2];

            double a_1_neg = sin(psi - gamma_);
            double a_2_neg = - cos(psi - gamma_);
            double b_neg = - a_1_neg * x_c - a_2_neg * y_c;

            double a_1_pos = sin(psi + gamma_);
            double a_2_pos = - cos(psi + gamma_);
            double b_pos = a_1_pos * x_c + a_2_pos * y_c;

            auto path_pos = sim_backend::msg::Point2D();

            double r_val_point = 1.0;
            double g_val_point = 0;
            double b_val_point = 0;

            auto ref_path_marker_msg = visualization_msgs::msg::MarkerArray();
            auto marker_i = visualization_msgs::msg::Marker();

            size_t idx = 0;
            bool first_visited = false;
            int marker_id = 0;

            for (size_t i = this->initial_idx_refloop_; i < (ref_points_global_.size() + this->initial_idx_refloop_); i++)
            {
                idx = i % ref_points_global_.size();
                path_pos.point_2d[0] = ref_points_global_[idx][0];
                path_pos.point_2d[1] = ref_points_global_[idx][1];

                RCLCPP_DEBUG_STREAM(this->get_logger(), "Ref path callback at track index " << idx << " with point (" << path_pos.point_2d[0] << ", " << path_pos.point_2d[1] << ").");

                // Check if global candidate point lies in the cone defined by the two "perception" halfspaces
                // And if global point lies inside a "perception donut"
                if(((a_1_neg * path_pos.point_2d[0] + a_2_neg * path_pos.point_2d[1] <= b_neg) && 
                    (a_1_pos * path_pos.point_2d[0] + a_2_pos * path_pos.point_2d[1] >= b_pos) &&
                    (sqrt(pow(path_pos.point_2d[0] - x_c, 2) + pow(path_pos.point_2d[1] - y_c, 2)) <= this->r_perception_max_)) ||
                    (sqrt(pow(path_pos.point_2d[0] - x_c, 2) + pow(path_pos.point_2d[1] - y_c, 2)) <= this->r_perception_min_))
                {
                    marker_i.header.frame_id = "world";
                    marker_i.header.stamp = this->now();
                    // set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
                    marker_i.type = 2;
                    marker_i.id = (int) idx;
                    // Set the scale of the marker
                    marker_i.scale.x = 0.18;
                    marker_i.scale.y = 0.18;
                    marker_i.scale.z = 0.18;
                    // Set the color
                    marker_i.color.r = r_val_point;
                    marker_i.color.g = g_val_point;
                    marker_i.color.b = b_val_point;
                    marker_i.color.a = 0.8;

                    // Set the pose of the marker
                    marker_i.pose.position.x = ref_points_global_[idx][0];
                    marker_i.pose.position.y = ref_points_global_[idx][1];
                    marker_i.pose.position.z = 0;
                    marker_i.pose.orientation.x = 0;
                    marker_i.pose.orientation.y = 0;
                    marker_i.pose.orientation.z = 0;
                    marker_i.pose.orientation.w = 1;

                    // append to marker array
                    ref_path_marker_msg.markers.push_back(marker_i);

                    marker_id++;
                    
                    if (!first_visited){
                        first_visited = true;
                        this->initial_idx_refloop_ = idx;
                    }
                }
            }

            ref_path_publisher_marker_array_->publish(ref_path_marker_msg);

        }

        // Timers
        rclcpp::TimerBase::SharedPtr solve_timer_;
        rclcpp::TimerBase::SharedPtr trackpub_timer_;

        // Publisher for vehicle state
        rclcpp::Publisher<sim_backend::msg::VehicleState>::SharedPtr state_publisher_;
        rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr velocity_vector_publisher_;
        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr ref_path_publisher_marker_array_;

        // Publisher for track
        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr track_publisher_;
        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr trackbounds_left_publisher_;
        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr trackbounds_right_publisher_;

        // Subscriber to update input signals
        rclcpp::Subscription<sim_backend::msg::SysInput>::SharedPtr input_subscription_;

        // Subscriber to reset message
        rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr reset_subscription_;

        // Cycle time of Simulation node
        std::chrono::milliseconds dt_{std::chrono::milliseconds(5)};
        std::chrono::milliseconds dt_trackpub_{std::chrono::milliseconds(5)};
        double dt_seconds_;

        // Maximum step time for ODE solver
        double max_dt_ = 1e-3;

        // Time scalar that gets updated each time step
        double t_;
        double start_time_ns_;

        // "Perception Cone angle"
        double gamma_;
        double r_perception_max_;
        double r_perception_min_;
        size_t initial_idx_refloop_;

        // State vector of dynamic system
        state_type x_{state_type(10)};

        // Init System
        DynamicSystem sys_;

        // filepaths
        std::string track_fpath_midline_;
        std::string track_fpath_leftbound_;
        std::string track_fpath_rightbound_;

        // Global reference path
        std::vector<std::vector<double>> ref_points_global_;
        // Local reference path
        std::vector<std::vector<double>> ref_points_local_;

        // Track bounds
        std::vector<std::vector<double>> trackbounds_left_;
        std::vector<std::vector<double>> trackbounds_right_;
    
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