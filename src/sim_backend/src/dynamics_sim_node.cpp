/*
 * Simple Vehicle Dynamics Simulator Project
 *
 * Copyright (c) 2023-2024 Authors:
 *   - Jonas Ohnemus <johnemus@ethz.ch>
 *
 * All rights reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */

#include <boost/numeric/odeint.hpp>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include "rclcpp/rclcpp.hpp"
#include "sim_backend/msg/point2_d.hpp"
#include "sim_backend/msg/point2_d_array.hpp"
#include "sim_backend/msg/sys_input.hpp"
#include "sim_backend/msg/vehicle_state.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/empty.hpp"
#include "std_msgs/msg/float64.hpp"
#include "std_msgs/msg/int32.hpp"

#include "sim_backend/dynamic_system.hpp"

using namespace std::chrono_literals;
using namespace boost::numeric::odeint;

class DynamicsSimulator : public rclcpp::Node {
public:
  DynamicsSimulator()
      : Node("dynamics_simulator",
             rclcpp::NodeOptions()
                 .allow_undeclared_parameters(true)
                 .automatically_declare_parameters_from_overrides(true)) {
    track_fpath_midline_ =
        this->get_parameter("track_filepath_midline").as_string();
    track_fpath_leftbound_ =
        this->get_parameter("track_filepath_leftbound").as_string();
    track_fpath_rightbound_ =
        this->get_parameter("track_filepath_rightbound").as_string();

    if (this->get_csv_ref_track()) {
      RCLCPP_ERROR_STREAM(this->get_logger(),
                          "Something went wrong reading CSV ref points file!");
      rclcpp::shutdown();
    }
    print_global_refpoints();

    // State of Simulation

    // X_C_I
    x_[0] = 0.0;
    // Y_C_I
    x_[1] = 0.0;
    // psi
    x_[2] = 0.0;

    // dX_cdt in vehicle frame
    x_[3] = 0.0;
    // dY_cdt in vehicle frame
    x_[4] = 0.0;
    // dpsidt
    x_[5] = 0.0;

    // del_s_act
    x_[6] = 0.0;

    // omega_wheel_fl
    x_[7] = 0.0;
    // omega_wheel_fr
    x_[8] = 0.0;
    // omega_wheel_rl
    x_[9] = 0.0;
    // omega_wheel_rr
    x_[10] = 0.0;

    // ==============
    // T_mot_fl_act
    x_[11] = 0.0;
    // dT_mot_fl_act
    x_[12] = 0.0;

    // T_mot_fr_act
    x_[13] = 0.0;
    // dT_mot_fr_act
    x_[14] = 0.0;

    // T_mot_rl_act
    x_[15] = 0.0;
    // dT_mot_rl_act
    x_[16] = 0.0;

    // T_mot_rr_act
    x_[17] = 0.0;
    // dT_mot_rr_act
    x_[18] = 0.0;

    // ===============

    // Fx_fl_act
    x_[19] = 0.0;
    // Fx_fr_act
    x_[20] = 0.0;
    // Fx_rl_act
    x_[21] = 0.0;
    // Fx_rr_act
    x_[22] = 0.0;
    
    // Fy_fl_act
    x_[23] = 0.0;
    // Fy_fr_act
    x_[24] = 0.0;
    // Fy_rl_act
    x_[25] = 0.0;
    // Fy_rr_act
    x_[26] = 0.0;
    

    // Get perception parameters from parameter server
    this->gamma_ = this->get_parameter("gamma").as_double();
    this->r_perception_max_ =
        this->get_parameter("r_perception_max").as_double();
    this->r_perception_min_ =
        this->get_parameter("r_perception_min").as_double();

    // Set initial index of reference path to 0
    this->initial_idx_refloop_ = 0;

    // Create Dynamic System and update parameters
    sys_ = DynamicSystem();

    param_struct_.l_f = this->get_parameter("l_f").as_double();
    param_struct_.l_r = this->get_parameter("l_r").as_double();
    param_struct_.wb_f = this->get_parameter("wb_f").as_double();
    param_struct_.wb_r = this->get_parameter("wb_r").as_double();
    param_struct_.h_cg = this->get_parameter("h_cg").as_double();
    param_struct_.r_wheel = this->get_parameter("r_wheel").as_double();

    param_struct_.m = this->get_parameter("m").as_double();
    param_struct_.Iz = this->get_parameter("Iz").as_double();
    param_struct_.g = this->get_parameter("g").as_double();
    param_struct_.Iwheel = this->get_parameter("Iwheel").as_double();

    param_struct_.D_tire_lon = this->get_parameter("D_tire_lon").as_double();
    param_struct_.C_tire_lon = this->get_parameter("C_tire_lon").as_double();
    param_struct_.B_tire_lon = this->get_parameter("B_tire_lon").as_double();
    param_struct_.tau_tire_x = this->get_parameter("tau_tire_x").as_double();

    param_struct_.D_tire_lat = this->get_parameter("D_tire_lat").as_double();
    param_struct_.C_tire_lat = this->get_parameter("C_tire_lat").as_double();
    param_struct_.B_tire_lat = this->get_parameter("B_tire_lat").as_double();
    param_struct_.tau_tire_y = this->get_parameter("tau_tire_y").as_double();

    param_struct_.C_d = this->get_parameter("C_d").as_double();
    param_struct_.C_r = this->get_parameter("C_r").as_double();
    param_struct_.C_l = this->get_parameter("C_l").as_double();

    param_struct_.tau_mot = this->get_parameter("tau_mot").as_double();
    param_struct_.D_mot = this->get_parameter("D_mot").as_double();

    param_struct_.iG = this->get_parameter("iG").as_double();
    param_struct_.tau_steer = this->get_parameter("tau_steer").as_double();

    param_struct_.delta_s_max = this->get_parameter("delta_s_max").as_double();
    param_struct_.delta_s_min = this->get_parameter("delta_s_min").as_double();
    param_struct_.T_mot_max = this->get_parameter("T_mot_max").as_double();
    param_struct_.T_mot_min = this->get_parameter("T_mot_min").as_double();

    sys_.update_parameters(param_struct_);

    // Time step for ODE solver in seconds
    dt_seconds_ = dt_.count() / 1e3;
    // Absolute Simulation time in seconds
    t_ = 0.00001;

    // Publisher for vehicle state
    state_publisher_ = this->create_publisher<sim_backend::msg::VehicleState>(
        "vehicle_state", 1);

    // Point2D Array Publishers
    track_publisher_ = this->create_publisher<sim_backend::msg::Point2DArray>(
        "track_points2d", 1);
    trackbounds_left_publisher_ =
        this->create_publisher<sim_backend::msg::Point2DArray>(
            "trackbounds_left_points2d", 1);
    trackbounds_right_publisher_ =
        this->create_publisher<sim_backend::msg::Point2DArray>(
            "trackbounds_right_points2d", 1);
    ref_path_publisher_ =
        this->create_publisher<sim_backend::msg::Point2DArray>(
            "reference_path_points2d", 1);

    // number of cones hit publisher
    num_cones_hit_publisher_ =
        this->create_publisher<std_msgs::msg::Int32>("num_cones_hit", 1);

    // lap count publisher
    lap_count_publisher_ =
        this->create_publisher<std_msgs::msg::Int32>("lap_count", 1);
    // lap time publisher
    lap_time_publisher_ =
        this->create_publisher<std_msgs::msg::Float64>("lap_time", 1);
    // last lap time publisher
    last_lap_time_publisher_ =
        this->create_publisher<std_msgs::msg::Float64>("last_lap_time", 1);

    // Publisher for controllers
    // start controller: empty message
    start_controller_publisher_ =
        this->create_publisher<std_msgs::msg::Empty>("start_controller", 1);
    // reset controller: empty message
    reset_controller_publisher_ =
        this->create_publisher<std_msgs::msg::Empty>("reset_controller", 1);
    // stop controller: empty message
    stop_controller_publisher_ =
        this->create_publisher<std_msgs::msg::Empty>("stop_controller", 1);

    // Timers for solve step and track publisher
    solve_timer_ =
        rclcpp::create_timer(this, this->get_clock(), this->dt_,
                             std::bind(&DynamicsSimulator::solve_step, this));
    trackpub_timer_ = rclcpp::create_timer(
        this, this->get_clock(), this->dt_trackpub_,
        std::bind(&DynamicsSimulator::track_callback, this));

    // Subscriber for input signals and reset message
    input_subscription_ = this->create_subscription<sim_backend::msg::SysInput>(
        "vehicle_input", 1,
        std::bind(&DynamicsSimulator::update_input, this,
                  std::placeholders::_1));
    reset_subscription_ = this->create_subscription<std_msgs::msg::Empty>(
        "reset_sim", 1,
        std::bind(&DynamicsSimulator::reset_simulator, this,
                  std::placeholders::_1));

    // subscriber to toggle_sim_input_acceptance to accept control inputs or not
    toggle_sim_input_acceptance_sub_ =
        this->create_subscription<std_msgs::msg::Empty>(
            "toggle_sim_input_acceptance", 1,
            std::bind(&DynamicsSimulator::toggle_sim_input_acceptance, this,
                      std::placeholders::_1));

    RCLCPP_INFO_STREAM(this->get_logger(),
                       "Node " << this->get_name() << " initialized.");
  }

private:
  void toggle_sim_input_acceptance(const std_msgs::msg::Empty &msg) {
    RCLCPP_INFO_STREAM(this->get_logger(), "Toggling input acceptance.");
    this->accept_inputs_ = !this->accept_inputs_;
  }

  void reset_simulator(const std_msgs::msg::Empty &msg) {
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
    x_[10] = 0.0;
    x_[11] = 0.0;
    x_[12] = 0.0;
    x_[13] = 0.0;
    x_[14] = 0.0;
    x_[15] = 0.0;
    x_[16] = 0.0;
    x_[17] = 0.0;
    x_[18] = 0.0;
    x_[19] = 0.0;
    x_[20] = 0.0;
    x_[21] = 0.0;
    x_[22] = 0.0;
    x_[23] = 0.0;
    x_[24] = 0.0;
    x_[25] = 0.0;
    x_[26] = 0.0;
    
    this->initial_idx_refloop_ = 0;
    num_cones_hit_ = 0;
    for (size_t i = 0; i < cones_hit_count_left_.size(); i++) {
      cones_hit_count_left_[i] = 0;
      cones_currently_hit_left_[i] = false;
    }
    for (size_t i = 0; i < cones_hit_count_right_.size(); i++) {
      cones_hit_count_right_[i] = 0;
      cones_currently_hit_right_[i] = false;
    }
    t_ = 0.00001;
    lap_count_ = 0;
    lap_time_start_ = 0.0;
    lap_time_ = 0.0;
    last_lap_time_ = 0.0;

    RCLCPP_DEBUG_STREAM(this->get_logger(), "Sim Reset.");
  }

  void update_input(const sim_backend::msg::SysInput &msg) {
    RCLCPP_DEBUG_STREAM(this->get_logger(),
                        "Received Input Tm_fl: " << msg.t_m_fl << ""
                    << ", Tm_fr: " << msg.t_m_fr
                    << ", Tm_rl: " << msg.t_m_rl
                    << ", Tm_rr: " << msg.t_m_rr
                    << ", delta_s: " << msg.del_s);
    if (this->accept_inputs_) {
      this->sys_.update_inputs(msg.t_m_fl, msg.t_m_fr, msg.t_m_rl, msg.t_m_rr, msg.del_s);
    } else {
      this->sys_.update_inputs(0.0, 0.0, 0.0, 0.0, 0.0);
    }
  }

  int get_csv_ref_track() {
    std::ifstream data_middle(track_fpath_midline_);
    std::string line;
    while (std::getline(data_middle, line)) {
      std::stringstream lineStream(line);
      std::string cell;
      std::vector<double> parsedRow;
      while (std::getline(lineStream, cell, ',')) {
        parsedRow.push_back(std::stod(cell));
      }
      ref_points_global_.push_back(parsedRow);
    }

    // left cones are blue
    std::ifstream data_blue(track_fpath_leftbound_);
    while (std::getline(data_blue, line)) {
      std::stringstream lineStream(line);
      std::string cell;
      std::vector<double> parsedRow;
      while (std::getline(lineStream, cell, ',')) {
        parsedRow.push_back(std::stod(cell));
      }
      trackbounds_left_.push_back(parsedRow);
      cones_hit_count_left_.push_back(0);
      cones_currently_hit_left_.push_back(false);
    }

    // right cones are yellow
    std::ifstream data_yellow(track_fpath_rightbound_);
    while (std::getline(data_yellow, line)) {
      std::stringstream lineStream(line);
      std::string cell;
      std::vector<double> parsedRow;
      while (std::getline(lineStream, cell, ',')) {
        parsedRow.push_back(std::stod(cell));
      }
      trackbounds_right_.push_back(parsedRow);
      cones_hit_count_right_.push_back(0);
      cones_currently_hit_right_.push_back(false);
    }

    return 0;
  }

  void print_global_refpoints() {
    RCLCPP_DEBUG_STREAM(this->get_logger(),
                        "=== \nGot 2D Points (reference) as an array:");
    for (size_t i = 0; i < ref_points_global_.size(); i++) {
      RCLCPP_DEBUG_STREAM(this->get_logger(),
                          "x: " << ref_points_global_[i][0]
                                << ", y: " << ref_points_global_[i][1]);
    }
    RCLCPP_DEBUG_STREAM(this->get_logger(), "===");
  }

  void updateAllConeCollisions() {
    // RCLCPP_DEBUG_STREAM(this->get_logger(), "Updating all cone collisions.");
    // create a bounding box around the car
    double x_c = x_[0];
    double y_c = x_[1];
    double psi = x_[2];

    // create 4 line segments that represent the car's bounding box
    // RCLCPP_DEBUG_STREAM(this->get_logger(), "Creating car bounding box.");
    std::vector<std::vector<double>> car_bbox;
    car_bbox.push_back({x_c + l_to_front_ * cos(psi) + l_to_right_ * sin(psi),
                        y_c + l_to_front_ * sin(psi) - l_to_right_ * cos(psi)});
    car_bbox.push_back({x_c + l_to_front_ * cos(psi) - l_to_left_ * sin(psi),
                        y_c + l_to_front_ * sin(psi) + l_to_left_ * cos(psi)});
    car_bbox.push_back({x_c - l_to_rear_ * cos(psi) - l_to_left_ * sin(psi),
                        y_c - l_to_rear_ * sin(psi) + l_to_left_ * cos(psi)});
    car_bbox.push_back({x_c - l_to_rear_ * cos(psi) + l_to_right_ * sin(psi),
                        y_c - l_to_rear_ * sin(psi) - l_to_right_ * cos(psi)});

    // iterate through both left and right cone arrays and check for
    // intersection with car bbox if the current cone is not in collision with
    // the car, it is counted as colliding with the car if the current cone is
    // in collision with the car, it is not counted as colliding with the car to
    // avoid double counting
    // RCLCPP_DEBUG_STREAM(this->get_logger(), "Iterating through cones left.");
    for (size_t i = 0; i < trackbounds_left_.size(); i++) {
      if (coneWithinBoundingBox(car_bbox, trackbounds_left_[i])) {
        if (!cones_currently_hit_left_[i]) {
          cones_currently_hit_left_[i] = true;
          cones_hit_count_left_[i]++;
        }
      } else {
        cones_currently_hit_left_[i] = false;
      }
    }
    // RCLCPP_DEBUG_STREAM(this->get_logger(), "Iterating through cones right.");
    for (size_t i = 0; i < trackbounds_right_.size(); i++) {
      if (coneWithinBoundingBox(car_bbox, trackbounds_right_[i])) {
        if (!cones_currently_hit_right_[i]) {
          cones_currently_hit_right_[i] = true;
          cones_hit_count_right_[i]++;
        }
      } else {
        cones_currently_hit_right_[i] = false;
      }
    }
    // RCLCPP_DEBUG_STREAM(this->get_logger(), "Updated all cone collisions.");
  }

  bool coneWithinBoundingBox(std::vector<std::vector<double>> bbox,
                             std::vector<double> cone) {
    // RCLCPP_DEBUG_STREAM(this->get_logger(), "Checking if cone is within bounding box.");
    // Check if the cone is within the bounding box defined by the four points
    // in counterclockwise order
    double x = cone[0];
    double y = cone[1];

    // Iterate through the four line segments of the bounding box
    for (size_t i = 0; i < bbox.size(); i++) {
      double x1 = bbox[i][0];
      double y1 = bbox[i][1];
      double x2 = bbox[(i + 1) % bbox.size()][0];
      double y2 = bbox[(i + 1) % bbox.size()][1];

      // Calculate the cross product between the vectors (x2-x1, y2-y1) and
      // (x-x1, y-y1)
      double crossProduct = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1);

      // If the cross product is negative, the point is outside the bounding box
      if (crossProduct < 0) {
        // RCLCPP_DEBUG_STREAM(this->get_logger(), "Cone is outside bounding box.");
        return false;
      }
    }
    // RCLCPP_DEBUG_STREAM(this->get_logger(), "Cone is within bounding box.");

    // If the point is inside all four line segments, it is within the bounding
    // box
    return true;
  }

  void count_cones_hit() {
    // RCLCPP_DEBUG_STREAM(this->get_logger(), "Counting cones hit.");
    num_cones_hit_ = 0;
    // iterate through both left and right cone arrays and check for
    // intersection with car bbox
    for (size_t i = 0; i < trackbounds_left_.size(); i++) {
      num_cones_hit_ += cones_hit_count_left_[i];
    }
    for (size_t i = 0; i < trackbounds_right_.size(); i++) {
      num_cones_hit_ += cones_hit_count_right_[i];
    }
    // RCLCPP_DEBUG_STREAM(this->get_logger(),
                        // "Number of cones hit: " << num_cones_hit_);
  }

  bool carCrossesStartFinishLine() {
    // RCLCPP_DEBUG_STREAM(this->get_logger(), "Checking if car crosses start/finish line.");
    // check if car has crossed the start/finish line, given as the fourth line
    // segment of ref_points_global_
    double x_c = x_[0];
    double y_c = x_[1];
    double psi = x_[2];

    // start/finish line as a line segment perpendicular to the segment from
    // point 4 to point 5 on the global reference path, going through the fifth
    // point
    double x1 = ref_points_global_[4][0];
    double y1 = ref_points_global_[4][1];
    double x2 = ref_points_global_[5][0];
    double y2 = ref_points_global_[5][1];
    // orthogonal segment to the line segment from point 4 to point 5
    double dx_norm = (x2 - x1) / sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
    double dy_norm = (y2 - y1) / sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
    double x3 = x2 + trackwidth_ * dy_norm;
    double y3 = y2 - trackwidth_ * dx_norm;
    double x4 = x2 - trackwidth_ * dy_norm;
    double y4 = y2 + trackwidth_ * dx_norm;

    // create bounding box
    std::vector<std::vector<double>> car_bbox;
    car_bbox.push_back({x_c + l_to_front_ * cos(psi) + l_to_right_ * sin(psi),
                        y_c + l_to_front_ * sin(psi) - l_to_right_ * cos(psi)});
    car_bbox.push_back({x_c + l_to_front_ * cos(psi) - l_to_left_ * sin(psi),
                        y_c + l_to_front_ * sin(psi) + l_to_left_ * cos(psi)});
    car_bbox.push_back({x_c - l_to_rear_ * cos(psi) - l_to_left_ * sin(psi),
                        y_c - l_to_rear_ * sin(psi) + l_to_left_ * cos(psi)});
    car_bbox.push_back({x_c - l_to_rear_ * cos(psi) + l_to_right_ * sin(psi),
                        y_c - l_to_rear_ * sin(psi) - l_to_right_ * cos(psi)});

    // check if bounding box collides with start/finish line
    for (size_t i = 0; i < car_bbox.size(); i++) {
      if (doIntersect({x3, y3}, {x4, y4}, car_bbox[i],
                      car_bbox[(i + 1) % car_bbox.size()])) {
        // RCLCPP_DEBUG_STREAM(this->get_logger(),
                          //  "Car currently crosses start/finish line!");

        return true;
      }
    }
    // RCLCPP_DEBUG_STREAM(this->get_logger(),
                      //  "Car currently does not cross start/finish line.");
    return false;
  }

  bool doIntersect(std::vector<double> p1, std::vector<double> q1,
                   std::vector<double> p2, std::vector<double> q2) {
    // RCLCPP_DEBUG_STREAM(this->get_logger(),
                      //  "Checking intersection between car bbox and start/finish line.");
    // Find the four orientations needed for general and special cases
    int o1 = orientation(p1, q1, p2);
    int o2 = orientation(p1, q1, q2);
    int o3 = orientation(p2, q2, p1);
    int o4 = orientation(p2, q2, q1);

    // General case
    if (o1 != o2 && o3 != o4) {
      return true;
    }

    // Special Cases
    // p1, q1 and p2 are colinear and p2 lies on segment p1q1
    if (o1 == 0 && onSegment(p1, p2, q1)) {
      return true;
    }

    // p1, q1 and q2 are colinear and q2 lies on segment p1q1
    if (o2 == 0 && onSegment(p1, q2, q1)) {
      return true;
    }

    // p2, q2 and p1 are colinear and p1 lies on segment p2q2
    if (o3 == 0 && onSegment(p2, p1, q2)) {
      return true;
    }

    // p2, q2 and q1 are colinear and q1 lies on segment p2q2
    if (o4 == 0 && onSegment(p2, q1, q2)) {
      return true;
    }
    // RCLCPP_DEBUG_STREAM(this->get_logger(),
    //                    "No intersection between car bbox and start/finish line.");
    return false; // Doesn't fall in any of the above cases
  }

  bool onSegment(std::vector<double> p, std::vector<double> q,
                 std::vector<double> r) {
    // RCLCPP_DEBUG_STREAM(this->get_logger(),
    //                    "Checking if point lies on segment.");
    if (q[0] <= std::max(p[0], r[0]) && q[0] >= std::min(p[0], r[0]) &&
        q[1] <= std::max(p[1], r[1]) && q[1] >= std::min(p[1], r[1])) {
      // RCLCPP_DEBUG_STREAM(this->get_logger(), "Point lies on segment.");
      return true;
    }
    // RCLCPP_DEBUG_STREAM(this->get_logger(), "Point does not lie on segment.");
    return false;
  }

  int orientation(std::vector<double> p, std::vector<double> q,
                  std::vector<double> r) {
    // RCLCPP_DEBUG_STREAM(this->get_logger(), "Checking orientation of points.");
    // See https://www.geeksforgeeks.org/orientation-3-ordered-points/
    // for details of below formula.
    double val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1]);

    if (val == 0) {
      // RCLCPP_DEBUG_STREAM(this->get_logger(), "Points are colinear.");
      return 0; // colinear
    }
    // RCLCPP_DEBUG_STREAM(this->get_logger(), "Points are not colinear.");
    return (val > 0) ? 1 : 2; // clock or counterclock wise
  }

  void solve_step() {
    RCLCPP_DEBUG_STREAM(this->get_logger(), "Solve step started.");
    /* Get start time of solve step */
    double t0 = (double_t)(this->now().nanoseconds()); // [ms]

    RCLCPP_DEBUG_STREAM(this->get_logger(),
                        "Step began at time t_ = " << t_ << " s.");

    RCLCPP_DEBUG_STREAM(this->get_logger(),
                        "State: " << x_[0] << ", " << x_[1] << ", "
                        << x_[2] << ", " << x_[3] << ", " << x_[4] << ", "
                        << x_[5] << ", " << x_[6] << ", " << x_[7] << ", "
                        << x_[8] << ", " << x_[9] << ", " << x_[10] << ", "
                        << x_[11] << ", " << x_[12] << ", " << x_[13] << ", "
                        << x_[14] << ", " << x_[15] << ", " << x_[16] << ", "
                        << x_[17] << ", " << x_[18] << ", " << x_[19] << ", "
                        << x_[20] << ", " << x_[21] << ", " << x_[22] << ", "
                        << x_[23] << ", " << x_[24] << ", " << x_[25] << ", "
                        << x_[26]);

    double adaptive_dt = 1e-4;
    size_t steps = integrate_adaptive(stepper_, sys_, x_, t_, t_ + dt_seconds_,
                                      adaptive_dt);
    // stepper_uncontrolled_.do_step(sys_, x_, t_, dt_seconds_);
    // size_t steps = 1;
    t_ = t_ + dt_seconds_;
    RCLCPP_DEBUG_STREAM(this->get_logger(),
                        "Solver produced a result integrating "
                            << steps << " step(s) forward.");

    auto state_msg = sim_backend::msg::VehicleState();
    /* x = [xc_I, yc_I, psi, dxc_V, dyc_V, dpsi, fx_f, dfx_f, fx_r, dfx_r] */
    state_msg.x_c = x_[0];
    state_msg.y_c = x_[1];
    state_msg.psi = x_[2];
    state_msg.dx_c =
        x_[3] * cos(x_[2]) - x_[4] * sin(x_[2]); // rotated into global frame
    state_msg.dy_c = x_[3] * sin(x_[2]) + x_[4] * cos(x_[2]);
    state_msg.dx_c_v = x_[3];
    state_msg.dy_c_v = x_[4];
    state_msg.dpsi = x_[5];
    
    state_msg.del_s_act = x_[6];
    state_msg.omega_wheel_fl = x_[7];
    state_msg.omega_wheel_fr = x_[8];
    state_msg.omega_wheel_rl = x_[9];
    state_msg.omega_wheel_rr = x_[10];

    state_msg.t_mot_fl_act = x_[11];
    state_msg.dt_mot_fl_act = x_[12];
    state_msg.t_mot_fr_act = x_[13];
    state_msg.dt_mot_fr_act = x_[14];
    state_msg.t_mot_rl_act = x_[15];
    state_msg.dt_mot_rl_act = x_[16];
    state_msg.t_mot_rr_act = x_[17];
    state_msg.dt_mot_rr_act = x_[18];

    state_msg.fx_fl = x_[19];
    state_msg.fx_fr = x_[20];
    state_msg.fx_rl = x_[21];
    state_msg.fx_rr = x_[22];
    state_msg.fy_fl = x_[23];
    state_msg.fy_fr = x_[24];
    state_msg.fy_rl = x_[25];
    state_msg.fy_rr = x_[26];

    state_msg.ax_c_v = ((x_[19] + x_[20]) * cos(x_[6])
          + (x_[21] + x_[21]) 
          - (x_[23] + x_[24]) * sin(x_[6]) 
          - tanh(x_[3] / 1e-6) * (param_struct_.C_d * pow(x_[3], 2))
          - tanh(x_[3] / 1e-6) * (param_struct_.C_r * param_struct_.m * param_struct_.g)) / param_struct_.m 
          + x_[3] * x_[5];
    state_msg.ay_c_v = (x_[25] + x_[26] 
          + (x_[23] + x_[24]) * cos(x_[6])
          + (x_[19] + x_[20]) * sin(x_[6])) / param_struct_.m 
          - x_[3] * x_[5];

    double t_mot_fl = 0.0;
    double t_mot_fr = 0.0;
    double t_mot_rl = 0.0;
    double t_mot_rr = 0.0;
    double del_s = 0.0;

    this->sys_.get_inputs(t_mot_fl, t_mot_fr, t_mot_rl, t_mot_rr, del_s);

    state_msg.del_s_ref = del_s;
    state_msg.t_mot_fl_ref = t_mot_fl;
    state_msg.t_mot_fr_ref = t_mot_fr;
    state_msg.t_mot_rl_ref = t_mot_rl;
    state_msg.t_mot_rr_ref = t_mot_rr;

    state_publisher_->publish(state_msg);

    this->ref_path_callback();

    // check update all cone collisions
    // RCLCPP_DEBUG_STREAM(this->get_logger(), "Checking for cone collisions.");
    this->updateAllConeCollisions();

    // check if car is on start/finish line and previously was not
    // RCLCPP_DEBUG_STREAM(this->get_logger(), "Checking if car is on start/finish line.");

    if (this->carCrossesStartFinishLine() && !car_on_start_finish_line_) {
      car_on_start_finish_line_ = true;

      // check if car crosses start/finish line for the first time
      if (lap_time_start_ == 0.0) {
        // This is the case for when the car crosses the start/finish line for
        // the first time
        RCLCPP_INFO_STREAM(this->get_logger(),
                           "Car crossed start/finish line for the first time!");
        // definitely first lap
        lap_count_ = 0;

        // set last lap time to zero for first lap
        last_lap_time_ = 0.0;

        // reset lap time
        lap_time_ = 0.0;
      } else {
        // car crossed start/finish line after the first lap
        RCLCPP_INFO_STREAM(this->get_logger(),
                           "Car crossed start/finish line!");
        RCLCPP_INFO_STREAM(this->get_logger(),
                           "Lap time: " << lap_time_ << " s");
        RCLCPP_INFO_STREAM(this->get_logger(), "Lap count: " << lap_count_);

        // increment lap count
        lap_count_++;

        // set last lap time to current lap time
        last_lap_time_ = lap_time_;

        // reset lap time
        lap_time_ = 0.0;

        auto last_lap_time_msg = std_msgs::msg::Float64();
        last_lap_time_msg.data = last_lap_time_;
        last_lap_time_publisher_->publish(last_lap_time_msg);
      }

      lap_time_start_ = t_;

    } else if (!carCrossesStartFinishLine()) {
      car_on_start_finish_line_ = false;
    }

    lap_time_ = t_ - lap_time_start_;

    auto lap_count_msg = std_msgs::msg::Int32();
    lap_count_msg.data = lap_count_;
    // RCLCPP_DEBUG_STREAM(this->get_logger(), "Publishing lap count: " << lap_count_);
    lap_count_publisher_->publish(lap_count_msg);

    auto lap_time_msg = std_msgs::msg::Float64();
    lap_time_msg.data = lap_time_;
    // RCLCPP_DEBUG_STREAM(this->get_logger(), "Publishing lap time: " << lap_time_);
    lap_time_publisher_->publish(lap_time_msg);

    // count cones hit
    // RCLCPP_DEBUG_STREAM(this->get_logger(), "Counting cones hit.");
    count_cones_hit();
    // publish number of cones hit
    auto num_cones_hit_msg = std_msgs::msg::Int32();
    num_cones_hit_msg.data = num_cones_hit_;
    // RCLCPP_DEBUG_STREAM(this->get_logger(), "Publishing number of cones hit: " << num_cones_hit_);
    num_cones_hit_publisher_->publish(num_cones_hit_msg);

    /* Get end time of solve step */
    double t1 = (double_t)(this->now().nanoseconds()); // [ms]
    // RCLCPP_DEBUG_STREAM(this->get_logger(),
                        // "Time needed for step: " << t1 - t0
                        //                          << " ms. \nSolver did "
                        //                          << steps << " step(s).");

    if (t1 - t0 > 1e6 * dt_.count()) {
      RCLCPP_ERROR_STREAM(this->get_logger(),
                          "Needed too long for solver step! Took: "
                              << (t1 - t0) / 1e6 << " ms, but dt is "
                              << dt_.count() << " ms!");
    }
    // RCLCPP_DEBUG_STREAM(this->get_logger(),
                        // "Step ended at time t_ = " << t_ << " s.");

    // RCLCPP_DEBUG_STREAM(this->get_logger(), "Solve step end.");
  }

  void track_callback() {
    // This callback publishes the left cones, the right cones,
    // and the middle points for the whole track (for visualization)

    // RCLCPP_DEBUG_STREAM(this->get_logger(), "Track callback started.");

    // ======= Track publisher ==========
    auto track_msg = sim_backend::msg::Point2DArray();
    auto point2d_i = sim_backend::msg::Point2D();

    for (size_t i = 0; i < ref_points_global_.size(); i++) {
      // get track midpoint coordinates
      point2d_i.x = ref_points_global_[i][0];
      point2d_i.y = ref_points_global_[i][1];
      point2d_i.id = uint32_t(i);

      // append to marker array
      track_msg.points.push_back(point2d_i);
    }
    track_publisher_->publish(track_msg);

    // ======= Left/Blue Boundary publisher ==========
    auto left_bound_msg = sim_backend::msg::Point2DArray();

    for (size_t i = 0; i < trackbounds_left_.size(); i++) {
      // get left/blue boundary point coordinates
      point2d_i.x = trackbounds_left_[i][0];
      point2d_i.y = trackbounds_left_[i][1];
      point2d_i.id = uint32_t(i);

      // append to marker array
      left_bound_msg.points.push_back(point2d_i);
    }
    trackbounds_left_publisher_->publish(left_bound_msg);

    // ======= Right/Yellow Boundary publisher ==========
    auto right_bound_msg = sim_backend::msg::Point2DArray();

    for (size_t i = 0; i < trackbounds_right_.size(); i++) {
      // get right/yellow boundary point coordinates
      point2d_i.x = trackbounds_right_[i][0];
      point2d_i.y = trackbounds_right_[i][1];
      point2d_i.id = uint32_t(i);

      // append to marker array
      right_bound_msg.points.push_back(point2d_i);
    }
    trackbounds_right_publisher_->publish(right_bound_msg);

    // RCLCPP_DEBUG_STREAM(this->get_logger(), "Track callback ended.");
  }

  void ref_path_callback() {
    // This callback publishes the middle points ahead of the vehicle
    // (defined by a minimum and maximum perception radius as well as a
    // perception angle gamma) As well as the previous 5 points of the
    // respective reference path

    // RCLCPP_DEBUG_STREAM(this->get_logger(), "Ref path callback started.");

    this->gamma_ = this->get_parameter("gamma").as_double();
    this->r_perception_max_ =
        this->get_parameter("r_perception_max").as_double();
    this->r_perception_min_ =
        this->get_parameter("r_perception_min").as_double();

    // current geometric state of the 2D vehicle model
    double x_c = x_[0];
    double y_c = x_[1];
    double psi = x_[2];

    // Necessary message container variables
    auto point2d_i = sim_backend::msg::Point2D();
    auto ref_path_msg = sim_backend::msg::Point2DArray();

    point2d_i.x = x_c;
    point2d_i.y = y_c;
    point2d_i.id = uint32_t(0);
    ref_path_msg.points.push_back(point2d_i);

    // temporary variables to store point positions
    double x_pos = 0.0;
    double y_pos = 0.0;

    // index and identifier variables
    size_t idx = 0;
    int marker_id = 1;
    bool first_visited = false;

    int idx_temp = 0;

    for (size_t i = this->initial_idx_refloop_;
         i < (ref_points_global_.size() + this->initial_idx_refloop_); i++) {
      auto point2d_i = sim_backend::msg::Point2D();

      idx = i % ref_points_global_.size();
      x_pos = ref_points_global_[idx][0];
      y_pos = ref_points_global_[idx][1];

      // RCLCPP_DEBUG_STREAM(this->get_logger(),
                          // "Ref path callback at track index "
                          //     << idx << " with point (" << x_pos << ", "
                          //     << y_pos << ").");

      // ===
      double angle = 0.0;

      double dx_car_heading = cos(psi);
      double dy_car_heading = sin(psi);

      double dx_diffvec = x_pos - x_c;
      double dy_diffvec = y_pos - y_c;

      // https://wumbo.net/formulas/angle-between-two-vectors-2d/
      // will be in the range [-pi, pi] by definition
      angle = atan2(dy_car_heading * dx_diffvec - dx_car_heading * dy_diffvec,
                    dx_car_heading * dx_diffvec + dy_car_heading * dy_diffvec);

      if ((
              // check if point lies in the perception cone
              abs(angle) <= this->gamma_ &&
              // "maximum perception radius"
              (sqrt(pow(x_pos - x_c, 2) + pow(y_pos - y_c, 2)) <=
               this->r_perception_max_) &&
              // "minimum perception radius"
              (sqrt(pow(x_pos - x_c, 2) + pow(y_pos - y_c, 2)) >=
               this->r_perception_min_))) {
        // RCLCPP_DEBUG_STREAM(this->get_logger(),
                            // "Ref path callback at track index "
                            //     << idx << " with point (" << x_pos << ", "
                            //     << y_pos << ") in perception cone.");
        point2d_i.x = x_pos;
        point2d_i.y = y_pos;
        point2d_i.id = uint32_t(marker_id);
        marker_id++;

        // append to marker array
        ref_path_msg.points.push_back(point2d_i);

        if (!first_visited) {
          first_visited = true;
          // RCLCPP_DEBUG_STREAM(this->get_logger(),
                              // "Ref path callback at track index "
                              //     << idx << " with point (" << x_pos << ", "
                              //     << y_pos << ") is the first point in the "
                              //     << "perception cone.");
          idx_temp = idx;
        }
      }
    }

    this->initial_idx_refloop_ = idx_temp;

    ref_path_publisher_->publish(ref_path_msg);
    // RCLCPP_DEBUG_STREAM(this->get_logger(), "Ref path callback ended.");
  }

  // Timers
  rclcpp::TimerBase::SharedPtr solve_timer_;
  rclcpp::TimerBase::SharedPtr trackpub_timer_;

  // Publisher for vehicle state
  rclcpp::Publisher<sim_backend::msg::VehicleState>::SharedPtr state_publisher_;
  rclcpp::Publisher<sim_backend::msg::Point2DArray>::SharedPtr
      ref_path_publisher_;

  // Publisher for track
  rclcpp::Publisher<sim_backend::msg::Point2DArray>::SharedPtr track_publisher_;
  rclcpp::Publisher<sim_backend::msg::Point2DArray>::SharedPtr
      trackbounds_left_publisher_;
  rclcpp::Publisher<sim_backend::msg::Point2DArray>::SharedPtr
      trackbounds_right_publisher_;

  // Publisher for number of cones hit
  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr num_cones_hit_publisher_;

  // Controller control message publishers
  rclcpp::Publisher<std_msgs::msg::Empty>::SharedPtr
      start_controller_publisher_;
  rclcpp::Publisher<std_msgs::msg::Empty>::SharedPtr
      reset_controller_publisher_;
  rclcpp::Publisher<std_msgs::msg::Empty>::SharedPtr stop_controller_publisher_;

  // Subscriber to update input signals
  rclcpp::Subscription<sim_backend::msg::SysInput>::SharedPtr
      input_subscription_;

  // Subscriber to reset message
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr reset_subscription_;

  // Subscriber to toggle_sim_input_acceptance to accept control inputs or not
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr
      toggle_sim_input_acceptance_sub_;

  // lap counter publisher
  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr lap_count_publisher_;

  // lap time publisher
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr lap_time_publisher_;

  // last lap time publisher
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr last_lap_time_publisher_;

  // Cycle time of Simulation node
  std::chrono::milliseconds dt_{std::chrono::milliseconds(5)};
  std::chrono::milliseconds dt_trackpub_{std::chrono::milliseconds(20)};
  double dt_seconds_;

  // Maximum step time for ODE solver
  double max_dt_ = 1e-3;

  // Time scalar that gets updated each time step
  double t_;

  // "Perception Cone angle"
  double gamma_;
  // Perception radii (min and max)
  double r_perception_max_;
  double r_perception_min_;

  // car dimensions
  double l_to_front_ = 1.0;
  double l_to_rear_ = 1.0;
  double l_to_right_ = 0.8;
  double l_to_left_ = 0.8;

  bool accept_inputs_ = true;

  int lap_count_ = 0;
  int num_cones_hit_ = 0;
  double trackwidth_ = 5.0;
  double lap_time_ = 0.0;       // seconds for the current lap
  double lap_time_start_ = 0.0; // seconds on the clock
  double last_lap_time_ = 0.0;  // seconds for the last lap

  // Index to keep track of initial point of reference path
  size_t initial_idx_refloop_;

  // State vector of dynamic system
  state_type x_{state_type(27)};

  // Init System
  DynamicSystem sys_;

  // parameters struct
  parameters param_struct_;

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

  // boolean vector to keep track of which cones are currently hit
  std::vector<bool> cones_currently_hit_left_;
  std::vector<bool> cones_currently_hit_right_;

  // count vector to keep track of how many times each cone has been hit
  std::vector<int> cones_hit_count_left_;
  std::vector<int> cones_hit_count_right_;

  bool car_on_start_finish_line_ = false;

  // Create uncontrolled ODE stepper
  runge_kutta_dopri5<state_type> stepper_uncontrolled_ =
      runge_kutta_dopri5<state_type>();
  // Create Default Error checker
  default_error_checker<double, range_algebra, default_operations>
      error_checker =
          default_error_checker<double, range_algebra, default_operations>(
              1e-6, 1e-6);
  // Create Default Step Adjuster
  default_step_adjuster<double, double> step_adjuster =
      default_step_adjuster<double, double>(static_cast<double>(max_dt_));

  // finally create controlled stepper for error-aware stepping in the ODE solve
  // process
  controlled_runge_kutta<
      runge_kutta_dopri5<state_type>,
      default_error_checker<double, range_algebra, default_operations>,
      default_step_adjuster<double, double>, initially_resizer>
      stepper_ = controlled_runge_kutta<
          runge_kutta_dopri5<state_type>,
          default_error_checker<double, range_algebra, default_operations>,
          default_step_adjuster<double, double>, initially_resizer>(
          error_checker, step_adjuster, stepper_uncontrolled_);
};

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DynamicsSimulator>());
  rclcpp::shutdown();
  return 0;
}
