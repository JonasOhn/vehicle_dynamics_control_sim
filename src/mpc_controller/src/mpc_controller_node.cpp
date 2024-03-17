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

// standard
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "rclcpp/rclcpp.hpp"

// message includes
#include "mpc_controller/msg/mpc_cost_parameters.hpp"
#include "mpc_controller/msg/mpc_input_trajectory.hpp"
#include "mpc_controller/msg/mpc_kappa_trajectory.hpp"
#include "mpc_controller/msg/mpc_solver_state.hpp"
#include "mpc_controller/msg/mpc_state_trajectory.hpp"
#include "sim_backend/msg/point2_d.hpp"
#include "sim_backend/msg/point2_d_array.hpp"
#include "sim_backend/msg/sys_input.hpp"
#include "sim_backend/msg/vehicle_state.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/empty.hpp"

// controller class include
#include "mpc_controller/mpc_controller_class.hpp"

using namespace std::chrono_literals;

class MPCControllerNode : public rclcpp::Node {
public:
  MPCControllerNode()
      : Node("mpc_controller",
             rclcpp::NodeOptions()
                 .allow_undeclared_parameters(true)
                 .automatically_declare_parameters_from_overrides(true)) {

    /* ===== NODE INIT ====== */

    /* ========= SUBSCRIBERS ============ */
    // Init State Subscriber
    this->state_subscriber_ =
        this->create_subscription<sim_backend::msg::VehicleState>(
            "vehicle_state", 1,
            std::bind(&MPCControllerNode::state_update, this,
                      std::placeholders::_1));
    // Init reference path subscriber
    this->ref_path_subscriber_ =
        this->create_subscription<sim_backend::msg::Point2DArray>(
            "reference_path_points2d", 1,
            std::bind(&MPCControllerNode::ref_path_update, this,
                      std::placeholders::_1));

    // Subscriber to reset_controller topic
    this->reset_mpc_subscriber_ =
        this->create_subscription<std_msgs::msg::Empty>(
            "reset_controller", 1,
            std::bind(&MPCControllerNode::reset_controller, this,
                      std::placeholders::_1));
    // Subscriber to start_controller topic
    this->start_mpc_subscriber_ =
        this->create_subscription<std_msgs::msg::Empty>(
            "start_controller", 1,
            std::bind(&MPCControllerNode::start_controller, this,
                      std::placeholders::_1));
    // Subscriber to stop_controller topic
    this->stop_mpc_subscriber_ =
        this->create_subscription<std_msgs::msg::Empty>(
            "stop_controller", 1,
            std::bind(&MPCControllerNode::stop_controller, this,
                      std::placeholders::_1));

    this->param_update_subscriber_ =
        this->create_subscription<mpc_controller::msg::MpcCostParameters>(
            "mpc_cost_parameters", 1,
            std::bind(&MPCControllerNode::update_cost_parameters_from_msg, this,
                      std::placeholders::_1));

    /* ========= PUBLISHERS ============ */
    // Init MPC <Functional> Prediction Horizon Publishers
    this->mpc_xtraj_publisher_ =
        this->create_publisher<mpc_controller::msg::MpcStateTrajectory>(
            "mpc_state_trajectory", 10);
    this->mpc_utraj_publisher_ =
        this->create_publisher<mpc_controller::msg::MpcInputTrajectory>(
            "mpc_input_trajectory", 10);
    this->mpc_solve_state_publisher_ =
        this->create_publisher<mpc_controller::msg::MpcSolverState>(
            "mpc_solver_state", 10);
    this->mpc_kappa_traj_publisher_ =
        this->create_publisher<mpc_controller::msg::MpcKappaTrajectory>(
            "mpc_kappa_trajectory", 10);

    // Init MPC <Visual> Prediction Horizon Publishers
    this->spline_fit_publisher_ =
        this->create_publisher<sim_backend::msg::Point2DArray>(
            "mpc_spline_points2d", 10);
    this->xy_predict_traj_publisher_ =
        this->create_publisher<sim_backend::msg::Point2DArray>(
            "mpc_xy_predict_trajectory", 10);

    // Init Control Command Publisher and corresponding Timer with respecitve
    // callback
    this->control_cmd_publisher_ =
        this->create_publisher<sim_backend::msg::SysInput>("vehicle_input", 10);
    this->control_cmd_timer_ = rclcpp::create_timer(
        this, this->get_clock(), this->dt_,
        std::bind(&MPCControllerNode::control_callback, this));

    // this->param_update_timer_ = rclcpp::create_timer(this, this->get_clock(),
    // 1ms, std::bind(&MPCControllerNode::update_cost_parameters, this));

    /* ========= CONTROLLER ============ */
    RCLCPP_DEBUG_STREAM(this->get_logger(), "Setting up MPC.");

    RCLCPP_DEBUG_STREAM(this->get_logger(),
                        "--- Initializing Horizon parameters.");
    this->mpc_controller_obj_.set_horizon_parameters(this->N_horizon_mpc_,
                                                     this->T_final_mpc_);

    RCLCPP_DEBUG_STREAM(this->get_logger(),
                        "--- Initializing Solver parameters.");
    this->mpc_controller_obj_.set_solver_parameters(
        this->sqp_max_iter_, this->rti_phase_, this->warm_start_first_qp_);

    RCLCPP_DEBUG_STREAM(this->get_logger(),
                        "--- Initializing Model parameters.");
    this->mpc_controller_obj_.set_model_parameters(
        this->l_f_, this->l_r_, this->m_, this->C_d_, this->C_r_);

    RCLCPP_DEBUG_STREAM(this->get_logger(),
                        "--- Initializing Cost parameters.");
    this->mpc_controller_obj_.set_cost_parameters(
        this->q_sd_, this->q_n_, this->q_mu_, this->q_axm_, this->q_dels_,
        this->r_daxm_, this->r_ddels_);

    RCLCPP_DEBUG_STREAM(this->get_logger(),
                        "--- Sending control loop time step to MPC.");
    this->mpc_controller_obj_.set_dt_control_feedback(this->dt_seconds_);

    RCLCPP_DEBUG_STREAM(this->get_logger(), "--- Initializing MPC Solver.");

    if (this->mpc_controller_obj_.init_solver() != 0) {
      RCLCPP_INFO_STREAM(this->get_logger(),
                         "Solver initialization failed. Node being shut down.");
      rclcpp::shutdown();
    }

    /* ========= SUCCESS MESSAGE ============ */

    RCLCPP_INFO_STREAM(this->get_logger(),
                       "Node " << this->get_name() << " initialized.");
  }

private:
  void ref_path_update(const sim_backend::msg::Point2DArray &refpath_msg) {
    RCLCPP_DEBUG_STREAM(
        this->get_logger(),
        "Reference path update called with refpath message containing "
            << refpath_msg.points.size() << " points.");

    // clear previous reference path
    this->reference_path_.clear();

    // Create container for single 2D point with data type from message
    auto point2d = sim_backend::msg::Point2D();

    // variable to store distance between points on the path
    double dist_to_prev_point = 0.0;

    // get all reference points
    for (size_t i = 0; i < refpath_msg.points.size(); i++) {
      // Get i-th point on published path
      point2d = refpath_msg.points[i];

      // 2D point
      std::vector<double> ref_point{0.0, 0.0};
      ref_point[0] = point2d.x;
      ref_point[1] = point2d.y;

      RCLCPP_DEBUG_STREAM(this->get_logger(), "Candidate ref_point = ("
                                                  << ref_point[0] << ", "
                                                  << ref_point[1] << ")");

      // From second point (i > 0) on: Check if point too far from previous
      // point
      if (i > 0) {
        dist_to_prev_point =
            std::hypot(ref_point[0] - this->reference_path_[i - 1][0],
                       ref_point[1] - this->reference_path_[i - 1][1]);

        if (dist_to_prev_point > this->max_dist_to_prev_path_point_) {
          RCLCPP_DEBUG_STREAM(this->get_logger(),
                              "Breaking out because: ref_point = ("
                                  << ref_point[0] << ", " << ref_point[1]
                                  << ") too far from last point.");
          break;
        }
      }

      // Add point to reference path
      this->reference_path_.push_back(ref_point);
      RCLCPP_DEBUG_STREAM(this->get_logger(), "Added: ref_point = ("
                                                  << ref_point[0] << ", "
                                                  << ref_point[1] << ")");
    }

    // Check if we have enough points for a bspline fit
    if (this->reference_path_.size() + 2 < 4) {
      RCLCPP_DEBUG_STREAM(this->get_logger(),
                          "Distance-filtered ref-path size too small for "
                          "bspline fit. Quitting Callback.");
      return;
    }

    RCLCPP_DEBUG_STREAM(
        this->get_logger(),
        "Iterated through reference points, size of reference path: "
            << this->reference_path_.size());

    // Invoke Reference Path update in MPC object (includes bspline fit etc.)
    this->mpc_controller_obj_.set_reference_path(this->reference_path_);

    // ========== END of path update ===============
    RCLCPP_DEBUG_STREAM(this->get_logger(), "Reference path update succesful.");
  }

  void reset_controller(const std_msgs::msg::Empty msg) {
    RCLCPP_DEBUG_STREAM(this->get_logger(), "Resetting Controller.");

    // stop controller
    this->controller_running_ = false;

    // reset MPC
    this->mpc_controller_obj_.reset_prediction_trajectories();

    // clear reference path
    this->reference_path_.clear();

    // clear visualization
    this->xy_spline_points_.clear();
    this->xy_predict_points_.clear();

    RCLCPP_DEBUG_STREAM(this->get_logger(), "Controller reset.");
  }

  void start_controller(const std_msgs::msg::Empty msg) {
    RCLCPP_DEBUG_STREAM(this->get_logger(), "Starting Controller.");

    // start controller
    this->controller_running_ = true;

    RCLCPP_DEBUG_STREAM(this->get_logger(), "Controller started.");
  }

  void stop_controller(const std_msgs::msg::Empty msg) {
    RCLCPP_DEBUG_STREAM(this->get_logger(), "Stopping Controller.");

    // stop controller
    this->controller_running_ = false;

    RCLCPP_DEBUG_STREAM(this->get_logger(), "Controller stopped.");
  }

  void update_cost_parameters_from_msg(
      const mpc_controller::msg::MpcCostParameters msg) {
    RCLCPP_INFO_STREAM(this->get_logger(),
                       "Updating MPC parameters from ROS message.");

    q_sd_ = msg.q_sd;
    q_n_ = msg.q_n;
    q_mu_ = msg.q_mu;
    q_dels_ = msg.q_dels;
    q_axm_ = msg.q_ax;
    r_daxm_ = msg.r_dax;
    r_ddels_ = msg.r_ddels;

    this->set_parameter(rclcpp::Parameter("cost.q_sd", msg.q_sd));
    this->set_parameter(rclcpp::Parameter("cost.q_n", msg.q_n));
    this->set_parameter(rclcpp::Parameter("cost.q_mu", msg.q_mu));
    this->set_parameter(rclcpp::Parameter("cost.q_dels", msg.q_dels));
    this->set_parameter(rclcpp::Parameter("cost.q_ax", msg.q_ax));
    this->set_parameter(rclcpp::Parameter("cost.r_dax", msg.r_dax));
    this->set_parameter(rclcpp::Parameter("cost.r_ddels", msg.r_ddels));

    this->mpc_controller_obj_.set_cost_parameters(msg.q_sd, msg.q_n, msg.q_mu,
                                                  msg.q_ax, msg.q_dels,
                                                  msg.r_dax, msg.r_ddels);
  }

  void update_cost_parameters() {
    RCLCPP_DEBUG_STREAM(this->get_logger(),
                        "Updating Parameters for Controller.");

    q_sd_ = this->get_parameter("cost.q_sd").as_double();
    q_n_ = this->get_parameter("cost.q_n").as_double();
    q_mu_ = this->get_parameter("cost.q_mu").as_double();
    q_dels_ = this->get_parameter("cost.q_dels").as_double();
    q_axm_ = this->get_parameter("cost.q_ax").as_double();
    r_daxm_ = this->get_parameter("cost.r_dax").as_double();
    r_ddels_ = this->get_parameter("cost.r_ddels").as_double();

    this->mpc_controller_obj_.set_cost_parameters(q_sd_, q_n_, q_mu_, q_axm_,
                                                  q_dels_, r_daxm_, r_ddels_);

    RCLCPP_DEBUG_STREAM(this->get_logger(), "Parameters updated.");
  }

  void control_callback() {
    if (this->controller_running_) {
      RCLCPP_DEBUG_STREAM(this->get_logger(), "Control Callback called.");

      // total time count
      double total_elapsed_time = 0.0;

      // =================== Initial State for MPC ========================
      RCLCPP_DEBUG_STREAM(this->get_logger(), "Setting x0 ...");
      this->mpc_controller_obj_.set_initial_state();

      // =================== Parametrization for MPC ========================
      RCLCPP_DEBUG_STREAM(this->get_logger(), "Setting p0 ...");
      this->mpc_controller_obj_.init_mpc_parameters();

      // ============== State and Input Trajectory Initialization for MPC
      // ==============
      RCLCPP_DEBUG_STREAM(this->get_logger(),
                          "Initializing solution (X, U, x_N) ...");
      this->mpc_controller_obj_.init_mpc_horizon();

      /* ===================== SOLVE MPC ===================== */
      RCLCPP_DEBUG_STREAM(this->get_logger(), "Calling NLP Solver.. ");
      this->mpc_controller_obj_.solve_mpc();

      // =================== Evaluate and Publish MPC Solve Process
      // ======================== prepare evaluation
      double kkt_norm_inf;
      double elapsed_time;
      int sqp_iter;
      int qp_status;
      int sqp_status;

      RCLCPP_DEBUG_STREAM(this->get_logger(), "Retrieving Solver Statistics..");
      this->mpc_controller_obj_.get_solver_statistics(
          sqp_status, qp_status, sqp_iter, kkt_norm_inf, elapsed_time);
      RCLCPP_INFO_STREAM(this->get_logger(),
                         "SQP Status: "
                             << sqp_status << "\n"
                             << "QP (in SQP) Status: " << qp_status << "\n"
                             << "SQP Iterations: " << sqp_iter << "\n"
                             << "KKT inf. norm: " << kkt_norm_inf << "\n"
                             << "Solve Time: " << elapsed_time);
      total_elapsed_time += elapsed_time;

      auto solver_state_msg = mpc_controller::msg::MpcSolverState();
      solver_state_msg.solver_status = sqp_status;
      solver_state_msg.kkt_norm_inf = kkt_norm_inf;
      solver_state_msg.total_solve_time = total_elapsed_time;
      solver_state_msg.sqp_iter = sqp_iter;
      solver_state_msg.qp_solver_status = qp_status;
      mpc_solve_state_publisher_->publish(solver_state_msg);

      // =================== Get Input for Plant from MPC Solver
      // ========================
      /* Get solver outputs and publish message */
      double u_eval[2];

      this->mpc_controller_obj_.get_input(u_eval);

      auto veh_input_msg = sim_backend::msg::SysInput();
      veh_input_msg.fx_r = u_eval[0] / 2.0;
      veh_input_msg.fx_f = u_eval[0] / 2.0;
      veh_input_msg.del_s = u_eval[1];

      control_cmd_publisher_->publish(veh_input_msg);

      RCLCPP_INFO_STREAM(this->get_logger(), "Published Input message.");

      // =================== Publish MPC Predictions ========================
      this->publish_predictions();

      // ===== FINISH Control Callback =========
      RCLCPP_DEBUG_STREAM(this->get_logger(), "Control Callback ended.");
    } else {
      auto veh_input_msg = sim_backend::msg::SysInput();
      veh_input_msg.fx_r = 0.0;
      veh_input_msg.fx_f = 0.0;
      veh_input_msg.del_s = 0.0;
      control_cmd_publisher_->publish(veh_input_msg);
      RCLCPP_DEBUG_STREAM(this->get_logger(),
                          "Controller not running. Control Callback ended.");
    }
  }

  void state_update(const sim_backend::msg::VehicleState &state_msg) {
    RCLCPP_DEBUG_STREAM(this->get_logger(), "Updating State for Controller.");

    double s = 0.0;
    double n = 0.0;
    double mu = 0.0;
    double x_path = 0.0;
    double y_path = 0.0;

    this->mpc_controller_obj_.set_state(
        state_msg.x_c, state_msg.y_c, state_msg.psi, state_msg.dx_c_v,
        state_msg.del_s_ref,
        (state_msg.fx_f_act + state_msg.fx_r_act) /
            this->mpc_controller_obj_.get_mass(),
        s, n, mu);

    RCLCPP_DEBUG_STREAM(this->get_logger(),
                        "s = " << s << ", n = " << n << ", mu = " << mu);
  }

  void publish_predictions() {
    if (this->controller_running_) {
      // =================== Evaluate and Publish MPC Predictions
      // ========================

      RCLCPP_DEBUG_STREAM(this->get_logger(), "Evaluating MPC Predictions.");

      auto state_traj_msg = mpc_controller::msg::MpcStateTrajectory();
      auto input_traj_msg = mpc_controller::msg::MpcInputTrajectory();
      auto kappa_traj_msg = mpc_controller::msg::MpcKappaTrajectory();

      this->mpc_controller_obj_.get_predictions(
          state_traj_msg.s, state_traj_msg.n, state_traj_msg.mu,
          state_traj_msg.vx_c, state_traj_msg.vy_c, state_traj_msg.dpsi,
          state_traj_msg.ax_m, state_traj_msg.del_s, input_traj_msg.dax_m,
          input_traj_msg.ddel_s, kappa_traj_msg.s_traj_mpc,
          kappa_traj_msg.kappa_traj_mpc, kappa_traj_msg.s_ref_spline,
          kappa_traj_msg.kappa_ref_spline, kappa_traj_msg.s_ref_mpc,
          kappa_traj_msg.kappa_ref_mpc);

      for (i_ = 0; i_ < (int)kappa_traj_msg.kappa_ref_mpc.size(); i_++) {
        RCLCPP_DEBUG_STREAM(this->get_logger(),
                            "s = " << kappa_traj_msg.s_ref_mpc[i_]
                                   << ", kappa = "
                                   << kappa_traj_msg.kappa_ref_mpc[i_]);
      }

      mpc_xtraj_publisher_->publish(state_traj_msg);
      mpc_utraj_publisher_->publish(input_traj_msg);
      mpc_kappa_traj_publisher_->publish(kappa_traj_msg);

      RCLCPP_DEBUG_STREAM(this->get_logger(), "Published MPC Predictions.");

      /* ========================== Publish Visualization Messages
       * ======================== */

      RCLCPP_DEBUG_STREAM(this->get_logger(),
                          "Clearing MPC Visual Predictions.");

      this->xy_spline_points_.clear();
      this->xy_predict_points_.clear();

      RCLCPP_DEBUG_STREAM(this->get_logger(),
                          "Retrieving new MPC Visual Predictions.");

      this->mpc_controller_obj_.get_visual_predictions(
          this->xy_spline_points_, this->xy_predict_points_);

      // ------ SPLINE FIT
      auto bspline_fit_msg = sim_backend::msg::Point2DArray();
      auto point2d_i = sim_backend::msg::Point2D();

      RCLCPP_DEBUG_STREAM(this->get_logger(),
                          "Filling Spline Fit Point2D Array.");

      // iterate through all spline points
      for (i_ = 0; i_ < (int)this->xy_spline_points_.size(); i_++) {
        point2d_i.id = i_;

        point2d_i.x = this->xy_spline_points_[i_][0];
        point2d_i.y = this->xy_spline_points_[i_][1];

        // append to point2d array
        bspline_fit_msg.points.push_back(point2d_i);
      }

      spline_fit_publisher_->publish(bspline_fit_msg);

      // ----- PATH PREDICTION MPC
      auto path_predict_msg = sim_backend::msg::Point2DArray();

      RCLCPP_DEBUG_STREAM(this->get_logger(),
                          "Filling XY-predictions Point2D Array.");

      // iterate through all spline points
      for (i_ = 0; i_ < (int)this->xy_predict_points_.size(); i_++) {
        point2d_i.id = i_;
        point2d_i.x = this->xy_predict_points_[i_][0];
        point2d_i.y = this->xy_predict_points_[i_][1];

        // append to point2d array
        path_predict_msg.points.push_back(point2d_i);
      }
      xy_predict_traj_publisher_->publish(path_predict_msg);

      RCLCPP_DEBUG_STREAM(this->get_logger(), "Published MPC Predictions.");
    }
  }

  // Initialize Controller Object
  MpcController mpc_controller_obj_ = MpcController();

  // Subscriber to state message published at faster frequency
  rclcpp::Subscription<sim_backend::msg::VehicleState>::SharedPtr
      state_subscriber_;

  // Subscriber to ref path provided by "perception"
  rclcpp::Subscription<sim_backend::msg::Point2DArray>::SharedPtr
      ref_path_subscriber_;

  // subscribers to reset, start and stop controller
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr reset_mpc_subscriber_;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr start_mpc_subscriber_;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr stop_mpc_subscriber_;

  rclcpp::Subscription<mpc_controller::msg::MpcCostParameters>::SharedPtr
      param_update_subscriber_;

  // Timer for control command publishing
  rclcpp::TimerBase::SharedPtr control_cmd_timer_;

  // Control command publisher
  rclcpp::Publisher<sim_backend::msg::SysInput>::SharedPtr
      control_cmd_publisher_;

  // Timer for parameter update
  // rclcpp::TimerBase::SharedPtr param_update_timer_;

  // mpc trajectory publishers
  rclcpp::Publisher<mpc_controller::msg::MpcStateTrajectory>::SharedPtr
      mpc_xtraj_publisher_;
  rclcpp::Publisher<mpc_controller::msg::MpcInputTrajectory>::SharedPtr
      mpc_utraj_publisher_;
  rclcpp::Publisher<mpc_controller::msg::MpcKappaTrajectory>::SharedPtr
      mpc_kappa_traj_publisher_;
  rclcpp::Publisher<mpc_controller::msg::MpcSolverState>::SharedPtr
      mpc_solve_state_publisher_;

  rclcpp::Publisher<sim_backend::msg::Point2DArray>::SharedPtr
      spline_fit_publisher_;
  rclcpp::Publisher<sim_backend::msg::Point2DArray>::SharedPtr
      xy_predict_traj_publisher_;

  // Reference Path to give to MPC
  std::vector<std::vector<double>> reference_path_;
  const double max_dist_to_prev_path_point_ =
      this->get_parameter("max_distance_path_points").as_double();

  // Step Time for controller publisher
  int dt_ms = this->get_parameter("dt_ms").as_int();
  std::chrono::milliseconds dt_{std::chrono::milliseconds(dt_ms)};
  const double dt_seconds_ = dt_.count() / 1e3;

  // MPC horizon parameters
  double T_final_mpc_ = this->get_parameter("horizon.T_final").as_double();
  int N_horizon_mpc_ = this->get_parameter("horizon.N_horizon").as_int();

  // Solver parameters
  int sqp_max_iter_ =
      this->get_parameter("solver_options.nlp_solver_max_iter").as_int();
  int rti_phase_ =
      this->get_parameter("solver_options.init_rti_phase").as_int();
  int ws_first_qp =
      this->get_parameter("solver_options.qp_solver_warm_start").as_int();
  bool warm_start_first_qp_ = (bool)ws_first_qp;

  // Controller Model parameters
  double l_f_ = this->get_parameter("model.l_f").as_double();
  double l_r_ = this->get_parameter("model.l_r").as_double();
  double m_ = this->get_parameter("model.m").as_double();
  double C_d_ = this->get_parameter("model.C_d").as_double();
  double C_r_ = this->get_parameter("model.C_r").as_double();

  double q_sd_ = this->get_parameter("cost.q_sd").as_double();
  double q_n_ = this->get_parameter("cost.q_n").as_double();
  double q_mu_ = this->get_parameter("cost.q_mu").as_double();
  double q_dels_ = this->get_parameter("cost.q_dels").as_double();
  double q_axm_ = this->get_parameter("cost.q_ax").as_double();
  double r_ddels_ = this->get_parameter("cost.r_ddels").as_double();
  double r_daxm_ = this->get_parameter("cost.r_dax").as_double();

  // For visualization
  std::vector<std::vector<double>> xy_spline_points_;
  std::vector<std::vector<double>> xy_predict_points_;

  // Iterators
  int i_ = 0;

  // boolean status variable for controller
  bool controller_running_ = false;
};

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MPCControllerNode>());
  rclcpp::shutdown();
  return 0;
}
