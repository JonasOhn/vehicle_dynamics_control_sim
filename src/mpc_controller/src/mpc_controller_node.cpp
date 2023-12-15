// standard
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <cmath>
#include <fstream>

#include "rclcpp/rclcpp.hpp"

// message includes
#include "sim_backend/msg/sys_input.hpp"
#include "sim_backend/msg/vehicle_state.hpp"
#include "sim_backend/msg/ref_path.hpp"
#include "mpc_controller/msg/mpc_state_trajectory.hpp"
#include "mpc_controller/msg/mpc_input_trajectory.hpp"
#include "mpc_controller/msg/mpc_kappa_trajectory.hpp"
#include "mpc_controller/msg/mpc_solver_state.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

// controller class include
#include "mpc_controller/mpc_controller_class.hpp"

#define DT_MS 50


using namespace std::chrono_literals;


class MPCControllerNode : public rclcpp::Node
{
  public:
    MPCControllerNode()
    : Node("mpc_controller",
            rclcpp::NodeOptions()
                .allow_undeclared_parameters(true)
                .automatically_declare_parameters_from_overrides(true))
    {

        /* ========= SUBSCRIBER ============ */
        // Init State Subscriber
        this->state_subscriber_ = this->create_subscription<sim_backend::msg::VehicleState>(
            "vehicle_state", 1, std::bind(&MPCControllerNode::state_update, this, std::placeholders::_1));
        // Init reference path subscriber
        this->ref_path_subscriber_ = this->create_subscription<sim_backend::msg::RefPath>(
            "reference_path", 1, std::bind(&MPCControllerNode::ref_path_update, this, std::placeholders::_1));


        /* ========= PUBLISHER ============ */
        // Init MPC <Numerical> Prediction Horizon Publishers
        this->mpc_xtraj_publisher_ = this->create_publisher<mpc_controller::msg::MpcStateTrajectory>(
            "mpc_state_trajectory", 10);
        this->mpc_utraj_publisher_ = this->create_publisher<mpc_controller::msg::MpcInputTrajectory>(
            "mpc_input_trajectory", 10);
        this->mpc_solve_state_publisher_ = this->create_publisher<mpc_controller::msg::MpcSolverState>(
            "mpc_solver_state", 10);
        this->mpc_kappa_traj_publisher_ = this->create_publisher<mpc_controller::msg::MpcKappaTrajectory>(
            "mpc_kappa_trajectory", 10);

        // Init MPC <Visual> Prediction Horizon Publishers
        this->spline_fit_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "mpc_spline_fit", 10);
        this->xy_predict_traj_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "mpc_xy_predict_trajectory", 10);

        // Init Control Command Publisher and corresponding Timer with respecitve callback
        this->control_cmd_publisher_ = this->create_publisher<sim_backend::msg::SysInput>("vehicle_input", 10);
        this->control_cmd_timer_ = rclcpp::create_timer(this, this->get_clock(), this->dt_, std::bind(&MPCControllerNode::control_callback, this));

        RCLCPP_DEBUG_STREAM(this->get_logger(), "Setting up MPC.");
        /* ========= CONTROLLER ============ */
        RCLCPP_DEBUG_STREAM(this->get_logger(), "--- Initializing Horizon parameters.");
        this->mpc_controller_obj_.set_horizon_parameters(this->n_s_mpc_,
                                                         this->N_horizon_mpc_,
                                                         this->T_final_mpc_,
                                                         this->s_max_mpc_);
        RCLCPP_DEBUG_STREAM(this->get_logger(), "--- Initializing Solver parameters.");
        this->mpc_controller_obj_.set_solver_parameters(this->sqp_max_iter_,
                                                        this->rti_phase_,
                                                        this->warm_start_first_qp_);
        RCLCPP_DEBUG_STREAM(this->get_logger(), "--- Initializing Model parameters.");
        this->mpc_controller_obj_.set_model_parameters(this->l_f_,
                                                       this->l_r_,
                                                       this->m_,
                                                       this->Iz_,
                                                       this->g_,
                                                       this->D_tire_,
                                                       this->C_tire_,
                                                       this->B_tire_,
                                                       this->C_d_,
                                                       this->C_r_);
        RCLCPP_DEBUG_STREAM(this->get_logger(), "--- Sending control loop time step to MPC.");
        this->mpc_controller_obj_.set_dt_control_feedback(this->dt_seconds_);
        
        RCLCPP_DEBUG_STREAM(this->get_logger(), "--- Initializing MPC Solver.");
        if(this->mpc_controller_obj_.init_solver() != 0){
            RCLCPP_INFO_STREAM(this->get_logger(), "Solver initialization failed. Node being shut down.");
            rclcpp::shutdown();
        }


        /* ========= SUCCESS MESSAGE ============ */
        RCLCPP_INFO_STREAM(this->get_logger(), "Node " << this->get_name() << " initialized.");
    }

  private:

    void ref_path_update(const sim_backend::msg::RefPath & refpath_msg)
    {
        RCLCPP_DEBUG_STREAM(this->get_logger(), "Reference path update called.");

        // clear previous reference path
        this->reference_path_.clear();

        // Create container for single 2D point with data type from message
        auto path_pos = sim_backend::msg::Point2D();

        // 2D point
        std::vector<double> ref_point{0.0, 0.0};

        // variable to store distance between points on the path
        double dist_to_prev_point = 0.0;

        // get all reference points
        for (size_t i = 0; i < refpath_msg.ref_path.size(); i++){
            // Get i-th point on published path
            path_pos = refpath_msg.ref_path[i];
            ref_point[0] = path_pos.point_2d[0];
            ref_point[1] = path_pos.point_2d[1];

            // From second point on: Check if point too far from previous point
            if(i > 0){
                dist_to_prev_point = sqrt(pow(ref_point[0] - this->reference_path_[i-1][0], 2.0) 
                    + pow(ref_point[1] - this->reference_path_[i-1][1], 2.0));

                if(dist_to_prev_point > this->max_dist_to_prev_path_point_){
                    RCLCPP_DEBUG_STREAM(this->get_logger(), 
                                        "Breaking out because: ref_point = (" 
                                        << ref_point[0] << ", " 
                                        << ref_point[1] << ") too far from last point.");
                    break;
                }
            }
            this->reference_path_.push_back(ref_point);
            RCLCPP_DEBUG_STREAM(this->get_logger(),
                                "Got: ref_point = (" << ref_point[0] << ", " << ref_point[1] << ")");
        }

        // Check if we have enough points for a bspline fit
        if(this->reference_path_.size() + 2 < 4){
            RCLCPP_DEBUG_STREAM(this->get_logger(),
                                "Distance-filtered ref-path size too small for bspline fit. Quitting Callback.");
            return;
        }

        RCLCPP_DEBUG_STREAM(this->get_logger(), 
                            "Iterated through reference points, size of reference path: " 
                            << this->reference_path_.size());

        // Invoke Reference Path update in MPC object (includes bspline fit etc.)
        this->mpc_controller_obj_.set_reference_path(this->reference_path_);


        // ========== END of path update ===============
        RCLCPP_DEBUG_STREAM(this->get_logger(), "Reference path update succesful.");
    }

    void control_callback()
    {
        RCLCPP_DEBUG_STREAM(this->get_logger(), "Control Callback called.");

        // total time count
        double total_elapsed_time = 0.0;


        // =================== Initial State for MPC ========================
        RCLCPP_DEBUG_STREAM(this->get_logger(), "Setting x0");
        this->mpc_controller_obj_.set_initial_state();


        // =================== Parametrization for MPC ========================
        RCLCPP_DEBUG_STREAM(this->get_logger(), "Setting p0 ...");
        this->mpc_controller_obj_.init_mpc_parameters();


        // ============== State and Input Trajectory Initialization for MPC ==============
        RCLCPP_DEBUG_STREAM(this->get_logger(), "Initializing solution (x, u, x_N)");
        this->mpc_controller_obj_.init_mpc_horizon();


        /* ===================== SOLVE MPC ===================== */
        RCLCPP_DEBUG_STREAM(this->get_logger(), "Calling NLP Solver.. ");
        this->mpc_controller_obj_.solve_mpc();


        // =================== Evaluate and Publish MPC Solve Process ========================
        // prepare evaluation
        double kkt_norm_inf;
        double elapsed_time;
        int sqp_iter;
        int qp_status;
        int sqp_status;

        RCLCPP_DEBUG_STREAM(this->get_logger(), "Retrieving Solver Statistics..");
        this->mpc_controller_obj_.get_solver_statistics(sqp_status,
                                                        qp_status,
                                                        sqp_iter,
                                                        kkt_norm_inf, 
                                                        elapsed_time);
        RCLCPP_INFO_STREAM(this->get_logger(), "SQP Status: " << sqp_status << "\n" <<
                                               "QP (in SQP) Status: " << qp_status << "\n" <<
                                               "SQP Iterations: " << sqp_iter << "\n" << 
                                               "KKT inf. norm: " << kkt_norm_inf << "\n" <<
                                               "Solve Time: " << elapsed_time );
        total_elapsed_time += elapsed_time;

        auto solver_state_msg = mpc_controller::msg::MpcSolverState();
        solver_state_msg.solver_status = sqp_status;
        solver_state_msg.kkt_norm_inf = kkt_norm_inf;
        solver_state_msg.total_solve_time = total_elapsed_time;
        solver_state_msg.sqp_iter = sqp_iter;
        solver_state_msg.qp_solver_status = qp_status;
        mpc_solve_state_publisher_->publish(solver_state_msg);


        // =================== Get Input for Plant from MPC Solver ========================
        /* Get solver outputs and publish message asap */
        double u_eval[2];

        this->mpc_controller_obj_.get_input(u_eval);

        auto veh_input_msg = sim_backend::msg::SysInput();
        veh_input_msg.fx_r = u_eval[0] / 2.0;
        veh_input_msg.fx_f = u_eval[0] / 2.0;
        veh_input_msg.del_s = u_eval[1];
        
        control_cmd_publisher_->publish(veh_input_msg);

        RCLCPP_INFO_STREAM(this->get_logger(), "Published Input message.");


        // =================== Evaluate and Publish MPC Predictions ========================
        
        RCLCPP_DEBUG_STREAM(this->get_logger(), "Evaluating MPC Predictions.");
        
        auto state_traj_msg = mpc_controller::msg::MpcStateTrajectory();
        auto input_traj_msg = mpc_controller::msg::MpcInputTrajectory();
        auto kappa_traj_msg = mpc_controller::msg::MpcKappaTrajectory();

        this->mpc_controller_obj_.get_predictions(state_traj_msg.s,
                                                  state_traj_msg.n,
                                                  state_traj_msg.mu,
                                                  state_traj_msg.vx_c,
                                                  state_traj_msg.vy_c,
                                                  state_traj_msg.dpsi,
                                                  input_traj_msg.fx_m,
                                                  input_traj_msg.del_s,
                                                  kappa_traj_msg.s_traj_mpc,
                                                  kappa_traj_msg.kappa_traj_mpc,
                                                  kappa_traj_msg.s_ref_spline,
                                                  kappa_traj_msg.kappa_ref_spline,
                                                  kappa_traj_msg.s_ref_mpc,
                                                  kappa_traj_msg.kappa_ref_mpc
                                                  );

        mpc_xtraj_publisher_->publish(state_traj_msg);
        mpc_utraj_publisher_->publish(input_traj_msg);
        mpc_kappa_traj_publisher_->publish(kappa_traj_msg);

        RCLCPP_DEBUG_STREAM(this->get_logger(), "Published MPC Predictions.");

        /* ========================== Publish Visualization Messages ======================== */

        RCLCPP_DEBUG_STREAM(this->get_logger(), "Clearing MPC Visual Predictions.");

        this->xy_spline_points_.clear();
        this->xy_predict_points_.clear();

        RCLCPP_DEBUG_STREAM(this->get_logger(), "Retrieving new MPC Visual Predictions.");

        this->mpc_controller_obj_.get_visual_predictions(this->xy_spline_points_,
                                                         this->xy_predict_points_);

        // ------ SPLINE FIT
        auto bspline_fit_msg = visualization_msgs::msg::MarkerArray();
        auto marker_i = visualization_msgs::msg::Marker();

        RCLCPP_DEBUG_STREAM(this->get_logger(), "Filling Spline Fit Marker Array.");
        int i = 0;
        // iterate through all spline points
        for (i = 0; i < (int)this->xy_spline_points_.size(); i++){
            marker_i.header.frame_id = "world";
            marker_i.header.stamp = this->now();
            // set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
            marker_i.type = 2;
            marker_i.id = i;

            // Set the scale of the marker
            marker_i.scale.x = 0.2;
            marker_i.scale.y = 0.2;
            marker_i.scale.z = 0.2;

            // Set the color
            marker_i.color.r = 0.0;
            marker_i.color.g = 1.0;
            marker_i.color.b = 0.0;
            marker_i.color.a = 1.0;

            // Set the pose of the marker
            marker_i.pose.position.x = this->xy_spline_points_[i][0];
            marker_i.pose.position.y = this->xy_spline_points_[i][1];
            marker_i.pose.position.z = 0;
            marker_i.pose.orientation.x = 0;
            marker_i.pose.orientation.y = 0;
            marker_i.pose.orientation.z = 0;
            marker_i.pose.orientation.w = 1;

            // append to marker array
            bspline_fit_msg.markers.push_back(marker_i);
        }
        
        spline_fit_publisher_->publish(bspline_fit_msg);

        // ----- PATH PREDICTION MPC
        auto path_predict_msg = visualization_msgs::msg::MarkerArray();

        RCLCPP_DEBUG_STREAM(this->get_logger(), "Filling XY-predictions Marker Array.");

        // iterate through all spline points
        for (i = 0; i < (int)this->xy_predict_points_.size(); i++){
            marker_i.header.frame_id = "vehicle_frame";
            marker_i.header.stamp = this->now();
            // set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
            marker_i.type = 1;
            marker_i.id = i;

            // Set the scale of the marker
            marker_i.scale.x = 0.2;
            marker_i.scale.y = 0.2;
            marker_i.scale.z = 0.2;

            // Set the color
            marker_i.color.r = 0.8;
            marker_i.color.g = 0.0;
            marker_i.color.b = 0.8;
            marker_i.color.a = 1.0;

            // Set the pose of the marker
            marker_i.pose.position.x = this->xy_predict_points_[i][0];
            marker_i.pose.position.y = this->xy_predict_points_[i][1];
            marker_i.pose.position.z = 0;
            marker_i.pose.orientation.x = 0;
            marker_i.pose.orientation.y = 0;
            marker_i.pose.orientation.z = 0;
            marker_i.pose.orientation.w = 1;

            // append to marker array
            path_predict_msg.markers.push_back(marker_i);
        }
        xy_predict_traj_publisher_->publish(path_predict_msg);


        // ===== FINISH Control Callback =========
        RCLCPP_DEBUG_STREAM(this->get_logger(), "Control Callback ended.");
    }

    void state_update(const sim_backend::msg::VehicleState & state_msg)
    {
        this->mpc_controller_obj_.set_state(state_msg.psi,
                                            state_msg.dx_c_v,
                                            state_msg.dy_c_v,
                                            state_msg.dpsi);
    }

    // Initialize Controller Object
    MpcController mpc_controller_obj_ = MpcController();

    // Subscriber to state message published at faster frequency
    rclcpp::Subscription<sim_backend::msg::VehicleState>::SharedPtr state_subscriber_;
    
    // Subscriber to ref path provided by "perception"
    rclcpp::Subscription<sim_backend::msg::RefPath>::SharedPtr ref_path_subscriber_;

    // Timer for control command publishing
    rclcpp::TimerBase::SharedPtr control_cmd_timer_;

    // Control command publisher
    rclcpp::Publisher<sim_backend::msg::SysInput>::SharedPtr control_cmd_publisher_;

    // mpc trajectory publishers
    rclcpp::Publisher<mpc_controller::msg::MpcStateTrajectory>::SharedPtr mpc_xtraj_publisher_;
    rclcpp::Publisher<mpc_controller::msg::MpcInputTrajectory>::SharedPtr mpc_utraj_publisher_;
    rclcpp::Publisher<mpc_controller::msg::MpcKappaTrajectory>::SharedPtr mpc_kappa_traj_publisher_;
    rclcpp::Publisher<mpc_controller::msg::MpcSolverState>::SharedPtr mpc_solve_state_publisher_;

    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr spline_fit_publisher_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr xy_predict_traj_publisher_;

    // Reference Path to give to MPC
    std::vector<std::vector<double>> reference_path_;
    const double max_dist_to_prev_path_point_ = 5.0;

    // Step Time for controller publisher
    std::chrono::milliseconds dt_{std::chrono::milliseconds(DT_MS)};
    const double dt_seconds_ = dt_.count() / 1e3;

    // MPC horizon parameters
    int n_s_mpc_ = this->get_parameter("horizon.n_s").as_int();
    double s_max_mpc_ = this->get_parameter("horizon.s_max").as_double();
    double T_final_mpc_ = this->get_parameter("horizon.T_final").as_double();
    int N_horizon_mpc_ = this->get_parameter("horizon.N_horizon").as_int();

    // Solver parameters
    int sqp_max_iter_ = this->get_parameter("solver_options.nlp_solver_max_iter").as_int();
    int rti_phase_ = 0;
    int ws_first_qp = this->get_parameter("solver_options.qp_solver_warm_start").as_int();
    bool warm_start_first_qp_ = (bool) ws_first_qp;

    // Controller Model parameters
    double l_f_ = this->get_parameter("model.l_f").as_double();
    double l_r_ = this->get_parameter("model.l_r").as_double();
    double m_ = this->get_parameter("model.m").as_double();
    double Iz_ = this->get_parameter("model.Iz").as_double();
    double g_ = this->get_parameter("model.g").as_double();
    double D_tire_ = this->get_parameter("model.D_tire").as_double();
    double C_tire_ = this->get_parameter("model.C_tire").as_double();
    double B_tire_ = this->get_parameter("model.B_tire").as_double();
    double C_d_ = this->get_parameter("model.C_d").as_double();
    double C_r_ = this->get_parameter("model.C_r").as_double();

    // For visualization
    std::vector<std::vector<double>> xy_spline_points_;
    std::vector<std::vector<double>> xy_predict_points_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MPCControllerNode>());
  rclcpp::shutdown();
  return 0;
}