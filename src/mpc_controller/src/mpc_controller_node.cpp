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
#include "sim_backend/msg/sys_input.hpp"
#include "sim_backend/msg/vehicle_state.hpp"
#include "sim_backend/msg/ref_path.hpp"
#include "mpc_controller/msg/mpc_state_trajectory.hpp"
#include "mpc_controller/msg/mpc_input_trajectory.hpp"
#include "mpc_controller/msg/mpc_solver_state.hpp"
#include <Eigen/Dense>

// acados
#include "acados/utils/print.h"
#include "acados/utils/math.h"
#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"
#include "acados_solver_veh_dynamics_ode.h"

// blasfeo
#include "blasfeo/include/blasfeo_d_aux_ext_dep.h"

using namespace std::chrono_literals;

#define NX     VEH_DYNAMICS_ODE_NX
#define NZ     VEH_DYNAMICS_ODE_NZ
#define NU     VEH_DYNAMICS_ODE_NU
#define NP     VEH_DYNAMICS_ODE_NP
#define NBX    VEH_DYNAMICS_ODE_NBX
#define NBX0   VEH_DYNAMICS_ODE_NBX0
#define NBU    VEH_DYNAMICS_ODE_NBU
#define NSBX   VEH_DYNAMICS_ODE_NSBX
#define NSBU   VEH_DYNAMICS_ODE_NSBU
#define NSH    VEH_DYNAMICS_ODE_NSH
#define NSG    VEH_DYNAMICS_ODE_NSG
#define NSPHI  VEH_DYNAMICS_ODE_NSPHI
#define NSHN   VEH_DYNAMICS_ODE_NSHN
#define NSGN   VEH_DYNAMICS_ODE_NSGN
#define NSPHIN VEH_DYNAMICS_ODE_NSPHIN
#define NSBXN  VEH_DYNAMICS_ODE_NSBXN
#define NS     VEH_DYNAMICS_ODE_NS
#define NSN    VEH_DYNAMICS_ODE_NSN
#define NG     VEH_DYNAMICS_ODE_NG
#define NBXN   VEH_DYNAMICS_ODE_NBXN
#define NGN    VEH_DYNAMICS_ODE_NGN
#define NY0    VEH_DYNAMICS_ODE_NY0
#define NY     VEH_DYNAMICS_ODE_NY
#define NYN    VEH_DYNAMICS_ODE_NYN
#define NH     VEH_DYNAMICS_ODE_NH
#define NPHI   VEH_DYNAMICS_ODE_NPHI
#define NHN    VEH_DYNAMICS_ODE_NHN
#define NH0    VEH_DYNAMICS_ODE_NH0
#define NPHIN  VEH_DYNAMICS_ODE_NPHIN
#define NR     VEH_DYNAMICS_ODE_NR


class MPCController : public rclcpp::Node
{
  public:
    MPCController()
    : Node("mpc_controller",
            rclcpp::NodeOptions()
                .allow_undeclared_parameters(true)
                .automatically_declare_parameters_from_overrides(true))
    {
        bspline_coeff_ << -1, 3, -3, 1,
                         3, -6, 0, 4,
                         -3, 3, 3, 1,
                         1, 0, 0, 0;
        bspline_coeff_ = bspline_coeff_ / 6.0;
        std::cout << bspline_coeff_ << std::endl;
        ctrl_points_ << 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0;
        std::cout << ctrl_points_ << std::endl;
        // Init s_ref_mpc_, which is constant
        for (i_ = 0; i_<n_s_mpc_; i_++)
        {
            this->s_ref_mpc_.push_back(i_ * ds_);
            //RCLCPP_INFO_STREAM(this->get_logger(), "Adding: s_i=" << i_ * ds_);
        }
        for (i_ = 0; i_<n_s_mpc_; i_++)
        {
            this->curv_ref_mpc_.push_back(0.01);
            //RCLCPP_INFO_STREAM(this->get_logger(), "Adding: curv_i=" << 0.01);
        }
        // init dt in seconds from previously defined dt in chrono
        this->dt_seconds_ = dt_.count() / 1e3;

        this->l_f_ = this->get_parameter("model.l_f").as_double();
        this->l_r_ = this->get_parameter("model.l_r").as_double();
        this->m_ = this->get_parameter("model.m").as_double();
        this->Iz_ = this->get_parameter("model.Iz").as_double();
        this->g_ = this->get_parameter("model.g").as_double();
        this->D_tire_ = this->get_parameter("model.D_tire").as_double();
        this->C_tire_ = this->get_parameter("model.C_tire").as_double();
        this->B_tire_ = this->get_parameter("model.B_tire").as_double();
        this->C_d_ = this->get_parameter("model.C_d").as_double();
        this->C_r_ = this->get_parameter("model.C_r").as_double();

        // Init State Subscriber
        state_subscriber_ = this->create_subscription<sim_backend::msg::VehicleState>(
            "vehicle_state", 1, std::bind(&MPCController::state_update, this, std::placeholders::_1));

        // Init reference path subscriber
        ref_path_subscriber_ = this->create_subscription<sim_backend::msg::RefPath>(
            "reference_path", 1, std::bind(&MPCController::ref_path_update, this, std::placeholders::_1));

        spline_ref_path_publisher_ = this->create_publisher<sim_backend::msg::RefPath>("spline_ref_path", 10);

        mpc_xtraj_publisher_ = this->create_publisher<mpc_controller::msg::MpcStateTrajectory>("mpc_state_trajectory", 10);
        mpc_utraj_publisher_ = this->create_publisher<mpc_controller::msg::MpcInputTrajectory>("mpc_input_trajectory", 10);
        
        // Init Control Command Publisher and corresponding Timer with respecitve callback
        control_cmd_publisher_ = this->create_publisher<sim_backend::msg::SysInput>("vehicle_input", 10);
        control_cmd_timer_ = this->create_wall_timer(this->dt_, std::bind(&MPCController::control_callback, this));
        
        // ---------------------

        // allocate memory and get empty ocp capsule
        RCLCPP_INFO_STREAM(this->get_logger(), "Allocating Memory for solver capsule ...");
        acados_ocp_capsule_ = veh_dynamics_ode_acados_create_capsule();

        // there is an opportunity to change the number of shooting intervals in C without new code generation
        N_ = VEH_DYNAMICS_ODE_N;
        
        // allocate the array and fill it accordingly
        new_time_steps_ = NULL;
        RCLCPP_INFO_STREAM(this->get_logger(), "Creating Solver and filling capsule ...");
        solver_status_ = veh_dynamics_ode_acados_create_with_discretization(acados_ocp_capsule_, N_, new_time_steps_);

        if (solver_status_)
        {
            RCLCPP_INFO_STREAM(this->get_logger(), "veh_dynamics_ode_acados_create() returned solver_status_ " << solver_status_ << ". Exiting.\n");
            // rclcpp::shutdown();
        }

        nlp_config_ = veh_dynamics_ode_acados_get_nlp_config(acados_ocp_capsule_);
        nlp_dims_ = veh_dynamics_ode_acados_get_nlp_dims(acados_ocp_capsule_);
        nlp_in_ = veh_dynamics_ode_acados_get_nlp_in(acados_ocp_capsule_);
        nlp_out_ = veh_dynamics_ode_acados_get_nlp_out(acados_ocp_capsule_);
        nlp_solver_ = veh_dynamics_ode_acados_get_nlp_solver(acados_ocp_capsule_);
        nlp_opts_ = veh_dynamics_ode_acados_get_nlp_opts(acados_ocp_capsule_);

        // Solver options set
        this->rti_phase_ = 1;
        this->warm_start_first_qp_ = true;
        this->nlp_solver_max_iter_ = this->get_parameter("solver_options.nlp_solver_max_iter").as_int();

        ocp_nlp_solver_opts_set(this->nlp_config_, this->nlp_opts_, "rti_phase", &this->rti_phase_);
        //ocp_nlp_solver_opts_set(this->nlp_config_, this->nlp_opts_, "nlp_solver_max_iter", &this->nlp_solver_max_iter_);
        ocp_nlp_solver_opts_set(this->nlp_config_, this->nlp_opts_, "warm_start_first_qp", &this->warm_start_first_qp_);

        setvbuf(stdout, NULL, _IONBF, 0);

        solver_initialized_ = false;

        //this->control_callback();

        // ==========================================
        RCLCPP_INFO_STREAM(this->get_logger(), "Node " << this->get_name() << " initialized.");
    }

  private:

    void ref_path_update(const sim_backend::msg::RefPath & refpath_msg){
        RCLCPP_INFO_STREAM(this->get_logger(), "Reference path update.");
        // Create container for single 2D point with data type from message
        auto path_pos = sim_backend::msg::Point2D();
        // Create container for single 2D point with internal data type
        Eigen::Vector2d ref_point;

        // Check amount of reference points
        if(refpath_msg.ref_path.size() < 2){
            RCLCPP_INFO_STREAM(this->get_logger(), "Reference path size less than 2, no ref path update.");
            return;
        }else{
            // RCLCPP_INFO_STREAM(this->get_logger(), "Clearing ref. points");
            // Clear all 2D reference points
            this->ref_points_.clear();
            this->t_ref_spline_.clear();
            this->curv_ref_spline_.clear();
            this->xy_ref_spline_.clear();
            this->dxy_ref_spline_.clear();
            this->ddxy_ref_spline_.clear();
            this->s_ref_spline_.clear();
        }

        // get all reference points
        for (size_t i = 0; i < refpath_msg.ref_path.size(); i++){
            path_pos = refpath_msg.ref_path[i];
            ref_point[0] = path_pos.point_2d[0];
            ref_point[1] = path_pos.point_2d[1];
            if(i > 0){
                if(sqrt(pow(ref_point[0] - this->ref_points_[i-1][0], 2.0) 
                    + pow(ref_point[1] - this->ref_points_[i-1][1], 2.0)) > 5.0){
                    RCLCPP_INFO_STREAM(this->get_logger(), "Breaking out because: ref_point = (" << ref_point[0] << ", " << ref_point[1] << ") too far from last point.");
                    break;
                }
            }
            this->ref_points_.push_back(ref_point);
            RCLCPP_INFO_STREAM(this->get_logger(), "Got: ref_point = (" << ref_point[0] << ", " << ref_point[1] << ")");
        }
        if(this->ref_points_.size() + 2 < 4){
            RCLCPP_INFO_STREAM(this->get_logger(), "Distance-filtered ref-path size too small. Quitting Callback.");
            return;
        }

        // create an additional point in the beginning of ref_points
        ref_point[0] = 2 * this->ref_points_[0][0] - this->ref_points_[1][0];
        ref_point[1] = 2 * this->ref_points_[0][1] - this->ref_points_[1][1];
        this->ref_points_.insert(this->ref_points_.begin(), ref_point);
        RCLCPP_INFO_STREAM(this->get_logger(), "Added: ref_point[0] = (" << this->ref_points_[0][0] << ", " << this->ref_points_[0][1] << ")");
        // create an additional point at the end of ref_points
        ref_point[0] = 2 * this->ref_points_[this->ref_points_.size()-1][0] - this->ref_points_[this->ref_points_.size()-2][0];
        ref_point[1] = 2 * this->ref_points_[this->ref_points_.size()-1][1] - this->ref_points_[this->ref_points_.size()-2][1];
        this->ref_points_.push_back(ref_point);
        RCLCPP_INFO_STREAM(this->get_logger(), "Added: ref_point[end] = (" << this->ref_points_[this->ref_points_.size()-1][0] << ", " << this->ref_points_[this->ref_points_.size()-1][1] << ")");

        // Calculate number of control points
        size_t num_control_points = this->ref_points_.size();

        // count number of points in the reference path
        this->n_s_spline_ = (int)num_control_points - 3;

        // Generate knot vector with necessary number of points
        for(double t = 0.0; t <= (double)this->n_s_spline_; t+=this->dt_spline_){
            this->t_ref_spline_.push_back(t);
            //RCLCPP_INFO_STREAM(this->get_logger(), "Generated: t_i = " << t);
        }

        // path coordinate s along spline-fitted reference path
        double s_ref_length = 0.0;
        // instant curvature
        double curv_instant = 0.0;
        for(size_t i = 0; i < (size_t)this->t_ref_spline_.size(); i++){

            // get integer value of knot vector
            this->j_ = (int) t_ref_spline_[i];
            // RCLCPP_INFO_STREAM(this->get_logger(), "t integer: " << this->j_);

            // Fill the control points 2x4 matrix
            // rows
            for(this->i_=0; this->i_ < 2; this->i_++){
                // columns
                for(this->k_=0; this->k_ < 4; this->k_++){
                    this->ctrl_points_(this->i_, this->k_) = this->ref_points_[this->j_+this->k_][this->i_];
                    // RCLCPP_INFO_STREAM(this->get_logger(), "control points matrix: " << this->ctrl_points_(this->i_, this->k_) << " at row: " << this->i_ << " and col: " << this->k_);
                }
            }

            // get fractional vatue of knot vector
            this->t_frac_ = t_ref_spline_[i] - this->j_;
            // RCLCPP_INFO_STREAM(this->get_logger(), "t fraction: " << this->t_frac_);

            // fill knot polynomial vector
            this->t_vec_ << pow(this->t_frac_, 3),
                            pow(this->t_frac_, 2),
                            this->t_frac_,
                            1.0;
            // fill 1st deriv knot polynomial vector
            this->dt_vec_ << 3 * pow(this->t_frac_, 2),
                            2 * this->t_frac_,
                            1.0,
                            0.0;
            // fill 2nd deriv knot polynomial vector
            this->ddt_vec_ << 6 * this->t_frac_,
                            2.0,
                            0.0,
                            0.0;
            this->xy_ref_spline_.push_back(this->ctrl_points_ * this->bspline_coeff_ * this->t_vec_);
            this->dxy_ref_spline_.push_back(this->ctrl_points_ * this->bspline_coeff_ * this->dt_vec_);
            this->ddxy_ref_spline_.push_back(this->ctrl_points_ * this->bspline_coeff_ * this->ddt_vec_);
            
            // Discretely integrate up the path coordinate s
            if(i > 0){
                s_ref_length += sqrt(pow(this->xy_ref_spline_[i][0] - this->xy_ref_spline_[i-1][0], 2.0)
                                    + pow(this->xy_ref_spline_[i][1] - this->xy_ref_spline_[i-1][1], 2.0));
            }
            this->s_ref_spline_.push_back(s_ref_length);

            RCLCPP_INFO_STREAM(this->get_logger(), "cumsum s: " << s_ref_length);

            RCLCPP_INFO_STREAM(this->get_logger(), "Spline interpolation: " << this->xy_ref_spline_[i][0] << " " << this->xy_ref_spline_[i][1]);

            // Calculate curvature from first and second derivatives
            curv_instant = (this->dxy_ref_spline_[i][0] * this->ddxy_ref_spline_[i][1] -
                            this->dxy_ref_spline_[i][1] * this->ddxy_ref_spline_[i][0]);
            curv_instant /= (pow((pow(this->dxy_ref_spline_[i][0], 2.0) + 
                                  pow(this->dxy_ref_spline_[i][1], 2.0)), 1.5));
            this->curv_ref_spline_.push_back(curv_instant);
            RCLCPP_INFO_STREAM(this->get_logger(), "Curvature here: " << curv_instant);
        }

        int idx_next_s = 0;
        // Fill Curvature for MPC
        // - if s_max is larger than the integrated s_ref_length, need to fill curv_ref_mpc (with zeros?)
        // - if s_max is smaller than the integrated s_ref_length, no need to take further steps 
        //   since curv_ref_mpc is interpolated correctly
        for (this->i_ = 0; this->i_<this->n_s_mpc_; this->i_++)
        {
            // Iterate through the previously calculated s_ref vector and go 
            // for first entry that is larger than in mpc s ref, interpolate between this and the previous point
            for(this->j_ = 0; this->j_ < (int)this->s_ref_spline_.size(); this->j_++){
                // As soon as s_ref_mpc (constant) is larger than s_ref, calculate curvature as 
                // linear interpolation and break out
                if (this->s_ref_mpc_[this->i_] < this->s_ref_spline_[this->j_]){
                    idx_next_s = this->j_;
                    // Linearly interpolate between given s_ref values on the spline to get curvature
                    this->curv_ref_mpc_[this->i_] = this->curv_ref_spline_[idx_next_s - 1] + 
                        (this->s_ref_mpc_[this->i_] - this->s_ref_spline_[idx_next_s - 1])/
                        (this->s_ref_spline_[idx_next_s] - this->s_ref_spline_[idx_next_s - 1]) 
                        * (this->curv_ref_spline_[idx_next_s] - this->curv_ref_spline_[idx_next_s - 1]);
                    // go for next s_ref_mpc curvature
                    break;
                }else if (this->s_max_mpc_ <= this->s_ref_spline_[this->j_]){
                    // Fill with zeros if the reference spline length is larger than the maximum s_ref_mpc
                    this->curv_ref_mpc_[this->i_] = this->curv_ref_mpc_[this->i_ - 1];
                }
            }
            //RCLCPP_INFO_STREAM(this->get_logger(), "Adding: curv_i=" << this->curv_ref_mpc_[this->i_]);
        }
    }

    void control_callback()
    {
        RCLCPP_INFO_STREAM(this->get_logger(), "Control Callback called.");
        RCLCPP_INFO_STREAM(this->get_logger(), "Setting x0");
        // initial condition
        int idxbx0[NBX0];
        idxbx0[0] = 0;
        idxbx0[1] = 1;
        idxbx0[2] = 2;
        idxbx0[3] = 3;
        idxbx0[4] = 4;
        idxbx0[5] = 5;
        idxbx0[6] = 6;
        idxbx0[7] = 7;

        double lbx0[NBX0];
        double ubx0[NBX0];
        lbx0[0] = this->x_[0];
        ubx0[0] = this->x_[0];
        lbx0[1] = this->x_[1];
        ubx0[1] = this->x_[1];
        lbx0[2] = this->x_[2];
        ubx0[2] = this->x_[2];
        lbx0[3] = this->x_[3];
        ubx0[3] = this->x_[3];
        lbx0[4] = this->x_[4];
        ubx0[4] = this->x_[4];
        lbx0[5] = this->x_[5];
        ubx0[5] = this->x_[5];
        lbx0[6] = this->x_[6];
        ubx0[6] = this->x_[6];
        lbx0[7] = this->x_[7];
        ubx0[7] = this->x_[7];
        RCLCPP_INFO_STREAM(this->get_logger(), "init with x_sim done");

        ocp_nlp_constraints_model_set(this->nlp_config_, this->nlp_dims_, this->nlp_in_, 0, "idxbx", idxbx0);
        ocp_nlp_constraints_model_set(this->nlp_config_, this->nlp_dims_, this->nlp_in_, 0, "lbx", lbx0);
        ocp_nlp_constraints_model_set(this->nlp_config_, this->nlp_dims_, this->nlp_in_, 0, "ubx", ubx0);

        // initialization for state values
        double x_init[NX];
        x_init[0] = this->x_[0];
        x_init[1] = this->x_[1];
        x_init[2] = this->x_[2];
        x_init[3] = this->x_[3];
        x_init[4] = this->x_[4];
        x_init[5] = this->x_[5];
        x_init[6] = this->x_[6];
        x_init[7] = this->x_[7];

        RCLCPP_INFO_STREAM(this->get_logger(), "Setting u0 ...");
        // initial value for control input
        double u0[NU];
        u0[0] = 0.0;
        u0[1] = 0.0;

        RCLCPP_INFO_STREAM(this->get_logger(), "Setting p0 ...");
        // set parameters
        double p[NP];
        p[0] = this->m_;
        p[1] = this->g_;
        p[2] = this->l_f_;
        p[3] = this->l_r_;
        p[4] = this->Iz_;
        p[5] = this->B_tire_;
        p[6] = this->C_tire_;
        p[7] = this->D_tire_;
        p[8] = this->C_d_;
        p[9] = this->C_r_;
        // if(this->x_[3]>0.5){
        //     p[10] = 1.0;
        // }else{
        //     p[10] = 0.0; 
        // }
        p[10] = 0.0;

        // Init curvature ref
        for(this->j_=11; this->j_<NP; this->j_++)
        {
            p[this->j_] = this->curv_ref_mpc_[this->j_];
        }

        RCLCPP_INFO_STREAM(this->get_logger(), "Setting parameters ...");
        for (this->i_ = 0; this->i_ <= this->N_; this->i_++)
        {
            veh_dynamics_ode_acados_update_params(this->acados_ocp_capsule_, this->i_, p, NP);
        }

        // prepare evaluation
        double kkt_norm_inf;
        double elapsed_time;
        double total_elapsed_time = 0.0;
        int sqp_iter;

        RCLCPP_INFO_STREAM(this->get_logger(), "Initializing solution (x, u, x_N)");
        // initialize solution
        for (this->i_ = 0; this->i_ < this->N_; this->i_++)
        {
            ocp_nlp_out_set(this->nlp_config_, this->nlp_dims_, this->nlp_out_, i_, "x", x_init);
            ocp_nlp_out_set(this->nlp_config_, this->nlp_dims_, this->nlp_out_, i_, "u", u0);
        }
        ocp_nlp_out_set(this->nlp_config_, this->nlp_dims_, this->nlp_out_, this->N_, "x", x_init);

        if(!solver_initialized_){
            this->rti_phase_ = 1;
            RCLCPP_INFO_STREAM(this->get_logger(), "=== Solver Call with rti_phase " << this->rti_phase_);
            ocp_nlp_solver_opts_set(this->nlp_config_, this->nlp_opts_, "rti_phase", &this->rti_phase_);
            this->solver_status_ = veh_dynamics_ode_acados_solve(this->acados_ocp_capsule_);
            ocp_nlp_get(this->nlp_config_, this->nlp_solver_, "time_tot", &elapsed_time);
            RCLCPP_INFO_STREAM(this->get_logger(), "Elapsed time: " << elapsed_time*1000 << " ms");
            RCLCPP_INFO_STREAM(this->get_logger(), "veh_dynamics_ode_acados_solve() status: " << this->solver_status_);
            RCLCPP_INFO_STREAM(this->get_logger(), "\n");
            total_elapsed_time += elapsed_time;

            ocp_nlp_constraints_model_set(this->nlp_config_, this->nlp_dims_, this->nlp_in_, 0, "idxbx", idxbx0);
            ocp_nlp_constraints_model_set(this->nlp_config_, this->nlp_dims_, this->nlp_in_, 0, "lbx", lbx0);
            ocp_nlp_constraints_model_set(this->nlp_config_, this->nlp_dims_, this->nlp_in_, 0, "ubx", ubx0);
            solver_initialized_ = true;
        }

        this->rti_phase_ = 2;
        RCLCPP_INFO_STREAM(this->get_logger(), "=== Solver Call with rti_phase " << this->rti_phase_);
        ocp_nlp_solver_opts_set(this->nlp_config_, this->nlp_opts_, "rti_phase", &this->rti_phase_);
        this->solver_status_ = veh_dynamics_ode_acados_solve(this->acados_ocp_capsule_);
        ocp_nlp_get(this->nlp_config_, this->nlp_solver_, "time_tot", &elapsed_time);
        RCLCPP_INFO_STREAM(this->get_logger(), "Elapsed time: " << elapsed_time*1000 << " ms");
        RCLCPP_INFO_STREAM(this->get_logger(), "veh_dynamics_ode_acados_solve() status: " << this->solver_status_);
        RCLCPP_INFO_STREAM(this->get_logger(), "\n");
        total_elapsed_time += elapsed_time;

        /* ======================================================================== */

        //RCLCPP_INFO_STREAM(this->get_logger(), "SYSTEM TRAJECTORIES:");
        auto state_traj_msg = mpc_controller::msg::MpcStateTrajectory();
        auto input_traj_msg = mpc_controller::msg::MpcInputTrajectory();
        /* Get solution */ //--> TODO: Put into class member variable
        //RCLCPP_INFO_STREAM(this->get_logger(), "--- x_traj ---");
        for (this->i_ = 0; this->i_ <= this->nlp_dims_->N; this->i_++){
            ocp_nlp_out_get(this->nlp_config_, this->nlp_dims_, this->nlp_out_, this->i_, "x", &this->x_traj[this->i_]);
            
            //RCLCPP_INFO_STREAM(this->get_logger(), "STAGE " << this->i_ << ":");
            state_traj_msg.s.push_back(x_traj[this->i_][0]);
            state_traj_msg.n.push_back(x_traj[this->i_][1]);
            state_traj_msg.mu.push_back(x_traj[this->i_][2]);
            state_traj_msg.vx_c.push_back(x_traj[this->i_][3]);
            state_traj_msg.vy_c.push_back(x_traj[this->i_][4]);
            state_traj_msg.dpsi.push_back(x_traj[this->i_][5]);
            state_traj_msg.del_s.push_back(x_traj[this->i_][6]);
            state_traj_msg.fx_m.push_back(x_traj[this->i_][7]);
            //RCLCPP_INFO_STREAM(this->get_logger(), "===");
        }
        //RCLCPP_INFO_STREAM(this->get_logger(), "--- u_traj ---");
        for (this->i_ = 0; this->i_ < this->nlp_dims_->N; this->i_++){
            ocp_nlp_out_get(this->nlp_config_, this->nlp_dims_, this->nlp_out_, this->i_, "u", &this->u_traj[this->i_]);
            
            // RCLCPP_INFO_STREAM(this->get_logger(), "STAGE " << this->i_ << ":");
            // for(this->j_ = 0; this->j_ < NU; this->j_++){
            //     RCLCPP_INFO_STREAM(this->get_logger(), u_traj[this->i_][this->j_]);
            // }
            input_traj_msg.ddel_s.push_back(u_traj[this->i_][0]);
            input_traj_msg.dfx_m.push_back(u_traj[this->i_][1]);
            // RCLCPP_INFO_STREAM(this->get_logger(), "===");
        }
        mpc_xtraj_publisher_->publish(state_traj_msg);
        mpc_utraj_publisher_->publish(input_traj_msg);

        /* ======================================================================== */

        // get KKT inf. norm
        ocp_nlp_out_get(this->nlp_config_, this->nlp_dims_, this->nlp_out_, 0, "kkt_norm_inf", &kkt_norm_inf);
        // get SQP iterations
        ocp_nlp_get(nlp_config_, this->nlp_solver_, "sqp_iter", &sqp_iter);

        veh_dynamics_ode_acados_print_stats(acados_ocp_capsule_);

        RCLCPP_INFO_STREAM(this->get_logger(), "===\nSolver info: sqp_iter=" << sqp_iter << ", kkt_inf_norm=" << kkt_norm_inf);

        double u_eval[] = {0.0, 0.0};
        double x_eval[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        // solve time delay compensation
        double dt_mpc = (this->T_final_mpc_ / this->N_horizon_mpc_);

        int stage_to_eval = (int) (total_elapsed_time / dt_mpc);
        stage_to_eval = 0;
        RCLCPP_INFO_STREAM(this->get_logger(), "Eval stage: " << stage_to_eval);
        // evaluate u and x at that stage
        ocp_nlp_out_get(this->nlp_config_, this->nlp_dims_, this->nlp_out_, stage_to_eval, "u", &u_eval);
        ocp_nlp_out_get(this->nlp_config_, this->nlp_dims_, this->nlp_out_, stage_to_eval, "x", &x_eval);

        RCLCPP_INFO_STREAM(this->get_logger(), " u_eval: (" << u_eval[0] << ", " << u_eval[1] << ")");
        RCLCPP_INFO_STREAM(this->get_logger(), " Fx_eval: " << x_eval[6]);
        RCLCPP_INFO_STREAM(this->get_logger(), " del_s_eval: " << x_eval[7]);

        auto veh_input_msg = sim_backend::msg::SysInput();
        if(this->solver_status_ == 0 || this->solver_status_ == 5){
            // veh_input_msg.fx_r = (this->x_[6] + dt_mpc * (stage_to_eval + 1) * u_eval[0]) / 2.0;
            // veh_input_msg.fx_f = (this->x_[6] + dt_mpc * (stage_to_eval + 1) * u_eval[0]) / 2.0;
            // veh_input_msg.del_s = (this->x_[7] + dt_mpc * (stage_to_eval + 1) * u_eval[1]);
            veh_input_msg.fx_r = (this->x_[6] + dt_seconds_ * u_eval[0]) / 2.0;
            veh_input_msg.fx_f = (this->x_[6] + dt_seconds_ * u_eval[0]) / 2.0;
            veh_input_msg.del_s = (this->x_[7] + dt_seconds_ * u_eval[1]);
        }else{
            veh_input_msg.fx_r = 0.0;
            veh_input_msg.fx_f = 0.0;
            veh_input_msg.del_s = 0.0;
        }
        control_cmd_publisher_->publish(veh_input_msg);

        // Create container for single 2D point with data type from message
        auto path_pos = sim_backend::msg::Point2D();
        auto spline_ref_path_msg = sim_backend::msg::RefPath();
        for(this->j_ = 0; this->j_ < (int)this->xy_ref_spline_.size(); this->j_++){
            path_pos.point_2d[0] = this->xy_ref_spline_[this->j_][0];
            path_pos.point_2d[1] = this->xy_ref_spline_[this->j_][1];
            spline_ref_path_msg.ref_path.push_back(path_pos);
        }
        spline_ref_path_publisher_->publish(spline_ref_path_msg);
    }

    void state_update(const sim_backend::msg::VehicleState & state_msg)
    {
        RCLCPP_INFO_STREAM(this->get_logger(), "State Update.");
        double x_c = state_msg.x_c;
        double y_c = state_msg.y_c;
        double psi = state_msg.psi;

        RCLCPP_INFO_STREAM(this->get_logger(), "car heading update");
        double dx_car_heading = cos(psi);
        double dy_car_heading = sin(psi);

        RCLCPP_INFO_STREAM(this->get_logger(), "d(x,y) path update");
        double dx_path = dx_car_heading;
        double dy_path = dy_car_heading;
        if(this->dxy_ref_spline_.size() > 0){
            dx_path = this->dxy_ref_spline_[0][0];
            dy_path = this->dxy_ref_spline_[0][1];
        }

        this->x_[0] = 0.0; // s
        this->x_[1] = 0.0; // n

        // TODO: Calculate mu from current reference path first entries in dxy and heading 
        // https://wumbo.net/formulas/angle-between-two-vectors-2d/
        RCLCPP_INFO_STREAM(this->get_logger(), "mu update");
        this->x_[2] = atan2(dy_car_heading * dx_path - dx_car_heading * dy_path,
                            dx_car_heading * dx_path + dy_car_heading * dy_path); // mu

        RCLCPP_INFO_STREAM(this->get_logger(), "velocities update");

        this->x_[3] = state_msg.dx_c * cos(psi) + state_msg.dy_c * sin(psi); // vx
        this->x_[4] = state_msg.dy_c * cos(psi) - state_msg.dy_c * sin(psi); // vy
        this->x_[5] = state_msg.dpsi; // r or dpsi

        // TODO: Calculate Fx and del_s (maybe as member variables that get integrated in control_callback)
        // then we don't have to update them here.
        RCLCPP_INFO_STREAM(this->get_logger(), "input state updates ");
        this->x_[6] = state_msg.fx_f_act + state_msg.fx_r_act; // Fx
        this->x_[7] = state_msg.del_s_ref; // del_s
    }

    // Subscriber to state message published at faster frequency
    rclcpp::Subscription<sim_backend::msg::VehicleState>::SharedPtr state_subscriber_;
    
    // Subscriber to ref path provided by "perception"
    rclcpp::Subscription<sim_backend::msg::RefPath>::SharedPtr ref_path_subscriber_;

    // Timer for control command publishing
    rclcpp::TimerBase::SharedPtr control_cmd_timer_;

    // Control command publisher
    rclcpp::Publisher<sim_backend::msg::SysInput>::SharedPtr control_cmd_publisher_;

    // Interpolated ref. path publisher
    rclcpp::Publisher<sim_backend::msg::RefPath>::SharedPtr spline_ref_path_publisher_;

    // mpc trajectory publishers
    rclcpp::Publisher<mpc_controller::msg::MpcStateTrajectory>::SharedPtr mpc_xtraj_publisher_;
    rclcpp::Publisher<mpc_controller::msg::MpcInputTrajectory>::SharedPtr mpc_utraj_publisher_;

    rclcpp::Publisher<mpc_controller::msg::MpcSolverState>::SharedPtr mpc_solver_state_publisher_;

    // Step Time for controller publisher
    std::chrono::milliseconds dt_{std::chrono::milliseconds(100)};
    double dt_seconds_;

    // Current State for MPC
    // [s:0, n:1, mu:2, vx:3, vy:4, dpsi:5, Fx:6, dels:7]
    double x_[8] = {0.0};

    // Reference Path ahead of the vehicle
    std::vector<Eigen::Vector2d> ref_points_;

    // some iterables
    int i_ = 0, j_ = 0, k_ = 0;

    // Solver interface path
    const int n_s_mpc_ = this->get_parameter("horizon.n_s").as_int();
    const double s_max_mpc_ = this->get_parameter("horizon.s_max").as_double();
    const double T_final_mpc_ = this->get_parameter("horizon.T_final").as_double();
    const int N_horizon_mpc_ = this->get_parameter("horizon.N_horizon").as_int();
    const double ds_ = s_max_mpc_ / (n_s_mpc_ - 1);

    bool solver_initialized_;

    // s vector
    std::vector<double> s_ref_mpc_;
    // curvature vector
    std::vector<double> curv_ref_mpc_;

    Eigen::Matrix4d bspline_coeff_;
    Eigen::Vector4d t_vec_;
    Eigen::Vector4d dt_vec_;
    Eigen::Vector4d ddt_vec_;
    double t_frac_;

    Eigen::Matrix<double, 2, 4> ctrl_points_;

    double x_traj[VEH_DYNAMICS_ODE_N + 1][NX];
    double u_traj[VEH_DYNAMICS_ODE_N][NU];

    // Spline vectors
    // knot step (uniform spline)
    double dt_spline_ = 0.2;
    // knot vector for b-spline fit to ref path
    std::vector<double> t_ref_spline_;
    // s vector of spline
    std::vector<double> s_ref_spline_;
    // curvature vector on spline
    std::vector<double> curv_ref_spline_;
    // spline xy-points, derivatives
    std::vector<Eigen::Vector2d> xy_ref_spline_;
    std::vector<Eigen::Vector2d> dxy_ref_spline_;
    std::vector<Eigen::Vector2d> ddxy_ref_spline_;
    int n_s_spline_;

    // Controller Model parameters
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

    // Solver variables
    ocp_nlp_config *nlp_config_;
    ocp_nlp_dims *nlp_dims_;
    ocp_nlp_in *nlp_in_;
    ocp_nlp_out *nlp_out_;
    ocp_nlp_solver *nlp_solver_;
    void *nlp_opts_;

    veh_dynamics_ode_solver_capsule *acados_ocp_capsule_;

    int rti_phase_;
    bool warm_start_first_qp_;
    int nlp_solver_max_iter_;

    int N_;
    double* new_time_steps_;
    int solver_status_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MPCController>());
  rclcpp::shutdown();
  return 0;
}