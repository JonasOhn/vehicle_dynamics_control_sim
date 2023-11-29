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

// acados
#include "acados/utils/print.h"
#include "acados/utils/math.h"
#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"
#include "acados_solver_veh_dynamics_ode.h"
#include "acados_c/sim_interface.h"
#include "acados_sim_solver_veh_dynamics_ode.h"

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
        // init dt in seconds from previously defined dt in chrono
        this->dt_seconds_ = dt_.count() / 1e3;

        this->l_f_ = this->get_parameter("l_f").as_double();
        this->l_r_ = this->get_parameter("l_r").as_double();
        this->m_ = this->get_parameter("m").as_double();
        this->Iz_ = this->get_parameter("Iz").as_double();
        this->g_ = this->get_parameter("g").as_double();
        this->D_tire_ = this->get_parameter("D_tire").as_double();
        this->C_tire_ = this->get_parameter("C_tire").as_double();
        this->B_tire_ = this->get_parameter("B_tire").as_double();
        this->C_d_ = this->get_parameter("C_d").as_double();
        this->C_r_ = this->get_parameter("C_r").as_double();

        // Init State Subscriber
        state_subscriber_ = this->create_subscription<sim_backend::msg::VehicleState>(
            "vehicle_state", 1, std::bind(&MPCController::cartesian_state_update, this, std::placeholders::_1));

        // Init reference path subscriber
        ref_path_subscriber_ = this->create_subscription<sim_backend::msg::RefPath>(
            "reference_path", 1, std::bind(&MPCController::ref_path_update, this, std::placeholders::_1));
        
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

        setvbuf(stdout, NULL, _IONBF, 0);

        // ==========================================
        RCLCPP_INFO_STREAM(this->get_logger(), "Checking Solver Initialization ...");
        this->check_solver_init();

        if (solver_status_ == ACADOS_SUCCESS)
        {
            RCLCPP_INFO_STREAM(this->get_logger(), "successfully initialized mpc solver.");
        }else{
            RCLCPP_INFO_STREAM(this->get_logger(), "something went wrong in the mpc initialization. solver_status_=" << solver_status_);
            // rclcpp::shutdown();
        }
        RCLCPP_INFO_STREAM(this->get_logger(), "Node " << this->get_name() << " initialized.");
    }

  private:

    void print_matrix(int row, int col, double *A, int lda)
    {
        int i, j;
        for(j=0; j<col; j++)
        {
            for(i=0; i<row; i++)
            {
                printf("%e \t", A[i+lda*j]);
            }
            printf("\n");
        }
        printf("\n");
    }

    void check_solver_init()
    {
        RCLCPP_INFO_STREAM(this->get_logger(), "Setting x0 ...");
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
        lbx0[0] = 0;
        ubx0[0] = 0;
        lbx0[1] = 0;
        ubx0[1] = 0;
        lbx0[2] = 0;
        ubx0[2] = 0;
        lbx0[3] = 0;
        ubx0[3] = 0;
        lbx0[4] = 0;
        ubx0[4] = 0;
        lbx0[5] = 0;
        ubx0[5] = 0;
        lbx0[6] = 0;
        ubx0[6] = 0;
        lbx0[7] = 0;
        ubx0[7] = 0;

        ocp_nlp_constraints_model_set(this->nlp_config_, this->nlp_dims_, this->nlp_in_, 0, "idxbx", idxbx0);
        ocp_nlp_constraints_model_set(this->nlp_config_, this->nlp_dims_, this->nlp_in_, 0, "lbx", lbx0);
        ocp_nlp_constraints_model_set(this->nlp_config_, this->nlp_dims_, this->nlp_in_, 0, "ubx", ubx0);

        // initialization for state values
        double x_init[NX];
        x_init[0] = 0.0;
        x_init[1] = 0.0;
        x_init[2] = 0.0;
        x_init[3] = 0.0;
        x_init[4] = 0.0;
        x_init[5] = 0.0;
        x_init[6] = 0.0;
        x_init[7] = 0.0;

        RCLCPP_INFO_STREAM(this->get_logger(), "Setting u0 ...");
        // initial value for control input
        double u0[NU];
        u0[0] = 0.0;
        u0[1] = 0.0;

        RCLCPP_INFO_STREAM(this->get_logger(), "Setting p0 ...");
        // set parameters
        double p[NP];
        p[0] = 220;
        p[1] = 9.81;
        p[2] = 0.8;
        p[3] = 0.7;
        p[4] = 100;
        p[5] = 9.5;
        p[6] = 1.7;
        p[7] = -1;
        p[8] = 1.133;
        p[9] = 0.5;
        p[10] = 0;
        p[11] = 0;
        p[12] = 0;
        p[13] = 0;
        p[14] = 0;
        p[15] = 0;
        p[16] = 0;
        p[17] = 0;
        p[18] = 0;
        p[19] = 0;
        p[20] = 0;
        p[21] = 0;
        p[22] = 0;
        p[23] = 0;
        p[24] = 0;
        p[25] = 0;
        p[26] = 0.01;
        p[27] = 0.01;
        p[28] = 0.01;
        p[29] = 0.01;
        p[30] = 0.01;
        p[31] = 0.01;
        p[32] = 0.01;
        p[33] = 0.01;
        p[34] = 0.01;
        p[35] = 0.01;
        p[36] = 0.01;
        p[37] = 0.01;
        p[38] = 0.01;
        p[39] = 0.01;
        p[40] = 0.01;

        for (int ii = 0; ii <= this->N_; ii++)
        {
            RCLCPP_INFO_STREAM(this->get_logger(), "Setting parameters for stage " << ii);
            veh_dynamics_ode_acados_update_params(this->acados_ocp_capsule_, ii, p, NP);
        }

        RCLCPP_INFO_STREAM(this->get_logger(), "Preparing Evaluation ...");
        // prepare evaluation
        int NTIMINGS = 5;
        double min_time = 1e6;
        double kkt_norm_inf;
        double elapsed_time;
        int sqp_iter;

        double *xtraj = new double[NX * (N_+1)];
        double *utraj = new double[NU * N_];

        // solve ocp in loop
        int rti_phase = 1;
        RCLCPP_INFO_STREAM(this->get_logger(), "RTI Phase set to: " << rti_phase);

        for (int ii = 0; ii < NTIMINGS; ii++)
        {
            RCLCPP_INFO_STREAM(this->get_logger(), "Initializing solution (x_i, u_i, x_N) for i = " << ii);
            // initialize solution
            for (int i = 0; i < N_; i++)
            {
                ocp_nlp_out_set(this->nlp_config_, this->nlp_dims_, this->nlp_out_, i, "x", x_init);
                ocp_nlp_out_set(this->nlp_config_, this->nlp_dims_, this->nlp_out_, i, "u", u0);
            }
            ocp_nlp_out_set(this->nlp_config_, this->nlp_dims_, this->nlp_out_, this->N_, "x", x_init);

            RCLCPP_INFO_STREAM(this->get_logger(), "Setting Solver OPtions ...");
            ocp_nlp_solver_opts_set(this->nlp_config_, this->nlp_opts_, "rti_phase", &rti_phase);

            RCLCPP_INFO_STREAM(this->get_logger(), "Solver Call ...");
            this->solver_status_ = veh_dynamics_ode_acados_solve(acados_ocp_capsule_);

            RCLCPP_INFO_STREAM(this->get_logger(), "Get elapsed time of solve ...");
            ocp_nlp_get(this->nlp_config_, this->nlp_solver_, "time_tot", &elapsed_time);
            min_time = MIN(elapsed_time, min_time);
        }

        RCLCPP_INFO_STREAM(this->get_logger(), "Print Statistics ... ");
        /* print solution and statistics */
        for (int ii = 0; ii <= this->nlp_dims_->N; ii++){
            ocp_nlp_out_get(this->nlp_config_, this->nlp_dims_, this->nlp_out_, ii, "x", &xtraj[ii*NX]);
        }
        for (int ii = 0; ii < nlp_dims_->N; ii++){
            ocp_nlp_out_get(this->nlp_config_, this->nlp_dims_, this->nlp_out_, ii, "u", &utraj[ii*NU]);
        }

        RCLCPP_INFO_STREAM(this->get_logger(), "\nSYSTEM TRAJECTORIES:\n");

        RCLCPP_INFO_STREAM(this->get_logger(), "\n--- xtraj ---\n");
        print_matrix( NX, this->N_+1, xtraj, NX);

        RCLCPP_INFO_STREAM(this->get_logger(), "\n--- utraj ---\n");
        print_matrix( NU, this->N_, utraj, NU );

        RCLCPP_INFO_STREAM(this->get_logger(), "\nsolved ocp " << NTIMINGS << " times, solution printed above\n\n");

        if (this->solver_status_ == ACADOS_SUCCESS)
        {
            RCLCPP_INFO_STREAM(this->get_logger(), "veh_dynamics_ode_acados_solve(): SUCCESS!\n");
        }
        else
        {
            RCLCPP_INFO_STREAM(this->get_logger(), "veh_dynamics_ode_acados_solve() failed with solver_status_ " << this->solver_status_ << ".\n");
        }

        // get solution
        ocp_nlp_out_get(this->nlp_config_, this->nlp_dims_, this->nlp_out_, 0, "kkt_norm_inf", &kkt_norm_inf);
        ocp_nlp_get(nlp_config_, this->nlp_solver_, "sqp_iter", &sqp_iter);

        veh_dynamics_ode_acados_print_stats(acados_ocp_capsule_);

        RCLCPP_INFO_STREAM(this->get_logger(), "\nSolver info:\n");
        RCLCPP_INFO_STREAM(this->get_logger(), " SQP iterations " << 
            sqp_iter << "\n minimum time for " << NTIMINGS << " solve " << min_time*1000 << " [ms]\n KKT " << kkt_norm_inf << " \n");

        // free solver
        int free_solver_status = veh_dynamics_ode_acados_free(this->acados_ocp_capsule_);
        if (free_solver_status) {
            RCLCPP_INFO_STREAM(this->get_logger(), "veh_dynamics_ode_acados_free() returned free_solver_status " << free_solver_status << ". \n");
        }
        // free solver capsule
        free_solver_status = veh_dynamics_ode_acados_free_capsule(this->acados_ocp_capsule_);
        if (free_solver_status) {
            RCLCPP_INFO_STREAM(this->get_logger(), "veh_dynamics_ode_acados_free_capsule() returned free_solver_status " << free_solver_status << ". \n");
        }

        RCLCPP_INFO_STREAM(this->get_logger(), "checked solver init.");

        delete [] xtraj;
        delete [] utraj;
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

    void control_callback()
    {

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
        lbx0[0] = 0;
        ubx0[0] = 0;
        lbx0[1] = 0;
        ubx0[1] = 0;
        lbx0[2] = 0;
        ubx0[2] = 0;
        lbx0[3] = this->x_sim_[3];
        ubx0[3] = this->x_sim_[3];
        lbx0[4] = this->x_sim_[4];
        ubx0[4] = this->x_sim_[4];
        lbx0[5] = this->x_sim_[5];
        ubx0[5] = this->x_sim_[5];
        lbx0[6] = 0;
        ubx0[6] = 0;
        lbx0[7] = 0;
        ubx0[7] = 0;

        ocp_nlp_constraints_model_set(this->nlp_config_, this->nlp_dims_, this->nlp_in_, 0, "idxbx", idxbx0);
        ocp_nlp_constraints_model_set(this->nlp_config_, this->nlp_dims_, this->nlp_in_, 0, "lbx", lbx0);
        ocp_nlp_constraints_model_set(this->nlp_config_, this->nlp_dims_, this->nlp_in_, 0, "ubx", ubx0);

        // initialization for state values
        double x_init[NX];
        x_init[0] = 0.0;
        x_init[1] = 0.0;
        x_init[2] = 0.0;
        x_init[3] = 0.0;
        x_init[4] = 0.0;
        x_init[5] = 0.0;
        x_init[6] = 0.0;
        x_init[7] = 0.0;

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
        p[10] = 0.0;

        // Init curvature ref
        for(j=11; j<41; j++)
        {
            p[j] = this->curv_ref_[j];
        }

        for (int ii = 0; ii <= this->N_; ii++)
        {
            RCLCPP_INFO_STREAM(this->get_logger(), "Setting parameters for stage " << ii);
            veh_dynamics_ode_acados_update_params(this->acados_ocp_capsule_, ii, p, NP);
        }

        RCLCPP_INFO_STREAM(this->get_logger(), "Preparing Evaluation ...");

        // prepare evaluation
        double min_time = 1e6;
        double kkt_norm_inf;
        double elapsed_time;
        int sqp_iter;

        double *xtraj = new double[NX * (N_+1)];
        double *utraj = new double[NU * N_];

        // solve ocp in loop
        int rti_phase = 1;
        RCLCPP_INFO_STREAM(this->get_logger(), "RTI Phase set to: " << rti_phase);

        RCLCPP_INFO_STREAM(this->get_logger(), "Initializing solution (x_i, u_i, x_N) for i = " << ii);
        // initialize solution
        for (int i = 0; i < N_; i++)
        {
            ocp_nlp_out_set(this->nlp_config_, this->nlp_dims_, this->nlp_out_, i, "x", x_init);
            ocp_nlp_out_set(this->nlp_config_, this->nlp_dims_, this->nlp_out_, i, "u", u0);
        }
        ocp_nlp_out_set(this->nlp_config_, this->nlp_dims_, this->nlp_out_, this->N_, "x", x_init);

        RCLCPP_INFO_STREAM(this->get_logger(), "Setting Solver OPtions ...");
        ocp_nlp_solver_opts_set(this->nlp_config_, this->nlp_opts_, "rti_phase", &rti_phase);

        RCLCPP_INFO_STREAM(this->get_logger(), "Solver Call ...");
        this->solver_status_ = veh_dynamics_ode_acados_solve(acados_ocp_capsule_);

        RCLCPP_INFO_STREAM(this->get_logger(), "Get elapsed time of solve ...");
        ocp_nlp_get(this->nlp_config_, this->nlp_solver_, "time_tot", &elapsed_time);
        min_time = MIN(elapsed_time, min_time);

        RCLCPP_INFO_STREAM(this->get_logger(), "Print Statistics ... ");
        /* print solution and statistics */
        for (int ii = 0; ii <= this->nlp_dims_->N; ii++){
            ocp_nlp_out_get(this->nlp_config_, this->nlp_dims_, this->nlp_out_, ii, "x", &xtraj[ii*NX]);
        }
        for (int ii = 0; ii < nlp_dims_->N; ii++){
            ocp_nlp_out_get(this->nlp_config_, this->nlp_dims_, this->nlp_out_, ii, "u", &utraj[ii*NU]);
        }

        RCLCPP_INFO_STREAM(this->get_logger(), "\nSYSTEM TRAJECTORIES:\n");

        RCLCPP_INFO_STREAM(this->get_logger(), "\n--- xtraj ---\n");
        print_matrix( NX, this->N_+1, xtraj, NX);

        RCLCPP_INFO_STREAM(this->get_logger(), "\n--- utraj ---\n");
        print_matrix( NU, this->N_, utraj, NU );

        if (this->solver_status_ == ACADOS_SUCCESS)
        {
            RCLCPP_INFO_STREAM(this->get_logger(), "veh_dynamics_ode_acados_solve(): SUCCESS!\n");
        }
        else
        {
            RCLCPP_INFO_STREAM(this->get_logger(), "veh_dynamics_ode_acados_solve() failed with solver_status_ " << this->solver_status_ << ".\n");
        }

        // get solution
        ocp_nlp_out_get(this->nlp_config_, this->nlp_dims_, this->nlp_out_, 0, "kkt_norm_inf", &kkt_norm_inf);
        ocp_nlp_get(nlp_config_, this->nlp_solver_, "sqp_iter", &sqp_iter);

        veh_dynamics_ode_acados_print_stats(acados_ocp_capsule_);

        RCLCPP_INFO_STREAM(this->get_logger(), "\nSolver info:\n");
        RCLCPP_INFO_STREAM(this->get_logger(), " SQP iterations " << 
            sqp_iter << "\n minimum time for " << NTIMINGS << " solve " << min_time*1000 << " [ms]\n KKT " << kkt_norm_inf << " \n");

        // free solver
        int free_solver_status = veh_dynamics_ode_acados_free(this->acados_ocp_capsule_);
        if (free_solver_status) {
            RCLCPP_INFO_STREAM(this->get_logger(), "veh_dynamics_ode_acados_free() returned free_solver_status " << free_solver_status << ". \n");
        }
        // free solver capsule
        free_solver_status = veh_dynamics_ode_acados_free_capsule(this->acados_ocp_capsule_);
        if (free_solver_status) {
            RCLCPP_INFO_STREAM(this->get_logger(), "veh_dynamics_ode_acados_free_capsule() returned free_solver_status " << free_solver_status << ". \n");
        }

        RCLCPP_INFO_STREAM(this->get_logger(), "checked solver init.");

        

        auto veh_input_msg = sim_backend::msg::SysInput();

        double u[] = {0.0, 0.0, 0.0};
        
        veh_input_msg.fx_r = u[0];
        veh_input_msg.fx_f = u[1];
        veh_input_msg.del_s = u[2];

        control_cmd_publisher_->publish(veh_input_msg);

        delete [] xtraj;
        delete [] utraj;
    }

    void cartesian_state_update(const sim_backend::msg::VehicleState & state_msg)
    {
        x_sim_[0] = state_msg.x_c;
        x_sim_[1] = state_msg.y_c;
        x_sim_[2] = state_msg.psi;
        x_sim_[3] = state_msg.dx_c;
        x_sim_[4] = state_msg.dy_c;
        x_sim_[5] = state_msg.dpsi;
    }

    // Subscriber to state message published at faster frequency
    rclcpp::Subscription<sim_backend::msg::VehicleState>::SharedPtr state_subscriber_;
    
    // Subscriber to ref path provided by "perception"
    rclcpp::Subscription<sim_backend::msg::RefPath>::SharedPtr ref_path_subscriber_;

    // Timer for control command publishing
    rclcpp::TimerBase::SharedPtr control_cmd_timer_;

    // Control command publisher
    rclcpp::Publisher<sim_backend::msg::SysInput>::SharedPtr control_cmd_publisher_;

    // Step Time for controller publisher
    std::chrono::milliseconds dt_{std::chrono::milliseconds(100)};
    double dt_seconds_;

    // State from sim
    double x_[6] = {0.0};

    // Current State for MPC
    double x_[8] = {0.0};

    // Reference Path ahead of the vehicle
    std::vector<std::vector<double>> ref_points_;

    // some iterables
    int i_, j_, k_ = 0, 0, 0;

    // Solver interface path
    int n_s_ = 30;
    double s_max_ = 15.0;
    double ds_ = s_max_ / (n_s_ - 1);
    // s vector
    std::vector<double> s_ref_;
    // Init s_ref_, which is constant
    for (i_ = 0; i_<n_s_; i_++)
    {
        this->s_ref_.push_back(i * ds_);
        RCLCPP_INFO_STREAM(this->get_logger(), "Adding: s_i=" << i * ds_);
    }
    // curvature vector
    std::vector<double> curv_ref_;
    for (i_ = 0; i_<n_s_; i_++)
    {
        this->curv_ref_.push_back(0.0);
        RCLCPP_INFO_STREAM(this->get_logger(), "Adding: curv_i=" << 0.0);
    }

    // Spline vectors
    // knot step (uniform spline)
    double dt_spline_ = 0.1;
    // knot vector for b-spline fit to ref path
    std::vector<double> t_ref_spline_;
    // curvature vector on spline
    std::vector<double> curv_ref_spline_;
    // spline xy-points, derivatives
    std::vector<std::vector<double>> xy_ref_spline_;
    std::vector<std::vector<double>> dxy_ref_spline_;
    std::vector<std::vector<double>> ddxy_ref_spline_;

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