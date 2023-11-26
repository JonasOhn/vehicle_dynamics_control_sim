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

void print_matrix(int row, int col, double *A, int lda)
{
	int i, j;
	for(j=0; j<col; j++)
		{
		for(i=0; i<row; i++)
			{
			printf("%e\t", A[i+lda*j]);
			}
		printf("\n");
		}
	printf("\n");
}

void check_solver_init()
{
    veh_dynamics_ode_solver_capsule *acados_ocp_capsule = veh_dynamics_ode_acados_create_capsule();
    // there is an opportunity to change the number of shooting intervals in C without new code generation
    int N = VEH_DYNAMICS_ODE_N;
    // allocate the array and fill it accordingly
    double* new_time_steps = NULL;
    int status = veh_dynamics_ode_acados_create_with_discretization(acados_ocp_capsule, N, new_time_steps);

    if (status)
    {
        printf("veh_dynamics_ode_acados_create() returned status %d. Exiting.\n", status);
        rclcpp::shutdown();
    }

    ocp_nlp_config *nlp_config = veh_dynamics_ode_acados_get_nlp_config(acados_ocp_capsule);
    ocp_nlp_dims *nlp_dims = veh_dynamics_ode_acados_get_nlp_dims(acados_ocp_capsule);
    ocp_nlp_in *nlp_in = veh_dynamics_ode_acados_get_nlp_in(acados_ocp_capsule);
    ocp_nlp_out *nlp_out = veh_dynamics_ode_acados_get_nlp_out(acados_ocp_capsule);
    ocp_nlp_solver *nlp_solver = veh_dynamics_ode_acados_get_nlp_solver(acados_ocp_capsule);
    void *nlp_opts = veh_dynamics_ode_acados_get_nlp_opts(acados_ocp_capsule);

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

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "idxbx", idxbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "lbx", lbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "ubx", ubx0);

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

    // initial value for control input
    double u0[NU];
    u0[0] = 0.0;
    u0[1] = 0.0;
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
    p[26] = 0.05;
    p[27] = 0.05;
    p[28] = 0.05;
    p[29] = 0.05;
    p[30] = 0.05;
    p[31] = 0.05;
    p[32] = 0.05;
    p[33] = 0.05;
    p[34] = 0.05;
    p[35] = 0.05;
    p[36] = 0.05;
    p[37] = 0.05;
    p[38] = 0.05;
    p[39] = 0.05;
    p[40] = 0.05;

    for (int ii = 0; ii <= N; ii++)
    {
        veh_dynamics_ode_acados_update_params(acados_ocp_capsule, ii, p, NP);
    }


    // prepare evaluation
    int NTIMINGS = 1;
    double min_time = 1;
    double kkt_norm_inf;
    double elapsed_time;
    int sqp_iter;

    double xtraj[NX * (N+1)];
    double utraj[NU * N];


    // solve ocp in loop
    int rti_phase = 0;

    for (int ii = 0; ii < NTIMINGS; ii++)
    {
        // initialize solution
        for (int i = 0; i < N; i++)
        {
            ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "x", x_init);
            ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "u", u0);
        }
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, N, "x", x_init);
        ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "rti_phase", &rti_phase);
        status = veh_dynamics_ode_acados_solve(acados_ocp_capsule);
        ocp_nlp_get(nlp_config, nlp_solver, "time_tot", &elapsed_time);
        min_time = MIN(elapsed_time, min_time);
    }

    /* print solution and statistics */
    for (int ii = 0; ii <= nlp_dims->N; ii++)
        ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, ii, "x", &xtraj[ii*NX]);
    for (int ii = 0; ii < nlp_dims->N; ii++)
        ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, ii, "u", &utraj[ii*NU]);

    printf("\nSYSTEM TRAJECTORIES:\n");
    printf("\n--- xtraj ---\n");
    print_matrix( NX, N+1, xtraj, NX);
    printf("\n--- utraj ---\n");
    print_matrix( NU, N, utraj, NU );

    printf("\nsolved ocp %d times, solution printed above\n\n", NTIMINGS);

    if (status == ACADOS_SUCCESS)
    {
        printf("veh_dynamics_ode_acados_solve(): SUCCESS!\n");
    }
    else
    {
        printf("veh_dynamics_ode_acados_solve() failed with status %d.\n", status);
    }

    // get solution
    ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, 0, "kkt_norm_inf", &kkt_norm_inf);
    ocp_nlp_get(nlp_config, nlp_solver, "sqp_iter", &sqp_iter);

    veh_dynamics_ode_acados_print_stats(acados_ocp_capsule);

    printf("\nSolver info:\n");
    printf(" SQP iterations %2d\n minimum time for %d solve %f [ms]\n KKT %e\n",
        sqp_iter, NTIMINGS, min_time*1000, kkt_norm_inf);

    // free solver
    status = veh_dynamics_ode_acados_free(acados_ocp_capsule);
    if (status) {
        printf("veh_dynamics_ode_acados_free() returned status %d. \n", status);
    }
    // free solver capsule
    status = veh_dynamics_ode_acados_free_capsule(acados_ocp_capsule);
    if (status) {
        printf("veh_dynamics_ode_acados_free_capsule() returned status %d. \n", status);
    }

    printf("END!");
}
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

        // state_subscriber_ = this->create_subscription<sim_backend::msg::VehicleState>(
        //     "vehicle_state", 1, std::bind(&MPCController::cartesian_state_update, this, std::placeholders::_1));

        // control_cmd_publisher_ = this->create_publisher<sim_backend::msg::SysInput>("vehicle_input", 10);
        // control_cmd_timer_ = this->create_wall_timer(this->dt_, std::bind(&MPCController::control_callback, this));

        // ==========================================
        check_solver_init();
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