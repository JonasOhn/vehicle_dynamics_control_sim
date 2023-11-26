#include "acados_solvers_library/acados_solver_library.hpp"


AcadosSolver::AcadosSolver(){
    solver_status_ = 0.0;
    acados_ocp_capsule_ = veh_dynamics_ode_acados_create_capsule();
    // there is an opportunity to change the number of shooting intervals in C without new code generation
    N_ = VEH_DYNAMICS_ODE_N;
    // allocate the array and fill it accordingly
    double* new_time_steps = NULL;
    int status = veh_dynamics_ode_acados_create_with_discretization(acados_ocp_capsule_, N_, new_time_steps);

    if (status)
    {
        printf("veh_dynamics_ode_acados_create() returned status %d. Exiting.\n", status);
        exit(1);
    }
}

int AcadosSolver::solve()
{
    int status = 0;
    ocp_nlp_config *nlp_config = veh_dynamics_ode_acados_get_nlp_config(acados_ocp_capsule_);
    ocp_nlp_dims *nlp_dims = veh_dynamics_ode_acados_get_nlp_dims(acados_ocp_capsule_);
    ocp_nlp_in *nlp_in = veh_dynamics_ode_acados_get_nlp_in(acados_ocp_capsule_);
    ocp_nlp_out *nlp_out = veh_dynamics_ode_acados_get_nlp_out(acados_ocp_capsule_);
    ocp_nlp_solver *nlp_solver = veh_dynamics_ode_acados_get_nlp_solver(acados_ocp_capsule_);
    void *nlp_opts = veh_dynamics_ode_acados_get_nlp_opts(acados_ocp_capsule_);

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
    p[26] = 0.1;
    p[27] = 0.1;
    p[28] = 0.1;
    p[29] = 0.1;
    p[30] = 0.1;
    p[31] = 0.1;
    p[32] = 0.1;
    p[33] = 0.1;
    p[34] = 0.1;
    p[35] = 0.1;
    p[36] = 0.1;
    p[37] = 0.1;
    p[38] = 0.1;
    p[39] = 0.1;
    p[40] = 0.1;

    for (int ii = 0; ii <= N_; ii++)
    {
        veh_dynamics_ode_acados_update_params(acados_ocp_capsule_, ii, p, NP);
    }


    // prepare evaluation
    int NTIMINGS = 1;
    double min_time = 1e12;
    double kkt_norm_inf;
    double elapsed_time;
    int sqp_iter;

    double *xtraj;
    xtraj = new double[NX * (N_+1)];
    double *utraj;
    utraj = new double[NU * N_];


    // solve ocp in loop
    int rti_phase = 0;

    for (int ii = 0; ii < NTIMINGS; ii++)
    {
        // initialize solution
        for (int i = 0; i < N_; i++)
        {
            ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "x", x_init);
            ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "u", u0);
        }
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, N_, "x", x_init);
        ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "rti_phase", &rti_phase);
        status = veh_dynamics_ode_acados_solve(acados_ocp_capsule_);
        ocp_nlp_get(nlp_config, nlp_solver, "time_tot", &elapsed_time);
        min_time = MIN(elapsed_time, min_time);
    }

    /* print solution and statistics */
    for (int ii = 0; ii <= nlp_dims->N; ii++)
        ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, ii, "x", &xtraj[ii*NX]);
    for (int ii = 0; ii < nlp_dims->N; ii++)
        ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, ii, "u", &utraj[ii*NU]);

    printf("\n--- xtraj ---\n");
    d_print_exp_tran_mat( NX, N_+1, xtraj, NX);
    printf("\n--- utraj ---\n");
    d_print_exp_tran_mat( NU, N_, utraj, NU );
    // ocp_nlp_out_print(nlp_solver->dims, nlp_out);

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

    veh_dynamics_ode_acados_print_stats(acados_ocp_capsule_);

    printf("\nSolver info:\n");
    printf(" SQP iterations %2d\n minimum time for %d solve %f [ms]\n KKT %e\n",
        sqp_iter, NTIMINGS, min_time*1000, kkt_norm_inf);

    // free solver
    status = veh_dynamics_ode_acados_free(acados_ocp_capsule_);
    if (status) {
        printf("veh_dynamics_ode_acados_free() returned status %d. \n", status);
    }
    // free solver capsule
    status = veh_dynamics_ode_acados_free_capsule(acados_ocp_capsule_);
    if (status) {
        printf("veh_dynamics_ode_acados_free_capsule() returned status %d. \n", status);
    }

    delete[]xtraj;
    delete[]utraj;

    return status;
}