#include "mpc_controller/mpc_controller_class.hpp"


MpcController::MpcController(){
    std::cout << "Initializing Controller Class." << std::endl;
    this->mpc_geometry_obj_ = MpcGeometry();
    std::cout << "Controller Class Initialized." << std::endl;
}

MpcController::~MpcController(){
    std::cout << "Controller Class Destructor called." << std::endl;
    // free solver
    // int status = veh_dynamics_ode_acados_free(this->acados_ocp_capsule_);
    // if (status) {
    //     std::cout << "veh_dynamics_ode_acados_free() returned status " << status << std::endl;
    // }
    // // free solver capsule
    // status = veh_dynamics_ode_acados_free_capsule(this->acados_ocp_capsule_);
    // if (status) {
    //     std::cout << "veh_dynamics_ode_acados_free_capsule() returned status " << status << std::endl;
    // }
    std::cout << "Destructor finished." << std::endl;
}

int MpcController::get_number_of_spline_evaluations(){
    std::cout << "Getting number of spline knots." << std::endl;
    return this->mpc_geometry_obj_.get_number_of_spline_evaluations();
}

void MpcController::set_dt_control_feedback(double dt_ctrl_callback){
    std::cout << "Setting control loop time step." << std::endl;
    this->dt_ctrl_callback_ = dt_ctrl_callback;
}

int8_t MpcController::set_model_parameters(double l_f, double l_r,
                            double m, double Iz,
                            double g, double D_tire,
                            double C_tire, double B_tire,
                            double C_d, double C_r)
{
    std::cout << "Setting model parameters." << std::endl;
    this->model_params_.l_f = l_f;
    this->model_params_.l_r = l_r;
    this->model_params_.m = m;
    this->model_params_.Iz = Iz;
    this->model_params_.g = g;
    this->model_params_.B_tire = B_tire;
    this->model_params_.C_tire = C_tire;
    this->model_params_.D_tire = D_tire;
    this->model_params_.C_d = C_d;
    this->model_params_.C_r = C_r;
    return 0;
}

int8_t MpcController::set_solver_parameters(int sqp_max_iter, int rti_phase, bool warm_start_first_qp)
{
    std::cout << "Setting solver parameters." << std::endl;
    this->solver_params_.sqp_max_iter = sqp_max_iter;
    this->solver_params_.rti_phase = rti_phase;
    this->solver_params_.warm_start_first_qp = warm_start_first_qp;
    return 0;
}

int8_t MpcController::set_horizon_parameters(int n_s, int N, double T_f, double s_max)
{
    std::cout << "Setting horizon parameters." << std::endl;
    this->horizon_params_.n_s_mpc = n_s;
    this->horizon_params_.N_horizon_mpc = N;
    this->horizon_params_.T_final_mpc = T_f;
    this->horizon_params_.s_max_mpc = s_max;
    this->horizon_params_.ds_mpc = s_max / (n_s - 1);
    this->horizon_params_.dt_mpc = (T_f / N);

    this->mpc_geometry_obj_.init_mpc_curvature_horizon(this->horizon_params_.n_s_mpc,
                                                       this->horizon_params_.ds_mpc);
    return 0;
}

int8_t MpcController::init_solver()
{
    std::cout << "Initializing NLP solver capsule etc." << std::endl;
    // allocate memory and get empty ocp capsule
    this->acados_ocp_capsule_ = veh_dynamics_ode_acados_create_capsule();

    // allocate the array and fill it accordingly
    int creation_status = veh_dynamics_ode_acados_create_with_discretization(this->acados_ocp_capsule_, 
                                                                              this->horizon_params_.N_horizon_mpc,
                                                                              NULL);
    if (creation_status)
    {
        std::cout << "Initializing NLP solver capsule failed." << std::endl;
        return 1;
    }
    this->nlp_config_ = veh_dynamics_ode_acados_get_nlp_config(this->acados_ocp_capsule_);
    this->nlp_dims_ = veh_dynamics_ode_acados_get_nlp_dims(this->acados_ocp_capsule_);
    this->nlp_in_ = veh_dynamics_ode_acados_get_nlp_in(this->acados_ocp_capsule_);
    this->nlp_out_ = veh_dynamics_ode_acados_get_nlp_out(this->acados_ocp_capsule_);
    this->nlp_solver_ = veh_dynamics_ode_acados_get_nlp_solver(this->acados_ocp_capsule_);
    this->nlp_opts_ = veh_dynamics_ode_acados_get_nlp_opts(this->acados_ocp_capsule_);

    // Solver options set
    ocp_nlp_solver_opts_set(this->nlp_config_, this->nlp_opts_, "rti_phase", &(this->solver_params_.rti_phase));
    ocp_nlp_solver_opts_set(this->nlp_config_, this->nlp_opts_, "max_iter", &(this->solver_params_.sqp_max_iter));
    ocp_nlp_solver_opts_set(this->nlp_config_, this->nlp_opts_, "warm_start_first_qp", &(this->solver_params_.warm_start_first_qp));

    std::cout << "Initializing QP solver capsule etc." << std::endl;
    // allocate memory and get empty QP OCP capsule
    this->acados_qp_capsule_ = veh_kinematics_ode_init_acados_create_capsule();

    // allocate the array and fill it accordingly
    creation_status = veh_kinematics_ode_init_acados_create_with_discretization(this->acados_qp_capsule_, 
                                                                                    this->horizon_params_.N_horizon_mpc,
                                                                                    NULL);
    if (creation_status)
    {
        std::cout << "Initializing QP solver capsule failed." << std::endl;
        return 1;
    }
    this->qp_config_ = veh_kinematics_ode_init_acados_get_nlp_config(this->acados_qp_capsule_);
    this->qp_dims_ = veh_kinematics_ode_init_acados_get_nlp_dims(this->acados_qp_capsule_);
    this->qp_in_ = veh_kinematics_ode_init_acados_get_nlp_in(this->acados_qp_capsule_);
    this->qp_out_ = veh_kinematics_ode_init_acados_get_nlp_out(this->acados_qp_capsule_);
    this->qp_solver_ = veh_kinematics_ode_init_acados_get_nlp_solver(this->acados_qp_capsule_);
    this->qp_opts_ = veh_kinematics_ode_init_acados_get_nlp_opts(this->acados_qp_capsule_);

    // Solver options set
    ocp_nlp_solver_opts_set(this->qp_config_, this->qp_opts_, "rti_phase", &(this->solver_params_.rti_phase));
    ocp_nlp_solver_opts_set(this->qp_config_, this->qp_opts_, "max_iter", &(this->qp_init_max_iter_));
    ocp_nlp_solver_opts_set(this->qp_config_, this->qp_opts_, "warm_start_first_qp", &(this->solver_params_.warm_start_first_qp));

    return 0;
}

int8_t MpcController::set_state(double x_c, double y_c, double psi, double vx_local, double vy_local, double dpsi)
{
    std::cout << "Setting state." << std::endl;
    // s: Progress coordinate
    this->x_[0] = 0.000001; // s
    this->x_qp_[0] = 0.000001; // s

    // n: Lateral deviation coordinate
    this->x_[1] = this->mpc_geometry_obj_.get_initial_lateral_deviation(x_c, y_c);
    this->x_qp_[1] = this->x_[1];

    // mu: Heading difference from path
    this->x_[2] = this->mpc_geometry_obj_.get_initial_heading_difference(psi); // mu
    this->x_qp_[2] = this->x_[2]; // mu

    // vx, vy: velocities in local frame
    this->x_[3] = vx_local; // vx
    this->vx_const_qp_ = fmax(vx_local, 0.5);

    this->x_[4] = vy_local; // vy
    this->x_qp_[3] = vy_local; // vy

    // dpsi: yaw rate
    this->x_[5] = dpsi; // r or dpsi
    this->x_qp_[4] = dpsi; // r or dpsi

    return 0;
}

int8_t MpcController::set_initial_state()
{
    std::cout << "Setting initial state x0 for NLP solver." << std::endl;
    int idxbx0[NBX0];
    idxbx0[0] = 0;
    idxbx0[1] = 1;
    idxbx0[2] = 2;
    idxbx0[3] = 3;
    idxbx0[4] = 4;
    idxbx0[5] = 5;

    double bx0[NBX0];
    bx0[0] = this->x_[0];
    bx0[1] = this->x_[1];
    bx0[2] = this->x_[2];
    bx0[3] = this->x_[3];
    bx0[4] = this->x_[4];
    bx0[5] = this->x_[5];

    ocp_nlp_constraints_model_set(this->nlp_config_, this->nlp_dims_, this->nlp_in_, 0, "idxbx", idxbx0);
    ocp_nlp_constraints_model_set(this->nlp_config_, this->nlp_dims_, this->nlp_in_, 0, "lbx", bx0);
    ocp_nlp_constraints_model_set(this->nlp_config_, this->nlp_dims_, this->nlp_in_, 0, "ubx", bx0);

    // ==== QP:

    std::cout << "Setting initial state x0 for QP solver." << std::endl;
    int idxbx0_qp[NBX0_QP];
    idxbx0_qp[0] = 0;
    idxbx0_qp[1] = 1;
    idxbx0_qp[2] = 2;
    idxbx0_qp[3] = 3;
    idxbx0_qp[4] = 4;
    idxbx0_qp[5] = 5;

    double bx0_qp[NBX0_QP];
    bx0_qp[0] = this->x_qp_[0];
    bx0_qp[1] = this->x_qp_[1];
    bx0_qp[2] = this->x_qp_[2];
    bx0_qp[3] = this->x_qp_[3];
    bx0_qp[4] = this->x_qp_[4];
    bx0_qp[5] = this->x_qp_[5];

    ocp_nlp_constraints_model_set(this->qp_config_, this->qp_dims_, this->qp_in_, 0, "idxbx", idxbx0_qp);
    ocp_nlp_constraints_model_set(this->qp_config_, this->qp_dims_, this->qp_in_, 0, "lbx", bx0_qp);
    ocp_nlp_constraints_model_set(this->qp_config_, this->qp_dims_, this->qp_in_, 0, "ubx", bx0_qp);

    return 0;
}

int8_t MpcController::init_mpc_parameters()
{
    std::cout << "Initializing NLP model parameters for all stages." << std::endl;
    // set parameters
    double p[NP];
    p[0] = this->model_params_.m;
    p[1] = this->model_params_.g;
    p[2] = this->model_params_.l_f;
    p[3] = this->model_params_.l_r;
    p[4] = this->model_params_.Iz;
    p[5] = this->model_params_.B_tire;
    p[6] = this->model_params_.C_tire;
    p[7] = this->model_params_.D_tire;
    p[8] = this->model_params_.C_d;
    p[9] = this->model_params_.C_r;

    // Init curvature ref using the last predicted s-trajectory
    for (this->i_ = 0; this->i_ <= this->horizon_params_.N_horizon_mpc; this->i_++)
    {
        // curvature value at stage i_ from geometry object, given s
        p[10] = this->mpc_geometry_obj_.get_mpc_curvature(this->x_traj_[this->i_][0]);
        veh_dynamics_ode_acados_update_params(this->acados_ocp_capsule_, this->i_, p, NP);
    }

    std::cout << "Initializing QP model parameters for all stages." << std::endl;
    // set QP parameters
    double p_qp[NP_QP];
    p_qp[0] = this->model_params_.m;
    p_qp[1] = this->model_params_.g;
    p_qp[2] = this->model_params_.l_f;
    p_qp[3] = this->model_params_.l_r;
    p_qp[4] = this->model_params_.Iz;
    p_qp[5] = this->model_params_.B_tire;
    p_qp[6] = this->model_params_.C_tire;
    p_qp[7] = this->model_params_.D_tire;
    p_qp[8] = this->vx_const_qp_;

    // Init curvature ref as previous solution for curvature
    for (this->i_ = 0; this->i_ <= this->horizon_params_.N_horizon_mpc; this->i_++)
    {
        p_qp[9] = this->curv_qp_[this->i_];
        veh_kinematics_ode_init_acados_update_params(this->acados_qp_capsule_, this->i_, p_qp, NP_QP);
    }

    return 0;
}

int8_t MpcController::init_mpc_horizon()
{
    std::cout << "Initializing state and input for all stages." << std::endl;

    // Solve QP for initialization
    double qp_time = 0.0;
    int qp_init_status = veh_kinematics_ode_init_acados_solve(this->acados_qp_capsule_);
    ocp_nlp_get(this->qp_config_, this->qp_solver_, "time_tot", &qp_time);
    std::cout << "Solving QP for init took " << qp_time*1000 << "ms." << std::endl;
    std::cout << "QP init status is " << qp_init_status << "." << std::endl;

    // Get QP predicted solution
    for (this->i_ = 0; this->i_ < VEH_KINEMATICS_ODE_INIT_N; this->i_++)
    {
        ocp_nlp_out_get(this->qp_config_, this->qp_dims_, this->qp_out_, i_, "x", &this->x_traj_qp_[this->i_]);
        ocp_nlp_out_get(this->qp_config_, this->qp_dims_, this->qp_out_, i_, "u", &this->u_traj_qp_[this->i_]);
    }
    ocp_nlp_out_get(this->qp_config_, this->qp_dims_, this->qp_out_, VEH_KINEMATICS_ODE_INIT_N, "x", &this->x_traj_qp_[VEH_KINEMATICS_ODE_INIT_N]);
    

    // this->x_stage_to_fill_[3] = this->vx_const_qp_; // vx constant
    // this->u_stage_to_fill_[0] = this->axm_const_qp_; // axm constant

    // initialize solution with QP predicted solution
    for (this->i_ = 0; this->i_ < this->horizon_params_.N_horizon_mpc; this->i_++)
    {
        // this->x_stage_to_fill_[0] = this->x_traj_qp_[this->i_][0]; // s
        // this->x_stage_to_fill_[1] = this->x_traj_qp_[this->i_][1]; // n
        // this->x_stage_to_fill_[2] = this->x_traj_qp_[this->i_][2]; // mu
        // this->x_stage_to_fill_[4] = this->x_traj_qp_[this->i_][3]; // vy
        // this->x_stage_to_fill_[5] = this->x_traj_qp_[this->i_][4]; // dpsi

        // this->u_stage_to_fill_[1] = this->u_traj_qp_[this->i_][0]; // del_s

        this->x_stage_to_fill_[0] = this->x_traj_[this->i_][0]; // s
        this->x_stage_to_fill_[1] = this->x_traj_[this->i_][1]; // n
        this->x_stage_to_fill_[2] = this->x_traj_[this->i_][2]; // mu
        this->x_stage_to_fill_[3] = this->x_traj_[this->i_][3]; // vx
        this->x_stage_to_fill_[4] = this->x_traj_[this->i_][4]; // vy
        this->x_stage_to_fill_[5] = this->x_traj_[this->i_][5]; // dpsi

        this->u_stage_to_fill_[0] = this->u_traj_[this->i_][0]; // ax_m
        this->u_stage_to_fill_[1] = this->u_traj_[this->i_][1]; // del_s

        ocp_nlp_out_set(this->nlp_config_, this->nlp_dims_, this->nlp_out_, i_, "x", this->x_stage_to_fill_);
        ocp_nlp_out_set(this->nlp_config_, this->nlp_dims_, this->nlp_out_, i_, "u", this->u_stage_to_fill_);
        // ocp_nlp_out_set(this->nlp_config_, this->nlp_dims_, this->nlp_out_, i_, "z", &this->curv_qp_[this->i_]);
    }
    this->x_stage_to_fill_[0] = this->x_traj_[VEH_KINEMATICS_ODE_INIT_N][0]; // s
    this->x_stage_to_fill_[1] = this->x_traj_[VEH_KINEMATICS_ODE_INIT_N][1]; // n
    this->x_stage_to_fill_[2] = this->x_traj_[VEH_KINEMATICS_ODE_INIT_N][2]; // mu
    this->x_stage_to_fill_[3] = this->x_traj_[VEH_KINEMATICS_ODE_INIT_N][3]; // vx
    this->x_stage_to_fill_[4] = this->x_traj_[VEH_KINEMATICS_ODE_INIT_N][4]; // vy
    this->x_stage_to_fill_[5] = this->x_traj_[VEH_KINEMATICS_ODE_INIT_N][5]; // dpsi

    ocp_nlp_out_set(this->nlp_config_, this->nlp_dims_, this->nlp_out_, this->horizon_params_.N_horizon_mpc, "x", this->x_stage_to_fill_);
    
    return 0;
}

int8_t MpcController::get_solver_statistics(int & nlp_solv_status,
                                           int & qp_solv_status,
                                           int & sqp_iter,
                                           double & kkt_norm, 
                                           double & solv_time)
{
    int8_t return_val = 0;
    return_val = this->update_solver_statistics();
    
    std::cout << "Retrieving solver statistics." << std::endl;

    nlp_solv_status = this->solver_out_.nlp_solver_status;
    qp_solv_status = this->solver_out_.qp_solver_status;
    sqp_iter = this->solver_out_.sqp_iterations;
    kkt_norm = this->solver_out_.kkt_norm_inf;
    solv_time = this->solver_out_.solve_time;

    return return_val;
}

int8_t MpcController::update_solver_statistics()
{
    std::cout << "Updating solver statistics." << std::endl;

    // get time
    ocp_nlp_get(this->nlp_config_, this->nlp_solver_, "time_tot", &(this->solver_out_.solve_time));

    // get stage 0 KKT inf. norm
    ocp_nlp_out_get(this->nlp_config_, this->nlp_dims_, this->nlp_out_, 0, "kkt_norm_inf", &(this->solver_out_.kkt_norm_inf));

    // get SQP iterations
    ocp_nlp_get(nlp_config_, this->nlp_solver_, "sqp_iter", &(this->solver_out_.sqp_iterations));

    // Get QP status
    ocp_nlp_get(this->nlp_config_, this->nlp_solver_, "qp_status", &(this->solver_out_.qp_solver_status));

    return 0;
}

int8_t MpcController::solve_mpc()
{
    std::cout << "Solve called." << std::endl;

    this->solver_out_.nlp_solver_status = veh_dynamics_ode_acados_solve(this->acados_ocp_capsule_);
    
    return (int8_t) this->solver_out_.nlp_solver_status;
}

int8_t MpcController::get_input(double (&u)[2])
{
    std::cout << "Retrieving plant-input from solver." << std::endl;

    // get stage to evaluate based on elapsed solver time
    //int stage_to_eval = (int) (total_elapsed_time / dt_mpc_);
    int stage_to_eval = 0;

    // If solver successful
    if(this->solver_out_.nlp_solver_status == 0
        || this->solver_out_.nlp_solver_status == 5
        || this->solver_out_.nlp_solver_status == 2){
        // evaluate u at stage
        ocp_nlp_out_get(this->nlp_config_, this->nlp_dims_, this->nlp_out_, stage_to_eval, "u", &u);
        this->axm_const_qp_ = fmax(u[0], 0.01);
        // scale optimization output by m to get Force back
        u[0] = this->model_params_.m * u[0];
    // If solver failed
    }else{
        u[0] = 0.0;
        u[1] = 0.0;
        this->axm_const_qp_ = 0.0;
    }

    return 0;
}

int8_t MpcController::get_predictions(std::vector<double> &s_predict,
                                     std::vector<double> &n_predict,
                                     std::vector<double> &mu_predict,
                                     std::vector<double> &vx_predict,
                                     std::vector<double> &vy_predict,
                                     std::vector<double> &dpsi_predict,
                                     std::vector<double> &axm_predict,
                                     std::vector<double> &dels_predict,
                                     std::vector<double> &s_traj_mpc,
                                     std::vector<double> &kappa_traj_mpc,
                                     std::vector<double> &s_ref_spline,
                                     std::vector<double> &kappa_ref_spline,
                                     std::vector<double> &s_ref_mpc,
                                     std::vector<double> &kappa_ref_mpc)
{
    std::cout << "Retrieving MPC numeric predictions." << std::endl;
    // Clear all vector just to be sure
    s_predict.clear();
    n_predict.clear();
    mu_predict.clear();
    vx_predict.clear();
    vy_predict.clear();
    dpsi_predict.clear();
    axm_predict.clear();
    dels_predict.clear();
    s_traj_mpc.clear();
    kappa_traj_mpc.clear();
    s_ref_spline.clear();
    kappa_ref_spline.clear();
    s_ref_mpc.clear();
    kappa_ref_mpc.clear();

    /* Get solution */ 
    for (int i = 0; i < this->horizon_params_.N_horizon_mpc; i++)
    {
        ocp_nlp_out_get(this->nlp_config_, this->nlp_dims_, this->nlp_out_, i, "x", &this->x_traj_[i]);
        ocp_nlp_out_get(this->nlp_config_, this->nlp_dims_, this->nlp_out_, i, "u", &this->u_traj_[i]);
        ocp_nlp_out_get(this->nlp_config_, this->nlp_dims_, this->nlp_out_, i, "z", &this->z_traj_[i]);
        ocp_nlp_out_get(this->nlp_config_, this->nlp_dims_, this->nlp_out_, i, "z", &this->curv_qp_[i]);

        s_predict.push_back(this->x_traj_[i][0]);
        n_predict.push_back(this->x_traj_[i][1]);
        mu_predict.push_back(this->x_traj_[i][2]);
        vx_predict.push_back(this->x_traj_[i][3]);
        vy_predict.push_back(this->x_traj_[i][4]);
        dpsi_predict.push_back(this->x_traj_[i][5]);

        axm_predict.push_back(this->u_traj_[i][0]);
        dels_predict.push_back(this->u_traj_[i][1]);

        kappa_traj_mpc.push_back(this->z_traj_[i]);
        s_traj_mpc.push_back(this->x_traj_[i][0]);
    }
    // get terminal state
    ocp_nlp_out_get(this->nlp_config_,
                    this->nlp_dims_, 
                    this->nlp_out_, 
                    this->nlp_dims_->N, 
                    "x", 
                    &this->x_traj_[this->nlp_dims_->N]);
    ocp_nlp_out_get(this->nlp_config_,
                    this->nlp_dims_,
                    this->nlp_out_, 
                    this->nlp_dims_->N,
                    "z", 
                    &this->curv_qp_[this->nlp_dims_->N]);

    s_predict.push_back(x_traj_[this->nlp_dims_->N][0]);
    n_predict.push_back(x_traj_[this->nlp_dims_->N][1]);
    mu_predict.push_back(x_traj_[this->nlp_dims_->N][2]);
    vx_predict.push_back(x_traj_[this->nlp_dims_->N][3]);
    vy_predict.push_back(x_traj_[this->nlp_dims_->N][4]);
    dpsi_predict.push_back(x_traj_[this->nlp_dims_->N][5]);

    this->mpc_geometry_obj_.get_s_ref_spline(s_ref_spline);
    this->mpc_geometry_obj_.get_s_ref_mpc(s_ref_mpc);
    this->mpc_geometry_obj_.get_kappa_ref_spline(kappa_ref_spline);
    this->mpc_geometry_obj_.get_kappa_ref_mpc(kappa_ref_mpc);

    return 0;
}

/**
 * Getter function for the fitted spline and the predicted positions along MPC horizon
 *
 * @param xy_spline_points vector to fill spline fit
 * @param xy_predict_points vector to fill predicted path
 * @return 0 if successful
 */
int8_t MpcController::get_visual_predictions(std::vector<std::vector<double>> &xy_spline_points,
                                             std::vector<std::vector<double>> &xy_predict_points)
{
    int i = 0;
    std::cout << "Retrieving MPC visual predictions." << std::endl;

    std::vector<double> xy_spline_point{0.0, 0.0};
    for(i = 0; i < this->mpc_geometry_obj_.get_number_of_spline_evaluations(); i++)
    {
        this->mpc_geometry_obj_.get_spline_eval_waypoint(xy_spline_point, i);
        xy_spline_points.push_back(xy_spline_point);
    }

    std::vector<double> xy_predicted_point{0.0, 0.0};
    xy_predict_points.push_back(xy_predicted_point);
    double psi_predict = 0.0;
    double dx = 0.0;
    double dy = 0.0;
    for(i = 0; i < this->horizon_params_.N_horizon_mpc; i++)
    {
        // psi_k+1 = psi_k + dpsi_k * dt
        psi_predict = psi_predict + this->x_traj_[i][5] * this->horizon_params_.dt_mpc;

        dx = this->x_traj_[i][3] * this->horizon_params_.dt_mpc;
        dy = this->x_traj_[i][4] * this->horizon_params_.dt_mpc;

        xy_predicted_point[0] = xy_predicted_point[0] + (dx * cos(psi_predict) - dy * sin(psi_predict));
        xy_predicted_point[1] = xy_predicted_point[1] + (dy * cos(psi_predict) + dx * sin(psi_predict));

        xy_predict_points.push_back(xy_predicted_point);
    }

    return 0;
}

/**
 * Setter function for the reference path that also invokes b-spline fit
 * and setter for curvature parameter vector of MPC
 *
 * @param path refpath points ahead of the car including the CoG of the car
 * @return 0 if successful
 */
int8_t MpcController::set_reference_path(std::vector<std::vector<double>> &path)
{
    std::cout << "Updating Reference Path using geometry class." << std::endl;

    this->mpc_geometry_obj_.set_control_points(path);
    this->mpc_geometry_obj_.fit_bspline_to_waypoint_path();
    this->mpc_geometry_obj_.set_mpc_curvature(this->horizon_params_.s_max_mpc, this->horizon_params_.n_s_mpc);

    return 0;
}