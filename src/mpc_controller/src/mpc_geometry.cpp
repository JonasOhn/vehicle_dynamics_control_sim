#include "mpc_controller/mpc_geometry.hpp"

/**
 * Constructor for the Geometry class, sets the 3rd degree bspline matrix
 *
 */
MpcGeometry::MpcGeometry(){
    std::cout << "Initializing MPC Geometry." << std::endl;
    // B-Spline Coefficient Matrix
    this->bspline_coeff_ << -1, 3, -3, 1,
                        3, -6, 0, 4,
                        -3, 3, 3, 1,
                        1, 0, 0, 0;
    this->bspline_coeff_ = this->bspline_coeff_ / 6.0;
    std::cout << "MPC Geometry initialized." << std::endl;
}

/**
 * Getter for the number of knots or spline evaluations
 *
 * @return total length of knot vector
 */
int MpcGeometry::get_number_of_spline_evaluations()
{
    std::cout << "Getting length of knot vector: " << this->n_t_evals_ << std::endl;
    return this->n_t_evals_;
}

/**
 * Get the initial heading difference from a heading of the car
 *
 * @param psi current heading of the car
 * @return mu, the heading difference state (initial) for MPC
 */
double MpcGeometry::get_initial_heading_difference(double psi)
{
    std::cout << "Retrieving initial heading difference." << std::endl;

    double dx_car_heading = cos(psi);
    double dy_car_heading = sin(psi);

    double dx_path = dx_car_heading;
    double dy_path = dy_car_heading;

    if(this->dxy_ref_spline_.size() > 0){
        dx_path = this->dxy_ref_spline_[0][0];
        dy_path = this->dxy_ref_spline_[0][1];
    }

    // https://wumbo.net/formulas/angle-between-two-vectors-2d/
    return atan2(dy_car_heading * dx_path - dx_car_heading * dy_path,
                 dx_car_heading * dx_path + dy_car_heading * dy_path);
}

double MpcGeometry::get_initial_lateral_deviation(double x_c, double y_c)
{
    // if(this->dxy_ref_spline_.size() > 1){
    //     // projection onto path normal vector to get n
    //     // https://en.wikipedia.org/wiki/Vector_projection
    //     // vector from path start point to CoG of car, vector a
    //     double x_delta_vector = x_c - this->xy_ref_spline_[0][0];
    //     double y_delta_vector = y_c - this->xy_ref_spline_[0][1];

    //     // unit normal vector on path, e_n, vector b_hat
    //     double b_hat_x = - dy_path / sqrt(pow(dx_path, 2.0) + pow(dy_path, 2.0));
    //     double b_hat_y = dx_path / sqrt(pow(dx_path, 2.0) + pow(dy_path, 2.0));

    //     // projection: a_1 = b_hat dot a
    //     a_1 = x_delta_vector * b_hat_x + y_delta_vector * b_hat_y;
    // }
    return 0.0;
}

/**
 * Initialize the Curvature Horizon (s, kappa) that is used in the MPC
 *
 * @param n_s number of progress points from 0 to s_max
 * @param ds progress increment
 * @return 0 if successful
 */
int8_t MpcGeometry::init_mpc_curvature_horizon(int n_s, double ds)
{
    std::cout << "Initializing s vector and curvature to zero." << std::endl;

    // Init s_ref_mpc and respective curvature values
    for (int i = 0; i < n_s; i++)
    {
        this->s_ref_mpc_.push_back(i * ds);
        this->curv_ref_mpc_.push_back(0.0);
    }
    return 0;
}

/**
 * Set the control points for the spline fit from given waypoint path
 *
 * @param waypoints 2D path in the form of waypoints
 * @return 0 if successful
 */
int8_t MpcGeometry::set_control_points(std::vector<std::vector<double>> &waypoints)
{
    std::cout << "Setting B-Spline Control Points." << std::endl;

    // Clear all 2D reference points
    this->ref_points_.clear();

    // Create container for single 2D point with internal data type
    Eigen::Vector2d ref_point;

    // set curvature to zero
    for(this->j_ = 0; this->j_ < (int)this->curv_ref_mpc_.size(); this->j_++)
    {
        this->curv_ref_mpc_[this->j_] = 0.0;
    }

    for(this->i_ = 0; this->i_ < (int)waypoints.size(); this->i_++)
    {
        ref_point[0] = waypoints[this->i_][0];
        ref_point[1] = waypoints[this->i_][1];
        this->ref_points_.push_back(ref_point);
    }

    /* === create an additional point in the beginning of ref_points */

    // option 1: mirror second point on first point
    ref_point[0] = 2 * this->ref_points_[0][0] - this->ref_points_[1][0];
    ref_point[1] = 2 * this->ref_points_[0][1] - this->ref_points_[1][1];

    // option 2: stack first point such that it exists twice
    //ref_point[0] = this->ref_points_[0][0];
    //ref_point[1] = this->ref_points_[0][1];

    this->ref_points_.insert(this->ref_points_.begin(), ref_point);
    
    /* === create an additional point at the end of ref_points */
    
    // option 1: mirror second last point on last point
    ref_point[0] = 2 * this->ref_points_[this->ref_points_.size()-1][0] - this->ref_points_[this->ref_points_.size()-2][0];
    ref_point[1] = 2 * this->ref_points_[this->ref_points_.size()-1][1] - this->ref_points_[this->ref_points_.size()-2][1];

    // option 2: stack last point such that it exists twice
    //ref_point[0] = this->ref_points_[this->ref_points_.size()-1][0];
    //ref_point[1] = this->ref_points_[this->ref_points_.size()-1][1];

    this->ref_points_.push_back(ref_point);

    return 0;
}

/**
 * Fit a third-degree B-Spline through the existing waypoint path
 *
 * @return 0 if fit was successful
 */
int8_t MpcGeometry::fit_bspline_to_waypoint_path()
{
    std::cout << "Fitting 3rd degree B-Spline." << std::endl;

    // Clear all vectors
    this->s_ref_spline_.clear();
    this->curv_ref_spline_.clear();
    this->t_ref_spline_.clear();
    this->xy_ref_spline_.clear();
    this->dxy_ref_spline_.clear();
    this->ddxy_ref_spline_.clear();

    // Calculate number of control points
    size_t num_control_points = this->ref_points_.size();

    // Get number of points for interpolation
    this->n_s_spline_ = (int)num_control_points - 3;

    this->n_t_evals_ = 0;
    // Generate knot vector with necessary number of points
    for(double t = 0.0; t <= (double)this->n_s_spline_; t += this->dt_spline_){
        this->t_ref_spline_.push_back(t);
        n_t_evals_++;
    }

    // path coordinate s along spline-fitted reference path
    double s_ref_length = 0.0;
    // instant curvature
    double curv_instant = 0.0;
    for(size_t i = 0; i < (size_t)this->t_ref_spline_.size(); i++){

        // get integer value of knot vector
        this->t_int_ = (int) t_ref_spline_[i];

        // Fill the control points 2x4 matrix with current points
        // rows
        for(this->i_=0; this->i_ < 2; this->i_++){
            // columns
            for(this->k_=0; this->k_ < 4; this->k_++){
                this->ctrl_points_(this->i_, this->k_) = this->ref_points_[this->t_int_+this->k_][this->i_];
            }
        }

        // get fractional vatue of knot vector
        this->t_frac_ = t_ref_spline_[i] - this->t_int_;

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



        // Calculate curvature from first and second derivatives
        curv_instant = (this->dxy_ref_spline_[i][0] * this->ddxy_ref_spline_[i][1] -
                        this->dxy_ref_spline_[i][1] * this->ddxy_ref_spline_[i][0]);
        curv_instant /= (pow((pow(this->dxy_ref_spline_[i][0], 2.0) + 
                                pow(this->dxy_ref_spline_[i][1], 2.0)), 1.5));
        this->curv_ref_spline_.push_back(curv_instant);
    }

    return 0;
}

/**
 * Generate Curvature Parameter Array for the MPC.
 *
 * @return 1 if something went wrong, 0 otherwise
 */
int8_t MpcGeometry::set_mpc_curvature(int s_max_mpc, int n_s_mpc)
{
    std::cout << "Setting MPC curvature." << std::endl;

    for (this->j_ = 0; this->j_ < n_s_mpc; this->j_++)
    {
        this->curv_ref_mpc_[this->j_] = 0.0;
    }

    int idx_next_s = 0;
    // - if s_max is larger than the integrated s_ref_length, need to fill curv_ref_mpc (with zeros?)
    // - if s_max is smaller than the integrated s_ref_length, no need to take further steps 
    //   since curv_ref_mpc is interpolated correctly
    for (int i = 0; i < n_s_mpc; i++)
    {
        // Iterate through the previously calculated s_ref vector and go 
        // for first entry that is larger than in mpc s ref, interpolate between this and the previous point
        for(this->j_ = 0; this->j_ < (int)this->s_ref_spline_.size(); this->j_++){
            // As soon as s_ref_mpc (constant) is larger than s_ref, calculate curvature as 
            // linear interpolation and break out
            if (this->s_ref_mpc_[i] < this->s_ref_spline_[this->j_]){
                idx_next_s = this->j_;
                // Linearly interpolate between given s_ref values on the spline to get curvature
                this->curv_ref_mpc_[i] = this->curv_ref_spline_[idx_next_s - 1] + 
                    (this->s_ref_mpc_[i] - this->s_ref_spline_[idx_next_s - 1])/
                    (this->s_ref_spline_[idx_next_s] - this->s_ref_spline_[idx_next_s - 1]) 
                    * (this->curv_ref_spline_[idx_next_s] - this->curv_ref_spline_[idx_next_s - 1]);
                // go for next s_ref_mpc curvature
                break;
            }else if (s_max_mpc <= this->s_ref_spline_[this->j_]){
                // Fill with previous value if the reference spline length is larger than the maximum s_ref_mpc
                this->curv_ref_mpc_[i] = this->curv_ref_mpc_[i - 1];
            }
        }
    }

    return 0;
}

/**
 * Get the curvature parameter for the MPC by s-value
 *
 * @param s_to_eval progress value where to evaluate curvature
 * @return the curvature at given index
 */
double MpcGeometry::get_mpc_curvature(double s_to_eval)
{
    double curv_ref_return = 0.0;
    std::cout << "Getting MPC curvature at s= " << s_to_eval << "." << std::endl;

    int idx_next_s = 0;
    // Iterate through the previously calculated s_ref vector (spline) and go 
    // for first entry that is larger than in mpc s ref, interpolate between this and the previous point
    for(this->j_ = 0; this->j_ < (int)this->s_ref_spline_.size(); this->j_++){
        // As soon as s_to_eval (input) is larger than s_ref from spline, calculate curvature as 
        // linear interpolation and break out
        if (s_to_eval <= this->s_ref_spline_[this->j_]){
            idx_next_s = this->j_;
            // Linearly interpolate between given s_ref values on the spline to get curvature
            curv_ref_return = this->curv_ref_spline_[idx_next_s - 1] + 
                (s_to_eval - this->s_ref_spline_[idx_next_s - 1])/
                (this->s_ref_spline_[idx_next_s] - this->s_ref_spline_[idx_next_s - 1]) 
                * (this->curv_ref_spline_[idx_next_s] - this->curv_ref_spline_[idx_next_s - 1]);

            break;
        }
    }

    return curv_ref_return;
}

/**
 * Get the progress (s) vector from spline fit
 *
 * @param vec pointer to vector to fill
 * @return 0
 */
int8_t MpcGeometry::get_s_ref_spline(std::vector<double> &vec)
{
    std::cout << "Getting fitted spline s vector." << std::endl;
    vec = this->s_ref_spline_;
    return 0;
}

/**
 * Get the progress (s) vector for mpc interpolation
 *
 * @param vec pointer to vector to fill
 * @return 0
 */
int8_t MpcGeometry::get_s_ref_mpc(std::vector<double> &vec)
{
    std::cout << "Getting MPC s vector." << std::endl;
    vec = this->s_ref_mpc_;
    return 0;
}

/**
 * Get the curvature (kappa) vector from spline fit
 *
 * @param vec pointer to vector to fill
 * @return 0
 */
int8_t MpcGeometry::get_kappa_ref_spline(std::vector<double> &vec)
{
    std::cout << "Getting fitted spline curvature vector." << std::endl;
    vec = this->curv_ref_spline_;
    return 0;
}

/**
 * Get the curvature (kappa) vector for mpc interpolation
 *
 * @param vec pointer to vector to fill
 * @return 0
 */
int8_t MpcGeometry::get_kappa_ref_mpc(std::vector<double> &vec)
{
    std::cout << "Getting MPC curvature vector." << std::endl;
    vec = this->curv_ref_mpc_;
    return 0;
}

/**
 * Get spline waypoint (2D) at given index
 *
 * @param vec pointer to vector to fill
 * @param idx index
 * @return 0 if successful
 */
int8_t MpcGeometry::get_spline_eval_waypoint(std::vector<double> & vec, int idx)
{
    std::cout << "Getting Spline waypoint at idx " << idx << "." << std::endl;
    if (idx > (int)this->xy_ref_spline_.size() - 1)
    {
        std::cout << "Waypoint idx " << idx << " out of range." << std::endl;
        return 1;
    }
    vec[0] = this->xy_ref_spline_[idx][0];
    vec[1] = this->xy_ref_spline_[idx][1];
    return 0;
}