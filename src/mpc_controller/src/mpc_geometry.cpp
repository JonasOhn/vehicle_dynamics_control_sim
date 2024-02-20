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

double MpcGeometry::get_x_ref(int idx)
{
    if(this->dxy_ref_spline_.size() > 0){
        return this->xy_ref_spline_[idx][0];
    }
    return 0.0;
}

double MpcGeometry::get_y_ref(int idx)
{
    if(this->dxy_ref_spline_.size() > 0){
        return this->xy_ref_spline_[idx][1];
    }
    return 0.0;
}

double MpcGeometry::get_dx_ref(int idx)
{
    if(this->dxy_ref_spline_.size() > 0){
        return this->dxy_ref_spline_[idx][0];
    }
    return 0.0;
}

double MpcGeometry::get_dy_ref(int idx)
{
    if(this->dxy_ref_spline_.size() > 0){
        return this->dxy_ref_spline_[idx][1];
    }
    return 0.0;
}

/**
 * Getter for the idx of the initial path reference point
 *
 * @return idx of initial path reference point
 */
int MpcGeometry::get_initial_path_reference_idx(double x_c, double y_c)
{
    int idx_path = 0;

    for (this->i_ = 0; this->i_ < (int)this->xy_ref_spline_.size(); this->i_++)
    {
        double x_delta_vector = x_c - this->xy_ref_spline_[this->i_][0];
        double y_delta_vector = y_c - this->xy_ref_spline_[this->i_][1];

        if(x_delta_vector * this->dxy_ref_spline_[this->i_][0] + y_delta_vector * this->dxy_ref_spline_[this->i_][1] <= 0)
        {
            if (this->i_ > 0)
            {
                idx_path = this->i_ - 1;
            }else
            {
                idx_path = 0;
            }

            break;
        }
    }

    return idx_path;
}

/**
 * Get the initial heading difference from a heading of the car
 *
 * @param psi current heading of the car
 * @param path_idx index of the path reference point
 * @return mu, the heading difference state (initial) for MPC
 */
double MpcGeometry::get_initial_heading_difference(double psi, int path_idx)
{
    std::cout << "Retrieving initial heading difference." << std::endl;

    double angle = 0.0;

    double dx_car_heading = cos(psi);
    double dy_car_heading = sin(psi);

    double dx_path = 0.0;
    double dy_path = 0.0;

    if(this->dxy_ref_spline_.size() > 0){
        dx_path = this->dxy_ref_spline_[path_idx][0];
        dy_path = this->dxy_ref_spline_[path_idx][1];
        // https://wumbo.net/formulas/angle-between-two-vectors-2d/
        angle = atan2(dy_car_heading * dx_path - dx_car_heading * dy_path,
                      dx_car_heading * dx_path + dy_car_heading * dy_path);
    }

    return angle;
}

/**
 * Get the initial progress from the car's position
 *
 * @param x_c current x-coordinate of the car
 * @param y_c current y-coordinate of the car
 * @param path_idx index of the path reference point
 * @return s, the progress state (initial) for MPC
 */
double MpcGeometry::get_initial_progress(double x_c, double y_c, int path_idx)
{
    double s = 0.0;

    if (this->dxy_ref_spline_.size() >= 1)
    {
            // projection onto initial path tangent vector to get n
            // https://en.wikipedia.org/wiki/Vector_projection
            // vector from path start point to CoG of car, vector a
            double x_delta_vector = x_c - this->xy_ref_spline_[path_idx][0];
            double y_delta_vector = y_c - this->xy_ref_spline_[path_idx][1];

            // tangent initial path vector
            double dx_path = this->dxy_ref_spline_[path_idx][0];
            double dy_path = this->dxy_ref_spline_[path_idx][1];

            // *unit* normal vector on path, e_n, vector b_hat
            double b_hat_x = dx_path / sqrt(pow(dx_path, 2.0) + pow(dy_path, 2.0));
            double b_hat_y = dy_path / sqrt(pow(dx_path, 2.0) + pow(dy_path, 2.0));

            // projection: s = a_1 = b_hat dot a
            s = x_delta_vector * b_hat_x + y_delta_vector * b_hat_y;
    }

    return s;
}

/**
 * Get the initial lateral deviation of the car
 *
 * @param x_c current x-coordinate of the car
 * @param y_c current y-coordinate of the car
 * @param path_idx index of the path reference point
 * @return n, the lateral deviation state (initial) for MPC
 */
double MpcGeometry::get_initial_lateral_deviation(double x_c, double y_c, int path_idx)
{
    double n = 0.0;

    if (this->dxy_ref_spline_.size() >= 1)
    {
        // projection onto initial path normal vector to get n
        // https://en.wikipedia.org/wiki/Vector_projection
        // vector from path start point to CoG of car, vector a
        double x_delta_vector = x_c - this->xy_ref_spline_[path_idx][0];
        double y_delta_vector = y_c - this->xy_ref_spline_[path_idx][1];

        // tangent initial path vector
        double dx_path = this->dxy_ref_spline_[path_idx][0];
        double dy_path = this->dxy_ref_spline_[path_idx][1];

        // *unit* normal vector on path, e_n, vector b_hat
        double b_hat_x = - dy_path / sqrt(pow(dx_path, 2.0) + pow(dy_path, 2.0));
        double b_hat_y = dx_path / sqrt(pow(dx_path, 2.0) + pow(dy_path, 2.0));

        // projection: n = a_1 = b_hat dot a
        n = x_delta_vector * b_hat_x + y_delta_vector * b_hat_y;
    }

    return n;
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
    for(size_t i = 0; i < (size_t)this->t_ref_spline_.size(); i++)
    {
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
 * Get the curvature parameter for the MPC by s-value
 *
 * @param s_to_eval progress value where to evaluate curvature
 * @return the curvature at given index
 */
double MpcGeometry::get_mpc_curvature(double s_to_eval)
{
    std::cout << "retrieving mpc curvature at s = " << s_to_eval << std::endl;

    double curv_ref_return = 0.0;

    if (this->s_ref_spline_.size() < 1 || s_to_eval < this->s_ref_spline_[0])
    {
        return curv_ref_return;
    }

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
    std::cout << "Got MPC curvature at s = " << s_to_eval << " to be " << curv_ref_return << std::endl;

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