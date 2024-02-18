#ifndef MPC_GEOMETRY_H
#define MPC_GEOMETRY_H

#include <iostream>
#include <Eigen/Dense>


class MpcGeometry {

    private:
        // Reference Path ahead of the vehicle, given by simulator (or estimation)
        std::vector<Eigen::Vector2d> ref_points_;

        // Matrix for bspline coefficients
        Eigen::Matrix4d bspline_coeff_;
        // Knot Vector
        Eigen::Vector4d t_vec_;
        // Derivative of know vector w.r.t. t
        Eigen::Vector4d dt_vec_;
        // Derivative of know vector w.r.t. t
        Eigen::Vector4d ddt_vec_;
        // instantaneous control points (2D), matrix form
        Eigen::Matrix<double, 2, 4> ctrl_points_;

        // storage variable for fractional part of t (knot)
        double t_frac_ = 0.0;
        // storage variable for integer part of t (knot)
        int t_int_ = 0;

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
        int n_s_spline_ = 0;
        int n_t_evals_ = 0;

        int i_ = 0;
        int j_ = 0;
        int k_ = 0;

    public:

        MpcGeometry();

        int8_t set_control_points(std::vector<std::vector<double>> &waypoints);

        int8_t fit_bspline_to_waypoint_path();

        double get_mpc_curvature(double s_to_eval);

        int get_number_of_spline_evaluations();

        double get_initial_heading_difference(double psi);

        double get_initial_lateral_deviation(double x_c, double y_c);

        double get_initial_progress(double x_c, double y_c);

        int8_t get_s_ref_spline(std::vector<double> &vec);

        int8_t get_kappa_ref_spline(std::vector<double> &vec);

        int8_t get_spline_eval_waypoint(std::vector<double> &vec, int idx);

};

#endif