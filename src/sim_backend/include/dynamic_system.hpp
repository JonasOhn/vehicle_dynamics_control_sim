#ifndef DYNAMIC_SYSTEM_H
#define DYNAMIC_SYSTEM_H

#include <iostream>
#include <boost/numeric/odeint.hpp>

using namespace std;
using namespace boost::numeric::odeint;

typedef struct{
    double l_f; // m
    double l_r; // m
    double m; // kg
    double Iz; // kg m m
    double g; // m s-2
    double D_tire;
    double C_tire;
    double B_tire;
    double C_d; // 0.5 * rho * CdA
    double C_r; // -
    double T_mot;
    double D_mot;
} parameters;

typedef std::vector<double> state_type;

/* The rhs of dxdt = f(x) defined as a class */
class DynamicSystem {

    private:
        // indep. params
        parameters params_;
        // dep. params
        double l_;
        // inputs
        double Fx_f_;
        double Fx_r_;
        double delta_s_;

    public:
        DynamicSystem();

        void update_inputs(double fx_f, double fx_r, double delta_steer);

        void operator() ( const state_type &x , state_type &dxdt , const double /* t */ );
};

#endif