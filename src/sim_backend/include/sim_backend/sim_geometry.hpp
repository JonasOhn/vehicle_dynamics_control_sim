#ifndef SIM_GEOMETRY_H
#define SIM_GEOMETRY_H

namespace sim_geometry{

struct Pose2D {
    double x;
    double y;
    double psi;
};

struct Point2D {
    double x;
    double y;
};

}

#endif