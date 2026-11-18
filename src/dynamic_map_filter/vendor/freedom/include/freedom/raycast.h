#ifndef _RAYCAST_H
#define _RAYCAST_H

#include "freedom/common_types.h"
#include <algorithm>
#include <cmath>

namespace freedom{
const double MAX_TIME = 1.0;
class RayCaster
{
public:
    RayCaster(const Point& start_, const Index& start_idx_, Index& idx_, const double& res_) : 
        start(start_), start_idx(start_idx_), idx(idx_), res(res_) {}

    void setRayEnd(const Point& end_, const Index& end_idx_);
    bool step();

private:
    Point start;
    Index start_idx;

    Point end;
    Index end_idx;

    int step_x;
    int step_y;
    int step_z;

    double t_step_x;
    double t_step_y;
    double t_step_z;

    double t_to_next_bound_x;
    double t_to_next_bound_y;
    double t_to_next_bound_z;

    Index& idx;
    double res;

    inline double tToNextBound(const double& pos,const double& vel)
    {
        if(vel == 0) return MAX_TIME;
        // floor defines the SAME half-open voxel as Map. At an exact
        // boundary, a positive ray crosses the NEXT upper face, whereas a
        // negative ray leaves this address cell immediately. fmod returned
        // zero for both and shifted positive rays by an entire voxel.
        const double boundary=(std::floor(pos/res)+(vel>0?1.:0.))*res;
        return std::max(0.,(boundary-pos)/vel);
    }
};
}
#endif
