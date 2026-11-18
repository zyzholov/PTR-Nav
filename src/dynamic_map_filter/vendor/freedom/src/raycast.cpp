#include "freedom/raycast.h"
#include <limits>

namespace freedom{
void RayCaster::setRayEnd(const Point& end_, const Index& end_idx_)
{
    // 重置idx到初始位置
    idx = start_idx;

    end = end_;
    end_idx = end_idx_;

    PointBias velocity = end - start;
    step_x = velocity.x() > 0? 1 : velocity.x()<0? -1 : 0;
    step_y = velocity.y() > 0? 1 : velocity.y()<0? -1 : 0;
    step_z = velocity.z() > 0? 1 : velocity.z()<0? -1 : 0;

    t_step_x = velocity.x() != 0 ? res / abs(velocity.x()) : MAX_TIME;
    t_step_y = velocity.y() != 0 ? res / abs(velocity.y()) : MAX_TIME;
    t_step_z = velocity.z() != 0 ? res / abs(velocity.z()) : MAX_TIME;

    t_to_next_bound_x = tToNextBound(start.x(),velocity.x());
    t_to_next_bound_y = tToNextBound(start.y(),velocity.y());
    t_to_next_bound_z = tToNextBound(start.z(),velocity.z());
    // A negative ray on a starting face immediately enters the lower
    // address cell. Do not report zero-length start/corner cells as free.
    if(t_to_next_bound_x==0. && step_x<0){idx.x()+=step_x;t_to_next_bound_x+=t_step_x;}
    if(t_to_next_bound_y==0. && step_y<0){idx.y()+=step_y;t_to_next_bound_y+=t_step_y;}
    if(t_to_next_bound_z==0. && step_z<0){idx.z()+=step_z;t_to_next_bound_z+=t_step_z;}
}

// 使用do{}while(step())的方式以从开始的体素遍历
// 将结束判断放在末尾以不包括末端体素
// 将结束判断放在开头以包括末端体素
bool RayCaster::step()
{   
    const double crossing=std::min({t_to_next_bound_x,t_to_next_bound_y,t_to_next_bound_z});
    if(crossing>=MAX_TIME)return false;
    // Cross simultaneous faces together. The few-ULP tie tolerance only
    // OMITs numerical sliver cells; it cannot add a fictitious free cell.
    const double tie=8*std::numeric_limits<double>::epsilon()*std::max(1.,std::abs(crossing));
    if(t_to_next_bound_x<=crossing+tie){idx.x()+=step_x;t_to_next_bound_x+=t_step_x;}
    if(t_to_next_bound_y<=crossing+tie){idx.y()+=step_y;t_to_next_bound_y+=t_step_y;}
    if(t_to_next_bound_z<=crossing+tie){idx.z()+=step_z;t_to_next_bound_z+=t_step_z;}
    return idx!=end_idx;
}

}  // namespace freedom
