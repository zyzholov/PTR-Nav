#pragma once
#include <cmath>
#include <set>
#include <utility>
#include <vector>

namespace terrain_nav
{
// World-grid offsets (dy,dx), matching probabilistic_risk._geometry exactly.
inline std::vector<std::pair<int,int>> heightSupportOffsets(
  double front, double back, double half_width, double res, double yaw, bool internal_steps,
  bool exact_footprint=false)
{
  const double c=std::cos(yaw),s=std::sin(yaw);
  const double pad=.5*res*(std::abs(c)+std::abs(s));
  const double half=(front+back)/2,offset=(front-back)/2;
  const double extent_x=std::abs(c)*half+std::abs(s)*half_width+.5*res;
  const double extent_y=std::abs(s)*half+std::abs(c)*half_width+.5*res;
  const int extent=static_cast<int>(std::ceil(std::hypot(std::max(front,back),half_width)/res))+1;
  std::set<std::pair<int,int>> footprint;
  for(int r=-extent;r<=extent;++r) for(int col=-extent;col<=extent;++col) {
    const double x=res*(c*col+s*r),y=res*(-s*col+c*r);
    const bool world_axes=std::abs(res*col-offset*c)<=extent_x+1e-9 &&
      std::abs(res*r-offset*s)<=extent_y+1e-9;
    if(x>=-back-pad && x<=front+pad && std::abs(y)<=half_width+pad &&
       (!exact_footprint || world_axes))footprint.emplace(r,col);
  }
  auto support=footprint;
  if(!internal_steps) for(const auto &q:footprint) for(int k=-1;k<=2;++k) {
    support.emplace(q.first+k,q.second);support.emplace(q.first,q.second+k);
  }
  return {support.begin(),support.end()};
}
}  // namespace terrain_nav
