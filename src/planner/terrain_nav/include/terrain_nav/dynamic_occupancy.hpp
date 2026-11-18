#pragma once
#include "terrain_nav/motion_prediction.hpp"
#include <map>
#include <unordered_map>
#include <unordered_set>

namespace terrain_nav {
// Explicit observation-limited dynamic scope, not free-space classification.
// Remove only finite PAST coasts. Invalid/future sources stay in the set so
// ordinary validity checks still reject them. Raw/held occupancy is separate.
inline std::vector<MotionTrackSnapshot> currentMotionObservations(
    const std::vector<MotionTrackSnapshot> &tracks,double stamp)
{
  if(!std::isfinite(stamp))return tracks;
  std::vector<MotionTrackSnapshot> result;
  for(const auto &t:tracks)
    if(!std::isfinite(t.stamp)||t.stamp>=stamp-5e-9)result.push_back(t);
  return result;
}
inline bool currentMeasuredIdentity(const MotionTrackSnapshot &t,double stamp,bool moving_only)
{
  return t.id&&t.motion_confirmed&&t.circular&&t.center_motion_fit&&validGaussianMotion(t.gaussian)&&
    t.gaussian.observation_count>=3&&std::isfinite(stamp)&&std::abs(t.stamp-stamp)<=5e-9&&
    t.gaussian.stamp==t.stamp&&(!moving_only||t.gaussian.mean.tail<2>().norm()>=.35)&&
    gaussianCircleEnclosesMeasurements(t);
}
// PROPOSAL eligibility only. A geometry-qualified causal centre model does
// not need a persistent label before a full-speed escape can be EVALUATED.
// A persistent ID is still required for ownership, history exclusion and
// conflict resolution; the full all-object execution proof is unchanged.
inline bool currentMeasuredCenterModel(const MotionTrackSnapshot &t,double stamp)
{
  if(currentMeasuredIdentity(t,stamp,false))return true;
  if(t.id||t.motion_confirmed||!t.circular||!t.center_motion_fit||
     !validGaussianMotion(t.gaussian)||t.gaussian.observation_count<3||
     !std::isfinite(stamp)||std::abs(t.stamp-stamp)>5e-9||t.gaussian.stamp!=t.stamp||
     t.center_observations.size()<3||!gaussianCircleEnclosesMeasurements(t))return false;
  const auto velocity=fitCausalCenterVelocity(t.center_observations);
  if(!velocity||std::abs(velocity->stamp_s-t.stamp)>5e-9||
     velocity->fit_residual_rms>t.resolution||velocity->max_step_innovation>2*t.resolution||
     std::hypot(velocity->velocity[0]-t.vx,velocity->velocity[1]-t.vy)>1e-8)return false;
  const auto &last=t.center_observations.back();
  return std::hypot(last.center[0]-t.center_x,last.center[1]-t.center_y)<=1e-8;
}
// Ownership labels, never free-space observations. Only an uninflated CURRENT
// measured cell uniquely belonging to a confirmed motion identity can be labeled.
// Ambiguous/unconfirmed/coasting hypotheses cannot erase static geometry.
inline std::map<uint64_t,std::vector<size_t>> measuredMovingCells(
    const std::vector<MotionTrackSnapshot> &tracks,const std::vector<uint8_t> &raw,
    int width,int height,double x0,double y0,double resolution,double stamp)
{
  std::map<uint64_t,std::vector<size_t>> out;
  if(width<=0||height<=0||raw.size()!=size_t(width)*height||!std::isfinite(resolution)||resolution<=0||
     !std::isfinite(x0)||!std::isfinite(y0)||!std::isfinite(stamp))return out;
  std::unordered_map<uint64_t,size_t> ids;
  for(const auto &t:tracks)if(t.id)++ids[t.id];
  std::unordered_map<size_t,uint64_t> claims;
  for(const auto &t:tracks) {
    if(!std::isfinite(t.resolution)||t.resolution<=0)continue;
    // Confirmation comes from actual past translation. A newly measured
    // slowdown/reversal does not turn the SAME body's historic positions into
    // independent stationary objects; its full Gaussian future stays checked.
    const bool eligible=ids[t.id]==1&&std::abs(t.resolution-resolution)<=1e-10&&currentMeasuredIdentity(t,stamp,false);
    std::unordered_set<size_t> once;
    for(const auto &p:t.cells) {
      if(!std::isfinite(p[0])||!std::isfinite(p[1]))continue;
      // Conservative overlap also lets a different/coarser hypothesis veto
      // ownership. No nearest-center association to an old component.
      const double half=t.resolution*.5;
      if(p[0]+half<=x0||p[0]-half>=x0+width*resolution||
         p[1]+half<=y0||p[1]-half>=y0+height*resolution)continue;
      // Clip in floating point BEFORE integer conversion, including malformed
      // but finite out-of-domain observations.
      const int xa=int(std::floor(std::clamp((p[0]-half-x0)/resolution+1e-7,0.,double(width-1))));
      const int xb=int(std::ceil(std::clamp((p[0]+half-x0)/resolution-1e-7,1.,double(width))))-1;
      const int ya=int(std::floor(std::clamp((p[1]-half-y0)/resolution+1e-7,0.,double(height-1))));
      const int yb=int(std::ceil(std::clamp((p[1]+half-y0)/resolution-1e-7,1.,double(height))))-1;
      for(int y=ya;y<=yb;++y)for(int x=xa;x<=xb;++x) {
        const size_t cell=size_t(y)*width+x;if(!raw[cell]||!once.insert(cell).second)continue;
        const uint64_t owner=eligible?t.id:0;
        const auto inserted=claims.emplace(cell,owner);
        if(!inserted.second)inserted.first->second=0;
      }
    }
  }
  for(const auto &[cell,id]:claims)if(id)out[id].push_back(cell);
  for(auto &[id,cells]:out)std::sort(cells.begin(),cells.end());
  return out;
}
inline bool rectangleInsideMeasuredCell(const std::array<double,4> &rectangle,
                                         const std::array<double,4> &cell)
{
  for(double v:rectangle)if(!std::isfinite(v))return false;
  for(double v:cell)if(!std::isfinite(v))return false;
  return rectangle[0]<rectangle[2]&&rectangle[1]<rectangle[3]&&cell[0]<cell[2]&&cell[1]<cell[3]&&
    rectangle[0]>=cell[0]-1e-7&&rectangle[1]>=cell[1]-1e-7&&
    rectangle[2]<=cell[2]+1e-7&&rectangle[3]<=cell[3]+1e-7;
}
} // namespace terrain_nav
