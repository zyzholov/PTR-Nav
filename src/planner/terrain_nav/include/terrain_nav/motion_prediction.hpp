#pragma once
// One time-indexed motion/body predicate for online control and raw-data replay.
#include "terrain_nav/dynamic_yield.hpp"
#include "terrain_nav/motion_guard.hpp"
#include "terrain_nav/temporal_geometry.hpp"
#include <string>

namespace terrain_nav {
struct MotionCheckGeometry {
  double front=.521,back=.521,half_width=.460,max_steering=.31432913231137,wheelbase=.430,extra_inflation=0.;
};
struct MotionConflict {
  uint64_t track_id=0;
  YieldState state;
  double age=0.,track_stamp=0.,end_dx=0.,end_dy=0.,reserve=0.;
  std::string reason="invalid_motion_input";
};
struct EarlyStopEnvelope {
  std::vector<YieldState> stationary;
  double expansion=0.;
  bool valid=false;
};
// A robust PRIORITY guard for every possible earlier rest footprint. Enclose
// the entire sampled body locus by the initial rectangle plus an isotropic
// radius, and test that stationary envelope over the ORIGINAL ordered times.
// Do not encode different rest positions as duplicate end-time "trajectory"
// samples: their artificial time gap and a reachable-model fallback silently
// replace a configured Gaussian model with a different, enormous event.
// This envelope never certifies an executable command or its terrain budget.
inline EarlyStopEnvelope earlyStopEnvelope(const std::vector<YieldState> &states,
                                           const MotionCheckGeometry &g)
{
  EarlyStopEnvelope out;
  if(states.empty()||!std::isfinite(g.front)||g.front<0||!std::isfinite(g.back)||g.back<0||
     !std::isfinite(g.half_width)||g.half_width<0||!std::isfinite(g.max_steering)||
     g.max_steering<0||g.max_steering>=1.57||!std::isfinite(g.wheelbase)||g.wheelbase<=0)return out;
  const auto &root=states.front();
  const double radius=std::hypot(std::max(g.front,g.back),g.half_width);
  double gap=0.,speed=0.;
  for(size_t i=0;i<states.size();++i) {
    const auto &s=states[i];
    if(!std::isfinite(s.t)||!std::isfinite(s.x)||!std::isfinite(s.y)||!std::isfinite(s.yaw)||
       !std::isfinite(s.speed)||(i&&s.t<=states[i-1].t))return {};
    if(i)gap=std::max(gap,s.t-states[i-1].t);
    speed=std::max(speed,std::abs(s.speed));
    const double turn=std::remainder(s.yaw-root.yaw,2*M_PI);
    out.expansion=std::max(out.expansion,std::hypot(s.x-root.x,s.y-root.y)+
        2*radius*std::abs(std::sin(turn/2)));
    auto at=root;at.t=s.t;at.speed=at.steering=0.;out.stationary.push_back(at);
  }
  // Cover body motion BETWEEN the supplied states as well. The configured
  // steering bound limits corner speed; obstacle interpolation is separately
  // covered by the ordinary motion predicate's temporal sweep.
  out.expansion+=.5*gap*speed*(1+radius*std::tan(g.max_steering)/g.wheelbase);
  out.valid=std::isfinite(out.expansion);return out;
}
// In the Gaussian model the fit's radius encloses measured cell squares;
// centre/velocity uncertainty belongs to the posterior, not a second copy
// of the legacy registration disk. Fallback reachable geometry is unchanged.
inline bool gaussianCircleEnclosesMeasurements(const MotionTrackSnapshot &t)
{
  if(!t.circular || !std::isfinite(t.center_x)||!std::isfinite(t.center_y)||
     !std::isfinite(t.radius_bound)||t.radius_bound<=0||!std::isfinite(t.resolution)||t.resolution<=0||t.cells.empty())return false;
  for(const auto &p:t.cells)if(!std::isfinite(p[0])||!std::isfinite(p[1])||
      std::hypot(p[0]-t.center_x,p[1]-t.center_y)+std::sqrt(.5)*t.resolution>t.radius_bound+1e-8)return false;
  return true;
}
// Advisory speed-policy ownership ONLY. Never remove occupancy, predictions
// or risk events. Require EXACT current measured bounds, a confirmed moving
// Gaussian identity and mutual uniqueness. Motion confirmation is historical:
// a CURRENT measured posterior remains modeled while that identity slows or
// reverses. Its zero mean speed is not a claim that it cannot move again;
// the complete Gaussian process and chance checks remain mandatory. Coasts,
// provisional fits and every unmatched component retain the approach limit.
inline std::vector<std::array<double,4>> unmodeledCompactBounds(
    const std::vector<std::array<double,4>> &cues,const std::vector<MotionTrackSnapshot> &tracks,
    double frame_stamp)
{
  std::vector<std::array<double,4>> boxes;std::vector<size_t> owners;
  if(!std::isfinite(frame_stamp))return cues;
  for(size_t i=0;i<tracks.size();++i) {
    const auto &t=tracks[i];
    if(!t.id||!t.motion_confirmed||!t.center_motion_fit||!validGaussianMotion(t.gaussian)||
       t.gaussian.observation_count<3||std::abs(t.stamp-frame_stamp)>1e-8||t.gaussian.stamp!=t.stamp||
       !gaussianCircleEnclosesMeasurements(t))continue;
    std::array<double,4> box{INFINITY,INFINITY,-INFINITY,-INFINITY};
    for(const auto &p:t.cells) {
      box[0]=std::min(box[0],p[0]-.5*t.resolution);box[1]=std::min(box[1],p[1]-.5*t.resolution);
      box[2]=std::max(box[2],p[0]+.5*t.resolution);box[3]=std::max(box[3],p[1]+.5*t.resolution);
    }
    boxes.push_back(box);owners.push_back(i);
  }
  const auto equal=[](const auto &a,const auto &b) {
    for(int k=0;k<4;++k)if(!std::isfinite(a[k])||std::abs(a[k]-b[k])>1e-8)return false;
    return true;
  };
  std::vector<std::array<double,4>> remaining;
  for(const auto &cue:cues) {
    size_t matches=0,matched=0;
    for(size_t i=0;i<boxes.size();++i)if(equal(cue,boxes[i])){++matches;matched=i;}
    size_t claims=0;if(matches==1)for(const auto &other:cues)claims+=equal(other,boxes[matched]);
    if(matches==1)for(size_t i=0;i<tracks.size();++i)
      if(i!=owners[matched]&&tracks[i].id==tracks[owners[matched]].id)++matches;
    if(matches!=1||claims!=1)remaining.push_back(cue);
  }
  return remaining;
}
inline double motionSampleHalfGap(const std::vector<YieldState> &states)
{
  if(states.empty()||!std::isfinite(states.front().t))return NAN;
  double half=0.;
  for(size_t i=1;i<states.size();++i) {
    const double dt=states[i].t-states[i-1].t;
    if(!std::isfinite(dt)||dt<0)return NAN;
    half=std::max(half,.5*dt);
  }
  return half;
}
struct TimedMotionAssessment {bool valid=false,unsafe=true;};
inline TimedMotionAssessment assessTimedMotion(const std::vector<YieldState> &states,double origin,
    const std::vector<MotionTrackSnapshot> &tracks,const MotionCheckGeometry &geometry,
    double velocity_reserve_mps=0.,double preference_clearance_m=0.,MotionConflict *conflict=nullptr)
{
    if(states.empty() || !std::isfinite(velocity_reserve_mps) || velocity_reserve_mps<0 ||
        !std::isfinite(preference_clearance_m) || preference_clearance_m<0 ||
        !std::isfinite(origin)||!std::isfinite(geometry.front)||geometry.front<0||
        !std::isfinite(geometry.back)||geometry.back<0||!std::isfinite(geometry.half_width)||geometry.half_width<0||
        !std::isfinite(geometry.wheelbase)||geometry.wheelbase<=0||!std::isfinite(geometry.max_steering)||
        geometry.max_steering<0||geometry.max_steering>=1.57||
        !std::isfinite(geometry.extra_inflation)||geometry.extra_inflation<0)return {};
    for(size_t i=0;i<states.size();++i) {
      const auto &s=states[i];
      if(!std::isfinite(s.t)||!std::isfinite(s.x)||!std::isfinite(s.y)||!std::isfinite(s.yaw)||
         !std::isfinite(s.speed)||(i&&s.t<states[i-1].t))return {};
    }
    const double f=geometry.front,b=geometry.back,w=geometry.half_width;
    if(conflict)*conflict=MotionConflict{};
    for(const auto &track:tracks) {
      if(conflict)conflict->track_id=track.id;
      if(!std::isfinite(track.resolution) || track.resolution<=0 || track.cells.empty() ||
         !std::isfinite(track.vx) || !std::isfinite(track.vy) || !std::isfinite(track.stamp) ||
         !std::isfinite(track.acceleration_bound) || track.acceleration_bound<0||
         !std::all_of(track.cells.begin(),track.cells.end(),[](const auto &p) {
           return std::isfinite(p[0])&&std::isfinite(p[1]);
         }))return {};
      if(track.circular && (!std::isfinite(track.center_x) || !std::isfinite(track.center_y) ||
          !std::isfinite(track.radius_bound) || track.radius_bound<=0 ||
          !std::isfinite(track.velocity_error) || track.velocity_error<0))return {};
      const double age=origin+states.front().t-track.stamp;
      if(age<-.05 || age>.6)return {};
      double cell_x0=INFINITY,cell_y0=INFINITY,cell_x1=-INFINITY,cell_y1=-INFINITY;
      if(!track.circular)for(const auto &cell:track.cells) {
        cell_x0=std::min(cell_x0,cell[0]);cell_x1=std::max(cell_x1,cell[0]);
        cell_y0=std::min(cell_y0,cell[1]);cell_y1=std::max(cell_y1,cell[1]);
      }
      for(const auto &state:states) {
        if(conflict){conflict->state=state;conflict->age=age;conflict->track_stamp=track.stamp;}
        const double t=origin+state.t-track.stamp;
        const double speed=std::hypot(track.vx,track.vy);
        const double cap=predictionSpeedCap(speed,track.velocity_error,track.prediction_maximum_speed);
        const double travel=terrain_nav::boundedTranslationDistance(speed,std::max(0.,t),track.acceleration_bound,cap);
        const double retreat=std::min(0.,terrain_nav::minimumTranslationDistance(speed,std::max(0.,t),track.acceleration_bound,cap));
        if(!std::isfinite(travel) || !std::isfinite(retreat) || (track.acceleration_bound>0 && speed<1e-6))return {};
        const double tx=speed>0?track.vx/speed*travel:0.,ty=speed>0?track.vy/speed*travel:0.;
        const double bx=speed>0?track.vx/speed*retreat:0.,by=speed>0?track.vy/speed*retreat:0.;
        const double sweep=.02*(std::min(cap,speed+track.acceleration_bound*std::max(0.,t))+std::abs(state.speed)*
          (1+std::hypot(std::max(f,b),w)*std::tan(geometry.max_steering)/geometry.wheelbase));
        // Keep the same positional residual accepted by association. A fixed
        // 20cm minimum silently preserves the former coarse-grid envelope
        // after switching to measured 5cm shapes. Weak first translations
        // still keep their original minimum and growing velocity reserve.
        const double association=terrain_nav::motionAssociationRadius(track.resolution);
        const double reserve=geometry.extra_inflation+
          terrain_nav::motionAssociationRadius(track.resolution,std::max(0.,age))+
          (velocity_reserve_mps>0?std::max(0.,.2-association):0.)+sweep+
          (velocity_reserve_mps+track.velocity_error)*std::max(0.,t)+preference_clearance_m;
        if(conflict) {conflict->end_dx=tx;conflict->end_dy=ty;conflict->reserve=reserve;
          conflict->reason=track.circular?"circle_reach":"cell_reach";}
        if(track.circular) {
          if(terrain_nav::bodyOverlapsTranslatingDisk(state.x,state.y,state.yaw,f,b,w,
            track.center_x+bx,track.center_y+by,track.center_x+tx,
            track.center_y+ty,track.radius_bound+reserve))return {true,true};
          continue;
        }
        // Conservative broad phase for a measured cell group. Only skip an
        // ego sample when the full translated/reserved group AABB and full
        // asymmetric vehicle AABB are disjoint. Every remaining cell retains
        // the original exact continuous SAT; no shape, time or reserve shrinks.
        const double c=std::cos(state.yaw),s=std::sin(state.yaw);
        const double cx=state.x+.5*(f-b)*c,cy=state.y+.5*(f-b)*s;
        const double ex=.5*(f+b)*std::abs(c)+w*std::abs(s);
        const double ey=.5*(f+b)*std::abs(s)+w*std::abs(c);
        const double pad=.5*track.resolution+reserve;
        const double x0=cell_x0+std::min(bx,tx)-pad,x1=cell_x1+std::max(bx,tx)+pad;
        const double y0=cell_y0+std::min(by,ty)-pad,y1=cell_y1+std::max(by,ty)+pad;
        const std::array<double,8> bounds{x0,y0,x1,y1,cx-ex,cy-ey,cx+ex,cy+ey};
        if(std::all_of(bounds.begin(),bounds.end(),[](double v){return std::isfinite(v);})) {
          double scale=1.;for(double v:bounds)scale=std::max(scale,std::abs(v));
          const double outward=1e-9*scale;
          if(x1<cx-ex-outward||x0>cx+ex+outward||y1<cy-ey-outward||y0>cy+ey+outward)continue;
        }
        for(const auto &cell:track.cells)
          // A tracked obstacle may slow, stop or reverse on its heading;
          // extrapolated departure is NOT evidence that it will vacate the
          // lane. Retain those positions at every FUTURE ego sample. The
          // model is directional bounded-speed motion, not reciprocity or a
          // calibrated probability of other agents yielding.
          if(!std::isfinite(cell[0]) || !std::isfinite(cell[1]))return {};
          else if(terrain_nav::bodyOverlapsTranslatingCell(state.x,state.y,state.yaw,f,b,w,
               cell[0]+bx,cell[1]+by,cell[0]+tx,cell[1]+ty,
               track.resolution+2*reserve))return {true,true};
      }
    }
    return {true,false};
}
inline bool timedMotionUnsafe(const std::vector<YieldState> &states,double origin,
    const std::vector<MotionTrackSnapshot> &tracks,const MotionCheckGeometry &geometry,
    double velocity_reserve_mps=0.,double preference_clearance_m=0.,MotionConflict *conflict=nullptr)
{
  const auto result=assessTimedMotion(states,origin,tracks,geometry,
    velocity_reserve_mps,preference_clearance_m,conflict);
  return !result.valid||result.unsafe;
}
} // namespace terrain_nav
