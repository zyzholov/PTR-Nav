#pragma once
#include "terrain_nav/motion_prediction.hpp"

namespace terrain_nav {
// A Gaussian residual plus a deterministic mean-displacement segment. The
// rectangle minus that segment is enclosed by its body-frame AABB; collision
// with the original swept disk implies membership in this enlarged box.
// Gaussian supporting-halfspace tails bound this event without independence.
inline double boundedMeanBodyCollisionUpper(const YieldState &s,const MotionCheckGeometry &g,
    double radius,const GaussianPosition &position,const Eigen::Vector2d &first,
    const Eigen::Vector2d &last)
{
  const double c=std::cos(s.yaw),sn=std::sin(s.yaw);
  Eigen::Matrix2d rotate;rotate<<c,sn,-sn,c;
  const Eigen::Vector2d a=rotate*first,b=rotate*last;
  const double padding=1e-9; // Outward rounding at deterministic tangency.
  const Eigen::Vector2d low(-g.back-radius-std::max(a.x(),b.x())-padding,
                          -g.half_width-radius-std::max(a.y(),b.y())-padding);
  const Eigen::Vector2d high(g.front+radius-std::min(a.x(),b.x())+padding,
                            g.half_width+radius-std::min(a.y(),b.y())+padding);
  GaussianPosition local;local.valid=position.valid;
  local.mean=rotate*(position.mean-Eigen::Vector2d(s.x,s.y))-.5*(low+high);
  local.covariance=rotate*position.covariance*rotate.transpose();
  return gaussianBodyCollisionUpper(0.,0.,0.,.5*(high.x()-low.x()),.5*(high.x()-low.x()),
    .5*(high.y()-low.y()),0.,local);
}
struct MotionRisk {
  bool valid=false;
  double upper=1.;
  size_t gaussian_objects=0,reachable_objects=0;
  std::vector<double> pose_upper;
};
// Conditional marginal bounds -> Boole union bound over ALL objects and
// samples. In particular, do NOT take their maximum or multiply survival
// probabilities: object/time errors need not be independent. Validity is
// conditional on the explicitly configured Gaussian position/process model.
// Non-fitted objects retain the original deterministic reachable envelope.
inline MotionRisk timedGaussianMotionRisk(const std::vector<YieldState> &states,double origin,
    const std::vector<MotionTrackSnapshot> &tracks,const MotionCheckGeometry &geometry,
    double preference_clearance=0.)
{
  MotionRisk out;
  const double f=geometry.front,b=geometry.back,w=geometry.half_width;
  if(states.empty() || !std::isfinite(origin) || !std::isfinite(f)||f<0||!std::isfinite(b)||b<0||
     !std::isfinite(w)||w<0||!std::isfinite(geometry.wheelbase)||geometry.wheelbase<=0||
     !std::isfinite(geometry.max_steering)||geometry.max_steering<0||geometry.max_steering>=1.57||
     !std::isfinite(geometry.extra_inflation)||geometry.extra_inflation<0||
     !std::isfinite(preference_clearance)||preference_clearance<0)return out;
  for(const auto &s:states)if(!std::isfinite(s.t)||!std::isfinite(s.x)||!std::isfinite(s.y)||
      !std::isfinite(s.yaw)||!std::isfinite(s.speed))return out;
  out.pose_upper.assign(states.size(),0.);
  const double half_gap=motionSampleHalfGap(states);if(!std::isfinite(half_gap))return out;
  for(const auto &track:tracks) {
    if(!track.circular || !track.center_motion_fit || !validGaussianMotion(track.gaussian)) {
      ++out.reachable_objects;
      if(timedMotionUnsafe(states,origin,{track},geometry,0.,preference_clearance))return out;
      continue;
    }
    ++out.gaussian_objects;
    const double age=origin+states.front().t-track.stamp;
    if(!std::isfinite(age)||age<-.05||age>.6||track.gaussian.stamp!=track.stamp||
       !std::isfinite(track.resolution)||track.resolution<=0||track.cells.empty()||
       !std::isfinite(track.radius_bound)||track.radius_bound<=0||
       !std::isfinite(track.prediction_maximum_speed)||track.prediction_maximum_speed<=0||
       track.prediction_maximum_speed>4.5||!gaussianCircleEnclosesMeasurements(track))return out;
    for(size_t i=0;i<states.size();++i) {
      const auto &s=states[i];
      // Same +/-20ms inter-sample geometric reserve as the original online
      // predicate. This uses the physical 4.5m/s operating-speed assumption;
      // it is NOT an almost-sure bound of the unbounded Gaussian process.
      // Thus the returned bound is for these expanded discrete poses, not
      // an unconditional continuous-time human safety guarantee.
      const double sweep=half_gap*(4.5+std::abs(s.speed)*
        (1+std::hypot(std::max(f,b),w)*std::tan(geometry.max_steering)/geometry.wheelbase));
      double radius=track.radius_bound+geometry.extra_inflation+
        sweep+preference_clearance;
      const double t=origin+s.t-track.stamp;
      const auto p=gaussian_motion_detail::predictValidated(track.gaussian,t);
      if(!p.valid)return out;
      double risk;
      const double acceleration=track.gaussian.longitudinal_acceleration_bound;
      if(acceleration>0.) {
        const double speed=track.gaussian.mean.tail<2>().norm();
        const double cap=predictionSpeedCap(speed,track.velocity_error,track.prediction_maximum_speed);
        const double maximum=boundedTranslationDistance(speed,t,acceleration,cap);
        if(!std::isfinite(maximum))return out;
        Eigen::Vector2d first=Eigen::Vector2d::Zero(),last=first;
        if(speed>1e-9) {
          first=track.gaussian.mean.tail<2>()/speed*(minimumTranslationDistance(speed,t,acceleration,cap)-speed*t);
          last=track.gaussian.mean.tail<2>()/speed*(maximum-speed*t);
        } else radius+=maximum; // Direction of a departure from rest is unknown.
        risk=boundedMeanBodyCollisionUpper(s,geometry,radius,p,first,last);
      } else risk=gaussianBodyCollisionUpper(s.x,s.y,s.yaw,f,b,w,radius,p);
      if(!std::isfinite(risk))return out;
      out.pose_upper[i]=std::min(1.,out.pose_upper[i]+risk);
    }
  }
  out.upper=0.;for(double p:out.pose_upper)out.upper=std::min(1.,out.upper+p);
  out.valid=true;return out;
}
} // namespace terrain_nav
