#pragma once
#include <cmath>

namespace terrain_nav {
// All poses remain base_link poses. The rear axle is offset metres BEHIND it.
// Speed and tan(steering)/wheelbase refer to the rear axle's longitudinal
// motion; path distances refer to the base_link arc, in either gear.
inline double bodyArcScale(double curvature,double offset) {
  return std::hypot(1.,offset*curvature);
}
template<class Pose> Pose integrateBodyVelocity(Pose p,double speed,double rate,
                                               double dt,double offset) {
  const double turn=rate*dt,half=.5*turn;
  const double sinc=std::abs(half)<1e-8?1.-half*half/6.:std::sin(half)/half;
  const double yaw=p.yaw+turn;
  p.x+=speed*dt*sinc*std::cos(p.yaw+half)+offset*(std::cos(yaw)-std::cos(p.yaw));
  p.y+=speed*dt*sinc*std::sin(p.yaw+half)+offset*(std::sin(yaw)-std::sin(p.yaw));
  p.yaw=std::atan2(std::sin(yaw),std::cos(yaw));
  return p;
}
template<class Pose> Pose advanceBodyArc(Pose p,double curvature,double signed_base_distance,
                                        double offset) {
  const double rear_distance=signed_base_distance/bodyArcScale(curvature,offset);
  return integrateBodyVelocity(p,rear_distance,rear_distance*curvature,1.,offset);
}
template<class Pose> Pose rearAxlePose(Pose p,double offset) {
  p.x-=offset*std::cos(p.yaw);p.y-=offset*std::sin(p.yaw);return p;
}
template<class Pose> Pose baseLinkPose(Pose p,double offset) {
  p.x+=offset*std::cos(p.yaw);p.y+=offset*std::sin(p.yaw);return p;
}
struct BodyArcConnection {bool valid=false;double curvature=0.,distance=0.;};
// A constant steering connector to a base_link POSITION (end yaw is free).
// Reject long/full-circle solutions. Admission and swept-body proofs belong
// to the caller. No endpoint snapping is used to hide a kinematic error.
inline BodyArcConnection connectBodyArc(double dx,double dy,int direction,double offset) {
  BodyArcConnection r;
  if(!std::isfinite(dx)||!std::isfinite(dy)||!std::isfinite(offset)||offset<0.||
     (direction!=1&&direction!=-1)||dx*direction<=1e-8)return r;
  const double denominator=dx*dx+dy*dy+2.*offset*dx;
  if(std::abs(dy)<1e-10){r.valid=true;r.distance=dx;return r;}
  if(std::abs(denominator)<1e-12)return r;
  const double k=2.*dy/denominator;
  const double angle=std::atan2(k*k*offset*dy+k*dx,
                               k*k*offset*(dx+offset)-k*dy+1.);
  const double rear_distance=angle/k;
  if(!std::isfinite(rear_distance)||rear_distance*direction<=0.||std::abs(angle)>=M_PI-1e-6)return r;
  r.valid=true;r.curvature=k;r.distance=rear_distance*bodyArcScale(k,offset);return r;
}
}  // namespace terrain_nav
