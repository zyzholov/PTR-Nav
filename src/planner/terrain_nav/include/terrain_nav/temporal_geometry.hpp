#pragma once
#include <algorithm>
#include <array>
#include <cmath>

namespace terrain_nav {
// Exact distance between a line segment and the full oriented vehicle body.
// The segment is the possible object-center locus; the radius includes the
// measured object's size and explicit engineering uncertainty, not its speed.
inline bool bodyOverlapsTranslatingDisk(double x,double y,double yaw,
    double front,double back,double half_width,double ax,double ay,double bx,double by,double radius)
{
  for(double v:{x,y,yaw,front,back,half_width,ax,ay,bx,by,radius})if(!std::isfinite(v))return true;
  if(front<=0 || back<=0 || half_width<=0 || radius<=0)return true;
  const double c=std::cos(yaw),s=std::sin(yaw);
  const double a=c*(ax-x)+s*(ay-y),b=-s*(ax-x)+c*(ay-y);
  const double d=c*(bx-ax)+s*(by-ay),e=-s*(bx-ax)+c*(by-ay);
  double low=0.,high=1.;
  for(const auto &axis:std::array<std::array<double,4>,2>{{{a,d,-back,front},{b,e,-half_width,half_width}}}) {
    if(std::abs(axis[1])<1e-14) {if(axis[0]<axis[2] || axis[0]>axis[3]){low=2;break;}}
    else {
      double l=(axis[2]-axis[0])/axis[1],h=(axis[3]-axis[0])/axis[1];
      if(l>h)std::swap(l,h);low=std::max(low,l);high=std::min(high,h);
    }
  }
  if(low<=high)return true;
  const auto point_rectangle=[&](double px,double py) {
    const double dx=std::max({-back-px,0.,px-front});
    const double dy=std::max(0.,std::abs(py)-half_width);return dx*dx+dy*dy;
  };
  double distance=std::min(point_rectangle(a,b),point_rectangle(a+d,b+e));
  const double length=d*d+e*e;
  for(double px:{-back,front})for(double py:{-half_width,half_width}) {
    const double t=length>0?std::clamp(((px-a)*d+(py-b)*e)/length,0.,1.):0.;
    distance=std::min(distance,std::pow(px-a-t*d,2)+std::pow(py-b-t*e,2));
  }
  return distance<=radius*radius+1e-12;
}
// Exact continuous SAT for an axis-aligned square translating along a segment
// against the complete oriented asymmetric vehicle rectangle. Contact is a
// collision. No temporal sampling or bounding-circle shrinkage is involved.
inline bool bodyOverlapsTranslatingCell(double x,double y,double yaw,
    double front,double back,double half_width,double ax,double ay,double bx,double by,double side)
{
  for(double v:{x,y,yaw,front,back,half_width,ax,ay,bx,by,side})if(!std::isfinite(v))return true;
  if(front<=0 || back<=0 || half_width<=0 || side<=0)return true;
  const double c=std::cos(yaw),s=std::sin(yaw),half_length=(front+back)/2.;
  const double cx=x+(front-back)*.5*c,cy=y+(front-back)*.5*s;
  const std::array<std::array<double,2>,4> axes={{{c,s},{-s,c},{1,0},{0,1}}};
  double low=0,high=1;
  for(const auto &n:axes) {
    const double radius=half_length*std::abs(n[0]*c+n[1]*s)+
      half_width*std::abs(-n[0]*s+n[1]*c)+side*.5*(std::abs(n[0])+std::abs(n[1]));
    const double p=n[0]*(ax-cx)+n[1]*(ay-cy),d=n[0]*(bx-ax)+n[1]*(by-ay);
    if(std::abs(d)<1e-14) {if(std::abs(p)>radius+1e-12)return false;continue;}
    double a=(-radius-p)/d,b=(radius-p)/d;if(a>b)std::swap(a,b);
    low=std::max(low,a);high=std::min(high,b);
    if(low>high+1e-12)return false;
  }
  return true;
}
}  // namespace terrain_nav
