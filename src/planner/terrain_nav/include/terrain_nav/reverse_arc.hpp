#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>
#include <vector>
#include "terrain_nav/body_kinematics.hpp"

namespace terrain_nav
{
struct TerminalGearManeuver {
  std::vector<std::array<double,3>> first,second;
  int first_gear=1;
  double length=0.;
};

// A nearby position behind base_link need not admit a single reverse arc:
// the base/rear-axle offset can require a small parking maneuver. Enumerate
// bounded two-gear connectors with exact BODY arcs and a stopped cusp. These
// are geometry proposals only; neither leg grants ground/collision permission.
inline std::vector<TerminalGearManeuver> terminalGearManeuvers(
    double x,double y,double yaw,double gx,double gy,double goal_radius,
    double max_distance,double max_curvature,double spacing=.025,double offset=0.)
{
  std::vector<TerminalGearManeuver> result;
  const double distance=std::hypot(gx-x,gy-y);
  if(!std::isfinite(x)||!std::isfinite(y)||!std::isfinite(yaw)||!std::isfinite(distance)||
      !std::isfinite(goal_radius)||goal_radius<=0||distance<=goal_radius||
      !std::isfinite(max_distance)||max_distance<=distance||
      !std::isfinite(max_curvature)||max_curvature<=0||
      !std::isfinite(spacing)||spacing<=0||!std::isfinite(offset)||offset<0)return result;
  struct Pose {double x,y,yaw;};
  for(int gear:{1,-1})for(double fraction:{0.,-.5,.5,-1.,1.})
    for(double first_length:{.3,.45,.6,.8,1.,1.2}) {
      if(first_length>=max_distance)continue;
      const double k=fraction*max_curvature;
      const auto cusp=advanceBodyArc(Pose{x,y,yaw},k,gear*first_length,offset);
      for(int target=0;target<9;++target) {
        const double angle=(target-1)*M_PI/4.,radius=target?.8*goal_radius:0.;
        const double dx=gx+radius*std::cos(angle)-cusp.x,dy=gy+radius*std::sin(angle)-cusp.y;
        const auto connection=connectBodyArc(std::cos(cusp.yaw)*dx+std::sin(cusp.yaw)*dy,
          -std::sin(cusp.yaw)*dx+std::cos(cusp.yaw)*dy,-gear,offset);
        const double length=first_length+std::abs(connection.distance);
        if(!connection.valid||std::abs(connection.curvature)>max_curvature||
            std::abs(connection.distance)<.1||length>max_distance)continue;
        TerminalGearManeuver proposal;proposal.first_gear=gear;proposal.length=length;
        const int steps1=std::max(2,int(std::ceil(first_length/spacing)));
        const int steps2=std::max(2,int(std::ceil(std::abs(connection.distance)/spacing)));
        for(int i=0;i<=steps1;++i) {
          const auto p=i==steps1?cusp:
            advanceBodyArc(Pose{x,y,yaw},k,gear*first_length*(double(i)/steps1),offset);
          proposal.first.push_back({p.x,p.y,p.yaw});
        }
        for(int i=0;i<=steps2;++i) {
          const auto p=advanceBodyArc(cusp,connection.curvature,connection.distance*i/steps2,offset);
          proposal.second.push_back({p.x,p.y,p.yaw});
        }
        const auto &end=proposal.second.back();
        if(std::hypot(end[0]-gx,end[1]-gy)<=goal_radius)result.push_back(std::move(proposal));
      }
    }
  std::stable_sort(result.begin(),result.end(),[](const auto &a,const auto &b){return a.length<b.length;});
  return result;
}

// Order bounded recovery travel by the distance needed to undo a heading
// error. Every candidate still needs the native retreat AND exit proof.
inline std::vector<double> topologyRecoveryDistances(double heading_error,double max_curvature,double max_distance) {
  std::vector<double> result;
  if(!std::isfinite(heading_error)||!std::isfinite(max_curvature)||!std::isfinite(max_distance)||
      max_curvature<=0||max_distance<=0)return result;
  const double angle=std::abs(std::atan2(std::sin(heading_error),std::cos(heading_error)));
  const double preferred=std::min(max_distance,std::max(.8,std::ceil(angle/max_curvature/.2)*.2));
  for(double distance:{preferred,.8,.4,1.2,1.6,2.,2.8,max_distance}) {
    if(distance>max_distance+1e-9)continue;
    if(std::none_of(result.begin(),result.end(),[&](double value){return std::abs(value-distance)<1e-6;}))
      result.push_back(distance);
  }
  return result;
}

// A single reverse gear, bounded-curvature Ackermann arc. Yaw is BODY yaw.
inline std::vector<std::array<double, 3>> reverseArc(
  double x, double y, double yaw, double gx, double gy,
  double max_distance, double max_curvature, double spacing = 0.05,double rear_axle_offset=0.)
{
  const double c = std::cos(yaw), s = std::sin(yaw);
  const double dx = gx - x, dy = gy - y;
  const double lx = c * dx + s * dy, ly = -s * dx + c * dy;
  const double distance = std::hypot(lx, ly);
  if (!std::isfinite(distance) || !std::isfinite(max_distance) || !std::isfinite(max_curvature) ||
      !std::isfinite(spacing) || spacing<=0 || max_curvature<=0 ||
      !std::isfinite(rear_axle_offset) || rear_axle_offset<0 ||
      lx >= -1e-6 || distance > max_distance || distance < 1e-6) return {};
  const auto connection=connectBodyArc(lx,ly,-1,rear_axle_offset);
  if(!connection.valid)return {};
  const double curvature = connection.curvature;
  if (std::abs(curvature) > max_curvature) return {};
  const double length = connection.distance;
  if (length >= 0.0 || std::abs(length)>max_distance || std::abs(length) > 1.5 * distance) return {};
  const int steps = std::max(2, static_cast<int>(std::ceil(std::abs(length) / spacing)));
  std::vector<std::array<double, 3>> path;
  path.reserve(steps + 1);
  for (int i = 0; i <= steps; ++i) {
    struct Pose {double x,y,yaw;};
    const auto p=advanceBodyArc(Pose{x,y,yaw},curvature,length*i/steps,rear_axle_offset);
    path.push_back({p.x,p.y,p.yaw});
  }
  return path;
}

// A terminal goal is a disk, not an exact point requiring arbitrarily tight
// curvature. Generate only bounded reverse motion; safety is checked by caller.
inline std::vector<std::vector<std::array<double, 3>>> reverseGoalRegionArcs(
  double x, double y, double yaw, double gx, double gy, double goal_radius,
  double max_distance, double max_curvature, double spacing = 0.025,double rear_axle_offset=0.)
{
  const double c=std::cos(yaw),s=std::sin(yaw),dx=gx-x,dy=gy-y;
  const double lx=c*dx+s*dy,ly=-s*dx+c*dy,distance=std::hypot(lx,ly);
  if (!std::isfinite(distance) || !std::isfinite(goal_radius) || goal_radius<=0 ||
      !std::isfinite(max_distance) || max_distance<=0 || !std::isfinite(max_curvature) ||
      max_curvature<=0 || !std::isfinite(spacing) || spacing<=0 ||
      !std::isfinite(rear_axle_offset) || rear_axle_offset<0 ||
      lx>=-1e-6 || distance<=goal_radius || distance>max_distance) return {};
  const auto connection=connectBodyArc(lx,ly,-1,rear_axle_offset);
  const double toward=std::clamp(connection.valid?connection.curvature:0.,-max_curvature,max_curvature);
  const int steps=std::max(2,static_cast<int>(std::ceil(max_distance/spacing)));
  std::vector<std::vector<std::array<double,3>>> candidates;
  for (double curvature : {0.,toward,-max_curvature,-.5*max_curvature,
                            .5*max_curvature,max_curvature}) {
    std::vector<std::array<double,3>> path={{x,y,yaw}};
    for (int i=1;i<=steps;++i) {
      struct Pose {double x,y,yaw;};
      const auto p=advanceBodyArc(Pose{x,y,yaw},curvature,-max_distance*i/steps,rear_axle_offset);
      path.push_back({p.x,p.y,p.yaw});
      if (std::hypot(path.back()[0]-gx,path.back()[1]-gy)<=goal_radius) {
        candidates.push_back(std::move(path));
        break;
      }
    }
  }
  return candidates;
}

inline std::vector<std::array<double,3>> reverseGoalRegionArc(
  double x,double y,double yaw,double gx,double gy,double goal_radius,
  double max_distance,double max_curvature,double spacing=.025,double rear_axle_offset=0.)
{
  auto candidates=reverseGoalRegionArcs(x,y,yaw,gx,gy,goal_radius,max_distance,max_curvature,spacing,rear_axle_offset);
  if(candidates.empty()) return {};
  return *std::min_element(candidates.begin(),candidates.end(),
    [](const auto &a,const auto &b){return a.size()<b.size();});
}
}  // namespace terrain_nav
