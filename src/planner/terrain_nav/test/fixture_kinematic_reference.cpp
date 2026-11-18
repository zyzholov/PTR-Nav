// OFFLINE fixture admissibility diagnosis. No ROS interfaces and no navigation
// inputs. Authored geometry is evaluation-only, never a runtime planner map.
#include <nlohmann/json.hpp>
#include "terrain_nav/body_kinematics.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <queue>
#include <unordered_map>
#include <vector>
using J=nlohmann::json;
struct Object {double x,y,l,w,r;bool circle;};
struct State {double x,y,yaw,g;int parent,gear;};
int main(int argc,char **argv) {
  if(argc!=3)return 2;J input;std::ifstream file(argv[1]);file>>input;
  std::vector<Object> objects;
  for(const auto &o:input.at("objects"))if(!o.at("terrain").get<bool>()) {
    const bool circle=o.at("shape")=="cylinder";
    objects.push_back({o.at("x"),o.at("y"),o.value("length",0.),o.value("width",0.),o.value("radius",0.),circle});
  }
  const double front=.68,back=.58,half=.50,kmax=.98*std::tan(.4)/.549185;
  const double rear_axle_offset=input.value("rear_axle_offset",.2078);
  const auto clear=[&](const State &p) {
    if(p.x<-2 || p.x>17 || std::abs(p.y)>6.)return false;
    const double c=std::cos(p.yaw),s=std::sin(p.yaw),a=(front+back)/2;
    const double x=p.x+(front-back)*.5*c,y=p.y+(front-back)*.5*s;
    for(const auto &o:objects) {
      const double dx=o.x-x,dy=o.y-y,u=c*dx+s*dy,v=-s*dx+c*dy;
      if(o.circle) {
        const double du=std::max(0.,std::abs(u)-a),dv=std::max(0.,std::abs(v)-half);
        if(du*du+dv*dv<=o.r*o.r)return false;
      } else {
        if(std::abs(dx)>std::abs(c)*a+std::abs(s)*half+o.l/2 ||
           std::abs(dy)>std::abs(s)*a+std::abs(c)*half+o.w/2 ||
           std::abs(u)>a+std::abs(c)*o.l/2+std::abs(s)*o.w/2 ||
           std::abs(v)>half+std::abs(s)*o.l/2+std::abs(c)*o.w/2)continue;
        return false;
      }
    }
    return true;
  };
  const auto key=[](const State &p) {
    const int x=static_cast<int>(std::floor(p.x/.1))+1000,y=static_cast<int>(std::floor(p.y/.1))+1000;
    const double angle=std::atan2(std::sin(p.yaw),std::cos(p.yaw));
    const int yaw=std::clamp(int(std::floor((angle+M_PI)*72/(2*M_PI))),0,71);
    return (uint64_t(x)*2000+y)*144+yaw*2+(p.gear<0);
  };
  const auto start=input.at("start").get<std::vector<double>>(),goal=input.at("goal").get<std::vector<double>>();
  const auto distance=[&](const State &p){return std::hypot(p.x-goal[0],p.y-goal[1]);};
  std::vector<State> nodes={{start[0],start[1],start[2],0,-1,1}};
  std::unordered_map<uint64_t,double> best;best[key(nodes[0])]=0;
  using Item=std::pair<double,int>;std::priority_queue<Item,std::vector<Item>,std::greater<Item>> open;
  open.push({distance(nodes[0]),0});int chosen=-1,expanded=0;
  const auto began=std::chrono::steady_clock::now();std::string reason="search exhausted";
  while(!open.empty()) {
    const int id=open.top().second;open.pop();const auto p=nodes[id];
    if(p.g>best[key(p)]+1e-8)continue;
    if(!clear(p))continue;
    if(distance(p)<=.25){chosen=id;reason="geometric route found";break;}
    if(++expanded>500000 || (expanded%128==0 &&
        std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count()>5.)){
      reason="bounded reference search; feasibility unresolved";break;
    }
    for(int gear:{1,-1})for(double ratio:{-1.,-.5,0.,.5,1.}) {
      State next=p;bool safe=true;const double k=ratio*kmax,ds=.05*gear;
      for(int step=0;step<8;++step) {
        next=terrain_nav::advanceBodyArc(next,k,ds,rear_axle_offset);
        if(!clear(next)){safe=false;break;}
      }
      if(!safe)continue;
      next.g=p.g+.4*(gear>0?1.:1.8)+.08*std::abs(ratio)+(gear==p.gear?0.:.8);
      next.gear=gear;next.parent=id;const auto address=key(next);
      if(best.count(address) && best[address]<=next.g+1e-8)continue;
      best[address]=next.g;nodes.push_back(next);open.push({next.g+distance(next),int(nodes.size()-1)});
    }
  }
  J path=J::array();std::vector<int> ids;
  for(int i=chosen;i!=-1;i=nodes[i].parent)ids.push_back(i);std::reverse(ids.begin(),ids.end());
  for(int i:ids)path.push_back({nodes[i].x,nodes[i].y,nodes[i].yaw,nodes[i].gear});
  J result={{"reached",chosen>=0},{"reason",reason},{"expanded",expanded},{"path",path},
    {"protected_body",{front,back,half}},{"max_curvature",kmax},{"rear_axle_offset",rear_axle_offset},
    {"elapsed_ms",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-began).count()},
    {"scope","offline authored-geometry full-body SE2 diagnosis; not a traction, timing or real navigation proof; no truth feedback"}};
  std::ofstream output(argv[2]);output<<result.dump(2)<<'\n';result.erase("path");std::cout<<result.dump()<<'\n';
  return 0;
}
