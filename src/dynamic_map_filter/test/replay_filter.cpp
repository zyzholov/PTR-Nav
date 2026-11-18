// Native deterministic qualification on actual recorded scans/estimated poses.
// ROI and floor arguments are evaluation-only; neither filter receives them.
#include "dynamic_map_filter/dufomap_filter.hpp"
#include "dynamic_map_filter/freedom_filter.hpp"
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <stdexcept>

using namespace ptr_dynamic_map;
using Key=std::array<int,3>;
static Key key(const Point& p) {return {int(std::floor(p[0]/.1f)),int(std::floor(p[1]/.1f)),int(std::floor(p[2]/.1f))};}
template<class T> void read(std::ifstream& f,T& x){f.read(reinterpret_cast<char*>(&x),sizeof(x));}
int main(int argc,char** argv) {
  if(argc!=8) {std::cerr<<"usage: replay_filter stream backend output.csv roi_x roi_y floor_z vacuum_stamp\n";return 2;}
  const std::string backend=argv[2];
  if(backend!="dufomap" && backend!="freedom" && backend!="freedom_depth")return 2;
  DufomapFilter dufo;FreedomFilter freedom(backend=="freedom_depth");
  std::ifstream input(argv[1],std::ios::binary);std::ofstream out(argv[3]);
  std::array<char,8> magic{};input.read(magic.data(),magic.size());
  if(std::string(magic.data(),magic.size())!="PTRMAP01")throw std::runtime_error("invalid measured stream");
  const double cx=std::stod(argv[4]),cy=std::stod(argv[5]),floor=std::stod(argv[6]),vacate=std::stod(argv[7]);
  std::map<Key,Point> historical,ground;std::set<Key> labels;
  out<<"stamp_s,age_s,input_points,old_obstacle_points,retained_ground_points,integration_ms,nodes\n";
  while(input.peek()!=EOF) {
    double stamp;Point origin;std::array<float,9> rotation;uint32_t count;
    read(input,stamp);read(input,origin);read(input,rotation);read(input,count);
    if(!input || count>1000000)throw std::runtime_error("invalid scan header");
    std::vector<Point> points(count);input.read(reinterpret_cast<char*>(points.data()),count*sizeof(Point));
    if(!input)throw std::runtime_error("truncated scan");
    Metrics m;
    if(backend=="dufomap")m=dufo.integrate(points,origin);
    else {m=freedom.integrate(points,origin,rotation);labels.clear();for(auto p:freedom.dynamicPoints())labels.insert(key(p));}
    const auto dynamic=[&](Point p){return backend=="dufomap"?dufo.seenFree(p):labels.count(key(p))>0;};
    for(auto p:points) {
      if(std::abs(p[0]-cx)>1.15 || std::abs(p[1]-cy)>.95)continue;
      if(p[2]>floor+.30 && !dynamic(p))historical[key(p)]=p;
      else if(std::abs(p[2]-floor)<.10 && !dynamic(p))ground[key(p)]=p;
    }
    for(auto it=historical.begin();it!=historical.end();)if(dynamic(it->second))it=historical.erase(it);else ++it;
    for(auto it=ground.begin();it!=ground.end();)if(dynamic(it->second))it=ground.erase(it);else ++it;
    out<<stamp<<','<<stamp-vacate<<','<<count<<','<<historical.size()<<','<<ground.size()<<','<<m.integration_ms<<','<<m.nodes<<'\n';
  }
}
