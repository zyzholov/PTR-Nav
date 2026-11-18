// Recorded router geometry and fused map only. No simulator or truth feedback.
#define TERRAIN_NAV_BENCHMARK
#include "../src/terrain_nav_node.cpp"
#include <fstream>
#include <filesystem>
#include <iostream>
#include <nlohmann/json.hpp>
#include <rclcpp/serialization.hpp>
#include <rcl/time.h>
using J=nlohmann::json;
struct TerrainNavBenchmark {
 template<class T> static T read(const std::filesystem::path &p){std::ifstream f(p,std::ios::binary|std::ios::ate);if(!f)throw std::runtime_error(p.string());auto n=f.tellg();rclcpp::SerializedMessage b(static_cast<size_t>(n));auto &r=b.get_rcl_serialized_message();f.seekg(0);f.read(reinterpret_cast<char*>(r.buffer),n);r.buffer_length=n;T m;rclcpp::Serialization<T>().deserialize_message(&b,&m);return m;}
 static J run(const std::filesystem::path &p,double gx,double gy){
  TerrainNavNode n(rclcpp::NodeOptions(),false);
  auto odom=read<nav_msgs::msg::Odometry>(p/"odom.cdr");
  auto *clock=n.get_clock()->get_clock_handle();
  if(rcl_enable_ros_time_override(clock)!=RCL_RET_OK ||
      rcl_set_ros_time_override(clock,rclcpp::Time(odom.header.stamp).nanoseconds())!=RCL_RET_OK)
    throw std::runtime_error("cannot set recorded clock");
  n.odomCallback(std::make_shared<nav_msgs::msg::Odometry>(odom));n.mapCallback(std::make_shared<se2_grid_msgs::msg::SE2Grid>(read<se2_grid_msgs::msg::SE2Grid>(p/"map.cdr")));
  if(!n.map_ready_)throw std::runtime_error("recorded contract rejected");
  auto g=read<nav_msgs::msg::OccupancyGrid>(p/"topology_memory.cdr");
  std::vector<terrain_nav::RouteMemory::Point> seen,occupied;
  for(size_t i=0;i<g.data.size();++i){terrain_nav::RouteMemory::Point q{g.info.origin.position.x+(i%g.info.width+.5)*g.info.resolution,g.info.origin.position.y+(i/g.info.width+.5)*g.info.resolution};if(g.data[i]==100)occupied.push_back(q);else if(g.data[i]==0)seen.push_back(q);}
  n.route_memory_.observe(rclcpp::Time(g.header.stamp).seconds(),{g.info.origin.position.x,g.info.origin.position.y,g.info.origin.position.x+g.info.width*g.info.resolution,g.info.origin.position.y+g.info.height*g.info.resolution},seen,occupied);
  n.route_memory_epoch_=n.input_epoch_.load();
  n.topology_guide_enable_=true;
  OdomState state=n.odom_;Pose2 active;std::vector<Pose2> unused;
  n.resolveTopologyGoalLocked(state,Pose2{gx,gy,0,0},active,unused,true);
  J results=J::array();
  for(int mode:{0,1,2})for(double budget:{24.,200.}){
   terrain_nav::RouteOptions o;o.heuristic_weight=2.5;o.allow_partial=true;
   const auto cost=[&](double x,double y,double yaw){if(mode==1)return 1.;double c=n.topologyTerrainCostLocked(x,y,yaw);return mode==2?std::min(c,3.):c;};
   auto r=n.route_memory_.route({n.odom_.x,n.odom_.y},{gx,gy},n.odom_.yaw,.5,n.maxReferenceCurvature(),.68,.58,cost,false,false,0,budget,o);
   J probes=J::array();for(size_t i=0;i<r.path.size();i+=4){const auto &q=r.path[i];auto c=n.queryCostLocked(q[0],q[1],r.yaws[i]);probes.push_back({{"xy",q},{"yaw",r.yaws[i]},{"cost",cost(q[0],q[1],r.yaws[i])},{"known",c.known},{"p",c.probability_upper}});}
   results.push_back({{"mode",mode},{"budget_ms",budget},{"reached",r.reached},{"partial",r.partial},{"expanded",r.expanded},{"reason",r.reason},{"path",r.path},{"yaws",r.yaws},{"probes",probes}});
  }
  return {{"scope","offline actual recorded router memory/map; diagnostic counterfactual costs are not executed paths"},{"start",{n.odom_.x,n.odom_.y,n.odom_.yaw}},{"goal",{gx,gy}},{"occupied",occupied.size()},{"runs",results}};
 }
};
int main(int argc,char**argv){if(argc<5)return 2;auto p=std::filesystem::path(argv[1]);std::string out=argv[2];double gx=std::stod(argv[3]),gy=std::stod(argv[4]);rclcpp::init(argc,argv);try{auto r=TerrainNavBenchmark::run(p,gx,gy);std::ofstream(out)<<r.dump(2)<<'\n';rclcpp::shutdown();return 0;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';rclcpp::shutdown();return 1;}}
