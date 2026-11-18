#include "terrain_nav/route_memory.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>

int main(int argc,char **argv) {
  if(argc!=3)return 2;
  using J=nlohmann::json;J input;std::ifstream file(argv[1]);file>>input;
  terrain_nav::RouteMemory memory;
  memory.observe(input.at("stamp"),input.at("window").get<std::array<double,4>>(),
      input.at("seen").get<std::vector<terrain_nav::RouteMemory::Point>>(),
      input.at("occupied").get<std::vector<terrain_nav::RouteMemory::Point>>());
  const auto begin=std::chrono::steady_clock::now();
  const auto result=memory.route(input.at("start").get<terrain_nav::RouteMemory::Point>(),
      input.at("goal").get<terrain_nav::RouteMemory::Point>(),input.at("yaw"),input.value("radius",.5),input.value("max_curvature",.7547),
      input.value("front",.68),input.value("back",.58),{},input.value("fine",false),
      input.value("allow_reverse",false),input.value("initial_direction",0));
  J out={{"scope","optimistic routing replay only, not physical navigation or a terrain safety certificate"},
         {"reached",result.reached},{"reason",result.reason},{"fine",result.fine},{"expanded",result.expanded},
         {"path",result.path},{"yaws",result.yaws},{"directions",result.directions},
         {"elapsed_ms",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count()}};
  std::ofstream output(argv[2]);output<<out.dump(2)<<'\n';out.erase("path");out.erase("yaws");std::cout<<out.dump()<<'\n';
}
