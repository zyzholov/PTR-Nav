// Entire measured masks -> all currently qualified circle observations.
// No truth input, spatial crop, foreground editing, or command publication.
#include "terrain_nav/motion_guard.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
int main(int argc,char **argv)
{
  try {
    if(argc!=3||std::filesystem::exists(argv[2]))throw std::runtime_error("usage: motion_observation_audit RAW_MASKS NEW_JSON");
    std::ifstream input(argv[1]);nlohmann::json raw;input>>raw;
    if(raw.at("overflow").get<bool>()||raw.at("schema_version")!=1)throw std::runtime_error("incomplete raw history");
    terrain_nav::GridMotionGuard guard;nlohmann::json frames=nlohmann::json::array();double previous=-INFINITY;
    for(const auto &f:raw.at("frames")) {
      const auto &g=f.at("geometry");const int width=g[0],height=g[1];const double stamp=f.at("stamp_s");
      if(width<=0||height<=0||!std::isfinite(stamp)||stamp<=previous)throw std::runtime_error("invalid source geometry/time");
      previous=stamp;std::vector<uint8_t> occupied(size_t(width)*height,0);
      for(int id:f.at("occupied")) {
        if(id<0||id>=int(occupied.size())||occupied[id])throw std::runtime_error("invalid measured cell");occupied[id]=1;
      }
      const auto original=occupied;guard.update(occupied,width,height,g[2],g[3],g[4],stamp);
      if(occupied!=original)throw std::runtime_error("raw input mutated");
      nlohmann::json observations=nlohmann::json::array();
      for(const auto &c:guard.measuredCircleObservations())observations.push_back({{"center",c.center},
        {"radius",c.radius},{"radius_bound",c.conservative_radius},{"rms",c.residual_rms},
        {"span",c.observed_span_rad},{"points",c.source_count},{"sensitivity_bound",c.center_error_bound}});
      frames.push_back({{"stamp_s",stamp},{"resolution",g[4]},{"observations",observations}});
    }
    nlohmann::json out={{"source_file",argv[1]},{"scope","all circle hypotheses on complete unchanged measured masks; not truth or semantic labels"},{"frames",frames}};
    std::ofstream output(argv[2]);if(!output)throw std::runtime_error("cannot create output");output<<out.dump(2)<<'\n';
    std::cout<<"audited "<<frames.size()<<" complete frames\n";return 0;
  }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
