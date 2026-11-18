// Deterministic replay of actual pre-inflation sensor occupancy, not truth.
#include "terrain_nav/motion_guard.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

int main(int argc,char **argv)
{
  using Json=nlohmann::json;
  try {
    if(argc<3 || std::filesystem::exists(argv[2]))
      throw std::runtime_error("usage: motion_guard_replay INPUT_JSON NEW_OUTPUT_JSON [AGGREGATION_FACTOR] [--preserve-components] [--gaussian] [--sim-observation-calibration] [--known-slow-centers] [--component-link-cells=N] [--acceleration-bound=VALUE]");
    int factor=1;bool preserve=false,gaussian=false,calibrated=false,turning=false;double acceleration=0.,identity_memory=3.;
    int component_link_cells=1;bool known_slow_centers=false;
    for(int i=3;i<argc;++i) {
      const std::string flag=argv[i];
      if(i==3 && flag.find("--")!=0)factor=std::stoi(flag);
      else if(flag=="--preserve-components")preserve=true;
      else if(flag=="--gaussian")gaussian=true;
      else if(flag=="--sim-observation-calibration")calibrated=true;
      else if(flag=="--turning-identity")turning=true;
      else if(flag=="--known-slow-centers")known_slow_centers=true;
      else if(flag.find("--component-link-cells=")==0)component_link_cells=std::stoi(flag.substr(23));
      else if(flag.find("--identity-memory=")==0)identity_memory=std::stod(flag.substr(18));
      else if(flag.find("--acceleration-bound=")==0)acceleration=std::stod(flag.substr(21));
      else throw std::invalid_argument("unknown replay flag: "+flag);
    }
    if(factor<1 || factor>4)throw std::runtime_error("aggregation factor must be 1..4");
    if(component_link_cells<1 || component_link_cells>3 || (factor!=1 && component_link_cells!=1))
      throw std::runtime_error("component link must be 1..3 cells and requires native resolution");
    if(!std::isfinite(acceleration)||acceleration<0||acceleration>4.5 || (calibrated&&!gaussian))
      throw std::invalid_argument("invalid disclosed replay motion model");
    std::ifstream input(argv[1]);Json data;input>>data;
    const bool tracking=data.contains("tracking_frames")&&!data.at("tracking_frames").empty();
    const auto &recording=data.at(tracking?"tracking_frames":"frames");
    if(data.at("schema_version")!=1 || data.at("overflow").get<bool>() ||
       recording.size()<2 || recording.size()>1500)
      throw std::runtime_error("incomplete or unsupported motion recording");
    terrain_nav::GridMotionGuard guard;guard.acceleration_bound_mps2=acceleration;
    guard.fit_gaussian_motion=gaussian;guard.sim_observation_calibration=calibrated;
    guard.identity_memory_seconds=identity_memory;guard.turning_identity_reacquisition=turning;
    guard.track_known_slow_centers=known_slow_centers;
    guard.component_link_cells=component_link_cells;
    Json frames=Json::array();std::string frame;
    for(const auto &sample:recording) {
      const std::string current_frame=sample.at("frame");
      if(current_frame.empty() || (!frame.empty() && frame!=current_frame))
        throw std::runtime_error("motion recording changes frame");
      frame=current_frame;
      const auto &g=sample.at("geometry");
      const int width=g.at(0),height=g.at(1);
      if(width<1 || height<1 || width>1024 || height>1024)
        throw std::runtime_error("invalid grid dimensions");
      if(width%factor || height%factor)throw std::runtime_error("aggregation must match live complete-square geometry");
      std::vector<uint8_t> raw(static_cast<size_t>(width)*height,0);
      for(int cell:sample.at("occupied")) {
        if(cell<0 || cell>=static_cast<int>(raw.size()) || raw[cell])
          throw std::runtime_error("invalid or duplicate occupied index");
        raw[cell]=1;
      }
      const auto begin=std::chrono::steady_clock::now();
      std::vector<uint8_t> forecast;
      if(factor==1)forecast=guard.update(raw,width,height,g.at(2),g.at(3),g.at(4),sample.at("stamp_s"));
      else {
        const int cw=width/factor,ch=height/factor;std::vector<uint8_t> coarse(cw*ch,0);
        std::vector<std::vector<int>> groups;
        if(preserve) {
          auto measured=terrain_nav::aggregateMotionComponents(raw,width,height,factor);
          coarse=std::move(measured.occupied);groups=std::move(measured.components);
        } else for(int y=0;y<height;++y)for(int x=0;x<width;++x)
          if(raw[y*width+x])coarse[(y/factor)*cw+x/factor]=1;
        const auto projected=guard.update(coarse,cw,ch,g.at(2),g.at(3),g.at(4).get<double>()*factor,sample.at("stamp_s"),1.5,groups);
        forecast.resize(raw.size());
        for(int y=0;y<height;++y)for(int x=0;x<width;++x)forecast[y*width+x]=projected[(y/factor)*cw+x/factor];
      }
      const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
      Json occupied=Json::array();
      for(size_t i=0;i<forecast.size();++i)if(forecast[i])occupied.push_back(i);
      const auto tracks=[](const auto &input) {
        Json out=Json::array();
        for(const auto &track:input)out.push_back({{"id",track.id},{"stamp_s",track.stamp},
          {"velocity",{track.vx,track.vy}},{"resolution",track.resolution},{"cells",track.cells},
          {"circular",track.circular},{"center",{track.center_x,track.center_y}},
          {"radius_bound",track.radius_bound},{"velocity_error",track.velocity_error},{"fit_rms",track.fit_rms},
          {"acceleration_bound",track.acceleration_bound},{"prediction_maximum_speed",track.prediction_maximum_speed},
          {"motion_confirmed",track.motion_confirmed},{"causal_center_fit",track.center_motion_fit},
          {"retained_radius",track.retained_radius}});
        return out;
      };
      frames.push_back({{"stamp_s",sample.at("stamp_s")},{"geometry",g},
        {"moving_components",guard.moving_components},{"forecast",occupied},{"compute_ms",ms},
        {"tracks",tracks(guard.snapshot())},{"brake_only_tracks",tracks(guard.provisionalSnapshot())},
        {"stationary_resolutions",tracks(guard.resolvedSnapshot())}});
    }
    Json result={{"scope","deterministic measured-mask replay; not physical navigation acceptance"},
      {"input_stream",tracking?"tracking_frames":"frames (legacy body-only input)"},
      {"motion_guard_sha256",MOTION_GUARD_SHA},{"aggregation_factor",factor},{"preserve_measured_components",preserve},
      {"acceleration_bound_mps2",acceleration},{"fit_gaussian",gaussian},{"sim_observation_calibration",calibrated},
      {"identity_memory_seconds",identity_memory},{"turning_identity_reacquisition",turning},
      {"component_link_cells",component_link_cells},{"known_slow_centers",known_slow_centers},
      {"horizon_s",1.5},{"frames",frames}};
    std::ofstream output(argv[2]);if(!output)throw std::runtime_error("cannot create replay result");
    output<<result.dump(2)<<'\n';
    std::cout<<"replayed "<<frames.size()<<" measured frames\n";
    return 0;
  } catch(const std::exception &error) {
    std::cerr<<error.what()<<'\n';return 1;
  }
}
