// Fixed computational workload only. No navigation, vehicle feasibility,
// external-object calibration or physical-success claim.
#include "terrain_nav/joint_motion_risk.hpp"
#include <nlohmann/json.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
namespace terrain_nav {
bool jointMotionCudaBatchHits(const std::vector<joint_detail::Actor>&,const std::vector<joint_detail::Pose>&,
  const std::vector<int>&,size_t,uint64_t,std::vector<unsigned>&,std::string&);
bool jointMotionCudaSharedBatchHits(const std::vector<joint_detail::Actor>&,const std::vector<joint_detail::Pose>&,
  const std::vector<int>&,const std::vector<double>&,const std::vector<int>&,
  size_t,uint64_t,std::vector<unsigned>&,std::string&);
}
int main(int argc,char **argv)
{
  try {
    if(argc!=2||std::filesystem::exists(argv[1]))throw std::runtime_error("new JSON output required");
    std::string error;if(!terrain_nav::jointMotionGpuAvailable(&error))throw std::runtime_error(error);
    std::vector<terrain_nav::joint_detail::Actor> actors(3);
    for(int k=0;k<3;++k) {
      auto &a=actors[k];a.mean[0]=4.+2*k;a.mean[1]=k==1?-2.5:2.5;
      a.mean[3]=k==1?.8:-.8;a.root[0]=a.root[5]=.018;a.root[10]=a.root[15]=.14;
      a.age=.2;a.density=.04;a.radius=.5;a.maximum_speed=4.5;
    }
    std::vector<terrain_nav::joint_detail::Pose> poses;
    std::vector<double> times;std::vector<int> offsets{0},indices;
    for(int i=0;i<=250;++i)times.push_back(.02*i);
    for(int c=0;c<172;++c) {
      const double lateral=(c/171.-.5)*4.;
      for(int i=0;i<=250;++i) {
        const double t=times[i],yaw=std::atan2(.2*lateral*std::cos(t),1.2);
        poses.push_back({t,1.2*t,lateral+.2*lateral*std::sin(t),std::cos(yaw),std::sin(yaw),.521,.521,.460,.065});
        indices.push_back(i);
      }
      offsets.push_back(poses.size());
    }
    const size_t samples=256;std::vector<unsigned> hits(actors.size()*172);
    const auto run=[&](bool shared) {
      const auto start=std::chrono::steady_clock::now();
      const bool ok=shared?terrain_nav::jointMotionCudaSharedBatchHits(actors,poses,offsets,times,indices,samples,1234,hits,error):
          terrain_nav::jointMotionCudaBatchHits(actors,poses,offsets,samples,1234,hits,error);
      if(!ok)throw std::runtime_error(error);
      return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    };
    run(false);run(true); // Both allocation/kernel warmups excluded.
    nlohmann::json results=nlohmann::json::array();
    for(int i=0;i<5;++i) {
      double legacy=0,common=0;
      if(i%2){common=run(true);legacy=run(false);}else{legacy=run(false);common=run(true);}
      results.push_back({{"iteration",i},{"per_candidate_forecast_ms",legacy},{"common_forecast_ms",common}});
    }
    nlohmann::json out={{"scope","fixed synthetic GPU compute microbenchmark; not navigation or risk calibration"},
      {"candidates",172},{"actors",3},{"samples",samples},{"times",times.size()},{"measurements",results},
      {"shared_forecast_bytes",actors.size()*samples*times.size()*sizeof(terrain_nav::joint_detail::Segment)},
      {"coupling_change","legacy uses candidate-specific PRNG streams; common screening uses identical actor draws across candidates. They have the same per-candidate model law, not identical sample counts. CPU/GPU/order parity is tested separately. Final fresh-sample confidence bounds are unchanged."}};
    std::ofstream(argv[1])<<out.dump(2)<<'\n';std::cout<<out.dump()<<'\n';
    return 0;
  }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
