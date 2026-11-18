#include "dynamic_map_filter/dufomap_filter.hpp"
#include <ufo/map/ufomap.hpp>
#include <ufo/map/integration/integration.hpp>
#include <ufo/map/predicate/spatial.hpp>
#include <tbb/global_control.h>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace ptr_dynamic_map {
static Config checked(Config c) {
  if(!std::isfinite(c.resolution) || c.resolution<.05f || c.resolution>.3f ||
     !std::isfinite(c.surface_margin) || c.surface_margin<.1f || c.surface_margin>.5f ||
     !std::isfinite(c.max_range) || c.max_range<2.f || c.max_range>100.f ||
     c.unknown_margin_cells<1 || c.unknown_margin_cells>4 || c.threads<1 || c.threads>8 ||
     c.max_nodes<10000 || !std::isfinite(c.min_range) || c.min_range<0 || c.min_range>=c.max_range)
    throw std::invalid_argument("invalid DUFOMap adapter geometry/resource limits");
  return c;
}
struct DufomapFilter::Impl {
  using Map=ufo::Map<ufo::MapType::SEEN_FREE | ufo::MapType::REFLECTION>;
  Config config;
  tbb::global_control threads;
  std::unique_ptr<Map> map;
  ufo::IntegrationParams params;
  explicit Impl(Config c):config(checked(c)),threads(tbb::global_control::max_allowed_parallelism,config.threads) {
    params.min_range=c.min_range;params.max_range=c.max_range;
    params.inflate_unknown=c.unknown_margin_cells;
    params.inflate_unknown_compensation=true;
    params.inflate_hits_dist=c.surface_margin;
    params.ray_passthrough_hits=false;
    params.ray_casting_method=ufo::RayCastingMethod::PROPER;
    params.down_sampling_method=ufo::DownSamplingMethod::CENTER;
    params.parallel=true;params.num_threads=c.threads;
    reset();
  }
  void reset() {map=std::make_unique<Map>(config.resolution,17);params.time=1;}
};
DufomapFilter::DufomapFilter(Config c):impl_(std::make_unique<Impl>(c)){}
DufomapFilter::~DufomapFilter()=default;
Metrics DufomapFilter::integrate(const std::vector<Point>& points,const Point& origin) {
  const auto start=std::chrono::steady_clock::now();
  for(float v:origin)if(!std::isfinite(v))throw std::invalid_argument("nonfinite LiDAR origin");
  Metrics m;
  if(impl_->map->numNodes()>impl_->config.max_nodes) {impl_->reset();m.reset_for_budget=true;}
  ufo::PointCloud cloud;cloud.reserve(points.size());
  const float lo=impl_->config.min_range*impl_->config.min_range;
  const float hi=impl_->config.max_range*impl_->config.max_range;
  for(const auto &p:points) {
    if(!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2]))continue;
    const float dx=p[0]-origin[0],dy=p[1]-origin[1],dz=p[2]-origin[2];
    const float r=dx*dx+dy*dy+dz*dz;
    if(r>lo && r<hi)cloud.push_back({ufo::Point(p[0],p[1],p[2])});
  }
  m.input_points=cloud.size();
  if(!cloud.empty())ufo::insertPointCloud(*impl_->map,std::move(cloud),
      ufo::Point(origin[0],origin[1],origin[2]),impl_->params,true);
  m.nodes=impl_->map->numNodes();m.memory_bytes=impl_->map->memoryUsageAllocated();
  m.integration_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
  return m;
}
bool DufomapFilter::seenFree(const Point& p) const {
  if(!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2]))return false;
  return impl_->map->seenFree(ufo::Point(p[0],p[1],p[2]));
}
std::vector<Point> DufomapFilter::dynamicVoxels(std::size_t maximum) const {
  std::vector<Point> out;
  if(maximum==0)return out;
  for(auto n:impl_->map->query(ufo::pred::Leaf(0) && ufo::pred::SeenFree() && ufo::pred::HitsMin(1))) {
    const auto p=impl_->map->center(n);out.push_back({p.x,p.y,p.z});
    if(out.size()>=maximum)break;
  }
  return out;
}
float DufomapFilter::resolution() const {return impl_->config.resolution;}
void DufomapFilter::reset(){impl_->reset();}
}
