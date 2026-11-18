#include "dynamic_map_filter/freedom_filter.hpp"
#include <freedom/freedom.h>
#include <chrono>
#include <stdexcept>

namespace ptr_dynamic_map {
struct FreedomFilter::Impl {
  bool enhancement;
  std::unique_ptr<freedom::FreeDOM> core;
  std::vector<Point> dynamic;
  std::size_t nodes=0;
  explicit Impl(bool e):enhancement(e){reset();}
  void reset() {
    dynamic.clear();nodes=0;core=std::make_unique<freedom::FreeDOM>();
    freedom::FreeDOM::Config c{};
    c.sensor_min_range=.2;c.sensor_max_range=40.;c.sensor_min_z=-5.;c.sensor_max_z=8.;
    c.sub_voxel_size=.1;c.voxel_depth=2;c.block_depth=5;
    c.enable_local_map=true;c.local_map_range=40.;c.local_map_min_z=-5.;c.local_map_max_z=8.;
    c.raycast_max_range=30.;c.raycast_min_z=-5.;c.raycast_max_z=8.;
    c.counts_to_free=3;c.counts_to_revert=20;
    c.conservative_connectivity=26;c.aggressive_connectivity=124;
    c.enable_raycast_enhancement=enhancement;
    c.lidar_horizon_fov=2.*M_PI;c.lidar_vertical_fov_upper=52.*M_PI/180.;
    c.lidar_vertical_fov_lower=-7.*M_PI/180.;c.depth_image_vertical_lines=96;
    c.depth_image_min_range=.2;c.max_raycast_enhancement_range=30.;c.raycast_enhancement_depth_margin=.4;
    c.inpaint_size=4;c.erosion_size=2;c.min_raycast_enhancement_area=.005;c.depth_image_top_margin=.2;
    c.learn_fov=false;c.enable_fov_mask=false;c.num_threads=2;
    core->set_params(c);
    core->set_raycast_enhancement_callback([](const freedom::DepthImage&){});
    core->set_map_removal_callback([this](const freedom::MRMap& map) {
      dynamic.clear();nodes=0;
      const auto nv=map.getVoxel2blockMultiples(),ns=map.getSubvoxel2voxelMultiples();
      for(const auto& entry:map.get_static_blocks()) {
        const auto block_origin=map.getBlockSize()*entry.first.cast<double>();
        freedom::Index vi=freedom::Index::Zero();
        for(unsigned i=0;i<nv*nv*nv;++i) {
          if(entry.second.is_static_voxel_allocated(i)) {
            const auto& voxel=entry.second.getStaticVoxel(i);++nodes;
            for(unsigned j=0;j<ns*ns*ns;++j) {
              if(voxel.scan_in_subvoxel[j]!=freedom::StaticVoxel::NOT_A_SCAN &&
                 voxel.dynamic_level[j]>freedom::DynamicLevel::STATIC) {
                const freedom::Point p=block_origin+map.getVoxelSize()*vi.cast<double>()+voxel.points[j].cast<double>();
                dynamic.push_back({float(p.x()),float(p.y()),float(p.z())});
              }
            }
          }
          freedom::incrementIdx(vi,nv);
        }
      }
    });
  }
};
FreedomFilter::FreedomFilter(bool e):impl_(std::make_unique<Impl>(e)){}
FreedomFilter::~FreedomFilter()=default;
Metrics FreedomFilter::integrate(const std::vector<Point>& points,const Point& origin,const std::array<float,9>& rotation) {
  const auto start=std::chrono::steady_clock::now();
  Eigen::Isometry3d transform=Eigen::Isometry3d::Identity();
  for(int i=0;i<3;++i) {
    transform.translation()[i]=origin[i];
    for(int j=0;j<3;++j)transform.linear()(i,j)=rotation[i*3+j];
  }
  if(!transform.matrix().allFinite() || (transform.linear().transpose()*transform.linear()-Eigen::Matrix3d::Identity()).norm()>.001)
    throw std::invalid_argument("invalid LiDAR pose for FreeDOM");
  pcl::PointCloud<pcl::PointXYZ> cloud;cloud.reserve(points.size());
  for(const auto& p:points) {
    const Eigen::Vector3d v(p[0],p[1],p[2]);if(!v.allFinite())continue;
    const Eigen::Vector3d local=transform.inverse()*v;
    if(local.norm()>.2 && local.norm()<40.)cloud.push_back(pcl::PointXYZ(local.x(),local.y(),local.z()));
  }
  Metrics m;
  if(impl_->nodes>300000){impl_->reset();m.reset_for_budget=true;}
  if(!cloud.empty())impl_->core->pointcloud_integrate(cloud,transform);
  m.input_points=cloud.size();m.nodes=impl_->nodes;
  m.integration_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
  return m;
}
const std::vector<Point>& FreedomFilter::dynamicPoints() const {return impl_->dynamic;}
void FreedomFilter::reset(){impl_->reset();}
}
