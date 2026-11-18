#include "dynamic_map_filter/freedom_filter.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <freedom/mrmap.h>
#include <freedom/raycast.h>
#include <set>
#include <algorithm>
using namespace ptr_dynamic_map;
const std::array<float,9> identity{1,0,0,0,1,0,0,0,1};

namespace {
using Cell=std::array<int,3>;
Cell cell(const freedom::Point& p,double size) {
  return {int(std::floor(p.x()/size)),int(std::floor(p.y()/size)),int(std::floor(p.z()/size))};
}
// Independent slab intersection, deliberately NOT the production DDA.
std::vector<std::pair<double,Cell>> intersected(const freedom::Point& origin,const freedom::Point& end,double size) {
  const auto a=cell(origin,size),b=cell(end,size);std::vector<std::pair<double,Cell>> out;
  for(int x=std::min(a[0],b[0])-1;x<=std::max(a[0],b[0])+1;++x)
  for(int y=std::min(a[1],b[1])-1;y<=std::max(a[1],b[1])+1;++y)
  for(int z=std::min(a[2],b[2])-1;z<=std::max(a[2],b[2])+1;++z) {
    const Cell q{x,y,z};double enter=0.,leave=1.;bool hit=true;
    for(int axis=0;axis<3;++axis) {
      const double low=q[axis]*size,high=low+size,v=end[axis]-origin[axis];
      if(v==0.){if(origin[axis]<low||origin[axis]>=high)hit=false;continue;}
      double u=(low-origin[axis])/v,w=(high-origin[axis])/v;if(u>w)std::swap(u,w);
      enter=std::max(enter,u);leave=std::min(leave,w);
    }
    if(hit&&leave>enter+1e-10&&enter<1.&&leave>0.)out.push_back({enter,q});
  }
  std::sort(out.begin(),out.end());return out;
}
}

TEST(Freedom, RayTraversalMatchesSegmentBoxesAtPositiveAndNegativeGridBoundaries) {
  const double size=.5;
  for(const auto& origin:std::vector<freedom::Point>{{0.,0.,.25},{-.5,.17,.31},{.5,-.5,.23},{-.13,.11,-.5}})
  for(const auto& end:std::vector<freedom::Point>{{3.17,1.31,1.73},{-2.13,-.71,1.13},{1.71,-2.31,-.87},
                                               {3.17,0.,.25},{-1.13,0.,.25}}) {
    const auto oi=cell(origin,size),ei=cell(end,size);std::set<Cell> expected,actual;
    for(const auto& item:intersected(origin,end,size))if(item.second!=ei)expected.insert(item.second);
    freedom::Index pos,start(oi[0],oi[1],oi[2]),finish(ei[0],ei[1],ei[2]);
    freedom::RayCaster ray(origin,start,pos,size);ray.setRayEnd(end,finish);int steps=0;
    do {
      if(pos==finish)break;
      // A negative ray at a start boundary exits its address cell at t=0;
      // production visits it, but it supplies no positive-length free ray.
      Cell q{pos.x(),pos.y(),pos.z()};if(steps>0||expected.count(q))actual.insert(q);
      ASSERT_LT(++steps,1000);
    }while(ray.step());
    EXPECT_EQ(actual,expected)<<"origin="<<origin.transpose()<<" end="<<end.transpose();
  }
}

TEST(Freedom, FreeRayEvidenceUsesEveryMeasuredEndpointAndStopsAtCurrentOccupiedCells) {
  freedom::MRMap map;freedom::MRMap::MRMapConfig c{};
  c.sub_voxel_size=.1;c.voxel_depth=2;c.block_depth=5;c.enable_local_map=true;
  c.local_map_range=8;c.local_map_min_z=-2;c.local_map_max_z=4;c.num_threads=2;
  c.sensor_max_range=8;c.sensor_min_z=-2;c.sensor_max_z=4;c.raycast_max_range=8;
  c.raycast_min_z=-2;c.raycast_max_z=4;c.counts_to_free=3;c.counts_to_revert=20;
  c.conservative_connectivity=26;c.aggressive_connectivity=124;map.set_params(c);
  freedom::ScanMap scan;freedom::ScanMap::ScanMapConfig s{};
  s.sub_voxel_size=.1;s.voxel_depth=2;s.block_depth=5;s.enable_local_map=true;
  s.local_map_range=8;s.local_map_min_z=-2;s.local_map_max_z=4;s.num_threads=2;
  s.sensor_min_range=.1;s.sensor_max_range=8;s.sensor_min_z=-2;s.sensor_max_z=4;scan.set_params(s);
  const freedom::Point origin(.123,.167,.271);pcl::PointCloud<pcl::PointXYZ> points;
  for(double x:{4.02,4.38})for(double y:{.42,.78})for(double z:{.82,1.18})
    points.push_back(pcl::PointXYZ(x-origin.x(),y-origin.y(),z-origin.z()));
  points.push_back(pcl::PointXYZ(1.31-origin.x(),.23-origin.y(),.47-origin.z()));
  Eigen::Isometry3d transform=Eigen::Isometry3d::Identity();transform.translation()=origin;
  scan.build_scan_map(points,transform);map.scan_removal(scan);
  std::vector<freedom::Point> measured;std::set<Cell> occupied,expected,actual;
  const double size=map.getVoxelSize();
  for(const auto& p:points) {
    const freedom::Point end=transform*freedom::Point(p.x,p.y,p.z);measured.push_back(end);occupied.insert(cell(end,size));
  }
  for(const auto& end:measured)for(const auto& item:intersected(origin,end,size)) {
    if(occupied.count(item.second))break;expected.insert(item.second);
  }
  freedom::DepthImage empty;freedom::Indices incremental;map.freespace_estimation(scan,empty,incremental);
  const auto minimum=map.get_raycast_map_min_idx(),shape=map.get_raycast_map_idx_size();
  for(int x=-1;x<=12;++x)for(int y=-1;y<=3;++y)for(int z=-1;z<=4;++z) {
    freedom::Index voxel(x,y,z),block;freedom::LinearIndex vi;map.getBlockIdxFromVoxelIdx(voxel,block);
    const auto d=(block-minimum).eval();if((d.array()<0).any()||(d.array()>=shape.array()).any())continue;
    const auto bi=d.x()*shape.y()*shape.z()+d.y()*shape.z()+d.z();
    map.getLocalVoxelLinearIdxFromVoxelIdx(voxel,vi);
    if(map.get_raycast_blocks()[bi].trversed_voxels[vi/64].load()&(1ULL<<(vi%64)))actual.insert({x,y,z});
  }
  EXPECT_EQ(actual,expected);
  for(const auto& q:occupied)EXPECT_FALSE(actual.count(q));
  // Many beams/duplicates in ONE scan must not satisfy the THREE-scan
  // evidence requirement. Local ray bits are unioned before temporal votes.
  const auto count=map.getVoxel2blockMultiples();
  for(const auto& block:map.get_free_blocks())if(!block.second.is_free())
    for(unsigned i=0;i<count*count*count;++i)EXPECT_LE(block.second.getFreeVoxel(i).free_counter,1u);
}

TEST(Freedom, MissingFramesDoNotDeclareTheInitialWallOrOccludedSpaceDynamic) {
  FreedomFilter f;std::vector<Point> wall;
  for(int y=-15;y<=15;++y)for(int z=-5;z<=15;++z)wall.push_back({3.,y*.1f,z*.1f});
  for(int i=0;i<5;++i)f.integrate(wall,{0,0,1},identity);
  EXPECT_TRUE(f.dynamicPoints().empty());
  for(int i=0;i<30;++i)f.integrate({},{0,0,1},identity);
  EXPECT_TRUE(f.dynamicPoints().empty());
}

TEST(Freedom, RejectsNonfinitePoseAndNonRigidRotation) {
  FreedomFilter f;auto wrong=identity;wrong[0]=2.;
  EXPECT_THROW(f.integrate({},{0,0,1},wrong),std::invalid_argument);
  wrong=identity;wrong[0]=NAN;
  EXPECT_THROW(f.integrate({},{0,0,1},wrong),std::invalid_argument);
  EXPECT_THROW(f.integrate({},{NAN,0,1},identity),std::invalid_argument);
}

TEST(Freedom, StaticSlopedGroundWithMovingSensorIsRetained) {
  FreedomFilter f;std::vector<Point> floor;
  for(int x=-25;x<=25;++x)for(int y=-25;y<=25;++y)
    floor.push_back({x*.1f,y*.1f,.06f*x*.1f+.04f*y*.1f});
  for(float x:{-.5f,0.f,.5f,1.f})f.integrate(floor,{x,.2f,1.f+.06f*x},identity);
  EXPECT_TRUE(f.dynamicPoints().empty());
}

TEST(Freedom, HistoricalMapAppliesTheConfiguredSecondAggressiveRing) {
  // Isolate map-label propagation, without raycast or a simulated ground truth.
  // Same-scan occupied cells form a line: seed -> moderate -> aggressive ->
  // second aggressive ring. A disconnected fifth cell must remain static.
  freedom::MRMap map;freedom::MRMap::MRMapConfig c{};
  c.sub_voxel_size=.1;c.voxel_depth=2;c.block_depth=5;
  c.enable_local_map=true;c.local_map_range=8;c.local_map_min_z=-2;c.local_map_max_z=4;
  c.num_threads=1;c.sensor_max_range=8;c.sensor_min_z=-2;c.sensor_max_z=4;
  c.raycast_max_range=8;c.raycast_min_z=-2;c.raycast_max_z=4;
  c.counts_to_free=3;c.counts_to_revert=20;c.conservative_connectivity=26;c.aggressive_connectivity=124;
  map.set_params(c);
  freedom::ScanMap scan;freedom::ScanMap::ScanMapConfig s{};
  s.sub_voxel_size=.1;s.voxel_depth=2;s.block_depth=5;
  s.enable_local_map=true;s.local_map_range=8;s.local_map_min_z=-2;s.local_map_max_z=4;
  s.num_threads=1;s.sensor_min_range=.1;s.sensor_max_range=8;s.sensor_min_z=-2;s.sensor_max_z=4;
  scan.set_params(s);pcl::PointCloud<pcl::PointXYZ> points;
  for(float x:{.2f,.6f,1.f,1.4f,2.2f})points.push_back(pcl::PointXYZ(x,.2f,1.f));
  scan.build_scan_map(points,Eigen::Isometry3d::Identity());map.scan_removal(scan);
  map.staticspace_integration(scan,7);
  map.map_removal({freedom::Index(0,0,2)});
  const auto level=[&](float x) {
    freedom::Index sub,voxel,block;freedom::LinearIndex vi,si;
    map.getSubVoxelIdxFromPoint(freedom::Point(x,.2,1.),sub);
    map.getVoxelIdxFromSubVoxelIdx(sub,voxel);map.getBlockIdxFromVoxelIdx(voxel,block);
    map.getLocalVoxelLinearIdxFromVoxelIdx(voxel,vi);map.getLocalSubVoxelLinearIdxFromSubvoxelIdx(sub,si);
    return map.get_static_blocks().at(block).getStaticVoxel(vi).dynamic_level[si];
  };
  EXPECT_EQ(level(.2f),freedom::DynamicLevel::CONSERVATIVE_DYNAMIC);
  EXPECT_EQ(level(.6f),freedom::DynamicLevel::MODERATE_DYNAMIC);
  EXPECT_EQ(level(1.f),freedom::DynamicLevel::AGGRESSIVE_DYNAMIC);
  EXPECT_EQ(level(1.4f),freedom::DynamicLevel::AGGRESSIVE_DYNAMIC);
  EXPECT_EQ(level(2.2f),freedom::DynamicLevel::STATIC);
}
