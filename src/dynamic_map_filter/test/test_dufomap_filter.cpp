#include "dynamic_map_filter/dufomap_filter.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <limits>
using namespace ptr_dynamic_map;
static std::vector<Point> wall(float x) {
  std::vector<Point> p;
  for(int y=-25;y<=25;++y)for(int z=-10;z<=20;++z)p.push_back({x,y*.1f,z*.1f});
  return p;
}
TEST(Dufomap,UnknownAndOcclusionCannotBecomeFreeByTimeoutOrMissingData) {
  DufomapFilter f;const Point origin{0,0,1};
  auto p=wall(3.f);
  for(int i=0;i<5;++i)f.integrate(p,origin);
  EXPECT_FALSE(f.seenFree({3.f,.03f,1.03f}));
  EXPECT_FALSE(f.seenFree({5.f,.03f,1.03f}));
  for(int i=0;i<20;++i)f.integrate({},origin);
  EXPECT_FALSE(f.seenFree({3.f,.03f,1.03f}));
  EXPECT_FALSE(f.seenFree({5.f,.03f,1.03f}));
}
TEST(Dufomap,ConfirmedFreeSpaceClassifiesAnEnteringObjectWithoutErasingItsRawScan) {
  DufomapFilter f;const Point origin{0,0,1};auto background=wall(8.f);
  for(int i=0;i<5;++i)f.integrate(background,origin);
  const Point moving{3.05f,.05f,1.05f};EXPECT_TRUE(f.seenFree(moving));
  auto with_object=background;with_object.push_back(moving);
  f.integrate(with_object,origin);
  EXPECT_EQ(with_object.size(),background.size()+1);
  EXPECT_TRUE(f.seenFree(moving));
  EXPECT_FALSE(f.dynamicVoxels().empty());
  EXPECT_FALSE(f.seenFree({8.f,0.f,1.f}));
}
TEST(Dufomap,RemovingAnInitiallyOccupiedSurfaceRequiresActualLaterRays) {
  DufomapFilter f;const Point origin{0,0,1};auto foreground=wall(3.f);
  for(int i=0;i<3;++i)f.integrate(foreground,origin);
  EXPECT_FALSE(f.seenFree({3.05f,.05f,1.05f}));
  for(int i=0;i<5;++i)f.integrate(wall(8.f),origin);
  EXPECT_TRUE(f.seenFree({3.05f,.05f,1.05f}));
  f.reset();EXPECT_FALSE(f.seenFree({3.05f,.05f,1.05f}));
}
TEST(Dufomap,RejectsInvalidGeometryAndNonfiniteOrigins) {
  Config c;c.resolution=0;EXPECT_THROW(DufomapFilter f(c),std::invalid_argument);
  c=Config{};c.threads=0;EXPECT_THROW(DufomapFilter f(c),std::invalid_argument);
  DufomapFilter f;EXPECT_THROW(f.integrate({}, {0,0,std::numeric_limits<float>::quiet_NaN()}),std::invalid_argument);
}
TEST(Dufomap,StaticSlopedGroundRemainsStaticAsTheSensorMoves) {
  DufomapFilter f;std::vector<Point> ground;
  for(int x=-35;x<=35;++x)for(int y=-35;y<=35;++y)
    ground.push_back({x*.1f,y*.1f,.06f*x*.1f+.04f*y*.1f});
  for(float x:{-.5f,0.f,.5f,1.f})f.integrate(ground,{x,.2f,1.f+.06f*x});
  std::size_t removed=0;for(auto p:ground)removed+=f.seenFree(p);
  EXPECT_EQ(removed,0u);
}
