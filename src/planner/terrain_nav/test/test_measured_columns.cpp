#include "terrain_nav/measured_columns.hpp"
#include <gtest/gtest.h>
#include <random>

namespace {
TEST(MeasuredColumns, RawMotionSurfaceRetainsOnlyExistingAboveFloorBodyAndWitnessReturns)
{
  terrain_nav::ColumnEvidence columns;columns.occupied.assign(9,0);columns.ground.assign(9,0.);
  columns.occupied[4]=1;auto tracking=columns.occupied;tracking[6]=1;
  std::vector<float> ceiling(9,INFINITY);ceiling[4]=.7;
  std::vector<std::array<float,3>>points={{{1.2F,1.2F,.02F}},{{1.3F,1.2F,.5F}},
    {{1.4F,1.2F,.8F}},{{.2F,2.2F,.2F}},{{.3F,2.2F,1.F}},{{.4F,2.2F,2.F}},{{.2F,.2F,.5F}}};
  const auto original=columns.occupied;
  const auto result=terrain_nav::measuredMotionSurfacePoints(points,columns,tracking,3,3,0,0,1.,.1,1.5,ceiling);
  ASSERT_EQ(result.size(),2u);EXPECT_NEAR(result[0][0],1.3F,1e-8);EXPECT_NEAR(result[1][0],.3F,1e-8);
  EXPECT_EQ(columns.occupied,original);EXPECT_FALSE(columns.occupied[6]);EXPECT_TRUE(tracking[6]);
}

auto detect(const std::vector<std::array<float,3>> &points,const std::vector<float> &ground={}) {
  return terrain_nav::measuredColumns(points,20,20,0,0,.2,.85,.32F,2,ground);
}

TEST(NearestFiniteSources, ExactLexicographicBruteForceAtEveryCellAndRadius) {
  std::mt19937 rng(20261107);
  for(int width:{1,2,7,23})for(int height:{1,3,11,19})for(int density:{0,5,50,100}) {
    std::vector<float> sources(width*height,NAN);
    for(auto &v:sources)if(int(rng()%100)<density)v=float(rng()%50);
    for(int radius:{0,1,3,30}) {
      const auto actual=terrain_nav::nearestFiniteSources(sources,width,height,radius*radius);
      for(int i=0;i<width*height;++i) {
        int best=-1,distance=radius*radius+1;
        for(int j=0;j<width*height;++j)if(std::isfinite(sources[j])) {
          const int dx=i%width-j%width,dy=i/width-j/width,d=dx*dx+dy*dy;
          if(d<distance){distance=d;best=j;}
        }
        EXPECT_EQ(actual[i],best)<<width<<'x'<<height<<" density="<<density<<" radius="<<radius<<" cell="<<i;
      }
    }
  }
  EXPECT_THROW(terrain_nav::nearestFiniteSources({},1,1,1),std::invalid_argument);
}

TEST(MeasuredColumns, WitnessRectanglesContainAllOriginalPaddedReturnsAndKeepMissingInvalid) {
  const std::vector<std::array<float,3>> points={{{.021F,.011F,.4F}},{{.045F,.041F,0.F}},{{.08F,.02F,.5F}},{{NAN,0,0}}};
  const std::vector<uint8_t> occupied={1,1,1};
  const auto rectangles=terrain_nav::columnWitnessRectangles(points,occupied,3,1,0.,0.,.05);
  ASSERT_TRUE(rectangles[0].valid());ASSERT_TRUE(rectangles[1].valid());EXPECT_FALSE(rectangles[2].valid());
  for(const auto &p:points)if(std::isfinite(p[0])) {
    const int cell=int(p[0]/.05);const auto &r=rectangles[cell];
    EXPECT_LE(r.x0,std::max(cell*.05,double(p[0])-.02));
    EXPECT_GE(r.x1,std::min((cell+1)*.05,double(p[0])+.02));
    EXPECT_LE(r.y0,std::max(0.,double(p[1])-.02));
    EXPECT_GE(r.y1,std::min(.05,double(p[1])+.02));
  }
  EXPECT_EQ(occupied,(std::vector<uint8_t>{1,1,1}));
  EXPECT_THROW(terrain_nav::columnWitnessRectangles(points,{},3,1,0,0,.05),std::invalid_argument);
}
}

TEST(ColumnInflation, WholeSquareDiskIntersectionCannotLoseTheRequestedReserve) {
  for(double resolution:{.1,.2,.4})for(double radius:{.01,.25,.35,.45}) {
    std::vector<uint8_t> raw(21*21,0);raw[10*21+10]=1;
    const auto mask=terrain_nav::inflateColumns(raw,21,21,resolution,radius);
    for(int y=0;y<21;++y)for(int x=0;x<21;++x) {
      const double dx=std::max(0.,(std::abs(x-10)-1)*resolution);
      const double dy=std::max(0.,(std::abs(y-10)-1)*resolution);
      EXPECT_EQ(mask[y*21+x]!=0,std::hypot(dx,dy)<=radius+1e-12);
    }
    EXPECT_EQ(std::count(raw.begin(),raw.end(),1),1);
  }
}

TEST(ColumnInflation, PartialCellsAndEdgesAreConservativeWithoutMapWrap) {
  std::vector<uint8_t> raw(7*7,0);raw[0]=1;
  const auto mask=terrain_nav::inflateColumns(raw,7,7,.2,.25);
  // x=.3 starts 20cm from the original square's far edge. A 25cm
  // reserve intersects it even though its center is 40cm from the seed.
  EXPECT_EQ(mask[2],1);
  EXPECT_EQ(mask[2*7+2],0);
  EXPECT_EQ(mask[6],0);EXPECT_EQ(mask[6*7],0);
  EXPECT_EQ(terrain_nav::inflateColumns(raw,7,7,.2,0.),raw);
  EXPECT_THROW(terrain_nav::inflateColumns(raw,7,7,.2,-.1),std::invalid_argument);
  EXPECT_THROW(terrain_nav::inflateColumns({},7,7,.2,.1),std::invalid_argument);
}

TEST(MeasuredColumns, SparseUpperReturnsNeedMeasuredGroundAndOwnReturn) {
  const std::vector<std::array<float,3>> points={{{1.9F,2.1F,.6F}},{{2.1F,2.1F,.6F}}};
  auto result=detect(points);
  EXPECT_EQ(std::count(result.occupied.begin(),result.occupied.end(),1),0);
  result=detect(points,std::vector<float>(400,0));
  EXPECT_TRUE(result.occupied[10*20+9]);EXPECT_TRUE(result.occupied[10*20+10]);
  EXPECT_EQ(std::count(result.occupied.begin(),result.occupied.end(),1),2);
  EXPECT_FALSE(result.occupied[10*20+11]);
}

TEST(MeasuredColumns, SparseNeighbourSearchMatchesBruteForceIncludingDistanceTiesAndUnknowns) {
  std::mt19937 rng(20261102);
  constexpr int width=23,height=19;
  const float nan=std::numeric_limits<float>::quiet_NaN();
  for(double resolution:{.05,.1,.2})for(int seed=0;seed<6;++seed) {
    std::vector<float> measured(width*height,nan);
    for(auto &z:measured)if(rng()%6==0)z=float(int(rng()%10)-5)*.04F;
    const auto result=terrain_nav::measuredColumns({},width,height,0,0,resolution,.85,.18F,1,measured);
    auto witnesses=measured;
    for(int y=0;y<height;++y)for(int x=0;x<width;++x)
      for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx) {
        const int nx=x+dx,ny=y+dy;
        if((!dx&&!dy)||nx<0||ny<0||nx>=width||ny>=height)continue;
        const float jump=measured[y*width+x]-measured[ny*width+nx];
        if(std::isfinite(jump)&&jump>std::max(.18,std::hypot(dx,dy)*resolution))witnesses[y*width+x]=nan;
      }
    const int radius=std::ceil(.85/resolution);
    for(int y=0;y<height;++y)for(int x=0;x<width;++x) {
      float expected=witnesses[y*width+x];int best=std::isfinite(expected)?0:std::numeric_limits<int>::max();
      for(int dy=-radius;dy<=radius;++dy)for(int dx=-radius;dx<=radius;++dx) {
        const int nx=x+dx,ny=y+dy,d=dx*dx+dy*dy;
        if(!d||d>radius*radius||nx<0||ny<0||nx>=width||ny>=height)continue;
        if(d<best&&std::isfinite(witnesses[ny*width+nx])) {best=d;expected=witnesses[ny*width+nx];}
      }
      const float actual=result.ground[y*width+x];
      EXPECT_TRUE(actual==expected||(std::isnan(actual)&&std::isnan(expected)))<<seed<<" "<<x<<" "<<y;
      EXPECT_FALSE(result.occupied[y*width+x]);
    }
  }
}

TEST(MeasuredColumns, SharedGroundRelativeCeilingRetainsLowBeamAndPillarButNotOverheadPanel) {
  std::vector<float> ceiling(400,.75F),ground(400,0);
  std::vector<std::array<float,3>> points={{{1.9F,2.1F,0}},{{1.9F,2.1F,1.1F}},
    {{2.1F,2.1F,.5F}},{{2.3F,2.1F,.6F}}};
  const auto mask=terrain_nav::measuredColumns(points,20,20,0,0,.2,.85,.32F,1,ground,ceiling);
  EXPECT_FALSE(mask.occupied[10*20+9]);EXPECT_TRUE(mask.occupied[10*20+10]);EXPECT_TRUE(mask.occupied[10*20+11]);
  ceiling[10*20+9]=std::numeric_limits<float>::infinity();
  const auto unknown=terrain_nav::measuredColumns(points,20,20,0,0,.2,.85,.32F,1,ground,ceiling);
  EXPECT_TRUE(unknown.occupied[10*20+9]);
}

TEST(MeasuredColumns, KnownSlopeAndChangingViewDoNotBecomePillars) {
  std::vector<std::array<float,3>> points;std::vector<float> ground(400);
  for(int y=0;y<20;++y)for(int x=0;x<20;++x) {
    const float z=.5F*x*.2F;ground[y*20+x]=z;
    points.push_back({(x+.5F)*.2F,(y+.5F)*.2F,z});
  }
  auto result=detect(points,ground);
  EXPECT_EQ(std::count(result.occupied.begin(),result.occupied.end(),1),0);
  result=detect(points);
  EXPECT_EQ(std::count(result.occupied.begin(),result.occupied.end(),1),0);
}

TEST(MeasuredColumns, CurrentVerticalSpanRemainsAnObstacleWithoutMeasuredGround) {
  std::vector<std::array<float,3>> points;
  for(int z=0;z<12;++z)points.push_back({1.9F,2.1F,z*.1F});
  const auto result=detect(points);
  EXPECT_TRUE(result.occupied[10*20+9]);
}

TEST(MeasuredColumns, SingleOutlierAndCanopyDoNotCreateCorroboration) {
  const auto result=detect({{{1.9F,2.1F,.8F}},{{2.1F,2.1F,5.F}},{{2.3F,2.1F,5.F}}},std::vector<float>(400,0));
  EXPECT_EQ(std::count(result.occupied.begin(),result.occupied.end(),1),0);
}

TEST(MeasuredColumns, NearestGroundDoesNotWrapMapOrCrossAHeightJump) {
  std::vector<float> ground(400,std::numeric_limits<float>::quiet_NaN());
  ground[0]=0;ground[399]=1;
  const auto result=detect({{{3.9F,3.9F,1.F}},{{3.9F,3.9F,1.01F}}},ground);
  EXPECT_EQ(result.ground[399],1.F);
  EXPECT_EQ(std::count(result.occupied.begin(),result.occupied.end(),1),0);
  const auto empty=detect({},ground);
  EXPECT_EQ(std::count(empty.occupied.begin(),empty.occupied.end(),1),0);
}

TEST(MeasuredColumns, ObstacleTopCannotWitnessItsOwnClearance) {
  std::vector<float> ground(400,0);
  ground[10*20+9]=ground[10*20+10]=.8F;
  const auto result=detect({{{1.9F,2.1F,.8F}},{{2.1F,2.1F,.8F}}},ground);
  EXPECT_TRUE(result.occupied[10*20+9]);EXPECT_TRUE(result.occupied[10*20+10]);
  EXPECT_EQ(result.ground[10*20+9],0.F);EXPECT_EQ(result.ground[10*20+10],0.F);
  EXPECT_EQ(ground[10*20+9],.8F);
}

TEST(MeasuredColumns, DiagonalSlopeGroundWitnessIsNotAnIsolatedTop) {
  for(double resolution:{.2,1.}) {
    std::vector<float> ground(400);std::vector<std::array<float,3>> points;
    for(int y=0;y<20;++y)for(int x=0;x<20;++x) {
      const float z=(.5*x+.4*y)*resolution;ground[y*20+x]=z;
      points.push_back({float((x+.5)*resolution),float((y+.5)*resolution),z});
    }
    const auto result=terrain_nav::measuredColumns(points,20,20,0,0,resolution,.85,.32F,2,ground);
    EXPECT_EQ(result.ground,ground);
    EXPECT_EQ(std::count(result.occupied.begin(),result.occupied.end(),1),0);
  }
}

TEST(MeasuredColumns, LowerGroundHypothesisCannotErasePreviousObstacle) {
  std::vector<float> ground(400,0);
  ground[10*20+9]=ground[10*20+10]=.8F;
  const auto result=detect({{{1.9F,2.1F,2.F}},{{2.1F,2.1F,2.F}}},ground);
  EXPECT_TRUE(result.occupied[10*20+9]);EXPECT_TRUE(result.occupied[10*20+10]);
  EXPECT_EQ(result.ground[10*20+9],0.F);
}

TEST(MeasuredColumns, ConservativeSingleReturnPolicyStillRequiresMeasuredGroundAndRejectsCanopy) {
  const std::vector<std::array<float,3>> points={{{1.9F,2.1F,.6F}},{{3.1F,3.1F,5.F}}};
  const auto result=terrain_nav::measuredColumns(points,20,20,0,0,.2,.85,.32F,1,std::vector<float>(400,0));
  EXPECT_TRUE(result.occupied[10*20+9]);EXPECT_FALSE(result.occupied[15*20+15]);
  const auto unsupported=terrain_nav::measuredColumns(points,20,20,0,0,.2,.85,.32F,1,{});
  EXPECT_EQ(std::count(unsupported.occupied.begin(),unsupported.occupied.end(),1),0);
}

TEST(MeasuredColumns, RoutingClearanceRequiresActualUnmixedGroundInTheSameFineColumn) {
  std::vector<float> measured(400,0.F);
  const auto floor=detect({{1.9F,2.1F,0.F},{1.9F,2.1F,.01F}},measured);
  const size_t cell=10*20+9;
  EXPECT_TRUE(terrain_nav::routingFloorWitness(floor,measured,cell));
  EXPECT_FALSE(terrain_nav::routingFloorWitness(floor,{},cell));
  EXPECT_FALSE(terrain_nav::routingFloorWitness(floor,measured,cell+1));
  for(const float z:{-.20F,.08F,.35F,.8F}) {
    const auto mixed=detect({{1.9F,2.1F,0.F},{1.9F,2.1F,z}},measured);
    EXPECT_FALSE(terrain_nav::routingFloorWitness(mixed,measured,cell));
  }
  measured[cell]=.8F;
  const auto top=detect({{1.9F,2.1F,.8F}},measured);
  EXPECT_FALSE(terrain_nav::routingFloorWitness(top,measured,cell));
}

TEST(MeasuredColumns, RepeatedStampMissingReturnLargeGapAndResetCannotConfirmClearance) {
  using terrain_nav::routingClearStreak;
  EXPECT_EQ(routingClearStreak(0,-1,10.,true),1);
  EXPECT_EQ(routingClearStreak(1,10.,10.,true),1);
  EXPECT_EQ(routingClearStreak(1,10.,10.1,true),2);
  EXPECT_EQ(routingClearStreak(2,10.1,10.2,false),0);
  EXPECT_EQ(routingClearStreak(1,10.,11.,true),1);
  EXPECT_EQ(routingClearStreak(2,10.,9.,true),1);
  EXPECT_EQ(routingClearStreak(1,10.,NAN,true),0);
}
