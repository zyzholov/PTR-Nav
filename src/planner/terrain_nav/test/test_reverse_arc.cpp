#include <gtest/gtest.h>
#include "terrain_nav/reverse_arc.hpp"

TEST(ReverseArc, RecoveryTravelFollowsHeadingAndRetainsTheConfiguredLimit) {
  auto distances=terrain_nav::topologyRecoveryDistances(1.4,.7547,4.);
  ASSERT_FALSE(distances.empty());EXPECT_NEAR(distances.front(),2.,1e-9);
  EXPECT_EQ(distances.back(),4.);
  for(double d:distances)EXPECT_LE(d,4.);
  EXPECT_EQ(terrain_nav::topologyRecoveryDistances(-1.4,.7547,4.),distances);
  distances=terrain_nav::topologyRecoveryDistances(1.4,.7547,.6);
  ASSERT_EQ(distances.size(),2u);EXPECT_EQ(distances[0],.6);EXPECT_EQ(distances[1],.4);
  EXPECT_TRUE(terrain_nav::topologyRecoveryDistances(1.,0.,4.).empty());
}

TEST(ReverseArc, StraightMaintainsBodyHeading)
{
  const auto path = terrain_nav::reverseArc(0, 0, 0, -2, 0, 4, 0.7);
  ASSERT_GT(path.size(), 20u);
  EXPECT_NEAR(path.back()[0], -2, 1e-6);
  for (const auto &p : path) EXPECT_NEAR(p[2], 0, 1e-6);
}

TEST(ReverseArc, NeosNearbyRearGoalNeedsStoppedGearChangeInsteadOfImpossibleSingleArc)
{
  const double offset=.215,k=1./std::sqrt(1.34*1.34-offset*offset);
  // Relative terminal pose of the 2026-11-11 recorded turning failure.
  const double gx=-.5454674958414,gy=-.1917677776949;
  EXPECT_TRUE(terrain_nav::reverseGoalRegionArcs(0,0,0,gx,gy,.1,1.,k,.025,offset).empty());
  const auto proposals=terrain_nav::terminalGearManeuvers(0,0,0,gx,gy,.1,3.,k,.025,offset);
  ASSERT_FALSE(proposals.empty());
  bool forward_first=false;
  for(const auto &p:proposals) {
    EXPECT_LE(p.length,3.);ASSERT_GE(p.first.size(),2u);ASSERT_GE(p.second.size(),2u);
    EXPECT_EQ(p.first.front(),(std::array<double,3>{0.,0.,0.}));
    EXPECT_EQ(p.first.back(),p.second.front());
    EXPECT_LE(std::hypot(p.second.back()[0]-gx,p.second.back()[1]-gy),.1);
    forward_first|=p.first_gear>0;
    for(const auto *leg:{&p.first,&p.second})for(size_t i=1;i<leg->size();++i) {
      const auto &a=leg->at(i-1),&b=leg->at(i);
      const double ds=std::hypot(b[0]-a[0],b[1]-a[1]);
      ASSERT_GT(ds,0.);EXPECT_LE(ds,.025001);
      EXPECT_LE(std::abs(b[2]-a[2])/ds,k+1e-4);
    }
  }
  EXPECT_TRUE(forward_first);
  EXPECT_TRUE(terrain_nav::terminalGearManeuvers(0,0,0,gx,gy,.1,0.,k).empty());
  EXPECT_TRUE(terrain_nav::terminalGearManeuvers(0,0,0,gx,gy,.1,3.,k,NAN).empty());
  EXPECT_TRUE(terrain_nav::terminalGearManeuvers(0,0,NAN,gx,gy,.1,3.,k).empty());
}

TEST(ReverseArc, LateralTargetAndRotatedWorld)
{
  const auto path = terrain_nav::reverseArc(5, 3, M_PI/2, 4.7, 1, 4, 0.7);
  ASSERT_FALSE(path.empty());
  EXPECT_NEAR(path.back()[0], 4.7, 1e-6);
  EXPECT_NEAR(path.back()[1], 1, 1e-6);
  EXPECT_LT(std::abs(path.back()[2] - M_PI/2), 0.5);
}

TEST(ReverseArc, RejectsForwardTooFarAndTooTight)
{
  EXPECT_TRUE(terrain_nav::reverseArc(0, 0, 0, 2, 0, 4, .7).empty());
  EXPECT_TRUE(terrain_nav::reverseArc(0, 0, 0, -5, 0, 4, .7).empty());
  EXPECT_TRUE(terrain_nav::reverseArc(0, 0, 0, -.5, 1, 4, .7).empty());
}

TEST(ReverseArc, TerminalGapAndInvalidSpacing)
{
  const auto path=terrain_nav::reverseArc(0,0,0,-.27,0,4,.7);
  ASSERT_FALSE(path.empty());
  EXPECT_NEAR(path.back()[0],-.27,1e-6);
  EXPECT_TRUE(terrain_nav::reverseArc(0,0,0,-1,0,4,.7,0).empty());
  EXPECT_TRUE(terrain_nav::reverseArc(0,0,0,-1,0,4,.7,NAN).empty());
}

TEST(ReverseArc, OffsetTerminalGoalUsesOriginalDiskWithoutExceedingCurvature)
{
  EXPECT_TRUE(terrain_nav::reverseArc(0,0,0,-.27,-.05,4,.7).empty());
  for (double lateral : {-.05,.05}) {
    const auto path=terrain_nav::reverseGoalRegionArc(0,0,0,-.27,lateral,.22,.5,.7);
    ASSERT_GT(path.size(),1u);
    EXPECT_LE(std::hypot(path.back()[0]+.27,path.back()[1]-lateral),.22);
    for (size_t i=1;i<path.size();++i) {
      const double ds=std::hypot(path[i][0]-path[i-1][0],path[i][1]-path[i-1][1]);
      EXPECT_LE(ds,.025001);
      EXPECT_LT(path[i][0],path[i-1][0]);
      EXPECT_LE(std::abs(path[i][2]-path[i-1][2])/ds,.7001);
    }
  }
  EXPECT_TRUE(terrain_nav::reverseGoalRegionArc(0,0,0,.27,0,.22,.5,.7).empty());
  EXPECT_TRUE(terrain_nav::reverseGoalRegionArc(0,0,0,-.1,.45,.22,.5,.7).empty());
  EXPECT_TRUE(terrain_nav::reverseGoalRegionArc(0,0,0,-.8,0,.22,.5,.7).empty());
  EXPECT_TRUE(terrain_nav::reverseGoalRegionArc(0,0,0,-.27,0,.22,.5,.7,0).empty());
}

TEST(ReverseArc, OffsetGoalBeyondHalfMetreCanReachDiskWithoutTighteningTurn)
{
  for (double lateral : {-.4,.4}) {
    EXPECT_TRUE(terrain_nav::reverseArc(0,0,0,-.84,lateral,4,.7).empty());
    const auto path=terrain_nav::reverseGoalRegionArc(0,0,0,-.84,lateral,.22,1.5,.7);
    ASSERT_GT(path.size(),2u);
    EXPECT_LE(std::hypot(path.back()[0]+.84,path.back()[1]-lateral),.22);
    for (size_t i=1;i<path.size();++i) {
      const double ds=std::hypot(path[i][0]-path[i-1][0],path[i][1]-path[i-1][1]);
      EXPECT_LE(std::abs(path[i][2]-path[i-1][2])/ds,.7001);
    }
  }
}

TEST(ReverseArc, TerminalDiskRetainsBothSteeringDirectionsForSafetyCertification)
{
  const auto paths=terrain_nav::reverseGoalRegionArcs(.0685,.169,.214,-.1846,.0147,.22,.45,.755);
  bool increasing=false,decreasing=false;
  for(const auto &path:paths) {
    ASSERT_GT(path.size(),1u);
    EXPECT_LE(std::hypot(path.back()[0]+.1846,path.back()[1]-.0147),.22);
    increasing|=path.back()[2]>.214;decreasing|=path.back()[2]<.214;
  }
  EXPECT_TRUE(increasing);EXPECT_TRUE(decreasing);
}

#include "terrain_nav/dynamic_yield.hpp"
TEST(BodyReference, RearAxleHasZeroLateralVelocityWhileBaseDoesNot) {
  struct Pose {double x=0.,y=0.,yaw=0.;};
  for(int direction:{-1,1})for(double k:{-.7,.7}) {
    const double a=.2078,v=.5*direction,dt=1e-5;
    const auto p=terrain_nav::integrateBodyVelocity(Pose{},v,v*k,dt,a);
    EXPECT_NEAR(p.y/dt,a*v*k,2e-6);
    const auto rear=terrain_nav::rearAxlePose(p,a);
    EXPECT_NEAR(rear.y/dt,0.,2e-6);
    const auto split=terrain_nav::advanceBodyArc(terrain_nav::advanceBodyArc(Pose{},k,.3*direction,a),k,.6*direction,a);
    const auto single=terrain_nav::advanceBodyArc(Pose{},k,.9*direction,a);
    EXPECT_NEAR(split.x,single.x,1e-12);EXPECT_NEAR(split.y,single.y,1e-12);
  }
}
TEST(BodyReference, AnalyticBaseConnectorAndReverseArcReachBothSidesWithoutSnapping) {
  struct Pose {double x=0.,y=0.,yaw=0.;};
  for(int direction:{-1,1})for(double k:{-.7,0.,.7}) {
    const double a=.2078,length=direction*1.8;
    const auto p=terrain_nav::advanceBodyArc(Pose{},k,length,a);
    const auto arc=terrain_nav::connectBodyArc(p.x,p.y,direction,a);
    ASSERT_TRUE(arc.valid);EXPECT_NEAR(arc.curvature,k,1e-12);EXPECT_NEAR(arc.distance,length,1e-12);
    if(direction<0) {
      const auto reverse=terrain_nav::reverseArc(0,0,0,p.x,p.y,4.,.8,.025,a);
      ASSERT_FALSE(reverse.empty());EXPECT_NEAR(reverse.back()[0],p.x,1e-12);
      EXPECT_NEAR(reverse.back()[1],p.y,1e-12);EXPECT_NEAR(reverse.back()[2],p.yaw,1e-12);
    }
  }
}
TEST(BodyReference, DynamicRolloutUsesTheSameBaseArcAsPlanningInBothGears) {
  for(int direction:{-1,1}) {
    terrain_nav::YieldModel model;model.rear_axle_offset=.2078;model.delay=0.;
    terrain_nav::YieldState initial;initial.speed=.4*direction;initial.steering=.3;
    const std::vector<terrain_nav::YieldCommand> commands={{0,initial.speed,initial.steering}};
    const auto states=terrain_nav::rolloutYield(initial,initial,commands,model,1.);
    const auto expected=terrain_nav::integrateBodyVelocity(initial,initial.speed,
      initial.speed*std::tan(initial.steering)/model.wheelbase,1.,model.rear_axle_offset);
    ASSERT_FALSE(states.empty());EXPECT_NEAR(states.back().x,expected.x,1e-12);
    EXPECT_NEAR(states.back().y,expected.y,1e-12);EXPECT_NEAR(states.back().yaw,expected.yaw,1e-12);
  }
}
