#include <gtest/gtest.h>
#include "terrain_nav/mobility_recovery.hpp"
using terrain_nav::MobilityRecovery;
using Action=MobilityRecovery::Action;

TEST(MobilityRecovery,SpinningOrMovingVehicleCannotRestartBeforeAStableSupportedStop) {
  MobilityRecovery r;r.goal(1);ASSERT_TRUE(r.stall(10.,2));
  EXPECT_EQ(r.tick(10.2,-.55,true,true),Action::Hold);
  EXPECT_EQ(r.tick(10.3,0.,true,true),Action::Hold);
  EXPECT_EQ(r.tick(10.7,-.04,true,true),Action::Hold);
  EXPECT_EQ(r.tick(10.8,0.,true,true),Action::Hold);
  EXPECT_EQ(r.tick(11.2,0.,true,true),Action::Hold);
  EXPECT_EQ(r.tick(11.31,0.,true,true),Action::Replan);
  EXPECT_EQ(r.tick(11.4,0.,true,true),Action::None);
  EXPECT_EQ(r.attempts(),1);
}

TEST(MobilityRecovery,StaleUnknownOrIntermittentSupportCannotSupplyStoppedConfirmation) {
  for(int mode=0;mode<2;++mode) {
    MobilityRecovery r;r.goal(1);ASSERT_TRUE(r.stall(10.,2));
    EXPECT_EQ(r.tick(10.1,0.,true,true),Action::Hold);
    EXPECT_EQ(r.tick(10.5,0.,mode!=0,mode!=1),Action::Hold);
    EXPECT_EQ(r.tick(10.9,0.,true,true),Action::Hold);
    EXPECT_EQ(r.tick(11.3,0.,true,true),Action::Hold);
    EXPECT_EQ(r.tick(11.41,0.,true,true),Action::Replan);
  }
}

TEST(MobilityRecovery,DuplicateGoalAndFreshMapDoNotReplenishTwoPhysicalRetries) {
  MobilityRecovery r;r.goal(7);
  for(int i=0;i<2;++i) {
    const double t=10.+2*i;ASSERT_TRUE(r.stall(t,2));
    r.goal(7);EXPECT_EQ(r.tick(t+.1,0.,true,true),Action::Hold);
    EXPECT_EQ(r.tick(t+.7,0.,true,true),Action::Replan);
  }
  EXPECT_FALSE(r.stall(15.,2));EXPECT_EQ(r.tick(15.1,0.,true,true),Action::Latched);
  r.goal(7);EXPECT_TRUE(r.latched());
  r.goal(8);EXPECT_FALSE(r.latched());EXPECT_EQ(r.attempts(),0);EXPECT_TRUE(r.stall(16.,2));
}

TEST(MobilityRecovery,DisabledAllowanceClockResetAndUnconfirmedStopFailClosed) {
  MobilityRecovery r;r.goal(1);EXPECT_FALSE(r.stall(10.,0));EXPECT_TRUE(r.latched());
  r.goal(2);ASSERT_TRUE(r.stall(11.,2));EXPECT_EQ(r.tick(10.,0.,true,true),Action::Latched);
  r.goal(3);ASSERT_TRUE(r.stall(12.,2));EXPECT_EQ(r.tick(17.01,0.,false,false),Action::Latched);
}
