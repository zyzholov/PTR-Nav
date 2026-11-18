#include "terrain_nav/chance_budget.hpp"
#include "terrain_nav/probability_support.hpp"
#include "terrain_nav/dynamic_yield.hpp"
#include <gtest/gtest.h>
#include <limits>

TEST(ClearanceSpeed, BrakingDistanceIncludesSourceAndActuatorDelay)
{
  for(double distance:{.16,.2,.4,.8,2.})for(double delay:{.35,.55,.8}) {
    const double v=terrain_nav::clearanceSpeedLimit(distance,.16,delay,1.2);
    EXPECT_GE(v,0.);
    EXPECT_NEAR(v*delay+v*v/2.4,std::max(0.,distance-.16),1e-12);
  }
  EXPECT_GT(terrain_nav::clearanceSpeedLimit(2.,.16,.6,1.2),1.2);
  EXPECT_LT(terrain_nav::clearanceSpeedLimit(.4,.16,.6,1.2),.4);
  EXPECT_LT(terrain_nav::clearanceSpeedLimit(.4,.16,.8,1.2),
            terrain_nav::clearanceSpeedLimit(.4,.16,.35,1.2));
}

TEST(ClearanceSpeed, InvalidGeometryFailsClosedAndReverseUsesSameMagnitude)
{
  const double nan=std::numeric_limits<double>::quiet_NaN();
  for(auto distance:{nan,-1.,.15})EXPECT_EQ(terrain_nav::clearanceSpeedLimit(distance,.16,.6,1.2),0.);
  EXPECT_EQ(terrain_nav::clearanceSpeedLimit(.4,.16,.6,0.),0.);
  EXPECT_EQ(terrain_nav::clearanceSpeedLimit(.4,-.16,.6,1.2),0.);
  const double bound=terrain_nav::clearanceSpeedLimit(.4,.16,.6,1.2);
  EXPECT_DOUBLE_EQ(std::clamp(-1.,-bound,bound),-bound);
  EXPECT_GT(terrain_nav::clearanceSpeedLimit(.4,.16,0.,1.2),bound);
}

TEST(ChanceBudget, UnionDoesNotAssumeIndependentCells)
{
  terrain_nav::ChanceBudget risk;
  EXPECT_TRUE(risk.add(1,.04,.05));
  EXPECT_FALSE(risk.add(2,.04,.05));
  EXPECT_NEAR(risk.upper(),.08,1e-12);
}

TEST(ChanceBudget, RevisitCountsSameDiscreteEventOnce)
{
  terrain_nav::ChanceBudget risk;
  EXPECT_TRUE(risk.add(1,.04,.05));
  EXPECT_TRUE(risk.add(1,.04,.05));
  EXPECT_TRUE(risk.add(1,.01,.05));
  EXPECT_NEAR(risk.upper(),.04,1e-12);
  EXPECT_FALSE(risk.add(1,.06,.05));
  EXPECT_NEAR(risk.upper(),.06,1e-12);
}

TEST(ChanceBudget, InvalidProbabilityFailsClosed)
{
  terrain_nav::ChanceBudget risk;
  EXPECT_FALSE(risk.add(1,std::numeric_limits<double>::quiet_NaN(),.1));
  EXPECT_FALSE(risk.add(1,-.1,.1));
  EXPECT_FALSE(risk.add(1,1.1,.1));
}

TEST(ChanceBudget, FootprintUsesActualSupportNotBoundingSquare)
{
  const auto internal=terrain_nav::heightSupportOffsets(.52,.52,.413,.2,M_PI/36,true);
  const auto external=terrain_nav::heightSupportOffsets(.52,.52,.413,.2,M_PI/36,false);
  EXPECT_EQ(internal.size(),33u);
  EXPECT_GT(external.size(),internal.size());
  EXPECT_LT(internal.size(),49u);
  for(const auto &offset:internal)
    EXPECT_NE(std::find(external.begin(),external.end(),offset),external.end());
}

TEST(ChanceBudget, ExactContactStencilExcludesOnlyDisjointGridCornersAndRetainsCrossingSquares)
{
  const double yaw=7*M_PI/36;
  const auto old=terrain_nav::heightSupportOffsets(.52,.52,.413,.2,yaw,true);
  const auto sat=terrain_nav::heightSupportOffsets(.52,.52,.413,.2,yaw,true,true);
  const std::pair<int,int> disjoint{4,1},crossing{3,1};
  EXPECT_NE(std::find(old.begin(),old.end(),disjoint),old.end());
  EXPECT_EQ(std::find(sat.begin(),sat.end(),disjoint),sat.end());
  EXPECT_NE(std::find(sat.begin(),sat.end(),crossing),sat.end());
  EXPECT_LT(sat.size(),old.size());
  for(const auto &q:sat)EXPECT_NE(std::find(old.begin(),old.end(),q),old.end());
}

TEST(ChanceBudget, MixedCertificatesCanAdmitAConfidentInterpolationWithoutLosingSourceReuse)
{
  constexpr uint64_t pose=uint64_t{1}<<63;
  terrain_nav::ChanceBudget risk;
  EXPECT_TRUE(risk.addAlternatives(pose,.025,{1,2,3,4},.001,.05));
  EXPECT_TRUE(risk.addAlternatives(pose+1,.003,{4,5,6,7},.03,.05));
  EXPECT_NEAR(risk.upper(),.007,1e-12);
  EXPECT_TRUE(risk.addAlternatives(pose+1,.003,{4,5,6,7},.03,.05));
  EXPECT_NEAR(risk.upper(),.007,1e-12);
}

TEST(ChanceBudget, OriginalAllSourceCertificateIsNeverLostByGreedyChoices)
{
  constexpr uint64_t pose=uint64_t{1}<<63;
  terrain_nav::ChanceBudget risk;
  for(unsigned i=0;i<30;++i) {
    EXPECT_TRUE(risk.addAlternatives(pose+i,.0009,{1,2,3,4},.001,.005));
    EXPECT_LE(risk.upper(),.004+1e-12);
  }
  EXPECT_FALSE(risk.add(pose+100,.1,.05));
}

TEST(ChanceBudget, InvalidAlternativeCannotReleaseUnknownOrNegativeVariance)
{
  terrain_nav::ChanceBudget risk;
  EXPECT_FALSE(risk.addAlternatives(1,.01,{},.01,.05));
  EXPECT_FALSE(risk.addAlternatives(1,std::numeric_limits<double>::quiet_NaN(),{2},.01,.05));
  EXPECT_FALSE(risk.addAlternatives(1,.01,{2},-.01,.05));
}

TEST(ChanceBudget, HeterogeneousVariancesDoNotUseWorstCellTailEverywhere)
{
  constexpr uint64_t pose=uint64_t{1}<<63;
  terrain_nav::ChanceBudget risk;
  EXPECT_TRUE(risk.addAlternatives(pose,.04,{1,2,3},.02,.05,{.001,.001,.01}));
  EXPECT_NEAR(risk.upper(),.012,1e-12);
  EXPECT_TRUE(risk.addAlternatives(pose+1,.04,{2,3,4},.02,.05,{.001,.01,.001}));
  EXPECT_NEAR(risk.upper(),.013,1e-12);
  EXPECT_FALSE(risk.addAlternatives(pose,.04,{1,2,3},.02,.05,{.001}));
  EXPECT_FALSE(risk.addAlternatives(pose,.04,{1,2,3},.02,.05,{.001,-.1,.001}));
}
