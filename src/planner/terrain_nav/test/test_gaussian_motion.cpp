#include <fstream>
#include <nlohmann/json.hpp>
#include <gtest/gtest.h>
#include "terrain_nav/dynamic_yield.hpp"
#include "terrain_nav/motion_buffer_capacity.hpp"
#include "terrain_nav/planning_budget.hpp"

TEST(JointMotion, CapacityHeadroomIsBoundedAndAvoidsPerFrameReallocation) {
  using terrain_nav::reservedMotionCapacity;
  size_t capacity=0,grows=0;
  for(size_t n=1000;n<1100;++n)if(n>capacity){capacity=reservedMotionCapacity(n,8192);++grows;}
  EXPECT_EQ(grows,1u);EXPECT_GE(capacity,1100u);EXPECT_LE(capacity,8192u);
  const size_t maximum=std::numeric_limits<size_t>::max();
  EXPECT_EQ(reservedMotionCapacity(maximum-1,maximum),maximum);
  EXPECT_EQ(reservedMotionCapacity(8190,8192),8192u);
  EXPECT_THROW(reservedMotionCapacity(8193,8192),std::invalid_argument);
  EXPECT_THROW(reservedMotionCapacity(0,8192),std::invalid_argument);
}

TEST(DynamicYield, TurnReleaseProposalsCanStopAfterDiagonalEscapeWithoutContinuingTheArc) {
  using namespace terrain_nav;
  YieldModel m;m.wheelbase=.43;m.rear_axle_offset=.215;
  m.speed_dead_time=0.;m.steering_dead_time=.06;
  YieldState root;root.speed=1.;
  const auto proposals=turnReleaseYieldProposals(1.33,.31432913231137);
  ASSERT_FALSE(proposals.empty());
  bool distinct_from_arc=false;
  for(const auto &p:proposals) {
    ASSERT_EQ(p.commands.size(),3u);
    EXPECT_GT(p.commands[1].t,0.);EXPECT_GT(p.commands[2].t,p.commands[1].t);
    EXPECT_GT(p.commands[0].speed,0.);EXPECT_LE(p.commands[0].speed,1.33);
    EXPECT_EQ(p.commands[1].speed,p.commands[0].speed);
    EXPECT_EQ(p.commands[1].steering,0.);EXPECT_EQ(p.commands.back().speed,0.);
    const auto path=finiteJointRollout(rolloutQueuedYield(root,p.commands,m,6.,100.,{}));
    ASSERT_FALSE(path.empty());EXPECT_NEAR(path.back().speed,0.,1e-12);
    EXPECT_GE(path.back().t,3.);EXPECT_NEAR(path.back().steering,0.,1e-12);
    const auto at=std::find_if(path.begin(),path.end(),[&](const auto&s){
      return s.t>p.commands[1].t+.2;});
    ASSERT_NE(at,path.end());EXPECT_NEAR(at->steering,0.,1e-12);
    if(std::abs(at->yaw)>.1)distinct_from_arc=true;
    // Constant arc must keep changing heading during the same coast interval.
    auto arc=p.commands;arc.erase(arc.begin()+1);
    const auto turning=rolloutQueuedYield(root,arc,m,path.back().t,100.,{});
    ASSERT_FALSE(turning.empty());
    EXPECT_GT(std::abs(turning.back().yaw),std::abs(path.back().yaw)+.01);
  }
  EXPECT_TRUE(distinct_from_arc);
  EXPECT_TRUE(turnReleaseYieldProposals(NAN,.3).empty());
  EXPECT_TRUE(turnReleaseYieldProposals(1.,0.).empty());
}

TEST(DynamicYield, ForwardEscapeExistsDuringReverseAndRetainsTheBackwardStoppingSweep) {
  using namespace terrain_nav;
  YieldModel m;m.wheelbase=.43;m.rear_axle_offset=.215;
  m.speed_dead_time=.06;m.steering_dead_time=.06;
  YieldState root;root.speed=-.4;
  const auto proposals=forwardYieldProposals(root,m,1.33,.31432913231137,6.,true);
  ASSERT_GT(proposals.size(),50u);
  const auto straight=makeAdvanceCommands(root,m,1.2,1.,0.,6.);
  ASSERT_FALSE(straight.empty());EXPECT_GT(straight.front().speed,0.);
  const auto path=finiteJointRollout(rolloutQueuedYield(root,straight,m,6.,100.,{}));
  ASSERT_FALSE(path.empty());
  EXPECT_LT(path[1].speed,0.);EXPECT_LT(path[1].x,root.x);
  const auto minimum=std::min_element(path.begin(),path.end(),[](const auto&a,const auto&b){return a.x<b.x;});
  // Delay travel plus physical stopping distance must remain in the proof.
  EXPECT_LT(minimum->x,-.4*.06-.4*.4/(2.*m.deceleration)+.005);
  EXPECT_GT(path.back().x,.8);EXPECT_NEAR(path.back().speed,0.,1e-12);
  for(size_t i=1;i<path.size();++i)
    EXPECT_LE(std::abs(path[i].speed-path[i-1].speed),1.2*(path[i].t-path[i-1].t)+1e-9);
  EXPECT_FALSE(makeBypassCommands(root,m,2.,1.,.2,.6,6.).empty());
}

TEST(DynamicYield, FreshnessEffortLimitKeepsAnAlreadyProvedFallback) {
  using namespace terrain_nav;
  double now=73.52;int proofs=0;
  const auto selected=bestCertifiedInteraction(std::vector<double>{1,2,3},std::vector<size_t>{0,1,2},
    [&](size_t){++proofs;now+=.030;return true;},
    [&]{return interactionProofFitsFreshness(now,73.615,.015);});
  EXPECT_EQ(selected,1u);EXPECT_EQ(proofs,2);
  EXPECT_LT(now,73.615); // Preserves time for final validation, not a permit.
}

TEST(DynamicYield, WaitThenAdvanceRetainsTheAbsoluteResumeTimeAfterRefresh) {
  using namespace terrain_nav;
  const auto wait=waitThenAdvance({{0.,.60,.1},{1.2,0.,.1}},.80,.0);
  ASSERT_EQ(wait.size(),3u);
  EXPECT_EQ(yieldCommandAt(wait,.79).speed,0.);
  EXPECT_EQ(yieldCommandAt(wait,.80).speed,.6);
  const auto remainder=remainingYieldCommands(wait,.35);
  ASSERT_EQ(remainder.size(),3u);
  EXPECT_EQ(yieldCommandAt(remainder,.44).speed,0.);
  EXPECT_EQ(yieldCommandAt(remainder,.45).speed,.6);
  EXPECT_DOUBLE_EQ(remainder.back().t,1.65);
  YieldModel model;
  const auto states=finiteJointRollout(rolloutQueuedYield({},wait,model,6.,100.,{}));
  ASSERT_FALSE(states.empty());
  EXPECT_GT(states.back().x,.35);
  EXPECT_NEAR(states.back().speed,0.,1e-12);
  EXPECT_TRUE(waitThenAdvance({{0.,-.5,0.}},.8,0.).empty());
}

TEST(DynamicYield, BudgetedSearchCannotSpendEveryProofOnOneBehaviorFamily) {
  const auto order=terrain_nav::interactionBranchOrder({1,1,1,1,1,1,-1,-1,0,0});
  ASSERT_EQ(order.size(),10u);
  EXPECT_EQ(order[0],0u);EXPECT_EQ(order[1],6u);EXPECT_EQ(order[2],8u);
  auto sorted=order;std::sort(sorted.begin(),sorted.end());
  for(size_t i=0;i<sorted.size();++i)EXPECT_EQ(sorted[i],i);
  EXPECT_EQ(terrain_nav::interactionBranchOrder({-1,-1,1}),
    (std::vector<size_t>{0,2,1}));
  EXPECT_THROW(terrain_nav::interactionBranchOrder({1,2}),std::invalid_argument);
}

TEST(DynamicYield, ScreenHeadroomSeedsFallbackButNeverReplacesProofOrBetterProvenUtility) {
  using namespace terrain_nav;
  const std::vector<int> family{1,1,-1,-1,0};
  const std::vector<double> screen{.04,.03,.045,.001,.02},utility{5,4,-1,-2,-3};
  const auto order=interactionProofOrder(family,screen);
  ASSERT_EQ(order,(std::vector<size_t>{1,3,4,0,2}));
  int checked=0;
  auto prove=[&](size_t i){++checked;return i==3;};
  auto budget=[&]{return checked<2;};
  EXPECT_EQ(bestCertifiedInteraction(utility,interactionBranchOrder(family),prove,budget),std::numeric_limits<size_t>::max());
  checked=0;EXPECT_EQ(bestCertifiedInteraction(utility,order,prove,budget),3u);
  // A low screen value itself grants nothing, and a later fully-proven
  // higher-utility action replaces the first fallback when budget permits.
  EXPECT_EQ(bestCertifiedInteraction(utility,order,[](size_t){return false;},[]{return true;}),std::numeric_limits<size_t>::max());
  EXPECT_EQ(bestCertifiedInteraction(utility,order,[](size_t i){return i==3||i==0;},[]{return true;}),0u);
  EXPECT_THROW(interactionProofOrder({1},{NAN}),std::invalid_argument);
  EXPECT_THROW(interactionProofOrder({1},{-.1}),std::invalid_argument);
  EXPECT_THROW(interactionProofOrder({2},{.1}),std::invalid_argument);
  EXPECT_THROW(interactionProofOrder({1},{}),std::invalid_argument);
}

TEST(DynamicYield, ThreatenedStopChecksASecondWellSeparatedRetreatWithinTheProofBudget) {
  using namespace terrain_nav;
  const std::vector<int> families{1,1,1,1,-1,-1};
  const std::vector<double> estimates{.01,.035,.038,.041,.002,.004};
  const std::vector<double> utility{10.,9.,8.,7.,-1.,-2.};
  const auto select=[&](bool threatened_stop) {
    int attempts=0;
    return bestCertifiedInteraction(utility,interactionProofOrder(families,estimates,threatened_stop),
      [&](size_t index){++attempts;return index==5;},[&]{return attempts<3;});
  };
  EXPECT_EQ(select(false),std::numeric_limits<size_t>::max());
  EXPECT_EQ(select(true),5u);
}

TEST(DynamicYield, CertifiedProgressCommitsBeforeOptionalRefinementConsumesItsFreshness) {
  using namespace terrain_nav;
  double clock=100.;int proofs=0,readiness=0;
  const double expiry=100.04;
  const auto selected=bestCertifiedInteraction(std::vector<double>{5.,6.},std::vector<size_t>{0,1},
    [&](size_t index){++proofs;clock+=index==0?.005:.10;return true;},
    [&]{return interactionProofFitsFreshness(clock,expiry,.005);},
    [&](size_t){++readiness;return true;});
  EXPECT_EQ(selected,0u);EXPECT_EQ(proofs,1);EXPECT_EQ(readiness,1);
  EXPECT_LT(clock,expiry);
}
TEST(DynamicYield, SeparateInputDeadTimesDoNotDoubleCountTheExplicitActuatorSlew) {
  using namespace terrain_nav;
  YieldModel m;m.delay=.2;m.speed_dead_time=0.;m.steering_dead_time=.08;m.step=.005;
  YieldState root;
  const auto early=rolloutQueuedYield(root,{{0.,1.,.2}},m,.04,100.,{});
  ASSERT_FALSE(early.empty());EXPECT_NEAR(early.back().speed,.048,1e-12);
  EXPECT_DOUBLE_EQ(early.back().steering,0.);EXPECT_DOUBLE_EQ(early.back().yaw,0.);
  const auto later=rolloutQueuedYield(root,{{0.,1.,.2}},m,.12,100.,{});
  ASSERT_FALSE(later.empty());EXPECT_NEAR(later.back().speed,.144,1e-12);
  EXPECT_GT(later.back().steering,.1);EXPECT_GT(later.back().yaw,0.);
  m.speed_dead_time=-.5;EXPECT_TRUE(rolloutQueuedYield(root,{{0,1,.2}},m,1.,100.,{}).empty());
}
TEST(DynamicYield, SeparateDeadTimesReplayEachActualPendingInputAndNeverFuturePublications) {
  using namespace terrain_nav;
  YieldModel m;m.delay=.2;m.speed_dead_time=0.;m.steering_dead_time=.1;m.step=.005;
  YieldState root;root.speed=.6;
  const std::vector<YieldCommand> issued{{99.9,1.,.2},{99.92,0.,-.2},{100.02,50.,1.}};
  const auto path=rolloutQueuedYield(root,{{0.,.4,0.}},m,.08,100.,issued);
  ASSERT_FALSE(path.empty());EXPECT_NEAR(path.back().speed,.504,1e-12);
  EXPECT_LT(path.back().steering,0.);
  EXPECT_GT(path.at(3).steering,0.);
  for(const auto &s:path){EXPECT_LE(s.speed,.6);EXPECT_LE(std::abs(s.steering),.2+1e-10);}
}
TEST(DynamicYield, ExplicitEqualDeadTimesExactlyPreserveLegacyPendingQueueResults) {
  using namespace terrain_nav;
  for(double delay:{0.,.05,.2}) {
    YieldModel inherited;inherited.delay=delay;YieldModel explicit_axes=inherited;
    explicit_axes.speed_dead_time=explicit_axes.steering_dead_time=delay;
    YieldState root;root.speed=.7;root.steering=.1;
    const std::vector<YieldCommand> commands{{0.,1.,-.2},{.31,.4,.2},{.8,0.,0.}},
      issued{{99.,.7,.1},{99.88,1.,.1},{99.96,.5,-.1}};
    const auto a=rolloutQueuedYield(root,commands,inherited,2.,100.,issued),
      b=rolloutQueuedYield(root,commands,explicit_axes,2.,100.,issued);
    ASSERT_EQ(a.size(),b.size());ASSERT_FALSE(a.empty());
    for(size_t i=0;i<a.size();++i) {
      EXPECT_DOUBLE_EQ(a[i].t,b[i].t);EXPECT_DOUBLE_EQ(a[i].x,b[i].x);EXPECT_DOUBLE_EQ(a[i].y,b[i].y);
      EXPECT_DOUBLE_EQ(a[i].yaw,b[i].yaw);EXPECT_DOUBLE_EQ(a[i].speed,b[i].speed);EXPECT_DOUBLE_EQ(a[i].steering,b[i].steering);
    }
  }
}

TEST(DynamicYield, EarlyCommitCannotUseFailedProofAndRetainsBrakeWhileCheckingProgress) {
  using namespace terrain_nav;
  const std::vector<double> utility{-1.,5.,4.,6.};
  const std::vector<int> family{0,1,1,1};
  std::vector<size_t> proved,ready;
  const auto selected=bestCertifiedInteraction(utility,std::vector<size_t>{0,1,2,3},
    [&](size_t i){proved.push_back(i);return i!=1;},[]{return true;},
    [&](size_t i){ready.push_back(i);return family[i]>0;});
  EXPECT_EQ(selected,2u);
  EXPECT_EQ(proved,(std::vector<size_t>{0,1,2}));
  EXPECT_EQ(ready,(std::vector<size_t>{0,2}));
  EXPECT_EQ(bestCertifiedInteraction(utility,std::vector<size_t>{0,1,2,3},
    [](size_t){return false;},[]{return true;},[](size_t){return true;}),std::numeric_limits<size_t>::max());
  EXPECT_EQ(bestCertifiedInteraction(utility,std::vector<size_t>{0,1,2,3},
    [](size_t){return true;},[]{return false;},[](size_t){return true;}),std::numeric_limits<size_t>::max());
}

TEST(DynamicYield, LongerRetreatsAreOptionalFiniteStoppingPoliciesWithinTheSameHorizon) {
  terrain_nav::YieldModel model;model.wheelbase=.430;model.rear_axle_offset=.215;
  model.delay=.20;model.acceleration=.82;model.deceleration=1.2;model.steering_rate=4.;
  terrain_nav::YieldState root;root.speed=1.2;
  const auto proposals=terrain_nav::reverseYieldProposals(root,model,.55,.31432913231137,6.,true);
  bool extended=false,short_retained=false;
  for(const auto &p:proposals) {
    extended|=p.distance>1.2;short_retained|=p.distance==.8;
    const auto states=terrain_nav::rolloutYield(root,root,p.commands,model,6.);
    ASSERT_FALSE(states.empty());EXPECT_DOUBLE_EQ(states.back().speed,0.);
    for(const auto &s:states){EXPECT_GE(s.speed,-.550001);EXPECT_LE(std::abs(s.steering),.314330);}
  }
  EXPECT_TRUE(extended);EXPECT_TRUE(short_retained);
  for(const auto &p:terrain_nav::reverseYieldProposals(root,model,.55,.31432913231137,3.,false))
    EXPECT_LE(p.distance,1.2);
}
#include "terrain_nav/gaussian_motion.hpp"
#include "terrain_nav/motion_risk.hpp"
#include "terrain_nav/joint_motion_risk.hpp"

TEST(JointMotion, AConfidenceRefinementNeverChangesTheRiskLimitOrAcceptsInvalidEvidence) {
  using namespace terrain_nav;
  JointMotionRisk r;r.valid=true;r.samples=8192;r.estimate=.04;
  r.upper=binomialCollisionUpper(8192,328,1e-12);
  ASSERT_GT(r.upper,.05);EXPECT_TRUE(refineMotionConfidence(r,.05));
  // Same estimated proportion, a new larger confidence calculation. This is
  // a statistical-resolution check, not a terrain/navigation permission.
  const auto larger=binomialCollisionUpper(65536,2622,motionCheckAlpha(2));
  EXPECT_LT(larger,.05);EXPECT_LT(larger,r.upper);
  r.samples=65536;EXPECT_FALSE(refineMotionConfidence(r,.05));
  r.samples=8192;r.valid=false;EXPECT_FALSE(refineMotionConfidence(r,.05));
  r.valid=true;r.estimate=.06;EXPECT_FALSE(refineMotionConfidence(r,.05));
  r.estimate=.04;r.upper=.049;EXPECT_FALSE(refineMotionConfidence(r,.05));
  r.upper=NAN;EXPECT_FALSE(refineMotionConfidence(r,.05));
  r.upper=.06;r.estimate=-.1;EXPECT_FALSE(refineMotionConfidence(r,.05));
  r.estimate=.04;EXPECT_FALSE(refineMotionConfidence(r,NAN));
  EXPECT_FALSE(refineMotionConfidence(r,.05,8192));
}
#include <random>

TEST(MotionGeometry, GroupBroadPhaseMatchesUnprunedPerCellSweptBodyChecks) {
  using namespace terrain_nav;
  std::mt19937_64 rng(661128);std::uniform_real_distribution<double> unit(-1.,1.);
  const MotionCheckGeometry g{.7,.3,.4,.3,.43,.05};
  for(int trial=0;trial<160;++trial) {
    MotionTrackSnapshot actor;actor.stamp=99.9;actor.resolution=trial%2?.05:.2;
    actor.vx=.7+.3*unit(rng);actor.vy=.6*unit(rng);
    actor.acceleration_bound=trial%3?.8:0.;actor.velocity_error=.1;actor.prediction_maximum_speed=4.5;
    const double x=5*unit(rng),y=5*unit(rng);
    for(int i=0;i<8;++i)for(int j=0;j<6;++j)
      actor.cells.push_back({x+(i-3.5)*actor.resolution,y+(j-2.5)*actor.resolution});
    std::vector<YieldState> states;
    for(int i=0;i<8;++i)states.push_back({i*.4,3*unit(rng),3*unit(rng),M_PI*unit(rng),unit(rng),0.});
    const double extra_velocity=trial%2?1.:0.;bool expected=false;
    for(const auto &s:states) {
      const double age=100.-actor.stamp,t=100.+s.t-actor.stamp,speed=std::hypot(actor.vx,actor.vy);
      const double cap=predictionSpeedCap(speed,actor.velocity_error,actor.prediction_maximum_speed);
      const double travel=boundedTranslationDistance(speed,t,actor.acceleration_bound,cap);
      const double retreat=std::min(0.,minimumTranslationDistance(speed,t,actor.acceleration_bound,cap));
      const double sweep=.02*(std::min(cap,speed+actor.acceleration_bound*t)+std::abs(s.speed)*
        (1+std::hypot(std::max(g.front,g.back),g.half_width)*std::tan(g.max_steering)/g.wheelbase));
      const double association=motionAssociationRadius(actor.resolution);
      const double reserve=g.extra_inflation+motionAssociationRadius(actor.resolution,age)+
        (extra_velocity>0?std::max(0.,.2-association):0.)+sweep+(extra_velocity+actor.velocity_error)*t;
      // Oracle calls the original exact cell primitive for EVERY cell;
      // it does not call the new group-bounds implementation.
      for(const auto &cell:actor.cells)
        expected|=bodyOverlapsTranslatingCell(s.x,s.y,s.yaw,g.front,g.back,g.half_width,
          cell[0]+actor.vx/speed*retreat,cell[1]+actor.vy/speed*retreat,
          cell[0]+actor.vx/speed*travel,cell[1]+actor.vy/speed*travel,actor.resolution+2*reserve);
    }
    const auto result=assessTimedMotion(states,100.,{actor},g,extra_velocity);
    ASSERT_TRUE(result.valid)<<trial;EXPECT_EQ(result.unsafe,expected)<<trial;
  }
  MotionTrackSnapshot tangent;tangent.stamp=100.;tangent.resolution=.05;tangent.acceleration_bound=0.;
  tangent.velocity_error=0.;tangent.prediction_maximum_speed=4.5;
  tangent.cells={{{g.front+.5*tangent.resolution+g.extra_inflation+motionAssociationRadius(.05,0.),0.}}};
  EXPECT_TRUE(timedMotionUnsafe({YieldState{}},100.,{tangent},g));
}

using namespace terrain_nav;
namespace {
std::vector<TimedCenter> measurements(double x=1.,double y=2.)
{
  std::vector<TimedCenter> h;
  for(int i=0;i<4;++i)h.push_back({10.+.1*i,{x+.1*i,y-.08*i},.45,.05});
  return h;
}
GaussianPosition position(double x,double y,double variance=.01)
{
  GaussianPosition p;p.valid=true;p.mean<<x,y;p.covariance=variance*Eigen::Matrix2d::Identity();return p;
}
}
TEST(GaussianMotion, ContinuousProcessHasAnalyticSemigroup)
{
  const auto f=gaussianMotionTransition(.7);
  EXPECT_TRUE(gaussianMotionProcess(1.,.2).isApprox(
    f*gaussianMotionProcess(.3,.2)*f.transpose()+gaussianMotionProcess(.7,.2),1e-12));
}
TEST(GaussianAccelerationMotion, WhiteJerkHasTheExactMarkovCovarianceSemigroup)
{
  const auto f=gaussianAccelerationTransition(.7);
  const auto q=gaussianJerkProcess(1.,.2);
  EXPECT_TRUE(q.isApprox(f*gaussianJerkProcess(.3,.2)*f.transpose()+gaussianJerkProcess(.7,.2),1e-12));
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,6,6>> e(q);
  EXPECT_GT(e.eigenvalues().minCoeff(),0.);
  EXPECT_NEAR(q(0,0),.2/20.,1e-15);EXPECT_NEAR(q(0,2),.2/8.,1e-15);
  EXPECT_NEAR(q(0,4),.2/6.,1e-15);EXPECT_NEAR(q(2,4),.2/2.,1e-15);
}
TEST(GaussianAccelerationMotion, CausalPosteriorLearnsReversalWithoutDiscardingAccelerationUncertainty)
{
  std::vector<TimedCenter> history;
  const auto observation=[](double t){return TimedCenter{10.+t,{1.+.4*t+.15*t*t,2.+.6*t-.35*t*t},.45,.05};};
  for(int i=0;i<4;++i)history.push_back(observation(.1*i));
  auto ca=fitGaussianAccelerationCenterMotion(history,.02,0.,4);
  ASSERT_TRUE(validGaussianMotion(ca));ASSERT_TRUE(ca.constant_acceleration);
  for(int i=4;i<=20;++i) {
    const auto old=ca;ca=updateGaussianCenterMotion(ca,observation(.1*i),.02);
    ASSERT_TRUE(ca.valid)<<i;EXPECT_EQ(ca.observation_count,old.observation_count+1);
    EXPECT_DOUBLE_EQ(old.stamp,10.+.1*(i-1));
  }
  EXPECT_LT(ca.mean[3],0.);EXPECT_NEAR(ca.acceleration_mean[4],.3,.01);EXPECT_NEAR(ca.acceleration_mean[5],-.7,.01);
  EXPECT_NEAR(ca.mean[2],1.,.01);EXPECT_NEAR(ca.mean[3],-.8,.01);
  const auto future=predictGaussianPosition(ca,.8);ASSERT_TRUE(future.valid);
  EXPECT_NEAR(future.mean[0],observation(2.8).center[0],.02);
  EXPECT_NEAR(future.mean[1],observation(2.8).center[1],.02);
  EXPECT_GT(future.covariance.trace(),(ca.covariance.topLeftCorner<2,2>().trace()));
  EXPECT_FALSE(updateGaussianCenterMotion(ca,observation(2.),.02).valid);
  auto corrupt=ca;corrupt.mean[0]+=.01;EXPECT_FALSE(validGaussianMotion(corrupt));
  corrupt=ca;corrupt.longitudinal_acceleration_bound=1.2;EXPECT_FALSE(validGaussianMotion(corrupt));
}
TEST(GaussianAccelerationMotion, SharedSamplerMatchesAllSixAnalyticWhiteJerkMoments)
{
  joint_detail::Actor a{};a.constant_acceleration=1;a.density=.2;a.acceleration_mean[4]=.3;
  Eigen::Matrix<double,6,1> mean=Eigen::Matrix<double,6,1>::Zero();
  Eigen::Matrix<double,6,6> second=Eigen::Matrix<double,6,6>::Zero();
  const double duration=1.3;const size_t n=100000;
  for(size_t k=0;k<n;++k) {
    double state[6];auto rng=joint_detail::initialState(a,2271,k,0,state);
    joint_detail::propagate(state,duration,a.density,rng,true);
    const Eigen::Map<Eigen::Matrix<double,6,1>> x(state);mean+=x;second+=x*x.transpose();
  }
  mean/=n;second=second/n-mean*mean.transpose();
  Eigen::Matrix<double,6,1> initial=Eigen::Matrix<double,6,1>::Zero();initial[4]=.3;
  EXPECT_LT((mean-gaussianAccelerationTransition(duration)*initial).norm(),.01);
  EXPECT_LT((second-gaussianJerkProcess(duration,a.density)).norm(),.01);
}
TEST(GaussianMotion, CausalFitRecoversMotionAndKeepsSource)
{
  const auto m=fitGaussianCenterMotion(measurements());ASSERT_TRUE(validGaussianMotion(m));
  EXPECT_DOUBLE_EQ(m.stamp,10.3);EXPECT_NEAR(m.mean.x(),1.3,.002);EXPECT_NEAR(m.mean.y(),1.76,.002);
  EXPECT_NEAR(m.mean[2],1.,.005);EXPECT_NEAR(m.mean[3],-.8,.005);
  const auto p=predictGaussianPosition(m,1.);ASSERT_TRUE(p.valid);
  EXPECT_NEAR(p.mean.x(),2.3,.01);EXPECT_NEAR(p.mean.y(),.96,.01);
  EXPECT_GT(p.covariance.trace(),(m.covariance.topLeftCorner<2,2>().trace()));
}
TEST(GaussianMotion, RecursiveIdentityUpdateEqualsSingleConsumptionOfAllObservations)
{
  const auto h=measurements();
  auto prefix=h;prefix.pop_back();
  const auto previous=fitGaussianCenterMotion(prefix);
  const auto recursive=updateGaussianCenterMotion(previous,h.back(),.03);
  const auto batch=fitGaussianCenterMotion(h);
  ASSERT_TRUE(recursive.valid&&batch.valid);
  EXPECT_TRUE(recursive.mean.isApprox(batch.mean,1e-10));
  EXPECT_TRUE(recursive.covariance.isApprox(batch.covariance,1e-10));
  EXPECT_EQ(recursive.observation_count,4u);
  EXPECT_EQ(recursive.oldest_stamp,previous.oldest_stamp);
  EXPECT_EQ(recursive.stamp,h.back().stamp_s);
  EXPECT_EQ(recursive.process_density,previous.process_density);
  EXPECT_LT(recursive.covariance(2,2),fitGaussianCenterMotion(h,.03,.04,3).covariance(2,2));
  // Replayed/old/missing data must never increase confidence or freshness.
  EXPECT_FALSE(updateGaussianCenterMotion(recursive,h.back(),.03).valid);
  auto bad=h.back();bad.stamp_s+=.5;
  EXPECT_FALSE(updateGaussianCenterMotion(recursive,bad,.03).valid);
  bad=h.back();bad.stamp_s+=.1;bad.center[0]+=2.;
  EXPECT_FALSE(updateGaussianCenterMotion(recursive,bad,.03).valid);
  EXPECT_FALSE(updateGaussianCenterMotion(previous,h.back(),0.).valid);
}
TEST(GaussianMotion, ExplicitReassociationPropagatesTheFullGapAndConsumesOnlyOneNewMeasurement)
{
  for(bool acceleration:{false,true}) {
    const auto prior=acceleration?fitGaussianAccelerationCenterMotion(measurements(),.03,.04,4):
      fitGaussianCenterMotion(measurements(),.03,.04,4);
    ASSERT_TRUE(validGaussianMotion(prior));
    const double gap=.8;
    const auto predicted=predictGaussianPosition(prior,gap);ASSERT_TRUE(predicted.valid);
    TimedCenter measured{prior.stamp+gap,{predicted.mean.x(),predicted.mean.y()},.45,.05};
    EXPECT_FALSE(updateGaussianCenterMotion(prior,measured,.03).valid);
    const auto posterior=updateGaussianCenterMotion(prior,measured,.03,13.815510557964274,2.);
    ASSERT_TRUE(validGaussianMotion(posterior));
    EXPECT_EQ(posterior.observation_count,prior.observation_count+1);
    EXPECT_DOUBLE_EQ(posterior.stamp,measured.stamp_s);
    EXPECT_DOUBLE_EQ(prior.stamp,10.3);
    EXPECT_DOUBLE_EQ(posterior.oldest_stamp,prior.oldest_stamp);
    const Eigen::Matrix2d r=.03*.03*Eigen::Matrix2d::Identity();
    // The complete elapsed F/Q covariance is used before conditioning on
    // this actual observation; the gap is not filled with virtual samples.
    const Eigen::Matrix2d expected=predicted.covariance-
      predicted.covariance*(predicted.covariance+r).inverse()*predicted.covariance;
    EXPECT_TRUE((posterior.covariance.topLeftCorner<2,2>().isApprox(expected,1e-9)));
    EXPECT_FALSE(updateGaussianCenterMotion(posterior,measured,.03,13.815510557964274,2.).valid);
    for(double invalid:{0.,-1.,6.1,double(NAN)})
      EXPECT_FALSE(updateGaussianCenterMotion(prior,measured,.03,13.815510557964274,invalid).valid);
    measured.center[0]+=100.;
    EXPECT_FALSE(updateGaussianCenterMotion(prior,measured,.03,13.815510557964274,2.).valid);
  }
}
TEST(GaussianMotion, ReassociatedWindowConsumesOnlyNewCentresAndRejectsReplayedOrMalformedHistory)
{
  for(bool acceleration:{false,true}) {
    const auto history=measurements();
    const auto prior=acceleration?fitGaussianAccelerationCenterMotion(history,.03,.04,4):
      fitGaussianCenterMotion(history,.03,.04,4);
    auto window=history;
    auto sequential=prior;
    for(double gap:{.8,.9,1.}) {
      const auto predicted=predictGaussianPosition(prior,gap);ASSERT_TRUE(predicted.valid);
      window.push_back({prior.stamp+gap,{predicted.mean.x(),predicted.mean.y()},.45,.05});
      sequential=updateGaussianCenterMotion(sequential,window.back(),.03,13.815510557964274,2.);
    }
    ASSERT_TRUE(sequential.valid);
    const auto result=updateGaussianCenterWindow(prior,window,.03,13.815510557964274,2.,3);
    ASSERT_TRUE(result.valid);EXPECT_EQ(result.observation_count,prior.observation_count+3);
    EXPECT_TRUE(result.mean.isApprox(sequential.mean,1e-12));
    EXPECT_TRUE(result.covariance.isApprox(sequential.covariance,1e-12));
    EXPECT_FALSE(updateGaussianCenterWindow(result,window,.03,13.815510557964274,2.,3).valid);
    auto insufficient=window;insufficient.pop_back();
    EXPECT_FALSE(updateGaussianCenterWindow(prior,insufficient,.03,13.815510557964274,2.,3).valid);
    auto bad=window;bad[1].stamp_s=bad[0].stamp_s;
    EXPECT_FALSE(updateGaussianCenterWindow(prior,bad,.03,13.815510557964274,2.,3).valid);
    bad=window;bad.back().resolution=.1;
    EXPECT_FALSE(updateGaussianCenterWindow(prior,bad,.03,13.815510557964274,2.,3).valid);
  }
}
TEST(GaussianAccelerationMotion, RecordedFarIdentityCannotBeDonatedByOneBroadPositionInnovation)
{
  nlohmann::json d;std::ifstream(std::string(GAUSSIAN_FIXTURE_DIR)+"/recorded_dormant_identity_donation.json")>>d;
  GaussianMotion prior;prior.valid=prior.constant_acceleration=true;
  prior.stamp=d["prior_stamp_s"];prior.oldest_stamp=prior.stamp-3.;
  prior.observation_count=d["prior_count"];prior.process_density=d["process_density"];
  prior.observation_std=d["observation_std"];
  for(int k=0;k<6;++k)prior.acceleration_mean[k]=d["prior_mean"][k];
  for(int i=0;i<6;++i)for(int j=0;j<6;++j)prior.acceleration_covariance(i,j)=d["prior_covariance"][i*6+j];
  prior.mean=prior.acceleration_mean.head<4>();prior.covariance=prior.acceleration_covariance.topLeftCorner<4,4>();
  ASSERT_TRUE(validGaussianMotion(prior));
  std::vector<TimedCenter> observed;
  for(const auto &s:d["new_centres"])observed.push_back({s["stamp_s"],{s["x"],s["y"]},.45,.05});
  const auto single=updateGaussianCenterMotion(prior,observed.front(),.03,13.815510557964274,5.);
  ASSERT_TRUE(single.valid);EXPECT_LT(single.mean[2],-3.9);
  // The actual later centres move in +Y with almost no X motion. A broad
  // one-point correction is not enough evidence for this old identity.
  EXPECT_FALSE(updateGaussianCenterWindow(prior,observed,.03,13.815510557964274,5.,3).valid);
}

TEST(GaussianMotion, InvalidHistoryCannotBecomePrediction)
{
  auto h=measurements();h[1].stamp_s=h[0].stamp_s;EXPECT_FALSE(fitGaussianCenterMotion(h).valid);
  h=measurements();h[1].center[0]=NAN;EXPECT_FALSE(fitGaussianCenterMotion(h).valid);
  h=measurements();h[1].resolution=.2;EXPECT_FALSE(fitGaussianCenterMotion(h).valid);
  h=measurements();h[1].radius=.8;EXPECT_FALSE(fitGaussianCenterMotion(h).valid);
  h=measurements();h.back().stamp_s=11.;EXPECT_FALSE(fitGaussianCenterMotion(h).valid);
  h.resize(2);EXPECT_FALSE(fitGaussianCenterMotion(h).valid);
  EXPECT_FALSE(fitGaussianCenterMotion(measurements(),-.03).valid);
  EXPECT_FALSE(fitGaussianCenterMotion(measurements(),.03,-.1).valid);
}
TEST(GaussianMotion, FitEquivariantUnderWorldTranslation)
{
  const auto a=fitGaussianCenterMotion(measurements()),b=fitGaussianCenterMotion(measurements(101.,-98.));
  ASSERT_TRUE(a.valid&&b.valid);EXPECT_NEAR((b.mean-a.mean)[0],100.,1e-10);
  EXPECT_NEAR((b.mean-a.mean)[1],-100.,1e-10);EXPECT_NEAR((b.mean-a.mean).tail<2>().norm(),0.,1e-10);
  EXPECT_TRUE(a.covariance.isApprox(b.covariance,1e-12));
}
TEST(GaussianMotion, InvalidCovarianceFailsClosed)
{
  auto m=fitGaussianCenterMotion(measurements());m.covariance(0,0)=-1.;
  EXPECT_FALSE(validGaussianMotion(m));EXPECT_FALSE(predictGaussianPosition(m,1.).valid);
  m=fitGaussianCenterMotion(measurements());EXPECT_FALSE(predictGaussianPosition(m,-.1).valid);
  EXPECT_FALSE(predictGaussianPosition(m,11.).valid);
  auto p=position(2.,0.);p.covariance(0,0)=-1.;
  EXPECT_EQ(gaussianBodyCollisionUpper(0,0,0,.5,.5,.46,.45,p),1.);
  p=position(2.,0.);p.covariance(0,1)=.01;
  EXPECT_EQ(gaussianBodyCollisionUpper(0,0,0,.5,.5,.46,.45,p),1.);
}
TEST(GaussianMotion, HalfspaceTailMatchesAnalyticAxisBound)
{
  const auto p=position(1.2,0.,.01);
  EXPECT_NEAR(gaussianBodyCollisionUpper(0,0,0,.5,.3,.46,.45,p),.5*std::erfc(.25/std::sqrt(.02)),1e-14);
}
TEST(GaussianMotion, ZeroVarianceUsesFullBodyAndDisk)
{
  EXPECT_EQ(gaussianBodyCollisionUpper(0,0,0,.5,.3,.46,.45,position(.8,0,0)),1.);
  EXPECT_EQ(gaussianBodyCollisionUpper(0,0,0,.5,.3,.46,.45,position(1.,0,0)),0.);
  EXPECT_EQ(gaussianBodyCollisionUpper(0,0,0,.5,.3,.46,.45,position(-.8,0,0)),0.);
  EXPECT_EQ(gaussianBodyCollisionUpper(0,0,0,.5,.3,.46,.45,position(.9,.86,0)),0.);
}
TEST(GaussianMotion, CollisionBoundRotatesWithVehicleAndCovariance)
{
  auto p=position(1.2,.7);p.covariance<<.02,.004,.004,.01;
  const double a=gaussianBodyCollisionUpper(0,0,.1,.5,.3,.46,.45,p),angle=.9;
  Eigen::Matrix2d r;r<<std::cos(angle),-std::sin(angle),std::sin(angle),std::cos(angle);
  p.mean=r*p.mean+Eigen::Vector2d(5,7);p.covariance=(r*p.covariance*r.transpose()).eval();
  EXPECT_NEAR(a,gaussianBodyCollisionUpper(5,7,1.,.5,.3,.46,.45,p),1e-13);
}
TEST(GaussianMotion, SupportingHalfspacesUpperBoundNumericallyIntegratedCollision)
{
  // Deterministic quadrature is an independent geometric check of the
  // one-sided bound, not data fitting, a physical test, or calibration.
  for(const auto &xy:std::vector<std::array<double,2>>{{1.1,0.},{.8,.8},{-.9,.5},{0.,1.1}}) {
    auto p=position(xy[0],xy[1],.04);p.covariance(0,1)=p.covariance(1,0)=.01;
    Eigen::LLT<Eigen::Matrix2d> llt(p.covariance);const Eigen::Matrix2d l=llt.matrixL();
    double mass=0.;const double step=.04;
    for(double i=-7.+step/2;i<7.;i+=step)for(double j=-7.+step/2;j<7.;j+=step) {
      const Eigen::Vector2d q=p.mean+l*Eigen::Vector2d(i,j);
      const double dx=std::max({q.x()-.5,-.3-q.x(),0.}),dy=std::max(std::abs(q.y())-.46,0.);
      if(std::hypot(dx,dy)<=.45)mass+=std::exp(-.5*(i*i+j*j))/(2*std::acos(-1.))*step*step;
    }
    EXPECT_LE(mass,gaussianBodyCollisionUpper(0,0,0,.5,.3,.46,.45,p)+.002);
  }
}
namespace {
MotionTrackSnapshot actor(double x,double y)
{
  MotionTrackSnapshot t;t.circular=t.center_motion_fit=true;t.stamp=10.3;t.resolution=.05;
  t.center_x=x;t.center_y=y;t.radius_bound=.45;t.vx=1.;t.vy=-.8;t.cells={{{x,y}}};
  t.gaussian=fitGaussianCenterMotion(measurements(x-.3,y+.24));return t;
}
}
TEST(GaussianMotion, JointBoundSumsTimeAndObjectsWithoutIndependence)
{
  const auto t=actor(1.4,0.);const MotionCheckGeometry g;
  const std::vector<YieldState> s={{0.,0.,0.,0.,0.,0.},{.02,0.,0.,0.,0.,0.}};
  const auto one=timedGaussianMotionRisk(s,10.3,{t},g),two=timedGaussianMotionRisk(s,10.3,{t,t},g);
  ASSERT_TRUE(one.valid&&two.valid);EXPECT_EQ(one.gaussian_objects,1u);EXPECT_EQ(two.gaussian_objects,2u);
  EXPECT_NEAR(one.upper,std::min(1.,one.pose_upper[0]+one.pose_upper[1]),1e-12);
  EXPECT_NEAR(two.upper,std::min(1.,2*one.upper),1e-12);
  for(size_t i=0;i<s.size();++i)EXPECT_NEAR(two.pose_upper[i],std::min(1.,2*one.pose_upper[i]),1e-12);
}
TEST(GaussianAccelerationMotion, MixedCvCaCpuCudaAndSharedForecastsDescribeTheSameJointEvent)
{
  auto ca=actor(1.7,1.5);ca.vx=0.;ca.vy=-.5;auto &g=ca.gaussian;
  g.constant_acceleration=true;g.acceleration_mean<<1.7,1.5,0.,-.5,0.,-.3;
  g.acceleration_covariance=Eigen::Matrix<double,6,6>::Identity()*.01;
  g.mean=g.acceleration_mean.head<4>();g.covariance=g.acceleration_covariance.topLeftCorner<4,4>();
  ASSERT_TRUE(validGaussianMotion(g));
  const std::vector<MotionTrackSnapshot> actors={actor(9.,8.),ca};
  std::vector<std::vector<YieldState>> paths(2);
  for(int i=0;i<=120;++i)for(int k=0;k<2;++k)paths[k].push_back({(.02+.003*k)*i,.02*i,.2*k,0.,1.,0.});
  const auto cpu=jointGaussianMotionRisk(paths[0],10.3,actors,MotionCheckGeometry{},1024,1e-8,3811,false);
  ASSERT_TRUE(cpu.valid)<<cpu.error;
  const auto analytic=timedGaussianMotionRisk(paths[0],10.3,actors,MotionCheckGeometry{});
  ASSERT_TRUE(analytic.valid);EXPECT_GE(analytic.upper,cpu.estimate-.01);
  const auto screen=screenJointMotionBatch(paths,10.3,actors,MotionCheckGeometry{},256,3811,false);
  ASSERT_TRUE(screen.valid)<<screen.error;
  std::string error;if(!jointMotionGpuAvailable(&error))GTEST_SKIP()<<error;
  const auto gpu=jointGaussianMotionRisk(paths[0],10.3,actors,MotionCheckGeometry{},1024,1e-8,3811,true);
  ASSERT_TRUE(gpu.valid)<<gpu.error;EXPECT_EQ(cpu.collisions,gpu.collisions);EXPECT_DOUBLE_EQ(cpu.upper,gpu.upper);
  const auto batch=screenJointMotionBatch(paths,10.3,actors,MotionCheckGeometry{},256,3811,true);
  ASSERT_TRUE(batch.valid)<<batch.error;EXPECT_EQ(screen.estimates,batch.estimates);EXPECT_EQ(screen.lower_bounds,batch.lower_bounds);
}

TEST(JointMotion, MeanAccelerationCanRejectEntryWhileAnEarlierStoppingPlaceRemainsAdmissible) {
  // A slower CURRENT center velocity must not pin future mean travel to CV
  // when the declared model also permits longitudinal acceleration. This
  // demonstrates a model boundary, not calibration or navigation success.
  auto cv=actor(5.,3.5);cv.vx=0.;cv.vy=-.2;cv.acceleration_bound=1.2;cv.velocity_error=.1;
  cv.gaussian.mean<<5.,3.5,0.,-.2;
  auto bounded=cv;bounded.gaussian.longitudinal_acceleration_bound=1.2;
  YieldState root;root.x=1.8;root.speed=.6;
  YieldModel model;model.wheelbase=.43;model.rear_axle_offset=.215;model.delay=.04;
  const auto commands=makeAdvanceCommands(root,model,2.,.6,0.,6.);
  const auto advance=finiteJointRollout(rolloutYield(root,root,commands,model,6.));
  const auto stop=finiteJointRollout(rolloutYield(root,root,{{0,0,0}},model,6.));
  const MotionCheckGeometry geometry{.521,.521,.46,.31432913231137,.43,0.};
  const auto nominal=jointGaussianMotionRisk(advance,10.3,{cv},geometry,8192,1e-8,112,false);
  const auto stress=jointGaussianMotionRisk(advance,10.3,{bounded},geometry,8192,1e-8,112,false);
  const auto waiting=jointGaussianMotionRisk(stop,10.3,{bounded},geometry,8192,1e-8,113,false);
  ASSERT_TRUE(nominal.valid&&stress.valid&&waiting.valid);
  EXPECT_LT(nominal.upper,.05);EXPECT_GT(stress.upper,.05);EXPECT_LT(waiting.upper,.05);
  std::string error;if(jointMotionGpuAvailable(&error)) {
    const auto gpu=jointGaussianMotionRisk(advance,10.3,{bounded},geometry,8192,1e-8,112,true);
    ASSERT_TRUE(gpu.valid);EXPECT_EQ(gpu.collisions,stress.collisions);EXPECT_DOUBLE_EQ(gpu.upper,stress.upper);
  }
}
TEST(GaussianMotion, EarlierRestEnvelopeContainsEveryRotatedBodyAndRetainsOrderedTimeCoverage)
{
  MotionCheckGeometry g;YieldState initial;initial.speed=1.2;initial.steering=.2;
  YieldModel model;model.wheelbase=g.wheelbase;model.rear_axle_offset=.215;
  const auto states=rolloutYield(initial,initial,{{0,0,.2}},model,3.);
  const auto envelope=earlyStopEnvelope(states,g);
  ASSERT_TRUE(envelope.valid);ASSERT_EQ(envelope.stationary.size(),states.size());
  for(size_t i=0;i<states.size();++i) {
    EXPECT_DOUBLE_EQ(envelope.stationary[i].t,states[i].t);
    EXPECT_DOUBLE_EQ(envelope.stationary[i].speed,0.);
    for(double x:{g.front,-g.back})for(double y:{-g.half_width,g.half_width}) {
      const auto &s=states[i];const double px=s.x+std::cos(s.yaw)*x-std::sin(s.yaw)*y;
      const double py=s.y+std::sin(s.yaw)*x+std::cos(s.yaw)*y;
      const double dx=std::max({-g.back-px,px-g.front,0.});
      const double dy=std::max(std::abs(py)-g.half_width,0.);
      EXPECT_LE(std::hypot(dx,dy),envelope.expansion+1e-12);
    }
  }
  auto invalid=states;invalid[1].t=invalid[0].t;EXPECT_FALSE(earlyStopEnvelope(invalid,g).valid);
  invalid=states;invalid[2].yaw=NAN;EXPECT_FALSE(earlyStopEnvelope(invalid,g).valid);
  EXPECT_FALSE(earlyStopEnvelope({},g).valid);
  YieldState rest;auto stationary=rolloutYield(rest,rest,{{0,0,0}},model,3.);
  EXPECT_DOUBLE_EQ(earlyStopEnvelope(stationary,g).expansion,0.);
}
TEST(GaussianMotion, CompactSpeedCueOwnershipNeedsExactFreshUniqueMeasuredConfirmedIdentity)
{
  auto t=actor(2.,1.);t.id=7;t.motion_confirmed=true;
  t.cells={{{1.8,.8}},{{2.2,.8}},{{1.8,1.2}},{{2.2,1.2}}};
  const std::vector<std::array<double,4>> one{{1.775,.775,2.225,1.225}};
  ASSERT_TRUE(gaussianCircleEnclosesMeasurements(t));
  EXPECT_TRUE(unmodeledCompactBounds(one,{t},10.3).empty());
  EXPECT_DOUBLE_EQ(compactEncounterSpeed(1.33,0,0,0,one,.6),.6);
  EXPECT_DOUBLE_EQ(compactEncounterSpeed(1.33,0,0,0,unmodeledCompactBounds(one,{t},10.3),.6),1.33);
  auto other=one;other.push_back({2.5,-1.2,2.9,-.8});
  const auto remaining=unmodeledCompactBounds(other,{t},10.3);
  ASSERT_EQ(remaining.size(),1u);EXPECT_EQ(remaining[0],other[1]);
  EXPECT_DOUBLE_EQ(compactEncounterSpeed(1.33,0,0,0,remaining,.6),.6);
  for(int variant=0;variant<7;++variant) {
    auto invalid=t;
    if(variant==0)invalid.id=0;
    if(variant==1)invalid.motion_confirmed=false;
    if(variant==2)invalid.stamp-=.1;
    if(variant==3)invalid.gaussian.valid=false;
    if(variant==4)invalid.gaussian.mean[2]=NAN;
    if(variant==5)invalid.gaussian.observation_count=2;
    if(variant==6)invalid.cells.push_back({4.,1.});
    EXPECT_EQ(unmodeledCompactBounds(one,{invalid},10.3),one)<<variant;
  }
  auto duplicate=one;duplicate.push_back(one[0]);
  EXPECT_EQ(unmodeledCompactBounds(duplicate,{t},10.3),duplicate);
  EXPECT_EQ(unmodeledCompactBounds(one,{t,t},10.3),one);
  auto shifted=one;shifted[0][0]-=.05;
  EXPECT_EQ(unmodeledCompactBounds(shifted,{t},10.3),shifted);
  // A confirmed actor is not unclassified again just because it has braked
  // below the NEW-identity speed threshold. Missing/future observations still
  // cannot refresh it, and no newly static component can claim this identity.
  for(double speed:{0.,.1,.349,-.1,-.8}) {
    auto slow=t;slow.gaussian.mean.tail<2>()<<speed,0.;
    EXPECT_TRUE(unmodeledCompactBounds(one,{slow},10.3).empty());
    slow.motion_confirmed=false;
    EXPECT_EQ(unmodeledCompactBounds(one,{slow},10.3),one);
  }
  EXPECT_EQ(t.cells.size(),4u);EXPECT_EQ(one.size(),1u);
}
TEST(GaussianMotion, InvalidOrNonFittedPredictionCannotEraseOccupiedObject)
{
  auto t=actor(0.,0.);const std::vector<YieldState> states={{0,0,0,0,0,0}};
  t.gaussian.valid=false;
  const auto blocked=timedGaussianMotionRisk(states,10.3,{t},MotionCheckGeometry{});
  EXPECT_FALSE(blocked.valid);EXPECT_EQ(blocked.upper,1.);EXPECT_EQ(blocked.reachable_objects,1u);
  t=actor(0,0);t.gaussian.stamp=9.;EXPECT_FALSE(timedGaussianMotionRisk(states,10.3,{t},MotionCheckGeometry{}).valid);
  t=actor(0,0);EXPECT_FALSE(timedGaussianMotionRisk(states,11.,{t},MotionCheckGeometry{}).valid);
}
TEST(JointMotion, InvalidSourceIsDifferentFromAValidDeterministicFallbackConflict)
{
  const std::vector<YieldState> states={{0,0,0,0,0,0},{.02,0,0,0,0,0}};
  auto t=actor(0.,0.);t.gaussian.valid=false;
  auto known=assessTimedMotion(states,10.3,{t},MotionCheckGeometry{});
  EXPECT_TRUE(known.valid&&known.unsafe);
  auto positive=jointGaussianMotionRisk(states,10.3,{t},MotionCheckGeometry{},256,1e-6,3,false);
  EXPECT_FALSE(positive.valid);EXPECT_TRUE(positive.deterministic_conflict);EXPECT_EQ(positive.upper,1.);
  t.stamp=9.;
  auto missing=jointGaussianMotionRisk(states,10.3,{t},MotionCheckGeometry{},256,1e-6,3,false);
  EXPECT_FALSE(missing.valid);EXPECT_FALSE(missing.deterministic_conflict);EXPECT_EQ(missing.upper,1.);
  EXPECT_FALSE(assessTimedMotion(states,10.3,{t},MotionCheckGeometry{}).valid);
  t=actor(0.,0.);t.gaussian.stamp-=.1;
  missing=jointGaussianMotionRisk(states,10.3,{t},MotionCheckGeometry{},256,1e-6,3,false);
  EXPECT_FALSE(missing.valid);EXPECT_FALSE(missing.deterministic_conflict);
  EXPECT_TRUE(timedMotionUnsafe(states,NAN,{t},MotionCheckGeometry{}));
}
TEST(GaussianMotion, FarObjectDoesNotBlockAndClearancePreferenceCannotDecreaseRisk)
{
  auto t=actor(4.,3.);std::vector<YieldState> states;
  // The trajectory must carry its actual temporal coverage. A 1s gap now
  // correctly requires a much larger swept reserve than a 20ms rollout.
  for(int i=0;i<=100;++i)states.push_back({.02*i,.02*i,0,0,i==100?0.:1.,0});
  const auto a=timedGaussianMotionRisk(states,10.3,{t},MotionCheckGeometry{}),
    b=timedGaussianMotionRisk(states,10.3,{t},MotionCheckGeometry{},.2);
  ASSERT_TRUE(a.valid&&b.valid);EXPECT_LT(a.upper,.02);EXPECT_GE(b.upper,a.upper);
  EXPECT_EQ(timedGaussianMotionRisk(states,10.3,{},MotionCheckGeometry{}).upper,0.);
}
TEST(JointMotion, BinomialUpperIsOneSidedAndFailsClosedOnInvalidCounts)
{
  EXPECT_NEAR(binomialCollisionUpper(100,0,.05),1-std::pow(.05,.01),1e-13);
  EXPECT_NEAR(binomialCollisionUpper(100,10,.05),.1637176233,1e-9);
  EXPECT_EQ(binomialCollisionUpper(0,0,.05),1.);EXPECT_EQ(binomialCollisionUpper(10,11,.05),1.);
  EXPECT_EQ(binomialCollisionUpper(10,0,0),1.);EXPECT_EQ(binomialCollisionUpper(10,10,.05),1.);
  EXPECT_GT(binomialCollisionUpper(8192,0,1e-20),binomialCollisionUpper(8192,0,1e-6));
}
TEST(JointMotion, LifetimeSamplingFailureSpendsTelescopingBudget)
{
  double sum=0.;for(uint64_t n=1;n<=10000;++n)sum+=motionCheckAlpha(n);
  EXPECT_NEAR(sum,1e-4*(1-1./10001),1e-16);EXPECT_LT(sum,1e-4);EXPECT_EQ(motionCheckAlpha(0),0.);
}
TEST(JointMotion, RepeatedSamePositionDoesNotMultiplyTimeCollisionProbability)
{
  auto t=actor(1.2,0.);t.gaussian.mean<<1.2,0.,0.,0.;t.gaussian.covariance.setZero();
  t.gaussian.covariance(0,0)=t.gaussian.covariance(1,1)=.04;t.gaussian.process_density=0.;
  const std::vector<YieldState> one={{0,0,0,0,0,0},{.02,0,0,0,0,0}};auto many=one;
  for(int i=2;i<=100;++i)many.push_back({.02*i,0,0,0,0,0});
  const auto a=jointGaussianMotionRisk(one,10.3,{t},MotionCheckGeometry{},1024,1e-6,452,false),
    b=jointGaussianMotionRisk(many,10.3,{t},MotionCheckGeometry{},1024,1e-6,452,false);
  ASSERT_TRUE(a.valid&&b.valid);EXPECT_EQ(a.collisions,b.collisions);EXPECT_EQ(a.upper,b.upper);
  EXPECT_GT(a.estimate,.01);EXPECT_LT(a.estimate,.5);
}
TEST(JointMotion, ObjectUnionDoesNotAssumeIndependentSurvival)
{
  const auto t=actor(4.,3.);const std::vector<YieldState> s={{0,0,0,0,0,0},{.02,0,0,0,0,0}};
  const auto p=jointGaussianMotionRisk(s,10.3,{t,t},MotionCheckGeometry{},1024,1e-6,457,false);
  ASSERT_TRUE(p.valid);ASSERT_EQ(p.collisions.size(),2u);
  const auto analytic=timedGaussianMotionRisk(s,10.3,{t},MotionCheckGeometry{});
  ASSERT_TRUE(analytic.valid);
  EXPECT_NEAR(p.upper,std::min(1.,std::min(analytic.upper,binomialCollisionUpper(1024,p.collisions[0],.5e-6))+
    std::min(analytic.upper,binomialCollisionUpper(1024,p.collisions[1],.5e-6))),1e-13);
}
TEST(JointMotion, RemoteActorsHaveNoSamplingFloorAndDoNotDiluteNearActorRisk)
{
  auto near=actor(.7,0.);near.gaussian.mean<<.7,0.,0.,0.;
  near.gaussian.covariance=Eigen::Matrix4d::Identity()*.002;
  const std::vector<YieldState> states={{0,0,0,0,0,0},{.02,0,0,0,0,0}};
  const MotionCheckGeometry g;
  const auto one=jointGaussianMotionRisk(states,10.3,{near},g,512,1e-9,91,false);
  std::vector<MotionTrackSnapshot> many{near};
  for(int i=0;i<5;++i)many.push_back(actor(8.+i,5.));
  const auto six=jointGaussianMotionRisk(states,10.3,many,g,512,1e-9,91,false);
  ASSERT_TRUE(one.valid&&six.valid);ASSERT_EQ(six.collisions.size(),6u);
  EXPECT_EQ(one.collisions[0],six.collisions[0]);EXPECT_GT(six.upper,.95);
  const auto remote=jointGaussianMotionRisk(states,10.3,{many.back()},g,512,1e-9,91,false);
  ASSERT_TRUE(remote.valid);EXPECT_LT(remote.upper,1e-20);
  EXPECT_EQ(remote.samples,512u);EXPECT_EQ(remote.gaussian_objects,1u);
}

TEST(JointMotion, AnalyticalDischargePreservesTheFullUpperBoundAndEveryRemainingSampleStream)
{
  using namespace terrain_nav;
  auto near=actor(1.12,.1);near.gaussian.mean<<1.12,.1,0.,0.;
  near.gaussian.covariance.setZero();near.gaussian.covariance(0,0)=near.gaussian.covariance(1,1)=.04;
  near.gaussian.process_density=0.;
  const std::vector<YieldState> states{{0,0,0,0,0,0},{.02,0,0,0,0,0}};
  // Put a remote body BEFORE the near body: compacting actor slots would
  // change its PRNG stream and make this equality fail.
  std::vector<MotionTrackSnapshot> tracks{actor(8,5),near,actor(-8,-5)};
  for(bool gpu:{false,true}) {
    std::string error;if(gpu&&!jointMotionGpuAvailable(&error))continue;
    for(size_t samples:{512u,8192u}) {
      const auto full=jointGaussianMotionRisk(states,10.3,tracks,MotionCheckGeometry{},samples,1e-10,89,gpu,0.,false);
      const auto lean=jointGaussianMotionRisk(states,10.3,tracks,MotionCheckGeometry{},samples,1e-10,89,gpu,0.,true);
      ASSERT_TRUE(full.valid&&lean.valid)<<lean.error;
      EXPECT_DOUBLE_EQ(lean.upper,full.upper);EXPECT_EQ(lean.collisions[1],full.collisions[1]);
      ASSERT_EQ(lean.sample_counts.size(),3u);EXPECT_EQ(lean.sample_counts[0],0u);
      EXPECT_EQ(lean.sample_counts[1],samples);EXPECT_EQ(lean.sample_counts[2],0u);
      EXPECT_EQ(lean.analytical_objects,2u);
    }
    // The NEW swept-mean enclosing box may discharge a truly remote bounded
    // acceleration event. Its upper bound must equal the fully sampled path.
    auto bounded=tracks;bounded[0].gaussian.longitudinal_acceleration_bound=1.2;
    auto result=jointGaussianMotionRisk(states,10.3,bounded,MotionCheckGeometry{},512,1e-8,89,gpu,0.,true);
    const auto original=jointGaussianMotionRisk(states,10.3,bounded,MotionCheckGeometry{},512,1e-8,89,gpu,0.,false);
    ASSERT_TRUE(result.valid&&original.valid)<<result.error;
    EXPECT_EQ(result.sample_counts[0],0u);EXPECT_DOUBLE_EQ(result.upper,original.upper);
    auto bad=tracks;bad[0].stamp=9.;
    EXPECT_FALSE(jointGaussianMotionRisk(states,10.3,bad,MotionCheckGeometry{},512,1e-8,89,gpu,0.,true).valid);
  }
}
TEST(JointMotion, AnalyticRefinementPreservesRoundedBodySweepAndPreferenceClearance)
{
  auto t=actor(1.35,0.);t.gaussian.mean<<1.35,0.,0.,0.;
  t.gaussian.covariance.setZero();t.gaussian.covariance(0,0)=t.gaussian.covariance(1,1)=.01;
  t.gaussian.process_density=0.;
  const std::vector<YieldState> states={{0,0,0,0,1.,0},{.02,.02,0,0,1.,0}};
  const auto a=jointGaussianMotionRisk(states,10.3,{t},MotionCheckGeometry{},1024,1e-8,531,false);
  const auto b=jointGaussianMotionRisk(states,10.3,{t},MotionCheckGeometry{},1024,1e-8,531,false,.3);
  ASSERT_TRUE(a.valid&&b.valid);EXPECT_GT(b.upper,a.upper);
  EXPECT_NEAR(a.upper,std::min(binomialCollisionUpper(1024,a.collisions[0],1e-8),
    timedGaussianMotionRisk(states,10.3,{t},MotionCheckGeometry{}).upper),1e-13);
}
TEST(JointMotion, InvalidOrStaleEvidenceAndReversedTimeDoNotAuthorizeMotion)
{
  auto t=actor(4,3);const std::vector<YieldState> s={{0,0,0,0,0,0},{.1,0,0,0,0,0}};
  EXPECT_FALSE(jointGaussianMotionRisk(s,11.,{t},MotionCheckGeometry{},512,1e-6,1,false).valid);
  auto reversed=s;std::reverse(reversed.begin(),reversed.end());
  EXPECT_FALSE(jointGaussianMotionRisk(reversed,10.3,{t},MotionCheckGeometry{},512,1e-6,1,false).valid);
  t.gaussian.valid=false;t.center_x=t.center_y=0;t.cells={{{0,0}}};
  EXPECT_FALSE(jointGaussianMotionRisk(s,10.3,{t},MotionCheckGeometry{},512,1e-6,1,false).valid);
}
TEST(JointMotion, CudaUsesSameSamplesAndWholeBodyAsCpu)
{
  std::string error;if(!jointMotionGpuAvailable(&error))GTEST_SKIP()<<error;
  std::vector<YieldState> states;
  for(int i=0;i<=100;++i)states.push_back({.02*i,.015*i,.1*std::sin(.02*i),.003*i,.75,0});
  for(double x:{1.0,1.8,3.}) {
    auto t=actor(x,1.3);auto other=actor(3.,-1.);
    const auto cpu=jointGaussianMotionRisk(states,10.4,{t,other},MotionCheckGeometry{},512,1e-6,895,true);
    const auto reference=jointGaussianMotionRisk(states,10.4,{t,other},MotionCheckGeometry{},512,1e-6,895,false);
    ASSERT_TRUE(cpu.valid)<<cpu.error;ASSERT_TRUE(reference.valid)<<reference.error;
    EXPECT_EQ(cpu.collisions,reference.collisions);EXPECT_DOUBLE_EQ(cpu.upper,reference.upper);
  }
}
TEST(JointMotion, BatchedScreenMatchesIndividualSamplesAndCpuCuda)
{
  std::vector<std::vector<YieldState>> candidates;
  for(int c=0;c<8;++c) {
    std::vector<YieldState> states;
    for(int i=0;i<=100;++i)states.push_back({.02*i,.015*i,.15*c+.1*std::sin(.02*i),.003*i,.75,0});
    candidates.push_back(std::move(states));
  }
  const std::vector<MotionTrackSnapshot> tracks={actor(1.8,1.3),actor(3.,-1.)};
  const uint64_t seed=567;
  const auto reference=screenJointMotionBatch(candidates,10.4,tracks,MotionCheckGeometry{},256,seed,false);
  ASSERT_TRUE(reference.valid)<<reference.error;
  ASSERT_EQ(reference.lower_bounds.size(),candidates.size());
  EXPECT_DOUBLE_EQ(reference.discard_confidence_error,.001);
  for(size_t c=0;c<candidates.size();++c)EXPECT_LE(reference.lower_bounds[c],reference.estimates[c]);
  for(size_t c=0;c<candidates.size();++c) {
    const auto single=jointGaussianMotionRisk(candidates[c],10.4,tracks,MotionCheckGeometry{},256,.05,
      seed,false);
    ASSERT_TRUE(single.valid);EXPECT_EQ(single.estimate,reference.estimates[c]);
  }
  auto bad=candidates;bad[1][0].t=.1;
  EXPECT_FALSE(screenJointMotionBatch(bad,10.4,tracks,MotionCheckGeometry{},256,seed,false).valid);
  std::string error;if(!jointMotionGpuAvailable(&error))GTEST_SKIP()<<error;
  const auto gpu=screenJointMotionBatch(candidates,10.4,tracks,MotionCheckGeometry{},256,seed,true);
  ASSERT_TRUE(gpu.valid)<<gpu.error;EXPECT_EQ(gpu.estimates,reference.estimates);
  EXPECT_EQ(gpu.lower_bounds,reference.lower_bounds);
}
TEST(JointMotion, ScreeningCommonForecastsAreOrderInvariantAndCpuCudaAgreeOnMixedTimes)
{
  std::vector<std::vector<YieldState>> candidates(4);
  for(int c=0;c<4;++c)for(int i=0;i<=60;++i)
    candidates[c].push_back({(.02+.003*c)*i,.02*i,.2*c,.003*i,1.,0.});
  candidates.push_back(candidates.front());
  auto tracks=std::vector<MotionTrackSnapshot>{actor(1.8,-.6),actor(3.,-1.)};
  tracks.front().vx=0.;tracks.front().vy=.2;
  tracks.front().gaussian.mean<<1.8,-.6,0.,.2;
  tracks.front().gaussian.covariance=Eigen::Matrix4d::Identity()*.04;
  tracks.back().gaussian.longitudinal_acceleration_bound=1.2;
  const auto cpu=screenJointMotionBatch(candidates,10.4,tracks,MotionCheckGeometry{},256,901,false);
  ASSERT_TRUE(cpu.valid)<<cpu.error;EXPECT_EQ(cpu.estimates.front(),cpu.estimates.back());
  EXPECT_TRUE(std::any_of(cpu.estimates.begin(),cpu.estimates.end(),[](double p){return p>0&&p<1;}));
  std::reverse(candidates.begin(),candidates.end());
  const auto reversed=screenJointMotionBatch(candidates,10.4,tracks,MotionCheckGeometry{},256,901,false);
  ASSERT_TRUE(reversed.valid);auto expected=cpu.estimates;std::reverse(expected.begin(),expected.end());
  EXPECT_EQ(reversed.estimates,expected);
  auto lower=cpu.lower_bounds;std::reverse(lower.begin(),lower.end());EXPECT_EQ(reversed.lower_bounds,lower);
  std::string error;if(!jointMotionGpuAvailable(&error))GTEST_SKIP()<<error;
  const auto gpu=screenJointMotionBatch(candidates,10.4,tracks,MotionCheckGeometry{},256,901,true);
  ASSERT_TRUE(gpu.valid)<<gpu.error;EXPECT_EQ(gpu.estimates,expected);
  EXPECT_EQ(gpu.lower_bounds,lower);
}
TEST(JointMotion, OversizedScreenFailsClosedBeforeAllocatingItsForecast)
{
  std::vector<YieldState> states;
  for(int i=0;i<200;++i)states.push_back({.02*i,.01*i,0.,0.,.5,0.});
  const auto result=screenJointMotionBatch({states,states},10.4,{actor(1.8,1.3)},
      MotionCheckGeometry{},65536,901,false);
  EXPECT_FALSE(result.valid);EXPECT_EQ(result.error,"screen candidate work budget exceeded");
}
TEST(JointMotion, IrregularSchedulesKeepActualTimesWithoutRequiringAHugeSharedForecast)
{
  std::vector<std::vector<YieldState>> candidates(8);
  for(int c=0;c<8;++c)for(int i=0;i<=600;++i)
    candidates[c].push_back({i?i*.007+c*1e-6:0.,.004*i,.08*c,0.,.55,0.});
  const std::vector<MotionTrackSnapshot> tracks{actor(1.4,.3),actor(3.,-.5)};
  const uint64_t seed=71651;const MotionCheckGeometry g;
  const auto cpu=screenJointMotionBatch(candidates,10.4,tracks,g,32,seed,false);
  ASSERT_TRUE(cpu.valid)<<cpu.error;EXPECT_TRUE(cpu.independent_candidate_forecasts);
  EXPECT_DOUBLE_EQ(cpu.discard_confidence_error,.001);
  for(size_t c:{0u,7u}) {
    const auto reference=jointGaussianMotionRisk(candidates[c],10.4,tracks,g,32,1e-6,
      seed^(0x9e3779b97f4a7c15ULL*c),false);
    ASSERT_TRUE(reference.valid);EXPECT_DOUBLE_EQ(cpu.estimates[c],reference.estimate);
  }
  std::string error;if(!jointMotionGpuAvailable(&error))GTEST_SKIP()<<error;
  const auto gpu=screenJointMotionBatch(candidates,10.4,tracks,g,32,seed,true);
  ASSERT_TRUE(gpu.valid)<<gpu.error;EXPECT_TRUE(gpu.independent_candidate_forecasts);
  EXPECT_EQ(gpu.estimates,cpu.estimates);EXPECT_EQ(gpu.lower_bounds,cpu.lower_bounds);
}
TEST(JointMotion, SweptDiskKernelMatchesIndependentFullBodyGeometry)
{
  std::mt19937 gen(983);std::uniform_real_distribution<double> d(-3,3);
  for(int i=0;i<400;++i) {
    const double x=d(gen),y=d(gen),yaw=d(gen),ax=d(gen),ay=d(gen),bx=d(gen),by=d(gen);
    joint_detail::Pose p{0,x,y,std::cos(yaw),std::sin(yaw),.521,.491,.46,0};
    EXPECT_EQ(joint_detail::collisionSegment(ax,ay,bx,by,p,.45),
      bodyOverlapsTranslatingDisk(x,y,yaw,.521,.491,.46,ax,ay,bx,by,.45));
  }
}
TEST(JointMotion, AccelerationEnvelopeCannotBeReplacedByGaussianProcessNoise)
{
  auto t=actor(0,3);t.gaussian.mean<<0.,3.,0.,-.5;t.gaussian.covariance.setZero();
  t.gaussian.process_density=0.;t.vx=0.;t.vy=-.5;
  std::vector<YieldState> states;for(int i=0;i<=150;++i)states.push_back({.02*i,0,0,0,0,0});
  const auto cv=jointGaussianMotionRisk(states,10.3,{t},MotionCheckGeometry{},256,1e-6,378,false);
  ASSERT_TRUE(cv.valid);EXPECT_EQ(cv.collisions[0],0u);
  t.acceleration_bound=t.gaussian.longitudinal_acceleration_bound=1.2;
  const auto accelerating=jointGaussianMotionRisk(states,10.3,{t},MotionCheckGeometry{},256,1e-6,378,false);
  ASSERT_TRUE(accelerating.valid);EXPECT_EQ(accelerating.collisions[0],256u);
  std::string error;if(!jointMotionGpuAvailable(&error))GTEST_SKIP()<<error;
  const auto gpu=jointGaussianMotionRisk(states,10.3,{t},MotionCheckGeometry{},256,1e-6,378,true);
  ASSERT_TRUE(gpu.valid)<<gpu.error;EXPECT_EQ(gpu.collisions,accelerating.collisions);
}
TEST(JointMotion, SignedMeanEnvelopeIncludesCappedReversalAndUnknownRestDeparture)
{
  joint_detail::Actor a{};a.acceleration=1.2;a.maximum_speed=4.5;a.age=.17;
  for(double speed:{.0,.2,.8,4.5})for(double angle:{-.8,0.,1.7})for(double future:{.0,.5,1.,3.,6.}) {
    a.mean[2]=speed*std::cos(angle);a.mean[3]=speed*std::sin(angle);
    const double t=a.age+future,state[4]={a.mean[2]*t,a.mean[3]*t,a.mean[2],a.mean[3]};
    const auto envelope=joint_detail::sampledSegment(a,state,future);
    const double radius=joint_detail::meanDepartureRadius(a,future);
    if(speed>0.) {
      const double minimum=minimumTranslationDistance(speed,t,a.acceleration,a.maximum_speed);
      EXPECT_NEAR(envelope.ax,std::cos(angle)*minimum,1e-12);
      EXPECT_NEAR(envelope.ay,std::sin(angle)*minimum,1e-12);
    }
    for(int profile=0;profile<5;++profile) {
      double v=speed,distance=0.;const int steps=1200;const double dt=t/steps;
      for(int k=0;k<steps;++k) {
        const double control=profile==0?-1.:profile==1?1.:profile==2?0.:((k<steps/2)==(profile==3)?-1.:1.);
        const double next=std::clamp(v+control*a.acceleration*dt,-a.maximum_speed,a.maximum_speed);
        distance+=(v+next)*dt/2.;v=next;
      }
      const double x=std::cos(angle)*distance,y=std::sin(angle)*distance;
      EXPECT_LE(joint_detail::pointSegmentSquared(x,y,envelope.ax,envelope.ay,envelope.bx,envelope.by),
                radius*radius+1e-4)<<speed<<" "<<future<<" "<<profile;
    }
  }
}
TEST(JointMotion, ReversingOrDepartingActorCannotReceiveAFalseClearCertificate)
{
  for(bool stopped:{false,true}) {
    // Source is moving AWAY, or is stationary; bounded acceleration can
    // return/depart towards the ego within the SAME three-second horizon.
    auto t=actor(0.,-2.);t.vx=0.;t.vy=stopped?0.:-.5;
    t.gaussian.mean<<0.,-2.,0.,t.vy;t.gaussian.covariance.setZero();t.gaussian.process_density=0.;
    t.acceleration_bound=t.gaussian.longitudinal_acceleration_bound=1.2;
    std::vector<YieldState> states;for(int i=0;i<=150;++i)states.push_back({.02*i,0,0,0,0,0});
    EXPECT_TRUE(timedMotionUnsafe(states,10.3,{t},MotionCheckGeometry{}));
    const auto cpu=jointGaussianMotionRisk(states,10.3,{t},MotionCheckGeometry{},256,1e-6,572,false);
    ASSERT_TRUE(cpu.valid)<<cpu.error;EXPECT_EQ(cpu.collisions[0],256u);EXPECT_GT(cpu.upper,.05);
    const auto analytic=timedGaussianMotionRisk(states,10.3,{t},MotionCheckGeometry{});
    ASSERT_TRUE(analytic.valid);EXPECT_DOUBLE_EQ(analytic.upper,1.);
    const auto discharged=jointGaussianMotionRisk(states,10.3,{t},MotionCheckGeometry{},256,1e-6,572,false,0.,true);
    ASSERT_TRUE(discharged.valid);EXPECT_EQ(discharged.sample_counts[0],256u);
    const auto screen=screenJointMotionBatch({states},10.3,{t},MotionCheckGeometry{},256,572,false);
    ASSERT_TRUE(screen.valid);EXPECT_DOUBLE_EQ(screen.estimates[0],1.);EXPECT_GT(screen.lower_bounds[0],.05);
    std::string error;if(jointMotionGpuAvailable(&error)) {
      const auto gpu=jointGaussianMotionRisk(states,10.3,{t},MotionCheckGeometry{},256,1e-6,572,true);
      ASSERT_TRUE(gpu.valid)<<gpu.error;EXPECT_EQ(gpu.collisions,cpu.collisions);
      const auto batch=screenJointMotionBatch({states},10.3,{t},MotionCheckGeometry{},256,572,true);
      ASSERT_TRUE(batch.valid);EXPECT_EQ(batch.estimates,screen.estimates);EXPECT_EQ(batch.lower_bounds,screen.lower_bounds);
    }
  }
}
TEST(JointMotion, SweptMeanAnalyticBoxContainsEveryDeterministicSegmentContact)
{
  const MotionCheckGeometry g;
  for(double yaw:{0.,.7,1.6})for(double angle:{-.5,.6,2.})for(double alpha:{0.,.4,1.}) {
    const YieldState state{0.,2.,-1.,yaw,0.,0.};
    const Eigen::Vector2d first(-1.2*std::cos(angle),-1.2*std::sin(angle));
    const Eigen::Vector2d last(2.3*std::cos(angle),2.3*std::sin(angle));
    const Eigen::Vector2d shift=(1-alpha)*first+alpha*last;
    for(double x:{-g.back,g.front})for(double y:{-g.half_width,g.half_width}) {
      GaussianPosition point;point.valid=true;
      point.mean<<state.x+std::cos(yaw)*x-std::sin(yaw)*y-shift.x(),
                  state.y+std::sin(yaw)*x+std::cos(yaw)*y-shift.y();
      EXPECT_DOUBLE_EQ(boundedMeanBodyCollisionUpper(state,g,0.,point,first,last),1.);
      point.mean.x()+=100.;
      EXPECT_DOUBLE_EQ(boundedMeanBodyCollisionUpper(state,g,0.,point,first,last),0.);
    }
  }
}
TEST(GaussianMotion, LocalWindowTracksMotionBirthWithoutArtificialVarianceReduction)
{
  std::vector<TimedCenter> h={{10.,{1.,3.},.45,.05},{10.1,{1.,3.},.45,.05},
    {10.2,{1.,2.92},.45,.05},{10.3,{1.,2.83},.45,.05}};
  const auto local=fitGaussianCenterMotion(h,.03,.04,3),all=fitGaussianCenterMotion(h,.03,.04,4);
  ASSERT_TRUE(local.valid&&all.valid);
  EXPECT_EQ(local.observation_count,3u);EXPECT_EQ(all.observation_count,4u);
  EXPECT_EQ(local.oldest_stamp,10.1);EXPECT_EQ(local.stamp,all.stamp);
  const auto a=predictGaussianPosition(local,.1),b=predictGaussianPosition(all,.1);
  EXPECT_LT(std::abs(a.mean.y()-2.73),std::abs(b.mean.y()-2.73));
  EXPECT_GT(a.covariance(1,1),b.covariance(1,1)); // Less data -> larger uncertainty.
}
TEST(JointMotion, MeasuredEnclosureAndPosteriorDoNotAcquireAnotherRegistrationDisk)
{
  const MotionCheckGeometry g;const double x=g.front+.45+.045+.05;
  auto t=actor(x,0.);t.gaussian.mean<<x,0.,0.,0.;t.gaussian.covariance.setZero();t.gaussian.process_density=0.;
  t.cells={{{x+.4,0.}}};ASSERT_TRUE(gaussianCircleEnclosesMeasurements(t));
  const std::vector<YieldState> states={{0,0,0,0,0,0},{.02,0,0,0,0,0}};
  const auto risk=jointGaussianMotionRisk(states,10.3,{t},g,1024,1e-6,893,false);
  ASSERT_TRUE(risk.valid)<<risk.error;EXPECT_EQ(risk.collisions[0],0u);EXPECT_LT(risk.upper,.02);
  t.cells={{{x+.6,0.}}};EXPECT_FALSE(gaussianCircleEnclosesMeasurements(t));
  EXPECT_FALSE(jointGaussianMotionRisk(states,10.3,{t},g,1024,1e-6,893,false).valid);
}
TEST(JointMotion, TemporalReserveUsesActualMaximumHalfInterval)
{
  EXPECT_DOUBLE_EQ(motionSampleHalfGap({{0,0,0,0,0,0},{.02,0,0,0,0,0}}),.01);
  EXPECT_DOUBLE_EQ(motionSampleHalfGap({{0,0,0,0,0,0},{.02,0,0,0,0,0},{.06,0,0,0,0,0}}),(.06-.02)/2);
  EXPECT_DOUBLE_EQ(motionSampleHalfGap({{0,0,0,0,0,0},{0,0,0,0,0,0}}),0.);
  EXPECT_TRUE(std::isnan(motionSampleHalfGap({{.1,0,0,0,0,0},{0,0,0,0,0,0}})));
}

TEST(JointMotion, ForwardLibraryIncludesFiniteGentleEscapeFromRecordedRearApproach)
{
  // Relative geometry of the first 71.599998399s refusal in the full-mask
  // replay of modular_smoke_20261110_193524. This is a conditional kinematic
  // regression, NOT terrain admission or a claim of physical navigation.
  YieldState initial{0.,0.,0.,-.12277973057625997,.855096132912273,-.24520705413525679};
  YieldModel model;model.wheelbase=.430;model.rear_axle_offset=.215;model.delay=.20;
  const MotionCheckGeometry g;
  auto t=actor(-2.267656714291719,-.213562685263086);
  std::vector<TimedCenter> history;
  for(int i=0;i<3;++i)history.push_back({10.1+.1*i,
    {t.center_x+.8377703558083819*(.1*i-.2),t.center_y-.0018426937962247323*(.1*i-.2)},.45,.05});
  t.gaussian=fitGaussianCenterMotion(history,.018,.04,3);ASSERT_TRUE(t.gaussian.valid);t.stamp=t.gaussian.stamp;
  t.radius_bound=.5052708660050846;t.vx=.8377703558083819;t.vy=-.0018426937962247323;
  bool found=false;
  for(const auto &p:forwardYieldProposals(initial,model,1.33,g.max_steering,6.,true)) {
    if(std::abs(p.distance-5.4)>1e-8||p.commands.size()!=2||
       std::abs(p.commands.front().steering+.5*g.max_steering)>1e-8)continue;
    const auto states=finiteJointRollout(rolloutYield(initial,initial,p.commands,model,6.));
    ASSERT_FALSE(states.empty());EXPECT_DOUBLE_EQ(states.back().speed,0.);
    EXPECT_LE(states.back().t,6.);EXPECT_GE(states.back().t,3.);
    for(const auto &s:states) {EXPECT_LE(std::abs(s.steering),g.max_steering+1e-10);EXPECT_LE(s.speed,1.33+1e-10);}
    const auto r=jointGaussianMotionRisk(states,10.585,{t},g,8192,1e-6,8342,false);
    ASSERT_TRUE(r.valid)<<r.error;EXPECT_LT(r.upper,.05);found=true;
  }
  EXPECT_TRUE(found);
  // An arbitrary stop is not interchangeable with the finite moving escape.
  const auto stopped=rolloutYield(initial,initial,{{0.,0.,0.}},model,3.);
  const auto r=jointGaussianMotionRisk(stopped,10.585,{t},g,8192,1e-6,8343,false);
  ASSERT_TRUE(r.valid)<<r.error;EXPECT_GT(r.upper,.05);
  EXPECT_TRUE(forwardYieldProposals(initial,model,1.33,NAN,6.,true).empty());
}

TEST(JointMotion, RiskHeadroomChangesPreferenceButDoesNotTurnAnEstimateIntoPermission)
{
  EXPECT_DOUBLE_EQ(interactionRiskPriority(3.,.04,.05,0.),3.);
  EXPECT_GT(interactionRiskPriority(2.8,.01,.05,1.),interactionRiskPriority(3.,.04,.05,1.));
  EXPECT_LT(interactionRiskPriority(3.,.8,.05,1.),0.); // Still requires the separate full proof.
  EXPECT_EQ(interactionRiskPriority(3.,NAN,.05,1.),-INFINITY);
  EXPECT_EQ(interactionRiskPriority(3.,.1,0.,1.),-INFINITY);
  EXPECT_EQ(interactionRiskPriority(3.,.1,.05,-1.),-INFINITY);
}

TEST(JointMotion, LongLateralEscapeCanRecoverHeadingInsideRoadInsteadOfContinuingAnArcIntoWall)
{
  YieldState initial;initial.speed=1.2;
  YieldModel model;model.wheelbase=.430;model.rear_axle_offset=.215;model.delay=.20;
  const MotionCheckGeometry body;
  const auto proposals=forwardYieldProposals(initial,model,1.33,body.max_steering,6.,true);
  ASSERT_LT(proposals.size()+10,512u); // Brake/reverse also fit the batch contract.
  bool found=false;
  for(const auto &p:proposals) {
    if(p.distance!=6. || p.commands.size()<3 || p.commands[1].t!=1.5 ||
       p.commands.front().steering!=-.5*body.max_steering)continue;
    const auto states=finiteJointRollout(rolloutYield(initial,initial,p.commands,model,6.));
    ASSERT_FALSE(states.empty());EXPECT_DOUBLE_EQ(states.back().speed,0.);
    EXPECT_GE(states.back().t,3.);EXPECT_LE(states.back().t,6.);
    EXPECT_LT(std::abs(states.back().yaw),.1);
    EXPECT_GT(states.back().x,4.);EXPECT_LT(states.back().y,-1.2);
    for(const auto &s:states) {
      const double side=std::abs(std::sin(s.yaw))*std::max(body.front,body.back)+
        std::abs(std::cos(s.yaw))*body.half_width;
      EXPECT_GT(s.y-side,-3.35);EXPECT_LT(s.y+side,3.35);
      EXPECT_LE(std::abs(s.steering),body.max_steering+1e-10);
    }
    found=true;
  }
  EXPECT_TRUE(found);
  // Holding that same gentle curvature cannot achieve this forward escape
  // inside the same walls. Compare swept full-body geometry, not centre only.
  const auto constant=makeAdvanceCommands(initial,model,6.,1.33,-.5*body.max_steering,6.);
  const auto arc=rolloutYield(initial,initial,constant,model,6.);
  ASSERT_FALSE(arc.empty());bool crosses_wall=false;
  for(const auto &s:arc) {
    const double side=std::abs(std::sin(s.yaw))*std::max(body.front,body.back)+
      std::abs(std::cos(s.yaw))*body.half_width;
    crosses_wall|=s.y-side<=-3.35;
  }
  EXPECT_TRUE(crosses_wall);
}


TEST(DynamicYield, RecordedCrossingNeedsFasterProposalsButRetainsFullDynamicProbabilityProof) {
 using J=nlohmann::json;J d;std::ifstream(std::string(GAUSSIAN_FIXTURE_DIR)+"/constrained_crossing_queue.json")>>d;
 const auto&r=d["root"];
 terrain_nav::YieldState root{r[0],r[1],r[2],r[3],r[4],r[5]};
 terrain_nav::YieldModel m;m.delay=.2;m.wheelbase=.43;m.rear_axle_offset=.215;
 std::vector<terrain_nav::YieldCommand> issued;for(const auto&c:d["issued"])issued.push_back({c[0],c[1],c[2]});
 std::vector<terrain_nav::MotionTrackSnapshot> tracks;
 for(const auto&a:d["actors"]){terrain_nav::MotionTrackSnapshot t;
  t.id=a["id"];t.motion_confirmed=t.center_motion_fit=t.circular=true;t.stamp=a["stamp"];t.resolution=a["resolution"];
  t.radius_bound=a["radius"];t.center_x=a["mean"][0];t.center_y=a["mean"][1];t.vx=a["vx"];t.vy=a["vy"];
  t.acceleration_bound=1.2;t.prediction_maximum_speed=4.5;t.velocity_error=.1;
  for(const auto&c:a["cells"])t.cells.push_back({c[0],c[1]});
  t.gaussian.valid=true;t.gaussian.stamp=t.stamp;t.gaussian.observation_count=3;t.gaussian.observation_std=a["std"];
  t.gaussian.process_density=.04;for(int i=0;i<4;++i){t.gaussian.mean[i]=a["mean"][i];for(int j=0;j<4;++j)t.gaussian.covariance(i,j)=a["cov"][4*i+j];}
  tracks.push_back(t);
 }
 
 terrain_nav::MotionCheckGeometry g{.521,.521,.46,.31432913231137,.43,0.};
 double low_best=1.,fast_best=1.;bool seen=false;
 for(double speed:{.6,1.33}) {
  std::vector<std::vector<terrain_nav::YieldState>> states;
  for(auto p:terrain_nav::forwardYieldProposals(root,m,speed,g.max_steering,6.,true)) {
   for(const auto&c:p.commands)EXPECT_LE(std::abs(c.speed),1.33);
   auto path=terrain_nav::finiteJointRollout(terrain_nav::rolloutQueuedYield(root,p.commands,m,6.,d["origin"],issued));
   if(path.size()>=11&&std::abs(path[path.size()-11].speed)<.001)states.push_back(std::move(path));
  }
  const auto screen=terrain_nav::screenJointMotionBatch(states,d["origin"],tracks,g,256,1729,false);
  ASSERT_TRUE(screen.valid);ASSERT_FALSE(screen.estimates.empty());
  const double best=*std::min_element(screen.estimates.begin(),screen.estimates.end());
  if(speed==.6) {low_best=best;continue;}
  fast_best=best;
  for(size_t i=0;i<states.size();++i)if(screen.estimates[i]<=.05) {
   const auto risk=terrain_nav::jointGaussianMotionRisk(states[i],d["origin"],tracks,g,65536,d["alpha"],1730+i,false,0.,true);
   ASSERT_TRUE(risk.valid);if(risk.upper<=.05){seen=true;break;}
  }
 }
 EXPECT_GT(low_best,.15);EXPECT_LT(fast_best,low_best);EXPECT_TRUE(seen);
 // Motion-only qualification: no assertion of terrain support or execution.
}

TEST(JointMotionRisk, LowerScreenBoundKeepsSamplingNoiseSeparateFromFinalPermission) {
  using namespace terrain_nav;
  const double alpha=.001/(512.*6.);
  // 13/256 = 5.078125%: an estimate alone wrongly discarded the policy.
  EXPECT_GT(13./256.,.05);EXPECT_LT(binomialCollisionLower(256,13,alpha),.05);
  EXPECT_GT(binomialCollisionUpper(8192,819,1e-10),.05); // Retaining is not final permission.
  EXPECT_DOUBLE_EQ(binomialCollisionLower(100,0,.05),0.);
  EXPECT_NEAR(binomialCollisionLower(100,100,.05),std::pow(.05,.01),1e-13);
  EXPECT_TRUE(std::isnan(binomialCollisionLower(0,0,.05)));
  EXPECT_TRUE(std::isnan(binomialCollisionLower(100,101,.05)));
  EXPECT_TRUE(std::isnan(binomialCollisionLower(100,1,0.)));
  double previous=0.;
  for(size_t k=0;k<=100;++k) {
    const double lower=binomialCollisionLower(100,k,.01);
    EXPECT_GE(lower,previous);EXPECT_LE(lower,double(k)/100.+1e-14);previous=lower;
  }
  // Exact finite-binomial coverage, not a simulated favourable draw.
  for(double p:{.001,.05,.2,.5,.9,.999})for(double delta:{.001,.05,.2}) {
    double false_discard=0.;constexpr size_t n=16;
    for(size_t k=0;k<=n;++k)if(binomialCollisionLower(n,k,delta)>p)
      false_discard+=std::exp(std::lgamma(n+1.)-std::lgamma(k+1.)-std::lgamma(n-k+1.)+
        k*std::log(p)+(n-k)*std::log1p(-p));
    EXPECT_LE(false_discard,delta+1e-12);
  }
}

TEST(JointMotion, ProgressiveRejectionSplitsOneRecallBudgetAndResamplesTheAdaptiveSubset) {
  auto t=actor(0.,0.);t.gaussian.mean.setZero();t.gaussian.process_density=0.;
  t.gaussian.covariance=Eigen::Matrix4d::Identity()*.002;
  auto fallback=t;fallback.gaussian.valid=false;fallback.circular=false;
  fallback.center_x=fallback.center_y=20.;fallback.cells={{{20.,20.}}};fallback.vx=fallback.vy=0.;
  std::vector<std::vector<YieldState>> candidates;
  for(const auto &xy:std::vector<std::array<double,2>>{{0.,0.},{3.,3.},{.85,0.},{20.,20.}})
    candidates.push_back({{0.,xy[0],xy[1],0.,0.,0.},{.02,xy[0],xy[1],0.,0.,0.}});
  constexpr uint64_t seed=9721,domain=0x6a09e667f3bcc909ULL;
  const MotionCheckGeometry geometry;
  const auto coarse=screenJointMotionBatch(candidates,10.3,{t,fallback},geometry,32,seed^domain,false,5e-4);
  ASSERT_TRUE(coarse.valid)<<coarse.error;
  std::vector<size_t> indices;std::vector<std::vector<YieldState>> kept;
  for(size_t i=0;i<candidates.size();++i)if(coarse.lower_bounds[i]<=.05) {
    indices.push_back(i);kept.push_back(candidates[i]);
  }
  ASSERT_FALSE(kept.empty());ASSERT_LT(kept.size(),candidates.size());
  const auto fine=screenJointMotionBatch(kept,10.3,{t,fallback},geometry,256,seed,false,5e-4);
  ASSERT_TRUE(fine.valid)<<fine.error;
  const auto cpu=screenJointMotionProgressiveBatch(candidates,10.3,{t,fallback},geometry,256,seed,false,.05);
  ASSERT_TRUE(cpu.valid)<<cpu.error;EXPECT_EQ(cpu.coarse_samples,32u);
  EXPECT_EQ(cpu.refined_candidates,kept.size());EXPECT_EQ(cpu.coarse_discarded,candidates.size()-kept.size());
  EXPECT_DOUBLE_EQ(cpu.discard_confidence_error,.001);
  for(size_t i=0;i<indices.size();++i) {
    EXPECT_DOUBLE_EQ(cpu.estimates[indices[i]],fine.estimates[i]);
    EXPECT_DOUBLE_EQ(cpu.lower_bounds[indices[i]],fine.lower_bounds[i]);
  }
  EXPECT_GT(cpu.lower_bounds[0],.05);EXPECT_DOUBLE_EQ(cpu.lower_bounds[3],1.);
  EXPECT_LT(cpu.lower_bounds[1],.05); // Survives screening; still has NO execution permission.
  std::string error;
  if(jointMotionGpuAvailable(&error)) {
    const auto gpu=screenJointMotionProgressiveBatch(candidates,10.3,{t,fallback},geometry,256,seed,true,.05);
    ASSERT_TRUE(gpu.valid)<<gpu.error;
    EXPECT_EQ(gpu.estimates,cpu.estimates);EXPECT_EQ(gpu.lower_bounds,cpu.lower_bounds);
    EXPECT_EQ(gpu.coarse_discarded,cpu.coarse_discarded);EXPECT_EQ(gpu.refined_candidates,cpu.refined_candidates);
  }
  const auto all_rejected=screenJointMotionProgressiveBatch({candidates[0]},10.3,{t},geometry,256,seed,false,.05);
  ASSERT_TRUE(all_rejected.valid);EXPECT_EQ(all_rejected.coarse_discarded,1u);
  EXPECT_EQ(all_rejected.refined_candidates,0u);EXPECT_DOUBLE_EQ(all_rejected.discard_confidence_error,.0005);
  auto bad=candidates;bad[0][1].x=NAN;
  EXPECT_FALSE(screenJointMotionProgressiveBatch(bad,10.3,{t},geometry,256,seed,false,.05).valid);
  EXPECT_FALSE(screenJointMotionProgressiveBatch(candidates,11.,{t},geometry,256,seed,false,.05).valid);
  EXPECT_FALSE(screenJointMotionProgressiveBatch(candidates,10.3,{t},geometry,16,seed,false,.05).valid);
  EXPECT_FALSE(screenJointMotionProgressiveBatch(candidates,10.3,{t},geometry,256,seed,false,NAN).valid);
  EXPECT_FALSE(screenJointMotionBatch(candidates,10.3,{t},geometry,256,seed,false,0.).valid);
  const auto exact=screenJointMotionProgressiveBatch(candidates,10.3,{fallback},geometry,256,seed,false,.05);
  ASSERT_TRUE(exact.valid);EXPECT_DOUBLE_EQ(exact.discard_confidence_error,0.);
  EXPECT_DOUBLE_EQ(exact.lower_bounds[3],1.);
}
