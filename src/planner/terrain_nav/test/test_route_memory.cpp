#include "terrain_nav/route_memory.hpp"
#include "terrain_nav/planning_budget.hpp"
#include <gtest/gtest.h>
#include <future>

using terrain_nav::RouteMemory;

TEST(RouteMemory, ObservationProposalKeepsCurrentWallsAndOriginalHistoricalEvidence) {
  RouteMemory history(.1),current(.1);
  history.observe(1.,{-4,-4,5,4},{},{{0.,.4}});
  current.observe(2.,{-4,-4,5,4},{},{});
  auto proposal=history.currentGeometryGuidance(current);
  terrain_nav::RouteOptions options;options.advisory_cost=12.;
  ASSERT_TRUE(proposal.route({0,0},{3,0},0.,.5,.77,.68,.58,{},true,false,1,200.,options).reached);
  EXPECT_TRUE(proposal.occupiedAt({0.,.4})); // retained as prior, not erased
  EXPECT_STREQ(history.route({0,0},{3,0},0.,.5,.77,.68,.58,{},true,false,1,200.,options).reason,"blocked start body");
  current.observe(3.,{-4,-4,5,4},{},{{0.,.4}});
  const auto with_current_wall=history.currentGeometryGuidance(current);
  EXPECT_STREQ(with_current_wall.route({0,0},{3,0},0.,.5,.77,.68,.58,{},true,false,1,200.,options).reason,"blocked start body");
  EXPECT_THROW(history.currentGeometryGuidance(RouteMemory(.2)),std::invalid_argument);
}

TEST(RouteMemory, RepeatedOverlappingWitnessesDoNotFillTheWholeHashCell) {
  RouteMemory memory(.1);
  // Nonidentical overlapping edges defeat containment/rectangular coalescing.
  // All witnesses remain above y=.54, outside the protected body at y=0.
  for(int i=0;i<100;++i) {
    const double t=i*.0003;
    memory.observe(1.+i*.1,{-4,-4,5,4},{},{},{},{},false,{{.025+t,.548+t,.005,.005}});
  }
  const auto stats=memory.geometryStats();
  EXPECT_EQ(stats.whole,0u);EXPECT_GT(stats.rectangles,0u);
  const auto clear=memory.route({0,0},{3,0},0.,.5,.77,.68,.58,{},true,false,1,200.);
  ASSERT_TRUE(clear.reached)<<clear.reason;
  // Compression must retain the original AND newest measured geometry.
  for(double y:{.05,.08})
    EXPECT_STREQ(memory.route({0,y},{3,y},0.,.5,.77,.68,.58,{},true,false,1,200.).reason,"blocked start body");
  // Absence of returns does not erase these hard witnesses.
  memory.observe(30.,{-4,-4,5,4},{{.05,.55}},{ });
  EXPECT_STREQ(memory.route({0,.05},{3,.05},0.,.5,.77,.68,.58,{},true,false,1,200.).reason,"blocked start body");
}

TEST(RouteMemory, FragmentCompressionPreservesBothArmsAndAFreeCorner) {
  RouteMemory memory(.1);
  for(int i=0;i<100;++i) {
    const double t=i*.0001;
    memory.observe(1.+i*.1,{-4,-4,5,4},{},{},{},{},false,
      {{.075+t,.575+t,.005,.005},{.025+t,.585+t,.005,.005}});
  }
  EXPECT_EQ(memory.geometryStats().whole,0u);
  ASSERT_GT(memory.geometryStats().raster,0u);
  EXPECT_TRUE(memory.route({0,0},{3,0},0.,.5,.77,.68,.58,{},true,false,1,200.).reached);
  EXPECT_STREQ(memory.route({0,.08},{3,.08},0.,.5,.77,.68,.58,{},true,false,1,200.).reason,"blocked start body");
  const auto rectangles=memory.geometryStats().rectangles;
  for(int i=0;i<1000;++i)
    memory.observe(11.+i*.001,{-4,-4,5,4},{},{},{},{},false,{{.0849,.5849,.005,.005},{.0349,.5949,.005,.005}});
  EXPECT_EQ(memory.geometryStats().rectangles,rectangles); // no cumulative microcell growth
  EXPECT_EQ(memory.geometryStats().whole,0u);
  auto copy=memory;
  copy.observe(15.,{-4,-4,5,4},{},{},{},{},false,{},{},{{.05,.55,.05}});
  EXPECT_EQ(copy.geometryStats().hard,0u);
  EXPECT_GT(memory.geometryStats().hard,0u); // worker copy has independent integer masks
  memory.observe(20.,{-4,-4,5,4},{},{{.05,.55}}); // legacy hard evidence stays full
  EXPECT_GT(memory.geometryStats().whole,0u);
}

TEST(RouteMemory, AdvisoryHistoryCanBeFiniteGuidanceButNeverClearsActualEvidence) {
  RouteMemory memory(.1);
  memory.observe(1.,{-4,-4,5,4},{},{},{},{{0.,.4}});
  EXPECT_STREQ(memory.route({0,0},{3,0},0.,.5,.77,.68,.58,{},true,false,1,200.).reason,"blocked start body");
  terrain_nav::RouteOptions options;options.advisory_cost=2.;
  const auto soft=memory.route({0,0},{3,0},0.,.5,.77,.68,.58,{},true,false,1,200.,options);
  ASSERT_TRUE(soft.reached)<<soft.reason;
  EXPECT_TRUE(memory.occupiedAt({0.,.4})); // provenance remains; no geometry is erased
  EXPECT_STREQ(memory.route({0,0},{3,0},0.,.5,.77,.68,.58,{},true,false,1,200.).reason,"blocked start body");
  options.advisory_cost=4.;
  const auto changed=memory.route({0,0},{3,0},0.,.5,.77,.68,.58,{},true,false,1,200.,options);
  EXPECT_TRUE(changed.reached);EXPECT_FALSE(changed.heuristic_reused);
  memory.observe(2.,{-4,-4,5,4},{},{{0.,.4}});
  EXPECT_STREQ(memory.route({0,0},{3,0},0.,.5,.77,.68,.58,{},true,false,1,200.,options).reason,"blocked start body");
}

TEST(RouteMemory, InvalidAdvisoryCostDoesNotGrantAnOptimisticRoute) {
  RouteMemory memory(.1);memory.observe(1.,{-4,-4,5,4},{},{});
  for(double value:{-1.,31.,double(NAN),double(INFINITY)}) {
    terrain_nav::RouteOptions options;options.advisory_cost=value;
    EXPECT_FALSE(memory.route({0,0},{3,0},0.,.5,.77,.68,.58,{},true,false,1,200.,options).reached);
  }
}

TEST(RouteMemory, CommonGuidanceScoreRetainsUnknownAndHistoricalPreferences) {
  RouteMemory memory(.1);memory.observe(1.,{-4,-4,5,4},{{0,0}},{},{},{{1,0}});
  EXPECT_DOUBLE_EQ(memory.guidanceCost({0,0},1.,2.),1.);
  EXPECT_DOUBLE_EQ(memory.guidanceCost({1,0},1.,2.),3.);
  EXPECT_DOUBLE_EQ(memory.guidanceCost({2,0},1.,2.),3.);
  EXPECT_DOUBLE_EQ(memory.guidanceCost({2,0},13.,2.),13.);
  EXPECT_FALSE(std::isfinite(memory.guidanceCost({0,0},NAN,2.)));
}
using terrain_nav::PlanningBudget;

TEST(RouteMemory, AnisotropicWitnessDoesNotGrowIntoABoundingSquare) {
  RouteMemory square(.1),rectangle(.1);
  square.observe(1.,{-4,-4,5,4},{},{},{},{},false,{{0.,.52,.03}});
  rectangle.observe(1.,{-4,-4,5,4},{},{},{},{},false,{{0.,.52,.03,.005}});
  EXPECT_STREQ(square.route({0,0},{3,0},0,.5,.77,.68,.58,{},true,false,1,100.).reason,"blocked start body");
  EXPECT_TRUE(rectangle.route({0,0},{3,0},0,.5,.77,.68,.58,{},true,false,1,100.).reached);
  EXPECT_STREQ(rectangle.route({0,.03},{3,.03},0,.5,.77,.68,.58,{},true,false,1,100.).reason,"blocked start body");
  EXPECT_THROW(rectangle.observe(2.,{-4,-4,5,4},{},{},{},{},false,{{0.,.52,.03,0.}}),std::invalid_argument);
}

TEST(RouteMemory, PreciseSquaresKeepTheirExtentInsteadOfExpandingToTheHashCell) {
  RouteMemory coarse(.1),precise(.1);
  coarse.observe(1.,{-4,-4,5,4},{},{{0.,.56}});
  precise.observe(1.,{-4,-4,5,4},{},{},{},{},false,{{0.,.56,.025}});
  EXPECT_STREQ(coarse.route({0,0},{3,0},0,.5,.77,.68,.58,{},false,false,0,200.).reason,"blocked start body");
  auto route=precise.route({0,0},{3,0},0,.5,.77,.68,.58,{},false,false,0,200.);
  ASSERT_TRUE(route.reached)<<route.reason;
  EXPECT_TRUE(precise.occupiedAt({-.01,.56}));EXPECT_TRUE(precise.occupiedAt({.01,.56}));
  // The source square crosses the spatial hash boundary: neither half is lost.
  for(double x:{-.02,.02})
    EXPECT_STREQ(precise.route({x,0},{3,0},M_PI/4,.5,.77,.68,.58,{},false,false,0,200.).reason,"blocked start body");
  precise.observe(2.,{-4,-4,5,4},{{0,.56}},{});
  EXPECT_TRUE(precise.occupiedAt({0.,.56})); // floor/occlusion cannot clear the old witness
  EXPECT_TRUE(precise.route({0,0},{3,0},0,.5,.77,.68,.58,{},false,false,0,200.).reached);
  precise.observe(3.,{-4,-4,5,4},{},{},{},{},false,{{0.,.49,.025}});
  EXPECT_STREQ(precise.route({0,0},{3,0},0,.5,.77,.68,.58,{},false,false,0,200.).reason,"blocked start body");
  EXPECT_THROW(precise.observe(4.,{-4,-4,5,4},{},{},{},{},false,{{0.,0.,-.1}}),std::invalid_argument);
}

TEST(RouteMemory, FineEvidenceCannotNarrowLegacyOrExistingOccupiedGeometry) {
  RouteMemory memory(.1);memory.observe(1.,{-4,-4,5,4},{},{{0.,.56}});
  memory.observe(2.,{-4,-4,5,4},{},{},{},{},false,{{0.,.56,.005}});
  EXPECT_STREQ(memory.route({0,0},{3,0},0,.5,.77,.68,.58,{},false,false,0,200.).reason,"blocked start body");
  memory.clear();
  memory.observe(1.,{-4,-4,5,4},{},{},{},{},false,{{0.,.49,.04}});
  memory.observe(2.,{-4,-4,5,4},{},{},{},{},false,{{0.,.56,.005}});
  EXPECT_STREQ(memory.route({0,0},{3,0},0,.5,.77,.68,.58,{},false,false,0,200.).reason,"blocked start body");
}

TEST(RouteMemory, LShapedFineWitnessDoesNotBecomeItsOccupiedBoundingHull) {
  // Frozen real source frame: the entire protected body is outside all three
  // squares, but a per-hash-cell bounding hull closes its free diagonal corner.
  RouteMemory memory(.1);
  memory.observe(1.,{0,-8,10,2},{},{},{},{},false,{
    {4.375000288709998,-3.325000660493970,.025},
    {4.325000287964940,-3.275000659748912,.025},
    {4.375000288709998,-3.275000659748912,.025}});
  const RouteMemory::Point start{4.189961284692159,-3.862728317969502};
  const double yaw=-.5949835848253485;
  auto route=memory.route(start,{start[0]+2*std::cos(yaw),start[1]+2*std::sin(yaw)},yaw,
    .5,.77,.68,.58,{},false,false,0,200.);
  ASSERT_TRUE(route.reached)<<route.reason;
  // The same source still collides after a real 10cm lateral approach.
  auto shifted=start;shifted[1]+=.1;
  EXPECT_STREQ(memory.route(shifted,{7,-5},yaw,.5,.77,.68,.58,{},false,false,0,200.).reason,"blocked start body");
}

TEST(RouteMemory, FineClearingCannotEraseTheUnseenHalfAndCurrentOccupancyWins) {
  RouteMemory memory(.1);
  memory.observe(1.,{-4,-4,5,4},{},{},{},{},false,{{.025,.525,.025},{.075,.575,.025}});
  memory.observe(2.,{-4,-4,5,4},{},{},{},{},false,{},{},{{.075,.575,.025}});
  EXPECT_TRUE(memory.occupiedAt({.025,.525})); // same hash cell, other actual witness survives
  EXPECT_STREQ(memory.route({0,0},{3,0},0.,.5,.77,.68,.58,{},false,false,0,200.).reason,"blocked start body");
  memory.observe(3.,{-4,-4,5,4},{},{},{},{},false,{},{},{{.025,.525,.025}});
  EXPECT_FALSE(memory.occupiedAt({.025,.525}));
  memory.observe(4.,{-4,-4,5,4},{},{},{},{},false,{{.025,.525,.025}},{},{{.025,.525,.025}});
  EXPECT_TRUE(memory.occupiedAt({.025,.525}));
  EXPECT_THROW(memory.observe(5.,{-4,-4,5,4},{},{},{},{},false,{},{},{{0.,0.,NAN}}),std::invalid_argument);
}

TEST(RouteMemory, TerrainCostHeuristicInvalidatesCacheWithoutRelaxingActualGeometry) {
  RouteMemory memory(.1);terrain_nav::RouteOptions options;
  options.heuristic_cost=[](double,double){return 5.;};
  const auto cost=[](double,double,double){return 5.;};
  auto first=memory.route({0,0},{4,0},0.,.5,.77,.68,.58,cost,false,false,0,200.,options);
  ASSERT_TRUE(first.reached);EXPECT_FALSE(first.heuristic_reused);
  auto second=memory.route({0,0},{4,0},0.,.5,.77,.68,.58,cost,false,false,0,200.,options);
  ASSERT_TRUE(second.reached);EXPECT_TRUE(second.heuristic_reused);
  options.heuristic_cost=[](double,double){return 6.;};
  auto changed=memory.route({0,0},{4,0},0.,.5,.77,.68,.58,cost,false,false,0,200.,options);
  EXPECT_TRUE(changed.reached);EXPECT_FALSE(changed.heuristic_reused);
  memory.observe(1.,{-2,-2,6,2},{},{{.35,.05}});
  EXPECT_STREQ(memory.route({0,0},{4,0},0.,.5,.77,.68,.58,cost,false,false,0,200.,options).reason,"blocked start body");
  memory.clear();options.heuristic_cost=[](double,double){return NAN;};
  auto invalid=memory.route({0,0},{4,0},0.,.5,.77,.68,.58,cost,false,false,0,200.,options);
  EXPECT_FALSE(invalid.reached);EXPECT_STREQ(invalid.reason,"invalid routing heuristic cost");
}

TEST(RouteMemory, FinalGoalConnectionIsACarArcInsteadOfAnInstantHeadingJump) {
  RouteMemory memory(.1);const double limit=.98*std::tan(.4)/.549;
  for(bool fine:{false,true})for(RouteMemory::Point goal:std::vector<RouteMemory::Point>{{1.1,.07},{3.2,.36},{-.9,.12}}) {
    auto route=memory.route({0,0},goal,0,.5,limit,.68,.58,{},fine,true,0,200.);
    ASSERT_TRUE(route.reached)<<route.reason;
    EXPECT_NEAR(route.path.back()[0],goal[0],1e-7);EXPECT_NEAR(route.path.back()[1],goal[1],1e-7);
    for(size_t i=1;i<route.path.size();++i) {
      const double ds=std::hypot(route.path[i][0]-route.path[i-1][0],route.path[i][1]-route.path[i-1][1]);
      const double turn=std::atan2(std::sin(route.yaws[i]-route.yaws[i-1]),std::cos(route.yaws[i]-route.yaws[i-1]));
      ASSERT_GT(ds,1e-9);
      EXPECT_LE(std::abs(turn)/ds,limit*1.001);
      const double heading=route.yaws[i-1]+turn/2;
      const double tangent=std::atan2(route.path[i][1]-route.path[i-1][1],route.path[i][0]-route.path[i-1][0]);
      EXPECT_NEAR(std::sin(tangent-heading),0.,1e-7);
    }
  }
}

TEST(RouteMemory, UsefulPartialRouteNeverAppendsAnUnreachedGoalOrInventsATerrainPassage) {
  RouteMemory memory(.1);terrain_nav::RouteOptions options;options.allow_partial=true;
  options.heuristic_weight=2.5;
  const auto corridor=[](double x,double y,double) {
    return x>2.||std::abs(y)>.1?std::numeric_limits<double>::infinity():1.;
  };
  auto route=memory.route({0,0},{6,0},0.,.5,.77,.68,.58,corridor,false,false,0,100.,options);
  ASSERT_FALSE(route.reached);ASSERT_TRUE(route.partial);ASSERT_GT(route.path.size(),2u);
  EXPECT_GT(route.path.back()[0],.7);EXPECT_LE(route.path.back()[0],2.);
  EXPECT_EQ(route.path.size(),route.yaws.size());EXPECT_EQ(route.path.size(),route.directions.size());
  for(const auto &p:route.path)EXPECT_TRUE(std::isfinite(corridor(p[0],p[1],0.)));
  options.allow_partial=false;
  route=memory.route({0,0},{6,0},0.,.5,.77,.68,.58,corridor,false,false,0,100.,options);
  EXPECT_FALSE(route.reached);EXPECT_FALSE(route.partial);EXPECT_TRUE(route.path.empty());
  options.allow_partial=true;
  const auto too_short=[](double x,double,double){return x>.3?std::numeric_limits<double>::infinity():1.;};
  route=memory.route({0,0},{6,0},0.,.5,.77,.68,.58,too_short,false,false,0,100.,options);
  EXPECT_FALSE(route.partial);EXPECT_TRUE(route.path.empty());
  route=memory.route({0,0},{6,0},0.,.5,.77,.68,.58,{},false,false,0,0.,options);
  EXPECT_FALSE(route.partial);EXPECT_TRUE(route.path.empty());
}

TEST(RouteMemory, RecoverySearchEnforcesItsInitialReverseLengthAndSingleGearChange) {
  RouteMemory memory(.1);terrain_nav::RouteOptions options;options.heuristic_weight=4.;
  options.single_reverse_then_forward=true;options.max_initial_reverse=.4;
  auto route=memory.route({0,0},{3,0},0.,.5,.77,.68,.58,{},true,true,-1,100.,options);
  ASSERT_TRUE(route.reached)<<route.reason;ASSERT_GT(route.path.size(),2u);
  ASSERT_EQ(route.directions.front(),-1);int changes=0;double reverse_length=0.;
  for(size_t i=1;i<route.path.size();++i) {
    if(route.directions[i]!=route.directions[i-1])++changes;
    if(route.directions[i]<0)reverse_length+=std::hypot(route.path[i][0]-route.path[i-1][0],route.path[i][1]-route.path[i-1][1]);
  }
  EXPECT_EQ(changes,1);EXPECT_LE(reverse_length,.4+1e-6);EXPECT_GT(reverse_length,.3);
  options.max_initial_reverse=.3;
  route=memory.route({0,0},{3,0},0.,.5,.77,.68,.58,{},true,true,-1,100.,options);
  EXPECT_FALSE(route.reached);EXPECT_TRUE(route.path.empty());
  options.max_initial_reverse=NAN;
  EXPECT_FALSE(memory.route({0,0},{3,0},0.,.5,.77,.68,.58,{},true,true,-1,100.,options).reached);
}

TEST(RouteMemory, CollisionResolutionCanMatchFineColumnsWithoutChangingTheVehicle) {
  RouteMemory coarse,fine(.1);
  coarse.observe(10.,{-4,-4,4,4},{},{{.75,.25}});
  fine.observe(10.,{-4,-4,4,4},{},{{.75,.25}});
  EXPECT_STREQ(coarse.route({0,0},{3,0},0.).reason,"blocked start body");
  const auto retreat=fine.route({0,0},{-3,0},0.,.5,.77,.68,.58,{},false,true,-1);
  EXPECT_TRUE(retreat.reached)<<retreat.reason; // close obstacle still forbids an impossible forward turn
  EXPECT_THROW(RouteMemory(0.),std::invalid_argument);
}

TEST(PlanningBudget, NestedPhaseCannotExtendDeadlineAndScopeRestoresIt) {
  EXPECT_TRUE(std::isinf(PlanningBudget::remaining()));
  {
    PlanningBudget outer(60000);
    EXPECT_GT(PlanningBudget::remaining(),10000.);
    {
      PlanningBudget exhausted(0);
      EXPECT_TRUE(PlanningBudget::expired());
      PlanningBudget cannot_extend(60000);
      EXPECT_TRUE(PlanningBudget::expired());
    }
    EXPECT_GT(PlanningBudget::remaining(),10000.);
    EXPECT_LE(PlanningBudget::remaining(17.),17.);
  }
  EXPECT_TRUE(std::isinf(PlanningBudget::remaining()));
}

TEST(PlanningBudget, InteractionReservesCandidateAndFinalProofBeforeSourceExpiry) {
  using terrain_nav::interactionProofFitsFreshness;
  EXPECT_TRUE(interactionProofFitsFreshness(73.41,73.615,.015));
  EXPECT_FALSE(interactionProofFitsFreshness(73.59,73.615,.015));
  EXPECT_FALSE(interactionProofFitsFreshness(73.62,73.615,0.));
  EXPECT_FALSE(interactionProofFitsFreshness(73.59,73.615,.040));
  EXPECT_FALSE(interactionProofFitsFreshness(NAN,73.615,.01));
  EXPECT_FALSE(interactionProofFitsFreshness(73.4,INFINITY,.01));
  EXPECT_FALSE(interactionProofFitsFreshness(73.4,73.615,-.01));
}

TEST(PlanningBudget, WorkerDeadlineDoesNotAffectControllerThreadOrSurviveException) {
  {
    PlanningBudget exhausted(0);
    EXPECT_TRUE(std::async(std::launch::async,[]{return !PlanningBudget::expired();}).get());
    EXPECT_TRUE(PlanningBudget::expired());
    EXPECT_THROW(PlanningBudget(-1.),std::invalid_argument);
    EXPECT_TRUE(PlanningBudget::expired());
  }
  EXPECT_THROW(PlanningBudget(std::numeric_limits<double>::infinity()),std::invalid_argument);
  EXPECT_FALSE(PlanningBudget::expired());
}

TEST(RouteMemory, ExhaustedRoutingBudgetCannotReturnAnUnsearchedRoute) {
  RouteMemory memory;
  const auto result=memory.route({0,0},{3,0},0.,.5,.77,.68,.58,{},false,false,0,0.);
  EXPECT_FALSE(result.reached);EXPECT_TRUE(result.path.empty());
  EXPECT_STREQ(result.reason,"routing search budget");
  EXPECT_TRUE(memory.route({0,0},{3,0},0.).reached);
}
namespace {
std::vector<RouteMemory::Point> walls(double a,double b) {
  std::vector<RouteMemory::Point> points;
  for(double x=a;x<=b+1e-6;x+=.2)for(double y:{-1.35,1.35})points.push_back({x,y});
  return points;
}
}

TEST(RouteMemory, InitiallyUnknownExitIsExploredFromOneGoalAcrossLongWall) {
  RouteMemory memory;
  memory.observe(10.,{-10.4,-10.4,10.4,10.4},{},walls(-10.2,10.2));
  auto route=memory.route({0,0},{2.5,4.2},0.);
  ASSERT_TRUE(route.reached);ASSERT_GT(route.path.size(),20u);
  double far=0;for(const auto &p:route.path)far=std::max(far,p[0]);
  EXPECT_GT(far,10.4);EXPECT_GT(route.path[2][0],0.);
  // New observations extend the same wall. Old occupied geometry behind the
  // rolling sensor window prevents the tempting but pointless return branch.
  memory.observe(11.,{-4.4,-10.4,16.4,10.4},{},walls(-4.4,12.));
  route=memory.route({6,0},{2.5,4.2},0.);
  ASSERT_TRUE(route.reached);EXPECT_GT(route.path[2][0],6.);
  far=0;for(const auto &p:route.path)far=std::max(far,p[0]);EXPECT_GT(far,12.);
}

TEST(RouteMemory, OnlyConfirmedClearanceRemovesMovedObjectsWithoutExpiryWait) {
  RouteMemory memory;memory.observe(10.,{-5,-5,6,5},{},{{2,0},{2,.4},{2,-.4}});
  auto blocked=memory.route({0,0},{5,0},0.);ASSERT_TRUE(blocked.reached);
  EXPECT_TRUE(std::any_of(blocked.path.begin(),blocked.path.end(),[](const auto &p){return std::abs(p[1])>.6;}));
  memory.observe(10.1,{-5,-5,6,5},{},{});
  EXPECT_TRUE(memory.occupiedAt({2,0}));
  memory.observe(10.2,{-5,-5,6,5},{}, {}, {{2,0},{2,.4},{2,-.4}});
  auto cleared=memory.route({0,0},{5,0},0.);ASSERT_TRUE(cleared.reached);
  EXPECT_TRUE(std::all_of(cleared.path.begin(),cleared.path.end(),[](const auto &p){return std::abs(p[1])<.41;}));
}

TEST(RouteMemory, RepeatedGearAndResolutionSearchReusesHeuristicButNewObstacleInvalidatesIt) {
  RouteMemory memory;memory.observe(10.,{-5,-5,9,5},{},{});
  auto first=memory.route({0,0},{7,0},0.);ASSERT_TRUE(first.reached);EXPECT_FALSE(first.heuristic_reused);
  auto fine=memory.route({0,0},{7,0},0.,.5,.77,.68,.58,{},true);
  ASSERT_TRUE(fine.reached);EXPECT_TRUE(fine.heuristic_reused);
  memory.observe(10.1,{-5,-5,9,5},{},{{3,0}});
  auto changed=memory.route({0,0},{7,0},0.);ASSERT_TRUE(changed.reached);
  EXPECT_FALSE(changed.heuristic_reused);
  EXPECT_TRUE(std::any_of(changed.path.begin(),changed.path.end(),[](const auto &p){return std::abs(p[1])>.6;}));
  memory.observe(10.2,{-5,-5,9,5},{},{},{{3,0}});
  auto freed=memory.route({0,0},{7,0},0.);ASSERT_TRUE(freed.reached);EXPECT_FALSE(freed.heuristic_reused);
  EXPECT_TRUE(std::all_of(freed.path.begin(),freed.path.end(),[](const auto &p){return std::abs(p[1])<.41;}));
}

TEST(RouteMemory, CachedHeuristicNeverCachesTerrainVetoAndExpiresOnClockGoalOrWidthChange) {
  RouteMemory memory;memory.observe(10.,{-5,-5,9,5},{},{});
  ASSERT_TRUE(memory.route({0,0},{7,0},0.).reached);
  const auto veto=[](double x,double y,double) {
    return x>2 && x<3.5 && std::abs(y)<.65?std::numeric_limits<double>::infinity():1.;
  };
  auto terrain=memory.route({0,0},{7,0},0.,.5,.77,.68,.58,veto);
  ASSERT_TRUE(terrain.reached);EXPECT_TRUE(terrain.heuristic_reused);
  for(const auto &p:terrain.path)EXPECT_TRUE(std::isfinite(veto(p[0],p[1],0.)));
  EXPECT_FALSE(memory.route({0,0},{6,0},0.).heuristic_reused);
  EXPECT_FALSE(memory.route({0,0},{6,0},0.,.6).heuristic_reused);
  memory.observe(11.,{-5,-5,9,5},{},{});
  EXPECT_FALSE(memory.route({0,0},{6,0},0.,.6).heuristic_reused);
  memory.observe(9.,{-5,-5,9,5},{},{});
  EXPECT_FALSE(memory.route({0,0},{6,0},0.,.6).heuristic_reused);
}

TEST(RouteMemory, QuantizationJitterReusesOnlyHintAndKeepsShiftedBodyCollisionCurrent) {
  RouteMemory memory;memory.observe(10.,{-5,-5,9,5},{},{{3,0}});
  ASSERT_TRUE(memory.route({0,0},{7,0},0.).reached);
  memory.observe(10.1,{-5,-5,9,5},{},{{3.2,0}},{{3,0}});
  auto shifted=memory.route({0,0},{7,0},0.);
  ASSERT_TRUE(shifted.reached);EXPECT_TRUE(shifted.quantized_heuristic_reused);
  for(size_t i=0;i<shifted.path.size();++i) {
    const double dx=3.3-shifted.path[i][0],dy=.1-shifted.path[i][1],yaw=shifted.yaws[i];
    const double forward=dx*std::cos(yaw)+dy*std::sin(yaw),lateral=-dx*std::sin(yaw)+dy*std::cos(yaw);
    EXPECT_FALSE(forward>=-.58 && forward<=.68 && std::abs(lateral)<=.5);
  }
  RouteMemory nose;nose.observe(10.,{-5,-5,9,5},{},{{.8,0}});
  nose.route({0,0},{7,0},0.);
  nose.observe(10.1,{-5,-5,9,5},{},{{.6,0}});
  auto hit=nose.route({0,0},{7,0},0.);
  EXPECT_FALSE(hit.reached);EXPECT_STREQ(hit.reason,"blocked start body");
}

TEST(RouteMemory, UnseenRearWallEndDoesNotPermitAnImpossiblePointRobotUTurn) {
  RouteMemory memory;
  // Recorded tilted-LiDAR failure mechanism: much shorter visible wall
  // behind the vehicle. A point-grid shortcut goes backward to the apparent
  // end; a car cannot reverse its heading inside this protected-width lane.
  memory.observe(10.,{-8,-10.4,12.8,10.4},{},walls(-1.2,11.8));
  const auto route=memory.route({2.,.2},{2.1,4.2},.4);
  ASSERT_TRUE(route.reached);
  EXPECT_GT(std::max_element(route.path.begin(),route.path.end(),
      [](const auto &a,const auto &b){return a[0]<b[0];})->at(0),11.8);
  EXPECT_GT(route.path[2][0],2.);
}

TEST(RouteMemory, UnknownRoutingIsOnlyOptimisticAndClockResetDropsHistory) {
  RouteMemory memory;EXPECT_TRUE(memory.route({0,0},{5,0},0).reached);
  memory.observe(10.,{-5,-5,5,5},{},{{2,0}});EXPECT_GT(memory.size(),0u);
  memory.observe(9.,{10,10,11,11},{},{});EXPECT_EQ(memory.size(),0u);
  EXPECT_FALSE(memory.route({0,0},{1000,1000},0).reached);
  EXPECT_THROW(memory.observe(10.,{0,0,-1,1},{},{}),std::invalid_argument);
}

TEST(RouteMemory, ReversibleRecoveryRouteKeepsBodyYawAndSeparateGearStates) {
  RouteMemory memory;auto occupied=walls(-7.,2.);
  for(double y=-1.4;y<=1.4;y+=.2)occupied.push_back({2.,y});
  memory.observe(10.,{-8,-4,4,4},{},occupied);
  const auto forward=memory.route({1.,0.},{-4.,0.},0.);
  EXPECT_FALSE(forward.reached);
  const auto reverse=memory.route({1.,0.},{-4.,0.},0.,.5,.7547,.68,.58,{},false,true);
  ASSERT_TRUE(reverse.reached)<<reverse.reason;
  ASSERT_EQ(reverse.path.size(),reverse.directions.size());
  for(size_t i=1;i<reverse.path.size();++i) {
    EXPECT_EQ(reverse.directions[i],-1);
    EXPECT_NEAR(reverse.yaws[i],0.,1e-6);
    EXPECT_LT(reverse.path[i][0],reverse.path[i-1][0]);
  }
}

TEST(RouteMemory, RecoveryCanReturnThroughStartInAnotherGearWithoutPruningItsExit) {
  RouteMemory memory;
  const auto route=memory.route({0.,0.},{3.,0.},0.,.5,.7547,.68,.58,{},false,true,-1);
  ASSERT_TRUE(route.reached)<<route.reason;
  ASSERT_GT(route.directions.size(),2u);
  EXPECT_EQ(route.directions.front(),-1);
  EXPECT_TRUE(std::find(route.directions.begin(),route.directions.end(),1)!=route.directions.end());
  EXPECT_NEAR(route.path.back()[0],3.,1e-9);
}

TEST(RouteMemory, ExitTurnUsesObstacleAwareRoutingRatherThanExhaustingSearchAcrossWall) {
  RouteMemory memory;std::vector<RouteMemory::Point> seen;
  for(double x=4.6;x<25.4;x+=.4)for(double y=-9.6;y<11.2;y+=.4)seen.push_back({x,y});
  memory.observe(175.,{4.6,-9.6,25.4,11.2},seen,walls(-12.,12.));
  const auto result=memory.route({15.01595,.84787},{2.14616,4.1998},.27159,.5,.7547);
  ASSERT_TRUE(result.reached)<<result.reason;ASSERT_EQ(result.yaws.size(),result.path.size());
  EXPECT_LT(result.expanded,2000);EXPECT_GT(result.yaws[3],.27159);
  for(size_t i=1;i+1<result.path.size();++i) {
    const double gap=std::atan2(std::sin(result.yaws[i]-result.yaws[i-1]),std::cos(result.yaws[i]-result.yaws[i-1]));
    EXPECT_LE(std::abs(gap),.075471);
  }
}

TEST(RouteMemory, RoutingUsesActualFrontOverhangAndChangesEnvelopeWithYaw) {
  RouteMemory memory;memory.observe(10.,{-5,-5,5,6},{},{{.65,0.}});
  // The 0.50m inscribed circle is clear, but the protected 0.68m nose hits.
  auto forward=memory.route({0,0},{4,0},0.,.5,.7547,.68,.58);
  EXPECT_FALSE(forward.reached);EXPECT_STREQ(forward.reason,"blocked start body");
  // Rotating the SAME complete body makes the obstacle lateral and clear.
  EXPECT_TRUE(memory.route({0,0},{0,4},M_PI/2,.5,.7547,.68,.58).reached);
}

TEST(RouteMemory, KnownUnsafeTerrainBlocksRoutingButUnknownRemainsAnExplorationHint) {
  RouteMemory memory;
  const auto cost=[](double x,double y,double) {
    return x>2 && x<3.5 && std::abs(y)<.65?std::numeric_limits<double>::infinity():1.;
  };
  const auto around=memory.route({0,0},{6,0},0.,.5,.7547,.68,.58,cost);
  ASSERT_TRUE(around.reached)<<around.reason;
  EXPECT_TRUE(std::any_of(around.path.begin(),around.path.end(),[](const auto &p){return std::abs(p[1])>.65;}));
  for(const auto &p:around.path)EXPECT_TRUE(std::isfinite(cost(p[0],p[1],0.)));
  const auto unknown=memory.route({0,0},{5,0},0.,.5,.7547,.68,.58,[](double,double,double){return 3.;});
  EXPECT_TRUE(unknown.reached); // still requires the separate native execution certificate
}

TEST(RouteMemory, OcclusionPredictedSupportAndTimeoutNeverOpenARecordedWall) {
  RouteMemory memory(.1);const auto wall=walls(-17.5,17.5);
  memory.observe(10.,{-20,-5,20,5},{},wall);
  memory.observe(11.,{5,-5,25,5},wall,{}); // a floor prediction is NOT clearing
  memory.observe(200.,{5,-5,25,5},{},{}); // long drive / no new return
  for(const auto &p:wall)EXPECT_TRUE(memory.occupiedAt(p));
  const auto route=memory.route({19.,0.},{2.5,4.2},0.,.5,.77,.68,.58,{},false,false,0,200.);
  ASSERT_TRUE(route.reached)<<route.reason;
  EXPECT_GT(route.yaws[3],0.); // correct upper side of the complete wall
  EXPECT_GT(route.path.back()[1],4.);
}

TEST(RouteMemory, CurrentOccupancyOverridesClearanceAndOutOfViewClearanceIsIgnored) {
  RouteMemory memory;
  memory.observe(10.,{-5,-5,5,5},{},{{2,0},{8,0}});
  memory.observe(11.,{-5,-5,5,5},{},{{2,0}},{{2,0},{8,0}});
  EXPECT_TRUE(memory.occupiedAt({2,0}));EXPECT_TRUE(memory.occupiedAt({8,0}));
  memory.observe(12.,{-5,-5,5,5},{},{},{{2,0}});
  EXPECT_FALSE(memory.occupiedAt({2,0}));EXPECT_TRUE(memory.occupiedAt({8,0}));
}

TEST(RouteMemory, AdvisoryPreservesTopologyWithoutASecondFullBodyObstacleAtTheStart) {
  RouteMemory memory(.1);
  const std::vector<RouteMemory::Point> wall={{.65,.45}};
  memory.observe(10.,{-5,-5,6,5},{},{},{},wall,true);
  EXPECT_TRUE(memory.occupiedAt(wall.front()));
  EXPECT_STREQ(memory.route({0,0},{4,0},0.).reason,"blocked start body");
  terrain_nav::RouteOptions qualified;qualified.allow_advisory_start_overlap=true;
  auto route=memory.route({0,0},{4,0},0.,.5,.77,.68,.58,{},true,false,0,100.,qualified);
  EXPECT_TRUE(route.reached)<<route.reason;
  memory.observe(11.,{-5,-5,6,5},{},wall);
  EXPECT_STREQ(memory.route({0,0},{4,0},0.,.5,.77,.68,.58,{},true,false,0,100.,qualified).reason,"blocked start body");
}

TEST(RouteMemory, AdvisoryRefreshDoesNotEraseUnobservedOutsideGeometryOrActualOccupancy) {
  RouteMemory memory(.1);
  memory.observe(10.,{-6,-5,6,5},{},{{2,0}},{},{{2,0},{-4,0}},true);
  memory.observe(11.,{-3,-5,6,5},{},{},{},{},true);
  EXPECT_TRUE(memory.occupiedAt({2,0})); // actual occupancy unchanged
  EXPECT_TRUE(memory.occupiedAt({-4,0})); // outside mapper view
  memory.observe(12.,{-6,-5,6,5},{},{},{},{{2,0}},true);
  EXPECT_FALSE(memory.occupiedAt({-4,0}));
}

TEST(RouteMemory, HistoricalWallsStillForbidAFullBodyUTurnInsideTheCorridor) {
  RouteMemory memory(.1);memory.observe(10.,{-9,-5,9,5},{},{},{},walls(-8,8),true);
  terrain_nav::RouteOptions options;options.heuristic_weight=2.5;options.allow_advisory_start_overlap=true;
  auto route=memory.route({0,0},{0,4.2},0.,.5,.77,.68,.58,{},false,false,0,200.,options);
  ASSERT_TRUE(route.reached)<<route.reason;
  double max_x=-1e9;for(const auto &p:route.path)max_x=std::max(max_x,p[0]);
  EXPECT_GT(max_x,8.5); // a centre-only turn would spuriously stay inside
}

TEST(RouteMemory, HunterTopologyArcUsesRearAxleKinematicsAtTheBasePose) {
  RouteMemory memory(.1);std::vector<RouteMemory::Point> floor;
  for(int y=-40;y<=40;++y)for(int x=-40;x<=60;++x)floor.push_back({x*.1,y*.1});
  memory.observe(1.,{-4,-4,6,4},floor,{});
  terrain_nav::RouteOptions options;options.rear_axle_offset=.2078;
  auto route=memory.route({0,0},{3.,1.},0.,.5,.75,.68,.58,{},true,true,1,250.,options);
  ASSERT_TRUE(route.reached)<<route.reason;ASSERT_EQ(route.path.size(),route.yaws.size());
  EXPECT_NEAR(route.path.back()[0],3.,1e-8);EXPECT_NEAR(route.path.back()[1],1.,1e-8);
  for(size_t i=1;i<route.path.size();++i) {
    const double a=options.rear_axle_offset,ya=route.yaws[i-1],yb=route.yaws[i];
    const double dx=route.path[i][0]-a*std::cos(yb)-route.path[i-1][0]+a*std::cos(ya);
    const double dy=route.path[i][1]-a*std::sin(yb)-route.path[i-1][1]+a*std::sin(ya);
    const double turn=std::atan2(std::sin(yb-ya),std::cos(yb-ya));
    if(std::hypot(dx,dy)<1e-9)continue;
    EXPECT_NEAR(std::sin(std::atan2(dy,dx)-ya-.5*turn-(route.directions[i]<0?M_PI:0.)),0.,1e-8);
    EXPECT_LE(std::abs(2.*std::sin(.5*turn))/std::hypot(dx,dy),.750001);
  }
  options.rear_axle_offset=NAN;
  EXPECT_FALSE(memory.route({0,0},{3.,1.},0.,.5,.75,.68,.58,{},true,true,1,250.,options).reached);
}

TEST(RouteMemory, FiniteReverseObservationPreferenceDoesNotBanAnOtherwiseAvailableRoute) {
  RouteMemory memory(.1);memory.observe(1.,{-6,-3,3,3},{},{});
  terrain_nav::RouteOptions options;options.rear_axle_offset=.2078;
  int evaluations=0;options.reverse_observation_cost=[&](double,double,double){++evaluations;return 40.;};
  const auto route=memory.route({0,0},{-2.,0.},0.,.5,.75,.68,.58,{},true,true,-1,250.,options);
  EXPECT_TRUE(route.reached)<<route.reason;EXPECT_GT(evaluations,0);
  ASSERT_FALSE(route.directions.empty());EXPECT_EQ(route.directions.front(),-1);
  // A finite cost may choose a forward detour after the required initial
  // reverse. It must retain that option rather than banning reverse motion.
  memory.observe(2.,{-6,-3,3,3},{},{{-.7,0.}});
  const auto blocked=memory.route({0,0},{-2.,0.},0.,.5,.75,.68,.58,{},true,true,-1,250.,options);
  EXPECT_FALSE(blocked.reached);
}
