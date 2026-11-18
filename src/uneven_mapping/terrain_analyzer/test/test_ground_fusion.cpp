#include <gtest/gtest.h>
#include <algorithm>
#include <vector>
#include "terrain_analyzer/ground_fusion.hpp"
#include "ground_witness_cases.hpp"
using namespace terrain_analyzer;
TEST(GroundFusion,SparseCurrentReturnUsesAnEnclosureWithoutFabricatingAScanPlane) {
  for(int mode=0;mode<=11;++mode) {
    const auto result=sparseReturnCase(mode);
    EXPECT_EQ(result.qualified,mode==0||mode==10)<<mode;
    if(result.qualified) {
      EXPECT_FLOAT_EQ(result.scan.height,0);EXPECT_FLOAT_EQ(result.scan.variance,.0001f);
      EXPECT_EQ(result.scan.count,mode==10?500:1);EXPECT_FALSE(result.scan.centered);
      auto old=GroundState{};old.observed=true;old.height=.39f;old.variance=.00055f;
      const auto repaired=fuseGroundScan(old,result.scan,.0001f,2.5f,true,.18f,true);
      EXPECT_FLOAT_EQ(repaired.height,0);EXPECT_EQ(repaired.pending_count,0);
    }
  }
}
TEST(GroundFusion,FreeSpaceRetiresOnlyElevatedLabelsWithEnclosedReliableFloor) {
  for(int mode=0;mode<=13;++mode) {
    const auto result=freeSpaceCase(mode);
    EXPECT_EQ(result.retired,mode==0||mode==13)<<mode;
    EXPECT_FLOAT_EQ(result.reference,mode==0||mode==13?0.f:99.f)<<mode;
  }
  // Retirement supplies no measurement; a future scan is still necessary.
  auto old=GroundState{};old.observed=true;old.height=.39f;old.variance=.0001f;
  float ref=NAN;EXPECT_TRUE(freeSpaceGroundReference(old,FreeSpaceReferences{0},3,3,.18f,ref));
  EXPECT_FLOAT_EQ(old.height,.39f);EXPECT_TRUE(old.observed);
  EXPECT_FALSE(freeSpaceGroundReference(old,FreeSpaceReferences{0},3,3,NAN,ref));
  EXPECT_FALSE(freeSpaceGroundReference(old,FreeSpaceReferences{0},3,0,.18f,ref));
  EXPECT_TRUE(freeRayIntersectsGround(old,.40f));
  EXPECT_FALSE(freeRayIntersectsGround(old,.44f));
  old.variance=.00055f;EXPECT_TRUE(freeRayIntersectsGround(old,.44f));
  old.variance=.1f;EXPECT_FALSE(freeRayIntersectsGround(old,.52f));
  old.observed=false;EXPECT_FALSE(freeRayIntersectsGround(old,.39f));
}
TEST(GroundFusion,CurrentFloorWitnessRejectsPitsStepsEmptyOneSidedPendingAndUncertainEvidence) {
  for(int mode=0;mode<=12;++mode) {
    const auto result=witnessCase(mode);
    EXPECT_EQ(result.qualified,mode==0||mode==1)<<mode;
    if(result.qualified) {
      EXPECT_NEAR(result.scan.height,0,1e-6);EXPECT_GE(result.scan.variance,.0000999f);
      EXPECT_EQ(result.scan.count,1);EXPECT_EQ(result.scan.coverage,1U);
    }
  }
}
struct Samples {
  std::vector<float> h;float noise=.0001f;
  std::vector<unsigned int> bits;
  int first() const {return h.empty()?-1:0;}
  int next(int i) const {return i+1<int(h.size())?i+1:-1;}
  float height(int i) const {return h[i];}
  float variance(int) const {return noise;}
  unsigned int coverage(int i) const {return bits.empty()?1U:bits[i];}
};
GroundState flat() {GroundState s;s.observed=true;s.variance=.0001f;return s;}
GroundState step(GroundState s,float h,int count=1) {
  return fuseGroundScan(s,robustGroundScan(Samples{std::vector<float>(count,h)},.0001f),
                        .0001f,2.5f,true,.18f);
}
TEST(GroundScan,ReturnOrderAndIsolatedLowOutlierDoNotMoveTheFloor) {
  Samples p{{-.08f,-.001f,0.f,.001f,.002f}};const auto a=robustGroundScan(p,.0001f);
  EXPECT_NEAR(a.height,0.f,1e-5f);EXPECT_GE(a.variance,.0001f);
  do {const auto b=robustGroundScan(p,.0001f);EXPECT_NEAR(a.height,b.height,1e-6f);}
  while(std::next_permutation(p.h.begin(),p.h.end()));
}
TEST(GroundScan,DuplicatedRaysAreNotIndependentObservations) {
  auto one=robustGroundScan(Samples{{.06f}},.0001f);
  auto many=robustGroundScan(Samples{std::vector<float>(500,.06f)},.0001f);
  EXPECT_FLOAT_EQ(one.variance,many.variance);
  auto s=step(flat(),.06f,500);EXPECT_EQ(s.pending_count,1);EXPECT_FLOAT_EQ(s.height,0.f);
}
TEST(GroundScan,EmptyOrInvalidReturnsCannotCreateAHeightOrEraseAPendingChange) {
  auto scan=robustGroundScan(Samples{{NAN,INFINITY}},.0001f);EXPECT_EQ(scan.count,0);
  auto s=step(flat(),-.08f);
  auto retained=fuseGroundScan(s,scan,.0001f,2.5f,true,.18f);
  EXPECT_EQ(retained.pending_count,1);EXPECT_FLOAT_EQ(retained.height,0.f);
}
TEST(GroundScan,TrueLowerSurfaceMajoritySurvivesAndMixedSurfaceScatterIsNotZero) {
  auto scan=robustGroundScan(Samples{{-.3f,-.299f,-.301f,0.f,0.f}},.0001f);
  EXPECT_NEAR(scan.height,-.299f,1e-5f);EXPECT_GE(scan.variance,.0001f);
  auto rough=robustGroundScan(Samples{{-.03f,-.01f,0.f,.01f,.03f}},.0001f);
  EXPECT_GT(rough.variance,.0003f);
}
TEST(GroundFusion,AnUnconfirmedLowPointDoesNotBecomePermanentGround) {
  auto s=step(flat(),-.08f);EXPECT_EQ(s.pending_count,1);EXPECT_FLOAT_EQ(s.height,0.f);
  s=step(s,0.f);EXPECT_EQ(s.pending_count,0);EXPECT_FLOAT_EQ(s.height,0.f);
}
TEST(GroundFusion,ConfirmedSmallRiseCanRepairAnOldLowFloorOrRepresentALedge) {
  auto s=flat();for(int i=0;i<2;++i){s=step(s,.06f);EXPECT_EQ(s.pending_count,i+1);EXPECT_FLOAT_EQ(s.height,0.f);}
  s=step(s,.06f);EXPECT_EQ(s.pending_count,0);EXPECT_NEAR(s.height,.06f,1e-6f);
}
TEST(GroundFusion,SpatiallyDiverseCoherentSmallLedgeIsOneImmediateObservation) {
  Samples p{{.059f,.06f,.061f},.0001f,{groundCoverageBit(.1,.1),groundCoverageBit(.1,.3),groundCoverageBit(.1,.6)}};
  auto scan=robustGroundScan(p,.0001f);ASSERT_TRUE(spatialGroundEvidence(scan));
  auto s=fuseGroundScan(flat(),scan,.0001f,2.5f,true,.18f);
  EXPECT_EQ(s.pending_count,0);EXPECT_NEAR(s.height,.06f,1e-5f);
  EXPECT_GE(s.variance,.0001f); // No division by number of returns/subcells.
}
TEST(GroundFusion,SpatialCoverageCannotAuthorizeDeepPitOrNoisyOrDuplicateEvidence) {
  Samples p{{-.301f,-.3f,-.299f},.0001f,{1U,16U,256U}};
  auto scan=robustGroundScan(p,.0001f);ASSERT_TRUE(spatialGroundEvidence(scan));
  auto s=fuseGroundScan(flat(),scan,.0001f,2.5f,true,.18f);
  EXPECT_EQ(s.pending_count,1);EXPECT_FLOAT_EQ(s.height,0.f);
  p.h={.06f,.06f,.06f};p.bits={1U,1U,1U};
  EXPECT_FALSE(spatialGroundEvidence(robustGroundScan(p,.0001f)));
  p.bits={1U,16U,256U};p.noise=.001f;
  EXPECT_FALSE(spatialGroundEvidence(robustGroundScan(p,.0001f)));
}
TEST(GroundFusion,PassingHighObjectDoesNotReplaceTheFloor) {
  auto s=flat();for(int i=0;i<8;++i)s=step(s,.6f,30);
  EXPECT_EQ(s.pending_count,0);EXPECT_FLOAT_EQ(s.height,0.f);
}
TEST(GroundFusion,OcclusionDoesNotSupplyConfirmationAndDifferentHeightsResetIt) {
  auto s=step(flat(),.06f);s=fuseGroundScan(s,{},.0001f,2.5f,true,.18f);
  EXPECT_EQ(s.pending_count,1);s=step(s,.13f);EXPECT_EQ(s.pending_count,1);
  EXPECT_NEAR(s.pending_height,.13f,1e-6f);EXPECT_FLOAT_EQ(s.height,0.f);
}
TEST(GroundFusion,DeepPitIsNotFusedFromOneReturnAndRemainsPendingUntilConfirmed) {
  auto s=step(flat(),-.3f);EXPECT_EQ(s.pending_count,1);EXPECT_FLOAT_EQ(s.height,0.f);
  // Immediate raw-pit veto is a separate node-level regression.
  s=step(step(s,-.3f),-.3f);EXPECT_EQ(s.pending_count,0);EXPECT_NEAR(s.height,-.3f,1e-6f);
}

TEST(GroundFusion,EnclosedMeasuredLowerPlaneCanRetireAnOldObstacleHeightWithoutExtraScans) {
  auto old=flat();old.height=.4f;old.pending_count=1;old.pending_height=0;
  GroundScan scan{0.f,.0001f,1,1,1U,true};GroundState neighbors[9];
  for(auto &n:neighbors)n=flat();neighbors[4]=old;
  ASSERT_TRUE(enclosedLowerGround(old,scan,neighbors));
  const auto actual=fuseGroundScan(old,scan,.0001f,2.5f,true,.18f,enclosedLowerGround(old,scan,neighbors));
  EXPECT_FLOAT_EQ(actual.height,0);EXPECT_FLOAT_EQ(actual.variance,scan.variance);
  EXPECT_EQ(actual.pending_count,0);EXPECT_EQ(scan.count,1);EXPECT_EQ(scan.coverage,1U);
  old.height=.073f;EXPECT_TRUE(enclosedLowerGround(old,scan,neighbors));
  old.variance=.01f;EXPECT_FALSE(enclosedLowerGround(old,scan,neighbors));
}

TEST(GroundFusion,EnclosureCannotTurnNewPitEmptyCellOrOneSidedUncertainEvidenceIntoGround) {
  auto old=flat();GroundScan scan{-.4f,.0001f,1,1,1U,true};GroundState neighbors[9];
  for(auto &n:neighbors)n=flat();
  EXPECT_FALSE(enclosedLowerGround(old,scan,neighbors)); // newly lower than surrounding terrain
  old.height=.4;scan.height=0;
  for(int mode=0;mode<6;++mode) {
    auto s=scan;auto o=old;for(auto &n:neighbors)n=flat();
    if(mode==0)s.count=0;
    if(mode==1)s.centered=false;
    if(mode==2)o.observed=false;
    if(mode==3)for(int k=3;k<9;++k)neighbors[k].observed=false;
    if(mode==4)for(auto &n:neighbors)n.pending_count=1;
    if(mode==5)for(auto &n:neighbors)n.variance=.01f;
    EXPECT_FALSE(enclosedLowerGround(o,s,neighbors))<<mode;
  }
  scan.height=.8;EXPECT_FALSE(enclosedLowerGround(old,scan,neighbors)); // never clear additive occupancy
}

struct PositionedSamples {
  std::vector<float> h,px,py;
  int first()const{return h.empty()?-1:0;}
  int next(int i)const{return ++i<int(h.size())?i:-1;}
  float height(int i)const{return h[i];}
  float variance(int)const{return .0001f;}
  float x(int i)const{return px[i];}
  float y(int i)const{return py[i];}
  unsigned int coverage(int i)const{return groundCoverageBit(px[i]+.5f,py[i]+.5f);}
};

TEST(GroundScan,SlopingCellCentreDoesNotMoveWithBeamPhaseAndUncertaintyDoesNotShrinkWithRayCount) {
  for(float phase:{-.06f,.06f})for(float slope:{-.65f,.35f,.65f}) {
    PositionedSamples p;
    for(float x:{-.35f,-.15f,.15f,.35f})for(float y:{-.35f,-.15f,.15f,.35f}) {
      p.px.push_back(x+phase);p.py.push_back(y);
      p.h.push_back(.12f+.2f*slope*(x+phase)+.2f*.2f*y);
    }
    const auto result=centeredGroundScan(p,.0001f);
    EXPECT_NEAR(result.height,.12f,2e-6f);
    EXPECT_GE(result.variance,.00009999f);
    EXPECT_NEAR(robustGroundScan(p,.0001f).height,.12f,.04f);
    EXPECT_TRUE(spatialGroundEvidence(result));
  }
}

TEST(GroundScan,CentredPlaneCannotSmoothVerticalStepsOrFitOneSidedOrCollinearRays) {
  PositionedSamples p;
  for(float x:{-.4f,-.2f,0.f,.2f,.4f})for(float y:{-.4f,-.2f,0.f,.2f,.4f}) {
    p.px.push_back(x);p.py.push_back(y);p.h.push_back(x>0?.06f:0.f);
  }
  auto expect_fallback=[&] {
    const auto old=robustGroundScan(p,.0001f),actual=centeredGroundScan(p,.0001f);
    EXPECT_FLOAT_EQ(actual.height,old.height);EXPECT_FLOAT_EQ(actual.variance,old.variance);
    EXPECT_EQ(actual.inliers,old.inliers);
  };
  expect_fallback();
  for(auto &x:p.px)x=x*.2f+.35f;
  expect_fallback();
  for(auto &y:p.py)y=0.f;
  expect_fallback();
}

TEST(GroundScan,SparseCellUsesRealPatchSlopeWithoutInventingCoverageOrConfirmation) {
  PositionedSamples patch;
  for(float x:{-1.2f,-.6f,0.f,.6f,1.2f})for(float y:{-1.2f,-.6f,0.f,.6f,1.2f}) {
    patch.px.push_back(x);patch.py.push_back(y);patch.h.push_back(.12f+.066f*x+.02f*y);
  }
  for(float phase:{-.4f,.4f}) {
    PositionedSamples own{{.12f+.066f*phase},{phase},{0.f}};
    const auto scan=patchCenteredGroundScan(own,patch,.0001f);
    EXPECT_TRUE(scan.centered);EXPECT_NEAR(scan.height,.12f,1e-6f);
    EXPECT_GE(scan.variance,.00009999f);EXPECT_EQ(scan.count,1);EXPECT_EQ(scan.inliers,1);
    EXPECT_EQ(scan.coverage,own.coverage(0));EXPECT_FALSE(spatialGroundEvidence(scan));
    auto previous=flat();previous.height=.06f;
    previous=fuseGroundScan(previous,scan,.0001f,2.5f,true,.18f);
    EXPECT_EQ(previous.pending_count,1); // A neighborhood is not three scans.
  }
  EXPECT_EQ(patchCenteredGroundScan(PositionedSamples{},patch,.0001f).count,0);
  PositionedSamples own{{.3f},{0.f},{0.f}};
  patch.h[12]=.3f;
  const auto rejected=patchCenteredGroundScan(own,patch,.0001f);
  EXPECT_FALSE(rejected.centered);EXPECT_FLOAT_EQ(rejected.height,.3f);
}

TEST(GroundScan,PatchDoesNotSmoothStepsPitsOrOneSidedEvidence) {
  PositionedSamples patch;
  for(float x:{-1.2f,-.6f,0.f,.6f,1.2f})for(float y:{-1.2f,-.6f,0.f,.6f,1.2f}) {
    patch.px.push_back(x);patch.py.push_back(y);patch.h.push_back(x>0?.12f:0.f);
  }
  const PositionedSamples own{{0.f},{-.3f},{0.f}};
  EXPECT_FALSE(patchCenteredGroundScan(own,patch,.0001f).centered);
  patch.h.assign(patch.h.size(),0.f);patch.h[12]=-.3f;
  EXPECT_FALSE(patchCenteredGroundScan(own,patch,.0001f).centered);
  patch.h.assign(patch.h.size(),0.f);
  for(auto &x:patch.px)x=fabsf(x)+.3f;
  EXPECT_FALSE(patchCenteredGroundScan(own,patch,.0001f).centered);
}

TEST(GroundScan,ScanPatchWrapsStorageButNeverAcrossOppositeLogicalEdges) {
  const int width=7,height=5,start_x=3,start_y=4;
  const int corner=start_y*width+start_x;
  for(int k:{0,1,2,3,6})EXPECT_EQ(groundPatchCell(corner,k,width,height,start_x,start_y),-1);
  EXPECT_EQ(groundPatchCell(corner,4,width,height,start_x,start_y),corner);
  EXPECT_EQ(groundPatchCell(corner,7,width,height,start_x,start_y),start_x);
  EXPECT_EQ(groundPatchDelta(0,6,7),1);EXPECT_EQ(groundPatchDelta(6,0,7),-1);
  EXPECT_EQ(groundPatchDelta(1,1,7),0);
}
