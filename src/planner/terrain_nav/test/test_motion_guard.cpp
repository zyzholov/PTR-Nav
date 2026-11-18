#include "terrain_nav/motion_guard.hpp"
#include "terrain_nav/motion_resolution.hpp"
#include "terrain_nav/dynamic_yield.hpp"
#include "terrain_nav/temporal_geometry.hpp"
#include "terrain_nav/dynamic_occupancy.hpp"
#include <gtest/gtest.h>
#include <fstream>
#include <nlohmann/json.hpp>
#include "data/closing_gate_translation_gap.hpp"
#include "data/recorded_radius_collapse.hpp"

TEST(DynamicYield, BoundedAccelerationCoversVariableSpeedAndSaturatesAtTrackingSpeedLimit) {
  using terrain_nav::boundedTranslationDistance;
  EXPECT_DOUBLE_EQ(boundedTranslationDistance(.8,2.,0.),1.6);
  EXPECT_NEAR(boundedTranslationDistance(.8,1.,1.2),1.4,1e-12);
  EXPECT_NEAR(boundedTranslationDistance(4.,2.,1.2),
    4.*(.5/1.2)+.6*std::pow(.5/1.2,2)+4.5*(2.-.5/1.2),1e-12);
  EXPECT_TRUE(std::isinf(boundedTranslationDistance(.8,-1.,1.2)));
  EXPECT_TRUE(std::isinf(boundedTranslationDistance(.8,1.,NAN)));
  EXPECT_TRUE(std::isinf(boundedTranslationDistance(5.,1.,1.2)));
  // A genuinely varying speed, not differently sampled constant translation.
  // v(t)=.8*(1+.5*sin(pi*t/2)); max |dv/dt|=.2*pi<1.2m/s².
  const auto position=[](double t){return .8*(t+(1.-std::cos(M_PI*t/2))/M_PI);};
  bool constant_velocity_misses=false;
  for(double source=0;source<6.;source+=.1)for(double dt=.1;dt<=2.;dt+=.1) {
    const double v=.8*(1.+.5*std::sin(M_PI*source/2));
    const double actual=position(source+dt)-position(source);
    EXPECT_LE(actual,boundedTranslationDistance(v,dt,1.2)+1e-12);
    constant_velocity_misses|=actual>v*dt+.1*dt;
  }
  EXPECT_TRUE(constant_velocity_misses);
}

TEST(MotionGuard, PredictionDomainIsExplicitAndCannotTruncateAnAlreadyFasterObservation) {
  EXPECT_DOUBLE_EQ(terrain_nav::predictionSpeedCap(.8,.1,2.),2.);
  EXPECT_DOUBLE_EQ(terrain_nav::predictionSpeedCap(2.1,.1,2.),4.5);
  EXPECT_DOUBLE_EQ(terrain_nav::predictionSpeedCap(1.95,.1,2.),4.5);
  EXPECT_DOUBLE_EQ(terrain_nav::predictionSpeedCap(.8,.1,4.5),4.5);
  EXPECT_TRUE(std::isnan(terrain_nav::predictionSpeedCap(.8,.1,NAN)));
  EXPECT_TRUE(std::isnan(terrain_nav::predictionSpeedCap(.8,-.1,2.)));
  const double cap=terrain_nav::predictionSpeedCap(3.,.1,2.);
  EXPECT_GT(terrain_nav::boundedTranslationDistance(3.,1.,1.2,cap),3.);
}

TEST(MotionGuard, IdentityAssociationIncludesBoundedAccelerationWithoutChangingCollisionGeometry) {
  // Integrating a bounded vector acceleration gives |p-p_CV| <= a*t^2/2.
  // This is an identity gate only: it neither clears cells nor reduces risk.
  for(double age:{.1,.3,.6})for(double heading:{0.,.7,1.57,2.8}) {
    const double offset=.5*1.2*age*age;
    const double actual_error=std::hypot(offset*std::cos(heading),offset*std::sin(heading));
    EXPECT_GE(terrain_nav::motionIdentityRadius(.05,age,.1,1.2),actual_error);
    EXPECT_DOUBLE_EQ(terrain_nav::motionIdentityRadius(.05,age,.1,0.),
      terrain_nav::motionAssociationRadius(.05,age)+.1*age);
  }
  EXPECT_TRUE(std::isnan(terrain_nav::motionIdentityRadius(.05,.2,-.1,1.2)));
  EXPECT_TRUE(std::isnan(terrain_nav::motionIdentityRadius(.05,.2,.1,NAN)));
  EXPECT_TRUE(std::isnan(terrain_nav::motionIdentityRadius(.05,.61,.1,1.2)));
  EXPECT_TRUE(std::isnan(terrain_nav::motionIdentityRadius(NAN,.2,.1,1.2)));
}

TEST(MotionGuard, RecordedOppositeMovingBodiesCannotDonateIdentityThroughLargeCircleFitReserves) {
  using terrain_nav::motionVelocityIdentityCompatible;
  // Same-ID jump reproduced from the COMPLETE recorded masks. A large
  // narrow-arc positional sensitivity reserve must not waive this test.
  EXPECT_FALSE(motionVelocityIdentityCompatible(.0368605553654647,.6929386066979554,.1,
    -.0571809343271365,-.4827797783626918,.1,.199999995,1.2));
  // Correcting each 0.3s-window averaging lag still rejects the recorded
  // opposite-body donation. This is not an arbitrary wider identity gate.
  EXPECT_FALSE(motionVelocityIdentityCompatible(.0368605553654647,.6929386066979554,.28,
    -.0571809343271365,-.4827797783626918,.28,.199999995,1.2));
  // A genuine bounded 90-degree turn remains admissible when enough causal
  // time has elapsed. This does not assert identity uniqueness or free space.
  EXPECT_TRUE(motionVelocityIdentityCompatible(.8,0.,.1,0.,.8,.1,1.,1.2));
  EXPECT_FALSE(motionVelocityIdentityCompatible(.8,0.,.1,0.,.8,.1,.1,1.2));
  EXPECT_FALSE(motionVelocityIdentityCompatible(.8,0.,.1,.8,0.,.1,0.,1.2));
  EXPECT_FALSE(motionVelocityIdentityCompatible(.8,0.,.1,NAN,0.,.1,.1,1.2));
  EXPECT_FALSE(motionVelocityIdentityCompatible(.8,0.,.1,.8,0.,-.1,.1,1.2));
}

namespace {
std::vector<uint8_t> circleReturn(double x,double y,double view=0.,double radius=.45,double span=2.6) {
  std::vector<uint8_t> raw(200*200,0);
  for(int k=0;k<=1200;++k) {
    const double angle=view+span*k/1200.;
    const int ix=int(std::floor((x+radius*std::cos(angle)+5)/.05));
    const int iy=int(std::floor((y+radius*std::sin(angle)+5)/.05));
    if(ix>=0 && iy>=0 && ix<200 && iy<200)raw[iy*200+ix]=1;
  }
  return raw;
}
}

TEST(MotionGuard, ObservationLimitedScopeDropsCoastsButRetainsInvalidSourcesForRejection) {
  terrain_nav::MotionTrackSnapshot current,coast,bad,future;
  current.id=1;current.stamp=10.;coast=current;coast.id=2;coast.stamp=9.9;
  bad=current;bad.id=3;bad.stamp=NAN;future=current;future.id=4;future.stamp=10.1;
  const auto scoped=terrain_nav::currentMotionObservations({current,coast,bad,future},10.);
  ASSERT_EQ(scoped.size(),3u);EXPECT_EQ(scoped[0].id,1u);EXPECT_EQ(scoped[1].id,3u);EXPECT_EQ(scoped[2].id,4u);
  EXPECT_DOUBLE_EQ(coast.stamp,9.9); // No source time or input is refreshed.
  EXPECT_EQ(terrain_nav::currentMotionObservations({current,coast},NAN).size(),2u);
}

TEST(MotionGuard, BothCvAndCaPosteriorsRemainCurrentThroughMeasuredRestAndReversal) {
 for(bool acceleration:{false,true}) {
  terrain_nav::GridMotionGuard guard;guard.acceleration_bound_mps2=1.2;
  guard.fit_gaussian_motion=guard.track_known_slow_centers=true;
  guard.gaussian_constant_acceleration=acceleration;
  guard.turning_identity_reacquisition=true;guard.identity_memory_seconds=5.;
  uint64_t identity=0;size_t retained_rest=0;uint64_t last_identity=0;size_t last_observations=0;
  for(int i=0;i<=42;++i) {
    const double t=i*.1;
    const double x=t<.6?.8*t:t<1.6?.48+.8*(t-.6)-.4*(t-.6)*(t-.6):
      t<2.4?.88:t<3.4?.88-.4*(t-2.4)*(t-2.4):.48-.8*(t-3.4);
    auto raw=circleReturn(x,0.,.006*i);const auto original=raw;
    guard.update(raw,200,200,-5,-5,.05,10.+t);EXPECT_EQ(raw,original);
    for(const auto &track:guard.actionHypotheses(true))if(track.id&&terrain_nav::validGaussianMotion(track.gaussian)) {
      EXPECT_EQ(track.gaussian.constant_acceleration,acceleration);
      if(!identity)identity=track.id;
      if(t>=2.&&t<=2.3) {
        EXPECT_EQ(track.id,identity);EXPECT_DOUBLE_EQ(track.stamp,10.+t);
        EXPECT_EQ(track.gaussian.stamp,track.stamp);++retained_rest;
      }
      last_identity=track.id;last_observations=track.gaussian.observation_count;
    }
  }
  ASSERT_NE(identity,0u);EXPECT_GE(retained_rest,3u);
  EXPECT_EQ(last_identity,identity);EXPECT_GT(last_observations,20u);
  const double latest=14.2;
  guard.update(std::vector<uint8_t>(40000,0),200,200,-5,-5,.05,latest+.1);
  for(const auto &track:guard.actionHypotheses(true)) {
    EXPECT_LE(track.stamp,latest);EXPECT_LE(track.gaussian.stamp,latest);
  }
  EXPECT_TRUE(terrain_nav::currentMotionObservations(guard.actionHypotheses(true),latest+.1).empty());
  terrain_nav::GridMotionGuard stationary;
  stationary.fit_gaussian_motion=stationary.track_known_slow_centers=true;
  stationary.gaussian_constant_acceleration=acceleration;
  for(int i=0;i<20;++i) {
    stationary.update(circleReturn(.88,0.,.006*i),200,200,-5,-5,.05,10.+.1*i);
    EXPECT_TRUE(stationary.actionHypotheses(true).empty());
  }
 }
}

TEST(MotionGuard, RawSurfaceSeparatesChangingNarrowViewFromTranslationAndKeepsSourceOccupancy)
{
  for(bool translating:{false,true}) {
    terrain_nav::GridMotionGuard guard;guard.acceleration_bound_mps2=1.2;
    guard.fit_gaussian_motion=guard.gaussian_constant_acceleration=true;
    guard.track_known_slow_centers=guard.turning_identity_reacquisition=true;
    guard.sim_observation_calibration=true;uint64_t identity=0;
    for(int i=0;i<18;++i) {
      const double t=.1*i,cx=.123+(translating?.8*t:0.),cy=.217;
      std::vector<std::array<double,2>> returns;std::vector<uint8_t> raw(40000,0);
      for(int j=0;j<200;++j) {
        const double angle=M_PI-.7+.12*std::sin(i*.7)+1.4*j/199.;
        const double x=cx+.4*std::cos(angle),y=cy+.4*std::sin(angle);
        returns.push_back({x,y});const int ix=std::floor((x+5)/.05),iy=std::floor((y+5)/.05);
        raw[iy*200+ix]=1;
      }
      const auto original=raw;
      guard.update(raw,200,200,-5,-5,.05,10.+t,1.5,{},&returns);EXPECT_EQ(raw,original);
      const auto circles=guard.measuredCircleObservations();ASSERT_EQ(circles.size(),1u);
      EXPECT_TRUE(circles[0].raw_surface);EXPECT_FALSE(guard.simRepeatabilityDomain(circles[0],.05));
      EXPECT_NEAR(circles[0].center[0],cx,1e-8);EXPECT_NEAR(circles[0].center[1],cy,1e-8);
      if(!translating)EXPECT_TRUE(guard.snapshot().empty());
      else if(i>=4) {
        const auto tracks=guard.snapshot();ASSERT_EQ(tracks.size(),1u)<<i;
        if(!identity)identity=tracks[0].id;
        EXPECT_EQ(tracks[0].id,identity);EXPECT_TRUE(tracks[0].raw_surface_fit);
        EXPECT_TRUE(tracks[0].gaussian.valid);EXPECT_NEAR(tracks[0].vx,.8,1e-8);
        EXPECT_NEAR(tracks[0].gaussian.observation_std,.03,1e-12);
        EXPECT_NEAR(tracks[0].velocity_averaging_lag_s,.15,1e-6);
        EXPECT_GE(tracks[0].velocity_error,.1+1.2*tracks[0].velocity_averaging_lag_s-1e-10);
      }
    }
    if(translating)EXPECT_NE(identity,0u);
  }
}

TEST(MotionGuard, GeometryDegradationCannotOverwriteTheOwnedMeasuredGaussianPosterior)
{
  for(bool acceleration:{false,true})for(bool missing:{false,true}) {
    terrain_nav::GridMotionGuard guard;guard.acceleration_bound_mps2=1.2;
    guard.fit_gaussian_motion=guard.track_known_slow_centers=guard.turning_identity_reacquisition=true;
    guard.gaussian_constant_acceleration=acceleration;guard.identity_memory_seconds=5.;
    uint64_t identity=0;size_t before_count=0,after_count=0;bool degraded_seen=false;
    for(int i=0;i<32;++i) {
      const double t=.1*i,cx=.123+.8*t,cy=.217;
      std::vector<std::array<double,2>> returns;std::vector<uint8_t> raw(40000,0);
      for(int j=0;j<200;++j) {
        const double a=M_PI-.7+1.4*j/199.;
        const double x=cx+.4*std::cos(a),y=cy+.4*std::sin(a);
        returns.push_back({x,y});
        raw[int(std::floor((y+5)/.05))*200+int(std::floor((x+5)/.05))]=1;
      }
      // Raster still exists in one case, but the independent surface source
      // is unavailable. In the other case all object observations are absent.
      const bool degraded=i>=10&&i<=15;
      if(degraded) {returns.clear();if(missing)std::fill(raw.begin(),raw.end(),0);}
      guard.update(raw,200,200,-5,-5,.05,10.+t,1.5,{},&returns);
      for(const auto &track:guard.actionHypotheses(true))if(track.id) {
        if(!identity)identity=track.id;
        EXPECT_EQ(track.id,identity)<<i;
        if(i==9&&track.gaussian.valid)before_count=track.gaussian.observation_count;
        if(degraded) {
          degraded_seen=true;
          // A fresh raster registration is not a fresh Gaussian centre.
          if(track.gaussian.valid)EXPECT_LE(track.gaussian.stamp,10.9);
        }
        if(i==31&&track.gaussian.valid) {
          EXPECT_DOUBLE_EQ(track.gaussian.stamp,13.1);
          after_count=track.gaussian.observation_count;
        }
      }
    }
    EXPECT_NE(identity,0u);EXPECT_GE(before_count,7u);EXPECT_TRUE(degraded_seen);
    EXPECT_GT(after_count,before_count+8u)<<"acceleration="<<acceleration<<" missing="<<missing;
  }
}

TEST(MotionGuard, QualifiedUnownedCentresMayRequestEscapeProposalsButCannotOwnOrClearCells)
{
  terrain_nav::GridMotionGuard guard;guard.acceleration_bound_mps2=1.2;
  guard.fit_gaussian_motion=true;bool tested=false;
  for(int i=0;i<6;++i) {
    const auto raw=circleReturn(.08*i,0.);const auto original=raw;
    guard.update(raw,200,200,-5,-5,.05,10.+.1*i);EXPECT_EQ(raw,original);
    for(auto track:guard.actionHypotheses(true))if(track.center_motion_fit&&track.gaussian.valid) {
      // Hold the actually measured model and source window fixed, vary ONLY
      // label availability. This fixture need not force a particular tracker
      // confirmation schedule; a recorded real ID-zero refusal is audited
      // separately in log/neos_early_candidate_audit_20261116/.
      track.id=0;track.motion_confirmed=false;tested=true;
      EXPECT_TRUE(terrain_nav::currentMeasuredCenterModel(track,10.+.1*i));
      EXPECT_FALSE(terrain_nav::currentMeasuredIdentity(track,10.+.1*i,false));
      EXPECT_TRUE(terrain_nav::measuredMovingCells({track},raw,200,200,-5,-5,.05,10.+.1*i).empty());
      EXPECT_FALSE(terrain_nav::currentMeasuredCenterModel(track,10.+.1*i+.1));
      auto bad=track;bad.center_observations.resize(2);
      EXPECT_FALSE(terrain_nav::currentMeasuredCenterModel(bad,10.+.1*i));
      bad=track;bad.vx+=1.;EXPECT_FALSE(terrain_nav::currentMeasuredCenterModel(bad,10.+.1*i));
      bad=track;bad.gaussian.stamp-=.1;EXPECT_FALSE(terrain_nav::currentMeasuredCenterModel(bad,10.+.1*i));
      bad=track;bad.center_observations[1].center[0]+=1.;
      EXPECT_FALSE(terrain_nav::currentMeasuredCenterModel(bad,10.+.1*i));
    }
  }
  EXPECT_TRUE(tested);
}

TEST(MotionGuard, CurvedBodyDecelerationKeepsIdentityAndNeedsActualStationaryHistory) {
  terrain_nav::GridMotionGuard guard;guard.acceleration_bound_mps2=1.2;
  uint64_t identity=0;bool resolved=false;
  // Smooth braking at 1 m/s^2, followed by observed stationary returns.
  // No missing frame is substituted with a stationary observation.
  for(int i=0;i<=20;++i) {
    const double t=i*.1;
    const double x=t<.6?.8*t:t<1.4?.48+.8*(t-.6)-.5*(t-.6)*(t-.6):.8;
    const auto raw=circleReturn(x,0,.01*i);const auto original=raw;
    guard.update(raw,200,200,-5,-5,.05,10.+t);EXPECT_EQ(raw,original);
    const auto tracks=guard.snapshot();
    if(i>=3 && i<=8) {
      ASSERT_EQ(tracks.size(),1u)<<i;
      if(!identity)identity=tracks[0].id;
      EXPECT_EQ(tracks[0].id,identity)<<i;
    }
    if(i<=13)EXPECT_TRUE(guard.resolvedSnapshot().empty())<<i;
    for(const auto &r:guard.resolvedSnapshot()) {
      EXPECT_EQ(r.id,identity);EXPECT_GE(t,1.4);resolved=true;
    }
  }
  EXPECT_GT(identity,0u);EXPECT_TRUE(resolved);
}

TEST(MotionGuard, KnownMeasuredBodyKeepsItsFreshPosteriorThroughSlowdownAndReversal) {
  terrain_nav::GridMotionGuard guard;guard.acceleration_bound_mps2=1.2;
  guard.fit_gaussian_motion=true;guard.track_known_slow_centers=true;
  guard.turning_identity_reacquisition=true;guard.identity_memory_seconds=5.;
  uint64_t identity=0;int slow_updates=0;
  for(int i=0;i<=25;++i) {
    const double t=.1*i,u=std::clamp(t-.5,0.,1.6);
    const double x=t<.5?.8*t:t<=2.1?.4+.8*u-.5*u*u:.4-.8*(t-2.1);
    const auto raw=circleReturn(x,0,.01*i),original=raw;
    guard.update(raw,200,200,-5,-5,.05,10.+t);EXPECT_EQ(raw,original);
    const auto tracks=guard.snapshot();
    if(i<4)continue;
    ASSERT_EQ(tracks.size(),1u)<<i;
    if(!identity)identity=tracks[0].id;
    EXPECT_EQ(tracks[0].id,identity)<<i;
    EXPECT_DOUBLE_EQ(tracks[0].stamp,10.+t)<<i;
    EXPECT_EQ(tracks[0].gaussian.stamp,tracks[0].stamp);
    EXPECT_TRUE(terrain_nav::validGaussianMotion(tracks[0].gaussian));
    if(i>=11&&i<=16)++slow_updates;
  }
  EXPECT_GT(identity,0u);EXPECT_GE(slow_updates,5);
  // Freshness belongs to actual observations, not to the predicted coast.
  const double last=guard.snapshot()[0].stamp;
  for(int i=1;i<=8;++i) {
    guard.update(std::vector<uint8_t>(40000),200,200,-5,-5,.05,12.5+.1*i);
    for(const auto &t:guard.snapshot())EXPECT_EQ(t.stamp,last);
  }
  EXPECT_TRUE(guard.snapshot().empty());EXPECT_TRUE(guard.resolvedSnapshot().empty());
  terrain_nav::GridMotionGuard stationary;stationary.fit_gaussian_motion=true;
  stationary.track_known_slow_centers=true;
  for(int i=0;i<20;++i) {
    stationary.update(circleReturn(1.,0,.01*i),200,200,-5,-5,.05,10.+.1*i);
    EXPECT_TRUE(stationary.snapshot().empty());
  }
}

TEST(MotionGuard, NearbyStationaryObjectsCannotResolveTwoMovingIdentitiesByOneReturn) {
 for(bool turning:{false,true}) {
  terrain_nav::GridMotionGuard guard;guard.acceleration_bound_mps2=1.2;
  terrain_nav::GridMotionGuard upper,lower;
  upper.acceleration_bound_mps2=lower.acceleration_bound_mps2=1.2;
  for(auto *g:{&guard,&upper,&lower}) {
    g->turning_identity_reacquisition=turning;g->identity_memory_seconds=turning?5.:3.;
  }
  const auto two=[](double x,double y) {
    auto first=circleReturn(x,y,0.,.2,2*M_PI);const auto second=circleReturn(x,-y,0.,.2,2*M_PI);
    for(size_t j=0;j<first.size();++j)first[j]|=second[j];return first;
  };
  for(int i=0;i<6;++i) {
    guard.update(two(.025+.08*i,.275),200,200,-5,-5,.05,10.+.1*i);
    upper.update(circleReturn(.025+.08*i,.275,0.,.2,2*M_PI),200,200,-5,-5,.05,10.+.1*i);
    lower.update(circleReturn(.025+.08*i,-.275,0.,.2,2*M_PI),200,200,-5,-5,.05,10.+.1*i);
  }
  ASSERT_EQ(guard.snapshot().size(),2u);
  // The two equal-radius objects approach the same hypothesized position.
  // One visible body cannot positively identify which old ID became static.
  bool upper_match=false,lower_match=false;
  for(int i=6;i<12;++i) {
    const auto raw=circleReturn(.825,.025,0.,.2,2*M_PI);
    guard.update(raw,200,200,-5,-5,.05,10.+.1*i);
    upper.update(raw,200,200,-5,-5,.05,10.+.1*i);
    lower.update(raw,200,200,-5,-5,.05,10.+.1*i);
    upper_match|=!upper.resolvedSnapshot().empty();lower_match|=!lower.resolvedSnapshot().empty();
    EXPECT_TRUE(guard.resolvedSnapshot().empty())<<i;
  }
  // Both identities individually fit the new gate; only mutual uniqueness
  // prevents the single return from cancelling both in the combined scene.
  EXPECT_TRUE(upper_match);EXPECT_TRUE(lower_match);
 }
}

TEST(MotionGuard, StationaryReturnCannotDonateItsIdentityToAnotherTrackOnLaterScans) {
 for(bool turning:{false,true}) {
  terrain_nav::GridMotionGuard guard,single;guard.acceleration_bound_mps2=1.2;
  guard.turning_identity_reacquisition=turning;guard.identity_memory_seconds=turning?5.:3.;
  single.acceleration_bound_mps2=1.2;single.turning_identity_reacquisition=turning;
  single.identity_memory_seconds=guard.identity_memory_seconds;
  for(int i=0;i<6;++i) {
    auto raw=circleReturn(.08*i,.3,0.,.2);const auto lower=circleReturn(.08*i,-.3,0.,.2);
    single.update(raw,200,200,-5,-5,.05,10.+.1*i);
    for(size_t j=0;j<raw.size();++j)raw[j]|=lower[j];
    guard.update(raw,200,200,-5,-5,.05,10.+.1*i);
  }
  ASSERT_EQ(guard.snapshot().size(),2u);
  std::unordered_set<uint64_t> resolved;bool single_resolved=false;
  // Asymmetric quantized centres: one old gate contains this body one scan
  // earlier than the other gate. It must not resolve both successively.
  for(int i=6;i<(turning?60:14);++i) {
    const auto raw=circleReturn(.8,0,0.,.2);const auto original=raw;
    guard.update(raw,200,200,-5,-5,.05,10.+.1*i);EXPECT_EQ(raw,original);
    single.update(raw,200,200,-5,-5,.05,10.+.1*i);
    single_resolved|=!single.resolvedSnapshot().empty();
    for(const auto &r:guard.resolvedSnapshot())resolved.insert(r.id);
  }
  // The velocity-window averaging reserve removes quantization's false
  // uniqueness. Ambiguous bodies may resolve neither ID, but a single-body
  // positive control must still resolve: this is not a blanket refusal.
  EXPECT_LE(resolved.size(),1u);EXPECT_TRUE(single_resolved);
 }
}

TEST(MotionGuard, ReacquisitionRemembersIdentityButNeverExportsAnExpiredPredictionOrOneWitness) {
  for(int return_frame:{18,39})for(int direction:{1,-1}) {
    terrain_nav::GridMotionGuard guard;guard.acceleration_bound_mps2=1.2;
    for(int i=0;i<5;++i)guard.update(circleReturn(.08*i,0),200,200,-5,-5,.05,10.+.1*i);
    ASSERT_EQ(guard.snapshot().size(),1u);const auto identity=guard.snapshot()[0].id;
    for(int i=5;i<return_frame;++i) {
      guard.update(std::vector<uint8_t>(40000,0),200,200,-5,-5,.05,10.+.1*i);
      EXPECT_TRUE(guard.resolvedSnapshot().empty());
      if(i>=11)EXPECT_TRUE(guard.snapshot().empty())<<i;
    }
    bool reacquired=false,provisional_seen=false;
    for(int i=return_frame;i<return_frame+8;++i) {
      const auto raw=circleReturn(.08*(direction==1?i:2*return_frame-i),0);const auto original=raw;
      guard.update(raw,200,200,-5,-5,.05,10.+.1*i);EXPECT_EQ(raw,original);
      const auto tracks=guard.snapshot();
      for(const auto &t:guard.actionHypotheses(true))
        provisional_seen|=t.id==0&&t.center_motion_fit&&!t.motion_confirmed;
      if(i<return_frame+2)EXPECT_TRUE(tracks.empty())<<i;
      for(const auto &t:tracks) {
        EXPECT_GE(t.stamp,10.+.1*(return_frame+2));
        if(return_frame==18 && direction==1)EXPECT_EQ(t.id,identity);else EXPECT_NE(t.id,identity);
        reacquired=true;
      }
    }
    const bool ambiguous_reversal=return_frame==18&&direction<0;
    EXPECT_EQ(reacquired,!ambiguous_reversal);
    if(ambiguous_reversal)EXPECT_TRUE(provisional_seen); // Current geometry is still checked.
  }
}

TEST(MotionGuard, ReacquisitionCannotChooseBetweenTwoCompatibleDormantIdentities) {
 for(bool turning:{false,true}) {
  terrain_nav::GridMotionGuard guard;guard.acceleration_bound_mps2=1.2;
  guard.turning_identity_reacquisition=turning;guard.identity_memory_seconds=turning?5.:3.;
  for(int i=0;i<5;++i) {
    auto raw=circleReturn(.08*i,.65);const auto other=circleReturn(.08*i,-.65);
    for(size_t j=0;j<raw.size();++j)raw[j]|=other[j];
    guard.update(raw,200,200,-5,-5,.05,10.+.1*i);
  }
  ASSERT_EQ(guard.snapshot().size(),2u);
  for(int i=5;i<18;++i)guard.update(std::vector<uint8_t>(40000,0),200,200,-5,-5,.05,10.+.1*i);
  for(int i=18;i<25;++i) {
    const auto raw=circleReturn(.08*i,0);const auto original=raw;
    guard.update(raw,200,200,-5,-5,.05,10.+.1*i);
    EXPECT_EQ(raw,original);EXPECT_TRUE(guard.snapshot().empty())<<i;
    EXPECT_TRUE(guard.resolvedSnapshot().empty());
  }
 }
}

TEST(MotionGuard, RecordedTurnGapNeedsFreshTurningReassociationNotExpiredPredictionsOrTimeoutRelease) {
  nlohmann::json fixture;std::ifstream(std::string(MOTION_FIXTURE_DIR)+"/recorded_turn_identity_gap.json")>>fixture;
  ASSERT_GT(fixture.at("frames").size(),70u);
  terrain_nav::GridMotionGuard guard;guard.acceleration_bound_mps2=1.2;
  guard.fit_gaussian_motion=true;guard.sim_observation_calibration=true;
  guard.identity_memory_seconds=5.;guard.turning_identity_reacquisition=true;
  uint64_t original_id=0;bool regained=false,resolved=false,expired_prediction_absent=false;
  for(const auto &f:fixture.at("frames")) {
    const auto &g=f.at("geometry");const int w=g[0],h=g[1];const double stamp=f.at("stamp_s");
    std::vector<uint8_t> raw(size_t(w)*h,0);for(int i:f.at("occupied"))raw.at(i)=1;
    const auto unchanged=raw;guard.update(raw,w,h,g[2],g[3],g[4],stamp);EXPECT_EQ(raw,unchanged);
    const auto tracks=guard.snapshot();
    for(const auto &t:tracks) {
      EXPECT_LE(stamp-t.stamp,.600001);
      if(stamp<61.5 && !original_id)original_id=t.id;
      if(stamp>64.5 && t.id==original_id && t.stamp>64.4)regained=true;
    }
    // The first NEW strict reacquisition in this recording is 63.015s.
    // During the actual preceding missing-prediction interval, labels alone
    // must never produce a motion prediction or a stationary resolution.
    if(stamp>62.2 && stamp<62.9) {
      EXPECT_TRUE(tracks.empty());EXPECT_TRUE(guard.resolvedSnapshot().empty());
      expired_prediction_absent=true;
    }
    for(const auto &t:guard.resolvedSnapshot())if(t.id==original_id && t.stamp>66.8)resolved=true;
  }
  EXPECT_GT(original_id,0u);EXPECT_TRUE(expired_prediction_absent);EXPECT_TRUE(regained);EXPECT_TRUE(resolved);
}

TEST(MotionGuard, IdentityMemoryIsBoundedAndDoesNotAcceptInvalidDurations) {
  for(double memory:{std::numeric_limits<double>::quiet_NaN(),.5,6.1}) {
    terrain_nav::GridMotionGuard guard;guard.identity_memory_seconds=memory;
    EXPECT_THROW(guard.update(std::vector<uint8_t>(16),4,4,0,0,.2,10.),std::invalid_argument);
  }
}

TEST(MotionGuard, RecordedRigidIdentityCannotLoseItsObservedExtentWhenVisibleArcChanges) {
  for(const char *name:{"recorded_turn_identity_gap.json","recorded_multi_identity_gap.json"}) {
    nlohmann::json fixture;std::ifstream(std::string(MOTION_FIXTURE_DIR)+"/"+name)>>fixture;
    terrain_nav::GridMotionGuard guard;guard.acceleration_bound_mps2=1.2;
    guard.fit_gaussian_motion=true;guard.sim_observation_calibration=true;
    guard.identity_memory_seconds=5.;guard.turning_identity_reacquisition=true;
    std::unordered_map<uint64_t,double> prior_extent;int checked=0;
    for(const auto &f:fixture.at("frames")) {
      const auto &g=f.at("geometry");const int w=g[0],h=g[1];const double stamp=f.at("stamp_s");
      std::vector<uint8_t> raw(size_t(w)*h,0);for(int i:f.at("occupied"))raw.at(i)=1;
      const auto original=raw;guard.update(raw,w,h,g[2],g[3],g[4],stamp);EXPECT_EQ(raw,original);
      for(const auto &t:guard.snapshot())if(t.id&&t.circular) {
        EXPECT_LE(stamp-t.stamp,.600001);
        if(prior_extent.count(t.id)) {
          EXPECT_GE(t.radius_bound+1e-12,prior_extent[t.id])<<name<<" id="<<t.id<<" stamp="<<stamp;
          ++checked;
        }
        prior_extent[t.id]=t.radius_bound;
      }
    }
    EXPECT_GT(checked,20)<<name;
  }
}

TEST(MotionGuard, SeparateRigidIdentitiesCannotBorrowEachOthersLargerExtent) {
  terrain_nav::GridMotionGuard guard;guard.acceleration_bound_mps2=1.2;
  for(int i=0;i<9;++i) {
    auto raw=circleReturn(.08*i,1.5,0.,.25);
    const auto large=circleReturn(.08*i,-1.5,0.,.65);
    for(size_t j=0;j<raw.size();++j)raw[j]|=large[j];
    guard.update(raw,200,200,-5,-5,.05,10.+.1*i);
  }
  const auto tracks=guard.snapshot();ASSERT_EQ(tracks.size(),2u);
  for(const auto &t:tracks) {
    ASSERT_TRUE(t.circular);
    if(t.center_y>0)EXPECT_LT(t.radius_bound,.4);
    else EXPECT_GT(t.radius_bound,.65);
  }
}

TEST(MotionGuard, UpperReturnsAreTrackingWitnessesOnlyNearExistingLabelsAndNeverInventMissingRays) {
  std::vector<uint8_t> body(400,0);body[0]=1;const auto original=body;
  std::vector<float> ground(400,0.F);
  const std::vector<std::array<float,3>> points={{{.025F,.025F,1.1F}},{{.075F,.025F,1.2F}},
    {{-.425F,-.425F,1.1F}},{{.025F,.025F,3.1F}},{{.025F,.025F,.1F}}};
  const std::vector<terrain_nav::MotionWitnessRegion> regions={{7,0,0,.2,10.}};
  EXPECT_EQ(terrain_nav::motionWitnessMask(body,points,ground,20,20,-.5,-.5,.05,{},2.2,1),body);
  EXPECT_EQ(terrain_nav::motionWitnessMask(body,{},ground,20,20,-.5,-.5,.05,regions,2.2,1),body);
  const auto measured=terrain_nav::motionWitnessMask(body,points,ground,20,20,-.5,-.5,.05,regions,2.2,1);
  EXPECT_EQ(body,original);EXPECT_EQ(measured[0],1);EXPECT_EQ(measured[10*20+10],1);
  EXPECT_EQ(measured[10*20+11],1);EXPECT_EQ(measured[1*20+1],0);
  EXPECT_EQ(std::count(measured.begin(),measured.end(),uint8_t(1)),3);
  ground.assign(400,NAN);
  EXPECT_EQ(terrain_nav::motionWitnessMask(body,points,ground,20,20,-.5,-.5,.05,regions,2.2,1),body);
  EXPECT_THROW(terrain_nav::motionWitnessMask(body,points,ground,20,20,-.5,-.5,.05,regions,NAN,1),std::invalid_argument);
}

TEST(MotionGuard, LabelSearchRegionsCannotRefreshExpiredMotionPredictionsOrResolveAbsence) {
  terrain_nav::GridMotionGuard guard;guard.acceleration_bound_mps2=1.2;
  guard.identity_memory_seconds=5.;guard.turning_identity_reacquisition=true;
  EXPECT_TRUE(guard.trackingWitnessRegions(10.).empty());
  for(int i=0;i<6;++i)guard.update(circleReturn(.08*i,0),200,200,-5,-5,.05,10.+.1*i);
  ASSERT_EQ(guard.snapshot().size(),1u);const auto original=guard.snapshot()[0];
  EXPECT_FALSE(guard.trackingWitnessRegions(10.6).empty());
  for(int i=6;i<25;++i) {
    guard.update(std::vector<uint8_t>(40000,0),200,200,-5,-5,.05,10.+.1*i);
    EXPECT_TRUE(guard.resolvedSnapshot().empty());
    if(i>=12)EXPECT_TRUE(guard.snapshot().empty());
  }
  const auto regions=guard.trackingWitnessRegions(12.5);ASSERT_EQ(regions.size(),1u);
  EXPECT_EQ(regions[0].id,original.id);EXPECT_DOUBLE_EQ(regions[0].source_stamp,original.stamp);
  EXPECT_TRUE(guard.trackingWitnessRegions(16.).empty());
  EXPECT_TRUE(guard.snapshot().empty()); // A nonempty search region was never a motion prediction.
}

TEST(MotionGuard, NewlyMeasuredUpperArcMaintainsIdentityWhileBodyProjectionStaysEmpty) {
  terrain_nav::GridMotionGuard guard;guard.acceleration_bound_mps2=1.2;
  guard.identity_memory_seconds=5.;guard.turning_identity_reacquisition=true;
  uint64_t identity=0;int fresh_upper_scans=0;
  for(int i=0;i<18;++i) {
    const double stamp=10.+.1*i;
    const auto visible=circleReturn(.08*i,0,.01*i);
    auto body=i<6?visible:std::vector<uint8_t>(40000,0);const auto untouched=body;
    std::vector<std::array<float,3>> points;
    for(size_t j=0;j<visible.size();++j)if(visible[j])points.push_back({
      float(-5.+(j%200+.5)*.05),float(-5.+(j/200+.5)*.05),1.2F});
    const auto tracking=terrain_nav::motionWitnessMask(body,points,std::vector<float>(40000,0.F),
      200,200,-5,-5,.05,guard.trackingWitnessRegions(stamp),2.2,1);
    guard.update(tracking,200,200,-5,-5,.05,stamp);
    EXPECT_EQ(body,untouched);
    if(i==5){ASSERT_EQ(guard.snapshot().size(),1u);identity=guard.snapshot()[0].id;}
    if(i>=8) {
      EXPECT_EQ(std::count(body.begin(),body.end(),uint8_t(1)),0);
      ASSERT_EQ(guard.snapshot().size(),1u)<<i;
      const auto track=guard.snapshot()[0];EXPECT_EQ(track.id,identity);
      EXPECT_DOUBLE_EQ(track.stamp,stamp);++fresh_upper_scans;
    }
  }
  EXPECT_GT(fresh_upper_scans,8);
  // Complete absence still expires prediction; labels alone cannot bridge it.
  for(int i=18;i<28;++i)guard.update(std::vector<uint8_t>(40000,0),200,200,-5,-5,.05,10.+.1*i);
  EXPECT_TRUE(guard.snapshot().empty());EXPECT_TRUE(guard.resolvedSnapshot().empty());
}

TEST(MotionGuard, RecordedMultipleBodiesKeepDirectedIdentityAndResolveOnlyOnNewStationaryReturns) {
  nlohmann::json fixture;std::ifstream(std::string(MOTION_FIXTURE_DIR)+"/recorded_multi_identity_gap.json")>>fixture;
  ASSERT_GT(fixture.at("frames").size(),90u);
  terrain_nav::GridMotionGuard guard;guard.acceleration_bound_mps2=1.2;
  guard.fit_gaussian_motion=true;guard.sim_observation_calibration=true;
  guard.identity_memory_seconds=5.;guard.turning_identity_reacquisition=true;
  bool second_reacquired=false;std::unordered_set<uint64_t> resolved;
  for(const auto &f:fixture.at("frames")) {
    const auto &g=f.at("geometry");const int w=g[0],h=g[1];const double stamp=f.at("stamp_s");
    std::vector<uint8_t> raw(size_t(w)*h,0);for(int i:f.at("occupied"))raw.at(i)=1;
    const auto original=raw;guard.update(raw,w,h,g[2],g[3],g[4],stamp);EXPECT_EQ(raw,original);
    for(const auto &t:guard.snapshot()) {
      EXPECT_LE(stamp-t.stamp,.600001);
      if(t.id==2 && t.stamp>70.7 && t.center_y>.8)second_reacquired=true;
    }
    for(const auto &t:guard.resolvedSnapshot())if(t.id==1||t.id==2) {
      EXPECT_GE(t.stamp,t.id==1?72.8:73.3);resolved.insert(t.id);
    }
  }
  EXPECT_TRUE(second_reacquired);EXPECT_EQ(resolved.size(),2u);
}

TEST(MotionGuard, RecordedWeakViewCannotCreateASecondIdentityInsideTheSameBody) {
  terrain_nav::GridMotionGuard guard;guard.acceleration_bound_mps2=1.2;
  uint64_t identity=0;bool fresh_after_weak=false;
  for(int i=0;i<17;++i) {
    const bool weak=i>=5&&i<=8;
    const auto points=weak?recorded_radius_collapse::current():recorded_radius_collapse::prior();
    const double dx=.08*(i-(weak?6:4));std::vector<uint8_t> raw(200*200,0);
    for(const auto &p:points) {
      const int x=int(std::floor((p[0]+dx)/.05)),y=int(std::floor((p[1]+5)/.05));
      ASSERT_GE(x,0);ASSERT_LT(x,200);ASSERT_GE(y,0);ASSERT_LT(y,200);raw[y*200+x]=1;
    }
    const auto original=raw;
    guard.update(raw,200,200,0.,-5.,.05,10.+.1*i);EXPECT_EQ(raw,original);
    const auto tracks=guard.snapshot();
    if(i==4){ASSERT_EQ(tracks.size(),1u);identity=tracks[0].id;}
    for(const auto &t:tracks) {
      if(identity)EXPECT_EQ(t.id,identity)<<i;
      if(i>=11&&std::abs(t.stamp-(10.+.1*i))<1e-8)fresh_after_weak=true;
    }
  }
  EXPECT_GT(identity,0u);EXPECT_TRUE(fresh_after_weak);
}

TEST(DynamicYield, EarlierSafeProgressIsPreferredOverAnAlmostEqualButMuchSlowerEndpoint) {
  terrain_nav::YieldModel model;model.wheelbase=.430;model.rear_axle_offset=.215;
  terrain_nav::YieldState initial;initial.speed=.8;
  const auto fast_commands=terrain_nav::makeAdvanceCommands(initial,model,3.,1.33,0.,6.);
  const auto slow_commands=terrain_nav::makeAdvanceCommands(initial,model,3.,.6,0.,6.);
  ASSERT_FALSE(fast_commands.empty());ASSERT_FALSE(slow_commands.empty());
  const auto fast=terrain_nav::rolloutYield(initial,initial,fast_commands,model,6.);
  const auto slow=terrain_nav::rolloutYield(initial,initial,slow_commands,model,6.);
  EXPECT_NEAR(fast.back().x,slow.back().x,.10);
  EXPECT_GT(terrain_nav::interactionUtility(initial,fast,10,0),terrain_nav::interactionUtility(initial,slow,10,0));
}

TEST(DynamicYield, FiniteAdvanceIncludesWholeStopAndGuardedTailWithoutReverseSizedWaiting) {
  terrain_nav::YieldState initial;initial.speed=1.2;
  terrain_nav::YieldModel model;model.wheelbase=.430;model.rear_axle_offset=.215;
  const auto commands=terrain_nav::makeAdvanceCommands(initial,model,1.8,1.33,0.,6.);
  ASSERT_FALSE(commands.empty());
  const auto full=terrain_nav::rolloutYield(initial,initial,commands,model,6.);
  const auto finite=terrain_nav::finiteAdvanceRollout(full);
  ASSERT_FALSE(finite.empty());
  EXPECT_GE(finite.back().t,3.);EXPECT_LT(finite.back().t,full.back().t);
  EXPECT_DOUBLE_EQ(finite.back().x,full.back().x);EXPECT_DOUBLE_EQ(finite.back().speed,0.);
  double last_moving=0.;for(const auto &s:full)if(std::abs(s.speed)>.001)last_moving=s.t;
  EXPECT_GE(finite.back().t-last_moving,1.2-1e-8);
  EXPECT_TRUE(terrain_nav::finiteAdvanceRollout(full,.6).empty());
  auto incomplete=full;incomplete.resize(30);
  EXPECT_TRUE(terrain_nav::finiteAdvanceRollout(incomplete).empty());
  auto gap=full;gap[1].t=gap[0].t;
  EXPECT_TRUE(terrain_nav::finiteAdvanceRollout(gap).empty());
  auto actual=finite.back();actual.t=finite.back().t-.1;
  const auto continued=terrain_nav::continuationYieldRollout(actual,initial,commands,model,finite.back().t);
  ASSERT_FALSE(continued.empty());
  EXPECT_NEAR(continued.back().t-continued.front().t,3.,1e-8);
  EXPECT_DOUBLE_EQ(continued.back().x,actual.x);EXPECT_DOUBLE_EQ(continued.back().speed,0.);
}
TEST(DynamicYield, ContinuationCannotUndoObservedSteeringOrBrakingBeforePendingCommand) {
  terrain_nav::YieldModel model;model.delay=.35;
  terrain_nav::YieldState initial;initial.speed=1.2;initial.steering=0.;
  auto actual=initial;actual.t=.10;actual.speed=.4;actual.steering=-.15;
  const auto held=terrain_nav::continuationYieldRollout(actual,initial,{{0,0,0}},model,.34);
  ASSERT_GT(held.size(),2u);
  for(const auto &s:held)if(s.t<.34){EXPECT_DOUBLE_EQ(s.speed,.4);EXPECT_DOUBLE_EQ(s.steering,-.15);}
  const auto stopped=terrain_nav::continuationYieldRollout(actual,initial,{{0,0,0}},model,2.);
  EXPECT_DOUBLE_EQ(stopped.back().speed,0.);EXPECT_NEAR(stopped.back().steering,0.,1e-12);
  const auto pending=terrain_nav::rolloutYield(actual,initial,{{0,0,0}},model,.34);
  EXPECT_GT(pending.back().speed,actual.speed); // The explicit pending hypothesis must survive.
}
TEST(DynamicYield, IssuedQueueReplaysPendingAccelerationRatherThanAssumingMeasuredRest) {
  terrain_nav::YieldState root;terrain_nav::YieldModel model;model.delay=.2;model.step=.005;
  const std::vector<terrain_nav::YieldCommand> brake{{0,0,0}};
  const auto held=terrain_nav::rolloutYield(root,root,brake,model,1.);
  const auto queued=terrain_nav::rolloutQueuedYield(root,brake,model,1.,100.,
    {{99.,0,0},{99.81,1.,0}});
  ASSERT_FALSE(queued.empty());EXPECT_DOUBLE_EQ(held.back().x,0.);
  // Accelerate for .19s then symmetrically brake: total distance a*T^2.
  EXPECT_NEAR(queued.back().x,1.2*.19*.19,1e-10);
  EXPECT_DOUBLE_EQ(queued.back().speed,0.);
}
TEST(DynamicYield, QueuedContinuationUsesPublishedStopInsteadOfAnUnsentOldPlan) {
  terrain_nav::YieldState root;root.t=.3;root.speed=.3;root.steering=-.1;
  terrain_nav::YieldModel model;model.delay=.2;model.step=.005;
  const std::vector<terrain_nav::YieldCommand> old_plan{{0,1.,.2},{.5,0.,0.}};
  const auto queued=terrain_nav::rolloutQueuedYield(root,old_plan,model,1.5,100.,
    {{99.,0,0},{100.05,0.,-.1}});
  ASSERT_GT(queued.size(),3u);
  for(const auto&s:queued)if(s.t<.5-1e-9) {
    EXPECT_LE(s.speed,root.speed);EXPECT_NEAR(s.steering,-.1,1e-12);
  }
  // At root+.2, the currently proposed continuation may reach the plant.
  EXPECT_DOUBLE_EQ(queued.back().speed,0.);
  auto held_root=root;held_root.speed=0.;
  const auto idle=terrain_nav::rolloutQueuedYield(held_root,{{0,0,0}},model,1.,100.,{});
  EXPECT_DOUBLE_EQ(idle.back().x,held_root.x); // No invented old plan without history.
}
TEST(DynamicYield, QueuedInputsPreserveSteeringGearChangeAndOffGridEvents) {
  terrain_nav::YieldState root;root.speed=.1;
  terrain_nav::YieldModel model;model.delay=.2;model.step=.02;
  const std::vector<terrain_nav::YieldCommand> history{{99.,0.,0.},{99.813,-.5,.3}};
  const auto queued=terrain_nav::rolloutQueuedYield(root,{{0,0,0}},model,1.,100.,history);
  ASSERT_FALSE(queued.empty());bool event=false,reversed=false,turned=false;
  for(const auto&s:queued) {
    event|=std::abs(s.t-.013)<1e-9;reversed|=s.speed<0.;turned|=s.steering>.2;
  }
  EXPECT_TRUE(event);EXPECT_TRUE(reversed);EXPECT_TRUE(turned);
  EXPECT_DOUBLE_EQ(queued.back().speed,0.);EXPECT_NEAR(queued.back().steering,0.,1e-12);
  EXPECT_TRUE(terrain_nav::rolloutQueuedYield(root,{},model,1.,100.,{{100.,0,0},{99.,0,0}}).empty());
  EXPECT_TRUE(terrain_nav::rolloutQueuedYield(root,{},model,1.,100.,{{99.,NAN,0}}).empty());
  model.step=0.;EXPECT_TRUE(terrain_nav::rolloutQueuedYield(root,{},model,1.,100.,history).empty());
}
TEST(DynamicYield, IssuedHistoryRetainsBoundaryPredecessorAndResetsOnClockRegression) {
  terrain_nav::IssuedYieldHistory history;
  ASSERT_TRUE(history.record({10.,.5,.1}));ASSERT_TRUE(history.record({10.5,0.,0.}));
  ASSERT_TRUE(history.record({13.,-.5,-.1}));
  auto saved=history.snapshot();ASSERT_EQ(saved.size(),2u);EXPECT_DOUBLE_EQ(saved.front().t,10.5);
  EXPECT_FALSE(history.record({NAN,0,0}));EXPECT_EQ(history.snapshot().size(),2u);
  ASSERT_TRUE(history.record({13.,0.,0.}));EXPECT_DOUBLE_EQ(history.snapshot().back().speed,0.);
  ASSERT_TRUE(history.record({2.,.1,0.}));saved=history.snapshot();ASSERT_EQ(saved.size(),1u);
  EXPECT_DOUBLE_EQ(saved.front().t,2.);EXPECT_DOUBLE_EQ(saved.front().speed,.1);
}
TEST(DynamicYield, JointFiniteEscapeRetainsWholeStopRatherThanRejectingLongerSafeEndpoints) {
  terrain_nav::YieldState initial;initial.speed=1.2;terrain_nav::YieldModel model;
  const auto commands=terrain_nav::makeAdvanceCommands(initial,model,4.2,1.33,0.,6.);
  ASSERT_FALSE(commands.empty());const auto full=terrain_nav::rolloutYield(initial,initial,commands,model,6.);
  const auto finite=terrain_nav::finiteJointRollout(full);ASSERT_FALSE(finite.empty());
  EXPECT_GT(finite.back().t,3.);EXPECT_LT(finite.back().t,6.);EXPECT_EQ(finite.back().speed,0.);
  EXPECT_EQ(finite.back().x,full.back().x);ASSERT_GE(finite.size(),11u);
  EXPECT_NEAR(finite[finite.size()-11].speed,0.,.001);
  auto incomplete=full;incomplete.resize(20);EXPECT_TRUE(terrain_nav::finiteJointRollout(incomplete).empty());
  auto invalid=full;invalid[1].t=invalid[0].t;EXPECT_TRUE(terrain_nav::finiteJointRollout(invalid).empty());
}

TEST(MotionGuard, ActualStationaryWallEndpointDoesNotBecomeMovingDueToChangingVisibility) {
  std::ifstream input(std::string(MOTION_FIXTURE_DIR)+"/static_wall_endpoint_parallax.json");
  ASSERT_TRUE(input.good());nlohmann::json data;input>>data;
  terrain_nav::GridMotionGuard guard;
  for(const auto &frame:data.at("frames")) {
    const auto &g=frame.at("geometry");const int w=g[0],h=g[1];
    std::vector<uint8_t> raw(size_t(w)*h,0);for(int cell:frame.at("occupied"))raw.at(cell)=1;
    const auto original=raw;
    guard.update(raw,w,h,g[2],g[3],g[4],frame.at("stamp_s"));
    EXPECT_TRUE(guard.snapshot().empty())<<frame.at("stamp_s");
    EXPECT_EQ(raw,original); // Ambiguity must never delete current occupancy.
  }
}

TEST(MotionGuard, CurvedObjectCenterHistorySeparatesViewChangeFromActualMotion) {
  const auto measured=[](double x,double y,double view) {
    std::vector<uint8_t> raw(160*160,0);
    for(int k=0;k<=1200;++k) {
      const double a=view+2.6*k/1200.;
      const int ix=int(std::floor((x+.55*std::cos(a)+4)/.05));
      const int iy=int(std::floor((y+.55*std::sin(a)+4)/.05));
      if(ix>=0 && iy>=0 && ix<160 && iy<160)raw[iy*160+ix]=1;
    }
    return raw;
  };
  terrain_nav::GridMotionGuard stationary,moving;uint64_t id=0;
  moving.fit_gaussian_motion=true;
  for(int i=0;i<10;++i) {
    const double t=10.+.1*i;
    stationary.update(measured(0,0,.1*i),160,160,-4,-4,.05,t);
    EXPECT_TRUE(stationary.snapshot().empty())<<i;
    EXPECT_TRUE(stationary.provisionalSnapshot().empty())<<i;
    moving.update(measured(0,1.-.08*i,.02*i),160,160,-4,-4,.05,t);
    if(i<3)continue;
    const auto tracks=moving.snapshot();ASSERT_EQ(tracks.size(),1u)<<i;
    const auto &track=tracks.front();if(!id)id=track.id;
    EXPECT_EQ(track.id,id);EXPECT_TRUE(track.circular);
    EXPECT_TRUE(track.motion_confirmed);EXPECT_TRUE(track.center_motion_fit);
    EXPECT_NEAR(track.vx,0.,.08);EXPECT_NEAR(track.vy,-.8,.08);
    EXPECT_GE(track.radius_bound,.55);EXPECT_GT(track.velocity_error,0.);
    ASSERT_TRUE(track.gaussian.valid);
    EXPECT_EQ(track.gaussian.stamp,t);
    EXPECT_DOUBLE_EQ(track.gaussian.process_density,.04);
    if(i==9)EXPECT_GT(track.gaussian.observation_count,3u);
  }
  // Disappearance is not positive stationary evidence and cannot resolve ID.
  moving.update(std::vector<uint8_t>(160*160,0),160,160,-4,-4,.05,11.);
  EXPECT_TRUE(moving.resolvedSnapshot().empty());
  for(const auto &coasted:moving.snapshot()) {
    EXPECT_LT(coasted.gaussian.stamp,11.);
    EXPECT_LT(coasted.stamp,11.);
  }
}

TEST(MotionGuard, RecordedQualifiedCenterCanParticipateBeforeIdentityWithoutChangingRawOccupancy) {
  std::ifstream input(std::string(MOTION_FIXTURE_DIR)+"/qualified_center_before_identity.json");
  ASSERT_TRUE(input.good());nlohmann::json data;input>>data;
  for(bool gaussian:{false,true}) {
  terrain_nav::GridMotionGuard guard;guard.fit_gaussian_motion=gaussian;bool exercised=false,promoted=false;
  const double target=data.at("expected_causal_candidate_source_s");
  for(const auto &frame:data.at("frames")) {
    const auto &g=frame.at("geometry");const int w=g[0],h=g[1];
    std::vector<uint8_t> raw(size_t(w)*h,0);for(int cell:frame.at("occupied"))raw.at(cell)=1;
    const auto original=raw;const double stamp=frame.at("stamp_s");
    guard.update(raw,w,h,g[2],g[3],g[4],stamp);EXPECT_EQ(raw,original);
    if(std::abs(stamp-target)<1e-7) {
      exercised=true;EXPECT_TRUE(guard.snapshot().empty());
      const auto hints=guard.actionHypotheses(true);ASSERT_EQ(hints.size(),1u);
      EXPECT_TRUE(hints[0].center_motion_fit);EXPECT_FALSE(hints[0].motion_confirmed);
      EXPECT_EQ(hints[0].id,0u);EXPECT_GT(hints[0].velocity_error,0.);
      EXPECT_EQ(hints[0].gaussian.valid,gaussian);
      if(gaussian) {
        EXPECT_TRUE(terrain_nav::validGaussianMotion(hints[0].gaussian));
        EXPECT_EQ(hints[0].gaussian.stamp,stamp);
      }
      EXPECT_TRUE(guard.actionHypotheses(false).empty());
      EXPECT_TRUE(guard.brakeOnlyHypotheses(true).empty());
      EXPECT_EQ(guard.brakeOnlyHypotheses(false).size(),1u);
      EXPECT_TRUE(guard.resolvedSnapshot().empty());
    } else if(stamp>target) {
      const auto confirmed=guard.snapshot();ASSERT_EQ(confirmed.size(),1u);
      EXPECT_TRUE(confirmed[0].motion_confirmed);EXPECT_GT(confirmed[0].id,0u);promoted=true;
    }
  }
  EXPECT_TRUE(exercised);EXPECT_TRUE(promoted);
  }
}
TEST(MotionGuard, SimulatorRepeatabilityCalibrationHasExplicitShapeAndResolutionDomain) {
  terrain_nav::CircleObservation c;c.radius=.45;c.observed_span_rad=2.5;
  EXPECT_TRUE(terrain_nav::GridMotionGuard::simRepeatabilityDomain(c,.05));
  c.retained_radius=true;EXPECT_FALSE(terrain_nav::GridMotionGuard::simRepeatabilityDomain(c,.05));
  c.retained_radius=false;
  EXPECT_FALSE(terrain_nav::GridMotionGuard::simRepeatabilityDomain(c,.2));
  c.radius=.8;EXPECT_FALSE(terrain_nav::GridMotionGuard::simRepeatabilityDomain(c,.05));
  c.radius=.45;c.observed_span_rad=1.8;EXPECT_FALSE(terrain_nav::GridMotionGuard::simRepeatabilityDomain(c,.05));
}
TEST(MotionGuard, RepeatabilityCalibrationCannotEditOccupancyVelocityOrPersistentIdentity) {
  std::ifstream input(std::string(MOTION_FIXTURE_DIR)+"/qualified_center_before_identity.json");
  ASSERT_TRUE(input.good());nlohmann::json data;input>>data;
  terrain_nav::GridMotionGuard reference,calibrated;reference.fit_gaussian_motion=calibrated.fit_gaussian_motion=true;
  calibrated.sim_observation_calibration=true;int exercised=0;
  for(const auto &frame:data.at("frames")) {
    const auto &g=frame.at("geometry");const int w=g[0],h=g[1];
    std::vector<uint8_t> raw(size_t(w)*h,0);for(int cell:frame.at("occupied"))raw.at(cell)=1;
    const auto original=raw;const double stamp=frame.at("stamp_s");
    reference.update(raw,w,h,g[2],g[3],g[4],stamp);calibrated.update(raw,w,h,g[2],g[3],g[4],stamp);
    EXPECT_EQ(raw,original);
    const auto a=reference.actionHypotheses(true),b=calibrated.actionHypotheses(true);
    ASSERT_EQ(a.size(),b.size());
    for(size_t i=0;i<a.size();++i) {
      EXPECT_EQ(a[i].cells,b[i].cells);EXPECT_EQ(a[i].vx,b[i].vx);EXPECT_EQ(a[i].vy,b[i].vy);EXPECT_EQ(a[i].id,b[i].id);
      if(b[i].gaussian.valid&&b[i].gaussian.observation_std==.018) {
        ++exercised;EXPECT_LT(b[i].gaussian.covariance.trace(),a[i].gaussian.covariance.trace());
      }
    }
  }
  EXPECT_GT(exercised,0);
}

TEST(DynamicYield, DiskReservationIncludesPhysicalSizeAndDoesNotConfuseSpeedWithSize) {
  using terrain_nav::bodyOverlapsTranslatingDisk;
  EXPECT_TRUE(bodyOverlapsTranslatingDisk(0,0,0,.521,.521,.46,0,3,0,-3,.45));
  EXPECT_FALSE(bodyOverlapsTranslatingDisk(2,0,0,.521,.521,.46,0,3,0,-3,.45));
  EXPECT_TRUE(bodyOverlapsTranslatingDisk(0,0,0,.521,.521,.46,1,0,1,0,.5));
  EXPECT_FALSE(bodyOverlapsTranslatingDisk(0,0,0,.521,.521,.46,1,1,1,1,.5));
  EXPECT_TRUE(bodyOverlapsTranslatingDisk(0,0,0,.521,.521,.46,NAN,0,1,1,.5));
  // Rigidly rotating/translating both bodies cannot change collision.
  for(double yaw:{-.9,0.,1.4}) {
    const double c=std::cos(yaw),s=std::sin(yaw);
    EXPECT_TRUE(bodyOverlapsTranslatingDisk(4,-2,yaw,.521,.521,.46,
      4-3*s,-2+3*c,4+3*s,-2-3*c,.45));
    EXPECT_FALSE(bodyOverlapsTranslatingDisk(4+2*c,-2+2*s,yaw,.521,.521,.46,
      4-3*s,-2+3*c,4+3*s,-2-3*c,.45));
  }
}

TEST(DynamicYield, BypassRecoversHeadingAndUtilityPrefersUsefulCertifiedProgress) {
  terrain_nav::YieldModel model;model.wheelbase=.430;model.rear_axle_offset=.215;
  terrain_nav::YieldState initial;initial.speed=1.;
  const auto commands=terrain_nav::makeBypassCommands(initial,model,3.,1.,.314329,.8,6.);
  ASSERT_FALSE(commands.empty());
  const auto states=terrain_nav::rolloutYield(initial,initial,commands,model,6.);
  EXPECT_NEAR(states.back().speed,0.,1e-9);
  EXPECT_LT(std::abs(states.back().yaw),.12);
  EXPECT_GT(states.back().y,.3);
  const auto reverse_commands=terrain_nav::makeYieldCommands(initial,model,.8,.55,0.,6.);
  ASSERT_FALSE(reverse_commands.empty());
  const auto reverse=terrain_nav::rolloutYield(initial,initial,reverse_commands,model,6.);
  const auto stop=terrain_nav::rolloutYield(initial,initial,{{0,0,0}},model,6.);
  EXPECT_GT(terrain_nav::interactionUtility(initial,states,10,0),
            terrain_nav::interactionUtility(initial,stop,10,0));
  EXPECT_GT(terrain_nav::interactionUtility(initial,stop,10,0),
            terrain_nav::interactionUtility(initial,reverse,10,0));
  for(size_t i=1;i<states.size();++i) {
    EXPECT_LE(std::abs(states[i].steering-states[i-1].steering),model.steering_rate*model.step+1e-8);
    EXPECT_LE(std::abs(states[i].speed-states[i-1].speed),model.acceleration*model.step+1e-8);
  }
}

TEST(DynamicYield, BypassCancelsActualTurningDuringAccelerationDelayAndSteeringSlew) {
  using namespace terrain_nav;
  for(double initial_speed:{0.,.4,1.33})for(double delay:{.04,.20,.35})
    for(double slew:{1.,4.})for(double side:{-1.,1.}) {
      YieldModel model;model.wheelbase=.43;model.rear_axle_offset=.215;
      model.delay=delay;model.steering_rate=slew;model.step=.01;
      YieldState initial;initial.speed=initial_speed;initial.yaw=.4;
      const auto commands=makeBypassCommands(initial,model,4.,1.33,side*.31432913231137,.6,6.);
      ASSERT_GE(commands.size(),4u)<<initial_speed<<' '<<delay<<' '<<slew;
      EXPECT_GT(commands[2].t,commands[1].t);
      const auto states=rolloutYield(initial,initial,commands,model,6.);
      ASSERT_FALSE(states.empty());EXPECT_DOUBLE_EQ(states.back().speed,0.);
      EXPECT_LT(std::abs(std::remainder(states.back().yaw-initial.yaw,2.*M_PI)),.03)
        <<initial_speed<<' '<<delay<<' '<<slew<<' '<<side;
      for(size_t i=1;i<states.size();++i) {
        EXPECT_LE(std::abs(states[i].steering),.31432913231137+1e-12);
        EXPECT_LE(std::abs(states[i].steering-states[i-1].steering),slew*model.step+1e-10);
      }
    }
}

TEST(DynamicYield, RecedingWarmSuffixNeverRestartsTheOldPulseOrItsStoppingDeadline) {
  using namespace terrain_nav;
  const std::vector<YieldCommand> original{{0.,.8,.2},{.7,.8,-.2},{1.4,.8,0.},{2.,0.,0.}};
  for(double elapsed:{0.,.25,.7,1.,1.999,2.,2.5}) {
    const auto suffix=remainingYieldCommands(original,elapsed);ASSERT_FALSE(suffix.empty());
    EXPECT_DOUBLE_EQ(suffix.front().t,0.);
    for(double future:{0.,.01,.2,.6,1.,1.8,3.}) {
      const auto a=yieldCommandAt(original,elapsed+future),b=yieldCommandAt(suffix,future);
      EXPECT_DOUBLE_EQ(a.speed,b.speed);EXPECT_DOUBLE_EQ(a.steering,b.steering);
    }
    if(elapsed<2.)EXPECT_NEAR(elapsed+suffix.back().t,2.,1e-12);
    else {ASSERT_EQ(suffix.size(),1u);EXPECT_DOUBLE_EQ(suffix.front().speed,0.);}
  }
  EXPECT_TRUE(remainingYieldCommands(original,-.1).empty());
  EXPECT_TRUE(remainingYieldCommands(original,NAN).empty());
  EXPECT_TRUE(remainingYieldCommands({{1.,.8,0.},{0.,0.,0.}},.5).empty());
  EXPECT_TRUE(remainingYieldCommands({{0.,NAN,0.}},.5).empty());
}

TEST(MotionGuard, MeasuredRetreatParallaxDoesNotInventNewCrossingIdentities) {
  // Full actual sensor masks, cropped only in source TIME. The former
  // centroid gate invented IDs 2/3/4 while the same actor stayed visible.
  std::ifstream input(std::string(MOTION_FIXTURE_DIR)+"/single_crossing_identity_gap.json");
  ASSERT_TRUE(input.good());nlohmann::json data;input>>data;
  terrain_nav::GridMotionGuard guard;uint64_t identity=0;size_t fresh=0;
  for(const auto &f:data.at("frames")) {
    const auto &g=f.at("geometry");const int width=g.at(0),height=g.at(1);
    std::vector<uint8_t> raw(size_t(width)*height,0);
    for(int i:f.at("occupied"))raw.at(i)=1;
    const double time=f.at("stamp_s");
    guard.update(raw,width,height,g.at(2),g.at(3),g.at(4),time);
    if(time<63.5 || time>70.65)continue;
    const auto tracks=guard.snapshot();ASSERT_EQ(tracks.size(),1u)<<time;
    const auto &track=tracks.front();
    if(!identity)identity=track.id;
    EXPECT_EQ(track.id,identity)<<time;
    EXPECT_GT(track.id,0u);EXPECT_LT(track.vy,-.4);
    EXPECT_GE(time-track.stamp,-1e-8);EXPECT_LE(time-track.stamp,.6);
    if(std::abs(time-track.stamp)<1e-8)++fresh;
    EXPECT_TRUE(guard.resolvedSnapshot().empty());
  }
  EXPECT_GT(fresh,40u);
}

TEST(MotionGuard, MeasuredClosingGateConfirmsStrictTranslationsAcrossAmbiguousScan) {
  terrain_nav::GridMotionGuard guard;
  double first=-1;
  for(const auto &f:measured_fixture::closingGateGap()) {
    std::vector<uint8_t> raw(104*104,0);
    for(int i:f.occupied)raw.at(i)=1;
    guard.update(raw,104,104,f.x,f.y,4*.05000000074505806,f.stamp);
    for(const auto &t:guard.snapshot())
      if(t.stamp>58.8 && t.cells.size()>=5 && first<0) {
        first=t.stamp;
        EXPECT_NEAR(t.vx,0.,1e-6); // Occlusion growth is not longitudinal motion.
        EXPECT_LT(t.vy,-.5);
      }
  }
  ASSERT_GT(first,0.);
  EXPECT_LE(first,58.9151); // Two unique strict registrations; not the first scan.
  EXPECT_GT(first,58.815);
}

namespace {
std::vector<uint8_t> obstacle(int x,int y) {
  std::vector<uint8_t> grid(60*60,0);
  for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++)grid[(y+dy)*60+x+dx]=1;
  return grid;
}

TEST(DynamicYield, MomentumAwareHorizonIncludesReverseAndFinalStop) {
  terrain_nav::YieldModel model;terrain_nav::YieldState initial;initial.speed=1.2;
  const double h=terrain_nav::interactionHorizon(initial,model,.55,1.2);
  EXPECT_GT(h,4.5);EXPECT_LE(h,6.);
  const auto commands=terrain_nav::makeYieldCommands(initial,model,1.2,.55,0.,h);
  ASSERT_FALSE(commands.empty());const auto rollout=terrain_nav::rolloutYield(initial,initial,commands,model,h);
  ASSERT_GT(rollout.size(),20u);EXPECT_NEAR(rollout.back().speed,0.,1e-9);
  EXPECT_NEAR(rollout[rollout.size()-11].speed,0.,1e-9);
  double reverse=0.;for(size_t i=1;i<rollout.size();++i)
    reverse+=std::max(0.,-.5*(rollout[i].speed+rollout[i-1].speed))*(rollout[i].t-rollout[i-1].t);
  EXPECT_GT(reverse,.9);
  model.deceleration=0.;EXPECT_EQ(terrain_nav::interactionHorizon(initial,model,.55,1.2),0.);
}

TEST(MotionGuard, CompactObservedCueDoesNotRequireMotionAndCannotReplaceRawCollision) {
  terrain_nav::GridMotionGuard guard;auto raw=obstacle(20,30);const auto original=raw;
  const auto forecast=guard.update(raw,60,60,0.,0.,.2,10.);
  const auto bounds=guard.compactObservedBounds();
  ASSERT_EQ(bounds.size(),1u);EXPECT_NEAR(bounds[0][0],3.8,1e-9);EXPECT_NEAR(bounds[0][2],4.4,1e-9);
  EXPECT_TRUE(guard.snapshot().empty());EXPECT_EQ(raw,original);
  EXPECT_EQ(std::count(forecast.begin(),forecast.end(),1),0);
  std::fill(raw.begin(),raw.end(),0);for(int y=0;y<60;++y)raw[y*60+20]=1;
  guard.update(raw,60,60,0.,0.,.2,10.1);
  EXPECT_TRUE(guard.compactObservedBounds().empty()); // Full wall is not a compact cue.
  std::fill(raw.begin(),raw.end(),0);guard.update(raw,60,60,0.,0.,.2,10.2);
  EXPECT_TRUE(guard.compactObservedBounds().empty());
}

TEST(DynamicYield, CompactEncounterOnlySlowsApproachAndUsesVehicleHeading) {
  const std::vector<std::array<double,4>> cue={{{3.8,2.5,4.6,3.4}}};
  EXPECT_DOUBLE_EQ(terrain_nav::compactEncounterSpeed(1.2,0,0,0,cue,.55),.55);
  EXPECT_DOUBLE_EQ(terrain_nav::compactEncounterSpeed(.3,0,0,0,cue,.55),.3);
  EXPECT_DOUBLE_EQ(terrain_nav::compactEncounterSpeed(-.5,0,0,0,cue,.55),-.5);
  EXPECT_DOUBLE_EQ(terrain_nav::compactEncounterSpeed(1.2,0,0,M_PI,cue,.55),1.2);
  EXPECT_DOUBLE_EQ(terrain_nav::compactEncounterSpeed(1.2,-5,0,0,cue,.55),1.2);
  EXPECT_DOUBLE_EQ(terrain_nav::compactEncounterSpeed(1.2,0,-5,0,cue,.55),1.2);
  EXPECT_DOUBLE_EQ(terrain_nav::compactEncounterSpeed(1.2,0,0,0,cue,0.),1.2);
  EXPECT_DOUBLE_EQ(terrain_nav::compactEncounterSpeed(1.2,0,0,0,{{{0,0,NAN,0}}},.55),0.);
}

TEST(MotionGuard, PredictionPositionReserveUsesTheActualAdmittedRegistrationResidual) {
  EXPECT_DOUBLE_EQ(terrain_nav::motionAssociationRadius(.05),1.5*.05);
  EXPECT_DOUBLE_EQ(terrain_nav::motionAssociationRadius(.20),1.5*.20);
  EXPECT_TRUE(std::isinf(terrain_nav::motionAssociationRadius(0.)));
  EXPECT_TRUE(std::isinf(terrain_nav::motionAssociationRadius(NAN)));
  EXPECT_DOUBLE_EQ(terrain_nav::motionAssociationRadius(.05,.5),.2);
  EXPECT_TRUE(std::isinf(terrain_nav::motionAssociationRadius(.05,.7)));
  EXPECT_TRUE(std::isinf(terrain_nav::motionAssociationRadius(.05,-.1)));
}

TEST(MotionGuard, MeasuredVelocityQuantizationCannotSplitIdentityInsideExistingCoastReserve) {
  std::ifstream input(std::string(MOTION_FIXTURE_DIR)+"/retreat_velocity_quantization_gap.json");
  ASSERT_TRUE(input.good());nlohmann::json data;input>>data;
  terrain_nav::GridMotionGuard guard;uint64_t identity=0;size_t fresh=0;
  for(const auto &f:data.at("frames")) {
    const auto &g=f.at("geometry");const int width=g.at(0),height=g.at(1);
    std::vector<uint8_t> raw(size_t(width)*height,0);for(int i:f.at("occupied"))raw.at(i)=1;
    const double time=f.at("stamp_s");guard.update(raw,width,height,g.at(2),g.at(3),g.at(4),time);
    if(time<73.1 || time>80.3)continue;
    const auto tracks=guard.snapshot();ASSERT_EQ(tracks.size(),1u)<<time;
    const auto &track=tracks.front();if(!identity)identity=track.id;
    EXPECT_EQ(track.id,identity)<<time;EXPECT_GT(identity,0u);
    EXPECT_LE(time-track.stamp,.6);EXPECT_LT(track.vy,-.4);
    if(std::abs(time-track.stamp)<1e-8)++fresh;
    EXPECT_TRUE(guard.resolvedSnapshot().empty());
  }
  EXPECT_GT(fresh,40u);
}

TEST(DynamicYield, LegalAccelerationCrossingIsDifferentFromStoppingOrSlowing) {
  terrain_nav::YieldModel model;terrain_nav::YieldState initial;initial.speed=.8;
  const auto commands=terrain_nav::makeAdvanceCommands(initial,model,3.,1.33,0.,5.);
  ASSERT_FALSE(commands.empty());const auto fast=terrain_nav::rolloutYield(initial,initial,commands,model,5.);
  const auto slow_commands=terrain_nav::makeAdvanceCommands(initial,model,1.,.6,0.,5.);
  ASSERT_FALSE(slow_commands.empty());const auto slow=terrain_nav::rolloutYield(initial,initial,slow_commands,model,5.);
  EXPECT_GT(fast[75].x,slow[75].x+.3);EXPECT_NEAR(fast.back().speed,0.,1e-9);
  for(size_t i=1;i<fast.size();++i) {
    EXPECT_LE(fast[i].speed,1.33+1e-9);
    EXPECT_LE(std::abs(fast[i].speed-fast[i-1].speed),model.acceleration*model.step+1e-9);
  }
}

TEST(DynamicYield, FutureDepartureCannotEraseAnObstacleThatMayStopInTheLane) {
  using terrain_nav::bodyOverlapsTranslatingCell;
  // Constant-speed endpoint has already left; a possible STOP at y=0 has not.
  EXPECT_FALSE(bodyOverlapsTranslatingCell(0,0,0,.521,.521,.46,0,-3,0,-3,.2));
  EXPECT_TRUE(bodyOverlapsTranslatingCell(0,0,0,.521,.521,.46,0,3,0,-3,.2));
  // Acceleration can clear its x corridor before lateral arrival.
  EXPECT_FALSE(bodyOverlapsTranslatingCell(2,0,0,.521,.521,.46,0,3,0,-3,.2));
  // Likewise a genuinely clear rear footprint is not blocked by a radial cap.
  EXPECT_FALSE(bodyOverlapsTranslatingCell(-2,0,0,.521,.521,.46,0,3,0,-3,.2));
  EXPECT_TRUE(bodyOverlapsTranslatingCell(0,0,0,.521,.521,.46,.621,0,.621,2,.2));
}

TEST(DynamicYield, ContinuousSegmentCollisionMatchesIndependentDensePoseSampling) {
  using terrain_nav::bodyOverlapsTranslatingCell;
  // Independent polygon test: transform vehicle corners and use edge
  // normals of both actual polygons. Dense samples only supply positive
  // witnesses; a very short continuous contact need not land on a sample.
  for(int k=0;k<60;++k) {
    const double yaw=k*.103,ax=-2.+(k%7)*.43,ay=-1.5+(k%11)*.27;
    const double bx=1.8-(k%5)*.53,by=1.7-(k%9)*.37,side=.05+(k%4)*.07;
    const double c=std::cos(yaw),s=std::sin(yaw);
    const std::array<std::array<double,2>,4> body={{{.7*c-.4*s,.7*s+.4*c},
      {.7*c+.4*s,.7*s-.4*c},{-.3*c+.4*s,-.3*s-.4*c},{-.3*c-.4*s,-.3*s+.4*c}}};
    const auto overlap=[&](double px,double py) {
      const std::array<std::array<double,2>,4> cell={{{px-side/2,py-side/2},{px+side/2,py-side/2},
        {px+side/2,py+side/2},{px-side/2,py+side/2}}};
      for(const auto &polygon:{body,cell})for(size_t j=0;j<4;++j) {
        const auto a=polygon[j],b=polygon[(j+1)%4];const double nx=b[1]-a[1],ny=a[0]-b[0];
        double a0=INFINITY,a1=-INFINITY,b0=INFINITY,b1=-INFINITY;
        for(const auto &p:body){const double v=p[0]*nx+p[1]*ny;a0=std::min(a0,v);a1=std::max(a1,v);}
        for(const auto &p:cell){const double v=p[0]*nx+p[1]*ny;b0=std::min(b0,v);b1=std::max(b1,v);}
        if(a1<b0 || b1<a0)return false;
      }
      return true;
    };
    bool witnessed=false;for(int i=0;i<=2000;++i) {
      const double f=i/2000.;witnessed|=overlap(ax+f*(bx-ax),ay+f*(by-ay));
    }
    const bool continuous=bodyOverlapsTranslatingCell(0,0,yaw,.7,.3,.4,ax,ay,bx,by,side);
    if(witnessed)EXPECT_TRUE(continuous)<<k;
    EXPECT_EQ(continuous,bodyOverlapsTranslatingCell(0,0,yaw,.7,.3,.4,bx,by,ax,ay,side));
    EXPECT_EQ(overlap(ax,ay),bodyOverlapsTranslatingCell(0,0,yaw,.7,.3,.4,ax,ay,ax,ay,side));
  }
}

TEST(MotionGuard, CoarseRegistrationMustNotMergeMeasuredWallAndObjectAcrossAFineGap) {
  std::vector<uint8_t> fine(24*24,0);
  for(int y=0;y<20;++y)fine[y*24+8]=1;
  for(int y=10;y<15;++y)for(int x=11;x<16;++x)fine[y*24+x]=1;
  const auto original=fine;const auto grouped=terrain_nav::aggregateMotionComponents(fine,24,24,4);
  ASSERT_EQ(grouped.components.size(),2u);EXPECT_EQ(fine,original);
  for(size_t i=0;i<fine.size();++i)if(fine[i])EXPECT_TRUE(grouped.occupied[(i/24/4)*6+(i%24/4)]);
  // The shared coarse column is NOT evidence of measured connectivity.
  bool shared=false;for(int i:grouped.components[0])
    shared|=std::binary_search(grouped.components[1].begin(),grouped.components[1].end(),i);
  EXPECT_TRUE(shared);
  EXPECT_THROW(terrain_nav::aggregateMotionComponents(fine,24,24,5),std::invalid_argument);
  EXPECT_THROW(terrain_nav::aggregateMotionComponents(fine,23,24,4),std::invalid_argument);
}

TEST(DynamicYield, ImmediateReverseStillTravelsForwardThroughLatencyAndBraking) {
  terrain_nav::YieldState start;start.speed=.78;
  terrain_nav::YieldModel model;
  const auto commands=terrain_nav::makeYieldCommands(start,model,.4,.55,.4,3);
  ASSERT_FALSE(commands.empty());EXPECT_LT(commands.front().speed,0);
  const auto states=terrain_nav::rolloutYield(start,start,commands,model,3);
  EXPECT_GT(states[10].x,.14);EXPECT_NEAR(states[10].speed,.78,1e-9);
  const auto first_reverse=std::find_if(states.begin(),states.end(),[](const auto &s){return s.speed<0;});
  ASSERT_NE(first_reverse,states.end());EXPECT_GT(first_reverse->t,.99);
  EXPECT_GT(first_reverse->x,.5);
  for(size_t i=1;i<states.size();++i) {
    EXPECT_LE(std::abs(states[i].steering-states[i-1].steering),model.steering_rate*model.step+1e-8);
    EXPECT_LE(std::abs(states[i].speed-states[i-1].speed),model.acceleration*model.step+1e-8);
  }
}

TEST(DynamicYield, ForwardAvoidanceRespectsLatencySteeringRateAndComesToRest) {
  terrain_nav::YieldState start;start.speed=.7;
  terrain_nav::YieldModel model;
  for(double steering:{-.4,0.,.4}) {
    const auto commands=terrain_nav::makeAdvanceCommands(start,model,1.,.6,steering,3.);
    ASSERT_EQ(commands.size(),2u);EXPECT_GT(commands.front().speed,0);
    EXPECT_EQ(commands.back().speed,0);
    const auto states=terrain_nav::rolloutYield(start,start,commands,model,3.);
    EXPECT_NEAR(states[10].speed,start.speed,1e-9);
    EXPECT_NEAR(states.back().speed,0.,1e-9);
    EXPECT_GT(states.back().x,.6);EXPECT_LT(std::hypot(states.back().x,states.back().y),1.1);
    if(steering==0)EXPECT_NEAR(states.back().yaw,0.,1e-9);
    else EXPECT_GT(states.back().yaw*steering,0.);
    for(size_t i=1;i<states.size();++i) {
      EXPECT_LE(std::abs(states[i].steering-states[i-1].steering),model.steering_rate*model.step+1e-8);
      EXPECT_GE(states[i].speed,-1e-9);
    }
  }
  EXPECT_TRUE(terrain_nav::makeAdvanceCommands(start,model,1.,.6,0.,.5).empty());
  start.speed=-.4;
  const auto reversing=terrain_nav::makeAdvanceCommands(start,model,1.,.6,0.,3.);
  ASSERT_FALSE(reversing.empty());
  const auto reversal=terrain_nav::rolloutYield(start,start,reversing,model,3.);
  EXPECT_LT(reversal[1].speed,0.);EXPECT_LT(reversal[1].x,0.);
  EXPECT_GT(reversal.back().x,0.);EXPECT_NEAR(reversal.back().speed,0.,1e-9);
}

TEST(MotionGuard, ExportedWorldTracksAreIndependentOfLaterUpdates) {
  terrain_nav::GridMotionGuard model;
  model.update(obstacle(40,48),60,60,-6,-6,.2,.9);
  model.update(obstacle(40,46),60,60,-6,-6,.2,1);
  model.update(obstacle(40,44),60,60,-6,-6,.2,1.1);
  const auto snapshot=model.snapshot();ASSERT_EQ(snapshot.size(),1u);
  EXPECT_NEAR(snapshot[0].vy,-4,1e-9);EXPECT_EQ(snapshot[0].cells.size(),9u);
  model.update(obstacle(40,44),60,60,-6,-6,.2,1.2);
  EXPECT_TRUE(model.snapshot().empty());EXPECT_EQ(snapshot[0].cells.size(),9u);
}
}

TEST(MotionGuard, StaticComponentsDoNotBecomePredictedTerrain) {
  terrain_nav::GridMotionGuard model;
  model.update(obstacle(30,45),60,60,-6,-6,.2,1);
  auto mask=model.update(obstacle(30,45),60,60,-6,-6,.2,1.1);
  EXPECT_EQ(model.moving_components,0);
  EXPECT_EQ(std::count(mask.begin(),mask.end(),1),0);
}
TEST(MotionGuard, IdentityPersistsWhileMovingAndExpiryIsNotStationaryResolution) {
  terrain_nav::GridMotionGuard model;
  model.update(obstacle(40,48),60,60,-6,-6,.2,1.);
  model.update(obstacle(40,46),60,60,-6,-6,.2,1.1);
  model.update(obstacle(40,44),60,60,-6,-6,.2,1.2);
  ASSERT_EQ(model.snapshot().size(),1u);const auto id=model.snapshot()[0].id;
  ASSERT_GT(id,0u);
  model.update(obstacle(40,42),60,60,-6,-6,.2,1.3);
  ASSERT_EQ(model.snapshot().size(),1u);EXPECT_EQ(model.snapshot()[0].id,id);
  for(double time:{1.4,1.5,1.6,1.7,1.8,1.9,2.01}) {
    model.update(std::vector<uint8_t>(3600,0),60,60,-6,-6,.2,time);
    EXPECT_TRUE(model.resolvedSnapshot().empty());
  }
  EXPECT_TRUE(model.snapshot().empty());
}
TEST(MotionGuard, PositiveStationaryShapeResolvesOnlyThePreviouslyObservedIdentity) {
  terrain_nav::GridMotionGuard model;
  model.update(obstacle(40,48),60,60,-6,-6,.2,1.);
  model.update(obstacle(40,46),60,60,-6,-6,.2,1.1);
  model.update(obstacle(40,44),60,60,-6,-6,.2,1.2);
  ASSERT_EQ(model.snapshot().size(),1u);const auto id=model.snapshot()[0].id;
  model.update(obstacle(40,44),60,60,-6,-6,.2,1.3);
  EXPECT_TRUE(model.resolvedSnapshot().empty()); // One repeated scan is insufficient.
  model.update(obstacle(40,44),60,60,-6,-6,.2,1.4);
  model.update(obstacle(40,44),60,60,-6,-6,.2,1.5);
  ASSERT_EQ(model.resolvedSnapshot().size(),1u);
  EXPECT_EQ(model.resolvedSnapshot()[0].id,id);
  EXPECT_NEAR(model.resolvedSnapshot()[0].stamp,1.5,1e-9);
  EXPECT_TRUE(model.snapshot().empty());
}
TEST(MotionGuard, EqualIntervalEndpointsDoNotProveStationaryWhenIntermediateScansMove) {
  terrain_nav::GridMotionGuard model;
  model.update(obstacle(40,48),60,60,-6,-6,.2,1.);
  model.update(obstacle(40,46),60,60,-6,-6,.2,1.1);
  model.update(obstacle(40,44),60,60,-6,-6,.2,1.2);
  model.update(obstacle(40,43),60,60,-6,-6,.2,1.3);
  model.update(obstacle(40,44),60,60,-6,-6,.2,1.4);
  model.update(obstacle(40,44),60,60,-6,-6,.2,1.5);
  EXPECT_TRUE(model.resolvedSnapshot().empty());
}
TEST(MotionGuard, CrossingIsForecastBeforeItsCurrentCellsReachTheLane) {
  terrain_nav::GridMotionGuard model;
  model.update(obstacle(40,48),60,60,-6,-6,.2,.9);
  model.update(obstacle(40,46),60,60,-6,-6,.2,1);
  auto mask=model.update(obstacle(40,44),60,60,-6,-6,.2,1.1);
  ASSERT_EQ(model.moving_components,1);
  EXPECT_TRUE(mask[30*60+40]);
  EXPECT_FALSE(mask[30*60+30]);
  mask=model.update(std::vector<uint8_t>(3600,0),60,60,-6,-6,.2,1.2);
  EXPECT_TRUE(mask[30*60+40]);
  for(double t:{1.3,1.4,1.5,1.6,1.71})
    mask=model.update(std::vector<uint8_t>(3600,0),60,60,-6,-6,.2,t);
  EXPECT_EQ(std::count(mask.begin(),mask.end(),1),0);
}

TEST(MotionGuard, SubcellPerFrameCrossingHasConfirmedVelocityRatherThanEndlessProvisionalBraking) {
  terrain_nav::GridMotionGuard model;
  int first_confirmed=-1;double best_error=10.;
  for(int frame=0;frame<30;++frame) {
    std::vector<uint8_t> raw(3600,0);
    const double cy=2.7-.08*frame;
    for(int y=0;y<60;++y)for(int x=0;x<60;++x)
      if(std::hypot(-6+(x+.5)*.2-2.,-6+(y+.5)*.2-cy)<=.45)raw[y*60+x]=1;
    model.update(raw,60,60,-6,-6,.2,1.+frame*.1);
    for(const auto &track:model.snapshot()) {
      if(first_confirmed<0)first_confirmed=frame;
      best_error=std::min(best_error,std::hypot(track.vx,track.vy+.8));
    }
  }
  ASSERT_GE(first_confirmed,2);EXPECT_LE(first_confirmed,6);
  EXPECT_LT(best_error,.25);
}
TEST(MotionGuard, RollingOriginDoesNotMakeStaticObstacleMove) {
  terrain_nav::GridMotionGuard model;
  model.update(obstacle(40,40),60,60,-6,-6,.2,1);
  auto mask=model.update(obstacle(39,40),60,60,-5.8,-6,.2,1.1);
  EXPECT_EQ(std::count(mask.begin(),mask.end(),1),0);
}
TEST(MotionGuard, TimeResetAndUnobservedGapDoNotReuseVelocity) {
  terrain_nav::GridMotionGuard model;
  model.update(obstacle(40,46),60,60,-6,-6,.2,5);
  auto mask=model.update(obstacle(40,44),60,60,-6,-6,.2,1);
  EXPECT_EQ(std::count(mask.begin(),mask.end(),1),0);
  mask=model.update(obstacle(40,42),60,60,-6,-6,.2,2);
  EXPECT_EQ(std::count(mask.begin(),mask.end(),1),0);
}
TEST(MotionGuard, ChangedVisibleSubsetOfStaticWallIsNotVelocityEvidence) {
  terrain_nav::GridMotionGuard model;
  std::vector<uint8_t> first(3600,0),second(3600,0);
  for(int y=40;y<=42;++y)for(int x=35;x<=44;++x) {
    first[y*60+x]=1;if(x>=39)second[y*60+x]=1;
  }
  model.update(first,60,60,-6,-6,.2,1);
  auto mask=model.update(second,60,60,-6,-6,.2,1.1);
  EXPECT_EQ(model.moving_components,0);
  EXPECT_EQ(std::count(mask.begin(),mask.end(),1),0);
}

TEST(MotionGuard, WholeScanStationaryEvidenceSurvivesComponentSplits) {
  const auto row=[](std::initializer_list<int> xs) {
    std::vector<uint8_t> mask(3600,0);for(int x:xs)mask[35*60+x]=1;return mask;
  };
  terrain_nav::GridMotionGuard model;
  model.update(row({25,26,27,29,30,31}),60,60,-6,-6,.2,1.);
  model.update(row({27,28,29,30}),60,60,-6,-6,.2,1.1);
  const auto raw=row({26,27,28,29}),original=raw;
  const auto forecast=model.update(raw,60,60,-6,-6,.2,1.2);
  EXPECT_TRUE(model.snapshot().empty());EXPECT_EQ(raw,original);
  EXPECT_EQ(std::count(forecast.begin(),forecast.end(),1),0);
}

TEST(MotionGuard, FixedWorldLineExplainsTangentialVisibilityWithoutInventedVelocity) {
  const auto fragment=[](int x) {
    std::vector<uint8_t> mask(3600,0);
    for(int c:{10,12,14,44,46,48,x,x+1})mask[45*60+c]=1;
    return mask;
  };
  terrain_nav::GridMotionGuard model;
  for(int i=0;i<4;++i) {
    const auto raw=fragment(25+i),original=raw;
    const auto forecast=model.update(raw,60,60,-6,-6,.2,1.+i*.1);
    EXPECT_TRUE(model.snapshot().empty());EXPECT_TRUE(model.provisionalSnapshot().empty());
    EXPECT_EQ(std::count(forecast.begin(),forecast.end(),1),0);EXPECT_EQ(raw,original);
  }
}

TEST(MotionGuard, ThinObjectWithVisibleEndpointsStillHasObservableTranslation) {
  terrain_nav::GridMotionGuard model;
  for(int i=0;i<3;++i) {
    std::vector<uint8_t> raw(3600,0);raw[30*60+25+i]=raw[30*60+26+i]=1;
    model.update(raw,60,60,-6,-6,.2,1.+i*.1);
  }
  ASSERT_EQ(model.snapshot().size(),1u);EXPECT_NEAR(model.snapshot()[0].vx,2.,1e-9);
}

TEST(MotionGuard, TwoFineCellsKeepCurrentOccupancyAndCautionWithoutInventingPersistentIdentity) {
  terrain_nav::GridMotionGuard guard;bool warned=false;
  for(int i=0;i<7;++i) {
    std::vector<uint8_t> raw(40000,0);raw[(130-2*i)*200+110]=raw[(131-2*i)*200+110]=1;
    const auto original=raw;guard.update(raw,200,200,-5,-5,.05,10.+.1*i);
    EXPECT_EQ(raw,original);EXPECT_TRUE(guard.snapshot().empty());
    for(const auto &t:guard.brakeOnlyHypotheses(true)) {
      EXPECT_EQ(t.id,0u);EXPECT_FALSE(t.motion_confirmed);warned=true;
    }
    EXPECT_TRUE(guard.resolvedSnapshot().empty());
  }
  EXPECT_TRUE(warned);
}

TEST(MotionGuard, CompleteRecordedSceneCannotPromoteSparseWallJitterToUnresolvedMotionIdentity) {
  nlohmann::json fixture;std::ifstream(std::string(MOTION_FIXTURE_DIR)+"/recorded_sparse_wall_identity.json")>>fixture;
  ASSERT_GE(fixture.at("frames").size(),50u);
  terrain_nav::GridMotionGuard guard;guard.acceleration_bound_mps2=1.2;
  guard.fit_gaussian_motion=true;guard.sim_observation_calibration=true;
  guard.identity_memory_seconds=5.;guard.turning_identity_reacquisition=true;
  bool sparse_cue=false;int confirmed=0;
  for(const auto &f:fixture.at("frames")) {
    const auto &g=f.at("geometry");const int w=g[0],h=g[1];const double stamp=f.at("stamp_s");
    const double resolution=g[4];std::vector<uint8_t> raw(size_t(w)*h,0);
    for(int i:f.at("occupied"))raw.at(i)=1;
    const auto original=raw;guard.update(raw,w,h,g[2],g[3],resolution,stamp);EXPECT_EQ(raw,original);
    for(const auto &t:guard.snapshot()) {
      if(!t.center_motion_fit)EXPECT_GE(t.cells.size()*resolution*resolution,.01-1e-12)<<stamp;
      EXPECT_LE(stamp-t.stamp,.600001);++confirmed;
    }
    for(const auto &t:guard.brakeOnlyHypotheses(true))
      sparse_cue|=t.cells.size()*resolution*resolution<.01-1e-12;
  }
  EXPECT_TRUE(sparse_cue);EXPECT_GT(confirmed,10); // Genuine measured actors remain.
}

TEST(MotionGuard, RestEvidenceRequiresLongDistinctMeasuredCentersAndRejectsMotionOrGaps) {
  std::vector<terrain_nav::TimedCenter> rest;
  for(int i=0;i<=10;++i)rest.push_back({10.+.1*i,{{1.+.01*std::sin(i),2.}},.45,.05});
  EXPECT_TRUE(terrain_nav::measuredStationaryCenters(rest));
  auto short_history=rest;short_history.resize(7);
  EXPECT_FALSE(terrain_nav::measuredStationaryCenters(short_history));
  auto moving=rest;for(size_t i=0;i<moving.size();++i)moving[i].center[0]=1.+.008*i;
  EXPECT_FALSE(terrain_nav::measuredStationaryCenters(moving)); // 8cm/s is not classified as rest.
  auto gap=rest;gap[6].stamp_s=gap[5].stamp_s;
  EXPECT_FALSE(terrain_nav::measuredStationaryCenters(gap));
  gap=rest;gap.erase(gap.begin()+4,gap.begin()+6);
  EXPECT_FALSE(terrain_nav::measuredStationaryCenters(gap));
  auto shape=rest;shape.back().radius=.7;
  EXPECT_FALSE(terrain_nav::measuredStationaryCenters(shape));
  shape=rest;shape.back().center[0]=NAN;
  EXPECT_FALSE(terrain_nav::measuredStationaryCenters(shape));
}

TEST(MotionGuard, RecordedSlowdownCannotSilentlyDeleteItsConfirmedConflictWithoutPositiveResolution) {
  nlohmann::json fixture;std::ifstream(std::string(MOTION_FIXTURE_DIR)+"/confirmed_stationary_transition.json")>>fixture;
  ASSERT_GT(fixture.at("tracking_frames").size(),200u);
  terrain_nav::GridMotionGuard guard;guard.acceleration_bound_mps2=1.2;
  guard.fit_gaussian_motion=guard.sim_observation_calibration=true;
  guard.identity_memory_seconds=5.;guard.turning_identity_reacquisition=guard.track_known_slow_centers=true;
  std::map<uint64_t,terrain_nav::MotionTrackSnapshot> previous;
  std::set<uint64_t> already_resolved;
  bool tracked_target=false,resolved_target=false;double resolution_time=INFINITY;
  std::array<terrain_nav::MotionResolutionMailbox,3> mailboxes;
  std::array<bool,3> delivered{{false,false,false}};
  size_t fresh_rest_posteriors=0,frame_index=0;
  for(const auto&f:fixture.at("tracking_frames")) {
    const auto&g=f.at("geometry");int w=g[0],h=g[1];double stamp=f.at("stamp_s");
    std::vector<uint8_t> raw(size_t(w)*h,0);for(int i:f.at("occupied"))raw.at(i)=1;
    const auto original=raw;guard.update(raw,w,h,g[2],g[3],g[4],stamp);EXPECT_EQ(raw,original);
    std::map<uint64_t,terrain_nav::MotionTrackSnapshot> current;
    const auto actions=guard.actionHypotheses(true);
    for(const auto&t:guard.snapshot()) {
      current[t.id]=t;tracked_target|=std::abs(t.center_x-7.33)<.2&&t.center_motion_fit;
    }
    const auto resolutions=guard.resolvedSnapshot();
    for(size_t phase=0;phase<mailboxes.size();++phase) {
      mailboxes[phase].observe(stamp,stamp+.14,1,g[4],actions,resolutions);
      if(frame_index%3!=phase)continue; // A slower consumer must not miss a one-scan event.
      for(const auto &event:mailboxes[phase].snapshot(stamp+.14))
        if(std::hypot(event.center_x-7.33,event.center_y-2.68)<.2) {
          delivered[phase]=true;
          EXPECT_LE(event.stamp,stamp);EXPECT_LE(stamp+.14-event.stamp,.6);
        }
    }
    for(const auto&t:resolutions)if(std::hypot(t.center_x-7.33,t.center_y-2.68)<.2) {
      resolved_target=true;resolution_time=std::min(resolution_time,stamp);
      EXPECT_TRUE(previous.count(t.id)||already_resolved.count(t.id));EXPECT_DOUBLE_EQ(t.stamp,stamp);
      already_resolved.insert(t.id);
      if(t.center_motion_fit&&terrain_nav::validGaussianMotion(t.gaussian)) {
        const auto current_posterior=std::find_if(actions.begin(),actions.end(),[&](const auto&a){return a.id==t.id;});
        ASSERT_NE(current_posterior,actions.end());
        EXPECT_DOUBLE_EQ(current_posterior->stamp,stamp);
        EXPECT_DOUBLE_EQ(current_posterior->gaussian.stamp,stamp);
        ++fresh_rest_posteriors;
      }
    }
    for(const auto&[id,t]:previous)if(!current.count(id)&&stamp-t.stamp<=.600001)
      EXPECT_TRUE(std::any_of(resolutions.begin(),resolutions.end(),[&](const auto&r){return r.id==id;}))<<stamp<<" id="<<id;
    previous=std::move(current);
    ++frame_index;
  }
  EXPECT_TRUE(tracked_target);EXPECT_TRUE(resolved_target);EXPECT_LT(resolution_time,81.214998184);
  for(bool received:delivered)EXPECT_TRUE(received);
  // Continuing measured rest now produces a fresh posterior/event on each
  // observed frame. All consumer phases must receive that actual evidence;
  // the separate mailbox test checks retention when no newer event exists.
  EXPECT_GE(fresh_rest_posteriors,3u);
}

TEST(MotionGuard, PositiveResolutionDeliveryPreservesSourceTimeAndCannotOutliveItsEvidence) {
  terrain_nav::MotionResolutionMailbox mailbox;
  terrain_nav::MotionTrackSnapshot event;event.id=7;event.stamp=10.1;event.resolution=.05;
  event.cells={{{1.,2.}}};event.vx=event.vy=0.;
  mailbox.observe(10.1,10.24,1,.05,{}, {event});
  mailbox.observe(10.2,10.34,1,.05,{}, {});
  auto out=mailbox.snapshot(10.35);ASSERT_EQ(out.size(),1u);
  EXPECT_DOUBLE_EQ(out[0].stamp,10.1);
  EXPECT_TRUE(terrain_nav::freshStationaryResolution(out[0],7,10.,10.35));
  EXPECT_FALSE(terrain_nav::freshStationaryResolution(out[0],7,10.1,10.35));
  EXPECT_FALSE(terrain_nav::freshStationaryResolution(out[0],8,10.,10.35));
  EXPECT_FALSE(terrain_nav::freshStationaryResolution(out[0],7,10.,10.71));
  mailbox.observe(10.4,10.54,1,.05,{}, {event}); // Old bytes cannot be relabelled as a new event.
  EXPECT_TRUE(mailbox.snapshot(10.71).empty());
  mailbox.observe(10.1,10.24,2,.05,{}, {event}); // Explicit new epoch.
  auto moving=event;moving.stamp=10.2;moving.vx=.5;
  const auto old=moving.cells;mailbox.observe(10.2,10.34,2,.05,{moving},{});
  EXPECT_TRUE(mailbox.snapshot(10.35).empty());EXPECT_EQ(moving.cells,old);
  mailbox.observe(10.3,10.44,2,.05,{}, {moving}); // A moving object is not a static resolution.
  EXPECT_TRUE(mailbox.snapshot(10.45).empty());
}

TEST(MotionGuard, CurrentStationaryPosteriorDoesNotSuppressItsExplicitSameSourceRestEvent) {
  terrain_nav::MotionResolutionMailbox mailbox;
  terrain_nav::MotionTrackSnapshot event;event.id=7;event.stamp=10.1;event.resolution=.05;
  event.cells={{{1.,2.}}};event.motion_confirmed=true;
  auto posterior=event;posterior.center_motion_fit=true;posterior.gaussian.valid=true;
  mailbox.observe(10.1,10.24,1,.05,{posterior},{event});
  auto result=mailbox.snapshot(10.25);ASSERT_EQ(result.size(),1u);
  EXPECT_DOUBLE_EQ(result[0].stamp,10.1);
  // Redelivery of that exact observation does not renew or remove it.
  mailbox.observe(10.1,10.3,1,.05,{posterior},{event});
  ASSERT_EQ(mailbox.snapshot(10.3).size(),1u);
  EXPECT_TRUE(mailbox.snapshot(10.71).empty());
  // A newer same-ID measurement revokes the event even without a new event.
  posterior.stamp=10.2;
  mailbox.observe(10.2,10.34,1,.05,{posterior},{});
  EXPECT_TRUE(mailbox.snapshot(10.35).empty());
  for(int fault=0;fault<2;++fault) {
    mailbox.reset();posterior=event;
    if(fault==0)posterior.vx=.1;else posterior.cells={{{1.2,2.}}};
    mailbox.observe(10.1,10.24,1,.05,{posterior},{event});
    EXPECT_TRUE(mailbox.snapshot(10.25).empty());
  }
}

TEST(MotionGuard, ResolutionDeliveryFailsClosedOnClockGeometryFutureAndResourceDiscontinuities) {
  terrain_nav::MotionResolutionMailbox mailbox;
  terrain_nav::MotionTrackSnapshot event;event.id=1;event.stamp=10.;event.resolution=.05;event.cells={{{1,2}}};
  mailbox.observe(10.,10.14,1,.05,{}, {event});ASSERT_EQ(mailbox.snapshot(10.2).size(),1u);
  mailbox.observe(10.1,10.24,1,.1,{}, {});EXPECT_TRUE(mailbox.snapshot(10.25).empty());
  mailbox.observe(10.,10.14,1,.05,{}, {event});
  mailbox.observe(9.9,10.04,1,.05,{}, {});EXPECT_TRUE(mailbox.snapshot(10.05).empty());
  mailbox.observe(10.1,10.24,1,.05,{}, {event});EXPECT_TRUE(mailbox.snapshot(10.25).empty());
  event.stamp=10.4;mailbox.observe(10.2,10.34,1,.05,{}, {event});
  EXPECT_TRUE(mailbox.snapshot(10.35).empty());
  event.stamp=10.3;event.motion_confirmed=false;mailbox.observe(10.3,10.44,1,.05,{}, {event});
  EXPECT_TRUE(mailbox.snapshot(10.45).empty());
  std::vector<terrain_nav::MotionTrackSnapshot> overflow(1025,event);
  for(size_t i=0;i<overflow.size();++i){overflow[i].id=i+1;overflow[i].stamp=10.4;overflow[i].motion_confirmed=true;}
  mailbox.observe(10.4,10.54,1,.05,{},overflow);EXPECT_TRUE(mailbox.snapshot(10.55).empty());
}

TEST(MotionGuard, AdjacentRasterRowsDoNotGiveLongWallFragmentsNormalVelocity) {
  // Three changing subsets generate mutually consistent diagonal centroid
  // velocities, despite every return belonging to the same thin world wall.
  // Rotate the complete recording so the test does not rely on horizontal X.
  for(int rotation=0;rotation<4;++rotation) {
    terrain_nav::GridMotionGuard model;
    for(int frame=0;frame<3;++frame) {
      std::vector<uint8_t> raw(3600,0);
      const auto put=[&](int x,int y) {
        for(int k=0;k<rotation;++k){const int old=x;x=59-y;y=old;}
        raw[y*60+x]=1;
      };
      for(int x:{8,10,12,14,16,18,20,35,37,39,41,43,45,47,49}) {
        put(x,29+(frame==2));if(frame==1)put(x,30);
      }
      if(frame==0)for(int x=23;x<=27;++x)put(x,29);
      if(frame==1){put(26,29);put(27,29);put(25,30);put(27,30);put(28,30);}
      if(frame==2)for(int x=26;x<=29;++x)put(x,30);
      const auto original=raw;
      const auto forecast=model.update(raw,60,60,-6,-6,.2,1.+frame*.1);
      EXPECT_TRUE(model.snapshot().empty());EXPECT_TRUE(model.provisionalSnapshot().empty());
      EXPECT_EQ(std::count(forecast.begin(),forecast.end(),1),0);EXPECT_EQ(raw,original);
    }
  }
}

TEST(MotionGuard, TranslationNormalToSupportingWallRemainsObservable) {
  terrain_nav::GridMotionGuard model;
  for(int frame=0;frame<3;++frame) {
    std::vector<uint8_t> raw(3600,0);
    for(int x:{10,12,14,16,18,40,42,44,46,48})raw[40*60+x]=1;
    for(int x=27;x<31;++x)raw[(40-2*frame)*60+x]=1;
    model.update(raw,60,60,-6,-6,.2,1.+frame*.1);
  }
  ASSERT_EQ(model.snapshot().size(),1u);EXPECT_NEAR(model.snapshot()[0].vy,-4.,1e-9);
}

TEST(MotionGuard, AmbiguousWallContextDoesNotCancelAnAlreadyConfirmedCoast) {
  terrain_nav::GridMotionGuard model;
  for(int frame=0;frame<4;++frame) {
    std::vector<uint8_t> raw(3600,0);
    for(int x=23+2*frame;x<27+2*frame;++x)raw[(29+(frame==3))*60+x]=1;
    if(frame>=2)for(int x:{8,10,12,14,16,18,20,38,40,42,44,46,48}) {
      raw[29*60+x]=1;raw[30*60+x]=1;
    }
    model.update(raw,60,60,-6,-6,.2,1.+frame*.1);
    if(frame>=2) {
      ASSERT_EQ(model.snapshot().size(),1u);
      EXPECT_NEAR(model.snapshot()[0].stamp,1.2,1e-9);
      EXPECT_NEAR(model.snapshot()[0].vx,4.,1e-9);
    }
  }
}
TEST(MotionGuard, MovingThinBoxFaceKeepsItsEndpointTranslationEvidence) {
  terrain_nav::GridMotionGuard model;
  for(int i=0;i<3;++i) {
    std::vector<uint8_t> raw(3600,0);for(int y=42-i;y<46-i;++y)raw[y*60+40]=1;
    model.update(raw,60,60,-6,-6,.2,1.+i*.1);
  }
  ASSERT_EQ(model.snapshot().size(),1u);EXPECT_NEAR(model.snapshot()[0].vy,-2.,1e-9);
}

TEST(MotionGuard, TruncatedThinFragmentCannotEstablishVelocityButRawOccupancySurvives) {
  terrain_nav::GridMotionGuard model;
  for(int i=0;i<3;++i) {
    std::vector<uint8_t> raw(3600,0);raw[30*60+55+i]=raw[30*60+56+i]=1;
    const auto original=raw;
    const auto forecast=model.update(raw,60,60,-6,-6,.2,1.+i*.1);
    EXPECT_TRUE(model.snapshot().empty());EXPECT_TRUE(model.provisionalSnapshot().empty());
    EXPECT_EQ(raw,original);EXPECT_EQ(std::count(forecast.begin(),forecast.end(),1),0);
  }
}

TEST(MotionGuard, RealStationaryObservationCancelsVelocityWithoutWaitingForExpiry) {
  terrain_nav::GridMotionGuard model;
  model.update(obstacle(40,48),60,60,-6,-6,.2,.9);
  model.update(obstacle(40,46),60,60,-6,-6,.2,1);
  model.update(obstacle(40,44),60,60,-6,-6,.2,1.1);
  auto mask=model.update(obstacle(40,44),60,60,-6,-6,.2,1.2);
  EXPECT_EQ(model.moving_components,0);
  EXPECT_EQ(std::count(mask.begin(),mask.end(),1),0);
}

TEST(MotionGuard, CoastingUsesWorldGeometryWhenGridRollsAndDoesNotPaintOldSweep) {
  terrain_nav::GridMotionGuard model;
  model.update(obstacle(40,48),60,60,-6,-6,.2,.9);
  model.update(obstacle(40,46),60,60,-6,-6,.2,1);
  model.update(obstacle(40,44),60,60,-6,-6,.2,1.1);
  auto mask=model.update(std::vector<uint8_t>(3600,0),60,60,-5.6,-6,.2,1.3);
  EXPECT_TRUE(mask[30*60+38]);
  EXPECT_FALSE(mask[46*60+38]);
  EXPECT_FALSE(mask[30*60+40+3]);
}

TEST(MotionGuard, TimeResetAlsoInvalidatesCoastingTracks) {
  terrain_nav::GridMotionGuard model;
  model.update(obstacle(40,48),60,60,-6,-6,.2,4.9);
  model.update(obstacle(40,46),60,60,-6,-6,.2,5);
  model.update(obstacle(40,44),60,60,-6,-6,.2,5.1);
  auto mask=model.update(std::vector<uint8_t>(3600,0),60,60,-6,-6,.2,1);
  EXPECT_EQ(std::count(mask.begin(),mask.end(),1),0);
}

TEST(MotionGuard, OneTranslationOrReversingVisibleSubsetCannotConfirmMotion) {
  terrain_nav::GridMotionGuard model;
  auto raw=obstacle(40,46);const auto original=raw;
  model.update(obstacle(40,48),60,60,-6,-6,.2,1.);
  auto mask=model.update(raw,60,60,-6,-6,.2,1.1);
  EXPECT_EQ(model.moving_components,0);EXPECT_TRUE(model.snapshot().empty());
  ASSERT_EQ(model.provisionalSnapshot().size(),1u);
  EXPECT_EQ(std::count(mask.begin(),mask.end(),1),0);EXPECT_EQ(raw,original);
  mask=model.update(obstacle(40,48),60,60,-6,-6,.2,1.2);
  EXPECT_TRUE(model.snapshot().empty());
  mask=model.update(obstacle(40,50),60,60,-6,-6,.2,1.3);
  EXPECT_EQ(model.snapshot().size(),1u);
  EXPECT_TRUE(model.provisionalSnapshot().empty());
}

TEST(MotionGuard, ConfirmationCannotBridgeUnobservedGap) {
  terrain_nav::GridMotionGuard model;
  model.update(obstacle(40,48),60,60,-6,-6,.2,1.);
  model.update(obstacle(40,46),60,60,-6,-6,.2,1.1);
  model.update(obstacle(40,44),60,60,-6,-6,.2,1.8);
  model.update(obstacle(40,42),60,60,-6,-6,.2,1.9);
  EXPECT_TRUE(model.snapshot().empty());
}

TEST(MotionGuard, PartialCellTranslationOnlyOffersFreshBrakingEvidence) {
  const auto wide=[](int x) {
    std::vector<uint8_t> mask(3600,0);
    for(int y=40;y<45;++y)for(int c=x;c<x+5;++c)mask[y*60+c]=1;
    return mask;
  };
  terrain_nav::GridMotionGuard model;
  model.update(wide(30),60,60,-6,-6,.2,1.);
  auto mask=model.update(wide(31),60,60,-6,-6,.2,1.1);
  EXPECT_EQ(std::count(mask.begin(),mask.end(),1),0);
  EXPECT_TRUE(model.snapshot().empty());
  ASSERT_EQ(model.provisionalSnapshot().size(),1u);
  EXPECT_NEAR(model.provisionalSnapshot().front().vx,2.,1e-9);
  mask=model.update(wide(32),60,60,-6,-6,.2,1.2);
  EXPECT_TRUE(model.snapshot().empty());
  EXPECT_EQ(std::count(mask.begin(),mask.end(),1),0);
  EXPECT_EQ(model.provisionalSnapshot().size(),1u);
  model.update(wide(32),60,60,-6,-6,.2,1.3);
  EXPECT_TRUE(model.provisionalSnapshot().empty());
  model.update(wide(33),60,60,-6,-6,.2,1.4);
  ASSERT_FALSE(model.provisionalSnapshot().empty());
  model.update(wide(34),60,60,-6,-6,.2,2.);
  EXPECT_TRUE(model.provisionalSnapshot().empty());
}

TEST(MotionGuard, SparseLatticeRegistrationCanBrakeBeforeCentroidRoundsToOneCell) {
  const auto raster=[](std::initializer_list<std::array<int,2>> cells) {
    std::vector<uint8_t> mask(3600,0);
    for(const auto &p:cells)mask[(30+p[1])*60+30+p[0]]=1;
    return mask;
  };
  const auto before=raster({{0,-2},{0,-1},{1,-1},{6,-1},{1,0},{2,0},{4,0},{5,0},{2,1},{3,1},{4,1}});
  const auto after=raster({{0,-2},{0,-1},{0,0},{1,0},{6,0},{1,1},{2,1},{3,1},{4,1},{5,1}});
  terrain_nav::GridMotionGuard model;
  model.update(before,60,60,-6,-6,.2,1.);
  const auto forecast=model.update(after,60,60,-6,-6,.2,1.1);
  EXPECT_TRUE(model.snapshot().empty());
  EXPECT_EQ(std::count(forecast.begin(),forecast.end(),1),0);
  ASSERT_EQ(model.provisionalSnapshot().size(),1u);
  EXPECT_NEAR(model.provisionalSnapshot().front().vx,0.,1e-9);
  EXPECT_NEAR(model.provisionalSnapshot().front().vy,2.,1e-9);
  model.update(after,60,60,-6,-6,.2,1.2);
  EXPECT_TRUE(model.provisionalSnapshot().empty());
}
