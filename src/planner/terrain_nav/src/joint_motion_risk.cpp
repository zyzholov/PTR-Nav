#include "terrain_nav/joint_motion_risk.hpp"
#include <boost/math/special_functions/beta.hpp>
#include <chrono>

namespace terrain_nav {
namespace {
bool packGaussianRoot(const GaussianMotion &m,joint_detail::Actor &a)
{
  if(m.constant_acceleration) {
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,6,6>> solver(m.acceleration_covariance);
    if(solver.info()!=Eigen::Success||solver.eigenvalues().minCoeff()< -1e-10)return false;
    const Eigen::Matrix<double,6,6> root=solver.eigenvectors()*solver.eigenvalues().cwiseMax(0.).cwiseSqrt().asDiagonal();
    a.constant_acceleration=1;
    for(int i=0;i<6;++i){a.acceleration_mean[i]=m.acceleration_mean[i];for(int j=0;j<6;++j)a.acceleration_root[6*i+j]=root(i,j);}
    for(int i=0;i<4;++i)a.mean[i]=m.mean[i];
  } else {
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> solver(m.covariance);
    if(solver.info()!=Eigen::Success||solver.eigenvalues().minCoeff()< -1e-10)return false;
    const Eigen::Matrix4d root=solver.eigenvectors()*solver.eigenvalues().cwiseMax(0.).cwiseSqrt().asDiagonal();
    for(int i=0;i<4;++i){a.mean[i]=m.mean[i];for(int j=0;j<4;++j)a.root[4*i+j]=root(i,j);}
  }
  return true;
}
}
#ifdef TERRAIN_NAV_MOTION_CUDA
bool jointMotionCudaAvailable(std::string &error);
bool jointMotionCudaHits(const std::vector<joint_detail::Actor> &actors,const std::vector<joint_detail::Pose> &poses,
  size_t samples,uint64_t seed,std::vector<unsigned> &hits,std::string &error);
bool jointMotionCudaBatchHits(const std::vector<joint_detail::Actor> &,const std::vector<joint_detail::Pose> &,
  const std::vector<int> &,size_t,uint64_t,std::vector<unsigned> &,std::string &);
bool jointMotionCudaSharedBatchHits(const std::vector<joint_detail::Actor> &,const std::vector<joint_detail::Pose> &,
  const std::vector<int> &,const std::vector<double> &,const std::vector<int> &,
  size_t,uint64_t,std::vector<unsigned> &,std::string &);
#endif
double binomialCollisionUpper(size_t trials,size_t hits,double alpha)
{
  if(!trials||hits>trials||!std::isfinite(alpha)||alpha<=0||alpha>=1)return 1.;
  if(hits==trials)return 1.;
  if(hits==0)return -std::expm1(std::log(alpha)/double(trials));
  // Invert the upper tail directly; 1-alpha loses precision for small alpha.
  return boost::math::ibetac_inv(double(hits+1),double(trials-hits),alpha);
}
double binomialCollisionLower(size_t trials,size_t hits,double alpha)
{
  if(!trials||hits>trials||!std::isfinite(alpha)||alpha<=0||alpha>=1)return NAN;
  if(!hits)return 0.;
  if(hits==trials)return std::exp(std::log(alpha)/double(trials));
  return boost::math::ibeta_inv(double(hits),double(trials-hits+1),alpha);
}
bool jointMotionGpuAvailable(std::string *error)
{
  std::string e;
#ifdef TERRAIN_NAV_MOTION_CUDA
  const bool available=jointMotionCudaAvailable(e);
#else
  const bool available=false;e="CUDA joint-motion backend not built";
#endif
  if(error)*error=e;return available;
}
JointMotionRisk jointGaussianMotionRisk(const std::vector<YieldState> &states,double origin,
  const std::vector<MotionTrackSnapshot> &tracks,const MotionCheckGeometry &g,
  size_t samples,double alpha,uint64_t seed,bool gpu,double preference_clearance,bool analytical_discharge)
{
  JointMotionRisk out;out.backend=gpu?"cuda":"cpu";out.alpha=alpha;out.samples=samples;
  const auto start=std::chrono::steady_clock::now();
  const auto done=[&]() {out.elapsed_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();return out;};
  const auto fail=[&](const char *e){out.error=e;return done();};
  if(states.empty()||states.size()>2048||!std::isfinite(origin)||samples<32||samples>65536||
     !std::isfinite(alpha)||alpha<=0||alpha>=1||!std::isfinite(g.front)||g.front<0||
     !std::isfinite(g.back)||g.back<0||!std::isfinite(g.half_width)||g.half_width<0||
     !std::isfinite(g.wheelbase)||g.wheelbase<=0||!std::isfinite(g.max_steering)||g.max_steering<0||g.max_steering>=1.57||
     !std::isfinite(g.extra_inflation)||g.extra_inflation<0||!std::isfinite(preference_clearance)||preference_clearance<0)
    return fail("invalid joint-risk request");
  std::vector<joint_detail::Pose> poses;poses.reserve(states.size());
  const double half_gap=motionSampleHalfGap(states);if(!std::isfinite(half_gap))return fail("invalid temporal coverage");
  const double base=origin+states.front().t;double last=-INFINITY;
  for(const auto &s:states) {
    const double t=s.t-states.front().t;
    if(!std::isfinite(t)||t<last||t>10||!std::isfinite(s.x)||!std::isfinite(s.y)||!std::isfinite(s.yaw)||
       !std::isfinite(s.speed))return fail("invalid/noncausal trajectory");
    last=t;
    const double sweep=half_gap*(4.5+std::abs(s.speed)*(1+std::hypot(std::max(g.front,g.back),g.half_width)*std::tan(g.max_steering)/g.wheelbase));
    poses.push_back({t,s.x,s.y,std::cos(s.yaw),std::sin(s.yaw),g.front,g.back,g.half_width,sweep});
  }
  std::vector<joint_detail::Actor> actors;
  std::vector<const MotionTrackSnapshot *> gaussian_sources;
  for(const auto &t:tracks) {
    if(!t.circular||!t.center_motion_fit||!validGaussianMotion(t.gaussian)) {
      ++out.reachable_objects;
      const auto fallback=assessTimedMotion(states,origin,{t},g,0.,preference_clearance);
      if(!fallback.valid)return fail("invalid reachable fallback evidence");
      if(fallback.unsafe) {
        out.deterministic_conflict=true;
        return fail("reachable fallback collision");
      }
      continue;
    }
    const double age=base-t.stamp;
    if(!std::isfinite(age)||age<0||age>.6)return fail("invalid/stale Gaussian source age");
    if(t.gaussian.stamp!=t.stamp)return fail("Gaussian posterior/source stamp mismatch");
    if(!std::isfinite(t.resolution)||t.resolution<=0||
       t.cells.empty()||!std::isfinite(t.radius_bound)||t.radius_bound<=0||
       !std::isfinite(t.acceleration_bound)||t.acceleration_bound<0)return fail("invalid Gaussian shape/acceleration source");
    if(!gaussianCircleEnclosesMeasurements(t))return fail("Gaussian shape does not enclose measured cells");
    joint_detail::Actor a{};a.age=age;a.density=t.gaussian.process_density;
    a.acceleration=t.gaussian.longitudinal_acceleration_bound;a.maximum_speed=predictionSpeedCap(t.gaussian.mean.tail<2>().norm(),t.velocity_error,t.prediction_maximum_speed);
    if(!std::isfinite(a.maximum_speed)||a.maximum_speed<t.gaussian.mean.tail<2>().norm())return fail("invalid acceleration speed domain");
    a.radius=t.radius_bound+g.extra_inflation+preference_clearance;
    if(!packGaussianRoot(t.gaussian,a))return fail("non-PSD covariance");
    actors.push_back(a);
    gaussian_sources.push_back(&t);
  }
  out.gaussian_objects=actors.size();out.collisions.assign(actors.size(),0);
  out.sample_counts.assign(actors.size(),samples);
  std::vector<double> analytical(actors.size(),INFINITY);
  const double confidence_floor=actors.empty()?0.:binomialCollisionUpper(samples,0,alpha/actors.size());
  for(size_t i=0;i<actors.size();++i) {
    const auto bound=timedGaussianMotionRisk(states,origin,{*gaussian_sources[i]},g,preference_clearance);
    if(bound.valid)analytical[i]=bound.upper;
    // CP(k,N,alpha) >= CP(0,N,alpha) for EVERY possible sample outcome.
    // If A <= CP(0), min(A,CP(k)) == A independently of the samples. Skip
    // only that provably redundant work. Keep actor slots/stream indices so
    // all actually sampled bodies retain EXACTLY the same random streams.
    if(analytical_discharge&&analytical[i]<=confidence_floor) {
      actors[i].analytical_only=1;out.sample_counts[i]=0;++out.analytical_objects;
    }
  }
  if(actors.size()>out.analytical_objects) {
    if(gpu) {
#ifdef TERRAIN_NAV_MOTION_CUDA
      if(!jointMotionCudaHits(actors,poses,samples,seed,out.collisions,out.error))return done();
#else
      return fail("CUDA joint-motion backend not built");
#endif
    } else {
      for(size_t object=0;object<actors.size();++object)if(!actors[object].analytical_only)
        for(size_t sample=0;sample<samples;++sample)
          out.collisions[object]+=joint_detail::trajectoryHit(actors[object],poses.data(),int(poses.size()),seed,sample,object);
    }
  }
  // Samples preserve temporal correlation WITHIN each actor. Use a union
  // bound ACROSS actor events, so no actor-independence assumption is needed.
  // Split sampling failure probability as well. No risk cost is discounted.
  out.upper=out.estimate=0.;
  for(size_t i=0;i<out.collisions.size();++i) {
    const unsigned hits=out.collisions[i];
    double upper=binomialCollisionUpper(samples,hits,alpha/actors.size());
    // A remote object must not acquire an artificial collision-probability
    // floor just because a finite sample cannot prove probability zero.
    // The deterministic halfspace/temporal union bound covers the SAME
    // Gaussian event and expanded poses. Its minimum with the independently
    // sampled upper bound is still an upper bound, with the same alpha.
    // Keep the object union (no independence assumption) and all samples.
    // Bounded mean acceleration uses a swept-segment enclosing rectangle
    // (or a departure disk at rest), never the old point-Gaussian bound.
    upper=std::min(upper,analytical[i]);
    out.upper=std::min(1.,out.upper+upper);
    // For an analytically discharged component, this is its analytic upper
    // contribution, not an invented Monte Carlo hit fraction. This score is
    // used only to decide whether another complete proof may be worthwhile.
    out.estimate=std::min(1.,out.estimate+(actors[i].analytical_only?analytical[i]:double(hits)/samples));
  }
  out.valid=true;return done();
}
JointMotionScreen screenJointMotionBatch(const std::vector<std::vector<YieldState>> &candidates,double origin,
  const std::vector<MotionTrackSnapshot> &tracks,const MotionCheckGeometry &g,
  size_t samples,uint64_t seed,bool gpu,double discard_error)
{
  JointMotionScreen out;const auto start=std::chrono::steady_clock::now();
  const auto finish=[&] {out.elapsed_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();return out;};
  if(candidates.empty()||candidates.size()>512||samples<32||samples>65536||!std::isfinite(origin)||
     !std::isfinite(discard_error)||discard_error<=0||discard_error>=1||
     !std::isfinite(g.front)||g.front<0||!std::isfinite(g.back)||g.back<0||!std::isfinite(g.half_width)||g.half_width<0||
     !std::isfinite(g.wheelbase)||g.wheelbase<=0||!std::isfinite(g.max_steering)||g.max_steering<0||g.max_steering>=1.57||
     !std::isfinite(g.extra_inflation)||g.extra_inflation<0) {out.error="invalid screening request";return finish();}
  out.estimates.assign(candidates.size(),0.);
  out.lower_bounds.assign(candidates.size(),0.);
  if(candidates.front().empty()){out.error="empty trajectory";return finish();}
  const double first=candidates.front().front().t,base=origin+first;
  std::vector<joint_detail::Pose> poses;std::vector<int> offsets{0};
  for(const auto &states:candidates) {
    if(states.empty()||states.size()>2048||states.front().t!=first) {out.error="inconsistent batch time origin";return finish();}
    double last=-INFINITY;
    const double half_gap=motionSampleHalfGap(states);if(!std::isfinite(half_gap)){out.error="invalid temporal coverage";return finish();}
    for(const auto &s:states) {
      const double t=s.t-first;
      if(!std::isfinite(t)||t<last||t>10||!std::isfinite(s.x)||!std::isfinite(s.y)||!std::isfinite(s.yaw)||!std::isfinite(s.speed))
        {out.error="invalid batch pose";return finish();}
      last=t;
      const double sweep=half_gap*(4.5+std::abs(s.speed)*(1+std::hypot(std::max(g.front,g.back),g.half_width)*std::tan(g.max_steering)/g.wheelbase));
      poses.push_back({t,s.x,s.y,std::cos(s.yaw),std::sin(s.yaw),g.front,g.back,g.half_width,sweep});
    }
    offsets.push_back(int(poses.size()));
  }
  std::vector<joint_detail::Actor> actors;
  for(const auto &t:tracks) {
    if(!t.circular||!t.center_motion_fit||!validGaussianMotion(t.gaussian)) {
      for(size_t c=0;c<candidates.size();++c)if(timedMotionUnsafe(candidates[c],origin,{t},g))
        out.estimates[c]=out.lower_bounds[c]=1.;
      continue;
    }
    const double age=base-t.stamp;
    if(!std::isfinite(age)||age<0||age>.6||t.gaussian.stamp!=t.stamp||!std::isfinite(t.resolution)||t.resolution<=0||
       t.cells.empty()||!std::isfinite(t.radius_bound)||t.radius_bound<=0||!std::isfinite(t.acceleration_bound)||t.acceleration_bound<0||
       !gaussianCircleEnclosesMeasurements(t))
      {out.error="invalid screening source";return finish();}
    joint_detail::Actor a{};a.age=age;a.density=t.gaussian.process_density;
    a.acceleration=t.gaussian.longitudinal_acceleration_bound;a.maximum_speed=predictionSpeedCap(t.gaussian.mean.tail<2>().norm(),t.velocity_error,t.prediction_maximum_speed);
    if(!std::isfinite(a.maximum_speed)||a.maximum_speed<t.gaussian.mean.tail<2>().norm())
      {out.error="invalid acceleration speed domain";return finish();}
    a.radius=t.radius_bound+g.extra_inflation;
    if(!packGaussianRoot(t.gaussian,a)){out.error="invalid covariance";return finish();}
    actors.push_back(a);
  }
  std::vector<unsigned> hits(actors.size()*candidates.size(),0);
  if(!actors.empty()) {
    // Screening only: common random trajectories across candidate policies.
    // Sample on the union of their ACTUAL times; Q's Markov composition
    // preserves each candidate marginal. Final permission still uses fresh,
    // independent samples in jointGaussianMotionRisk, not these ranked paths.
    std::vector<double> times;times.reserve(poses.size());
    for(const auto &p:poses)times.push_back(p.t);
    std::sort(times.begin(),times.end());times.erase(std::unique(times.begin(),times.end()),times.end());
    const bool shared=times.size()<=4096 &&
      double(actors.size())*samples*times.size()*sizeof(joint_detail::Segment)<=256.*1024*1024;
    if(!shared&&double(actors.size())*samples*poses.size()>16.*1024*1024)
      {out.error="screen candidate work budget exceeded";return finish();}
    out.independent_candidate_forecasts=!shared;
    std::vector<int> indices;indices.reserve(poses.size());
    if(shared)for(const auto &p:poses)indices.push_back(std::lower_bound(times.begin(),times.end(),p.t)-times.begin());
    if(gpu) {
#ifdef TERRAIN_NAV_MOTION_CUDA
      const bool success=shared?
        jointMotionCudaSharedBatchHits(actors,poses,offsets,times,indices,samples,seed,hits,out.error):
        jointMotionCudaBatchHits(actors,poses,offsets,samples,seed,hits,out.error);
      if(!success)return finish();
#else
      out.error="CUDA joint-motion backend not built";return finish();
#endif
    } else if(!shared) {
      // Continuous switch times can produce thousands of distinct times.
      // Keep those ACTUAL times (no rounding or dropped states) and the
      // original covariance/process model. Only cross-policy random-number
      // sharing changes; simultaneous lower bounds do not require it.
      for(size_t c=0;c<candidates.size();++c)for(size_t o=0;o<actors.size();++o)for(size_t s=0;s<samples;++s)
        hits[c*actors.size()+o]+=joint_detail::trajectoryHit(actors[o],poses.data()+offsets[c],
          offsets[c+1]-offsets[c],seed^(0x9e3779b97f4a7c15ULL*c),s,o);
    } else {
      std::vector<joint_detail::Segment> forecast(actors.size()*times.size()*samples);
      for(size_t o=0;o<actors.size();++o)for(size_t s=0;s<samples;++s) {
        double state[6],previous=0.;auto rng=joint_detail::initialState(actors[o],seed,s,o,state);
        for(size_t t=0;t<times.size();++t) {
          joint_detail::propagate(state,times[t]-previous,actors[o].density,rng,actors[o].constant_acceleration);previous=times[t];
          forecast[(o*times.size()+t)*samples+s]=joint_detail::sampledSegment(actors[o],state,times[t]);
        }
      }
      for(size_t c=0;c<candidates.size();++c)for(size_t o=0;o<actors.size();++o)for(size_t s=0;s<samples;++s)
        hits[c*actors.size()+o]+=joint_detail::sharedTrajectoryHit(actors[o],poses.data()+offsets[c],
          indices.data()+offsets[c],offsets[c+1]-offsets[c],forecast.data()+o*times.size()*samples,samples,s);
    }
    // A 256-draw estimate just above the final risk limit is not evidence
    // that a candidate is infeasible. Use exact one-sided lower bounds for
    // REJECTION, splitting this recall-error allowance over every actor and
    // fixed candidate. Shared random paths need no cross-policy independence.
    // Final execution still needs fresh independent upper-bound proofs with
    // the ORIGINAL lifetime alpha allocation and risk limit.
    out.discard_confidence_error=discard_error;
    const double alpha=out.discard_confidence_error/double(candidates.size()*actors.size());
    std::vector<double> lower_table(samples+1,NAN);lower_table[0]=0.;
    for(size_t c=0;c<candidates.size();++c)for(size_t o=0;o<actors.size();++o) {
      const unsigned k=hits[c*actors.size()+o];
      if(k>samples){out.error="invalid screening hit count";return finish();}
      if(!std::isfinite(lower_table[k]))lower_table[k]=binomialCollisionLower(samples,k,alpha);
      out.estimates[c]=std::min(1.,out.estimates[c]+double(k)/samples);
      out.lower_bounds[c]=std::min(1.,out.lower_bounds[c]+lower_table[k]);
    }
  }
  out.valid=true;return finish();
}

JointMotionScreen screenJointMotionProgressiveBatch(const std::vector<std::vector<YieldState>> &candidates,
    double origin,const std::vector<MotionTrackSnapshot> &tracks,const MotionCheckGeometry &g,
    size_t samples,uint64_t seed,bool gpu,double risk_limit)
{
  JointMotionScreen out;const auto start=std::chrono::steady_clock::now();
  const auto finish=[&] {out.elapsed_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();return out;};
  if(!std::isfinite(risk_limit)||risk_limit<=0||risk_limit>=1||samples<32||samples>65536) {
    out.error="invalid progressive screening request";return finish();
  }
  if(samples==32)return screenJointMotionBatch(candidates,origin,tracks,g,samples,seed,gpu);
  constexpr uint64_t coarse_domain=0x6a09e667f3bcc909ULL;
  // Validate EVERY original candidate and source before forming a subset.
  out=screenJointMotionBatch(candidates,origin,tracks,g,32,seed^coarse_domain,gpu,5e-4);
  out.coarse_samples=32;
  if(!out.valid)return finish();
  // With no Gaussian objects, the first pass has exact reachable fallbacks,
  // so another random pass cannot add any information.
  if(out.discard_confidence_error==0.)return finish();
  std::vector<size_t> indices;std::vector<std::vector<YieldState>> survivors;
  for(size_t i=0;i<candidates.size();++i)if(out.lower_bounds[i]<=risk_limit) {
    indices.push_back(i);survivors.push_back(candidates[i]);
  }
  out.coarse_discarded=candidates.size()-survivors.size();out.refined_candidates=survivors.size();
  if(survivors.empty())return finish();
  // Conditional on the first stream, this is a fixed subset. Fresh second
  // samples and its own simultaneous 0.0005 budget keep total recall error
  // <=0.001 without assuming independence BETWEEN candidate policies.
  const auto fine=screenJointMotionBatch(survivors,origin,tracks,g,samples,seed,gpu,5e-4);
  if(!fine.valid){out.valid=false;out.error="fine screening: "+fine.error;return finish();}
  out.independent_candidate_forecasts|=fine.independent_candidate_forecasts;
  out.discard_confidence_error+=fine.discard_confidence_error;
  for(size_t i=0;i<indices.size();++i) {
    out.estimates[indices[i]]=fine.estimates[i];out.lower_bounds[indices[i]]=fine.lower_bounds[i];
  }
  return finish();
}
} // namespace terrain_nav
