#pragma once
#include "terrain_nav/motion_risk.hpp"
#include "terrain_nav/joint_motion_kernel.hpp"
#include <string>

namespace terrain_nav {
struct JointMotionRisk {
  bool valid=false;
  // A valid deterministic fallback collision is distinct from unavailable
  // evidence. Neither permits motion; only the former can enroll an identity
  // as a positively witnessed potential conflict.
  bool deterministic_conflict=false;
  double upper=1.,estimate=1.,alpha=0.,elapsed_ms=0.;
  size_t samples=0,gaussian_objects=0,reachable_objects=0;
  size_t analytical_objects=0;
  std::vector<size_t> sample_counts; // Zero means analytically discharged, NOT zero observed collisions.
  std::vector<unsigned> collisions;
  std::string backend,error;
};
struct JointMotionScreen {
  bool valid=false;
  double elapsed_ms=0.;
  // Shared time union when bounded; otherwise independent trajectories per
  // candidate preserve the same marginal model without a huge forecast grid.
  bool independent_candidate_forecasts=false;
  std::vector<double> estimates; // Screening ONLY: never a motion certificate.
  // Gaussian-event lower bounds, simultaneous over this fixed candidate
  // batch. A reachable fallback failure is the hard-rejection sentinel 1.
  // These may reject a proposal, NEVER authorize a command or replace an UCB.
  std::vector<double> lower_bounds;
  double discard_confidence_error=0.;
  size_t coarse_samples=0,coarse_discarded=0,refined_candidates=0;
  std::string error;
};
// More independent samples can reduce a statistical confidence remainder;
// they never lower the configured collision-risk limit or change the model.
// Refinement is useful only for a valid estimated sub-limit event whose
// current confidence bound still fails. Invalid evidence remains a refusal.
inline bool refineMotionConfidence(const JointMotionRisk &risk,double limit,size_t maximum=65536)
{
  return risk.valid&&risk.samples>0&&risk.samples<maximum&&maximum<=65536&&
    std::isfinite(limit)&&limit>0&&limit<1&&std::isfinite(risk.estimate)&&risk.estimate>=0&&
    risk.estimate<limit&&std::isfinite(risk.upper)&&risk.upper>limit&&risk.upper<=1;
}
// One-sided exact binomial confidence bound. Alpha is a SAMPLING error
// probability, distinct from the MODEL's collision risk and its calibration.
double binomialCollisionUpper(size_t trials,size_t hits,double alpha);
double binomialCollisionLower(size_t trials,size_t hits,double alpha);
inline double motionCheckAlpha(uint64_t check,double total=1e-4)
{
  return check && std::isfinite(total) && total>0 && total<1 ?
    total/(double(check)*(double(check)+1.)):0.; // Sum over all checks <= total.
}
bool jointMotionGpuAvailable(std::string *error=nullptr);
JointMotionRisk jointGaussianMotionRisk(const std::vector<YieldState> &states,double origin,
  const std::vector<MotionTrackSnapshot> &tracks,const MotionCheckGeometry &geometry,
  size_t samples,double alpha,uint64_t seed,bool gpu,double preference_clearance=0.,bool analytical_discharge=false);
JointMotionScreen screenJointMotionBatch(const std::vector<std::vector<YieldState>> &candidates,double origin,
  const std::vector<MotionTrackSnapshot> &tracks,const MotionCheckGeometry &geometry,
  size_t samples,uint64_t seed,bool gpu,double discard_error=1e-3);
// Two rejection stages, not two chances to authorize motion. Half the same
// batch recall-error budget goes to each stage. The adaptive second subset
// uses a new domain-separated sample stream; every execution proof remains
// independently resampled with its original lifetime alpha and risk limit.
JointMotionScreen screenJointMotionProgressiveBatch(const std::vector<std::vector<YieldState>> &candidates,
  double origin,const std::vector<MotionTrackSnapshot> &tracks,const MotionCheckGeometry &geometry,
  size_t samples,uint64_t seed,bool gpu,double risk_limit);
} // namespace terrain_nav
