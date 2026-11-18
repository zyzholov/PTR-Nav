#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>

namespace terrain_nav {
// Effort scheduling only, never an input-age extension or a motion permit.
// A candidate proof is followed by a fresh-root final proof. Avoid spending
// the rest of the search budget improving a fallback whose source will expire
// before those two operations finish. Observed cost is a scheduling estimate,
// not a worst-case execution bound; final source-age gates remain mandatory.
inline bool interactionProofFitsFreshness(double now,double source_expiry,
                                         double observed_proof_seconds)
{
  return std::isfinite(now)&&std::isfinite(source_expiry)&&
    std::isfinite(observed_proof_seconds)&&observed_proof_seconds>=0&&
    now+2.*std::max(.005,observed_proof_seconds)+.010<source_expiry;
}

// Nested search phases share the earliest deadline on THIS worker thread.
// Controller callbacks are independent; this never grants motion permission.
class PlanningBudget {
  using Clock=std::chrono::steady_clock;
  using Time=Clock::time_point;
  std::optional<Time> previous_;
  static std::optional<Time> &deadline() {
    static thread_local std::optional<Time> value;
    return value;
  }
public:
  explicit PlanningBudget(double milliseconds):previous_(deadline()) {
    if(!std::isfinite(milliseconds)||milliseconds<0)throw std::invalid_argument("invalid planning budget");
    const auto proposed=Clock::now()+std::chrono::duration_cast<Clock::duration>(
      std::chrono::duration<double,std::milli>(milliseconds));
    deadline()=previous_?std::min(*previous_,proposed):proposed;
  }
  ~PlanningBudget(){deadline()=previous_;}
  PlanningBudget(const PlanningBudget&)=delete;
  PlanningBudget &operator=(const PlanningBudget&)=delete;
  static double remaining(double cap=std::numeric_limits<double>::infinity()) {
    if(std::isnan(cap)||cap<=0)return 0;
    return deadline()?std::max(0.,std::min(cap,
      std::chrono::duration<double,std::milli>(*deadline()-Clock::now()).count())):cap;
  }
  static bool expired(){return remaining()<=0;}
};
} // namespace terrain_nav
