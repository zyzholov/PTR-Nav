#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace terrain_nav
{
// A union bound, not an independence product. Revisited discrete map events
// count once; a more conservative bound for the same event replaces the old.
class ChanceBudget
{
public:
  bool add(uint64_t event, double upper, double budget)
  {
    if (!valid(upper)) return false;
    shared_.add(event,upper);absolute_.add(event,upper);poses_.add(event,upper);mixed_.add(event,upper);
    return this->upper() <= budget;
  }

  bool addAlternatives(uint64_t pose, double pose_upper,
                       const std::vector<uint64_t> &sources,double tail,double budget,
                       const std::vector<double> &absolute_tails={})
  {
    if(!valid(pose_upper) || !valid(tail) || sources.empty()) return false;
    if(!absolute_tails.empty() && (absolute_tails.size()!=sources.size() ||
       !std::all_of(absolute_tails.begin(),absolute_tails.end(),valid))) return false;
    // Each ledger covers every pose failure. Their minimum is a choice of
    // valid proofs, not a product or an independence assumption. Keeping the
    // original all-source ledger prevents a greedy choice from losing a path
    // that the old source certificate could already accept.
    for(auto id:sources) shared_.add(id,tail);
    poses_.add(pose,pose_upper);
    double source_delta=0,absolute_delta=0;
    for(size_t i=0;i<sources.size();++i) {
      const double p=absolute_tails.empty()?tail:absolute_tails[i];
      absolute_.add(sources[i],p);
      source_delta+=mixed_.increment(sources[i],tail);
      absolute_delta+=mixed_.increment(sources[i],p);
    }
    const double pose_delta=mixed_.increment(pose,pose_upper);
    if(absolute_delta<std::min(source_delta,pose_delta)) {
      for(size_t i=0;i<sources.size();++i) mixed_.add(sources[i],absolute_tails.empty()?tail:absolute_tails[i]);
    } else if(source_delta<pose_delta) {
      for(auto id:sources) mixed_.add(id,tail);
    } else mixed_.add(pose,pose_upper);
    return upper()<=budget;
  }
  double upper() const { return std::min({1.0,shared_.sum,absolute_.sum,poses_.sum,mixed_.sum}); }

private:
  static bool valid(double p) {return std::isfinite(p) && p>=0 && p<=1;}
  struct Account {
    double sum=0;
    std::unordered_map<uint64_t,double> events;
    double increment(uint64_t id,double value) const {
      const auto it=events.find(id);
      return std::max(0.0,value-(it==events.end()?0.0:it->second));
    }
    void add(uint64_t id,double value) {
      auto &previous=events[id];sum+=std::max(0.0,value-previous);previous=std::max(value,previous);
    }
  };
  Account shared_,absolute_,poses_,mixed_;
};
}  // namespace terrain_nav
