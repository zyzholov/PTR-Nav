#pragma once
#include <cmath>
#include <cstdint>
#include <limits>

namespace terrain_nav {
// Supervises a detected traction stall; it NEVER generates motion. A restart
// is only a request for fresh normal planning after a confirmed physical stop.
// Retry allowance belongs to the external goal, not a path/map/input epoch.
class MobilityRecovery {
public:
  enum class Action {None,Hold,Replan,Latched};
  void goal(uint64_t sequence) {
    if(sequence==sequence_)return;
    sequence_=sequence;attempts_=0;active_=latched_=false;stopped_=nan();last_=nan();
  }
  bool stall(double now,int allowance) {
    if(active_||latched_)return false;
    if(!std::isfinite(now)||now<=0||allowance<0||allowance>2||attempts_>=allowance) {
      latched_=true;return false;
    }
    ++attempts_;active_=true;begin_=last_=now;stopped_=nan();return true;
  }
  Action tick(double now,double speed,bool fresh,bool supported) {
    if(latched_)return Action::Latched;
    if(!active_)return Action::None;
    if(!std::isfinite(now)||!std::isfinite(speed)||now<last_||now-begin_>5.) {
      active_=false;latched_=true;return Action::Latched;
    }
    last_=now;
    if(!fresh||!supported||std::abs(speed)>=.03) {stopped_=nan();return Action::Hold;}
    if(!std::isfinite(stopped_))stopped_=now;
    if(now-stopped_<.5)return Action::Hold;
    active_=false;stopped_=nan();return Action::Replan;
  }
  int attempts()const{return attempts_;}
  bool active()const{return active_;}
  bool latched()const{return latched_;}
private:
  static double nan(){return std::numeric_limits<double>::quiet_NaN();}
  uint64_t sequence_=std::numeric_limits<uint64_t>::max();
  int attempts_=0;bool active_=false,latched_=false;
  double begin_=0,last_=nan(),stopped_=nan();
};
}
