#pragma once
#include "terrain_nav/motion_guard.hpp"
#include <map>

namespace terrain_nav {
constexpr double motion_resolution_maximum_age=.6;

inline bool validStationaryResolution(const MotionTrackSnapshot &t)
{
  return t.id&&t.motion_confirmed&&std::isfinite(t.stamp)&&std::isfinite(t.resolution)&&
    t.resolution>0&&!t.cells.empty()&&t.vx==0.&&t.vy==0.&&
    std::all_of(t.cells.begin(),t.cells.end(),[](const auto &p) {
      return std::isfinite(p[0])&&std::isfinite(p[1]);
    });
}
inline bool freshStationaryResolution(const MotionTrackSnapshot &t,uint64_t id,
                                     double conflict_source,double now)
{
  return validStationaryResolution(t)&&t.id==id&&std::isfinite(conflict_source)&&
    std::isfinite(now)&&t.stamp>conflict_source&&now>=t.stamp-.05&&
    now-t.stamp<=motion_resolution_maximum_age;
}

// A current stationary posterior and its explicit measured-rest event can
// describe the SAME source observation. Keeping the former for future
// restart prediction must not suppress delivery of the latter. A later
// observation, different geometry, or nonzero measured velocity still revokes
// an older event. This predicate cannot manufacture an event on its own.
inline bool sameStationaryObservation(const MotionTrackSnapshot &track,const MotionTrackSnapshot &event)
{
  return validStationaryResolution(track)&&validStationaryResolution(event)&&
    track.id==event.id&&track.stamp==event.stamp&&track.resolution==event.resolution&&track.cells==event.cells;
}

// A latest-value scan can overwrite a one-scan positive resolution before
// control runs. Deliver that actual observation across the next few scans;
// never refresh its source time, infer a resolution from expiry, or clear an
// occupied cell. A newer same-ID observation revokes the older event.
class MotionResolutionMailbox {
  std::map<uint64_t,MotionTrackSnapshot> pending_;
  double source_=-INFINITY,resolution_=0.;
  uint64_t epoch_=0;
  bool initialized_=false;
 public:
  void reset(){pending_.clear();source_=-INFINITY;resolution_=0.;initialized_=false;}
  void observe(double source,double receipt,uint64_t epoch,double resolution,
      const std::vector<MotionTrackSnapshot> &tracks,const std::vector<MotionTrackSnapshot> &events)
  {
    if(!std::isfinite(source)||!std::isfinite(receipt)||!std::isfinite(resolution)||
       resolution<=0||receipt<source-.05){reset();return;}
    if(initialized_&&(epoch!=epoch_||resolution!=resolution_||source<source_))reset();
    for(auto i=pending_.begin();i!=pending_.end();) {
      const bool superseded=std::any_of(tracks.begin(),tracks.end(),[&](const auto &t) {
        return t.motion_confirmed&&t.id==i->first&&std::isfinite(t.stamp)&&t.stamp>=i->second.stamp&&
          !sameStationaryObservation(t,i->second);
      });
      if(receipt-i->second.stamp>motion_resolution_maximum_age||superseded)i=pending_.erase(i);else ++i;
    }
    if(initialized_&&source==source_)return; // Re-delivery cannot renew evidence.
    source_=source;resolution_=resolution;epoch_=epoch;initialized_=true;
    for(const auto &event:events) {
      if(!validStationaryResolution(event)||std::abs(event.stamp-source)>5e-9||
          receipt-event.stamp>motion_resolution_maximum_age)continue;
      if(std::any_of(tracks.begin(),tracks.end(),[&](const auto &t) {
          return t.motion_confirmed&&t.id==event.id&&std::isfinite(t.stamp)&&t.stamp>=event.stamp&&
            !sameStationaryObservation(t,event);
        }))continue;
      const auto old=pending_.find(event.id);
      if(old==pending_.end()||event.stamp>old->second.stamp)pending_[event.id]=event;
    }
    if(pending_.size()>1024)pending_.clear(); // Lose permission, never manufacture it.
  }
  std::vector<MotionTrackSnapshot> snapshot(double now) const
  {
    std::vector<MotionTrackSnapshot> out;
    if(!std::isfinite(now))return out;
    for(const auto &[id,event]:pending_)if(now>=event.stamp-.05&&
        now-event.stamp<=motion_resolution_maximum_age)out.push_back(event);
    return out;
  }
};
}
