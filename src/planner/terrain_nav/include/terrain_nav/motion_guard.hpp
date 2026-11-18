#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include "terrain_nav/object_observation.hpp"
#include "terrain_nav/gaussian_motion.hpp"

namespace terrain_nav
{
// Positive measured rest evidence for an ALREADY confirmed identity. This
// closes a motion episode, not occupancy or a permanent no-restart promise.
// Use raw fitted centres (not filtered posterior means), a full source-time
// interval and distinct scans. Empty/partial/ambiguous association cannot
// accumulate this evidence. Navigation still checks its raw stationary body.
inline bool measuredStationaryCenters(const std::vector<TimedCenter>&history)
{
  if(history.size()<8)return false;
  const auto&first=history.front();const auto&last=history.back();
  if(!std::isfinite(first.resolution)||first.resolution<=0||
     !std::isfinite(first.stamp_s)||!std::isfinite(last.stamp_s)||
     last.stamp_s-first.stamp_s<.8-1e-6)return false;
  double mt=0,mx=0,my=0;
  for(size_t i=0;i<history.size();++i) {
    const auto&s=history[i];
    if(!std::isfinite(s.stamp_s)||!std::isfinite(s.center[0])||!std::isfinite(s.center[1])||
       !std::isfinite(s.radius)||s.radius<.15||s.radius>1.5||s.resolution!=first.resolution||
       (i&&(s.stamp_s<=history[i-1].stamp_s||s.stamp_s-history[i-1].stamp_s>.200001)))return false;
    for(size_t j=0;j<i;++j)
      if(std::hypot(s.center[0]-history[j].center[0],s.center[1]-history[j].center[1])>1.5*first.resolution||
         std::abs(s.radius-history[j].radius)>2*first.resolution)return false;
    mt+=s.stamp_s-first.stamp_s;mx+=s.center[0];my+=s.center[1];
  }
  mt/=history.size();mx/=history.size();my/=history.size();
  double xx=0,xy=0,tt=0;
  for(const auto&s:history) {
    const double t=s.stamp_s-first.stamp_s-mt;
    xx+=t*(s.center[0]-mx);xy+=t*(s.center[1]-my);tt+=t*t;
  }
  return tt>0&&std::hypot(xx,xy)/tt<=.05;
}

inline double boundedTranslationDistance(double speed,double t,double acceleration,double maximum_speed=4.5)
{
  if(!std::isfinite(speed) || speed<0 || !std::isfinite(t) || t<0 ||
     !std::isfinite(acceleration) || acceleration<0 || !std::isfinite(maximum_speed) ||
     maximum_speed<=0 || speed>maximum_speed+1e-8)return INFINITY;
  if(acceleration==0)return speed*t;
  const double accelerating=std::min(t,std::max(0.,(maximum_speed-speed)/acceleration));
  return speed*accelerating+.5*acceleration*accelerating*accelerating+
         maximum_speed*(t-accelerating);
}
inline double minimumTranslationDistance(double speed,double t,double acceleration,double maximum_speed=4.5)
{
  if(!std::isfinite(boundedTranslationDistance(speed,t,acceleration,maximum_speed)))return -INFINITY;
  if(acceleration==0)return speed*t;
  const double reversing=std::min(t,(speed+maximum_speed)/acceleration);
  return speed*reversing-.5*acceleration*reversing*reversing-maximum_speed*(t-reversing);
}
inline double predictionSpeedCap(double speed,double velocity_error,double profile_maximum)
{
  if(!std::isfinite(speed)||speed<0||!std::isfinite(velocity_error)||velocity_error<0||
     !std::isfinite(profile_maximum)||profile_maximum<=0||profile_maximum>4.5)return NAN;
  // A walking-domain prior may NOT truncate an already faster observation.
  // Leaving that domain immediately restores the original full-speed model.
  return speed+velocity_error>profile_maximum?4.5:profile_maximum;
}
inline double motionAssociationRadius(double resolution,double age=0.)
{
  // Same positional + velocity-error reserve as the temporal controller.
  // A point-valued quantized velocity is not exact over a partial coast.
  // This is the existing engineering model, not a calibrated confidence set.
  return std::isfinite(resolution) && resolution>0 && std::isfinite(age) && age>=0 && age<=.600001 ?
    1.5*resolution+.25*age :
    std::numeric_limits<double>::infinity();
}
// Identity association only, never free-space evidence or a collision margin.
// The same declared acceleration domain used in prediction also bounds the
// deviation from the last centre's constant-velocity extrapolation.
inline double motionIdentityRadius(double resolution,double age,double velocity_error,double acceleration)
{
  if(!std::isfinite(resolution)||resolution<=0||!std::isfinite(age)||age<0||age>.600001||
     !std::isfinite(velocity_error)||velocity_error<0||!std::isfinite(acceleration)||acceleration<0||acceleration>4.5)
    return NAN;
  return motionAssociationRadius(resolution,age)+velocity_error*age+.5*acceleration*age*age;
}
// Necessary consistency under the declared vector-acceleration model. This
// only vetoes identity reuse; failure never erases the new measured body.
// A poorly conditioned circle's positional reserve cannot excuse an
// incompatible velocity change or authorize donation of another object's ID.
inline bool motionVelocityIdentityCompatible(double old_vx,double old_vy,double old_error,
    double vx,double vy,double error,double age,double acceleration)
{
  for(double value:{old_vx,old_vy,old_error,vx,vy,error,age,acceleration})
    if(!std::isfinite(value))return false;
  if(age<=0||old_error<0||error<0||acceleration<0||acceleration>4.5||
     std::hypot(old_vx,old_vy)>4.5+1e-8||std::hypot(vx,vy)>4.5+1e-8)return false;
  return std::hypot(vx-old_vx,vy-old_vy)<=acceleration*age+old_error+error+1e-8;
}
struct AggregatedMotionInput {
  int width=0,height=0;
  std::vector<uint8_t> occupied;
  std::vector<std::vector<int>> components;
};

// Velocity registration may use coarse cells, but segmentation must retain
// the original measured connectivity. Otherwise a real narrow gap between a
// moving object and a static wall vanishes before association even starts.
// Coarse groups may share cells; this never deletes any collision occupancy.
inline AggregatedMotionInput aggregateMotionComponents(const std::vector<uint8_t>&raw,
    int width,int height,int factor)
{
  if(width<=0 || height<=0 || raw.size()!=size_t(width)*height || factor<1 || factor>4 ||
     width%factor || height%factor)throw std::invalid_argument("invalid motion aggregation");
  AggregatedMotionInput out;out.width=width/factor;out.height=height/factor;
  out.occupied.assign(size_t(out.width)*out.height,0);
  std::vector<uint8_t> seen(raw.size(),0);
  for(size_t seed=0;seed<raw.size();++seed) {
    if(!raw[seed] || seen[seed])continue;
    std::vector<int> fine{int(seed)},coarse;seen[seed]=1;
    for(size_t j=0;j<fine.size();++j) {
      const int q=fine[j],x=q%width,y=q/width;
      const int c=(y/factor)*out.width+x/factor;
      out.occupied[c]=1;coarse.push_back(c);
      for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx) {
        const int nx=x+dx,ny=y+dy;
        if(nx<0 || ny<0 || nx>=width || ny>=height)continue;
        const int n=ny*width+nx;
        if(raw[n] && !seen[n]){seen[n]=1;fine.push_back(n);}
      }
    }
    std::sort(coarse.begin(),coarse.end());coarse.erase(std::unique(coarse.begin(),coarse.end()),coarse.end());
    out.components.push_back(std::move(coarse));
  }
  return out;
}

struct MotionTrackSnapshot {
  std::vector<std::array<double,2>> cells;
  double vx=0,vy=0,stamp=0,resolution=0;
  uint64_t id=0;
  // Optional measured curved-body hypothesis. Non-circular objects retain
  // their measured-cell model; these are engineering bounds, not semantics
  // or calibrated human motion probabilities.
  bool circular=false;
  double center_x=0,center_y=0,radius_bound=0,velocity_error=0,fit_rms=0;
  double acceleration_bound=0;
  double prediction_maximum_speed=4.5;
  double measured_extent_x=0,measured_extent_y=0;
  // Geometry-qualified causal center velocity is separate from a confirmed
  // persistent identity. Unconfirmed hypotheses keep id=0 and cannot resolve
  // a lost confirmed track, create free space, or acquire a coasting lifetime.
  bool motion_confirmed=true,center_motion_fit=false;
  GaussianMotion gaussian;
  bool retained_radius=false,raw_surface_fit=false;
  bool raw_surface_input=false;
  size_t raw_surface_return_count=0;
  double velocity_averaging_lag_s=0.;
  std::vector<TimedCenter> center_observations; // Actual causal sources, diagnostic/reassociation only.
};
// A LABEL search region, not an occupancy/velocity prediction or free space.
struct MotionWitnessRegion {uint64_t id=0;double x=0,y=0,radius=0,source_stamp=0;};
inline std::vector<uint8_t> motionWitnessMask(const std::vector<uint8_t> &body,
    const std::vector<std::array<float,3>> &points,const std::vector<float> &ground,
    int width,int height,double min_x,double min_y,double resolution,
    const std::vector<MotionWitnessRegion> &regions,double maximum_height,int minimum_hits)
{
  if(width<=0||height<=0||body.size()!=size_t(width)*height||ground.size()!=body.size()||
      !std::isfinite(min_x)||!std::isfinite(min_y)||!std::isfinite(resolution)||resolution<=0||
      !std::isfinite(maximum_height)||maximum_height<0||maximum_height>3.||minimum_hits<1)
    throw std::invalid_argument("invalid motion witness geometry");
  auto result=body;if(maximum_height==0||regions.empty())return result;
  std::vector<uint8_t> hits(body.size(),0);
  for(const auto &p:points) {
    if(!std::isfinite(p[0])||!std::isfinite(p[1])||!std::isfinite(p[2]))continue;
    const double col=std::floor((p[0]-min_x)/resolution),row=std::floor((p[1]-min_y)/resolution);
    if(col<0||row<0||col>=width||row>=height)continue;
    const size_t i=size_t(row)*width+size_t(col);
    if(body[i]||!std::isfinite(ground[i])||p[2]-ground[i]<=.30F||p[2]-ground[i]>maximum_height)continue;
    if(std::any_of(regions.begin(),regions.end(),[&](const auto &r) {
        return r.id && std::isfinite(r.x)&&std::isfinite(r.y)&&std::isfinite(r.radius)&&r.radius>0 &&
          std::hypot(p[0]-r.x,p[1]-r.y)<=r.radius;
      }) && hits[i]<255)++hits[i];
  }
  for(int y=0;y<height;++y)for(int x=0;x<width;++x)if(hits[y*width+x]) {
    int count=0;
    for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx)
      if(x+dx>=0&&y+dy>=0&&x+dx<width&&y+dy<height)count+=hits[(y+dy)*width+x+dx];
    if(count>=minimum_hits)result[y*width+x]=1;
  }
  return result;
}
// Short-horizon constant-translation guard, separate from terrain probabilities.
// Mutual nearest component association is deliberately bounded; no truth input.
class GridMotionGuard
{
  struct Component {
    double x=0,y=0;
    std::vector<int> cells;
    std::unordered_set<uint64_t> lattice;
    std::vector<std::array<double,2>> circle_points;
    bool raw_surface_input=false;
    std::optional<CircleObservation> circle;
  };
  static uint64_t key(int x,int y) {return (uint64_t(static_cast<uint32_t>(x))<<32)|static_cast<uint32_t>(y);}
  std::vector<Component> previous_;
  std::unordered_set<uint64_t> previous_lattice_;
  std::array<double,4> previous_window_{};
  struct Observation {
    std::vector<Component> components;
    std::unordered_set<uint64_t> lattice;
    std::array<double,4> window;
    double stamp=0;
  };
  std::vector<Observation> observations_;
  struct Track {
    Component shape;
    double vx=0,vy=0,stamp=0;
    bool confirmed=false;
    uint64_t id=0;
    double velocity_error=0;
    double velocity_averaging_lag_s=0.;
    bool center_motion_fit=false;
    GaussianMotion gaussian;
    std::vector<TimedCenter> center_observations;
  };
  std::vector<Track> tracks_;
  std::vector<Track> candidates_;
  std::vector<Track> brake_candidates_;
  std::vector<Track> recent_confirmed_,resolved_;
  // Identity memory is NOT a coasting prediction. Reacquisition below needs
  // fresh strict motion witnesses; only their NEW source may be exported.
  std::vector<Track> dormant_identities_;
  // An observed shape may temporarily have only raster registration. Keep
  // its last VALID posterior separately: replacing current geometry must
  // not destroy Bayesian state. Never export this cache as a fresh track.
  std::map<uint64_t,GaussianMotion> gaussian_history_;
  // A positively identified stationary return remains owned by that ID.
  // Otherwise it can resolve another old ID on the NEXT scan as that other
  // identity's association envelope grows. This registry only vetoes reuse;
  // it never creates occupancy, free space, or an executable prediction.
  std::vector<Track> stationary_identity_history_;
  std::map<uint64_t,std::vector<TimedCenter>> measured_rest_history_;
  double resolution_history_overflow_until_=-INFINITY;
  uint64_t next_id_=1;
  double stamp_=std::numeric_limits<double>::quiet_NaN();
  double resolution_=0;

public:
  int moving_components=0;
  // Association connectivity only. Every component still consists solely of
  // measured cells, and the collision grid retains its original resolution.
  int component_link_cells=1;
  // Labels only, never an executable prediction lifetime. Turning recovery
  // additionally requires fresh causal fits, physical velocity/position gates
  // and mutual uniqueness; absence cannot resolve an old conflict.
  double identity_memory_seconds=3.;
  bool turning_identity_reacquisition=false;
  double acceleration_bound_mps2=0.; // Explicit operating model, not fitted truth.
  double prediction_maximum_speed_mps=4.5;
  bool fit_gaussian_motion=false;
  double gaussian_observation_std=.03,gaussian_process_density=.04;
  bool gaussian_constant_acceleration=false;
  double gaussian_jerk_density=.04;
  int gaussian_maximum_observations=3;
  bool gaussian_acceleration_envelope=false;
  // Optional continuity for an ALREADY identified, currently measured body.
  // Birth still requires the original strict translation evidence. Slowdown
  // and reversal are not missing observations, nor proof of permanent rest.
  bool track_known_slow_centers=false;
  bool sim_observation_calibration=false;
  double sim_observation_std=.018;
  std::vector<MotionWitnessRegion> trackingWitnessRegions(double stamp) const {
    std::vector<MotionWitnessRegion> result;
    if(!std::isfinite(stamp))return result;
    const auto collect=[&](const auto &history) {
      for(const auto &t:history) {
        const double age=stamp-t.stamp;
        if(!t.id||!t.shape.circle||age<=0||age>identity_memory_seconds||
           std::any_of(result.begin(),result.end(),[&](const auto &r){return r.id==t.id;}))continue;
        const auto &c=*t.shape.circle;
        result.push_back({t.id,c.center[0]+t.vx*age,c.center[1]+t.vy*age,
          c.conservative_radius+1.5*resolution_+.25*age+t.velocity_error*age+
            .5*acceleration_bound_mps2*age*age,t.stamp});
      }
    };
    collect(recent_confirmed_);collect(dormant_identities_);return result;
  }
  static bool simRepeatabilityDomain(const CircleObservation &circle,double resolution) {
    return std::isfinite(resolution)&&std::abs(resolution-.05)<=1e-8 &&
      !circle.retained_radius && !circle.raw_surface &&
      std::isfinite(circle.radius)&&circle.radius>=.30&&circle.radius<=.60&&
      std::isfinite(circle.observed_span_rad)&&circle.observed_span_rad>=2.;
  }

private:
  std::vector<MotionTrackSnapshot> exportTracks(const std::vector<Track> &tracks,bool provisional) const
  {
    std::vector<MotionTrackSnapshot> result;
    for(const auto &track:tracks) {
      if(provisional && track.confirmed)continue;
      MotionTrackSnapshot out;
      out.vx=track.vx;out.vy=track.vy;out.stamp=track.stamp;out.resolution=resolution_;out.id=track.id;
      out.acceleration_bound=acceleration_bound_mps2;
      out.prediction_maximum_speed=prediction_maximum_speed_mps;
      out.motion_confirmed=track.confirmed;out.center_motion_fit=track.center_motion_fit;
      out.gaussian=track.gaussian;
      out.raw_surface_input=track.shape.raw_surface_input;
      out.raw_surface_return_count=track.shape.raw_surface_input?track.shape.circle_points.size():0;
      out.velocity_averaging_lag_s=track.velocity_averaging_lag_s;
      out.center_observations=track.center_observations;
      if(track.shape.circle) {
        out.raw_surface_fit=track.shape.circle->raw_surface;
        out.circular=true;out.center_x=track.shape.circle->center[0];out.center_y=track.shape.circle->center[1];
        out.radius_bound=track.shape.circle->conservative_radius;
        out.fit_rms=track.shape.circle->residual_rms;out.velocity_error=track.velocity_error;
        out.retained_radius=track.shape.circle->retained_radius;
      }
      for(uint64_t cell:track.shape.lattice) {
        const int gx=static_cast<int32_t>(cell>>32),gy=static_cast<int32_t>(cell&0xffffffffU);
        out.cells.push_back({(gx+.5)*resolution_,(gy+.5)*resolution_});
      }
      double x0=INFINITY,x1=-INFINITY,y0=INFINITY,y1=-INFINITY;
      for(const auto &cell:out.cells) {
        x0=std::min(x0,cell[0]);x1=std::max(x1,cell[0]);
        y0=std::min(y0,cell[1]);y1=std::max(y1,cell[1]);
      }
      out.measured_extent_x=x1-x0+resolution_;out.measured_extent_y=y1-y0+resolution_;
      if(!out.circular){out.center_x=track.shape.x;out.center_y=track.shape.y;}
      result.push_back(std::move(out));
    }
    return result;
  }

public:
  std::vector<MotionTrackSnapshot> snapshot() const {return exportTracks(tracks_,false);}
  // Read-only diagnostic observations, with no motion/identity/free-space
  // implication. Used to audit measurement repeatability on recorded scans.
  std::vector<CircleObservation> measuredCircleObservations()const {
    std::vector<CircleObservation> out;
    for(const auto &component:previous_)if(component.circle)out.push_back(*component.circle);
    return out;
  }
  std::vector<std::array<double,4>> compactObservedBounds() const
  {
    std::vector<std::array<double,4>> result;
    if(!std::isfinite(resolution_) || resolution_<=0)return result;
    for(const auto &component:previous_) {
      if(component.lattice.size()<3 || component.lattice.size()>128)continue;
      std::array<double,4> b={INFINITY,INFINITY,-INFINITY,-INFINITY};
      for(const auto cell:component.lattice) {
        const double x=(static_cast<int32_t>(cell>>32)+.5)*resolution_;
        const double y=(static_cast<int32_t>(cell&0xffffffffU)+.5)*resolution_;
        b[0]=std::min(b[0],x-.5*resolution_);b[1]=std::min(b[1],y-.5*resolution_);
        b[2]=std::max(b[2],x+.5*resolution_);b[3]=std::max(b[3],y+.5*resolution_);
      }
      // A cautious encounter cue, NOT a human classifier or motion proof.
      // Full walls are excluded; compact static posts may legitimately slow
      // an approach. No occupancy is deleted or expanded by this product.
      if(b[2]-b[0]>=.3 && b[3]-b[1]>=.3 && b[2]-b[0]<=1.8 && b[3]-b[1]<=1.8)
        result.push_back(b);
    }
    return result;
  }
  // Positive current observation of the SAME component stationary over the
  // history interval. Missing returns or an expired velocity are not this.
  std::vector<MotionTrackSnapshot> resolvedSnapshot() const {return exportTracks(resolved_,false);}
  // Fresh first-translation evidence can request braking, never a reverse
  // maneuver or a long-lived obstacle reservation.
  std::vector<MotionTrackSnapshot> provisionalSnapshot() const {return exportTracks(brake_candidates_,true);}
  std::vector<MotionTrackSnapshot> actionHypotheses(bool fitted_candidates)const {
    auto result=snapshot();
    if(fit_gaussian_motion&&(gaussian_constant_acceleration||track_known_slow_centers))
      for(const auto &known:exportTracks(resolved_,false))
      if(known.center_motion_fit&&known.gaussian.constant_acceleration==gaussian_constant_acceleration&&validGaussianMotion(known.gaussian)&&
         std::none_of(result.begin(),result.end(),[&](const auto &t){return t.id==known.id;}))result.push_back(known);
    if(fitted_candidates)for(const auto &hint:provisionalSnapshot())
      if(hint.center_motion_fit)result.push_back(hint);
    return result;
  }
  std::vector<MotionTrackSnapshot> brakeOnlyHypotheses(bool fitted_candidates)const {
    auto result=provisionalSnapshot();
    if(fitted_candidates)result.erase(std::remove_if(result.begin(),result.end(),
      [](const auto &hint){return hint.center_motion_fit;}),result.end());
    return result;
  }

  std::vector<uint8_t> update(const std::vector<uint8_t> &raw,int width,int height,
      double min_x,double min_y,double resolution,double stamp,double horizon=1.5,
      const std::vector<std::vector<int>> &groups={},
      const std::vector<std::array<double,2>> *surface_points=nullptr)
  {
    if(width<=0 || height<=0 || raw.size()!=static_cast<size_t>(width*height) ||
       !std::isfinite(resolution) || resolution<=0 || !std::isfinite(stamp) ||
       !std::isfinite(min_x) || !std::isfinite(min_y) ||
       !std::isfinite(acceleration_bound_mps2) || acceleration_bound_mps2<0 || acceleration_bound_mps2>4.5 ||
       !std::isfinite(identity_memory_seconds)||identity_memory_seconds<.6||identity_memory_seconds>6. ||
       !std::isfinite(prediction_maximum_speed_mps)||prediction_maximum_speed_mps<=0||prediction_maximum_speed_mps>4.5 ||
       component_link_cells<1 || component_link_cells>3 ||
       !std::isfinite(horizon) || horizon<=0 || horizon>3)
      throw std::invalid_argument("invalid motion-guard geometry or time");
    if(surface_points&&!groups.empty())throw std::invalid_argument("raw circle fitting requires native component geometry");
    std::unordered_map<int,std::vector<std::array<double,2>>> point_cells;
    if(surface_points)for(const auto &p:*surface_points) {
      if(!std::isfinite(p[0])||!std::isfinite(p[1]))throw std::invalid_argument("nonfinite motion surface point");
      const double x=std::floor((p[0]-min_x)/resolution),y=std::floor((p[1]-min_y)/resolution);
      if(x<0||y<0||x>=width||y>=height||!raw[size_t(y)*width+size_t(x)])
        throw std::invalid_argument("motion surface point has no measured occupied cell");
      point_cells[int(y)*width+int(x)].push_back(p);
    }
    std::vector<uint8_t> seen(raw.size(),0),forecast(raw.size(),0);
    std::vector<Component> current;
    std::unordered_set<uint64_t> current_lattice;
    std::vector<std::array<int,2>> neighbors;
    if(groups.empty()) {
      const int radius=component_link_cells;
      const int radius_squared=std::max(2,radius*radius);
      for(int dy=-radius;dy<=radius;++dy)for(int dx=-radius;dx<=radius;++dx)
        if(dx*dx+dy*dy>0&&dx*dx+dy*dy<=radius_squared)neighbors.push_back({dx,dy});
    }
    for(size_t seed=0;seed<(groups.empty()?raw.size():groups.size());++seed) {
      if(groups.empty() && (!raw[seed] || seen[seed]))continue;
      Component item;
      if(groups.empty()){item.cells.push_back(static_cast<int>(seed));seen[seed]=1;}
      else {
        item.cells=groups[seed];
        if(!std::is_sorted(item.cells.begin(),item.cells.end()) ||
           std::adjacent_find(item.cells.begin(),item.cells.end())!=item.cells.end())
          throw std::invalid_argument("motion groups must contain sorted unique indices");
        for(int q:item.cells)if(q<0 || size_t(q)>=raw.size() || !raw[q])
          throw std::invalid_argument("motion group has no measured occupancy");
      }
      for(size_t p=0;p<item.cells.size();++p) {
        const int q=item.cells[p],x=q%width,y=q/width;
        const double wx=min_x+(x+.5)*resolution,wy=min_y+(y+.5)*resolution;
        item.x+=wx;item.y+=wy;
        const auto address=key(static_cast<int>(std::floor(wx/resolution)),static_cast<int>(std::floor(wy/resolution)));
        item.lattice.insert(address);current_lattice.insert(address);
        if(!groups.empty())continue;
        for(const auto &offset:neighbors) {
          const int nx=x+offset[0],ny=y+offset[1];
          if(nx<0 || ny<0 || nx>=width || ny>=height)continue;
          const int n=ny*width+nx;
          if(raw[n] && !seen[n]){seen[n]=1;item.cells.push_back(n);}
        }
      }
      if(item.cells.size()<2 || item.cells.size()>512)continue;
      item.x/=item.cells.size();item.y/=item.cells.size();
      std::vector<std::array<double,2>> measured;measured.reserve(item.lattice.size());
      for(auto cell:item.lattice)measured.push_back({(static_cast<int32_t>(cell>>32)+.5)*resolution,
        (static_cast<int32_t>(cell&0xffffffffU)+.5)*resolution});
      if(surface_points) {
        item.raw_surface_input=true;
        for(int cell:item.cells) {
          const auto found=point_cells.find(cell);
          if(found!=point_cells.end())item.circle_points.insert(item.circle_points.end(),found->second.begin(),found->second.end());
        }
        item.circle=fitRawSurfaceCircle(item.circle_points,measured,resolution);
      } else {
        item.circle_points=measured;
        item.circle=fitMeasuredCircle(measured,resolution);
      }
      if(stamp>stamp_&&stamp-stamp_<=.4&&resolution==resolution_) {
        std::vector<CircleObservation> retained;
        for(const auto &old:recent_confirmed_)if(old.id&&old.center_motion_fit&&old.shape.circle) {
          const auto &prior=*old.shape.circle;const double age=stamp-old.stamp;
          if(age<=0||age>.6)continue;
          const std::array<double,2> predicted={prior.center[0]+old.vx*age,prior.center[1]+old.vy*age};
          const double base_gate=motionIdentityRadius(resolution,age,old.velocity_error,acceleration_bound_mps2);
          if(!prior.retained_radius&&item.circle&&std::abs(prior.radius-item.circle->radius)<=2*resolution&&
              std::hypot(item.circle->center[0]-predicted[0],item.circle->center[1]-predicted[1])<=base_gate)continue;
          // Position conditioning of the old partial arc matters for shape
          // association. This does not reduce the new body's collision size.
          const double gate=base_gate+prior.center_error_bound;
          if(std::hypot(item.x-predicted[0],item.y-predicted[1])>prior.conservative_radius+gate)continue;
          auto fit=fitMeasuredCircleWithRadius(item.circle_points,resolution,prior.radius,predicted,surface_points?8192:512);
          if(fit&&surface_points) {
            if(!encloseMeasuredCircleCells(*fit,measured,resolution))fit.reset();
            else fit->raw_surface=true;
          }
          if(fit&&std::hypot(fit->center[0]-predicted[0],fit->center[1]-predicted[1])<=gate) {
            fit->conservative_radius=std::max(fit->conservative_radius,prior.conservative_radius);
            retained.push_back(*fit);
          }
        }
        if(retained.size()==1)item.circle=retained.front();
      }
      current.push_back(std::move(item));
    }
    moving_components=0;
    resolved_.clear();
    const double input_dt=stamp-stamp_;
    constexpr double coast_seconds=.6;
    if(input_dt<=0 || input_dt>.4 || resolution!=resolution_) {
      tracks_.clear();candidates_.clear();brake_candidates_.clear();observations_.clear();recent_confirmed_.clear();
      dormant_identities_.clear();
      gaussian_history_.clear();
      stationary_identity_history_.clear();resolution_history_overflow_until_=-INFINITY;
      measured_rest_history_.clear();
    }
    for(auto i=measured_rest_history_.begin();i!=measured_rest_history_.end();)
      if(i->second.empty()||stamp-i->second.back().stamp_s>.200001)i=measured_rest_history_.erase(i);else ++i;
    for(auto i=gaussian_history_.begin();i!=gaussian_history_.end();)
      if(stamp-i->second.stamp>identity_memory_seconds)i=gaussian_history_.erase(i);else ++i;
    const double stationary_ownership_seconds=turning_identity_reacquisition?identity_memory_seconds:1.;
    stationary_identity_history_.erase(std::remove_if(stationary_identity_history_.begin(),stationary_identity_history_.end(),
      [&](const Track &t){return stamp-t.stamp>stationary_ownership_seconds;}),stationary_identity_history_.end());
    // At .20m / 10Hz, an .8m/s object advances only .08m per scan. Requiring
    // two consecutive one-frame raster translations cannot confirm it, and
    // an unchanged raster frame is not evidence of zero velocity. Use up to
    // .30s of actual measured shape displacement. The estimator still needs
    // two consistent translations and the existing wall/occlusion vetoes.
    observations_.erase(std::remove_if(observations_.begin(),observations_.end(),
      [&](const Observation &o){return stamp-o.stamp>.300001;}),observations_.end());
    double association_stamp=stamp_;
    if(!observations_.empty()) {
      const auto &baseline=observations_.front();
      previous_=baseline.components;previous_lattice_=baseline.lattice;
      previous_window_=baseline.window;association_stamp=baseline.stamp;
    }
    const double dt=stamp-association_stamp;
    recent_confirmed_.erase(std::remove_if(recent_confirmed_.begin(),recent_confirmed_.end(),
      [&](const Track &t){return stamp-t.stamp>1.;}),recent_confirmed_.end());
    dormant_identities_.erase(std::remove_if(dormant_identities_.begin(),dormant_identities_.end(),
      [&](const Track &t){return stamp-t.stamp>identity_memory_seconds;}),dormant_identities_.end());
    tracks_.erase(std::remove_if(tracks_.begin(),tracks_.end(),[&](const Track &track) {
      return stamp-track.stamp>coast_seconds;
    }),tracks_.end());
    // A scan that cannot establish translation is not evidence that an
    // earlier STRICT translation did not occur. Preserve its confirmation
    // opportunity across subcell/partly occluded frames, for the same .4s
    // interval accepted by consistent(). Only distinct source scans count.
    candidates_.erase(std::remove_if(candidates_.begin(),candidates_.end(),[&](const Track &track) {
      return stamp-track.stamp>.4;
    }),candidates_.end());
    std::vector<Track> detected,early_brakes;
    if(dt>=.04 && dt<=.4 && resolution==resolution_ && current.size()<=256 && previous_.size()<=256) {
      std::vector<int> old_for_new(current.size(),-1),new_for_old(previous_.size(),-1);
      std::vector<double> best_new(current.size(),1e30),best_old(previous_.size(),1e30);
      for(size_t i=0;i<current.size();++i)for(size_t j=0;j<previous_.size();++j) {
        const double ratio=double(current[i].cells.size())/previous_[j].cells.size();
        if(ratio<.25 || ratio>4)continue;
        const double distance=std::hypot(current[i].x-previous_[j].x,current[i].y-previous_[j].y);
        if(distance<best_new[i]){best_new[i]=distance;old_for_new[i]=j;}
        if(distance<best_old[j]){best_old[j]=distance;new_for_old[j]=i;}
      }
      for(size_t i=0;i<current.size();++i) {
        const int j=old_for_new[i];
        if(j<0 || new_for_old[j]!=static_cast<int>(i))continue;
        const Component *reference=&previous_[j];
        const auto *reference_lattice=&previous_lattice_;
        auto reference_window=previous_window_;
        double pair_dt=dt;
        // A changing visible face shifts its centroid even when its physical
        // center does not move. Qualify a curved-body hypothesis from the
        // COMPLETE measured component, then associate centers mutually across
        // distinct source scans. No geometric fit alone confirms motion.
        std::vector<TimedCenter> centers;
        if(current[i].circle) {
          const auto &now_circle=*current[i].circle;
          bool calibration_domain=sim_observation_calibration&&simRepeatabilityDomain(now_circle,resolution);
          for(const auto &observation:observations_) {
            const Component *best=nullptr;double closest=INFINITY,second=INFINITY;
            for(const auto &old:observation.components)if(old.circle &&
                std::abs(old.circle->radius-now_circle.radius)<=2*resolution) {
              const double d=std::hypot(old.circle->center[0]-now_circle.center[0],
                                       old.circle->center[1]-now_circle.center[1]);
              if(d<closest){second=closest;closest=d;best=&old;}else second=std::min(second,d);
            }
            if(!best || closest>4.5*(stamp-observation.stamp)+2*resolution || second-closest<resolution)continue;
            bool mutual=true;
            for(size_t other=0;other<current.size();++other)if(other!=i && current[other].circle &&
                std::abs(current[other].circle->radius-best->circle->radius)<=2*resolution &&
                std::hypot(current[other].circle->center[0]-best->circle->center[0],
                           current[other].circle->center[1]-best->circle->center[1])<=closest+resolution)
              mutual=false;
            if(mutual) {
              centers.push_back({observation.stamp,best->circle->center,best->circle->radius,resolution});
              calibration_domain&=simRepeatabilityDomain(*best->circle,resolution);
            }
          }
          centers.push_back({stamp,now_circle.center,now_circle.radius,resolution});
          const auto velocity=fitCausalCenterVelocity(centers);
          if(velocity && velocity->fit_residual_rms<=resolution &&
              velocity->max_step_innovation<=2*resolution) {
            const double speed=std::hypot(velocity->velocity[0],velocity->velocity[1]);
            const double displacement=std::hypot(centers.back().center[0]-centers.front().center[0],
                                                  centers.back().center[1]-centers.front().center[1]);
            std::vector<uint64_t> slow_identities;
            if(track_known_slow_centers && fit_gaussian_motion && speed<.35) {
              for(const auto &old:recent_confirmed_)if(old.id && old.center_motion_fit && old.shape.circle) {
                const auto &c=*old.shape.circle;const double age=stamp-old.stamp;
                if(age<=0 || age>.6 || std::abs(c.radius-now_circle.radius)>2*resolution)continue;
                const double gate=motionIdentityRadius(resolution,age,old.velocity_error,acceleration_bound_mps2);
                if(std::hypot(c.center[0]+old.vx*age-now_circle.center[0],
                              c.center[1]+old.vy*age-now_circle.center[1])<=gate)
                  slow_identities.push_back(old.id);
              }
              std::sort(slow_identities.begin(),slow_identities.end());
              slow_identities.erase(std::unique(slow_identities.begin(),slow_identities.end()),slow_identities.end());
            }
            if(speed<=4.5 && ((speed>=.35 && displacement>=2*resolution) || slow_identities.size()==1)) {
              Track track{current[i],velocity->velocity[0],velocity->velocity[1],stamp};
              // In-sample error reserve with an explicit engineering floor.
              // This is not a calibrated future acceleration/intent model.
              track.velocity_averaging_lag_s=velocity->averaging_lag_s;
              track.velocity_error=std::max(.10,2*velocity->fit_residual_rms/velocity->span_s)+
                acceleration_bound_mps2*velocity->averaging_lag_s;
              track.center_motion_fit=true;
              track.center_observations=centers;
              if(fit_gaussian_motion) {
                track.gaussian=gaussian_constant_acceleration?fitGaussianAccelerationCenterMotion(centers,
                  calibration_domain?sim_observation_std:gaussian_observation_std,
                  gaussian_jerk_density,gaussian_maximum_observations):fitGaussianCenterMotion(centers,
                  calibration_domain?sim_observation_std:gaussian_observation_std,
                  gaussian_process_density,gaussian_maximum_observations);
                track.gaussian.longitudinal_acceleration_bound=gaussian_acceleration_envelope?acceleration_bound_mps2:0.;
              }
              detected.push_back(std::move(track));continue;
            }
            const bool stationary=speed<.20 && std::all_of(centers.begin(),centers.end(),[&](const auto &c) {
              return std::hypot(c.center[0]-now_circle.center[0],c.center[1]-now_circle.center[1])<=1.5*resolution;
            });
            if(stationary) {
              auto stationary_history=turning_identity_reacquisition?dormant_identities_:recent_confirmed_;
              if(turning_identity_reacquisition)for(const auto &old:recent_confirmed_)
                if(std::none_of(stationary_history.begin(),stationary_history.end(),
                    [&](const Track &known){return known.id==old.id;}))stationary_history.push_back(old);
              for(const auto &old:stationary_history)if(old.id && old.shape.circle) {
                const auto &c=*old.shape.circle;const double age=stamp-old.stamp;
                const double maximum_age=turning_identity_reacquisition?identity_memory_seconds:coast_seconds;
                // Positive NEW multi-scan stationary geometry, not absence or
                // an expired predicted velocity, may close an identity gap.
                // Check physically possible braking as well as position.
                const double reserve=1.5*resolution+.25*age+old.velocity_error*age+
                  .5*acceleration_bound_mps2*age*age;
                if(age>0 && age<=maximum_age && std::abs(c.radius-now_circle.radius)<=2*resolution &&
                   (!turning_identity_reacquisition || std::hypot(old.vx,old.vy)<=
                     acceleration_bound_mps2*age+old.velocity_error+.20) &&
                   std::hypot(c.center[0]+old.vx*age-now_circle.center[0],
                              c.center[1]+old.vy*age-now_circle.center[1])<=reserve)
                  resolved_.push_back(Track{current[i],0,0,stamp,true,old.id});
              }
              continue;
            }
          }
        }
        const bool stationary_interval=pair_dt>=.15 && current[i].lattice==reference->lattice &&
          std::all_of(observations_.begin(),observations_.end(),[&](const Observation &o) {
            return std::any_of(o.components.begin(),o.components.end(),[&](const Component &c) {
              return c.lattice==current[i].lattice;
            });
          });
        if(stationary_interval) {
          for(const auto &old:recent_confirmed_) {
            const auto a=current[i].lattice.size(),b=old.shape.lattice.size();
            if(!old.id || a<.5*b || a>2*b)continue;
            size_t overlap=0;for(auto cell:current[i].lattice)overlap+=old.shape.lattice.count(cell);
            if(overlap>=.5*std::max(a,b)) {
              Track resolved{current[i],0,0,stamp,true,old.id};resolved_.push_back(std::move(resolved));
            }
          }
        }
        if(!observations_.empty()) {
          const auto &latest=observations_.back();
          const Component *recent=nullptr;double distance=std::numeric_limits<double>::infinity();
          for(const auto &component:latest.components) {
            const double ratio=double(current[i].cells.size())/component.cells.size();
            const double d=std::hypot(current[i].x-component.x,current[i].y-component.y);
            if(ratio>=.25 && ratio<=4 && d<distance){distance=d;recent=&component;}
          }
          if(recent && current[i].lattice==recent->lattice) {
            // Pending evidence has not earned a coast. A completely repeated
            // shape cannot bridge confirmation if its proposed displacement
            // over this interval exceeds half a cell. Subcell translations
            // remain unresolved; a partial/missing shape is not this veto.
            candidates_.erase(std::remove_if(candidates_.begin(),candidates_.end(),[&](const Track &old) {
              if(std::hypot(old.vx,old.vy)*(stamp-latest.stamp)<=.5*resolution)return false;
              size_t overlap=0;
              for(const auto cell:recent->lattice)overlap+=old.shape.lattice.count(cell);
              return overlap>=.5*std::min(recent->lattice.size(),old.shape.lattice.size());
            }),candidates_.end());
            // Reusing an OLD translation would create stale brake requests.
            // A fast track that should have crossed >1.5 cells is disproved
            // by a fully repeated shape. Subcell-speed tracks may only coast.
            tracks_.erase(std::remove_if(tracks_.begin(),tracks_.end(),[&](const Track &track) {
              return track.shape.lattice==recent->lattice &&
                std::hypot(track.vx,track.vy)*(stamp-latest.stamp)>1.5*resolution;
            }),tracks_.end());
            continue;
          }
          // Generic visible subsets need immediate registration to reject
          // reversals. Curved bodies with a qualified center history use the
          // object estimator below rather than averaging arbitrary surfaces.
          if(recent && distance>=1.5*resolution) {
            reference=recent;reference_lattice=&latest.lattice;
            reference_window=latest.window;pair_dt=stamp-latest.stamp;
          }
        }
        const double vx=(current[i].x-reference->x)/pair_dt,vy=(current[i].y-reference->y)/pair_dt;
        const double speed=std::hypot(vx,vy);
        // Tangential displacement of a short visible wall fragment is not
        // observable motion: a world-fixed line explains it equally well.
        // Require fixed returns on BOTH sides, outside the fragment itself.
        // Wide objects and motion across the line are not suppressed. This
        // only qualifies velocity evidence; raw collision geometry is intact.
        bool fixed_line=false,clipped_line=false,ambiguous_tangent=false;
        if(speed>=.35) {
          const double ux=vx/speed,uy=vy/speed;bool thin=true;
          double low=0,high=0;
          for(const auto cell:current[i].lattice) {
            const double x=(static_cast<int32_t>(cell>>32)+.5)*resolution-current[i].x;
            const double y=(static_cast<int32_t>(cell&0xffffffffU)+.5)*resolution-current[i].y;
            if(std::abs(ux*y-uy*x)>.51*resolution){thin=false;break;}
            const double along=ux*x+uy*y;low=std::min(low,along);high=std::max(high,along);
          }
          if(thin) {
            const double left=std::max(min_x,reference_window[0]);
            const double bottom=std::max(min_y,reference_window[1]);
            const double right=std::min(min_x+width*resolution,reference_window[2]);
            const double top=std::min(min_y+height*resolution,reference_window[3]);
            // A thin fragment lacks identifiable tangent features if the
            // registration neighborhood is cut off. Use the maximum admitted
            // interval (.4s) and translation speed (4.5m/s), not a fitted speed
            // from the very centroid whose observability is in question.
            const double reach=4.5*.4+resolution;
            const double dx=std::abs(ux)*reach,dy=std::abs(uy)*reach;
            clipped_line=current[i].x-dx<left || current[i].x+dx>right ||
              current[i].y-dy<bottom || current[i].y+dy>top;
            const auto supports_line=[&](const auto &lattice) {
              std::unordered_set<int> before,after;
              for(const auto cell:lattice) {
                const double x=(static_cast<int32_t>(cell>>32)+.5)*resolution-current[i].x;
                const double y=(static_cast<int32_t>(cell&0xffffffffU)+.5)*resolution-current[i].y;
                const double along=ux*x+uy*y;
                if(std::abs(ux*y-uy*x)>.51*resolution || std::abs(along)>20*resolution)continue;
                // Several transverse wall returns at the same longitudinal
                // position are not an extension along the motion axis.
                const int longitudinal=static_cast<int>(std::llround(along/resolution));
                if(along<low-resolution)before.insert(longitudinal);
                if(along>high+resolution)after.insert(longitudinal);
              }
              return before.size()>=2 && after.size()>=2;
            };
            // Sampling locations may change while the supporting world line
            // stays fixed. Exact shared cells are not required, but both
            // independently observed frames must support both extensions.
            fixed_line=supports_line(current_lattice)&&supports_line((*reference_lattice));
          }
          // The velocity direction itself is noisy for tiny wall fragments:
          // alternating returns in adjacent raster rows add a spurious normal
          // velocity. Fit the nearby returns instead of fitting that velocity.
          // This is an observability veto, NOT stationary evidence: it may not
          // delete an already confirmed coast or erase current occupancy.
          double mx=0,my=0,xx=0,xy=0,yy=0,lx=ux,ly=uy;int n=0;
          const double context_radius=std::max(20*resolution,4.5*.4+2*resolution);
          // One refinement removes the directional selection bias of the
          // initial noisy velocity band. The radius and band stay bounded.
          for(int pass=0;pass<2;++pass) {
            const double cx=mx,cy=my;mx=my=xx=xy=yy=0;n=0;
            const auto collect=[&](const auto &lattice) {
              for(const auto cell:lattice) {
                const double x=(static_cast<int32_t>(cell>>32)+.5)*resolution-current[i].x;
                const double y=(static_cast<int32_t>(cell&0xffffffffU)+.5)*resolution-current[i].y;
                if(std::hypot(x,y)>context_radius || std::abs(lx*(y-cy)-ly*(x-cx))>2*resolution)continue;
                ++n;mx+=x;my+=y;xx+=x*x;xy+=x*y;yy+=y*y;
              }
            };
            collect(current_lattice);collect((*reference_lattice));
            if(n<8)break;
            mx/=n;my/=n;xx=xx/n-mx*mx;xy=xy/n-mx*my;yy=yy/n-my*my;
            const double angle=.5*std::atan2(2*xy,xx-yy);lx=std::cos(angle);ly=std::sin(angle);
          }
          if(n>=8) {
            const double disc=std::hypot(xx-yy,2*xy),major=.5*(xx+yy+disc),minor=.5*(xx+yy-disc);
            // A one-cell normal change can be raster sampling, not motion.
            // Larger normal translations remain observable by this guard.
            if(major>=9*std::max(minor,1e-12) && std::abs(lx*ux+ly*uy)>=.7071067811865476 &&
               std::abs(lx*vy-ly*vx)*pair_dt<=1.01*resolution) {
              double low=std::numeric_limits<double>::infinity(),high=-low;
              bool in_band=true;
              for(const auto cell:current[i].lattice) {
                const double x=(static_cast<int32_t>(cell>>32)+.5)*resolution-current[i].x-mx;
                const double y=(static_cast<int32_t>(cell&0xffffffffU)+.5)*resolution-current[i].y-my;
                if(std::abs(lx*y-ly*x)>1.01*resolution){in_band=false;break;}
                const double along=lx*x+ly*y;low=std::min(low,along);high=std::max(high,along);
              }
              const auto extended=[&](const auto &lattice) {
                std::unordered_set<int> before,after;
                for(const auto cell:lattice) {
                  const double x=(static_cast<int32_t>(cell>>32)+.5)*resolution-current[i].x-mx;
                  const double y=(static_cast<int32_t>(cell&0xffffffffU)+.5)*resolution-current[i].y-my;
                  const double along=lx*x+ly*y;
                  if(std::abs(lx*y-ly*x)>1.01*resolution || std::abs(along)>context_radius)continue;
                  const int bin=static_cast<int>(std::llround(along/resolution));
                  if(along<low-resolution)before.insert(bin);
                  if(along>high+resolution)after.insert(bin);
                }
                // A wall endpoint has support on ONE side. Its varying
                // visible subset still cannot establish tangent motion.
                // Treat this as ambiguity, not positive stationary proof;
                // a prior confirmed coast is therefore left untouched.
                return (before.size()>=2 && after.size()>=2) ||
                  before.size()>=5 || after.size()>=5;
              };
              const double reach=4.5*.4+resolution;
              const double left=std::max(min_x,reference_window[0]);
              const double bottom=std::max(min_y,reference_window[1]);
              const double right=std::min(min_x+width*resolution,reference_window[2]);
              const double top=std::min(min_y+height*resolution,reference_window[3]);
              const bool context_clipped=current[i].x-std::abs(lx)*reach<left ||
                current[i].x+std::abs(lx)*reach>right ||
                current[i].y-std::abs(ly)*reach<bottom || current[i].y+std::abs(ly)*reach>top;
              ambiguous_tangent=in_band && (context_clipped ||
                (extended(current_lattice) && extended((*reference_lattice))));
            }
          }
        }
        // A changing visible subset of a static wall can move its centroid.
        // Translation must explain shape overlap better than zero motion.
        const int sx=static_cast<int>(std::round(vx*pair_dt/resolution));
        const int sy=static_cast<int>(std::round(vy*pair_dt/resolution));
        int unchanged=0,translated=0;
        for(int cell:current[i].cells) {
          const int gx=static_cast<int>(std::floor((min_x+(cell%width+.5)*resolution)/resolution));
          const int gy=static_cast<int>(std::floor((min_y+(cell/width+.5)*resolution)/resolution));
          unchanged+=(*reference_lattice).count(key(gx,gy));
          translated+=(*reference).lattice.count(key(gx-sx,gy-sy));
        }
        const double count=std::min(current[i].cells.size(),(*reference).cells.size());
        const bool centroid_strict=!fixed_line && !clipped_line && !ambiguous_tangent && speed>=.5 && speed<=4.5 && unchanged<.8*count &&
          translated>=.5*count && translated>=unchanged+.15*count;
        // A visible subset's centroid moves when an occlusion opens. Estimate
        // velocity from UNIQUE best shape registration instead; the centroid
        // only supplies this bounded search neighborhood. Its direction is
        // not a veto: occlusion can bias that direction too. A unique
        // match alone still cannot confirm motion. The zero-motion veto and
        // two independent, consistent strict observations remain required.
        bool registered=false;double registered_vx=0,registered_vy=0;
        if(!fixed_line && !clipped_line && !ambiguous_tangent && speed>=.35 && speed<=4.5 &&
            (centroid_strict || unchanged<=count-2)) {
          int best=unchanged,second=unchanged,best_x=0,best_y=0;
          for(int tx=sx-1;tx<=sx+1;++tx)for(int ty=sy-1;ty<=sy+1;++ty) {
            const double shift=std::hypot(tx,ty)*resolution;
            if(!shift || shift/pair_dt>4.5)continue;
            int overlap=0;
            for(const auto cell:current[i].lattice) {
              const int gx=static_cast<int32_t>(cell>>32),gy=static_cast<int32_t>(cell&0xffffffffU);
              overlap+=(*reference).lattice.count(key(gx-tx,gy-ty));
            }
            if(overlap>best) {second=best;best=overlap;best_x=tx;best_y=ty;}
            else second=std::max(second,overlap);
          }
          if(best>=.65*count && best>=unchanged+.15*count && best>second) {
            registered=true;registered_vx=best_x*resolution/pair_dt;registered_vy=best_y*resolution/pair_dt;
            if(speed<.5 || unchanged>=.8*count)
              early_brakes.push_back({current[i],registered_vx,registered_vy,stamp});
          }
        }
        // Ambiguous strong centroid evidence may request a stop, but cannot
        // furnish a point-valued velocity to authorize a maneuver.
        if(!registered && centroid_strict)early_brakes.push_back({current[i],vx,vy,stamp});
        if(ambiguous_tangent && !fixed_line && unchanged<.8*count)continue;
        if(clipped_line && !fixed_line)continue; // Missing features cannot cancel an already confirmed coast.
        if(fixed_line || speed<.5 || unchanged>=.8*count) {
          // A local zero-motion raster match may reject an unconfirmed
          // proposal, but cannot silently erase a confirmed identity. Its
          // multi-scan, mutually unique stationary resolution below is the
          // only positive close-out. Otherwise navigation sees a missing
          // conflict with NO resolution even while its measured body remains.
          const auto contradicted=[&](const Track &track) {
            int overlap=0;
            for(const auto cell:(*reference).lattice)overlap+=track.shape.lattice.count(cell);
            return overlap>=.5*std::min((*reference).lattice.size(),track.shape.lattice.size());
          };
          tracks_.erase(std::remove_if(tracks_.begin(),tracks_.end(),[&](const Track&track) {
            return !track.confirmed&&contradicted(track);
          }),tracks_.end());
          candidates_.erase(std::remove_if(candidates_.begin(),candidates_.end(),contradicted),candidates_.end());
          continue;
        }
        if(speed>4.5)continue;
        if(!registered)continue;
        // Two fine raster returns cannot distinguish a moving tiny body
        // from changing visibility/row quantization of a wall. Keep their
        // actual occupancy and NEW brake-only cue, but do not create an
        // enduring identity from sub-0.01m² generic registration support.
        // This metric threshold preserves the original two-cell .20m model;
        // qualified curved-centre fits have their separate observability test.
        if(count*resolution*resolution<.01-1e-12) {
          early_brakes.push_back({current[i],registered_vx,registered_vy,stamp});
          continue;
        }
        Track registered_track{current[i],registered_vx,registered_vy,stamp};
        // A curved shape whose center model was unavailable keeps the generic
        // raster velocity fallback, explicitly less certain than the fit.
        if(current[i].circle)registered_track.velocity_error=.5;
        detected.push_back(std::move(registered_track));
      }
    }
    // A visible surface's centroid changes as the ego backs away. Identity
    // must follow registered measured geometry, not that changing centroid.
    // Keep identity association separate from TWO strict velocity witnesses:
    // one new displacement after a gap cannot refresh the motion coast.
    const auto shape_consistent=[&](const Track &old,const Track &track,double maximum_age) {
      const double age=stamp-old.stamp;
      if(age<=0 || age>maximum_age+1e-8)return false;
      if(old.shape.circle && track.shape.circle) {
        // This is an ID-reuse veto. Unowned short-baseline proposals have
        // not earned a persistent velocity/ID; their existing independent
        // confirmation and full trajectory-risk checks remain authoritative.
        if(old.id&&old.center_motion_fit&&track.center_motion_fit&&
           !motionVelocityIdentityCompatible(old.vx,old.vy,old.velocity_error,
             track.vx,track.vy,track.velocity_error,age,acceleration_bound_mps2))return false;
        const auto &a=*old.shape.circle,&b=*track.shape.circle;
        const double fit_reserve=(a.retained_radius||b.retained_radius)?a.center_error_bound+b.center_error_bound:0.;
        return std::abs(a.radius-b.radius)<=2*resolution &&
          std::hypot(a.center[0]+old.vx*age-b.center[0],a.center[1]+old.vy*age-b.center[1])<=
            motionIdentityRadius(resolution,age,old.velocity_error,acceleration_bound_mps2)+fit_reserve;
      }
      const double a=std::hypot(old.vx,old.vy),b=std::hypot(track.vx,track.vy);
      if(old.vx*track.vx+old.vy*track.vy<.8*a*b ||
         std::hypot(old.vx-track.vx,old.vy-track.vy)>std::max(1.,.5*std::max(a,b)))return false;
      const auto n=old.shape.lattice.size(),m=track.shape.lattice.size();
      if(!n || !m || m<.5*n || m>2*n)return false;
      const int sx=std::lround(old.vx*age/resolution),sy=std::lround(old.vy*age/resolution);
      const double radius=motionAssociationRadius(resolution,age);
      const int span=static_cast<int>(std::ceil(radius/resolution))+1;
      size_t overlap=0;
      for(int dx=-span;dx<=span;++dx)for(int dy=-span;dy<=span;++dy) {
        // Bound the actual shift, including fractional-cell quantization.
        if(std::hypot((sx+dx)*resolution-old.vx*age,
                      (sy+dy)*resolution-old.vy*age)>radius)continue;
        size_t match=0;
        for(const auto cell:track.shape.lattice)
          match+=old.shape.lattice.count(key(static_cast<int32_t>(cell>>32)-sx-dx,
                                            static_cast<int32_t>(cell&0xffffffffU)-sy-dy));
        overlap=std::max(overlap,match);
      }
      return overlap>=.65*std::min(n,m);
    };
    // A split/merge or two compatible objects must not donate an arbitrary
    // first ID. Mutual uniqueness is required across ALL strict detections.
    std::vector<std::vector<uint64_t>> identities(detected.size());
    for(size_t i=0;i<detected.size();++i) {
      std::vector<uint64_t> directed_identities;
      for(const auto &old:recent_confirmed_)
        if(old.id && shape_consistent(old,detected[i],coast_seconds))identities[i].push_back(old.id);
      const auto &track=detected[i];
      for(const auto &old:dormant_identities_) {
        const double age=stamp-old.stamp;
        if(age<=coast_seconds || age>identity_memory_seconds || !old.id || !old.center_motion_fit || !track.center_motion_fit ||
            !old.shape.circle || !track.shape.circle)continue;
        const auto &a=*old.shape.circle,&b=*track.shape.circle;
        if(std::abs(a.radius-b.radius)>2*resolution)continue;
        const double av=std::hypot(old.vx,old.vy),bv=std::hypot(track.vx,track.vy);
        if(av>4.5||bv>4.5||((av<.35||bv<.35)&&!track_known_slow_centers))continue;
        if(turning_identity_reacquisition) {
          // |v_new-v_old| <= a*dt, including both fitted velocity reserves.
          // A fixed heading dot product rules out a physically admissible
          // 90-degree turn regardless of how much causal time elapsed.
          if(std::hypot(track.vx-old.vx,track.vy-old.vy)>
             acceleration_bound_mps2*age+old.velocity_error+track.velocity_error)continue;
        } else if(old.vx*track.vx+old.vy*track.vy<.8*av*bv||
            std::abs(av-bv)>std::max(.5,.5*std::max(av,bv)))continue;
        const double gate=1.5*resolution+.25*age+old.velocity_error*age+.5*acceleration_bound_mps2*age*age;
        if(std::hypot(a.center[0]+old.vx*age-b.center[0],a.center[1]+old.vy*age-b.center[1])>gate ||
            std::hypot(b.center[0]-track.vx*age-a.center[0],b.center[1]-track.vy*age-a.center[1])>gate)continue;
        identities[i].push_back(old.id);
        if(turning_identity_reacquisition) {
          // Preserve a unique continuously directed translation before
          // considering broad acceleration/turn alternatives. A distant
          // weak identity's ever-growing gate must not steal a fresh object
          // from its matching directed track. This is a conditional geometric
          // association, not an absolute semantic-identity probability.
          const double directed_gate=1.5*resolution+.25*age+old.velocity_error*age;
          if(old.vx*track.vx+old.vy*track.vy>=.8*av*bv &&
             std::abs(av-bv)<=std::max(.5,.5*std::max(av,bv)) &&
             std::hypot(a.center[0]+old.vx*age-b.center[0],a.center[1]+old.vy*age-b.center[1])<=directed_gate &&
             std::hypot(b.center[0]-track.vx*age-a.center[0],b.center[1]-track.vy*age-a.center[1])<=directed_gate)
            directed_identities.push_back(old.id);
        }
      }
      std::sort(identities[i].begin(),identities[i].end());
      identities[i].erase(std::unique(identities[i].begin(),identities[i].end()),identities[i].end());
      // No older, wider history is allowed to override contradictory NEWER
      // measured velocity evidence for the same identity. History remembers
      // labels; it is neither a new observation nor a reason to delete one.
      identities[i].erase(std::remove_if(identities[i].begin(),identities[i].end(),[&](uint64_t id) {
        const Track *latest=nullptr;
        const auto newest=[&](const auto &history) {
          for(const auto &old:history)if(old.id==id&&(!latest||old.stamp>latest->stamp))latest=&old;
        };
        newest(tracks_);newest(recent_confirmed_);newest(dormant_identities_);newest(stationary_identity_history_);
        const bool velocity_mismatch=latest&&latest->center_motion_fit&&track.center_motion_fit&&
          latest->shape.circle&&track.shape.circle&&
          !motionVelocityIdentityCompatible(latest->vx,latest->vy,latest->velocity_error,
            track.vx,track.vy,track.velocity_error,stamp-latest->stamp,acceleration_bound_mps2);
        if(velocity_mismatch)return true;
        const auto saved=gaussian_history_.find(id);
        if(!fit_gaussian_motion||!track.center_motion_fit||!track.gaussian.valid||
           saved==gaussian_history_.end()||!validGaussianMotion(saved->second)||
           stamp-saved->second.stamp<=coast_seconds+1e-6||
           track.gaussian.constant_acceleration!=saved->second.constant_acceleration||
           track.gaussian.process_density!=saved->second.process_density||
           track.gaussian.longitudinal_acceleration_bound!=saved->second.longitudinal_acceleration_bound)
          return false;
        // This gate rejects reuse only. Actual occupancy and unowned motion
        // hypotheses remain; absence/rejection never resolves a conflict.
        return !updateGaussianCenterWindow(saved->second,track.center_observations,
          track.gaussian.observation_std,13.815510557964274,identity_memory_seconds,3).valid;
      }),identities[i].end());
      if(turning_identity_reacquisition && !directed_identities.empty()) {
        for(const auto &old:recent_confirmed_)
          if(old.id && shape_consistent(old,track,coast_seconds))directed_identities.push_back(old.id);
        std::vector<uint64_t> preferred;
        for(const auto id:identities[i])
          if(std::find(directed_identities.begin(),directed_identities.end(),id)!=directed_identities.end())
            preferred.push_back(id);
        // An inconsistent directed candidate must not discard an admissible
        // turning identity before the temporal evidence has been checked.
        if(!preferred.empty())identities[i]=std::move(preferred);
      }
    }
    for(size_t i=0;i<detected.size();++i) {
      auto &track=detected[i];
      bool confirmed=false;
      for(const auto &old:recent_confirmed_)confirmed|=shape_consistent(old,track,.4);
      for(const auto &old:candidates_)confirmed|=shape_consistent(old,track,.4);
      if(!confirmed)continue;
      uint64_t identity=0;
      if(identities[i].empty()) {
        // A weaker view inside an unresolved physical body's envelope is
        // not positive evidence for a SECOND object. Retain the raw shape
        // and current provisional motion, without inventing another ID that
        // will later remain unresolved after the original object stops.
        const bool ambiguous_birth=std::any_of(dormant_identities_.begin(),dormant_identities_.end(),[&](const Track &old) {
          const double age=stamp-old.stamp;
          if(age<=0||age>identity_memory_seconds||!old.shape.circle)return false;
          const auto &c=*old.shape.circle;
          const double gate=1.5*resolution+.25*age+old.velocity_error*age+.5*acceleration_bound_mps2*age*age;
          const double x=c.center[0]+old.vx*age,y=c.center[1]+old.vy*age;
          for(const auto cell:track.shape.lattice) {
            const double px=(static_cast<int32_t>(cell>>32)+.5)*resolution,
                         py=(static_cast<int32_t>(cell&0xffffffffU)+.5)*resolution;
            if(std::hypot(px-x,py-y)<=c.conservative_radius+gate)return true;
          }
          return false;
        });
        if(ambiguous_birth)continue;
      }
      if(!identities[i].empty()) {
        if(identities[i].size()!=1)continue;
        identity=identities[i].front();
        size_t claims=0;
        for(const auto &matches:identities)
          claims+=std::find(matches.begin(),matches.end(),identity)!=matches.end();
        if(claims!=1)continue;
      }
      track.confirmed=true;track.id=identity?identity:next_id_++;
      if(identity && track.shape.circle) {
        // Association is already mutually unique. Use only the most recent
        // actual posterior of THIS identity, with one new measured centre.
        // A gap, invalid fit or rejected innovation retains the fresh local
        // fit above; no covariance or timestamp is copied across identities.
        const auto saved=gaussian_history_.find(identity);
        const GaussianMotion *prior=saved==gaussian_history_.end()?nullptr:&saved->second;
        if(fit_gaussian_motion && prior && track.gaussian.valid && validGaussianMotion(*prior) &&
           track.gaussian.constant_acceleration==prior->constant_acceleration &&
           track.gaussian.process_density==prior->process_density &&
           track.gaussian.longitudinal_acceleration_bound==prior->longitudinal_acceleration_bound) {
          const auto updated=updateGaussianCenterWindow(*prior,track.center_observations,
            track.gaussian.observation_std,13.815510557964274,identity_memory_seconds,
            stamp-prior->stamp>coast_seconds+1e-6?3:1);
          if(updated.valid)track.gaussian=updated;
        }
        // A different visible arc does not prove that the SAME rigid body
        // shrank. Preserve its previously observed collision extent only
        // AFTER mutually unique identity association. Never donate size
        // across ambiguous/new IDs, refresh a timestamp, or invent returns.
        // This remains a conditional rigid-circle enclosure, not a bound on
        // an arbitrary unseen object or on unmodelled centre-fit bias.
        const auto retain_extent=[&](const auto &history) {
          for(const auto &old:history)if(old.id==identity && old.shape.circle)
            track.shape.circle->conservative_radius=std::max(
              track.shape.circle->conservative_radius,old.shape.circle->conservative_radius);
        };
        retain_extent(recent_confirmed_);retain_extent(dormant_identities_);
      }
      // Nearby objects are not evidence that an earlier identity vanished.
      tracks_.erase(std::remove_if(tracks_.begin(),tracks_.end(),[&](const Track &old) {
        return old.id==track.id;
      }),tracks_.end());
      tracks_.push_back(track);
      if(track_known_slow_centers&&track.center_motion_fit&&track.shape.circle) {
        auto&history=measured_rest_history_[track.id];
        if(!history.empty()&&(stamp<=history.back().stamp_s||resolution!=history.back().resolution))history.clear();
        const auto&circle=*track.shape.circle;
        history.push_back({stamp,circle.center,circle.radius,resolution});
        while(!history.empty()&&stamp-history.front().stamp_s>1.+1e-6)history.erase(history.begin());
        if(measuredStationaryCenters(history))resolved_.push_back(Track{track.shape,0,0,stamp,true,track.id});
      }
    }
    // Stationary evidence must identify ONE previous object mutually. A
    // widened physical association gate must not let two nearby stationary
    // bodies resolve the same identity, or one body resolve two identities.
    std::vector<Track> unique_resolutions;
    for(const auto &candidate:resolved_) {
      bool ambiguous=stamp<=resolution_history_overflow_until_;
      for(const auto &known:stationary_identity_history_)if(known.id!=candidate.id) {
        bool same_body=candidate.shape.lattice==known.shape.lattice;
        if(candidate.shape.circle && known.shape.circle) {
          const auto &a=*candidate.shape.circle,&b=*known.shape.circle;
          same_body|=std::abs(a.radius-b.radius)<=2*resolution &&
            std::hypot(a.center[0]-b.center[0],a.center[1]-b.center[1])<=1.5*resolution;
        }
        ambiguous|=same_body;
      }
      for(const auto &other:resolved_) {
        const bool same_shape=candidate.shape.lattice==other.shape.lattice;
        if((same_shape && candidate.id!=other.id) || (!same_shape && candidate.id==other.id))ambiguous=true;
      }
      if(!ambiguous && std::none_of(unique_resolutions.begin(),unique_resolutions.end(),
          [&](const Track &r){return r.id==candidate.id;}))unique_resolutions.push_back(candidate);
    }
    resolved_=std::move(unique_resolutions);
    if(fit_gaussian_motion&&(gaussian_constant_acceleration||track_known_slow_centers))for(auto &r:resolved_) {
      if(!r.id||!r.shape.circle)continue;
      const Track *prior=nullptr;
      const auto newest=[&](const auto &history) {
        for(const auto &old:history)if(old.id==r.id&&old.shape.circle&&(!prior||old.stamp>prior->stamp))prior=&old;
      };
      newest(recent_confirmed_);newest(dormant_identities_);
      const auto saved=gaussian_history_.find(r.id);
      if(!prior||saved==gaussian_history_.end()||
         saved->second.constant_acceleration!=gaussian_constant_acceleration||!validGaussianMotion(saved->second))continue;
      const auto &circle=*r.shape.circle;
      const double sigma=sim_observation_calibration&&simRepeatabilityDomain(circle,resolution)?
        sim_observation_std:gaussian_observation_std;
      const auto posterior=updateGaussianCenterMotion(saved->second,
        TimedCenter{stamp,circle.center,circle.radius,resolution},sigma,13.815510557964274,identity_memory_seconds);
      if(!posterior.valid)continue;
      r.gaussian=posterior;r.center_motion_fit=true;r.velocity_error=prior->velocity_error;
      r.shape.circle->conservative_radius=std::max(r.shape.circle->conservative_radius,prior->shape.circle->conservative_radius);
    }
    for(const auto &r:resolved_)stationary_identity_history_.push_back(r);
    if(stationary_identity_history_.size()>1024) {
      // Storage pressure loses qualification, never invents a new resolution.
      resolution_history_overflow_until_=stamp+stationary_ownership_seconds;
      stationary_identity_history_.clear();resolved_.clear();
    }
    const auto resolved=[&](const Track &t) {
      return t.id && std::any_of(resolved_.begin(),resolved_.end(),[&](const Track &r){return r.id==t.id;});
    };
    tracks_.erase(std::remove_if(tracks_.begin(),tracks_.end(),resolved),tracks_.end());
    const auto unmodeled_resolution=[&](const Track &t) {
      if(!resolved(t))return false;
      return !fit_gaussian_motion||(!gaussian_constant_acceleration&&!track_known_slow_centers)||
        std::none_of(resolved_.begin(),resolved_.end(),[&](const Track &r) {
        return r.id==t.id&&r.gaussian.constant_acceleration==gaussian_constant_acceleration&&validGaussianMotion(r.gaussian);
      });
    };
    recent_confirmed_.erase(std::remove_if(recent_confirmed_.begin(),recent_confirmed_.end(),unmodeled_resolution),recent_confirmed_.end());
    dormant_identities_.erase(std::remove_if(dormant_identities_.begin(),dormant_identities_.end(),unmodeled_resolution),dormant_identities_.end());
    // A confirmed stop changes observed velocity, not physical identity.
    // Retain the configured CV/CA posterior only after this NEW unambiguous measured body
    // update. Missing returns, expired velocities and arbitrary static posts
    // never enter this path or refresh the source clock.
    if(fit_gaussian_motion&&(gaussian_constant_acceleration||track_known_slow_centers))
      for(const auto &r:resolved_)if(r.gaussian.constant_acceleration==gaussian_constant_acceleration&&validGaussianMotion(r.gaussian)) {
      for(auto *memory:{&recent_confirmed_,&dormant_identities_}) {
        auto old=std::find_if(memory->begin(),memory->end(),[&](const Track &t){return t.id==r.id;});
        if(old==memory->end())memory->push_back(r);else if(r.stamp>old->stamp)*old=r;
      }
    }
    for(const auto &track:tracks_) {
      auto old=std::find_if(recent_confirmed_.begin(),recent_confirmed_.end(),[&](const Track &t){return t.id==track.id;});
      if(old==recent_confirmed_.end())recent_confirmed_.push_back(track);
      else if(track.stamp>old->stamp)*old=track;
      if(track.id && track.confirmed && track.center_motion_fit && track.shape.circle) {
        auto memory=std::find_if(dormant_identities_.begin(),dormant_identities_.end(),[&](const Track &t){return t.id==track.id;});
        if(memory==dormant_identities_.end())dormant_identities_.push_back(track);
        else if(track.stamp>memory->stamp)*memory=track;
      }
    }
    if(recent_confirmed_.size()>1024)recent_confirmed_.erase(recent_confirmed_.begin(),recent_confirmed_.end()-1024);
    if(dormant_identities_.size()>1024)dormant_identities_.erase(dormant_identities_.begin(),dormant_identities_.end()-1024);
    const auto remember_posterior=[&](const Track &track) {
      if(!track.id||!track.shape.circle||!track.center_motion_fit||track.stamp!=stamp||
         track.gaussian.stamp!=track.stamp||!validGaussianMotion(track.gaussian))return;
      auto old=gaussian_history_.find(track.id);
      if(old==gaussian_history_.end())gaussian_history_.emplace(track.id,track.gaussian);
      else if(track.gaussian.stamp>old->second.stamp)old->second=track.gaussian;
    };
    for(const auto &track:tracks_)remember_posterior(track);
    for(const auto &track:resolved_)remember_posterior(track);
    while(gaussian_history_.size()>1024) {
      const auto oldest=std::min_element(gaussian_history_.begin(),gaussian_history_.end(),
        [](const auto &a,const auto &b){return a.second.stamp<b.second.stamp;});
      gaussian_history_.erase(oldest); // Losing a prior is never a resolution.
    }
    brake_candidates_=detected;
    brake_candidates_.insert(brake_candidates_.end(),early_brakes.begin(),early_brakes.end());
    for(auto &track:detected)candidates_.push_back(std::move(track));
    // Keep bounded storage even if a different sensor dispatches many scans
    // within .4s. Discarding old candidates can only postpone confirmation.
    if(candidates_.size()>1024)candidates_.erase(candidates_.begin(),candidates_.end()-1024);
    for(const auto &track:tracks_) {
        ++moving_components;
        const double speed=std::hypot(track.vx,track.vy),age=stamp-track.stamp;
        const double cap=predictionSpeedCap(speed,track.velocity_error,prediction_maximum_speed_mps);
        const double distance=boundedTranslationDistance(speed,age+horizon,acceleration_bound_mps2,cap);
        const int steps=std::max(1,static_cast<int>(std::ceil(distance/(.5*resolution))));
        // Start at the extrapolated present, not the old swept path. No input
        // other than a newly verified translation refreshes the coast timer.
        const int margin=static_cast<int>(std::ceil((.2+.25*age)/resolution));
        for(int step=0;step<=steps;++step) {
          const double t=age+horizon*step/steps;
          const double travel=boundedTranslationDistance(speed,t,acceleration_bound_mps2,cap);
          for(uint64_t cell:track.shape.lattice) {
            const int gx=static_cast<int32_t>(cell>>32),gy=static_cast<int32_t>(cell&0xffffffffU);
            const int x=static_cast<int>(std::floor(((gx+.5)*resolution+(speed>1e-9?track.vx/speed*travel:0.)-min_x)/resolution));
            const int y=static_cast<int>(std::floor(((gy+.5)*resolution+(speed>1e-9?track.vy/speed*travel:0.)-min_y)/resolution));
            // Engineering association margin, not a probabilistic bound.
            for(int oy=-margin;oy<=margin;oy++)for(int ox=-margin;ox<=margin;ox++) {
              const int nx=x+ox,ny=y+oy;
              if(nx>=0 && ny>=0 && nx<width && ny<height)forecast[ny*width+nx]=1;
            }
          }
        }
    }
    // Raw association history is replaced every scan, including empty input.
    observations_.push_back({current,current_lattice,
        {min_x,min_y,min_x+width*resolution,min_y+height*resolution},stamp});
    if(observations_.size()>32)observations_.erase(observations_.begin());
    previous_=std::move(current);previous_lattice_=std::move(current_lattice);
    previous_window_={min_x,min_y,min_x+width*resolution,min_y+height*resolution};
    stamp_=stamp;resolution_=resolution;
    return forecast;
  }
};
}  // namespace terrain_nav
