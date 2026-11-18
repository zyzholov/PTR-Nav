#pragma once
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include "terrain_nav/body_kinematics.hpp"

namespace terrain_nav {
// Optimistic global ROUTING ONLY. Unknown cells may be considered while
// choosing where to observe next. This class never certifies a vehicle pose,
// changes terrain probabilities, or permits executing an unknown path.
struct RouteOptions {
  double rear_axle_offset=0.;
  double heuristic_weight=1.;
  double max_initial_reverse=std::numeric_limits<double>::infinity();
  bool single_reverse_then_forward=false;
  bool allow_partial=false;
  bool allow_advisory_start_overlap=false;
  // Zero retains legacy hard historical geometry. A positive value treats
  // explicitly ADVISORY history as a finite per-metre routing preference.
  // Actual occupied evidence is always hard; this grants no execution.
  double advisory_cost=0.;
  // Heading-relaxed search cost only. It cannot mark occupancy free or
  // authorize execution. Use the same terrain scale as the car primitives.
  std::function<double(double,double)> heuristic_cost;
  // Finite routing preference for reverse terrain that cannot currently be
  // supported. This never certifies it, clears occupancy, or changes a bound.
  std::function<double(double,double,double)> reverse_observation_cost;
};
class RouteMemory {
public:
  using Point=std::array<double,2>;
  // Three-field legacy evidence is square; an explicit second half extent
  // retains a measured rectangle without expanding it to a bounding square.
  struct Square {double x,y,half;double half_y=-1.;};
  struct Result {std::vector<Point> path;std::vector<double> yaws;std::vector<int> directions;
    int expanded=0;bool reached=false;bool partial=false;bool fine=false;bool heuristic_reused=false;bool quantized_heuristic_reused=false;
    double preprocessing_ms=0;const char *reason="invalid";};
private:
  struct Bounds {double x0=0,y0=0,x1=0,y1=0;bool valid=false;};
  using Mask=std::array<uint32_t,32>;
  struct Cell {bool occupied=false;double stamp=0;bool advisory=false;std::vector<Bounds> hard_bounds,advisory_bounds;
    std::optional<Mask> hard_mask,advisory_mask;};
  std::unordered_map<uint64_t,Cell> cells_;
  double stamp_=-1;
  double resolution_=.2;
  struct HeuristicCache {
    std::vector<double> distance;
    std::vector<uint64_t> occupied;
    std::vector<uint64_t> advisory;
    std::vector<uint64_t> advisory_relief;
    std::vector<float> terrain_cost;
    int lowx=0,lowy=0,width=0,height=0,gx=0,gy=0;
    int relief_x=0,relief_y=0;bool source_relief=false;
    double radius=0,stamp=-1,advisory_cost=0.;
    std::chrono::steady_clock::time_point built{};
  };
  mutable HeuristicCache heuristic_cache_;
  static uint64_t key(int x,int y) {return (uint64_t(static_cast<uint32_t>(x))<<32)|static_cast<uint32_t>(y);}
  static int ix(uint64_t key) {return static_cast<int32_t>(key>>32);}
  static int iy(uint64_t key) {return static_cast<int32_t>(key&0xffffffffU);}
  int index(double value) const {return static_cast<int>(std::floor(value/resolution_));}
  // Repeated, overlapping sensor rectangles need not have identical edges.
  // The old 64-fragment overflow filled the WHOLE hash cell, progressively
  // closing a passage even though no new occupied surface was observed.
  // Sweep the exact union instead: vertical slabs have disjoint y intervals,
  // and equal adjacent intervals are joined without adding a bounding hull.
  static void markMask(Mask &mask,const Bounds &b,double cell_x,double cell_y,double resolution) {
    const double step=resolution/32.;
    const int x0=std::clamp(int(std::floor((b.x0-cell_x)/step)),0,31);
    const int x1=std::clamp(int(std::ceil((b.x1-cell_x)/step))-1,0,31);
    const int y0=std::clamp(int(std::floor((b.y0-cell_y)/step)),0,31);
    const int y1=std::clamp(int(std::ceil((b.y1-cell_y)/step))-1,0,31);
    for(int y=y0;y<=y1;++y)for(int x=x0;x<=x1;++x)mask[y]|=uint32_t(1)<<x;
  }
  static std::vector<Bounds> maskBounds(const Mask &mask,double cell_x,double cell_y,double resolution) {
    std::vector<Bounds> result;const double step=resolution/32.;
    for(int y=0;y<32;++y)for(int x=0;x<32;) {
      if(!(mask[y]&(uint32_t(1)<<x))){++x;continue;}
      const int x0=x++;while(x<32&&(mask[y]&(uint32_t(1)<<x)))++x;
      result.push_back({std::nextafter(cell_x+x0*step,-INFINITY),
        std::nextafter(cell_y+y*step,-INFINITY),std::nextafter(cell_x+x*step,INFINITY),
        std::nextafter(cell_y+(y+1)*step,INFINITY),true});
    }
    return result;
  }
  static void compactBounds(std::vector<Bounds> &bounds,std::optional<Mask> &raster,
      double cell_x,double cell_y,double resolution) {
    if(bounds.size()<=64)return;
    std::vector<double> edges;edges.reserve(bounds.size()*2);
    for(const auto &b:bounds){edges.push_back(b.x0);edges.push_back(b.x1);}
    std::sort(edges.begin(),edges.end());edges.erase(std::unique(edges.begin(),edges.end()),edges.end());
    std::vector<Bounds> result;
    std::vector<std::pair<double,double>> intervals;
    for(size_t x=1;x<edges.size();++x) {
      const double x0=edges[x-1],x1=edges[x];if(x0>=x1)continue;
      intervals.clear();
      for(const auto &b:bounds)if(b.x0<x1 && b.x1>x0)intervals.emplace_back(b.y0,b.y1);
      std::sort(intervals.begin(),intervals.end());
      for(size_t i=0;i<intervals.size();) {
        const double y0=intervals[i].first;double y1=intervals[i++].second;
        while(i<intervals.size() && intervals[i].first<=y1){y1=std::max(y1,intervals[i].second);++i;}
        auto previous=std::find_if(result.rbegin(),result.rend(),[&](const Bounds &b){
          return b.x1==x0 && b.y0==y0 && b.y1==y1;
        });
        if(previous!=result.rend())previous->x1=x1;
        else result.push_back({x0,y0,x1,y1,true});
      }
    }
    if(result.size()<=256){bounds=std::move(result);return;}
    // Adversarial/staircase unions may remain fragmented. Outward quantize
    // onto a FIXED 32x32 lattice inside this hash cell, never its full hull.
    // All source rectangles remain covered; added extent is at most res/32
    // on each coordinate and cannot accumulate with successive compactions.
    raster=Mask{};
    for(const auto &b:bounds)markMask(*raster,b,cell_x,cell_y,resolution);
    bounds=maskBounds(*raster,cell_x,cell_y,resolution);
  }
  static bool representedWithinOneCell(const std::vector<uint64_t> &a,const std::vector<uint64_t> &b) {
    for(const auto &p:a) {
      if(std::binary_search(b.begin(),b.end(),p))continue;
      bool found=false;
      for(int dy=-1;dy<=1&&!found;++dy)for(int dx=-1;dx<=1&&!found;++dx)
        found=std::binary_search(b.begin(),b.end(),key(ix(p)+dx,iy(p)+dy));
      if(!found)return false;
    }
    return true;
  }
public:
  explicit RouteMemory(double resolution=.2):resolution_(resolution) {
    if(!std::isfinite(resolution) || resolution<.05 || resolution>1.)
      throw std::invalid_argument("invalid routing resolution");
  }
  void clear(){cells_.clear();stamp_=-1;heuristic_cache_=HeuristicCache{};}
  size_t size() const{return cells_.size();}
  std::vector<std::array<double,4>> hardRectangles()const {
    std::vector<std::array<double,4>> result;
    for(const auto &entry:cells_)if(entry.second.occupied) {
      if(entry.second.hard_bounds.empty())result.push_back({ix(entry.first)*resolution_,iy(entry.first)*resolution_,
        (ix(entry.first)+1)*resolution_,(iy(entry.first)+1)*resolution_});
      else for(const auto &b:entry.second.hard_bounds)result.push_back({b.x0,b.y0,b.x1,b.y1});
    }
    return result;
  }
  // A SECOND optimistic observation proposal, not a replacement occupancy
  // map. Preserve every current hard obstacle. Retained older geometry stays
  // in the original memory and becomes a finite prior ONLY in this copy.
  // No caller may execute this route without current full-pose certification.
  RouteMemory currentGeometryGuidance(const RouteMemory &current)const {
    if(current.resolution_!=resolution_)throw std::invalid_argument("routing resolution mismatch");
    auto result=current;result.heuristic_cache_=HeuristicCache{};
    for(const auto &entry:cells_) {
      if(!entry.second.occupied&&!entry.second.advisory)continue;
      auto &cell=result.cells_[entry.first];cell.advisory=true;
      cell.advisory_bounds.clear();cell.advisory_mask.reset();
    }
    return result;
  }
  bool occupiedAt(Point point) const {
    const auto found=cells_.find(key(index(point[0]),index(point[1])));
    return found!=cells_.end() && (found->second.occupied || found->second.advisory);
  }
  double guidanceCost(Point point,double base,double advisory_cost) const {
    if(!std::isfinite(point[0])||!std::isfinite(point[1])||!std::isfinite(base)||base<1.||
       !std::isfinite(advisory_cost)||advisory_cost<0.||advisory_cost>30.)return INFINITY;
    const auto found=cells_.find(key(index(point[0]),index(point[1])));
    return std::max(found==cells_.end()?3.:1.,base+
      (found!=cells_.end()&&found->second.advisory?advisory_cost:0.));
  }
  struct Evidence {Point position;bool occupied;bool advisory;};
  std::vector<Evidence> evidence() const {
    std::vector<Evidence> result;result.reserve(cells_.size());
    for(const auto &entry:cells_)result.push_back({
      {(ix(entry.first)+.5)*resolution_,(iy(entry.first)+.5)*resolution_},entry.second.occupied,entry.second.advisory});
    return result;
  }
  double resolution()const{return resolution_;}
  struct GeometryStats {size_t hard=0,whole=0,rectangles=0,raster=0;};
  GeometryStats geometryStats()const {
    GeometryStats result;
    for(const auto &entry:cells_)if(entry.second.occupied) {
      ++result.hard;result.whole+=entry.second.hard_bounds.empty();result.rectangles+=entry.second.hard_bounds.size();
      result.raster+=bool(entry.second.hard_mask);
    }
    return result;
  }
  void observe(double stamp,const std::array<double,4> &window,
               const std::vector<Point> &seen,const std::vector<Point> &occupied,
               const std::vector<Point> &confirmed_clear={},const std::vector<Point> &advisory={},
               bool refresh_advisory=false,const std::vector<Square> &occupied_squares={},
               const std::vector<Square> &advisory_squares={},const std::vector<Square> &clear_squares={}) {
    if(!std::isfinite(stamp) || !std::all_of(window.begin(),window.end(),[](double v){return std::isfinite(v);}) ||
       window[2]<=window[0] || window[3]<=window[1])throw std::invalid_argument("invalid routing observation");
    if(stamp<stamp_)clear();
    for(auto it=cells_.begin();it!=cells_.end();) {
      // Occlusion, a rolling-window shift and GP floor support are NOT
      // evidence that a wall disappeared. Retain occupied routing geometry
      // until explicit observation contradicts it (or the frame is reset).
      // Only the traversed/observed preference may expire with time.
      if(refresh_advisory) {
        const double x=(ix(it->first)+.5)*resolution_,y=(iy(it->first)+.5)*resolution_;
        if(x>=window[0] && x<window[2] && y>=window[1] && y<window[3]) {
          it->second.advisory=false;it->second.advisory_bounds.clear();it->second.advisory_mask.reset();
        }
      }
      if(!it->second.occupied && !it->second.advisory && stamp-it->second.stamp>120.)it=cells_.erase(it);else ++it;
    }
    const auto insert=[&](const auto &points,bool obstacle) {
      for(const auto &p:points) {
        if(!std::isfinite(p[0]) || !std::isfinite(p[1]) || std::max(std::abs(p[0]),std::abs(p[1]))>1e5)
          throw std::invalid_argument("invalid routing point");
        auto &c=cells_[key(index(p[0]),index(p[1]))];c.occupied|=obstacle;c.stamp=stamp;
        if(obstacle){c.hard_bounds.clear();c.hard_mask.reset();} // legacy evidence occupies the whole cell
      }
    };
    for(const auto &p:confirmed_clear) {
      if(!std::isfinite(p[0]) || !std::isfinite(p[1]) || std::max(std::abs(p[0]),std::abs(p[1]))>1e5)
        throw std::invalid_argument("invalid routing clearance");
      if(p[0]<window[0] || p[0]>=window[2] || p[1]<window[1] || p[1]>=window[3])continue;
      auto &c=cells_[key(index(p[0]),index(p[1]))];c.occupied=false;c.hard_bounds.clear();c.hard_mask.reset();c.stamp=stamp;
    }
    for(const auto &p:clear_squares) {
      if(!std::isfinite(p.x)||!std::isfinite(p.y)||!std::isfinite(p.half)||p.half<=0||p.half>1.)
        throw std::invalid_argument("invalid routing clear square");
      const Bounds free{p.x-p.half,p.y-p.half,p.x+p.half,p.y+p.half,true};
      if(free.x0<window[0]||free.y0<window[1]||free.x1>window[2]||free.y1>window[3])continue;
      for(int y=index(free.y0);y<=index(std::nextafter(free.y1,-INFINITY));++y)
        for(int x=index(free.x0);x<=index(std::nextafter(free.x1,-INFINITY));++x) {
          auto found=cells_.find(key(x,y));if(found==cells_.end()||!found->second.occupied)continue;
          auto &c=found->second;
          if(c.hard_mask) {
            const double step=resolution_/32.;
            for(int my=0;my<32;++my)for(int mx=0;mx<32;++mx) {
              const double bx=x*resolution_+mx*step,by=y*resolution_+my*step;
              // A partial free witness cannot erase an occupied microcell.
              if(bx>=free.x0&&by>=free.y0&&bx+step<=free.x1&&by+step<=free.y1)
                (*c.hard_mask)[my]&=~(uint32_t(1)<<mx);
            }
            c.hard_bounds=maskBounds(*c.hard_mask,x*resolution_,y*resolution_,resolution_);
            c.occupied=!c.hard_bounds.empty();c.stamp=stamp;
            if(!c.occupied)c.hard_mask.reset();
            continue;
          }
          auto source=c.hard_bounds;
          if(source.empty())source.push_back({x*resolution_,y*resolution_,(x+1)*resolution_,(y+1)*resolution_,true});
          std::vector<Bounds> remainder;
          for(const auto &b:source) {
            const double x0=std::max(b.x0,free.x0),y0=std::max(b.y0,free.y0),
                         x1=std::min(b.x1,free.x1),y1=std::min(b.y1,free.y1);
            if(x0>=x1||y0>=y1){remainder.push_back(b);continue;}
            // A 5cm floor witness cannot erase the other half of a 10cm
            // memory cell. Subtract only its actual witnessed rectangle.
            if(b.x0<x0)remainder.push_back({b.x0,b.y0,x0,b.y1,true});
            if(x1<b.x1)remainder.push_back({x1,b.y0,b.x1,b.y1,true});
            if(b.y0<y0)remainder.push_back({x0,b.y0,x1,y0,true});
            if(y1<b.y1)remainder.push_back({x0,y1,x1,b.y1,true});
          }
          c.occupied=!remainder.empty();c.hard_bounds=std::move(remainder);c.stamp=stamp;
          compactBounds(c.hard_bounds,c.hard_mask,x*resolution_,y*resolution_,resolution_);
        }
    }
    // Conflicting current occupancy always wins over a clearance witness.
    insert(seen,false);insert(occupied,true);stamp_=stamp;
    for(const auto &p:advisory) {
      if(!std::isfinite(p[0]) || !std::isfinite(p[1]) || std::max(std::abs(p[0]),std::abs(p[1]))>1e5)
        throw std::invalid_argument("invalid advisory routing point");
      auto &c=cells_[key(index(p[0]),index(p[1]))];c.advisory=true;c.stamp=stamp;
      c.advisory_bounds.clear();c.advisory_mask.reset();
    }
    const auto squares=[&](const auto &evidence,bool advisory_role) {
      for(const auto &p:evidence) {
        if(!std::isfinite(p.x)||!std::isfinite(p.y)||!std::isfinite(p.half)||p.half<=0||p.half>1.||
           std::max(std::abs(p.x),std::abs(p.y))>1e5)throw std::invalid_argument("invalid routing square");
        if(!std::isfinite(p.half_y)||(p.half_y!=-1.&&(p.half_y<=0||p.half_y>1.)))
          throw std::invalid_argument("invalid routing rectangle");
        const double yh=p.half_y>0?p.half_y:p.half;
        const double x0=p.x-p.half,y0=p.y-yh,x1=p.x+p.half,y1=p.y+yh;
        for(int y=index(y0);y<=index(std::nextafter(y1,-INFINITY));++y)
          for(int x=index(x0);x<=index(std::nextafter(x1,-INFINITY));++x) {
            auto &c=cells_[key(x,y)];auto &bound=advisory_role?c.advisory_bounds:c.hard_bounds;
            auto &raster=advisory_role?c.advisory_mask:c.hard_mask;
            auto &present=advisory_role?c.advisory:c.occupied;
            const Bounds clipped{std::max(x0,x*resolution_),std::max(y0,y*resolution_),
              std::min(x1,(x+1)*resolution_),std::min(y1,(y+1)*resolution_),true};
            // Keep the UNION, not its bounding hull: diagonal returns can have
            // a real free corner inside one hash cell. Coalescing is allowed
            // only for containment or an exactly rectangular union. No old
            // occupancy is narrowed; legacy/overflow retains the whole cell.
            if(present && raster) {
              // Keep integer bits across observations. Re-rasterizing the
              // outward-rounded display rectangles would grow one bin at
              // each compaction and eventually recreate the original bug.
              markMask(*raster,clipped,x*resolution_,y*resolution_,resolution_);
              bound=maskBounds(*raster,x*resolution_,y*resolution_,resolution_);
            }
            else if(!present)bound.push_back(clipped);
            else if(!bound.empty()) {
              auto merged=clipped;bool covered=false;
              for(size_t i=0;i<bound.size();) {
                const auto old=bound[i];
                const bool contains=old.x0<=merged.x0&&old.y0<=merged.y0&&old.x1>=merged.x1&&old.y1>=merged.y1;
                if(contains){covered=true;break;}
                const bool inside=merged.x0<=old.x0&&merged.y0<=old.y0&&merged.x1>=old.x1&&merged.y1>=old.y1;
                const bool horizontal=merged.y0==old.y0&&merged.y1==old.y1&&merged.x0<=old.x1&&old.x0<=merged.x1;
                const bool vertical=merged.x0==old.x0&&merged.x1==old.x1&&merged.y0<=old.y1&&old.y0<=merged.y1;
                if(inside||horizontal||vertical) {
                  merged={std::min(old.x0,merged.x0),std::min(old.y0,merged.y0),
                    std::max(old.x1,merged.x1),std::max(old.y1,merged.y1),true};
                  bound.erase(bound.begin()+i);i=0;
                } else ++i;
              }
              if(!covered)bound.push_back(merged);
              compactBounds(bound,raster,x*resolution_,y*resolution_,resolution_);
            }
            present=true;c.stamp=stamp;
          }
      }
    };
    squares(occupied_squares,false);squares(advisory_squares,true);
    if(cells_.size()>120000)clear(); // bounded memory; no unsafe fallback
  }

  Result route(Point start,Point goal,double yaw,double radius=.5,double max_curvature=.77,
               double front=.68,double back=.58,
               const std::function<double(double,double,double)> &pose_cost={},bool fine=false,
               bool allow_reverse=false,int initial_direction=0,double budget_ms=24.,
               const RouteOptions &options={}) const {
    Result result;result.fine=fine;
    const auto preprocessing_begin=std::chrono::steady_clock::now();
    if(!std::isfinite(budget_ms)||budget_ms<=0){result.reason="routing search budget";return result;}
    if(!std::isfinite(options.rear_axle_offset)||options.rear_axle_offset<0.||options.rear_axle_offset>3.||
       !std::isfinite(options.heuristic_weight)||options.heuristic_weight<1.||options.heuristic_weight>4.||
       !std::isfinite(options.advisory_cost)||options.advisory_cost<0.||options.advisory_cost>30.||
       !(options.max_initial_reverse>0.) ||
       (options.single_reverse_then_forward&&(!allow_reverse||initial_direction!=-1)))return result;
    const int yaw_bins=fine?72:32,primitive_steps=fine?4:8;
    if(initial_direction < -1 || initial_direction > 1 || (initial_direction<0 && !allow_reverse))return result;
    for(double v:{start[0],start[1],goal[0],goal[1],yaw,radius,max_curvature,front,back})if(!std::isfinite(v))return result;
    if(radius<=0 || radius>2. || front<=0 || back<=0 || front>3 || back>3 || max_curvature<=0 || max_curvature>3. || std::max({std::abs(start[0]),std::abs(start[1]),std::abs(goal[0]),std::abs(goal[1])})>1e5)return result;
    const int sx=index(start[0]),sy=index(start[1]),gx=index(goal[0]),gy=index(goal[1]);
    int lowx=std::min(sx,gx),highx=std::max(sx,gx),lowy=std::min(sy,gy),highy=std::max(sy,gy);
    for(const auto &entry:cells_) {
      lowx=std::min(lowx,ix(entry.first));highx=std::max(highx,ix(entry.first));
      lowy=std::min(lowy,iy(entry.first));highy=std::max(highy,iy(entry.first));
    }
    // Physical maneuver space, independent of grid resolution. Ten cells
    // shrank from 2m to 1m when collision columns became finer, clipping a
    // car's turning circle at the observed wall end and losing valid exits.
    const int pad=static_cast<int>(std::ceil(
      std::max(2.,2./max_curvature+std::max(front,back)+radius)/resolution_));
    lowx-=pad;lowy-=pad;highx+=pad;highy+=pad;
    const int width=highx-lowx+1,height=highy-lowy+1;
    if(width>512 || height>512 || width*height>120000)return result;
    std::vector<uint8_t> blocked(width*height,0),raw_occupied(width*height,0),observed(width*height,0);
    std::vector<uint8_t> advisory_cells(width*height,0);
    std::vector<std::vector<Bounds>> actual_bounds(width*height);
    std::vector<uint64_t> occupied_keys;
    std::vector<uint64_t> advisory_keys;
    std::vector<uint64_t> advisory_relief;
    const auto overlaps_start=[&](int x,int y) {
      const double c=std::cos(yaw),s=std::sin(yaw),a=(front+back)*.5,half=.5*resolution_;
      const double dx=(x+.5)*resolution_-start[0]-(front-back)*.5*c;
      const double dy=(y+.5)*resolution_-start[1]-(front-back)*.5*s;
      const double projected=half*(std::abs(c)+std::abs(s));
      return std::abs(c*dx+s*dy)<=a+projected && std::abs(-s*dx+c*dy)<=radius+projected &&
        std::abs(dx)<=a*std::abs(c)+radius*std::abs(s)+half &&
        std::abs(dy)<=a*std::abs(s)+radius*std::abs(c)+half;
    };
    const int reach=static_cast<int>(std::ceil(radius/resolution_+.5));
    for(const auto &entry:cells_) {
      const int x=ix(entry.first)-lowx,y=iy(entry.first)-lowy;
      observed[y*width+x]=1;
      if(entry.second.advisory) {
        advisory_keys.push_back(entry.first);advisory_cells[y*width+x]=1;
      }
      if(!entry.second.occupied && !entry.second.advisory)continue;
      // Distinguish source roles in the heuristic cache as well as geometry.
      occupied_keys.push_back(entry.first);
      const bool soft_advisory=options.advisory_cost>0.;
      if(!entry.second.occupied && soft_advisory)continue;
      // A caller may qualify the CURRENT stance with independent, fresh
      // actual collision/terrain evidence. Only advisory cells intersecting
      // that fixed stance may then be omitted from this ROUTING search.
      // Actual occupancy never gets relief. Every later execution is still
      // certified independently, including these same world coordinates.
      const bool relieve_advisory=entry.second.advisory && options.allow_advisory_start_overlap &&
          overlaps_start(ix(entry.first),iy(entry.first));
      if(relieve_advisory)advisory_relief.push_back(entry.first);
      if(!entry.second.occupied && relieve_advisory)continue;
      const Bounds whole{ix(entry.first)*resolution_,iy(entry.first)*resolution_,
        (ix(entry.first)+1)*resolution_,(iy(entry.first)+1)*resolution_,true};
      const auto append=[&](const std::vector<Bounds> &source) {
        auto &to=actual_bounds[y*width+x];
        if(source.empty())to.push_back(whole);
        else to.insert(to.end(),source.begin(),source.end());
      };
      if(entry.second.occupied)append(entry.second.hard_bounds);
      if(entry.second.advisory && !relieve_advisory && !soft_advisory)append(entry.second.advisory_bounds);
      raw_occupied[y*width+x]=1;
      for(int dy=-reach;dy<=reach;++dy)for(int dx=-reach;dx<=reach;++dx) {
        const int xx=x+dx,yy=y+dy;
        if(xx<0 || yy<0 || xx>=width || yy>=height)continue;
        const double px=(lowx+xx+.5)*resolution_,py=(lowy+yy+.5)*resolution_;
        for(const auto &b:actual_bounds[y*width+x]) {
          const double nx=std::max({b.x0-px,0.,px-b.x1}),ny=std::max({b.y0-py,0.,py-b.y1});
          if(std::hypot(nx,ny)<=radius+1e-9){blocked[yy*width+xx]=1;break;}
        }
      }
    }
    // Exact broad phase. Most body AABBs contain no occupied cell; scanning
    // every empty cell for every car-arc sample dominated dense routing.
    // The summed-area table can only skip an EMPTY box. Nonempty boxes still
    // test the same clipped source rectangles with the same SAT inequalities.
    const int pitch=width+1;
    std::vector<int> occupancy_sum((height+1)*pitch,0);
    std::vector<std::vector<int>> occupied_rows(height);
    for(int y=0;y<height;++y) {
      int total=0;
      for(int x=0;x<width;++x) {
        if(raw_occupied[y*width+x]){++total;occupied_rows[y].push_back(x);}
        occupancy_sum[(y+1)*pitch+x+1]=occupancy_sum[y*pitch+x+1]+total;
      }
    }
    const int source=(sy-lowy)*width+sx-lowx,target=(gy-lowy)*width+gx-lowx;
    const auto clear_pose=[&](double x,double y,double yaw) {
      // Both actual and historical geometry retain the full oriented body.
      // A centre-only historical check invents car U-turns in narrow lanes.
      const double c=std::cos(yaw),s=std::sin(yaw),a=(front+back)*.5;
      const double cx=x+(front-back)*.5*c,cy=y+(front-back)*.5*s;
      const double hx=std::abs(c)*a+std::abs(s)*radius,hy=std::abs(s)*a+std::abs(c)*radius;
      const int x0=index(cx-hx)-lowx,x1=index(cx+hx)-lowx,y0=index(cy-hy)-lowy,y1=index(cy+hy)-lowy;
      if(x0<0 || y0<0 || x1>=width || y1>=height)return false;
      const int count=occupancy_sum[(y1+1)*pitch+x1+1]-occupancy_sum[y0*pitch+x1+1]-
        occupancy_sum[(y1+1)*pitch+x0]+occupancy_sum[y0*pitch+x0];
      if(!count)return true;
      for(int yy=y0;yy<=y1;++yy) {
        const auto &row=occupied_rows[yy];
        for(auto it=std::lower_bound(row.begin(),row.end(),x0);it!=row.end()&&*it<=x1;++it) {
        const int xx=*it;
        for(const auto &b:actual_bounds[yy*width+xx]) {
        const double bx=(b.x1-b.x0)*.5,by=(b.y1-b.y0)*.5;
        const double dx=(b.x0+b.x1)*.5-cx,dy=(b.y0+b.y1)*.5-cy;
        if(std::abs(dx)>hx+bx || std::abs(dy)>hy+by ||
           std::abs(c*dx+s*dy)>a+bx*std::abs(c)+by*std::abs(s) ||
           std::abs(-s*dx+c*dy)>radius+bx*std::abs(s)+by*std::abs(c))continue;
        return false;
        }
      }
      }
      return true;
    };
    if(!clear_pose(start[0],start[1],yaw)){result.reason="blocked start body";return result;}
    if(pose_cost && !std::isfinite(pose_cost(start[0],start[1],yaw))){result.reason="unsafe start terrain";return result;}
    // Only the 2D heuristic gets endpoint relief for cell quantization. All
    // actual arc samples use the complete oriented vehicle against raw cells.
    const bool source_relief=blocked[source];
    blocked[source]=0;blocked[target]=0;
    // Obstacle-aware cost-to-go avoids expanding thousands of headings on
    // the wrong side of a long wall. It remains ROUTING guidance: this grid
    // does not authorize motion or replace the SE(2) safety certificate.
    std::sort(occupied_keys.begin(),occupied_keys.end());
    std::sort(advisory_keys.begin(),advisory_keys.end());
    std::sort(advisory_relief.begin(),advisory_relief.end());
    const auto &cached=heuristic_cache_;
    std::vector<float> terrain_cost;
    if(options.heuristic_cost || options.advisory_cost>0.) {
      terrain_cost.reserve(width*height);
      for(int y=0;y<height;++y)for(int x=0;x<width;++x) {
        const double value=(options.heuristic_cost?options.heuristic_cost((lowx+x+.5)*resolution_,(lowy+y+.5)*resolution_):1.)+
          options.advisory_cost*advisory_cells[y*width+x];
        if(!std::isfinite(value) || value<1.){result.reason="invalid routing heuristic cost";return result;}
        terrain_cost.push_back(static_cast<float>(value));
      }
    }
    const bool same_terrain=cached.terrain_cost==terrain_cost && (terrain_cost.empty() ||
      (cached.lowx==lowx && cached.lowy==lowy && cached.width==width && cached.height==height));
    const auto steady_now=std::chrono::steady_clock::now();
    const bool same_occupancy=cached.occupied==occupied_keys && cached.advisory==advisory_keys;
    const bool same_quantized_surfaces=same_occupancy ||
      (representedWithinOneCell(cached.occupied,occupied_keys) && representedWithinOneCell(occupied_keys,cached.occupied) &&
       representedWithinOneCell(cached.advisory,advisory_keys) && representedWithinOneCell(advisory_keys,cached.advisory));
    const bool reuse=!cached.distance.empty() && cached.advisory_relief==advisory_relief && cached.gx==gx && cached.gy==gy &&
      cached.radius==radius && cached.advisory_cost==options.advisory_cost && same_quantized_surfaces && same_terrain && stamp_>=cached.stamp &&
      stamp_-cached.stamp<=.5 &&
      std::chrono::duration<double>(steady_now-cached.built).count()<=.5 &&
      cached.source_relief==source_relief &&
      (!source_relief || (cached.relief_x==sx && cached.relief_y==sy));
    // Cache ONLY the optimistic cost-to-go heuristic, never occupancy,
    // terrain evidence, paths or execution certificates. Coarse/fine and
    // recovery searches repeatedly rebuilt the same Dijkstra field. Current
    // collision cells and pose costs below are ALWAYS evaluated afresh.
    // An unmatched added/removed surface invalidates reuse immediately. One
    // grid-cell boundary jitter may reuse the HEURISTIC for at most .5s.
    // Every arc STILL checks the exact new occupied cells below, including
    // shifted cells; neither collision nor a probability certificate is
    // cached. This bounded preference is not an optimality guarantee.
    std::vector<double> distance_to_goal;
    if(!reuse) {
    distance_to_goal.assign(width*height,std::numeric_limits<double>::infinity());
    using GridItem=std::pair<double,int>;
    std::priority_queue<GridItem,std::vector<GridItem>,std::greater<GridItem>> grid_open;
    distance_to_goal[target]=0.;grid_open.push({0.,target});
    size_t grid_iterations=0;
    while(!grid_open.empty()) {
      if((++grid_iterations%256)==0 && std::chrono::duration<double,std::milli>(
          std::chrono::steady_clock::now()-preprocessing_begin).count()>=budget_ms) {
        result.reason="routing preparation budget";return result;
      }
      const auto [distance,cell]=grid_open.top();grid_open.pop();
      if(distance>distance_to_goal[cell]+1e-9)continue;
      const int x=cell%width,y=cell/width;
      for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx) {
        if(!dx && !dy)continue;
        const int xx=x+dx,yy=y+dy;
        if(xx<0 || yy<0 || xx>=width || yy>=height)continue;
        const int next=yy*width+xx;if(blocked[next])continue;
        if(dx && dy && (blocked[y*width+xx] || blocked[yy*width+x]))continue;
        const double lower=terrain_cost.empty()?1.:std::min(terrain_cost[cell],terrain_cost[next]);
        const double edge=resolution_*std::hypot(dx,dy)*std::max(lower,observed[cell] && observed[next]?1.:3.);
        const double candidate=distance+edge;
        if(candidate+1e-9<distance_to_goal[next]){distance_to_goal[next]=candidate;grid_open.push({candidate,next});}
      }
    }
    heuristic_cache_.distance=std::move(distance_to_goal);
    heuristic_cache_.occupied=std::move(occupied_keys);
    heuristic_cache_.advisory=std::move(advisory_keys);
    heuristic_cache_.advisory_relief=std::move(advisory_relief);
    heuristic_cache_.terrain_cost=std::move(terrain_cost);
    heuristic_cache_.lowx=lowx;heuristic_cache_.lowy=lowy;
    heuristic_cache_.width=width;heuristic_cache_.height=height;
    heuristic_cache_.gx=gx;heuristic_cache_.gy=gy;
    heuristic_cache_.radius=radius;heuristic_cache_.stamp=stamp_;heuristic_cache_.built=steady_now;
    heuristic_cache_.advisory_cost=options.advisory_cost;
    heuristic_cache_.source_relief=source_relief;heuristic_cache_.relief_x=sx;heuristic_cache_.relief_y=sy;
    }
    result.heuristic_reused=reuse;
    result.quantized_heuristic_reused=reuse&&!same_occupancy;
    result.preprocessing_ms=std::chrono::duration<double,std::milli>(
      std::chrono::steady_clock::now()-preprocessing_begin).count();
    struct Node {double x,y,yaw,cost,curvature;int parent;int direction=1;double initial_reverse_distance=0.;};
    std::vector<Node> nodes;nodes.reserve(10000);
    std::unordered_map<uint64_t,double> cost;cost.reserve(20000);
    using Item=std::pair<double,int>;
    std::priority_queue<Item,std::vector<Item>,std::greater<Item>> open;
    const auto wrap=[](double angle){return std::atan2(std::sin(angle),std::cos(angle));};
    const auto address=[&](double x,double y,double angle,int direction=1) {
      const int stride=fine?1:2,key_width=(width+stride-1)/stride;
      const int cell=((index(y)-lowy)/stride)*key_width+(index(x)-lowx)/stride;
      int bin=static_cast<int>(std::floor((wrap(angle)+M_PI)*yaw_bins/(2*M_PI)));bin=std::clamp(bin,0,yaw_bins-1);
      const auto pose=static_cast<uint64_t>(cell)*yaw_bins+bin;
      return allow_reverse?pose*2+(direction<0):pose;
    };
    const auto sample=[&](double x,double y) {
      const int xx=index(x)-lowx,yy=index(y)-lowy;
      return xx<0 || yy<0 || xx>=width || yy>=height ? -1 : yy*width+xx;
    };
    const auto advance=[&](Node p,double k,double ds) {
      return advanceBodyArc(p,k,ds,options.rear_axle_offset);
    };
    const auto goal_distance=[&](double x,double y){return std::hypot(x-goal[0],y-goal[1]);};
    const auto heuristic=[&](double x,double y) {
      const auto &cache=heuristic_cache_;
      const int xx=index(x)-cache.lowx,yy=index(y)-cache.lowy;
      const int cell=xx<0 || yy<0 || xx>=cache.width || yy>=cache.height?-1:yy*cache.width+xx;
      return cell>=0 && std::isfinite(cache.distance[cell])?
        std::max(goal_distance(x,y),cache.distance[cell]):goal_distance(x,y);
    };
    const int start_direction=initial_direction?initial_direction:1;
    nodes.push_back({start[0],start[1],yaw,0.,0.,-1,start_direction});
    cost[address(start[0],start[1],yaw,start_direction)]=0;
    const double initial_h=heuristic(start[0],start[1]);
    open.push({options.heuristic_weight*initial_h,0});int chosen=-1,frontier=-1;
    double goal_connection_curvature=0.,goal_connection_length=0.;
    double best_h=initial_h;
    while(!open.empty()) {
      const int i=open.top().second;open.pop();const auto p=nodes[i];
      if(p.cost>cost[address(p.x,p.y,p.yaw,p.direction)]+1e-8)continue;
      const double distance=goal_distance(p.x,p.y),h=heuristic(p.x,p.y);
      if(options.allow_partial && initial_h-h>=.75 && h<best_h &&
          std::hypot(p.x-start[0],p.y-start[1])>=.5) {frontier=i;best_h=h;}
      if(distance<=.600001 && (distance<1e-6 || std::abs(wrap(std::atan2(goal[1]-p.y,goal[0]-p.x)-
          p.yaw-(p.direction<0?M_PI:0.)))<.6)) {
        // Connect with a bounded-curvature arc. A straight positional append
        // followed by a tangent-yaw jump used to mark geometrically valid
        // searches "reached" which the actual car then correctly rejected.
        const double c=std::cos(p.yaw),s=std::sin(p.yaw);
        const double dx=c*(goal[0]-p.x)+s*(goal[1]-p.y);
        const double dy=-s*(goal[0]-p.x)+c*(goal[1]-p.y);
        const auto connection=connectBodyArc(dx,dy,p.direction,options.rear_axle_offset);
        const double curvature=connection.curvature,length=connection.distance;
        bool clear=distance<1e-8 || (connection.valid &&
          std::abs(curvature)<=max_curvature+1e-10);
        const int steps=std::max(1,int(std::ceil(std::abs(length)/.05)));
        for(int k=1;clear && k<=steps;++k) {
          const auto q=advance(p,curvature,length*k/steps);
          if(!clear_pose(q.x,q.y,q.yaw) || (pose_cost && !std::isfinite(pose_cost(q.x,q.y,q.yaw))))clear=false;
        }
        if(clear){chosen=i;result.reached=true;goal_connection_curvature=curvature;goal_connection_length=length;break;}
      }
      if(++result.expanded>40000 || (result.expanded%16==0 &&
         std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-preprocessing_begin).count()>budget_ms)){
        result.reason="routing search budget";
        if(frontier<0)return result;
        chosen=frontier;result.partial=true;break;
      }
      // A point-grid U-turn is not an available car maneuver inside a narrow
      // corridor. Both gears preserve body yaw and the steering-radius limit.
      // Reversing is an opt-in recovery ROUTE, never an execution exemption.
      for(int direction:{1,-1})for(double ratio:{-1.,-.5,0.,.5,1.}) {
        if(direction<0 && !allow_reverse)continue;
        if(options.single_reverse_then_forward && p.parent>=0 && p.direction>0 && direction<0)continue;
        if(p.parent<0 && initial_direction && direction!=initial_direction)continue;
        const double curvature=ratio*max_curvature;auto next=p;bool clear=true;double travel_cost=0;
        if(initial_direction<0 && p.initial_reverse_distance>=0) {
          next.initial_reverse_distance=direction>0?-1.:p.initial_reverse_distance+.1*primitive_steps;
          if(next.initial_reverse_distance>options.max_initial_reverse+1e-9)continue;
        }
        for(int step=0;step<primitive_steps;++step) {
          next=advance(next,curvature,.1*direction);const int cell=sample(next.x,next.y);
          if(cell<0 || !clear_pose(next.x,next.y,next.yaw)){clear=false;break;}
          double terrain=(pose_cost?pose_cost(next.x,next.y,next.yaw):1.)+options.advisory_cost*advisory_cells[cell];
          if(direction<0 && options.reverse_observation_cost) {
            const double extra=options.reverse_observation_cost(next.x,next.y,next.yaw);
            if(!std::isfinite(extra)||extra<0.){clear=false;break;}
            terrain+=extra;
          }
          if(!std::isfinite(terrain) || terrain<1.){clear=false;break;}
          travel_cost+=.1*(direction<0?1.35:1.)*std::max(observed[cell]?1.:3.,terrain);
        }
        if(!clear)continue;
        next.cost=p.cost+travel_cost+.2*std::abs(ratio)+.1*std::abs(curvature-p.curvature)+
          (p.parent>=0 && p.direction!=direction?1.5:0.);
        next.direction=direction;
        const auto key=address(next.x,next.y,next.yaw,direction);const auto found=cost.find(key);
        if(found!=cost.end() && found->second<=next.cost+1e-9)continue;
        next.parent=i;next.curvature=curvature;cost[key]=next.cost;
        nodes.push_back(next);open.push({next.cost+options.heuristic_weight*heuristic(next.x,next.y),static_cast<int>(nodes.size()-1)});
      }
    }
    if(!result.reached && !result.partial) {
      if(frontier<0){result.reason=allow_reverse?"no reversible routing maneuver":"no forward routing maneuver";return result;}
      chosen=frontier;result.partial=true;
    }
    std::vector<int> sequence;for(int i=chosen;i!=-1;i=nodes[i].parent)sequence.push_back(i);
    std::reverse(sequence.begin(),sequence.end());result.path.push_back(start);result.yaws.push_back(yaw);
    result.directions.push_back(sequence.size()>1?nodes[sequence[1]].direction:1);
    for(size_t j=1;j<sequence.size();++j) {
      auto p=nodes[sequence[j-1]];const double k=nodes[sequence[j]].curvature;
      const int direction=nodes[sequence[j]].direction;
      for(int step=0;step<primitive_steps;++step){p=advance(p,k,.1*direction);result.path.push_back({p.x,p.y});
        result.yaws.push_back(wrap(p.yaw));result.directions.push_back(direction);}
    }
    if(result.reached && std::abs(goal_connection_length)>1e-8) {
      const auto p=nodes[chosen];const int steps=std::max(1,int(std::ceil(std::abs(goal_connection_length)/.05)));
      for(int i=1;i<=steps;++i) {
        const auto q=advance(p,goal_connection_curvature,goal_connection_length*i/steps);
        result.path.push_back({q.x,q.y});result.yaws.push_back(wrap(q.yaw));result.directions.push_back(p.direction);
      }
    }
    result.reason=result.reached?"reached":"useful routing frontier; final goal not reached";
    return result;
  }
};
} // namespace terrain_nav
