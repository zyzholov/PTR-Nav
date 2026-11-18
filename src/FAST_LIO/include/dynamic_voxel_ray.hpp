#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <unordered_set>
#include <unordered_map>

namespace dynamic_voxel_ray {

// Classification affects registration features only, never the raw scan or
// collision observations. Do not use future labels, labels after a clock reset,
// or an old accumulated-map label as current motion evidence.
inline bool fresh_registration_label(bool enabled,double scan_stamp,double label_stamp,bool labelled) {
  return enabled&&labelled&&std::isfinite(scan_stamp)&&std::isfinite(label_stamp)&&label_stamp>=0&&
    scan_stamp>=label_stamp&&scan_stamp-label_stamp<=.25;
}

// Exact Float32-coordinate preimage of floor(float(x / size)). The label
// producer and static_map_excluded already use that convention. Multiplying
// its integer ID by size does NOT invert the rounded division at cell faces.
// Return [lower,upper), matching IKD-tree box semantics. Unsupported/numerically
// ambiguous ranges fail closed, rather than widen a deletion into neighbours.
inline bool float_label_interval(int64_t id,float size,std::array<float,2>& interval) {
  if(!std::isfinite(size)||size<=0.f||size>1.f||id<=-9007199254740990LL||id>=9007199254740990LL)return false;
  const auto first=[&](double threshold,float& value) {
    value=static_cast<float>(threshold*static_cast<double>(size));
    for(int attempt=0;attempt<8;++attempt) {
      if(!std::isfinite(value))return false;
      const float quotient=value/size;
      if(static_cast<double>(quotient)<threshold) {
        value=std::nextafter(value,std::numeric_limits<float>::infinity());continue;
      }
      const float previous=std::nextafter(value,-std::numeric_limits<float>::infinity());
      const float previous_quotient=previous/size;
      if(static_cast<double>(previous_quotient)<threshold)return true;
      value=previous;
    }
    return false;
  };
  if(!first(static_cast<double>(id),interval[0])||!first(static_cast<double>(id)+1.,interval[1]))return false;
  return interval[0]<interval[1];
}

struct Key {
  int64_t x;
  int64_t y;
  int64_t z;

  bool operator==(const Key &other) const {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct Hash {
  size_t operator()(const Key &key) const {
    size_t seed = std::hash<int64_t>{}(key.x);
    seed ^= std::hash<int64_t>{}(key.y) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed ^= std::hash<int64_t>{}(key.z) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
  }
};

// Exact sparse bit index, not free-space evidence. A ray normally stays in an
// 8x8x8 block for several visits, avoiding repeated empty-voxel hash lookups.
class VoxelPresence {
  using Block=std::array<uint64_t,8>;
public:
  struct Cache {
    Key key{};
    const Block *block=nullptr;
    uint64_t epoch=0;
  };

  void insert(const Key &key) {
    const auto block=block_key(key);
    auto entry=blocks_.try_emplace(block,Block{});
    if (entry.second) ++epoch_;
    const unsigned bit=offset(key,block);
    entry.first->second[bit/64] |= uint64_t{1}<<(bit%64);
  }

  void erase(const Key &key) {
    const auto block=block_key(key);
    auto entry=blocks_.find(block);
    if (entry==blocks_.end()) return;
    const unsigned bit=offset(key,block);
    entry->second[bit/64] &= ~(uint64_t{1}<<(bit%64));
    if (std::all_of(entry->second.begin(),entry->second.end(),[](uint64_t word){return word==0;})) {
      blocks_.erase(entry);++epoch_;
    }
  }

  bool contains(const Key &key,Cache &cache) const {
    const auto block=block_key(key);
    if (cache.epoch!=epoch_ || !(cache.key==block)) {
      auto entry=blocks_.find(block);
      cache={block,entry==blocks_.end()?nullptr:&entry->second,epoch_};
    }
    const unsigned bit=offset(key,block);
    return cache.block && ((*cache.block)[bit/64] & (uint64_t{1}<<(bit%64)));
  }

  void clear() {blocks_.clear();++epoch_;}

private:
  static int64_t floor_divide(int64_t x) {return x/8-(x%8<0);}
  static Key block_key(const Key &key) {
    return {floor_divide(key.x),floor_divide(key.y),floor_divide(key.z)};
  }
  static unsigned offset(const Key &key,const Key &block) {
    return static_cast<unsigned>((key.x-block.x*8)+8*(key.y-block.y*8)+64*(key.z-block.z*8));
  }
  std::unordered_map<Key,Block,Hash> blocks_;
  uint64_t epoch_=1;
};

// Three enclosing quadrants must contain locally flat measured ground. The
// third lowest per-quadrant height is exactly the original popcount>=3 test.
// Bounded memory of actual locally flat returns, solely to protect ground
// during historical-map cleanup. Missing scans never create free-space labels.
class GroundHistory {
 public:
  void begin(double stamp,double cell_size=.4) {
    if(!std::isfinite(stamp) || !std::isfinite(cell_size) || cell_size<=0) {
      cells_.clear();stamp_=-std::numeric_limits<double>::infinity();return;
    }
    if(stamp<stamp_ || std::abs(cell_size-cell_size_)>1e-9)cells_.clear();
    cell_size_=cell_size;
    stamp_=stamp;
    for(auto it=cells_.begin();it!=cells_.end();) {
      if(stamp-it->second.stamp>3. || stamp<it->second.stamp)it=cells_.erase(it);else ++it;
    }
    if(cells_.size()>50000)cells_.clear();
  }
  void observe(const Key& key,float low,float high) {
    if(!std::isfinite(stamp_) || !std::isfinite(low) || !std::isfinite(high) ||
       high<low || high-low>.20f)return;
    cells_[key]={low,high,stamp_};
  }
  bool lookup(const Key& key,float& low,float& high)const {
    const auto it=cells_.find(key);if(it==cells_.end())return false;
    const auto age=stamp_-it->second.stamp;
    if(age<0 || age>3.)return false;
    // Raise the protective floor as its measured reference ages; never lower
    // it to obtain more deletions. This is an engineering preservation bound.
    const float bound=.01f*age;
    low=it->second.low+bound;high=it->second.high+bound;return true;
  }
  size_t size()const{return cells_.size();}
 private:
  struct Span {float low,high;double stamp;};
  std::unordered_map<Key,Span,Hash> cells_;
  double stamp_=-std::numeric_limits<double>::infinity();
  double cell_size_=.4;
};

template <typename Lookup>
float enclosing_ground_floor(int64_t x,int64_t y,Lookup lookup) {
  std::array<float,4> minima;
  minima.fill(std::numeric_limits<float>::infinity());
  for (int gy=-2;gy<=2;gy++) for (int gx=-2;gx<=2;gx++) {
    float low,high;
    if (!lookup(x+gx,y+gy,low,high) || high-low>.20f) continue;
    const int quadrant=(gx>=0?1:0)+(gy>=0?2:0);
    minima[quadrant]=std::min(minima[quadrant],low);
  }
  std::nth_element(minima.begin(),minima.begin()+2,minima.end());
  return minima[2];
}

struct EvidenceWindow {
  uint32_t last_frame = 0;
  uint8_t scans = 0;

  void reset() {
    last_frame = 0;
    scans = 0;
  }

  bool observe(uint32_t frame, uint8_t required_scans, uint32_t max_gap_frames) {
    if (frame == last_frame) return scans >= required_scans;
    if (last_frame == 0 || frame < last_frame || frame - last_frame > max_gap_frames)
      scans = 0;
    last_frame = frame;
    if (scans < required_scans) ++scans;
    return scans >= required_scans;
  }
};

template <typename Contains, typename Visit>
void flood_connected(const Key &seed, Contains contains, Visit visit, size_t max_cells) {
  if (max_cells == 0) return;
  std::unordered_set<Key, Hash> seen;
  seen.reserve(max_cells);
  std::deque<Key> pending{seed};
  seen.insert(seed);
  while (!pending.empty()) {
    const Key current = pending.front();
    pending.pop_front();
    visit(current);
    for (int dz = -1; dz <= 1; ++dz) {
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          if (dx == 0 && dy == 0 && dz == 0) continue;
          const Key next{current.x + dx, current.y + dy, current.z + dz};
          if (!contains(next) || seen.find(next) != seen.end()) continue;
          if (seen.size() >= max_cells) return;
          seen.insert(next);
          pending.push_back(next);
        }
      }
    }
  }
}

template <typename Candidate,typename Visitor>
void traverse_candidates(const std::array<double, 3> &origin,
              const std::array<double, 3> &endpoint,
              double voxel_size, double max_range, double endpoint_margin,
              Candidate candidate,Visitor visit) {
  std::array<double, 3> delta, direction, t_max, t_step;
  double distance2 = 0.0;
  for (int axis = 0; axis < 3; ++axis) {
    if (!std::isfinite(origin[axis]) || !std::isfinite(endpoint[axis])) return;
    delta[axis] = endpoint[axis] - origin[axis];
    distance2 += delta[axis] * delta[axis];
  }
  if (!std::isfinite(distance2) || voxel_size <= 0.0 || max_range <= 0.0) return;
  const double distance = std::sqrt(distance2);
  const double limit = std::min(max_range, distance - endpoint_margin);
  if (limit <= 0.0) return;

  Key key{};
  int64_t *coordinate[3] = {&key.x, &key.y, &key.z};
  for (int axis = 0; axis < 3; ++axis) {
    direction[axis] = delta[axis] / distance;
    *coordinate[axis] = static_cast<int64_t>(std::floor(origin[axis] / voxel_size));
    if (direction[axis] == 0.0) {
      t_max[axis] = t_step[axis] = std::numeric_limits<double>::infinity();
    } else {
      const double boundary = (*coordinate[axis] + (direction[axis] > 0.0 ? 1 : 0)) * voxel_size;
      t_max[axis] = std::max(0.0, (boundary - origin[axis]) / direction[axis]);
      t_step[axis] = voxel_size / std::abs(direction[axis]);
    }
  }

  double t = 0.0;
  while (t <= limit) {
    // Candidate presence is not free-space evidence. It only avoids computing
    // ray-to-center geometry where the caller has no stored voxel to inspect.
    // DDA steps and every geometric/evidence test on a present voxel stay exact.
    if(candidate(key)) {
    const std::array<double, 3> center = {
      (key.x + 0.5) * voxel_size,
      (key.y + 0.5) * voxel_size,
      (key.z + 0.5) * voxel_size};
    double along = 0.0;
    double perpendicular2 = 0.0;
    for (int axis = 0; axis < 3; ++axis)
      along += (center[axis] - origin[axis]) * direction[axis];
    for (int axis = 0; axis < 3; ++axis) {
      const double error = center[axis] - origin[axis] - along * direction[axis];
      perpendicular2 += error * error;
    }
    if (along > voxel_size * 2.0 && along <= limit &&
        perpendicular2 <= 0.49 * voxel_size * voxel_size)
      visit(key);
    }

    int axis = 0;
    if (t_max[1] < t_max[axis]) axis = 1;
    if (t_max[2] < t_max[axis]) axis = 2;
    t = t_max[axis];
    *coordinate[axis] += direction[axis] > 0.0 ? 1 : -1;
    t_max[axis] += t_step[axis];
  }
}

template <typename Visitor>
void traverse(const std::array<double, 3> &origin,
              const std::array<double, 3> &endpoint,
              double voxel_size,double max_range,double endpoint_margin,Visitor visit) {
  traverse_candidates(origin,endpoint,voxel_size,max_range,endpoint_margin,
      [](const Key &){return true;},visit);
}

}  // namespace dynamic_voxel_ray
