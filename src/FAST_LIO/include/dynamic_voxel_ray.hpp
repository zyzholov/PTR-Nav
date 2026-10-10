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

namespace dynamic_voxel_ray {

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

template <typename Visitor>
void traverse(const std::array<double, 3> &origin,
              const std::array<double, 3> &endpoint,
              double voxel_size, double max_range, double endpoint_margin,
              Visitor visit) {
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

    int axis = 0;
    if (t_max[1] < t_max[axis]) axis = 1;
    if (t_max[2] < t_max[axis]) axis = 2;
    t = t_max[axis];
    *coordinate[axis] += direction[axis] > 0.0 ? 1 : -1;
    t_max[axis] += t_step[axis];
  }
}

}  // namespace dynamic_voxel_ray
