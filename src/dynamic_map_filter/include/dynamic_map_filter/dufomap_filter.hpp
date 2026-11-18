#pragma once
#include <array>
#include <cstddef>
#include <memory>
#include <vector>

namespace ptr_dynamic_map {
using Point = std::array<float,3>;
struct Config {
  float resolution=.1f;
  float surface_margin=.2f;
  unsigned unknown_margin_cells=2;
  float min_range=.2f;
  float max_range=50.f;
  unsigned threads=2;
  std::size_t max_nodes=4'000'000;
};
struct Metrics {
  double integration_ms=0.;
  std::size_t nodes=0;
  std::size_t memory_bytes=0;
  std::size_t input_points=0;
  bool reset_for_budget=false;
};
// A C++17 public interface; the pinned upstream core is compiled privately as
// C++20. This module classifies static-map points, never live collision input.
class DufomapFilter {
 public:
  explicit DufomapFilter(Config config={});
  ~DufomapFilter();
  DufomapFilter(const DufomapFilter&)=delete;
  DufomapFilter& operator=(const DufomapFilter&)=delete;
  Metrics integrate(const std::vector<Point>& world_points,const Point& sensor_origin);
  bool seenFree(const Point& world_point) const;
  std::vector<Point> dynamicVoxels(std::size_t maximum=20000) const;
  float resolution() const;
  void reset();
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}
