#pragma once
#include "dynamic_map_filter/dufomap_filter.hpp"

namespace ptr_dynamic_map {
// Experimental pinned FreeDOM static-map classifier. Completion is a model
// assumption; it never supplies measured terrain or clears current obstacles.
class FreedomFilter {
 public:
  explicit FreedomFilter(bool depth_enhancement=false);
  ~FreedomFilter();
  FreedomFilter(const FreedomFilter&)=delete;
  FreedomFilter& operator=(const FreedomFilter&)=delete;
  Metrics integrate(const std::vector<Point>& world_points,const Point& origin,
                    const std::array<float,9>& rotation);
  const std::vector<Point>& dynamicPoints() const;
  void reset();
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}
