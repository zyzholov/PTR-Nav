#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace se2_grid_converter
{
using SurfaceTriangle = std::array<std::size_t, 3>;

inline std::vector<SurfaceTriangle> surfaceTriangles(
  const std::vector<float> & heights, std::size_t rows, std::size_t cols,
  std::size_t start_row, std::size_t start_col, double max_step)
{
  std::vector<SurfaceTriangle> triangles;
  if (rows < 2 || cols < 2 || heights.size() != rows * cols ||
      !std::isfinite(max_step) || max_step < 0.0) {
    return triangles;
  }
  triangles.reserve((rows - 1) * (cols - 1) * 2);
  auto buffer = [&](std::size_t row, std::size_t col) {
      return ((row + start_row) % rows) * cols + (col + start_col) % cols;
    };
  // Connect logical neighbours, not neighbours across the circular buffer's edge.
  for (std::size_t row = 0; row + 1 < rows; ++row) {
    for (std::size_t col = 0; col + 1 < cols; ++col) {
      const auto a = buffer(row, col), b = buffer(row + 1, col);
      const auto c = buffer(row, col + 1), d = buffer(row + 1, col + 1);
      const std::array<float, 4> z{heights[a], heights[b], heights[c], heights[d]};
      if (!std::all_of(z.begin(), z.end(), [](float h) { return std::isfinite(h); })) {
        continue;
      }
      const auto bounds = std::minmax_element(z.begin(), z.end());
      if (*bounds.second - *bounds.first > max_step) continue;
      triangles.push_back({a, b, c});
      triangles.push_back({b, d, c});
    }
  }
  return triangles;
}
}  // namespace se2_grid_converter
