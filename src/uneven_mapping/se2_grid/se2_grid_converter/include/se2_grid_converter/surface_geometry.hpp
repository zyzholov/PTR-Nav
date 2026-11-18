#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace se2_grid_converter
{
using SurfaceTriangle = std::array<std::size_t, 3>;

// Sides run around WORLD XY corners (-,-),(+,-),(+,+),(-,+).
// Native rows/cols descend world X/Y. Storage wrapping must never make the
// opposite logical map edge an occupied neighbour. This is display only.
inline std::array<bool,4> occupiedBoundarySides(const std::vector<unsigned char> &mask,
  std::size_t rows,std::size_t cols,std::size_t sr,std::size_t sc,std::size_t r,std::size_t c)
{
  std::array<bool,4> edges{false,false,false,false};
  if(!rows||!cols||mask.size()!=rows*cols||r>=rows||c>=cols||!mask[r*cols+c])return edges;
  const int lr=(r+rows-sr%rows)%rows,lc=(c+cols-sc%cols)%cols;
  constexpr int dr[]={0,-1,0,1},dc[]={1,0,-1,0};
  for(int side=0;side<4;++side) {
    const int nr=lr+dr[side],nc=lc+dc[side];
    edges[side]=nr<0||nc<0||nr>=static_cast<int>(rows)||nc>=static_cast<int>(cols)||
      !mask[((nr+sr)%rows)*cols+(nc+sc)%cols];
  }
  return edges;
}

inline float displayElevation(float measured,bool observed,float predicted,bool navigation_qualified)
{
  if(observed && std::isfinite(measured))return measured;
  if(navigation_qualified && std::isfinite(predicted))return predicted;
  return std::numeric_limits<float>::quiet_NaN();
}

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
