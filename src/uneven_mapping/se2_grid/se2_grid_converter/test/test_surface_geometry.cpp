#include "se2_grid_converter/surface_geometry.hpp"
#include "se2_grid_converter/grid_projection.hpp"

#include <gtest/gtest.h>
#include <limits>

TEST(SurfaceGeometry, FlatGridHasLocalUpwardTriangles)
{
  const auto triangles = se2_grid_converter::surfaceTriangles(
    std::vector<float>(12, 0.0F), 3, 4, 0, 0, 0.65);
  ASSERT_EQ(triangles.size(), 12u);
  EXPECT_EQ(triangles.front(), (se2_grid_converter::SurfaceTriangle{0, 4, 1}));
}

TEST(SurfaceGeometry, CircularBufferNeverBridgesOppositeMapEdges)
{
  for (std::size_t sr = 0; sr < 3; ++sr) {
    for (std::size_t sc = 0; sc < 4; ++sc) {
      const auto triangles = se2_grid_converter::surfaceTriangles(
        std::vector<float>(12, 0.0F), 3, 4, sr, sc, 0.65);
      ASSERT_EQ(triangles.size(), 12u);
      for (const auto & t : triangles) {
        std::array<double, 3> x{}, y{};
        for (std::size_t i = 0; i < 3; ++i) {
          x[i] = se2_grid_converter::cellCoordinate(0, 0.6, 0.2, t[i] / 4, sr, 3);
          y[i] = se2_grid_converter::cellCoordinate(0, 0.8, 0.2, t[i] % 4, sc, 4);
        }
        for (std::size_t i = 0; i < 3; ++i) {
          const auto j = (i + 1) % 3;
          EXPECT_LE(std::hypot(x[i] - x[j], y[i] - y[j]), std::sqrt(2.0) * 0.2 + 1e-6);
        }
        EXPECT_GT((x[1] - x[0]) * (y[2] - y[0]) -
          (y[1] - y[0]) * (x[2] - x[0]), 0.0);
      }
    }
  }
}

TEST(SurfaceGeometry, UnknownCellsAndStepsAreNotFilled)
{
  const float nan = std::numeric_limits<float>::quiet_NaN();
  EXPECT_TRUE(se2_grid_converter::surfaceTriangles({0, 0, nan, 0}, 2, 2, 0, 0, 0.65).empty());
  EXPECT_TRUE(se2_grid_converter::surfaceTriangles({0, 0, 0, 0.8F}, 2, 2, 0, 0, 0.65).empty());
  EXPECT_EQ(se2_grid_converter::surfaceTriangles({0, 0.1F, 0.2F, 0.3F}, 2, 2, 1, 1, 0.65).size(), 2u);
}

TEST(SurfaceGeometry, InvalidGeometryProducesNoSurface)
{
  EXPECT_TRUE(se2_grid_converter::surfaceTriangles({0, 0}, 1, 2, 0, 0, 0.65).empty());
  EXPECT_TRUE(se2_grid_converter::surfaceTriangles({0, 0}, 2, 2, 0, 0, 0.65).empty());
  EXPECT_TRUE(se2_grid_converter::surfaceTriangles({0, 0, 0, 0}, 2, 2, 0, 0, -0.1).empty());
}
