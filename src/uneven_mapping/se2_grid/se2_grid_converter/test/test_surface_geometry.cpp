#include "se2_grid_converter/surface_geometry.hpp"
#include "se2_grid_converter/grid_projection.hpp"
#include "se2_grid_converter/display_colors.hpp"

#include <gtest/gtest.h>
#include <limits>

TEST(SurfaceGeometry, OccupiedBoundariesRemoveInternalEdgesWithoutWrappingLogicalMap)
{
  using se2_grid_converter::occupiedBoundarySides;
  for(std::size_t sr=0;sr<3;++sr)for(std::size_t sc=0;sc<4;++sc) {
    std::vector<unsigned char> mask(12,1);
    int edges=0;
    for(std::size_t r=0;r<3;++r)for(std::size_t c=0;c<4;++c)
      for(bool e:occupiedBoundarySides(mask,3,4,sr,sc,r,c))edges+=e;
    EXPECT_EQ(edges,14); // The rectangle perimeter, never internal cell seams.
    mask.assign(12,0);mask[sr*4+sc]=1;
    EXPECT_EQ(occupiedBoundarySides(mask,3,4,sr,sc,sr,sc),(std::array<bool,4>{true,true,true,true}));
    // A logical neighbour across the physical buffer seam is still internal.
    const auto nr=(sr+1)%3;mask[nr*4+sc]=1;
    EXPECT_FALSE(occupiedBoundarySides(mask,3,4,sr,sc,sr,sc)[3]);
    EXPECT_FALSE(occupiedBoundarySides(mask,3,4,sr,sc,nr,sc)[1]);
  }
}

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

TEST(SurfaceGeometry, DisplayHeightUsesObservedBeforeQualifiedCompletionAndNeverCandidates)
{
  using se2_grid_converter::displayElevation;
  const float nan=std::numeric_limits<float>::quiet_NaN();
  EXPECT_FLOAT_EQ(displayElevation(1.F,true,2.F,true),1.F);
  EXPECT_FLOAT_EQ(displayElevation(nan,false,2.F,true),2.F);
  EXPECT_TRUE(std::isnan(displayElevation(1.F,false,2.F,false)));
  EXPECT_TRUE(std::isnan(displayElevation(nan,false,nan,true)));
  EXPECT_TRUE(std::isnan(displayElevation(nan,true,2.F,false)));
}

TEST(DisplayColors, ProbabilityIsVertexLocalAndUnknownIsDistinct)
{
  using se2_grid_converter::probabilityColor;
  const auto safe=probabilityColor(0),hazard=probabilityColor(1);
  EXPECT_NE(safe,hazard);
  EXPECT_EQ(probabilityColor(-1),safe);EXPECT_EQ(probabilityColor(2),hazard);
  const auto unknown=probabilityColor(std::numeric_limits<float>::quiet_NaN());
  EXPECT_NE(unknown,safe);EXPECT_NE(unknown,hazard);
  const std::array<float,3> values{0,0,1};
  EXPECT_EQ(probabilityColor(values[0]),safe);EXPECT_EQ(probabilityColor(values[1]),safe);
  EXPECT_EQ(probabilityColor(values[2]),hazard);
}

TEST(DisplayColors, ActualReturnRangeAndAccumulatedHeightHaveBoundedDistinctColors)
{
  using namespace se2_grid_converter;
  EXPECT_NE(lidarReturnColor(1),lidarReturnColor(20));
  EXPECT_NE(lidarMapColor(-.4F),lidarMapColor(2.F));
  for(float value:{-100.F,0.F,8.F,100.F,std::numeric_limits<float>::quiet_NaN()})
    for(const auto color:{lidarReturnColor(value),lidarMapColor(value),probabilityColor(value)})
      for(const auto channel:color) {EXPECT_TRUE(std::isfinite(channel));EXPECT_GE(channel,0);EXPECT_LE(channel,1);}
}
