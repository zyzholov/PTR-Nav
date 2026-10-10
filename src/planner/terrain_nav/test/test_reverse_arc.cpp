#include <gtest/gtest.h>
#include "terrain_nav/reverse_arc.hpp"

TEST(ReverseArc, StraightMaintainsBodyHeading)
{
  const auto path = terrain_nav::reverseArc(0, 0, 0, -2, 0, 4, 0.7);
  ASSERT_GT(path.size(), 20u);
  EXPECT_NEAR(path.back()[0], -2, 1e-6);
  for (const auto &p : path) EXPECT_NEAR(p[2], 0, 1e-6);
}

TEST(ReverseArc, LateralTargetAndRotatedWorld)
{
  const auto path = terrain_nav::reverseArc(5, 3, M_PI/2, 4.7, 1, 4, 0.7);
  ASSERT_FALSE(path.empty());
  EXPECT_NEAR(path.back()[0], 4.7, 1e-6);
  EXPECT_NEAR(path.back()[1], 1, 1e-6);
  EXPECT_LT(std::abs(path.back()[2] - M_PI/2), 0.5);
}

TEST(ReverseArc, RejectsForwardTooFarAndTooTight)
{
  EXPECT_TRUE(terrain_nav::reverseArc(0, 0, 0, 2, 0, 4, .7).empty());
  EXPECT_TRUE(terrain_nav::reverseArc(0, 0, 0, -5, 0, 4, .7).empty());
  EXPECT_TRUE(terrain_nav::reverseArc(0, 0, 0, -.5, 1, 4, .7).empty());
}
