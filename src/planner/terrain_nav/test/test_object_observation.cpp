#include "terrain_nav/object_observation.hpp"
#include "data/recorded_radius_collapse.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <set>
#include <vector>

namespace {
using Points = std::vector<std::array<double, 2>>;
using terrain_nav::TimedCenter;
using terrain_nav::fitCausalCenterVelocity;
using terrain_nav::fitMeasuredCircle;
using terrain_nav::fitRawSurfaceCircle;
constexpr double pi = 3.14159265358979323846;

Points arc(std::array<double, 2> center, double radius, double start, double span,
           std::size_t n = 40)
{
  Points points;
  for (std::size_t i = 0; i < n; ++i) {
    const double angle = start + span * static_cast<double>(i) / static_cast<double>(n - 1);
    points.push_back({{center[0] + radius * std::cos(angle),
                       center[1] + radius * std::sin(angle)}});
  }
  return points;
}

// Rasterize just the observed shell, then supply UNIQUE occupied cell centers.
// Different views genuinely have different cells and visible-face centroids.
Points rasterArc(std::array<double, 2> center, double radius, double start,
                 double span, double resolution)
{
  std::set<std::array<int, 2>> cells;
  for (const auto& p : arc(center, radius, start, span, 4097))
    cells.insert({{static_cast<int>(std::floor(p[0] / resolution)),
                   static_cast<int>(std::floor(p[1] / resolution))}});
  Points points;
  for (const auto& cell : cells)
    points.push_back({{(cell[0] + .5) * resolution, (cell[1] + .5) * resolution}});
  return points;
}

std::array<double, 2> centroid(const Points& points)
{
  std::array<double, 2> mean{};
  for (const auto& p : points)
    for (std::size_t axis = 0; axis < 2; ++axis)
      mean[axis] += p[axis] / static_cast<double>(points.size());
  return mean;
}

double distance(const std::array<double, 2>& a, const std::array<double, 2>& b)
{
  return std::hypot(a[0] - b[0], a[1] - b[1]);
}

std::vector<TimedCenter> translatingCenters()
{
  std::vector<TimedCenter> history;
  for (double t : {0., .1, .2, .3, .4})
    history.push_back({t, {{2. + .8 * t, -1. - .4 * t}}, .63, .02});
  return history;
}

TEST(ObjectObservation, ChangingRasterFacesMoveCentroidsButPreserveCircleCenter)
{
  const std::array<double, 2> center{{2.137, -1.173}};
  const double radius = .63, resolution = .02;
  const auto face_a = rasterArc(center, radius, -.9, 2.4, resolution);
  const auto face_b = rasterArc(center, radius, 1.6, 2.4, resolution);
  ASSERT_GT(distance(centroid(face_a), centroid(face_b)), .6);
  const auto a = fitMeasuredCircle(face_a, resolution);
  const auto b = fitMeasuredCircle(face_b, resolution);
  ASSERT_TRUE(a);
  ASSERT_TRUE(b);
  EXPECT_LT(distance(a->center, center), .015);
  EXPECT_LT(distance(b->center, center), .015);
  EXPECT_LT(distance(a->center, b->center), .02);
  EXPECT_NEAR(a->radius, radius, .015);
  EXPECT_NEAR(b->radius, radius, .015);
  EXPECT_EQ(a->source_count, face_a.size());
  EXPECT_LE(a->residual_rms, .8 * resolution);
  EXPECT_GE(a->observed_span_rad, pi / 3.);
  EXPECT_TRUE(std::isfinite(a->center_error_bound));
  EXPECT_GT(a->center_error_bound, resolution);
}

TEST(ObjectObservation, ExactPartialArcsHaveGenericRadiiAndOrderIndependentFits)
{
  for (double radius : {.19, .37, .83, 1.43}) {
    auto points = arc({{-3.1, 4.7}}, radius, 2.8, 2.3);
    const auto fit = fitMeasuredCircle(points, .02);
    ASSERT_TRUE(fit) << radius;
    EXPECT_NEAR(fit->center[0], -3.1, 1.e-11);
    EXPECT_NEAR(fit->center[1], 4.7, 1.e-11);
    EXPECT_NEAR(fit->radius, radius, 1.e-11);
    EXPECT_NEAR(fit->observed_span_rad, 2.3, 1.e-11);
    EXPECT_LT(fit->residual_rms, 1.e-11);
    std::reverse(points.begin(), points.end());
    const auto reversed = fitMeasuredCircle(points, .02);
    ASSERT_TRUE(reversed);
    EXPECT_LT(distance(fit->center, reversed->center), 1.e-11);
  }
}

TEST(ObjectObservation, CenterReserveReflectsPartialArcConditioning)
{
  const auto full = fitMeasuredCircle(arc({{0., 0.}}, .7, 0., 2. * pi, 64), .02);
  const auto partial = fitMeasuredCircle(arc({{0., 0.}}, .7, 0., 1.4, 64), .02);
  ASSERT_TRUE(full);
  ASSERT_TRUE(partial);
  EXPECT_GT(partial->center_error_bound, 3. * full->center_error_bound);
}

TEST(ObjectObservation, RetainedRigidRadiusCorrectsRecordedWeakArcWithoutCroppingMeasurements)
{
  const auto previous=recorded_radius_collapse::prior(),current=recorded_radius_collapse::current();
  const auto original=current;const double r=recorded_radius_collapse::resolution;
  const auto prior=fitMeasuredCircle(previous,r),free=fitMeasuredCircle(current,r);
  ASSERT_TRUE(prior&&free);ASSERT_GT(std::abs(prior->radius-free->radius),2*r);
  const std::array<double,2> predicted={prior->center[0]+.533843*recorded_radius_collapse::dt,
    prior->center[1]+.00709004*recorded_radius_collapse::dt};
  const auto retained=terrain_nav::fitMeasuredCircleWithRadius(current,r,prior->radius,predicted);
  ASSERT_TRUE(retained);EXPECT_TRUE(retained->retained_radius);
  EXPECT_DOUBLE_EQ(retained->radius,prior->radius);EXPECT_EQ(current,original);
  EXPECT_GT(free->center[0]-retained->center[0],r);
  EXPECT_GE(retained->conservative_radius,prior->conservative_radius-1e-12);
  for(const auto &p:current)EXPECT_GE(retained->conservative_radius+1e-12,
    distance(p,retained->center)+std::sqrt(.5)*r);
  EXPECT_FALSE(terrain_nav::fitMeasuredCircleWithRadius(current,r,NAN,predicted));
  EXPECT_FALSE(terrain_nav::fitMeasuredCircleWithRadius(current,r,prior->radius,{{100.,100.}}));
  auto broken=current;broken[0][0]=NAN;
  EXPECT_FALSE(terrain_nav::fitMeasuredCircleWithRadius(broken,r,prior->radius,predicted));
}

TEST(ObjectObservation, ConservativeRadiusCoversAllMeasuredRasterCorners)
{
  const double resolution = .035;
  const auto points = rasterArc({{-.327, 1.413}}, .52, -2.7, 3.8, resolution);
  const auto fit = fitMeasuredCircle(points, resolution);
  ASSERT_TRUE(fit);
  EXPECT_GE(fit->conservative_radius, fit->radius + 1.5 * resolution);
  for (const auto& p : points) {
    EXPECT_GE(fit->conservative_radius + 1.e-12,
              distance(p, fit->center) + std::sqrt(.5) * resolution);
    for (double dx : {-.5 * resolution, .5 * resolution})
      for (double dy : {-.5 * resolution, .5 * resolution})
        EXPECT_LE(distance({{p[0] + dx, p[1] + dy}}, fit->center),
                  fit->conservative_radius + 1.e-12);
  }
}

TEST(ObjectObservation, MostDistantReturnCanSetConservativeRadius)
{
  for (double offset : {0., 1.e8}) {
    auto points = arc({{offset, -offset}}, .55, 0., 2. * pi * 31. / 32., 32);
    points.front()[0] += .11;  // Sparse outward evidence must remain covered.
    const auto fit = fitMeasuredCircle(points, .04);
    ASSERT_TRUE(fit);
    double max_distance = 0.;
    for (const auto& p : points) max_distance = std::max(max_distance, distance(p, fit->center));
    EXPECT_GT(max_distance + std::sqrt(.5) * .04, fit->radius + 1.5 * .04);
    EXPECT_NEAR(fit->conservative_radius, max_distance + std::sqrt(.5) * .04, 1.e-12);
  }
}

TEST(ObjectObservation, RejectsLinesThickWallFragmentsAndCorners)
{
  Points line, noisy_line, wall, corner;
  for (int i = 0; i < 40; ++i) {
    const double x = .04 * i;
    line.push_back({{x, .7}});
    noisy_line.push_back({{x, .7 + .008 * std::sin(i)}});
    wall.push_back({{x, .69}});
    wall.push_back({{x, .71}});
    corner.push_back({{x, 0.}});
    if (i != 0) corner.push_back({{0., x}});
  }
  EXPECT_FALSE(fitMeasuredCircle(line, .02));
  EXPECT_FALSE(fitMeasuredCircle(noisy_line, .02));
  EXPECT_FALSE(fitMeasuredCircle(wall, .02));
  EXPECT_FALSE(fitMeasuredCircle(corner, .02));
}

TEST(ObjectObservation, RejectsFilledDiscsIncludingCoarseDiscThatPassesPrototypeRms)
{
  Points fine_disc, coarse_disc;
  for (int y = -10; y <= 10; ++y)
    for (int x = -10; x <= 10; ++x)
      if (std::hypot(x * .04, y * .04) <= .36)
        fine_disc.push_back({{x * .04, y * .04}});
  // Twelve cells at radii sqrt(.5)*res and sqrt(2.5)*res. The prototype
  // fits radius sqrt(11/6)*res with RMS < .8*res and full angular coverage.
  for (int y = -2; y < 2; ++y)
    for (int x = -2; x < 2; ++x)
      if (std::hypot(x + .5, y + .5) < 1.6)
        coarse_disc.push_back({{(x + .5) * .15, (y + .5) * .15}});
  ASSERT_EQ(coarse_disc.size(), 12u);
  EXPECT_FALSE(fitMeasuredCircle(fine_disc, .04));
  EXPECT_FALSE(fitMeasuredCircle(coarse_disc, .15));
}

TEST(ObjectObservation, RejectsLowAngularCoverageAndNoncircularRadialSupport)
{
  Points short_arc;
  // Endpoints plus mid-arc support make the PCA gate less restrictive than
  // the angular gate here; the measured angular span is only one radian.
  for (double angle : {-.5, -.49, -.48, -.47, -.03, -.02, -.01, 0.,
                       .01, .02, .03, .04, .47, .48, .49, .5})
    short_arc.push_back({{.8 * std::cos(angle), .8 * std::sin(angle)}});
  EXPECT_FALSE(fitMeasuredCircle(short_arc, .01));
  auto ellipse = arc({{0., 0.}}, .7, 0., 6.2, 64);
  for (auto& p : ellipse) p[1] *= .55;
  EXPECT_FALSE(fitMeasuredCircle(ellipse, .02));
}

TEST(ObjectObservation, RejectsInvalidSizeResolutionDegeneracyAndRadius)
{
  EXPECT_FALSE(fitMeasuredCircle({}, .02));
  EXPECT_FALSE(fitMeasuredCircle(arc({{0., 0.}}, .5, 0., 2., 7), .02));
  EXPECT_FALSE(fitMeasuredCircle(arc({{0., 0.}}, .5, 0., 2., 513), .02));
  EXPECT_TRUE(fitMeasuredCircle(arc({{0., 0.}}, .5, 0., 2., 8), .02));
  EXPECT_TRUE(fitMeasuredCircle(arc({{0., 0.}}, .5, 0., 2., 512), .02));
  EXPECT_FALSE(fitMeasuredCircle(Points(16, {{1., 2.}}), .02));
  EXPECT_FALSE(fitMeasuredCircle(arc({{0., 0.}}, .14, 0., 2.), .02));
  EXPECT_FALSE(fitMeasuredCircle(arc({{0., 0.}}, 1.51, 0., 2.), .02));
  const auto points = arc({{0., 0.}}, .5, 0., 2.);
  for (double resolution : {0., -.02, std::numeric_limits<double>::infinity(),
                            std::numeric_limits<double>::quiet_NaN()})
    EXPECT_FALSE(fitMeasuredCircle(points, resolution));
}

TEST(ObjectObservation, RejectsAnyNonfiniteSampleAndOverflowingGeometry)
{
  for (double invalid : {std::numeric_limits<double>::quiet_NaN(),
                         std::numeric_limits<double>::infinity(),
                         -std::numeric_limits<double>::infinity()})
    for (std::size_t axis = 0; axis < 2; ++axis) {
      auto points = arc({{0., 0.}}, .5, 0., 2.);
      points[9][axis] = invalid;
      EXPECT_FALSE(fitMeasuredCircle(points, .02));
    }
  auto points = arc({{0., 0.}}, .5, 0., 2.);
  points[0][0] = std::numeric_limits<double>::max();
  points[1][0] = -std::numeric_limits<double>::max();
  EXPECT_FALSE(fitMeasuredCircle(points, .02));
}

TEST(ObjectObservation, CenteringPreservesLocalGeometryAtLargeWorldCoordinates)
{
  const auto fit = fitMeasuredCircle(arc({{1.e8, -2.e8}}, .73, -1., 2.5), .02);
  ASSERT_TRUE(fit);
  EXPECT_NEAR(fit->center[0], 1.e8, 1.e-6);
  EXPECT_NEAR(fit->center[1], -2.e8, 1.e-6);
  EXPECT_NEAR(fit->radius, .73, 1.e-6);
}

TEST(ObjectCenterVelocity, TranslatingRasterArcsWithChangingFacesRecoverVelocity)
{
  std::vector<TimedCenter> history;
  for (int i = 0; i < 5; ++i) {
    const double t = .1 * i;
    const auto points = rasterArc({{2.137 + .8 * t, -1.173 - .4 * t}},
                                  .63, -.9 + .55 * i, 2.4, .02);
    const auto circle = fitMeasuredCircle(points, .02);
    ASSERT_TRUE(circle);
    history.push_back({t, circle->center, circle->radius, .02});
  }
  const auto fit = fitCausalCenterVelocity(history);
  ASSERT_TRUE(fit);
  EXPECT_NEAR(fit->velocity[0], .8, .035);
  EXPECT_NEAR(fit->velocity[1], -.4, .035);
  EXPECT_LT(fit->fit_residual_rms, .012);
  EXPECT_LT(fit->max_step_innovation, .02);
  EXPECT_EQ(fit->source_count, 5u);
  EXPECT_NEAR(fit->span_s, .4, 1.e-12);
  EXPECT_NEAR(fit->stamp_s, .4, 1.e-12);
}

TEST(ObjectCenterVelocity, StationaryObjectWithChangingRasterFacesHasNearZeroVelocity)
{
  std::vector<TimedCenter> history;
  for (int i = 0; i < 5; ++i) {
    const auto circle = fitMeasuredCircle(
        rasterArc({{2.137, -1.173}}, .63, -.9 + .55 * i, 2.4, .02), .02);
    ASSERT_TRUE(circle);
    history.push_back({.1 * i, circle->center, circle->radius, .02});
  }
  const auto fit = fitCausalCenterVelocity(history);
  ASSERT_TRUE(fit);
  EXPECT_LT(std::hypot(fit->velocity[0], fit->velocity[1]), .05);
}

TEST(ObjectCenterVelocity, FitsIrregularTimesAndExactMinimumSpan)
{
  const std::vector<TimedCenter> history = {
      {0., {{3., -2.}}, .3, .02},
      {.037, {{3.074, -2.111}}, .3, .02},
      {.15, {{3.3, -2.45}}, .3, .02}};
  const auto fit = fitCausalCenterVelocity(history);
  ASSERT_TRUE(fit);
  EXPECT_NEAR(fit->velocity[0], 2., 1.e-12);
  EXPECT_NEAR(fit->velocity[1], -3., 1.e-12);
  EXPECT_LT(distance(fit->fitted_center, history.back().center), 1.e-12);
  EXPECT_LT(fit->fit_residual_rms, 1.e-12);
  EXPECT_LT(fit->max_step_innovation, 1.e-12);
}

TEST(ObjectCenterVelocity, ReportsVectorRmsAndAdjacentStepInnovationInMetres)
{
  const std::vector<TimedCenter> history = {
      {0., {{0., 0.}}, .3, .02},
      {.1, {{.1, .03}}, .3, .02},
      {.2, {{.2, 0.}}, .3, .02}};
  const auto fit = fitCausalCenterVelocity(history);
  ASSERT_TRUE(fit);
  EXPECT_NEAR(fit->velocity[0], 1., 1.e-12);
  EXPECT_NEAR(fit->velocity[1], 0., 1.e-12);
  EXPECT_NEAR(fit->fitted_center[1], .01, 1.e-12);
  EXPECT_NEAR(fit->fit_residual_rms, std::sqrt(.0002), 1.e-12);
  EXPECT_NEAR(fit->max_step_innovation, .03, 1.e-12);
}

TEST(ObjectCenterVelocity, UsesOnlyLastPointFourSecondsAndDoesNotMutateInput)
{
  auto history = translatingCenters();
  history.insert(history.begin(), {-.001, {{100., -200.}}, .63, .02});
  const auto before = history;
  const auto fit = fitCausalCenterVelocity(history);
  ASSERT_TRUE(fit);
  EXPECT_EQ(fit->source_count, 5u);
  EXPECT_NEAR(fit->velocity[0], .8, 1.e-12);
  EXPECT_NEAR(fit->velocity[1], -.4, 1.e-12);
  for (std::size_t i = 0; i < history.size(); ++i) {
    EXPECT_EQ(history[i].stamp_s, before[i].stamp_s);
    EXPECT_EQ(history[i].center, before[i].center);
  }
}

TEST(ObjectCenterVelocity, RejectsTooFewScansTooShortSpanOrExpiredHistory)
{
  auto history = translatingCenters();
  EXPECT_FALSE(fitCausalCenterVelocity({}));
  EXPECT_FALSE(fitCausalCenterVelocity({history[0], history[2]}));
  history.resize(3);
  history[1].stamp_s = .07;
  history[2].stamp_s = .14;
  EXPECT_FALSE(fitCausalCenterVelocity(history));
  history[2].stamp_s = .50001;
  EXPECT_FALSE(fitCausalCenterVelocity(history));
}

TEST(ObjectCenterVelocity, RejectsDuplicateReversedAndNonfiniteTimestamps)
{
  auto history = translatingCenters();
  history[2].stamp_s = history[1].stamp_s;
  EXPECT_FALSE(fitCausalCenterVelocity(history));
  history = translatingCenters();
  std::swap(history[1], history[2]);
  EXPECT_FALSE(fitCausalCenterVelocity(history));
  for (double invalid : {std::numeric_limits<double>::quiet_NaN(),
                         std::numeric_limits<double>::infinity()}) {
    history = translatingCenters();
    history[1].stamp_s = invalid;
    EXPECT_FALSE(fitCausalCenterVelocity(history));
  }
}

TEST(ObjectCenterVelocity, LatestInvalidGeometryCannotReturnStaleVelocity)
{
  for (double invalid : {std::numeric_limits<double>::quiet_NaN(),
                         std::numeric_limits<double>::infinity()}) {
    auto history = translatingCenters();
    history.back().center[1] = invalid;
    EXPECT_FALSE(fitCausalCenterVelocity(history));
    history = translatingCenters();
    history.back().radius = invalid;
    EXPECT_FALSE(fitCausalCenterVelocity(history));
    history = translatingCenters();
    history.back().resolution = invalid;
    EXPECT_FALSE(fitCausalCenterVelocity(history));
  }
  auto history = translatingCenters();
  history.back().radius = 0.;
  EXPECT_FALSE(fitCausalCenterVelocity(history));
  history = translatingCenters();
  history.back().resolution = 0.;
  EXPECT_FALSE(fitCausalCenterVelocity(history));
}

TEST(ObjectCenterVelocity, InvalidMiddleGeometryResetsAndRequiresFreshSpan)
{
  auto history = translatingCenters();
  history[1].center[0] = std::numeric_limits<double>::quiet_NaN();
  auto fit = fitCausalCenterVelocity(history);
  ASSERT_TRUE(fit);
  EXPECT_EQ(fit->source_count, 3u);
  EXPECT_NEAR(fit->span_s, .2, 1.e-12);
  EXPECT_NEAR(fit->velocity[0], .8, 1.e-12);
  history[2].center[0] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(fitCausalCenterVelocity(history));
}

TEST(ObjectCenterVelocity, RadiusAndResolutionChangesResetUsableSuffix)
{
  auto history = translatingCenters();
  history[2].radius = history[3].radius = history[4].radius = 1.1;
  auto fit = fitCausalCenterVelocity(history);
  ASSERT_TRUE(fit);
  EXPECT_EQ(fit->source_count, 3u);
  EXPECT_NEAR(fit->velocity[0], .8, 1.e-12);
  history = translatingCenters();
  history[2].resolution = history[3].resolution = history[4].resolution = .04;
  fit = fitCausalCenterVelocity(history);
  ASSERT_TRUE(fit);
  EXPECT_EQ(fit->source_count, 3u);
  history[4].resolution = .08;
  EXPECT_FALSE(fitCausalCenterVelocity(history));
}

TEST(ObjectCenterVelocity, CumulativeShapeDriftCannotEvadeRadiusReset)
{
  auto history = translatingCenters();
  for (std::size_t i = 0; i < history.size(); ++i) history[i].radius = .3 + .015 * i;
  EXPECT_FALSE(fitCausalCenterVelocity(history));
}

TEST(ObjectCenterVelocity, RelativeFitHandlesLargeWorldCoordinatesAndEpochTimes)
{
  auto history = translatingCenters();
  const double epoch = 1.7e9;
  for (auto& sample : history) {
    sample.center = {{1.e8 + .8 * sample.stamp_s, -2.e8 - .4 * sample.stamp_s}};
    sample.stamp_s += epoch;
  }
  const auto fit = fitCausalCenterVelocity(history);
  ASSERT_TRUE(fit);
  EXPECT_NEAR(fit->velocity[0], .8, 2.e-6);
  EXPECT_NEAR(fit->velocity[1], -.4, 2.e-6);
  EXPECT_EQ(fit->source_count, 5u);
}

TEST(ObjectObservation, ActualReturnsAvoidNarrowArcVoxelBiasWithoutRemovingOccupiedCorners)
{
  double worst_voxel_error=0.;
  for(int phase=0;phase<30;++phase) {
    const std::array<double,2> centre={5.+.049*phase/29.,.17+.049*phase/29.};
    const auto returns=arc(centre,.4,pi-.7,1.4,200);
    const auto cells=rasterArc(centre,.4,pi-.7,1.4,.05);
    const auto original=cells;const auto fit=fitRawSurfaceCircle(returns,cells,.05);
    ASSERT_TRUE(fit)<<phase;EXPECT_TRUE(fit->raw_surface);EXPECT_EQ(cells,original);
    EXPECT_NEAR(fit->center[0],centre[0],1e-9);EXPECT_NEAR(fit->center[1],centre[1],1e-9);
    EXPECT_NEAR(fit->radius,.4,1e-9);
    for(const auto &c:cells)for(double dx:{-.025,.025})for(double dy:{-.025,.025})
      EXPECT_LE(std::hypot(c[0]+dx-fit->center[0],c[1]+dy-fit->center[1]),fit->conservative_radius+1e-12);
    if(const auto voxel=fitMeasuredCircle(cells,.05))worst_voxel_error=std::max(worst_voxel_error,
      std::hypot(voxel->center[0]-centre[0],voxel->center[1]-centre[1]));
  }
  EXPECT_GT(worst_voxel_error,.1); // Controlled quantization example, not sensor calibration.
}

TEST(ObjectObservation, InvalidRawEvidenceCannotAcquireAVoxelDerivedCircle)
{
  const auto cells=rasterArc({0,0},.4,pi-.7,1.4,.05);
  EXPECT_FALSE(fitRawSurfaceCircle({},cells,.05));
  // Existing PCA observability limit stays .025. A 1.2-rad exact arc has
  // ratio .02475; quantization must not manufacture missing observability.
  EXPECT_FALSE(fitRawSurfaceCircle(arc({0,0},.4,pi-.6,1.2,200),
    rasterArc({0,0},.4,pi-.6,1.2,.05),.05));
  auto returns=arc({0,0},.4,pi-.7,1.4,100);returns[3][0]=NAN;
  EXPECT_FALSE(fitRawSurfaceCircle(returns,cells,.05));
  EXPECT_FALSE(fitRawSurfaceCircle(arc({2,2},.4,pi-.7,1.4,100),cells,.05));
}

TEST(ObjectObservation, DenseRawReturnsKeepAllSamplesWithoutExpandingTheOccupiedComponentLimit)
{
  const auto cells=rasterArc({.123,.217},.4,pi-.8,1.6,.05);
  const auto returns=arc({.123,.217},.4,pi-.8,1.6,2000);
  EXPECT_FALSE(fitMeasuredCircle(returns,.05)); // Old voxel-facing bound stays.
  const auto fit=fitRawSurfaceCircle(returns,cells,.05);ASSERT_TRUE(fit);
  EXPECT_EQ(fit->source_count,returns.size());EXPECT_NEAR(fit->center[0],.123,1e-9);
  EXPECT_NEAR(fit->center[1],.217,1e-9);EXPECT_TRUE(fit->raw_surface);
  EXPECT_FALSE(fitRawSurfaceCircle(arc({.123,.217},.4,pi-.8,1.6,8193),cells,.05));
}

TEST(ObjectCenterVelocity, RejectsFiniteInputsWhoseArithmeticOverflows)
{
  auto history = translatingCenters();
  history[1].center[0] = std::numeric_limits<double>::max();
  history.back().center[0] = -std::numeric_limits<double>::max();
  EXPECT_FALSE(fitCausalCenterVelocity(history));
}

TEST(ObjectCenterVelocity, AveragingLagBoundsInstantaneousVelocityWithRegularAndIrregularSources)
{
  const std::vector<std::vector<double>> schedules={{-.3,-.2,-.1,0.},{-.35,-.21,-.08,0.},{-.39,-.2,0.}};
  for(const auto &times:schedules)for(double angle:{0.,.8,2.4}) {
    const double a=1.2,ax=a*std::cos(angle),ay=a*std::sin(angle);
    std::vector<TimedCenter> history;
    for(double t:times)history.push_back({10.+t,{5.+.6*t+.5*ax*t*t,-2.+.5*t+.5*ay*t*t},.4,.05});
    const auto fit=fitCausalCenterVelocity(history);ASSERT_TRUE(fit);
    EXPECT_DOUBLE_EQ(fit->stamp_s,10.);EXPECT_GT(fit->averaging_lag_s,0.);
    EXPECT_LE(fit->averaging_lag_s,fit->span_s);
    EXPECT_NEAR(fit->velocity[0],.6-ax*fit->averaging_lag_s,1e-10);
    EXPECT_NEAR(fit->velocity[1],.5-ay*fit->averaging_lag_s,1e-10);
    EXPECT_NEAR(std::hypot(fit->velocity[0]-.6,fit->velocity[1]-.5),a*fit->averaging_lag_s,1e-10);
  }
  for(const auto &times:schedules)for(double frequency:{.5,2.,7.,15.}) {
    std::vector<TimedCenter> history;const double a=1.2,w=frequency;
    for(double t:times)history.push_back({10.+t,{2.+.6*t+a/(w*w)*(1-std::cos(w*t)),
      -1.+.3*t+a/(w*w)*std::sin(w*t)},.4,.05});
    const auto fit=fitCausalCenterVelocity(history);ASSERT_TRUE(fit);
    // The direction of acceleration changes continuously, |a(t)|=1.2.
    EXPECT_LE(std::hypot(fit->velocity[0]-.6,fit->velocity[1]-(.3+a/w)),a*fit->averaging_lag_s+1e-10);
  }
}
}  // namespace
