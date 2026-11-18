#pragma once

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <Eigen/Cholesky>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <vector>

namespace terrain_nav {

// A geometric circle hypothesis from ONE measured component in a common metric
// frame. No semantic class, association, confirmation, or motion is implied.
struct CircleObservation {
  std::array<double, 2> center{};
  double radius = 0.;                  // Algebraic least-squares radius, metres.
  double conservative_radius = 0.;     // About center; includes measured cell squares.
  double residual_rms = 0.;            // Radial RMS, metres.
  double observed_span_rad = 0.;       // 2*pi minus the largest angular gap.
  std::size_t source_count = 0;
  // Local linear sensitivity reserve, NOT a calibrated probability/confidence
  // bound or a proof of coverage of the true/unseen object. No 1/sqrt(n) gain.
  double center_error_bound = 0.;
  bool retained_radius = false; // Conditional rigid-shape fit, not side-arc calibration.
  bool raw_surface = false; // Fit uses actual returns, not occupied-cell centres.
};

namespace object_observation_detail {
constexpr double pi = 3.141592653589793238462643383279502884;
constexpr double cell_half_diagonal = 0.707106781186547524400844362104849039;

inline bool finiteCenter(const std::array<double, 2>& center)
{
  return std::isfinite(center[0]) && std::isfinite(center[1]);
}
}  // namespace object_observation_detail

namespace object_observation_detail {
inline std::optional<CircleObservation> boundCircleObservation(
    const std::vector<std::array<double,2>> &occupied_centers,double resolution,
    const Eigen::Vector2d &center,double radius)
{
  const std::size_t n=occupied_centers.size();
  std::vector<double> angles;
  angles.reserve(n);
  double squared_error = 0., max_distance = 0.;
  Eigen::Vector2d direction_sum = Eigen::Vector2d::Zero();
  Eigen::Matrix2d direction_outer_sum = Eigen::Matrix2d::Zero();
  for (std::size_t i = 0; i < n; ++i) {
    // Measure about the actual returned center, including world-coordinate
    // rounding, so the raster enclosure also holds at large frame offsets.
    const Eigen::Vector2d d(occupied_centers[i][0] - center.x(),
                            occupied_centers[i][1] - center.y());
    const double distance = std::hypot(d.x(), d.y());
    if (!std::isfinite(distance) || distance < .6 * radius) return std::nullopt;
    const double error = distance - radius;
    squared_error += error * error;
    max_distance = std::max(max_distance, distance);
    angles.push_back(std::atan2(d.y(), d.x()));
    const Eigen::Vector2d direction = d / distance;
    direction_sum += direction;
    direction_outer_sum.noalias() += direction * direction.transpose();
  }
  const double residual = std::sqrt(squared_error / static_cast<double>(n));
  if (!std::isfinite(residual) || residual > .8 * resolution) return std::nullopt;
  std::sort(angles.begin(), angles.end());
  double largest_gap = angles.front() + 2. * pi - angles.back();
  for (std::size_t i = 1; i < n; ++i)
    largest_gap = std::max(largest_gap, angles[i] - angles[i - 1]);
  const double span = 2. * pi - largest_gap;
  if (span < pi / 3.) return std::nullopt;

  // Radial-fit Jacobian, eliminating the unknown radius (the mean direction).
  // Worst-direction local amplification grows for poorly observed arcs. This
  // engineering reserve assumes bounded radial perturbations and local model
  // validity; systematic map/pose error and model bias are not bounded here.
  const Eigen::Vector2d mean_direction = direction_sum / static_cast<double>(n);
  const Eigen::Matrix2d sensitivity = direction_outer_sum / static_cast<double>(n) -
      mean_direction * mean_direction.transpose();
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> conditioning(sensitivity);
  if (conditioning.info() != Eigen::Success || !conditioning.eigenvalues().allFinite() ||
      conditioning.eigenvalues()[0] <= 0.) return std::nullopt;

  CircleObservation observation;
  observation.center = {{center.x(), center.y()}};
  observation.radius = radius;
  // sqrt(.5) rather than the truncated .707 covers every corner of every
  // observed resolution-square cell, by the triangle inequality.
  observation.conservative_radius = std::max(
      radius + 1.5 * resolution, max_distance + cell_half_diagonal * resolution);
  observation.residual_rms = residual;
  observation.observed_span_rad = span;
  observation.source_count = n;
  observation.center_error_bound = (residual + 1.5 * resolution) /
      std::sqrt(conditioning.eigenvalues()[0]);
  if (!std::isfinite(observation.conservative_radius) ||
      !std::isfinite(observation.center_error_bound)) return std::nullopt;
  return observation;
}
} // namespace object_observation_detail

// Reproduces the centered algebraic fit in fit_observed_shapes.py: 8..512
// samples, PCA eigenratio >= .025, full rank, radius .15..1.5 m, radial RMS
// <= .8*resolution, and angular span >= pi/3. Reject the WHOLE component on
// nonfinite input; do not silently remove samples or subsample large walls.
// An additional deep-interior veto (any distance < .6*fitted radius) rejects
// filled-disc support even when a coarse raster passes the RMS gate. This is
// deliberately restrictive for thick/noisy shells. Inputs should be unique
// measured occupied cell centers of one component, never predicted/fill cells.
inline std::optional<CircleObservation> fitMeasuredCircle(
    const std::vector<std::array<double, 2>>& occupied_centers,
    double resolution,std::size_t maximum_samples=512)
{
  using namespace object_observation_detail;
  const std::size_t n = occupied_centers.size();
  if (maximum_samples<8||maximum_samples>8192||n < 8 || n > maximum_samples || !std::isfinite(resolution) || resolution <= 0.)
    return std::nullopt;

  // Subtract a measured origin BEFORE averaging/squaring to preserve local
  // geometry at large world coordinates and avoid summing world coordinates.
  if (!finiteCenter(occupied_centers.front())) return std::nullopt;
  const Eigen::Vector2d origin(occupied_centers.front()[0], occupied_centers.front()[1]);
  Eigen::MatrixXd q(static_cast<Eigen::Index>(n), 2);
  for (std::size_t i = 0; i < n; ++i) {
    if (!finiteCenter(occupied_centers[i])) return std::nullopt;
    q(static_cast<Eigen::Index>(i), 0) = occupied_centers[i][0] - origin.x();
    q(static_cast<Eigen::Index>(i), 1) = occupied_centers[i][1] - origin.y();
  }
  if (!q.allFinite()) return std::nullopt;
  const Eigen::Vector2d mean = q.colwise().mean();
  q.rowwise() -= mean.transpose();
  const Eigen::Matrix2d covariance = q.transpose() * q / static_cast<double>(n);
  if (!covariance.allFinite()) return std::nullopt;
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> pca(covariance);
  if (pca.info() != Eigen::Success || !pca.eigenvalues().allFinite() ||
      pca.eigenvalues()[1] <= 0. ||
      pca.eigenvalues()[0] < .025 * pca.eigenvalues()[1])
    return std::nullopt;

  Eigen::MatrixXd a(static_cast<Eigen::Index>(n), 3);
  a.leftCols(2) = 2. * q;
  a.col(2).setOnes();
  const Eigen::VectorXd b = q.rowwise().squaredNorm();
  if (!a.allFinite() || !b.allFinite()) return std::nullopt;
  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(a);
  qr.setThreshold(std::numeric_limits<double>::epsilon() * static_cast<double>(n));
  if (qr.rank() != 3) return std::nullopt;
  const Eigen::Vector3d fit = qr.solve(b);
  if (!fit.allFinite()) return std::nullopt;
  const double radius_squared = fit[2] + fit.head<2>().squaredNorm();
  if (!std::isfinite(radius_squared) || radius_squared <= 0.) return std::nullopt;
  const double radius = std::sqrt(radius_squared);
  if (radius < .15 || radius > 1.5) return std::nullopt;
  const Eigen::Vector2d center = origin + mean + fit.head<2>();
  if (!center.allFinite()) return std::nullopt;

  return object_observation_detail::boundCircleObservation(occupied_centers,resolution,center,radius);
}

// Refine the centre/radius from ACTUAL surface returns while retaining every
// occupied square in collision geometry. Empty, incompatible or unobservable
// raw evidence cannot silently fall back to an overconfident voxel fit.
inline bool encloseMeasuredCircleCells(CircleObservation &circle,
    const std::vector<std::array<double,2>> &cells,double resolution)
{
  if(cells.size()<2||cells.size()>512||!std::isfinite(resolution)||resolution<=0)return false;
  const auto enclosure=object_observation_detail::boundCircleObservation(cells,resolution,
    Eigen::Vector2d(circle.center[0],circle.center[1]),circle.radius);
  if(!enclosure)return false;
  circle.conservative_radius=std::max(circle.conservative_radius,enclosure->conservative_radius);
  circle.center_error_bound=std::max(circle.center_error_bound,enclosure->center_error_bound);
  return true;
}
inline std::optional<CircleObservation> fitRawSurfaceCircle(
    const std::vector<std::array<double,2>> &returns,
    const std::vector<std::array<double,2>> &cells,double resolution)
{
  // The 512-square component bound remains above. A square can contain many
  // actual LiDAR returns: keep ALL (up to a bounded 8192), never subsample a
  // convenient arc or divide the centre error by sqrt(return count).
  auto fit=fitMeasuredCircle(returns,resolution,8192);
  if(!fit||!encloseMeasuredCircleCells(*fit,cells,resolution))return std::nullopt;
  fit->raw_surface=true;return fit;
}

// A recent independently observed rigid radius can constrain a weak new arc.
// Every current point must still satisfy the original radial residual,
// angular-span, conditioning and enclosure gates; no returns are cropped.
inline std::optional<CircleObservation> fitMeasuredCircleWithRadius(
    const std::vector<std::array<double,2>> &points,double resolution,double radius,
    const std::array<double,2> &predicted_center,std::size_t maximum_samples=512)
{
  if(maximum_samples<8||maximum_samples>8192||points.size()<8||points.size()>maximum_samples||!std::isfinite(resolution)||resolution<=0||
     !std::isfinite(radius)||radius<.15||radius>1.5||!object_observation_detail::finiteCenter(predicted_center))return std::nullopt;
  for(const auto &p:points)if(!object_observation_detail::finiteCenter(p))return std::nullopt;
  Eigen::Vector2d center(predicted_center[0],predicted_center[1]);
  for(int iteration=0;iteration<8;++iteration) {
    Eigen::Matrix2d h=Eigen::Matrix2d::Zero();Eigen::Vector2d g=Eigen::Vector2d::Zero();
    for(const auto &p:points) {
      const Eigen::Vector2d d=center-Eigen::Vector2d(p[0],p[1]);const double length=d.norm();
      if(length<.1*radius)return std::nullopt;
      const Eigen::Vector2d j=d/length;h.noalias()+=j*j.transpose();g+=j*(length-radius);
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> e(h);
    if(e.info()!=Eigen::Success||e.eigenvalues()[0]<.001*e.eigenvalues()[1])return std::nullopt;
    const Eigen::Vector2d step=h.ldlt().solve(g);
    if(!step.allFinite()||step.norm()>radius)return std::nullopt;
    center-=step;if(step.norm()<1e-8)break;
  }
  auto fit=object_observation_detail::boundCircleObservation(points,resolution,center,radius);
  if(fit)fit->retained_radius=true;return fit;
}

// One center from one distinct measured scan, already associated by the caller.
// Use the fitted radius, not conservative_radius. All samples share a frame and
// clock; scan timestamps are seconds. Failed geometry can be represented by a
// nonfinite center or nonpositive radius/resolution to break history explicitly.
struct TimedCenter {
  double stamp_s = 0.;
  std::array<double, 2> center{};
  double radius = 0.;
  double resolution = 0.;
};

struct CenterVelocityObservation {
  std::array<double, 2> velocity{};       // Metres/second, constant-velocity LS.
  std::array<double, 2> fitted_center{};  // LS position at stamp_s (not a forecast).
  double stamp_s = 0.;
  double fit_residual_rms = 0.;           // RMS of vector position errors, metres.
  // max ||(center[i]-center[i-1]) - velocity*(stamp[i]-stamp[i-1])||,
  // metres. In-sample diagnostic, NOT an independent prediction error bound.
  double max_step_innovation = 0.;
  double span_s = 0.;
  // OLS slope is a nonnegative weighted average of velocities over the
  // source interval, not instantaneous velocity at stamp_s. For |a|<=A,
  // its latest-velocity discrepancy is <= A*averaging_lag_s (no noise term).
  // This is NOT a different observation/source timestamp.
  double averaging_lag_s = 0.;
  std::size_t source_count = 0;
};

// Pure causal LS over the valid suffix ending at history.back(), using only
// the last .4 s, at least 3 distinct scans and at least .15 s. Does not sort,
// deduplicate, extrapolate, smooth with future scans, gate speed, associate, or
// confirm. Nonfinite/nonincreasing timestamps reject the entire call. Invalid
// geometry, resolution changes, or a fitted-radius range > 2*resolution reset
// the usable suffix; recovery needs 3 new valid scans over .15 s. A failed latest
// sample therefore cannot return a stale velocity. Caller owns/clears history
// across frame/clock changes, missed detections, and association changes.
inline std::optional<CenterVelocityObservation> fitCausalCenterVelocity(
    const std::vector<TimedCenter>& history)
{
  using namespace object_observation_detail;
  if (history.size() < 3) return std::nullopt;
  for (std::size_t i = 0; i < history.size(); ++i)
    if (!std::isfinite(history[i].stamp_s) ||
        (i != 0 && history[i].stamp_s <= history[i - 1].stamp_s))
      return std::nullopt;

  const double latest = history.back().stamp_s;
  // Allow floating-point representation error at epoch-sized timestamps.
  const double time_tolerance = std::min(1.e-6,
      8. * std::numeric_limits<double>::epsilon() * std::max(1., std::abs(latest)));
  std::size_t first = history.size();
  double min_radius = 0., max_radius = 0.;
  for (std::size_t i = 0; i < history.size(); ++i) {
    const auto& sample = history[i];
    const double age = latest - sample.stamp_s;
    if (!std::isfinite(age)) return std::nullopt;
    if (age > .4 + time_tolerance) continue;
    if (!finiteCenter(sample.center) || !std::isfinite(sample.radius) ||
        sample.radius < .15 || sample.radius > 1.5 ||
        !std::isfinite(sample.resolution) || sample.resolution <= 0.) {
      first = history.size();
      continue;
    }
    const double resolution_tolerance = 32. * std::numeric_limits<double>::epsilon() *
        std::max(sample.resolution, first == history.size() ? sample.resolution :
                 history[first].resolution);
    if (first == history.size() ||
        std::abs(sample.resolution - history[first].resolution) > resolution_tolerance ||
        std::max(max_radius, sample.radius) - std::min(min_radius, sample.radius) >
            2. * sample.resolution) {
      first = i;
      min_radius = max_radius = sample.radius;
    } else {
      min_radius = std::min(min_radius, sample.radius);
      max_radius = std::max(max_radius, sample.radius);
    }
  }
  if (first == history.size() || history.size() - first < 3) return std::nullopt;
  const double span = latest - history[first].stamp_s;
  if (!std::isfinite(span) || span + time_tolerance < .15) return std::nullopt;

  // Fit relative times/positions for numerical stability at large coordinates.
  const std::size_t n = history.size() - first;
  const auto& origin = history.back().center;
  double mean_t = 0.;
  std::array<double, 2> mean_position{};
  for (std::size_t i = first; i < history.size(); ++i) {
    mean_t += (history[i].stamp_s - latest) / static_cast<double>(n);
    for (std::size_t axis = 0; axis < 2; ++axis)
      mean_position[axis] += (history[i].center[axis] - origin[axis]) / static_cast<double>(n);
  }
  double time_squared_sum = 0.,time_cubed_sum=0.;
  std::array<double, 2> cross_sum{};
  for (std::size_t i = first; i < history.size(); ++i) {
    const double dt = history[i].stamp_s - latest - mean_t;
    time_squared_sum += dt * dt;
    time_cubed_sum += dt * dt * dt;
    for (std::size_t axis = 0; axis < 2; ++axis)
      cross_sum[axis] += dt * (history[i].center[axis] - origin[axis] - mean_position[axis]);
  }
  if (!std::isfinite(time_squared_sum) || time_squared_sum <= 0.) return std::nullopt;
  CenterVelocityObservation observation;
  observation.stamp_s = latest;
  observation.span_s = span;
  observation.source_count = n;
  const double lag=-mean_t-.5*time_cubed_sum/time_squared_sum;
  if(!std::isfinite(lag)||lag < -time_tolerance||lag>span+time_tolerance)return std::nullopt;
  observation.averaging_lag_s=std::clamp(lag,0.,span);
  std::array<double, 2> intercept{};
  for (std::size_t axis = 0; axis < 2; ++axis) {
    observation.velocity[axis] = cross_sum[axis] / time_squared_sum;
    intercept[axis] = mean_position[axis] - observation.velocity[axis] * mean_t;
    observation.fitted_center[axis] = origin[axis] + intercept[axis];
  }
  if (!finiteCenter(observation.velocity) || !finiteCenter(observation.fitted_center))
    return std::nullopt;
  double squared_error = 0.;
  for (std::size_t i = first; i < history.size(); ++i) {
    std::array<double, 2> error{}, step_error{};
    for (std::size_t axis = 0; axis < 2; ++axis) {
      error[axis] = history[i].center[axis] - origin[axis] - intercept[axis] -
          observation.velocity[axis] * (history[i].stamp_s - latest);
      if (i != first)
        step_error[axis] = history[i].center[axis] - history[i - 1].center[axis] -
            observation.velocity[axis] * (history[i].stamp_s - history[i - 1].stamp_s);
    }
    const double residual = std::hypot(error[0], error[1]);
    const double innovation = std::hypot(step_error[0], step_error[1]);
    if (!std::isfinite(residual) || !std::isfinite(innovation)) return std::nullopt;
    squared_error += residual * residual / static_cast<double>(n);
    observation.max_step_innovation = std::max(observation.max_step_innovation, innovation);
  }
  observation.fit_residual_rms = std::sqrt(squared_error);
  if (!std::isfinite(observation.fit_residual_rms)) return std::nullopt;
  return observation;
}

}  // namespace terrain_nav
