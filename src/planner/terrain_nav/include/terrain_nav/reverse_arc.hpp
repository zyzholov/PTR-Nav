#pragma once

#include <array>
#include <cmath>
#include <vector>

namespace terrain_nav
{
// A single reverse gear, bounded-curvature Ackermann arc. Yaw is BODY yaw.
inline std::vector<std::array<double, 3>> reverseArc(
  double x, double y, double yaw, double gx, double gy,
  double max_distance, double max_curvature, double spacing = 0.05)
{
  const double c = std::cos(yaw), s = std::sin(yaw);
  const double dx = gx - x, dy = gy - y;
  const double lx = c * dx + s * dy, ly = -s * dx + c * dy;
  const double distance = std::hypot(lx, ly);
  if (lx >= -0.30 || distance > max_distance || distance < 0.30) return {};
  const double curvature = 2.0 * ly / (lx * lx + ly * ly);
  if (std::abs(curvature) > max_curvature) return {};
  const double turn = 2.0 * std::atan2(-ly, -lx);
  const double length = std::abs(curvature) > 1e-6 ? turn / curvature : -distance;
  if (length >= 0.0 || std::abs(length) > 1.5 * distance) return {};
  const int steps = std::max(2, static_cast<int>(std::ceil(std::abs(length) / spacing)));
  std::vector<std::array<double, 3>> path;
  path.reserve(steps + 1);
  for (int i = 0; i <= steps; ++i) {
    const double ds = length * i / steps, th = ds * curvature;
    const double px = std::abs(curvature) > 1e-6 ? std::sin(th) / curvature : ds;
    const double py = std::abs(curvature) > 1e-6 ? (1.0 - std::cos(th)) / curvature : 0.0;
    path.push_back({x + c * px - s * py, y + s * px + c * py, yaw + th});
  }
  return path;
}
}  // namespace terrain_nav
