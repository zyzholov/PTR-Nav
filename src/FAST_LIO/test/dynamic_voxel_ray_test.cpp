#include "dynamic_voxel_ray.hpp"

#include <array>
#include <cassert>
#include <unordered_set>

int main() {
  using dynamic_voxel_ray::Key;
  std::unordered_set<Key, dynamic_voxel_ray::Hash> passed;
  const auto scan = [&](std::array<double, 3> endpoint) {
    passed.clear();
    dynamic_voxel_ray::traverse({0.0, 0.0, 0.0}, endpoint, 0.2, 20.0, 0.65,
                                [&](const Key &key) { passed.insert(key); });
  };

  scan({10.0, 0.1, 0.1});
  assert(passed.count({25, 0, 0}) == 1);  // A farther return proves the old cell free.
  assert(passed.count({49, 0, 0}) == 0);  // The endpoint is occupied, not free.

  scan({4.0, 0.1, 0.1});
  assert(passed.count({25, 0, 0}) == 0);  // An occluding return cannot clear behind it.

  scan({10.0, 2.0, 0.1});
  assert(passed.count({25, 0, 0}) == 0);  // A nearby, nonintersecting ray is not proof.

  dynamic_voxel_ray::EvidenceWindow evidence;
  assert(!evidence.observe(1, 3, 3));
  assert(!evidence.observe(1, 3, 3));  // Repeated rays in one scan count once.
  assert(!evidence.observe(5, 3, 3));  // Old isolated evidence expires.
  assert(!evidence.observe(6, 3, 3));
  assert(evidence.observe(8, 3, 3));
  evidence.reset();                     // A current hit makes the cell occupied again.
  assert(!evidence.observe(9, 3, 3));

  std::unordered_set<Key, dynamic_voxel_ray::Hash> cells{
      {0, 0, 0}, {1, 0, 0}, {2, 1, 0}, {5, 0, 0}, {6, 0, 0}};
  std::unordered_set<Key, dynamic_voxel_ray::Hash> connected;
  dynamic_voxel_ray::flood_connected(
      {0, 0, 0}, [&](const Key &key) { return cells.count(key) != 0; },
      [&](const Key &key) { connected.insert(key); }, 20);
  assert(connected.count({2, 1, 0}) == 1);
  assert(connected.count({5, 0, 0}) == 0);  // A separated static object is not part of the cluster.
}
