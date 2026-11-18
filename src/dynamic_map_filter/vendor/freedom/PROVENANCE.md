# FreeDOM core

Source: https://github.com/LC-Robotics/FreeDOM
Commit: dcfd5690cafddf121a80bd8ea477807d9656748e
License: MIT (original LICENSE included).

Only the native mapping/depth/raycast core is vendored. ROS1 node/visualization wrappers are excluded. `ROS_FREE_PORT.patch` lists the ROS-free changes: remove unused ROS includes/helpers, add explicit standard/PCL includes, suppress per-stage synchronous stdout, initialize unused-depth-image `learn_fov` for safe destruction.

`HISTORY_RING_FIX.patch` fixes one upstream logic error in `MRMap::map_removal`: the second aggressive ring was gated by `!cluster_aggressive_twice`, opposite to `scan_removal` and the configured connectivity. The condition is changed to `cluster_aggressive_twice`. The controlled same-scan line test `Freedom.HistoricalMapAppliesTheConfiguredSecondAggressiveRing` fails before this change and passes after it; a disconnected occupied cell stays static. This is a locally corrected FreeDOM core, not an unmodified reproduction of its published metrics. No raycasting, inpainting, connectivity radii, or risk budgets are changed by this fix.

This is an experimental auxiliary backend. Its inferred depth completion is **not** a measured hit or proof of navigable ground. Raw returns must remain unfiltered in live collision/terrain input.

`MEASURED_RAYS_AND_BOUNDARIES.patch` records a further local correction (2026-11-08). Free-space traversal now uses each actual deskewed return retained in the scan block instead of the occupied-voxel centroid. The DDA follows the map's half-open floor convention at starting faces and crosses simultaneous faces together, without visiting zero-length corner cells. Current occupied cells still stop traversal; multiple rays in a frame still contribute at most one temporal free vote. Independent segment/box tests cover both signs, exact faces, shared endpoint voxels and occlusion. Depth completion remains disabled in production. This correction does not establish a real-time clearing guarantee; the old recorded stream had the same sustained clearing delay and slightly higher native runtime after the change.
