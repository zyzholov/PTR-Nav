"""Current-scan vertical occupancy, kept separate from smoothed terrain risk."""

import numpy as np


def vertical_columns(points, shape, origin, resolution, *, min_height=0.32,
                     min_span=0.18, min_hits=2, neighbor_radius=0.6, xp=np):
    rows, cols = shape
    points = xp.asarray(points, dtype=xp.float32)
    low = xp.full(rows * cols, xp.inf, dtype=xp.float32)
    high = xp.full(rows * cols, -xp.inf, dtype=xp.float32)
    if points.size == 0:
        return xp.zeros(shape, dtype=bool), low.reshape(shape)
    points = points[xp.isfinite(points).all(axis=1)]
    ix = xp.floor((points[:, 0] - origin[0]) / resolution).astype(xp.int32)
    iy = xp.floor((points[:, 1] - origin[1]) / resolution).astype(xp.int32)
    inside = (ix >= 0) & (ix < cols) & (iy >= 0) & (iy < rows)
    ids = iy[inside] * cols + ix[inside]
    z = points[inside, 2]
    xp.minimum.at(low, ids, z)
    xp.maximum.at(high, ids, z)
    low, high = low.reshape(shape), high.reshape(shape)
    ground = low.copy()
    radius = max(1, int(np.ceil(neighbor_radius / resolution)))
    # Bounded shifts avoid wrap-around contamination at rolling-map edges.
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            y0, y1 = max(0, -dy), min(rows, rows - dy)
            x0, x1 = max(0, -dx), min(cols, cols - dx)
            ground[y0:y1, x0:x1] = xp.minimum(
                ground[y0:y1, x0:x1], low[y0+dy:y1+dy, x0+dx:x1+dx])
    hits = xp.zeros(rows * cols, dtype=xp.int32)
    relative = z - ground.ravel()[ids]
    # Do not classify a high canopy as a bumper-height obstacle.
    colliding = (relative >= min_height) & (relative <= 1.5)
    xp.add.at(hits, ids[colliding], 1)
    observed = xp.isfinite(low) & xp.isfinite(high)
    span = xp.where(observed, high, 0) - xp.where(observed, low, 0)
    occupied = observed & (span >= min_span) & (hits.reshape(shape) >= min_hits)
    return occupied, ground


def preserve_collision_risk(result, occupied):
    """Hard geometry cannot be averaged away by BGK/SEB fusion weights."""
    result.baseline_risk = np.maximum(result.baseline_risk, occupied.astype(np.float32))
    result.final_risk_slices = [np.where(occupied, 1.0, risk).astype(np.float32)
                               for risk in result.final_risk_slices]
