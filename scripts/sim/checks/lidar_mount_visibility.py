#!/usr/bin/env python3
"""Compare MID360 pitch choices on recorded vehicle and cylinder poses.

This observer-only geometric check omits occlusion, reflectivity, scan
distortion, registration and tracker filtering. A possible intersection is
not a measured return. The recorded tracking cells remain separate.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src/isaac_sim_hunter_livo2/scripts'))
from mid360_scan import Mid360ScanTable
from lidar_mount import DEFAULT_PITCH_DEG, LIVOX_BASE_TRANSLATE, LIDAR_FRAME_TRANSLATE


def rotation(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([[cy, -sy, 0.], [sy, cy, 0.], [0., 0., 1.]]) @ np.array(
        [[cp, 0., sp], [0., 1., 0.], [-sp, 0., cp]]) @ np.array(
        [[1., 0., 0.], [0., cr, -sr], [0., sr, cr]])


def side_hits(rays, origin, xy, radius, min_z, max_z):
    """Ray/vertical cylinder side intersections over the requested height."""
    relative = origin[:2] - xy
    a = np.sum(rays[:, :2]**2, axis=1)
    b = rays[:, :2] @ relative
    disc = b*b - a*(relative @ relative - radius*radius)
    usable = (disc >= 0.) & (a > 1e-12)
    distance = (-b[usable] - np.sqrt(disc[usable])) / a[usable]
    z = origin[2] + rays[usable, 2]*distance
    return int(np.count_nonzero((distance >= .1) & (distance <= 70.) &
                                (z >= min_z) & (z <= max_z)))


def analyse(run, target, start, end, radius, height, pitches):
    nav = json.loads((run / 'navigation_data.json').read_text())
    raw = json.loads((run / 'motion_grids.json').read_text())
    if raw['overflow'] or not raw['tracking_frames']:
        raise ValueError('complete tracking occupancy log required')
    truth = np.asarray(nav['truth'])
    odom = np.asarray(nav['localization_poses'])
    actor = np.asarray(nav['moving_obstacles'][target])
    ref = next(row for row in odom if truth[0, 6] <= row[0] <= truth[-1, 6])
    truth_yaw = np.unwrap(truth[:, 7])
    ref_xyz = np.array([np.interp(ref[0], truth[:, 6], truth[:, i]) for i in (1, 2, 3)])
    world_to_map = rotation(0., 0., ref[4] - np.interp(ref[0], truth[:, 6], truth_yaw))

    table = Mid360ScanTable()
    azimuth, elevation = np.radians(table.azimuth), np.radians(table.elevation)
    directions = np.stack((np.cos(elevation)*np.cos(azimuth),
                           np.cos(elevation)*np.sin(azimuth), np.sin(elevation)), axis=-1)
    mounts = {str(p): rotation(0., math.radians(p), 0.) for p in pitches}
    rows = []
    for scan in raw['tracking_frames']:
        stamp = scan['stamp_s']
        if not max(start, truth[0, 6], actor[0, 0]) <= stamp <= min(end, truth[-1, 6], actor[-1, 0]):
            continue
        x, y, z, pitch, roll = (np.interp(stamp, truth[:, 6], truth[:, k]) for k in (1, 2, 3, 4, 5))
        yaw = np.interp(stamp, truth[:, 6], truth_yaw)
        body = rotation(roll, pitch, yaw)
        xy = np.array([np.interp(stamp, actor[:, 0], actor[:, k]) for k in (1, 2)])
        center_in_map = (world_to_map @ (np.r_[xy, 0.] - ref_xyz))[:2] + ref[1:3]
        width, _, grid_x, grid_y, res = scan['geometry']
        cells = np.asarray(scan['occupied'], dtype=int)
        cell_xy = np.column_stack((grid_x + (cells % width + .5)*res,
                                   grid_y + (cells // width + .5)*res))
        nearby = int(np.count_nonzero(np.linalg.norm(cell_xy-center_in_map, axis=1) <= radius+.11))
        rays = directions[table.phase_at(stamp)]
        estimates = {}
        for p, mount in mounts.items():
            pos = np.array([x, y, z]) + body @ (LIVOX_BASE_TRANSLATE + mount @ LIDAR_FRAME_TRANSLATE)
            world_rays = rays @ (body @ mount).T
            estimates[p] = {
                'body_band': side_hits(world_rays, pos, xy, radius, .10, min(.65, height)),
                'full_height': side_hits(world_rays, pos, xy, radius, .10, height),
            }
        rows.append({'stamp_s': stamp, 'tracking_cells_near_target': nearby, 'possible_hits': estimates})
    if not rows:
        raise ValueError('no aligned frames in requested interval')
    return {
        'scope': __doc__.strip(), 'run': str(run), 'target': target, 'ray_table_sha256': table.sha256,
        'frame_count': len(rows),
        'measured_zero_cell_frames': sum(r['tracking_cells_near_target'] == 0 for r in rows),
        'possible_zero_hit_frames': {
            p: {band: sum(r['possible_hits'][p][band] == 0 for r in rows)
                for band in ('body_band', 'full_height')} for p in mounts},
        'rows': rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--target', required=True)
    parser.add_argument('--start', type=float, required=True)
    parser.add_argument('--end', type=float, required=True)
    parser.add_argument('--radius', type=float, required=True)
    parser.add_argument('--height', type=float, required=True)
    parser.add_argument('--pitch', type=float, nargs='+', default=[DEFAULT_PITCH_DEG, 5., 0.])
    args = parser.parse_args()
    if (not all(map(math.isfinite, [args.start, args.end, args.radius, args.height, *args.pitch])) or
            not (0 <= args.start < args.end and args.radius > 0 and args.height >= .10)):
        parser.error('invalid geometry or time range')
    result = analyse(args.run, args.target, args.start, args.end, args.radius, args.height, args.pitch)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
