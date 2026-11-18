#!/usr/bin/env python3
"""Audit a frozen map/scan projection; never runs a simulator or navigation trial."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from fsgp_bgk_se2.overhead_clearance import measured_ceiling, clip_overhead
from fsgp_bgk_se2.obstacle_columns import vertical_columns


def analyze(elevation, variance, observed, points, origin, resolution, height, margin):
    raw = np.asarray(points, dtype=np.float32)
    raw = raw[np.isfinite(raw).all(axis=1)]
    ground = np.where(observed, elevation, np.nan)
    floor = observed & np.isfinite(variance) & (variance >= 0) & (variance <= .02**2)
    start = perf_counter()
    ceiling = measured_ceiling(elevation, floor, raw, origin, resolution,
                               vehicle_height_m=height, margin_m=margin)
    clipped = clip_overhead(raw, ceiling, origin, resolution)
    elapsed_ms = (perf_counter()-start)*1000
    before, _ = vertical_columns(raw, elevation.shape, origin, resolution,
                                 min_hits=1, ground_elevation=ground)
    after, _ = vertical_columns(clipped, elevation.shape, origin, resolution,
                                min_hits=1, ground_elevation=ground)
    col = np.floor((raw[:, 0]-origin[0])/resolution).astype(int)
    row = np.floor((raw[:, 1]-origin[1])/resolution).astype(int)
    inside = (row >= 0) & (col >= 0) & (row < elevation.shape[0]) & (col < elevation.shape[1])
    removed = np.zeros(len(raw), bool)
    ids = np.flatnonzero(inside)
    removed[ids] = raw[ids, 2] > ceiling[row[ids], col[ids]]
    summary = dict(raw_points=len(raw), kept_points=len(clipped), removed_high_points=int(removed.sum()),
                   qualified_cells=int(np.isfinite(ceiling).sum()),
                   occupied_before=int(before.sum()), occupied_after=int(after.sum()),
                   removed_occupied_cells=int((before & ~after).sum()),
                   added_occupied_cells=int((after & ~before).sum()),
                   removed_xyz_map_m=raw[removed].tolist(),
                   slice_single_call_ms=elapsed_ms,
                   timing_scope='one diagnostic call, not a latency benchmark',
                   raw_input_unchanged=bool(np.array_equal(raw, points)))
    arrays = dict(raw=raw, removed=removed, ceiling=ceiling, before=before, after=after,
                  elevation=elevation, observed=observed, origin=np.array(origin), resolution=resolution)
    return summary, arrays


def plot(arrays, pose, config, output):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Polygon
    from plot_style import configure_chinese
    configure_chinese()
    raw, removed = arrays['raw'], arrays['removed']
    origin, res = arrays['origin'], arrays['resolution']
    h, w = arrays['elevation'].shape
    extent = (origin[0], origin[0]+w*res, origin[1], origin[1]+h*res)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), layout='constrained')
    im = axes[0, 0].imshow(np.where(arrays['observed'], arrays['elevation'], np.nan),
                           origin='lower', extent=extent, cmap='terrain', interpolation='nearest')
    fig.colorbar(im, ax=axes[0, 0], label='实测地面高程（米）')
    axes[0, 0].set_title('实测地面；空白不是已知安全区域')
    im = axes[0, 1].imshow(np.where(np.isfinite(arrays['ceiling']), arrays['ceiling'], np.nan),
                           origin='lower', extent=extent, cmap='cividis', interpolation='nearest')
    fig.colorbar(im, ax=axes[0, 1], label='碰撞切片上界高程（米）')
    axes[0, 1].set_title('有实测平面依据的上界；不向地图添加地面')
    occupancy = np.full((h, w), np.nan)
    occupancy[arrays['after']] = 0
    occupancy[arrays['before'] & ~arrays['after']] = 1
    axes[1, 0].imshow(occupancy, origin='lower', extent=extent,
                      cmap=ListedColormap(['#b7474d', '#db972f']), vmin=0, vmax=1, interpolation='nearest')
    axes[1, 0].scatter([], [], marker='s', color='#b7474d', label='仍保留的障碍格')
    axes[1, 0].scatter([], [], marker='s', color='#db972f', label='仅因净空切片移除的格')
    c, s = np.cos(pose[3]), np.sin(pose[3])
    f = max(config['footprint_front'], config['body_collision_front'])+config['footprint_margin']
    b = max(config['footprint_back'], config['body_collision_back'])+config['footprint_margin']
    width = max(config['footprint_half_width'], config['body_collision_half_width'])+config['footprint_margin']
    corners = np.array([[f,width],[-b,width],[-b,-width],[f,-width]])
    xy = corners @ np.array([[c,s],[-s,c]]) + pose[:2]
    axes[1, 0].add_patch(Polygon(xy, fill=False, edgecolor='#237d8f', linewidth=1.8, label='当前保护包络'))
    axes[1, 0].set_title('占据格对照；保留的支柱不能随面板一并清除')
    axes[1, 0].legend(fontsize=8)
    near = (abs(raw[:,0]-pose[0]) <= 3) & (abs(raw[:,1]-pose[1]) <= 3)
    axes[1, 1].scatter(raw[near & ~removed,0], raw[near & ~removed,2], s=7,
                       color='#416d89', label='保留回波')
    axes[1, 1].scatter(raw[near & removed,0], raw[near & removed,2], s=35,
                       marker='x', color='#db972f', label='过滤的高处回波')
    axes[1, 1].set(title='局部 X-Z 回波投影（Y 方向合并）', xlabel='地图 X（米）', ylabel='地图 Z（米）')
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat[:3]:
        ax.set(xlim=(pose[0]-3,pose[0]+3), ylim=(pose[1]-3,pose[1]+3),
               xlabel='地图 X（米）', ylabel='地图 Y（米）', aspect='equal')
    fig.suptitle('冻结失败现场的碰撞几何审计；不是重新导航或通过结果')
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    from rclpy.serialization import deserialize_message
    from se2_grid_msgs.msg import SE2Grid
    from sensor_msgs.msg import PointCloud2
    from sensor_msgs_py.point_cloud2 import read_points
    import yaml
    from fsgp_bgk_se2.se2_grid_tools import get_layer_2d, grid_buffer_to_world, matrix_shape
    root = Path(__file__).resolve().parents[3]
    diagnostic = json.loads((args.input/'diagnostic.json').read_text())
    payloads = {name:(args.input/name).read_bytes() for name in ('map.cdr','completion_cloud.cdr')}
    hashes = {name:hashlib.sha256(value).hexdigest() for name,value in payloads.items()}
    if (hashes['map.cdr'] != diagnostic['map_sha256'] or
            hashes['completion_cloud.cdr'] != diagnostic['completion_cloud']['sha256']):
        raise ValueError('snapshot hash mismatch')
    msg = deserialize_message(payloads['map.cdr'], SE2Grid)
    cloud = deserialize_message(payloads['completion_cloud.cdr'], PointCloud2)
    delta = (msg.info.header.stamp.sec-cloud.header.stamp.sec +
             1e-9*(msg.info.header.stamp.nanosec-cloud.header.stamp.nanosec))
    if (cloud.header.frame_id != msg.info.header.frame_id or
            abs(delta-diagnostic['completion_cloud']['source_age_s']) > 5e-9 or not 0 <= delta <= .3):
        raise ValueError('snapshot frame or conditioning time mismatch')
    rows = read_points(cloud, field_names=('x','y','z'), skip_nans=True)
    points = np.column_stack([rows[key] for key in ('x','y','z')])
    shape = matrix_shape(msg.data[0])
    layer = lambda name:grid_buffer_to_world(get_layer_2d(msg,name,np.nan,shape)[0],msg)
    config_path = root/'src/planner/terrain_nav/config/hunter_gazebo.yaml'
    config = next(iter(yaml.safe_load(config_path.read_text()).values()))['ros__parameters']
    origin = (msg.info.pose.position.x-msg.info.length_x/2, msg.info.pose.position.y-msg.info.length_y/2)
    summary, arrays = analyze(layer('gp_elevation_observed'), layer('gp_variance_observed'),
        layer('terrain_observed')>.5, points, origin, msg.info.pos_resolution,
        config['vehicle_envelope_height_m'], config['overhead_clearance_margin_m'])
    summary.update(scope='frozen geometry audit, not physical navigation acceptance',
        input=str(args.input), input_sha256=hashes, pose=diagnostic['pose'],
        current_geometry=dict(height=config['vehicle_envelope_height_m'],headroom=config['overhead_clearance_margin_m']),
        source_sha256={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), config_path, root/'src/fsgp_bgk_se2/fsgp_bgk_se2/overhead_clearance.py',
                         root/'src/fsgp_bgk_se2/fsgp_bgk_se2/obstacle_columns.py')})
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output/'geometry.npz', **arrays)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    if args.plot:plot(arrays,np.array(diagnostic['pose']),config,args.output/'projection.png')
    print(json.dumps(summary,indent=2,allow_nan=False))


if __name__ == '__main__':main()
