#!/usr/bin/env python3
"""Validate synchronized cloud and terrain-mesh products used by RViz."""

import argparse
import math
import statistics
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import Marker
from se2_grid_msgs.msg import SE2Grid
from fsgp_bgk_se2.se2_grid_tools import get_layer_2d, get_layer_slices, grid_buffer_to_world, layer_refs, matrix_shape, serialized_header


class SerializedSamples:
    """Bounded original-stamp content samples; no decoding on unmatched frames."""
    def __init__(self,names,depth=6):
        self.pending={name:{} for name in names}
        self.depth=depth

    def add(self,name,payload):
        stamp=serialized_header(payload).stamp
        key=(stamp.sec,stamp.nanosec)
        self.pending[name][key]=payload
        while len(self.pending[name])>self.depth:del self.pending[name][min(self.pending[name])]
        common=set.intersection(*(set(cache) for cache in self.pending.values()))
        return (max(common),{n:c[max(common)] for n,c in self.pending.items()}) if common else None


def surface_evidence(msg,measured_only=False):
    refs=layer_refs(msg);shape=matrix_shape(msg.data[0])
    def layer(name,default=np.nan):
        return grid_buffer_to_world(get_layer_2d(msg,name,default,shape)[0],msg)
    measured=layer('gp_elevation_observed') if 'gp_elevation_observed' in refs else layer('elevation')
    observed=layer('terrain_observed',1)>.5
    predicted=layer('probability_elevation');qualified=layer('probability_gp_used',0)>.5
    display=np.where(observed&np.isfinite(measured),measured,
                     np.nan if measured_only else np.where(qualified,predicted,np.nan))
    resolution=msg.info.pos_resolution
    x0=msg.info.pose.position.x-msg.info.length_x/2+.5*resolution
    y0=msg.info.pose.position.y-msg.info.length_y/2+.5*resolution
    expected={};prediction_positions=set()
    for iy,ix in zip(*np.where(np.isfinite(display))):
        key=(round(x0+ix*resolution,2),round(y0+iy*resolution,2))
        expected[key]=float(display[iy,ix]+.02)
        if not observed[iy,ix] and qualified[iy,ix]:prediction_positions.add(key)
    return expected,prediction_positions


def cloud_evidence(msg,name):
    """Every eligible same-stamp source cell, independent of scene size.

    Risk yaw is chosen asynchronously by the renderer's odometry subscriber;
    return each actual slice as a permitted complete product, not their union.
    """
    refs=layer_refs(msg);shape=matrix_shape(msg.data[0])
    def scalar(n,default=np.nan):return grid_buffer_to_world(get_layer_2d(msg,n,default,shape)[0],msg)
    if name=='elevation':
        z=scalar('elevation')+.02;masks=[np.isfinite(z)&(scalar('valid',1)>=.5)]
    else:
        z=scalar('probability_elevation')+.035
        risks=get_layer_slices(msg,'failure_probability_upper')[0]
        valid=get_layer_slices(msg,'probability_valid')[0]
        masks=[np.isfinite(grid_buffer_to_world(a,msg))&
               (grid_buffer_to_world(valid[min(i,len(valid)-1)],msg)>=.5)&np.isfinite(z)
               for i,a in enumerate(risks)] if valid else []
    res=msg.info.pos_resolution;x0=msg.info.pose.position.x-msg.info.length_x/2+.5*res
    y0=msg.info.pose.position.y-msg.info.length_y/2+.5*res
    return [{(round(x0+c*res,2),round(y0+r*res,2)):float(z[r,c])
             for r,c in zip(*np.nonzero(mask))}for mask in masks]


def require_cloud_fidelity(points,candidates,name):
    actual={(round(x,2),round(y,2)):z for x,y,z in points}
    if len(actual)!=len(points):raise RuntimeError(f'{name}: duplicate displayed cells')
    if not any(actual.keys()==expected.keys() and all(abs(actual[k]-v)<=1e-4 for k,v in expected.items())
               for expected in candidates):
        raise RuntimeError(f'{name}: missing, fabricated or misplaced same-stamp source cells')


def smooth_quad_exists(evidence,resolution):
    for (x,y),z in evidence.items():
        heights=[evidence.get((round(x+dx*resolution,2),round(y+dy*resolution,2)))
                 for dx,dy in ((0,0),(1,0),(0,1),(1,1))]
        if all(v is not None for v in heights) and max(heights)-min(heights)<=.25:return True
    return False


class VisualizationProbe(Node):
    def __init__(self):
        super().__init__("hunter_visualization_probe")
        self.elevation = None
        self.risk = None
        self.surface = None
        self.height_surface = None
        self.elevation_fields = set()
        self.risk_fields = set()
        self.pending = {name: {} for name in ("elevation", "risk", "surface", "height_surface", "grid")}
        self.sample_stamp = None
        self.lidar_pending={name:{} for name in ('raw_map','map','raw_live','live')}
        self.navigation_scans={}
        self.density_matches=[]
        self.lidar_matches={}
        self.lidar_checked={}
        self.lidar_received={name:0 for name in (*self.lidar_pending,'nav_live')}
        self.serialized_surface=SerializedSamples(self.pending)
        self.serialized_lidar={'map':SerializedSamples(('map','raw_map'),4),
                              'live':SerializedSamples(('live','raw_live','nav_live'),12)}
        # Content qualification takes ONE exact-stamp sample. Decoding the
        # mesh, two clouds and grid on every callback can itself force the
        # two depth-1 LiDAR readers into permanently disjoint frame phases.
        # Collect bounded serialized originals first; decode a matched set.
        self.surface_types={
            'elevation':(PointCloud2,self._elevation_cb), 'risk':(PointCloud2,self._risk_cb),
            'surface':(Marker,self._surface_cb), 'height_surface':(Marker,self._height_surface_cb),
            'grid':(SE2Grid,self._grid_cb)}
        for name,topic in (
            ('elevation','/car_0_terrain_analyzer/elevation_cloud'),
            ('risk','/car_0_terrain_analyzer/risk_cloud'),
            ('surface','/car_0_terrain_analyzer/terrain_surface'),
            ('height_surface','/terrain_display/elevation_surface'),
            ('grid','/car_0_fsgp_bgk/fused_map')):
            self.create_subscription(self.surface_types[name][0],topic,
                lambda payload,n=name:self._raw_surface_cb(n,payload),1,raw=True)
        for name,topic in [('raw_map','/Laser_map'),('map','/terrain_display/lidar_map'),
                           ('raw_live','/cloud_registered_visual'),('live','/terrain_display/lidar_returns')]:
            self.create_subscription(PointCloud2,topic,lambda payload,n=name:self._raw_lidar_cb(n,payload),1,raw=True)
        self.create_subscription(PointCloud2,'/cloud_registered',lambda payload:self._raw_lidar_cb('nav_live',payload),1,raw=True)

    def _raw_surface_cb(self,name,payload):
        if self.sample_stamp is not None:return
        sample=self.serialized_surface.add(name,payload)
        if sample is None:return
        _,payloads=sample
        for n,p in payloads.items():
            kind,callback=self.surface_types[n]
            callback(deserialize_message(p,kind))
        if self.sample_stamp is not None:self.serialized_surface.pending={n:{} for n in self.pending}

    def _raw_lidar_cb(self,name,payload):
        self.lidar_received[name]+=1
        base='live' if name=='nav_live' else name.removeprefix('raw_')
        if base in self.lidar_matches:return
        sample=self.serialized_lidar[base].add(name,payload)
        if sample is None:return
        _,payloads=sample
        for n,p in payloads.items():
            msg=deserialize_message(p,PointCloud2)
            if n=='nav_live':self._navigation_scan_cb(msg)
            else:self._lidar_cb(n,msg,count_received=False)
        if base in self.lidar_matches:self.serialized_lidar[base].pending={n:{} for n in payloads}

    def _navigation_scan_cb(self,msg):
        key=(msg.header.stamp.sec,msg.header.stamp.nanosec)
        self.navigation_scans[key]=msg.width*msg.height
        while len(self.navigation_scans)>12:del self.navigation_scans[min(self.navigation_scans)]
        self._check_density()

    def _check_density(self):
        common=self.navigation_scans.keys()&self.lidar_pending['raw_live'].keys()
        if common:
            key=max(common);dense=self.lidar_pending['raw_live'][key]
            self.density_matches.append((self.navigation_scans[key],dense.width*dense.height))

    def _lidar_cb(self,name,msg,count_received=True):
        if count_received:self.lidar_received[name]+=1
        key=(msg.header.stamp.sec,msg.header.stamp.nanosec)
        self.lidar_pending[name][key]=msg
        while len(self.lidar_pending[name])>12:del self.lidar_pending[name][min(self.lidar_pending[name])]
        if name=='raw_live':self._check_density()
        base=name.removeprefix('raw_')
        common=self.lidar_pending[base].keys()&self.lidar_pending['raw_'+base].keys()
        if common:
            key=max(common)
            if self.lidar_checked.get(base)==key:return
            raw=self.lidar_pending['raw_'+base][key];colored=self.lidar_pending[base][key]
            a=point_cloud2.read_points(raw,field_names=('x','y','z'))
            b=point_cloud2.read_points(colored,field_names=('x','y','z','rgb'))
            if len(a)!=len(b) or raw.header!=colored.header:
                raise RuntimeError(f'colored {base} changed source point count or header')
            for axis in ('x','y','z'):
                if not np.array_equal(a[axis],b[axis],equal_nan=True):
                    raise RuntimeError(f'colored {base} changed measured coordinates')
            self.lidar_matches[base]=dict(points=len(a),colors=len(np.unique(b['rgb'].view(np.uint32))))
            self.lidar_checked[base]=key

    def _grid_cb(self,msg):
        self._remember('grid',msg.info,msg)

    @staticmethod
    def _read(msg):
        fields = {field.name for field in msg.fields}
        wanted = [name for name in ("x", "y", "z") if name in fields]
        rows = list(
            point_cloud2.read_points(msg, field_names=wanted, skip_nans=True)
        )
        return fields, [tuple(float(value) for value in row) for row in rows]

    def _elevation_cb(self, msg):
        self._remember("elevation", msg, self._read(msg))

    def _risk_cb(self, msg):
        self._remember("risk", msg, self._read(msg))

    def _surface_cb(self, msg):
        if msg.ns == "terrain_surface" and msg.action == Marker.ADD:
            self._remember("surface", msg, msg)

    def _height_surface_cb(self,msg):
        if msg.ns=="measured_elevation" and msg.action==Marker.ADD:
            self._remember("height_surface",msg,msg)

    def _remember(self, name, msg, value):
        stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        self.pending[name][stamp] = value
        while len(self.pending[name]) > 5:
            del self.pending[name][min(self.pending[name])]
        common = set.intersection(*(set(cache) for cache in self.pending.values()))
        if common:
            self.sample_stamp = max(common)
            self.elevation_fields, self.elevation = self.pending["elevation"][self.sample_stamp]
            self.risk_fields, self.risk = self.pending["risk"][self.sample_stamp]
            self.surface = self.pending["surface"][self.sample_stamp]
            self.height_surface = self.pending["height_surface"][self.sample_stamp]
            self.grid = self.pending['grid'][self.sample_stamp]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()
    rclpy.init()
    node = VisualizationProbe()
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline and (
        node.elevation is None or node.risk is None or node.surface is None or node.height_surface is None or len(node.lidar_matches)<2 or not node.density_matches
    ):
        rclpy.spin_once(node, timeout_sec=0.25)
    try:
        if node.elevation is None or node.risk is None or node.surface is None or node.height_surface is None:
            raise RuntimeError("no elevation/risk/surface products with a common timestamp arrived")
        if len(node.lidar_matches)<2 or any(v['points']<100 or v['colors']<3 for v in node.lidar_matches.values()):
            stamps={name:sorted(cache)[-3:] for name,cache in node.lidar_pending.items()}
            raise RuntimeError(f'missing/invalid actual colored LiDAR products: {node.lidar_matches}; '
                               f'received={node.lidar_received}; recent_stamps={stamps}')
        if not node.density_matches or any(dense<nav for nav,dense in node.density_matches):
            raise RuntimeError(f'full measured display scan is missing/sparser than navigation: {node.density_matches}')
        for name, fields, points in (
            ("elevation", node.elevation_fields, node.elevation),
            ("risk", node.risk_fields, node.risk),
        ):
            if not {"x", "y", "z"}.issubset(fields):
                raise RuntimeError(f"{name} cloud is missing x/y/z fields: {sorted(fields)}")
            if not all(math.isfinite(value) for point in points for value in point):
                raise RuntimeError(f"{name} cloud contains non-finite points")
            require_cloud_fidelity(points,cloud_evidence(node.grid,name),name)

        # Masks differ: a height observation need not qualify a whole pose.
        # On common cells both converters use raw elevation; offsets are
        # +0.035m and +0.020m, not two independent terrain reconstructions.
        elevation_by_xy = {
            (round(point[0], 2), round(point[1], 2)): point[2]
            for point in node.elevation
        }
        deltas = []
        for x, y, z in node.risk:
            base = elevation_by_xy.get((round(x, 2), round(y, 2)))
            if base is not None:
                deltas.append(z - base)
        # The GP posterior may refine measured height. Check each product
        # against its OWN published source above, not a fixed raw/GP difference.
        median_delta = statistics.median(deltas) if deltas else math.nan
        surface = node.surface
        expected_surface,prediction_positions=surface_evidence(node.grid)
        if surface.type != Marker.TRIANGLE_LIST or (not surface.points and
                smooth_quad_exists(expected_surface,node.grid.info.pos_resolution)):
            raise RuntimeError(
                f"terrain surface is not a useful triangle mesh: type={surface.type} "
                f"points={len(surface.points)}"
            )
        if len(surface.points) % 3 != 0:
            raise RuntimeError("terrain surface point count is not divisible by three")
        if len(surface.colors) != len(surface.points):
            raise RuntimeError(
                f"terrain surface colors do not match vertices: "
                f"colors={len(surface.colors)} points={len(surface.points)}"
            )
        surface_xyz = [
            (float(point.x), float(point.y), float(point.z)) for point in surface.points
        ]
        if not all(math.isfinite(value) for point in surface_xyz for value in point):
            raise RuntimeError("terrain surface contains non-finite vertices")
        if not all(
            math.isfinite(value) and 0.0 <= value <= 1.0
            for color in surface.colors for value in (color.r, color.g, color.b, color.a)
        ):
            raise RuntimeError("terrain surface contains invalid vertex colours")
        displayed_predictions=set()
        for point in surface_xyz:
            key=(round(point[0],2),round(point[1],2))
            z = expected_surface.get(key)
            if z is None or abs(z - point[2]) > 1e-4:
                raise RuntimeError("risk surface vertex is not on observed/navigation-qualified elevation")
            if key in prediction_positions:displayed_predictions.add(key)
        maximum_edge = 0.0
        for offset in range(0, len(surface_xyz), 3):
            triangle = surface_xyz[offset:offset + 3]
            heights = [point[2] for point in triangle]
            if max(heights) - min(heights) > 0.2501:
                raise RuntimeError("terrain surface bridges a rejected height discontinuity")
            for i in range(3):
                a, b = triangle[i], triangle[(i + 1) % 3]
                maximum_edge = max(maximum_edge, math.hypot(a[0] - b[0], a[1] - b[1]))
        # The Hunter profile uses 0.2m cells. A triangle must not connect
        # opposite sides of the rolling circular buffer.
        if maximum_edge > math.sqrt(2.0) * node.grid.info.pos_resolution + 1e-4:
            raise RuntimeError(f"terrain surface has a nonlocal edge: {maximum_edge:.3f}m")
        z_span = (max(point[2] for point in surface_xyz) - min(point[2] for point in surface_xyz)) if surface_xyz else 0.
        # A physically flat course must stay flat. The exact per-vertex source
        # comparison above rejects flattening nonflat evidence without forcing
        # invented height variation into flat-world qualification scenes.
        height_mesh=node.height_surface
        expected_height,_=surface_evidence(node.grid,measured_only=True)
        if (height_mesh.type!=Marker.TRIANGLE_LIST or len(height_mesh.points)%3 or len(height_mesh.points)!=len(height_mesh.colors) or
                (not height_mesh.points and smooth_quad_exists(expected_height,node.grid.info.pos_resolution))):
            raise RuntimeError("missing measured elevation mesh")
        for point in height_mesh.points:
            expected=expected_height.get((round(point.x,2),round(point.y,2)))
            if expected is None or abs(expected-point.z)>1e-4:
                raise RuntimeError("measured height mesh does not follow observed elevation")
        print(
            f"PASS elevation={len(node.elevation)} risk={len(node.risk)} "
            f"common={len(deltas)} drape_delta={median_delta:.3f}m "
            f"surface_triangles={len(surface.points) // 3} surface_z_span={z_span:.2f}m "
            f"max_edge={maximum_edge:.3f}m stamp={node.sample_stamp}",
            flush=True,
        )
        print(f"PASS independent elevation mesh triangles={len(height_mesh.points)//3}, synchronized, no height fabrication")
        print(f'PASS risk drape includes {len(displayed_predictions)} qualified GP vertices; candidates/unknown are not fabricated')
        print(f'PASS actual LiDAR XYZ/count/frame/stamp unchanged; RGB products={node.lidar_matches}')
        print(f'PASS same-stamp navigation/display counts={node.density_matches[-1]}; no display downsampling applied to navigation')
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
