#!/usr/bin/env python3
"""Validate synchronized cloud and terrain-mesh products used by RViz."""

import argparse
import math
import statistics
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import Marker


class VisualizationProbe(Node):
    def __init__(self):
        super().__init__("hunter_visualization_probe")
        self.elevation = None
        self.risk = None
        self.surface = None
        self.elevation_fields = set()
        self.risk_fields = set()
        self.pending = {name: {} for name in ("elevation", "risk", "surface")}
        self.sample_stamp = None
        self.create_subscription(
            PointCloud2, "/car_0_terrain_analyzer/elevation_cloud", self._elevation_cb, 1
        )
        self.create_subscription(
            PointCloud2, "/car_0_terrain_analyzer/risk_cloud", self._risk_cb, 1
        )
        self.create_subscription(
            Marker, "/car_0_terrain_analyzer/terrain_surface", self._surface_cb, 1
        )

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()
    rclpy.init()
    node = VisualizationProbe()
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline and (
        node.elevation is None or node.risk is None or node.surface is None
    ):
        rclpy.spin_once(node, timeout_sec=0.25)
    try:
        if node.elevation is None or node.risk is None or node.surface is None:
            raise RuntimeError("no elevation/risk/surface products with a common timestamp arrived")
        if len(node.elevation) < 100 or len(node.risk) < 100:
            raise RuntimeError(
                f"visualization clouds too sparse: elevation={len(node.elevation)} risk={len(node.risk)}"
            )
        for name, fields, points in (
            ("elevation", node.elevation_fields, node.elevation),
            ("risk", node.risk_fields, node.risk),
        ):
            if not {"x", "y", "z"}.issubset(fields):
                raise RuntimeError(f"{name} cloud is missing x/y/z fields: {sorted(fields)}")
            if not all(math.isfinite(value) for point in points for value in point):
                raise RuntimeError(f"{name} cloud contains non-finite points")
            extent = max(max(point[i] for point in points) - min(point[i] for point in points)
                         for i in (0, 1))
            if extent < 5.0:
                raise RuntimeError(f"{name} cloud extent is unexpectedly small: {extent:.2f}m")

        # Both converters use the same valid mask and bgk_mean. Risk is draped
        # at +0.04m while elevation is at +0.02m, so common cells should have
        # a stable 2cm separation instead of a floating risk sheet.
        elevation_by_xy = {
            (round(point[0], 2), round(point[1], 2)): point[2]
            for point in node.elevation
        }
        deltas = []
        for x, y, z in node.risk:
            base = elevation_by_xy.get((round(x, 2), round(y, 2)))
            if base is not None:
                deltas.append(z - base)
        if len(deltas) < 100:
            raise RuntimeError(f"too few common elevation/risk cells: {len(deltas)}")
        median_delta = statistics.median(deltas)
        if abs(median_delta - 0.02) > 0.006:
            raise RuntimeError(
                f"risk is not draped on elevation: median z delta={median_delta:.3f}m"
            )
        surface = node.surface
        if surface.type != Marker.TRIANGLE_LIST or len(surface.points) < 300:
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
        for point in surface_xyz:
            z = elevation_by_xy.get((round(point[0], 2), round(point[1], 2)))
            if z is None or abs(z - point[2]) > 1e-4:
                raise RuntimeError("terrain surface vertex is not on the synchronized elevation grid")
        maximum_edge = 0.0
        for offset in range(0, len(surface_xyz), 3):
            triangle = surface_xyz[offset:offset + 3]
            heights = [point[2] for point in triangle]
            if max(heights) - min(heights) > 0.6501:
                raise RuntimeError("terrain surface bridges a rejected height discontinuity")
            for i in range(3):
                a, b = triangle[i], triangle[(i + 1) % 3]
                maximum_edge = max(maximum_edge, math.hypot(a[0] - b[0], a[1] - b[1]))
        # The Hunter profile uses 0.2m cells. A triangle must not connect
        # opposite sides of the rolling circular buffer.
        if maximum_edge > math.sqrt(2.0) * 0.2 + 1e-4:
            raise RuntimeError(f"terrain surface has a nonlocal edge: {maximum_edge:.3f}m")
        xy_extent = max(
            max(point[i] for point in surface_xyz) - min(point[i] for point in surface_xyz)
            for i in (0, 1)
        )
        if xy_extent < 5.0:
            raise RuntimeError(f"terrain surface extent is unexpectedly small: {xy_extent:.2f}m")
        z_span = max(point[2] for point in surface_xyz) - min(point[2] for point in surface_xyz)
        if z_span < 0.05:
            raise RuntimeError("terrain surface is flat; elevation geometry was not rendered")
        print(
            f"PASS elevation={len(node.elevation)} risk={len(node.risk)} "
            f"common={len(deltas)} drape_delta={median_delta:.3f}m "
            f"surface_triangles={len(surface.points) // 3} surface_z_span={z_span:.2f}m "
            f"max_edge={maximum_edge:.3f}m stamp={node.sample_stamp}",
            flush=True,
        )
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
