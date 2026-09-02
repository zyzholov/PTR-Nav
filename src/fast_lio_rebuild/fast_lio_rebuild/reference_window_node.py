#!/usr/bin/env python3
from __future__ import annotations

import bisect
from copy import deepcopy
import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import rclpy
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


@dataclass
class StampedPose:
    stamp_s: float
    pose: PoseStamped


def stamp_to_float(msg) -> float:
    return float(msg.sec) + float(msg.nanosec) * 1e-9


def pose_to_point(pose: PoseStamped) -> Point:
    p = pose.pose.position
    return Point(x=float(p.x), y=float(p.y), z=float(p.z))


def make_point(x: float, y: float, z: float) -> Point:
    return Point(x=float(x), y=float(y), z=float(z))


def lerp(a: float, b: float, t: float) -> float:
    return float(a + (b - a) * t)


def lerp_point(a: Point, b: Point, t: float) -> Point:
    return make_point(lerp(a.x, b.x, t), lerp(a.y, b.y, t), lerp(a.z, b.z, t))


def lerp_color(a: ColorRGBA, b: ColorRGBA, t: float) -> ColorRGBA:
    out = ColorRGBA()
    out.r = lerp(a.r, b.r, t)
    out.g = lerp(a.g, b.g, t)
    out.b = lerp(a.b, b.b, t)
    out.a = lerp(a.a, b.a, t)
    return out


def make_color(r: float, g: float, b: float, a: float) -> ColorRGBA:
    color = ColorRGBA()
    color.r = float(r)
    color.g = float(g)
    color.b = float(b)
    color.a = float(a)
    return color


def distance_xy(a: PoseStamped, b: PoseStamped) -> float:
    ax, ay = a.pose.position.x, a.pose.position.y
    bx, by = b.pose.position.x, b.pose.position.y
    return math.hypot(ax - bx, ay - by)


class ReferenceWindowNode(Node):
    """Publish a short reference path from current rebuilt odom to ahead reference odom."""

    def __init__(self) -> None:
        super().__init__("reference_window_node")
        self.declare_parameter("new_odom_topic", "/fastlio_rebuild/Odometry")
        self.declare_parameter("reference_odom_topic", "/fastlio_reference/Odometry")
        self.declare_parameter("path_topic", "/fastlio_rebuild/reference_window_path")
        self.declare_parameter("marker_topic", "/fastlio_rebuild/reference_window_marker")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("reference_lead_time", 0.45)
        self.declare_parameter("window_duration", 3.0)
        self.declare_parameter("max_points", 80)
        self.declare_parameter("min_point_spacing", 0.08)
        self.declare_parameter("buffer_duration", 12.0)
        self.declare_parameter("line_width", 0.18)
        self.declare_parameter("point_scale", 0.24)
        self.declare_parameter("z_offset", 0.10)
        self.declare_parameter("pre_simplify_tolerance", 0.08)
        self.declare_parameter("smooth_iterations", 5)
        self.declare_parameter("smooth_spacing", 0.04)
        self.declare_parameter("min_smooth_points", 108)
        self.declare_parameter("max_smooth_points", 340)
        self.declare_parameter("post_smooth_passes", 6)
        self.declare_parameter("post_smooth_strength", 0.62)
        self.declare_parameter("publish_empty_delete", True)

        self.new_odom_topic = self._param_str("new_odom_topic")
        self.reference_odom_topic = self._param_str("reference_odom_topic")
        self.path_topic = self._param_str("path_topic")
        self.marker_topic = self._param_str("marker_topic")
        self.map_frame = self._param_str("map_frame")
        self.reference_lead_time = self._param_float("reference_lead_time")
        self.window_duration = self._param_float("window_duration")
        self.max_points = max(2, self._param_int("max_points"))
        self.min_point_spacing = max(0.0, self._param_float("min_point_spacing"))
        self.buffer_duration = max(1.0, self._param_float("buffer_duration"))
        self.line_width = max(0.01, self._param_float("line_width"))
        self.point_scale = max(0.01, self._param_float("point_scale"))
        self.z_offset = self._param_float("z_offset")
        self.pre_simplify_tolerance = max(0.0, self._param_float("pre_simplify_tolerance"))
        self.smooth_iterations = max(0, self._param_int("smooth_iterations"))
        self.smooth_spacing = max(0.03, self._param_float("smooth_spacing"))
        self.min_smooth_points = max(2, self._param_int("min_smooth_points"))
        self.max_smooth_points = max(8, self._param_int("max_smooth_points"))
        self.post_smooth_passes = max(0, self._param_int("post_smooth_passes"))
        self.post_smooth_strength = min(0.85, max(0.0, self._param_float("post_smooth_strength")))
        self.publish_empty_delete = self._param_bool("publish_empty_delete")

        self.reference_times: List[float] = []
        self.reference_poses: List[PoseStamped] = []
        self.last_warn_s = 0.0
        self.last_reference_status_s = 0.0
        self.last_publish_status_s = 0.0
        self.path_publish_count = 0

        self.path_pub = self.create_publisher(Path, self.path_topic, 10)
        self.marker_pub = self.create_publisher(MarkerArray, self.marker_topic, 10)
        self.create_subscription(Odometry, self.reference_odom_topic, self._reference_callback, 100)
        self.create_subscription(Odometry, self.new_odom_topic, self._new_odom_callback, 50)

        self.get_logger().info(
            "Reference window: new=%s reference=%s path=%s marker=%s lead=%.2fs window=%.2fs"
            % (
                self.new_odom_topic,
                self.reference_odom_topic,
                self.path_topic,
                self.marker_topic,
                self.reference_lead_time,
                self.window_duration,
            )
        )

    def _param_str(self, name: str) -> str:
        return str(self.get_parameter(name).value)

    def _param_float(self, name: str) -> float:
        return float(self.get_parameter(name).value)

    def _param_int(self, name: str) -> int:
        return int(self.get_parameter(name).value)

    def _param_bool(self, name: str) -> bool:
        return bool(self.get_parameter(name).value)

    def _reference_callback(self, msg: Odometry) -> None:
        stamp_s = stamp_to_float(msg.header.stamp)
        pose = self._odom_to_pose(msg)
        idx = bisect.bisect_right(self.reference_times, stamp_s)
        self.reference_times.insert(idx, stamp_s)
        self.reference_poses.insert(idx, pose)
        self._prune_reference(stamp_s - self.buffer_duration)
        now_s = self.get_clock().now().nanoseconds * 1e-9
        if now_s - self.last_reference_status_s > 2.0:
            self.last_reference_status_s = now_s
            self.get_logger().info(
                "Buffered reference odom: %d poses, latest stamp %.3f"
                % (len(self.reference_poses), stamp_s)
            )

    def _new_odom_callback(self, msg: Odometry) -> None:
        stamp_s = stamp_to_float(msg.header.stamp)
        current = self._odom_to_pose(msg)
        selected = self._select_reference_window(stamp_s)
        if not selected:
            self._maybe_warn_no_reference(stamp_s)
            if self.publish_empty_delete:
                self._publish_delete_markers(msg)
            return

        poses = [self._pose_with_z_offset(current)]
        for pose in selected:
            if len(poses) >= self.max_points:
                break
            if self.min_point_spacing > 0.0 and distance_xy(poses[-1], pose) < self.min_point_spacing:
                continue
            poses.append(self._pose_with_z_offset(pose))

        if len(poses) < 2:
            return

        poses = self._smooth_pose_window(poses)

        path = Path()
        path.header.stamp = msg.header.stamp
        path.header.frame_id = self.map_frame or current.header.frame_id
        path.poses = poses
        self.path_pub.publish(path)
        self.marker_pub.publish(self._make_markers(path))
        self.path_publish_count += 1
        now_s = self.get_clock().now().nanoseconds * 1e-9
        if now_s - self.last_publish_status_s > 2.0:
            self.last_publish_status_s = now_s
            self.get_logger().info(
                "Published reference guide path #%d with %d poses from new stamp %.3f"
                % (self.path_publish_count, len(path.poses), stamp_s)
            )

    def _select_reference_window(self, new_stamp_s: float) -> List[PoseStamped]:
        if not self.reference_times:
            return []
        start_s = new_stamp_s + self.reference_lead_time
        end_s = start_s + self.window_duration
        start_idx = bisect.bisect_left(self.reference_times, start_s)
        end_idx = bisect.bisect_right(self.reference_times, end_s)
        if start_idx < end_idx:
            return self.reference_poses[start_idx:end_idx]

        # If playback timing is not ahead enough, still use the closest future
        # poses when available. This makes the visualization degrade gracefully.
        fallback_idx = bisect.bisect_left(self.reference_times, new_stamp_s)
        if fallback_idx < len(self.reference_poses):
            return self.reference_poses[fallback_idx : min(len(self.reference_poses), fallback_idx + self.max_points)]
        return []

    def _prune_reference(self, before_stamp_s: float) -> None:
        idx = bisect.bisect_left(self.reference_times, before_stamp_s)
        if idx <= 0:
            return
        del self.reference_times[:idx]
        del self.reference_poses[:idx]

    def _odom_to_pose(self, msg: Odometry) -> PoseStamped:
        pose = PoseStamped()
        pose.header = msg.header
        if not pose.header.frame_id:
            pose.header.frame_id = self.map_frame
        pose.pose = msg.pose.pose
        return pose

    def _pose_with_z_offset(self, pose: PoseStamped) -> PoseStamped:
        out = PoseStamped()
        out.header = deepcopy(pose.header)
        out.pose = deepcopy(pose.pose)
        out.pose.position.z = float(out.pose.position.z) + self.z_offset
        return out

    def _smooth_pose_window(self, poses: List[PoseStamped]) -> List[PoseStamped]:
        points = [pose_to_point(pose) for pose in poses]
        simplified = self._rdp_simplify(points, self.pre_simplify_tolerance)
        if len(simplified) >= 2:
            points = simplified
        points = self._chaikin_smooth(points, self.smooth_iterations)
        points = self._resample_points(
            points, self.smooth_spacing, self.max_smooth_points, self.min_smooth_points
        )
        points = self._laplacian_smooth(points, self.post_smooth_passes, self.post_smooth_strength)
        points = self._resample_points(
            points, self.smooth_spacing, self.max_smooth_points, self.min_smooth_points
        )
        if len(points) < 2:
            return poses
        smooth_poses: List[PoseStamped] = []
        for idx, point in enumerate(points):
            pose = PoseStamped()
            pose.header = deepcopy(poses[min(idx, len(poses) - 1)].header)
            pose.pose.position = deepcopy(point)
            yaw = self._point_yaw(points, idx)
            pose.pose.orientation.z = math.sin(yaw * 0.5)
            pose.pose.orientation.w = math.cos(yaw * 0.5)
            smooth_poses.append(pose)
        return smooth_poses

    @staticmethod
    def _chaikin_smooth(points: List[Point], iterations: int) -> List[Point]:
        if len(points) < 3 or iterations <= 0:
            return points
        smoothed = points
        for _ in range(iterations):
            next_points = [smoothed[0]]
            for p0, p1 in zip(smoothed[:-1], smoothed[1:]):
                next_points.append(lerp_point(p0, p1, 0.25))
                next_points.append(lerp_point(p0, p1, 0.75))
            next_points.append(smoothed[-1])
            smoothed = next_points
        return smoothed

    @staticmethod
    def _rdp_simplify(points: List[Point], tolerance: float) -> List[Point]:
        if len(points) < 4 or tolerance <= 0.0:
            return points

        keep = [False] * len(points)
        keep[0] = True
        keep[-1] = True

        def perpendicular_distance(p: Point, a: Point, b: Point) -> float:
            abx = b.x - a.x
            aby = b.y - a.y
            denom = abx * abx + aby * aby
            if denom <= 1e-9:
                return math.hypot(p.x - a.x, p.y - a.y)
            t = max(0.0, min(1.0, ((p.x - a.x) * abx + (p.y - a.y) * aby) / denom))
            proj_x = a.x + t * abx
            proj_y = a.y + t * aby
            return math.hypot(p.x - proj_x, p.y - proj_y)

        def simplify_range(start: int, end: int) -> None:
            if end <= start + 1:
                return
            max_dist = -1.0
            max_idx = start
            for idx in range(start + 1, end):
                dist = perpendicular_distance(points[idx], points[start], points[end])
                if dist > max_dist:
                    max_dist = dist
                    max_idx = idx
            if max_dist > tolerance:
                keep[max_idx] = True
                simplify_range(start, max_idx)
                simplify_range(max_idx, end)

        simplify_range(0, len(points) - 1)
        return [point for point, should_keep in zip(points, keep) if should_keep]

    @staticmethod
    def _resample_points(
        points: List[Point], spacing: float, max_points: int, min_points: int = 2
    ) -> List[Point]:
        if len(points) < 2:
            return points
        distances = [0.0]
        for p0, p1 in zip(points[:-1], points[1:]):
            distances.append(distances[-1] + math.hypot(p1.x - p0.x, p1.y - p0.y))
        total = distances[-1]
        if total <= 1e-6:
            return [points[0], points[-1]]
        sample_count = min(max_points, max(2, min_points, int(total / max(1e-3, spacing)) + 1))
        out: List[Point] = []
        seg_idx = 0
        for i in range(sample_count):
            target = total * i / max(1, sample_count - 1)
            while seg_idx + 1 < len(distances) and distances[seg_idx + 1] < target:
                seg_idx += 1
            if seg_idx + 1 >= len(points):
                out.append(points[-1])
                continue
            span = max(1e-6, distances[seg_idx + 1] - distances[seg_idx])
            t = (target - distances[seg_idx]) / span
            out.append(lerp_point(points[seg_idx], points[seg_idx + 1], t))
        return out

    @staticmethod
    def _laplacian_smooth(points: List[Point], passes: int, strength: float) -> List[Point]:
        if len(points) < 4 or passes <= 0 or strength <= 0.0:
            return points
        smoothed = points
        keep_radius = max(2, min(8, len(points) // 8))
        for _ in range(passes):
            next_points = [smoothed[0]]
            for idx in range(1, len(smoothed) - 1):
                prev_p = smoothed[idx - 1]
                curr_p = smoothed[idx]
                next_p = smoothed[idx + 1]
                endpoint_t = min(idx, len(smoothed) - 1 - idx) / float(keep_radius)
                local_strength = strength * min(1.0, max(0.0, endpoint_t))
                avg = make_point(
                    (prev_p.x + next_p.x) * 0.5,
                    (prev_p.y + next_p.y) * 0.5,
                    (prev_p.z + next_p.z) * 0.5,
                )
                next_points.append(lerp_point(curr_p, avg, local_strength))
            next_points.append(smoothed[-1])
            smoothed = next_points
        return smoothed

    @staticmethod
    def _point_yaw(points: List[Point], idx: int) -> float:
        if len(points) < 2:
            return 0.0
        if idx == 0:
            p0, p1 = points[0], points[1]
        elif idx >= len(points) - 1:
            p0, p1 = points[-2], points[-1]
        else:
            p0, p1 = points[idx - 1], points[idx + 1]
        return math.atan2(p1.y - p0.y, p1.x - p0.x)

    def _make_markers(self, path: Path) -> MarkerArray:
        points = [pose_to_point(pose) for pose in path.poses]
        clear = Marker()
        clear.header = path.header
        clear.ns = "__reference_window_clear__"
        clear.id = 0
        clear.action = Marker.DELETEALL
        clear.pose.orientation.w = 1.0

        glow = Marker()
        glow.header = path.header
        glow.ns = "reference_window_glow"
        glow.id = 0
        glow.type = Marker.LINE_STRIP
        glow.action = Marker.ADD
        glow.pose.orientation.w = 1.0
        glow.scale.x = self.line_width * 3.2
        glow.color = make_color(1.0, 0.22, 0.52, 0.30)
        glow.points = points

        outer = Marker()
        outer.header = path.header
        outer.ns = "reference_window_outer"
        outer.id = 1
        outer.type = Marker.LINE_STRIP
        outer.action = Marker.ADD
        outer.pose.orientation.w = 1.0
        outer.scale.x = self.line_width * 2.1
        outer.color = make_color(1.0, 1.0, 1.0, 1.0)
        outer.points = points
        outer.colors = self._outer_gradient_colors(len(points), 0.84)

        inner = Marker()
        inner.header = path.header
        inner.ns = "reference_window_inner"
        inner.id = 2
        inner.type = Marker.LINE_STRIP
        inner.action = Marker.ADD
        inner.pose.orientation.w = 1.0
        inner.scale.x = self.line_width * 1.0
        inner.color = make_color(1.0, 1.0, 1.0, 1.0)
        inner.points = points
        inner.colors = self._inner_gradient_colors(len(points), 0.96)

        shine = Marker()
        shine.header = path.header
        shine.ns = "reference_window_shine"
        shine.id = 5
        shine.type = Marker.LINE_STRIP
        shine.action = Marker.ADD
        shine.pose.orientation.w = 1.0
        shine.scale.x = self.line_width * 0.26
        shine.color = make_color(1.0, 0.93, 1.0, 0.48)
        shine.points = points
        return MarkerArray(markers=[clear, glow, outer, inner, shine])

    @staticmethod
    def _outer_gradient_colors(count: int, alpha: float) -> List[ColorRGBA]:
        near = make_color(0.78, 0.52, 1.00, alpha)
        mid = make_color(1.00, 0.24, 0.66, alpha)
        far = make_color(1.00, 0.34, 0.28, alpha)
        return ReferenceWindowNode._three_stop_gradient(count, near, mid, far)

    @staticmethod
    def _inner_gradient_colors(count: int, alpha: float) -> List[ColorRGBA]:
        near = make_color(1.00, 0.86, 1.00, alpha)
        mid = make_color(1.00, 0.30, 0.70, alpha)
        far = make_color(1.00, 0.12, 0.34, alpha)
        return ReferenceWindowNode._three_stop_gradient(count, near, mid, far)

    @staticmethod
    def _three_stop_gradient(
        count: int, near: ColorRGBA, mid: ColorRGBA, far: ColorRGBA
    ) -> List[ColorRGBA]:
        colors: List[ColorRGBA] = []
        for idx in range(max(0, count)):
            t = idx / max(1, count - 1)
            if t < 0.58:
                colors.append(lerp_color(near, mid, t / 0.58))
            else:
                colors.append(lerp_color(mid, far, (t - 0.58) / 0.42))
        return colors

    def _publish_delete_markers(self, msg: Odometry) -> None:
        marker = Marker()
        marker.header = msg.header
        marker.header.frame_id = self.map_frame or msg.header.frame_id
        marker.ns = "__reference_window_clear__"
        marker.id = 0
        marker.action = Marker.DELETEALL
        marker.pose.orientation.w = 1.0
        self.marker_pub.publish(MarkerArray(markers=[marker]))

    def _maybe_warn_no_reference(self, stamp_s: float) -> None:
        now_s = self.get_clock().now().nanoseconds * 1e-9
        if now_s - self.last_warn_s < 2.0:
            return
        self.last_warn_s = now_s
        self.get_logger().warn(
            "No ahead reference odometry buffered for new stamp %.3f. "
            "Start the reference odom bag a little earlier or reduce reference_lead_time." % stamp_s
        )


def main(args: Optional[Iterable[str]] = None) -> None:
    rclpy.init(args=args)
    node = ReferenceWindowNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
