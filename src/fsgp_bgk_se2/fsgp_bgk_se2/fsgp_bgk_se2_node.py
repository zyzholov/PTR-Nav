from __future__ import annotations

import copy
import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from se2_grid_msgs.msg import SE2Grid
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Float32MultiArray, Header

from .bgk_seb import BgkSebEstimator, BgkSebParams, BgkSebResult
from .obstacle_columns import preserve_collision_risk, vertical_columns
from .se2_grid_tools import (
    first_data_template,
    get_layer_2d,
    get_layer_slices,
    grid_buffer_to_world,
    layer_refs,
    matrix_shape,
    matrix_to_multiarray_like,
    multiarray_to_matrix,
    repeat_or_trim_slices,
    yaw_slices,
    world_to_grid_buffer,
)


SE2_GRID_LENGTH_YAW = 2.0 * math.pi + 0.02


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


class FsgpBgkSe2Node(Node):
    def __init__(self) -> None:
        super().__init__("fsgp_bgk_se2_node")
        self._declare_parameters()
        self._load_parameters()

        self.estimator = BgkSebEstimator(self.params)
        self.latest_odom: Optional[Odometry] = None
        self.latest_cloud: Optional[PointCloud2] = None
        self.last_backend_warn = ""
        self._last_process_wall = 0.0
        self._last_skip_log_wall = 0.0
        self._skipped_maps = 0

        self.map_sub = self.create_subscription(
            SE2Grid,
            self.input_fused_map_topic,
            self._map_callback,
            QoSProfile(depth=1),
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self._odom_callback,
            QoSProfile(depth=5),
        )
        if self.subscribe_cloud:
            self.cloud_sub = self.create_subscription(
                PointCloud2,
                self.input_cloud_topic,
                self._cloud_callback,
                QoSProfile(depth=1, reliability=qos_profile_sensor_data.reliability),
            )
        else:
            self.cloud_sub = None

        self.map_pub = self.create_publisher(
            SE2Grid,
            self.output_fused_map_topic,
            QoSProfile(depth=1),
        )
        self.bgk_cloud_pub = self.create_publisher(
            PointCloud2,
            self.bgk_cloud_topic,
            QoSProfile(depth=1),
        )
        self.seb_cloud_pub = self.create_publisher(
            PointCloud2,
            self.seb_cloud_topic,
            QoSProfile(depth=1),
        )

        self.get_logger().info(
            f"fsgp_bgk_se2 ready: input={self.input_fused_map_topic} "
            f"output={self.output_fused_map_topic} backend={self.estimator.backend_name} "
            f"gpu_requested={self.params.prefer_gpu} max_update={self.max_update_rate_hz:.2f}Hz "
            f"diagnostic_layers={self.include_diagnostic_layers} "
            f"reason={self.estimator.backend_reason}"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("input_fused_map_topic", "/car_0_terrain_analyzer/fused_map")
        self.declare_parameter("input_cloud_topic", "/cloud_registered")
        self.declare_parameter("odom_topic", "/Odometry")
        self.declare_parameter("output_fused_map_topic", "/car_0_fsgp_bgk/fused_map")
        self.declare_parameter("bgk_cloud_topic", "/car_0_fsgp_bgk/bgk_traversability_cloud")
        self.declare_parameter("seb_cloud_topic", "/car_0_fsgp_bgk/seb_risk_cloud")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("subscribe_cloud", False)

        self.declare_parameter("use_gpu", True)
        self.declare_parameter("gpu_backend", "auto")
        self.declare_parameter("auto_gpu_min_work_items", 100_000)
        self.declare_parameter("yaw_bins", 0)
        self.declare_parameter("bgk_sigma_m", 0.45)
        self.declare_parameter("bgk_radius_m", 1.20)
        self.declare_parameter("variance_free", 0.01)
        self.declare_parameter("variance_hard", 0.20)
        self.declare_parameter("slope_free_rad", 0.10)
        self.declare_parameter("max_slope_rad", 0.55)
        self.declare_parameter("step_free_m", 0.04)
        self.declare_parameter("max_step_m", 0.18)
        self.declare_parameter("max_pitch_rad", 0.55)
        self.declare_parameter("max_roll_rad", 0.45)
        self.declare_parameter("pitch_free_rad", 0.08)
        self.declare_parameter("roll_free_rad", 0.08)
        self.declare_parameter("min_support_ratio", 0.70)
        self.declare_parameter("footprint_front", 0.42)
        self.declare_parameter("footprint_back", 0.42)
        self.declare_parameter("footprint_half_width", 0.313)
        self.declare_parameter("footprint_margin", 0.10)
        self.declare_parameter("footprint_samples_x", 5)
        self.declare_parameter("footprint_samples_y", 5)
        self.declare_parameter("risk_fusion_mode", "weighted")
        self.declare_parameter("baseline_weight", 0.45)
        self.declare_parameter("seb_weight", 0.45)
        self.declare_parameter("uncertainty_weight", 0.10)
        self.declare_parameter("unknown_risk", 0.65)
        self.declare_parameter("include_diagnostic_layers", False)
        self.declare_parameter("publish_debug_clouds", True)
        self.declare_parameter("debug_yaw_index", 0)
        self.declare_parameter("debug_cloud_stride", 2)
        self.declare_parameter("max_update_rate_hz", 8.0)
        self.declare_parameter("log_period_sec", 2.0)

    def _load_parameters(self) -> None:
        self.input_fused_map_topic = self.get_parameter("input_fused_map_topic").value
        self.input_cloud_topic = self.get_parameter("input_cloud_topic").value
        self.odom_topic = self.get_parameter("odom_topic").value
        self.output_fused_map_topic = self.get_parameter("output_fused_map_topic").value
        self.bgk_cloud_topic = self.get_parameter("bgk_cloud_topic").value
        self.seb_cloud_topic = self.get_parameter("seb_cloud_topic").value
        self.frame_id = self.get_parameter("frame_id").value
        self.subscribe_cloud = _as_bool(self.get_parameter("subscribe_cloud").value)
        self.include_diagnostic_layers = _as_bool(
            self.get_parameter("include_diagnostic_layers").value
        )
        self.publish_debug_clouds = _as_bool(self.get_parameter("publish_debug_clouds").value)
        self.debug_yaw_index = int(self.get_parameter("debug_yaw_index").value)
        self.debug_cloud_stride = max(1, int(self.get_parameter("debug_cloud_stride").value))
        self.max_update_rate_hz = max(0.0, float(self.get_parameter("max_update_rate_hz").value))
        self.min_update_period_sec = (
            1.0 / self.max_update_rate_hz if self.max_update_rate_hz > 0.0 else 0.0
        )
        self.log_period_sec = max(0.2, float(self.get_parameter("log_period_sec").value))

        self.configured_yaw_bins = int(self.get_parameter("yaw_bins").value)
        self.params = BgkSebParams(
            resolution=0.20,
            yaw_bins=max(1, self.configured_yaw_bins or 72),
            prefer_gpu=_as_bool(self.get_parameter("use_gpu").value),
            gpu_backend=str(self.get_parameter("gpu_backend").value),
            auto_gpu_min_work_items=int(self.get_parameter("auto_gpu_min_work_items").value),
            bgk_sigma_m=float(self.get_parameter("bgk_sigma_m").value),
            bgk_radius_m=float(self.get_parameter("bgk_radius_m").value),
            variance_free=float(self.get_parameter("variance_free").value),
            variance_hard=float(self.get_parameter("variance_hard").value),
            slope_free_rad=float(self.get_parameter("slope_free_rad").value),
            max_slope_rad=float(self.get_parameter("max_slope_rad").value),
            step_free_m=float(self.get_parameter("step_free_m").value),
            max_step_m=float(self.get_parameter("max_step_m").value),
            max_pitch_rad=float(self.get_parameter("max_pitch_rad").value),
            max_roll_rad=float(self.get_parameter("max_roll_rad").value),
            pitch_free_rad=float(self.get_parameter("pitch_free_rad").value),
            roll_free_rad=float(self.get_parameter("roll_free_rad").value),
            min_support_ratio=float(self.get_parameter("min_support_ratio").value),
            footprint_front=float(self.get_parameter("footprint_front").value),
            footprint_back=float(self.get_parameter("footprint_back").value),
            footprint_half_width=float(self.get_parameter("footprint_half_width").value),
            footprint_margin=float(self.get_parameter("footprint_margin").value),
            footprint_samples_x=int(self.get_parameter("footprint_samples_x").value),
            footprint_samples_y=int(self.get_parameter("footprint_samples_y").value),
            fusion_mode=str(self.get_parameter("risk_fusion_mode").value),
            baseline_weight=float(self.get_parameter("baseline_weight").value),
            seb_weight=float(self.get_parameter("seb_weight").value),
            uncertainty_weight=float(self.get_parameter("uncertainty_weight").value),
            unknown_risk=float(self.get_parameter("unknown_risk").value),
            store_diagnostic_slices=self.include_diagnostic_layers or self.publish_debug_clouds,
        )

    def _odom_callback(self, msg: Odometry) -> None:
        self.latest_odom = msg

    def _cloud_callback(self, msg: PointCloud2) -> None:
        self.latest_cloud = msg

    def _map_callback(self, msg: SE2Grid) -> None:
        now = time.monotonic()
        if (
            self.min_update_period_sec > 0.0
            and self._last_process_wall > 0.0
            and now - self._last_process_wall < self.min_update_period_sec
        ):
            self._skipped_maps += 1
            if now - self._last_skip_log_wall >= self.log_period_sec:
                self.get_logger().info(
                    f"FSGP-BGK/SEB throttled {self._skipped_maps} input maps; "
                    f"max_update={self.max_update_rate_hz:.2f}Hz",
                )
                self._skipped_maps = 0
                self._last_skip_log_wall = now
            return

        self._last_process_wall = now
        try:
            start = time.perf_counter()
            result, out_msg = self._process_map(msg)
            self.map_pub.publish(out_msg)
            backend_state = f"{result.backend}: {result.backend_reason}"
            if backend_state != self.last_backend_warn:
                self.get_logger().info(f"FSGP-BGK compute backend: {backend_state}")
                self.last_backend_warn = backend_state
            if self.publish_debug_clouds:
                self._publish_debug_clouds(msg, result)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.get_logger().info(
                f"FSGP-BGK/SEB map: {result.bgk_mean.shape[1]}x{result.bgk_mean.shape[0]} "
                f"yaw={len(result.final_risk_slices)} backend={result.backend} {elapsed_ms:.1f}ms",
                throttle_duration_sec=self.log_period_sec,
            )
        except Exception as exc:
            self.get_logger().warn(
                f"FSGP-BGK/SEB update failed: {exc}",
                throttle_duration_sec=1.0,
            )

    def _process_map(self, msg: SE2Grid) -> Tuple[BgkSebResult, SE2Grid]:
        template = first_data_template(msg)
        if template is None:
            raise ValueError("input SE2Grid has no data layers")

        shape = matrix_shape(template)
        elevation, elevation_template = get_layer_2d(msg, "elevation", 0.0, shape)
        variance, _ = get_layer_2d(msg, "var", 0.0, elevation.shape)
        refs = layer_refs(msg)
        if "valid" in refs:
            valid, _ = get_layer_2d(msg, "valid", 0.0, elevation.shape)
            valid = np.nan_to_num(valid, nan=0.0) >= 0.5
        else:
            # The native CUDA terrain analyzer encodes unobserved cells as NaN
            # in elevation and intentionally omits a separate validity layer.
            valid = np.isfinite(elevation)

        risk_slices, _, risk_template = get_layer_slices(msg, "risk")
        if not risk_slices:
            risk_slices = [np.zeros_like(elevation, dtype=np.float32)]
        yaws = self.configured_yaw_bins or yaw_slices(msg)
        yaws = max(1, int(yaws))

        elevation = grid_buffer_to_world(elevation, msg)
        variance = grid_buffer_to_world(variance, msg)
        valid = grid_buffer_to_world(valid, msg)
        risk_slices = [grid_buffer_to_world(layer, msg) for layer in risk_slices]
        self.params.resolution = max(float(msg.info.pos_resolution), 1.0e-3)
        self.params.yaw_bins = yaws
        baseline_risk_slices = repeat_or_trim_slices(
            risk_slices,
            yaws,
            np.zeros_like(elevation, dtype=np.float32),
        )

        result = self.estimator.compute(elevation, variance, valid, baseline_risk_slices)
        collision = np.zeros_like(valid, dtype=bool)
        cloud = self.latest_cloud
        if cloud is not None and cloud.header.frame_id == msg.info.header.frame_id:
            map_time = msg.info.header.stamp.sec + msg.info.header.stamp.nanosec * 1e-9
            cloud_time = cloud.header.stamp.sec + cloud.header.stamp.nanosec * 1e-9
            if abs(map_time - cloud_time) <= 0.35:
                fields = {f.name: f for f in cloud.fields}
                if all(name in fields and fields[name].datatype == PointField.FLOAT32
                       for name in ("x", "y", "z")):
                    dtype = np.dtype({"names": ["x", "y", "z"],
                                      "formats": [">f4" if cloud.is_bigendian else "<f4"] * 3,
                                      "offsets": [fields[n].offset for n in ("x", "y", "z")],
                                      "itemsize": cloud.point_step})
                    data = np.ndarray((cloud.height, cloud.width), dtype=dtype,
                                      buffer=cloud.data, strides=(cloud.row_step, cloud.point_step))
                    points = np.column_stack([data[n].ravel() for n in ("x", "y", "z")])
                    xp = np
                    if result.backend == "cupy":
                        import cupy as xp
                    mask, _ = vertical_columns(
                        points, valid.shape,
                        (msg.info.pose.position.x - msg.info.length_x / 2,
                         msg.info.pose.position.y - msg.info.length_y / 2),
                        self.params.resolution, xp=xp)
                    collision = xp.asnumpy(mask) if xp is not np else mask
                    preserve_collision_risk(result, collision)
        out_msg = self._build_output_message(
            msg,
            result,
            # Occupied cells are observed even when the ground estimator has
            # insufficient support; otherwise RViz filters out the red column.
            (valid | collision).astype(np.float32, copy=False),
            elevation_template or risk_template or template,
        )
        self._append_layer(out_msg, "collision_risk",
                           (False, [world_to_grid_buffer(collision.astype(np.float32), msg)]),
                           elevation_template or risk_template or template)
        return result, out_msg

    def _build_output_message(
        self,
        msg: SE2Grid,
        result: BgkSebResult,
        valid: np.ndarray,
        template: Float32MultiArray,
    ) -> SE2Grid:
        overrides: Dict[str, Tuple[bool, List[np.ndarray]]] = {
            "valid": (False, [valid]),
            "risk": (True, result.final_risk_slices),
            "baseline_risk": (False, [result.baseline_risk]),
            "bgk_mean": (False, [result.bgk_mean]),
            "bgk_var": (False, [result.bgk_var]),
            "bgk_traversability": (False, [result.bgk_traversability]),
            "seb_risk": (True, result.seb_risk_slices),
            "seb_pitch_risk": (True, result.seb_pitch_risk_slices),
            "seb_roll_risk": (True, result.seb_roll_risk_slices),
            "seb_step_risk": (True, result.seb_step_risk_slices),
            "seb_support": (True, result.seb_support_slices),
        }
        overrides = {
            name: (so2, [world_to_grid_buffer(matrix, msg) for matrix in matrices])
            for name, (so2, matrices) in overrides.items()
        }

        refs = layer_refs(msg)
        out = SE2Grid()
        out.info = copy.deepcopy(msg.info)
        out.info.header.stamp = msg.info.header.stamp
        if self.frame_id:
            out.info.header.frame_id = self.frame_id
        yaw_count = max(1, len(result.final_risk_slices))
        out.info.yaw_resolution = SE2_GRID_LENGTH_YAW / max(1.0e-6, float(yaw_count) - 1.0e-6)
        out.info.length_yaw = SE2_GRID_LENGTH_YAW
        out.start_index_row = msg.start_index_row
        out.start_index_col = msg.start_index_col

        emitted = set()
        for layer_name in msg.layers:
            if layer_name in overrides:
                self._append_layer(out, layer_name, overrides[layer_name], template)
                emitted.add(layer_name)
                continue

            ref = refs[layer_name]
            if not self.include_diagnostic_layers and ref.has_so2:
                continue
            matrices = [
                msg.data[idx]
                for idx in ref.data_indices
                if idx < len(msg.data)
            ]
            if not matrices:
                continue
            out.layers.append(layer_name)
            out.has_so2.append(ref.has_so2)
            if ref.has_so2 and len(matrices) != len(result.final_risk_slices):
                np_mats = [multiarray_to_matrix(m) for m in matrices]
                np_mats = repeat_or_trim_slices(
                    np_mats,
                    len(result.final_risk_slices),
                    np_mats[0],
                )
                out.data.extend(matrix_to_multiarray_like(m, template) for m in np_mats)
            else:
                out.data.extend(copy.deepcopy(matrices))
            emitted.add(layer_name)

        for layer_name in (
            "valid",
            "baseline_risk",
            "bgk_mean",
            "bgk_var",
            "bgk_traversability",
        ):
            if layer_name not in emitted:
                self._append_layer(out, layer_name, overrides[layer_name], template)
                emitted.add(layer_name)

        if self.include_diagnostic_layers:
            for layer_name in (
                "seb_risk",
                "seb_pitch_risk",
                "seb_roll_risk",
                "seb_step_risk",
                "seb_support",
            ):
                if layer_name not in emitted:
                    self._append_layer(out, layer_name, overrides[layer_name], template)
                    emitted.add(layer_name)

        return out

    def _append_layer(
        self,
        out: SE2Grid,
        layer_name: str,
        layer_data: Tuple[bool, List[np.ndarray]],
        template: Float32MultiArray,
    ) -> None:
        has_so2, matrices = layer_data
        out.layers.append(layer_name)
        out.has_so2.append(has_so2)
        for matrix in matrices:
            out.data.append(matrix_to_multiarray_like(matrix, template))

    def _publish_debug_clouds(self, msg: SE2Grid, result: BgkSebResult) -> None:
        self.bgk_cloud_pub.publish(
            self._grid_to_cloud(
                msg,
                result.bgk_mean,
                result.bgk_traversability,
                "bgk_traversability",
            )
        )
        if not result.seb_risk_slices:
            return
        yaw_idx = max(0, min(self.debug_yaw_index, len(result.seb_risk_slices) - 1))
        self.seb_cloud_pub.publish(
            self._grid_to_cloud(
                msg,
                result.bgk_mean,
                result.seb_risk_slices[yaw_idx],
                "seb_risk",
            )
        )

    def _grid_to_cloud(
        self,
        msg: SE2Grid,
        z_layer: np.ndarray,
        intensity_layer: np.ndarray,
        cloud_name: str,
    ) -> PointCloud2:
        rows, cols = z_layer.shape
        res = float(msg.info.pos_resolution)
        center_x = float(msg.info.pose.position.x)
        center_y = float(msg.info.pose.position.y)
        start_x = center_x - 0.5 * float(msg.info.length_x) + 0.5 * res
        start_y = center_y - 0.5 * float(msg.info.length_y) + 0.5 * res

        points = []
        stride = self.debug_cloud_stride
        for r in range(0, rows, stride):
            y = start_y + r * res
            for c in range(0, cols, stride):
                z = float(z_layer[r, c])
                intensity = float(intensity_layer[r, c])
                if not np.isfinite(z) or not np.isfinite(intensity):
                    continue
                x = start_x + c * res
                points.append((x, y, z + 0.05, intensity))

        header = Header()
        header.stamp = msg.info.header.stamp
        header.frame_id = msg.info.header.frame_id or self.frame_id
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        cloud = point_cloud2.create_cloud(header, fields, points)
        self.get_logger().debug(f"{cloud_name} cloud points={len(points)}")
        return cloud


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = FsgpBgkSe2Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
