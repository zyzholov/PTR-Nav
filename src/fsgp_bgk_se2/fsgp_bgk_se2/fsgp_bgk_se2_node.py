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
from .se2_grid_tools import (
    first_data_template,
    get_layer_2d,
    get_layer_slices,
    layer_refs,
    matrix_shape,
    matrix_to_multiarray_like,
    multiarray_to_matrix,
    repeat_or_trim_slices,
    yaw_slices,
)


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
                qos_profile_sensor_data,
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
            f"gpu={self.params.prefer_gpu}"
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
        self.declare_parameter("risk_fusion_mode", "max")
        self.declare_parameter("baseline_weight", 0.40)
        self.declare_parameter("seb_weight", 0.45)
        self.declare_parameter("uncertainty_weight", 0.15)
        self.declare_parameter("unknown_risk", 0.85)
        self.declare_parameter("publish_debug_clouds", True)
        self.declare_parameter("debug_yaw_index", 0)
        self.declare_parameter("debug_cloud_stride", 2)
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
        self.publish_debug_clouds = _as_bool(self.get_parameter("publish_debug_clouds").value)
        self.debug_yaw_index = int(self.get_parameter("debug_yaw_index").value)
        self.debug_cloud_stride = max(1, int(self.get_parameter("debug_cloud_stride").value))
        self.log_period_sec = max(0.2, float(self.get_parameter("log_period_sec").value))

        self.configured_yaw_bins = int(self.get_parameter("yaw_bins").value)
        self.params = BgkSebParams(
            resolution=0.20,
            yaw_bins=max(1, self.configured_yaw_bins or 72),
            prefer_gpu=_as_bool(self.get_parameter("use_gpu").value),
            gpu_backend=str(self.get_parameter("gpu_backend").value),
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
        )

    def _odom_callback(self, msg: Odometry) -> None:
        self.latest_odom = msg

    def _cloud_callback(self, msg: PointCloud2) -> None:
        self.latest_cloud = msg

    def _map_callback(self, msg: SE2Grid) -> None:
        try:
            start = time.perf_counter()
            result, out_msg = self._process_map(msg)
            self.map_pub.publish(out_msg)
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
        valid, _ = get_layer_2d(msg, "valid", 1.0, elevation.shape)
        valid = np.nan_to_num(valid, nan=0.0) >= 0.5

        risk_slices, _, risk_template = get_layer_slices(msg, "risk")
        if not risk_slices:
            risk_slices = [np.zeros_like(elevation, dtype=np.float32)]
        yaws = self.configured_yaw_bins or yaw_slices(msg)
        yaws = max(1, int(yaws))

        self.params.resolution = max(float(msg.info.pos_resolution), 1.0e-3)
        self.params.yaw_bins = yaws
        baseline_risk_slices = repeat_or_trim_slices(
            risk_slices,
            yaws,
            np.zeros_like(elevation, dtype=np.float32),
        )

        result = self.estimator.compute(elevation, variance, valid, baseline_risk_slices)
        out_msg = self._build_output_message(
            msg,
            result,
            elevation_template or risk_template or template,
        )
        return result, out_msg

    def _build_output_message(
        self,
        msg: SE2Grid,
        result: BgkSebResult,
        template: Float32MultiArray,
    ) -> SE2Grid:
        overrides: Dict[str, Tuple[bool, List[np.ndarray]]] = {
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

        refs = layer_refs(msg)
        out = SE2Grid()
        out.info = copy.deepcopy(msg.info)
        out.info.header.stamp = msg.info.header.stamp
        if self.frame_id:
            out.info.header.frame_id = self.frame_id
        out.info.yaw_resolution = 2.0 * math.pi / max(1, len(result.final_risk_slices))
        out.info.length_yaw = 2.0 * math.pi
        out.start_index_row = msg.start_index_row
        out.start_index_col = msg.start_index_col

        emitted = set()
        for layer_name in msg.layers:
            if layer_name in overrides:
                self._append_layer(out, layer_name, overrides[layer_name], template)
                emitted.add(layer_name)
                continue

            ref = refs[layer_name]
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
            "baseline_risk",
            "bgk_mean",
            "bgk_var",
            "bgk_traversability",
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
        start_x = center_x - 0.5 * float(msg.info.length_x) + float(msg.start_index_col) * res
        start_y = center_y - 0.5 * float(msg.info.length_y) + float(msg.start_index_row) * res

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
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
