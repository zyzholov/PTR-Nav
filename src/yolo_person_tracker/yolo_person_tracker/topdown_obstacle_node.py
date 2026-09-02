#!/usr/bin/env python3
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String


Point2 = Tuple[float, float]
BBox = Tuple[float, float, float, float]


@dataclass
class TopdownDetection:
    bbox: BBox
    confidence: float
    class_id: int
    class_name: str

    @property
    def center(self) -> Point2:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass
class TopdownTrack:
    track_id: int
    class_name: str
    bbox: BBox
    confidence: float
    history: List[Point2] = field(default_factory=list)
    misses: int = 0

    def update(self, det: TopdownDetection, history_len: int) -> None:
        keep = 0.62 if self.history else 0.0
        self.bbox = tuple(float(self.bbox[i] * keep + det.bbox[i] * (1.0 - keep)) for i in range(4))  # type: ignore[assignment]
        self.class_name = det.class_name
        self.confidence = det.confidence
        self.misses = 0
        self.history.append(self.center)
        if len(self.history) > history_len:
            self.history = self.history[-history_len:]

    @property
    def center(self) -> Point2:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def bbox_iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 1e-6 else 0.0


def center_distance(a: Point2, b: Point2) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def clamp_point(point: Point2, width: int, height: int) -> Tuple[int, int]:
    x = int(round(max(0.0, min(float(width - 1), point[0]))))
    y = int(round(max(0.0, min(float(height - 1), point[1]))))
    return x, y


class TopdownYoloDetector:
    COCO_IDS = {"person": 0, "bicycle": 1, "car": 2, "motorcycle": 3}

    def __init__(
        self,
        model_path: str,
        ultralytics_root: str,
        target_names: Sequence[str],
        conf_thres: float,
        iou_thres: float,
        imgsz: int,
        device: str,
        half: bool,
        max_det: int,
    ) -> None:
        root = Path(ultralytics_root).expanduser()
        if root.exists() and str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from ultralytics import YOLO

        self.model = YOLO(str(Path(model_path).expanduser()))
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.imgsz = imgsz
        self.device = self._resolve_device(device)
        self.half = bool(half and self.device != "cpu")
        self.max_det = max_det
        wanted = [name.strip().lower() for name in target_names if name.strip()]
        self.target_ids = [self.COCO_IDS[name] for name in wanted if name in self.COCO_IDS]
        self.id_to_name = {v: k for k, v in self.COCO_IDS.items()}

    @staticmethod
    def _resolve_device(device: str) -> str:
        requested = str(device).strip().lower()
        if requested in ("", "auto", "none"):
            try:
                import torch

                return "0" if torch.cuda.is_available() else "cpu"
            except Exception:
                return "cpu"
        if requested in ("cuda", "gpu"):
            return "0"
        return requested

    def detect(self, frame: np.ndarray) -> List[TopdownDetection]:
        kwargs = {
            "source": frame,
            "conf": self.conf_thres,
            "iou": self.iou_thres,
            "imgsz": self.imgsz,
            "device": self.device,
            "half": self.half,
            "verbose": False,
            "max_det": self.max_det,
        }
        if self.target_ids:
            kwargs["classes"] = self.target_ids
        results = self.model.predict(**kwargs)
        if not results or results[0].boxes is None:
            return []
        boxes = results[0].boxes
        xyxy = boxes.xyxy.detach().cpu().numpy()
        conf = boxes.conf.detach().cpu().numpy()
        cls = boxes.cls.detach().cpu().numpy().astype(int)
        detections: List[TopdownDetection] = []
        for bbox, score, cls_id in zip(xyxy, conf, cls):
            if self.target_ids and int(cls_id) not in self.target_ids:
                continue
            detections.append(
                TopdownDetection(
                    bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                    confidence=float(score),
                    class_id=int(cls_id),
                    class_name=self.id_to_name.get(int(cls_id), str(cls_id)),
                )
            )
        return detections


class TopdownBicycleColorFallback:
    """Practical fallback for the light-blue bicycles in the topdown demo videos."""

    def __init__(
        self,
        enabled: bool,
        min_area: float,
        expand_ratio: float,
        border_margin_ratio: float,
    ) -> None:
        self.enabled = enabled
        self.min_area = max(1.0, float(min_area))
        self.expand_ratio = max(1.0, float(expand_ratio))
        self.border_margin_ratio = max(0.0, min(0.2, float(border_margin_ratio)))

    def detect(self, frame: np.ndarray) -> List[TopdownDetection]:
        if not self.enabled:
            return []
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Demo bicycles are pale cyan/blue in the aerial videos. This intentionally
        # complements YOLO; it is not a general bicycle detector.
        mask = cv2.inRange(hsv, (88, 45, 78), (116, 255, 255))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        margin_x = int(w * self.border_margin_ratio)
        margin_y = int(h * self.border_margin_ratio)
        part_boxes: List[BBox] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < max(70.0, self.min_area * 0.35) or area > 1900.0:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            if x <= margin_x or y <= margin_y or x + bw >= w - margin_x or y + bh >= h - margin_y:
                continue
            aspect = max(float(bw) / max(1.0, float(bh)), float(bh) / max(1.0, float(bw)))
            if aspect > 6.4 or max(bw, bh) > 95:
                continue
            part_boxes.append((float(x), float(y), float(x + bw), float(y + bh)))

        merged = self._merge_part_boxes(part_boxes)
        boxes: List[BBox] = []
        for box in merged:
            x1, y1, x2, y2 = box
            bw = x2 - x1
            bh = y2 - y1
            if bw < 14.0 or bh < 10.0:
                continue
            if max(bw, bh) > 230.0:
                continue
            aspect = max(bw / max(1.0, bh), bh / max(1.0, bw))
            if aspect > 5.9:
                continue
            pad_x = max(22.0, bw * 0.48)
            pad_y = max(20.0, bh * 0.72)
            boxes.append(
                (
                    max(0.0, x1 - pad_x),
                    max(0.0, y1 - pad_y),
                    min(float(w - 1), x2 + pad_x),
                    min(float(h - 1), y2 + pad_y),
                )
            )
        boxes = [bbox for bbox in boxes if self._passes_context_filter(bbox, hsv, gray)]
        return [
            TopdownDetection(bbox=bbox, confidence=0.42, class_id=1, class_name="bicycle")
            for bbox in boxes
        ]

    @staticmethod
    def _passes_context_filter(bbox: BBox, hsv: np.ndarray, gray: np.ndarray) -> bool:
        h, w = gray.shape[:2]
        x1 = int(round(max(0.0, min(float(w - 1), bbox[0]))))
        y1 = int(round(max(0.0, min(float(h - 1), bbox[1]))))
        x2 = int(round(max(0.0, min(float(w), bbox[2]))))
        y2 = int(round(max(0.0, min(float(h), bbox[3]))))
        if x2 <= x1 or y2 <= y1:
            return False
        gray_roi = gray[y1:y2, x1:x2]
        hsv_roi = hsv[y1:y2, x1:x2]
        if gray_roi.size == 0:
            return False
        dark_ratio = float(np.mean(gray_roi < 80))
        blue_ratio = float(np.mean(cv2.inRange(hsv_roi, (88, 45, 78), (116, 255, 255)) > 0))
        return dark_ratio < 0.105 and blue_ratio > 0.085

    @staticmethod
    def _merge_part_boxes(boxes: Sequence[BBox]) -> List[BBox]:
        merged: List[BBox] = []
        for box in sorted(boxes, key=lambda b: (b[0], b[1])):
            did_merge = False
            for idx, old in enumerate(merged):
                gap = center_distance(
                    ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5),
                    ((old[0] + old[2]) * 0.5, (old[1] + old[3]) * 0.5),
                )
                if bbox_iou(box, old) > 0.0 or gap < 86.0:
                    merged[idx] = (
                        min(box[0], old[0]),
                        min(box[1], old[1]),
                        max(box[2], old[2]),
                        max(box[3], old[3]),
                    )
                    did_merge = True
                    break
            if not did_merge:
                merged.append(box)
        return merged


class TopdownObstacleNode(Node):
    def __init__(self) -> None:
        super().__init__("topdown_obstacle_node")
        self.declare_parameter("video_path", "/home/z/Videos/tw/媒体4.mp4")
        self.declare_parameter("model_path", "/home/z/workspace_zyz/yolo_v8_man_detection/ultralytics-main/yolov8n.pt")
        self.declare_parameter("ultralytics_root", "/home/z/workspace_zyz/yolo_v8_man_detection/ultralytics-main")
        self.declare_parameter("target_class_names", "person,bicycle")
        self.declare_parameter("conf_thres", 0.18)
        self.declare_parameter("iou_thres", 0.45)
        self.declare_parameter("imgsz", 960)
        self.declare_parameter("device", "0")
        self.declare_parameter("half", True)
        self.declare_parameter("max_det", 60)
        self.declare_parameter("show_window", True)
        self.declare_parameter("loop_video", True)
        self.declare_parameter("max_fps", 60.0)
        self.declare_parameter("publish_annotated", True)
        self.declare_parameter("publish_image_stride", 2)
        self.declare_parameter("max_frames", 0)
        self.declare_parameter("track_history_len", 80)
        self.declare_parameter("max_missing_frames", 16)
        self.declare_parameter("match_max_distance_px", 120.0)
        self.declare_parameter("draw_track_history", False)
        self.declare_parameter("robot_anchor_x_ratio", 0.50)
        self.declare_parameter("robot_anchor_y_ratio", 0.78)
        self.declare_parameter("forward_goal_y_ratio", 0.20)
        self.declare_parameter("corridor_width_ratio", 0.22)
        self.declare_parameter("stop_radius_ratio", 0.15)
        self.declare_parameter("warn_radius_ratio", 0.26)
        self.declare_parameter("draw_semantic_path", False)
        self.declare_parameter("enable_bicycle_color_fallback", True)
        self.declare_parameter("bicycle_color_min_area", 260.0)
        self.declare_parameter("bicycle_color_expand_ratio", 3.4)
        self.declare_parameter("bicycle_color_border_margin_ratio", 0.035)

        self.video_path = self._param_str("video_path")
        target_names = [name.strip() for name in self._param_str("target_class_names").split(",")]
        self.target_names_lower = {name.strip().lower() for name in target_names}
        self.detector = TopdownYoloDetector(
            model_path=self._param_str("model_path"),
            ultralytics_root=self._param_str("ultralytics_root"),
            target_names=target_names,
            conf_thres=self._param_float("conf_thres"),
            iou_thres=self._param_float("iou_thres"),
            imgsz=self._param_int("imgsz"),
            device=self._param_str("device"),
            half=self._param_bool("half"),
            max_det=self._param_int("max_det"),
        )
        self.show_window = self._param_bool("show_window")
        self.loop_video = self._param_bool("loop_video")
        self.max_fps = self._param_float("max_fps")
        self.publish_annotated = self._param_bool("publish_annotated")
        self.publish_image_stride = max(1, self._param_int("publish_image_stride"))
        self.max_frames = self._param_int("max_frames")
        self.track_history_len = max(2, self._param_int("track_history_len"))
        self.max_missing_frames = max(0, self._param_int("max_missing_frames"))
        self.match_max_distance_px = self._param_float("match_max_distance_px")
        self.draw_track_history = self._param_bool("draw_track_history")
        self.robot_anchor_x_ratio = self._param_float("robot_anchor_x_ratio")
        self.robot_anchor_y_ratio = self._param_float("robot_anchor_y_ratio")
        self.forward_goal_y_ratio = self._param_float("forward_goal_y_ratio")
        self.corridor_width_ratio = self._param_float("corridor_width_ratio")
        self.stop_radius_ratio = self._param_float("stop_radius_ratio")
        self.warn_radius_ratio = self._param_float("warn_radius_ratio")
        self.draw_semantic_path = self._param_bool("draw_semantic_path")
        self.bicycle_fallback = TopdownBicycleColorFallback(
            enabled=self._param_bool("enable_bicycle_color_fallback") and "bicycle" in self.target_names_lower,
            min_area=self._param_float("bicycle_color_min_area"),
            expand_ratio=self._param_float("bicycle_color_expand_ratio"),
            border_margin_ratio=self._param_float("bicycle_color_border_margin_ratio"),
        )

        if self.show_window and not os.environ.get("DISPLAY"):
            self.show_window = False
        self.cap = cv2.VideoCapture(str(Path(self.video_path).expanduser()))
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count_total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.frame_index = 0
        self.last_process_wall = 0.0
        self.next_track_id = 1
        self.tracks: Dict[int, TopdownTrack] = {}
        self.last_detections: List[TopdownDetection] = []
        self.last_inference_fps = 0.0
        self.stop_requested = False

        qos = 10
        self.image_pub = self.create_publisher(Image, "~/annotated_image", qos)
        self.detection_pub = self.create_publisher(String, "~/detections", qos)
        self.stop_pub = self.create_publisher(Bool, "~/semantic_stop", qos)
        self.status_pub = self.create_publisher(String, "~/status", qos)
        self.timer = self.create_timer(0.001, self._process_once)
        self.get_logger().info(
            "Topdown semantic obstacle node: video=%s, target=%s, device=%s, half=%s, fps=%.1f, frames=%d"
            % (self.video_path, ",".join(target_names), self.detector.device, self.detector.half, self.video_fps, self.frame_count_total)
        )

    def _param_str(self, name: str) -> str:
        return str(self.get_parameter(name).value)

    def _param_float(self, name: str) -> float:
        return float(self.get_parameter(name).value)

    def _param_int(self, name: str) -> int:
        return int(self.get_parameter(name).value)

    def _param_bool(self, name: str) -> bool:
        return bool(self.get_parameter(name).value)

    def _process_once(self) -> None:
        now = time.monotonic()
        target_fps = min(self.video_fps, self.max_fps) if self.max_fps > 0.0 else self.video_fps
        if target_fps > 0.0 and now - self.last_process_wall < 1.0 / target_fps:
            return
        self.last_process_wall = now

        ok, frame = self.cap.read()
        if not ok:
            if self.loop_video:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.frame_index = 0
                ok, frame = self.cap.read()
            if not ok:
                self._request_stop()
                return
        self.frame_index += 1
        if self.max_frames > 0 and self.frame_index > self.max_frames:
            self._request_stop()
            return

        start = time.perf_counter()
        detections = self.detector.detect(frame)
        detections = self._augment_bicycle_detections(frame, detections)
        self.last_inference_fps = 1.0 / max(1e-6, time.perf_counter() - start)
        self.last_detections = detections
        self._update_tracks(detections)
        stop, risk_level = self._risk_state(frame.shape[:2])
        annotated = self._draw(frame.copy(), stop, risk_level)
        stamp = self.get_clock().now().to_msg()
        self._publish(stamp, annotated, stop, risk_level)
        if self.show_window:
            cv2.imshow("Topdown Semantic Obstacle", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                self._request_stop()

    def _update_tracks(self, detections: Sequence[TopdownDetection]) -> None:
        unmatched_tracks = set(self.tracks.keys())
        unmatched_dets = set(range(len(detections)))
        pairs: List[Tuple[float, int, int]] = []
        for track_id, track in self.tracks.items():
            for det_idx, det in enumerate(detections):
                iou = bbox_iou(track.bbox, det.bbox)
                dist = center_distance(track.center, det.center)
                if track.class_name != det.class_name:
                    continue
                if iou < 0.02 and dist > self.match_max_distance_px:
                    continue
                score = iou * 1.4 + max(0.0, 1.0 - dist / max(1.0, self.match_max_distance_px))
                pairs.append((score, track_id, det_idx))
        for _, track_id, det_idx in sorted(pairs, reverse=True):
            if track_id not in unmatched_tracks or det_idx not in unmatched_dets:
                continue
            self.tracks[track_id].update(detections[det_idx], self.track_history_len)
            unmatched_tracks.remove(track_id)
            unmatched_dets.remove(det_idx)
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            track = TopdownTrack(self.next_track_id, det.class_name, det.bbox, det.confidence)
            track.update(det, self.track_history_len)
            self.tracks[self.next_track_id] = track
            self.next_track_id += 1
        for track_id in list(unmatched_tracks):
            self.tracks[track_id].misses += 1
            if self.tracks[track_id].misses > self.max_missing_frames:
                del self.tracks[track_id]

    def _risk_state(self, shape_hw: Tuple[int, int]) -> Tuple[bool, str]:
        h, w = shape_hw
        anchor = (w * self.robot_anchor_x_ratio, h * self.robot_anchor_y_ratio)
        stop_radius = min(w, h) * self.stop_radius_ratio
        warn_radius = min(w, h) * self.warn_radius_ratio
        risk = "CLEAR"
        stop = False
        for track in self.tracks.values():
            if track.misses > 0:
                continue
            dist = center_distance(anchor, track.center)
            if dist <= stop_radius and track.class_name == "person":
                return True, "STOP_PERSON"
            if dist <= stop_radius:
                stop = True
                risk = "STOP_OBSTACLE"
            elif dist <= warn_radius and risk == "CLEAR":
                risk = "AVOID"
        return stop, risk

    def _draw(self, image: np.ndarray, stop: bool, risk_level: str) -> np.ndarray:
        h, w = image.shape[:2]
        if self.draw_semantic_path:
            self._draw_semantic_path(image, risk_level)
        for track in sorted(self.tracks.values(), key=lambda t: t.track_id):
            if track.misses > 0:
                continue
            color = self._class_color(track.class_name)
            x1, y1, x2, y2 = [int(round(v)) for v in track.bbox]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
            cv2.putText(
                image,
                f"{track.class_name} {track.confidence:.2f}",
                (x1, max(24, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                color,
                2,
                cv2.LINE_AA,
            )
            if self.draw_track_history and len(track.history) >= 2:
                pts = np.array([clamp_point(p, w, h) for p in track.history], dtype=np.int32)
                cv2.polylines(image, [pts], False, color, 2, cv2.LINE_AA)
        self._draw_hud(image, stop, risk_level)
        return image

    def _draw_semantic_path(self, image: np.ndarray, risk_level: str) -> None:
        h, w = image.shape[:2]
        anchor = np.array([w * self.robot_anchor_x_ratio, h * self.robot_anchor_y_ratio], dtype=float)
        goal = np.array([w * 0.50, h * self.forward_goal_y_ratio], dtype=float)
        offset_x = 0.0
        for track in self.tracks.values():
            if track.misses > 0:
                continue
            cx, cy = track.center
            if min(anchor[1], goal[1]) <= cy <= max(anchor[1], goal[1]) and abs(cx - anchor[0]) < w * self.corridor_width_ratio:
                sign = -1.0 if cx >= anchor[0] else 1.0
                offset_x += sign * w * 0.12
        offset_x = max(-w * 0.22, min(w * 0.22, offset_x))
        control = (anchor + goal) * 0.5 + np.array([offset_x, 0.0])
        points = []
        for i in range(28):
            t = i / 27.0
            p = (1 - t) * (1 - t) * anchor + 2 * (1 - t) * t * control + t * t * goal
            points.append((float(p[0]), float(p[1])))
        half_width = w * self.corridor_width_ratio * 0.42
        left = [(x - half_width * (1.0 - i / 27.0) - 16, y) for i, (x, y) in enumerate(points)]
        right = [(x + half_width * (1.0 - i / 27.0) + 16, y) for i, (x, y) in enumerate(points)]
        poly = np.array([clamp_point(p, w, h) for p in left + list(reversed(right))], dtype=np.int32)
        overlay = image.copy()
        fill = (36, 190, 82) if risk_level == "CLEAR" else (34, 178, 255) if risk_level == "AVOID" else (42, 64, 232)
        cv2.fillPoly(overlay, [poly], fill, cv2.LINE_AA)
        image[:] = cv2.addWeighted(overlay, 0.24, image, 0.76, 0.0)
        center = np.array([clamp_point(p, w, h) for p in points], dtype=np.int32)
        left_arr = np.array([clamp_point(p, w, h) for p in left], dtype=np.int32)
        right_arr = np.array([clamp_point(p, w, h) for p in right], dtype=np.int32)
        cv2.polylines(image, [left_arr, right_arr], False, fill, 3, cv2.LINE_AA)
        cv2.polylines(image, [center], False, (255, 255, 255), 2, cv2.LINE_AA)

    def _draw_hud(self, image: np.ndarray, stop: bool, risk_level: str) -> None:
        h, w = image.shape[:2]
        overlay = image[:76, :].copy()
        cv2.rectangle(overlay, (0, 0), (w, 76), (18, 24, 30), -1)
        image[:76, :] = cv2.addWeighted(overlay, 0.58, image[:76, :], 0.42, 0)
        status_color = (40, 66, 235) if stop else (42, 210, 255) if risk_level != "CLEAR" else (88, 232, 96)
        cv2.putText(image, "Topdown YOLO Semantic Obstacle Avoidance", (22, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (244, 248, 255), 2, cv2.LINE_AA)
        cv2.putText(
            image,
            f"risk={risk_level}  stop={stop}  infer={self.last_inference_fps:.1f} FPS  tracks={len(self.tracks)}",
            (22, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            status_color,
            2,
            cv2.LINE_AA,
        )

    @staticmethod
    def _class_color(name: str) -> Tuple[int, int, int]:
        if name == "person":
            return (236, 88, 255)
        if name == "bicycle":
            return (38, 220, 255)
        return (100, 220, 120)

    def _augment_bicycle_detections(
        self, frame: np.ndarray, detections: Sequence[TopdownDetection]
    ) -> List[TopdownDetection]:
        if "bicycle" not in self.target_names_lower:
            return list(detections)
        augmented = list(detections)
        existing = [det for det in detections if det.class_name == "bicycle"]
        for det in self.bicycle_fallback.detect(frame):
            if any(bbox_iou(det.bbox, old.bbox) > 0.10 or center_distance(det.center, old.center) < 56.0 for old in existing):
                continue
            augmented.append(det)
        return augmented

    def _publish(self, stamp, annotated: np.ndarray, stop: bool, risk_level: str) -> None:
        if self.publish_annotated and self.frame_index % self.publish_image_stride == 0 and self.image_pub.get_subscription_count() > 0:
            msg = Image()
            msg.header.stamp = stamp
            msg.header.frame_id = "topdown_view"
            msg.height = int(annotated.shape[0])
            msg.width = int(annotated.shape[1])
            msg.encoding = "bgr8"
            msg.is_bigendian = False
            msg.step = int(annotated.shape[1] * 3)
            msg.data = annotated.tobytes()
            self.image_pub.publish(msg)
        payload = {
            "frame_index": self.frame_index,
            "risk": risk_level,
            "stop": stop,
            "detections": [
                {
                    "class": det.class_name,
                    "confidence": round(det.confidence, 4),
                    "bbox_xyxy": [round(v, 2) for v in det.bbox],
                    "center_px": [round(det.center[0], 2), round(det.center[1], 2)],
                }
                for det in self.last_detections
            ],
        }
        self.detection_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
        self.stop_pub.publish(Bool(data=bool(stop)))
        self.status_pub.publish(String(data=json.dumps({"frame": self.frame_index, "risk": risk_level, "tracks": len(self.tracks)})))

    def _request_stop(self) -> None:
        self.stop_requested = True

    def destroy_node(self) -> bool:
        try:
            if hasattr(self, "cap"):
                self.cap.release()
            cv2.destroyAllWindows()
        finally:
            return super().destroy_node()


def main(args: Optional[Iterable[str]] = None) -> None:
    rclpy.init(args=args)
    node: Optional[TopdownObstacleNode] = None
    try:
        node = TopdownObstacleNode()
        while rclpy.ok() and not node.stop_requested:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
