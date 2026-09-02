#!/usr/bin/env python3
import json
import math
import os
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String


Point2 = Tuple[float, float]
BBox = Tuple[float, float, float, float]


@dataclass
class CameraMotionState:
    yaw: float = 0.0
    flow_dx_px: float = 0.0
    flow_dy_px: float = 0.0
    confidence: float = 0.0


@dataclass
class Detection:
    bbox: BBox
    confidence: float
    class_id: int
    class_name: str
    contour: Optional[List[Point2]] = None

    @property
    def center(self) -> Point2:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

    @property
    def foot_point(self) -> Point2:
        x1, y1, x2, y2 = self.bbox
        # Use a lower-body anchor instead of the exact bbox bottom so the trail
        # follows the person, not the heel/ground contact jitter.
        return ((x1 + x2) * 0.5, y1 + (y2 - y1) * 0.78)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass
class Track:
    track_id: int
    bbox: BBox
    confidence: float
    class_name: str
    history: Deque[Tuple[float, float, float]] = field(default_factory=lambda: deque(maxlen=48))
    bbox_history: Deque[Tuple[float, float, float, float, float]] = field(default_factory=lambda: deque(maxlen=48))
    contour_history: Deque[Tuple[List[Point2], float]] = field(default_factory=lambda: deque(maxlen=48))
    velocity: Point2 = (0.0, 0.0)
    last_stamp_s: float = 0.0
    misses: int = 0

    def update(self, det: Detection, stamp_s: float, trail_length: int) -> None:
        self.bbox = self._smooth_bbox(det.bbox)
        self.confidence = det.confidence
        self.class_name = det.class_name
        self.misses = 0
        self.history = deque(self.history, maxlen=trail_length)
        self.bbox_history = deque(self.bbox_history, maxlen=trail_length)
        self.contour_history = deque(self.contour_history, maxlen=trail_length)
        foot = det.foot_point

        if not self.history:
            self.history.append((foot[0], foot[1], stamp_s))
            self.bbox_history.append((*self.bbox, stamp_s))
            self.contour_history.append((self._contour_for_bbox(det.contour, self.bbox, det.foot_point, foot), stamp_s))
            self.last_stamp_s = stamp_s
            self.velocity = (0.0, 0.0)
            return

        dt = max(1e-3, min(0.45, stamp_s - self.last_stamp_s))
        pred_x, pred_y = self.predicted_foot(stamp_s)
        err_x = foot[0] - pred_x
        err_y = foot[1] - pred_y

        # Alpha-beta filter on the foot point: enough damping for bbox jitter,
        # but still responsive to a walking person.
        alpha = 0.36
        beta = 0.08
        smooth_x = pred_x + alpha * err_x
        smooth_y = pred_y + alpha * err_y
        vel_x = self.velocity[0] + beta * err_x / dt
        vel_y = self.velocity[1] + beta * err_y / dt
        self.velocity = self._limit_velocity((vel_x, vel_y))
        self.bbox = self._bbox_with_anchor(self.bbox, smooth_x, smooth_y)
        self.history.append((smooth_x, smooth_y, stamp_s))
        self.bbox_history.append((*self.bbox, stamp_s))
        self.contour_history.append((self._contour_for_bbox(det.contour, self.bbox, det.foot_point, (smooth_x, smooth_y)), stamp_s))
        self.last_stamp_s = stamp_s

    def _smooth_bbox(self, bbox: BBox) -> BBox:
        if not self.history:
            return bbox
        keep = 0.62
        return tuple(float(self.bbox[i] * keep + bbox[i] * (1.0 - keep)) for i in range(4))  # type: ignore[return-value]

    @staticmethod
    def _bbox_with_anchor(bbox: BBox, anchor_x: float, anchor_y: float) -> BBox:
        x1, y1, x2, y2 = bbox
        width = max(4.0, x2 - x1)
        height = max(8.0, y2 - y1)
        new_x1 = anchor_x - width * 0.5
        new_y1 = anchor_y - height * 0.78
        return (new_x1, new_y1, new_x1 + width, new_y1 + height)

    def _contour_for_bbox(
        self,
        contour: Optional[List[Point2]],
        bbox: BBox,
        observed_anchor: Point2,
        smoothed_anchor: Point2,
    ) -> List[Point2]:
        if contour:
            dx = smoothed_anchor[0] - observed_anchor[0]
            dy = smoothed_anchor[1] - observed_anchor[1]
            return [(float(x + dx), float(y + dy)) for x, y in contour]
        if self.contour_history and self.history:
            prev_contour, _ = self.contour_history[-1]
            prev_anchor = (self.history[-1][0], self.history[-1][1])
            dx = smoothed_anchor[0] - prev_anchor[0]
            dy = smoothed_anchor[1] - prev_anchor[1]
            return [(float(x + dx), float(y + dy)) for x, y in prev_contour]
        x1, y1, x2, y2 = bbox
        return self._stylized_person_contour(x1, y1, x2, y2)

    @staticmethod
    def _stylized_person_contour(x1: float, y1: float, x2: float, y2: float) -> List[Point2]:
        width = max(4.0, x2 - x1)
        height = max(8.0, y2 - y1)
        cx = (x1 + x2) * 0.5
        return [
            (cx - width * 0.10, y1 + height * 0.02),
            (cx - width * 0.24, y1 + height * 0.08),
            (cx - width * 0.30, y1 + height * 0.18),
            (cx - width * 0.48, y1 + height * 0.26),
            (cx - width * 0.42, y1 + height * 0.42),
            (cx - width * 0.30, y1 + height * 0.55),
            (cx - width * 0.22, y1 + height * 0.76),
            (cx - width * 0.18, y1 + height * 0.98),
            (cx - width * 0.02, y1 + height * 0.98),
            (cx, y1 + height * 0.70),
            (cx + width * 0.04, y1 + height * 0.98),
            (cx + width * 0.22, y1 + height * 0.98),
            (cx + width * 0.18, y1 + height * 0.75),
            (cx + width * 0.28, y1 + height * 0.55),
            (cx + width * 0.42, y1 + height * 0.42),
            (cx + width * 0.48, y1 + height * 0.26),
            (cx + width * 0.30, y1 + height * 0.18),
            (cx + width * 0.24, y1 + height * 0.08),
            (cx + width * 0.10, y1 + height * 0.02),
        ]

    @staticmethod
    def _limit_velocity(velocity: Point2, max_speed_px_s: float = 950.0) -> Point2:
        speed = math.hypot(velocity[0], velocity[1])
        if speed <= max_speed_px_s or speed <= 1e-6:
            return velocity
        scale = max_speed_px_s / speed
        return (velocity[0] * scale, velocity[1] * scale)

    @property
    def center(self) -> Point2:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

    @property
    def foot_point(self) -> Point2:
        if self.history:
            x, y, _ = self.history[-1]
            return (x, y)
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) * 0.5, y1 + (y2 - y1) * 0.78)

    def predicted_foot(self, stamp_s: float) -> Point2:
        foot = self.foot_point
        if self.last_stamp_s <= 0.0:
            return foot
        dt = max(0.0, min(0.35, stamp_s - self.last_stamp_s))
        return (foot[0] + self.velocity[0] * dt, foot[1] + self.velocity[1] * dt)

    def velocity_px_s(self) -> Point2:
        return self.velocity

    def predicted_points(self, horizon_s: float, step_s: float, stamp_s: float) -> List[Point2]:
        if horizon_s <= 0.0 or step_s <= 0.0:
            return []
        vx, vy = self.velocity_px_s()
        if abs(vx) + abs(vy) < 8.0:
            return []
        cx, cy = self.predicted_foot(stamp_s)
        points = []
        t = step_s
        while t <= horizon_s + 1e-6:
            points.append((cx + vx * t, cy + vy * t))
            t += step_s
        return points


class YoloDetector:
    def __init__(
        self,
        model_path: str,
        ultralytics_root: str,
        target_class_names: Sequence[str],
        conf_thres: float,
        iou_thres: float,
        imgsz: int,
        device: str,
        half: bool,
        extract_contours: bool,
        contour_max_side: int,
        contour_max_points: int,
        contour_interval: int,
    ) -> None:
        root = Path(ultralytics_root).expanduser()
        if root.exists() and str(root) not in sys.path:
            sys.path.insert(0, str(root))

        try:
            from ultralytics import YOLO
        except Exception as exc:
            hint = (
                "Failed to import ultralytics. Check that the selected Python environment "
                "can import torch, torchvision, and ultralytics."
            )
            raise RuntimeError(f"{hint}\n{traceback.format_exc()}") from exc

        model_file = Path(model_path).expanduser()
        if not model_file.exists():
            raise FileNotFoundError(f"YOLO model not found: {model_file}")

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.imgsz = imgsz
        self.device = self._resolve_device(device)
        self.half = bool(half and self.device != "cpu")
        self.extract_contours = extract_contours
        self.contour_max_side = max(64, int(contour_max_side))
        self.contour_max_points = max(12, int(contour_max_points))
        self.contour_interval = max(1, int(contour_interval))
        self.detect_count = self.contour_interval - 1
        self.model = YOLO(str(model_file))

        wanted = {name.strip().lower() for name in target_class_names if name.strip()}
        self.target_ids = [0] if wanted.intersection({"person", "man", "pedestrian"}) else []
        self.names: Dict[int, str] = {0: "person"}

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

    @staticmethod
    def _normalize_names(names) -> Dict[int, str]:
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        return {i: str(v) for i, v in enumerate(names)}

    def detect(self, frame: np.ndarray) -> List[Detection]:
        kwargs = {
            "source": frame,
            "conf": self.conf_thres,
            "iou": self.iou_thres,
            "imgsz": self.imgsz,
            "verbose": False,
            "device": self.device,
            "half": self.half,
            "max_det": 20,
        }
        if self.target_ids:
            kwargs["classes"] = self.target_ids

        results = self.model.predict(**kwargs)
        detections: List[Detection] = []
        if not results:
            return detections

        boxes = getattr(results[0], "boxes", None)
        if boxes is None or boxes.xyxy is None:
            return detections

        xyxy = boxes.xyxy.detach().cpu().numpy()
        conf = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.ones((len(xyxy),), dtype=float)
        cls = boxes.cls.detach().cpu().numpy().astype(int) if boxes.cls is not None else np.zeros((len(xyxy),), dtype=int)
        self.detect_count += 1
        extract_contours_this_frame = self.extract_contours and (self.detect_count % self.contour_interval == 0)
        for bbox, score, cls_id in zip(xyxy, conf, cls):
            name = self.names.get(int(cls_id), str(cls_id))
            if self.target_ids and int(cls_id) not in self.target_ids:
                continue
            detections.append(
                Detection(
                    bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                    confidence=float(score),
                    class_id=int(cls_id),
                    class_name=name,
                    contour=self._extract_person_contour(frame, bbox) if extract_contours_this_frame else None,
                )
            )
        return detections

    def _extract_person_contour(self, frame: np.ndarray, bbox: Sequence[float]) -> Optional[List[Point2]]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in bbox]
        bw = max(4.0, x2 - x1)
        bh = max(8.0, y2 - y1)
        px = bw * 0.08
        py = bh * 0.04
        ix1 = max(0, int(x1 - px))
        iy1 = max(0, int(y1 - py))
        ix2 = min(w, int(x2 + px))
        iy2 = min(h, int(y2 + py))
        if ix2 <= ix1 + 8 or iy2 <= iy1 + 16:
            return None

        crop = frame[iy1:iy2, ix1:ix2]
        ch, cw = crop.shape[:2]
        scale = min(1.0, float(self.contour_max_side) / float(max(ch, cw)))
        small = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 0.999 else crop
        sh, sw = small.shape[:2]
        if sw < 8 or sh < 16:
            return None

        rect = (2, 2, max(2, sw - 4), max(2, sh - 4))
        mask = np.zeros((sh, sw), np.uint8)
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(small, mask, rect, bgd, fgd, 1, cv2.GC_INIT_WITH_RECT)
        except Exception:
            return self._fallback_edge_contour(ix1, iy1, x1, y1, x2, y2)

        fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return self._fallback_edge_contour(ix1, iy1, x1, y1, x2, y2)

        contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(contour)
        cx, cy, cw, ch = cv2.boundingRect(contour)
        fills_crop = contour_area > sw * sh * 0.86 or (cw > sw * 0.94 and ch > sh * 0.94)
        if contour_area < max(24.0, sw * sh * 0.035) or fills_crop:
            return self._fallback_edge_contour(ix1, iy1, x1, y1, x2, y2)

        epsilon = max(1.0, 0.006 * cv2.arcLength(contour, True))
        approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(approx) > self.contour_max_points:
            idx = np.linspace(0, len(approx) - 1, self.contour_max_points).astype(int)
            approx = approx[idx]

        inv = 1.0 / scale if scale > 1e-6 else 1.0
        points = [(float(ix1 + p[0] * inv), float(iy1 + p[1] * inv)) for p in approx]
        return points if len(points) >= 3 else self._fallback_edge_contour(ix1, iy1, x1, y1, x2, y2)

    def _fallback_edge_contour(self, ix1: int, iy1: int, x1: float, y1: float, x2: float, y2: float) -> List[Point2]:
        return Track._stylized_person_contour(float(x1), float(y1), float(x2), float(y2))


class HogDetector:
    """A small fallback/debug backend. Use YOLO for real demos."""

    def __init__(self, conf_thres: float) -> None:
        self.conf_thres = conf_thres
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame: np.ndarray) -> List[Detection]:
        scale = 0.5 if max(frame.shape[:2]) > 900 else 1.0
        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1.0 else frame
        rects, weights = self.hog.detectMultiScale(small, winStride=(8, 8), padding=(8, 8), scale=1.05)
        detections: List[Detection] = []
        inv = 1.0 / scale
        for (x, y, w, h), score in zip(rects, weights):
            confidence = float(max(0.0, min(1.0, score / 3.0)))
            if confidence < self.conf_thres:
                continue
            detections.append(
                Detection(
                    bbox=(float(x * inv), float(y * inv), float((x + w) * inv), float((y + h) * inv)),
                    confidence=confidence,
                    class_id=0,
                    class_name="person",
                )
            )
        return detections


def bbox_iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 1e-6 else 0.0


def center_distance(a: Point2, b: Point2) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def clamp_point(point: Point2, width: int, height: int) -> Tuple[int, int]:
    x = int(round(max(0.0, min(float(width - 1), point[0]))))
    y = int(round(max(0.0, min(float(height - 1), point[1]))))
    return x, y


class CameraMotionEstimator:
    def __init__(
        self,
        max_width: int,
        roi_y_ratio: float,
        smoothing: float,
        yaw_gain_px: float,
        update_stride: int,
    ) -> None:
        self.max_width = max(240, int(max_width))
        self.roi_y_ratio = max(0.25, min(0.85, float(roi_y_ratio)))
        self.smoothing = max(0.0, min(0.98, float(smoothing)))
        self.yaw_gain_px = max(1.0, float(yaw_gain_px))
        self.update_stride = max(1, int(update_stride))
        self.prev_gray: Optional[np.ndarray] = None
        self.state = CameraMotionState()

    def update(self, frame: np.ndarray, frame_index: int) -> CameraMotionState:
        if frame_index % self.update_stride != 0:
            return self.state

        h, w = frame.shape[:2]
        scale = min(1.0, float(self.max_width) / max(1.0, float(w)))
        if scale < 0.999:
            small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            small = frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray
            return self.state

        mask = np.zeros_like(self.prev_gray)
        y0 = int(mask.shape[0] * self.roi_y_ratio)
        y1 = int(mask.shape[0] * 0.98)
        mask[y0:y1, :] = 255

        p0 = cv2.goodFeaturesToTrack(
            self.prev_gray,
            maxCorners=260,
            qualityLevel=0.01,
            minDistance=8,
            blockSize=7,
            mask=mask,
        )
        if p0 is None or len(p0) < 16:
            self.prev_gray = gray
            self._decay_state()
            return self.state

        p1, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            p0,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        self.prev_gray = gray
        if p1 is None or status is None:
            self._decay_state()
            return self.state

        good = status.reshape(-1).astype(bool)
        before = p0.reshape(-1, 2)[good]
        after = p1.reshape(-1, 2)[good]
        if len(before) < 16:
            self._decay_state()
            return self.state

        flow = (after - before) / max(1e-6, scale)
        mag = np.linalg.norm(flow, axis=1)
        keep = mag < max(18.0, float(w) * 0.08)
        flow = flow[keep]
        if len(flow) < 16:
            self._decay_state()
            return self.state

        dx = float(np.median(flow[:, 0]))
        dy = float(np.median(flow[:, 1]))
        yaw_raw = max(-1.0, min(1.0, -dx / self.yaw_gain_px))
        conf = max(0.0, min(1.0, len(flow) / 180.0))
        keep_old = self.smoothing
        take_new = (1.0 - keep_old) * max(0.25, conf)
        self.state.yaw = self.state.yaw * keep_old + yaw_raw * take_new
        self.state.flow_dx_px = self.state.flow_dx_px * keep_old + dx * (1.0 - keep_old)
        self.state.flow_dy_px = self.state.flow_dy_px * keep_old + dy * (1.0 - keep_old)
        self.state.confidence = self.state.confidence * keep_old + conf * (1.0 - keep_old)
        return self.state

    def _decay_state(self) -> None:
        self.state.yaw *= self.smoothing
        self.state.flow_dx_px *= self.smoothing
        self.state.flow_dy_px *= self.smoothing
        self.state.confidence *= self.smoothing


def draw_corner_box(image: np.ndarray, bbox: BBox, color: Tuple[int, int, int], thickness: int = 2) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1] - 1, x2), min(image.shape[0] - 1, y2)
    line = max(14, int(0.18 * min(x2 - x1, y2 - y1)))
    cv2.rectangle(image, (x1, y1), (x2, y2), (18, 18, 18), thickness + 2, cv2.LINE_AA)
    for sx, sy in [(1, 1), (-1, 1), (1, -1), (-1, -1)]:
        px = x1 if sx > 0 else x2
        py = y1 if sy > 0 else y2
        cv2.line(image, (px, py), (px + sx * line, py), color, thickness, cv2.LINE_AA)
        cv2.line(image, (px, py), (px, py + sy * line), color, thickness, cv2.LINE_AA)


def draw_label(image: np.ndarray, origin: Tuple[int, int], text: str, color: Tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    y = max(th + 8, y)
    x = max(0, min(image.shape[1] - tw - 12, x))
    cv2.rectangle(image, (x, y - th - 8), (x + tw + 12, y + baseline + 3), (22, 24, 28), -1, cv2.LINE_AA)
    cv2.rectangle(image, (x, y - th - 8), (x + 5, y + baseline + 3), color, -1, cv2.LINE_AA)
    cv2.putText(image, text, (x + 8, y - 4), font, scale, (245, 248, 252), thickness, cv2.LINE_AA)


class PersonTrackerNode(Node):
    def __init__(self) -> None:
        super().__init__("yolo_person_tracker")

        self.declare_parameter("video_path", "/home/z/Videos/track_camera_view.mp4")
        self.declare_parameter("model_path", "/home/z/workspace_zyz/yolo_v8_man_detection/ultralytics-main/yolov8n.pt")
        self.declare_parameter("ultralytics_root", "/home/z/workspace_zyz/yolo_v8_man_detection/ultralytics-main")
        self.declare_parameter("backend", "yolo")
        self.declare_parameter("device", "0")
        self.declare_parameter("half", True)
        self.declare_parameter("async_inference", True)
        self.declare_parameter("warmup", True)
        self.declare_parameter("target_class_names", "person,man,pedestrian")
        self.declare_parameter("conf_thres", 0.35)
        self.declare_parameter("iou_thres", 0.45)
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("extract_contours", True)
        self.declare_parameter("contour_max_side", 128)
        self.declare_parameter("contour_max_points", 36)
        self.declare_parameter("contour_interval", 2)
        self.declare_parameter("max_fps", 60.0)
        self.declare_parameter("realtime_playback", True)
        self.declare_parameter("max_frame_skip", 3)
        self.declare_parameter("frame_scale", 1.0)
        self.declare_parameter("loop_video", True)
        self.declare_parameter("show_window", True)
        self.declare_parameter("window_scale", 1.0)
        self.declare_parameter("publish_annotated", True)
        self.declare_parameter("publish_image_stride", 2)
        self.declare_parameter("trail_length", 180)
        self.declare_parameter("ghost_max_count", 18)
        self.declare_parameter("ghost_alpha", 0.36)
        self.declare_parameter("ghost_recent_skip", 8)
        self.declare_parameter("ghost_min_spacing_px", 20.0)
        self.declare_parameter("ghost_path_alpha", 0.0)
        self.declare_parameter("draw_camera_trajectory", True)
        self.declare_parameter("camera_trajectory_alpha", 0.30)
        self.declare_parameter("camera_trajectory_width_ratio", 0.30)
        self.declare_parameter("camera_trajectory_origin_y_ratio", 0.96)
        self.declare_parameter("camera_trajectory_horizon_y_ratio", 0.62)
        self.declare_parameter("camera_trajectory_curve", 0.0)
        self.declare_parameter("camera_trajectory_target_weight", 0.72)
        self.declare_parameter("camera_motion_max_width", 480)
        self.declare_parameter("camera_motion_roi_y_ratio", 0.46)
        self.declare_parameter("camera_motion_smoothing", 0.82)
        self.declare_parameter("camera_motion_yaw_gain_px", 10.0)
        self.declare_parameter("camera_motion_update_stride", 2)
        self.declare_parameter("max_missing_frames", 12)
        self.declare_parameter("match_max_distance_px", 150.0)
        self.declare_parameter("prediction_horizon_sec", 0.0)
        self.declare_parameter("prediction_step_sec", 0.1)
        self.declare_parameter("safety_stop_area_ratio", 0.18)
        self.declare_parameter("max_frames", 0)

        self.video_path = self._param_str("video_path")
        self.model_path = self._param_str("model_path")
        self.ultralytics_root = self._param_str("ultralytics_root")
        self.backend_name = self._param_str("backend").lower()
        self.device = self._param_str("device")
        self.half = self._param_bool("half")
        self.async_inference = self._param_bool("async_inference")
        self.warmup = self._param_bool("warmup")
        self.target_class_names = [x.strip() for x in self._param_str("target_class_names").split(",")]
        self.conf_thres = self._param_float("conf_thres")
        self.iou_thres = self._param_float("iou_thres")
        self.imgsz = self._param_int("imgsz")
        self.extract_contours = self._param_bool("extract_contours")
        self.contour_max_side = self._param_int("contour_max_side")
        self.contour_max_points = self._param_int("contour_max_points")
        self.contour_interval = max(1, self._param_int("contour_interval"))
        self.max_fps = self._param_float("max_fps")
        self.realtime_playback = self._param_bool("realtime_playback")
        self.max_frame_skip = max(0, self._param_int("max_frame_skip"))
        self.frame_scale = self._param_float("frame_scale")
        self.loop_video = self._param_bool("loop_video")
        self.show_window = self._param_bool("show_window")
        self.window_scale = self._param_float("window_scale")
        self.publish_annotated = self._param_bool("publish_annotated")
        self.publish_image_stride = max(1, self._param_int("publish_image_stride"))
        self.trail_length = max(2, self._param_int("trail_length"))
        self.ghost_max_count = max(4, self._param_int("ghost_max_count"))
        self.ghost_alpha = max(0.02, min(0.95, self._param_float("ghost_alpha")))
        self.ghost_recent_skip = max(0, self._param_int("ghost_recent_skip"))
        self.ghost_min_spacing_px = max(0.0, self._param_float("ghost_min_spacing_px"))
        self.ghost_path_alpha = max(0.0, min(0.85, self._param_float("ghost_path_alpha")))
        self.draw_camera_trajectory = self._param_bool("draw_camera_trajectory")
        self.camera_trajectory_alpha = max(0.02, min(0.85, self._param_float("camera_trajectory_alpha")))
        self.camera_trajectory_width_ratio = max(0.10, min(0.90, self._param_float("camera_trajectory_width_ratio")))
        self.camera_trajectory_origin_y_ratio = max(0.55, min(1.05, self._param_float("camera_trajectory_origin_y_ratio")))
        self.camera_trajectory_horizon_y_ratio = max(0.18, min(0.85, self._param_float("camera_trajectory_horizon_y_ratio")))
        self.camera_trajectory_curve = max(-0.55, min(0.55, self._param_float("camera_trajectory_curve")))
        self.camera_trajectory_target_weight = max(0.0, min(1.0, self._param_float("camera_trajectory_target_weight")))
        self.camera_motion_max_width = self._param_int("camera_motion_max_width")
        self.camera_motion_roi_y_ratio = self._param_float("camera_motion_roi_y_ratio")
        self.camera_motion_smoothing = self._param_float("camera_motion_smoothing")
        self.camera_motion_yaw_gain_px = self._param_float("camera_motion_yaw_gain_px")
        self.camera_motion_update_stride = self._param_int("camera_motion_update_stride")
        self.max_missing_frames = max(0, self._param_int("max_missing_frames"))
        self.match_max_distance_px = self._param_float("match_max_distance_px")
        self.prediction_horizon_sec = self._param_float("prediction_horizon_sec")
        self.prediction_step_sec = self._param_float("prediction_step_sec")
        self.safety_stop_area_ratio = self._param_float("safety_stop_area_ratio")
        self.max_frames = self._param_int("max_frames")

        if self.show_window and not os.environ.get("DISPLAY"):
            self.get_logger().warn("DISPLAY is not set; disabling OpenCV window display.")
            self.show_window = False

        self.detector = self._make_detector()
        self.cap = self._open_video(self.video_path)
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count_total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._warmup_detector()
        self.frame_index = 0
        self.playback_start_wall = time.monotonic()
        self.last_process_wall = 0.0
        self.next_track_id = 1
        self.tracks: Dict[int, Track] = {}
        self.tracks_lock = threading.RLock()
        self.latest_frame_lock = threading.Lock()
        self.latest_frame_event = threading.Event()
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_frame_index = 0
        self.latest_stamp_s = 0.0
        self.worker_thread: Optional[threading.Thread] = None
        self.worker_stop = threading.Event()
        self.last_detection_count = 0
        self.last_inference_fps = 0.0
        self.display_fps = 0.0
        self._display_fps_last_wall = time.monotonic()
        self._display_fps_count = 0
        self.camera_motion = CameraMotionState()
        self.camera_motion_estimator = CameraMotionEstimator(
            max_width=self.camera_motion_max_width,
            roi_y_ratio=self.camera_motion_roi_y_ratio,
            smoothing=self.camera_motion_smoothing,
            yaw_gain_px=self.camera_motion_yaw_gain_px,
            update_stride=self.camera_motion_update_stride,
        )
        self.camera_path_endpoint: Optional[Point2] = None
        self.stop_requested = False

        qos = 10
        self.image_pub = self.create_publisher(Image, "~/annotated_image", qos)
        self.detection_pub = self.create_publisher(String, "~/detections", qos)
        self.primary_pub = self.create_publisher(PointStamped, "~/primary_person", qos)
        self.stop_pub = self.create_publisher(Bool, "~/person_safety_stop", qos)
        self.status_pub = self.create_publisher(String, "~/status", qos)

        if self.async_inference:
            self.worker_thread = threading.Thread(target=self._inference_worker, name="yolo_inference_worker", daemon=False)
            self.worker_thread.start()

        self.timer = self.create_timer(0.001, self._process_once)
        self.get_logger().info(
            "Person tracker started: video=%s, model=%s, backend=%s, device=%s, half=%s, async=%s, fps=%.1f, frames=%d"
            % (
                self.video_path,
                self.model_path,
                self.backend_name,
                getattr(self.detector, "device", "cpu"),
                getattr(self.detector, "half", False),
                self.async_inference,
                self.video_fps,
                self.frame_count_total,
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

    def _make_detector(self):
        if self.backend_name == "hog":
            self.get_logger().warn("Using OpenCV HOG fallback backend. This is for debugging only, not the requested YOLO model.")
            return HogDetector(self.conf_thres)

        try:
            detector = YoloDetector(
                model_path=self.model_path,
                ultralytics_root=self.ultralytics_root,
                target_class_names=self.target_class_names,
                conf_thres=self.conf_thres,
                iou_thres=self.iou_thres,
                imgsz=self.imgsz,
                device=self.device,
                half=self.half,
                extract_contours=self.extract_contours,
                contour_max_side=self.contour_max_side,
                contour_max_points=self.contour_max_points,
                contour_interval=self.contour_interval,
            )
            return detector
        except Exception as exc:
            if self.backend_name == "auto":
                self.get_logger().error(f"YOLO backend failed, falling back to HOG: {exc}")
                return HogDetector(self.conf_thres)
            self.get_logger().fatal(f"YOLO backend failed: {exc}")
            raise

    def _open_video(self, path: str):
        cap = cv2.VideoCapture(str(Path(path).expanduser()))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {path}")
        return cap

    def _warmup_detector(self) -> None:
        if not self.warmup or self.backend_name == "hog":
            return
        ok, frame = self.cap.read()
        if not ok:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return
        if 0.05 < self.frame_scale < 0.999:
            frame = cv2.resize(frame, None, fx=self.frame_scale, fy=self.frame_scale, interpolation=cv2.INTER_AREA)
        start = time.perf_counter()
        try:
            _ = self.detector.detect(frame)
            warmup_dt = time.perf_counter() - start
            self.get_logger().info(f"YOLO GPU warmup finished in {warmup_dt:.2f}s")
        finally:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def _process_once(self) -> None:
        now_wall = time.monotonic()
        if self.realtime_playback and self.video_fps > 1e-3:
            target_fps = self.video_fps
            if self.max_fps > 0.0:
                target_fps = min(target_fps, self.max_fps)
        else:
            target_fps = self.max_fps if self.max_fps > 0.0 else self.video_fps
        if target_fps > 0.0 and (now_wall - self.last_process_wall) < (1.0 / target_fps):
            return
        self.last_process_wall = now_wall

        if self.realtime_playback and self.video_fps > 1e-3 and self.max_frame_skip > 0:
            expected_next = int((now_wall - self.playback_start_wall) * self.video_fps)
            skip = min(self.max_frame_skip, max(0, expected_next - self.frame_index))
            if skip > 1:
                for _ in range(skip):
                    ok_skip, _ = self.cap.read()
                    if not ok_skip:
                        break
                    self.frame_index += 1

        ok, frame = self.cap.read()
        if not ok:
            if self.loop_video:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.frame_index = 0
                self.playback_start_wall = time.monotonic()
                ok, frame = self.cap.read()
            if not ok:
                self.get_logger().info("Video stream ended.")
                self._request_stop()
                return

        self.frame_index += 1
        if self.max_frames > 0 and self.frame_index > self.max_frames:
            self.get_logger().info(f"Reached max_frames={self.max_frames}; stopping.")
            self._request_stop()
            return

        if 0.05 < self.frame_scale < 0.999:
            frame = cv2.resize(frame, None, fx=self.frame_scale, fy=self.frame_scale, interpolation=cv2.INTER_AREA)

        if self.draw_camera_trajectory:
            self.camera_motion = self.camera_motion_estimator.update(frame, self.frame_index)

        stamp_msg = self.get_clock().now().to_msg()
        stamp_s = float(stamp_msg.sec) + float(stamp_msg.nanosec) * 1e-9
        detections: Sequence[Detection] = []
        if self.async_inference:
            self._submit_latest_frame(frame, self.frame_index, stamp_s)
        else:
            start = time.perf_counter()
            detections = self.detector.detect(frame)
            inference_dt = max(1e-6, time.perf_counter() - start)
            self.last_inference_fps = 1.0 / inference_dt
            self.last_detection_count = len(detections)
            with self.tracks_lock:
                self._update_tracks(detections, stamp_s)

        self._update_display_fps()
        needs_annotated = (
            self.show_window
            or (
                self.publish_annotated
                and (self.frame_index % self.publish_image_stride == 0)
                and self.image_pub.get_subscription_count() > 0
            )
        )
        annotated = self._draw(frame.copy(), stamp_s) if needs_annotated else frame
        self._publish(stamp_msg, annotated, detections)
        self._show(annotated)

    def _submit_latest_frame(self, frame: np.ndarray, frame_index: int, stamp_s: float) -> None:
        with self.latest_frame_lock:
            self.latest_frame = frame
            self.latest_frame_index = frame_index
            self.latest_stamp_s = stamp_s
        self.latest_frame_event.set()

    def _inference_worker(self) -> None:
        processed_frame_index = -1
        while not self.worker_stop.is_set():
            self.latest_frame_event.wait(0.05)
            if self.worker_stop.is_set():
                break
            with self.latest_frame_lock:
                if self.latest_frame is None or self.latest_frame_index == processed_frame_index:
                    self.latest_frame_event.clear()
                    continue
                frame = self.latest_frame
                frame_index = self.latest_frame_index
                stamp_s = self.latest_stamp_s
                self.latest_frame_event.clear()

            start = time.perf_counter()
            detections = self.detector.detect(frame)
            inference_dt = max(1e-6, time.perf_counter() - start)
            with self.tracks_lock:
                self.last_inference_fps = 1.0 / inference_dt
                self.last_detection_count = len(detections)
                self._update_tracks(detections, stamp_s)
            processed_frame_index = frame_index

    def _update_display_fps(self) -> None:
        self._display_fps_count += 1
        now = time.monotonic()
        elapsed = now - self._display_fps_last_wall
        if elapsed >= 0.5:
            self.display_fps = self._display_fps_count / elapsed
            self._display_fps_count = 0
            self._display_fps_last_wall = now

    def _update_tracks(self, detections: Sequence[Detection], stamp_s: float) -> None:
        unmatched_track_ids = set(self.tracks.keys())
        unmatched_detection_ids = set(range(len(detections)))
        matches: List[Tuple[int, int]] = []

        scored_pairs: List[Tuple[float, int, int]] = []
        for track_id, track in self.tracks.items():
            for det_idx, det in enumerate(detections):
                dist = center_distance(track.predicted_foot(stamp_s), det.foot_point)
                iou = bbox_iou(track.bbox, det.bbox)
                if dist > self.match_max_distance_px and iou < 0.05:
                    continue
                score = iou * 1.25 + max(0.0, 1.0 - dist / max(1.0, self.match_max_distance_px)) * 1.6
                scored_pairs.append((score, track_id, det_idx))

        for _, track_id, det_idx in sorted(scored_pairs, reverse=True):
            if track_id not in unmatched_track_ids or det_idx not in unmatched_detection_ids:
                continue
            matches.append((track_id, det_idx))
            unmatched_track_ids.remove(track_id)
            unmatched_detection_ids.remove(det_idx)

        for track_id, det_idx in matches:
            self.tracks[track_id].update(detections[det_idx], stamp_s, self.trail_length)

        for det_idx in unmatched_detection_ids:
            det = detections[det_idx]
            track = Track(
                track_id=self.next_track_id,
                bbox=det.bbox,
                confidence=det.confidence,
                class_name=det.class_name,
                history=deque(maxlen=self.trail_length),
            )
            track.update(det, stamp_s, self.trail_length)
            self.tracks[self.next_track_id] = track
            self.next_track_id += 1

        for track_id in list(unmatched_track_ids):
            self.tracks[track_id].misses += 1
            if self.tracks[track_id].misses > self.max_missing_frames:
                del self.tracks[track_id]

    def _draw(self, image: np.ndarray, stamp_s: float) -> np.ndarray:
        height, width = image.shape[:2]
        header_h = min(76, height)
        header = image[:header_h, :]
        overlay = header.copy()
        cv2.rectangle(overlay, (0, 0), (width, header_h), (18, 24, 30), -1)
        image[:header_h, :] = cv2.addWeighted(overlay, 0.55, header, 0.45, 0)

        title = f"YOLO Person Tracking  |  frame {self.frame_index}"
        with self.tracks_lock:
            tracks = list(sorted(self.tracks.values(), key=lambda t: t.track_id))
            track_count = len(self.tracks)
            detection_count = self.last_detection_count
            inference_fps = self.last_inference_fps
        metrics = (
            f"video {self.display_fps:.1f} FPS   infer {inference_fps:.1f} FPS   "
            f"persons {detection_count}   tracks {track_count}"
        )
        cv2.putText(image, title, (22, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (240, 246, 255), 2, cv2.LINE_AA)
        cv2.putText(image, metrics, (22, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (118, 226, 255), 1, cv2.LINE_AA)

        if self.draw_camera_trajectory:
            target_track = self._camera_trajectory_target(tracks, width, height)
            self._draw_camera_trajectory(image, width, height, target_track)

        for track in tracks:
            if track.misses > 0:
                continue
            color = self._track_color(track.track_id)
            self._draw_track_history(image, track, color)
            draw_corner_box(image, track.bbox, color, thickness=3)
            x1, y1, _, _ = track.bbox
            draw_label(
                image,
                (int(x1), int(y1) - 6),
                f"{track.class_name} {track.confidence:.2f}",
                color,
            )
            fx, fy = clamp_point(track.predicted_foot(stamp_s), width, height)
            cv2.ellipse(image, (fx, fy), (16, 6), 0, 0, 360, (10, 14, 18), -1, cv2.LINE_AA)
            cv2.ellipse(image, (fx, fy), (16, 6), 0, 0, 360, color, 2, cv2.LINE_AA)
            cv2.circle(image, (fx, fy), 4, (255, 255, 255), -1, cv2.LINE_AA)

        return image

    @staticmethod
    def _track_color(track_id: int) -> Tuple[int, int, int]:
        palette = [
            (232, 176, 255),
            (214, 150, 255),
            (244, 188, 255),
            (198, 134, 255),
            (225, 170, 246),
            (238, 198, 255),
        ]
        return palette[(track_id - 1) % len(palette)]

    @staticmethod
    def _lavender_color(alpha: float, shift: float = 0.0) -> Tuple[int, int, int]:
        alpha = max(0.0, min(1.0, alpha))
        stops = [
            (178, 98, 238),
            (210, 132, 255),
            (242, 180, 255),
            (255, 216, 250),
        ]
        scaled = min(2.999, (alpha + shift) % 1.0 * 3.0)
        idx = int(scaled)
        frac = scaled - idx
        c0 = stops[idx]
        c1 = stops[min(idx + 1, len(stops) - 1)]
        return tuple(int(c0[i] * (1.0 - frac) + c1[i] * frac) for i in range(3))

    def _draw_track_history(self, image: np.ndarray, track: Track, color: Tuple[int, int, int]) -> None:
        if len(track.bbox_history) < 2:
            return
        height, width = image.shape[:2]
        history = list(track.bbox_history)
        skip = min(self.ghost_recent_skip, max(1, len(history) - 1))
        history = history[:-skip] if skip > 0 else history
        if len(history) < 2:
            return

        sampled = self._sample_ghost_history(history)
        if len(sampled) < 2:
            return

        silhouettes: List[np.ndarray] = []
        all_points: List[Tuple[int, int]] = []
        for item in sampled:
            contour = Track._stylized_person_contour(*item[:4])
            pts = np.array([clamp_point(point, width, height) for point in contour], dtype=np.int32)
            silhouettes.append(pts)
            all_points.extend((int(x), int(y)) for x, y in pts)

        path_px: List[Tuple[int, int]] = []
        if self.ghost_path_alpha > 1e-3:
            dense_step = max(1, len(history) // 90)
            path_points = [self._history_foot_from_bbox(item[:4]) for item in history[::dense_step]]
            path_px = [clamp_point(point, width, height) for point in path_points]
            all_points.extend(path_px)
        if not all_points:
            return

        all_arr = np.array(all_points, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(all_arr)
        pad = max(6, int(max(w, h) * 0.04))
        rx1 = max(0, x - pad)
        ry1 = max(0, y - pad)
        rx2 = min(width, x + w + pad)
        ry2 = min(height, y + h + pad)
        if rx2 <= rx1 or ry2 <= ry1:
            return

        roi = image[ry1:ry2, rx1:rx2]
        overlay = roi.copy()

        def to_local(points: np.ndarray) -> np.ndarray:
            local = points.copy()
            local[:, 0] -= rx1
            local[:, 1] -= ry1
            return local

        if len(path_px) >= 2 and self.ghost_path_alpha > 1e-3:
            path_arr = np.array(path_px, dtype=np.int32)
            path_arr[:, 0] -= rx1
            path_arr[:, 1] -= ry1
            path_color = self._lavender_color(0.82, track.track_id * 0.035)
            dark_path = tuple(int(v * 0.34) for v in path_color)
            bright_path = tuple(int(v * 0.70) for v in path_color)
            cv2.polylines(overlay, [path_arr], False, dark_path, 5, cv2.LINE_AA)
            cv2.polylines(overlay, [path_arr], False, bright_path, 2, cv2.LINE_AA)

        for idx, pts_abs in enumerate(silhouettes):
            age = idx / max(1, len(silhouettes) - 1)
            if age < 0.04:
                continue
            pts = to_local(pts_abs)
            ghost_color = self._lavender_color(age, track.track_id * 0.035)
            strength = 0.34 + 0.52 * age
            glow_color = tuple(int(v * 0.22 * strength) for v in ghost_color)
            line_color = tuple(min(255, int(v * strength)) for v in ghost_color)
            cv2.polylines(overlay, [pts], True, glow_color, 3, cv2.LINE_AA)
            cv2.polylines(overlay, [pts], True, line_color, 1 if age < 0.74 else 2, cv2.LINE_AA)

            if self.ghost_path_alpha > 1e-3:
                foot = self._history_foot_from_bbox(sampled[idx][:4])
                fx, fy = clamp_point(foot, width, height)
                fx -= rx1
                fy -= ry1
                dot_radius = 1 if age < 0.65 else 2
                cv2.circle(overlay, (fx, fy), dot_radius, line_color, -1, cv2.LINE_AA)

        blend_alpha = max(self.ghost_alpha, self.ghost_path_alpha)
        cv2.addWeighted(overlay, blend_alpha, roi, 1.0 - blend_alpha, 0.0, roi)

    def _sample_ghost_history(self, history: Sequence[Tuple[float, float, float, float, float]]) -> List[Tuple[float, float, float, float, float]]:
        selected: List[Tuple[float, float, float, float, float]] = []
        last_foot: Optional[Point2] = None
        for item in reversed(history):
            foot = self._history_foot_from_bbox(item[:4])
            if last_foot is None or center_distance(foot, last_foot) >= self.ghost_min_spacing_px:
                selected.append(item)
                last_foot = foot
            if len(selected) >= self.ghost_max_count:
                break

        if len(selected) < min(6, len(history)):
            step = max(1, len(history) // self.ghost_max_count)
            return list(history[::step][-self.ghost_max_count :])
        selected.reverse()
        return selected

    @staticmethod
    def _history_foot_from_bbox(bbox: Sequence[float]) -> Point2:
        x1, _, x2, y2 = [float(v) for v in bbox]
        return ((x1 + x2) * 0.5, y2)

    @staticmethod
    def _mix_color(c0: Tuple[int, int, int], c1: Tuple[int, int, int], ratio: float) -> Tuple[int, int, int]:
        ratio = max(0.0, min(1.0, ratio))
        return tuple(int(c0[i] * (1.0 - ratio) + c1[i] * ratio) for i in range(3))

    @classmethod
    def _camera_trajectory_color(cls, t: float) -> Tuple[int, int, int]:
        # BGR: near cool lavender -> violet -> signal red.
        t = max(0.0, min(1.0, t))
        near = (255, 172, 182)
        mid = (246, 82, 138)
        far = (48, 58, 224)
        if t < 0.56:
            return cls._mix_color(near, mid, t / 0.56)
        return cls._mix_color(mid, far, (t - 0.56) / 0.44)

    @staticmethod
    def _scale_color(color: Tuple[int, int, int], scale: float) -> Tuple[int, int, int]:
        return tuple(max(0, min(255, int(v * scale))) for v in color)

    def _draw_contour_ghost(
        self,
        image: np.ndarray,
        contour: List[Point2],
        track_id: int,
        alpha: float,
        width: int,
        height: int,
    ) -> None:
        if len(contour) < 3:
            return
        pts_abs = np.array([clamp_point(p, width, height) for p in contour], dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts_abs)
        pad = max(3, int(max(w, h) * 0.03))
        rx1 = max(0, x - pad)
        ry1 = max(0, y - pad)
        rx2 = min(width, x + w + pad)
        ry2 = min(height, y + h + pad)
        if rx2 <= rx1 or ry2 <= ry1:
            return

        roi = image[ry1:ry2, rx1:rx2]
        pts = pts_abs.copy()
        pts[:, 0] -= rx1
        pts[:, 1] -= ry1
        color = self._lavender_color(alpha, track_id * 0.035)
        visibility = 0.48 + 0.36 * self.ghost_alpha
        strength = min(1.0, (0.54 + 0.46 * alpha) * visibility)
        glow_color = tuple(max(0, int(v * 0.42 * strength)) for v in color)
        line_color = tuple(min(255, int(v * strength)) for v in color)
        cv2.polylines(roi, [pts], True, glow_color, 4, cv2.LINE_AA)
        cv2.polylines(roi, [pts], True, line_color, 1 if alpha < 0.78 else 2, cv2.LINE_AA)

    def _camera_trajectory_target(self, tracks: Sequence[Track], width: int, height: int) -> Optional[Point2]:
        active = [track for track in tracks if track.misses == 0]
        if not active:
            return None
        origin_y = float(height) * self.camera_trajectory_origin_y_ratio
        candidates = []
        for track in active:
            x1, y1, x2, y2 = track.bbox
            area_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / max(1.0, float(width * height))
            if area_ratio > 0.23 or y2 > origin_y - 42.0:
                continue
            foot_x = (x1 + x2) * 0.5
            foot_y = y2
            score = track.confidence * max(0.1, 1.0 - area_ratio) + max(0.0, 1.0 - abs(foot_x - width * 0.5) / max(1.0, width * 0.5)) * 0.25
            candidates.append((score, (foot_x, foot_y)))
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[0])[1]

    def _draw_camera_trajectory(self, image: np.ndarray, width: int, height: int, target_point: Optional[Point2]) -> None:
        y0 = float(height) * self.camera_trajectory_origin_y_ratio
        ground_horizon_y = float(height) * self.camera_trajectory_horizon_y_ratio
        y0 = min(float(height - 1), max(float(height) * 0.58, y0))
        ground_horizon_y = min(y0 - 92.0, max(float(height) * 0.52, ground_horizon_y))
        if y0 <= ground_horizon_y + 16.0:
            return

        x0 = float(width) * 0.50
        motion_curve = self.camera_motion.yaw * max(0.25, self.camera_motion.confidence)
        curve = max(-0.85, min(0.85, self.camera_trajectory_curve + motion_curve))
        x_end = x0 + curve * float(width) * 0.26
        y_end = ground_horizon_y
        if target_point is not None:
            tx = max(float(width) * 0.08, min(float(width) * 0.92, target_point[0]))
            ty = max(ground_horizon_y + 28.0, min(y0 - 82.0, target_point[1]))
            weight = self.camera_trajectory_target_weight
            x_end = x_end * (1.0 - weight) + tx * weight
            y_end = y_end * (1.0 - weight) + ty * weight

        if self.camera_path_endpoint is None:
            self.camera_path_endpoint = (x_end, y_end)
        else:
            keep = 0.82
            self.camera_path_endpoint = (
                self.camera_path_endpoint[0] * keep + x_end * (1.0 - keep),
                self.camera_path_endpoint[1] * keep + y_end * (1.0 - keep),
            )
        x_end, y_end = self.camera_path_endpoint
        y_end = max(ground_horizon_y + 18.0, min(y0 - 82.0, y_end))

        control_x = (x0 + x_end) * 0.5 + curve * float(width) * 0.12
        control_y = y0 + (y_end - y0) * 0.58
        bottom_half_width = float(width) * self.camera_trajectory_width_ratio * 0.50
        top_half_width = max(12.0, float(width) * 0.022)
        centers: List[Tuple[int, int]] = []
        left: List[Tuple[int, int]] = []
        right: List[Tuple[int, int]] = []

        center_f: List[Point2] = []
        for i in range(28):
            t = i / 27.0
            omt = 1.0 - t
            x = omt * omt * x0 + 2.0 * omt * t * control_x + t * t * x_end
            y = omt * omt * y0 + 2.0 * omt * t * control_y + t * t * y_end
            y = max(ground_horizon_y + 10.0 + 8.0 * (1.0 - t), min(y0, y))
            center_f.append((x, y))

        for i, (x, y) in enumerate(center_f):
            t = i / max(1, len(center_f) - 1)
            half_w = bottom_half_width * ((1.0 - t) ** 1.28) + top_half_width * (t ** 0.9)
            centers.append((int(round(x)), int(round(y))))
            left.append((int(round(x - half_w)), int(round(y))))
            right.append((int(round(x + half_w)), int(round(y))))

        all_pts = np.array(left + right + centers, dtype=np.int32)
        rx1 = max(0, int(np.min(all_pts[:, 0])) - 18)
        ry1 = max(0, int(np.min(all_pts[:, 1])) - 18)
        rx2 = min(width, int(np.max(all_pts[:, 0])) + 18)
        ry2 = min(height, int(np.max(all_pts[:, 1])) + 18)
        if rx2 <= rx1 or ry2 <= ry1:
            return

        roi = image[ry1:ry2, rx1:rx2]
        overlay = roi.copy()

        def local_points(points: List[Tuple[int, int]]) -> np.ndarray:
            arr = np.array(points, dtype=np.int32)
            arr[:, 0] -= rx1
            arr[:, 1] -= ry1
            return arr

        left_arr = local_points(left)
        right_arr = local_points(right)
        center_arr = local_points(centers)
        segment_count = max(1, len(centers) - 1)
        for i in range(segment_count):
            t_mid = (i + 0.5) / segment_count
            fill_color = self._scale_color(self._camera_trajectory_color(t_mid), 0.78)
            quad = np.array([left_arr[i], left_arr[i + 1], right_arr[i + 1], right_arr[i]], dtype=np.int32)
            cv2.fillConvexPoly(overlay, quad, fill_color, cv2.LINE_AA)
        cv2.addWeighted(overlay, self.camera_trajectory_alpha * 0.82, roi, 1.0 - self.camera_trajectory_alpha * 0.82, 0.0, roi)

        for i in range(segment_count):
            t_mid = (i + 0.5) / segment_count
            color = self._camera_trajectory_color(t_mid)
            glow = self._scale_color(color, 0.46)
            center_glow = self._scale_color(color, 0.62)
            cv2.line(roi, tuple(left_arr[i]), tuple(left_arr[i + 1]), glow, 6, cv2.LINE_AA)
            cv2.line(roi, tuple(right_arr[i]), tuple(right_arr[i + 1]), glow, 6, cv2.LINE_AA)
            cv2.line(roi, tuple(left_arr[i]), tuple(left_arr[i + 1]), color, 2, cv2.LINE_AA)
            cv2.line(roi, tuple(right_arr[i]), tuple(right_arr[i + 1]), color, 2, cv2.LINE_AA)
            cv2.line(roi, tuple(center_arr[i]), tuple(center_arr[i + 1]), center_glow, 5, cv2.LINE_AA)
            cv2.line(roi, tuple(center_arr[i]), tuple(center_arr[i + 1]), color, 2, cv2.LINE_AA)

        phase = (self.frame_index % 90) / 90.0
        for k in range(8):
            t = (phase + k / 8.0) % 1.0
            idx = min(len(centers) - 1, max(0, int(t * (len(centers) - 1))))
            px, py = centers[idx]
            px -= rx1
            py -= ry1
            radius = max(2, int(7 * (1.0 - t) + 2))
            dot_color = self._camera_trajectory_color(t)
            cv2.circle(roi, (px, py), radius + 2, self._scale_color(dot_color, 0.36), -1, cv2.LINE_AA)
            cv2.circle(roi, (px, py), radius, dot_color, -1, cv2.LINE_AA)

        for i in range(4, len(left), 6):
            t = i / max(1, len(left) - 1)
            grid_color = self._scale_color(self._camera_trajectory_color(t), 0.58)
            cv2.line(
                roi,
                (left[i][0] - rx1, left[i][1] - ry1),
                (right[i][0] - rx1, right[i][1] - ry1),
                grid_color,
                1,
                cv2.LINE_AA,
            )

    def _publish(self, stamp_msg, annotated: np.ndarray, detections: Sequence[Detection]) -> None:
        if (
            self.publish_annotated
            and (self.frame_index % self.publish_image_stride == 0)
            and self.image_pub.get_subscription_count() > 0
        ):
            msg = Image()
            msg.header.stamp = stamp_msg
            msg.header.frame_id = "track_camera_view"
            msg.height = int(annotated.shape[0])
            msg.width = int(annotated.shape[1])
            msg.encoding = "bgr8"
            msg.is_bigendian = False
            msg.step = int(annotated.shape[1] * 3)
            msg.data = annotated.tobytes()
            self.image_pub.publish(msg)

        stamp_s = float(stamp_msg.sec) + float(stamp_msg.nanosec) * 1e-9
        payload = self._make_detection_payload(stamp_msg, annotated.shape[:2], stamp_s)
        self.detection_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
        self.status_pub.publish(String(data=json.dumps({"frame": self.frame_index, "tracks": len(self.tracks)})))

        primary = self._primary_track()
        stop = False
        if primary is not None:
            point = PointStamped()
            point.header.stamp = stamp_msg
            point.header.frame_id = "track_camera_view"
            foot = primary.predicted_foot(stamp_s)
            point.point.x = float(foot[0])
            point.point.y = float(foot[1])
            point.point.z = float(primary.confidence)
            self.primary_pub.publish(point)
            frame_area = float(annotated.shape[0] * annotated.shape[1])
            x1, y1, x2, y2 = primary.bbox
            area_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / max(1.0, frame_area)
            stop = area_ratio >= self.safety_stop_area_ratio
        self.stop_pub.publish(Bool(data=stop))

    def _make_detection_payload(self, stamp_msg, shape_hw: Tuple[int, int], stamp_s: float) -> Dict:
        height, width = shape_hw
        tracks = []
        with self.tracks_lock:
            for track in sorted(self.tracks.values(), key=lambda t: t.track_id):
                if track.misses > 0:
                    continue
                x1, y1, x2, y2 = track.bbox
                vx, vy = track.velocity_px_s()
                foot = track.predicted_foot(stamp_s)
                sampled_history = list(track.history)[:: max(1, len(track.history) // 80)]
                tracks.append(
                    {
                        "id": track.track_id,
                        "class": track.class_name,
                        "confidence": round(track.confidence, 4),
                        "bbox_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                        "center_px": [round(track.center[0], 2), round(track.center[1], 2)],
                        "foot_px": [round(foot[0], 2), round(foot[1], 2)],
                        "velocity_px_s": [round(vx, 2), round(vy, 2)],
                        "trajectory_px": [[round(x, 2), round(y, 2)] for x, y, _ in sampled_history],
                    }
                )
        return {
            "stamp": {"sec": int(stamp_msg.sec), "nanosec": int(stamp_msg.nanosec)},
            "frame_index": self.frame_index,
            "image_size": {"width": width, "height": height},
            "inference_fps": round(self.last_inference_fps, 2),
            "tracks": tracks,
        }

    def _primary_track(self) -> Optional[Track]:
        with self.tracks_lock:
            active = [track for track in self.tracks.values() if track.misses == 0]
        if not active:
            return None
        return max(active, key=lambda t: (t.bbox[2] - t.bbox[0]) * (t.bbox[3] - t.bbox[1]) * max(0.01, t.confidence))

    def _show(self, image: np.ndarray) -> None:
        if not self.show_window:
            return
        display = image
        if self.window_scale > 0.0 and abs(self.window_scale - 1.0) > 1e-3:
            display = cv2.resize(image, None, fx=self.window_scale, fy=self.window_scale, interpolation=cv2.INTER_AREA)
        cv2.imshow("YOLO Person Tracker", display)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            self.get_logger().info("Window close requested.")
            self._request_stop()

    def _request_stop(self) -> None:
        self.stop_requested = True
        self.worker_stop.set()
        self.latest_frame_event.set()

    def destroy_node(self) -> bool:
        try:
            self.worker_stop.set()
            self.latest_frame_event.set()
            if self.worker_thread is not None and self.worker_thread.is_alive():
                self.worker_thread.join(timeout=10.0)
            if hasattr(self, "cap") and self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()
        finally:
            return super().destroy_node()


def main(args: Optional[Iterable[str]] = None) -> None:
    rclpy.init(args=args)
    node: Optional[PersonTrackerNode] = None
    try:
        node = PersonTrackerNode()
        while rclpy.ok() and not node.stop_requested:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
