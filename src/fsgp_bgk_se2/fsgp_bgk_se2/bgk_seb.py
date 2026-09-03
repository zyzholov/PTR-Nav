from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class BgkSebParams:
    resolution: float
    yaw_bins: int
    prefer_gpu: bool = True
    gpu_backend: str = "auto"
    bgk_sigma_m: float = 0.45
    bgk_radius_m: float = 1.20
    bgk_min_weight: float = 1.0e-3
    variance_free: float = 0.01
    variance_hard: float = 0.20
    slope_free_rad: float = 0.10
    max_slope_rad: float = 0.55
    step_free_m: float = 0.04
    max_step_m: float = 0.18
    max_pitch_rad: float = 0.55
    max_roll_rad: float = 0.45
    pitch_free_rad: float = 0.08
    roll_free_rad: float = 0.08
    min_support_ratio: float = 0.70
    footprint_front: float = 0.42
    footprint_back: float = 0.42
    footprint_half_width: float = 0.313
    footprint_margin: float = 0.10
    footprint_samples_x: int = 5
    footprint_samples_y: int = 5
    fusion_mode: str = "max"
    baseline_weight: float = 0.40
    seb_weight: float = 0.45
    uncertainty_weight: float = 0.15
    unknown_risk: float = 0.85


@dataclass
class BgkSebResult:
    backend: str
    bgk_mean: np.ndarray
    bgk_var: np.ndarray
    bgk_traversability: np.ndarray
    final_risk_slices: List[np.ndarray]
    seb_risk_slices: List[np.ndarray]
    seb_pitch_risk_slices: List[np.ndarray]
    seb_roll_risk_slices: List[np.ndarray]
    seb_step_risk_slices: List[np.ndarray]
    seb_support_slices: List[np.ndarray]
    baseline_risk: np.ndarray


class BgkSebEstimator:
    def __init__(self, params: BgkSebParams):
        self.params = params
        self._xp = np
        self.backend_name = "cpu"
        if params.prefer_gpu:
            self._try_select_gpu(params.gpu_backend)

    def _try_select_gpu(self, requested: str) -> None:
        if requested not in ("auto", "cupy"):
            self.backend_name = "cpu"
            self._xp = np
            return
        try:
            import cupy as cp  # type: ignore

            if cp.cuda.runtime.getDeviceCount() <= 0:
                return
            self._xp = cp
            self.backend_name = "cupy"
        except Exception:
            self._xp = np
            self.backend_name = "cpu"

    def compute(
        self,
        elevation: np.ndarray,
        variance: np.ndarray,
        valid: np.ndarray,
        baseline_risk_slices: Sequence[np.ndarray],
    ) -> BgkSebResult:
        xp = self._xp
        p = self.params
        elevation_d = xp.asarray(np.nan_to_num(elevation, nan=0.0), dtype=xp.float32)
        variance_d = xp.asarray(np.nan_to_num(variance, nan=p.variance_hard), dtype=xp.float32)
        valid_d = xp.asarray(valid.astype(np.float32), dtype=xp.float32)
        baseline_d = [
            xp.asarray(np.nan_to_num(r, nan=p.unknown_risk), dtype=xp.float32)
            for r in baseline_risk_slices
        ]

        bgk_mean, bgk_var = self._bgk_smooth(elevation_d, variance_d, valid_d)
        bgk_risk = self._bgk_risk(bgk_mean, bgk_var, valid_d)
        bgk_trav = 1.0 - bgk_risk

        (
            final_slices,
            seb_slices,
            pitch_slices,
            roll_slices,
            step_slices,
            support_slices,
        ) = self._seb_yaw_sweep(bgk_mean, bgk_var, valid_d, baseline_d)

        return BgkSebResult(
            backend=self.backend_name,
            bgk_mean=self._to_numpy(bgk_mean),
            bgk_var=self._to_numpy(bgk_var),
            bgk_traversability=self._to_numpy(bgk_trav),
            final_risk_slices=[self._to_numpy(x) for x in final_slices],
            seb_risk_slices=[self._to_numpy(x) for x in seb_slices],
            seb_pitch_risk_slices=[self._to_numpy(x) for x in pitch_slices],
            seb_roll_risk_slices=[self._to_numpy(x) for x in roll_slices],
            seb_step_risk_slices=[self._to_numpy(x) for x in step_slices],
            seb_support_slices=[self._to_numpy(x) for x in support_slices],
            baseline_risk=self._to_numpy(baseline_d[0]) if baseline_d else np.zeros_like(elevation),
        )

    def _to_numpy(self, arr):
        if self.backend_name == "cupy":
            return self._xp.asnumpy(arr).astype(np.float32, copy=False)
        return np.asarray(arr, dtype=np.float32)

    def _shift(self, arr, row_shift: int, col_shift: int, fill_value: float):
        xp = self._xp
        rows, cols = arr.shape
        out = xp.full(arr.shape, fill_value, dtype=arr.dtype)

        dst_r0 = max(0, -row_shift)
        dst_r1 = min(rows, rows - row_shift)
        dst_c0 = max(0, -col_shift)
        dst_c1 = min(cols, cols - col_shift)
        if dst_r0 >= dst_r1 or dst_c0 >= dst_c1:
            return out

        src_r0 = dst_r0 + row_shift
        src_r1 = dst_r1 + row_shift
        src_c0 = dst_c0 + col_shift
        src_c1 = dst_c1 + col_shift
        out[dst_r0:dst_r1, dst_c0:dst_c1] = arr[src_r0:src_r1, src_c0:src_c1]
        return out

    @staticmethod
    def _soft_score(xp, value, free_value: float, hard_value: float):
        denom = max(1.0e-6, hard_value - free_value)
        return xp.clip((value - free_value) / denom, 0.0, 1.0)

    def _kernel_offsets(self) -> List[Tuple[int, int, float]]:
        p = self.params
        radius_cells = max(1, int(math.ceil(p.bgk_radius_m / max(p.resolution, 1.0e-3))))
        sigma_cells = max(0.5, p.bgk_sigma_m / max(p.resolution, 1.0e-3))
        offsets: List[Tuple[int, int, float]] = []
        for dr in range(-radius_cells, radius_cells + 1):
            for dc in range(-radius_cells, radius_cells + 1):
                dist2 = float(dr * dr + dc * dc)
                if dist2 > radius_cells * radius_cells:
                    continue
                weight = math.exp(-0.5 * dist2 / (sigma_cells * sigma_cells))
                offsets.append((dr, dc, weight))
        return offsets

    def _bgk_smooth(self, elevation, variance, valid):
        xp = self._xp
        p = self.params
        weight_sum = xp.zeros_like(elevation, dtype=xp.float32)
        height_sum = xp.zeros_like(elevation, dtype=xp.float32)
        height2_sum = xp.zeros_like(elevation, dtype=xp.float32)
        var_sum = xp.zeros_like(elevation, dtype=xp.float32)

        for dr, dc, weight in self._kernel_offsets():
            v = self._shift(valid, dr, dc, 0.0)
            h = self._shift(elevation, dr, dc, 0.0)
            var = self._shift(variance, dr, dc, p.variance_hard)
            wv = v * xp.float32(weight)
            weight_sum += wv
            height_sum += wv * h
            height2_sum += wv * h * h
            var_sum += wv * var

        safe_weight = xp.maximum(weight_sum, xp.float32(p.bgk_min_weight))
        mean = height_sum / safe_weight
        second = height2_sum / safe_weight
        local_var = xp.maximum(second - mean * mean, 0.0)
        obs_var = var_sum / safe_weight
        bgk_var = local_var + obs_var

        enough = weight_sum > xp.float32(p.bgk_min_weight)
        mean = xp.where(enough, mean, elevation)
        bgk_var = xp.where(enough, bgk_var, xp.float32(p.variance_hard))
        return mean, bgk_var

    def _bgk_risk(self, mean, bgk_var, valid):
        xp = self._xp
        p = self.params
        res = max(p.resolution, 1.0e-3)
        left = self._shift(mean, 0, -1, xp.nan)
        right = self._shift(mean, 0, 1, xp.nan)
        down = self._shift(mean, -1, 0, xp.nan)
        up = self._shift(mean, 1, 0, xp.nan)
        gx = xp.nan_to_num((right - left) / (2.0 * res), nan=0.0)
        gy = xp.nan_to_num((up - down) / (2.0 * res), nan=0.0)
        slope_angle = xp.arctan(xp.sqrt(gx * gx + gy * gy))

        center = mean
        left_f = xp.where(xp.isfinite(left), left, center)
        right_f = xp.where(xp.isfinite(right), right, center)
        down_f = xp.where(xp.isfinite(down), down, center)
        up_f = xp.where(xp.isfinite(up), up, center)
        step = xp.maximum(
            xp.maximum(xp.abs(center - left_f), xp.abs(center - right_f)),
            xp.maximum(xp.abs(center - down_f), xp.abs(center - up_f)),
        )
        slope_risk = self._soft_score(xp, slope_angle, p.slope_free_rad, p.max_slope_rad)
        step_risk = self._soft_score(xp, step, p.step_free_m, p.max_step_m)
        var_risk = self._soft_score(xp, bgk_var, p.variance_free, p.variance_hard)
        risk = xp.clip(0.45 * slope_risk + 0.35 * step_risk + 0.20 * var_risk, 0.0, 1.0)
        return xp.where(valid > 0.5, risk, xp.float32(p.unknown_risk))

    def _footprint_offsets(self) -> List[Tuple[float, float]]:
        p = self.params
        front = p.footprint_front + p.footprint_margin
        back = p.footprint_back + p.footprint_margin
        half_width = p.footprint_half_width + p.footprint_margin
        xs = np.linspace(-back, front, max(2, int(p.footprint_samples_x)))
        ys = np.linspace(-half_width, half_width, max(2, int(p.footprint_samples_y)))
        offsets = [(float(x), float(y)) for x in xs for y in ys]
        extra = [
            (front, 0.0),
            (-back, 0.0),
            (0.0, half_width),
            (0.0, -half_width),
            (front, half_width),
            (front, -half_width),
            (-back, half_width),
            (-back, -half_width),
        ]
        offsets.extend(extra)
        return offsets

    def _offset_to_shift(self, yaw: float, off_x: float, off_y: float) -> Tuple[int, int]:
        c = math.cos(yaw)
        s = math.sin(yaw)
        dx = c * off_x - s * off_y
        dy = s * off_x + c * off_y
        col_shift = int(round(dx / max(self.params.resolution, 1.0e-3)))
        row_shift = int(round(dy / max(self.params.resolution, 1.0e-3)))
        return row_shift, col_shift

    def _edge_risk(self, mean, valid, yaw: float, positive_axis: str):
        xp = self._xp
        p = self.params
        half_width = p.footprint_half_width + p.footprint_margin
        front = p.footprint_front + p.footprint_margin
        back = p.footprint_back + p.footprint_margin
        if positive_axis == "x":
            pos_offsets = [(front, -half_width), (front, 0.0), (front, half_width)]
            neg_offsets = [(-back, -half_width), (-back, 0.0), (-back, half_width)]
            distance = max(0.05, front + back)
            free = p.pitch_free_rad
            hard = p.max_pitch_rad
        else:
            pos_offsets = [(-back, half_width), (0.0, half_width), (front, half_width)]
            neg_offsets = [(-back, -half_width), (0.0, -half_width), (front, -half_width)]
            distance = max(0.05, 2.0 * half_width)
            free = p.roll_free_rad
            hard = p.max_roll_rad

        def mean_for(offsets):
            h_sum = xp.zeros_like(mean, dtype=xp.float32)
            v_sum = xp.zeros_like(mean, dtype=xp.float32)
            for ox, oy in offsets:
                dr, dc = self._offset_to_shift(yaw, ox, oy)
                h = self._shift(mean, dr, dc, 0.0)
                v = self._shift(valid, dr, dc, 0.0)
                h_sum += h * v
                v_sum += v
            return h_sum / xp.maximum(v_sum, xp.float32(1.0e-3)), v_sum

        pos_h, pos_count = mean_for(pos_offsets)
        neg_h, neg_count = mean_for(neg_offsets)
        angle = xp.arctan(xp.abs(pos_h - neg_h) / xp.float32(distance))
        risk = self._soft_score(xp, angle, free, hard)
        enough = (pos_count > 1.0) & (neg_count > 1.0)
        return xp.where(enough, risk, xp.float32(p.unknown_risk))

    def _seb_yaw_sweep(self, mean, bgk_var, valid, baseline_slices):
        xp = self._xp
        p = self.params
        offsets = self._footprint_offsets()
        yaw_bins = max(1, int(p.yaw_bins))

        final_slices = []
        seb_slices = []
        pitch_slices = []
        roll_slices = []
        step_slices = []
        support_slices = []
        uncertainty_risk = self._soft_score(xp, bgk_var, p.variance_free, p.variance_hard)

        for yaw_idx in range(yaw_bins):
            yaw = (float(yaw_idx) + 0.5) * 2.0 * math.pi / float(yaw_bins)
            valid_count = xp.zeros_like(mean, dtype=xp.float32)
            h_min = xp.full(mean.shape, xp.inf, dtype=xp.float32)
            h_max = xp.full(mean.shape, -xp.inf, dtype=xp.float32)
            base_risk = baseline_slices[min(yaw_idx, len(baseline_slices) - 1)]
            footprint_baseline = base_risk.copy()

            for ox, oy in offsets:
                dr, dc = self._offset_to_shift(yaw, ox, oy)
                h = self._shift(mean, dr, dc, 0.0)
                v = self._shift(valid, dr, dc, 0.0)
                r = self._shift(base_risk, dr, dc, p.unknown_risk)
                footprint_baseline = xp.maximum(footprint_baseline, r)
                valid_count += v
                h_min = xp.where(v > 0.5, xp.minimum(h_min, h), h_min)
                h_max = xp.where(v > 0.5, xp.maximum(h_max, h), h_max)

            support = xp.clip(valid_count / xp.float32(len(offsets)), 0.0, 1.0)
            support_risk = xp.clip(
                (xp.float32(p.min_support_ratio) - support) / max(p.min_support_ratio, 1.0e-3),
                0.0,
                1.0,
            )
            height_span = xp.where(valid_count > 0.0, h_max - h_min, xp.float32(p.max_step_m))
            step_risk = self._soft_score(xp, height_span, p.step_free_m, p.max_step_m)
            pitch_risk = self._edge_risk(mean, valid, yaw, "x")
            roll_risk = self._edge_risk(mean, valid, yaw, "y")

            seb_risk = xp.maximum(
                xp.maximum(step_risk, support_risk),
                xp.maximum(pitch_risk, roll_risk),
            )
            if p.fusion_mode == "weighted":
                final = (
                    xp.float32(p.baseline_weight) * footprint_baseline
                    + xp.float32(p.seb_weight) * seb_risk
                    + xp.float32(p.uncertainty_weight) * uncertainty_risk
                )
                final = xp.clip(final, 0.0, 1.0)
            else:
                final = xp.maximum(
                    footprint_baseline,
                    xp.maximum(seb_risk, xp.float32(p.uncertainty_weight) * uncertainty_risk),
                )
            final = xp.where(valid > 0.5, final, xp.float32(p.unknown_risk))

            final_slices.append(final.astype(xp.float32))
            seb_slices.append(seb_risk.astype(xp.float32))
            pitch_slices.append(pitch_risk.astype(xp.float32))
            roll_slices.append(roll_risk.astype(xp.float32))
            step_slices.append(step_risk.astype(xp.float32))
            support_slices.append(support.astype(xp.float32))

        return final_slices, seb_slices, pitch_slices, roll_slices, step_slices, support_slices
