from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


_CUPY_KERNEL_SOURCE = r"""
__device__ __forceinline__ float clamp01(const float value)
{
    return fminf(1.0f, fmaxf(0.0f, value));
}

__device__ __forceinline__ float soft_score(
    const float value, const float free_value, const float hard_value)
{
    return clamp01((value - free_value) / fmaxf(1.0e-6f, hard_value - free_value));
}

extern "C" __global__ void bgk_smooth_kernel(
    const float* elevation,
    const float* variance,
    const float* valid,
    const int* offset_rows,
    const int* offset_cols,
    const float* weights,
    const int offset_count,
    const int rows,
    const int cols,
    const float min_weight,
    const float variance_hard,
    float* mean_out,
    float* variance_out)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;
    const int cell_count = rows * cols;
    if (index >= cell_count)
        return;

    const int row = index / cols;
    const int col = index - row * cols;
    float weight_sum = 0.0f;
    float height_sum = 0.0f;
    float height2_sum = 0.0f;
    float variance_sum = 0.0f;

    for (int k = 0; k < offset_count; ++k)
    {
        const int source_row = row + offset_rows[k];
        const int source_col = col + offset_cols[k];
        if (source_row < 0 || source_row >= rows || source_col < 0 || source_col >= cols)
            continue;
        const int source_index = source_row * cols + source_col;
        const float weighted_valid = weights[k] * valid[source_index];
        const float height = elevation[source_index];
        weight_sum += weighted_valid;
        height_sum += weighted_valid * height;
        height2_sum += weighted_valid * height * height;
        variance_sum += weighted_valid * variance[source_index];
    }

    if (weight_sum <= min_weight)
    {
        mean_out[index] = elevation[index];
        variance_out[index] = variance_hard;
        return;
    }

    const float safe_weight = fmaxf(weight_sum, min_weight);
    const float mean = height_sum / safe_weight;
    const float second_moment = height2_sum / safe_weight;
    mean_out[index] = mean;
    variance_out[index] =
        fmaxf(second_moment - mean * mean, 0.0f) + variance_sum / safe_weight;
}

extern "C" __global__ void bgk_risk_kernel(
    const float* mean,
    const float* variance,
    const float* valid,
    const int rows,
    const int cols,
    const float resolution,
    const float slope_free,
    const float slope_hard,
    const float step_free,
    const float step_hard,
    const float variance_free,
    const float variance_hard,
    const float unknown_risk,
    float* risk_out)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;
    const int cell_count = rows * cols;
    if (index >= cell_count)
        return;

    if (valid[index] <= 0.5f)
    {
        risk_out[index] = unknown_risk;
        return;
    }

    const int row = index / cols;
    const int col = index - row * cols;
    const float center = mean[index];
    const bool has_left = col > 0;
    const bool has_right = col + 1 < cols;
    const bool has_down = row > 0;
    const bool has_up = row + 1 < rows;
    const float left = has_left ? mean[index - 1] : center;
    const float right = has_right ? mean[index + 1] : center;
    const float down = has_down ? mean[index - cols] : center;
    const float up = has_up ? mean[index + cols] : center;
    const float gx = (has_left && has_right) ? (right - left) / (2.0f * resolution) : 0.0f;
    const float gy = (has_down && has_up) ? (up - down) / (2.0f * resolution) : 0.0f;
    const float slope_angle = atanf(sqrtf(gx * gx + gy * gy));
    float step = 0.0f;
    if (has_left) step = fmaxf(step, fabsf(center - left));
    if (has_right) step = fmaxf(step, fabsf(center - right));
    if (has_down) step = fmaxf(step, fabsf(center - down));
    if (has_up) step = fmaxf(step, fabsf(center - up));
    const float slope_risk = soft_score(slope_angle, slope_free, slope_hard);
    const float step_risk = soft_score(step, step_free, step_hard);
    const float variance_risk = soft_score(variance[index], variance_free, variance_hard);
    risk_out[index] = clamp01(
        0.45f * slope_risk + 0.35f * step_risk + 0.20f * variance_risk);
}

extern "C" __global__ void seb_yaw_kernel(
    const float* mean,
    const float* variance,
    const float* valid,
    const float* baseline,
    const int* footprint_rows,
    const int* footprint_cols,
    const float* footprint_x,
    const float* footprint_y,
    const int footprint_count,
    const int* edge_rows,
    const int* edge_cols,
    const int rows,
    const int cols,
    const int yaw_bins,
    const float variance_free,
    const float variance_hard,
    const float step_free,
    const float step_hard,
    const float pitch_free,
    const float pitch_hard,
    const float roll_free,
    const float roll_hard,
    const float min_support_ratio,
    const float unknown_risk,
    const float baseline_weight,
    const float seb_weight,
    const float uncertainty_weight,
    const int weighted_fusion,
    const float pitch_distance,
    const float roll_distance,
    float* final_out,
    float* seb_out,
    float* pitch_out,
    float* roll_out,
    float* step_out,
    float* support_out)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;
    const int cell_count = rows * cols;
    const int total_count = cell_count * yaw_bins;
    if (index >= total_count)
        return;

    const int yaw_index = index / cell_count;
    const int cell_index = index - yaw_index * cell_count;
    const int row = cell_index / cols;
    const int col = cell_index - row * cols;
    const int footprint_base = yaw_index * footprint_count;
    const int edge_base = yaw_index * 12;
    const float* baseline_yaw = baseline + yaw_index * cell_count;

    float valid_count = 0.0f;
    float footprint_baseline = baseline_yaw[cell_index];
    for (int k = 0; k < footprint_count; ++k)
    {
        const int source_row = row + footprint_rows[footprint_base + k];
        const int source_col = col + footprint_cols[footprint_base + k];
        if (source_row < 0 || source_row >= rows || source_col < 0 || source_col >= cols)
        {
            footprint_baseline = fmaxf(footprint_baseline, unknown_risk);
            continue;
        }
        const int source_index = source_row * cols + source_col;
        const float source_valid = valid[source_index];
        footprint_baseline = fmaxf(footprint_baseline, baseline_yaw[source_index]);
        valid_count += source_valid;
    }

    const float support = clamp01(valid_count / (float)footprint_count);
    const float support_risk = clamp01(
        (min_support_ratio - support) / fmaxf(min_support_ratio, 1.0e-3f));
    float edge_height_sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float edge_valid_sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int k = 0; k < 12; ++k)
    {
        const int source_row = row + edge_rows[edge_base + k];
        const int source_col = col + edge_cols[edge_base + k];
        if (source_row < 0 || source_row >= rows || source_col < 0 || source_col >= cols)
            continue;
        const int source_index = source_row * cols + source_col;
        const int group = k / 3;
        const float source_valid = valid[source_index];
        edge_height_sum[group] += mean[source_index] * source_valid;
        edge_valid_sum[group] += source_valid;
    }

    float pitch_risk = unknown_risk;
    float pitch_gradient = 0.0f;
    if (edge_valid_sum[0] > 1.0f && edge_valid_sum[1] > 1.0f)
    {
        const float positive = edge_height_sum[0] / fmaxf(edge_valid_sum[0], 1.0e-3f);
        const float negative = edge_height_sum[1] / fmaxf(edge_valid_sum[1], 1.0e-3f);
        pitch_gradient = (positive - negative) / pitch_distance;
        pitch_risk = soft_score(atanf(fabsf(pitch_gradient)), pitch_free, pitch_hard);
    }
    float roll_risk = unknown_risk;
    float roll_gradient = 0.0f;
    if (edge_valid_sum[2] > 1.0f && edge_valid_sum[3] > 1.0f)
    {
        const float positive = edge_height_sum[2] / fmaxf(edge_valid_sum[2], 1.0e-3f);
        const float negative = edge_height_sum[3] / fmaxf(edge_valid_sum[3], 1.0e-3f);
        roll_gradient = (positive - negative) / roll_distance;
        roll_risk = soft_score(atanf(fabsf(roll_gradient)), roll_free, roll_hard);
    }

    float residual_min = 3.402823466e+38F;
    float residual_max = -3.402823466e+38F;
    for (int k = 0; k < footprint_count; ++k)
    {
        const int source_row = row + footprint_rows[footprint_base + k];
        const int source_col = col + footprint_cols[footprint_base + k];
        if (source_row < 0 || source_row >= rows || source_col < 0 || source_col >= cols)
            continue;
        const int source_index = source_row * cols + source_col;
        if (valid[source_index] <= 0.5f)
            continue;
        const float residual = mean[source_index] -
            pitch_gradient * footprint_x[footprint_base + k] -
            roll_gradient * footprint_y[footprint_base + k];
        residual_min = fminf(residual_min, residual);
        residual_max = fmaxf(residual_max, residual);
    }
    const float residual_span = valid_count > 0.0f ? residual_max - residual_min : step_hard;
    // Plane removal attenuates a sharp step across the sampled footprint.
    const float step_risk = soft_score(3.0f * residual_span, step_free, step_hard);

    const float seb_risk = fmaxf(fmaxf(step_risk, support_risk), fmaxf(pitch_risk, roll_risk));
    const float uncertainty_risk = soft_score(
        variance[cell_index], variance_free, variance_hard);
    float final_risk;
    if (weighted_fusion != 0)
    {
        final_risk = clamp01(
            baseline_weight * footprint_baseline + seb_weight * seb_risk +
            uncertainty_weight * uncertainty_risk);
    }
    else
    {
        final_risk = fmaxf(
            footprint_baseline, fmaxf(seb_risk, uncertainty_weight * uncertainty_risk));
    }
    final_risk = fmaxf(final_risk, footprint_baseline);
    if (step_risk >= 0.999f || pitch_risk >= 0.999f || roll_risk >= 0.999f)
        final_risk = 1.0f;
    if (valid[cell_index] <= 0.5f)
        final_risk = unknown_risk;

    final_out[index] = final_risk;
    seb_out[index] = seb_risk;
    pitch_out[index] = pitch_risk;
    roll_out[index] = roll_risk;
    step_out[index] = step_risk;
    support_out[index] = support;
}
"""


@dataclass
class BgkSebParams:
    resolution: float
    yaw_bins: int
    prefer_gpu: bool = True
    gpu_backend: str = "auto"
    auto_gpu_min_work_items: int = 100_000
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
    fusion_mode: str = "weighted"
    baseline_weight: float = 0.45
    seb_weight: float = 0.45
    uncertainty_weight: float = 0.10
    unknown_risk: float = 0.65
    store_diagnostic_slices: bool = False


@dataclass
class BgkSebResult:
    backend: str
    backend_reason: str
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
        self._cupy = None
        self._cupy_module = None
        self._auto_backend = False
        self.backend_name = "cpu"
        self.backend_reason = "GPU disabled"
        if params.prefer_gpu:
            self._try_select_gpu(params.gpu_backend)

    def _try_select_gpu(self, requested: str) -> None:
        if requested == "cpu":
            self.backend_name = "cpu"
            self.backend_reason = "gpu_backend=cpu"
            self._xp = np
            return
        if requested not in ("auto", "cupy"):
            self.backend_name = "cpu"
            self.backend_reason = f"unsupported gpu_backend={requested}; using CPU"
            self._xp = np
            return
        if requested == "auto":
            # Delay importing CuPy until the first map proves large enough to
            # benefit from it. The common CPU path stays lightweight.
            self._auto_backend = True
            self.backend_name = "cpu"
            self.backend_reason = "auto backend pending first map"
            return

        cp = self._load_cupy()
        if cp is None:
            return
        self._xp = cp
        self.backend_name = "cupy"
        self.backend_reason = "gpu_backend=cupy"

    def _load_cupy(self):
        try:
            import cupy as cp  # type: ignore

            if cp.cuda.runtime.getDeviceCount() <= 0:
                self.backend_reason = "no CUDA device found; using CPU"
                return None
            self._cupy = cp
            return cp
        except Exception as exc:
            self._xp = np
            self.backend_name = "cpu"
            self.backend_reason = (
                f"CuPy/CUDA unavailable ({type(exc).__name__}); using CPU"
            )
            return None

    def _select_auto_backend(self, rows: int, cols: int, yaw_bins: int) -> None:
        if not self._auto_backend:
            return
        work_items = int(rows) * int(cols) * max(1, int(yaw_bins))
        threshold = max(1, int(self.params.auto_gpu_min_work_items))
        if work_items >= threshold:
            cp = self._load_cupy()
            if cp is not None:
                self._xp = cp
                self.backend_name = "cupy"
                self.backend_reason = (
                    f"auto selected CuPy for {work_items} work items "
                    f"(threshold {threshold})"
                )
        else:
            self._xp = np
            self.backend_name = "cpu"
            self.backend_reason = (
                f"auto selected CPU for {work_items} work items "
                f"(threshold {threshold})"
            )
        self._auto_backend = False

    def compute(
        self,
        elevation: np.ndarray,
        variance: np.ndarray,
        valid: np.ndarray,
        baseline_risk_slices: Sequence[np.ndarray],
    ) -> BgkSebResult:
        self._select_auto_backend(
            elevation.shape[0],
            elevation.shape[1],
            self.params.yaw_bins,
        )
        try:
            return self._compute_impl(elevation, variance, valid, baseline_risk_slices)
        except Exception as exc:
            if self.backend_name != "cupy":
                raise
            self._xp = np
            self.backend_name = "cpu"
            self.backend_reason = f"CuPy failed ({type(exc).__name__}); fell back to CPU"
            return self._compute_impl(elevation, variance, valid, baseline_risk_slices)

    def _compute_impl(
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
        if not baseline_d:
            baseline_d = [xp.zeros_like(elevation_d, dtype=xp.float32)]

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
            backend_reason=self.backend_reason,
            bgk_mean=self._to_numpy(bgk_mean),
            bgk_var=self._to_numpy(bgk_var),
            bgk_traversability=self._to_numpy(bgk_trav),
            final_risk_slices=self._slices_to_numpy(final_slices),
            seb_risk_slices=self._slices_to_numpy(seb_slices),
            seb_pitch_risk_slices=self._slices_to_numpy(pitch_slices),
            seb_roll_risk_slices=self._slices_to_numpy(roll_slices),
            seb_step_risk_slices=self._slices_to_numpy(step_slices),
            seb_support_slices=self._slices_to_numpy(support_slices),
            baseline_risk=self._to_numpy(baseline_d[0]),
        )

    def _to_numpy(self, arr):
        if self.backend_name == "cupy":
            return self._xp.asnumpy(arr).astype(np.float32, copy=False)
        return np.asarray(arr, dtype=np.float32)

    def _slices_to_numpy(self, slices) -> List[np.ndarray]:
        if not slices:
            return []
        if self.backend_name == "cupy":
            stacked = self._xp.stack(slices, axis=0)
            host = self._xp.asnumpy(stacked).astype(np.float32, copy=False)
            return [host[i] for i in range(host.shape[0])]
        return [np.asarray(x, dtype=np.float32) for x in slices]

    def _get_cupy_kernel(self, name: str):
        cp = self._cupy
        if cp is None:
            raise RuntimeError("CuPy backend selected without an initialized CuPy module")
        if self._cupy_module is None:
            self._cupy_module = cp.RawModule(
                code=_CUPY_KERNEL_SOURCE,
                options=("--std=c++11",),
                name_expressions=(
                    "bgk_smooth_kernel",
                    "bgk_risk_kernel",
                    "seb_yaw_kernel",
                ),
            )
        return self._cupy_module.get_function(name)

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
        if self.backend_name == "cupy":
            return self._bgk_smooth_cupy(elevation, variance, valid)

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

    def _bgk_smooth_cupy(self, elevation, variance, valid):
        cp = self._cupy
        if cp is None:
            raise RuntimeError("CuPy smoothing requested without a CUDA backend")
        offsets = self._kernel_offsets()
        offset_rows = cp.asarray([x[0] for x in offsets], dtype=cp.int32)
        offset_cols = cp.asarray([x[1] for x in offsets], dtype=cp.int32)
        weights = cp.asarray([x[2] for x in offsets], dtype=cp.float32)
        mean = cp.empty_like(elevation, dtype=cp.float32)
        bgk_var = cp.empty_like(elevation, dtype=cp.float32)
        rows, cols = elevation.shape
        threads = 256
        blocks = (rows * cols + threads - 1) // threads
        self._get_cupy_kernel("bgk_smooth_kernel")(
            (blocks,),
            (threads,),
            (
                elevation,
                variance,
                valid,
                offset_rows,
                offset_cols,
                weights,
                np.int32(len(offsets)),
                np.int32(rows),
                np.int32(cols),
                np.float32(self.params.bgk_min_weight),
                np.float32(self.params.variance_hard),
                mean,
                bgk_var,
            ),
        )
        return mean, bgk_var

    def _bgk_risk(self, mean, bgk_var, valid):
        if self.backend_name == "cupy":
            return self._bgk_risk_cupy(mean, bgk_var, valid)

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

    def _bgk_risk_cupy(self, mean, bgk_var, valid):
        cp = self._cupy
        if cp is None:
            raise RuntimeError("CuPy BGK risk requested without a CUDA backend")
        p = self.params
        rows, cols = mean.shape
        risk = cp.empty_like(mean, dtype=cp.float32)
        threads = 256
        blocks = (rows * cols + threads - 1) // threads
        self._get_cupy_kernel("bgk_risk_kernel")(
            (blocks,),
            (threads,),
            (
                mean,
                bgk_var,
                valid,
                np.int32(rows),
                np.int32(cols),
                np.float32(max(p.resolution, 1.0e-3)),
                np.float32(p.slope_free_rad),
                np.float32(p.max_slope_rad),
                np.float32(p.step_free_m),
                np.float32(p.max_step_m),
                np.float32(p.variance_free),
                np.float32(p.variance_hard),
                np.float32(p.unknown_risk),
                risk,
            ),
        )
        return risk

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
        gradient = (pos_h - neg_h) / xp.float32(distance)
        angle = xp.arctan(xp.abs(gradient))
        risk = self._soft_score(xp, angle, free, hard)
        enough = (pos_count > 1.0) & (neg_count > 1.0)
        return (
            xp.where(enough, risk, xp.float32(p.unknown_risk)),
            xp.where(enough, gradient, xp.float32(0.0)),
        )

    def _seb_yaw_sweep(self, mean, bgk_var, valid, baseline_slices):
        if self.backend_name == "cupy":
            return self._seb_yaw_sweep_cupy(mean, bgk_var, valid, baseline_slices)

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
            base_risk = baseline_slices[min(yaw_idx, len(baseline_slices) - 1)]
            footprint_baseline = base_risk.copy()

            for ox, oy in offsets:
                dr, dc = self._offset_to_shift(yaw, ox, oy)
                v = self._shift(valid, dr, dc, 0.0)
                r = self._shift(base_risk, dr, dc, p.unknown_risk)
                footprint_baseline = xp.maximum(footprint_baseline, r)
                valid_count += v

            support = xp.clip(valid_count / xp.float32(len(offsets)), 0.0, 1.0)
            support_risk = xp.clip(
                (xp.float32(p.min_support_ratio) - support) / max(p.min_support_ratio, 1.0e-3),
                0.0,
                1.0,
            )
            pitch_risk, pitch_gradient = self._edge_risk(mean, valid, yaw, "x")
            roll_risk, roll_gradient = self._edge_risk(mean, valid, yaw, "y")
            residual_min = xp.full(mean.shape, xp.inf, dtype=xp.float32)
            residual_max = xp.full(mean.shape, -xp.inf, dtype=xp.float32)
            for ox, oy in offsets:
                dr, dc = self._offset_to_shift(yaw, ox, oy)
                h = self._shift(mean, dr, dc, 0.0)
                v = self._shift(valid, dr, dc, 0.0)
                residual = h - pitch_gradient * xp.float32(ox) - roll_gradient * xp.float32(oy)
                residual_min = xp.where(v > 0.5, xp.minimum(residual_min, residual), residual_min)
                residual_max = xp.where(v > 0.5, xp.maximum(residual_max, residual), residual_max)
            residual_span = xp.where(valid_count > 0.0, residual_max - residual_min, xp.float32(p.max_step_m))
            step_risk = self._soft_score(xp, 3.0 * residual_span, p.step_free_m, p.max_step_m)

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
            final = xp.maximum(final, footprint_baseline)
            hard_geometry = xp.maximum(step_risk, xp.maximum(pitch_risk, roll_risk)) >= 0.999
            final = xp.where(hard_geometry, xp.float32(1.0), final)
            final = xp.where(valid > 0.5, final, xp.float32(p.unknown_risk))

            final_slices.append(final.astype(xp.float32))
            if p.store_diagnostic_slices:
                seb_slices.append(seb_risk.astype(xp.float32))
                pitch_slices.append(pitch_risk.astype(xp.float32))
                roll_slices.append(roll_risk.astype(xp.float32))
                step_slices.append(step_risk.astype(xp.float32))
                support_slices.append(support.astype(xp.float32))

        return final_slices, seb_slices, pitch_slices, roll_slices, step_slices, support_slices

    def _seb_yaw_sweep_cupy(self, mean, bgk_var, valid, baseline_slices):
        cp = self._cupy
        if cp is None:
            raise RuntimeError("CuPy SEB sweep requested without a CUDA backend")
        p = self.params
        rows, cols = mean.shape
        yaw_bins = max(1, int(p.yaw_bins))
        offsets = self._footprint_offsets()
        footprint_rows = np.empty((yaw_bins, len(offsets)), dtype=np.int32)
        footprint_cols = np.empty_like(footprint_rows)
        footprint_x = np.broadcast_to(np.asarray([item[0] for item in offsets], dtype=np.float32), footprint_rows.shape).copy()
        footprint_y = np.broadcast_to(np.asarray([item[1] for item in offsets], dtype=np.float32), footprint_rows.shape).copy()
        edge_rows = np.empty((yaw_bins, 12), dtype=np.int32)
        edge_cols = np.empty_like(edge_rows)
        half_width = p.footprint_half_width + p.footprint_margin
        front = p.footprint_front + p.footprint_margin
        back = p.footprint_back + p.footprint_margin

        for yaw_index in range(yaw_bins):
            yaw = (float(yaw_index) + 0.5) * 2.0 * math.pi / float(yaw_bins)
            for offset_index, (off_x, off_y) in enumerate(offsets):
                dr, dc = self._offset_to_shift(yaw, off_x, off_y)
                footprint_rows[yaw_index, offset_index] = dr
                footprint_cols[yaw_index, offset_index] = dc

            edge_offsets = [
                (front, -half_width),
                (front, 0.0),
                (front, half_width),
                (-back, -half_width),
                (-back, 0.0),
                (-back, half_width),
                (-back, half_width),
                (0.0, half_width),
                (front, half_width),
                (-back, -half_width),
                (0.0, -half_width),
                (front, -half_width),
            ]
            for edge_index, (off_x, off_y) in enumerate(edge_offsets):
                dr, dc = self._offset_to_shift(yaw, off_x, off_y)
                edge_rows[yaw_index, edge_index] = dr
                edge_cols[yaw_index, edge_index] = dc

        baseline_stack = cp.stack(
            [baseline_slices[min(i, len(baseline_slices) - 1)] for i in range(yaw_bins)],
            axis=0,
        )
        footprint_rows_d = cp.asarray(footprint_rows)
        footprint_cols_d = cp.asarray(footprint_cols)
        footprint_x_d = cp.asarray(footprint_x)
        footprint_y_d = cp.asarray(footprint_y)
        edge_rows_d = cp.asarray(edge_rows)
        edge_cols_d = cp.asarray(edge_cols)
        shape = (yaw_bins, rows, cols)
        final = cp.empty(shape, dtype=cp.float32)
        seb = cp.empty(shape, dtype=cp.float32)
        pitch = cp.empty(shape, dtype=cp.float32)
        roll = cp.empty(shape, dtype=cp.float32)
        step = cp.empty(shape, dtype=cp.float32)
        support = cp.empty(shape, dtype=cp.float32)
        total = yaw_bins * rows * cols
        threads = 256
        blocks = (total + threads - 1) // threads
        self._get_cupy_kernel("seb_yaw_kernel")(
            (blocks,),
            (threads,),
            (
                mean,
                bgk_var,
                valid,
                baseline_stack,
                footprint_rows_d,
                footprint_cols_d,
                footprint_x_d,
                footprint_y_d,
                np.int32(len(offsets)),
                edge_rows_d,
                edge_cols_d,
                np.int32(rows),
                np.int32(cols),
                np.int32(yaw_bins),
                np.float32(p.variance_free),
                np.float32(p.variance_hard),
                np.float32(p.step_free_m),
                np.float32(p.max_step_m),
                np.float32(p.pitch_free_rad),
                np.float32(p.max_pitch_rad),
                np.float32(p.roll_free_rad),
                np.float32(p.max_roll_rad),
                np.float32(p.min_support_ratio),
                np.float32(p.unknown_risk),
                np.float32(p.baseline_weight),
                np.float32(p.seb_weight),
                np.float32(p.uncertainty_weight),
                np.int32(p.fusion_mode == "weighted"),
                np.float32(max(0.05, front + back)),
                np.float32(max(0.05, 2.0 * half_width)),
                final,
                seb,
                pitch,
                roll,
                step,
                support,
            ),
        )

        final_slices = [final[i] for i in range(yaw_bins)]
        if not p.store_diagnostic_slices:
            return final_slices, [], [], [], [], []
        return (
            final_slices,
            [seb[i] for i in range(yaw_bins)],
            [pitch[i] for i in range(yaw_bins)],
            [roll[i] for i in range(yaw_bins)],
            [step[i] for i in range(yaw_bins)],
            [support[i] for i in range(yaw_bins)],
        )
