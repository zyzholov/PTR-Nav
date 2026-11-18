#!/usr/bin/env python3
"""Known-plane SEB yaw diagnostic with an independent analytic attitude oracle."""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_style import configure_chinese
from terrain_risk_benchmark import coordinates, params_from_config

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src/fsgp_bgk_se2"))
from fsgp_bgk_se2.bgk_seb import BgkSebEstimator


def analytic_plane_risk(slope_rad, direction_rad, params):
    yaw = (np.arange(params.yaw_bins) + 0.5) * 2 * math.pi / params.yaw_bins
    gradient = math.tan(slope_rad)
    pitch = np.arctan(np.abs(gradient * np.cos(yaw - direction_rad)))
    roll = np.arctan(np.abs(gradient * np.sin(yaw - direction_rad)))
    score = np.maximum(
        np.clip((pitch - params.pitch_free_rad) / (params.max_pitch_rad - params.pitch_free_rad), 0, 1),
        np.clip((roll - params.roll_free_rad) / (params.max_roll_rad - params.roll_free_rad), 0, 1))
    unsafe = (pitch >= params.max_pitch_rad) | (roll >= params.max_roll_rad)
    return yaw, score, unsafe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if (args.output/"results.json").exists():
        parser.error("existing result directory; use a new versioned output")
    args.output.mkdir(parents=True, exist_ok=True)
    configure_chinese()
    rows, curves = [], []
    size, resolution, yaw_bins = 48, 0.2, 36
    slopes = (0.10, 0.20, 0.35, 0.45, 0.50, 0.60)
    directions = (0.0, 0.37, 0.91, 1.43)
    paths=(Path(__file__),Path(__file__).with_name("terrain_risk_benchmark.py"),
           ROOT/"src/fsgp_bgk_se2/config/hunter_fastlio.yaml",
           ROOT/"src/fsgp_bgk_se2/fsgp_bgk_se2/bgk_seb.py")
    fingerprints={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    parameters={}
    x, y = coordinates(size, resolution)
    interior = (np.abs(x) < 1.5) & (np.abs(y) < 1.5)
    for backend in ("cpu", "cupy"):
        params = params_from_config(backend, resolution, yaw_bins)
        params.store_diagnostic_slices = True
        parameters[backend]=asdict(params)
        estimator = BgkSebEstimator(params)
        for si, slope in enumerate(slopes):
            for di, direction in enumerate(directions):
                truth = math.tan(slope) * (x * math.cos(direction) + y * math.sin(direction))
                yaw, oracle_score, unsafe = analytic_plane_risk(slope, direction, params)
                for repeat in range(args.repeats):
                    seed = args.seed + si * 1000 + di * 100 + repeat
                    noisy = truth + np.random.default_rng(seed).normal(0.0, 0.02, truth.shape)
                    start = time.perf_counter()
                    result = estimator.compute(noisy.astype(np.float32),
                        np.full(truth.shape, 0.02 ** 2, dtype=np.float32),
                        np.ones(truth.shape, dtype=bool), [np.zeros_like(truth, dtype=np.float32)])
                    elapsed = (time.perf_counter() - start) * 1000
                    if result.backend != backend:
                        raise RuntimeError(f"requested {backend}, received {result.backend}: {result.backend_reason}")
                    seb = np.stack(result.seb_risk_slices)[:, interior]
                    fused = np.stack(result.final_risk_slices)[:, interior]
                    target = np.broadcast_to(unsafe[:, None], seb.shape)
                    predicted = fused >= 0.58
                    fn = int(np.count_nonzero(~predicted & target))
                    fp = int(np.count_nonzero(predicted & ~target))
                    hazard_count = int(np.count_nonzero(target))
                    safe_count = target.size - hazard_count
                    rows.append({"backend": result.backend, "slope_rad": slope,
                        "direction_rad": direction, "geometry_id": f"s{si}_d{di}",
                        "sensor_seed": seed, "repeat": repeat, "compute_ms": elapsed,
                        "seb_risk_mae": float(np.abs(seb - oracle_score[:, None]).mean()),
                        "fused_risk_mae": float(np.abs(fused - oracle_score[:, None]).mean()),
                        "false_safe_count": fn, "false_hazard_count": fp,
                        "unsafe_states": hazard_count, "safe_states": safe_count,
                        "false_safe_rate": fn / hazard_count if hazard_count else None,
                        "false_hazard_rate": fp / safe_count if safe_count else None,
                        "seb_yaw_span": float(np.ptp(seb, axis=0).mean())})
                    if backend == "cupy" and di == 0 and repeat == 0:
                        curves.append({"slope_rad": slope, "yaw": yaw.tolist(),
                            "oracle": oracle_score.tolist(), "seb": seb.mean(axis=1).tolist(),
                            "fused": fused.mean(axis=1).tolist(), "unsafe": unsafe.tolist()})
        print(f"[seb] {backend}: {len(slopes) * len(directions) * args.repeats} runs", flush=True)
    data = {"scope": "observed noisy planes only; not GP unknown prediction or PhysX traversal",
        "geometry_count": len(slopes) * len(directions), "runs": rows, "curves": curves,
        "arguments": {"repeats": args.repeats, "seed": args.seed},
        "adapter_sha256": hashlib.sha256((ROOT / "src/fsgp_bgk_se2/fsgp_bgk_se2/bgk_seb.py").read_bytes()).hexdigest(),
        "hard_threshold": 0.58, "noise_std_m": 0.02, "size": size, "yaw_bins": yaw_bins}
    if any(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=value for p,value in fingerprints.items()):
        raise RuntimeError("method/configuration changed during paired experiment")
    data["source_sha256"]=fingerprints
    data["parameters"]=parameters
    (args.output / "results.json").write_text(json.dumps(data, indent=2) + "\n")
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for ax, curve in zip(axes.flat, curves):
        degrees = np.degrees(curve["yaw"])
        ax.plot(degrees, curve["oracle"], color="black", label="解析姿态风险真值")
        ax.plot(degrees, curve["seb"], label="SEB 分量")
        ax.plot(degrees, curve["fused"], linestyle="--", label="保守融合输出")
        ax.axhline(0.58, color="#bd3939", linewidth=0.8, label="规划危险门限")
        ax.fill_between(degrees, 0, 1, where=curve["unsafe"], color="#bd3939", alpha=0.08)
        ax.set_title(f"坡面角度 {math.degrees(curve['slope_rad']):.1f} 度")
        ax.set_xlabel("车辆航向（度）"); ax.set_ylabel("风险分数")
        ax.set_ylim(0, 1.05); ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("同一点位航向扫描：方向性风险与解析坡面真值（阴影为超硬限制航向）")
    for suffix in ("png", "pdf"):
        fig.savefig(args.output / f"01_seb_yaw_curves.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    lines = ["# SEB 航向几何诊断", "",
        f"共 {len(rows)} 次 CPU/GPU 计算，24 个坡度/方向组合，每组合 {args.repeats} 个配对噪声种子。",
        "CPU/GPU 输出、航向和栅格不重复计作独立世界。输入全为已观测高程，不包含 GP 未知区预测。",
        "真值由平面梯度在车辆坐标下解析转换得到；这是几何合同诊断，不是动力学通行试验。",
        "融合风险包含SEB加权、方差项和硬几何保护；其误差不能混称 SEB 分量误差。", "",
        "| 后端 | 次数 | SEB 分数 MAE | 融合分数 MAE | 危险漏报率 | 安全误报率 |",
        "|---|---:|---:|---:|---:|---:|"]
    for backend in ("cpu", "cupy"):
        selected = [r for r in rows if r["backend"] == backend]
        fn, unsafe_n = sum(r["false_safe_count"] for r in selected), sum(r["unsafe_states"] for r in selected)
        fp, safe_n = sum(r["false_hazard_count"] for r in selected), sum(r["safe_states"] for r in selected)
        lines.append(f"| {backend} | {len(selected)} | {np.mean([r['seb_risk_mae'] for r in selected]):.4f} | "
            f"{np.mean([r['fused_risk_mae'] for r in selected]):.4f} | {fn / unsafe_n:.2%} | {fp / safe_n:.2%} |")
    lines.extend(["", "![航向风险曲线](01_seb_yaw_curves.png)", "",
        "漏报/误报统计使用规划分数门限 0.58 和车辆硬姿态限制两套不同定义，门限保守时可能有较高误报。",
        "本结果不覆盖台阶边缘、悬空障碍、软土或高速侧翻，也不构成风险概率标定。"])
    (args.output / "seb_direction_report.md").write_text("\n".join(lines) + "\n")
    print(f"[seb] results={args.output}")


if __name__ == "__main__":
    main()
