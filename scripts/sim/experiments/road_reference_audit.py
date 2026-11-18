"""Reference-only road re-scoring of frozen predictions, not new GP or navigation runs."""

import csv
from pathlib import Path

import numpy as np

from paper_clouds import grid, height_at
from pointcloud_batch import atomic_json, digest
from spatial_evidence import provenance, reconstruct_spatial_case
from thesis_statistics import source_values


def height_score(mean, reference, observed, variance):
    reference = np.asarray(reference).ravel()
    hidden = np.isfinite(reference) & ~np.asarray(observed).ravel().astype(bool)
    error = np.asarray(mean).ravel()[hidden] - reference[hidden]
    std = np.sqrt(np.asarray(variance).ravel()[hidden])
    return {"hidden_cells": int(hidden.sum()), "rmse_m": float(np.sqrt(np.mean(error ** 2))) if len(error) else None,
        "mae_m": float(np.abs(error).mean()) if len(error) else None,
        "picp95": float(np.mean(np.abs(error) <= 1.96 * std)) if len(error) else None,
        "width95_m": float(np.mean(3.92 * std)) if len(error) else None}


def audit_road_reference(record, catalog, results, dataset, output):
    sources = {d["id"]: d for d in catalog["datasets"]}
    references = {}; rows = []
    for row in record["runs"]:
        if row["family"] != "road" or row["status"] != "ok":
            continue
        if row["dataset_id"] not in references:
            item = sources[row["dataset_id"]]; source = dataset.parent / item["path"]
            if digest(source) != item["sha256"]:
                raise ValueError("road audit source changed")
            with np.load(source, allow_pickle=False) as a:
                references[row["dataset_id"]] = height_at(a["truth_height"], a["truth_origin"],
                    float(a["truth_resolution"]), grid(tuple(a["shape"]), a["query_origin"], .2)).reshape(tuple(a["shape"]))
        corrected = references[row["dataset_id"]]
        with np.load(results.parent / (row["id"] + ".npz"), allow_pickle=False) as a:
            legacy = a["reference_height"]; observed = a["observed_cells"]
            if (np.isfinite(legacy) & ~np.isfinite(corrected)).any():
                raise ValueError("reference correction removed previously scored cells")
            common = np.isfinite(legacy) & np.isfinite(corrected)
            if not np.allclose(legacy[common], corrected[common], atol=1e-5):
                raise ValueError("reference correction changed shared supported heights")
            old = height_score(a["height_mean"], legacy, observed, a["latent_var"])
            new = height_score(a["height_mean"], corrected, observed, a["latent_var"])
        if old["hidden_cells"] != row["hidden_cells"] or (old["rmse_m"] is not None and
                not np.isclose(old["rmse_m"], row["hidden_rmse_m"], atol=1e-6)):
            raise ValueError("archived prediction no longer matches frozen height scores")
        rows.append({"id": row["id"], "cluster_id": row["cluster_id"], "dataset_id": row["dataset_id"],
            "condition": row["condition"], "status": "ok", "legacy_reference_cells": int(np.isfinite(legacy).sum()),
            "corrected_reference_cells": int(np.isfinite(corrected).sum()),
            **{"legacy_" + k: v for k, v in old.items()}, **{"corrected_" + k: v for k, v in new.items()}})
        if len(rows) % 1000 == 0:
            print(f"[road-reference] re-scored {len(rows)} frozen predictions", flush=True)
    expected = [r for r in record["runs"] if r["family"] == "road" and r["status"] == "ok"]
    if len(rows) != len(expected):
        raise ValueError("partial road reference audit")
    statistics = {}
    for condition in record["settings"]["conditions"]:
        selected = [r for r in rows if r["condition"] == condition]
        statistics[condition] = {}
        for key in ("rmse_m", "mae_m", "picp95", "width95_m", "hidden_cells", "reference_cells"):
            for version in ("legacy", "corrected"):
                values = source_values(selected, version + "_" + key)
                statistics[condition][version + "_" + key] = {
                    "source_groups": len(values), "mean": float(np.mean(list(values.values()))) if values else None}
    example = reconstruct_spatial_case(record, catalog, results, dataset,
                                       next(r for r in expected if r["condition"] == "hole16"))
    report = {"scope": "v1/v2 reference-only re-scoring; same frozen GP means/variances and observation masks; not algorithm improvement",
        "risk_reference_rescored": False, "new_gp_runs": 0, "cases": len(rows), "sources": len(references),
        "results_sha256": digest(results), "dataset_sha256": digest(dataset),
        "sampler_sha256": digest(Path(__file__).with_name("paper_clouds.py")),
        "statistics": statistics, "example": provenance(example)}
    atomic_json(output / "road_reference_audit.json", report)
    table = []
    labels = {"point15": r"删点15\%", "point30": r"删点30\%", "point60": r"删点60\%",
        "hole08": "0.8米孔洞", "hole16": "1.6米孔洞", "hole32": "3.2米孔洞",
        "frontier25": r"单侧隐藏25\%", "frontier50": r"单侧隐藏50\%"}
    for condition, s in statistics.items():
        table.append(f"{labels[condition]} & {s['legacy_hidden_cells']['mean']:.1f} & {s['corrected_hidden_cells']['mean']:.1f} & "
            f"{s['legacy_rmse_m']['mean']*100:.2f} & {s['corrected_rmse_m']['mean']*100:.2f} & "
            f"{s['legacy_picp95']['mean']*100:.1f} & {s['corrected_picp95']['mean']*100:.1f} " + r"\\")
    (output / "road_reference_rows.tex").write_text("\n".join(table) + "\n" + r"\bottomrule" + "\n", encoding="utf-8")
    with (output / "road_reference_cases.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    return report, example


def plot_road_deletion(case, output):
    from plot_paper_scale import plt, save
    from matplotlib import colors
    from matplotlib.patches import Rectangle, Patch
    n = case["counts"]; origin = case["origin"]
    extent = [origin[0], origin[0] + 20.8, origin[1], origin[1] + 20.8]
    low, high = np.nanmin(case["before"]), np.nanmax(case["before"])
    mask = np.zeros_like(case["before"], dtype=int)
    mask[np.isfinite(case["observed"])] = 1
    mask[np.isfinite(case["before"]) & ~np.isfinite(case["observed"])] = 2
    fields = (case["before"], case["after_before_noise"], case["mean"],
              case["legacy_reference"], case["corrected_reference"], mask)
    titles = (f"(a) 删孔前源观测\n{n['before_cells']}格", f"(b) 删孔后源观测\n{n['after_cells']}格",
        "(c) GP预测均值\n10816格（非实测）", f"(d) 旧评分参考\n{n['legacy_reference_cells']}格",
        f"(e) 修正采样参考\n{n['corrected_reference_cells']}格", f"(f) 删除效果\n{n['removed_cells']}格被删除")
    fig, axes = plt.subplots(2, 3, figsize=(8.2, 6.0), layout="constrained")
    for i, (ax, field, title) in enumerate(zip(axes.flat, fields, titles)):
        if i < 5:
            im = ax.imshow(field, origin="lower", extent=extent, cmap="terrain", vmin=low, vmax=high,
                           interpolation="nearest")
            fig.colorbar(im, ax=ax, shrink=.75, pad=.02, label="高程（米）")
        else:
            ax.imshow(field, origin="lower", extent=extent,
                cmap=colors.ListedColormap(["#e1e3e5", "#24788e", "#bd6547"]), vmin=0, vmax=2, interpolation="nearest")
            ax.legend(handles=[Patch(color=c, label=l) for c, l in
                (("#e1e3e5", "原本无观测"), ("#24788e", "保留观测"), ("#bd6547", "本次删除"))], fontsize=7, loc="lower right")
        center = origin + 10.4
        ax.add_patch(Rectangle(center - .8, 1.6, 1.6, fill=False, edgecolor="black", ls="--", lw=1.1))
        ax.set_title(title, fontsize=9); ax.set_xlabel("局部 x（米）"); ax.set_ylabel("局部 y（米）")
        ax.set_aspect("equal"); ax.set_xticks(center[0] + np.array([-10, 0, 10])); ax.set_yticks(center[1] + np.array([-10, 0, 10]))
    fig.suptitle("同一个道路样例：实测删除、模型预测与评分参考必须分开\n"
        "(a)(b)同为加噪前源点；黑框是1.6米孔洞；GP预测不是新增观测", fontsize=10)
    save(fig, output, "M1_road_deletion")
