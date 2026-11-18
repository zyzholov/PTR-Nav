"""Explain sparse-input scoring denominators without generating new predictions."""

from collections import Counter
import csv
import hashlib
from pathlib import Path

import numpy as np

from paper_clouds import grid, height_at, raster
from paper_scale_evaluate import partial_points
from pointcloud_batch import atomic_json, digest
from thesis_statistics import source_values


def scoring_partition(before, observed, reference):
    before = np.asarray(before, dtype=bool)
    observed = np.asarray(observed, dtype=bool)
    reference = np.asarray(reference, dtype=bool)
    if before.shape != observed.shape or observed.shape != reference.shape or not observed.size:
        raise ValueError("scoring masks must share a nonempty grid")
    if (observed & ~before).any():
        raise ValueError("deleting points created a measured cell")
    hidden = reference & ~observed
    removed = before & ~observed
    counts = {
        "query_cells": observed.size,
        "before_observed_cells": int(before.sum()),
        "observed_cells": int(observed.sum()),
        "reference_cells": int(reference.sum()),
        "removed_observed_cells": int(removed.sum()),
        "observed_referenced_cells": int((observed & reference).sum()),
        "hidden_reference_cells": int(hidden.sum()),
        "removed_referenced_cells": int((removed & reference).sum()),
        "naturally_unobserved_referenced_cells": int((reference & ~before).sum()),
        "observed_unreferenced_cells": int((observed & ~reference).sum()),
        "unobserved_unreferenced_cells": int((~observed & ~reference).sum()),
        "unreferenced_cells": int((~reference).sum()),
    }
    assert counts["hidden_reference_cells"] == (counts["removed_referenced_cells"] +
        counts["naturally_unobserved_referenced_cells"])
    assert counts["query_cells"] == sum(counts[k] for k in (
        "observed_referenced_cells", "hidden_reference_cells",
        "observed_unreferenced_cells", "unobserved_unreferenced_cells"))
    return counts


def sparse_reference_audit(record, catalog, results_path, dataset_path, output):
    sources = {d["id"]: d for d in catalog["datasets"]}
    rows = []; cache = {}; examples = {}; chain = hashlib.sha256()
    for row in record["runs"]:
        if row["family"] not in ("road", "indoor"):
            continue
        name = row["dataset_id"]
        if name not in cache:
            item = sources[name]; path = dataset_path.parent / item["path"]
            require = digest(path)
            if require != item["sha256"]:
                raise ValueError("sparse-reference source changed")
            chain.update((name + require).encode())
            with np.load(path, allow_pickle=False) as a:
                shape = tuple(a["shape"]); origin = a["query_origin"].copy()
                before = raster(a["xyz"][a["is_ground"]], shape, origin, .2)
                reference = height_at(a["truth_height"], a["truth_origin"], float(a["truth_resolution"]),
                    grid(shape, origin, .2)).reshape(shape)
            cache[name] = before, reference, origin
        before, reference, origin = cache[name]
        if row["status"] == "ok":
            path = results_path.parent / (row["id"] + ".npz")
            chain.update((row["id"] + digest(path)).encode())
            with np.load(path, allow_pickle=False) as a:
                observed = a["observed_cells"].astype(bool)
                legacy = np.isfinite(a["reference_height"])
                states = a["supported"]
                if int(states.sum()) != row["reference_states"]:
                    raise ValueError("archived risk reference denominator changed")
                state_count, supported_count = states.size, int(states.sum())
        else:
            # Failed/no-input cases remain in the spatial accounting, never as zero-error results.
            path = dataset_path.parent / sources[name]["path"]
            with np.load(path, allow_pickle=False) as a:
                keep = partial_points(a["xyz"], row["condition"], origin, row["case_seed"])
                observed = np.isfinite(raster(a["xyz"][keep & a["is_ground"]], before.shape, origin, .2))
            legacy = None; state_count = supported_count = None
        counts = scoring_partition(np.isfinite(before), observed, np.isfinite(reference))
        entry = {"id": row["id"], "dataset_id": name, "cluster_id": row["cluster_id"],
            "family": row["family"], "condition": row["condition"], "algorithm_status": row["status"],
            "status": "ok", "reason": row.get("reason"), **counts,
            "legacy_reference_cells": int(legacy.sum()) if legacy is not None else None,
            "legacy_hidden_cells": int((legacy & ~observed).sum()) if legacy is not None else None,
            "legacy_risk_selected_states": state_count, "legacy_risk_supported_states": supported_count,
            "legacy_risk_reference_eligibility": supported_count / state_count if state_count else None}
        for key in ("observed_cells", "reference_cells", "unreferenced_cells"):
            entry[key + "_fraction"] = counts[key] / counts["query_cells"]
        rows.append(entry)
        if row["condition"] == "hole16" and row["status"] == "ok" and row["family"] not in examples:
            examples[row["family"]] = {"row": entry, "before": np.isfinite(before), "observed": observed,
                "reference": np.isfinite(reference), "origin": origin}
        if len(rows) % 2000 == 0:
            print(f"[sparse-reference] accounted for {len(rows)} frozen conditions", flush=True)
    expected = [r for r in record["runs"] if r["family"] in ("road", "indoor")]
    if len(rows) != len(expected) or set(examples) != {"road", "indoor"}:
        raise ValueError("incomplete sparse-reference accounting")
    summaries = {}; table = []
    labels = {"point15": r"删点15\%", "hole16": "1.6米孔洞", "frontier50": r"单侧50\%"}
    for family in ("road", "indoor"):
        summaries[family] = {}
        for condition in record["settings"]["conditions"]:
            selected = [r for r in rows if r["family"] == family and r["condition"] == condition]
            summary = {"cases": len(selected), "algorithm_statuses": dict(Counter(r["algorithm_status"] for r in selected))}
            for key in ("observed_cells_fraction", "reference_cells_fraction", "unreferenced_cells_fraction",
                        "hidden_reference_cells", "removed_referenced_cells", "naturally_unobserved_referenced_cells",
                        "legacy_hidden_cells", "legacy_risk_reference_eligibility"):
                values = source_values(selected, key)
                summary[key] = {"groups": len(values), "mean": float(np.mean(list(values.values()))) if values else None}
            summary["cases_without_v2_hidden_reference"] = sum(r["hidden_reference_cells"] == 0 for r in selected)
            summary["cases_with_v2_hidden_prediction"] = sum(r["hidden_reference_cells"] > 0 and
                r["algorithm_status"] == "ok" for r in selected)
            summaries[family][condition] = summary
            if condition in labels:
                risk = summary["legacy_risk_reference_eligibility"]["mean"]
                table.append(f"{'道路' if family == 'road' else '室内'} & {labels[condition]} & " +
                    " & ".join(f"{summary[k]['mean']*100:.1f}" for k in (
                        "observed_cells_fraction", "reference_cells_fraction", "unreferenced_cells_fraction")) +
                    f" & {summary['hidden_reference_cells']['mean']:.1f} & " +
                    f"{summary['cases_with_v2_hidden_prediction']}/{len(selected)} & {risk*100:.1f} " + r"\\")
    report = {"scope": "spatial denominators only; v2 height reference, unchanged v1 sampled footprint support; no new GP or risk scores",
        "new_algorithm_runs": 0, "cases": len(rows), "sources": len(cache),
        "algorithm_statuses": dict(Counter(r["algorithm_status"] for r in rows)),
        "results_sha256": digest(results_path), "dataset_sha256": digest(dataset_path),
        "source_archive_chain_sha256": chain.hexdigest(), "statistics": summaries,
        "examples": {k: v["row"] for k, v in examples.items()},
        "weighting": "within-source case means, then equal source groups; all conditions retained; unsupported risk entries omitted, not zero"}
    atomic_json(output / "sparse_reference_audit.json", report)
    with (output / "sparse_reference_cases.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    (output / "sparse_reference_rows.tex").write_text("\n".join(table) + "\n" + r"\bottomrule" + "\n", encoding="utf-8")
    return report, examples


def plot_scoring_domains(examples, output):
    from matplotlib import colors
    from matplotlib.patches import Patch, Rectangle
    from plot_paper_scale import plt, save
    palette = ["#e2e5e7", "#4c927b", "#cb8b40", "#507da8"]
    fig, axes = plt.subplots(2, 3, figsize=(8.3, 6.9), layout="constrained")
    for i, family in enumerate(("road", "indoor")):
        example = examples[family]; row = example["row"]; origin = example["origin"]
        observed, reference = example["observed"], example["reference"]
        partition = np.zeros_like(reference, dtype=int)
        partition[observed & reference] = 1
        partition[~observed & reference] = 2
        partition[observed & ~reference] = 3
        fields = (example["before"], reference, partition)
        family_label = "城市道路" if family == "road" else "室内"
        titles = (f"{family_label}：删孔前源输入 O0\n{row['dataset_id']} / {row['before_observed_cells']}格",
            f"来源高程参考 R\n{row['reference_cells']}格",
            f"删孔后评分域 H\n{row['hidden_reference_cells']}格")
        extent = [origin[0], origin[0]+20.8, origin[1], origin[1]+20.8]
        if family == "indoor":
            rr, cc = np.nonzero(example["before"] | reference)
            center_view = origin + .2*np.array([(cc.min()+cc.max()+1)/2, (rr.min()+rr.max()+1)/2])
            half_view = .2*max(np.ptp(rr)+1, np.ptp(cc)+1)/2 + .6
        for col, (field, title) in enumerate(zip(fields, titles)):
            ax = axes[i, col]
            cmap = colors.ListedColormap(palette if col == 2 else [palette[0], palette[1]])
            ax.imshow(field, origin="lower", extent=extent, cmap=cmap, interpolation="nearest", vmin=0, vmax=3 if col == 2 else 1)
            center = origin + 10.4
            ax.add_patch(Rectangle(center-.8, 1.6, 1.6, fill=False, edgecolor="#32383d", ls="--", lw=1.))
            ax.set_title(f"({chr(97+i*3+col)}) {title}", fontsize=9)
            ax.set_xlabel("局部 x（米）"); ax.set_ylabel("局部 y（米）")
            if family == "indoor":
                ax.set_xlim(center_view[0]-half_view, center_view[0]+half_view)
                ax.set_ylim(center_view[1]-half_view, center_view[1]+half_view)
            else:
                ax.set_xticks(center[0]+np.array([-10, 0, 10])); ax.set_yticks(center[1]+np.array([-10, 0, 10]))
            ax.set_aspect("equal")
    handles = [Patch(color=c, label=l) for c, l in zip(palette, (
        "无参考：不评分", "有观测且有参考：不计隐藏误差", "无观测但有参考：评分隐藏误差", "有观测但无参考：不评分"))]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, fontsize=8)
    fig.suptitle("输入稀疏不等于参考稀疏：右列只有橙色格评分隐藏高程\n"
        "黑框为1.6米人工孔洞；室内放大地板邻域，两类计数仍用20.8米全窗", fontsize=10)
    save(fig, output, "M1_scoring_domains")
