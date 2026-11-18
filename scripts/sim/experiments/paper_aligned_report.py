"""Audited paper-style analysis of the new terrain-budget and randomized-query campaigns."""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from pointcloud_batch import ROOT, atomic_json, digest
from plot_paper_scale import plt, save, clustered_summary, grouped_values, ratio_rows, FAMILIES
from systematic_study import style
from paper_clouds import raster
from paper_scale_evaluate import height_metrics, risk_metrics
from types import SimpleNamespace
from plot_style import wilson_interval

BASE = ROOT / "log/paper_aligned_20261015"
OUT = ROOT / "docs/reports/systematic_study/assets/paper03"
PROTOCOL = ROOT / "scripts/sim/experiments/protocols/paper_aligned_v1.json"
HELPERS = ("plot_paper_scale.py", "systematic_study.py", "plot_style.py", "thesis_statistics.py", "paper_scale_evaluate.py", "paper_clouds.py")
CONDITIONS = {"point15": "随机删去15%输入点", "hole16": "内部1.6米孔洞", "frontier50": "单侧隐藏50%宽度"}
BUDGETS = (50, 125, 250, 500)
PALETTE = ("#237a91", "#398268", "#ae6745", "#766299", "#aa8b32")
NAMES = {"turn":"转弯", "reverse":"倒车", "detour":"静态绕障", "corridor":"窄通道", "ramp":"纵坡",
    "cross_slope":"横坡", "waves":"起伏", "steep_ramp":"陡坡拒绝", "step":"台阶拒绝", "unknown":"未知拒绝"}


def read(path): return json.loads(path.read_text())


def paired_deltas(rows, metric, baseline=125, target=500):
    pairs = defaultdict(dict)
    for row in rows:
        if row["requested_M"] in (baseline, target):
            key = (row["dataset_id"], row["condition"])
            if row["requested_M"] in pairs[key]: raise ValueError("duplicate paired budget")
            pairs[key][row["requested_M"]] = row
    result = []
    for key, pair in sorted(pairs.items()):
        if set(pair) != {baseline, target}: raise ValueError("unpaired budget")
        a, b = pair[baseline], pair[target]
        if a["status"] == b["status"] == "ok" and a.get(metric) is not None and b.get(metric) is not None:
            result.append({"cluster_id": a["cluster_id"], "status": "ok", metric: b[metric]-a[metric]})
    return result


def validate_budget(record, catalog, allow_preflight=False):
    expected = {d["id"]+"__"+c+f"__M{m}" for d in catalog["datasets"] for c in CONDITIONS for m in BUDGETS}
    actual = [r["id"] for r in record["runs"]]
    if not allow_preflight and (len(actual) != len(expected) or set(actual) != expected):
        raise ValueError("incomplete or duplicate budget factorial")
    if len(actual) != len(set(actual)): raise ValueError("duplicate run")
    pairs = defaultdict(list)
    for r in record["runs"]: pairs[(r["dataset_id"],r["condition"])].append(r)
    for pair in pairs.values():
        if {r["requested_M"] for r in pair} != set(BUDGETS): raise ValueError("missing budget")
        for key in ("case_seed", "source_sha256", "input_points", "ground_input_points"):
            if len({r[key] for r in pair}) != 1: raise ValueError("unpaired input: "+key)
        good = [r for r in pair if r["status"] == "ok"]
        if len({r["reference_sha256"] for r in good}) > 1: raise ValueError("different reference across budgets")
        for r in good:
            if not 1 <= r["effective_M"] <= min(r["requested_M"], r["train_points"]): raise ValueError("inducing budget contract")
            if r["actual_device"] != "cuda" or r["actual_backend"] != "cupy": raise ValueError("GPU fallback")


def audit_arrays(record, output):
    chain = []
    for i,r in enumerate(record["runs"]):
        if r["status"] != "ok": continue
        p, q = output/r["archive"], output/r["reference"]
        if digest(p) != r["archive_sha256"] or digest(q) != r["reference_sha256"]: raise ValueError("array hash mismatch")
        with np.load(p) as a, np.load(q) as ref:
            pred=SimpleNamespace(mean=a["mean"],latent_var=a["variance"],accepted=a["accepted"])
            h=height_metrics(pred,ref["reference_height"].ravel(),ref["observed"].ravel())
            risk=risk_metrics(a["risk"],ref["reference_risk"],ref["unsafe"],ref["supported"])
            for key,value in {**h,**risk}.items():
                if value is None:
                    if r[key] is not None: raise ValueError("missing denominator became a score")
                elif not np.isclose(value,r[key],rtol=1e-6,atol=1e-7): raise ValueError("recomputed metric differs: "+key)
            if a["risk"].shape != ref["supported"].shape: raise ValueError("risk/support mismatch")
        chain.append(r["id"]+":"+r["archive_sha256"]+":"+r["reference_sha256"])
        if (i+1)%1000==0: print("[paper audit]",i+1,"arrays",flush=True)
    import hashlib
    return {"verified_prediction_arrays":len(chain),"chain_sha256":hashlib.sha256("\n".join(chain).encode()).hexdigest()}


def summarize_terrain(rows):
    keys=("hidden_rmse_m","hidden_mae_m","hidden_crps_m","hidden_picp95","hidden_width95_m",
        "hidden_candidate_coverage","local_hidden_fraction","risk_mae","risk_error_variance","risk_rmse",
        "fit_ms","predict_ms","risk_ms","total_wall_ms","effective_M","hidden_cells","reference_states")
    summary={}
    for c in CONDITIONS:
        summary[c]={}
        for m in BUDGETS:
            summary[c][str(m)]={}
            for f in FAMILIES:
                subset=[r for r in rows if r["condition"]==c and r["requested_M"]==m and r["family"]==f]
                stats={k:clustered_summary(subset,k) for k in keys}
                stats.update({k:clustered_summary(ratio_rows(subset),k) for k in ("false_safe","false_hazard")})
                stats["statuses"]=dict(Counter(r["status"] for r in subset))
                stats["zero_hidden_cases"]=sum(r.get("hidden_cells")==0 for r in subset)
                stats["zero_supported_cases"]=sum(r.get("reference_states")==0 for r in subset)
                summary[c][str(m)][f]=stats
    return summary


def lines(summary, metrics, labels, filename):
    fig,axes=plt.subplots(len(metrics),3,figsize=(11,3.05*len(metrics)),squeeze=False,layout="constrained")
    x=np.arange(4)
    for col,c in enumerate(CONDITIONS):
        for row,(metric,label,scale) in enumerate(zip(metrics,labels,[100 if k.endswith("_m") or k in ("hidden_picp95","false_safe","false_hazard","hidden_candidate_coverage","local_hidden_fraction") else 1 for k in metrics])):
            ax=axes[row,col]
            for f,color in zip(FAMILIES,PALETTE):
                s=[summary[c][str(m)][f][metric] for m in BUDGETS]
                y=np.array([t["mean"] if t["mean"] is not None else np.nan for t in s])*scale
                lo=np.array([t["low"] if t["low"] is not None else np.nan for t in s])*scale
                hi=np.array([t["high"] if t["high"] is not None else np.nan for t in s])*scale
                ax.plot(x,y,"o-",ms=3,color=color,label=FAMILIES[f]);ax.fill_between(x,lo,hi,color=color,alpha=.10)
            if metric=="hidden_picp95": ax.axhline(95,ls="--",lw=.9,color="#b44e4b")
            ax.set_xticks(x,list(map(str,BUDGETS)));ax.set_xlabel("请求的全局诱导点预算 M");ax.set_ylabel(label)
            ax.grid(axis="y",alpha=.2)
            if row==0: ax.set_title(CONDITIONS[c])
            if "coverage" in metric or metric in ("hidden_picp95","false_safe","false_hazard","local_hidden_fraction"): ax.set_ylim(0,102)
    handles,names=axes[0,0].get_legend_handles_labels();fig.legend(handles,names,loc="outside upper center",ncols=5,frameon=False)
    save(fig,OUT,filename)


def boxplot(ax, data, labels, colors):
    boxes=ax.boxplot(data,tick_labels=labels,patch_artist=True,widths=.58,
        flierprops={"marker":".","markersize":2,"alpha":.5},medianprops={"color":"#263943","linewidth":1.2})
    for patch,color in zip(boxes["boxes"],colors): patch.set_facecolor(color);patch.set_alpha(.55)
    ax.grid(axis="y",alpha=.2)


def terrain_figures(record,summary,catalog):
    rows=record["runs"]
    lines(summary,["hidden_rmse_m","hidden_mae_m"],["隐藏地面RMSE（厘米）","隐藏地面MAE（厘米）"],"T1_height")
    lines(summary,["hidden_picp95","hidden_width95_m"],["95%区间实际覆盖率（%）","95%区间总宽（厘米）"],"T2_uncertainty")
    lines(summary,["risk_mae","risk_error_variance"],["姿态风险分数MAE（无量纲）","风险分数误差方差（无量纲）"],"T3_risk")
    lines(summary,["false_safe","false_hazard"],["危险漏判率 FNR（%）","安全误拒率 FPR（%）"],"T4_decisions")
    lines(summary,["local_hidden_fraction","hidden_candidate_coverage"],["隐藏位置使用局部条件分支（%）","隐藏地面候选接受比例（%）"],"T5_gates")
    fig,axes=plt.subplots(1,5,figsize=(11,3.3),layout="constrained")
    for ax,(f,title),color in zip(axes,FAMILIES.items(),PALETTE):
        data=[grouped_values([r for r in rows if r["family"]==f and r["condition"]=="hole16" and r["requested_M"]==m],"hidden_rmse_m")*100 for m in BUDGETS]
        boxplot(ax,data,list(map(str,BUDGETS)),[color]*4);ax.set_title(title);ax.set_xlabel("诱导点预算 M");ax.set_ylabel("来源组RMSE（厘米）")
    save(fig,OUT,"T6_boxes")
    fig,axes=plt.subplots(2,2,figsize=(9,6),layout="constrained")
    timing={}
    for ax,key,title in zip(axes.ravel(),("fit_ms","predict_ms","risk_ms","total_wall_ms"),("GP拟合","全图预测（含局部分支）","BGK/SEB及碰撞层","后端整次墙钟")):
        timing[key]={}
        for q,ls,label in ((.5,"-","P50"),(.95,"--","P95")):
            values=[]
            for m in BUDGETS:
                a=[r[key] for r in rows if r["status"]=="ok" and r["requested_M"]==m]
                value=float(np.quantile(a,q));values.append(value);timing[key].setdefault(str(m),{})[label]=value
            ax.plot(np.arange(4),values,"o"+ls,color="#237a91" if q==.5 else "#ae6745",label=label)
        ax.set_xticks(np.arange(4),list(map(str,BUDGETS)));ax.set_xlabel("诱导点预算 M");ax.set_ylabel("墙钟耗时（毫秒）");ax.set_title(title);ax.grid(alpha=.2);ax.legend(frameon=False)
    save(fig,OUT,"T7_timing")
    fig,axes=plt.subplots(1,3,figsize=(11,3.5),layout="constrained")
    for ax,c in zip(axes,CONDITIONS):
        for (f,label),color in zip(FAMILIES.items(),PALETTE):
            points=[(summary[c][str(m)][f]["total_wall_ms"]["mean"],summary[c][str(m)][f]["hidden_rmse_m"]["mean"]) for m in BUDGETS]
            x=[p[0] for p in points];y=[p[1]*100 if p[1] is not None else np.nan for p in points]
            ax.plot(x,y,"o-",ms=4,color=color,label=label)
            for m,a,b in zip(BUDGETS,x,y):
                if b is not None and np.isfinite(b): ax.annotate(str(m),(a,b),fontsize=6,xytext=(3,3),textcoords="offset points")
        ax.set_xlabel("来源组平均整次耗时（毫秒）");ax.set_ylabel("隐藏RMSE（厘米）");ax.set_title(CONDITIONS[c]);ax.grid(alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(),loc="outside upper center",ncols=5,frameon=False)
    save(fig,OUT,"T8_tradeoff")
    fig,axes=plt.subplots(2,3,figsize=(11,6),layout="constrained")
    for col,c in enumerate(CONDITIONS):
        selected=[[r for r in rows if r["family"]==f and r["condition"]==c and r["requested_M"]==125 and r["status"]=="ok"] for f in FAMILIES]
        for row,key,title in ((0,"hidden_cells","每输入可评分隐藏格数"),(1,"reference_states","每输入完整足迹参考状态数")):
            boxplot(axes[row,col],[[r[key] for r in ss] for ss in selected],list(FAMILIES.values()),PALETTE)
            axes[row,col].set_ylabel(title);axes[row,col].tick_params(axis="x",rotation=30)
            if row==0: axes[row,col].set_title(CONDITIONS[c])
    save(fig,OUT,"T9_denominators")
    fig,ax=plt.subplots(figsize=(7,3.5),layout="constrained")
    for f,color in zip(FAMILIES,PALETTE):
        vals=[summary["point15"][str(m)][f]["effective_M"]["mean"] for m in BUDGETS]
        ax.plot(BUDGETS,vals,"o-",color=color,label=FAMILIES[f])
    ax.plot([50,500],[50,500],"--",lw=1,color="#777777",label="实际=请求")
    ax.set_xlabel("请求诱导点预算 M");ax.set_ylabel("实际使用诱导点数");ax.grid(alpha=.2);ax.legend(ncols=3,frameon=False)
    save(fig,OUT,"T10_effective_budget")
    return timing


def representative_figures(record,catalog):
    selected={}
    for family in FAMILIES:
        candidates=sorted([r for r in record["runs"] if r["family"]==family and r["condition"]=="hole16" and r["requested_M"]==125
            and r["status"]=="ok" and r["hidden_cells"]>=20],key=lambda r:r["dataset_id"])
        if not candidates: continue
        row=candidates[0];selected[family]={"id":row["id"],"eligible_inputs":len(candidates),"selection":"first lexicographic ID with >=20 hidden reference cells, not lowest error"}
        item=next(d for d in catalog["datasets"] if d["id"]==row["dataset_id"])
        source=(BASE/"catalog.json").parent/item["path"]
        with np.load(source) as a:
            original=raster(a["xyz"][a["is_ground"]],tuple(a["shape"]),a["query_origin"],.2)
        with np.load(BASE/"budget"/row["reference"]) as a:
            observed=a["measured"].copy();ref=a["reference_height"].copy();origin=a["query_origin"].copy()
        with np.load(BASE/"budget"/row["archive"]) as a: pred=a["mean"].reshape(ref.shape);std=np.sqrt(a["variance"].reshape(ref.shape))
        if np.any(np.isfinite(observed)&~np.isfinite(original)): raise ValueError("deleted input gained observed cells")
        hidden=np.isfinite(ref)&~np.isfinite(observed)
        extent=[origin[0],origin[0]+20.8,origin[1],origin[1]+20.8]
        lo,hi=np.nanpercentile(np.r_[original.ravel(),ref.ravel()],[1,99])
        fig,axes=plt.subplots(2,3,figsize=(10,6.1),layout="constrained")
        fields=(original,observed,ref,pred,np.where(hidden,(pred-ref)*100,np.nan),std*100)
        titles=("(a) 缺测前输入地面格", "(b) 删孔后输入（加2厘米噪声）", "(c) 独立高程评分参考",
            "(d) GP全图均值（不等于放行）", "(e) 仅隐藏参考处误差", "(f) GP标准差（非误差）")
        for i,(ax,field,title) in enumerate(zip(axes.ravel(),fields,titles)):
            options=dict(cmap="terrain",vmin=lo,vmax=hi) if i<4 else dict(cmap="RdBu_r",vmin=-30,vmax=30) if i==4 else dict(cmap="magma",vmin=0,vmax=np.nanpercentile(std*100,98))
            im=ax.imshow(field,origin="lower",extent=extent,**options);ax.set_title(title,fontsize=9)
            ax.set_xlabel("局部X（米）");ax.set_ylabel("局部Y（米）")
            fig.colorbar(im,ax=ax,shrink=.8,label="高程（米）" if i<4 else "预测−参考（厘米）" if i==4 else "标准差（厘米）")
        fig.suptitle(FAMILIES[family]+"：输入删减与评分支撑分开显示；白色=无值，不代表安全",fontsize=11)
        save(fig,OUT,"V_"+family)
        if family in ("hilly","ruin"):
            fig=plt.figure(figsize=(9,6.5),layout="constrained")
            xs=origin[0]+(np.arange(104)+.5)*.2;ys=origin[1]+(np.arange(104)+.5)*.2;xx,yy=np.meshgrid(xs[::2],ys[::2])
            for k,(field,title) in enumerate(zip((original,observed,pred,np.where(hidden,np.abs(pred-ref)*100,np.nan)),
                    ("缺测前输入地面","删孔后输入地面","GP预测地面","隐藏参考处绝对误差"))):
                ax=fig.add_subplot(2,2,k+1,projection="3d")
                zz=field[::2,::2]
                # Plot observed supports only; do not triangulate across missing cells.
                im=ax.scatter(xx.ravel(),yy.ravel(),(zz if k<3 else ref[::2,::2]).ravel(),c=zz.ravel(),s=2,
                    cmap="terrain" if k<3 else "magma",vmin=lo if k<3 else 0,vmax=hi if k<3 else 30,rasterized=True)
                ax.set_title(title);ax.set_xlabel("X（米）",labelpad=0);ax.set_ylabel("Y（米）",labelpad=0);ax.set_zlabel("Z（米）",labelpad=0)
                ax.view_init(30,-60);ax.set_box_aspect((1,1,.45));fig.colorbar(im,ax=ax,shrink=.55,pad=.05,label="高程（米）" if k<3 else "误差（厘米）")
            save(fig,OUT,"V3_"+family)
    return selected


def validate_queries(plan,record):
    expected={(r["id"],i) for r in plan["cases"] for i in range(plan["settings"]["timing_repeats"])}
    actual=[(r["id"],r["repeat"]) for r in record["runs"]]
    if len(actual)!=len(expected) or set(actual)!=expected or record["settings"]!=plan["settings"]: raise ValueError("incomplete or changed query plan")


def path_quality(path):
    points=np.asarray(path,dtype=float)
    if len(points)<2:return 0.,None,0
    d=np.diff(points[:,:2],axis=0);length=np.linalg.norm(d,axis=1)
    delta=(np.diff(points[:,2])+np.pi)%(2*np.pi)-np.pi
    same=points[:-1,3]==points[1:,3];valid=same&(length>1e-8)
    curvature=float(np.abs(delta[valid]).sum()/length[valid].sum()) if valid.any() else None
    return float(length.sum()),curvature,int((~same).sum())


def query_analysis(plan,record):
    validate_queries(plan,record);grouped=defaultdict(list);summaries={};examples={}
    for r in record["runs"]:
        grouped[r["id"]].append(r)
        if r["status"]=="ok":
            length,curvature,gears=path_quality(r["path"])
            if not np.isclose(length,r["path_length_m"]) or gears!=r["gear_switches"]: raise ValueError("path metric discrepancy")
            if curvature is None:
                if r["mean_abs_curvature_per_m"] is not None: raise ValueError("null curvature denominator")
            elif not np.isclose(curvature,r["mean_abs_curvature_per_m"]): raise ValueError("curvature discrepancy")
    queries=[]
    for case in plan["cases"]:
        reps=grouped[case["id"]];ok=[r for r in reps if r["status"]=="ok"]
        # A query is robustly complete only if all timing repetitions pass the same check.
        q={"id":case["id"],"scenario":case["scenario"],"layout_id":case["layout_id"],"expected_stop":case["expected_stop"],
            "eligible":len(ok)==len(reps),"all_returned":len(ok)==len(reps) and all(r["returned"] for r in ok),
            "all_safe":len(ok)==len(reps) and all(r["independent_path_safe"] for r in ok),
            "complete_safe":len(ok)==len(reps) and all(r["returned"] and r["independent_path_safe"] and r["full_goal_connector"] for r in ok),
            "unsafe_return":any(r["returned"] and not r["independent_path_safe"] for r in ok),
            "boundary_respected":len(ok)==len(reps) and all(not r["returned"] or r["independent_path_safe"] for r in ok),
            "median_wall_ms":float(np.median([r["planner_wall_ms"] for r in ok])) if ok else None,
            "path_length_m":float(np.median([r["path_length_m"] for r in ok if r["full_goal_connector"] and r["independent_path_safe"]])) if any(r["full_goal_connector"] and r["independent_path_safe"] for r in ok) else None,
            "mean_abs_curvature_per_m":float(np.median([r["mean_abs_curvature_per_m"] for r in ok if r["full_goal_connector"] and r["independent_path_safe"] and r["mean_abs_curvature_per_m"] is not None])) if any(r["full_goal_connector"] and r["independent_path_safe"] and r["mean_abs_curvature_per_m"] is not None for r in ok) else None}
        queries.append(q)
    for f in NAMES:
        cases=[q for q in queries if q["scenario"]==f];calls=[r for r in record["runs"] if r["scenario"]==f and r["status"]=="ok"]
        times=[r["planner_wall_ms"] for r in calls];eligible=sum(q["eligible"] for q in cases)
        summaries[f]={"queries":len(cases),"eligible":eligible,"unique_layouts":len({q["layout_id"] for q in cases}),"calls":len(calls),
            "complete_safe":sum(q["complete_safe"] for q in cases),"returned_all":sum(q["all_returned"] for q in cases),
            "unsafe_queries":sum(q["unsafe_return"] for q in cases),"boundary_respected":sum(q["boundary_respected"] for q in cases),
            "wall_p50_ms":float(np.quantile(times,.5)) if times else None,"wall_p95_ms":float(np.quantile(times,.95)) if times else None,
            "over90ms":sum(t>90 for t in times),"over200ms":sum(t>200 for t in times),
            "complete_safe_interval":wilson_interval(sum(q["complete_safe"] for q in cases),len(cases))}
        # First ID, rather than the shortest or fastest path.
        first=sorted([c for c in plan["cases"] if c["scenario"]==f],key=lambda c:c["id"])[0]
        examples[f]={"case":first,"run":next(r for r in grouped[first["id"]] if r["repeat"]==0)}
    return summaries,queries,examples


def query_figures(summaries,queries,record,examples):
    fig,axes=plt.subplots(1,2,figsize=(11,3.8),layout="constrained")
    for ax,group,reject in zip(axes,(list(NAMES)[:7],list(NAMES)[7:]),(False,True)):
        vals=[summaries[f]["boundary_respected" if reject else "complete_safe"]/summaries[f]["queries"]*100 for f in group]
        lohi=[wilson_interval(summaries[f]["boundary_respected" if reject else "complete_safe"],summaries[f]["queries"]) for f in group]
        ax.bar(np.arange(len(group)),vals,color=PALETTE[0] if not reject else PALETTE[2],width=.6)
        ax.errorbar(np.arange(len(group)),vals,yerr=np.array([[v-p[0]*100 for v,p in zip(vals,lohi)],[p[1]*100-v for v,p in zip(vals,lohi)]]),fmt="none",color="#33434a",capsize=3)
        ax.set_xticks(np.arange(len(group)),[NAMES[f] for f in group],rotation=25);ax.set_ylim(0,108);ax.set_ylabel("查询比例（%）")
        ax.set_title("到达类：三次均完整连接且独立检查安全" if not reject else "拒绝类：未返回越界路径（不等于到达）")
        for i,v in enumerate(vals): ax.text(i,max(4,v-12),f"{v:.0f}",ha="center",color="white" if v>20 else "#333333",fontsize=9)
        ax.grid(axis="y",alpha=.15)
    save(fig,OUT,"Q1_feasibility")
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
    times=[[q["median_wall_ms"] for q in queries if q["scenario"]==f and q["median_wall_ms"] is not None] for f in NAMES]
    boxplot(axes[0],times,list(NAMES.values()),[PALETTE[i%5] for i in range(10)])
    axes[0].set_yscale("log");axes[0].set_ylabel("每查询三次计时中位数（毫秒）");axes[0].tick_params(axis="x",rotation=50)
    axes[0].set_title("固定查询的计时分布（非三倍样本量）")
    for f,color in zip(list(NAMES)[:7],list(PALETTE)+["#547da4","#7c8355"]):
        a=np.sort(times[list(NAMES).index(f)]);axes[1].plot(a,np.arange(1,len(a)+1)/len(a),label=NAMES[f],color=color)
    axes[1].set_xlabel("整次规划调用墙钟（毫秒）");axes[1].set_ylabel("查询累计比例");axes[1].set_xscale("log");axes[1].grid(alpha=.2);axes[1].legend(frameon=False,ncols=2)
    save(fig,OUT,"Q2_latency")
    fig,axes=plt.subplots(1,2,figsize=(11,3.8),layout="constrained")
    group=list(NAMES)[:7]
    for ax,key,label in zip(axes,("path_length_m","mean_abs_curvature_per_m"),("完整安全返回路径长度（米）","离散航向总变差/弧长（1/米）")):
        data=[[q[key] for q in queries if q["scenario"]==f and q["complete_safe"] and q[key] is not None] for f in group]
        boxplot(ax,data,[NAMES[f]+f"\n(n={len(d)})" for f,d in zip(group,data)],[PALETTE[i%5] for i in range(7)])
        ax.set_ylabel(label);ax.tick_params(axis="x",rotation=25)
    save(fig,OUT,"Q3_quality")
    fig,axes=plt.subplots(2,3,figsize=(10,6.4),layout="constrained")
    for ax,f in zip(axes.ravel(),("turn","reverse","detour","corridor","step","unknown")):
        item=examples[f];case=item["case"];run=item["run"]
        for b in case["obstacles"]:
            from matplotlib.patches import Rectangle
            ax.add_patch(Rectangle((b["x"]-b["hx"],b["y"]-b["hy"]),2*b["hx"],2*b["hy"],color="#75838a",alpha=.7))
        if case["expected_stop"]: ax.axvspan(3.5,9,color="#c87567",alpha=.22,label="边界另一侧")
        path=np.asarray(run.get("path",[]))
        if len(path):ax.plot(path[:,0],path[:,1],color=PALETTE[0],lw=1.8,label="实际返回路径")
        ax.scatter(*case["start"][:2],c=PALETTE[1],marker="o",s=25,label="起点");ax.scatter(*case["goal"],c=PALETTE[2],marker="*",s=65,label="目标")
        ax.set_xlim(-4,9);ax.set_ylim(-3.5,3.5);ax.set_aspect("equal");ax.grid(alpha=.2);ax.set_xlabel("X（米）");ax.set_ylabel("Y（米）")
        ax.set_title(NAMES[f]+f"：{run.get('planner_wall_ms',0):.1f}毫秒",fontsize=9)
    axes[0,0].legend(frameon=False,fontsize=7,loc="upper left");save(fig,OUT,"Q4_paths")


def fmt(value,digits=2): return "--" if value is None else f"{value:.{digits}f}"


def export_tables(summary,timing,qs,record):
    rows=[]
    for family in FAMILIES:
        for c in CONDITIONS:
            s=summary[c]["125"][family]
            rows.append(" & ".join([FAMILIES[family],CONDITIONS[c].replace("输入", "").replace("%",r"\%"),
                str(s["hidden_rmse_m"]["cases"]),str(s["hidden_rmse_m"]["clusters"]),fmt(None if s["hidden_rmse_m"]["mean"] is None else s["hidden_rmse_m"]["mean"]*100),
                fmt(None if s["hidden_picp95"]["mean"] is None else s["hidden_picp95"]["mean"]*100,1),fmt(None if s["hidden_width95_m"]["mean"] is None else s["hidden_width95_m"]["mean"]*100),
                fmt(s["risk_mae"]["mean"],3),fmt(None if s["false_safe"]["mean"] is None else s["false_safe"]["mean"]*100,1)])+r" \\")
    (OUT/"terrain_rows.tex").write_text("\n".join(rows)+"\n")
    rows=[]
    for m in BUDGETS:
        rs=[r for r in record["runs"] if r["requested_M"]==m and r["status"]=="ok"]
        rows.append(" & ".join([str(m),str(len(rs)),str(min(r["effective_M"] for r in rs))+"--"+str(max(r["effective_M"] for r in rs))]+[fmt(timing[k][str(m)][q],1) for k,q in (("fit_ms","P50"),("predict_ms","P50"),("risk_ms","P50"),("total_wall_ms","P50"),("total_wall_ms","P95"))])+r" \\")
    (OUT/"timing_rows.tex").write_text("\n".join(rows)+"\n")
    rows=[]
    for f,label in NAMES.items():
        s=qs[f];rows.append(" & ".join([label,str(s["queries"]),str(s["unique_layouts"]),str(s["complete_safe"]),str(s["unsafe_queries"]),fmt(s["wall_p50_ms"],1),fmt(s["wall_p95_ms"],1)])+r" \\")
    (OUT/"query_rows.tex").write_text("\n".join(rows)+"\n")
    rows=[]
    for c in CONDITIONS:
        for f,label in FAMILIES.items():
            s=summary[c]["125"][f];rows.append(" & ".join([label,CONDITIONS[c].replace("%",r"\%"),str(s["zero_hidden_cases"]),str(s["zero_supported_cases"]),str(s["false_safe"]["clusters"]),str(s["false_hazard"]["clusters"])])+r" \\")
    (OUT/"denominator_rows.tex").write_text("\n".join(rows)+"\n")
    paired=[]
    for f,label in FAMILIES.items():
        chosen=[r for r in record["runs"] if r["family"]==f and r["condition"]=="hole16"]
        values=[]
        for key,scale in (("hidden_rmse_m",100),("risk_mae",1),("total_wall_ms",1)):
            s=clustered_summary(paired_deltas(chosen,key),key)
            values.append("--" if s["mean"] is None else f"{s['mean']*scale:.3f} [{s['low']*scale:.3f}, {s['high']*scale:.3f}]")
        paired.append(" & ".join([label,*values])+r" \\")
    (OUT/"paired_rows.tex").write_text("\n".join(paired)+"\n")
    counts={"BudgetRuns":len(record["runs"]),"BudgetOK":sum(r["status"]=="ok" for r in record["runs"]),
        "QueryCount":sum(s["queries"] for s in qs.values()),"QueryCalls":sum(s["calls"] for s in qs.values()),
        "QueryLayouts":len({r["layout_id"] for r in read(BASE/"planning/cases.json")["cases"]}),
        "QueryComplete":sum(s["complete_safe"] for f,s in qs.items() if f not in ("steep_ramp","step","unknown")),
        "QueryUnsafe":sum(s["unsafe_queries"] for s in qs.values()),
        "QueryReject":sum(s["boundary_respected"] for f,s in qs.items() if f in ("steep_ramp","step","unknown"))}
    (OUT/"counts.tex").write_text("\n".join("\\newcommand{\\"+key+"}{"+str(value)+"}" for key,value in counts.items())+"\n")


def csv_rows(path,rows):
    keys=sorted(set().union(*(r.keys() for r in rows)))
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--verify-assets",action="store_true")
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    inputs={"protocol":PROTOCOL,"catalog":BASE/"catalog.json","terrain":BASE/"budget/results.json",
        "query_plan":BASE/"planning/cases.json","queries":BASE/"planning/runs/results.json"}
    hashes={str(p.relative_to(ROOT)):digest(p) for p in inputs.values()}
    hashes[str(Path(__file__).relative_to(ROOT))]=digest(Path(__file__))
    hashes.update({str((Path(__file__).parent/n).relative_to(ROOT)):digest(Path(__file__).parent/n) for n in HELPERS})
    manifest=OUT/"manifest.json"
    if args.verify_assets:
        old=read(manifest)
        if old["inputs"]!=hashes:raise ValueError("report inputs changed")
        for p,h in old["outputs"].items():
            if digest(OUT/p)!=h:raise ValueError("report asset changed")
        print("[paper report] frozen input and output hashes verified");return
    data={k:read(p) for k,p in inputs.items()};record=data["terrain"];catalog=data["catalog"]
    if record["settings"]["dataset_sha256"]!=digest(inputs["catalog"]):raise ValueError("catalog changed")
    for name,h in record["settings"]["source_sha256"].items():
        if digest(ROOT/name)!=h:raise ValueError("terrain source changed: "+name)
    validate_budget(record,catalog);audit=audit_arrays(record,BASE/"budget")
    plan=data["query_plan"]
    if plan["settings"]["protocol_sha256"]!=digest(PROTOCOL):raise ValueError("query protocol changed")
    for case in plan["cases"]:
        if digest(BASE/"planning"/case["map_file"])!=case["map_sha256"]:raise ValueError("map changed")
    for name,h in plan["settings"]["compiled_source_sha256"].items():
        if digest(ROOT/name)!=h:raise ValueError("planner source changed")
    for name,h in plan["settings"]["source_sha256"].items():
        if digest(ROOT/name)!=h:raise ValueError("query map generator changed")
    summary=summarize_terrain(record["runs"]);qs,queries,examples=query_analysis(plan,data["queries"])
    style();timing=terrain_figures(record,summary,catalog);selections=representative_figures(record,catalog)
    query_figures(qs,queries,data["queries"],examples)
    paired={c:{f:{key:clustered_summary(paired_deltas([r for r in record["runs"] if r["condition"]==c and r["family"]==f],key),key)
        for key in ("hidden_rmse_m","risk_mae","total_wall_ms")} for f in FAMILIES} for c in CONDITIONS}
    result={"study_id":"PTR-PAPER-03","new_terrain_computations":len(record["runs"]),"statuses":dict(Counter(r["status"] for r in record["runs"])),
        "source_groups":catalog["groups"],"audit":audit,"terrain":summary,"timing":timing,"paired_M500_minus_M125":paired,
        "query_calls":len(data["queries"]["runs"]),"queries":qs,"unique_query_layouts":len({c["layout_id"] for c in plan["cases"]}),
        "representative_selection":selections,"query_statuses":dict(Counter(r["status"] for r in data["queries"]["runs"]))}
    atomic_json(OUT/"results.json",result);export_tables(summary,timing,qs,record)
    csv_rows(OUT/"terrain_runs.csv",record["runs"]);csv_rows(OUT/"query_summary.csv",queries)
    csv_rows(OUT/"planning_calls.csv",[{k:v for k,v in r.items() if k!="path"} for r in data["queries"]["runs"]])
    required={"T"+str(i)+"_"+n for i,n in enumerate(("height","uncertainty","risk","decisions","gates","boxes","timing","tradeoff","denominators","effective_budget"),1)}
    required|={"Q1_feasibility","Q2_latency","Q3_quality","Q4_paths"}|{"V_"+f for f in FAMILIES}|{"V3_hilly","V3_ruin"}
    if not all((OUT/(n+".pdf")).exists() for n in required):raise ValueError("missing protocol figure")
    outputs={p.name:digest(p) for p in OUT.iterdir() if p.is_file() and p.name!="manifest.json"}
    atomic_json(manifest,{"inputs":hashes,"outputs":outputs,"selection":selections,"audit":audit})
    print("[paper report] complete",len([p for p in outputs if p.endswith('.pdf')]),"figures",flush=True)


if __name__=="__main__":main()
