#!/usr/bin/env python3
"""Chinese figures for frozen terrain and native closed-loop campaigns.

Only completed plans qualify. Isaac/DDS measurements remain a separate layer.
"""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from plot_paper_scale import (FAMILIES, clustered_summary, ratio_rows, save,
                             plt, font_manager, source_views)
from plot_style import configure_chinese, wilson_interval
from pointcloud_batch import atomic_json, digest
import numpy as np
from matplotlib.patches import Rectangle

CONDITIONS={"point15":"删点15%","point30":"删点30%","point60":"删点60%",
    "hole08":"0.8米孔洞","hole16":"1.6米孔洞","hole32":"3.2米孔洞",
    "frontier25":"隐藏右侧25%","frontier50":"隐藏右侧50%"}
SCENARIOS={"straight":"直行","turn":"转弯","reverse":"近距倒车","detour":"静态绕障",
    "corridor":"窄通道","ramp":"纵向缓坡","cross_slope":"横向斜坡","waves":"连续起伏",
    "steep_ramp":"陡坡拒绝","step":"台阶拒绝","unknown":"未知边界拒绝","emerge":"突现障碍绕行",
    "crossing":"移动横穿","stale_map":"地图过期停车"}
COLORS=["#217c91","#bd6547","#59926a","#776b91","#a89935"]


def validate_terrain(record,catalog):
    sources={d["id"] for d in catalog["datasets"]}
    conditions=record["settings"]["conditions"]
    expected={i+"__"+c for i in sources for c in conditions}
    actual=[r["id"] for r in record["runs"]]
    if (set(record["settings"]["selected_ids"])!=sources or set(conditions)!=set(CONDITIONS)
            or len(actual)!=len(expected) or set(actual)!=expected
            or record["planned_cases"]!=len(expected)):
        raise ValueError("incomplete expanded terrain plan; refuse final statistics")


def validate_navigation(plan,record):
    expected=[c["id"] for c in plan["cases"]];actual=[r["id"] for r in record["runs"]]
    if (len(expected)!=len(set(expected)) or len(actual)!=len(expected) or set(actual)!=set(expected)
            or record["settings"]!=plan["settings"]):
        raise ValueError("incomplete native navigation plan; refuse final statistics")


def heatmap(ax,stats,key,factor,title,limit=None):
    data=np.array([[stats[f][c][key]["mean"] if stats[f][c][key]["mean"] is not None else np.nan
                    for c in CONDITIONS] for f in FAMILIES])*factor
    im=ax.imshow(data,aspect="auto",cmap="viridis",vmin=0,vmax=limit)
    ax.set_xticks(range(8),list(CONDITIONS.values()),rotation=40,ha="right",fontsize=8)
    ax.set_yticks(range(5),list(FAMILIES.values()));ax.set_title(title)
    ax.set_xlabel("人工缺测方式（先删点，再计算模型）")
    for (i,j),v in np.ndenumerate(data):
        if np.isfinite(v):ax.text(j,i,f"{v:.1f}" if factor==100 else f"{v:.3f}",ha="center",va="center",fontsize=8,
            color="white" if im.norm(v)<.52 else "black")
    plt.colorbar(im,ax=ax,shrink=.75)


def terrain_figures(record,catalog,output):
    validate_terrain(record,catalog)
    metrics=("risk_mae","risk_rmse","risk_error_variance","hidden_rmse_m","accepted_hidden_rmse_m","hidden_picp95",
             "hidden_width95_m","hidden_candidate_coverage")
    stats={}
    for f in FAMILIES:
        stats[f]={}
        for c in CONDITIONS:
            rows=[r for r in record["runs"] if r["family"]==f and r["condition"]==c]
            stat={k:clustered_summary(rows,k) for k in metrics}
            ratios=ratio_rows(rows)
            stat.update({k:clustered_summary(ratios,k) for k in ("recall","false_safe","false_hazard")})
            stat["statuses"]=dict(Counter(r["status"] for r in rows));stats[f][c]=stat
    fig,axes=plt.subplots(1,3,figsize=(16,5),layout="constrained")
    heatmap(axes[0],stats,"risk_mae",1,"通行风险MAE（无量纲，越低越好）",1)
    heatmap(axes[1],stats,"risk_rmse",1,"通行风险RMSE（无量纲，越低越好）",1)
    heatmap(axes[2],stats,"risk_error_variance",1,"风险误差经验方差（非GP方差）")
    fig.suptitle("地形核心测试：128位置 × 36航向的足迹风险偏差\n每格数值为来源组等权均值，不是高程误差或实车通过概率")
    save(fig,output,"01_terrain_risk")
    fig,axes=plt.subplots(3,1,figsize=(11,13),layout="constrained")
    for ax,k,title in zip(axes,("recall","false_safe","false_hazard"),
        ("危险召回 TP/(TP+FN)（%，越高越好）","危险漏判 FN/(TP+FN)（%，越低越好）",
         "安全参考被拒绝 FP/(FP+TN)（%，越低越好）")):
        heatmap(ax,stats,k,100,title,100)
    fig.suptitle("风险分类阈值0.58：安全识别与过度保守必须同时看\n未知区域保留0.65风险；高召回不等于能够完成导航")
    save(fig,output,"02_terrain_safety")
    fig,axes=plt.subplots(2,2,figsize=(13,10),layout="constrained")
    for ax,k,factor,title,limit in zip(axes.flat,
        ("hidden_rmse_m","hidden_picp95","hidden_width95_m","hidden_candidate_coverage"),
        (100,100,100,100),
        ("无输入地面格子：高程RMSE（厘米）","名义95%区间覆盖参考比例（%）",
         "名义95%区间平均宽度（厘米）","隐藏格子通过距离/方差门限（%）"),(None,100,None,100)):
        heatmap(ax,stats,k,factor,title,limit)
    fig.suptitle("辅助测试：隐藏高程与不确定性\n全部隐藏格子参与误差；候选不是有效地面，也不是安全可通过的证明")
    save(fig,output,"03_hidden_elevation")
    timing={}
    fig,axes=plt.subplots(1,3,figsize=(13,4.5),layout="constrained")
    for ax,key,title in zip(axes,("fit_ms","predict_ms","risk_ms"),
        ("观测特征与GP拟合","10816格查询预测","当前柱状占据和BGK/SEB")):
        timing[key]={}
        for f,color in zip(FAMILIES,COLORS):
            values=np.sort([r[key] for r in record["runs"] if r["family"]==f and r["status"]=="ok"])
            if not len(values):continue
            timing[key][f]={"n":len(values),"p50_ms":float(np.median(values)),"p95_ms":float(np.quantile(values,.95))}
            ax.plot(values,np.arange(1,len(values)+1)/len(values),label=FAMILIES[f],color=color)
        ax.set_xscale("log");ax.set_xlabel("组件墙钟耗时（毫秒，对数刻度）")
        ax.set_ylabel("累计案例比例");ax.set_title(title);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("实测CUDA/CuPy组件速度：包含初始化，不含ROS、仿真或参考评分\n邻域索引由CPU构建，局部GP矩阵求解与SEB批量计算使用GPU")
    save(fig,output,"04_terrain_timing")
    return {"statistics":stats,"timing":timing,"statuses":dict(Counter(r["status"] for r in record["runs"])),
        "actual_devices":dict(Counter(r.get("actual_device","none") for r in record["runs"])),
        "actual_backends":dict(Counter(r.get("actual_backend","none") for r in record["runs"])),
        "source_count":len(catalog["datasets"]),"runs":len(record["runs"]),
        "source_groups":{f:len({d["cluster_id"] for d in catalog["datasets"] if d["family"]==f}) for f in FAMILIES},
        "splits":dict(Counter(d.get("split","unspecified") for d in catalog["datasets"]))}


def spatial_prediction_views(results,output):
    fig,axes=plt.subplots(3,4,figsize=(14,11),layout="constrained")
    for row,kind in enumerate(("point60","hole32","frontier50")):
        with np.load(results.parent/("forest__"+kind+"__snapshot.npz")) as a:
            origin=a["query_origin"];extent=[origin[0],origin[0]+20.8,origin[1],origin[1]+20.8]
            valid=a["elevation"];mean=a["mean"];ref=a["reference"]
            fields=(valid,mean,np.abs(mean-ref),a["risk"])
            low,high=np.nanmin(ref),np.nanmax(ref)
            for col,(field,title,cmap) in enumerate(zip(fields,
                ("实际输入地面高程（米）","模型预测高程（米）","相对完整参考的绝对误差（米）","航向0号切片风险（无量纲）"),
                ("terrain","terrain","magma","RdYlGn_r"))):
                ax=axes[row,col]
                limits={"vmin":low,"vmax":high} if col<2 else ({"vmin":0,"vmax":1} if col==3 else {"vmin":0})
                im=ax.imshow(field,origin="lower",extent=extent,cmap=cmap,**limits)
                ax.set_title(title,fontsize=10);ax.set_xlabel("局部 x（米）");ax.set_ylabel("局部 y（米）")
                plt.colorbar(im,ax=ax,shrink=.75)
                if col==0:ax.text(.02,.02,CONDITIONS[kind],transform=ax.transAxes,fontsize=10,
                    bbox={"facecolor":"white","alpha":.8,"edgecolor":"none"})
    fig.suptitle("同一预定森林样例：输入、预测、完整参考误差与规划风险不是一回事\n无输入处为空白；预测图有值不代表风险图将其标为可通过；全部参数在评分前冻结")
    save(fig,output,"10_spatial_prediction")


def navigation_figures(plan,record,output):
    validate_navigation(plan,record)
    rows=record["runs"];cases={c["id"]:c for c in plan["cases"]}
    stats={};names=list(SCENARIOS)
    for name in names:
        selected=[r for r in rows if r["scenario"]==name];ok=sum(r["status"]=="passed" for r in selected)
        lo,hi=wilson_interval(ok,len(selected))
        stats[name]={"n":len(selected),"qualified":ok,"qualified_rate":ok/len(selected),"wilson95":[lo,hi],
            "expected_stop":cases[selected[0]["id"]]["expected_stop"],
            "collisions":sum(r.get("collision",False) for r in selected),
            "terrain_violations":sum(r.get("unsafe_terrain",False) for r in selected),
            "unknown_violations":sum(r.get("unknown_policy_violation",False) for r in selected),
            "arrivals":sum(r.get("reached",False) for r in selected),
            "errors":sum(r["status"]=="error" for r in selected)}
        for key in ("planner_p95_ms","control_p95_ms","tracking_p95_m","path_length_m","goal_error_m"):
            values=[r[key] for r in selected if key in r]
            stats[name][key]=float(np.median(values)) if values else None
    fig,ax=plt.subplots(figsize=(13,5.5),layout="constrained")
    x=np.arange(len(names));means=[stats[n]["qualified_rate"]*100 for n in names]
    errors=np.maximum(0,np.array([[stats[n]["qualified_rate"]-stats[n]["wilson95"][0] for n in names],
          [stats[n]["wilson95"][1]-stats[n]["qualified_rate"] for n in names]])*100)
    ax.bar(x,means,yerr=errors,capsize=3,color=["#bd6547" if stats[n]["expected_stop"] else "#217c91" for n in names])
    for i,n in enumerate(names):ax.text(i,3,f"{stats[n]['qualified']}/{stats[n]['n']}",ha="center",va="bottom",rotation=90,fontsize=8)
    ax.set_xticks(x,[SCENARIOS[n] for n in names],rotation=35,ha="right");ax.set_ylim(0,108)
    ax.set_ylabel("闭环任务合格率及Wilson 95%区间（%）");ax.grid(axis="y",alpha=.2)
    ax.set_title("实际生产规划/控制回调 + 一阶自行车模型\n蓝色：无碰撞到达并停车；棕色：拒绝危险任务并停车，不算导航到达")
    save(fig,output,"05_navigation_qualification")
    fig,axes=plt.subplots(1,3,figsize=(13,4.5),layout="constrained")
    for ax,key,title in zip(axes,("collisions","terrain_violations","unknown_violations"),
        ("独立矩形碰撞检查","独立密集足迹坡度/台阶越限","车体前缘进入未知区域")):
        counts=[stats[n][key] for n in names];ax.bar(x,counts,color="#bd6547")
        ax.set_xticks(x,[SCENARIOS[n] for n in names],rotation=60,ha="right",fontsize=7)
        ax.set_ylabel("违规案例数（次）");ax.set_title(title);ax.grid(axis="y",alpha=.2)
        ax.set_ylim(0,max(1,max(counts))*1.15)
    fig.suptitle("安全违规按案例计数：不能用到达率掩盖碰撞或跨越未知边界")
    save(fig,output,"06_navigation_safety")
    for part in range(4):
        selected_names=names[part*4:(part+1)*4]
        fig,axes=plt.subplots(1,len(selected_names),figsize=(4*len(selected_names),4.5),squeeze=False,layout="constrained")
        for ax,name in zip(axes.flat,selected_names):
            selected=[r for r in rows if r["scenario"]==name];representative=selected[0];case=cases[representative["id"]]
            for row in selected:
                trace=np.asarray(row.get("trajectory",[]))
                if len(trace):ax.plot(trace[:,1],trace[:,2],color="#217c91" if row["status"]=="passed" else "#bd6547",alpha=.15,lw=.7)
            for b in case["obstacles"]:
                ax.add_patch(Rectangle((b["x"]-b["hx"],b["y"]-b["hy"]),2*b["hx"],2*b["hy"],color="#555f67",alpha=.35))
            if name in ("step","unknown"):
                ax.axvline(3.5,color="#bd6547",ls="--",label="危险/未知边界")
            if name=="crossing":ax.axvline(4.5,color="#555f67",ls=":",label="移动障碍x位置")
            ax.plot(0,0,"o",color="black",ms=4);ax.plot(*case["goal"],"*",color="#bd6547",ms=9)
            ax.set_title(f"{SCENARIOS[name]}（{len(selected)}次）");ax.set_xlabel("地图 x（米）");ax.set_ylabel("地图 y（米）")
            ax.set_aspect("equal",adjustable="datalim");ax.grid(alpha=.2)
        fig.suptitle("所有批次执行轨迹：青色合格，棕色失败\n灰色仅为第一个预定案例的障碍；每次目标、障碍或响应延迟有随机变化")
        save(fig,output,f"07_trajectories_{part+1}")
    fig,axes=plt.subplots(2,3,figsize=(13,7.5),layout="constrained")
    for ax,name in zip(axes.flat,("turn","reverse","step","emerge","crossing","stale_map")):
        row=next(r for r in rows if r["scenario"]==name and r.get("trajectory"))
        trace=np.asarray(row["trajectory"])
        ax.plot(trace[:,0],trace[:,4],label="实际模型速度",color="#217c91")
        ax.plot(trace[:,0],trace[:,6],"--",label="发布速度指令",color="#bd6547")
        ax.set_xlabel("仿真时间（秒）");ax.set_ylabel("纵向速度（米/秒，负值为倒车）")
        ax.set_title(f"{SCENARIOS[name]}：第一个预定案例");ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.suptitle("控制对象：速度指令与模型响应，含到达后停车过程\n不是仅看路径是否生成；未挑选最优轨迹")
    save(fig,output,"08_control_response")
    fig,axes=plt.subplots(1,3,figsize=(13,4.5),layout="constrained")
    for ax,key,title in zip(axes,("planner_p95_ms","control_p95_ms","tracking_p95_m"),
        ("每案例规划调用P95（毫秒）","每案例控制回调P95（毫秒）","每案例跟踪偏差P95（米）")):
        for name,color in zip(("straight","detour","reverse","emerge","crossing"),COLORS):
            values=np.sort([r[key] for r in rows if r["scenario"]==name and key in r])
            ax.plot(values,np.arange(1,len(values)+1)/len(values),label=SCENARIOS[name],color=color)
        if "ms" in key:ax.set_xscale("log")
        ax.set_xlabel(title);ax.set_ylabel("累计案例比例");ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("实际C++组件耗时与跟踪误差：失败案例也保留\n同步加速沙盒不包含ROS调度、网络、仿真渲染和真实执行器误差")
    save(fig,output,"09_control_timing")
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout="constrained")
    corridor=[r for r in rows if r["scenario"]=="corridor"]
    axes[0].scatter([cases[r["id"]]["corridor_width_m"] for r in corridor],
        [r.get("tracking_p95_m",np.nan) for r in corridor],
        c=["#217c91" if r["status"]=="passed" else "#bd6547" for r in corridor],s=15)
    axes[0].set_xlabel("实际墙间通道宽度（米）");axes[0].set_ylabel("每案例跟踪偏差P95（米）")
    axes[0].set_title("窄通道：不是只统计容易宽度")
    dynamic=[r for r in rows if r["scenario"] in ("emerge","crossing")]
    axes[1].scatter([cases[r["id"]]["speed_tau_s"] for r in dynamic],
        [r.get("goal_error_m",np.nan) for r in dynamic],
        c=["#217c91" if r["status"]=="passed" else "#bd6547" for r in dynamic],s=15)
    axes[1].set_xlabel("模型速度一阶响应时间常数（秒）");axes[1].set_ylabel("终止时距目标距离（米）")
    axes[1].set_title("动态任务：执行器响应变化与失败")
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle("测试难度参数与结果：青色合格，棕色失败；不删除高误差案例")
    save(fig,output,"11_difficulty_response")
    failures=[{k:v for k,v in r.items() if k not in ("trajectory","planning")} for r in rows if r["status"]!="passed"]
    atomic_json(output/"navigation_failures.json",failures)
    with (output/"navigation_runs.csv").open("w",newline="",encoding="utf-8-sig") as stream:
        keys=sorted({k for r in rows for k in r if k not in ("trajectory","planning")})
        writer=csv.DictWriter(stream,keys);writer.writeheader()
        writer.writerows({k:v for k,v in r.items() if k in keys} for r in rows)
    return {"runs":len(rows),"qualified":sum(r["status"]=="passed" for r in rows),"statistics":stats,
        "scope":record["runs"][0].get("scope"),"statuses":dict(Counter(r["status"] for r in rows))}


def tables(summary,output):
    values={"TerrainRuns":summary["terrain"]["runs"],"TerrainInputs":summary["terrain"]["source_count"],
        "NavigationRuns":summary["navigation"]["runs"],"NavigationQualified":summary["navigation"]["qualified"]}
    (output/"counts.tex").write_text("\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k,v in values.items())+"\n")
    lines=[]
    for n,s in summary["navigation"]["statistics"].items():
        planner="--" if s['planner_p95_ms'] is None else f"{s['planner_p95_ms']:.1f}"
        controller="--" if s['control_p95_ms'] is None else f"{s['control_p95_ms']:.2f}"
        lines.append(f"{SCENARIOS[n]} & {s['n']} & {s['qualified']} & {s['collisions']} & {s['terrain_violations']} & "
            f"{s['unknown_violations']} & {planner} & {controller} "+r"\\")
    (output/"navigation_rows.tex").write_text("\n".join(lines)+"\n"+r"\bottomrule"+"\n")
    lines=[]
    for f in FAMILIES:
        for c in ("point15","hole16","frontier50"):
            s=summary["terrain"]["statistics"][f][c]
            fmt=lambda k,factor=1:"--" if s[k]["mean"] is None else f"{s[k]['mean']*factor:.2f}"
            label=CONDITIONS[c].replace('%',r'\%')
            lines.append(f"{FAMILIES[f]} & {label} & {fmt('risk_mae')} & "
                f"{fmt('recall',100)} & {fmt('false_hazard',100)} & {fmt('hidden_rmse_m',100)} "+r"\\")
    (output/"terrain_rows.tex").write_text("\n".join(lines)+"\n"+r"\bottomrule"+"\n")


def completion_figures(results,audit_path,output):
    record=json.loads(results.read_text());audit=json.loads(audit_path.read_text())
    if audit["status"]!="passed" or audit["results_sha256"]!=digest(results):
        raise ValueError("completion audit does not match result file")
    rows=record["runs"];details=audit["completion_details"]
    stats={};paired=[]
    for family in FAMILIES:
        stats[family]={}
        for condition in CONDITIONS:
            selected=[r for r in rows if r["family"]==family and r["condition"]==condition]
            original=[dict(r,**{k:r.get("observed_only_"+k) for k in ("tp","fn","fp","tn")})
                      for r in selected if r["status"]=="ok"]
            now={r["cluster_id"]:r for r in ratio_rows(selected)}
            old={r["cluster_id"]:r for r in ratio_rows(original)}
            delta=[]
            for group in now.keys()&old.keys():
                delta.append({"cluster_id":group,"status":"ok",**{
                    "delta_"+k:now[group][k]-old[group][k]
                    if now[group][k] is not None and old[group][k] is not None else None
                    for k in ("false_safe","false_hazard")}})
            extra=[r for r in details if r["family"]==family and r["condition"]==condition]
            s={k:clustered_summary(selected,k) for k in ("risk_mae","observed_only_risk_mae")}
            s.update({k:clustered_summary(extra,k) for k in ("filled_gt_coverage","filled_rmse_m")})
            s.update({k:clustered_summary(delta,k) for k in ("delta_false_safe","delta_false_hazard")})
            s["new_false_safe_states"]=sum(max(0,r["fn"]-r["observed_only_fn"]) for r in selected if r["status"]=="ok")
            stats[family][condition]=s
            paired.extend(dict(r,family=family,condition=condition) for r in delta)
    fig,axes=plt.subplots(1,2,figsize=(15,5),layout="constrained")
    for ax,key,title in zip(axes,("delta_false_hazard","delta_false_safe"),
        ("安全参考误拒变化（百分点，负值改善）","危险漏判变化（百分点，正值恶化）")):
        values=np.array([[stats[f][c][key]["mean"] if stats[f][c][key]["mean"] is not None else np.nan
                         for c in CONDITIONS] for f in FAMILIES])*100
        bound=max(.01,float(np.nanmax(np.abs(values))))
        im=ax.imshow(values,aspect="auto",cmap="RdBu_r",vmin=-bound,vmax=bound)
        ax.set_xticks(range(8),list(CONDITIONS.values()),rotation=40,ha="right",fontsize=8)
        ax.set_yticks(range(5),list(FAMILIES.values()));ax.set_title(title)
        ax.set_xlabel("人工缺测方式")
        for (i,j),v in np.ndenumerate(values):
            if np.isfinite(v):ax.text(j,i,f"{v:+.2f}",ha="center",va="center",fontsize=8,
                                     color="white" if abs(v)>.6*bound else "black")
        plt.colorbar(im,ax=ax,shrink=.8)
    fig.suptitle("有界补全的配对工程回归：补全开启 − 仅观测（同一输入、同一次GP拟合）\n500份旧来源 × 8条件；不是外部方法比较；减少误拒不能抵消增加危险漏判")
    save(fig,output,"12_completion_safety")
    fig,axes=plt.subplots(1,2,figsize=(15,5),layout="constrained")
    heatmap(axes[0],stats,"filled_gt_coverage",100,"有参考隐藏格子的实际补全比例（%）",100)
    heatmap(axes[1],stats,"filled_rmse_m",100,"被补全且有参考的格子：高程RMSE（厘米）")
    fig.suptitle("只补观测包围的近邻小洞：最近距离≤0.35米、模型标准差≤4厘米\n覆盖分母是有真值参考的隐藏地面，不是所有未知格子；误差只代表被补格子")
    save(fig,output,"13_completion_coverage")
    lines=[]
    for family in FAMILIES:
        for c in ("point60","hole16","frontier50"):
            s=stats[family][c]
            fmt=lambda k,factor=1:"--" if s[k]["mean"] is None else f"{s[k]['mean']*factor:.2f}"
            label=CONDITIONS[c].replace('%',r'\%')
            lines.append(f"{FAMILIES[family]} & {label} & {fmt('filled_gt_coverage',100)} & {fmt('filled_rmse_m',100)} & "
                         f"{fmt('delta_false_hazard',100)} & {fmt('delta_false_safe',100)} "+r"\\")
    (output/"completion_rows.tex").write_text("\n".join(lines)+"\n"+r"\bottomrule"+"\n")
    return {"runs":len(rows),"statuses":dict(Counter(r["status"] for r in rows)),"statistics":stats,
        "new_false_safe_states":sum(s["new_false_safe_states"] for f in stats.values() for s in f.values()),
        "filled_outside_reference_cells":sum(r["filled_outside_reference_cells"] for r in details),
        "results_sha256":digest(results),"audit_sha256":digest(audit_path)}


def safety_figures(directory,output):
    plan=json.loads((directory/"cases.json").read_text());record=json.loads((directory/"runs/results.json").read_text())
    validate_navigation(plan,record);stats={};lines=[]
    fig,axes=plt.subplots(1,3,figsize=(13,4.5),layout="constrained")
    for i,name in enumerate(("steep_ramp","step")):
        rows=[r for r in record["runs"] if r["scenario"]==name]
        success=sum(r["status"]=="passed" for r in rows);lo,hi=wilson_interval(success,len(rows))
        stats[name]={"n":len(rows),"passed":success,"wilson95":[lo,hi],
            "terrain_violations":sum(r.get("unsafe_terrain",False) for r in rows),
            "collisions":sum(r.get("collision",False) for r in rows)}
        axes[0].bar(i,success/len(rows)*100,color=COLORS[i])
        axes[0].errorbar(i,success/len(rows)*100,yerr=[[max(0,success/len(rows)-lo)*100],
             [max(0,hi-success/len(rows))*100]],fmt="none",color="black",capsize=4)
        axes[0].text(i,5,f"{success}/{len(rows)}",ha="center")
        for ax,key,title in zip(axes[1:],("peak_pitch_rad","peak_residual_span_m"),
            ("实际模型最大足迹俯仰（弧度）","实际模型最大足迹残差跨度（米）")):
            values=np.sort([r[key] for r in rows if key in r]);stats[name][key]=float(values.max()) if len(values) else None
            ax.plot(values,np.arange(1,len(values)+1)/len(values),label=SCENARIOS[name],color=COLORS[i])
            ax.set_xlabel(title);ax.set_ylabel("累计案例比例");ax.grid(alpha=.2)
        lines.append(f"{SCENARIOS[name]} & {len(rows)} & {success} & {stats[name]['terrain_violations']} & "
                     f"{stats[name]['collisions']} & {lo*100:.1f}--{hi*100:.1f} "+r"\\")
    axes[0].set_xticks(range(2),[SCENARIOS[n] for n in stats]);axes[0].set_ylim(0,108)
    axes[0].set_ylabel("安全拒绝合格率（%）");axes[0].grid(axis="y",alpha=.2)
    axes[1].axvline(.55,ls="--",color="black",label="俯仰上限")
    axes[2].axvline(.18,ls="--",color="black",label="台阶代理上限")
    axes[1].legend(fontsize=8);axes[2].legend(fontsize=8)
    fig.suptitle("I版本针对性安全回归：半格保守查询 + 周期角度边界修复\n同种子200任务，包含H版本7个陡坡失败；不代表I版本完整导航通过率")
    save(fig,output,"14_safety_revalidation")
    (output/"safety_rows.tex").write_text("\n".join(lines)+"\n"+r"\bottomrule"+"\n")
    return {"runs":len(record["runs"]),"statistics":stats,"results_sha256":digest(directory/"runs/results.json")}


def isaac_figures(manifest,output):
    output.mkdir(parents=True,exist_ok=True)
    from analyze_navigation_subject import read_trials,plot_trajectories,plot_latency
    record,trials=read_trials(manifest)
    ids=[r["index"] for r in record["runs"]]
    if len(ids)!=len(set(ids)) or set(ids)!={j["index"] for j in record["jobs"]}:
        raise ValueError("incomplete Isaac job plan; refuse final statistics")
    plot_trajectories(output,trials,columns=3);plot_latency(output,trials)
    ghost_trials=[t for t in trials if t["summary"].get("scenario")=="clearance"]
    if ghost_trials:
        fig,axes=plt.subplots(3,len(ghost_trials),figsize=(6*len(ghost_trials),10),squeeze=False,layout="constrained")
        for column,trial in enumerate(ghost_trials):
            s=trial["summary"]
            for row,(key,title,ylabel,value_column) in enumerate((
                ("registered_after_vacate","当前注册帧：原障碍高度带","点数（个）",1),
                ("ghost_laser_obstacle_after_vacate","FAST-LIO累计地图：原区域高度带尾段","点数（个）",1),
                ("ghost_risk_after_vacate","增强消息中的前端基线风险尾段","基线风险分数（无量纲）",2))):
                ax=axes[row,column];a=np.asarray(s.get(key,[]),dtype=float)
                if a.ndim==2 and len(a):ax.plot(a[:,0],a[:,value_column],color=COLORS[column%5],marker=".")
                else:ax.text(.5,.5,"没有保存有效读数",transform=ax.transAxes,ha="center")
                ax.set_title(f"任务{trial['row']['index']}：{title}",fontsize=10)
                ax.set_xlabel("障碍移开后的仿真时间（秒，各图窗口不同）");ax.set_ylabel(ylabel)
                ax.grid(alpha=.2)
                if row==2:ax.set_ylim(-.02,1.05);ax.axhline(.58,ls="--",color="black",label="分类阈值0.58")
        fig.suptitle("动态去残影诊断：当前帧、累计地图、前端基线风险必须分别验证\n旧记录未独立保存增强后的SEB风险全程；首次下降不代表持续清除；点数不是语义真值")
        save(fig,output,"15_dynamic_ghosts")
    rows=[];stats={}
    for name in sorted({r["scenario"] for r in record["runs"]}):
        subset=[r for r in record["runs"] if r["scenario"]==name]
        def metric(k):
            a=[r[k] for r in subset if isinstance(r.get(k),(int,float))]
            return float(np.median(a)) if a else None
        s={"n":len(subset),"pipeline_qualified":sum(r["passed"] for r in subset),
            "arrivals":sum(r.get("reached",False) for r in subset),
            "settled":sum(r.get("settled",False) for r in subset),
            **{k:metric(k) for k in ("risk_map_age_p95_s","lidar_wall_hz","imu_wall_hz","planner_p95_ms",
                "ghost_laser_obstacle_clear_delay_s","ghost_risk_clear_delay_s","ghost_live_clear_delay_s")}}
        stats[name]=s
        fmt=lambda k:"--" if s[k] is None else f"{s[k]:.2f}"
        rows.append(f"{SCENARIOS.get(name,name)} & {s['n']} & {s['pipeline_qualified']} & {s['arrivals']} & "
            f"{s['settled']} & {fmt('risk_map_age_p95_s')} & {fmt('lidar_wall_hz')} & {fmt('imu_wall_hz')} "+r"\\")
    (output/"isaac_rows.tex").write_text("\n".join(rows)+"\n"+r"\bottomrule"+"\n")
    return {"runs":len(record["runs"]),"statistics":stats,"manifest_sha256":digest(manifest)}


def capture_figure(manifest,output):
    from analyze_navigation_subject import read_trials
    record,trials=read_trials(manifest)
    if len(record["runs"])!=len(record["jobs"]) or len(trials)!=1:
        raise ValueError("capture smoke must be a completed single-case plan")
    raw=trials[0]["raw"]
    if raw.get("schema_version",0)<3:
        raise ValueError("capture does not distinguish baseline and enhanced risk")
    series=raw["clearance_series"]
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout="constrained")
    for ax,key,column,title,ylabel in zip(axes.flat,
        ("registered","laser_obstacle","baseline_risk","enhanced_risk"),(1,1,2,2),
        ("当前注册点云：原高度带","累计点云：原障碍高度带","前端基线风险","增强后SEB风险"),
        ("点数（个）","点数（个）","风险分数（无量纲）","风险分数（无量纲）")):
        a=np.asarray(series[key],dtype=float)
        if a.ndim==2 and len(a):ax.plot(a[:,0],a[:,column],color="#217c91")
        ax.set_title(title);ax.set_xlabel("障碍移开后的仿真时间（秒）");ax.set_ylabel(ylabel);ax.grid(alpha=.2)
        if "risk" in key:
            ax.set_ylim(-.02,1.05);ax.axhline(.58,ls="--",color="#bd6547")
    fig.suptitle("新增完整时序采集的实际链路验证：单独预定案例，非成功率估计\n前端与增强风险分别保存；高风险也可能是未知/足迹约束，不能自动等同于动态残影")
    save(fig,output,"16_full_clearance_capture")
    return {"manifest_sha256":digest(manifest),"schema_version":raw["schema_version"],
        "pipeline_passed":record["runs"][0]["passed"],
        "diagnostics":raw["clearance_temporal_diagnostics"],
        "series_counts":{k:len(v) for k,v in series.items() if isinstance(v,list)}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",type=Path,required=True);p.add_argument("--terrain-results",type=Path,required=True)
    p.add_argument("--navigation",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--isaac",type=Path,required=True,help="complete, separate physical/ROS campaign")
    p.add_argument("--isaac-candidate",type=Path,help="separate completed risk-contract correction campaign")
    p.add_argument("--completion",type=Path,required=True);p.add_argument("--completion-audit",type=Path,required=True)
    p.add_argument("--safety",type=Path,required=True)
    p.add_argument("--capture",type=Path,help="completed schema-v3 collection smoke, separate from repeated batches")
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True);configure_chinese()
    font=Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
    font_manager.fontManager.addfont(str(font));plt.rcParams["font.family"]=["DejaVu Sans",font_manager.FontProperties(fname=str(font)).get_name()]
    catalog=json.loads(args.dataset.read_text());terrain=json.loads(args.terrain_results.read_text())
    if digest(args.dataset)!=terrain["settings"]["dataset_sha256"]:raise ValueError("terrain dataset hash mismatch")
    plan=json.loads((args.navigation/"cases.json").read_text())
    nav=json.loads((args.navigation/"runs/results.json").read_text())
    summary={"terrain":terrain_figures(terrain,catalog,args.output),
        "navigation":navigation_figures(plan,nav,args.output),
        "hashes":{"terrain":digest(args.terrain_results),"navigation":digest(args.navigation/"runs/results.json")}}
    summary["source_examples"]=source_views(catalog,args.dataset,args.output)
    spatial_prediction_views(args.terrain_results,args.output)
    summary["isaac"]=isaac_figures(args.isaac,args.output)
    if args.isaac_candidate:
        summary["isaac_candidate"]=isaac_figures(args.isaac_candidate,args.output/"isaac_candidate")
    summary["completion"]=completion_figures(args.completion,args.completion_audit,args.output)
    summary["safety_revalidation"]=safety_figures(args.safety,args.output)
    if args.capture:summary["capture_smoke"]=capture_figure(args.capture,args.output)
    tables(summary,args.output);atomic_json(args.output/"summary.json",summary)
    print(json.dumps({"terrain":summary["terrain"]["statuses"],"navigation":summary["navigation"]["statuses"],
        "figures":len(list(args.output.glob("*.pdf")))},ensure_ascii=False))


if __name__=="__main__":main()
