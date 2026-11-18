#!/usr/bin/env python3
"""Distribution, calibration and genuine 3D evidence from immutable recorded runs."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from matplotlib import colors
from plot_paper_scale import FAMILIES, save, plt, font_manager, ratio_rows
from plot_expanded_campaign import CONDITIONS, SCENARIOS, validate_terrain, validate_navigation
from plot_style import configure_chinese
from pointcloud_batch import atomic_json, digest
from thesis_statistics import (box_summary, equal_source_curve, height_diagnostics,
                               source_values, source_mean_interval, threshold_counts, zero_failure_upper)

SHOW=("point15","hole16","frontier50")
PALETTE=("#207d91","#b56c46","#65718b")
NOMINAL=np.array([.5,.8,.9,.95,.99])
THRESHOLDS=np.unique(np.r_[np.linspace(0,1,41),.58,.65,1.001])


def plain(value):
    if isinstance(value,np.ndarray):return plain(value.tolist())
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    if isinstance(value,np.generic):return plain(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def collect_arrays(results,record,output):
    counts={};heights={};chain=hashlib.sha256();verified=0
    for row in record["runs"]:
        if row["status"]!="ok":continue
        path=results.parent/(row["id"]+".npz")
        chain.update(row["id"].encode());chain.update(bytes.fromhex(digest(path)))
        key=(row["family"],row["condition"],row["cluster_id"])
        with np.load(path,allow_pickle=False) as a:
            score,unsafe,support=(a[k] for k in ("predicted_risk","unsafe","supported"))
            confusion=threshold_counts(score,unsafe,support,THRESHOLDS)
            operating=confusion[np.flatnonzero(np.isclose(THRESHOLDS,.58))[0]]
            if not np.array_equal(operating,[row[k] for k in ("tp","fn","fp","tn")]):
                raise ValueError("saved array does not match recorded confusion: "+row["id"])
            if support.sum()!=row["reference_states"]:
                raise ValueError("reference denominator mismatch")
            error=score[support]-a["reference_risk"][support]
            if len(error) and not np.isclose(np.abs(error).mean(),row["risk_mae"],atol=1e-7):
                raise ValueError("saved array does not match recorded risk MAE")
            d=height_diagnostics(a["height_mean"],a["latent_var"],a["reference_height"],a["observed_cells"],NOMINAL)
        counts[key]=counts.get(key,np.zeros_like(confusion))+confusion
        if d["n"]:
            if key not in heights:heights[key]=d
            else:
                for field in d:heights[key][field]+=d[field]
        verified+=1
        if verified%2000==0:print(f"[evidence] audited {verified} saved arrays",flush=True)
    curves={};calibration={}
    for family in FAMILIES:
        curves[family]={};calibration[family]={}
        for condition in CONDITIONS:
            selected=[v for k,v in counts.items() if k[:2]==(family,condition) and v[0].sum()>0]
            curves[family][condition]=plain(equal_source_curve(selected))
            h=[v for k,v in heights.items() if k[:2]==(family,condition)]
            coverage=np.array([v["coverage_counts"]/v["n"] for v in h])
            width=np.array([v["width_sums"]/v["n"] for v in h])
            interval=source_mean_interval(coverage) if len(h) else np.full((2,5),np.nan)
            calibration[family][condition]={"source_groups":len(h),
                "coverage_mean":coverage.mean(axis=0) if len(h) else np.full(5,np.nan),
                "coverage_low":interval[0],"coverage_high":interval[1],
                "width_mean_m":width.mean(axis=0) if len(h) else np.full(5,np.nan),
                "crps_mean_m":float(np.mean([v["crps_sum_m"]/v["n"] for v in h])) if h else None,
                "bias_mean_m":float(np.mean([v["signed_error_sum_m"]/v["n"] for v in h])) if h else None}
    result=plain({"curves":curves,"calibration":calibration,"thresholds":THRESHOLDS,
        "nominal":NOMINAL,"arrays_verified":verified,"array_digest_chain":chain.hexdigest(),
        "results_sha256":digest(results),"script_sha256":digest(__file__),
        "statistics_sha256":digest(Path(__file__).with_name("thesis_statistics.py")),
        "scope":"source-equal reference-state weighting for curves; source-pooled hidden cells then equal sources for coverage; not IID points or calibrated risk probabilities"})
    atomic_json(output/"array_statistics.json",result)
    return result


def box_group(ax,groups,labels,ylabel,title,log=False,unit="来源组",rotation=20,ticksize=8):
    positions=np.arange(1,len(groups)+1)
    if any(len(v)==0 for v in groups):raise ValueError("empty cohort cannot be a zero-valued box")
    b=ax.boxplot(groups,positions=positions,widths=.58,patch_artist=True,showmeans=True,
        meanprops={"marker":"D","markerfacecolor":"white","markeredgecolor":"black","markersize":3},
        flierprops={"marker":".","markersize":2,"alpha":.45},medianprops={"color":"black"})
    for patch in b["boxes"]:patch.set_facecolor("#a8ced2");patch.set_alpha(.8)
    ax.set_xticks(positions,[f"{label}\n{unit}={len(v)}" for label,v in zip(labels,groups)],rotation=rotation,ha="right",fontsize=ticksize)
    ax.set_ylabel(ylabel);ax.set_title(title);ax.grid(axis="y",alpha=.2)
    if log:ax.set_yscale("log")
    else:ax.set_ylim(bottom=0)


def distribution_figures(record,output):
    summary={};csv_rows=[]
    for number,metric,factor,label in (("01","risk_mae",1,"来源组平均风险MAE（无量纲）"),
        ("02","hidden_rmse_m",100,"来源组平均隐藏高程RMSE（厘米）")):
        fig,axes=plt.subplots(3,1,figsize=(7,8),layout="constrained")
        for ax,condition in zip(axes,SHOW):
            groups=[]
            for family in FAMILIES:
                rows=[r for r in record["runs"] if r["family"]==family and r["condition"]==condition]
                values=source_values(rows,metric);a=np.array(list(values.values()))*factor;groups.append(a)
                summary[family+"/"+condition+"/"+metric]=box_summary(a)
                csv_rows.extend({"family":family,"condition":condition,"metric":metric,"cluster_id":k,"value":v}
                                for k,v in values.items())
            box_group(ax,groups,list(FAMILIES.values()),label,CONDITIONS[condition])
            if metric=="hidden_rmse_m":
                ax.set_yscale("symlog",linthresh=1,linscale=.5)
                ax.set_ylim(bottom=0)
        note="；高程纵轴≤1厘米线性、以上对数" if metric=="hidden_rmse_m" else ""
        fig.suptitle("箱体Q1—Q3，中线中位数，须按1.5×IQR，菱形均值\n一个值是来源组内案例均值；离群点不删除"+note)
        save(fig,output,number+"_terrain_boxplots")
    fig,axes=plt.subplots(3,2,figsize=(7.4,8),layout="constrained")
    for col,condition in enumerate(SHOW):
        for row,(metric,label) in enumerate((("false_safe","危险漏判（%）"),("false_hazard","安全参考误拒（%）"))):
            groups=[]
            for family in FAMILIES:
                selected=[r for r in record["runs"] if r["family"]==family and r["condition"]==condition]
                groups.append(np.array(list(source_values(ratio_rows(selected),metric).values()))*100)
            box_group(axes[col,row],groups,list(FAMILIES.values()),label,CONDITIONS[condition])
            axes[col,row].set_ylim(0,105)
    fig.suptitle("漏判与误拒必须同时展示，来源组内汇总混淆计数\n没有类别分母的组不填零；有效来源组数见横轴")
    save(fig,output,"03_safety_boxplots")
    with (output/"boxplot_source_values.csv").open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,["family","condition","metric","cluster_id","value"]);writer.writeheader();writer.writerows(csv_rows)
    atomic_json(output/"boxplot_statistics.json",summary)


def curves_and_calibration(stats,output):
    fig,axes=plt.subplots(3,2,figsize=(7.4,8),layout="constrained")
    for col,condition in enumerate(SHOW):
        for family in FAMILIES:
            c=stats["curves"][family][condition]
            r=np.asarray(c["recall"],float);p=np.asarray(c["precision"],float);fpr=np.asarray(c["fpr"],float)
            axes[col,0].plot(fpr,r,label=FAMILIES[family],lw=1.4)
            axes[col,1].plot(r,p,label=FAMILIES[family],lw=1.4)
            i=np.flatnonzero(np.isclose(THRESHOLDS,.58))[0]
            axes[col,0].plot(fpr[i],r[i],"o",ms=3);axes[col,1].plot(r[i],p[i],"o",ms=3)
        axes[col,0].plot([0,1],[0,1],":",color="gray")
        for ax in axes[col,:]:ax.set_xlim(0,1.02);ax.set_ylim(0,1.02);ax.grid(alpha=.2)
        axes[col,0].set_title(CONDITIONS[condition]+"：ROC");axes[col,1].set_title(CONDITIONS[condition]+"：PR")
        axes[col,0].set_xlabel("安全状态误报比例");axes[col,0].set_ylabel("危险召回比例")
        axes[col,1].set_xlabel("危险召回比例");axes[col,1].set_ylabel("危险预测精确率")
    axes[0,0].legend(fontsize=8)
    fig.suptitle("危险识别ROC/PR，圆点为固定门限0.58\n来源等权；PR受危险比例影响；风险不是事件概率")
    save(fig,output,"04_roc_pr")
    fig,axes=plt.subplots(3,1,figsize=(7,8),layout="constrained")
    for ax,condition in zip(axes,SHOW):
        for family in FAMILIES:
            c=stats["calibration"][family][condition]
            line,=ax.plot(NOMINAL*100,np.asarray(c["coverage_mean"])*100,"o-",label=FAMILIES[family],ms=3)
            ax.fill_between(NOMINAL*100,np.asarray(c["coverage_low"])*100,np.asarray(c["coverage_high"])*100,
                            color=line.get_color(),alpha=.12)
        ax.plot([45,100],[45,100],"--",color="black",label="名义=实际")
        ax.set_xlim(45,100);ax.set_ylim(0,103);ax.set_title(CONDITIONS[condition]);ax.grid(alpha=.2)
        ax.set_xlabel("高程区间名义覆盖（%）");ax.set_ylabel("隐藏参考实际覆盖（%）")
    axes[0].legend(fontsize=8)
    fig.suptitle("高程覆盖诊断：阴影是来源组bootstrap均值区间\n只用隐藏且参考有效地面；覆盖须结合宽度与误差")
    save(fig,output,"05_height_coverage_curve")
    fig,axes=plt.subplots(2,1,figsize=(7,7),layout="constrained")
    family_colors=dict(zip(FAMILIES,("#207d91","#b56c46","#59926a","#776b91","#ad973c")))
    for family in FAMILIES:
        for condition,marker in zip(SHOW,("o","s","^")):
            c=stats["calibration"][family][condition]
            axes[0].scatter(c["width_mean_m"][3]*100,c["coverage_mean"][3]*100,marker=marker,s=45,color=family_colors[family])
            axes[1].scatter(c["crps_mean_m"]*100,c["bias_mean_m"]*100,marker=marker,s=45,color=family_colors[family])
    axes[0].axhline(95,ls="--",color="black");axes[0].set_xscale("log")
    axes[0].set_xlabel("95%区间平均宽度（厘米，对数）");axes[0].set_ylabel("实际覆盖（%）")
    axes[1].axhline(0,ls="--",color="black");axes[1].set_xlabel("高程Gaussian CRPS（厘米，越小越好）")
    axes[1].set_ylabel("高程有符号偏差（厘米）")
    for ax in axes:ax.grid(alpha=.2)
    from matplotlib.lines import Line2D
    handles=[Line2D([],[],color=color,marker="o",ls="none",label=FAMILIES[f]) for f,color in family_colors.items()]
    handles += [Line2D([],[],color="black",marker=marker,ls="none",label=CONDITIONS[c]) for c,marker in zip(SHOW,("o","s","^"))]
    axes[0].legend(handles=handles,loc="upper left",bbox_to_anchor=(1.01,1),fontsize=8)
    fig.suptitle("同时看覆盖、宽度、CRPS与偏差（来源组等权）\n每点是来源/缺测组合；CRPS不是车辆通过概率")
    save(fig,output,"06_uncertainty_width_bias")


def surface_figures(results,record,catalog,dataset_path,output):
    from spatial_evidence import reconstruct_spatial_case, provenance
    selected={}
    source_audit={}
    for family in FAMILIES:
        row=next(r for r in record["runs"] if r["family"]==family and r["condition"]=="hole16" and r["status"]=="ok")
        selected[family]=row["id"]
        d=reconstruct_spatial_case(record,catalog,results,dataset_path,row)
        source_audit[family]=provenance(d)
        reference,mean,observed,origin=d["legacy_reference"],d["mean"],d["observed"],d["origin"]
        xx,yy=np.meshgrid(origin[0]+(np.arange(mean.shape[1])+.5)*.2,origin[1]+(np.arange(mean.shape[0])+.5)*.2)
        bounds=(reference,mean,observed,d["before"])
        low=min(np.nanmin(v) for v in bounds);high=max(np.nanmax(v) for v in bounds)
        zspan=max(.2,high-low);error=np.abs(mean-reference);norm=colors.Normalize(0,max(.01,np.nanmax(error)))
        fig=plt.figure(figsize=(7.4,7.8),layout="constrained")
        fields=((observed,"输入地面观测（孔洞为空白）"),(mean,"GP预测地面"),
                (reference,"已保存评分参考"),(reference,"参考表面上的绝对预测误差"))
        if family=="road":
            n=d["counts"]
            fields=((d["before"],f"删孔前地面观测：{n['before_cells']}格"),
                    (d["after_before_noise"],f"删孔后地面观测：{n['after_cells']}格"),
                    (mean,"GP预测均值（不是新增观测）"),
                    (reference,f"旧评分参考：{n['legacy_reference_cells']}格"))
        for i,(field,title) in enumerate(fields):
            ax=fig.add_subplot(2,2,i+1,projection="3d")
            is_error=i==3 and family!="road"
            kwargs={"cmap":"terrain","vmin":low,"vmax":high} if not is_error else {
                "facecolors":plt.cm.magma(norm(error)),"shade":False}
            mesh=ax.plot_surface(xx,yy,field,rcount=52,ccount=52,linewidth=0,antialiased=False,rasterized=True,**kwargs)
            ax.set_xlim(xx.min(),xx.max());ax.set_ylim(yy.min(),yy.max());ax.set_zlim(low,high if high>low else low+.2)
            ax.set_box_aspect((20.8,20.8,4*zspan));ax.view_init(elev=32,azim=-62)
            ax.set_xticks(np.linspace(xx.min(),xx.max(),5)[1:-1]);ax.set_yticks(np.linspace(yy.min(),yy.max(),5)[1:-1])
            from matplotlib.ticker import FormatStrFormatter, MaxNLocator
            ax.set_zticks([low,high]);ax.zaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            ax.set_xlabel("局部x（米）");ax.set_ylabel("局部y（米）");ax.set_zlabel("")
            ax.text2D(.98,.90,"z（米）",transform=ax.transAxes,fontsize=9,ha="right")
            ax.set_title(title,fontsize=9);ax.tick_params(labelsize=8)
            bar=mesh if not is_error else plt.cm.ScalarMappable(norm=norm,cmap="magma")
            colorbar=fig.colorbar(bar,ax=ax,orientation="horizontal",shrink=.7,pad=.12,aspect=20,
                                 label="绝对误差（米）" if is_error else "高程（米）")
            colorbar.locator=MaxNLocator(nbins=3);colorbar.update_ticks();colorbar.ax.tick_params(labelsize=8)
        fig.suptitle(f"{FAMILIES[family]}：首个可评价样例 / 内部1.6米孔洞\n竖向比例×4，数值为米；坐标/高程范围相同\n绘图降采样52×52；预测有值不代表可通行")
        save(fig,output,"07_surface_"+family)
    atomic_json(output/"spatial_selection.json",selected)
    atomic_json(output/"spatial_provenance.json",source_audit)
    row=next(r for r in record["runs"] if r["family"]=="forest" and r["condition"]=="hole16" and r["status"]=="ok")
    with np.load(results.parent/(row["id"]+".npz")) as a:
        xy=a["query_xy"].copy();prediction=a["predicted_risk"].copy();reference=a["reference_risk"].copy();valid=a["supported"].copy()
    fig=plt.figure(figsize=(7.4,4.5),layout="constrained")
    for i,(values,title) in enumerate(((reference,"独立足迹几何参考"),(prediction,"本工程预测风险"))):
        ax=fig.add_subplot(1,2,i+1,projection="3d")
        for yaw_index in (0,9,18,27):
            selected=valid[yaw_index];yaw=np.full(selected.sum(),(yaw_index+.5)*10.)
            cloud=ax.scatter(xy[selected,0],xy[selected,1],yaw,c=values[yaw_index,selected],vmin=0,vmax=1,
                cmap="RdYlGn_r",s=9,depthshade=False,rasterized=True)
        ax.set_xlabel("局部x（米）");ax.set_ylabel("局部y（米）");ax.set_zlabel("")
        ax.text2D(.98,.90,"θ（度）",transform=ax.transAxes,fontsize=9,ha="right")
        ax.set_zticks([5,95,185,275]);ax.set_title(title);ax.view_init(elev=22,azim=-58)
        fig.colorbar(cloud,ax=ax,orientation="horizontal",shrink=.7,pad=.12,aspect=20,label="风险分数（无量纲）")
    fig.suptitle("SE(2)空间(x,y,θ)：第三轴是航向，非地形高度\n128位置、四个切片；未插值；参考无支撑状态不画")
    save(fig,output,"08_se2_risk_3d")


def navigation_boxes(plan,record,output):
    validate_navigation(plan,record)
    names=list(SCENARIOS)
    fig,axes=plt.subplots(3,1,figsize=(7.5,9.2),layout="constrained")
    statistics={}
    for ax,key,label,log in zip(axes,("planner_p95_ms","control_p95_ms","tracking_p95_m"),
        ("案例内规划P95（毫秒，对数）","案例内控制P95（毫秒，对数）","案例内跟踪偏差P95（米）"),(True,True,False)):
        groups=[];labels=[]
        for name in names:
            rows=[r for r in record["runs"] if r["scenario"]==name and r.get(key) is not None]
            values=np.array([r[key] for r in rows]);groups.append(values);labels.append(SCENARIOS[name])
            statistics[name+"/"+key]=box_summary(values)
        box_group(ax,groups,labels,label,"H版本模型闭环：包括失败",log=log,unit="任务",rotation=50,ticksize=7)
        rng=np.random.default_rng(20261014)
        for i,name in enumerate(names,1):
            failed=[r[key] for r in record["runs"] if r["scenario"]==name and r["status"]!="passed" and r.get(key) is not None]
            if failed:ax.scatter(i+rng.uniform(-.16,.16,len(failed)),failed,s=7,color="#bd6547",alpha=.7)
    fig.suptitle("每类100个模型任务，红点失败；每值是案例内P95\n不是总体回调P95，不含DDS或PhysX实时性")
    save(fig,output,"09_navigation_boxplots")
    atomic_json(output/"navigation_boxplot_statistics.json",statistics)
    failures=Counter(r["scenario"] for r in record["runs"] if r["status"]!="passed")
    ranked=failures.most_common();fig,ax=plt.subplots(figsize=(7,3.5),layout="constrained")
    ax.bar([SCENARIOS[k] for k,_ in ranked],[v for _,v in ranked],color="#b56c46")
    for i,(_,v) in enumerate(ranked):ax.text(i,v+.5,str(v),ha="center")
    ax.set_ylabel("不合格案例数（次）");ax.set_title("失败集中在哪类任务（不是未经证明的因果归因）");ax.grid(axis="y",alpha=.2)
    save(fig,output,"10_failure_distribution")


def analytic_figures(path,output):
    record=json.loads(path.read_text());rows=record["runs"]
    expected={(backend,slope,direction,repeat) for backend in ("cpu","cupy")
              for slope in (.10,.20,.35,.45,.50,.60) for direction in (0.,.37,.91,1.43)
              for repeat in range(record["arguments"]["repeats"])}
    actual=[(r["backend"],r["slope_rad"],r["direction_rad"],r["repeat"]) for r in rows]
    if len(actual)!=len(expected) or set(actual)!=expected:raise ValueError("incomplete analytic paired plan")
    pairs=defaultdict(dict)
    for r in rows:pairs[(r["geometry_id"],r["sensor_seed"])][r["backend"]]=r
    metrics=("seb_risk_mae","fused_risk_mae","false_safe_count","false_hazard_count")
    differences={k:max(abs(pair["cpu"][k]-pair["cupy"][k]) for pair in pairs.values()) for k in metrics}
    fig,axes=plt.subplots(3,2,figsize=(7.4,8),layout="constrained")
    for ax,curve in zip(axes.flat,record["curves"]):
        degrees=np.degrees(curve["yaw"])
        ax.plot(degrees,curve["oracle"],color="black",label="解析姿态分数")
        ax.plot(degrees,curve["seb"],label="SEB几何分量")
        ax.plot(degrees,curve["fused"],"--",label="加权输出＋硬几何保护")
        ax.fill_between(degrees,0,1,where=curve["unsafe"],color="#bd6547",alpha=.12)
        ax.axhline(.58,ls=":",color="#bd6547");ax.set_title(f"坡面倾角{np.degrees(curve['slope_rad']):.1f}度")
        ax.set_xlabel("车辆航向（度）");ax.set_ylabel("几何风险分数（无量纲）");ax.set_ylim(0,1.05);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7)
    repeats=record["arguments"]["repeats"]
    fig.suptitle(f"新增配对诊断：24种解析坡面 × {repeats}噪声种子 × CPU/GPU\n{len(rows)}次计算，{len(pairs)}对输入，不是独立地形；阴影为解析姿态硬越限")
    save(fig,output,"11_analytic_yaw")
    fig,axes=plt.subplots(1,2,figsize=(7.4,4.3),layout="constrained")
    timing=[[r["compute_ms"] for r in rows if r["backend"]==backend] for backend in ("cpu","cupy")]
    box_group(axes[0],timing,["CPU","CuPy GPU"],"计算墙钟（毫秒，对数）","包括首次冷启动",log=True,unit="调用")
    geometry_pairs=defaultdict(lambda:defaultdict(list))
    for r in rows:geometry_pairs[r["geometry_id"]][r["backend"]].append(r["compute_ms"])
    speedups=[np.median(v["cpu"])/np.median(v["cupy"]) for v in geometry_pairs.values()]
    axes[1].bar(range(24),speedups,color="#207d91");axes[1].axhline(1,ls="--",color="black")
    axes[1].set_xlabel("24种预定坡度/坡向组合索引");axes[1].set_ylabel("CPU中位耗时/GPU中位耗时（倍）")
    axes[1].set_title("每组合20个配对输入；不是全系统加速比");axes[1].grid(axis="y",alpha=.2)
    fig.suptitle("48×48×36 SEB计算耗时：含复制/同步，不含GP/ROS\n固定小图与机器；不能推断渲染或全系统速度")
    save(fig,output,"13_backend_runtime")
    totals={b:{k:sum(r[k] for r in rows if r["backend"]==b)
               for k in ("false_safe_count","unsafe_states","false_hazard_count","safe_states")} for b in ("cpu","cupy")}
    return {"runs":len(rows),"paired_inputs":len(pairs),"geometries":record["geometry_count"],"totals":totals,
        "runtime_ms":{b:box_summary(v) for b,v in zip(("cpu","cupy"),timing)},
        "geometry_median_speedup_range":[float(min(speedups)),float(max(speedups))],
        "max_cpu_gpu_metric_difference":differences,"results_sha256":digest(path),
        "algorithm_sha256":record["adapter_sha256"]}


def sample_size_figure(output):
    n=np.arange(1,501);upper=np.array([zero_failure_upper(int(v)) for v in n])*100
    fig,ax=plt.subplots(figsize=(7,4.5),layout="constrained");ax.plot(n,upper,color="#207d91")
    for count in (2,30,100,299):
        ax.plot(count,zero_failure_upper(count)*100,"o",color="#b56c46")
        ax.annotate(f"n={count}：{zero_failure_upper(count)*100:.2f}%",(count,zero_failure_upper(count)*100),
            xytext=(8,10 if count<100 else 25),textcoords="offset points",fontsize=9)
    ax.axhline(1,ls="--",color="black");ax.set_yscale("log");ax.set_xlabel("独立同分布任务数n（且观察到零失败）")
    ax.set_ylabel("真实失败概率的单侧95%上限（%，对数）")
    ax.set_title("样本量服从要证明的主张，不是毕业论文硬性次数")
    ax.grid(alpha=.2);save(fig,output,"12_sample_size_bound")


def evidence_tables(stats,analytic,output):
    lines=[]
    for family in FAMILIES:
        for condition in SHOW:
            c=stats["calibration"][family][condition];label=CONDITIONS[condition].replace('%',r'\%')
            lines.append(f"{FAMILIES[family]} & {label} & {c['source_groups']} & "
                f"{c['crps_mean_m']*100:.2f} & {c['bias_mean_m']*100:.2f} & "
                f"{c['coverage_mean'][3]*100:.1f} & {c['width_mean_m'][3]*100:.1f} "+r"\\")
    (output/"calibration_rows.tex").write_text("\n".join(lines)+"\n"+r"\bottomrule"+"\n")
    lines=[]
    for backend in ("cpu","cupy"):
        t=analytic["totals"][backend];runtime=analytic["runtime_ms"][backend]
        lines.append(f"{backend} & {analytic['runs']//2} & {t['false_safe_count']}/{t['unsafe_states']} & "
            f"{100*t['false_hazard_count']/t['safe_states']:.3f} & {runtime['median']:.2f} & {runtime['p95']:.2f} "+r"\\")
    (output/"analytic_rows.tex").write_text("\n".join(lines)+"\n"+r"\bottomrule"+"\n")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study",type=Path,default=Path("log/optimization_navigation_20261013_v2"))
    p.add_argument("--analytic",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--spatial-only",action="store_true",help="rebuild matched spatial views without replacing audited statistical caches")
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True);configure_chinese()
    font=Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf");font_manager.fontManager.addfont(str(font))
    plt.rcParams["font.family"]=["DejaVu Sans",font_manager.FontProperties(fname=str(font)).get_name()]
    plt.rcParams.update({"font.size":10,"axes.labelsize":9,"axes.titlesize":10,
        "xtick.labelsize":8,"ytick.labelsize":8,"figure.titlesize":11})
    results=args.study/"holdout_final/results.json";dataset=args.study/"holdout_clouds/evaluation_manifest.json"
    record=json.loads(results.read_text());catalog=json.loads(dataset.read_text());validate_terrain(record,catalog)
    if digest(dataset)!=record["settings"]["dataset_sha256"]:raise ValueError("dataset hash changed")
    if args.spatial_only:
        surface_figures(results,record,catalog,dataset,args.output)
        print("[spatial] matched 3D views rebuilt; old numeric scores unchanged")
        return
    stats=collect_arrays(results,record,args.output)
    distribution_figures(record,args.output);curves_and_calibration(stats,args.output)
    surface_figures(results,record,catalog,dataset,args.output)
    nav=args.study/"navigation_holdout_h";plan=json.loads((nav/"cases.json").read_text());runs=json.loads((nav/"runs/results.json").read_text())
    navigation_boxes(plan,runs,args.output);analytic=analytic_figures(args.analytic,args.output);sample_size_figure(args.output)
    evidence_tables(stats,analytic,args.output)
    atomic_json(args.output/"summary.json",{"date":"2026-10-14","read_only_reanalysis":True,
        "old_terrain_runs":len(record["runs"]),"arrays_verified":stats["arrays_verified"],"model_runs":len(runs["runs"]),
        "new_analytic":analytic,"plot_count":len(list(args.output.glob("*.pdf"))),
        "hashes":{"terrain":digest(results),"dataset":digest(dataset),"navigation":digest(nav/"runs/results.json")}})
    print(json.dumps({"arrays_verified":stats["arrays_verified"],"new_analytic":analytic,"figures":len(list(args.output.glob("*.pdf")))},ensure_ascii=False))


if __name__=="__main__":main()
