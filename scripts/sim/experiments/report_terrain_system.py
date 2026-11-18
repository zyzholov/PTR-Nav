"""Audit saved arrays and make the single Chinese report for terrain study v3."""

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from terrain_system_study import BASE, PROTOCOL, fingerprints, evaluation_domains, domain_metrics, risk_metrics
from pointcloud_batch import ROOT, atomic_json, digest
from plot_paper_scale import plt, save, FAMILIES
from plot_style import configure_chinese
from controlled_terrain_study import NAMES as MECHANISMS
from paper_aligned_report import query_analysis, NAMES as PLANNING_NAMES
from thesis_statistics import threshold_counts, equal_source_curve

OUT = ROOT/"docs/reports/systematic_study/assets/system03"
CONDITIONS = {"point15":"删点15%", "point30":"删点30%", "point60":"删点60%",
    "hole08":"孔洞0.8米", "hole16":"孔洞1.6米", "hole32":"孔洞3.2米",
    "frontier25":"单侧隐藏25%", "frontier50":"单侧隐藏50%"}
COLORS = ("#247b92", "#538456", "#be714c", "#77669d", "#ac8d30")


def read(path):
    return json.loads(path.read_text())


def equal(a, b):
    if isinstance(a, dict):
        return set(a) == set(b) and all(equal(a[k], b[k]) for k in a)
    if a is None or b is None:
        return a is b
    return bool(np.allclose(a, b, atol=1e-7, rtol=1e-6))


def audit(record, phase, protocol):
    settings = record["settings"]
    if settings["source_sha256"] != fingerprints() or settings["protocol_sha256"] != digest(PROTOCOL):
        raise ValueError("frozen source/protocol changed; do not relabel earlier results")
    rows = record["runs"]; ids = [r["id"] for r in rows]
    catalog = read(BASE/"catalog.json")
    if phase == "public":
        if settings["catalog_sha256"] != digest(BASE/"catalog.json"):
            raise ValueError("catalog changed")
        expected = {d["id"]+"__"+c for d in catalog["datasets"] for c in protocol[phase]["conditions"]}
        for item in catalog["datasets"]:
            if digest(BASE/item["path"]) != item["sha256"]:
                raise ValueError("input changed")
    else:
        expected = {f+f"_{i:03d}__"+c for f in protocol[phase]["families"]
            for i in range(protocol[phase]["seeds_per_family"]) for c in protocol[phase]["conditions"]}
    if len(ids) != protocol[phase]["planned_cases"] or len(set(ids)) != len(ids) or set(ids) != expected:
        raise ValueError("incomplete or duplicate factorial")
    chain = []
    for row in rows:
        path = BASE/phase/row["archive"]
        if row["status"] != "ok" or row["actual_device"] != "cuda" or row["actual_backend"] != "cupy":
            raise ValueError("unsuccessful execution cannot become a valid result")
        if digest(path) != row["archive_sha256"]:
            raise ValueError("archive changed")
        with np.load(path) as a:
            p = SimpleNamespace(mean=a["mean"], latent_var=a["variance"], accepted=a["accepted"])
            if not np.isfinite(p.mean).all() or not np.isfinite(p.latent_var).all() or (p.latent_var < 0).any():
                raise ValueError("nonfinite posterior")
            domains = evaluation_domains(a["reference"], a["full_observed"], a["observed"])
            for label, selected in domains.items():
                for suffix, mask in (("", selected), ("_accepted", selected & p.accepted)):
                    value = domain_metrics(p, a["reference"], mask, protocol["nominal_levels"])
                    if not equal(value, row["height"][label+suffix]):
                        raise ValueError("height metric mismatch: "+row["id"])
            risk = risk_metrics(a["predicted_risk"], a["reference_risk"], a["unsafe"],
                a["risk_supported"], protocol["risk_threshold"])
            if not equal(risk, row["risk"]):
                raise ValueError("risk metric mismatch")
            measured_center=np.broadcast_to(a["observed"].ravel()[a["query_indices"]],a["risk_supported"].shape)
            for label,selected in (("measured_center",measured_center),("unknown_center",~measured_center)):
                row["risk_"+label]=risk_metrics(a["predicted_risk"],a["reference_risk"],a["unsafe"],
                    a["risk_supported"]&selected,protocol["risk_threshold"])
            if (a["outside_hull"] & p.accepted).any():
                raise ValueError("outside-hull candidate accepted")
            row["threshold_counts"] = threshold_counts(a["predicted_risk"],a["unsafe"],a["risk_supported"],
                np.r_[np.linspace(.1,.9,17), .58]).tolist()
        chain.append(row["id"]+":"+row["archive_sha256"])
    return dict(arrays=len(chain), chain_sha256=hashlib.sha256("\n".join(chain).encode()).hexdigest())


def cluster_values(rows, getter):
    groups = defaultdict(list)
    for r in rows:
        value = getter(r)
        if value is not None and np.isfinite(value):
            groups[r["cluster_id"]].append(float(value))
    return np.asarray([np.mean(v) for _, v in sorted(groups.items())])


def summary(rows, getter):
    values = cluster_values(rows, getter)
    if not len(values):
        return dict(n=0, mean=None, low=None, high=None)
    rng = np.random.default_rng(261015123)
    samples = values[rng.integers(0, len(values), (2000, len(values)))].mean(axis=1)
    lo, hi = np.quantile(samples, [.025, .975])
    return dict(n=len(values), mean=float(values.mean()), low=float(lo), high=float(hi))


def proportion(row, domain="artificial"):
    a, b = row["height"][domain+"_accepted"]["n"], row["height"][domain]["n"]
    return a/b if b else None


def subset(rows, family=None, condition=None):
    return [r for r in rows if (family is None or r["family"] == family)
            and (condition is None or r["condition"] == condition)]


def box(ax, values, labels, ylabel):
    b = ax.boxplot([v if len(v) else [np.nan] for v in values], tick_labels=labels,
        widths=.55, patch_artist=True, medianprops=dict(color="#263b46", linewidth=1.4),
        flierprops=dict(marker=".", markersize=3, alpha=.5))
    for i, patch in enumerate(b["boxes"]):
        patch.set_facecolor(COLORS[i % len(COLORS)]); patch.set_alpha(.55)
    ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=.2)


def height_figures(rows):
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 10.2), layout="constrained")
    for ax, domain, key, title, cmap in zip(axes,
        ("artificial", "artificial_accepted", None), ("rmse_m", "rmse_m", None),
        ("删点新造成的隐藏格：原始RMSE", "通过门控的隐藏格：RMSE", "通过门控格 / 人工隐藏格"),
        ("YlOrRd", "YlOrRd", "YlGnBu")):
        a = np.full((5, 8), np.nan)
        for i, f in enumerate(FAMILIES):
            for j, c in enumerate(CONDITIONS):
                ss = subset(rows, f, c)
                value = summary(ss, (lambda r: proportion(r)) if domain is None else (lambda r: r["height"][domain][key]))["mean"]
                if value is not None: a[i, j] = value*100
        im = ax.imshow(a, cmap=cmap, aspect="auto", vmin=0, vmax=100 if domain is None else None)
        ax.set_xticks(range(8), CONDITIONS.values(), rotation=25, ha="right")
        ax.set_yticks(range(5), FAMILIES.values()); ax.set_title(title, fontsize=12)
        for i in range(5):
            for j in range(8):
                ax.text(j, i, "--" if np.isnan(a[i,j]) else f"{a[i,j]:.1f}", ha="center", va="center", fontsize=10,
                    color="#222222", bbox=dict(facecolor="white", alpha=.65, edgecolor="none", pad=.5))
        fig.colorbar(im, ax=ax, shrink=.7, label="比例（%）" if domain is None else "厘米")
    save(fig, OUT, "S1_accuracy")
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.2), layout="constrained")
    for col, (cs, xs, label) in enumerate(((list(CONDITIONS)[:3], [15,30,60], "删去输入点（%）"),
        (list(CONDITIONS)[3:6], [.8,1.6,3.2], "内部孔洞边长（米）"),
        (list(CONDITIONS)[6:], [25,50], "隐藏右侧地图宽度（%）"))):
        for f, color in zip(FAMILIES, COLORS):
            points = [summary(subset(rows, f, c), lambda r:r["height"]["artificial"]["rmse_m"]) for c in cs]
            means = [p["mean"] for p in points]
            coverage = [summary(subset(rows, f, c), proportion)["mean"] for c in cs]
            axes[0,col].plot(xs, np.asarray(means,dtype=float)*100, "o-", color=color, label=FAMILIES[f], ms=4)
            axes[0,col].fill_between(xs,np.asarray([p["low"] for p in points],dtype=float)*100,
                np.asarray([p["high"] for p in points],dtype=float)*100,color=color,alpha=.1)
            axes[1,col].plot(xs, np.asarray(coverage,dtype=float)*100, "o-", color=color, ms=4)
        axes[0,col].set_yscale("log"); axes[0,col].set_ylabel("人工隐藏格RMSE（厘米，对数）")
        axes[1,col].set_ylabel("人工隐藏格接受比例（%）"); axes[1,col].set_ylim(-2, 102)
        for ax in axes[:,col]: ax.set_xlabel(label); ax.grid(alpha=.2)
    fig.legend(*axes[0,0].get_legend_handles_labels(), loc="outside upper center", ncols=5, frameon=False)
    save(fig, OUT, "S2_severity")
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), layout="constrained")
    for ax, domain, title in zip(axes, ("artificial", "artificial_accepted", "natural"),
        ("人工隐藏格：全部预测", "人工隐藏格：仅门控通过", "自然缺测格：仅有参考的位置")):
        for f, color in zip(FAMILIES, COLORS):
            ss = subset(rows, f)
            y = [summary(ss, lambda r,i=i:r["height"][domain]["picp"][i] if r["height"][domain]["n"] else None)["mean"] for i in range(5)]
            ax.plot([50,80,90,95,99], [np.nan if v is None else v*100 for v in y], "o-", ms=3, label=FAMILIES[f], color=color)
        ax.plot([50,99], [50,99], "--", color="#666666", lw=1)
        ax.set(xlabel="名义区间覆盖率（%）", ylabel="真实高程落入区间的比例（%）", title=title, ylim=(0,102))
        ax.grid(alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncols=5, frameon=False)
    save(fig, OUT, "S3_calibration")
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4), layout="constrained")
    getters = (lambda r:r["height"]["artificial_accepted"]["width_m"][3]*100 if r["height"]["artificial_accepted"]["n"] else None,
        lambda r:r["height"]["artificial_accepted"]["crps_m"]*100 if r["height"]["artificial_accepted"]["n"] else None,
        lambda r:r["height"]["natural"]["n"])
    for ax, getter, title in zip(axes, getters, ("接受域95%区间总宽（厘米）", "接受域CRPS（厘米，越低越好）", "自然缺测且有参考的格数 / 输入")):
        box(ax, [cluster_values(subset(rows,f),getter) for f in FAMILIES], list(FAMILIES.values()), title)
    save(fig, OUT, "S4_width_boxes")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4), layout="constrained")
    labels = ["0–0.2", "0.2–0.35", "0.35–0.7", "0.7–1.2", ">=1.2"]
    for f, color in zip(FAMILIES, COLORS):
        values = [summary(subset(rows,f),lambda r,i=i:r["distance"][i]["rmse_m"])["mean"] for i in range(5)]
        axes[0].plot(range(5), [np.nan if v is None else v*100 for v in values], "o-", label=FAMILIES[f], color=color)
    axes[0].set_xticks(range(5), labels); axes[0].set_yscale("log")
    axes[0].set(xlabel="距最近实测地面点的XY距离（米）", ylabel="隐藏格RMSE（厘米，对数）")
    names = list(FAMILIES)
    box(axes[1], [cluster_values(subset(rows,f,"hole16"), lambda r:r["height"]["artificial"]["rmse_m"]*100) for f in names],
        list(FAMILIES.values()), "1.6米孔洞：来源组平均RMSE（厘米）")
    axes[0].legend(frameon=False); axes[0].grid(alpha=.2)
    save(fig, OUT, "S5_distance")


def risk_figures(public, mechanism):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7), layout="constrained")
    for col, (rows, names) in enumerate(((public,FAMILIES),(mechanism,{f:MECHANISMS[f] for f in read(PROTOCOL)["mechanism"]["families"]}))):
        for line, key, title in ((0,"fnr","危险状态漏判率 FNR（%）"),(1,"fpr","安全状态误拒率 FPR（%）")):
            values = [cluster_values(subset(rows,f),lambda r:r["risk"][key])*100 for f in names]
            box(axes[line,col], values, list(names.values()), title)
            axes[line,col].tick_params(axis="x", rotation=35)
            if line == 0: axes[line,col].axhline(1, color="#ba594f", ls="--", lw=1)
        axes[0,col].set_title("多来源地形代理参考" if col == 0 else "解析专项参考：旋转、边界和细柱")
    save(fig, OUT, "S6_risk")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6), layout="constrained")
    for phase, rows, color in (("104×104 多来源",public,COLORS[0]),("64×64 专项",mechanism,COLORS[2])):
        for ax, key, title in zip(axes,("gp_wall_ms","risk_wall_ms"),("GP拟合+全图预测", "垂直占据+BGK/SEB+结果拷回")):
            a = np.sort([r[key] for r in rows]); ax.plot(a,np.arange(1,len(a)+1)/len(a),label=phase,color=color)
            ax.set(xlabel="同步墙钟耗时（毫秒）", ylabel="调用累计比例", title=title)
            ax.grid(alpha=.2); ax.legend(frameon=False)
    axes[1].axvline(100, color="#ba594f", ls="--", lw=1)
    save(fig, OUT, "S7_latency")
    fig,axes=plt.subplots(1,2,figsize=(11.5,3.8),layout="constrained")
    for f,color in zip(FAMILIES,COLORS):
        groups=defaultdict(list)
        for row in subset(public,f):groups[row["cluster_id"]].append(row["threshold_counts"])
        counts=np.asarray([np.asarray(v).sum(axis=0) for _,v in sorted(groups.items())])
        curve=equal_source_curve(counts)
        axes[0].plot(curve["fpr"][:-1]*100,curve["recall"][:-1]*100,".-",color=color,label=FAMILIES[f])
        axes[1].plot(curve["recall"][:-1]*100,curve["precision"][:-1]*100,".-",color=color)
        axes[0].scatter(curve["fpr"][-1]*100,curve["recall"][-1]*100,marker="*",s=70,color=color)
        axes[1].scatter(curve["recall"][-1]*100,curve["precision"][-1]*100,marker="*",s=70,color=color)
    axes[0].set(xlabel="安全误拒率 FPR（%）",ylabel="危险检出率 Recall（%）",title="阈值扫描：安全/保守性折中")
    axes[1].set(xlabel="危险检出率 Recall（%）",ylabel="危险判定精确率 Precision（%）",title="星号为冻结阈值0.58，未据图调参")
    axes[0].legend(frameon=False)
    for ax in axes:ax.grid(alpha=.2);ax.set_xlim(0,102);ax.set_ylim(0,102)
    save(fig,OUT,"S13_thresholds")
    fig,axes=plt.subplots(1,2,figsize=(11.5,3.8),layout="constrained")
    for ax,rows,names in ((axes[0],public,FAMILIES),(axes[1],mechanism,{f:MECHANISMS[f] for f in read(PROTOCOL)["mechanism"]["families"]})):
        x=np.arange(len(names))
        for offset,key,title,color in ((-.18,"measured_center","中心有实测地面",COLORS[0]),(.18,"unknown_center","中心无实测地面",COLORS[2])):
            values=[]
            for f in names:
                ss=subset(rows,f);fp=sum(r["risk_"+key]["fp"] for r in ss);tn=sum(r["risk_"+key]["tn"] for r in ss)
                values.append(100*fp/(fp+tn) if fp+tn else np.nan)
            ax.bar(x+offset,values,width=.34,color=color,label=title)
        ax.set_xticks(x,list(names.values()),rotation=35);ax.set_ylabel("参考安全状态被拒绝的比例（%）")
        ax.set_ylim(0,105);ax.grid(axis="y",alpha=.2)
    axes[0].legend(frameon=False,fontsize=8)
    save(fig,OUT,"S14_observation_domains")
    fig,ax=plt.subplots(figsize=(7.5,3.8),layout="constrained")
    names=("slope_boundary","ridge","step_low","step_high","column")
    cs=("point30","hole16","frontier50")
    values=np.full((len(names),len(cs)),np.nan)
    for i,f in enumerate(names):
        for j,c in enumerate(cs):
            ss=subset(mechanism,f,c);tp=sum(r["risk"]["tp"] for r in ss);fn=sum(r["risk"]["fn"] for r in ss)
            if tp+fn:values[i,j]=100*fn/(tp+fn)
    im=ax.imshow(values,cmap="YlOrRd",aspect="auto",vmin=0)
    ax.set_xticks(range(3),[CONDITIONS[c] for c in cs]);ax.set_yticks(range(len(names)),[MECHANISMS[f] for f in names])
    for i in range(len(names)):
        for j in range(3):ax.text(j,i,f"{values[i,j]:.2f}%",ha="center",va="center",fontsize=9,
            bbox=dict(facecolor="white",alpha=.7,edgecolor="none",pad=1))
    fig.colorbar(im,ax=ax,label="危险状态漏判率 FNR（%）")
    save(fig,OUT,"S15_mechanism_conditions")


def spatial_figures(public, mechanism):
    examples = {}
    for family in ("hilly", "road"):
        row = sorted(subset(public,family,"hole16"), key=lambda r:r["id"])[0]
        examples[family] = row["id"]
        with np.load(BASE/"public"/row["archive"]) as a:
            shape=a["measured"].shape; ref=a["reference"].reshape(shape)
            extent=[a["origin"][0],a["origin"][0]+shape[1]*.2,a["origin"][1],a["origin"][1]+shape[0]*.2]
            domains=evaluation_domains(a["reference"],a["full_observed"],a["observed"])
            observed=a["observed"]; full=a["full_observed"]
            fig, axes=plt.subplots(2,3,figsize=(11,6.5),layout="constrained")
            limits=np.nanquantile(ref,[0,1])
            from matplotlib.patches import Rectangle
            for ax, field, title in zip(axes[0], (ref,np.where(full,ref,np.nan),np.where(observed,ref,np.nan)),
                ("(a) 独立参考：仅有限值可评分", "(b) 删点前观测支撑上的同一参考", "(c) 删点后支撑：必为(b)子集")):
                im=ax.imshow(field,origin="lower",extent=extent,cmap="terrain",vmin=limits[0],vmax=limits[1]);ax.set_title(title,fontsize=9)
                center=np.asarray(a["origin"])+np.asarray(shape[::-1])*.1
                ax.add_patch(Rectangle(center-.8,1.6,1.6,fill=False,edgecolor="#bb4c50",linewidth=1.1))
            fig.colorbar(im,ax=list(axes[0]),shrink=.7,label="高程（米）")
            fields=(np.where(domains["artificial"],a["mean"]-a["reference"],np.nan).reshape(shape)*100,
                np.where(a["accepted"],a["mean"],np.nan).reshape(shape),
                np.where(np.isfinite(ref),np.sqrt(a["variance"].reshape(shape))*100,np.nan))
            titles=("(d) 人工隐藏格误差：预测减参考", "(e) 门控通过的GP均值（未接入规划）", "(f) GP潜在高程标准差")
            for ax, field, title, cm, unit in zip(axes[1],fields,titles,("RdBu_r","terrain","magma"),("厘米","米","厘米")):
                im=ax.imshow(field,origin="lower",extent=extent,cmap=cm);ax.set_title(title,fontsize=9)
                fig.colorbar(im,ax=ax,shrink=.7,label=unit)
            for ax in axes.ravel(): ax.set(xlabel="局部X（米）",ylabel="局部Y（米）")
            fig.suptitle(FAMILIES[family]+"：首个预定样例；白色是无值，不表示安全",fontsize=12)
            save(fig,OUT,"S8_"+family)
            if family=="hilly":
                fig=plt.figure(figsize=(10,7.5),layout="constrained")
                xx=a["xy"][:,0].reshape(shape);yy=a["xy"][:,1].reshape(shape)
                fields=(np.where(observed,ref,np.nan),ref,fields[1],np.where(domains["artificial"],np.abs(a["mean"]-a["reference"]),np.nan).reshape(shape))
                titles=("输入观测支撑：孔洞不绘制", "完整生成参考表面", "仅门控通过的GP表面", "人工隐藏处绝对高程误差")
                for i,(field,title) in enumerate(zip(fields,titles)):
                    ax=fig.add_subplot(2,2,i+1,projection="3d")
                    surf=ax.plot_surface(xx,yy,field,cmap="terrain" if i<3 else "magma",rcount=shape[0],ccount=shape[1],linewidth=0,antialiased=False)
                    ax.set(xlabel="X（米）",ylabel="Y（米）",zlabel="Z / m" if i<3 else "误差 / m",title=title)
                    if i<3: ax.set_zlim(*limits)
                    ax.view_init(35,-60);fig.colorbar(surf,ax=ax,shrink=.45,pad=.16)
                save(fig,OUT,"S9_surface")
    row=sorted(subset(mechanism,"slope_boundary","point30"),key=lambda r:r["id"])[0]
    with np.load(BASE/"mechanism"/row["archive"]) as a:
        yaws=(np.arange(36)+.5)*10
        fig,axes=plt.subplots(1,2,figsize=(11,3.5),layout="constrained")
        axes[0].plot(yaws,a["reference_risk"].mean(axis=1),color=COLORS[0],label="密集足迹参考")
        axes[0].plot(yaws,a["predicted_risk"].mean(axis=1),"--",color=COLORS[2],label="当前SEB输出")
        axes[0].axhline(.58,color="#666666",lw=1,ls=":")
        axes[0].set(xlabel="车辆航向（度）",ylabel="风险分数（不是概率）",title="相同坡面随航向改变的风险")
        axes[0].legend(frameon=False)
        unsafe=a["unsafe"].sum(axis=1);miss=((a["predicted_risk"]<.58)&a["unsafe"]).sum(axis=1)
        axes[1].plot(yaws,unsafe,"-",color=COLORS[0],label="参考危险状态数")
        axes[1].plot(yaws,miss,"--",color=COLORS[2],label="被判低风险的危险状态数")
        axes[1].set(xlabel="车辆航向（度）",ylabel="状态数",title="危险分母与漏判数同时展示");axes[1].legend(frameon=False)
        save(fig,OUT,"S10_heading")
    return examples


def planning_figures():
    plan=read(BASE/"planning/cases.json");record=read(BASE/"planning/runs/results.json")
    if plan["settings"]["risk_input"]!="measured_only" or plan["settings"]["unknown_completion"]:
        raise ValueError("planning experiment does not match declared deployment contract")
    if plan["settings"]["protocol_sha256"]!=digest(PROTOCOL):raise ValueError("planning protocol mismatch")
    from navigation_sandbox import verify_native_build
    binary=ROOT/"build/terrain_nav/terrain_nav_query_benchmark"
    verify_native_build(binary)
    if digest(binary)!=plan["settings"]["binary_sha256"]:raise ValueError("native binary changed")
    for name,value in plan["settings"]["source_sha256"].items():
        if digest(ROOT/name)!=value:raise ValueError("planning input generator changed")
    for c in plan["cases"]:
        if digest(BASE/"planning"/c["map_file"])!=c["map_sha256"]:raise ValueError("planning map changed")
    stats, queries, examples=query_analysis(plan,record)
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.1),layout="constrained")
    box(axes[0],[[q["median_wall_ms"] for q in queries if q["scenario"]==f] for f in PLANNING_NAMES],list(PLANNING_NAMES.values()),"每查询3次耗时中位数（毫秒）")
    axes[0].set_yscale("log");axes[0].tick_params(axis="x",rotation=50)
    labels=["完整安全连接","安全局部段","未返回路径","存在不安全返回"]
    bottom=np.zeros(10)
    for label,color in zip(labels,(COLORS[0],"#b0cfc3","#c7cbce","#c36b5d")):
        y=[]
        for f in PLANNING_NAMES:
            qs=[q for q in queries if q["scenario"]==f]
            if label==labels[0]:n=sum(q["complete_safe"] for q in qs)
            elif label==labels[1]:n=sum(q["all_returned"] and q["all_safe"] and not q["complete_safe"] for q in qs)
            elif label==labels[3]:n=sum(q["unsafe_return"] for q in qs)
            else:n=sum(not q["unsafe_return"] and not(q["all_returned"] and q["all_safe"]) for q in qs)
            y.append(n)
        axes[1].bar(range(10),y,bottom=bottom,color=color,label=label);bottom+=np.asarray(y)
    axes[1].set_xticks(range(10),PLANNING_NAMES.values(),rotation=50)
    axes[1].set_ylabel("查询数（每类10；不是30次计时）");axes[1].legend(frameon=False,fontsize=8)
    save(fig,OUT,"S11_planning")
    fig,axes=plt.subplots(1,3,figsize=(11.5,3.5),layout="constrained")
    from matplotlib.patches import Rectangle
    for ax,f in zip(axes,("reverse","detour","step")):
        case=examples[f]["case"];run=examples[f]["run"]
        for b in case["obstacles"]:ax.add_patch(Rectangle((b["x"]-b["hx"],b["y"]-b["hy"]),2*b["hx"],2*b["hy"],color="#73818b"))
        if case["expected_stop"]:ax.axvspan(3.5,9,color="#c36b5d",alpha=.2)
        path=np.asarray(run["path"])
        if len(path):ax.plot(path[:,0],path[:,1],color=COLORS[0],lw=1.8)
        ax.scatter(*case["start"][:2],s=30,color="#538456",label="起点")
        ax.scatter(*case["goal"],s=40,color="#ba594f",marker="*",label="目标")
        ax.set(xlabel="X（米）",ylabel="Y（米）",title=PLANNING_NAMES[f]+"：预定首个样例")
        ax.axis("equal");ax.grid(alpha=.2)
    axes[0].legend(frameon=False)
    save(fig,OUT,"S12_paths")
    return stats,queries,record


def number(value,scale=1,digits=2):
    return "--" if value is None else f"{value*scale:.{digits}f}"


def table_rows(path, rows):
    # Keep the final noalign inside the input file, before TeX sees endinput.
    path.write_text("\n".join(rows)+"\n\\bottomrule\n")


def export_tables(public,mechanism,stats,protocol):
    text=[];details={};gates=[]
    for f,label in FAMILIES.items():
        details[f]={}
        for c in ("point30","hole16","frontier50"):
            ss=subset(public,f,c);h=lambda domain,key:summary(ss,lambda r:r["height"][domain][key])["mean"]
            raw=h("artificial","rmse_m");accepted=h("artificial_accepted","rmse_m")
            picp=summary(ss,lambda r:r["height"]["artificial_accepted"]["picp"][3] if r["height"]["artificial_accepted"]["n"] else None)["mean"]
            width=summary(ss,lambda r:r["height"]["artificial_accepted"]["width_m"][3] if r["height"]["artificial_accepted"]["n"] else None)["mean"]
            n=sum(r["height"]["artificial"]["n"] for r in ss);a=sum(r["height"]["artificial_accepted"]["n"] for r in ss)
            condition_tex=CONDITIONS[c].replace("%",r"\%")
            text.append(f"{label}&{condition_tex}&{number(raw,100)}&{number(accepted,100)}&{a}/{n}&{number(picp,100,1)}&{number(width,100)}\\\\")
            details[f][c]=dict(raw_rmse_m=raw,accepted_rmse_m=accepted,accepted_picp95=picp,accepted_width95_m=width,accepted=a,hidden=n)
            gates.append(dict(family=f,condition=c,accepted_error=accepted is not None and accepted<=protocol["gates"]["accepted_hidden_rmse_m"],
                accepted_calibration=picp is not None and picp>=protocol["gates"]["accepted_hidden_picp95_min"],empty_accepted=a==0))
    table_rows(OUT/"height_rows.tex",text)
    text=[];risk_summary={}
    for phase,rows,names in (("多来源",public,FAMILIES),("解析专项",mechanism,{f:MECHANISMS[f] for f in protocol["mechanism"]["families"]})):
        risk_summary[phase]={}
        for f,label in names.items():
            ss=subset(rows,f);counts={k:sum(r["risk"][k] for r in ss) for k in ("tp","fn","fp","tn")}
            n=counts["tp"]+counts["fn"];negative=counts["fp"]+counts["tn"]
            fnr=counts["fn"]/n if n else None;fpr=counts["fp"]/negative if negative else None
            risk_summary[phase][f]=dict(**counts,fnr=fnr,fpr=fpr,
                safety_screen=fnr is not None and fnr<=protocol["gates"]["unsafe_false_negative_rate_max"])
            text.append(f"{phase}&{label}&{len(ss)}&{n}&{counts['fn']}&{number(fnr,100)}&{number(fpr,100)}\\\\")
    table_rows(OUT/"risk_rows.tex",text)
    text=[]
    for f,label in PLANNING_NAMES.items():
        s=stats[f];text.append(f"{label}&{s['queries']}&{s['unique_layouts']}&{s['complete_safe']}&{s['unsafe_queries']}&{number(s['wall_p50_ms'])}&{number(s['wall_p95_ms'])}\\\\")
    table_rows(OUT/"planning_rows.tex",text)
    timings={}
    text=[]
    for phase,rows in (("多来源104",public),("专项64",mechanism)):
        timings[phase]={}
        for key,label in (("gp_wall_ms","GP拟合及预测"),("risk_wall_ms","占据及BGK/SEB"),("diagnostics_total_ms","两者顺序合计")):
            values=[r[key] for r in rows];q=np.quantile(values,[.5,.95,.99,1])
            timings[phase][key]=dict(zip(("p50_ms","p95_ms","p99_ms","max_ms"),map(float,q)))
            text.append(f"{phase}&{label}&{len(rows)}&"+"&".join(f"{x:.2f}" for x in q)+"\\\\")
    table_rows(OUT/"timing_rows.tex",text)
    conclusions=[]
    for phase,label in (("多来源","880次多来源实验"),("解析专项","480次解析专项实验")):
        rows=risk_summary[phase]
        misses=sum(r["fn"] for r in rows.values());hazards=sum(r["fn"]+r["tp"] for r in rows.values())
        failed=[(FAMILIES if phase=="多来源" else MECHANISMS)[f] for f,r in rows.items() if r["fnr"] is not None and not r["safety_screen"]]
        conclusions.append(f"{label}的风险池化漏判为{misses}/{hazards}（{misses/hazards*100:.2f}\\%）。"
            +("未满足1\\%筛查目标的类别为："+"、".join(failed)+"。" if failed else "有危险分母的类别满足本轮FNR筛查目标。"))
    empty=sum(g["empty_accepted"] for g in gates)
    both=sum(g["accepted_error"] and g["accepted_calibration"] for g in gates)
    conclusions.append(f"在表中15个类别--条件单元中，仅{both}个同时通过接受域误差及覆盖率筛查，{empty}个接受域为空。"
        "空接受域体现拒绝，不是高精度通过；本批没有因失败改变门控阈值。")
    arrival=list(PLANNING_NAMES)[:7]
    complete=sum(stats[f]["complete_safe"] for f in arrival)
    unsafe=sum(s["unsafe_queries"] for s in stats.values())
    conclusions.append(f"70个到达类静态查询中，{complete}个三次均完整安全连接；100个全部查询中有{unsafe}个出现不安全返回。"
        "部分路径不等于车辆已经到达，静态安全也不证明动态闭环安全。")
    gp=timings["多来源104"]["gp_wall_ms"]["p95_ms"];risk=timings["多来源104"]["risk_wall_ms"]["p95_ms"]
    conclusions.append(f"104$\\times$104地图的风险组件P95为{risk:.2f}毫秒，GP每次重新拟合及全图预测P95为{gp:.2f}毫秒。"
        "因此不能以风险组件很快为由声称全链路10Hz，更不能宣布渲染不卡顿。")
    (OUT/"conclusions.tex").write_text("\n\n".join(conclusions)+"\n")
    return dict(height=details,risk=risk_summary,height_gates=gates,timing=timings)


def detailed_statistics(rows):
    result={}
    for f in FAMILIES:
        result[f]={}
        for c in CONDITIONS:
            selected=subset(rows,f,c)
            cell=dict(inputs=len(selected),source_groups=len({r["cluster_id"] for r in selected}))
            for domain in ("artificial","artificial_accepted","natural","natural_accepted"):
                cell[domain]={key:summary(selected,lambda r,key=key:r["height"][domain][key])
                    for key in ("n","rmse_m","mae_m","p95_m","crps_m")}
                for key,index in (("picp95",3),("width95_m",3)):
                    raw="picp" if key=="picp95" else "width_m"
                    cell[domain][key]=summary(selected,lambda r,raw=raw,index=index:r["height"][domain][raw][index]
                        if r["height"][domain]["n"] else None)
            cell["artificial_acceptance"]=summary(selected,proportion)
            result[f][c]=cell
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True);configure_chinese()
    plt.rcParams.update({"font.size":12,"axes.titlesize":12,"axes.labelsize":12,
        "xtick.labelsize":11,"ytick.labelsize":11,"savefig.dpi":220})
    protocol=read(PROTOCOL)
    records={phase:read(BASE/phase/"results.json") for phase in ("public","mechanism")}
    audited={phase:audit(record,phase,protocol) for phase,record in records.items()}
    public=records["public"]["runs"];mechanism=records["mechanism"]["runs"]
    height_figures(public);risk_figures(public,mechanism);examples=spatial_figures(public,mechanism)
    stats,queries,planning=planning_figures()
    details=export_tables(public,mechanism,stats,protocol)
    atomic_json(OUT/"results.json",dict(study_id=protocol["study_id"],audit=audited,summary=details,
        source_statistics=detailed_statistics(public),
        planning=stats,queries=queries,examples=examples,protocol_sha256=digest(PROTOCOL),
        current_source_sha256=fingerprints(),environments={k:v["environment_start"] for k,v in records.items()},
        counts=dict(terrain_computations=len(public)+len(mechanism),planning_calls=len(planning["runs"]),
            planning_queries=len(queries),public_inputs=len({r["dataset_id"] for r in public}),
            public_groups=len({r["cluster_id"] for r in public}),
            mechanism_groups=len({r["cluster_id"] for r in mechanism}))))
    flattened=[]
    for phase,rows in (("public",public),("mechanism",mechanism)):
        for row in rows:
            flat={k:row[k] for k in ("id","family","cluster_id","condition","gp_wall_ms","risk_wall_ms","archive_sha256")}
            flat["phase"]=phase
            for domain in ("artificial","artificial_accepted","natural"):
                for key in ("n","rmse_m","mae_m","crps_m"):flat[domain+"_"+key]=row["height"][domain][key]
            flat.update({"risk_"+k:v for k,v in row["risk"].items()})
            for domain in ("measured_center","unknown_center"):
                flat.update({"risk_"+domain+"_"+k:v for k,v in row["risk_"+domain].items()})
            flattened.append(flat)
    with (OUT/"runs.csv").open("w") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flattened[0]));writer.writeheader();writer.writerows(flattened)
    atomic_json(OUT/"manifest.json",dict(data={str(BASE/p/"results.json"):digest(BASE/p/"results.json") for p in ("public","mechanism")},
        reporter_sha256=digest(Path(__file__)),assets={p.name:digest(p) for p in sorted(OUT.iterdir()) if p.suffix in (".png",".pdf",".tex",".csv")}))
    print("[report audited]",audited,"planning queries",len(queries),flush=True)


if __name__=="__main__":
    main()
