"""Read-only audit, clustered statistics and Chinese figures for campaign v4."""

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
from matplotlib.font_manager import FontProperties, findfont

from comprehensive_study import BASE, PROTOCOL, fingerprints, migration_compatible
from pointcloud_batch import ROOT, atomic_json, digest
from terrain_system_study import evaluation_domains, domain_metrics, risk_metrics
from report_terrain_system import equal, table_rows, box, subset
from plot_paper_scale import plt, save, FAMILIES
from plot_style import configure_chinese, wilson_interval
from controlled_terrain_study import NAMES as MECHANISMS
from paper_aligned_report import query_analysis, NAMES as QUERY_NAMES
from plot_expanded_campaign import SCENARIOS, validate_navigation
from thesis_statistics import threshold_counts, equal_source_curve
from paper_clouds import height_at, grid, raster
from navigation_sandbox import verify_native_build

OUT=ROOT/"docs/reports/comprehensive_study/assets"
DOC=OUT.parent
CN={"point15":"随机删点15%","point30":"随机删点30%","point60":"随机删点60%",
    "hole08":"0.8米孔洞","hole16":"1.6米孔洞","hole32":"3.2米孔洞",
    "frontier25":"单侧隐藏25%","frontier50":"单侧隐藏50%"}
COLORS=("#247b92","#4d8864","#ba7454","#7b6895","#a68a32")
THRESHOLDS=np.r_[np.linspace(.1,.95,18),.58]
STD_LIMITS=np.array([.02,.04,.06,.08,.10,.15,.20])
HIST_EDGES=np.linspace(-5,5,51)
P=None


def read(path):return json.loads(path.read_text())


def group_values(rows,getter):
    groups=defaultdict(list)
    for row in rows:
        value=getter(row)
        if value is not None and np.isfinite(value):groups[row["cluster_id"]].append(float(value))
    return {k:float(np.mean(v)) for k,v in sorted(groups.items())}


def summarize(rows,getter):
    values=np.asarray(list(group_values(rows,getter).values()))
    if not len(values):return dict(groups=0,cases=0,mean=None,median=None,p95=None,ci95=[None,None])
    rng=np.random.default_rng(P["statistics"]["seed"])
    indices=rng.integers(0,len(values),(P["statistics"]["bootstrap_resamples"],len(values)))
    ci=np.quantile(values[indices].mean(axis=1),[.025,.975]).tolist() if len(values)>1 else [None,None]
    return dict(groups=len(values),cases=sum(getter(r) is not None for r in rows),mean=float(values.mean()),
        median=float(np.median(values)),p95=float(np.quantile(values,.95)),ci95=ci)


def ratio(row,domain="artificial"):
    n=row["height"][domain]["n"]
    return row["height"][domain+"_accepted"]["n"]/n if n else None


def check_settings(record):
    if record["settings"]["source_sha256"]!=fingerprints() or record["settings"]["protocol_sha256"]!=digest(PROTOCOL):
        raise ValueError("source/protocol changed; cannot relabel frozen results")


def check_factorial(rows,expected,keys):
    actual=[tuple(r[k] for k in keys) for r in rows]
    if len(actual)!=len(expected) or set(actual)!=set(expected):raise ValueError("incomplete or duplicate factorial")


def audit_execution_history(record,phase):
    key="scripts/sim/experiments/comprehensive_study.py"
    current=record["settings"]["source_sha256"][key]
    allowed={current}
    for history in record.get("execution_history",[]):
        if not migration_compatible(history["settings"],record["settings"]):
            raise ValueError("driver migration changed model/data settings")
        prior=history["previous_driver_sha256"];allowed.add(prior)
        folder=BASE/"provenance/input_refusal_policy"
        if digest(folder/"previous_driver.py")!=prior:raise ValueError("previous driver snapshot changed")
        previous=read(folder/(phase+"_before.json"))
        if previous["settings"]!=history["settings"] or len(previous["runs"])!=history["completed_before_migration"]:
            raise ValueError("migration prefix provenance changed")
        by_id={r["id"]:r for r in record["runs"]}
        for row in previous["runs"]:
            if row["id"] not in by_id or any(value!=by_id[row["id"]].get(k) for k,value in row.items()):
                raise ValueError("pre-migration numerical result changed")
            if by_id[row["id"]]["execution_driver_sha256"]!=prior:raise ValueError("historical driver mislabeled")
    if any(r.get("execution_driver_sha256",current) not in allowed for r in record["runs"]):
        raise ValueError("unrecognized execution driver")


def audit_gp(phase):
    record=read(BASE/phase/"results.json");check_settings(record)
    audit_execution_history(record,phase)
    if record["settings"]["catalog_sha256"]!=digest(BASE/"catalog.json"):raise ValueError("changed catalog")
    rows=record["runs"]
    if phase=="public":
        ids=[d["id"] for d in read(BASE/"catalog.json")["datasets"]]
        expected={(i,c,.02) for i in ids for c in CN}
    else:
        s=P["mechanism"]
        expected={(f+f"_{i:03d}",c,n) for f in s["families"] for i in range(s["seeds_per_family"])
            for c in s["conditions"] for n in s["noise_levels_m"]}
    check_factorial(rows,expected,("dataset_id","condition","noise_std_m"))
    hashes=[]
    for index,row in enumerate(rows):
        path=BASE/phase/row["archive"]
        if digest(path)!=row["archive_sha256"]:raise ValueError("changed prediction array")
        if row["status"]=="input_refused":
            with np.load(path) as a:
                if row["reason"]=="at least eight finite observed points are required" and len(a["ground_xyz"])>=8:
                    raise ValueError("invalid minimum-observation refusal")
            hashes.append(row["id"]+":"+row["archive_sha256"])
            continue
        if row["status"]!="ok" or row["actual_device"]!="cuda" or row["actual_backend"]!="cupy":raise ValueError("unsuccessful GPU run")
        with np.load(path) as a:
            pred=SimpleNamespace(mean=a["mean"],latent_var=a["variance"],accepted=a["accepted"])
            if not np.isfinite(pred.mean).all() or not np.isfinite(pred.latent_var).all() or (pred.latent_var<0).any():raise ValueError("invalid posterior")
            domains=evaluation_domains(a["reference"],a["full_observed"],a["observed"])
            for name,domain in domains.items():
                for suffix,mask in (("",domain),("_accepted",domain&pred.accepted)):
                    m=domain_metrics(pred,a["reference"],mask,P["nominal_levels"])
                    if not equal(m,row["height"][name+suffix]):raise ValueError("height metric mismatch")
            m=risk_metrics(a["predicted_risk"],a["reference_risk"],a["unsafe"],a["risk_supported"],.58)
            if not equal(m,row["risk"]):raise ValueError("risk metric mismatch")
            if (a["outside_hull"]&a["accepted"]).any():raise ValueError("accepted exterior prediction")
            measured=np.broadcast_to(a["observed"].ravel()[a["query_indices"]],a["risk_supported"].shape)
            for label,mask in (("measured",measured),("unknown",~measured)):
                row["risk_"+label]=risk_metrics(a["predicted_risk"],a["reference_risk"],a["unsafe"],a["risk_supported"]&mask,.58)
            row["threshold_counts"]=threshold_counts(a["predicted_risk"],a["unsafe"],a["risk_supported"],THRESHOLDS).tolist()
            std=np.sqrt(a["variance"]);hidden=domains["artificial"]
            row["selective"]=[]
            for limit in STD_LIMITS:
                selected=hidden&a["supported"]&(std<=limit)
                error=a["mean"][selected]-a["reference"][selected]
                row["selective"].append(dict(n=int(selected.sum()),rmse_m=float(np.sqrt(np.mean(error**2))) if len(error) else None,
                    acceptance=float(selected.sum()/hidden.sum()) if hidden.any() else None))
            z=(a["mean"][hidden]-a["reference"][hidden])/np.maximum(std[hidden],1e-9)
            row["standardized_residual_hist"]=np.histogram(z,HIST_EDGES)[0].tolist()
            row["standardized_residual_n"]=len(z)
        hashes.append(row["id"]+":"+row["archive_sha256"])
        if (index+1)%1000==0:print("[v4 array audit]",phase,index+1,flush=True)
    paired=0
    if phase=="mechanism":
        triplets=defaultdict(dict)
        for row in rows:
            if row["status"]=="ok":triplets[row["paired_key"]][row["noise_std_m"]]=row
        for key,pair in triplets.items():
            if set(pair)!={0,.02,.05} or len({r["case_seed"] for r in pair.values()})!=1:raise ValueError("unpaired noise inputs")
            with np.load(BASE/phase/pair[0]["archive"]) as left,np.load(BASE/phase/pair[.02]["archive"]) as mid,np.load(BASE/phase/pair[.05]["archive"]) as right:
                for name in ("reference","observed","reference_risk","unsafe","risk_supported","xy"):
                    if not np.array_equal(left[name],mid[name],equal_nan=True) or not np.array_equal(left[name],right[name],equal_nan=True):
                        raise ValueError("noise triplet changed observation support/reference")
                if not np.array_equal(left["raw_xyz"][:,:2],right["raw_xyz"][:,:2]):raise ValueError("unpaired XY")
                if not np.allclose((mid["raw_xyz"][:,2]-left["raw_xyz"][:,2])/.02,
                    (right["raw_xyz"][:,2]-left["raw_xyz"][:,2])/.05,atol=1e-4):raise ValueError("standardized noise changed")
            paired+=1
    return record,dict(arrays=len(hashes),successful_fits=sum(r["status"]=="ok" for r in rows),
        input_refusals=sum(r["status"]=="input_refused" for r in rows),paired_triplets=paired,
        chain_sha256=hashlib.sha256("\n".join(hashes).encode()).hexdigest())


def audit_resolution():
    record=read(BASE/"discretization/results.json");check_settings(record);rows=record["runs"]
    s=P["discretization"]
    expected={(f"{f}_{P['seed']+1000000+fi*10000+i}",res,yaw) for fi,f in enumerate(s["families"])
        for i in range(s["geometries_per_family"]) for res in s["resolution_m"] for yaw in s["yaw_bins"]}
    check_factorial(rows,expected,("cluster_id","resolution_m","yaw_bins"))
    truth_hash={}
    for row in rows:
        path=BASE/"discretization"/row["archive"]
        if digest(path)!=row["archive_sha256"]:raise ValueError("resolution array changed")
        with np.load(path) as a:
            score=risk_metrics(a["predicted_risk"],a["reference_risk"],a["unsafe"],a["supported"],.58)
            if not equal(score,row["risk"]):raise ValueError("resolution score changed")
            sha=hashlib.sha256(a["xy"].tobytes()+a["yaws"].tobytes()+a["reference_risk"].tobytes()+a["unsafe"].tobytes()).hexdigest()
            if row["cluster_id"] in truth_hash and truth_hash[row["cluster_id"]]!=sha:raise ValueError("physical reference changed between resolutions")
            truth_hash[row["cluster_id"]]=sha
    return record


def audit_native(phase):
    path=BASE/phase;plan=read(path/"cases.json");record=read(path/"runs/results.json")
    if plan["settings"]["risk_input"]!="measured_only" or plan["settings"]["unknown_completion"]:raise ValueError("unexpected navigation completion")
    for name,sha in plan["settings"]["source_sha256"].items():
        if digest(ROOT/name)!=sha:raise ValueError("native source changed")
    binary=ROOT/"build/terrain_nav"/("terrain_nav_query_benchmark" if phase=="planning" else "terrain_nav_benchmark")
    verify_native_build(binary)
    if digest(binary)!=plan["settings"]["binary_sha256"]:raise ValueError("native binary changed")
    for case in plan["cases"]:
        if digest(path/case["map_file"])!=case["map_sha256"]:raise ValueError("native map changed")
    if phase=="planning":query_analysis(plan,record)
    else:validate_navigation(plan,record)
    return plan,record


def fmt(v,scale=1,digits=2):return "--" if v is None else f"{v*scale:.{digits}f}"


def ci_text(s,scale=1):
    if s["mean"] is None:return "--"
    if s["ci95"][0] is None:return f"{s['mean']*scale:.2f} [单组]"
    return f"{s['mean']*scale:.2f} [{s['ci95'][0]*scale:.2f}, {s['ci95'][1]*scale:.2f}]"


def stats_public(rows):
    result={}
    for f in FAMILIES:
        result[f]={}
        for c in CN:
            selected=subset(rows,f,c)
            cell={"cases":len(selected),"groups":len({r["cluster_id"] for r in selected})}
            for domain in ("artificial","artificial_accepted","natural","natural_accepted"):
                cell[domain]={k:summarize(selected,lambda r,k=k,d=domain:r["height"][d][k]) for k in ("rmse_m","mae_m","p95_m","crps_m","n")}
                for label,key in (("picp95","picp"),("width95_m","width_m")):
                    cell[domain][label]=summarize(selected,lambda r,k=key,d=domain:r["height"][d][k][3] if r["height"][d]["n"] else None)
            cell["acceptance"]=summarize(selected,ratio)
            cell["risk"]={k:summarize(selected,lambda r,k=k:r["risk"][k]) for k in ("fnr","fpr","mae","recall","precision")}
            result[f][c]=cell
    return result


def public_figures(rows,stats):
    figures=[]
    for family,title in FAMILIES.items():
        fig,axes=plt.subplots(2,2,figsize=(9.4,7),layout="constrained")
        for ax,key,label,scale in ((axes[0,0],"rmse_m","高程RMSE（厘米）",100),(axes[0,1],"mae_m","高程MAE（厘米）",100)):
            for domain,color,name in (("artificial",COLORS[0],"全部人工隐藏"),("artificial_accepted",COLORS[2],"仅门控通过")):
                values=[stats[family][c][domain][key] for c in CN]
                mean=np.asarray([s["mean"] for s in values],float)*scale
                low=np.asarray([s["ci95"][0] for s in values],float)*scale;high=np.asarray([s["ci95"][1] for s in values],float)*scale
                ax.plot(range(8),mean,"o-",color=color,label=name,ms=4);ax.fill_between(range(8),low,high,color=color,alpha=.12)
            ax.set_ylabel(label);ax.set_yscale("log");ax.legend(frameon=False,fontsize=9)
        accept=[stats[family][c]["acceptance"]["mean"] for c in CN]
        axes[1,0].bar(range(8),np.asarray(accept,float)*100,color=COLORS[0]);axes[1,0].set(ylabel="人工隐藏格接受比例（%）",ylim=(0,102))
        a=[sum(r["height"]["artificial"]["n"] for r in subset(rows,family,c)) for c in CN]
        b=[sum(r["height"]["natural"]["n"] for r in subset(rows,family,c)) for c in CN]
        axes[1,1].plot(range(8),a,"o-",label="人工隐藏参考格",color=COLORS[0]);axes[1,1].plot(range(8),b,"o--",label="自然缺测参考格",color=COLORS[2])
        axes[1,1].set_ylabel("累计可评分格数（非独立样本量）");axes[1,1].legend(frameon=False,fontsize=8)
        for ax in axes.ravel():ax.set_xticks(range(8),CN.values(),rotation=50,ha="right");ax.grid(axis="y",alpha=.2)
        fig.suptitle(title+"：误差、拒绝及评价分母；带状区间为来源组bootstrap",fontsize=13)
        name="T_"+family;save(fig,OUT,name);figures.append(name)
    fig,axes=plt.subplots(1,2,figsize=(9.5,4),layout="constrained")
    for ax,domain,label in zip(axes,("artificial","artificial_accepted"),("全部人工隐藏RMSE（厘米）","接受域RMSE（厘米）")):
        values=[np.asarray(list(group_values(subset(rows,f,"hole16"),lambda r:r["height"][domain]["rmse_m"]).values()))*100 for f in FAMILIES]
        box(ax,values,FAMILIES.values(),label);ax.tick_params(axis="x",rotation=20)
    save(fig,OUT,"T_boxes")
    fig,axes=plt.subplots(1,2,figsize=(9.5,4),layout="constrained")
    for f,color in zip(FAMILIES,COLORS):
        selected=subset(rows,f,"hole16")
        xs=[];ys=[]
        for i in range(len(STD_LIMITS)):
            xs.append(summarize(selected,lambda r,i=i:r["selective"][i]["acceptance"])["mean"])
            ys.append(summarize(selected,lambda r,i=i:r["selective"][i]["rmse_m"])["mean"])
        xx=np.asarray(xs,float)*100;yy=np.asarray(ys,float)*100
        axes[0].plot(xx,yy,"o-",color=color,label=FAMILIES[f],ms=4);axes[0].scatter(xx[4],yy[4],color=color,marker="*",s=90)
        vals=list(group_values(subset(rows,f),lambda r:r["height"]["natural"]["rmse_m"]).values())
        if vals:axes[1].scatter([FAMILIES[f]]*len(vals),np.asarray(vals)*100,s=10,color=color,alpha=.5)
    axes[0].set(xlabel="人工隐藏格接受比例（%）",ylabel="接受域RMSE（厘米）",title="选择性预测：星号为固定10厘米标准差门控")
    axes[1].set(ylabel="自然缺测域来源组RMSE（厘米）",title="有参考的自然缺测单独评价")
    axes[0].legend(frameon=False,fontsize=9);axes[1].tick_params(axis="x",rotation=25)
    for ax in axes:ax.grid(alpha=.2)
    save(fig,OUT,"T_selective")


def uncertainty_figures(rows):
    fig,axes=plt.subplots(1,3,figsize=(10.8,3.9),layout="constrained")
    for ax,domain,title in zip(axes,("artificial","artificial_accepted","natural"),("人工隐藏：全部","人工隐藏：门控通过","自然缺测：有参考位置")):
        for f,color in zip(FAMILIES,COLORS):
            ss=subset(rows,f);values=[summarize(ss,lambda r,i=i:r["height"][domain]["picp"][i] if r["height"][domain]["n"] else None) for i in range(5)]
            y=np.asarray([v["mean"] for v in values],float)*100;lo=np.asarray([v["ci95"][0] for v in values],float)*100;hi=np.asarray([v["ci95"][1] for v in values],float)*100
            ax.plot(np.asarray(P["nominal_levels"])*100,y,"o-",color=color,label=FAMILIES[f],ms=3);ax.fill_between(np.asarray(P["nominal_levels"])*100,lo,hi,color=color,alpha=.08)
        ax.plot([50,100],[50,100],"--",color="#777777");ax.set(xlabel="名义覆盖率（%）",ylabel="实际覆盖率（%）",ylim=(0,103),title=title);ax.grid(alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(),loc="outside upper center",ncols=5,frameon=False)
    save(fig,OUT,"U_calibration")
    fig,axes=plt.subplots(1,2,figsize=(9.5,4),layout="constrained")
    for ax,key,title in zip(axes,("width_m","crps_m"),("接受域95%区间总宽（厘米）","接受域CRPS（厘米，越低越好）")):
        values=[]
        for f in FAMILIES:
            getter=(lambda r:r["height"]["artificial_accepted"][key][3]*100 if r["height"]["artificial_accepted"]["n"] else None) if key=="width_m" else lambda r:r["height"]["artificial_accepted"][key]*100 if r["height"]["artificial_accepted"]["n"] else None
            values.append(list(group_values(subset(rows,f),getter).values()))
        box(ax,values,FAMILIES.values(),title);ax.tick_params(axis="x",rotation=20)
    save(fig,OUT,"U_sharpness")
    fig,ax=plt.subplots(figsize=(7,3.8),layout="constrained")
    centers=(HIST_EDGES[:-1]+HIST_EDGES[1:])/2
    for f,color in zip(FAMILIES,COLORS):
        groups=defaultdict(list)
        for r in subset(rows,f,"hole16"):
            if r["standardized_residual_n"]:groups[r["cluster_id"]].append(np.asarray(r["standardized_residual_hist"])/r["standardized_residual_n"]/.2)
        if groups:
            value=np.mean([np.mean(v,axis=0) for v in groups.values()],axis=0)
            ax.plot(centers,value,color=color,label=FAMILIES[f])
    ax.plot(centers,np.exp(-centers**2/2)/np.sqrt(2*np.pi),"--",color="#555555",label="标准正态参考，不是实测方法")
    ax.set(xlabel="标准化残差（预测−参考）/模型标准差",ylabel="来源组等权密度",title="标准化误差：宽区间是否掩盖了精度问题")
    ax.legend(frameon=False,ncols=2,fontsize=9);ax.grid(alpha=.2);save(fig,OUT,"U_residuals")


def risk_figures(rows):
    fig,axes=plt.subplots(1,2,figsize=(9.5,4),layout="constrained")
    for ax,key,label in zip(axes,("fnr","fpr"),("危险漏判 FNR（%）","安全误拒 FPR（%）")):
        box(ax,[np.asarray(list(group_values(subset(rows,f),lambda r:r["risk"][key]).values()))*100 for f in FAMILIES],FAMILIES.values(),label)
        ax.tick_params(axis="x",rotation=20)
    axes[0].axhline(1,color="#ba594f",ls="--");save(fig,OUT,"R_boxes")
    fig,axes=plt.subplots(1,2,figsize=(9.5,4),layout="constrained")
    for ax,key,title in zip(axes,("fnr","fpr"),("危险漏判（%）：包括有参考的未知中心","安全误拒（%）：包括保守未知拒绝")):
        matrix=np.array([[summarize(subset(rows,f,c),lambda r:r["risk"][key])["mean"] for c in CN] for f in FAMILIES],float)*100
        im=ax.imshow(matrix,cmap="YlOrRd",aspect="auto",vmin=0);ax.set_xticks(range(8),CN.values(),rotation=55,ha="right");ax.set_yticks(range(5),FAMILIES.values());ax.set_title(title,fontsize=10)
        for (i,j),v in np.ndenumerate(matrix):ax.text(j,i,"--" if not np.isfinite(v) else f"{v:.1f}",ha="center",va="center",fontsize=8,bbox=dict(facecolor="white",alpha=.7,edgecolor="none",pad=.5))
        fig.colorbar(im,ax=ax,shrink=.7)
    save(fig,OUT,"R_conditions")
    fig,axes=plt.subplots(1,2,figsize=(9.5,4),layout="constrained")
    for f,color in zip(FAMILIES,COLORS):
        groups=defaultdict(list)
        for r in subset(rows,f):groups[r["cluster_id"]].append(r["threshold_counts"])
        counts=np.array([np.asarray(v).sum(axis=0) for v in groups.values()]);curve=equal_source_curve(counts)
        axes[0].plot(curve["fpr"][:-1]*100,curve["recall"][:-1]*100,".-",color=color,label=FAMILIES[f]);axes[1].plot(curve["recall"][:-1]*100,curve["precision"][:-1]*100,".-",color=color)
        axes[0].scatter(curve["fpr"][-1]*100,curve["recall"][-1]*100,marker="*",color=color,s=70);axes[1].scatter(curve["recall"][-1]*100,curve["precision"][-1]*100,marker="*",color=color,s=70)
    axes[0].set(xlabel="安全误拒率（%）",ylabel="危险检出率（%）",title="阈值折中：来源组等权混淆质量")
    axes[1].set(xlabel="危险检出率（%）",ylabel="危险判定精确率（%）",title="星号是冻结阈值0.58，不据图重新选参")
    axes[0].legend(frameon=False,fontsize=9)
    for ax in axes:ax.grid(alpha=.2);ax.set_ylim(0,102)
    save(fig,OUT,"R_thresholds")
    fig,ax=plt.subplots(figsize=(7.5,4),layout="constrained")
    for offset,domain,label,color in ((-.18,"measured","中心有实测",COLORS[0]),(.18,"unknown","中心无实测",COLORS[2])):
        values=[]
        for f in FAMILIES:
            ss=subset(rows,f);fp=sum(r["risk_"+domain]["fp"] for r in ss);tn=sum(r["risk_"+domain]["tn"] for r in ss)
            values.append(100*fp/(fp+tn) if fp+tn else np.nan)
        ax.bar(np.arange(5)+offset,values,width=.34,color=color,label=label)
    ax.set_xticks(range(5),FAMILIES.values());ax.set(ylabel="参考安全状态误拒率（%）",ylim=(0,105));ax.legend(frameon=False);ax.grid(axis="y",alpha=.2)
    save(fig,OUT,"R_observation")


def paired_noise(rows,key):
    groups=defaultdict(dict)
    for row in rows:groups[row["paired_key"]][row["noise_std_m"]]=row
    result=[]
    for pair in groups.values():
        zero=key(pair[0])
        for noise in (.02,.05):
            value=key(pair[noise])
            if value is not None and zero is not None:result.append(dict(cluster_id=pair[noise]["cluster_id"],family=pair[noise]["family"],noise=noise,value=value-zero))
    return result


def noise_figures(rows):
    families=P["mechanism"]["families"];levels=P["mechanism"]["noise_levels_m"]
    fig,axes=plt.subplots(2,2,figsize=(9.5,7),layout="constrained")
    getters=(lambda r:r["height"]["artificial"]["rmse_m"]*100 if r["height"]["artificial"]["n"] else None,
        lambda r:r["height"]["artificial_accepted"]["picp"][3]*100 if r["height"]["artificial_accepted"]["n"] else None,
        lambda r:r["risk"]["fnr"]*100 if r["risk"]["fnr"] is not None else None,
        lambda r:r["risk"]["fpr"]*100 if r["risk"]["fpr"] is not None else None)
    labels=("人工隐藏RMSE（厘米）","接受域PICP95（%）","危险漏判FNR（%）","安全误拒FPR（%）")
    for ax,getter,label in zip(axes.ravel(),getters,labels):
        for fi,f in enumerate(families):
            values=[summarize([r for r in rows if r["family"]==f and r["noise_std_m"]==n],getter) for n in levels]
            y=np.asarray([v["mean"] for v in values],float);lo=np.asarray([v["ci95"][0] for v in values],float);hi=np.asarray([v["ci95"][1] for v in values],float)
            color=plt.get_cmap("tab10")(fi);ax.plot(np.asarray(levels)*100,y,"o-",color=color,label=MECHANISMS[f],ms=3);ax.fill_between(np.asarray(levels)*100,lo,hi,color=color,alpha=.08)
        ax.set(xlabel="实际注入Z噪声标准差（厘米）",ylabel=label);ax.grid(alpha=.2)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc="outside upper center",ncols=4,frameon=False,fontsize=9)
    save(fig,OUT,"N_response")
    deltas=paired_noise(rows,lambda r:r["height"]["artificial"]["rmse_m"])
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout="constrained")
    for ax,noise in zip(axes,(.02,.05)):
        vals=[np.asarray(list(group_values([r for r in deltas if r["family"]==f and r["noise"]==noise],lambda r:r["value"]).values()))*100 for f in families]
        box(ax,vals,[MECHANISMS[f] for f in families],"相对零噪声的配对RMSE变化（厘米）");ax.tick_params(axis="x",rotation=45);ax.axhline(0,color="#777777",lw=1);ax.set_title(f"注入噪声 {noise*100:.0f} 厘米")
    save(fig,OUT,"N_paired")
    fig,axes=plt.subplots(1,3,figsize=(10.5,3.8),layout="constrained")
    for ax,c in zip(axes,P["mechanism"]["conditions"]):
        matrix=[]
        for f in families:
            matrix.append([summarize([r for r in rows if r["family"]==f and r["condition"]==c and r["noise_std_m"]==n],lambda r:r["risk"]["fnr"])["mean"] for n in levels])
        matrix=np.asarray(matrix,float)*100;im=ax.imshow(matrix,cmap="YlOrRd",aspect="auto",vmin=0)
        ax.set_xticks(range(3),["0厘米","2厘米","5厘米"]);ax.set_yticks(range(8),[MECHANISMS[f] for f in families]);ax.set_title(CN[c])
        for (i,j),v in np.ndenumerate(matrix):ax.text(j,i,"--" if not np.isfinite(v) else f"{v:.1f}",ha="center",va="center",fontsize=8,bbox=dict(facecolor="white",alpha=.65,edgecolor="none",pad=.4))
        fig.colorbar(im,ax=ax,shrink=.65,label="FNR（%）")
    save(fig,OUT,"N_interactions")
    return deltas


def resolution_figures(rows):
    families=P["discretization"]["families"];resolutions=P["discretization"]["resolution_m"];bins=P["discretization"]["yaw_bins"]
    fig,axes=plt.subplots(2,2,figsize=(9,7),layout="constrained")
    for ax,f in zip(axes.ravel(),families):
        matrix=np.array([[summarize([r for r in rows if r["family"]==f and r["resolution_m"]==res and r["yaw_bins"]==yaw],lambda r:r["risk"]["fnr"])["mean"] for yaw in bins] for res in resolutions],float)*100
        im=ax.imshow(matrix,cmap="YlOrRd",aspect="auto",vmin=0);ax.set_xticks(range(3),bins);ax.set_yticks(range(3),resolutions);ax.set(xlabel="航向栅格数",ylabel="位置分辨率（米）",title=MECHANISMS[f])
        for (i,j),v in np.ndenumerate(matrix):ax.text(j,i,f"{v:.2f}%",ha="center",va="center",fontsize=10,bbox=dict(facecolor="white",alpha=.7,edgecolor="none",pad=.5))
        fig.colorbar(im,ax=ax,label="危险漏判（%）",shrink=.75)
    save(fig,OUT,"D_resolution")
    fig,axes=plt.subplots(2,2,figsize=(9.5,7),layout="constrained")
    selected=[r for r in rows if r["resolution_m"]==.2 and r["yaw_bins"]==36]
    for ax,f in zip(axes.ravel(),families):
        for key,label,color in (("fnr","危险漏判",COLORS[0]),("fpr","安全误拒",COLORS[2])):
            vals=[r["risk"][key]*100 for r in selected if r["family"]==f and r["risk"][key] is not None]
            x=[r["spec"]["ridge_width"] if f=="ridge" else r["spec"]["column_radius"] if f=="column" else r["spec"][f] for r in selected if r["family"]==f and r["risk"][key] is not None]
            if f=="slope_boundary":x=np.rad2deg(x)
            ax.scatter(x,vals,color=color,s=17,alpha=.7,label=label)
        labels={"slope_boundary":"坡面角（度）","ridge":"窄脊标准差宽度（米）","step_high":"台阶高度（米）","column":"柱体半径（米）"}
        ax.set(ylabel="状态比例（%）",xlabel=labels[f],title=MECHANISMS[f]);ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=9)
    save(fig,OUT,"D_geometry")
    # Fixed identifiers for typical cases; worst-case selection is explicitly diagnostic.
    fig,axes=plt.subplots(3,3,figsize=(9.5,9),layout="constrained")
    failures={}
    for line,f in enumerate(("ridge","step_high","column")):
        candidates=[r for r in selected if r["family"]==f]
        row=max(candidates,key=lambda r:r["risk"]["fnr"] or -1);failures[f]=row["id"]
        with np.load(BASE/"discretization"/row["archive"]) as a:
            fn=(a["predicted_risk"]<.58)&a["unsafe"];yaw=int(np.argmax(fn.sum(axis=1)))
            fields=(a["reference_risk"][yaw].reshape(11,11),a["predicted_risk"][yaw].reshape(11,11),fn[yaw].reshape(11,11).astype(float))
            for ax,field,title in zip(axes[line],fields,("独立足迹参考","网格风险输出","危险漏判位置")):
                im=ax.imshow(field,origin="lower",extent=[-2.2,2.2,-2.2,2.2],cmap="YlOrRd",vmin=0,vmax=1)
                ax.set(xlabel="X（米）",ylabel="Y（米）",title=MECHANISMS[f]+" / "+title)
            fig.colorbar(im,ax=list(axes[line]),shrink=.7,label="风险分数 / 漏判指示")
    save(fig,OUT,"D_failures")
    return failures


def source_quality(catalog):
    rows=[]
    for item in catalog["datasets"]:
        path=BASE/item["path"]
        if digest(path)!=item["sha256"]:raise ValueError("source changed")
        with np.load(path) as a:
            shape=tuple(a["shape"]);xy=grid(shape,a["query_origin"],.2)
            ref=height_at(a["truth_height"],a["truth_origin"],float(a["truth_resolution"]),xy)
            measured=raster(a["xyz"][a["is_ground"]],shape,a["query_origin"],.2).ravel()
            valid=np.isfinite(ref)&np.isfinite(measured)
            error=measured[valid]-ref[valid]
            rows.append(dict(id=item["id"],family=item["family"],cluster_id=item["cluster_id"],
                input_points=len(a["xyz"]),ground_points=int(a["is_ground"].sum()),
                reference_fraction=float(np.isfinite(ref).mean()),observed_fraction=float(np.isfinite(measured).mean()),
                raster_reference_rmse_m=float(np.sqrt(np.mean(error**2))) if len(error) else None,
                reference_kind=item["reference_kind"],source_sha256=item["sha256"]))
    fig,axes=plt.subplots(1,3,figsize=(10,4),layout="constrained")
    for ax,key,label,scale in zip(axes,("ground_points","reference_fraction","raster_reference_rmse_m"),
        ("输入地面点数 / 子图","独立参考有限值比例（%）","原始栅格与来源参考偏差（厘米）"),(1,100,100)):
        values=[np.asarray(list(group_values(subset(rows,f),lambda r:r[key]).values()))*scale for f in FAMILIES]
        box(ax,values,FAMILIES.values(),label);ax.tick_params(axis="x",rotation=30)
    axes[0].set_yscale("log");axes[2].set_yscale("log");save(fig,OUT,"A_sources")
    atomic_json(OUT/"source_audit.json",dict(datasets=rows,groups=catalog["source_groups"],
        interpretation="Input representation residual is before GP, not independent localization/registration truth. Sparse public reference coverage is explicit; no cases filtered."))
    return rows


def refusal_figure(records):
    rows=[r for record in records for r in record["runs"] if r["status"]=="input_refused"]
    matrix=np.array([[sum(r["family"]==f and r["condition"]==c for r in rows) for c in CN] for f in FAMILIES])
    fig,ax=plt.subplots(figsize=(8,3.8),layout="constrained")
    im=ax.imshow(matrix,cmap="YlOrRd",aspect="auto",vmin=0)
    ax.set_xticks(range(8),CN.values(),rotation=35,ha="right");ax.set_yticks(range(5),FAMILIES.values())
    for (i,j),v in np.ndenumerate(matrix):ax.text(j,i,str(v),ha="center",va="center",fontsize=11)
    fig.colorbar(im,ax=ax,label="输入不足而拒绝拟合的条件数")
    save(fig,OUT,"A_refusals")
    export_csv("input_refusals.csv",rows)
    return len(rows)


def precision_figure(rows):
    fig,axes=plt.subplots(1,2,figsize=(9.5,4),layout="constrained")
    for f,color in zip(FAMILIES,COLORS):
        values=np.asarray(list(group_values(subset(rows,f,"hole16"),lambda r:r["height"]["artificial"]["rmse_m"]).values()))
        if not len(values):continue
        order=np.random.default_rng(P["statistics"]["seed"]).permutation(len(values));values=values[order]
        sizes=sorted({n for n in (5,10,20,30,50,75,100,len(values)) if n<=len(values)})
        means=[];width=[]
        for n in sizes:
            selected=values[:n];rng=np.random.default_rng(P["statistics"]["seed"])
            lo,hi=np.quantile(selected[rng.integers(0,n,(4000,n))].mean(axis=1),[.025,.975])
            means.append(selected.mean()*100);width.append((hi-lo)*100)
        axes[0].plot(sizes,means,"o-",color=color,label=FAMILIES[f]);axes[1].plot(sizes,width,"o-",color=color)
    axes[0].set(xlabel="累计来源组数量（预定随机顺序）",ylabel="孔洞RMSE组均值（厘米）")
    axes[1].set(xlabel="累计来源组数量",ylabel="均值95%置信区间总宽（厘米）")
    axes[0].legend(frameon=False,fontsize=9)
    for ax in axes:ax.grid(alpha=.2)
    save(fig,OUT,"A_precision")


def spatial_layers(data, joint, *, max_std_m=.04, max_distance_m=.35):
    from fsgp_bgk_se2.supported_completion import supported_completion

    shape=data["measured"].shape
    observed=np.asarray(data["observed"],bool).reshape(shape)
    measured=np.asarray(data["measured"])
    if not np.array_equal(np.isfinite(measured),observed):
        raise ValueError("measured support does not match observed cells")
    for name in ("measured","observed","occupied","mean","variance","accepted","nearest","xy","reference","origin","resolution"):
        if not np.array_equal(data[name],joint[name],equal_nan=True):
            raise ValueError("spatial joint replay changed "+name)
    prediction=SimpleNamespace(mean=data["mean"],latent_var=data["variance"],accepted=data["accepted"],
        nearest_distance=data["nearest"],local_conditioned=joint["local"])
    completed,_,valid,filled=supported_completion(measured,np.full(shape,.0004,np.float32),observed,prediction,
        max_std_m=max_std_m,max_distance_m=max_distance_m,blocked=data["occupied"])
    if not np.array_equal(filled,joint["filled"]):raise ValueError("spatial completion differs from archived experiment")
    if not np.array_equal(completed[observed],measured[observed]) or (observed&~valid).any():
        raise ValueError("spatial completion removed or modified measurements")
    reference=data["reference"].reshape(shape)
    artificial=evaluation_domains(data["reference"],data["full_observed"],observed)["artificial"].reshape(shape)
    accepted=data["accepted"].reshape(shape);supported=data["supported"].reshape(shape)
    mean=data["mean"].reshape(shape)
    # GP acceptance is a prediction diagnostic, never a mask on existing measurements.
    status=np.zeros(shape,np.int8)
    status[observed&accepted]=1;status[observed&~accepted]=2;status[~observed&accepted]=3
    audit=dict(reference_cells=int(np.isfinite(reference).sum()),observed_cells=int(observed.sum()),
        observed_without_reference=int((observed&~np.isfinite(reference)).sum()),finite_gp_means=int(np.isfinite(mean).sum()),
        old_plot_omitted_observed=int((observed&~accepted).sum()),
        observed_unsupported=int((observed&~supported).sum()),observed_variance_rejected=int((observed&supported&~accepted).sum()),
        artificial_hidden_cells=int(artificial.sum()),hidden_gp_accepted=int((artificial&accepted).sum()),
        hidden_filled=int((artificial&filled).sum()),filled_cells=int(filled.sum()),
        filled_without_reference=int((filled&~np.isfinite(reference)).sum()),completed_cells=int(valid.sum()),
        observed_lost=0,observed_max_change_m=float(np.max(np.abs(completed[observed]-measured[observed]))) if observed.any() else 0.,
        hidden_rmse_cm=float(100*np.sqrt(np.mean((mean[artificial]-reference[artificial])**2))) if artificial.any() else None)
    return dict(reference=reference,measured=measured,completed=np.where(valid,completed,np.nan),mean=mean,
        std_cm=np.sqrt(data["variance"].reshape(shape))*100,status=status,filled=filled,
        error_cm=np.where(artificial,np.abs(mean-reference)*100,np.nan),audit=audit)


def grid_surface(ax, xx, yy, height, resolution, cmap, vmin, vmax):
    from matplotlib.colors import Normalize
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    finite=np.isfinite(height);x=xx[finite];y=yy[finite];z=height[finite]
    # Draw every finite cell, without triangulating or bridging any unknown cell.
    offsets=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*resolution/2
    vertices=np.stack((x[:,None]+offsets[:,0],y[:,None]+offsets[:,1],np.broadcast_to(z[:,None],(len(z),4))),axis=-1)
    colors=plt.get_cmap(cmap)(Normalize(vmin,vmax)(z))
    artist=Poly3DCollection(vertices,facecolors=colors,edgecolors="none",linewidths=0,antialiased=False,rasterized=True)
    ax.add_collection3d(artist)
    ax.auto_scale_xyz([xx.min()-resolution/2,xx.max()+resolution/2],
        [yy.min()-resolution/2,yy.max()+resolution/2],[vmin,vmax])
    return plt.cm.ScalarMappable(norm=Normalize(vmin,vmax),cmap=cmap)


def spatial_figures(rows):
    from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm
    from matplotlib.patches import Patch, Rectangle
    from joint_integration_study import PROTOCOL as JOINT_PROTOCOL

    joint_record=read(BASE/"integration/results.json");joint_plan=read(JOINT_PROTOCOL)
    lookup={(r["dataset_id"],r["condition"]):r for r in joint_record["runs"]}
    examples={};audits=[]
    for f,label in FAMILIES.items():
        row=sorted(subset(rows,f,"hole16"),key=lambda r:r["dataset_id"])[0];examples[f]=row["id"]
        jrow=lookup[row["dataset_id"],row["condition"]]
        if digest(BASE/"public"/row["archive"])!=row["archive_sha256"] or digest(BASE/"integration"/jrow["archive"])!=jrow["archive_sha256"]:
            raise ValueError("spatial example archive changed")
        with np.load(BASE/"public"/row["archive"]) as a,np.load(BASE/"integration"/jrow["archive"]) as joint:
            layers=spatial_layers(a,joint,max_std_m=joint_plan["max_std_m"],max_distance_m=joint_plan["max_distance_m"])
            audit=dict(family=f,id=row["id"],public_archive_sha256=row["archive_sha256"],joint_archive_sha256=jrow["archive_sha256"],**layers["audit"])
            audits.append(audit)
            shape=a["measured"].shape;reference=layers["reference"];res=float(a["resolution"])
            extent=[a["origin"][0],a["origin"][0]+shape[1]*res,a["origin"][1],a["origin"][1]+shape[0]*res]
            center=a["origin"]+np.array([shape[1],shape[0]])*res/2
            fields=[layers[k] for k in ("reference","measured","completed","error_cm")]
            zmin=min(np.nanmin(field) for field in fields[:3]);zmax=max(np.nanmax(field) for field in fields[:3])
            titles=(f"(a) 来源参考：{audit['reference_cells']}格",f"(b) 删除后实测：{audit['observed_cells']}格",
                f"(c) 保留实测 + 有界补全：新增{audit['filled_cells']}格","(d) 隐藏区GP均值绝对误差")
            fig,axes=plt.subplots(2,2,figsize=(9.5,7.8),layout="constrained")
            for i,(ax,field,title) in enumerate(zip(axes.ravel(),fields,titles)):
                im=ax.imshow(field,origin="lower",extent=extent,cmap="terrain" if i<3 else "magma",
                    vmin=zmin if i<3 else 0,vmax=zmax if i<3 else None,interpolation="nearest")
                ax.set(xlabel="局部X（米）",ylabel="局部Y（米）",title=title);fig.colorbar(im,ax=ax,shrink=.8,label="高程（米）" if i<3 else "误差（厘米）")
                if i<3:ax.add_patch(Rectangle(center-.8,1.6,1.6,fill=False,edgecolor="#b44f4f",lw=1))
                if i==3:
                    zoom=ax.inset_axes([.57,.55,.39,.39])
                    zoom.imshow(field,origin="lower",extent=extent,cmap="magma",vmin=0,vmax=im.get_clim()[1],interpolation="nearest")
                    zoom.set(xlim=(center[0]-1.2,center[0]+1.2),ylim=(center[1]-1.2,center[1]+1.2))
                    zoom.set_title("孔洞附近放大（同色标）",fontsize=8);zoom.tick_params(labelsize=7)
            fig.suptitle(label+"：参考、实测与补全不是同一层\n(c) 为单独接入专项，主实验补全关闭；白色不是安全",fontsize=13)
            save(fig,OUT,"S_"+f)

            fig,axes=plt.subplots(1,3,figsize=(12,4.8),layout="constrained")
            im=axes[0].imshow(layers["mean"],origin="lower",extent=extent,cmap="terrain",
                vmin=zmin,vmax=zmax,interpolation="nearest")
            fig.colorbar(im,ax=axes[0],shrink=.75,label="高程（米）")
            axes[0].set_title(f"GP原始均值：{audit['finite_gp_means']}格")
            std=layers["std_cm"];im=axes[1].imshow(std,origin="lower",extent=extent,cmap="cividis",
                norm=LogNorm(vmin=2,vmax=max(2.01,float(np.nanmax(std)))),interpolation="nearest")
            fig.colorbar(im,ax=axes[1],shrink=.75,label="预测标准差（厘米，对数色标）")
            axes[1].contour(layers["std_cm"]>10,levels=[.5],origin="lower",extent=extent,colors="#b44f4f",linewidths=.5)
            axes[1].set_title("标准差：红线为10厘米门槛")
            palette=["#e1e5e8","#527d93","#d99b45","#459e80"]
            axes[2].imshow(layers["status"],origin="lower",extent=extent,cmap=ListedColormap(palette),
                norm=BoundaryNorm(np.arange(-.5,4.5),4),interpolation="nearest")
            axes[2].set_title(f"GP门控：{audit['old_plot_omitted_observed']}个实测格被拒")
            for ax in axes:
                ax.set(xlabel="局部X（米）",ylabel="局部Y（米）")
                ax.add_patch(Rectangle(center-.8,1.6,1.6,fill=False,edgecolor="#b44f4f",lw=1))
            fig.legend(handles=[Patch(facecolor=c,label=t) for c,t in zip(palette,
                ("未知且GP拒绝","实测且GP接受","实测但GP拒绝（仍保留实测）","未知且GP接受（仅候选）"))],
                loc="outside lower center",ncols=2,frameon=False,fontsize=10)
            fig.suptitle(label+"：GP诊断层不应遮掉实测；GP接受也不等于通过有界补全",fontsize=13)
            save(fig,OUT,"S_gp_"+f)
            if f in ("hilly","road"):
                xx=a["xy"][:,0].reshape(shape);yy=a["xy"][:,1].reshape(shape)
                fig=plt.figure(figsize=(10,8),layout="constrained")
                for i,(field,title) in enumerate(zip(fields,("来源参考：逐格高程面","删除后实测：不跨空白连面","保留实测 + 有界补全（专项）","隐藏处绝对误差（厘米）"))):
                    ax=fig.add_subplot(2,2,i+1,projection="3d")
                    surf=grid_surface(ax,xx,yy,field,res,"terrain" if i<3 else "magma",
                        zmin if i<3 else 0,zmax if i<3 else np.nanmax(field))
                    ax.set(xlabel="X / m",ylabel="Y / m",zlabel="Z / m" if i<3 else "误差 / cm",title=title+f"\n有值{np.isfinite(field).sum()}格")
                    if i<3:ax.set_zlim(zmin,zmax)
                    if i<3:
                        square=center+np.array([[-.8,-.8],[.8,-.8],[.8,.8],[-.8,.8],[-.8,-.8]])
                        ax.plot(square[:,0],square[:,1],np.full(5,zmin),color="#b44f4f",lw=1.2)
                    ax.view_init(35,-60);fig.colorbar(surf,ax=ax,shrink=.4,pad=.16)
                save(fig,OUT,"S_3d_"+f)
    export_csv("spatial_audit.csv",audits)
    table_rows(OUT/"spatial_audit_rows.tex",[f"{FAMILIES[a['family']]}&{a['observed_cells']}&{a['old_plot_omitted_observed']}&"
        f"{a['artificial_hidden_cells']}&{a['hidden_gp_accepted']}&{a['hidden_filled']}\\\\" for a in audits])
    (OUT/"spatial_findings.tex").write_text("\n".join(
        f"{FAMILIES[a['family']]}样例：旧图将{a['old_plot_omitted_observed']}个已有实测格误画为空白，其中"
        f"{a['observed_unsupported']}个未通过支撑门控、{a['observed_variance_rejected']}个因预测方差被拒；"
        f"实测保留的有界补全丢失实测为0格。人工隐藏{a['artificial_hidden_cells']}格中，"
        f"普通GP候选{a['hidden_gp_accepted']}格，专项实际填入{a['hidden_filled']}格；"
        f"全部隐藏格的原始GP均值RMSE为{a['hidden_rmse_cm']:.2f}厘米。\\par"
        for a in audits)+"\n")
    return examples,audits


def scaling_figures(record):
    if record["protocol_sha256"]!=digest(PROTOCOL) or len(record["runs"])!=P["scaling"]["planned_measured_calls"]:raise ValueError("scaling mismatch")
    for name,value in record["source_sha256"].items():
        if digest(ROOT/name)!=value:raise ValueError("scaling source changed")
    rows=record["runs"]
    expected={(size,yaw,backend,i) for size in P["scaling"]["sizes"] for yaw in P["scaling"]["yaw_bins"]
        for backend in ("cpu","cupy") for i in range(1+P["scaling"]["warmup_calls"],1+P["scaling"]["warmup_calls"]+P["scaling"]["measured_calls"])}
    check_factorial(rows,expected,("size","yaw_bins","backend","repeat"))
    fig,axes=plt.subplots(1,3,figsize=(10,4),layout="constrained");data={}
    for backend in ("cpu","cupy"):
        data[backend]=np.array([[np.quantile([r["elapsed_ms"] for r in rows if r["backend"]==backend and r["size"]==size and r["yaw_bins"]==yaw],.95)
            for yaw in P["scaling"]["yaw_bins"]] for size in P["scaling"]["sizes"]])
    for ax,matrix,title in zip(axes,(data["cpu"],data["cupy"],data["cpu"]/data["cupy"]),("CPU P95（毫秒）","GPU P95（毫秒）","CPU/GPU P95比值（倍）")):
        im=ax.imshow(matrix,cmap="YlGnBu",aspect="auto");ax.set_xticks(range(3),P["scaling"]["yaw_bins"]);ax.set_yticks(range(3),P["scaling"]["sizes"])
        ax.set(xlabel="航向格数",ylabel="方形地图边长（格）",title=title)
        for (i,j),v in np.ndenumerate(matrix):ax.text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=9,bbox=dict(facecolor="white",alpha=.65,edgecolor="none",pad=.5))
        fig.colorbar(im,ax=ax,shrink=.7)
    save(fig,OUT,"L_scaling")
    fig,axes=plt.subplots(1,2,figsize=(9.5,4),layout="constrained")
    for ax,backend in zip(axes,("cpu","cupy")):
        keys=[(size,yaw) for size in P["scaling"]["sizes"] for yaw in P["scaling"]["yaw_bins"]]
        box(ax,[[r["elapsed_ms"] for r in rows if r["backend"]==backend and (r["size"],r["yaw_bins"])==key] for key in keys],
            [f"{size}×{size}\n{yaw}航向" for size,yaw in keys],"同步组件墙钟（毫秒）")
        ax.set_title(backend.upper()+"：每负载30次，不是30个地形");ax.tick_params(axis="x",rotation=45);ax.set_yscale("log")
    save(fig,OUT,"L_boxes")
    return {backend:matrix.tolist() for backend,matrix in data.items()}


def gp_latency_figures(rows):
    for row in rows:
        if row["gp_wall_ms"]<=0 or row["risk_wall_ms"]<=0 or not np.isclose(row["diagnostics_total_ms"],row["gp_wall_ms"]+row["risk_wall_ms"]):
            raise ValueError("invalid stage timing identity")
    fig,axes=plt.subplots(1,3,figsize=(11,4.1),layout="constrained")
    keys=("gp_wall_ms","risk_wall_ms","diagnostics_total_ms")
    labels=("GP拟合与查询墙钟（毫秒）","占据检测与风险墙钟（毫秒）","两个计时阶段之和（毫秒）")
    for ax,key,label in zip(axes,keys,labels):
        box(ax,[[r[key] for r in subset(rows,f)] for f in FAMILIES],FAMILIES.values(),label)
        ax.set_yscale("log");ax.tick_params(axis="x",rotation=35)
    save(fig,OUT,"L_gp_latency")
    result={};table=[]
    for f,label in FAMILIES.items():
        selected=subset(rows,f)
        result[f]={k:{"p50_ms":float(np.quantile([r[k] for r in selected],.5)),
            "p95_ms":float(np.quantile([r[k] for r in selected],.95)),
            "p99_ms":float(np.quantile([r[k] for r in selected],.99))} for k in keys}
        s=result[f]
        table.append(f"{label}&{len(selected)}&{s['gp_wall_ms']['p50_ms']:.2f}&{s['gp_wall_ms']['p95_ms']:.2f}&"
            f"{s['risk_wall_ms']['p95_ms']:.2f}&{s['diagnostics_total_ms']['p95_ms']:.2f}\\\\")
    table_rows(OUT/"gp_timing_rows.tex",table)
    return result


def planning_figures(plan,record):
    stats,queries,examples=query_analysis(plan,record)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout="constrained")
    box(axes[0],[[q["median_wall_ms"] for q in queries if q["scenario"]==f and q["median_wall_ms"] is not None] for f in QUERY_NAMES],
        QUERY_NAMES.values(),"同查询3次耗时中位数（毫秒）")
    axes[0].set_yscale("log");axes[0].tick_params(axis="x",rotation=55)
    for f,color in zip(list(QUERY_NAMES)[:7],plt.get_cmap("tab10").colors):
        a=np.sort([q["median_wall_ms"] for q in queries if q["scenario"]==f and q["median_wall_ms"] is not None])
        axes[1].plot(a,np.arange(1,len(a)+1)/len(a),color=color,label=QUERY_NAMES[f])
    axes[1].set(xlabel="每查询耗时中位数（毫秒）",ylabel="查询累计比例");axes[1].set_xscale("log");axes[1].legend(frameon=False,ncols=2,fontsize=8)
    save(fig,OUT,"Q_latency")
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout="constrained")
    names=list(QUERY_NAMES);bottom=np.zeros(10)
    for label,color in (("完整安全连接",COLORS[0]),("安全局部段","#afcaba"),("未返回/异常","#bcbfc3"),("出现不安全返回","#c06a5e")):
        values=[]
        for f in names:
            qs=[q for q in queries if q["scenario"]==f]
            if label=="完整安全连接":n=sum(q["complete_safe"] for q in qs)
            elif label=="安全局部段":n=sum(q["all_returned"] and q["all_safe"] and not q["complete_safe"] for q in qs)
            elif label=="出现不安全返回":n=sum(q["unsafe_return"] for q in qs)
            else:n=sum(not q["unsafe_return"] and not(q["all_returned"] and q["all_safe"]) for q in qs)
            values.append(n)
        axes[0].bar(range(10),values,bottom=bottom,color=color,label=label);bottom+=values
    axes[0].set_xticks(range(10),QUERY_NAMES.values(),rotation=55);axes[0].set_ylabel("查询数（每类100，不是300次计时）");axes[0].legend(frameon=False,fontsize=8)
    kinds=list(QUERY_NAMES)[:7]
    box(axes[1],[[q["path_length_m"] for q in queries if q["scenario"]==f and q["complete_safe"] and q["path_length_m"] is not None] for f in kinds],
        [QUERY_NAMES[f] for f in kinds],"完整安全路径长度（米）")
    axes[1].tick_params(axis="x",rotation=45);save(fig,OUT,"Q_outputs")
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout="constrained")
    by_id=defaultdict(list)
    for row in record["runs"]:by_id[row["id"]].append(row)
    selected=[q for q in queries if q["complete_safe"]]
    values=[[q["mean_abs_curvature_per_m"] for q in selected if q["scenario"]==f and q["mean_abs_curvature_per_m"] is not None] for f in QUERY_NAMES]
    box(axes[0],values,QUERY_NAMES.values(),"离散路径绝对曲率均值（1/米）")
    switches=[[float(np.median([r["gear_switches"] for r in by_id[q["id"]]])) for q in selected if q["scenario"]==f] for f in QUERY_NAMES]
    box(axes[1],switches,QUERY_NAMES.values(),"前进/倒车方向切换数（次）")
    for ax in axes:ax.tick_params(axis="x",rotation=55)
    save(fig,OUT,"Q_geometry")
    fig,axes=plt.subplots(2,3,figsize=(10,7),layout="constrained")
    from matplotlib.patches import Rectangle
    for ax,f in zip(axes.ravel(),("turn","reverse","detour","corridor","step","unknown")):
        case=examples[f]["case"];row=examples[f]["run"]
        for b in case["obstacles"]:ax.add_patch(Rectangle((b["x"]-b["hx"],b["y"]-b["hy"]),2*b["hx"],2*b["hy"],color="#78868c"))
        if case["expected_stop"]:ax.axvspan(3.5,9,color="#c06a5e",alpha=.2)
        path=np.asarray(row["path"])
        if len(path):ax.plot(path[:,0],path[:,1],color=COLORS[0],lw=1.7)
        ax.scatter(*case["start"][:2],color=COLORS[1],s=25,label="起点");ax.scatter(*case["goal"],color="#b5534b",marker="*",s=45,label="目标")
        ax.set(xlabel="X（米）",ylabel="Y（米）",title=QUERY_NAMES[f]+"：首个预定ID");ax.axis("equal");ax.grid(alpha=.2)
    axes[0,0].legend(frameon=False,fontsize=9);save(fig,OUT,"Q_paths")
    return stats,queries


def navigation_figures(plan,record):
    rows=record["runs"];stats={};case_map={c["id"]:c for c in plan["cases"]}
    from planning_query_campaign import layout_id
    for f in SCENARIOS:
        ss=[r for r in rows if r["scenario"]==f];cases=[c for c in plan["cases"] if c["scenario"]==f]
        safe=lambda r:not(r.get("collision",True) or r.get("unsafe_terrain",True) or r.get("unknown_policy_violation",True))
        stats[f]=dict(tasks=len(ss),passed=sum(r["status"]=="passed" for r in ss),safe_arrival=sum(safe(r) and r.get("reached",False) for r in ss),
            safe_rejection=sum(safe(r) and r["status"]=="passed" and r.get("expected_stop",False) for r in ss),
            collisions=sum(r.get("collision",False) for r in ss),terrain_violations=sum(r.get("unsafe_terrain",False) for r in ss),
            unknown_violations=sum(r.get("unknown_policy_violation",False) for r in ss),errors=sum(r["status"]=="error" for r in ss),
            unique_layouts=len({layout_id(c) for c in cases}))
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout="constrained")
    x=np.arange(14)
    axes[0].bar(x,[stats[f]["safe_arrival"] for f in SCENARIOS],color=COLORS[0],label="安全到达")
    axes[0].bar(x,[stats[f]["safe_rejection"] for f in SCENARIOS],bottom=[stats[f]["safe_arrival"] for f in SCENARIOS],color=COLORS[1],label="正确安全拒绝")
    completed=np.array([stats[f]["safe_arrival"]+stats[f]["safe_rejection"] for f in SCENARIOS])
    axes[0].bar(x,[stats[f]["tasks"]-n for f,n in zip(SCENARIOS,completed)],bottom=completed,color="#bcbfc3",label="未完成或不安全/异常")
    axes[0].set_xticks(x,SCENARIOS.values(),rotation=65,ha="right");axes[0].set_ylabel("任务数 / 每类30次");axes[0].legend(frameon=False,fontsize=8)
    for k,label,color in (("collisions","碰撞",COLORS[2]),("terrain_violations","姿态/地形超限",COLORS[3]),("unknown_violations","未知越界",COLORS[4])):
        axes[1].plot(x,[stats[f][k] for f in SCENARIOS],"o-",label=label,color=color,ms=4)
    axes[1].set_xticks(x,SCENARIOS.values(),rotation=65,ha="right");axes[1].set_ylabel("失败事件数");axes[1].legend(frameon=False,fontsize=8)
    save(fig,OUT,"C_outcomes")
    fig,axes=plt.subplots(1,3,figsize=(11,4.2),layout="constrained")
    families=list(SCENARIOS)[:8]+["emerge","crossing"]
    for ax,key,label in zip(axes,("tracking_p95_m","planner_p95_ms","control_p95_ms"),
        ("当时局部路径最近距离P95（米）","单任务规划调用P95（毫秒）","单任务控制回调P95（毫秒）")):
        values=[[r[key] for r in rows if r["scenario"]==f and r.get(key) is not None and r["status"]!="error"] for f in families]
        box(ax,values,[SCENARIOS[f] for f in families],label);ax.tick_params(axis="x",rotation=65, labelsize=8)
    save(fig,OUT,"C_tracking")
    fig,axes=plt.subplots(2,3,figsize=(10,7),layout="constrained")
    from matplotlib.patches import Rectangle
    for ax,f in zip(axes.ravel(),("reverse","detour","ramp","emerge","crossing","unknown")):
        ss=sorted([r for r in rows if r["scenario"]==f],key=lambda r:r["id"])
        for row in ss:
            traj=np.asarray(row.get("trajectory",[]))
            if len(traj):ax.plot(traj[:,1],traj[:,2],color=COLORS[0],alpha=.18,lw=.8)
        first=ss[0];case=case_map[first["id"]];traj=np.asarray(first.get("trajectory",[]))
        if len(traj):ax.plot(traj[:,1],traj[:,2],color=COLORS[2],lw=1.6,label="预定首个任务")
        for b in case["obstacles"]:ax.add_patch(Rectangle((b["x"]-b["hx"],b["y"]-b["hy"]),2*b["hx"],2*b["hy"],color="#71818b",alpha=.5))
        ax.scatter(*case["goal"],marker="*",s=40,color="#b55249");ax.set(xlabel="X（米）",ylabel="Y（米）",title=SCENARIOS[f]);ax.axis("equal");ax.grid(alpha=.2)
    axes[0,0].legend(frameon=False,fontsize=9);save(fig,OUT,"C_trajectories")
    fig,axes=plt.subplots(2,2,figsize=(10,6.5),layout="constrained")
    for ax,f in zip(axes.ravel(),("reverse","detour","emerge","crossing")):
        row=sorted([r for r in rows if r["scenario"]==f],key=lambda r:r["id"])[0];traj=np.asarray(row.get("trajectory",[]))
        if len(traj):
            ax.plot(traj[:,0],traj[:,4],color=COLORS[0],label="模型实际速度");ax.plot(traj[:,0],traj[:,6],"--",color=COLORS[2],label="控制速度指令")
            twin=ax.twinx();twin.plot(traj[:,0],np.rad2deg(traj[:,5]),color=COLORS[1],alpha=.65,label="模型转角");twin.set_ylabel("转角（度）",color=COLORS[1])
        ax.set(xlabel="模型时间（秒）",ylabel="纵向速度（米/秒）",title=SCENARIOS[f]+"：预定首个任务");ax.grid(alpha=.2)
    axes[0,0].legend(frameon=False,fontsize=9);save(fig,OUT,"C_commands")
    fig,axes=plt.subplots(1,2,figsize=(9,4),layout="constrained")
    names=("emerge","crossing","stale_map")
    response=[[r["event_stop_response_s"] for r in rows if r["scenario"]==f and r.get("event_stop_response_s",-1)>=0] for f in names]
    box(axes[0],response,[SCENARIOS[f]+f"\n(n={len(v)})" for f,v in zip(names,response)],"事件后首次零速度指令时间（秒）")
    axes[0].set_title("不是传感器到执行器的因果反应时延",fontsize=10)
    for key,label,color in (("peak_pitch_rad","俯仰峰值",COLORS[0]),("peak_roll_rad","横滚峰值",COLORS[2])):
        vals=[np.rad2deg(np.median([r.get(key,0) for r in rows if r["scenario"]==f])) for f in SCENARIOS]
        axes[1].plot(range(14),vals,"o-",color=color,label=label,ms=3)
    axes[1].set_xticks(range(14),SCENARIOS.values(),rotation=65,ha="right",fontsize=8);axes[1].set_ylabel("任务峰值的中位数（度）");axes[1].legend(frameon=False,fontsize=9)
    save(fig,OUT,"C_events")
    failed=sorted([r for r in rows if r["status"]!="passed" and r.get("trajectory")],key=lambda r:r["id"])
    diagnostic=[("C_failure_trace","control_failure_case.tex",failed[0])] if failed else []
    emerging=[r for r in failed if r["scenario"]=="emerge"]
    if emerging and emerging[0]["id"]!=failed[0]["id"]:diagnostic.append(("C_failure_emerge","control_failure_emerge.tex",emerging[0]))
    for name,filename,row in diagnostic:
        case=case_map[row["id"]];traj=np.asarray(row["trajectory"]);event=case.get("event_s")
        fig,axes=plt.subplots(2,2,figsize=(9.5,7),layout="constrained")
        ax=axes[0,0];ax.plot(traj[:,1],traj[:,2],color=COLORS[2],lw=1.6)
        for b in case["obstacles"]:
            y=b["y"]+b.get("vy",0)*(event or 0)
            ax.add_patch(Rectangle((b["x"]-b["hx"],y-b["hy"]),2*b["hx"],2*b["hy"],color="#71818b",alpha=.5))
        ax.scatter(*case["goal"],marker="*",s=50,color="#b55249");ax.axis("equal")
        ax.set(xlabel="X（米）",ylabel="Y（米）",title=SCENARIOS[row["scenario"]]+("：事件时刻障碍" if event is not None else "：初始静态障碍"))
        distance=np.hypot(traj[:,1]-case["goal"][0],traj[:,2]-case["goal"][1])
        axes[0,1].plot(traj[:,0],distance,color=COLORS[0]);axes[0,1].set(ylabel="剩余目标距离（米）",title="安全停止是否恢复了目标进展")
        axes[1,0].plot(traj[:,0],traj[:,4],color=COLORS[0],label="模型实际")
        axes[1,0].plot(traj[:,0],traj[:,6],"--",color=COLORS[2],label="控制指令")
        axes[1,0].set_ylabel("纵向速度（米/秒）");axes[1,0].legend(frameon=False,fontsize=9)
        for success,color,label in ((True,COLORS[0],"有返回"),(False,COLORS[2],"无返回")):
            plans=[p for p in row["planning"] if p["success"]==success]
            axes[1,1].scatter([p["time_s"] for p in plans],[p["elapsed_ms"] for p in plans],s=12,color=color,label=label)
        axes[1,1].set(ylabel="实际规划调用耗时（毫秒）",yscale="log");axes[1,1].legend(frameon=False,fontsize=9)
        for ax in (axes[0,1],axes[1,0],axes[1,1]):
            if event is not None:ax.axvline(event,ls="--",color="#777777",lw=1)
            ax.set_xlabel("模型时间（秒）")
        for ax in axes.ravel():ax.grid(alpha=.2)
        save(fig,OUT,name)
        (OUT/filename).write_text("对应诊断任务为\\path{"+row["id"]+"}，最终目标误差为"
            +f"{row['goal_error_m']:.2f}米。该样例用于诊断，不作为无偏代表样例。\n")
    return stats


def export_tables(public,noise,resolution,scaling,query_stats,nav_stats,nav_record,public_stats):
    rows=[];intervals=[];risk=[]
    for f,label in FAMILIES.items():
        for c in CN:
            ss=subset(public,f,c);s=public_stats[f][c]
            n=sum(r["height"]["artificial"]["n"] for r in ss);a=sum(r["height"]["artificial_accepted"]["n"] for r in ss)
            condition=CN[c].replace("%",r"\%")
            rows.append(f"{label}&{condition}&{ci_text(s['artificial']['rmse_m'],100)}&{ci_text(s['artificial_accepted']['rmse_m'],100)}&{a}/{n}\\\\")
            h=s["artificial_accepted"]
            intervals.append(f"{label}&{condition}&{ci_text(h['picp95'],100)}&{ci_text(h['width95_m'],100)}&{ci_text(h['crps_m'],100)}\\\\")
            tp=sum(r["risk"]["tp"] for r in ss);fn=sum(r["risk"]["fn"] for r in ss)
            fp=sum(r["risk"]["fp"] for r in ss);tn=sum(r["risk"]["tn"] for r in ss)
            risk.append(f"{label}&{condition}&{fn}/{tp+fn}&{fmt(fn/(tp+fn) if tp+fn else None,100)}&{fmt(fp/(fp+tn) if fp+tn else None,100)}&{ci_text(s['risk']['fnr'],100)}\\\\")
    table_rows(OUT/"height_rows.tex",rows);table_rows(OUT/"uncertainty_rows.tex",intervals);table_rows(OUT/"risk_rows.tex",risk)
    rows=[];noise_stats={}
    for f in P["mechanism"]["families"]:
        noise_stats[f]={}
        for n in P["mechanism"]["noise_levels_m"]:
            ss=[r for r in noise if r["family"]==f and r["noise_std_m"]==n]
            h=summarize(ss,lambda r:r["height"]["artificial"]["rmse_m"])
            fr=summarize(ss,lambda r:r["risk"]["fnr"]);fp=summarize(ss,lambda r:r["risk"]["fpr"])
            noise_stats[f][str(n)]=dict(height=h,fnr=fr,fpr=fp)
            rows.append(f"{MECHANISMS[f]}&{n*100:.0f}&{h['groups']}&{ci_text(h,100)}&{ci_text(fr,100)}&{ci_text(fp,100)}\\\\")
    table_rows(OUT/"noise_rows.tex",rows)
    rows=[]
    for f in P["discretization"]["families"]:
        for res in P["discretization"]["resolution_m"]:
            for yaw in P["discretization"]["yaw_bins"]:
                ss=[r for r in resolution if r["family"]==f and r["resolution_m"]==res and r["yaw_bins"]==yaw]
                rows.append(f"{MECHANISMS[f]}&{res}&{yaw}&{ci_text(summarize(ss,lambda r:r['risk']['fnr']),100)}&{ci_text(summarize(ss,lambda r:r['risk']['fpr']),100)}\\\\")
    table_rows(OUT/"resolution_rows.tex",rows)
    rows=[]
    for size in P["scaling"]["sizes"]:
        for yaw in P["scaling"]["yaw_bins"]:
            for backend in ("cpu","cupy"):
                a=[r["elapsed_ms"] for r in scaling["runs"] if r["size"]==size and r["yaw_bins"]==yaw and r["backend"]==backend]
                rows.append(f"{size}&{yaw}&{backend.upper()}&{len(a)}&"+"&".join(fmt(float(np.quantile(a,q))) for q in (.5,.95,.99,1))+"\\\\")
    table_rows(OUT/"scaling_rows.tex",rows)
    rows=[]
    for f,label in QUERY_NAMES.items():
        s=query_stats[f]
        rows.append(f"{label}&{s['queries']}&{s['unique_layouts']}&{s['complete_safe']}&{s['unsafe_queries']}&{fmt(s['wall_p50_ms'])}&{fmt(s['wall_p95_ms'])}\\\\")
    table_rows(OUT/"query_rows.tex",rows)
    rows=[]
    for f,label in SCENARIOS.items():
        s=nav_stats[f];ss=[r for r in nav_record["runs"] if r["scenario"]==f and r["status"]!="error"]
        med=lambda k:float(np.median([r[k] for r in ss])) if ss else None
        rows.append(f"{label}&{s['tasks']}&{s['unique_layouts']}&{s['safe_arrival']}&{s['safe_rejection']}&{s['collisions']}&{s['terrain_violations']}&{fmt(med('tracking_p95_m'),100)}&{fmt(med('control_p95_ms'))}\\\\")
    table_rows(OUT/"control_rows.tex",rows)
    return noise_stats


def export_csv(name,rows):
    if not rows:return
    fields=sorted({k for r in rows for k in r})
    with (OUT/name).open("w") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def integration_figures():
    from joint_integration_study import PROTOCOL as JOINT_PROTOCOL
    record=read(BASE/"integration/results.json");settings=record["settings"];plan=read(JOINT_PROTOCOL)
    if settings["protocol_sha256"]!=digest(JOINT_PROTOCOL) or settings["source_results_sha256"]!=digest(BASE/"public/results.json"):
        raise ValueError("joint source/protocol changed")
    for name,sha in settings["source_sha256"].items():
        if digest(ROOT/name)!=sha:raise ValueError("joint method changed")
    expected={(i,c) for i in settings["selected_ids"] for c in plan["conditions"]}
    check_factorial(record["runs"],expected,("dataset_id","condition"))
    for row in record["runs"]:
        if row["replay_mean_max_delta_m"]>1e-4 or row["replay_risk_max_delta"]>1e-6:
            raise ValueError("joint baseline was not reproduced")
        target=BASE/"integration"/row["archive"]
        if digest(target)!=row["archive_sha256"]:raise ValueError("joint array changed")
        with np.load(target) as a:
            for i,mode in enumerate(plan["modes"]):
                score=risk_metrics(a["risk"][i],a["reference_risk"],a["unsafe"],a["supported"],.58)
                if not equal(score,row["risk"][mode]):raise ValueError("joint score changed")
                new=((a["risk"][i]<.58)&a["unsafe"]&a["supported"]&~((a["risk"][0]<.58)&a["unsafe"]&a["supported"])).sum()
                if int(new)!=row["newly_missed"][mode]:raise ValueError("new misses changed")
            filled=a["filled"]
            if (filled&a["observed"]).any() or (filled&a["occupied"]).any():raise ValueError("completion touched protected geometry")
            gates=(a["local"].reshape(filled.shape)&a["accepted"].reshape(filled.shape)&
                (a["variance"].reshape(filled.shape)<=.04**2)&(a["nearest"].reshape(filled.shape)<=.35))
            if (filled&~gates).any():raise ValueError("completion bypassed gate")
    rows=record["runs"];fig,axes=plt.subplots(1,2,figsize=(9.5,4),layout="constrained")
    names={"measured_only":"仅实测","refine_observed":"修正实测","bounded_completion":"有界补全"}
    for ax,key,label in zip(axes,("fnr","fpr"),("危险漏判FNR（%）","安全误拒FPR（%）")):
        for f,color in zip(FAMILIES,COLORS):
            values=[summarize(subset(rows,f),lambda r,mode=mode:r["risk"][mode][key])["mean"] for mode in plan["modes"]]
            ax.plot(range(3),np.asarray(values,float)*100,"o-",color=color,label=FAMILIES[f],ms=4)
        ax.set_xticks(range(3),names.values());ax.set_ylabel(label);ax.grid(alpha=.2)
    axes[0].legend(frameon=False,fontsize=9);save(fig,OUT,"J_modes")
    fig,axes=plt.subplots(1,2,figsize=(9.5,4),layout="constrained")
    selected=[[r for r in rows if r["family"]==f] for f in FAMILIES]
    box(axes[0],[[100*r["filled_height"]["n"]/r["hidden_reference_cells"] for r in ss if r["hidden_reference_cells"]] for ss in selected],
        FAMILIES.values(),"有参考隐藏格中的实际补全比例（%）")
    box(axes[1],[[r["filled_height"]["rmse_m"]*100 for r in ss if r["filled_height"]["n"]] for ss in selected],
        FAMILIES.values(),"实际补全位置高程RMSE（厘米）")
    for ax in axes:ax.tick_params(axis="x",rotation=25)
    save(fig,OUT,"J_completion")
    table=[];summary={}
    for f,label in FAMILIES.items():
        ss=subset(rows,f);summary[f]={}
        for mode in plan["modes"]:
            tp=sum(r["risk"][mode]["tp"] for r in ss);fn=sum(r["risk"][mode]["fn"] for r in ss)
            fp=sum(r["risk"][mode]["fp"] for r in ss);tn=sum(r["risk"][mode]["tn"] for r in ss)
            new=sum(r["newly_missed"][mode] for r in ss)
            summary[f][mode]=dict(tp=tp,fn=fn,fp=fp,tn=tn,new_missed=new)
            table.append(f"{label}&{names[mode]}&{fn}/{tp+fn}&{fmt(fn/(tp+fn) if tp+fn else None,100)}&{fmt(fp/(fp+tn) if fp+tn else None,100)}&{new}\\\\")
    table_rows(OUT/"joint_rows.tex",table)
    added=sum(r["newly_missed"]["bounded_completion"] for r in rows)
    refined=sum(r["newly_missed"]["refine_observed"] for r in rows)
    filled=sum(r["filled_cells"] for r in rows)
    (OUT/"joint_conclusion.tex").write_text(f"300次共享GP拟合支持900次风险图计算；有界补全共填入{filled}格。"
        f"实测修正、有界补全分别逐状态累计新增危险漏判{refined}、{added}次；不同缺测和相关航向下的计数不代表独立事故。"
        +("存在新增漏判，不能仅因误拒下降就开启部署补全。" if added else "本次未观测到新增漏判不构成零风险证明，仍需独立地形及闭环复核。")+"\n")
    return dict(fits=len(rows),risk_calls=len(rows)*3,arrays=len(rows),new_missed_completion=added,new_missed_refinement=refined,filled_cells=filled,by_family=summary)


def conclusions(public,noise,resolution,query_stats,nav_stats,scaling):
    lines=[]
    for label,rows in (("多来源缺测",public),("噪声配对专项",noise)):
        tp=sum(r["risk"]["tp"] for r in rows);fn=sum(r["risk"]["fn"] for r in rows)
        lines.append(f"{label}的状态池化危险漏判为{fn}/{tp+fn}（{fn/(tp+fn)*100:.2f}\\%）。"
            "分母是相关的位置--航向状态，不是独立试验次数；组均值与置信区间另列于表。")
    failed=[]
    for f in P["discretization"]["families"]:
        ss=[r for r in resolution if r["family"]==f and r["resolution_m"]==.2 and r["yaw_bins"]==36]
        tp=sum(r["risk"]["tp"] for r in ss);fn=sum(r["risk"]["fn"] for r in ss)
        if tp+fn and fn/(tp+fn)>.01:failed.append(MECHANISMS[f]+f"（{fn/(tp+fn)*100:.2f}\\%）")
    lines.append("在零噪声、完整观测的0.2米/36航向离散专项中，仍未达到1\\%筛查目标的类别："+("、".join(failed) if failed else "无有危险分母的超目标类别")+"。这项结果比仅在缺测条件下看漏判更能暴露离散/足迹几何问题，但仍不是实车接触真值。")
    complete=sum(query_stats[f]["complete_safe"] for f in list(QUERY_NAMES)[:7]);unsafe=sum(s["unsafe_queries"] for s in query_stats.values())
    lines.append(f"700个到达类静态查询中，{complete}个三次都完整安全连接；1000查询中{unsafe}个出现独立检查不安全返回。安全局部段不当作最终到达，拒绝类另列。")
    arrived=sum(s["safe_arrival"] for s in nav_stats.values());reject=sum(s["safe_rejection"] for s in nav_stats.values());collision=sum(s["collisions"] for s in nav_stats.values())
    lines.append(f"420个模型闭环任务中，安全到达{arrived}个，正确安全拒绝{reject}个，碰撞任务{collision}个。它们来自生产回调及一阶自行车模型，不是新增Isaac物理仿真成绩。")
    paired_disagreements=sum(a["threshold_disagreements"] for a in scaling["agreement"])
    max_delta=max(a["max_abs_delta"] for a in scaling["agreement"])
    lines.append(f"九个CPU/GPU配对工作负载的最大风险差为{max_delta:.6g}，阈值判定分歧累计{paired_disagreements}个状态；该数只对应所测负载和最后一遍输出，不能代替所有输入上的数值等价证明。")
    lines.append("实验数量增加没有改变未知补全默认关闭的事实，也没有把未实现的时序后验融合或动态残影清除标成已验证。下一步应优先用本轮失败区域做受控方法修正，而非按结果选择有利样本继续累加。")
    (OUT/"conclusions.tex").write_text("\n\n".join(lines)+"\n")


def section_findings(public,stats,resolution,qstats,nstats,scaling):
    height=[];uncertainty=[];risk=[]
    for f,label in FAMILIES.items():
        cell=stats[f]["hole16"];h=cell["artificial_accepted"]
        height.append(f"{label}在1.6米孔洞条件下，接受域RMSE组均值为{ci_text(h['rmse_m'],100)}厘米，"
            f"人工隐藏格接受比例组均值为{fmt(cell['acceptance']['mean'],100)}\\%。")
        uncertainty.append(f"{label}对应接受域实际PICP95为{fmt(h['picp95']['mean'],100)}\\%，"
            f"区间总宽为{fmt(h['width95_m']['mean'],100)}厘米，CRPS为{fmt(h['crps_m']['mean'],100)}厘米。")
        rows=subset(public,f)
        fn=sum(r['risk']['fn'] for r in rows);tp=sum(r['risk']['tp'] for r in rows)
        fp=sum(r['risk']['fp'] for r in rows);tn=sum(r['risk']['tn'] for r in rows)
        risk.append(f"{label}全部成功缺测条件的池化FN/危险为{fn}/{tp+fn}，FNR为{fmt(fn/(tp+fn) if tp+fn else None,100)}\\%，"
            f"安全误拒FPR为{fmt(fp/(fp+tn) if fp+tn else None,100)}\\%。")
    (OUT/"height_findings.tex").write_text("\n\n".join(height)+"\n\n以上区间是来源组均值CI；接受范围和误差必须一起引用。\n")
    (OUT/"uncertainty_findings.tex").write_text("\n\n".join(uncertainty)+"\n\n这些是各来源组指标的均值，不是所有格合并后的覆盖频率。\n")
    (OUT/"risk_findings.tex").write_text("\n\n".join(risk)+"\n\n状态池化强调逐状态错误负担；它不提供数百万独立样本的统计保证，也不能省略资格拒绝和来源组结果。\n")
    lines=[]
    for f,label in QUERY_NAMES.items():
        s=qstats[f]
        lines.append(f"{label}100查询中，三次均完整安全连接{s['complete_safe']}个，出现不安全返回{s['unsafe_queries']}个；"
            f"实际规划调用耗时P95为{fmt(s['wall_p95_ms'])}毫秒。")
    (OUT/"query_findings.tex").write_text("\n\n".join(lines)+"\n")
    incomplete=[]
    for f,label in SCENARIOS.items():
        s=nstats[f];n=s['tasks']-s['safe_arrival']-s['safe_rejection']
        if n:incomplete.append(f"{label}{n}个")
    (OUT/"control_findings.tex").write_text("按任务逐项核验后，未完成或不安全/异常任务为："
        +("、".join(incomplete) if incomplete else "本轮未观测到")+"。无碰撞但超时到不了目标仍不合格；这些任务的有效轨迹和时延未从统计中删除。\n")
    size=104;yaw=36
    values={b:np.quantile([r['elapsed_ms'] for r in scaling['runs'] if r['size']==size and r['yaw_bins']==yaw and r['backend']==b],.95) for b in ('cpu','cupy')}
    (OUT/"timing_findings.tex").write_text(f"在104$\\times$104格、36航向的工作负载下，CPU P95为{values['cpu']:.2f}毫秒，"
        f"GPU P95为{values['cupy']:.2f}毫秒，比值为{values['cpu']/values['cupy']:.2f}倍。其他负载另列，不能只用这一点推断整个框架实时。\n")


def environment_table(environment):
    escape=lambda text:str(text).translate(str.maketrans({"_":r"\_","%":r"\%","&":r"\&","#":r"\#","$":r"\$"}))
    cpu=next((line.split(":",1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")),"unavailable")
    gpu=next(csv.reader(environment["gpu"].splitlines()))
    values=[("复核主机CPU",cpu),("实验GPU",gpu[0]),("驱动 / 显存",gpu[1]+" / "+gpu[2]),
        ("系统",environment["platform"]),("NumPy / PyTorch / CuPy",environment["numpy"]+" / "+environment["torch"]+" / "+environment["cupy"]),
        ("线程与共享边界","OpenBLAS=1, OMP=1, PyTorch=1; shared desktop GPU")]
    table_rows(OUT/"environment_rows.tex",[escape(k)+" & "+escape(v)+r" \\" for k,v in values])


def main():
    global P
    P=read(PROTOCOL);OUT.mkdir(parents=True,exist_ok=True);configure_chinese()
    # Use TrueType CJK outlines: CFF collections cannot be embedded as FontFile2.
    font_path=Path(findfont(FontProperties(family="Droid Sans Fallback"),fallback_to_default=False))
    rendering=dict(chinese_font="Droid Sans Fallback",latin_font="DejaVu Sans",chinese_font_sha256=digest(font_path),pdf_fonttype=42)
    plt.rcParams.update({"font.family":["DejaVu Sans","Droid Sans Fallback"],"pdf.fonttype":42,
        "font.size":11,"axes.titlesize":11,"axes.labelsize":11,"xtick.labelsize":10,"ytick.labelsize":10})
    catalog=read(BASE/"catalog.json")
    if catalog["protocol_sha256"]!=digest(PROTOCOL):raise ValueError("catalog protocol changed")
    for name,sha in fingerprints().items():
        if digest(BASE/"frozen_sources"/name)!=sha:raise ValueError("frozen source snapshot differs")
    public_record,paudit=audit_gp("public");noise_record,naudit=audit_gp("mechanism")
    resolution_record=audit_resolution();scaling=read(BASE/"scaling/results.json")
    qplan,qrecord=audit_native("planning");nplan,nrecord=audit_native("navigation")
    public=[r for r in public_record["runs"] if r["status"]=="ok"]
    noise=[r for r in noise_record["runs"] if r["status"]=="ok"];resolution=resolution_record["runs"]
    refusal_figure((public_record,noise_record))
    sources=source_quality(catalog);stats=stats_public(public)
    public_figures(public,stats);uncertainty_figures(public);risk_figures(public);precision_figure(public)
    noise_deltas=noise_figures(noise);failure_examples=resolution_figures(resolution)
    joint=integration_figures();examples,spatial_audit=spatial_figures(public)
    runtime=scaling_figures(scaling);gp_runtime=gp_latency_figures(public)
    qstats,queries=planning_figures(qplan,qrecord);nstats=navigation_figures(nplan,nrecord)
    nsummary=export_tables(public,noise,resolution,scaling,qstats,nstats,nrecord,stats)
    conclusions(public,noise,resolution,qstats,nstats,scaling)
    section_findings(public,stats,resolution,qstats,nstats,scaling)
    environment_table(public_record["environment_start"])
    regression_path=BASE/"regression_tests.xml"
    regression_cases=ET.parse(regression_path).findall(".//testcase")
    if not regression_cases or any(c.find(tag) is not None for c in regression_cases for tag in ("failure","error","skipped")):
        raise ValueError("regression suite incomplete or unsuccessful")
    regression=dict(tests=len(regression_cases),failures=0,errors=0,skipped=0,sha256=digest(regression_path))
    (OUT/"regression_summary.tex").write_text(f"本轮回归测试{len(regression_cases)}项通过，无失败、错误或跳过；"
        "其中包含CPU/CUDA支撑边界、风险几何、接入保护、来源分组及实验配对契约。单元测试通过不代替场景安全验收。\n")
    scalar=[]
    for phase,rows in (("public",public),("mechanism",noise)):
        for row in rows:
            flat={k:row[k] for k in ("id","dataset_id","family","cluster_id","condition","noise_std_m","gp_wall_ms","risk_wall_ms","archive_sha256")}
            flat["phase"]=phase
            for domain in ("artificial","artificial_accepted","natural"):
                flat.update({domain+"_"+k:row["height"][domain][k] for k in ("n","rmse_m","mae_m","p95_m","crps_m")})
            flat.update({"risk_"+k:v for k,v in row["risk"].items()});scalar.append(flat)
    export_csv("terrain_runs.csv",scalar);export_csv("noise_paired_deltas.csv",noise_deltas)
    export_csv("query_summary.csv",queries)
    export_csv("navigation_summary.csv",[{k:v for k,v in r.items() if k not in ("trajectory","planning","first_unsafe_pose")} for r in nrecord["runs"]])
    result=dict(study_id=P["study_id"],protocol_sha256=digest(PROTOCOL),source_sha256=fingerprints(),
        audit=dict(public=paudit,mechanism=naudit,resolution_arrays=len(resolution)),
        counts=dict(public_inputs=len(catalog["datasets"]),source_groups=catalog["source_groups"],public_attempts=len(public_record["runs"]),
            public_input_refusals=paudit["input_refusals"],public_computations=len(public),
            noise_computations=len(noise),paired_noise_triplets=naudit["paired_triplets"],discretization_computations=len(resolution),
            scaling_measured_calls=len(scaling["runs"]),planning_queries=len(queries),planning_calls=len(qrecord["runs"]),navigation_tasks=len(nrecord["runs"])),
        public_statistics=stats,noise_statistics=nsummary,runtime=runtime,gp_runtime=gp_runtime,query_statistics=qstats,navigation_statistics=nstats,
        examples=examples,spatial_audit=spatial_audit,failure_examples=failure_examples,cpu_gpu_agreement=scaling["agreement"],joint_integration=joint,
        environment=public_record["environment_start"],regression=regression,rendering=rendering)
    atomic_json(OUT/"results.json",result)
    (OUT/"counts.tex").write_text(
        f"\\newcommand{{\\PublicFits}}{{{len(public)}}}\n"
        f"\\newcommand{{\\PublicRefusals}}{{{paudit['input_refusals']}}}\n"
        f"\\newcommand{{\\PlotCount}}{{{len(list(OUT.glob('*.png')))}}}\n")
    atomic_json(OUT/"manifest.json",dict(protocol_sha256=digest(PROTOCOL),reporter_sha256=digest(Path(__file__)),
        report_tex_sha256=digest(DOC/"main.tex"),joint_protocol_sha256=digest(ROOT/"scripts/sim/experiments/protocols/joint_integration_v1.json"),
        inputs={str(p.relative_to(BASE)):digest(p) for p in [BASE/"catalog.json",*[BASE/phase/"results.json" for phase in ("public","mechanism","discretization","scaling")],
            BASE/"planning/cases.json",BASE/"planning/runs/results.json",BASE/"navigation/cases.json",BASE/"navigation/runs/results.json",BASE/"integration/results.json",regression_path]},
        assets={p.name:digest(p) for p in sorted(OUT.iterdir()) if p.suffix in (".csv",".pdf",".png",".tex")}))
    print("[v4 report audited]",result["counts"],flush=True)


if __name__=="__main__":main()
