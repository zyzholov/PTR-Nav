"""Audit and visualize correction evidence, keeping data changes separate from algorithm changes."""

from collections import defaultdict
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import ConvexHull

from pointcloud_batch import ROOT, atomic_json, digest
from paper_clouds import grid, height_at
from paper_scale_evaluate import height_metrics, risk_metrics, partial_points
from plot_paper_scale import plt, save, FAMILIES, clustered_summary
from systematic_study import style
from diagnose_terrain_prediction import errors

BASE=ROOT/"log/terrain_diagnosis_20261015"
OUT=ROOT/"docs/reports/systematic_study/assets/correction"
CONDITIONS=("point15","hole16","frontier50")
CN=("随机删点15%","内部1.6米孔洞","单侧隐藏50%")
BLUE,RED="#247b92","#b3664e"


def read(path):return json.loads(path.read_text())


def validate(record,n,variants):
    rows=record["runs"];ids={r["id"] for r in rows}
    expected={(i,c,v) for i in ids for c in CONDITIONS for v in variants}
    actual=[(r["id"],r["condition"],r["variant"]) for r in rows]
    if len(ids)!=n or len(actual)!=len(expected) or set(actual)!=expected:raise ValueError("incomplete or duplicate correction batch")
    if any(r["actual_device"]!="cuda" or r["actual_backend"]!="cupy" for r in rows):raise ValueError("GPU fallback")


def audit(record,folder):
    checked=0
    for row in record["runs"]:
        archive=folder/row["archive"]
        if digest(archive)!=row["archive_sha256"]:raise ValueError("saved correction prediction changed")
        with np.load(archive) as a:
            h=height_metrics(SimpleNamespace(mean=a["mean"],latent_var=a["variance"],accepted=a["accepted"]),a["reference"],a["observed"].ravel())
            r=risk_metrics(a["predicted_risk"],a["reference_risk"],a["unsafe"],a["risk_supported"])
            for k,v in {**h,**r}.items():
                if v is None:
                    if row[k] is not None:raise ValueError("zero-denominator metric")
                elif not np.isclose(v,row[k],atol=1e-7,rtol=1e-6):raise ValueError("recomputed metric differs: "+k)
            if not np.isfinite(a["mean"]).all() or not np.isfinite(a["variance"]).all() or (a["variance"]<0).any():raise ValueError("invalid posterior")
        checked+=1
    return checked


def method_pairs(record,catalog):
    pairs=defaultdict(dict)
    items={d["id"]:d for d in catalog["datasets"]}
    for row in record["runs"]:pairs[row["id"],row["condition"]][row["variant"]]=row
    for key,pair in pairs.items():
        a,b=pair["before"],pair["fixed"]
        if a["case_seed"]!=b["case_seed"] or a["source_sha256"]!=b["source_sha256"]:raise ValueError("unpaired method input")
        with np.load(BASE/"regression_v2"/a["archive"]) as left,np.load(BASE/"regression_v2"/b["archive"]) as right:
            for k in ("raw_xyz","reference","observed","xy","reference_risk","unsafe","risk_supported"):
                if not np.array_equal(left[k],right[k],equal_nan=True):raise ValueError("method reference or observation changed")
            item=items[key[0]];source=(BASE/"regression_v2_catalog.json").parent/item["path"]
            if digest(source)!=item["sha256"]:raise ValueError("source changed")
            with np.load(source) as original:
                keep=partial_points(original["xyz"],key[1],original["query_origin"],b["case_seed"])
                ground=right["raw_xyz"][original["is_ground"][keep]]
            center=ground[:,:2].mean(axis=0,dtype=np.float64)
            eq=ConvexHull(ground[:,:2].astype(np.float64)-center).equations
            outside=np.any((right["xy"]-center)@eq[:,:2].T+eq[:,2]>1e-6,axis=1)
            if np.any(outside&right["accepted"]) or np.any(outside&right["local"]):raise ValueError("exterior prediction still accepted")
    return pairs


def summary(rows):
    result={}
    for f in FAMILIES:
        result[f]={}
        for c in CONDITIONS:
            result[f][c]={}
            for v in sorted({r["variant"] for r in rows}):
                group=[r for r in rows if r["family"]==f and r["condition"]==c and r["variant"]==v]
                if not group:continue
                flat=[dict(r,status="ok",accepted_rmse_m=r["accepted_hidden"]["rmse_m"]) for r in group]
                result[f][c][v]={"cases":len(group),"groups":len({r["cluster_id"] for r in group}),
                    "raw_rmse":clustered_summary(flat,"hidden_rmse_m"),"accepted_rmse":clustered_summary(flat,"accepted_rmse_m"),
                    "hidden_cells":sum(r["hidden_cells"] for r in group),"accepted_cells":sum(r["accepted_hidden"]["n"] for r in group),
                    "outside_accepted":sum(r["outside_hull_accepted"] for r in group),
                    **{k:sum(r[k] for r in group) for k in ("tp","fn","fp","tn")}}
    return result


def source_figures(audit_rows,source_pair):
    catalog_path=BASE/"repaired_pair/dataset_manifest.json";catalog=read(catalog_path)
    corrected=[]
    for item in catalog["datasets"]:
        with np.load(catalog_path.parent/item["path"]) as a:
            xy=grid(tuple(a["shape"]),a["query_origin"],.2);z=a["truth_height"]
            ref=height_at(z,a["truth_origin"],.1,xy)
            interpolator=RegularGridInterpolator((np.arange(0,500,2),np.arange(0,500,2)),z[::2,::2])
            guess=interpolator(((xy-a["truth_origin"])/.1-.5)[:,::-1])
            corrected.append(dict(id=item["id"],shader_type=item["shader_parameters"]["type"],
                rmse_m=errors(guess,ref,np.isfinite(ref))["rmse_m"]))
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout="constrained")
    groups=[[r["complete_input_interpolation"]["rmse_m"]*100 for r in audit_rows if r["shader_type"]==i] for i in range(4)]
    groups.append([r["rmse_m"]*100 for r in corrected if r["shader_type"]==3])
    b=axes[0].boxplot(groups,tick_labels=["fBm\n旧75","脊状\n旧75","云状\n旧75","Swiss\n旧75","Swiss\n修正6"],patch_artist=True,
        flierprops={"marker":".","markersize":3})
    for p,color in zip(b["boxes"],[BLUE]*3+[RED,BLUE]):p.set_facecolor(color);p.set_alpha(.6)
    axes[0].set_yscale("log");axes[0].set_ylabel("完整0.2米输入插值RMSE（厘米）");axes[0].set_title("先检查输入能否表达0.1米参考")
    for v,label,color in (("legacy_data","旧生成配置",RED),("fixed_data","修正生成配置",BLUE)):
        data=[np.mean([r["hidden_rmse_m"] for r in source_pair["runs"] if r["variant"]==v and r["condition"]==c]) *100 for c in CONDITIONS]
        axes[1].plot(np.arange(3),data,"o-",label=label,color=color)
    axes[1].set_xticks(np.arange(3),CN,rotation=15);axes[1].set_ylabel("每输入隐藏RMSE均值（厘米）")
    axes[1].set_title("同一旧预测器：仅改变数据生成配置");axes[1].legend(frameon=False)
    for ax in axes:ax.grid(axis="y",alpha=.2)
    save(fig,OUT,"D1_source")
    oldcatalog_path=ROOT/"log/paper_aligned_20261015/catalog.json";old=read(oldcatalog_path)
    item=next(r for r in catalog["datasets"] if r["family"]=="hilly" and r["shader_parameters"]["type"]==3)
    earlier=next(r for r in old["datasets"] if r["id"]==item["id"])
    fig,axes=plt.subplots(1,2,figsize=(10,3.4),layout="constrained")
    for ax,d,path,label in zip(axes,(earlier,item),(oldcatalog_path,catalog_path),("旧Swiss：增益大于1","修正Swiss：衰减增益")):
        with np.load(path.parent/d["path"]) as a:
            z=a["truth_height"];row=250;x=-25+(np.arange(500)+.5)*.1
            ax.plot(x,z[row],color=BLUE,lw=.8,label="0.1米参考切片")
            ax.plot(x[::2],z[row,::2],".",ms=2,color=RED,label="0.2米输入位置")
        ax.set_title(label);ax.set_xlabel("X（米）");ax.set_ylabel("高程Z（米）");ax.grid(alpha=.2);ax.legend(frameon=False)
    save(fig,OUT,"D2_profiles")
    return corrected


def road_figures(road_audit,road_pair):
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout="constrained")
    for source,label,color in (("kitti_gt","旧GPS/IMU位姿",RED),("semantic_suma","官方SuMa位姿",BLUE)):
        vals=[r for r in road_audit["runs"] if r["source"]==source]
        labels=[f"{r['sequence']:02d}/{r['frame']}" for r in vals]
        axes[0].plot(np.arange(len(vals)),[r["median_abs_m"]*100 for r in vals],"o",color=color,label=label)
    for boundary in (2.5,5.5):axes[0].axvline(boundary,color="#bbbbbb",lw=.8)
    axes[0].set_xticks(np.arange(len(labels)),labels,rotation=50);axes[0].set_yscale("log")
    axes[0].set_xlabel("序列号/与首帧配对的帧号");axes[0].set_ylabel("近邻地面高差中位数（厘米）");axes[0].set_title("跨帧对齐质量；匹配数量另存JSON")
    for v,label,color in (("legacy_data","旧注册子图",RED),("fixed_data","SuMa注册子图",BLUE)):
        r=[r for r in road_pair["runs"] if r["variant"]==v and r["condition"]=="hole16"]
        vals=[np.mean([x["hidden_rmse_m"] for x in r if x["id"].startswith("kitti_"+seq)])*100 for seq in ("08","09","10")]
        axes[1].plot(np.arange(3),vals,"o-",color=color,label=label)
    axes[1].set_xticks(np.arange(3),["序列08","序列09","序列10"]);axes[1].set_yscale("log")
    axes[1].set_ylabel("孔洞隐藏RMSE均值（厘米）");axes[1].set_title("冻结旧预测器；只修正注册来源")
    for ax in axes:ax.grid(alpha=.2);ax.legend(frameon=False)
    save(fig,OUT,"D3_road")


def method_figures(record):
    fig,axes=plt.subplots(2,2,figsize=(10,6.5),layout="constrained");metrics={}
    for v,label,color,offset in (("before","原工程",RED,-.18),("fixed","支撑修正",BLUE,.18)):
        selected=[r for r in record["runs"] if r["variant"]==v];metrics[v]={}
        for c in CONDITIONS:
            r=[r for r in selected if r["condition"]==c]
            metrics[v][c]={"outside":sum(x["outside_hull_accepted"] for x in r),
                "accepted_percent":sum(x["accepted_hidden"]["n"] for x in r)/sum(x["hidden_cells"] for x in r)*100,
                "raw_rmse_cm":float(np.mean([x["hidden_rmse_m"] for x in r if x["hidden_rmse_m"] is not None]))*100,
                "fn":sum(x["fn"] for x in r)}
        for ax,key,title,ylabel in zip(axes.ravel(),("outside","accepted_percent","raw_rmse_cm","fn"),
                ("观测凸包外被接受的查询","有参考隐藏格的候选比例","所有隐藏格：原始预测仍保留","风险危险漏判出现次数"),
                ("查询数（相关，非事故）","接受比例（%）","每输入RMSE均值（厘米）","FN计数（相关位置/航向）")):
            values=[metrics[v][c][key] for c in CONDITIONS]
            ax.bar(np.arange(3)+offset,values,width=.34,color=color,label=label)
            ax.set_xticks(np.arange(3),CN,rotation=15);ax.set_title(title);ax.set_ylabel(ylabel);ax.grid(axis="y",alpha=.2)
            if key=="outside":
                for i,y in enumerate(values):ax.text(i+offset,y+max(values+[1])*.02,str(y),ha="center",fontsize=8)
    axes[0,0].legend(frameon=False);save(fig,OUT,"D4_method")
    return metrics


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    paths={"source_audit":BASE/"all_generated_source_audit.json","source_pair":BASE/"source_pair_final/results.json",
        "road_audit":BASE/"road_registration_audit.json","road_pair":BASE/"road_source_pair/results.json",
        "regression":BASE/"regression_v2/results.json","catalog":BASE/"regression_v2_catalog.json"}
    d={k:read(p) for k,p in paths.items()}
    if digest(ROOT/"src/fsgp_bgk_se2/fsgp_bgk_se2/feature_sgp.py")!=d["regression"]["settings"]["fixed_sha256"]:
        raise ValueError("fixed predictor changed after regression")
    if digest(BASE/"frozen_before/feature_sgp.py")!=d["regression"]["settings"]["baseline_sha256"]:
        raise ValueError("frozen baseline changed")
    validate(d["source_pair"],24,("legacy_data","fixed_data"));validate(d["road_pair"],9,("legacy_data","fixed_data"))
    validate(d["regression"],80,("before","fixed"))
    checked=sum(audit(d[key],paths[key].parent) for key in ("source_pair","road_pair","regression"))
    pairs=method_pairs(d["regression"],d["catalog"])
    style();corrected=source_figures(d["source_audit"]["runs"],d["source_pair"])
    road_figures(d["road_audit"],d["road_pair"]);metrics=method_figures(d["regression"])
    generated_source={str(t):{"n":sum(r["shader_type"]==t for r in d["source_audit"]["runs"]),
        "mean_interpolation_rmse_m":float(np.mean([r["complete_input_interpolation"]["rmse_m"] for r in d["source_audit"]["runs"] if r["shader_type"]==t]))}
        for t in range(4)}
    increased=[{"id":a["id"],"condition":a["condition"],"delta_fn":b["fn"]-a["fn"]}
        for a,b in ((p["before"],p["fixed"]) for p in pairs.values()) if b["fn"]>a["fn"]]
    result={"study_id":"PTR-TERRAIN-DIAG-02","verified_arrays":checked,"method_inputs":80,"method_pairs":len(pairs),
        "method_computations":len(d["regression"]["runs"]),"source_computations":len(d["source_pair"]["runs"]),
        "road_computations":len(d["road_pair"]["runs"]),"generated_source":generated_source,
        "corrected_representation":corrected,"terrain":summary(d["regression"]["runs"]),"method":metrics,
        "fn_increase_cases":increased,"outside_hull_accepted_fixed":sum(r["outside_hull_accepted"] for r in d["regression"]["runs"] if r["variant"]=="fixed"),
        "scope":"targeted correction regression only; source changes are not algorithm gains; no new physical navigation or external-method comparison"}
    atomic_json(OUT/"results.json",result)
    tables=[]
    for f in FAMILIES:
        for c,cn in zip(CONDITIONS,CN):
            s=result["terrain"][f][c]["fixed"]
            fmt=lambda x:"--" if x is None else f"{100*x:.2f}"
            tables.append(" & ".join([FAMILIES[f],cn.replace("%",r"\%"),str(s["cases"]),str(s["groups"]),fmt(s["raw_rmse"]["mean"]),
                fmt(s["accepted_rmse"]["mean"]),str(s["accepted_cells"])+"/"+str(s["hidden_cells"])])+r" \\")
    (OUT/"rows.tex").write_text("\n".join(tables)+"\n"+r"\bottomrule"+"\n")
    with (OUT/"method_runs.csv").open("w",newline="") as stream:
        scalar=[{k:v for k,v in r.items() if not isinstance(v,dict)} for r in d["regression"]["runs"]]
        writer=csv.DictWriter(stream,fieldnames=sorted(set().union(*(r.keys() for r in scalar))));writer.writeheader();writer.writerows(scalar)
    atomic_json(OUT/"manifest.json",{"inputs":{str(p.relative_to(ROOT)):digest(p) for p in paths.values()},
        "analyzer_sha256":digest(Path(__file__)),"outputs":{p.name:digest(p) for p in OUT.iterdir() if p.is_file() and p.name!="manifest.json"},
        "array_audit_count":checked})
    print("[correction audit]",checked,"arrays; exterior accepted",result["outside_hull_accepted_fixed"],"FN increase cases",len(increased),flush=True)


if __name__=="__main__":main()
