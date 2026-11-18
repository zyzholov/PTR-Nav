#!/usr/bin/env python3
"""Describe directed capability checks; task/job counts are not IID repetitions."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Rectangle,Patch
import numpy as np
from plot_style import configure_chinese

ROOT=Path(__file__).resolve().parents[3]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();data=json.loads(a.manifest.read_text());a.output.mkdir(parents=True,exist_ok=False)
    rows=[];jobs=[];trials={}
    for record in data['runs']:
        run=Path(record['run_directory']);s=record.get('navigation_summary',{})
        trace=json.loads((run/'navigation_data.json').read_text()) if (run/'navigation_data.json').exists() else {}
        trials[record['case']]=trace
        durations=[float(v) for v in re.findall(r'terrain_nav planner: result=\w+ reason=.*? elapsed=([\d.]+)ms',
                                               (run/'planning.log').read_text())] if (run/'planning.log').exists() else []
        jobs.append(durations)
        rows.append(dict(case=record['case'],description=record['description'],expectation=record['expectation'],
            qualified=record['exit_code']==0,source_unchanged=record['source_unchanged'],reached=s.get('reached'),
            duration_s=s.get('duration_s'),final_error_m=s.get('truth_error_m'),
            risk_age_p95_s=s.get('risk_map_age_p95_s'),plan_jobs=len(durations),
            plan_job_p95_ms=float(np.quantile(durations,.95)) if durations else None,
            plan_job_max_ms=max(durations,default=None),run_directory=str(run),
            failure_reason=record.get('result',{}).get('failure_reason')))
    configure_chinese()
    labels=[r['description'] for r in rows];ys=np.arange(len(rows))
    fig,ax=plt.subplots(figsize=(11,max(5,len(rows)*.42)),layout='constrained')
    boxes=ax.boxplot([d or [np.nan] for d in jobs],vert=False,positions=ys,patch_artist=True,
        widths=.55,showfliers=True,flierprops=dict(marker='.',markersize=2,alpha=.35),
        medianprops=dict(color='#17354b',linewidth=1.3))
    for box,row in zip(boxes['boxes'],rows):box.set_facecolor('#68b0a3' if row['qualified'] else '#d6a46a')
    ax.set_yticks(ys,[f"{name}（n={row['plan_jobs']}）" for name,row in zip(labels,rows)])
    ax.invert_yaxis();budget_line=ax.axvline(90,color='#b14e58',ls='--',lw=1,label='规划协作式预算 90 ms')
    ax.set(xlabel='一次规划任务的实际墙钟耗时（毫秒）',title='各场景规划耗时分布；每箱包含多次规划任务，并非多次独立导航实验')
    ax.set_xlim(left=0);ax.grid(axis='x',alpha=.15)
    ax.legend(handles=[Patch(facecolor='#68b0a3',label='完整验收通过'),Patch(facecolor='#d6a46a',label='未通过'),budget_line],loc='lower right',fontsize=8)
    for ext in ('png','pdf'):fig.savefig(a.output/f'planning_latency.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,max(5,len(rows)*.42)),layout='constrained')
    for ax,key,xlabel in zip(axes,('final_error_m','risk_age_p95_s'),('终点位置误差（米）；拒绝通行场景另行解释','风险图源时间年龄 P95（秒；仿真时间）')):
        for i,row in enumerate(rows):
            value=row[key]
            if value is not None:ax.scatter(value,i,color='#27877e' if row['qualified'] else '#bb694a',s=35)
        ax.set_yticks(ys,labels if ax is axes[0] else []);ax.invert_yaxis();ax.set_xlabel(xlabel);ax.grid(axis='x',alpha=.15)
    axes[1].axvline(.5,color='#b14e58',ls='--',lw=1,label='当前资格上限 0.5 s');axes[1].legend(fontsize=8)
    fig.suptitle('同一源码版本的定向任务结果；绿色为完整验收通过，棕色为未通过')
    for ext in ('png','pdf'):fig.savefig(a.output/f'capability_outcomes.{ext}',dpi=180)
    plt.close(fig)
    selected=[r for r in rows if trials[r['case']].get('spec')][:6]
    if selected:
        spec=importlib.util.spec_from_file_location('physical_boundary_courses',ROOT/'src/isaac_sim_hunter_livo2/scripts/boundary_courses.py')
        course=importlib.util.module_from_spec(spec);spec.loader.exec_module(course)
        fig,axes=plt.subplots(2,3,figsize=(14,9),layout='constrained')
        for ax,row in zip(axes.flat,selected):
            t=trials[row['case']];truth=np.asarray(t['truth'])
            for obj in course.fixtures(t['spec']):
                color='#cfb288' if obj['terrain'] else '#a8b4be'
                if obj['shape']=='cylinder':patch=Circle((obj['x'],obj['y']),obj['radius'],facecolor=color,alpha=.65)
                else:patch=Rectangle((obj['x']-obj['length']/2,obj['y']-obj['width']/2),obj['length'],obj['width'],facecolor=color,alpha=.65)
                ax.add_patch(patch)
            ax.plot(truth[:,1],truth[:,2],color='#236b98',lw=1.4,label='独立物理真值轨迹')
            for i,(name,moving) in enumerate(t.get('moving_obstacles',{}).items()):
                values=np.asarray(moving)
                if values.size:ax.plot(values[:,1],values[:,2],ls='--',color='#bb694a',lw=1,label='移动障碍真值轨迹' if i==0 else None)
            ax.scatter(*truth[0,1:3],color='#27877e',s=20,label='起点')
            ax.scatter(*t['truth_goal'],marker='*',color='#d49a2f',s=80,label='唯一外部目标')
            ax.scatter(*truth[-1,1:3],facecolor='none',edgecolor='#bb694a',s=45,label='试验终点')
            ax.set(title=f"{row['case']}：{'完整通过' if row['qualified'] else '未通过'}",xlabel='世界 X（米）',ylabel='世界 Y（米）',aspect='equal')
            ax.relim();ax.autoscale_view();ax.grid(alpha=.12)
        for ax in axes.flat[len(selected):]:ax.set_visible(False)
        axes.flat[0].legend(fontsize=7,loc='best')
        fig.suptitle('真实物理轨迹及场景实体；图中未绘制风险预测或虚构参考路径')
        for ext in ('png','pdf'):fig.savefig(a.output/f'physical_trajectories.{ext}',dpi=180)
        plt.close(fig)
    with (a.output/'cases.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    result=dict(scope=__doc__,manifest=str(a.manifest.resolve()),manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
        cases=rows,interpretation='每场景一次；不能用任务种类数替代重复次数，不能据此声称总体高成功率。规划箱线图样本是同一任务内的相关规划调用。')
    (a.output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(cases=len(rows),qualified=sum(r['qualified'] for r in rows),output=str(a.output)),ensure_ascii=False))


if __name__=='__main__':main()
