#!/usr/bin/env python3
"""Plot whole planner jobs from two retained corridor qualifications, including failures.

This is an engineering before/after check, not a paired statistical study.
Never filter unsuccessful jobs or infer a navigation success rate from job counts.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from plot_single_goal_qualification import configure_chinese


def load(run):
    data=json.loads((run/'navigation_data.json').read_text())
    result=json.loads((run/'result.json').read_text())
    metadata=json.loads((run/'run_metadata.json').read_text())
    jobs=[dict(index=i+1,result=m[0],elapsed_ms=float(m[1])) for i,m in enumerate(re.findall(
        r'terrain_nav planner: result=(success|failed).*?elapsed=([0-9.]+)ms',(run/'planning.log').read_text()))]
    if not jobs or data['summary'].get('kind')!='side_goal_corridor':
        raise ValueError('requires complete single-external-goal corridor qualifications')
    values=np.array([j['elapsed_ms'] for j in jobs])
    s=data['summary']
    return dict(run=str(run),passed=result['passed'],summary=s,jobs=jobs,
        planner_ms=dict(count=len(values),p50=float(np.percentile(values,50)),
                        p95=float(np.percentile(values,95)),maximum=float(values.max())),
        xy=np.asarray(data['truth'])[:,[1,2]],
        goal_requests=len(data['goal_requests']),goal_acknowledgements=len(data['goal_acknowledgements']),
        provenance=dict(source_snapshot_sha256=metadata['source_snapshot_sha256'],
            method_hashes=metadata['method_hashes'],arguments=metadata['arguments'],
            evidence_sha256={n:hashlib.sha256((run/n).read_bytes()).hexdigest() for n in
                ('navigation_data.json','planning.log','result.json','run_metadata.json')}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('before',type=Path);p.add_argument('after',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);configure_chinese()
    rows=[load(a.before.resolve()),load(a.after.resolve())]
    fig,axes=plt.subplots(2,2,figsize=(12,8.8));colors=('#527D9B','#C27C47');names=('修改前','共享时间预算')
    for row,color,name in zip(rows,colors,names):
        v=np.array([j['elapsed_ms'] for j in row['jobs']])
        axes[0,0].plot(np.arange(len(v))+1,v,label=name,color=color,lw=1.,alpha=.85)
        axes[0,1].step(np.sort(v),100*np.arange(1,len(v)+1)/len(v),where='post',color=color,label=name)
        axes[1,0].plot(*row['xy'].T,color=color,label=name,lw=1.4)
    axes[0,0].set(xlabel='规划任务序号（每次运行单独编号）',ylabel='完整规划任务耗时（毫秒）',title='成功和失败任务均纳入统计')
    axes[0,1].set(xlabel='完整规划任务耗时（毫秒）',ylabel='任务累计比例（%）',title='工作线程耗时分布；不是导航成功率',ylim=(0,101))
    axes[1,0].set(xlabel='仿真世界 X（米）',ylabel='仿真世界 Y（米）',title='车辆实际轨迹：外部目标各发布一次')
    axes[1,0].set_aspect('equal',adjustable='datalim')
    for ax in axes.flat[:3]:ax.grid(alpha=.18);ax.legend(frameon=False)
    axes[1,1].axis('off')
    table=[]
    for row in rows:
        s=row['summary'];m=row['planner_ms']
        table.append(['通过' if row['passed'] else '失败',str(m['count']),f"{m['p95']:.1f}",f"{m['maximum']:.1f}",
                      f"{s['duration_s']:.2f}",f"{s['truth_error_m']:.3f}",f"{row['goal_requests']}/{row['goal_acknowledgements']}"])
    metrics=['完整链路资格','规划任务数','任务 P95 / 毫秒','任务最大值 / 毫秒','到达并停稳 / 仿真秒','终点位置误差 / 米','目标发送 / 接收次数']
    t=axes[1,1].table(cellText=list(map(list,zip(*table))),rowLabels=metrics,colLabels=names,cellLoc='center',loc='center',colWidths=[.27,.31])
    t.auto_set_font_size(False);t.set_fontsize(10);t.scale(1.,1.65)
    fig.suptitle('单目标长走廊：共享规划预算的定向资格检查\n每版本各一次运行；任务不是独立重复实验；不据此推断通用导航成功率',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.93))
    for suffix in ('png','pdf'):fig.savefig(a.output/f'planning_budget.{suffix}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    for label,row in zip(('before','after'),rows):
        with (a.output/f'{label}_planner_jobs.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=('index','result','elapsed_ms'));w.writeheader();w.writerows(row['jobs'])
        del row['xy']
    (a.output/'inventory.json').write_text(json.dumps(dict(
        scope=__doc__,plotter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),runs=rows),ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
