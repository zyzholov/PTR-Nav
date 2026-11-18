#!/usr/bin/env python3
"""Plot recorded stationary risk streams; frames are not independent terrain trials."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def read_trace(path,warmup=5.):
    payload=path.read_bytes();data=json.loads(payload)
    if not data['accepted'] or data['maximum_speed_command_mps']>.01:
        raise ValueError(f'not a qualified stationary capture: {path}')
    samples=data['samples'];begin=samples[0]['stamp_s']
    rows=[r for r in samples if r['stamp_s']-begin>=warmup]
    if len(rows)<40:raise ValueError('insufficient frames after warmup')
    if any(r['probes'][0]['values']['probability_initial_affine_basis']!=1. for r in rows):
        raise ValueError('initial-stance assumptions differ within the evaluated interval')
    return rows,begin,hashlib.sha256(payload).hexdigest()


def summarize(rows,probe):
    values=np.array([np.nan if r['probes'][probe]['probability_by_yaw'][r['yaw_index']] is None
                     else r['probes'][probe]['probability_by_yaw'][r['yaw_index']] for r in rows])
    finite=values[np.isfinite(values)];adjacent=np.diff(values)
    return dict(frames=len(rows),known_frames=len(finite),
        minimum=float(finite.min()) if len(finite) else None,
        maximum=float(finite.max()) if len(finite) else None,
        jumps_over_005=int(np.count_nonzero(np.abs(adjacent)>.05)),
        xy_cells=sorted({tuple(r['probes'][probe]['xy']) for r in rows}),
        collision_values=sorted({r['probes'][probe]['values']['collision_risk'] for r in rows}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before',type=Path,required=True);parser.add_argument('--after',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    inputs={};statistics={};traces={}
    for name,path in (('before',args.before),('after',args.after)):
        rows,begin,digest=read_trace(path);traces[name]=(rows,begin)
        statistics[name]=[summarize(rows,j) for j in range(3)]
        inputs[name]=dict(path=str(path.resolve()),sha256=digest)
    report=dict(scope='two stationary simulator startups; correlated frames, no arrival-rate or calibration claim',
        warmup_source_seconds=5.,jump_definition='adjacent finite current-yaw bounds differ by more than 0.05',
        inputs=inputs,statistics=statistics,
        same_xy_cells=all(a['xy_cells']==b['xy_cells'] for a,b in zip(statistics['before'],statistics['after'])))
    (args.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with (args.output/'samples.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(('run','time_s','offset_m','x_m','y_m','yaw_bin','upper'))
        for name,(rows,begin) in traces.items():
            for r in rows:
                for q in r['probes']:writer.writerow((name,r['stamp_s']-begin,q['offset_m'],*q['xy'],
                    r['yaw_index'],q['probability_by_yaw'][r['yaw_index']]))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,
        'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':200})
    fig,axes=plt.subplots(3,1,figsize=(10,9.5),sharex=True,layout='constrained')
    titles=('车体基准前方 0.6 米探针（接近前缘）','车体基准前方 1.0 米探针',
            '第一条凸条附近：车体基准前方 2.5 米探针')
    for j,ax in enumerate(axes):
        for name,color,label in (('before','#a45660','修正前'),('after','#207c87','修正后')):
            rows,begin=traces[name]
            t=np.array([r['stamp_s']-begin for r in rows]);v=np.array([
                np.nan if r['probes'][j]['probability_by_yaw'][r['yaw_index']] is None else
                100*r['probes'][j]['probability_by_yaw'][r['yaw_index']] for r in rows])
            ax.plot(t,v,color=color,label=label,lw=1.5,marker='.',ms=3)
        ax.axhline(5.,color='#69717b',ls='--',lw=1,label='姿态风险预算 5%')
        ax.set(yscale='log',ylim=(.01,110),title=titles[j],ylabel='模型条件风险上界（%）')
        ax.set_yticks((.01,.1,1,5,10,100));ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.grid(axis='y',alpha=.18);ax.legend(loc='upper right',ncol=3,fontsize=9)
    axes[-1].set_xlabel('从诊断目标发出后首帧算起的仿真时间（秒；前 5 秒不统计）')
    fig.suptitle('风险闪变源层复测：没有对风险数值或颜色做时间平滑',fontsize=15)
    locations='同一场景分别启动；探针不保证落在相同栅格，精确坐标见原始 CSV。'
    if all(len(statistics[n][0]['xy_cells'])==1 for n in ('before','after')):
        old=statistics['before'][0]['xy_cells'][0][1];new=statistics['after'][0]['xy_cells'][0][1]
        locations=f'同一场景分别启动；近车探针栅格 Y={old:+.2f} / {new:+.2f} 米。'
    fig.supxlabel(locations+'\n时间帧相互相关；曲线表示条件模型上界，不是实测碰撞概率或独立实验成功率。',fontsize=9)
    for suffix in ('png','pdf'):fig.savefig(args.output/f'risk_time_series.{suffix}')
    plt.close(fig)
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
