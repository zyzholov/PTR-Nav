#!/usr/bin/env python3
"""Plot retained single-external-goal trials, including failed qualifications.

Never reconstructs a ground-truth route or sends navigation inputs. Each
source hash and protocol verdict is retained so different revisions cannot
silently become repeated trials of one method.
"""
import argparse
import hashlib
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src/isaac_sim_hunter_livo2/scripts'))
from boundary_courses import fixtures
from plot_style import configure_chinese
sys.path.insert(0,str(ROOT/'scripts/sim/checks'))
from clearance_metrics import sampled_body_clearances
from boundary_courses import movers


def plot_run(run,output):
    data_path=run/'navigation_data.json'
    d=json.loads(data_path.read_text());s=d['summary'];a=np.asarray(d['truth'])
    t=a[:,6]-a[0,6];c=np.asarray(d['controls'])
    moving=d.get('moving_obstacles',{})
    fig=plt.figure(figsize=(12,10 if moving else 8),layout='constrained')
    gs=fig.add_gridspec(4 if moving else 3,2,width_ratios=(1.15,1))
    ax=fig.add_subplot(gs[:,0]);distance=fig.add_subplot(gs[0,1])
    clearance=fig.add_subplot(gs[1,1]) if moving else None
    speed=fig.add_subplot(gs[2 if moving else 1,1]);steer=fig.add_subplot(gs[3 if moving else 2,1])
    for obj in fixtures(d['spec']):
        if obj['shape']=='box':
            patch=Rectangle((obj['x']-obj['length']/2,obj['y']-obj['width']/2),obj['length'],obj['width'],
                facecolor='#d1d7dd',edgecolor='#66717d',linewidth=.5)
        else:patch=Circle((obj['x'],obj['y']),obj['radius'],facecolor='#d1d7dd',edgecolor='#66717d',linewidth=.5)
        ax.add_patch(patch)
    ax.plot(a[:,1],a[:,2],color='#147d92',lw=2,label='独立物理真值轨迹')
    clearance_rows=[]
    colors=('#c66c30','#7762a8','#3d7581')
    for i,obj in enumerate(movers(d['spec'],0.) if moving else []):
        actor=obj['name'];positions=np.asarray(moving.get(actor,[]))
        if len(positions)<2:continue
        color=colors[i%len(colors)]
        ax.plot(positions[:,1],positions[:,2],color=color,ls='--',lw=1.2,label=f'{actor} 障碍轨迹')
        ax.scatter(*positions[0,1:3],s=22,c=color,marker='s')
        values=sampled_body_clearances(d['truth'],moving[actor],radius=obj['radius'],
                                     front=.58,back=.48,half_width=.4)
        if values:
            values=np.asarray(values)
            clearance.plot(values[:,0]-a[0,6],values[:,1],c=color,lw=1.1,
                           label=f'{actor}：最小 {values[:,1].min():.3f} m')
            clearance_rows.extend((row[0],row[0]-a[0,6],actor,row[1],row[2]) for row in values)
    ax.scatter(*a[0,1:3],c='#147d92',s=45,label='启动位置',zorder=4)
    ax.scatter(*d['truth_goal'],marker='*',s=180,c='#e4a530',edgecolor='#76510c',label='启动时唯一最终目标',zorder=4)
    ax.scatter(*a[-1,1:3],marker='x',s=60,c='#b44159',label='终止位置',zorder=4)
    ax.set(xlabel='世界 X（米）',ylabel='世界 Y（米）',title='从启动到结束：未提供中间目标或人工路线',aspect='equal')
    xmin=min(a[:,1].min(),d['truth_goal'][0])-1.5;xmax=max(a[:,1].max(),d['truth_goal'][0])+1.5
    if d['spec']['kind']=='side_goal_corridor':xmin=min(xmin,-d['spec']['length_m']/2-1)
    ymin=min(a[:,2].min(),d['truth_goal'][1])-2;ymax=max(a[:,2].max(),d['truth_goal'][1])+2
    for positions in moving.values():
        if not positions:continue
        values=np.asarray(positions)
        xmin=min(xmin,values[:,1].min()-.7);xmax=max(xmax,values[:,1].max()+.7)
        ymin=min(ymin,values[:,2].min()-.7);ymax=max(ymax,values[:,2].max()+.7)
    ax.set_xlim(xmin,xmax);ax.set_ylim(ymin,ymax)
    ax.legend(fontsize=8,loc='upper left');ax.grid(alpha=.2)
    error=np.linalg.norm(a[:,1:3]-np.asarray(d['truth_goal']),axis=1)
    distance.plot(t,error,color='#147d92');distance.axhline(.3,color='#b44159',ls='--',label='到达阈值 0.30 m')
    distance.set(ylabel='到最终目标的距离（米）',title='到达过程：绕障允许暂时远离最终目标');distance.legend(fontsize=8)
    if len(c):
        tc=c[:,0]-a[0,6]
        speed.plot(tc,c[:,1],color='#147d92',lw=1,label='速度指令')
        steer.plot(tc,np.rad2deg(c[:,2]),color='#6e5a91',lw=1,label='前轮转向指令')
    if moving and len(a)>2:
        delta=np.diff(a[:,1:3],axis=0);dt=np.diff(a[:,6]);good=dt>0
        measured=(delta[:,0]*np.cos(a[:-1,7])+delta[:,1]*np.sin(a[:-1,7]))/np.maximum(dt,1e-9)
        speed.plot(t[1:][good],measured[good],color='#af7734',lw=.7,alpha=.8,label='真值纵向速度（仅评估）')
        clearance.axhline(.10,color='#b44159',ls='--',lw=1,label='资格要求：至少 0.10 m')
        clearance.axhline(0,color='#3f4650',lw=.7)
        clearance.set(ylabel='车身包络至障碍净空（米）',title='正值为分离；负值为包络重叠')
        clearance.legend(fontsize=7)
        with (output/(run.name+'.clearance.csv')).open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['stamp_s','since_goal_s','actor','body_envelope_clearance_m','truth_bracket_gap_s'])
            writer.writerows(clearance_rows)
    speed.set(ylabel='速度（米/秒）');steer.set(ylabel='转向角（度）',xlabel='收到初始目标后的仿真时间（秒）')
    speed.legend(fontsize=8);steer.legend(fontsize=8)
    for right in (distance,speed,steer,*([clearance] if moving else [])):
        right.grid(alpha=.2);right.set_xlim(0,max(t[-1],1))
    pipeline=json.loads((run/'result.json').read_text())
    accepted=bool(pipeline['passed']);protocol=s.get('single_external_goal',False)
    fig.suptitle(f"单个外部最终目标完整链路资格：{'通过' if accepted else '未通过'}\n"
        f"外部请求 {len(d.get('goal_requests',[]))} 次 / 规划器确认 {len(d.get('goal_acknowledgements',[]))} 次；"
        f"单目标协议{'通过' if protocol else '未通过或尚未记录'}；运行 {run.name}",fontsize=12)
    for ext in ('png','pdf'):fig.savefig(output/(run.name+'.'+ext),dpi=190)
    plt.close(fig)
    metadata=json.loads((run/'run_metadata.json').read_text())
    return dict(run=str(run.relative_to(ROOT)),accepted=accepted,task_checks_accepted=bool(d['accepted']),
        pipeline_failure_reason=pipeline.get('failure_reason'),summary=s,
        requested_goal=d.get('requested_goal'),goal_requests=d.get('goal_requests'),
        goal_acknowledgements=d.get('goal_acknowledgements'),
        clearance_scope=('timestamp-bracketed planar enclosing body vs scripted cylinder; '
            'not continuous mesh contact; all moving objects retained') if moving else None,
        navigation_data_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
        method_hashes=metadata.get('method_hashes'),source_snapshot_sha256=metadata.get('source_snapshot_sha256'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('runs',nargs='+',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);configure_chinese()
    rows=[plot_run(run.resolve(),a.output) for run in a.runs]
    result=dict(scope='定向工程资格检查；失败与不同版本逐次保留，不合并为总体成功率；真值只作评估。',
        plotter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),runs=rows)
    (a.output/'inventory.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps([dict(run=r['run'],accepted=r['accepted']) for r in rows],ensure_ascii=False))


if __name__=='__main__':main()
