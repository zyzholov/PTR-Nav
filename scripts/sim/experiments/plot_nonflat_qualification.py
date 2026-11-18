#!/usr/bin/env python3
"""Plot complete recorded nonflat qualifications, retaining failures and hashes.

Truth is used only for assessment/plots. No point or route is sent to navigation.
Individual engineering revisions must not become a pooled success-rate study.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Rectangle
import numpy as np
from plot_style import configure_chinese


def plot_run(run,output):
    path=run/'navigation_data.json';d=json.loads(path.read_text());s=d['summary']
    verdict=json.loads((run/'result.json').read_text());a=np.asarray(d['truth'],float)
    if a.ndim!=2 or a.shape[1]!=8 or len(a)<2 or np.any(np.diff(a[:,6])<=0):
        raise ValueError('expected monotonic independent truth with documented eight columns')
    t=a[:,6]-a[0,6];c=np.asarray(d['controls'],float);goal=np.asarray(d['truth_goal'])
    fig,axes=plt.subplots(2,2,figsize=(11,7.4),layout='constrained')
    xy,height,speed,angle=axes.ravel();teal='#147d92';gold='#c88e28';red='#b44159'
    for obj in s.get('course_geometry_world_m',[]):
        if obj['shape']=='cylinder':patch=Circle((obj['x'],obj['y']),obj['radius'])
        else:patch=Rectangle((obj['x']-obj['length']/2,obj['y']-obj['width']/2),obj['length'],obj['width'])
        patch.set(facecolor='#d6dce2',edgecolor='#5b6672',linewidth=.8);xy.add_patch(patch)
    xy.plot(a[:,1],a[:,2],c=teal,lw=1.8,label='独立物理真值轨迹')
    xy.scatter(*a[0,1:3],c=teal,s=32,label='初始位置',zorder=3)
    xy.scatter(*goal,c=gold,marker='*',s=130,label='最终目标',zorder=3)
    xy.scatter(*a[-1,1:3],c=red,marker='x',s=40,label='终止位置',zorder=3)
    xy.set(xlabel='世界 X（米）',ylabel='世界 Y（米）',title='俯视轨迹与物理障碍物',aspect='equal')
    xy.set_xlim(min(a[:,1].min(),goal[0])-1,max(a[:,1].max(),goal[0])+1)
    xy.set_ylim(min(a[:,2].min(),goal[1])-.9,max(a[:,2].max(),goal[1])+.9)
    xy.legend(fontsize=8,loc='best')
    height.plot(t,a[:,3]-a[0,3],c=teal,lw=1.5)
    height.set(xlabel='目标发出后的仿真时间（秒）',ylabel='车体基准点相对初始高度（米）',
               title=f"实际升降过程；终点误差 {s['truth_error_m']:.3f} 米")
    speed.plot(c[:,0]-a[0,6],c[:,1],c=teal,lw=1.2,label='速度指令')
    delta=np.diff(a[:,1:3],axis=0);dt=np.diff(a[:,6])
    v=(delta[:,0]*np.cos(a[:-1,7])+delta[:,1]*np.sin(a[:-1,7]))/dt
    speed.plot(t[1:],v,c=gold,lw=.65,alpha=.85,label='真值纵向速度（仅评估）')
    speed.set(xlabel='目标发出后的仿真时间（秒）',ylabel='速度（米/秒）',title='速度响应与停稳')
    speed.legend(fontsize=8)
    angle.plot(t,np.rad2deg(a[:,4]),c=teal,lw=1.1,label='俯仰角')
    angle.plot(t,np.rad2deg(a[:,5]),c=red,lw=1.1,label='侧倾角')
    angle.set(xlabel='目标发出后的仿真时间（秒）',ylabel='姿态角（度）',title='车辆实际姿态；不等同于地面坡度')
    angle.legend(fontsize=8)
    for ax in axes.ravel():ax.grid(alpha=.18)
    for ax in (height,speed,angle):ax.set_xlim(0,t[-1])
    case={'ramp':'普通坡道','ramp_detour':'坡道绕柱','ramp_crossing':'坡道动态障碍'}.get(s['scenario'],s['scenario'])
    direction='下坡' if s.get('ramp_direction')=='downhill' else '上坡'
    fig.suptitle(f"非平坦导航资格：{direction} · {case} · 完整链路{'通过' if verdict['passed'] else '未通过'}\n"
        f"{run.name}；每次保留完整失败记录，真值仅作评估",fontsize=12)
    for ext in ('png','pdf'):fig.savefig(output/(run.name+'.'+ext),dpi=190)
    plt.close(fig)
    meta=json.loads((run/'run_metadata.json').read_text())
    return dict(run=str(run),pipeline_passed=bool(verdict['passed']),task_checks_accepted=bool(d['accepted']),
        summary=s,navigation_data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_snapshot_sha256=meta.get('source_snapshot_sha256'),method_hashes=meta.get('method_hashes'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('runs',nargs='+',type=Path)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True);configure_chinese()
    rows=[plot_run(run.resolve(),args.output) for run in args.runs]
    (args.output/'inventory.json').write_text(json.dumps(dict(
        scope='定向工程资格检查；逐个源码版本报告，不合并为重复试验成功率；时间为仿真时间。',
        plotter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),runs=rows),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps([dict(run=r['run'],passed=r['pipeline_passed']) for r in rows],ensure_ascii=False))


if __name__=='__main__':main()
