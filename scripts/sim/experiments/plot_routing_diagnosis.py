#!/usr/bin/env python3
"""Chinese single-frame routing diagnosis; not a physical success/ablation study."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from rclpy.serialization import deserialize_message
from nav_msgs.msg import OccupancyGrid
from se2_grid_msgs.msg import SE2Grid
from fsgp_bgk_se2.se2_grid_tools import layer_refs,multiarray_to_matrix,grid_buffer_to_world
from plot_style import configure_chinese


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('snapshot',type=Path);p.add_argument('before',type=Path);p.add_argument('after',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    memory=deserialize_message((a.snapshot/'topology_memory.cdr').read_bytes(),OccupancyGrid)
    fused=deserialize_message((a.snapshot/'map.cdr').read_bytes(),SE2Grid)
    old=np.asarray(memory.data).reshape(memory.info.height,memory.info.width)
    new=old.copy();res=memory.info.resolution;ox=memory.info.origin.position.x;oy=memory.info.origin.position.y
    refs=layer_refs(fused);history=grid_buffer_to_world(multiarray_to_matrix(fused.data[refs['historical_obstacle_advisory'].data_indices[0]]),fused)
    hres=fused.info.pos_resolution;hx=fused.info.pose.position.x-fused.info.length_x/2;hy=fused.info.pose.position.y-fused.info.length_y/2
    for row,col in np.argwhere(history>=.5):
        for dx in (.25,.75):
            for dy in (.25,.75):
                x=int(np.floor((hx+(col+dx)*hres-ox)/res));y=int(np.floor((hy+(row+dy)*hres-oy)/res))
                if 0<=x<new.shape[1] and 0<=y<new.shape[0]:new[y,x]=100
    before=json.loads(a.before.read_text());after=json.loads(a.after.read_text())
    b=next(v for v in before['runs'] if v['mode']==0 and v['budget_ms']==200.)
    c=next(v for v in after['runs'] if v['mode']==0 and v['budget_ms']==24.)
    configure_chinese();fig,axes=plt.subplots(2,1,figsize=(12,8),layout='constrained')
    extent=[ox,ox+old.shape[1]*res,oy,oy+old.shape[0]*res]
    for ax,values,result,title in zip(axes,(old,new),(b,c),('遗漏历史墙体：引导路径尝试返回原走廊内的虚假缺口','补入同帧历史几何：从已知墙端绕向目标一侧')):
        ax.imshow(np.where(values==100,2,np.where(values==0,1,0)),origin='lower',extent=extent,
            cmap=ListedColormap(['#fafbfc','#dbe6eb','#394d5c']),vmin=0,vmax=2,interpolation='nearest')
        xy=np.asarray(result['path']);ax.plot(xy[:,0],xy[:,1],color='#187e9a',lw=1.7,label='离线全局引导（含未知区域）')
        ax.scatter(*before['start'][:2],color='#24865c',s=28,zorder=5,label='同一帧车辆位置')
        ax.scatter(*before['goal'],marker='*',s=130,color='#ca8a21',zorder=5,label='同一最终目标')
        ax.set(xlim=(-1.5,22),ylim=(-2.7,5.5),aspect='equal',xlabel='地图 X 坐标（米）',ylabel='地图 Y 坐标（米）',title=title)
        ax.grid(alpha=.12);ax.legend(loc='upper right',fontsize=8)
    fig.suptitle('长走廊路由失败诊断：同一帧融合地图与车辆位置',fontsize=14)
    fig.text(.5,-.025,'记录时刻约 85 秒；这是单帧信息遗漏诊断，包含未知区域的引导线不代表可执行轨迹或实际导航成功。',ha='center',fontsize=9)
    for ext in ('png','pdf'):fig.savefig(a.output/f'routing_diagnosis.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    files=[a.snapshot/'topology_memory.cdr',a.snapshot/'map.cdr',a.before,a.after,Path(__file__)]
    inventory={'scope':__doc__,'source_files':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        'before':{'search_budget_ms':200,'expanded':b['expanded'],'reached':b['reached']},
        'after':{'search_budget_ms':24,'expanded':c['expanded'],'reached':c['reached']},
        'interpretation':'Different allowed search budgets; do not interpret this single diagnostic as a timing benchmark or repeated statistical ablation.'}
    (a.output/'inventory.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
