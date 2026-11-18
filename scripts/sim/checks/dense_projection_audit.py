#!/usr/bin/env python3
"""Read-only dense-scene failure diagnosis, not a navigation success experiment.

Separate raw obstacle extent, heading-dependent vehicle configuration space,
and statistical terrain support on one source-stamped recorded input. Candidate
results never enter a live controller; an any-heading free cell is not a path.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from rclpy.serialization import deserialize_message
from se2_grid_msgs.msg import SE2Grid
from sensor_msgs.msg import PointCloud2

from fsgp_bgk_se2.fsgp_bgk_se2_node import FsgpBgkSe2Node
from fsgp_bgk_se2.obstacle_columns import collision_subcells,collision_pose_masks
from fsgp_bgk_se2.overhead_clearance import clip_overhead
from fsgp_bgk_se2.se2_grid_tools import get_layer_slices,grid_buffer_to_world
from shared_certificate_qualification import frozen_prediction


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    root=a.run/'probability_after_trial';payload=(root/'map.cdr').read_bytes()
    msg=deserialize_message(payload,SE2Grid)
    def layers(name):
        v,_,_=get_layer_slices(msg,name)
        if not v:raise ValueError(f'missing {name}')
        return np.stack([grid_buffer_to_world(s,msg) for s in v])
    def layer(name):return layers(name)[0]
    metadata=json.loads((a.run/'run_metadata.json').read_text())
    if metadata['arguments']['obstacle_memory_role']!='advisory':
        raise ValueError('a single current scan cannot refine hard historical occupancy')
    cloud=deserialize_message((root/'completion_cloud.cdr').read_bytes(),PointCloud2)
    points=FsgpBgkSe2Node._decode_points(cloud)
    res=msg.info.pos_resolution;rows,cols=layer('collision_risk').shape
    origin=(msg.info.pose.position.x-msg.info.length_x/2,msg.info.pose.position.y-msg.info.length_y/2)
    points=clip_overhead(points,layer('collision_ceiling_elevation'),origin,res)
    occupied=layer('collision_risk')>.5
    observed=layer('terrain_observed')>.5;variance=layer('gp_variance_observed')
    floor=np.where(observed&np.isfinite(variance)&(variance>=0)&(variance<=.02**2),layer('gp_elevation_observed'),np.nan)
    old_bits=layer('collision_subcell_bits').astype(np.uint8)
    # Reproduce old extent before attributing changes to the floor distinction.
    replay=collision_subcells(points,occupied,origin,res)
    if not np.array_equal(old_bits,replay):raise ValueError('archived collision extent does not reproduce')
    new_bits=collision_subcells(points,occupied,origin,res,measured_floor=floor)
    if not np.array_equal(new_bits>0,occupied):raise AssertionError('a coarse occupied cell was erased')
    diagnostic=json.loads((root/'diagnostic.json').read_text());yaw=diagnostic['pose'][3]
    bins=round(2*np.pi/msg.info.yaw_resolution);k=int((yaw%(2*np.pi))/(2*np.pi)*bins)
    backend='cupy' if a.device=='cuda' else 'cpu'
    before=collision_pose_masks(old_bits,res,.68,.58,.5,bins,backend=backend)
    after=collision_pose_masks(new_bits,res,.68,.58,.5,bins,backend=backend)
    # Replay checks input hashes, frames, source time and old posterior values.
    old,_,_,provenance=frozen_prediction(root,a.device,replay_initial=True)
    new,_,_,candidate=frozen_prediction(root,a.device,replay_initial=True,
        complete_halo_override=True,column_seams_override=True,verify_posterior=False)
    x,y=np.meshgrid(origin[0]+(np.arange(cols)+.5)*res,origin[1]+(np.arange(rows)+.5)*res)
    focus=(x>-.4)&(x<12)&(abs(y)<3.9)
    raw_upper=layers('failure_probability_upper')[k];valid=layers('probability_valid')[k]>.5
    invalid_red=focus&~valid&np.isfinite(raw_upper)&(raw_upper>.05)
    bit_count=lambda b:np.array([int(v).bit_count() for v in b.ravel()]).reshape(b.shape)
    report=dict(scope=__doc__,source_run=str(a.run.resolve()),pose=diagnostic['pose'],
        yaw_bin_center_deg=(k+.5)*360/bins,focus_bounds_m=[-.4,12,-3.9,3.9],
        body_protected_front_back_halfwidth_m=[.68,.58,.5],single_margin_m=.10,
        focus_cells=int(focus.sum()),raw_occupied_cells=int((focus&occupied).sum()),
        invalid_red_cells=int(invalid_red.sum()),invalid_red_without_geometry=int((invalid_red&~before[k]).sum()),
        before=dict(occupied_subcells=int(bit_count(old_bits)[focus].sum()),
            geometry_clear_current_heading=int((focus&~before[k]).sum()),
            geometry_clear_some_heading=int((focus&~before.all(axis=0)).sum()),
            gp_completed_cells=int((focus&old.completed).sum()),conditioning_cells=old.training_points),
        candidate=dict(occupied_subcells=int(bit_count(new_bits)[focus].sum()),
            geometry_clear_current_heading=int((focus&~after[k]).sum()),
            geometry_clear_some_heading=int((focus&~after.all(axis=0)).sum()),
            gp_completed_cells=int((focus&new.completed).sum()),conditioning_cells=new.training_points),
        original_model=provenance,candidate_model=candidate,
        interpretation='Geometry clear is not measured ground or a chance certificate; some-heading counts are not connected SE(2) paths. Candidate GP counts use its own gates, not the final mixed-source navigation proof. This one archived frame is not the user screenshot or a success rate.')
    paths=[root/'map.cdr',root/'completion_cloud.cdr',root/'diagnostic.json',Path(__file__),
        Path('src/fsgp_bgk_se2/fsgp_bgk_se2/obstacle_columns.py'),Path('src/fsgp_bgk_se2/fsgp_bgk_se2/shared_terrain.py')]
    report['sha256']={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths}
    (a.output/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
    from plot_style import configure_chinese
    configure_chinese();fig,axes=plt.subplots(2,3,figsize=(13,8.2),constrained_layout=True)
    extent=[origin[0],origin[0]+cols*res,origin[1],origin[1]+rows*res]
    cmap=ListedColormap(['#edf3f5','#c34d61'])
    def image(ax,values,title,cm=cmap):
        ax.imshow(values,origin='lower',extent=extent,cmap=cm,interpolation='nearest',vmin=0,vmax=cm.N-1)
        ax.set(title=title,xlim=(-.4,12),ylim=(-3.9,3.9),aspect='equal',xlabel='地图 X（米）',ylabel='地图 Y（米）')
        ax.scatter(*diagnostic['pose'][:2],s=18,c='#20363f',zorder=5)
    image(axes[0,0],occupied,'原始占据柱：不是车身膨胀图')
    image(axes[0,1],before[k],f'原投影：{(k+.5)*360/bins:.0f}° 航向的车身碰撞区域')
    image(axes[0,2],after[k],'修正后：同一车身、同一航向、同一余量')
    image(axes[1,0],after.all(axis=0),'修正后：所有离散航向均碰撞的位置')
    for ax,pred,title in ((axes[1,1],old,'原 GP：实测与获准补全'),(axes[1,2],new,'修正后 GP：保留近柱实测与足迹边界')):
        support=np.zeros((rows,cols),np.int32);support[observed]=1;support[pred.completed]=2;support[occupied]=3
        image(ax,support,title,ListedColormap(['#e4e7eb','#72ada5','#c7a45a','#c34d61']))
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([],[],marker='s',ls='',color=c,label=t) for c,t in
        [('#72ada5','实测地面'),('#c7a45a','通过门控的 GP 补全'),('#c34d61','占据 / 指定姿态碰撞'),('#e4e7eb','未知：不等于安全')]],
        loc='lower center',ncol=4,bbox_to_anchor=(.5,-.045))
    fig.suptitle('密集柱阵的同帧诊断：障碍轮廓、完整车身与地形支撑分开检查\n至少存在一个无碰撞航向，不等于存在可执行路径；本图不是导航通过结果',fontsize=13)
    for ext in ('png','pdf'):fig.savefig(a.output/f'dense_projection_audit.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    print(json.dumps({k:report[k] for k in ('focus_cells','raw_occupied_cells','invalid_red_cells','invalid_red_without_geometry','before','candidate')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
