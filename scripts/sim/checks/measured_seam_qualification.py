#!/usr/bin/env python3
"""Source-matched diagnosis of measured-ground likelihood loss near columns.

One archived model and one changed model per input, not a navigation trial or
empirical probability calibration. Occupancy and physical risk limits stay fixed.
Pure-GP pose bounds exclude footprints selecting the separate initial stance.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from rclpy.serialization import deserialize_message
from se2_grid_msgs.msg import SE2Grid
from fsgp_bgk_se2.se2_grid_tools import get_layer_slices,grid_buffer_to_world
from fsgp_bgk_se2.shared_terrain import shared_geometric_probability
from fsgp_bgk_se2.probabilistic_risk import _geometry
from fsgp_bgk_se2.obstacle_columns import inspect_body_collision
from shared_certificate_qualification import frozen_prediction


def run(snapshot,output,device,column_ground=False):
    old,p,occupied,provenance=frozen_prediction(snapshot,device,replay_initial=True)
    new,_,same,changed=frozen_prediction(snapshot,device,replay_initial=True,
        measured_seams_override=True,column_ground_override=True if column_ground else None,verify_posterior=False)
    if not np.array_equal(occupied,same):raise ValueError('occupancy changed')
    m=deserialize_message((snapshot/'map.cdr').read_bytes(),SE2Grid)
    def layer(name):return grid_buffer_to_world(get_layer_slices(m,name)[0][0],m)
    pose=provenance['query_pose'];x,y,z,yaw=pose;minimum=np.asarray(provenance['map_min']);res=p.resolution
    rows,cols=old.mean.shape
    xx,yy=np.meshgrid(minimum[0]+(np.arange(cols)+.5)*res,minimum[1]+(np.arange(rows)+.5)*res)
    roi=(abs(xx-x)<=2.5)&(abs(yy-y)<=2.5)
    k=int((yaw%(2*math.pi))/m.info.yaw_resolution)
    mask=np.zeros((p.yaw_bins,rows,cols),bool);mask[k]=roi
    initial=layer('gp_model_initial_support')>.5
    offsets,_,_=_geometry(p,k)
    for dr,dc in offsets:
        # Boundary handling is performed by the production proof as well.
        r,c=np.where(mask[k]);inside=(r+dr>=0)&(r+dr<rows)&(c+dc>=0)&(c+dc<cols)
        own=np.ones(len(r),bool);own[inside]=initial[r[inside]+dr,c[inside]+dc]
        mask[k,r[own],c[own]]=False
    bounds=[]
    for prediction in (old,new):
        result=shared_geometric_probability(prediction,p,np.zeros_like(occupied),query_mask=mask)
        field=np.full((rows,cols),np.nan,np.float32)
        r,c,u=result[k];field[r,c]=u;bounds.append(field)
    pure_old=bounds[0].copy()
    if column_ground:
        # The old selected field includes original column-side heights where
        # its GP was inadmissible. Compare the actual published component,
        # not a different pure-GP field which never supplied those cells.
        bounds[0]=grid_buffer_to_world(get_layer_slices(m,'terrain_failure_probability_upper')[0][k],m)
        bounds[0]=np.where(mask[k],bounds[0],np.nan)
    bits=layer('collision_subcell_bits')
    body=[float(layer(n).flat[0]) for n in ('collision_body_front','collision_body_back','collision_body_half_width')]
    clear=np.zeros((rows,cols),bool)
    for r,c in zip(*np.where(mask[k])):
        unsafe,known=inspect_body_collision(occupied,bits,minimum,res,xx[r,c],yy[r,c],yaw,*body,
            factor=int(layer('collision_subcell_factor').flat[0]))
        clear[r,c]=known and not unsafe
    shared=mask[k]&clear&np.isfinite(bounds[0])&np.isfinite(bounds[1])
    probes=[]
    for distance in (-1.5,-1.,-.5,0.,.5,1.,1.5):
        qx,qy=x+distance*math.cos(yaw),y+distance*math.sin(yaw)
        r,c=int(math.floor((qy-minimum[1])/res)),int(math.floor((qx-minimum[0])/res))
        if not (0<=r<rows and 0<=c<cols):continue
        clean=lambda v:float(v) if np.isfinite(v) else None
        probes.append(dict(offset_m=distance,xy=[qx,qy],body_clear=bool(clear[r,c]),
            old_bound=clean(bounds[0][r,c]),new_bound=clean(bounds[1][r,c]),
            original_measured=bool(layer('terrain_observed')[r,c]>.5),
            raw_sigma_m=clean(np.sqrt(layer('gp_variance_observed')[r,c])),
            old_sigma_m=clean(np.sqrt(old.variance[r,c])),new_sigma_m=clean(np.sqrt(new.variance[r,c]))))
    record=dict(scope=__doc__,snapshot=str(snapshot.resolve()),device=device,source=provenance,
        separate_ground_and_columns=column_ground,baseline_bound_source='published terrain component' if column_ground else 'pure GP alternative proof',
        changed_model=changed['model'],physical_limits=asdict(p),body_envelope_m=body,
        original_training_cells=old.training_points,changed_training_cells=new.training_points,
        occupied_columns=int(occupied.sum()),same_supported_and_body_clear_queries=int(shared.sum()),
        old_above_state_limit=int(np.count_nonzero(shared&(bounds[0]>.05))),
        new_above_state_limit=int(np.count_nonzero(shared&(bounds[1]>.05))),probes=probes,
        caveat='conditional GP alternative proof; not the composed initial/GP/native live-body/path/braking certificate; no physical success claim')
    output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'arrays.npz',x=xx,y=yy,roi=roi,observed=layer('terrain_observed'),
        occupied=occupied,bits=bits,raw_height=layer('gp_elevation_observed'),raw_variance=layer('gp_variance_observed'),
        old_mean=old.mean,new_mean=new.mean,old_variance=old.variance,new_variance=new.variance,
        old_bound=bounds[0],new_bound=bounds[1],old_pure_gp_bound=pure_old,body_clear=clear)
    record['arrays_sha256']=hashlib.sha256((output/'arrays.npz').read_bytes()).hexdigest()
    (output/'qualification.json').write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    plot(record,output)
    print(json.dumps({key:record[key] for key in ('snapshot','original_training_cells','changed_training_cells',
        'same_supported_and_body_clear_queries','old_above_state_limit','new_above_state_limit','probes')},ensure_ascii=False),flush=True)


def plot(record,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib import font_manager
    font=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':[font_manager.FontProperties(fname=str(font)).get_name(),'DejaVu Sans'],
        'axes.unicode_minus':False,'font.size':9})
    a=np.load(output/'arrays.npz');roi=a['roi'];rr,cc=np.where(roi)
    r0,r1,c0,c1=rr.min(),rr.max()+1,cc.min(),cc.max()+1
    x=a['x'][r0:r1,c0:c1];y=a['y'][r0:r1,c0:c1]
    extent=[x.min()-.1,x.max()+.1,y.min()-.1,y.max()+.1]
    pose=record['source']['query_pose'];yaw=pose[3];c,s=math.cos(yaw),math.sin(yaw)
    front,back,width=record['body_envelope_m'];body=np.array([[front,width],[front,-width],[-back,-width],[-back,width]])@np.array([[c,s],[-s,c]])+pose[:2]
    fig,axes=plt.subplots(2,3,figsize=(12,7.8),layout='constrained')
    entries=[('raw_variance','原始实测地面标准差','标准差（厘米）','magma',0,6,True),
        ('old_bound','原模型：地形条件风险上界','上界（非实测失败率）','viridis',0,1,False),
        ('new_bound','修正后：地形条件风险上界','上界（非实测失败率）','viridis',0,1,False),
        ('old_variance','原模型高程标准差','标准差（厘米）','magma',0,6,True),
        ('new_variance','修正后高程标准差','标准差（厘米）','magma',0,6,True),
        ('mean_change','高程均值变化：修正后减原模型','变化（厘米）','RdBu_r',-3,3,False)]
    for ax,(key,title,label,cmap,low,high,sigma) in zip(axes.flat,entries):
        field=(a['new_mean']-a['old_mean'])*100 if key=='mean_change' else a[key]
        if sigma:field=np.sqrt(field)*100
        im=ax.imshow(field[r0:r1,c0:c1],origin='lower',extent=extent,cmap=cmap,vmin=low,vmax=high,interpolation='none')
        ax.contour(x,y,a['occupied'][r0:r1,c0:c1],levels=[.5],colors=['#313844'],linewidths=.8)
        ax.add_patch(Polygon(body,fill=False,color='#efb94c',linewidth=1.3))
        ax.set(title=title,xlabel='地图X（米）',ylabel='地图Y（米）',aspect='equal')
        fig.colorbar(im,ax=ax,shrink=.82,label=label)
    fig.suptitle('墙边已测地面不应因当前遮挡退回缺测\n同一源帧；车框含单次余量，黑线为原始占据；空白不代表安全',fontsize=12)
    for suffix in ('png','pdf'):fig.savefig(output/('measured_ground_seams.'+suffix),dpi=190)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('snapshots',type=Path,nargs='+')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--column-ground',action='store_true')
    a=p.parse_args();torch.set_num_threads(1)
    if a.output.exists():raise ValueError('output exists')
    a.output.mkdir(parents=True)
    for snapshot in a.snapshots:run(snapshot,a.output/snapshot.parent.name,a.device,a.column_ground)


if __name__=='__main__':main()
