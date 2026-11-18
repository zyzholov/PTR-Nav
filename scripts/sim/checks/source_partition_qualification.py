#!/usr/bin/env python3
"""Source-matched failure replay of a tighter covariance bound, not navigation.

The archived posterior and selected height field must reproduce before any
certificate comparison. No change to support, measurements, budgets or body.
The optional output map is a counterfactual replay for native checks only.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import shutil
import numpy as np
import torch
from rclpy.serialization import deserialize_message,serialize_message
from se2_grid_msgs.msg import SE2Grid
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from fsgp_bgk_se2.se2_grid_tools import get_layer_slices,grid_buffer_to_world,world_to_grid_buffer,layer_refs,matrix_to_multiarray_like
from fsgp_bgk_se2.shared_terrain import shared_geometric_probability
from shared_certificate_qualification import frozen_prediction


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    shutil.copyfile(a.run/'run_metadata.json',a.output/'run_metadata.json')
    source=a.output/'source';source.mkdir()
    for name in ('map.cdr','odom.cdr'):shutil.copyfile(a.snapshot/name,source/name)
    cloud=a.snapshot/('completion_cloud.cdr' if (a.snapshot/'completion_cloud.cdr').exists() else 'nearby_cloud.cdr')
    shutil.copyfile(cloud,source/'completion_cloud.cdr')
    digest=lambda b:hashlib.sha256(b).hexdigest()
    raw=(source/'map.cdr').read_bytes();m=deserialize_message(raw,SE2Grid)
    c=deserialize_message(cloud.read_bytes(),PointCloud2)
    o=deserialize_message((source/'odom.cdr').read_bytes(),Odometry)
    q=o.pose.pose.orientation;yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
    def layer(n):return grid_buffer_to_world(get_layer_slices(m,n)[0][0],m)
    age=float(layer('gp_completion_age').flat[0])
    info=dict(map_sha256=digest(raw),pose=[o.pose.pose.position.x,o.pose.pose.position.y,o.pose.pose.position.z,yaw],
              completion_cloud=dict(sha256=digest(cloud.read_bytes()),source_age_s=age))
    (source/'diagnostic.json').write_text(json.dumps(info,indent=2))
    prediction,params,collision,provenance=frozen_prediction(source,'cpu',replay_initial=True)
    if prediction.plane_components!=3:raise ValueError('this qualification requires both recorded covariance sources')
    # Prior cells and GP-selected cells must describe the SAME height field.
    selected=(layer('startup_prior_used')>.5)|(layer('probability_joint_used')>.5)
    delta=float(np.nanmax(abs(prediction.mean[selected]-layer('probability_elevation')[selected])))
    if delta>2e-5:raise ValueError(f'selected field mismatch: {delta}')
    shape=(params.yaw_bins,*collision.shape)
    def raster(updates):
        v=np.full(shape,np.nan,np.float32)
        for k,(r,c,u) in enumerate(updates):v[k,r,c]=u
        return v
    empty=np.zeros_like(collision)
    old=raster(shared_geometric_probability(replace(prediction,plane_components=0),params,empty))
    new=raster(shared_geometric_probability(prediction,params,empty))
    if not np.array_equal(np.isfinite(old),np.isfinite(new)):raise ValueError('support changed')
    finite=np.isfinite(old)
    if np.max(new[finite]-old[finite])>3e-5:raise ValueError('new bound exceeds original envelope')
    gpu_error=None
    if torch.cuda.is_available():
        gpu=replace(prediction,factor=prediction.factor.cuda())
        actual=raster(shared_geometric_probability(gpu,params,empty))
        if not np.array_equal(np.isfinite(actual),finite):raise ValueError('GPU support changed')
        gpu_error=float(np.max(abs(actual[finite]-new[finite])))
        if gpu_error>8e-5:raise ValueError(f'GPU certificate mismatch: {gpu_error}')
    # Preserve the original inexpensive/conditional proofs where tighter.
    refs=layer_refs(m)
    for field in ('terrain_failure_probability_upper','failure_probability_upper'):
        ref=refs[field]
        for k,i in enumerate(ref.data_indices):
            previous=grid_buffer_to_world(get_layer_slices(m,field)[0][k],m)
            proposed=np.minimum(previous,np.where(np.isfinite(new[k]),new[k],previous))
            if field=='failure_probability_upper':
                # Never remove the recorded diagnostic whole-body overlay.
                original_terrain=grid_buffer_to_world(get_layer_slices(m,'terrain_failure_probability_upper')[0][k],m)
                proposed=np.where(previous>original_terrain+1e-6,previous,proposed)
            array=world_to_grid_buffer(proposed,m)
            m.data[i]=matrix_to_multiarray_like(array,m.data[i])
    counterfactual=a.output/'counterfactual';counterfactual.mkdir()
    (counterfactual/'map.cdr').write_bytes(serialize_message(m))
    for name in ('odom.cdr','completion_cloud.cdr'):shutil.copyfile(source/name,counterfactual/name)
    if (a.snapshot/'braking.json').exists():shutil.copyfile(a.snapshot/'braking.json',counterfactual/'braking.json')
    nav=json.loads((a.run/'navigation_data.json').read_text());g=nav['requested_goal']
    x0=g['x']-12*math.cos(g['yaw']);y0=g['y']-12*math.sin(g['yaw'])
    rows,cols=collision.shape;x,y=np.meshgrid(m.info.pose.position.x-m.info.length_x/2+(np.arange(cols)+.5)*params.resolution,
        m.info.pose.position.y-m.info.length_y/2+(np.arange(rows)+.5)*params.resolution)
    patch=(x>x0-.8)&(x<x0+1.4)&(abs(y-y0)<1.2)&~collision
    k=int((yaw%(2*math.pi))/m.info.yaw_resolution)
    valid=finite[k]&patch
    summary=dict(scope=__doc__,source_run=str(a.run.resolve()),snapshot=str(a.snapshot.resolve()),
        provenance=provenance,selected_height_replay_difference_m=delta,gpu_max_difference=gpu_error,
        unchanged=['posterior mean','posterior covariance envelope','support domain','body geometry','risk budgets'],
        eligible_poses=int(finite.sum()),old_admissible_poses=int((old<=.05).sum()),new_admissible_poses=int((new<=.05).sum()),
        start_patch=dict(heading_rad=yaw,evaluable=int(valid.sum()),old_above_005=int((old[k][valid]>.05).sum()),
            new_above_005=int((new[k][valid]>.05).sum())),
        counterfactual_map_sha256=digest((counterfactual/'map.cdr').read_bytes()),
        original_map_sha256=digest(raw),source_sha256={str(Path(__file__)):digest(Path(__file__).read_bytes())})
    np.savez_compressed(a.output/'bounds.npz',old=old,new=new,x=x,y=y,collision=collision)
    (a.output/'qualification.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
