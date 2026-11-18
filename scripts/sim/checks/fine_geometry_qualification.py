#!/usr/bin/env python3
"""Reconstruct old occupancy witnesses, then vary ONLY their raster resolution.

No truth, reference route or commands. The candidate CDR is for read-only
native classification, never a simulator/navigation input or success claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from rclpy.serialization import deserialize_message,serialize_message
from se2_grid_msgs.msg import SE2Grid
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import read_points
from fsgp_bgk_se2.se2_grid_tools import (get_layer_slices,grid_buffer_to_world,world_to_grid_buffer,
    matrix_to_multiarray_like,layer_refs)
from fsgp_bgk_se2.obstacle_columns import collision_subcells,collision_pose_masks,inspect_body_collision
from fsgp_bgk_se2.overhead_clearance import clip_overhead


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('snapshot',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('output exists')
    s=a.snapshot;diagnostic=json.loads((s/'diagnostic.json').read_text())
    msg=deserialize_message((s/'map.cdr').read_bytes(),SE2Grid)
    cloud=deserialize_message((s/'completion_cloud.cdr').read_bytes(),PointCloud2)
    if hashlib.sha256((s/'map.cdr').read_bytes()).hexdigest()!=diagnostic['map_sha256'] or \
       hashlib.sha256((s/'completion_cloud.cdr').read_bytes()).hexdigest()!=diagnostic['completion_cloud']['sha256']:
        raise ValueError('source hash mismatch')
    def layer(name,k=0):return grid_buffer_to_world(get_layer_slices(msg,name)[0][k],msg)
    if float(layer('collision_subcell_factor').flat[0])!=2:raise ValueError('requires archived 10cm/2x source')
    metadata=json.loads((s.parent/'run_metadata.json').read_text())
    if metadata['arguments']['obstacle_memory_role']!='advisory':raise ValueError('hard memory reconstruction requires its original CDR too')
    origin=(msg.info.pose.position.x-msg.info.length_x/2,msg.info.pose.position.y-msg.info.length_y/2)
    res=msg.info.pos_resolution
    pts=read_points(cloud,field_names=('x','y','z'));pts=np.c_[pts['x'],pts['y'],pts['z']]
    pts=clip_overhead(pts,layer('collision_ceiling_elevation'),origin,res)
    occupied=layer('collision_risk')>=.5;variance=layer('gp_variance_observed')
    floor=np.where((layer('terrain_observed')>.5)&np.isfinite(variance)&(variance>=0)&(variance<=.02**2),layer('gp_elevation_observed'),np.nan)
    step=float(layer('probability_max_step').flat[0])
    old=collision_subcells(pts,occupied,origin,res,measured_floor=floor,max_step_m=step)
    mismatches=int(np.count_nonzero(old!=layer('collision_subcell_bits')))
    if mismatches:raise ValueError(f'old witness reconstruction mismatch: {mismatches} cells')
    fine=collision_subcells(pts,occupied,origin,res,measured_floor=floor,max_step_m=step,factor=4)
    x,y,z,yaw=diagnostic['pose'];body=[float(layer(n).flat[0]) for n in ('collision_body_front','collision_body_back','collision_body_half_width')]
    checks={str(f):inspect_body_collision(occupied,bits,origin,res,x,y,yaw,*body,factor=f) for f,bits in ((2,old),(4,fine))}
    refs=layer_refs(msg)
    def replace(name,matrix,k=0):
        index=refs[name].data_indices[k]
        msg.data[index]=matrix_to_multiarray_like(world_to_grid_buffer(matrix,msg),msg.data[index])
    replace('collision_subcell_factor',np.full_like(fine,4,dtype=np.float32))
    replace('collision_subcell_bits',fine.astype(np.float32))
    bins=round(2*np.pi/msg.info.yaw_resolution)
    masks=collision_pose_masks(fine,res,*body,bins,factor=4,backend='cupy')
    for k in range(bins):
        upper=layer('terrain_failure_probability_upper',k).copy();upper[masks[k]]=1.
        replace('failure_probability_upper',upper,k)
    a.output.mkdir(parents=True)
    for name in ('odom.cdr','completion_cloud.cdr'):(a.output/name).write_bytes((s/name).read_bytes())
    payload=serialize_message(msg);(a.output/'map.cdr').write_bytes(payload)
    queries=[[x+.05*i*np.cos(yaw),y+.05*i*np.sin(yaw),yaw] for i in range(9)]
    (a.output/'queries.json').write_text(json.dumps(queries)+'\n')
    record=dict(scope=__doc__,source=str(s.resolve()),source_map_sha256=diagnostic['map_sha256'],
        candidate_map_sha256=hashlib.sha256(payload).hexdigest(),original_reconstruction_mismatch_cells=mismatches,
        coarse_occupied_cells=int(occupied.sum()),original_body_envelope_m=body,xy_padding_m=.02,
        current_pose_clearance_inspection=checks,pose=diagnostic['pose'],
        changed_layers=['collision_subcell_factor','collision_subcell_bits','failure_probability_upper raster overlay'],
        unchanged='all terrain means/covariances/support/state-path-braking limits, original coarse occupancy and physical margins')
    (a.output/'geometry_qualification.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)


if __name__=='__main__':main()
