#!/usr/bin/env python3
"""Reconstruct recorded collision witnesses before refining ONLY their XY raster.

No truth or live navigation input. Coarse occupancy, fine bits, 2cm witness
padding, protected body, terrain field and budgets must remain unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from rclpy.serialization import deserialize_message,serialize_message
from se2_grid_msgs.msg import SE2Grid
from sensor_msgs.msg import PointCloud2
from fsgp_bgk_se2.fsgp_bgk_se2_node import FsgpBgkSe2Node
from fsgp_bgk_se2.overhead_clearance import clip_overhead
from fsgp_bgk_se2.obstacle_columns import collision_subcells,inspect_body_collision
from fsgp_bgk_se2.se2_grid_tools import get_layer_slices,grid_buffer_to_world,world_to_grid_buffer,matrix_to_multiarray_like


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('snapshot',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('output exists')
    digest=lambda b:hashlib.sha256(b).hexdigest()
    raw=(a.snapshot/'map.cdr').read_bytes();cloud=(a.snapshot/'completion_cloud.cdr').read_bytes()
    d=json.loads((a.snapshot/'diagnostic.json').read_text())
    if digest(raw)!=d['map_sha256'] or digest(cloud)!=d['completion_cloud']['sha256']:raise ValueError('source hash mismatch')
    metadata=json.loads((a.snapshot.parent/'run_metadata.json').read_text())
    if metadata['arguments']['obstacle_memory_role']!='advisory':raise ValueError('hard history requires its original source cloud')
    m=deserialize_message(raw,SE2Grid)
    if 'collision_subcell_xy_bounds_contract' in m.layers:raise ValueError('requires an archived whole-subcell source')
    def layer(n):return grid_buffer_to_world(get_layer_slices(m,n)[0][0],m)
    res=m.info.pos_resolution;origin=(m.info.pose.position.x-m.info.length_x/2,m.info.pose.position.y-m.info.length_y/2)
    points=FsgpBgkSe2Node._decode_points(deserialize_message(cloud,PointCloud2))
    points=clip_overhead(points,layer('collision_ceiling_elevation'),origin,res)
    v=layer('gp_variance_observed')
    floor=np.where((layer('terrain_observed')>.5)&np.isfinite(v)&(v>=0)&(v<=.02**2),layer('gp_elevation_observed'),np.nan)
    bits,bounds=collision_subcells(points,layer('collision_risk')>.5,origin,res,measured_floor=floor,
        factor=4,return_bounds=True)
    if not np.array_equal(bits,layer('collision_subcell_bits')):raise ValueError('original occupancy witness reconstruction mismatch')
    x,y,z,yaw=d['pose'];body=[float(layer(n).flat[0]) for n in ('collision_body_front','collision_body_back','collision_body_half_width')]
    args=(layer('collision_risk'),bits,origin,res,x,y,yaw,*body)
    original=inspect_body_collision(*args,factor=4);candidate=inspect_body_collision(*args,factor=4,bounds=bounds)
    a.output.mkdir();dest=a.output/'candidate';dest.mkdir()
    template=m.data[0]
    for name,values in [('collision_subcell_xy_bounds_contract',np.ones_like(bits)),
                        *[(f'collision_subcell_xy_bounds_{k}',b) for k,b in enumerate(bounds)]]:
        m.layers.append(name);m.has_so2.append(False)
        m.data.append(matrix_to_multiarray_like(world_to_grid_buffer(values.astype(np.float32),m),template))
    payload=serialize_message(m);(dest/'map.cdr').write_bytes(payload)
    for name in ('odom.cdr','completion_cloud.cdr'):shutil.copyfile(a.snapshot/name,dest/name)
    d['map_sha256']=digest(payload);d['scope']='counterfactual XY representation only; not physical execution'
    (dest/'diagnostic.json').write_text(json.dumps(d,indent=2))
    nav=json.loads((a.snapshot.parent/'navigation_data.json').read_text());goal=nav['requested_goal']
    queries=dict(poses=[[x,y,yaw]],recorded_external_goal=[goal['x'],goal['y']],seed_diagnosis=True,recovery_diagnosis=True)
    (a.output/'queries.json').write_text(json.dumps(queries))
    report=dict(scope=__doc__,source=str(a.snapshot.resolve()),source_map_sha256=digest(raw),candidate_map_sha256=digest(payload),
        source_cloud_sha256=digest(cloud),original_witness_reconstruction_exact=True,unchanged_body_front_back_halfwidth_m=body,
        unchanged_xy_padding_m=.02,quantization_outward_error_max_m=res/4/15,
        original_body_inspection=original,candidate_body_inspection=candidate,
        source_code_sha256={str(Path(__file__)):digest(Path(__file__).read_bytes())})
    (a.output/'qualification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
