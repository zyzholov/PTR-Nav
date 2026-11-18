#!/usr/bin/env python3
"""Source-frozen classification replay, never a navigation or calibration test.

Reproduce the archived fine geometry first. Change only obstacle subcell bits
and their XY bounds; verify all terrain/probability/other fields are identical
after serialized round-trip. Authored scene geometry is not an input.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np
from rclpy.serialization import deserialize_message, serialize_message
from se2_grid_msgs.msg import SE2Grid
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import read_points
from fsgp_bgk_se2.obstacle_columns import (
    measured_collision_floor_witness, collision_subcells, inspect_body_collision)
from fsgp_bgk_se2.overhead_clearance import clip_overhead
from fsgp_bgk_se2.se2_grid_tools import (
    get_layer_slices, grid_buffer_to_world, world_to_grid_buffer,
    layer_refs, matrix_to_multiarray_like)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('snapshot',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pose',type=float,nargs=3,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('output exists')
    data=(a.snapshot/'map.cdr').read_bytes();g=deserialize_message(data,SE2Grid)
    source=deserialize_message(data,SE2Grid)
    def layer(n):return grid_buffer_to_world(get_layer_slices(source,n)[0][0],source)
    res=g.info.pos_resolution
    origin=np.array([g.info.pose.position.x-g.info.length_x/2,g.info.pose.position.y-g.info.length_y/2])
    cloud=(a.snapshot/'completion_cloud.cdr').read_bytes()
    raw=read_points(deserialize_message(cloud,PointCloud2),field_names=('x','y','z'),skip_nans=True)
    points=clip_overhead(np.c_[raw['x'],raw['y'],raw['z']],layer('collision_ceiling_elevation'),origin,res)
    occ=layer('collision_risk')>.5;z=layer('gp_elevation_observed');v=layer('gp_variance_observed')
    obs=layer('terrain_observed')>.5
    floor=np.where(obs&np.isfinite(v)&(v>=0)&(v<=.02**2),z,np.nan)
    old,oldbounds=collision_subcells(points,occ,origin,res,measured_floor=floor,factor=4,return_bounds=True)
    np.testing.assert_array_equal(old,layer('collision_subcell_bits'))
    np.testing.assert_array_equal(oldbounds,np.stack([layer(f'collision_subcell_xy_bounds_{k}') for k in range(16)]))
    times=[]
    for _ in range(8):
        t=perf_counter();witness,gradient=measured_collision_floor_witness(z,v,obs,occ,res)
        times.append((perf_counter()-t)*1000)
    new,bounds=collision_subcells(points,occ,origin,res,measured_floor=witness,
        measured_floor_gradient=gradient,factor=4,return_bounds=True)
    np.testing.assert_array_equal(new>0,old>0)
    body=[float(layer(n).flat[0]) for n in ('collision_body_front','collision_body_back','collision_body_half_width')]
    args=(origin,res,*a.pose,*body)
    before=inspect_body_collision(occ,old,*args,factor=4,bounds=oldbounds)
    after=inspect_body_collision(occ,new,*args,factor=4,bounds=bounds)
    fields={'collision_subcell_bits':new,**{f'collision_subcell_xy_bounds_{k}':b for k,b in enumerate(bounds)}}
    refs=layer_refs(g)
    for name,values in fields.items():
        ref=refs[name]
        if ref.has_so2 or len(ref.data_indices)!=1:raise ValueError('expected scalar fine geometry')
        index=ref.data_indices[0] # Layer names and flattened data are DIFFERENT indices.
        g.data[index]=matrix_to_multiarray_like(world_to_grid_buffer(values.astype(np.float32),g),g.data[index])
    derived=serialize_message(g);roundtrip=deserialize_message(derived,SE2Grid)
    modified_indices={refs[n].data_indices[0] for n in fields}
    for i,(original,actual) in enumerate(zip(source.data,roundtrip.data)):
        if i not in modified_indices:
            if serialize_message(original)!=serialize_message(actual):raise AssertionError(f'unrelated data changed: {i}')
    for name,values in fields.items():
        np.testing.assert_array_equal(grid_buffer_to_world(get_layer_slices(roundtrip,name)[0][0],roundtrip),values)
    a.output.mkdir(parents=True);target=a.output/'counterfactual';target.mkdir()
    (target/'map.cdr').write_bytes(derived)
    for name in ('odom.cdr','completion_cloud.cdr','path.cdr','rejection.json'):
        if (a.snapshot/name).exists():shutil.copyfile(a.snapshot/name,target/name)
    sha=lambda b:hashlib.sha256(b).hexdigest()
    report=dict(scope=__doc__,source=str(a.snapshot.resolve()),source_map_sha256=sha(data),
        source_cloud_sha256=sha(cloud),counterfactual_map_sha256=sha(derived),
        source_code_sha256={str(f):sha(f.read_bytes()) for f in
            (Path(__file__),Path('src/fsgp_bgk_se2/fsgp_bgk_se2/obstacle_columns.py'))},
        original_geometry_reproduced=True,all_other_fields_bitwise_unchanged=True,
        original_occupied_columns=int(occ.sum()),coarse_occupancy_unchanged=True,
        changed_fine_cells=int((old!=new).sum()),new_plane_reference_cells=int((~np.isfinite(floor)&np.isfinite(witness)).sum()),
        pose=a.pose,protected_body_front_back_halfwidth_m=body,
        body_before=[bool(v) for v in before],body_after=[bool(v) for v in after],
        floor_classification_median_ms=float(np.median(times)),
        timing_scope='eight read-only calls on one frame; not runtime latency or a success-rate experiment')
    (a.output/'qualification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
