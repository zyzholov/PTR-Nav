#!/usr/bin/env python3
"""Derived collision-geometry candidate on a retained snapshot; NOT physical replay.

Only the three optional geometry layers are added. Original height, occupancy,
probability, source stamps, goals and point coordinates are unchanged. Requires
advisory history: a partial current scan cannot refine hard historical hazards.
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
from fsgp_bgk_se2.obstacle_columns import collision_subcells
from fsgp_bgk_se2.overhead_clearance import clip_overhead
from fsgp_bgk_se2.se2_grid_tools import get_layer_slices,grid_buffer_to_world,world_to_grid_buffer,matrix_to_multiarray_like


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();meta=json.loads((a.run/'run_metadata.json').read_text())
    if meta['arguments']['obstacle_memory_role']!='advisory':raise ValueError('hard history cannot be refined from one partial scan')
    root=a.run/'probability_after_trial';payload=(root/'map.cdr').read_bytes()
    msg=deserialize_message(payload,SE2Grid);cloud=deserialize_message((root/'completion_cloud.cdr').read_bytes(),PointCloud2)
    stamp=lambda h:h.stamp.sec*10**9+h.stamp.nanosec
    if msg.info.header.frame_id!=cloud.header.frame_id or not 0<=stamp(msg.info.header)-stamp(cloud.header)<=150000000:
        raise ValueError('current scan is missing, future, stale or in another frame')
    def layer(name):
        slices,_,_=get_layer_slices(msg,name);return grid_buffer_to_world(slices[0],msg)
    origin=(msg.info.pose.position.x-msg.info.length_x/2,msg.info.pose.position.y-msg.info.length_y/2)
    points=FsgpBgkSe2Node._decode_points(cloud)
    points=clip_overhead(points,layer('collision_ceiling_elevation'),origin,msg.info.pos_resolution)
    occupied=layer('collision_risk')>=.5
    bits=collision_subcells(points,occupied,origin,msg.info.pos_resolution)
    for name,values in (('collision_subcell_bits',bits.astype(np.float32)),
                         ('collision_subcell_factor',np.full(bits.shape,2.,np.float32)),
                         ('collision_subcell_contract',np.full(bits.shape,1.,np.float32))):
        if name in msg.layers:raise ValueError('source is already refined')
        msg.layers.append(name);msg.has_so2.append(False)
        msg.data.append(matrix_to_multiarray_like(world_to_grid_buffer(values,msg),msg.data[0]))
    a.output.mkdir(parents=True,exist_ok=False)
    shutil.copytree(root,a.output/'probability_after_trial')
    shutil.copy2(a.run/'navigation_data.json',a.output/'navigation_data.json')
    derived=serialize_message(msg);(a.output/'probability_after_trial/map.cdr').write_bytes(derived)
    record=dict(scope=__doc__,source_run=str(a.run.resolve()),source_map_sha256=hashlib.sha256(payload).hexdigest(),
        candidate_map_sha256=hashlib.sha256(derived).hexdigest(),source_cloud_sha256=hashlib.sha256((root/'completion_cloud.cdr').read_bytes()).hexdigest(),
        coarse_occupied=int(occupied.sum()),refined_occupied=int(((bits>0)&(bits<15)).sum()),
        source_stamp_ns=stamp(msg.info.header),geometry_resolution_m=msg.info.pos_resolution/2.)
    (a.output/'candidate_provenance.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))


if __name__=='__main__':main()
