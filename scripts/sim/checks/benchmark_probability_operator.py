#!/usr/bin/env python3
"""Pair two CUDA evaluations on the same reconditioned frozen measurements.

This is an operator regression/timing check, not navigation acceptance, terrain
accuracy, an archived posterior replay, or a comparison against another method.
No initial stance is reconstructed. Runtime inputs and source hashes are saved.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2
from se2_grid_msgs.msg import SE2Grid
from fsgp_bgk_se2.se2_grid_tools import get_layer_slices,grid_buffer_to_world,yaw_slices
from fsgp_bgk_se2.fsgp_bgk_se2_node import FsgpBgkSe2Node
from fsgp_bgk_se2.overhead_clearance import clip_overhead
from fsgp_bgk_se2.shared_terrain import shared_terrain_prediction,SharedTerrainParams
from fsgp_bgk_se2.probabilistic_risk import ProbabilityParams
from fsgp_bgk_se2.shared_terrain_cuda import shared_probability_cuda


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_dir',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--repeats',type=int,default=20)
    parser.add_argument('--support-distance',type=float,default=1.25)
    args=parser.parse_args()
    if args.output.exists() or not 3<=args.repeats<=100:parser.error('use a new output and 3..100 paired repeats')
    base=args.input_dir;inputs={n:(base/n).read_bytes() for n in ('map.cdr','completion_cloud.cdr')}
    msg=deserialize_message(inputs['map.cdr'],SE2Grid)
    cloud=deserialize_message(inputs['completion_cloud.cdr'],PointCloud2)
    if cloud.header.frame_id!=msg.info.header.frame_id:raise ValueError('different source frames')
    def layer(name):return grid_buffer_to_world(get_layer_slices(msg,name)[0][0],msg)
    def scalar(name):
        v=layer(name)
        if not np.isfinite(v).all() or not (v==v.flat[0]).all():raise ValueError('invalid scalar metadata '+name)
        return float(v.flat[0])
    z=layer('gp_elevation_observed');var=layer('gp_variance_observed');seen=layer('terrain_observed')>=.5
    r=msg.info.pos_resolution;origin=(msg.info.pose.position.x-msg.info.length_x/2,msg.info.pose.position.y-msg.info.length_y/2)
    x,y=np.meshgrid(origin[0]+(np.arange(z.shape[1])+.5)*r,origin[1]+(np.arange(z.shape[0])+.5)*r)
    raw=clip_overhead(FsgpBgkSe2Node._decode_points(cloud),layer('collision_ceiling_elevation'),origin,r)
    blocked=(layer('collision_risk')>=.5)|(layer('probability_raw_veto')>=.5)
    age=((msg.info.header.stamp.sec-cloud.header.stamp.sec)*1_000_000_000+
         msg.info.header.stamp.nanosec-cloud.header.stamp.nanosec)*1e-9
    torch.set_num_threads(1)
    prediction=shared_terrain_prediction(z,var,seen,x,y,raw,age,r,blocked,device='cuda',
        params=SharedTerrainParams(max_distance_m=args.support_distance),positive_veto_height=None)
    if prediction is None or not prediction.valid.any():raise ValueError('no eligible current posterior')
    params=ProbabilityParams(resolution=r,yaw_bins=yaw_slices(msg),internal_steps=True,margin=.1,
        front=scalar('probability_front')-.1,back=scalar('probability_back')-.1,
        half_width=scalar('probability_half_width')-.1,max_pitch=scalar('probability_max_pitch'),
        max_roll=scalar('probability_max_roll'),max_step=scalar('probability_max_step'),
        height_std_floor=scalar('probability_height_std_floor'),variance_scale=scalar('probability_variance_scale'))
    actual=shared_probability_cuda(prediction,params,blocked)
    reference=shared_probability_cuda(prediction,params,blocked,factorized=False)
    errors=[]
    for a,b in zip(actual,reference):
        np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
        np.testing.assert_allclose(a[2],b[2],atol=2e-6)
        if len(a[2]):errors.append(float(np.max(np.abs(a[2]-b[2]))))
    if not errors:raise ValueError('no eligible pose contrasts')
    times={'original':[],'factored':[]}
    for i in range(args.repeats+3):
        order=[(False,'original'),(True,'factored')]
        for flag,name in order if i%2 else reversed(order):
            t=time.perf_counter();shared_probability_cuda(prediction,params,blocked,factorized=flag)
            if i>=3:times[name].append((time.perf_counter()-t)*1000)
    root=Path(__file__).resolve().parents[3]
    code=[Path(__file__),*[root/'src/fsgp_bgk_se2/fsgp_bgk_se2'/n for n in
        ('shared_terrain.py','shared_terrain_cuda.py','probabilistic_risk.py','se2_grid_tools.py','overhead_clearance.py')]]
    result=dict(scope=__doc__,arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        shape=list(z.shape),valid_cells=int(prediction.valid.sum()),source_age_s=age,
        maximum_absolute_probability_difference=max(errors),times_ms=times,
        percentiles_ms={k:np.percentile(v,[50,95,100]).tolist() for k,v in times.items()},
        input_sha256={n:hashlib.sha256(b).hexdigest() for n,b in inputs.items()},
        code_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in code},
        gpu=torch.cuda.get_device_name(),torch_version=str(torch.__version__))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as output:json.dump(result,output,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('percentiles_ms','maximum_absolute_probability_difference','valid_cells')}))


if __name__=='__main__':main()
