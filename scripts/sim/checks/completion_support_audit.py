#!/usr/bin/env python3
"""Replay missing footprint supports on one frozen frame, not a navigation test."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from rclpy.serialization import deserialize_message
from se2_grid_msgs.msg import SE2Grid
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import read_points

from fsgp_bgk_se2.feature_sgp import FeatureSgpParams, FeatureSgpPredictor
from fsgp_bgk_se2.fresh_completion import fresh_local_prediction
from fsgp_bgk_se2.probabilistic_risk import ProbabilityParams, _geometry
from fsgp_bgk_se2.se2_grid_tools import get_layer_2d, grid_buffer_to_world, matrix_shape


def finite(value):
    if isinstance(value,dict):return {k:finite(v) for k,v in value.items()}
    if isinstance(value,list):return [finite(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--ahead-m',type=float,default=0.,help='Probe a nearby pose, without skipping the intervening path')
    parser.add_argument('--distances',type=float,nargs='+',default=[.35],help='Diagnostic raw-support limits; does not change runtime configuration')
    parser.add_argument('--neighbors',type=int,nargs='+',default=[20,32,48,64])
    args=parser.parse_args()
    if (not math.isfinite(args.ahead_m) or any(not math.isfinite(v) or not 0<v<=.75 for v in args.distances)
            or any(not 10<=v<=128 for v in args.neighbors)):
        parser.error('use finite offsets, support distances in (0,.75], and 10..128 neighbors')
    payload=(args.input/'map.cdr').read_bytes()
    cloud_payload=(args.input/'completion_cloud.cdr').read_bytes()
    msg=deserialize_message(payload,SE2Grid)
    cloud=deserialize_message(cloud_payload,PointCloud2)
    diagnostic=json.loads((args.input/'diagnostic.json').read_text())
    record=diagnostic['completion_cloud']
    if not record or hashlib.sha256(cloud_payload).hexdigest()!=record['sha256']:
        raise ValueError('not the archived timestamp-matched conditioning scan')
    age=record['source_age_s']
    map_ns=msg.info.header.stamp.sec*10**9+msg.info.header.stamp.nanosec
    cloud_ns=cloud.header.stamp.sec*10**9+cloud.header.stamp.nanosec
    if cloud.header.frame_id!=msg.info.header.frame_id or abs(map_ns-cloud_ns-round(age*1e9))>5:
        raise ValueError('scan frame or stamp does not match completion age')
    shape=matrix_shape(msg.data[0]);resolution=msg.info.pos_resolution
    def layer(name):
        return grid_buffer_to_world(get_layer_2d(msg,name,np.nan,shape)[0],msg)
    left=msg.info.pose.position.x-msg.info.length_x/2
    bottom=msg.info.pose.position.y-msg.info.length_y/2
    x,y=np.meshgrid(left+(np.arange(shape[1])+.5)*resolution,
                    bottom+(np.arange(shape[0])+.5)*resolution)
    pose=diagnostic['pose'].copy()
    pose[0]+=args.ahead_m*math.cos(pose[3]);pose[1]+=args.ahead_m*math.sin(pose[3])
    r=int((pose[1]-bottom)/resolution);c=int((pose[0]-left)/resolution)
    yaw_index=int((pose[3]%(2*math.pi))/msg.info.yaw_resolution)
    params=ProbabilityParams(resolution=resolution,
        yaw_bins=round(2*math.pi/msg.info.yaw_resolution),
        front=float(layer('probability_front').flat[0]),
        back=float(layer('probability_back').flat[0]),
        half_width=float(layer('probability_half_width').flat[0]),margin=0,internal_steps=True)
    height=layer('probability_elevation');variance=layer('probability_variance')
    missing=[]
    for dr,dc in _geometry(params,yaw_index)[0]:
        rr,cc=r+dr,c+dc
        if not 0<=rr<shape[0] or not 0<=cc<shape[1]:
            raise ValueError('footprint reaches outside recorded map; not an interior completion audit')
        if not np.isfinite(height[rr,cc]) or not np.isfinite(variance[rr,cc]):
            missing.append((rr,cc))
    rows=read_points(cloud,field_names=('x','y','z'))
    points=np.column_stack([rows[name] for name in ('x','y','z')])
    points=points[np.isfinite(points).all(axis=1)]
    cc=np.floor((points[:,0]-left)/resolution).astype(int)
    rr=np.floor((points[:,1]-bottom)/resolution).astype(int)
    inside=(rr>=0)&(rr<shape[0])&(cc>=0)&(cc<shape[1])
    points=points[inside];rr=rr[inside];cc=cc[inside]
    observed=layer('terrain_observed')>.5
    observed_h=layer('gp_elevation_observed');observed_v=layer('gp_variance_observed')
    usable=observed&np.isfinite(observed_h)&np.isfinite(observed_v)&(observed_v>=0)&(observed_v<=.02**2)
    tolerance=np.minimum(.05,np.maximum(.03,2*np.sqrt(np.maximum(observed_v,0))))
    agrees=usable[rr,cc]&(np.abs(points[:,2]-observed_h[rr,cc])<=tolerance[rr,cc])
    evidence=cKDTree(points[agrees,:2])
    blocked=np.ones(shape,bool)
    for rr,cc in missing:blocked[rr,cc]=layer('collision_risk')[rr,cc]>=.5
    result=dict(scope='one recorded frame; neighbor diagnostic, not physical acceptance',
        map_sha256=hashlib.sha256(payload).hexdigest(),cloud_sha256=record['sha256'],
        pose=pose,yaw_index=yaw_index,missing_xy=[[float(x[i,j]),float(y[i,j])] for i,j in missing],
        diagnostic_raw_support_limits_m=args.distances,
        raw_support_distance_m=[float(evidence.query((x[i,j],y[i,j]))[0]) for i,j in missing],
        variants=[])
    for count,limit in ((n,d) for n in args.neighbors for d in args.distances):
        predictor=FeatureSgpPredictor(FeatureSgpParams(device=args.device,training_steps=0,
            local_neighbors=count,local_nearest_m=limit))
        prediction=fresh_local_prediction(predictor,layer('gp_elevation_observed'),
            layer('gp_variance_observed'),layer('terrain_observed')>.5,x,y,points,
            record['source_age_s'],resolution,blocked=blocked)
        result['variants'].append(dict(neighbors=count,maximum_raw_support_distance_m=limit,training_points=prediction.training_points,
            supports=[dict(x=float(x[i,j]),y=float(y[i,j]),accepted=bool(prediction.accepted[i,j]),
                mean=float(prediction.mean[i,j]),sigma=float(np.sqrt(prediction.latent_var[i,j])),
                nearest_m=float(prediction.nearest_distance[i,j])) for i,j in missing]))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as output:json.dump(finite(result),output,indent=2,allow_nan=False)
    print(json.dumps(finite(result),indent=2,allow_nan=False))


if __name__=='__main__':main()
