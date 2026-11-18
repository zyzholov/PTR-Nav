#!/usr/bin/env python3
"""Replay saved real inputs to audit the tilted-sensor startup certificate.

This reconstructs a stance-plane diagnostic from a captured snapshot. It does
not replay the stability history and is NOT navigation/stability qualification.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from rclpy.serialization import deserialize_message
from se2_grid_msgs.msg import SE2Grid
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from fsgp_bgk_se2.se2_grid_tools import layer_refs,multiarray_to_matrix,grid_buffer_to_world
from fsgp_bgk_se2.startup_support import StartupSupport,Stance
from fsgp_bgk_se2.probabilistic_risk import ProbabilityParams
from fsgp_bgk_se2.shared_terrain import SharedTerrainParams,shared_terrain_prediction,shared_geometric_probability
from mid360_scan_probe import xyz_array


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();p=args.input
    grid=deserialize_message((p/'map.cdr').read_bytes(),SE2Grid)
    odom=deserialize_message((p/'odom.cdr').read_bytes(),Odometry)
    cloud=deserialize_message((p/'completion_cloud.cdr').read_bytes(),PointCloud2)
    refs=layer_refs(grid)
    def a(n):return grid_buffer_to_world(multiarray_to_matrix(grid.data[refs[n].data_indices[0]]),grid)
    h=a('gp_elevation_observed');v=a('gp_variance_observed');seen=a('ground_observed')>.5
    hit=a('collision_risk')>=.5;resolution=float(grid.info.pos_resolution)
    x,y=np.meshgrid(grid.info.pose.position.x-grid.info.length_x/2+(np.arange(h.shape[1])+.5)*resolution,
                    grid.info.pose.position.y-grid.info.length_y/2+(np.arange(h.shape[0])+.5)*resolution)
    q=odom.pose.pose.orientation;z=odom.pose.pose.position
    roll=np.arctan2(2*(q.w*q.x+q.y*q.z),1-2*(q.x*q.x+q.y*q.y))
    pitch=np.arcsin(2*(q.w*q.y-q.z*q.x));yaw=np.arctan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
    t=grid.info.header.stamp.sec+grid.info.header.stamp.nanosec*1e-9
    params=ProbabilityParams(front=.42,back=.42,half_width=.313,margin=.1,yaw_bins=36,internal_steps=True)
    model=StartupSupport(allow_one_sided=True)
    for stamp in (t-2,t):
        _,mean,_=model.update(h,v,seen,hit,x,y,Stance(z.x,z.y,yaw,roll,pitch,0,stamp,z.z),stamp,params)
    prior=a('startup_prior_used')>.5
    prediction=shared_terrain_prediction(h,v,seen,x,y,xyz_array(cloud),0.,resolution,hit,device='cuda',
        params=SharedTerrainParams(max_distance_m=1.25,max_extrapolation_m=.75),
        initial_support=prior,initial_height=mean,positive_veto_height=None)
    consistent=prior&~hit&~prediction.contradicted&np.isfinite(prediction.mean)&(np.abs(prediction.mean-mean)<=.05)
    r=int((z.y-(grid.info.pose.position.y-grid.info.length_y/2))/resolution)
    c=int((z.x-(grid.info.pose.position.x-grid.info.length_x/2))/resolution)
    records=[]
    for limit in (.04,.08,.12):
        admitted=consistent&(prediction.variance<=limit**2)
        candidate=replace(prediction,valid=prediction.valid|admitted)
        out=shared_geometric_probability(candidate,params,hit)
        k=int((yaw%(2*np.pi))/(2*np.pi/params.yaw_bins))
        rr,cc,values=out[k];lookup={(int(a),int(b)):float(v) for a,b,v in zip(rr,cc,values)}
        row=dict(initial_absolute_std_limit_m=limit,initial_admitted=int(admitted.sum()),initial_total=int(prior.sum()),
                 pose_upper=[lookup.get((r,c+j)) for j in range(8)],
                 initial_std_max=float(np.sqrt(prediction.variance[prior]).max()))
        records.append(row);print(json.dumps(row),flush=True)
    args.output.write_text(json.dumps(dict(scope='snapshot covariance diagnostic only; stability not replayed',records=records),indent=2)+'\n')


if __name__=='__main__':main()
