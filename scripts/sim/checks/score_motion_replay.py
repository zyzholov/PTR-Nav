#!/usr/bin/env python3
"""Assess recorded motion masks using truth only after the detector has run."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def interpolate_pose(rows, stamp):
    values=np.asarray(rows,float)
    if values.ndim!=2 or values.shape[1]!=4 or not np.isfinite(values).all():
        raise ValueError('expected finite timestamped planar poses')
    if np.any(np.diff(values[:,0])<=0) or stamp<values[0,0] or stamp>values[-1,0]:
        raise ValueError('pose time is not strictly increasing or does not cover this scan')
    return np.array([np.interp(stamp,values[:,0],values[:,1]),
                     np.interp(stamp,values[:,0],values[:,2]),
                     np.interp(stamp,values[:,0],np.unwrap(values[:,3]))])


def world_to_map(point, robot_truth, robot_odom):
    angle=robot_odom[2]-robot_truth[2]
    c,s=np.cos(angle),np.sin(angle)
    return np.array([[c,-s],[s,c]])@(np.asarray(point)-robot_truth[:2])+robot_odom[:2]


def mask_points(sample, key):
    width,height,x,y,resolution=sample['geometry']
    ids=np.asarray(sample[key],int)
    if width<1 or height<1 or not resolution>0 or np.any(ids<0) or np.any(ids>=width*height):
        raise ValueError('invalid mask geometry')
    return np.column_stack((x+(ids%width+.5)*resolution,y+(ids//width+.5)*resolution))


def circle_hit(points, center, radius):
    return bool(len(points) and np.any(np.linalg.norm(points-center,axis=1)<=radius))


def score(capture,replay,radius=.65):
    if not capture.get('accepted') or capture.get('overflow') or len(capture['frames'])!=len(replay['frames']):
        raise ValueError('capture incomplete or replay frame count differs')
    obstacle=np.asarray(capture['obstacle_truth_assessment_only'],float)
    if obstacle.ndim!=2 or obstacle.shape[1]!=3 or not np.isfinite(obstacle).all() or np.any(np.diff(obstacle[:,0])<=0):
        raise ValueError('invalid assessment-only obstacle truth')
    rows=[]
    for raw,prediction in zip(capture['frames'],replay['frames']):
        t=raw['stamp_s']
        if t!=prediction['stamp_s'] or raw['geometry']!=prediction['geometry']:
            raise ValueError('replay changed scan timing or geometry')
        if t<obstacle[0,0] or t+.1>obstacle[-1,0]:continue
        truth=interpolate_pose(capture['robot_truth_assessment_only'],t)
        odom=interpolate_pose(capture['robot_odom_assessment_only'],t)
        def location(at):
            return world_to_map([np.interp(at,obstacle[:,0],obstacle[:,1]),
                                 np.interp(at,obstacle[:,0],obstacle[:,2])],truth,odom)
        center=location(t)
        speed=float(np.linalg.norm(location(t+.1)-center)/.1)
        forecast=mask_points(prediction,'forecast')
        raw_points=mask_points(raw,'occupied')
        # Cell-center tolerance accounts for rasterization, not robot clearance.
        tolerance=np.sqrt(2)*raw['geometry'][4]/2
        row={'stamp_s':t,'elapsed_s':t-capture['trigger_stamp_s'],'truth_y_m':float(center[1]),
             'moving_truth':speed>.5,'raw_hit':circle_hit(raw_points,center,radius+tolerance),
             'moving_components':prediction['moving_components'],
             'forecast_now_hit':circle_hit(forecast,center,radius+tolerance),
             'lane_warned':bool(len(forecast) and np.any((np.abs(forecast[:,0]-center[0])<=radius+tolerance)&
                                  (np.abs(forecast[:,1])<=.5+tolerance))),
             'compute_ms':prediction['compute_ms']}
        for horizon in (.5,1.):
            row[f'forecast_{horizon:g}s_hit']=(circle_hit(forecast,location(t+horizon),radius+tolerance)
                                             if t+horizon<=obstacle[-1,0] else None)
        rows.append(row)
    moving=[row for row in rows if row['moving_truth']]
    lane=[row for row in moving if abs(row['truth_y_m'])<=radius+.5]
    if not moving or not lane:raise ValueError('no moving crossing or no lane exposure')
    summary={'scope':'stationary measured-mask replay, not navigation or collision acceptance',
             'moving_scan_count':len(moving),'lane_exposure_scan_count':len(lane),
             'raw_hit_fraction_while_moving':float(np.mean([r['raw_hit'] for r in moving])),
             'lane_warning_fraction_during_exposure':float(np.mean([r['lane_warned'] for r in lane])),
             'forecast_now_hit_fraction_while_moving':float(np.mean([r['forecast_now_hit'] for r in moving])),
             'compute_ms_p95':float(np.percentile([r['compute_ms'] for r in rows],95)),
             'compute_ms_max':float(max(r['compute_ms'] for r in rows)),
             'pre_trigger_forecast_scan_count':sum(r['moving_components']>0 for r in rows if r['elapsed_s']<0),
             'late_stationary_forecast_scan_count':sum(r['moving_components']>0 for r in rows
                  if r['elapsed_s']>3.2),
             'motion_guard_sha256':replay['motion_guard_sha256']}
    for horizon in (.5,1.):
        values=[r[f'forecast_{horizon:g}s_hit'] for r in moving if r[f'forecast_{horizon:g}s_hit'] is not None]
        summary[f'forecast_{horizon:g}s_endpoint_hit_fraction']=float(np.mean(values))
    return {'summary':summary,'frames':rows}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture',type=Path);parser.add_argument('replay',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    if args.output.exists():parser.error('output must not exist')
    result=score(json.loads(args.capture.read_text()),json.loads(args.replay.read_text()))
    result['inputs_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.capture,args.replay)}
    with args.output.open('x') as output:json.dump(result,output,indent=2,allow_nan=False)
    print(json.dumps(result['summary'],ensure_ascii=False,allow_nan=False))


if __name__=='__main__':main()
