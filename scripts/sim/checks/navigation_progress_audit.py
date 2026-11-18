#!/usr/bin/env python3
"""Offline entrance/progress qualification, separate from safe goal arrival.

Only the assessor reads fixture geometry and truth. No ROS publishers or
authored intermediate goals. This is a directed diagnostic, not a success-rate
experiment, and never rewrites the original navigation acceptance record.
"""
import argparse
import bisect
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src/isaac_sim_hunter_livo2/scripts'))
from boundary_courses import fixtures

DEFAULT_LIMITS=dict(first_path_s=6.,first_row_s=30.,approach_distance_m=8.)


def audit(record,limits=None):
    limits=dict(DEFAULT_LIMITS if limits is None else limits)
    if set(limits)!=set(DEFAULT_LIMITS) or any(not math.isfinite(v) or v<=0 for v in limits.values()):
        raise ValueError('finite positive, explicit entrance qualification limits required')
    if record['spec']['kind']!='dense':raise ValueError('this entrance assessor requires the dense fixture')
    requests=record['goal_requests']
    if len(requests)!=1:raise ValueError('qualification requires one external final goal')
    start=float(requests[0]['receipt_sim_stamp_s'])
    samples=record['truth'];columns=record['truth_columns']
    index={k:columns.index(k) for k in ('stamp_s','x','y','yaw')}
    selected=[r for r in samples if r[index['stamp_s']]>=start-1e-7]
    if len(selected)<2:raise ValueError('insufficient post-goal truth samples')
    if any(not all(math.isfinite(r[i])for i in index.values())for r in selected):
        raise ValueError('invalid truth sample')
    if any(b[index['stamp_s']]<a[index['stamp_s']] for a,b in zip(selected,selected[1:])):
        raise ValueError('nonmonotonic truth; cannot certify progress timing')
    row=[o for o in fixtures(record['spec'])if o['name'].startswith('Dense_0_')]
    if not row:raise ValueError('first-row fixture provenance missing')
    face=max(o['x']+o['radius']for o in row)
    body=record['summary']['evaluation_body_geometry']
    front,back,width=(float(body[k])for k in ('front','back','half_width'))
    first=selected[0];before=[];crossed=False
    for sample in selected:
        before.append(sample);yaw=sample[index['yaw']];c=math.cos(yaw)
        # Support function of the actual enclosing rectangle at the plane.
        rear_x=(back*c if c>=0 else -front*c)+width*abs(math.sin(yaw))
        if sample[index['x']]-rear_x>=face:
            crossed=True;break
    distance=sum(math.hypot(b[index['x']]-a[index['x']],b[index['y']]-a[index['y']])
        for a,b in zip(before,before[1:]))
    first_motion=next((r[index['stamp_s']]-start for r in selected
        if math.hypot(r[index['x']]-first[index['x']],r[index['y']]-first[index['y']])>=.10),None)
    paths=[p for p in record['path_history']if p[0]>=start-1e-7 and len(p[1])>=2]
    first_path=paths[0][0]-start if paths else None
    # Downsample only the diagnostic yaw variation, never navigation inputs.
    yaw_samples=[]
    for r in before:
        if not yaw_samples or r[index['stamp_s']]-yaw_samples[-1][index['stamp_s']]>=.1:
            yaw_samples.append(r)
    variation=sum(abs(math.atan2(math.sin(b[index['yaw']]-a[index['yaw']]),
        math.cos(b[index['yaw']]-a[index['yaw']])))for a,b in zip(yaw_samples,yaw_samples[1:]))
    away=False;revisits=0
    for r in before:
        separation=math.hypot(r[index['x']]-first[index['x']],r[index['y']]-first[index['y']])
        if separation>=1.:away=True
        elif away and separation<=.5:revisits+=1;away=False
    # Count proposed entrance-gap switches, not yellow lines as real driving.
    # One fixed initial map->truth alignment is assessor-only; not a collision
    # proof and not a route supplied to the planner. Short paths that do not
    # reach this plane are explicitly excluded, never extrapolated.
    origin=record['truth_start'];map_start=record.get('map_start')
    if map_start is None:
        pose=record['localization_poses'][0];cols=record['localization_pose_columns']
        map_start=[pose[cols.index(k)]for k in ('x','y','z','yaw')]
    angle=samples[0][index['yaw']]-map_start[3];ca,sa=math.cos(angle),math.sin(angle)
    def world(point):
        x,y=point[0]-map_start[0],point[1]-map_start[1]
        return origin[0]+ca*x-sa*y,origin[1]+sa*x+ca*y
    gaps=[];centres=sorted(o['y']for o in row)
    for stamp,points in paths:
        if stamp>before[-1][index['stamp_s']]:break
        transformed=[world(p)for p in points]
        for a,b in zip(transformed,transformed[1:]):
            if a[0]<face<=b[0] and b[0]>a[0]:
                y=a[1]+(b[1]-a[1])*(face-a[0])/(b[0]-a[0])
                gaps.append(dict(stamp_s=stamp,gap=bisect.bisect_left(centres,y),y_m=y));break
    first_row=before[-1][index['stamp_s']]-start if crossed else None
    failures=[]
    for name,value in [('first_path_s',first_path),('first_row_s',first_row),('approach_distance_m',distance)]:
        if value is None or value>limits[name]:failures.append(name)
    return dict(scope=__doc__,navigation_arrived=bool(record['summary']['reached']),
        navigation_settled=bool(record['summary']['settled']),entrance_quality_passed=not failures,
        limits=limits,failed_metrics=failures,goal_receipt_s=start,
        first_published_path_delay_s=first_path,first_10cm_motion_delay_s=first_motion,
        first_row_fully_cleared_s=first_row,approach_distance_m=distance,
        first_row_face_world_x_m=face,body_geometry_m=body,
        measured_yaw_variation_before_entry_rad=variation,start_area_revisits=revisits,
        plans_before_entry=sum(p[0]<=before[-1][index['stamp_s']]for p in paths),
        proposed_gap_switches=sum(a['gap']!=b['gap']for a,b in zip(gaps,gaps[1:])),
        entrance_gap_proposals=gaps,
        censoring='null first-row time means the entire body never cleared the first row; distance then covers the full mission',
        definitions=dict(first_path='first published nonempty path after the goal receipt; not a fresh certificate for every later command',
            first_row='entire uninflated physical enclosing body passes the far face of ALL first-row columns; no specific gap prescribed',
            approach_distance='actual truth XY path length before the above milestone, not the sum of planned yellow curves',
            gap_switches='proposals with a real forward crossing of the first-row plane, after a fixed initial frame alignment; diagnostic only'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('inputs',type=Path,nargs='+');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();rows=[]
    for path in a.inputs:rows.append(dict(input=str(path.resolve()),**audit(json.loads(path.read_text()))))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(dict(scope=__doc__,runs=rows),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    for r in rows:print({k:r[k]for k in ['input','navigation_arrived','entrance_quality_passed',
        'first_published_path_delay_s','first_row_fully_cleared_s','approach_distance_m','proposed_gap_switches']})
    return int(any(not r['entrance_quality_passed']for r in rows))


if __name__=='__main__':raise SystemExit(main())
