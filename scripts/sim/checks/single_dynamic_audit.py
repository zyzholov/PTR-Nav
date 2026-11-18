#!/usr/bin/env python3
"""Read-only action/encounter qualification. Truth never enters navigation.

This does not assert interaction with a cooperative agent. A scripted body
does not yield. Arrival and independent body clearance remain required.
"""
import json
import math
from collections import Counter
from pathlib import Path
import numpy as np
from score_motion_replay import interpolate_pose,mask_points
from boundary_courses import fixtures


def sustained_mode_spans(executed):
    spans={};mode=None;begin=last=None;count=0
    def retain():
        if count>=3 and last-begin>=.2-1e-8:spans[mode]=max(spans.get(mode,0.),last-begin)
    for action in executed:
        at=action['stamp_s'];current=action['mode']
        if current is None:
            retain();mode=None;begin=last=None;count=0
            continue
        if mode!=current or last is not None and not 0<at-last<=.3:
            retain();mode=current;begin=at;count=0
        last=at;count+=1
    retain();return spans


def assessment(record,motion):
    if record['spec']['kind'] not in ('single_crossing','single_head_on'):
        raise ValueError('single visible obstacle fixture required')
    if len(record['moving_obstacles'])!=1 or len(record['goal_requests'])!=1:
        raise ValueError('one moving body and one external goal required')
    if motion['overflow']:raise ValueError('motion recorder overflow')
    truth=np.array([[r[6],r[1],r[2],r[7]]for r in record['truth']],float)
    poses=np.asarray(record['localization_poses'],float)[:,[0,1,2,4]]
    if len(truth)<20:raise ValueError('insufficient truth')
    first=next((p for p in poses if truth[0,0]<=p[0]<=truth[-1,0]),None)
    if first is None:raise ValueError('source-time alignment missing')
    t=interpolate_pose(truth,first[0]);angle=first[3]-t[2];c,s=math.cos(angle),math.sin(angle)
    rotation=np.array([[c,-s],[s,c]]);shift=first[1:3]-rotation@t[:2]
    name,mover=next(iter(record['moving_obstacles'].items()));mover=np.asarray(mover,float)
    if len(mover)<20 or not np.isfinite(mover).all() or np.any(np.diff(mover[:,0])<=0):
        raise ValueError('invalid moving body record')
    moved=np.linalg.norm(mover[:,1:]-mover[0,1:],axis=1)
    moving=mover[moved>.02]
    if not len(moving):raise ValueError('body never moved')
    onset=float(moving[0,0]);raw_hits=[];static=fixtures(record['spec'])
    for frame in motion['frames']:
        at=frame['stamp_s']
        if not onset<=at<=min(onset+4.,mover[-1,0]):continue
        xy=(mask_points(frame,'occupied')-shift)@rotation
        center=np.array([np.interp(at,mover[:,0],mover[:,j])for j in (1,2)])
        tolerance=math.sqrt(2)*frame['geometry'][4]/2
        # The known fixture walls are assessment-only. Their raster cells
        # near the cylinder may not stand in for real observations of it.
        keep=np.ones(len(xy),bool)
        for wall in static:
            if wall['shape']=='box':
                keep&=~((np.abs(xy[:,0]-wall['x'])<=wall['length']/2+tolerance+.10)&
                        (np.abs(xy[:,1]-wall['y'])<=wall['width']/2+tolerance+.10))
        xy=xy[keep]
        if len(xy) and np.any(np.linalg.norm(xy-center,axis=1)<=.45+tolerance+.10):raw_hits.append(at)
    modes=Counter();executed=[];action_timeline=[];decision_ms=[];decision_by_mode={};reasons=Counter()
    for row in record['safety_status']:
        at=row['stamp_s']
        if not onset<=at<=truth[-1,0]:continue
        for status in row['status']:
            if status['name']!='terrain_nav/chance_constraints':continue
            values=status['values'];mode=values.get('dynamic_yield_state','unknown');modes[mode]+=1
            reasons[status['message']]+=1
            # A rejected/stop/nominal status between accepted commands breaks
            # the sequence even when its duration is under the gap limit.
            action_timeline.append(dict(stamp_s=at,mode=None))
            try:
                if values.get('dynamic_evaluation_current')=='1':
                    elapsed=float(values['dynamic_evaluation_ms'])
                    if math.isfinite(elapsed) and elapsed>=0:
                        decision_ms.append(elapsed);decision_by_mode.setdefault(mode,[]).append(elapsed)
                if (mode not in ('advance','retreat') or values['command_accepted']!='1' or
                        abs(float(values['command_speed_mps']))<=.05):continue
                limits=[('state_probability_upper','state_limit'),('dynamic_path_probability_upper','local_path_budget'),
                        ('dynamic_braking_probability_upper','braking_budget')]
                if any(not 0<=float(values[a])<=float(values[b])+1e-8 for a,b in limits):continue
                if not truth[0,0]+.1<=at<=truth[-1,0]:continue
                a=interpolate_pose(truth,at-.1);b=interpolate_pose(truth,at)
                actual=float((b[:2]-a[:2])@np.array([math.cos(b[2]),math.sin(b[2])])/.1)
                direction=1 if mode=='advance' else -1
                if direction*actual>.05:
                    action_timeline[-1]['mode']=mode
                    executed.append(dict(stamp_s=at,mode=mode,actual_signed_speed_mps=actual,
                        commanded_speed_mps=float(values['command_speed_mps'])))
            except (KeyError,ValueError,TypeError):continue
    spans=sustained_mode_spans(action_timeline);interaction=bool(spans)
    summary=record['summary'];failures=[]
    if not record['accepted']:failures.append('original navigation/physical/temporal acceptance')
    if len(raw_hits)<3 or raw_hits[-1]-raw_hits[0]<.18:failures.append('insufficient real obstacle observations')
    if not interaction:failures.append('no sustained certified advance/retreat with actual vehicle motion')
    return dict(accepted=not failures,scope=__doc__,failed_checks=failures,body=name,
        onset_s=onset,raw_observed_frames=len(raw_hits),first_raw_observation_delay_s=raw_hits[0]-onset if raw_hits else None,
        mode_status_counts=dict(modes),certified_executed_actions=executed,executed_mode_spans_s=spans,
        decision_wall_ms_p95=float(np.percentile(decision_ms,95))if decision_ms else None,
        decision_wall_ms_max=max(decision_ms)if decision_ms else None,
        decision_wall_ms_by_mode={mode:dict(count=len(samples),p50=float(np.percentile(samples,50)),
            p95=float(np.percentile(samples,95)),maximum=max(samples))for mode,samples in decision_by_mode.items()},
        rejection_messages=dict(reasons),arrival_duration_s=summary.get('duration_s'),
        minimum_body_clearance_m=summary.get('moving_clearance',{}).get(name,{}).get('minimum_m'),
        timing_scope='control diagnostic elapsed wall time; pooled P95 includes cheap waiting decisions; '
            'per-mode distributions separate actual maneuver checks; not command-source period or hard-real-time proof',
        alignment_scope='one early fixed source-time-matched transform used only by this assessor')


def audit(directory):
    directory=Path(directory)
    return assessment(json.loads((directory/'navigation_data.json').read_text()),
        json.loads((directory/'motion_grids.json').read_text()))
