"""Read-only qualification geometry and sustained-clearing diagnostics."""

from bisect import bisect_left
import math

# A recovery at the 1.5s deadline still needs the full 2s confirmation plus
# one 2Hz cumulative-map publication interval. A 3s window cannot test it.
EARLY_RECOVERY_WINDOW_S=1.5+2.0+.5


def moving_obstacle_observation(grids, poses, alignment, *, radius, start_s, end_s,
                                max_pose_gap_s=.12):
    """Read-only actual-cylinder evidence, using RAW columns at SOURCE time.

    Truth locates the evaluation ROI only. No truth enters the planner. Match
    bracketed poses without extrapolation; allow only half a cell diagonal for
    rasterization, never the planner's inflated obstacle envelope. Keep misses
    and unmatched frames so an empty or stale recorder cannot pass.
    """
    import numpy as np
    result=dict(accepted=False,hit_frames=0,matched_frames=0,hit_span_s=0.,
        limits=dict(min_hit_frames=3,min_hit_span_s=.18,max_pose_gap_s=max_pose_gap_s),
        columns=['source_stamp_s','occupied_cells_in_actual_cylinder'],samples=[],
        scope='raw measured columns vs time-aligned actual cylinder; truth is observer-only')
    if (alignment is None or start_s is None or end_s is None or
            len(alignment)!=3 or not all(math.isfinite(v) for v in (*alignment,radius,start_s,end_s)) or
            radius<=0 or end_s<start_s or len(poses)<2):return result
    p=np.asarray(poses,float)
    if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all() or np.any(np.diff(p[:,0])<=0):return result
    tx,ty,yaw=alignment;c,s=math.cos(yaw),math.sin(yaw)
    hits=[];last=-math.inf
    for grid in grids:
        stamp=grid['stamp_s']
        if not start_s<=stamp<=end_s:continue
        if not math.isfinite(stamp) or stamp<=last:return dict(result,reason='nonmonotonic raw source frames')
        last=stamp
        if grid['frame']!='map':continue
        i=bisect_left(p[:,0],stamp)
        if i<len(p) and abs(p[i,0]-stamp)<1e-8:x,y=p[i,1:]
        elif i==0 or i==len(p) or p[i,0]-p[i-1,0]>max_pose_gap_s:continue
        else:
            f=(stamp-p[i-1,0])/(p[i,0]-p[i-1,0]);x,y=p[i-1,1:]+f*(p[i,1:]-p[i-1,1:])
        px,py=tx+c*x-s*y,ty+s*x+c*y
        w,h,ox,oy,res=grid['geometry']
        if not all(math.isfinite(v) for v in (w,h,ox,oy,res)) or min(w,h,res)<=0:continue
        cells=np.asarray(grid['occupied'],int)
        if np.any((cells<0)|(cells>=w*h)):continue
        xx=ox+(cells%w+.5)*res;yy=oy+(cells//w+.5)*res
        count=int(np.count_nonzero((xx-px)**2+(yy-py)**2<=(radius+res/math.sqrt(2))**2))
        result['matched_frames']+=1;result['samples'].append([stamp,count])
        if count:hits.append(stamp)
    result['hit_frames']=len(hits)
    result['hit_span_s']=hits[-1]-hits[0] if hits else 0.
    result['accepted']=len(hits)>=3 and result['hit_span_s']>=.18
    return result


def localization_agreement(poses, truth, *, max_gap_s=.12, min_matches=20,
                           min_fraction=.90, xy_limit_m=.30, z_limit_m=.15,
                           yaw_limit_rad=.15):
    """Observer-only SE(2)+vertical alignment; truth must never enter navigation.

    poses: [source_stamp,x,y,z,yaw]; truth: recorder's eight-column body poses.
    One fixed transform at the first matched sample removes arbitrary map origin.
    Subsequent alignment is forbidden: it would erase actual SLAM drift.
    """
    limits=dict(xy_max_m=xy_limit_m,z_max_m=z_limit_m,yaw_max_rad=yaw_limit_rad,
                min_matches=min_matches,min_matched_fraction=min_fraction,max_truth_gap_s=max_gap_s)
    result=dict(accepted=False,limits=limits,matched_count=0,matched_fraction=0.,
                scope='relative localization since observer attachment; fixed initial yaw/translation alignment, no truth feedback, not absolute georeferencing',
                error_columns=['stamp_s','xy_m','abs_z_m','abs_yaw_rad'],errors=[])
    def valid(rows, indices, time_index):
        return (len(rows)>=2 and all(len(row)>max(indices) and
            all(math.isfinite(row[i]) for i in indices) for row in rows) and
            all(b[time_index]>a[time_index] for a,b in zip(rows,rows[1:])))
    if not valid(poses,(0,1,2,3,4),0) or not valid(truth,(1,2,3,6,7),6):
        return dict(result,reason='missing, nonfinite or nonmonotonic source poses')
    times=[row[6] for row in truth]
    wrap=lambda value: math.atan2(math.sin(value),math.cos(value))
    alignment=None
    for stamp,x,y,z,yaw in poses:
        i=bisect_left(times,stamp)
        if i<len(times) and abs(times[i]-stamp)<1e-8:
            target=truth[i];tx,ty,tz,tyaw=target[1],target[2],target[3],target[7]
        elif i==0 or i==len(times) or times[i]-times[i-1]>max_gap_s:
            continue
        else:
            a,b=truth[i-1],truth[i];f=(stamp-a[6])/(b[6]-a[6])
            tx,ty,tz=(a[j]+f*(b[j]-a[j]) for j in (1,2,3))
            tyaw=a[7]+f*wrap(b[7]-a[7])
        if alignment is None:
            angle=wrap(tyaw-yaw);c,s=math.cos(angle),math.sin(angle)
            alignment=(c,s,tx-c*x+s*y,ty-s*x-c*y,tz-z,angle)
        c,s,dx,dy,dz,angle=alignment
        result['errors'].append([stamp,math.hypot(c*x-s*y+dx-tx,s*x+c*y+dy-ty),
                                 abs(z+dz-tz),abs(wrap(yaw+angle-tyaw))])
    errors=result['errors'];n=len(errors)
    result.update(matched_count=n,matched_fraction=n/len(poses))
    if not n:return dict(result,reason='no time-matched truth')
    result.update(xy_rmse_m=math.sqrt(sum(row[1]**2 for row in errors)/n),
                  xy_max_m=max(row[1] for row in errors),z_max_m=max(row[2] for row in errors),
                  yaw_max_rad=max(row[3] for row in errors))
    result['accepted']=(n>=min_matches and n/len(poses)>=min_fraction and
        result['xy_max_m']<=xy_limit_m and result['z_max_m']<=z_limit_m and
        result['yaw_max_rad']<=yaw_limit_rad)
    result['reason']='within qualification limits' if result['accepted'] else 'localization error or matched coverage outside qualification limits'
    return result


def terminal_stop_evidence(truth,controls,window_s=.5,max_gap_s=.1):
    """Sampled terminal stop, independent of a planner's zero command/state.

    Recorder rows: truth [wall,x,y,z,pitch,roll,stamp,yaw], commands [stamp,v,steer].
    No extrapolation over a missing tail or long gap; not a continuous proof.
    """
    result=dict(accepted=False,window_s=window_s,max_gap_s=max_gap_s,
        speed_limit_mps=.05,yaw_rate_limit_radps=.10,command_limit_mps=.03)
    if (not math.isfinite(window_s) or not math.isfinite(max_gap_s) or
            window_s<=0 or max_gap_s<=0 or max_gap_s>=window_s):raise ValueError('invalid stop window')
    if len(truth)<2 or len(controls)<2:return dict(result,reason='missing stop samples')
    if (any(len(r)<8 or not all(math.isfinite(r[k])for k in (1,2,6,7))for r in truth) or
        any(len(r)<2 or not all(math.isfinite(v)for v in r[:2])for r in controls) or
        any(b[6]<=a[6]for a,b in zip(truth,truth[1:])) or
        any(b[0]<a[0]for a,b in zip(controls,controls[1:]))):
        return dict(result,reason='invalid source samples')
    end=max(truth[-1][6],controls[-1][0]);begin=end-window_s
    def tail(rows,index):
        j=bisect_left([r[index]for r in rows],begin)
        return rows[max(0,j-1):]
    body=tail(truth,6);commands=tail(controls,0)
    for rows,index in ((body,6),(commands,0)):
        if (rows[0][index]>begin+1e-8 or end-rows[-1][index]>max_gap_s or
                any(b[index]-a[index]>max_gap_s for a,b in zip(rows,rows[1:]))):
            return dict(result,reason='stop window not covered')
    speed=max(math.hypot(b[1]-a[1],b[2]-a[2])/(b[6]-a[6])for a,b in zip(body,body[1:]))
    yaw=max(abs(math.atan2(math.sin(b[7]-a[7]),math.cos(b[7]-a[7])))/(b[6]-a[6])for a,b in zip(body,body[1:]))
    command=max(abs(r[1])for r in commands)
    return dict(result,accepted=speed<=.05 and yaw<=.10 and command<=.03,
        sampled_speed_max_mps=speed,sampled_yaw_rate_max_radps=yaw,command_max_mps=command,
        reason='sampled terminal stop' if speed<=.05 and yaw<=.10 and command<=.03 else 'moving or nonzero command')


def clearance_acceptance(temporal, *, max_delay_s=1.5, require_probability=False):
    """All measured products must remain clear, without unknown gaps or relapse."""
    required=("registered","laser_obstacle","baseline_risk","enhanced_risk","live")
    if require_probability:required+=("probability_pose",)
    return all(name in temporal and
        temporal[name].get("first_sustained_clear_s") is not None and
        0<=temporal[name]["first_sustained_clear_s"]<=max_delay_s and
        temporal[name].get("clear_at_end") is True and
        temporal[name].get("reappearance_count")==0 for name in required)


def rectangle_circle_clearance(x, y, yaw, obstacle_x, obstacle_y, *, front,
                               back, half_width, radius):
    if (not all(math.isfinite(v) for v in
            (x,y,yaw,obstacle_x,obstacle_y,front,back,half_width,radius))
            or min(front,back,half_width,radius)<=0):
        raise ValueError("invalid physical clearance geometry")
    dx,dy=obstacle_x-x,obstacle_y-y
    u=math.cos(yaw)*dx+math.sin(yaw)*dy
    v=-math.sin(yaw)*dx+math.cos(yaw)*dy
    return math.hypot(max(-back-u,0.,u-front),max(abs(v)-half_width,0.))-radius


def rectangle_front_wall_clearance(x,yaw,wall_x,*,front,back,half_width):
    """Signed separation from a vertical plane x=wall_x, approached from x<wall_x."""
    if (not all(math.isfinite(v) for v in (x,yaw,wall_x,front,back,half_width))
            or min(front,back,half_width)<=0):
        raise ValueError('invalid front-wall geometry')
    c,s=math.cos(yaw),math.sin(yaw)
    return wall_x-x-max(front*c,-back*c)-half_width*abs(s)


def rectangle_box_clearance(x, y, yaw, ox, oy, length, width, *, front, back, half_width):
    """Planar oriented body vs axis-aligned fixture: distance or negative SAT overlap."""
    values = (x, y, yaw, ox, oy, length, width, front, back, half_width)
    if not all(math.isfinite(v) for v in values) or min(length, width, front, back, half_width) <= 0:
        raise ValueError('invalid box clearance geometry')
    c, s = math.cos(yaw), math.sin(yaw)
    body = [(x+c*u-s*v, y+s*u+c*v)
            for u, v in ((front, half_width), (-back, half_width),
                         (-back, -half_width), (front, -half_width))]
    box = [(ox+u*length/2., oy+v*width/2.) for u, v in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
    overlaps = []
    for ax, ay in ((1., 0.), (0., 1.), (c, s), (-s, c)):
        a = [px*ax+py*ay for px, py in body]
        b = [px*ax+py*ay for px, py in box]
        overlaps.append(min(max(a)-min(b), max(b)-min(a)))
    if min(overlaps) >= 0.:
        return -min(overlaps)

    def point_edge(p, a, b):
        dx, dy = b[0]-a[0], b[1]-a[1]
        t = max(0., min(1., ((p[0]-a[0])*dx+(p[1]-a[1])*dy)/(dx*dx+dy*dy)))
        return math.hypot(p[0]-a[0]-t*dx, p[1]-a[1]-t*dy)

    return min(point_edge(p, polygon[i], polygon[(i+1) % 4])
               for points, polygon in ((body, box), (box, body))
               for p in points for i in range(4))


def static_course_clearances(truth, obstacles, *, front=.58, back=.48, half_width=.4,
                             body_z_bounds=None):
    """Independent fixed-world fixture evaluation; no estimated odometry or map input."""
    geometry = dict(front=front, back=back, half_width=half_width)
    result = {o['name']: [] for o in obstacles}
    for sample in truth:
        if len(sample) < 8 or not all(math.isfinite(sample[i]) for i in (1, 2, 6, 7)):
            raise ValueError('invalid physical truth sample')
        projected=geometry
        zlow=zhigh=None
        if body_z_bounds is not None:
            low,high=body_z_bounds
            if (not all(math.isfinite(v) for v in (low,high,*sample[3:6])) or low>=high):
                raise ValueError('invalid full-vehicle vertical bounds or attitude')
            cp,sp=math.cos(sample[4]),math.sin(sample[4]);cr,sr=math.cos(sample[5]),math.sin(sample[5])
            corners=[(cp*u+sp*sr*v+sp*cr*z,cr*v-sr*z,
                      sample[3]-sp*u+cp*sr*v+cp*cr*z)
                     for u in (-back,front) for v in (-half_width,half_width) for z in (low,high)]
            projected=dict(front=max(p[0] for p in corners),back=-min(p[0] for p in corners),
                           half_width=max(abs(p[1]) for p in corners))
            zlow=min(p[2] for p in corners);zhigh=max(p[2] for p in corners)
        for o in obstacles:
            if o['shape'] == 'cylinder':
                value = rectangle_circle_clearance(sample[1], sample[2], sample[7],
                    o['x'], o['y'], radius=o['radius'], **projected)
            elif o['shape'] == 'box':
                value = rectangle_box_clearance(sample[1], sample[2], sample[7],
                    o['x'], o['y'], o['length'], o['width'], **projected)
            else:
                raise ValueError('unsupported fixture shape')
            if body_z_bounds is not None:
                # Independent separating-axis lower bound, not an exact 3D
                # mesh distance. Positive vertical clearance permits underpass;
                # the post remains blocked when its height overlaps the body.
                if 'z' not in o or 'height' not in o:raise ValueError('missing fixture height')
                separation=max(o['z']-o['height']/2-zhigh,zlow-o['z']-o['height']/2)
                value=max(value,separation)
            result[o['name']].append((sample[6], value))
    return result


def sampled_body_clearances(truth, obstacles, *, max_gap_s=.12, **geometry):
    """Interpolate bracketed obstacle poses; never extrapolate or prove intersample safety.

    Truth rows use the archived navigation-trial format (stamp index 6, yaw 7).
    Obstacle rows are (simulation stamp, x, y). Duplicate publications at one
    stamp retain their last value; source arrays are not modified.
    """
    if not math.isfinite(max_gap_s) or max_gap_s<=0:
        raise ValueError("invalid alignment gap")
    unique={float(s[0]):s for s in obstacles if all(math.isfinite(v) for v in s[:3])}
    times=sorted(unique)
    result=[]
    for sample in truth:
        if len(sample)<8 or not all(math.isfinite(sample[i]) for i in (1,2,6,7)):
            continue
        t=sample[6]
        right=bisect_left(times,t)
        if right<len(times) and times[right]==t:
            point=unique[t];ox,oy=point[1:3];gap=0.
        elif 0<right<len(times):
            a,b=unique[times[right-1]],unique[times[right]]
            gap=b[0]-a[0]
            if gap>max_gap_s:continue
            weight=(t-a[0])/gap
            ox=a[1]+weight*(b[1]-a[1]);oy=a[2]+weight*(b[2]-a[2])
        else:continue
        clearance=rectangle_circle_clearance(sample[1],sample[2],sample[7],ox,oy,**geometry)
        result.append((t,clearance,gap))
    return result


def held_velocity_encounter(truth, obstacles, *, horizon_s=2.4, **geometry):
    """Diagnostic counterfactual, not a prediction of the actual controlled car.

    Freeze the pre-event measured planar velocity and heading. Test whether
    this straight unresponsive projection would approach the recorded obstacle.
    """
    if not math.isfinite(horizon_s) or horizon_s<=0:raise ValueError("invalid encounter horizon")
    motion=next(((a,b) for a,b in zip(obstacles,obstacles[1:])
        if b[0]>a[0] and math.hypot(b[1]-a[1],b[2]-a[2])>1e-4),None)
    if motion is None:return None
    stamp=motion[0][0]
    before=[s for s in truth if len(s)>=8 and s[6]<=stamp]
    if not before or stamp-before[-1][6]>.12:return None
    last=before[-1]
    prior=next((s for s in reversed(before[:-1]) if .08<=last[6]-s[6]<=.3),None)
    if prior is None:return None
    dt=last[6]-prior[6];vx=(last[1]-prior[1])/dt;vy=(last[2]-prior[2])/dt
    values=[]
    for t,ox,oy in obstacles:
        if stamp<=t<=stamp+horizon_s:
            age=t-last[6]
            clearance=rectangle_circle_clearance(last[1]+age*vx,last[2]+age*vy,last[7],ox,oy,**geometry)
            values.append(clearance)
    if len(values)<2:return None
    return dict(event_stamp_s=stamp,velocity_mps=[vx,vy],horizon_s=horizon_s,
        minimum_clearance_m=min(values),samples=len(values),
        scope="unresponsive held velocity and heading; not actual collision or control proof")


def event_stop_response(truth, controls, event_stamp, *, horizon_s=2.4):
    """Stop command latency and sustained physical rest after the moving event."""
    zero=next((c[0]-event_stamp for c in controls
        if event_stamp<=c[0]<=event_stamp+horizon_s and abs(c[1])<.03),None)
    speeds=[]
    for i,sample in enumerate(truth):
        if not event_stamp<=sample[6]<=event_stamp+horizon_s:continue
        prior=next((p for p in reversed(truth[:i]) if .08<=sample[6]-p[6]<=.3),None)
        if prior is None:continue
        speed=math.hypot(sample[1]-prior[1],sample[2]-prior[2])/(sample[6]-prior[6])
        speeds.append((sample[6]-event_stamp,speed))
    rest=sustained_clearance(speeds,.05,min_duration_s=.2,max_gap_s=.12)
    return dict(first_zero_command_delay_s=zero,first_sustained_truth_stop_delay_s=rest["first_sustained_clear_s"],
        physical_stop_duration_required_s=.2,scope="simulation timestamps relative to observed obstacle-motion onset")


def sustained_clearance(samples, threshold, *, min_duration_s=2.0, max_gap_s=.6):
    if min_duration_s<=0 or max_gap_s<=0 or not math.isfinite(threshold):
        raise ValueError("invalid clearance observation contract")
    rows=[(float(t),float(v) if v is not None and math.isfinite(float(v)) else None)
          for t,v in samples if math.isfinite(float(t))]
    if not rows:
        return {"samples":0,"first_sustained_clear_s":None,"clear_at_end":None,
                "reappearance_count":None,"observation_end_s":None}
    if any(b[0]<=a[0] for a,b in zip(rows,rows[1:])):
        raise ValueError("clearance timestamps must be strictly increasing")
    start=None;first=None;previous=None;was_cleared=False;reappeared=0
    for t,value in rows:
        if previous is not None and t-previous>max_gap_s:
            start=None;was_cleared=False
        if value is None:
            start=None;was_cleared=False
        elif value<=threshold:
            if start is None:start=t
            if t-start>=min_duration_s:
                was_cleared=True
                if first is None:first=start
        else:
            if was_cleared:reappeared+=1
            start=None;was_cleared=False
        previous=t
    return {"samples":sum(v is not None for _,v in rows),"first_sustained_clear_s":first,
            "clear_at_end":was_cleared,"reappearance_count":reappeared,
            "observation_end_s":rows[-1][0]}
