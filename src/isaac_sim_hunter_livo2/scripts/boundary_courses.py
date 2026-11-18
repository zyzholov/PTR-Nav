"""Controlled fixtures and input manifests. Never imported by navigation.

Geometry is shared only by scene authoring and the independent truth observer.
No poses, maps, routes or traversability labels are sent to perception/planning.
"""
import json
import math
import random
from pathlib import Path

KINDS=('long_corridor','side_goal_corridor','dense','chicane','rubble','multi_dynamic','single_crossing','single_head_on','deadend','deadend_entry','deadend_entry_observed_approach')


def load_spec(path):
    d=json.loads(Path(path).read_text())
    allowed={'kind','length_m','width_m','height_m','seed','features','speed_mps','motion_profile','moving_body_count'}
    if set(d)-allowed or d.get('kind') not in KINDS:raise ValueError('invalid boundary fixture schema')
    defaults=dict(length_m=24.,width_m=2.4,height_m=.06,seed=1,features=True,speed_mps=.8)
    defaults.update(d);d=defaults
    profiles={'single_crossing':('constant','varying_speed','stop_go','turning'),
              'multi_dynamic':('constant','varying_speed','periodic')}
    if 'motion_profile' in d and d['motion_profile'] not in profiles.get(d['kind'],()):
        raise ValueError('unsupported motion profile for this fixture')
    if 'moving_body_count' in d and (d['kind']!='multi_dynamic' or
            type(d['moving_body_count']) is not int or d['moving_body_count'] not in (3,6) or
            d['moving_body_count']==6 and d.get('motion_profile')!='periodic'):
        raise ValueError('six-body fixture requires the explicit periodic profile')
    if (not all(math.isfinite(d[k]) for k in ('length_m','width_m','height_m','speed_mps')) or
            not 12<=d['length_m']<=35 or not 1.2<=d['width_m']<=5. or
            not .015<=d['height_m']<=.35 or not .2<=d['speed_mps']<=2. or
            not isinstance(d['seed'],int) or not isinstance(d['features'],bool)):
        raise ValueError('boundary fixture dimensions outside approved sandbox bounds')
    return d


def goal_offset(d):
    if d['kind']=='deadend':return 6.,0.
    if d['kind']=='deadend_entry':return 8.,0.
    if d['kind']=='deadend_entry_observed_approach':return 10.,0.
    if d['kind']=='long_corridor':return d['length_m']+3.,3.5
    if d['kind']=='side_goal_corridor':return 2.5,d['width_m']/2+3.
    if d['kind']=='multi_dynamic':return 15.,0.
    return 12.,0.


def fixtures(d):
    k=d['kind'];out=[]
    def box(name,x,y,length,width,height=1.5,terrain=False):
        out.append(dict(name=name.replace('-','m'),shape='box',x=x,y=y,z=-.02+height/2,
                        length=length,width=width,height=height,terrain=terrain))
    def pillar(name,x,y,radius=.4):
        out.append(dict(name=name,shape='cylinder',x=x,y=y,z=.88,radius=radius,height=1.8,terrain=False))
    if k in ('long_corridor','side_goal_corridor'):
        offset=-d['length_m']/2 if k=='side_goal_corridor' else 0.
        for side in (-1,1):
            box('Wall'+str(side),offset+d['length_m']/2,side*(d['width_m']/2+.15),d['length_m'],.30)
            if d['features']:
                for i in range(2,int(d['length_m']),4):
                    box(f'Seam_{side}_{i}',offset+i,side*(d['width_m']/2-.05),.20,.10)
    elif k in ('deadend_entry','deadend_entry_observed_approach'):
        # Start outside. Closing the far end only after entry leaves REAL
        # traversed ground behind the vehicle. This exercises recovery from a
        # newly blocked branch, unlike assuming unseen rear ground at startup.
        approach=2. if k=='deadend_entry_observed_approach' else 0.
        for side in (-1,1):box('EntrySide'+str(side),2.5+approach,side*.85,3.,.30)
    elif k=='deadend':
        # Vehicle starts facing the closed end. The 1.40 m inner width admits
        # a protected body but cannot fit its minimum-radius U-turn. The exit
        # is BEHIND the start; only one final goal beyond the end wall is sent.
        # This geometry is scene/observer data only, not a planner guide.
        for side in (-1,1):box('DeadendSide'+str(side),-.5,side*.85,4.,.30)
        box('DeadendCap',1.65,0.,.30,2.)
    elif k=='dense':
        # Enclose the field: going around the outermost row in empty world is
        # not evidence of navigating inside a dense obstacle region.
        for side in (-1,1):box('Boundary'+str(side),6.,side*5.1,24.,.30)
        box('RearBoundary',-1.5,0.,.30,10.2)
        rng=random.Random(d['seed'])
        # Unjittered centre corridor is deliberately NOT reserved. Feasibility
        # is evaluated by planning/physics, not assumed from this generator.
        for i in range(5):
            for j in range(5):
                pillar(f'Dense_{i}_{j}',3.+1.65*i+rng.uniform(-.18,.18),
                       -4.+2.*j+(.8 if i%2 else 0)+rng.uniform(-.15,.15),.30)
    elif k=='chicane':
        for side in (-1,1):box('Channel'+str(side),6.,side*2.5,26.,.3)
        for i,x in enumerate((3.,5.8,8.6)):
            gap_center=.8 if i%2==0 else -.8
            for side in (-1,1):
                edge=gap_center+side*d['width_m']/2
                wall_edge=side*2.35
                width=abs(wall_edge-edge)
                if width>.01:box(f'Gate_{i}_{side}',x,(wall_edge+edge)/2,.25,width)
    elif k=='rubble':
        for side in (-1,1):box('Lane'+str(side),6.,side*2.,24.,.3)
        rng=random.Random(d['seed'])
        for i in range(8):
            # Calibrated full-width ledges force actual wheel contact: a clear
            # strip around decorative rocks cannot count as vertical crossing.
            box(f'Ledge_{i}',2.5+i,0.,.28,3.7,d['height_m']*(.75+.25*rng.random()),True)
        for i in range(24):
            box(f'Clast_{i}',rng.uniform(2.5,10.),rng.uniform(-1.65,1.65),
                rng.uniform(.18,.35),rng.uniform(.18,.35),d['height_m']*rng.uniform(.35,.7),True)
    elif k in ('multi_dynamic','single_crossing','single_head_on'):
        # A 0.45m-radius single mover at |y|=3 must not intersect
        # the wall or merge into its measured component at either endpoint.
        half_width=3.8 if k in ('single_crossing','single_head_on') else 3.5
        for side in (-1,1):box('Lane'+str(side),6.,side*half_width,30.,.3)
    return out


def movers(d,elapsed):
    if d['kind']=='single_crossing':
        t=max(0.,elapsed);v=d['speed_mps'];profile=d.get('motion_profile','constant');x=4.3
        if profile=='varying_speed':
            # Positive continuous speed v*(1+.5*sin(w*t)), not a constant
            # speed disguised as different sampling. Only the simulator and
            # independent assessor import this file, never navigation.
            w=math.pi/2;distance=v*(t+.5/w*(1-math.cos(w*t)))
        elif profile=='stop_go':
            # Continuous velocity through finite acceleration, braking, one
            # full second stopped, and re-acceleration. No position teleport.
            remaining=t;distance=0.;speed=0.
            for duration,acceleration in ((1.,v),(1.5,0.),(1.,-v),(1.,0.),(1.,v)):
                dt=min(remaining,duration);distance+=speed*dt+.5*acceleration*dt*dt
                speed+=acceleration*dt;remaining-=dt
                if remaining<=0:break
            distance+=max(0.,remaining)*speed
        elif profile=='turning':
            # Straight approach, tangent-continuous 90-degree turn (R=1m),
            # then longitudinal travel. The turn is not sent as an intent.
            if v*t<=1.5:distance=v*t
            else:
                angle=min(math.pi/2,v*t-1.5)
                x+=1.-math.cos(angle)+max(0.,v*t-1.5-math.pi/2)
                x=min(x,8.5);distance=1.5+math.sin(angle)
        else:distance=v*t
        return [dict(name='SingleCrossing',x=x,y=3.-min(6.,distance),radius=.45,height=1.6)]
    if d['kind']=='single_head_on':
        return [dict(name='SingleApproach',x=8.-min(3.,d['speed_mps']*max(0.,elapsed)),y=0.,radius=.45,height=1.6)]
    if d['kind'] in ('deadend_entry','deadend_entry_observed_approach'):
        approach=2. if d['kind']=='deadend_entry_observed_approach' else 0.
        return [dict(name='ClosingGate',x=4.85+approach,y=3.-min(3.,d['speed_mps']*max(0.,elapsed)),
                     radius=.72,height=1.6)]
    if d['kind']!='multi_dynamic':return []
    profile=d.get('motion_profile','constant')
    t=max(0.,elapsed);v=d['speed_mps']
    if profile=='periodic':
        # Continuous physical trajectories, including changing velocity and
        # reversals. No future trajectories are published to navigation.
        out=[dict(name='CrossA',x=5.,y=2.7*math.cos(v*t/2.7),radius=.45,height=1.6),
             dict(name='CrossB',x=7.5,y=-2.7*math.cos(v*max(0.,t-.5)/2.7),radius=.45,height=1.6),
             dict(name='Lead',x=9.5+.5*math.sin(.35*t),y=.4,radius=.40,height=1.6)]
        if d.get('moving_body_count',3)==6:
            out.extend([dict(name='CrossC',x=12.5,y=2.3*math.cos(.28*t+math.pi/2),radius=.40,height=1.6),
                        dict(name='CrossD',x=14.,y=-2.3*math.cos(.30*t+math.pi/4),radius=.40,height=1.6),
                        dict(name='Counter',x=16.+.7*math.sin(.25*t),y=-1.5,radius=.35,height=1.6)])
        return out
    def distance(at,phase):
        at=max(0.,at)
        if profile=='varying_speed':
            w=math.pi/2
            return v*(at+.35/w*(math.cos(phase)-math.cos(w*at+phase)))
        return v*at
    # Three independent physical kinematic bodies, including opposite lateral
    # directions and a slower longitudinal body. No teleport at observation.
    return [dict(name='CrossA',x=5.,y=2.7-min(5.4,distance(t,0.)),radius=.45,height=1.6),
            dict(name='CrossB',x=7.5,y=-2.7+min(5.4,distance(t-.5,math.pi)),radius=.45,height=1.6),
            dict(name='Lead',x=8.5+min(3.,.25*t),y=.4,radius=.40,height=1.6)]


def terrain_height(d,x,y):
    h=-.02
    if d['kind']=='rubble':
        for o in fixtures(d):
            if o['terrain'] and abs(x-o['x'])<=o['length']/2 and abs(y-o['y'])<=o['width']/2:
                h=max(h,-.02+o['height'])
    return h
