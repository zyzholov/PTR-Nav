from pathlib import Path
import sys
import math
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'src/isaac_sim_hunter_livo2/scripts'))
from boundary_courses import load_spec,fixtures,movers,goal_offset

CASES=Path(__file__).resolve().parents[1]/'experiments/boundary_cases'

@pytest.mark.parametrize('path',sorted(CASES.glob('*.json')))
def test_fixture_names_are_valid_usd_identifiers_and_geometry_is_positive(path):
    s=load_spec(path)
    for o in (*fixtures(s),*movers(s,0.)):
        assert o['name'].isascii() and o['name'].isidentifier()
        assert o['height']>0 and math.isfinite(o['x']) and math.isfinite(o['y'])
    assert len({o['name'] for o in fixtures(s)})==len(fixtures(s))

def test_long_corridor_goal_and_side_goal_exercise_different_unknown_topologies():
    a=load_spec(CASES/'long_corridor.json');b=load_spec(CASES/'side_goal_corridor.json')
    assert goal_offset(a)[0]>a['length_m']>20.8
    assert 0<goal_offset(b)[0]<b['length_m'] and goal_offset(b)[1]>b['width_m']/2
    walls=[o for o in fixtures(b) if o['name'].startswith('Wall')]
    assert all(o['x']==0 and o['length']/2>10.4 for o in walls)

def test_multiple_movers_have_opposite_velocities_and_no_time_step_teleport():
    s=load_spec(CASES/'multi_dynamic.json');a=movers(s,1.);b=movers(s,1.01)
    assert len(a)==3 and b[0]['y']<a[0]['y'] and b[1]['y']>a[1]['y'] and b[2]['x']>a[2]['x']
    assert all(math.hypot(y['x']-x['x'],y['y']-x['y'])<=.02001 for x,y in zip(a,b))


@pytest.mark.parametrize('name',['multi_dynamic_varying','multi_dynamic_six'])
def test_multi_body_profiles_change_velocity_without_body_intersections_or_wall_penetration(name):
    s=load_spec(CASES/(name+'.json'));velocities=[]
    for i in range(1200):
        t=i*.05;a=movers(s,t);b=movers(s,t+.001)
        assert len(a)==(6 if name.endswith('six') else 3)
        velocities.append((b[0]['y']-a[0]['y'])/.001)
        for k,body in enumerate(a):
            assert abs(body['y'])+body['radius']<3.35
            assert math.hypot(b[k]['x']-body['x'],b[k]['y']-body['y'])<=.0028
            for other in a[:k]:
                assert math.hypot(body['x']-other['x'],body['y']-other['y'])>body['radius']+other['radius']+.05
    if name.endswith('six'):assert min(velocities)<-.7 and max(velocities)>.7
    else:assert max(velocities[:80])-min(velocities[:80])>.3


@pytest.mark.parametrize('profile',['varying_speed','stop_go','turning'])
def test_nonlinear_movers_have_distinct_motion_without_teleports_or_wall_intersections(profile):
    import numpy as np
    spec=load_spec(CASES/f'single_{profile}.json');times=np.arange(0.,12.,.005)
    xy=np.array([[movers(spec,float(t))[0][k]for k in ('x','y')]for t in times])
    velocity=np.diff(xy,axis=0)/.005;speed=np.linalg.norm(velocity,axis=1)
    assert np.isfinite(xy).all() and speed.max()<=1.201
    assert np.max(abs(xy[:,1]))+.45<3.8-.15
    if profile=='varying_speed':assert speed[:600].max()-speed[:600].min()>.5
    elif profile=='stop_go':
        assert np.max(speed[(times[:-1]>3.6)&(times[:-1]<4.4)])<1e-8
        assert np.max(speed[(times[:-1]>5)&(times[:-1]<6)])>.7
    else:
        assert abs(velocity[200,0])<1e-6 and velocity[200,1]<-.7
        assert velocity[1200,0]>.7 and abs(velocity[1200,1])<1e-6


def test_low_ledge_case_keeps_original_layout_and_goal_while_halving_terrain_heights():
    low=load_spec(CASES/'rubble_3cm.json');high=load_spec(CASES/'rubble_6cm.json')
    assert goal_offset(low)==goal_offset(high)
    for a,b in zip(fixtures(low),fixtures(high)):
        assert (a['name'],a['x'],a['y'],a['length'],a['width'])==(b['name'],b['x'],b['y'],b['length'],b['width'])
        assert a['height']==pytest.approx(b['height']*(.5 if a['terrain'] else 1.))

def test_rubble_has_forced_contact_ledges_without_a_centre_bypass_strip():
    s=load_spec(CASES/'rubble_6cm.json');ledges=[o for o in fixtures(s) if o['name'].startswith('Ledge')]
    assert len(ledges)==8 and all(o['terrain'] and o['width']>=3.7 and .04<o['height']<=.06 for o in ledges)


def test_dense_field_has_bounding_walls_instead_of_an_empty_outer_bypass():
    objects=fixtures(load_spec(CASES/'dense.json'))
    assert len([o for o in objects if o['shape']=='cylinder'])==25
    assert {o['name'] for o in objects if o['shape']=='box'}=={'Boundarym1','Boundary1','RearBoundary'}


@pytest.mark.parametrize('path',sorted(CASES.glob('*.json')))
def test_nominal_terminal_body_pose_is_not_authored_inside_a_static_or_stopped_dynamic_body(path):
    from clearance_metrics import static_course_clearances,rectangle_circle_clearance
    s=load_spec(path);gx,gy=goal_offset(s)
    rows=[[0.,gx,gy,.26,0.,0.,1.,0.]]
    clr=static_course_clearances(rows,[o for o in fixtures(s) if not o['terrain']])
    assert all(v[0][1]>=.1 for v in clr.values())
    for o in movers(s,1000.):
        assert rectangle_circle_clearance(gx,gy,0.,o['x'],o['y'],front=.58,back=.48,half_width=.4,radius=o['radius'])>=.1


def test_dynamic_paths_never_interpenetrate_authored_lane_walls():
    from clearance_metrics import rectangle_box_clearance
    s=load_spec(CASES/'multi_dynamic.json')
    for t in (0.,1.,4.,10.,1000.):
        for o in movers(s,t):
            for w in fixtures(s):
                assert rectangle_box_clearance(o['x'],o['y'],0.,w['x'],w['y'],w['length'],w['width'],
                    front=o['radius'],back=o['radius'],half_width=o['radius'])>=.1


def test_deadend_has_a_reverse_exit_and_only_a_final_external_goal(tmp_path):
    path=tmp_path/'deadend.json'
    path.write_text('{"kind":"deadend"}')
    spec=load_spec(path)
    assert goal_offset(spec)==(6.,0.)
    objects=fixtures(spec)
    assert len(objects)==3
    sides=[o for o in objects if o['name'].startswith('DeadendSide')]
    assert len(sides)==2
    assert all(abs(o['y'])-o['width']/2==.7 for o in sides)
    assert all(o['x']-o['length']/2==-2.5 for o in sides)
    # No rear cap, route, waypoints, hidden support labels or moving obstacles.
    assert movers(spec,0.)==[]
    assert not any('Rear' in o['name'] for o in objects)


def test_entered_deadend_closes_smoothly_without_interpenetrating_side_walls():
    from clearance_metrics import rectangle_box_clearance
    spec=load_spec(CASES/'deadend_entry.json')
    assert goal_offset(spec)==(8.,0.)
    for t in (0.,.01,1.,2.,4.,10.):
        gate=movers(spec,t)[0]
        for wall in fixtures(spec):
            assert rectangle_box_clearance(gate['x'],gate['y'],0.,wall['x'],wall['y'],wall['length'],wall['width'],
                front=gate['radius'],back=gate['radius'],half_width=gate['radius'])>=.1
    assert abs(movers(spec,.01)[0]['y']-movers(spec,0.)[0]['y'])<=.008001


def test_observed_approach_control_preserves_closed_lane_but_adds_two_metres_before_entry():
    near=load_spec(CASES/'deadend_entry.json');far=load_spec(CASES/'deadend_entry_observed_approach.json')
    assert goal_offset(far)[0]-goal_offset(near)[0]==2.
    for a,b in zip(fixtures(near),fixtures(far)):
        assert b['x']-a['x']==2.
        assert (a['length'],a['width'],a['height'],a['y'])==(b['length'],b['width'],b['height'],b['y'])
    for t in (0.,2.,4.):
        a,b=movers(near,t)[0],movers(far,t)[0]
        assert b['x']-a['x']==2. and (a['radius'],a['height'],a['y'])==(b['radius'],b['height'],b['y'])
