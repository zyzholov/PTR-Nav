import ast
import pytest
from single_dynamic_audit import assessment,sustained_mode_spans
from boundary_courses import fixtures,movers,load_spec
from pathlib import Path


def sample():
    times=[1.+i*.05 for i in range(61)]
    truth=[(0.,.2*t,0.,.26,0.,0.,t,0.)for t in times]
    statuses=[]
    for at in (1.4,1.6,1.8,2.,2.2):
        statuses.append(dict(stamp_s=at,status=[dict(name='terrain_nav/chance_constraints',message='certified',values={
            'dynamic_yield_state':'advance','command_accepted':'1','command_speed_mps':'.3',
            'state_probability_upper':'.001','state_limit':'.05','dynamic_path_probability_upper':'.002',
            'local_path_budget':'.1','dynamic_braking_probability_upper':'.002','braking_budget':'.05'})]))
    trace=dict(spec=dict(kind='single_crossing'),accepted=True,goal_requests=[{}],truth=truth,
        localization_poses=[(t,.2*t,0.,.26,0.)for t in times],safety_status=statuses,
        moving_obstacles={'SingleCrossing':[(t,4.3,3.-.8*(t-1.))for t in times]},
        summary=dict(duration_s=3.,moving_clearance={'SingleCrossing':dict(minimum_m=.5)}))
    frames=[]
    for t in times:
        x,y=4.3,3.-.8*(t-1.)
        ix=int((x+10)/.2);iy=int((y+10)/.2)
        frames.append(dict(stamp_s=t,geometry=[100,100,-10.,-10.,.2],occupied=[iy*100+ix]))
    return trace,dict(overflow=False,frames=frames)


def test_accepts_only_observed_certified_executed_action_and_original_navigation():
    trace,motion=sample();result=assessment(trace,motion)
    assert result['accepted'] and result['executed_mode_spans_s']['advance']>.5
    assert len(result['certified_executed_actions'])==5 and result['raw_observed_frames']>=3


def test_decision_timing_does_not_hide_maneuvers_behind_cheap_waiting():
    trace,motion=sample()
    for row in trace['safety_status']:
        row['status'][0]['values'].update(dynamic_evaluation_current='1',dynamic_evaluation_ms='12')
    for i in range(100):
        trace['safety_status'].append(dict(stamp_s=2.01+i*.001,status=[dict(
            name='terrain_nav/chance_constraints',message='lost',values=dict(
                dynamic_yield_state='prediction_lost_hold',dynamic_evaluation_current='1',
                dynamic_evaluation_ms='.1',command_accepted='0'))]))
    result=assessment(trace,motion)
    assert result['decision_wall_ms_p95']==.1
    assert result['decision_wall_ms_by_mode']['advance']['p95']==12
    assert result['decision_wall_ms_by_mode']['prediction_lost_hold']['count']==100


@pytest.mark.parametrize('bad',['stop_only','no_actual_motion','no_certificate','unsafe_original','missing_raw','wall_only','interrupted_actions','intervening_stop'])
def test_rejects_lucky_arrival_or_advertised_but_unexecuted_interaction(bad):
    trace,motion=sample()
    if bad=='unsafe_original':trace['accepted']=False
    elif bad=='missing_raw':
        for f in motion['frames']:f['occupied']=[]
    elif bad=='wall_only':
        # A real wall cell lies close enough to the synthetic mover's early
        # ROI to fake observations unless static fixture returns are removed.
        trace['moving_obstacles']['SingleCrossing']=[(t,x,y+.3)for t,x,y in trace['moving_obstacles']['SingleCrossing']]
        for f in motion['frames']:f['occupied']=[68*100+71] # y=3.7, x=4.3: the wall's inner face.
    elif bad=='no_actual_motion':
        trace['truth']=[(r[0],0.,*r[2:])for r in trace['truth']]
        trace['localization_poses']=[(r[0],0.,*r[2:])for r in trace['localization_poses']]
    elif bad=='interrupted_actions':trace['safety_status']=trace['safety_status'][::2]
    elif bad=='intervening_stop':
        stops=[]
        for row in trace['safety_status'][:-1]:
            stops.append(dict(stamp_s=row['stamp_s']+.1,status=[dict(name='terrain_nav/chance_constraints',
                message='rejected',values=dict(dynamic_yield_state='refused',command_accepted='0',command_speed_mps='0'))]))
        trace['safety_status']=sorted(trace['safety_status']+stops,key=lambda r:r['stamp_s'])
    else:
        for row in trace['safety_status']:
            values=row['status'][0]['values']
            if bad=='stop_only':values['dynamic_yield_state']='brake';values['command_speed_mps']='0'
            else:values['dynamic_braking_probability_upper']='nan'
    assert not assessment(trace,motion)['accepted']


def test_spans_cannot_join_distant_bursts_or_different_gears():
    assert sustained_mode_spans([dict(stamp_s=t,mode='advance')for t in [0.,.2,.8]])=={}
    assert sustained_mode_spans([dict(stamp_s=t,mode=m)for t,m in [(0.,'advance'),(.2,'retreat'),(.4,'advance')]])=={}
    assert sustained_mode_spans([dict(stamp_s=t,mode=m)for t,m in [(0.,'advance'),(.1,None),(.2,'advance'),(.4,'advance')]])=={}


def test_two_fixtures_have_one_continuous_obstacle_and_no_hidden_goal_route():
    root=Path(__file__).parents[1]/'experiments/boundary_cases'
    for kind in ('single_crossing','single_head_on'):
        spec=load_spec(root/(kind+'.json'));assert len(movers(spec,0.))==1
        assert len(fixtures(spec))==2
        for step in range(201):
            body=movers(spec,step*.05)[0]
            for wall in fixtures(spec):
                # Independent circle-to-AABB distance; no mover may start
                # or finish inside a wall in a visibility qualification.
                dx=max(0.,abs(body['x']-wall['x'])-wall['length']/2)
                dy=max(0.,abs(body['y']-wall['y'])-wall['width']/2)
                assert (dx*dx+dy*dy)**.5-body['radius']>=.1
        a,b=movers(spec,1.)[0],movers(spec,1.01)[0]
        assert ((a['x']-b['x'])**2+(a['y']-b['y'])**2)**.5<=.00800001
    end=movers(load_spec(root/'single_head_on.json'),100.)[0]
    assert end['x']==5. and end['y']==0.


def test_scene_updates_every_manifest_with_movers_in_its_actual_physics_loop():
    # Pure fixture tests cannot establish that the Isaac plant calls them.
    # Inspect the executable loop without importing Isaac or starting a GPU.
    scene=Path(__file__).parents[3]/'src/isaac_sim_hunter_livo2/scripts/hunter_livo2_isaac_scene.py'
    tree=ast.parse(scene.read_text())
    assignment=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and
        any(isinstance(t,ast.Name) and t.id=='BOUNDARY_MOVING' for t in n.targets))
    assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='boundary_movers'
               for n in ast.walk(assignment.value))
    loop=next(n for n in ast.walk(tree) if isinstance(n,ast.While) and
        any(isinstance(c,ast.If) and isinstance(c.test,ast.Name) and c.test.id=='BOUNDARY_MOVING'
            for c in n.body))
    gate=next(n for n in loop.body if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='BOUNDARY_MOVING')
    assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='SetTranslate'
               for n in ast.walk(gate))
