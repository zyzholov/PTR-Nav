import math
import json
import ast
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from clearance_metrics import rectangle_box_clearance, static_course_clearances
from navigation_trial import COMPOSITE_SCENARIOS, course_obstacles
from navigation_courses import (CROSSING_SCENARIOS, RAMP_SCENARIOS, OVERHEAD_SCENARIOS, course_distance,
                                default_obstacle_trigger_distance, minimum_course_distance, front_sign_obstacles)
import navigation_trial


SHAPE = dict(front=.58, back=.48, half_width=.4)


def test_background_sign_is_evaluated_as_both_post_and_panel_not_just_three_pillars():
    fixtures = front_sign_obstacles()
    assert len(fixtures) == 2
    values = static_course_clearances([(0., 5.4, -2.37, .26, 0., 0., .1, 0.)], fixtures)
    assert values['FrontSignPost'][0][1] > .4
    assert values['FrontSignPanel'][0][1] == pytest.approx(.1)
    # A 1cm smaller clearance must fail the existing 10cm safety criterion.
    values = static_course_clearances([(0., 5.4, -2.38, .26, 0., 0., .1, 0.)], fixtures)
    assert values['FrontSignPanel'][0][1] < .1
    args = navigation_trial.parse_args(['--scenario', 'staggered', '--distance', '12', '--evaluate-front-sign'])
    assert args.evaluate_front_sign


def test_axis_aligned_box_separation_and_penetration():
    distance = lambda x, y, yaw=0.: rectangle_box_clearance(x, y, yaw, 2., 0., 1., 1., **SHAPE)
    assert distance(0., 0.) == pytest.approx(.92)
    assert distance(0., 0., math.pi) == pytest.approx(1.02)
    assert distance(1., 0.) == pytest.approx(-.08)
    assert distance(0., 2.) == pytest.approx(math.hypot(.92, 1.1))


def test_rotated_body_uses_edges_and_corners_not_center_distance():
    assert rectangle_box_clearance(0., 0., math.pi/2, 0., 2., 1., 1., **SHAPE) == pytest.approx(.92)
    assert rectangle_box_clearance(0., 0., math.pi/4, 0., 1., 5., .2, **SHAPE) == pytest.approx(.9-.98/math.sqrt(2))
    assert rectangle_box_clearance(0., 0., 0., 0., 0., 5., 5., **SHAPE) < 0.
    with pytest.raises(ValueError):
        rectangle_box_clearance(0., 0., 0., 1., 1., 0., 1., **SHAPE)


def test_corridor_has_genuine_body_margin_and_checks_both_walls():
    fixtures = course_obstacles('corridor')
    truth = [(0., 4.5, 0., .3, 0., 0., .1, 0.)]
    values = static_course_clearances(truth, fixtures)
    assert len(values) == 2
    assert all(rows[0][1] == pytest.approx(.8) for rows in values.values())
    truth[0] = (0., 4.5, 1., .3, 0., 0., .1, 0.)
    assert min(rows[0][1] for rows in static_course_clearances(truth, fixtures).values()) < 0.


def test_course_results_include_every_fixture_and_cannot_mutate_scene():
    assert set(COMPOSITE_SCENARIOS) == {'staggered', 'corridor', 'ramp_detour', 'reserve_overlap', 'overhead_pass', 'overhead_block', 'overhead_plain'}
    for name in COMPOSITE_SCENARIOS:
        fixtures = course_obstacles(name)
        assert len({o['name'] for o in fixtures}) == len(fixtures)
        assert all(o['name'].isascii() and o['name'].isidentifier() for o in fixtures)
        end=fixtures[-1]['x'] if name in OVERHEAD_SCENARIOS else max(o['x'] for o in fixtures)
        assert course_distance(name)>end
        assert course_distance(name) >= minimum_course_distance(name)
        result = static_course_clearances([(0., 0., 0., .3, 0., 0., .1, 0.)], fixtures)
        assert set(result) == {o['name'] for o in fixtures}
        fixtures[0]['x'] = -100.
        assert course_obstacles(name)[0]['x'] > 0.
    with pytest.raises(ValueError):
        static_course_clearances([(0., float('nan'), 0., .3, 0., 0., .1, 0.)], course_obstacles('corridor'))


def test_overhead_pair_has_identical_channel_and_goal_but_different_physical_clearance():
    high=course_obstacles('overhead_pass');low=course_obstacles('overhead_block')
    assert high[:-1]==low[:-1]
    plain=course_obstacles('overhead_plain')
    assert plain[:2]==high[:2] and plain[-1]==high[-1]
    assert len(plain)==3 and len(high)>3
    assert course_distance('overhead_pass')==course_distance('overhead_block')==7.
    assert minimum_course_distance('overhead_pass')<7.
    pose=(1.,4.,0.,.27,0.,0.,1.,0.)
    for fixture in (high[-1],low[-1]):
        assert fixture['z']==pytest.approx(-.02+fixture['bottom_m']+fixture['height']/2)
    assert static_course_clearances([pose],high)['OverheadBeam'][0][1]<0
    assert static_course_clearances([pose],high,body_z_bounds=(-.30,.35))['OverheadBeam'][0][1]>.30
    assert static_course_clearances([pose],low,body_z_bounds=(-.30,.35))['OverheadBeam'][0][1]<0


def test_campaign_plan_preserves_whole_courses_and_explicit_model_without_running(tmp_path):
    root = Path(__file__).resolve().parents[3]
    command = [sys.executable, str(root / 'scripts/sim/experiments/run_matrix.py'),
               '--case', 'staggered:2', '--case', 'corridor:1', '--case', 'ramp_detour:1',
               '--prediction-backend', 'feature_sgp', '--gp-complete-supported',
               '--probability-model', 'shared_gp', '--startup-support', 'stable_stance',
               '--gp-support-distance', '.75', '--reverse-recovery', '--vary-goals',
               '--dry-run', '--output', str(tmp_path / 'must_not_exist')]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, check=True)
    plan = json.loads(result.stdout)
    assert len(plan['jobs']) == 4
    for job in plan['jobs']:
        assert job['goal_distance_m'] == course_distance(job['scenario'])
        assert job['goal_lateral_m'] == 0.
    assert plan['settings']['probability_model'] == 'shared_gp'
    assert plan['settings']['reverse_recovery'] is True
    assert not (tmp_path / 'must_not_exist').exists()
    rejected = subprocess.run([*command, '--reverse-goal'], cwd=root, capture_output=True, text=True)
    assert rejected.returncode == 2


def test_reserve_fixture_has_physical_clearance_and_is_not_a_hidden_spawn_collision():
    fixtures = course_obstacles('reserve_overlap')
    values = static_course_clearances([(0., 2., 0., .3, 0., 0., .1, 0.)], fixtures)
    assert values['FrontReserve'][0][1] == pytest.approx(.42)
    assert course_distance('reserve_overlap') == 4.


def test_reserve_qualification_requires_certificates_actual_reversal_and_budgets():
    from navigation_trial import reserve_recovery_evidence
    values = dict(clearance_recovery_state='retreat', command_accepted='1',
                  command_speed_mps='-.3', clearance_recovery_path_upper='.001',
                  clearance_recovery_braking_upper='.001')
    samples = [dict(stamp_s=t,status=[dict(values=values)]) for t in (.1,.3)]
    samples.append(dict(stamp_s=1.5,status=[dict(values={**values,
        'clearance_recovery_state':'completed','command_speed_mps':'0'})]))
    truth = [(t,x,0.,.3,0.,0.,t,0.) for t,x in [(0.,0.),(1.,-.1),(2.,-.1),(3.,4.)]]
    assert reserve_recovery_evidence(samples, truth, 0.)['accepted']
    assert not reserve_recovery_evidence(samples[:-1], truth, 0.)['accepted']
    assert not reserve_recovery_evidence([], truth, 0.)['accepted']
    assert not reserve_recovery_evidence([samples[0],samples[-1]], truth, 0.)['accepted']
    # A different controller reversing after completion cannot satisfy this gate.
    outside = [(t,x,0.,.3,0.,0.,t,0.) for t,x in [(0.,0.),(1.,0.),(2.,0.),(3.,-.6)]]
    assert not reserve_recovery_evidence(samples, outside, 0.)['accepted']
    for key, value in [('clearance_recovery_path_upper', '.10001'),
                       ('clearance_recovery_braking_upper', '.05001'),
                       ('clearance_recovery_braking_upper', 'nan'),
                       ('command_speed_mps', '-.31'), ('command_accepted', '0')]:
        broken = [dict(stamp_s=t,status=[dict(values={**values, key:value})]) for t in (.1,.3)]
        assert not reserve_recovery_evidence([*broken,samples[-1]], truth, 0.)['accepted']


def test_box_fixture_cannot_report_cylinder_clearance_or_encounter():
    tree = ast.parse(Path(navigation_trial.__file__).read_text())
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    def assignment(name):
        return next(i for i, n in enumerate(run.body) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
    def forbidden(*args, **kwargs):
        raise AssertionError('box was incorrectly evaluated as a cylinder')
    scope = dict(vars(navigation_trial), args=SimpleNamespace(scenario='reserve_overlap'),
                 node=SimpleNamespace(obstacle_pose_samples=[], control_samples=[]),
                 samples=[(0., 0., 0., .3, 0., 0., .1, 0.)],
                 sampled_body_clearances=forbidden, held_velocity_encounter=forbidden)
    block = run.body[assignment('body_geometry'):assignment('summary')]
    exec(compile(ast.Module(body=block, type_ignores=[]), '<actual geometry dispatch>', 'exec'), scope)
    assert scope['body_min'] is None
    assert scope['encounter'] is None
    assert not scope['body_clearances']


def test_attempt_inventory_retains_failure_and_verifies_frozen_sources(tmp_path):
    import hashlib
    import zipfile
    from summarize_navigation_courses import index_attempt
    (tmp_path / 'result.json').write_text(json.dumps({'passed':False,'failure_reason':'test latency'}))
    (tmp_path / 'run_metadata.json').write_text('{}')
    with pytest.raises(ValueError,match='snapshot'):
        index_attempt(tmp_path)
    with zipfile.ZipFile(tmp_path / 'source_snapshot.zip','w') as archive:
        archive.writestr('src/node.py', 'version = 1\n')
    digest=hashlib.sha256((tmp_path / 'source_snapshot.zip').read_bytes()).hexdigest()
    (tmp_path / 'run_metadata.json').write_text(json.dumps({'source_snapshot_sha256':digest}))
    result=index_attempt(tmp_path)
    assert result['result']['passed'] is False
    assert result['result']['failure_reason'] == 'test latency'
    assert result['artifact_sha256']['source_snapshot.zip'] == digest
    (tmp_path / 'source_snapshot.zip').write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='snapshot'):
        index_attempt(tmp_path)


def test_ramp_crossing_is_dynamic_and_requires_the_whole_ramp():
    assert set(RAMP_SCENARIOS) == {'ramp', 'ramp_detour', 'ramp_crossing'}
    assert set(CROSSING_SCENARIOS) == {'crossing', 'ramp_crossing'}
    assert 'ramp_crossing' not in COMPOSITE_SCENARIOS
    assert course_obstacles('ramp_crossing') == ()
    assert course_distance('ramp_crossing') == 7.
    assert minimum_course_distance('ramp_crossing') == pytest.approx(6.28)
    assert default_obstacle_trigger_distance('ramp_crossing') == 2.
    assert default_obstacle_trigger_distance('crossing') == 1.10


def evaluate_trial_gates(scenario, direction='uphill', **overrides):
    # Execute the production acceptance block without a ROS node or simulator.
    tree = ast.parse(Path(navigation_trial.__file__).read_text())
    run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run')
    start = next(i for i, node in enumerate(run.body) if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == 'common_ok' for target in node.targets))
    end = next(i for i, node in enumerate(run.body) if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == 'temporal' for target in node.targets))
    scope = dict(vars(navigation_trial))
    scope.update(args=SimpleNamespace(scenario=scenario, ramp_direction=direction, reach_distance=.3,
                                     require_dynamic_encounter=False),
                 node=SimpleNamespace(map_samples=10, commands=5, live_center_samples=3),
                 samples=[(0., i*.1, 1.6, .35, 0., 0., 1.+i*.05, 0.) for i in range(71)],
                 summary=dict(truth_z_rise_max_m=.45 if direction == 'uphill' else 0.,
                              truth_z_drop_max_m=.45 if direction == 'downhill' else 0.,
                              truth_pitch_max_rad=.2, truth_roll_max_rad=.1, front_stop_commands=1),
                 map_age_p95=.7, reached=True, truth_error=.3, settled=True,
                 physical_separations=[1.05]*20, body_clearance_ok=True,
                 encounter=dict(minimum_clearance_m=.09),
                 event_response=dict(first_zero_command_delay_s=.6, first_sustained_truth_stop_delay_s=.5))
    scope.update(overrides)
    exec(compile(ast.Module(body=run.body[start:end], type_ignores=[]), '<actual trial gates>', 'exec'), scope)
    return bool(scope['passed'])


@pytest.mark.parametrize('scenario', RAMP_SCENARIOS)
@pytest.mark.parametrize('direction', ['uphill', 'downhill'])
def test_every_ramp_requires_the_directed_truth_excursion_and_attitude_limits(scenario, direction):
    assert evaluate_trial_gates(scenario, direction)
    metric = 'truth_z_drop_max_m' if direction == 'downhill' else 'truth_z_rise_max_m'
    other = 'truth_z_rise_max_m' if direction == 'downhill' else 'truth_z_drop_max_m'
    good = dict(truth_z_rise_max_m=0., truth_z_drop_max_m=0., truth_pitch_max_rad=.55,
                truth_roll_max_rad=.45, front_stop_commands=1)
    good[metric] = .28
    assert evaluate_trial_gates(scenario, direction, summary=good)
    for field, value in ((metric, .279), ('truth_pitch_max_rad', .551), ('truth_roll_max_rad', .451)):
        assert not evaluate_trial_gates(scenario, direction, summary={**good, field: value})
    assert not evaluate_trial_gates(scenario, direction, summary={**good, metric: .1, other: 1.2})
    assert not evaluate_trial_gates(scenario, direction, settled=False)


@pytest.mark.parametrize('scenario', CROSSING_SCENARIOS)
@pytest.mark.parametrize('failure', ['map_age', 'commands', 'live', 'center_margin', 'samples', 'body', 'stop', 'arrival'])
def test_flat_and_ramp_crossings_share_all_physical_safety_gates(scenario, failure):
    assert evaluate_trial_gates(scenario)
    changes = {'map_age': dict(map_age_p95=.701),
               'commands': dict(node=SimpleNamespace(map_samples=10, commands=4, live_center_samples=3)),
               'live': dict(node=SimpleNamespace(map_samples=10, commands=5, live_center_samples=2)),
               'center_margin': dict(physical_separations=[1.049]*20),
               'samples': dict(physical_separations=[1.05]*19),
               'body': dict(body_clearance_ok=False), 'arrival': dict(truth_error=.301),
               'stop': dict(summary=dict(truth_z_rise_max_m=.45, truth_z_drop_max_m=0.,
                                        truth_pitch_max_rad=.2, truth_roll_max_rad=.1, front_stop_commands=0))}
    assert not evaluate_trial_gates(scenario, **changes[failure])


@pytest.mark.parametrize('scenario', CROSSING_SCENARIOS)
def test_both_crossings_enforce_held_velocity_encounter_and_prompt_truth_stop(scenario):
    args = SimpleNamespace(scenario=scenario, ramp_direction='uphill', reach_distance=.3,
                           require_dynamic_encounter=True)
    assert evaluate_trial_gates(scenario, args=args)
    for changes in (dict(encounter=None), dict(encounter=dict(minimum_clearance_m=.10)),
                    dict(event_response=None),
                    dict(event_response=dict(first_zero_command_delay_s=None, first_sustained_truth_stop_delay_s=.5)),
                    dict(event_response=dict(first_zero_command_delay_s=.601, first_sustained_truth_stop_delay_s=.5)),
                    dict(event_response=dict(first_zero_command_delay_s=.6, first_sustained_truth_stop_delay_s=None))):
        assert not evaluate_trial_gates(scenario, args=args, **changes)


@pytest.mark.parametrize('start_x,x,heading', [(0., 2., 0.), (7., 5., math.pi)])
def test_stop_response_progress_is_forward_in_both_travel_directions(start_x, x, heading):
    node = SimpleNamespace(recording=False, truth=SimpleNamespace(pose=SimpleNamespace(
        pose=SimpleNamespace(position=SimpleNamespace(x=x, y=0.)))), truth_start=(start_x, 0., .35),
        truth_yaw_start=heading, command_history=[], commands=0, steering_commands=0)
    navigation_trial.Trial.on_command(node, SimpleNamespace(linear=SimpleNamespace(x=0.), angular=SimpleNamespace(z=0.)))
    assert node.command_history[-1][2] == pytest.approx(2.)


def test_ramp_crossing_live_observations_are_counted_at_the_slope_obstacle():
    grid = SimpleNamespace(info=SimpleNamespace(resolution=.1, width=120, height=60,
        origin=SimpleNamespace(position=SimpleNamespace(x=0., y=-3.))),
        header=SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=0)), data=[0]*(120*60))
    grid.data[30*120+35] = 100
    for scenario, expected in (('ramp_crossing', 1), ('crossing', 0)):
        node = SimpleNamespace(scenario=scenario, obstacle_x=3.5 if scenario == 'ramp_crossing' else 5.,
                               vacated_stamp=None, live_center_samples=0, live_peak_cells=0)
        navigation_trial.Trial.on_live(node, grid)
        assert node.live_center_samples == expected
