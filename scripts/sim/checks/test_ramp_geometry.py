import ast
import argparse
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

SCENE_PATH = Path(__file__).resolve().parents[3] / 'src/isaac_sim_hunter_livo2/scripts/hunter_livo2_isaac_scene.py'
sys.path.insert(0, str(SCENE_PATH.parent))
from navigation_courses import (COMPOSITE_SCENARIOS, CROSSING_SCENARIOS, RAMP_SCENARIOS,
                                course_obstacles, default_obstacle_trigger_distance)


def scene_functions(*names, **overrides):
    scope = dict(args=SimpleNamespace(scene_profile='light', nav_test_scenario='ramp_crossing',
                                     nav_test_ramp_rise=.45, nav_test_trigger_distance=2., robot_z=.35),
                 math=math, is_mars_profile=lambda: False, RAMP_SCENARIOS=RAMP_SCENARIOS,
                 CROSSING_SCENARIOS=CROSSING_SCENARIOS, COMPOSITE_SCENARIOS=COMPOSITE_SCENARIOS,
                 course_obstacles=course_obstacles, log=lambda _: None)
    scope.update(overrides)
    functions = [node for node in ast.parse(SCENE_PATH.read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(functions) == len(names)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(SCENE_PATH), 'exec'), scope)
    return scope


def test_ramp_spawn_matches_the_same_surface_used_by_collision_mesh():
    path = Path(__file__).resolve().parents[3] / 'src/isaac_sim_hunter_livo2/scripts/hunter_livo2_isaac_scene.py'
    tree = ast.parse(path.read_text())
    functions = [item for item in tree.body if isinstance(item, ast.FunctionDef)
                 and item.name in ('scene_robot_base_z', 'navigation_ramp_height')]
    scope = dict(args=SimpleNamespace(scene_profile='light', nav_test_scenario='ramp', robot_z=.35,
                                     nav_test_ramp_rise=.45),
                 is_mars_profile=lambda:False, RAMP_SCENARIOS=RAMP_SCENARIOS)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), scope)
    height, spawn = scope['navigation_ramp_height'], scope['scene_robot_base_z']
    assert height(0, 0) == -.02
    assert height(7, 0) == pytest.approx(.43)
    for x in (0, 1.5, 3.5, 5.5, 7):
        assert spawn(x, 0) - height(x, 0) == pytest.approx(.37)
    scope['args'].nav_test_scenario = 'step'
    assert spawn(7, 0) == .35


def test_static_and_moving_obstacle_truth_share_the_course_coordinates():
    path = Path(__file__).resolve().parents[3] / 'src/isaac_sim_hunter_livo2/scripts/hunter_livo2_isaac_scene.py'
    tree = ast.parse(path.read_text())
    functions = [item for item in tree.body if isinstance(item,ast.FunctionDef)
                 and item.name=='navigation_obstacle_initial_xy']
    scope={}
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),scope)
    pose=scope['navigation_obstacle_initial_xy']
    assert pose('detour')==(3.,0.)
    assert pose('clearance_guard')==(3.,0.)
    assert pose('clearance')==(5.,0.)
    assert pose('crossing')==pose('emerge')==(5.,3.2)


def test_explicit_crossing_trigger_does_not_add_a_clearance_guard_obstacle():
    path=Path(__file__).resolve().parents[3]/'src/isaac_sim_hunter_livo2/scripts/hunter_livo2_isaac_scene.py'
    tree=ast.parse(path.read_text())
    candidates=[node for node in ast.walk(tree) if isinstance(node,ast.If) and
        any(isinstance(x,ast.Constant) and x.value=='/StaticGuard' for x in ast.walk(node))]
    guard=min(candidates,key=lambda node:len(list(ast.walk(node))))
    trigger=next(node for node in ast.walk(tree) if isinstance(node,ast.If) and node.body and
        isinstance(node.body[0],ast.Assign) and any(isinstance(t,ast.Name) and t.id=='NAV_TEST_TRIGGERED'
            for t in node.body[0].targets))
    for scenario in ('crossing','ramp_crossing','clearance_guard','detour'):
        scope={'args':SimpleNamespace(nav_test_scenario=scenario), 'CROSSING_SCENARIOS': CROSSING_SCENARIOS}
        condition=lambda node:eval(compile(ast.Expression(node.test),str(path),'eval'),scope)
        assert condition(guard)==(scenario=='clearance_guard')
        assert condition(trigger)==(scenario in (*CROSSING_SCENARIOS,'clearance_guard'))


@pytest.mark.parametrize('rise', [.1, .45, 1.2])
def test_four_metre_ramp_has_the_analytic_smoothstep_slope(rise):
    scope = scene_functions('navigation_ramp_height')
    scope['args'].nav_test_ramp_rise = rise
    height = scope['navigation_ramp_height']
    assert height(1.5, 0.) == pytest.approx(-.02)
    assert height(5.5, 0.) == pytest.approx(rise-.02)
    assert height(3.5, 0.) == pytest.approx(rise/2.-.02)
    for u in (.1, .25, .5, .75, .9):
        x, eps = 1.5+4.*u, 1e-5
        derivative = (height(x+eps, 0.)-height(x-eps, 0.))/(2.*eps)
        assert derivative == pytest.approx(6.*rise*u*(1.-u)/4., abs=1e-9)
    assert height(-1., 0.) == height(1.5, 0.)
    assert height(7., 0.) == height(5.5, 0.)
    peak_slope = (height(3.5+1e-5, 0.)-height(3.5-1e-5, 0.))/2e-5
    assert math.degrees(math.atan(peak_slope)) == pytest.approx(math.degrees(math.atan(1.5*rise/4.)))
    if rise == 1.2:
        assert math.degrees(math.atan(peak_slope)) == pytest.approx(24.2277, abs=.0001)


@pytest.mark.parametrize('scenario', RAMP_SCENARIOS)
@pytest.mark.parametrize('rise', [.1, .45, 1.2])
def test_ramp_mesh_spawn_landmarks_and_obstacles_use_the_same_ground(scenario, rise):
    cylinders, boxes, meshes, rigid_flags = [], [], [], []
    prim = SimpleNamespace(GetPrim=lambda: 'moving cylinder')
    rigid = SimpleNamespace(CreateRigidBodyEnabledAttr=lambda value: rigid_flags.append(('rigid', value)),
                            CreateKinematicEnabledAttr=lambda value: rigid_flags.append(('kinematic', value)))
    scope = scene_functions('navigation_ramp_height', 'scene_robot_base_z', 'navigation_obstacle_initial_xy',
                            'navigation_obstacle_center_z', 'add_navigation_test_course', 'add_project_map',
                            define_cylinder=lambda *values: cylinders.append(values) or prim,
                            define_box=lambda *values: boxes.append(values),
                            add_heightfield_terrain=lambda *values: meshes.append(values),
                            resolve_map_asset=lambda _: None,
                            UsdPhysics=SimpleNamespace(RigidBodyAPI=SimpleNamespace(Apply=lambda _: rigid)))
    scope['args'].nav_test_scenario = scenario
    scope['args'].nav_test_ramp_rise = rise
    scope['args'].map_asset = 'none'
    stage = SimpleNamespace(DefinePrim=lambda *args: None)
    scope['add_project_map'](stage, {})
    scope['add_navigation_test_course'](stage, {})
    height = scope['navigation_ramp_height']
    assert len(meshes) == 1 and meshes[0][2:5] == (30., 150, height)
    for x in (0., 1.5, 3.5, 5.5, 7.):
        assert scope['scene_robot_base_z'](x, 0.)-height(x, 0.) == pytest.approx(.37)
    for values in boxes:
        x, y, z = values[2]
        assert z-height(x, y) == pytest.approx(.85)
    assert len(cylinders) == (0 if scenario == 'ramp' else 1)
    for values in cylinders:
        x, y, z = values[2]
        assert x == 3.5 and 1.5 < x < 5.5
        assert y == (3.2 if scenario == 'ramp_crossing' else 0.)
        assert (values[3], values[4], values[6]) == (.65, 1.8, True)
        assert z-values[4]/2. == pytest.approx(height(x, y))
    assert rigid_flags == ([('rigid', True), ('kinematic', True)] if scenario == 'ramp_crossing' else [])


@pytest.mark.parametrize('spawn_x,heading', [(0., 0.), (7., math.pi)])
def test_ramp_crossing_trigger_uses_spawn_heading_and_supports_explicit_trigger(spawn_x, heading):
    scope = scene_functions('navigation_obstacle_should_start', NAV_TEST_TRIGGERED=False,
                            scene_spawn_xy=lambda: (spawn_x, 0.), scene_spawn_yaw_rad=lambda: heading)
    start = scope['navigation_obstacle_should_start']
    assert not start((spawn_x, 0., 0.), 1.)
    assert not start((spawn_x+1.9*math.cos(heading), 0., 0.), 1.)
    assert not start((spawn_x, 4., 0.), 1.)
    assert not start((spawn_x-3.*math.cos(heading), 0., 0.), 1.)
    position = (spawn_x+2.*math.cos(heading), 0., 0.)
    assert not start(position, .2)
    assert start(position, .21)
    assert 1.5 < position[0] < 5.5
    scope['NAV_TEST_TRIGGERED'] = True
    assert start((spawn_x, 0., 0.), 0.)


@pytest.mark.parametrize('spawn_x,heading', [(0., 0.), (7., math.pi)])
@pytest.mark.parametrize('rise', [.1, .45, 1.2])
def test_ramp_crossing_runtime_motion_and_truth_stay_grounded(spawn_x, heading, rise):
    translations, published, logs = [], [], []
    position = [spawn_x+2.*math.cos(heading), 0., 0.]
    scope = scene_functions('navigation_ramp_height', 'navigation_obstacle_center_z',
                            'navigation_obstacle_should_start', 'publish_navigation_obstacle_pose',
                            NAV_TEST_TRIGGERED=False, scene_spawn_xy=lambda: (spawn_x, 0.),
                            scene_spawn_yaw_rad=lambda: heading)
    scope['args'].motion_model = 'physics'
    scope['args'].nav_test_ramp_rise = rise
    scope.update(stage=SimpleNamespace(GetPrimAtPath=lambda _: 'moving cylinder'),
                 hunter_articulation=SimpleNamespace(get_world_pose=lambda: (position, None)),
                 speed=.5, emerge_started_at=None, emerge_finished=False,
                 nav_obstacle_x=3.5, nav_obstacle_y=3.2, log=logs.append,
                 UsdGeom=SimpleNamespace(XformCommonAPI=lambda _: SimpleNamespace(SetTranslate=translations.append)),
                 Gf=SimpleNamespace(Vec3d=lambda *values: values),
                 NAV_OBSTACLE_PUBLISHER=SimpleNamespace(publish=published.append),
                 NAV_OBSTACLE_MSG_CLS=lambda: SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace()),
                     pose=SimpleNamespace(position=SimpleNamespace(), orientation=SimpleNamespace())))
    main = next(node for node in ast.parse(SCENE_PATH.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == 'main')
    motion = next(node for node in ast.walk(main) if isinstance(node, ast.If) and node.body
                  and isinstance(node.body[0], ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == 'robot_position'
                          for target in node.body[0].targets))
    code = compile(ast.Module(body=[motion], type_ignores=[]), str(SCENE_PATH), 'exec')
    for stamp, expected_y in ((10., 3.2), (11.2, 0.), (12.4, -3.2)):
        scope['t'] = stamp
        exec(code, scope)
        x, y, z = translations[-1]
        assert x == 3.5 and y == pytest.approx(expected_y)
        assert z-.9 == pytest.approx(scope['navigation_ramp_height'](x, y))
        scope['publish_navigation_obstacle_pose'](stamp, y, x)
        assert vars(published[-1].pose.position) == {'x': x, 'y': y, 'z': z}
        assert published[-1].header.frame_id == 'world'
    assert scope['emerge_finished']
    assert logs[-1] == 'navigation test obstacle has crossed the lane'


def scene_navigation_args(argv):
    tree = ast.parse(SCENE_PATH.read_text())
    parse = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'parse_args')
    options = ('--nav-test-scenario', '--nav-test-ramp-rise', '--nav-test-trigger-distance')
    additions = [node for node in parse.body if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                 and any(isinstance(value, ast.Constant) and value.value in options for value in node.value.args)]
    scope = dict(parser=argparse.ArgumentParser(), COMPOSITE_SCENARIOS=COMPOSITE_SCENARIOS,
                 CROSSING_SCENARIOS=CROSSING_SCENARIOS, math=math,
                 default_obstacle_trigger_distance=default_obstacle_trigger_distance)
    exec(compile(ast.Module(body=additions, type_ignores=[]), str(SCENE_PATH), 'exec'), scope)
    scope['args'] = scope['parser'].parse_args(argv)
    validations = [node for node in tree.body if isinstance(node, ast.If)
                   and any(isinstance(value, ast.Attribute) and value.attr in
                           ('nav_test_ramp_rise', 'nav_test_trigger_distance') for value in ast.walk(node.test))]
    exec(compile(ast.Module(body=validations, type_ignores=[]), str(SCENE_PATH), 'exec'), scope)
    return scope['args']


@pytest.mark.parametrize('rise', [.1, .45, 1.2])
def test_scene_accepts_ramp_rise_endpoints_and_resolves_slope_trigger(rise):
    args = scene_navigation_args(['--nav-test-scenario', 'ramp_crossing', '--nav-test-ramp-rise', str(rise)])
    assert args.nav_test_ramp_rise == rise and args.nav_test_trigger_distance == 2.
    assert scene_navigation_args(['--nav-test-scenario', 'crossing']).nav_test_trigger_distance == 1.1
    assert scene_navigation_args([]).nav_test_ramp_rise == .45


@pytest.mark.parametrize('value', ['nan', 'inf', '-inf', '0', '-.1', '.099', '1.201'])
def test_scene_rejects_invalid_ramp_rise_before_isaac_is_started(value):
    with pytest.raises(ValueError, match='nav-test-ramp-rise'):
        scene_navigation_args(['--nav-test-scenario', 'ramp_crossing', '--nav-test-ramp-rise='+value])


@pytest.mark.parametrize('value', ['nan', 'inf', '.49', '1.1', '1.5', '3.51'])
def test_scene_rejects_numeric_crossing_triggers_outside_slope_encounter(value):
    with pytest.raises(ValueError, match='trigger'):
        scene_navigation_args(['--nav-test-scenario', 'ramp_crossing', '--nav-test-trigger-distance', value])
