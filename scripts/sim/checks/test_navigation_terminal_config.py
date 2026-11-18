from pathlib import Path

import yaml


def test_estimated_terminal_gate_leaves_margin_for_independent_truth_accuracy():
    root = Path(__file__).resolve().parents[3]
    params = yaml.safe_load((root / 'src/planner/terrain_nav/config/hunter_gazebo.yaml').read_text())
    config = next(iter(params.values()))['ros__parameters']
    assert 0. < config['goal_tolerance'] <= config['goal_reached_distance'] <= .12
    assert config['goal_reached_distance'] < .30


def test_sim_collision_contract_has_full_body_and_one_explicit_margin():
    root = Path(__file__).resolve().parents[3]
    params = yaml.safe_load((root / 'src/planner/terrain_nav/config/hunter_gazebo.yaml').read_text())
    config = next(iter(params.values()))['ros__parameters']
    # The raw SEB launch retains its legacy risk model; the enhanced launch
    # explicitly enables probability constraints on top of this same geometry.
    assert config['chance_constrained'] is False
    launch = (root / 'src/planner/terrain_nav/launch/fsgp_bgk_terrain_nav.launch.py').read_text()
    assert 'DeclareLaunchArgument("chance_constrained", default_value="true")' in launch
    assert config['body_collision_front'] == .58
    assert config['body_collision_back'] == .48
    assert config['body_collision_half_width'] == .40
    assert config['footprint_margin'] == .10
    assert config['vehicle_envelope_height_m'] == .65
    assert config['overhead_clearance_margin_m'] == .10
    # Collision is full-body vs occupied squares, not a point robot requiring
    # another half-width inflation or a second hard distance-transform reserve.
    assert config['live_obstacle_inflation_radius'] == 0.
    assert config['hard_clearance_floor'] == 0.
    assert config['clearance_distance'] > config['footprint_margin']


def test_height_slice_is_explicitly_sim_only_and_user_overridable():
    import ast
    root = Path(__file__).resolve().parents[3]
    launch = ast.parse((root / 'src/fsgp_bgk_se2/launch/terrain_analysis.launch.py').read_text())
    declarations = [node for node in ast.walk(launch) if isinstance(node, ast.Call) and
                    isinstance(node.func, ast.Name) and node.func.id == 'DeclareLaunchArgument']
    declaration = next(node for node in declarations if node.args and
                       ast.literal_eval(node.args[0]) == 'overhead_filter_enabled')
    default = next(value.value for value in declaration.keywords if value.arg == 'default_value')
    assert ast.literal_eval(default) == 'false'
    wrapper = (root / 'scripts/sim/start_terrain_analysis.sh').read_text()
    assert 'hunter_arg_value overhead_filter_enabled true "${ARGS[@]}"' in wrapper
    assert wrapper.index('ARGS+=("overhead_filter_enabled:=') < wrapper.index('hunter_source_ros sim')


def test_body_bounds_enclose_all_authored_sim_perimeter_colliders():
    import ast
    root = Path(__file__).resolve().parents[3]
    config = next(iter(yaml.safe_load((root / 'src/planner/terrain_nav/config/hunter_gazebo.yaml')
                                    .read_text()).values()))['ros__parameters']
    tree = ast.parse((root / 'src/isaac_sim_hunter_livo2/scripts/hunter_livo2_isaac_scene.py').read_text())
    function = next(node for node in tree.body if isinstance(node,ast.FunctionDef) and
                    node.name=='add_hunter_auxiliary_colliders')
    specs = next(node.value for node in function.body if isinstance(node,ast.Assign) and
                 any(isinstance(target,ast.Name) and target.id=='collider_specs' for target in node.targets))
    for name, position, size in ast.literal_eval(specs):
        assert position[0]+size[0]/2 <= config['body_collision_front']+1e-9, name
        assert -position[0]+size[0]/2 <= config['body_collision_back']+1e-9, name
        assert abs(position[1])+size[1]/2 <= config['body_collision_half_width']+1e-9, name
