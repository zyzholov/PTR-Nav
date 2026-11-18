import importlib.util
from pathlib import Path
from launch import LaunchContext
from launch.actions import DeclareLaunchArgument,OpaqueFunction
import pytest

path=Path(__file__).resolve().parents[3]/'src/planner/terrain_nav/launch/fsgp_bgk_terrain_nav.launch.py'
spec=importlib.util.spec_from_file_location('planning_motion_launch',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def test_auto_preserves_custom_yaml_and_explicit_overrides_are_numeric(monkeypatch):
    monkeypatch.setattr(module,'Node',lambda **kwargs:kwargs)
    context=LaunchContext();context.launch_configurations['params']='/custom/vehicle.yaml'
    actions=module.generate_launch_description().entities
    for action in actions:
        if isinstance(action,DeclareLaunchArgument):action.execute(context)
    opaque=next(a for a in actions if isinstance(a,OpaqueFunction))
    node=opaque.execute(context)[0]
    assert node['parameters'][0].perform(context)=='/custom/vehicle.yaml'
    assert node['parameters'][-1]=={}
    context.launch_configurations['dynamic_yield_actuator_delay_sec']='0.20'
    context.launch_configurations['live_motion_joint_risk_limit']='0.05'
    assert opaque.execute(context)[0]['parameters'][-1]==dict(dynamic_yield_actuator_delay_sec=.2,live_motion_joint_risk_limit=.05)
    context.launch_configurations['unclassified_encounter_speed_mps']='0.60'
    assert opaque.execute(context)[0]['parameters'][-1]['unclassified_encounter_speed_mps']==.6
    context.launch_configurations['live_motion_track_known_slow_centers']='true'
    assert opaque.execute(context)[0]['parameters'][-1]['live_motion_track_known_slow_centers'] is True
    context.launch_configurations['live_motion_track_known_slow_centers']='false'
    assert opaque.execute(context)[0]['parameters'][-1]['live_motion_track_known_slow_centers'] is False
    context.launch_configurations['unclassified_encounter_model_release']='true'
    assert opaque.execute(context)[0]['parameters'][-1]['unclassified_encounter_model_release'] is True
    context.launch_configurations['live_motion_gaussian_acceleration_envelope']='true'
    assert opaque.execute(context)[0]['parameters'][-1]['live_motion_gaussian_acceleration_envelope'] is True

@pytest.mark.parametrize('key,value',[('dynamic_yield_actuator_delay_sec','nan'),
    ('dynamic_yield_actuator_delay_sec','-1'),('live_motion_joint_risk_limit','0'),
    ('live_motion_joint_risk_limit','inf'),('unclassified_encounter_speed_mps','nan'),
    ('unclassified_encounter_speed_mps','-0.1'),('unclassified_encounter_speed_mps','6'),
    ('live_motion_track_known_slow_centers','1'),('live_motion_track_known_slow_centers','True'),
    ('unclassified_encounter_model_release','TRUE'),('live_motion_gaussian_acceleration_envelope','1')])
def test_invalid_override_cannot_reach_planner(key,value):
    with pytest.raises(ValueError):module.optional_motion_parameters({key:value})


def test_raw_motion_surface_override_is_typed_and_does_not_change_auto_yaml():
    name='live_motion_raw_surface_fit'
    assert module.optional_motion_parameters({name:'auto'})=={}
    assert module.optional_motion_parameters({name:'true'})=={name:True}
    assert module.optional_motion_parameters({name:'false'})=={name:False}
    with pytest.raises(ValueError):module.optional_motion_parameters({name:'1'})

def test_observation_scope_is_an_explicit_boolean_and_preserves_auto_yaml():
    name='dynamic_observed_only'
    assert module.optional_motion_parameters({name:'auto'})=={}
    assert module.optional_motion_parameters({name:'true'})=={name:True}
    assert module.optional_motion_parameters({name:'false'})=={name:False}
    with pytest.raises(ValueError):module.optional_motion_parameters({name:'1'})

def test_operating_speed_is_explicit_typed_and_bounded_by_vehicle_datasheet():
    assert module.optional_motion_parameters({'max_speed':'auto'})=={}
    assert module.optional_motion_parameters({'max_speed':'1.8'})=={'max_speed':1.8}
    for value in ('0','-1','nan','inf','6'):
        with pytest.raises(ValueError):module.optional_motion_parameters({'max_speed':value})

@pytest.mark.parametrize('name',['dynamic_yield_speed_dead_time_sec','dynamic_yield_steering_dead_time_sec'])
def test_independent_pure_dead_times_preserve_inheritance_and_zero_delay(name):
    assert module.optional_motion_parameters({name:'auto'})=={}
    assert module.optional_motion_parameters({name:'-1'})=={name:-1.}
    assert module.optional_motion_parameters({name:'0'})=={name:0.}
    assert module.optional_motion_parameters({name:'.06'})=={name:.06}
    for value in ('-.5','-2','nan','inf','1.01'):
        with pytest.raises(ValueError):module.optional_motion_parameters({name:value})
