from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
from launch import LaunchContext
from launch.utilities import perform_substitutions
from launch_ros.actions import Node


@pytest.mark.parametrize("filename", ["terrain_analysis.launch.py",
    "hunter_fsgp_bgk_nav.launch.py", "fsgp_bgk_se2.launch.py"])
@pytest.mark.parametrize("override", [None, "SYNCHRONOUS"])
def test_async_transport_is_scoped_to_map_node_and_respects_override(filename, override, monkeypatch):
    spec=spec_from_file_location("terrain_launch",Path(__file__).parents[1]/"launch"/filename)
    module=module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.delenv("RMW_FASTRTPS_PUBLICATION_MODE",raising=False)
    if override is not None:
        monkeypatch.setenv("RMW_FASTRTPS_PUBLICATION_MODE",override)
    context=LaunchContext()
    found=0
    for action in module.generate_launch_description().entities:
        if not isinstance(action,Node):continue
        env={perform_substitutions(context,key):perform_substitutions(context,value)
             for key,value in (action.additional_env or [])}
        if action.node_executable=="fsgp_bgk_se2_node":
            found+=1
            assert env["RMW_FASTRTPS_PUBLICATION_MODE"]==(override or "ASYNCHRONOUS")
        else:
            assert "RMW_FASTRTPS_PUBLICATION_MODE" not in env
    assert found==1


@pytest.mark.parametrize('model,sim,expected',[
    ('marginal','true',False),('shared_gp','true',True),
    ('marginal','false',False),('shared_gp','false',False)])
def test_lower_ground_preservation_is_paired_with_simulated_shared_model(model,sim,expected):
    from launch_ros.utilities import evaluate_parameters
    spec=spec_from_file_location('ground_launch',Path(__file__).parents[1]/'launch/terrain_analysis.launch.py')
    module=module_from_spec(spec);spec.loader.exec_module(module)
    context=LaunchContext()
    context.launch_configurations.update(probability_model=model,use_sim_time=sim,
        terrain_dynamic_clear_enabled='true',terrain_dynamic_clear_frames='3')
    gpu=next(a for a in module.generate_launch_description().entities
             if isinstance(a,Node) and a.node_executable=='terrain_analyzer_gpu')
    parameters=evaluate_parameters(context,[gpu._Node__parameters[-1]])[0]
    assert parameters['elevation.preserve_measured_ground'] is expected


@pytest.mark.parametrize('filename',['terrain_analysis.launch.py',
    'hunter_fsgp_bgk_nav.launch.py','fsgp_bgk_se2.launch.py'])
@pytest.mark.parametrize('settings,expected',[
    ({},('1','1','1')),
    ({'HUNTER_TERRAIN_CPU_THREADS':'4'},('4','4','4')),
    ({'HUNTER_FSGP_CPU_THREADS':'2'},('2','2','2')),
    ({'HUNTER_FSGP_CPU_THREADS':'2','HUNTER_TERRAIN_CPU_THREADS':'4'},('2','2','2')),
    ({'HUNTER_TERRAIN_CPU_THREADS':'4','OPENBLAS_NUM_THREADS':'3'},('3','4','4')),
    ({'OPENBLAS_NUM_THREADS':'16'},('16','1','1')),
])
def test_small_matrix_thread_budget_is_scoped_to_fsgp_node(filename,settings,expected,monkeypatch):
    spec=spec_from_file_location('thread_launch',Path(__file__).parents[1]/'launch'/filename)
    module=module_from_spec(spec);spec.loader.exec_module(module)
    keys=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')
    for name in (*keys,'HUNTER_FSGP_CPU_THREADS','HUNTER_TERRAIN_CPU_THREADS'):
        monkeypatch.delenv(name,raising=False)
    for name,value in settings.items():monkeypatch.setenv(name,value)
    context=LaunchContext();found=0
    for action in module.generate_launch_description().entities:
        if not isinstance(action,Node):continue
        env={perform_substitutions(context,k):perform_substitutions(context,v)
             for k,v in (action.additional_env or [])}
        for key,value in zip(keys,expected):
            if action.node_executable=='fsgp_bgk_se2_node':assert env[key]==value
            else:assert key not in env
        found+=action.node_executable=='fsgp_bgk_se2_node'
    assert found==1
