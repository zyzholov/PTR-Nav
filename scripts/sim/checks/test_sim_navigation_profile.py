"""A module preset must keep producer/receiver assumptions aligned and opt-in."""
from pathlib import Path
import json
import subprocess
import pytest
import yaml

ROOT=Path(__file__).resolve().parents[3]


def expand(stage,*args):
    return subprocess.run(['bash','-c',
        'ROOT_DIR="$1"; source "$ROOT_DIR/scripts/lib/sim_navigation_profile.sh"; shift; '
        'hunter_sim_navigation_profile "$@" || exit $?; printf "%s\\0" "${HUNTER_STAGE_ARGS[@]}"',
        'profile',str(ROOT),stage,*args],capture_output=True)


@pytest.mark.parametrize('stage',['simulation','mapping','terrain','planning'])
def test_no_profile_leaves_original_arguments_and_real_defaults_alone(stage):
    result=expand(stage,'a:=one','a:=two','space value')
    assert result.returncode==0
    assert result.stdout.split(b'\0')[:-1]==[b'a:=one',b'a:=two',b'space value']
    assert not result.stderr


@pytest.mark.parametrize('stage',['simulation','mapping','terrain','planning'])
def test_preset_is_explicit_and_caller_overrides_come_last(stage):
    result=expand(stage,'navigation_profile:=neos_joint_sim','caller:=kept')
    manifest=json.loads((ROOT/'scripts/sim/config/neos_joint_sim.json').read_text())
    assert result.returncode==0
    assert result.stdout.split(b'\0')[:-1]==[v.encode() for v in manifest['stages'][stage]+['caller:=kept']]
    assert b'not calibrated real-vehicle risk' in result.stderr


def test_preset_permissions_match_the_producer_and_never_expand_into_real_wrappers():
    manifest=json.loads((ROOT/'scripts/sim/config/neos_joint_sim.json').read_text())['stages']
    terrain=dict(s.split(':=',1)for s in manifest['terrain'] if ':=' in s)
    planning=dict(s.split(':=',1)for s in manifest['planning'] if ':=' in s)
    assert terrain['probability_model']=='shared_gp'
    assert terrain['fsgp_processing_executor']=='single_ingress'
    # A ceiling matched to the 10Hz producer, not a measured throughput claim.
    assert float(terrain['fsgp_max_update_rate_hz'])==10.
    # Real acquisition stamps include one 100ms RTX render interval.
    assert float(terrain['fsgp_max_input_source_age_sec'])==.16
    assert float(terrain['fsgp_max_output_source_age_sec'])==.30
    assert float(terrain['fsgp_output_delivery_reserve_sec'])==.015
    assert terrain['shared_gp_independent_model_floor']==planning['chance_allow_independent_model_floor']=='true'
    assert terrain['probability_use_gp_completion']==planning['chance_allow_gp_completion']=='true'
    assert terrain['startup_support_mode']=='stable_stance' and planning['chance_allow_startup_prior']=='true'
    for path in (ROOT/'scripts/real').glob('*.sh'):
        assert 'source "$ROOT_DIR/scripts/lib/sim_navigation_profile.sh"' not in path.read_text()


def test_simulation_actuator_slew_matches_the_selected_navigation_motion_model():
    manifest=json.loads((ROOT/'scripts/sim/config/neos_joint_sim.json').read_text())['stages']
    arguments=manifest['simulation']
    planner=yaml.safe_load((ROOT/'src/planner/terrain_nav/config/hunter_gazebo.yaml').read_text())['terrain_nav_node']['ros__parameters']
    assert float(arguments[arguments.index('--cmd-vel-max-decel')+1])==planner['dynamic_yield_min_braking_decel']
    assert float(arguments[arguments.index('--cmd-vel-max-steer-rate')+1])==planner['max_steer_rate']
    # Clamps align commanded references; they do not assert that tyre contact,
    # slopes or torque limits make measured physical motion exact.


@pytest.mark.parametrize('name',['unknown','../../arbitrary','neos_joint_sim;exit 0'])
def test_unknown_presets_are_rejected_as_data_without_evaluation(name):
    result=expand('terrain','navigation_profile:='+name)
    assert result.returncode==2 and b'Unknown simulation' in result.stderr
