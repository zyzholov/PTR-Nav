"""Motor target dynamics only; does not claim a physical stopping distance."""
import ast
import math
from pathlib import Path
import importlib.util
from types import SimpleNamespace
import numpy as np

import pytest


tree=ast.parse((Path(__file__).resolve().parents[3]/
    'src/isaac_sim_hunter_livo2/scripts/hunter_livo2_isaac_scene.py').read_text())
definition=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='limited_drive_speed')
namespace={'math':math}
exec(compile(ast.Module(body=[definition],type_ignores=[]),'<actual drive helper>','exec'),namespace)
step=namespace['limited_drive_speed']


@pytest.mark.parametrize('sign',[-1,1])
def test_braking_faster_than_acceleration_without_speed_or_pose_teleport(sign):
    assert step(0,sign*1.2,.1,1.2,2.4)==pytest.approx(sign*.12)
    assert step(sign*1.2,0,.1,1.2,2.4)==pytest.approx(sign*.96)
    v=sign*1.2
    for _ in range(100):v=step(v,0,.005,1.2,2.4)
    assert abs(v)<1e-12


@pytest.mark.parametrize('sign',[-1,1])
def test_reverse_command_brakes_before_accelerating_in_new_direction(sign):
    assert step(sign*.12,-sign*1,.025,1.2,2.4)==pytest.approx(sign*.06)
    assert step(sign*.12,-sign*1,.1,1.2,2.4)==pytest.approx(-sign*.06)
    assert step(sign*.12,-sign*.02,.1,1.2,2.4)==pytest.approx(-sign*.02)


@pytest.mark.parametrize('values',[(0,1,-1,1,2),(0,1,.1,0,2),(0,1,.1,1,0),(math.nan,1,.1,1,2)])
def test_invalid_ramp_rejected(values):
    with pytest.raises(ValueError):step(*values)


def test_runtime_command_uses_both_limits():
    source=ast.unparse(next(n for n in tree.body if isinstance(n,ast.FunctionDef)
        and n.name=='cmd_vel_drive_command'))
    assert 'limited_drive_speed' in source and 'args.cmd_vel_max_decel' in source


@pytest.mark.parametrize('sign',[-1,1])
def test_emergency_motor_target_ramp_uses_finite_stronger_deceleration(sign):
    v=sign*1.2
    assert step(v,0,.1,1.2,3.6)==pytest.approx(sign*.84)
    for _ in range(67):v=step(v,0,.005,1.2,3.6)
    assert abs(v)<1e-12


module_spec=importlib.util.spec_from_file_location('wheel_control',Path(__file__).resolve().parents[3]/
    'src/isaac_sim_hunter_livo2/scripts/wheel_control.py')
servo=importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(servo)


@pytest.mark.parametrize('load',[-20.,-6.,6.,20.])
@pytest.mark.parametrize('inertia',[.4,1.4])
def test_zero_velocity_integral_balances_load_without_raising_motor_limit(load,inertia):
    # Analytical wheel/load sandbox only; not tire contact or a stopping-distance test.
    velocity=integral=0.
    torques=[]
    for _ in range(2400):
        torque,integral=servo.wheel_velocity_pi(0.,velocity,integral,.005,60.,40.,45.)
        velocity+=(torque-load)/inertia*.005
        torques.append(torque)
    assert abs(velocity)<.0005
    assert abs(integral-load)<.03
    assert max(map(abs,torques))<=45.


def test_saturated_wheel_cannot_wind_up_and_can_release_with_changed_command():
    integral=0.
    for _ in range(2000):
        torque,integral=servo.wheel_velocity_pi(10.,0.,integral,.005,60.,40.,45.)
    assert torque==45. and integral==0.
    torque,integral=servo.wheel_velocity_pi(-1.,0.,integral,.005,60.,40.,45.)
    assert torque==-45. and integral==0.
    assert servo.wheel_velocity_pi(0.,0.,4.,.005,60.,0.,45.)==(0.,0.)


@pytest.mark.parametrize('values',[(0.,0.,0.,0.,1.,1.,45.),(0.,math.nan,0.,.005,1.,1.,45.),
    (0.,0.,0.,.005,-1.,1.,45.),(0.,0.,0.,.005,1.,1.,-1.)])
def test_nonphysical_servo_input_rejected(values):
    with pytest.raises(ValueError):servo.wheel_velocity_pi(*values)


@pytest.mark.parametrize('idle_limit',[0.,12.,100.])
def test_idle_limit_never_disables_default_braking_or_exceeds_original_motor(idle_limit):
    definition=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='wheel_effort_limit')
    namespace=dict(args=SimpleNamespace(wheel_motor_effort=45.,wheel_idle_brake_effort=idle_limit,
        powered_wheels='rear'),FRONT_DRIVE_JOINTS=())
    exec(compile(ast.Module(body=[definition],type_ignores=[]),'<actual motor limit>','exec'),namespace)
    assert namespace['wheel_effort_limit']('rear',idle_brake=True)==min(45.,idle_limit or 45.)


def test_indexed_action_changes_only_drive_targets_and_preserves_combined_force_submission():
    definition=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='apply_hunter_joint_action')
    calls=[]
    class View:
        def set_joint_position_targets(self,values,**kw):calls.append(('position_target',values,kw))
        def set_joint_velocity_targets(self,values,**kw):calls.append(('velocity_target',values,kw))
        def set_joint_efforts(self,values,**kw):calls.append(('efforts',values,kw))
    class Articulation:
        _articulation_view=View()
        def apply_action(self,action):calls.append(('compatibility',action))
    namespace=dict(args=SimpleNamespace(joint_action_interface='targets'),np=np,ArticulationAction=SimpleNamespace)
    exec(compile(ast.Module(body=[definition],type_ignores=[]),'<actual indexed action>','exec'),namespace)
    apply=namespace['apply_hunter_joint_action'];joints=np.array([2,3],np.int32)
    apply(Articulation(),joint_positions=[.1,.2],joint_velocities=[0.,0.],joint_indices=joints)
    assert [x[0] for x in calls]==['position_target','velocity_target']
    np.testing.assert_array_equal(calls[0][1],np.array([[.1,.2]],np.float32))
    assert calls[0][2]['joint_indices'] is joints
    calls.clear();apply(Articulation(),joint_efforts=[12.,-10.],joint_indices=joints)
    assert len(calls)==1 and calls[0][0]=='efforts'
    np.testing.assert_array_equal(calls[0][1],[[12.,-10.]])
    namespace['args'].joint_action_interface='controller';calls.clear()
    apply(Articulation(),joint_velocities=[1.,2.],joint_indices=joints)
    assert len(calls)==1 and calls[0][0]=='compatibility'
    assert calls[0][1].joint_velocities==[1.,2.]


def test_exact_target_repeats_are_suppressed_but_changes_overlap_efforts_and_interfaces_are_not():
    definition=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='apply_hunter_joint_action')
    calls=[]
    class View:
        fail=False
        def set_joint_position_targets(self,values,**kw):
            calls.append(('position',values.copy(),kw))
            if self.fail:raise RuntimeError('failed write')
        def set_joint_velocity_targets(self,values,**kw):calls.append(('velocity',values.copy(),kw))
        def set_joint_efforts(self,values,**kw):calls.append(('efforts',values.copy(),kw))
    class Articulation:
        def __init__(self):self._articulation_view=View()
        def apply_action(self,action):calls.append(('compatibility',action))
    arguments=SimpleNamespace(joint_action_interface='targets')
    namespace=dict(args=arguments,np=np,ArticulationAction=SimpleNamespace)
    exec(compile(ast.Module(body=[definition],type_ignores=[]),'<actual indexed action>','exec'),namespace)
    apply=namespace['apply_hunter_joint_action'];car=Articulation();a=np.array([2,3],np.int32);b=np.array([5],np.int32)
    apply(car,joint_positions=[.1,.2],joint_indices=a)
    apply(car,joint_positions=[.1,.2],joint_indices=a.copy())
    assert len(calls)==1
    apply(car,joint_positions=[.3],joint_indices=b)
    apply(car,joint_positions=[.1,.2],joint_indices=a)
    assert len(calls)==2  # Disjoint DOFs do not invalidate each other's unchanged targets.
    apply(car,joint_positions=[.4],joint_indices=np.array([2],np.int32))
    apply(car,joint_positions=[.1,.2],joint_indices=a)
    assert len(calls)==4  # Overlapping subset DID invalidate the previous target.
    for _ in range(2):apply(car,joint_efforts=[4.,5.],joint_indices=a)
    assert [r[0]for r in calls[-2:]]==['efforts','efforts']
    arguments.joint_action_interface='controller';apply(car,joint_positions=[.4,.5],joint_indices=a)
    arguments.joint_action_interface='targets';apply(car,joint_positions=[.1,.2],joint_indices=a)
    assert calls[-2][0]=='compatibility' and calls[-1][0]=='position'
    car._articulation_view=View();apply(car,joint_positions=[.1,.2],joint_indices=a)
    assert calls[-1][0]=='position'
    car._articulation_view.fail=True
    with pytest.raises(RuntimeError):apply(car,joint_positions=[.5,.6],joint_indices=a)
    car._articulation_view.fail=False
    before=len(calls);apply(car,joint_positions=[.1,.2],joint_indices=a)
    assert len(calls)==before+1  # A failed setter cannot leave a cached permit.
