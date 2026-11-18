"""Independent no-slip geometry and actual scene target submission checks.

These checks do not assert that PhysX contacts track the ideal bicycle model.
"""
import ast
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / 'src/isaac_sim_hunter_livo2/scripts'
spec = importlib.util.spec_from_file_location('ackermann_servo', SCRIPTS/'wheel_control.py')
servo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(servo)
sys.path.insert(0,str(SCRIPTS))
from vehicle_geometry import geometry,load_spec,fit_neos_urdf
TREE = ast.parse((SCRIPTS/'hunter_livo2_isaac_scene.py').read_text())


def scene_namespace(max_steer=.4):
    names = {'WHEEL_BASE', 'TRACK_WIDTH', 'FRONT_AXLE_X', 'FRONT_AXLE_Y',
             'STEERING_JOINT_LIMIT', 'WHEEL_RADIUS', 'WHEEL_JOINT_SIGNS', 'SUSPENSION_JOINTS'}
    functions = {'finite_or_zero', 'clamp_signed', 'bounded_bicycle_steer',
                 'yaw_rate_from_speed_steer', 'steer_from_speed_yaw_rate',
                 'ackermann_wheel_targets', 'apply_hunter_articulation_targets'}
    selected = [n for n in TREE.body if
        (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)) or
        (isinstance(n, ast.FunctionDef) and n.name in functions)]
    ns = dict(math=math, np=np, ackermann_joint_targets=servo.ackermann_joint_targets,
              VEHICLE_GEOMETRY=geometry(),VEHICLE_SPEC=load_spec(),
              args=SimpleNamespace(cmd_vel_max_steer=max_steer, cmd_vel_min_turn_speed=.015,
                  steering_zero_snap=0., motion_model='physics', drive_speed_feedback_kp=1.2,
                  drive_yaw_feedback_kp=2., cmd_vel_max_speed=1.5, cmd_vel_max_yaw_rate=1.,
                  physics_hz=200., steering_direct_zero_hold=False, wheel_drive_mode='velocity'),
              PHYSICS_IMU_STATE={}, hunter_rollover_interlocked=lambda _:False)
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<actual scene functions>', 'exec'), ns)
    return ns


def test_scene_geometry_matches_fitted_neos_urdf_pivots_and_mechanical_limits():
    tree = ET.fromstring(fit_neos_urdf((ROOT/'src/simulator/fake_ugv/hunter_se/hunter_se_description/urdf/hunter_se_base.xacro').read_text()))
    joints = {e.attrib['name'].removeprefix('${prefix}'):e for e in tree.iter('joint')}
    origin = lambda name:tuple(map(float, joints[name].find('origin').attrib['xyz'].split()))
    rear = [origin('re_left_joint'), origin('re_right_joint')]
    front = [origin('fr_steer_left_joint'), origin('fr_steer_right_joint')]
    ns = scene_namespace()
    assert rear[0][0] == rear[1][0]
    assert ns['FRONT_AXLE_X'] == pytest.approx([p[0]-rear[0][0] for p in front])
    assert ns['FRONT_AXLE_Y'] == pytest.approx([p[1] for p in front])
    assert ns['TRACK_WIDTH'] == pytest.approx(rear[0][1]-rear[1][1]-2*.0145)
    assert ns['WHEEL_BASE'] == pytest.approx(sum(ns['FRONT_AXLE_X'])/2)
    for name in ('fr_steer_left_joint', 'fr_steer_right_joint'):
        limits = joints[name].find('limit').attrib
        assert float(limits['upper']) == ns['STEERING_JOINT_LIMIT']
        assert float(limits['lower']) == -ns['STEERING_JOINT_LIMIT']


@pytest.mark.parametrize('speed', [-1.2, -.25, 0., .25, 1.2])
@pytest.mark.parametrize('steer', [-1.2, -.4, -.02, 0., .02, .4, 1.2])
def test_each_wheel_velocity_is_tangent_to_one_common_instantaneous_circle(speed, steer):
    ns = scene_namespace()
    x,y = ns['FRONT_AXLE_X'],ns['FRONT_AXLE_Y']
    angles, front, rear, effective = servo.ackermann_joint_targets(speed, steer,
        ns['WHEEL_BASE'], x, y, (ns['TRACK_WIDTH']/2,-ns['TRACK_WIDTH']/2), ns['STEERING_JOINT_LIMIT'])
    curvature = math.tan(effective)/ns['WHEEL_BASE']
    for i in range(2):
        # Recover ICR curvature independently from each wheel's orientation.
        recovered = math.tan(angles[i])/(x[i]+y[i]*math.tan(angles[i]))
        assert recovered == pytest.approx(curvature, abs=1e-12)
        vx,vy = front[i]*math.cos(angles[i]),front[i]*math.sin(angles[i])
        assert (vx,vy) == pytest.approx((speed-speed*curvature*y[i],speed*curvature*x[i]))
    assert rear == pytest.approx((speed-speed*curvature*ns['TRACK_WIDTH']/2,speed+speed*curvature*ns['TRACK_WIDTH']/2))
    assert max(map(abs, angles)) <= .69+1e-12
    if abs(steer)<=.4:
        assert effective == pytest.approx(steer)
    else:
        assert max(map(abs, angles)) == pytest.approx(.69)


@pytest.mark.parametrize('steer', [-math.pi/2, math.pi/2, math.nan, math.inf])
def test_invalid_bicycle_angles_cannot_wrap_to_opposite_curvature(steer):
    with pytest.raises(ValueError):
        servo.ackermann_joint_targets(.5, steer, .549185, (.54838,.54999),
                                      (.24619,-.24619), (.252,-.252), .69)


@pytest.mark.parametrize('max_steer', [.4, 1.2])
@pytest.mark.parametrize('speed,steer', [(.25,.4),(.25,-.4),(-.25,.4),(-.25,-.4),(0.,.4)])
def test_actual_feedback_submits_angles_and_rear_targets_for_same_curvature(max_steer,speed,steer):
    ns = scene_namespace(max_steer)
    names = ['fr_steer_left_joint','fr_steer_right_joint','re_left_joint','re_right_joint']
    calls = []
    ns.update(physics_body_speed_yaw_rate=lambda _:(.7*speed,.2*speed*math.tan(steer)/ns['WHEEL_BASE']),
              powered_wheel_joints=lambda:names[2:],
              apply_hunter_joint_action=lambda _,**kw:calls.append(kw))
    articulation = SimpleNamespace(dof_names=names,get_dof_index=names.index)
    ns['apply_hunter_articulation_targets'](articulation,speed,steer,.005)
    angle_call = next(c for c in calls if 'joint_positions' in c)
    wheel_call = next(c for c in calls if 'joint_positions' not in c)
    angles = angle_call['joint_positions']
    wheel_speeds = wheel_call['joint_velocities']*ns['WHEEL_RADIUS']
    velocity = sum(wheel_speeds)/2
    yaw_rate = (wheel_speeds[1]-wheel_speeds[0])/ns['TRACK_WIDTH']
    for i in range(2):
        # No lateral slip at either steering pivot for the submitted rear twist.
        assert (velocity-yaw_rate*ns['FRONT_AXLE_Y'][i])*math.sin(angles[i]) == pytest.approx(
            yaw_rate*ns['FRONT_AXLE_X'][i]*math.cos(angles[i]),abs=1e-12)
    assert max(map(abs, angles))<=.69+1e-12
    assert ns['PHYSICS_IMU_STATE']['last_drive_feedback'] == pytest.approx((velocity,yaw_rate))
    if speed==0:
        assert np.all(wheel_speeds==0) and np.all(angles==0)


def test_maximum_bicycle_command_does_not_clip_inner_joint_to_bicycle_angle():
    ns = scene_namespace()
    angles,_ = ns['ackermann_wheel_targets'](.25,.4)
    assert .4<angles['fr_steer_left_joint']<.69
    assert angles['fr_steer_right_joint']<.4


def test_yaw_rate_and_joint_helpers_share_mechanical_saturation():
    ns = scene_namespace(1.2)
    steer = ns['steer_from_speed_yaw_rate'](.25,10.)
    angles,_ = ns['ackermann_wheel_targets'](.25,steer)
    assert max(abs(angles[n]) for n in ('fr_steer_left_joint','fr_steer_right_joint')) == pytest.approx(.69)
    assert ns['yaw_rate_from_speed_steer'](.25,1.2) == pytest.approx(.25*math.tan(steer)/ns['WHEEL_BASE'])
