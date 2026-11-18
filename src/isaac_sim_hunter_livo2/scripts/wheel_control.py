"""Bounded wheel servo; torques act through joints, never through base poses."""
import math


def ackermann_joint_targets(speed,steer,wheelbase,front_x,front_y,rear_y,joint_limit):
    """One common ICR, with actual steering-pivot / rear wheel coordinates.

    steer is the EQUIVALENT bicycle angle, not an inner-wheel joint limit.
    Mechanical saturation limits the common curvature before deriving either
    joint target. Front rolling speeds include their lateral axle velocity.
    """
    values=(speed,steer,wheelbase,*front_x,*front_y,*rear_y,joint_limit)
    if (not all(math.isfinite(v) for v in values) or abs(steer)>=math.pi/2 or wheelbase<=0 or
            len(front_x)!=2 or len(front_y)!=2 or len(rear_y)!=2 or min(front_x)<=0 or
            not front_y[0]>0>front_y[1] or not rear_y[0]>0>rear_y[1] or not 0<joint_limit<math.pi/2):
        raise ValueError('invalid Ackermann joint geometry')
    tangent=math.tan(joint_limit)
    positive=min(tangent/(x+y*tangent) for x,y in zip(front_x,front_y) if x+y*tangent>0)
    negative=min(tangent/(x-y*tangent) for x,y in zip(front_x,front_y) if x-y*tangent>0)
    curvature=max(-negative,min(positive,math.tan(steer)/wheelbase))
    angles=tuple(math.atan2(curvature*x,1-curvature*y) for x,y in zip(front_x,front_y))
    front_speed=tuple(speed*math.hypot(1-curvature*y,curvature*x) for x,y in zip(front_x,front_y))
    rear_speed=tuple(speed*(1-curvature*y) for y in rear_y)
    return angles,front_speed,rear_speed,math.atan(curvature*wheelbase)


def wheel_velocity_pi(target, measured, previous_integral, dt, kp, ki,
                      torque_limit, integral_decay=0.0):
    """One conditional-integration PI step, in rad/s, seconds and N m.

    A zero speed request is a velocity hold, not a request to remove the
    motor. Keeping its integral permits balancing a slope load without a
    permanent speed error. The caller can leak the integral during motion.
    """
    values = (target, measured, previous_integral, dt, kp, ki, torque_limit, integral_decay)
    if (not all(math.isfinite(v) for v in values) or dt <= 0 or kp < 0 or ki < 0
            or torque_limit < 0 or integral_decay < 0):
        raise ValueError('invalid wheel servo input')
    clamp = lambda value: max(-torque_limit, min(torque_limit, value))
    error = target - measured
    previous = clamp(previous_integral * math.exp(-integral_decay * dt)) if ki > 0 else 0.0
    integral = clamp(previous + ki * error * dt)
    unsaturated = kp * error + integral
    # Do not accumulate against a saturated motor; permit integration that
    # unwinds saturation. The returned integral is bounded by the same motor.
    if abs(unsaturated) > torque_limit and unsaturated * error > 0:
        integral = previous
    return clamp(kp * error + integral), integral
