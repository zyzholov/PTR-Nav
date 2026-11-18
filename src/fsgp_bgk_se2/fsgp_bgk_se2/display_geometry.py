"""Geometry for display/measurement only; never consumed by navigation."""
import math
import numpy as np


def project_path(x, y, xyz):
    points=np.asarray(xyz,dtype=float)
    if points.ndim!=2 or points.shape[0]<2 or points.shape[1]!=3 or not np.isfinite(points).all():
        return None
    a,b=points[:-1],points[1:]
    delta=b[:,:2]-a[:,:2]
    length2=np.sum(delta*delta,axis=1)
    valid=length2>1e-10
    if not valid.any(): return None
    u=np.clip(np.sum((np.array([x,y])-a[:,:2])*delta,axis=1)/np.maximum(length2,1e-10),0,1)
    q=a+u[:,None]*(b-a)
    distance2=np.sum((q[:,:2]-np.array([x,y]))**2,axis=1)
    distance2[~valid]=np.inf
    index=int(np.argmin(distance2))
    return float(math.sqrt(distance2[index])),q[index]


def command_projection(x,y,yaw,speed,steering,wheel_base,horizon=1.0,steps=20):
    if not all(math.isfinite(v) for v in (x,y,yaw,speed,steering,wheel_base,horizon)) or wheel_base<=0:
        return []
    rate=speed*math.tan(steering)/wheel_base
    dt=horizon/max(1,steps)
    result=[(x,y)]
    for _ in range(max(1,steps)):
        x+=speed*math.cos(yaw+.5*rate*dt)*dt
        y+=speed*math.sin(yaw+.5*rate*dt)*dt
        yaw+=rate*dt
        result.append((x,y))
    return result
