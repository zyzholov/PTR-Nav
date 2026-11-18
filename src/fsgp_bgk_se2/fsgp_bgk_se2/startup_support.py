"""Explicit simulation-only stable-stance assumption; never a measured layer."""
from dataclasses import dataclass,replace
import math
import numpy as np
from scipy.spatial import ConvexHull,QhullError


@dataclass
class Stance:
    x: float
    y: float
    yaw: float
    roll: float
    pitch: float
    speed: float
    stamp: float
    z: float | None = None


class StartupSupport:
    def __init__(self,allow_one_sided=False,require_goal=False,require_geometry=False,retire_after_departure=False):
        self.allow_one_sided=allow_one_sided
        self.require_goal=require_goal;self.armed=not require_goal
        self.require_geometry=require_geometry;self.footprint=None
        self.retire_after_departure=retire_after_departure;self.retired=False
        self.raster_extra=None
        self.raster_area=None
        self.reference=None;self.anchor=None;self.since=None;self.last_stamp=-math.inf
        self.plane=None;self.variance=None;self.revoked=False;self.reason="waiting for stable stance"
        self.measurement_sigma=None;self.plane_weights=None

    def set_footprint(self,front,back,half_width):
        """Freeze the planner's already protected body envelope; no extra margin."""
        values=(front,back,half_width)
        if not np.isfinite(values).all() or min(values)<=0 or max(values)>3.:
            raise ValueError('invalid initial protected body footprint')
        if self.footprint is not None and not np.allclose(values,self.footprint,atol=1e-8,rtol=0):
            if self.anchor is not None:
                self.revoked=True;self.reason='initial body footprint contract changed; revoked';return
            self.reference=None;self.since=None
        self.footprint=values

    def arm_initial_goal(self):
        # The first requested navigation goal starts the stationary evidence
        # window. Early SLAM settling must not freeze a stale map-frame anchor.
        # Later goals cannot reset, move or restore this one initial hypothesis.
        if not self.revoked and self.anchor is None:self.armed=True

    def observe_command(self,speed):
        if self.require_goal and self.anchor is None and (not math.isfinite(speed) or abs(speed)>.01):
            self.revoked=True;self.reason='motion before initial stance qualification; no assumption'

    def speed_limits(self,x,y):
        """Bootstrap slowdown only; normal terrain/braking limits still apply."""
        limit=np.full_like(x,100.,dtype=np.float32)
        if self.anchor is not None and not self.retired:
            limit[np.hypot(x-self.anchor.x,y-self.anchor.y)<1.5]=.25
        return limit

    def update(self,height,variance,observed,collision,x,y,pose,stamp,params):
        mask=np.zeros_like(observed,dtype=bool)
        if stamp<self.last_stamp:
            self.__init__(self.allow_one_sided,self.require_goal,self.require_geometry,self.retire_after_departure)
        self.last_stamp=stamp
        if self.revoked or self.retired:return mask,None,None
        if not self.armed:
            self.reason='waiting for first navigation goal before fixed stance qualification'
            return mask,None,None
        if self.require_geometry and self.footprint is None:
            self.reason='waiting for planner protected body geometry';return mask,None,None
        if self.anchor is None:
            # Fresh odometry is needed to establish the stance, not to move a
            # frozen world-frame hypothesis. Current observations still revoke it.
            if (pose is None or not np.isfinite([pose.x,pose.y,pose.yaw,pose.roll,pose.pitch,pose.speed,pose.stamp]).all()
                    or abs(stamp-pose.stamp)>.3):
                self.reason="waiting for fresh finite stance";return mask,None,None
            stable=abs(pose.speed)<=.03 and max(abs(pose.roll),abs(pose.pitch))<=.20
            if self.reference is not None:
                stable &= math.hypot(pose.x-self.reference.x,pose.y-self.reference.y)<=.035
                stable &= abs(math.atan2(math.sin(pose.yaw-self.reference.yaw),math.cos(pose.yaw-self.reference.yaw)))<=.025
            if not stable or self.reference is None:
                self.reference=pose;self.since=stamp
                return mask,None,None
            if stamp-self.since<1.5:return mask,None,None
            dx=x-pose.x;dy=y-pose.y;radius=np.hypot(dx,dy)
            ring=observed & ~collision & np.isfinite(height) & np.isfinite(variance) & (variance>=0)
            ring &= (radius>=.65)&(radius<=2.)
            if pose.z is not None:
                if not math.isfinite(pose.z):
                    self.reason="nonfinite Hunter stance height";return mask,None,None
                # A vertical surface can leave a single low return in a cell
                # labelled ground. Only fit sources compatible with the known
                # body-to-contact height and measured attitude. This selects
                # plane evidence; it never clears that cell's collision/risk.
                # The final full-footprint contradiction check below still
                # revokes the hypothesis for a real step or pit under the car.
                c,s=math.cos(pose.yaw),math.sin(pose.yaw)
                u=c*dx+s*dy;v=-s*dx+c*dy
                relative=height+math.tan(pose.pitch)*u-math.tan(pose.roll)*v
                ring &= (relative>=pose.z-.40)&(relative<=pose.z-.18)
            if np.count_nonzero(ring)<24:
                self.reason="insufficient surrounding measured ground";return mask,None,None
            try:
                enclosing=np.max(ConvexHull(np.c_[dx[ring],dy[ring]]).equations[:,2])<=-.1
            except QhullError:
                enclosing=False
            design=np.c_[dx[ring],dy[ring],np.ones(np.count_nonzero(ring))]
            if not enclosing:
                # Explicit simulation-only extension of the fixed stable-stance
                # hypothesis for a tilted sensor's one-sided blind region.
                # Never infer this permission merely from missing observations.
                c,s=math.cos(pose.yaw),math.sin(pose.yaw)
                u=c*dx[ring]+s*dy[ring];v=-s*dx[ring]+c*dy[ring]
                if (not self.allow_one_sided or pose.z is None or
                        np.ptp(u)<.5 or np.ptp(v)<.8 or np.linalg.matrix_rank(design)<3):
                    self.reason="surrounding observations do not enclose stance";return mask,None,None
            coefficient=np.linalg.lstsq(design,height[ring],rcond=None)[0]
            residual=height[ring]-design@coefficient
            if (np.quantile(np.abs(residual),.95)>.05 or
                    np.hypot(*coefficient[:2])>math.tan(.20)):
                self.reason="surrounding terrain is not locally planar";return mask,None,None
            if not enclosing:
                c,s=math.cos(pose.yaw),math.sin(pose.yaw)
                along=coefficient[0]*c+coefficient[1]*s
                across=-coefficient[0]*s+coefficient[1]*c
                if (abs(math.atan(along)+pose.pitch)>.08 or
                        abs(math.atan(across)-pose.roll)>.08 or np.quantile(np.abs(residual),.95)>.03):
                    self.reason="one-sided ground disagrees with stable measured attitude";return mask,None,None
            if pose.z is not None and (not math.isfinite(pose.z) or not .18<=pose.z-coefficient[2]<=.40):
                self.reason="Hunter body clearance disagrees with surrounding ground";return mask,None,None
            # A nearest-cell anchor shifts (rather than rasterizes) the real
            # stance, dropping intersecting cells on one side of a grid seam.
            self.anchor=replace(pose);self.plane=coefficient
            self.plane_weights=np.linalg.pinv(design)
            self.measurement_sigma=np.sqrt(variance[ring])
            self.variance=.01**2
            self.reason=("conditional initial-footprint assumption; not measured" if enclosing else
                         "conditional one-sided initial-footprint assumption; not measured")
        dx=x-self.anchor.x;dy=y-self.anchor.y
        area=np.zeros_like(observed,dtype=bool)
        front,back,width=self.footprint or (params.front+params.margin,params.back+params.margin,params.half_width+params.margin)
        # Freeze the raster footprint including the same small heading jitter
        # accepted by the stability check, especially across the 0/2pi seam.
        for angle in (self.anchor.yaw-.025,self.anchor.yaw,self.anchor.yaw+.025):
            yaw=(math.floor((angle%(2*math.pi))/(2*math.pi/params.yaw_bins))+.5)*2*math.pi/params.yaw_bins
            c,s=math.cos(yaw),math.sin(yaw);u=c*dx+s*dy;v=-s*dx+c*dy
            pad=.5*params.resolution*(abs(c)+abs(s))
            area |= (u>=-back-pad)&(u<=front+pad)&(np.abs(v)<=width+pad)
        body_area=area.copy()
        if self.footprint is not None:
            # Preserve the real body raster AND the finite source cells used
            # by a pose query at this fixed initial stance. Unioning is needed
            # at cell seams; snapping the physical hypothesis would shift it.
            from .probabilistic_risk import _geometry
            rr=int(math.floor((self.anchor.y-(y[0,0]-.5*params.resolution))/params.resolution))
            cc=int(math.floor((self.anchor.x-(x[0,0]-.5*params.resolution))/params.resolution))
            for angle in (self.anchor.yaw-.025,self.anchor.yaw,self.anchor.yaw+.025):
                k=int((angle%(2*math.pi))/(2*math.pi/params.yaw_bins))
                for dr,dc in _geometry(params,k)[0]:
                    r,c=rr+dr,cc+dc
                    if 0<=r<area.shape[0] and 0<=c<area.shape[1]:area[r,c]=True
        self.raster_extra=area&~body_area
        self.raster_area=area.copy()
        if self.retire_after_departure and pose is not None and np.isfinite(
                [pose.x,pose.y,pose.yaw,pose.stamp]).all() and abs(stamp-pose.stamp)<=.3:
            # This is a bootstrap hypothesis, not a permanently frozen height
            # island in the SLAM map. Retire it only after the CURRENT contact
            # stencil, including its heading tolerance, has NO overlap with
            # the ENTIRE fixed initial raster. Nothing moves with the car.
            # A later return must obtain ordinary measured/GP certificates.
            from .probabilistic_risk import _geometry
            rr=int(math.floor((pose.y-(y[0,0]-.5*params.resolution))/params.resolution))
            cc=int(math.floor((pose.x-(x[0,0]-.5*params.resolution))/params.resolution))
            inside=True;overlap=False
            for angle in (pose.yaw-.025,pose.yaw,pose.yaw+.025):
                k=int((angle%(2*math.pi))/(2*math.pi/params.yaw_bins))
                for dr,dc in _geometry(params,k)[0]:
                    r,c=rr+dr,cc+dc
                    if not (0<=r<area.shape[0] and 0<=c<area.shape[1]):inside=False
                    elif area[r,c]:overlap=True
            if inside and not overlap:
                self.retired=True;self.reason='initial bootstrap retired after full contact departure; ordinary support required'
                return mask,None,None
        predicted=self.plane[0]*dx+self.plane[1]*dy+self.plane[2]
        contradicted=area & (collision | (observed & np.isfinite(height) & (np.abs(height-predicted)>.06)))
        if contradicted.any():
            self.revoked=True;self.reason="initial assumption contradicted; revoked"
            return mask,None,None
        # Frozen world coordinates, not a free bubble following the robot.
        mask=area & ~observed & ~collision
        model_variance=np.full_like(height,self.variance)
        query=np.c_[dx[area],dy[area],np.ones(np.count_nonzero(area))]
        # Conditional on the explicit planar-stance assumption, measurement
        # errors may be arbitrarily correlated. No 1/sqrt(N) confidence gain.
        original_sigma=np.sqrt(self.measurement_sigma**2+params.height_std_floor**2/params.variance_scale)
        std_bound=np.abs(query@self.plane_weights)@original_sigma
        # The consumer adds sigma0 in quadrature. Inflate its stored marginal
        # so the resulting sigma instead bounds shared-source + model errors
        # by the triangle inequality, without assuming their independence.
        model_variance[area]=np.maximum(self.variance,std_bound**2+
            2*std_bound*params.height_std_floor/np.sqrt(params.variance_scale))
        return mask,predicted,model_variance
