"""Ground-relative navigation height envelope; never modifies the LiDAR map.

Only an enclosed locally planar measured floor can qualify an overhead slice.
Current raw floor returns remain mandatory; explicitly age-bounded real floor
witnesses may bridge the moving chassis' temporary occlusion. They never add
terrain support. Missing/discontinuous ground retains legacy checks.
This is a point-cloud envelope, not proof that unsampled volume is empty.
"""
import math
import numpy as np
from scipy.ndimage import minimum_filter,distance_transform_edt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from .obstacle_columns import vertical_columns


class FloorReferenceHistory:
    """Static-floor references only; NEVER terrain/GP support or obstacle clearing.

    Bound this independent reference cache to 1.5s by default, revalidate
    against measured height, and add an explicit temporal height reserve.
    Larger geometric blind spots require enclosing floor observations over a
    larger spatial neighbourhood, not indefinite reuse of old measurements.
    Frame/time/localization discontinuities invalidate the cache.
    """
    def __init__(self,max_age_s=1.5,voxel_m=.10):
        if not 0<max_age_s<=2 or not 0<voxel_m<=.20:raise ValueError('invalid floor-reference bounds')
        self.max_age_s=max_age_s;self.voxel_m=voxel_m
        self.points=np.empty((0,3),np.float32);self.stamps=np.empty(0)
        self.frame=None;self.stamp=None;self.pose=None

    def update(self,points,stamp,frame,pose_xy,source_age_s=0.):
        if (not np.isfinite([stamp,source_age_s,*pose_xy]).all() or source_age_s<0 or not frame):
            raise ValueError('invalid floor-reference coordinates/time')
        if (frame!=self.frame or self.stamp is not None and stamp<self.stamp or
                self.pose is not None and np.linalg.norm(np.asarray(pose_xy)-self.pose)>1.):
            self.points=np.empty((0,3),np.float32);self.stamps=np.empty(0)
        incoming=np.asarray(points,np.float32)
        if incoming.ndim!=2 or incoming.shape[1]!=3:raise ValueError('expected measured floor XYZ')
        incoming=incoming[np.isfinite(incoming).all(axis=1)]
        values=np.r_[self.points,incoming];stamps=np.r_[self.stamps,np.full(len(incoming),stamp-source_age_s)]
        keep=(stamp-stamps>=0)&(stamp-stamps<=self.max_age_s)
        values,stamps=values[keep],stamps[keep]
        if len(values):
            cells=np.floor(values[:,:2]/self.voxel_m).astype(np.int64)
            order=np.lexsort((-stamps,cells[:,0],cells[:,1]))
            first=np.r_[True,np.any(np.diff(cells[order],axis=0),axis=1)]
            ids=order[first];values,stamps=values[ids],stamps[ids]
        # An unexpectedly huge geometry fails closed instead of silently
        # dropping evidence and pretending a complete reference was retained.
        if len(values)>100000:values=np.empty((0,3),np.float32);stamps=np.empty(0)
        self.points,self.stamps=values,stamps
        self.frame,self.stamp,self.pose=frame,stamp,np.asarray(pose_xy).copy()
        return values.copy(),stamp-stamps


def floor_witness_points(elevation,observed,points,origin,resolution):
    """Actual returns agreeing with measured ground; no GP or stance heights."""
    if points is None:return np.empty((0,3),np.float32)
    z=np.asarray(elevation);p=np.asarray(points)
    p=p[np.isfinite(p[:,:3]).all(axis=1),:3]
    col=np.floor((p[:,0]-origin[0])/resolution).astype(int)
    row=np.floor((p[:,1]-origin[1])/resolution).astype(int)
    inside=(row>=0)&(row<z.shape[0])&(col>=0)&(col<z.shape[1])
    row,col,p=row[inside],col[inside],p[inside]
    floor=np.asarray(observed,bool)&np.isfinite(z)
    lower=minimum_filter(np.where(floor,z,np.inf),size=3,mode='constant',cval=np.inf)
    # Thin wall/step-top cells are not floor-plane witnesses when adjacent
    # lower measured ground exposes a >18cm discontinuity. Their measurements
    # and raw returns stay intact for collision and contradiction checks.
    compatible=floor&(z-lower<=.18)
    agrees=compatible[row,col]&(np.abs(p[:,2]-z[row,col])<=.05)
    return p[agrees]


def measured_ceiling(elevation, observed, points, origin, resolution, *,
                     vehicle_height_m=.65, margin_m=.10, reference_points=None,reference_error_m=0.,backend='cpu',
                     reference_radius_m=.9,max_residual_m=.04,measured_fallback=False):
    z=np.asarray(elevation,dtype=np.float32)
    if (z.ndim!=2 or np.shape(observed)!=z.shape or not math.isfinite(resolution) or resolution<=0 or
            not math.isfinite(vehicle_height_m) or vehicle_height_m<=0 or
            not math.isfinite(margin_m) or margin_m<.05 or
            not math.isfinite(reference_error_m) or reference_error_m<0 or backend not in ('cpu','cupy') or
            not math.isfinite(reference_radius_m) or not .9<=reference_radius_m<=3. or
            not math.isfinite(max_residual_m) or not .04<=max_residual_m<=.10 or
            not isinstance(measured_fallback,bool)):
        raise ValueError('invalid measured vehicle-height envelope')
    ceiling=np.full(z.shape,np.inf,np.float32)
    if points is None or min(z.shape)<3:return ceiling
    p=floor_witness_points(z,observed,points,origin,resolution)
    if len(p)<8:return ceiling
    if reference_points is not None:
        # Revalidate historical returns against today's measured height. A
        # changed floor cannot survive solely through cached witnesses.
        p=np.r_[p,floor_witness_points(z,observed,reference_points,origin,resolution)]
    col=np.floor((p[:,0]-origin[0])/resolution).astype(int)
    row=np.floor((p[:,1]-origin[1])/resolution).astype(int)
    witnessed=np.zeros(z.shape,bool)
    floor=np.asarray(observed,bool)&np.isfinite(z)
    witnessed[row,col]=True
    direct_floor=witnessed.copy()
    # A wall's lowest hit may vary gradually between cells, so height-component
    # membership alone cannot identify it as floor. Its current vertical return
    # column is independent evidence that the cell must not condition a floor
    # plane. Keep the original measured height for the target consistency check.
    xp=np
    if backend=='cupy':
        import cupy as xp
    columns,_=vertical_columns(points,z.shape,origin,resolution,min_hits=1,
                               ground_elevation=np.where(floor,z,np.nan),xp=xp)
    if xp is not np:columns=xp.asnumpy(columns)
    witnessed &= ~columns
    if np.count_nonzero(witnessed)<8:return ceiling
    # Interpolate a local measured plane only inside enclosing raw floor
    # witnesses. Sparse scans need not hit a cross of exactly adjacent cells.
    # This reference is NEVER inserted into terrain/GP ground support.
    # Height continuity separates a raised wall/step top from its adjacent
    # floor. A single global least-squares plane through all nearby "ground"
    # labels fails even when the actual supporting floor is perfectly planar.
    ids=np.arange(z.size).reshape(z.shape)
    horizontal=floor[:,:-1]&floor[:,1:]&(np.abs(z[:,:-1]-z[:,1:])<=.18)
    vertical=floor[:-1,:]&floor[1:,:]&(np.abs(z[:-1,:]-z[1:,:])<=.18)
    source=np.r_[ids[:,:-1][horizontal],ids[:-1,:][vertical]]
    dest=np.r_[ids[:,1:][horizontal],ids[1:,:][vertical]]
    graph=coo_matrix((np.ones(len(source)),(source,dest)),shape=(z.size,z.size)).tocsr()
    _,labels=connected_components(graph,directed=False)
    labels=labels.reshape(z.shape)
    nearest=distance_transform_edt(~floor,return_distances=False,return_indices=True)
    owner=labels[nearest[0],nearest[1]]
    evidence=(z,floor,witnessed,direct_floor,columns,labels,owner,resolution,
              vehicle_height_m,margin_m,reference_error_m,backend)
    if backend=='cupy':
        from .overhead_clearance_cuda import resident_plane_envelopes
        return resident_plane_envelopes(*evidence[:-1],reference_radius_m,max_residual_m,
                                       measured_fallback=measured_fallback)
    local=_prepared_envelope(*evidence,reference_radius_m,max_residual_m)
    missing=~np.isfinite(local)&floor
    if not measured_fallback or not missing.any():return local
    # The two radii must use identical raw witnesses and height components.
    # Reuse that evidence within this callback only; never cache it across
    # scans or skip a current contradiction to reduce computation.
    # Only these targets consume the broad fallback. Its normal-transfer
    # seeds still require a complete .4m halo; no source witness is removed.
    broad=_prepared_envelope(*evidence,3.,.10,query_mask=missing)
    return np.where(missing,broad,local)


def _prepared_envelope(z,floor,witnessed,direct_floor,columns,labels,owner,resolution,
                       vehicle_height_m,margin_m,reference_error_m,backend,
                       reference_radius_m,max_residual_m,query_mask=None):
    ceiling=np.full(z.shape,np.inf,np.float32)
    if backend=='cupy':
        from .overhead_clearance_cuda import plane_envelope
        return plane_envelope(z,floor,witnessed,direct_floor,columns,labels,owner,resolution,
                              vehicle_height_m,margin_m,reference_error_m,reference_radius_m,max_residual_m,
                              query_mask=query_mask)
    radius=max(1,int(np.ceil(reference_radius_m/resolution)))
    matrix=np.zeros((*z.shape,3,3),np.float64)
    rhs=np.zeros((*z.shape,3),np.float64)
    quadrants=np.zeros((*z.shape,4),np.int32)
    shifts=[]
    for dy in range(-radius,radius+1):
        for dx in range(-radius,radius+1):
            if np.hypot(dx,dy)*resolution>reference_radius_m:continue
            y0,y1=max(0,-dy),min(z.shape[0],z.shape[0]-dy)
            x0,x1=max(0,-dx),min(z.shape[1],z.shape[1]-dx)
            if y1<=y0 or x1<=x0:continue
            target=np.s_[y0:y1,x0:x1];source=np.s_[y0+dy:y1+dy,x0+dx:x1+dx]
            shifts.append((target,source,dx*resolution,dy*resolution))
    for target,source,u,v in shifts:
        weight=witnessed[source]&(labels[source]==owner[target])
        height=np.where(weight,z[source],0.)
        terms=(1.,u,v)
        for i,a in enumerate(terms):
            rhs[target+(i,)]+=height*a
            for j,b in enumerate(terms):
                if j>=i:matrix[target+(i,j)]+=weight*(a*b)
        for k,(sx,sy) in enumerate(((-1,-1),(1,-1),(-1,1),(1,1))):
            if u*sx>0 and v*sy>0:quadrants[target+(k,)]+=weight
    for i in range(3):
        for j in range(i):matrix[...,i,j]=matrix[...,j,i]
    qualified=(matrix[...,0,0]>=8)&np.all(quadrants>0,axis=-1)&(np.linalg.det(matrix)>1e-6)
    model=np.zeros((*z.shape,3),np.float64)
    model[qualified]=np.linalg.solve(matrix[qualified],rhs[qualified][...,None])[...,0]
    residual=np.zeros(z.shape,np.float64)
    for target,source,u,v in shifts:
        local=model[target]
        error=np.abs(z[source]-(local[...,0]+local[...,1]*u+local[...,2]*v))
        compatible=witnessed[source]&(labels[source]==owner[target])
        residual[target]=np.maximum(residual[target],np.where(compatible,error,0.))
    slope2=model[...,1]**2+model[...,2]**2
    qualified &= (slope2<=1.)&(residual<=max_residual_m)
    qualified &= ~floor|(np.abs(model[...,0]-z)<=.05)
    # Direct target ground can use a nearby enclosing patch to estimate its
    # normal. This is not interpolation of missing height: the target needs
    # its own age-qualified real return, measured height, and local continuity.
    # Freeze the seeds so this cannot cascade into unobserved space.
    seeds=qualified.copy();seed_model=model.copy();seed_residual=residual.copy()
    interior=np.ones(z.shape,bool);interior[[0,-1],:]=False;interior[:,[0,-1]]=False
    for target,source,u,v in sorted(shifts,key=lambda item:item[2]**2+item[3]**2):
        if u*u+v*v>.4**2+1e-9:continue
        local=seed_model[source]
        predicted=local[...,0]-local[...,1]*u-local[...,2]*v
        use=(~qualified[target]&seeds[source]&floor[target]&direct_floor[target]&interior[target]
             &(np.abs(predicted-z[target])<=.04))
        if not use.any():continue
        # A low step/pit near the target must not be ignored by selecting a
        # planar patch on only one side. Tall occupied columns remain hard
        # geometry and cannot condition this ground-normal estimate.
        for oy in (-1,0,1):
            for ox in (-1,0,1):
                rr,cc=np.indices(z[target].shape);rr+=target[0].start+oy;cc+=target[1].start+ox
                inside=(rr>=0)&(rr<z.shape[0])&(cc>=0)&(cc<z.shape[1])
                rr=np.clip(rr,0,z.shape[0]-1);cc=np.clip(cc,0,z.shape[1]-1)
                known=inside&floor[rr,cc]&~columns[rr,cc]
                expected=predicted+local[...,1]*(ox*resolution)+local[...,2]*(oy*resolution)
                use &= ~known|(np.abs(z[rr,cc]-expected)<=.04)
        if not use.any():continue
        qualified[target] |= use
        model[target]=np.where(use[...,None],np.stack((predicted,local[...,1],local[...,2]),axis=-1),model[target])
        residual[target]=np.where(use,seed_residual[source],residual[target])
    slope2=model[...,1]**2+model[...,2]**2
    # A body normal-height H above a supporting plane has vertical separation
    # H/n_z at the same XY. Retain local nonplanarity and explicit headroom.
    floor_upper=np.maximum(model[...,0]+residual,np.where(floor,z,-np.inf))
    upper=floor_upper+vehicle_height_m*np.sqrt(1+slope2)+margin_m+reference_error_m
    ceiling[:]=np.where(qualified,upper,np.inf)
    return ceiling if query_mask is None else np.where(query_mask,ceiling,np.inf)


def clip_overhead(points, ceiling, origin, resolution, *, xp=np):
    if points is None:return None
    p=xp.asarray(points)
    if not p.size:return p
    col=xp.floor((p[:,0]-origin[0])/resolution).astype(xp.int32)
    row=xp.floor((p[:,1]-origin[1])/resolution).astype(xp.int32)
    inside=(row>=0)&(row<ceiling.shape[0])&(col>=0)&(col<ceiling.shape[1])
    limit=xp.asarray(ceiling)
    ids=xp.flatnonzero(inside)
    remove=xp.zeros(len(p),dtype=bool)
    remove[ids]=p[ids,2]>limit[row[ids],col[ids]]
    return p[~remove]
