"""Current-scan vertical occupancy, kept separate from smoothed terrain risk."""

import numpy as np


def measured_floor_return_mask(points,elevation,variance,observed,occupied,origin,resolution,*,
                               max_step_m=.18,max_slope_rad=np.deg2rad(10.)):
    """Identify actual low returns near independent current measured floor.

    This refines a collision silhouette ONLY. It supplies no terrain/support
    value, clears no occupied coarse cell, and never uses a GP prediction.
    Six distinct fine-cell witnesses must agree in height and have 2-D spread
    inside the existing five-cell reference window. Unlike extrapolating a
    floor into empty space, each classified target is itself a measured XYZ.
    High, negative, ambiguous and unsupported returns remain obstacles.
    """
    from scipy.ndimage import minimum_filter
    from scipy.spatial import cKDTree
    raw=np.asarray(points);z=np.asarray(elevation);v=np.asarray(variance)
    obs=np.asarray(observed,bool);occ=np.asarray(occupied,bool)
    if (raw.ndim!=2 or raw.shape[1]<3 or z.ndim!=2 or
            any(a.shape!=z.shape for a in (v,obs,occ)) or
            not np.isfinite([*origin,resolution,max_step_m,max_slope_rad]).all() or
            resolution<=0 or max_step_m<=.04 or not 0<=max_slope_rad<np.pi/2):
        raise ValueError('invalid measured floor-return geometry')
    result=np.zeros(len(raw),bool)
    ids=np.flatnonzero(np.isfinite(raw[:,:3]).all(axis=1))
    xy=(raw[ids,:2].astype(np.float64)-np.asarray(origin))/resolution
    cell=np.floor(xy).astype(np.int64);rows,cols=z.shape
    inside=(cell[:,0]>=0)&(cell[:,0]<cols)&(cell[:,1]>=0)&(cell[:,1]<rows)
    ids,cell,xy=ids[inside],cell[inside],xy[inside]
    if not len(ids):return result
    r,c=cell[:,1],cell[:,0]
    reliable=obs&~occ&np.isfinite(z)&np.isfinite(v)&(v>=0)&(v<=.02**2)
    lower=minimum_filter(np.where(reliable,z,np.inf),size=3,mode='constant',cval=np.inf)
    reliable &= z-lower<=max_step_m
    source=reliable[r,c]&(np.abs(raw[ids,2]-z[r,c])<=.02)
    source_ids=ids[source]
    # Repeated rays in one fine cell cannot count as independent spatial
    # witnesses. Retain an actual return, never a synthesized centroid.
    lattice=np.floor(xy[source]*4).astype(np.int64)
    _,unique=np.unique(lattice[:,1]*(cols*4)+lattice[:,0],return_index=True)
    source_ids=source_ids[unique]
    query_ids=ids[occ[r,c]]
    if len(source_ids)<8 or not len(query_ids):return result
    distance,index=cKDTree(raw[source_ids,:2]).query(raw[query_ids,:2],k=8,
        distance_upper_bound=2*resolution)
    valid=np.isfinite(distance);index=np.minimum(index,len(source_ids)-1)
    witnesses=np.asarray(raw[source_ids[index],:3],dtype=np.float64)
    tolerance=np.minimum(.04,.02+np.tan(max_slope_rad)*distance)
    agrees=valid&(np.abs(witnesses[:,:,2]-raw[query_ids,None,2])<=tolerance)
    count=agrees.sum(axis=1);weight=agrees/np.maximum(1,count[:,None])
    delta=witnesses[:,:,:2]-raw[query_ids,None,:2]
    delta-=np.sum(weight[:,:,None]*delta,axis=1)[:,None,:]
    xx=np.sum(weight*delta[:,:,0]**2,axis=1);yy=np.sum(weight*delta[:,:,1]**2,axis=1)
    cross=np.sum(weight*delta[:,:,0]*delta[:,:,1],axis=1)
    spread=.5*(xx+yy-np.hypot(xx-yy,2*cross))
    result[query_ids[(count>=6)&(spread>(resolution/40)**2)]]=True
    return result


def measured_collision_floor_witness(elevation,variance,observed,occupied,resolution,*,max_step_m=.18):
    """Independent measured-plane reference for classifying ORIGINAL returns.

    A mixed coarse obstacle cell may have no terrain measurement even though
    its scan contains real floor. Interpolate a plane only inside the convex
    hull of >=6 reliable, nonoccupied measured neighbours, with 2-D spread and
    <=15mm residual. This is NOT a terrain observation or GP training value.
    Actual XYZ returns still have to match the plane at their own XY position.
    Missing/ambiguous references retain all obstacle witnesses.
    """
    from scipy.ndimage import correlate,minimum_filter
    z=np.asarray(elevation);v=np.asarray(variance);obs=np.asarray(observed,bool);occ=np.asarray(occupied,bool)
    if (z.ndim!=2 or v.shape!=z.shape or obs.shape!=z.shape or occ.shape!=z.shape or
            not np.isfinite([resolution,max_step_m]).all() or min(resolution,max_step_m)<=0):raise ValueError('invalid collision floor reference')
    reliable=obs&np.isfinite(z)&np.isfinite(v)&(v>=0)&(v<=.02**2)
    floor=np.where(reliable,z,np.nan).astype(np.float64)
    gradient=np.zeros((2,*z.shape),np.float64);training=reliable&~occ
    # A stored side/top can have tiny variance, yet was ALREADY rejected as
    # ground by collision_subcells against a lower measured neighbour. Do
    # that rejection BEFORE interpolation, so a valid independent floor can
    # replace the bad reference rather than leave all floor rays in the hard
    # silhouette. The measured height itself and every non-floor ray survive.
    lower=minimum_filter(np.where(training,z,np.inf),size=3,mode='constant',cval=np.inf)
    floor[occ&reliable&np.isfinite(lower)&(z-lower>max_step_m)]=np.nan
    if training.sum()<6:return floor,gradient
    dy,dx=np.mgrid[-2:3,-2:3].astype(float)*resolution;one=np.ones((5,5))
    w=training.astype(float);datum=float(np.median(z[training]));values=np.where(training,z-datum,0.)
    def sums(a,k):return correlate(a,k,mode='constant',cval=0.)
    n=sums(w,one);sx=sums(w,dx);sy=sums(w,dy);xx=sums(w,dx*dx);xy=sums(w,dx*dy);yy=sums(w,dy*dy)
    cx=xx-sx*sx/np.maximum(n,1.);cy=yy-sy*sy/np.maximum(n,1.);cxy=xy-sx*sy/np.maximum(n,1.)
    spread=.5*(cx+cy-np.sqrt(np.maximum(0.,(cx-cy)**2+4*cxy*cxy)))
    candidate=occ&~np.isfinite(floor)&(n>=6)&(spread>.1*resolution**2*n)
    if not candidate.any():return floor,gradient
    # No one-sided extrapolation of a road plane through an unseen cliff.
    window=np.lib.stride_tricks.sliding_window_view(np.pad(training,2),(5,5))[candidate].reshape(-1,25)
    angles=np.sort(np.where(window,np.arctan2(dy,dx).reshape(1,25),np.inf),axis=1)
    count=n[candidate].astype(int);last=angles[np.arange(len(count)),count-1]
    gap=np.zeros_like(angles[:,1:]);np.subtract(angles[:,1:],angles[:,:-1],
        out=gap,where=np.isfinite(angles[:,1:]))
    surrounded=np.maximum(gap.max(axis=1),angles[:,0]+2*np.pi-last)<np.pi-1e-6
    rows,cols=np.nonzero(candidate);rows=rows[surrounded];cols=cols[surrounded]
    candidate[:]=False;candidate[rows,cols]=True
    if not candidate.any():return floor,gradient
    sz=sums(values,one);zx=sums(values,dx);zy=sums(values,dy)
    rhs=np.stack((sz[candidate],zx[candidate],zy[candidate]),axis=-1)
    m=np.empty((len(rows),3,3));m[:,0,0]=n[candidate];m[:,0,1]=m[:,1,0]=sx[candidate]
    m[:,0,2]=m[:,2,0]=sy[candidate];m[:,1,1]=xx[candidate];m[:,1,2]=m[:,2,1]=xy[candidate];m[:,2,2]=yy[candidate]
    b=np.linalg.solve(m,rhs[...,None])[...,0]
    residual=np.maximum(0.,sums(values*values,one)[candidate]-np.sum(b*rhs,axis=1))/np.maximum(1.,n[candidate]-3.)
    keep=(residual<=.015**2)&(np.hypot(b[:,1],b[:,2])<=1.)
    floor[rows[keep],cols[keep]]=datum+b[keep,0]
    gradient[:,rows[keep],cols[keep]]=b[keep,1:].T
    return floor,gradient


def separate_pending_column_heights(pending,pending_elevation,pending_variance,current_columns,
                                    elevation,variance,observed,resolution,*,max_step_m,
                                    max_ground_slope_rad,ground_exclusions=None):
    """Separate a positive occupied side from an unconfirmed ground change.

    A pending height in a mixed coarse cell can be a column SIDE rather than
    the floor. Require CURRENT positive occupancy and a surrounding, independent
    measured floor reference (>=6 independent floor neighbours, 2-D convex enclosure,
    <=15mm fit residual). Even the pending height's 3-sigma LOWER bound must
    exceed that reference by the physical step limit. Pending-side noise is
    propagated into that test, rather than subjected to the 2cm FLOOR training
    threshold. Negative evidence, nonfinite/ambiguous candidates and uncertain
    references retain their veto.

    Returns an effective ground-pending mask and a provenance mask. This does
    not create observed ground, condition the GP, clear occupancy or certify a
    vehicle pose. Latent floor retains its original covariance and the native
    continuous full-body collision and chance checks remain mandatory.
    """
    pen=np.asarray(pending,bool);pz=np.asarray(pending_elevation);pv=np.asarray(pending_variance)
    col=np.asarray(current_columns,bool);z=np.asarray(elevation);v=np.asarray(variance);obs=np.asarray(observed,bool)
    exclusions=col if ground_exclusions is None else np.asarray(ground_exclusions,bool)
    if (pen.ndim!=2 or any(a.shape!=pen.shape for a in (pz,pv,col,z,v,obs,exclusions)) or
            not np.isfinite([resolution,max_step_m,max_ground_slope_rad]).all() or
            min(resolution,max_step_m,max_ground_slope_rad)<=0 or max_ground_slope_rad>=np.pi/2):
        raise ValueError('invalid pending-column separation geometry')
    from scipy.ndimage import minimum_filter,correlate
    if not np.any(pen&col&np.isfinite(pz)&np.isfinite(pv)&(pv>=0)):
        return pen.copy(),np.zeros_like(pen)
    # Physical occupancy and floor-source exclusion are DIFFERENT masks.
    # A separate original floor can exist in a mixed occupied coarse square.
    # Already-qualified real floor there must not be erased a second time.
    # The pending candidate never enters its own independent reference.
    reference_exclusions=exclusions|(pen&col)
    sources=obs&~reference_exclusions&np.isfinite(z)&np.isfinite(v)&(v>=0)&(v<=.02**2)
    local_lower=minimum_filter(np.where(sources,z,np.inf),size=3,mode='constant',cval=np.inf)
    selected=sources&(z-local_lower<=max_step_m)
    original_count=correlate(sources.astype(float),np.ones((5,5)),mode='constant',cval=0.)
    selected_count=correlate(selected.astype(float),np.ones((5,5)),mode='constant',cval=0.)
    # Reject isolated high side returns from the REFERENCE only. Retain a
    # strict majority of its original sources; rough/multimodal terrain must
    # not be flattened by cherry-picking a small lower cluster. The final
    # enclosure/spread/residual/slope checks still apply to every selected ray.
    floor,gradient=measured_collision_floor_witness(z,v,selected,reference_exclusions,resolution,max_step_m=max_step_m)
    separated=pen&col&np.isfinite(pz)&np.isfinite(pv)&(pv>=0)&np.isfinite(floor)
    separated &= np.hypot(gradient[0],gradient[1])<=np.tan(max_ground_slope_rad)
    separated &= 2*selected_count>original_count
    separated &= pz-floor-3*np.sqrt(np.maximum(0.,pv)+.015**2)>max_step_m
    return pen&~separated,separated


def separate_measured_floor_labels(columns,elevation,variance,observed,resolution,*,max_step_m=.18,
                                  max_ground_slope_rad=.55):
    """Keep independently supported measured floor in mixed column/grid cells.

    This returns a GROUND-LABEL exclusion mask, not collision occupancy. A
    coarse XY column can contain both a narrow vertical object and real floor.
    Removing that floor and predicting it again discards its likelihood and
    creates a second, coarse uncertainty halo around the exact body geometry.
    Fit only reliable NON-column observations in a 5x5 neighbourhood. A few
    undetected pole-side heights must not contaminate that reference plane.
    Select a strict majority compatible with the candidate's ORIGINAL height
    and the declared terrain slope plus bounded measurement tolerance. The
    candidate itself never enters the fit. Preserve an existing measurement
    only if this independent, well-spread, low-residual plane agrees;
    raised tops, pits, noisy/unknown cells and nonplanar support stay excluded.
    Heights, variances, observation ages and every occupied volume are intact.
    """
    col=np.asarray(columns,dtype=bool);z=np.asarray(elevation);v=np.asarray(variance);obs=np.asarray(observed,dtype=bool)
    if (col.ndim!=2 or z.shape!=col.shape or v.shape!=col.shape or obs.shape!=col.shape or
            not np.isfinite([resolution,max_step_m,max_ground_slope_rad]).all() or
            min(resolution,max_step_m,max_ground_slope_rad)<=0 or max_ground_slope_rad>=np.pi/2):
        raise ValueError('invalid measured-floor separation geometry')
    reliable=obs&np.isfinite(z)&np.isfinite(v)&(v>=0)&(v<=.02**2)
    training=reliable&~col;preserved=np.zeros_like(col)
    if np.count_nonzero(training)<6 or not np.any(col&reliable):return col.copy(),preserved
    rows,cols=np.nonzero(col&reliable)
    def windows(a,fill):
        return np.lib.stride_tricks.sliding_window_view(np.pad(a,2,constant_values=fill),(5,5))[rows,cols].reshape(-1,25)
    original=windows(training,False)
    neighbours=windows(z,0.).astype(np.float64)
    neighbour_variance=windows(v,np.inf)
    dy,dx=np.mgrid[-2:3,-2:3].astype(np.float64)*resolution
    xx,yy=dx.ravel(),dy.ravel()
    tolerance=np.minimum(min(.5*max_step_m,.05),np.maximum(.03,
        3*np.sqrt(np.maximum(0.,v[rows,cols,None]+neighbour_variance))))
    compatible=original&(np.abs(neighbours-z[rows,cols,None])<=
        np.tan(max_ground_slope_rad)*np.hypot(xx,yy)+tolerance)
    for iteration in range(3):
        weight=compatible.astype(np.float64);n=weight.sum(axis=1)
        sx=weight@xx;sy=weight@yy;xx_sum=weight@(xx*xx);xy=weight@(xx*yy);yy_sum=weight@(yy*yy)
        divisor=np.maximum(n,1.);cx=xx_sum-sx*sx/divisor;cy=yy_sum-sy*sy/divisor;cxy=xy-sx*sy/divisor
        spread=.5*(cx+cy-np.sqrt(np.maximum(0.,(cx-cy)**2+4*cxy*cxy)))
        candidate=(n>=6)&(2*n>original.sum(axis=1))&(spread>.1*resolution**2*n)
        if not candidate.any():return col.copy(),preserved
        # Subtract each measured candidate's datum only for conditioning;
        # the fit still uses independent neighbours, not the candidate.
        values=np.where(compatible,neighbours-z[rows,cols,None],0.)
        sz=values.sum(axis=1);szx=values@xx;szy=values@yy
        count=int(candidate.sum());matrix=np.empty((count,3,3))
        rhs=np.stack((sz[candidate],szx[candidate],szy[candidate]),axis=-1)
        matrix[:,0,0]=n[candidate];matrix[:,0,1]=matrix[:,1,0]=sx[candidate]
        matrix[:,0,2]=matrix[:,2,0]=sy[candidate];matrix[:,1,1]=xx_sum[candidate]
        matrix[:,1,2]=matrix[:,2,1]=xy[candidate];matrix[:,2,2]=yy_sum[candidate]
        coefficients=np.linalg.solve(matrix,rhs[...,None])[...,0]
        squared=np.square(values).sum(axis=1)[candidate]
        residual=np.maximum(0.,squared-np.sum(coefficients*rhs,axis=-1))/np.maximum(1.,n[candidate]-3.)
        if iteration==2:break
        # Bounded hard rejection of remaining pole-side outliers. Membership
        # can only shrink, requires a strict majority of ORIGINAL neighbours
        # at every refit, and never loosens the final residual/agreement tests.
        estimate=coefficients[:,0,None]+coefficients[:,1,None]*xx+coefficients[:,2,None]*yy
        bound=np.minimum(min(.5*max_step_m,.05),np.maximum(.03,
            3*np.sqrt(np.maximum(0.,neighbour_variance[candidate]))))
        selected=compatible[candidate]&(np.abs(values[candidate]-estimate)<=bound)
        if np.array_equal(selected,compatible[candidate]):break
        compatible[candidate]=selected
    tolerance=np.minimum(.5*max_step_m,np.maximum(.03,3*np.sqrt(v[rows[candidate],cols[candidate]]+residual)))
    keep=(residual<=.025**2)&(np.abs(coefficients[:,0])<=tolerance)
    preserved[rows[candidate][keep],cols[candidate][keep]]=True
    return col&~preserved,preserved


def inspect_body_collision(occupied,bits,origin,resolution,x,y,yaw,front,back,half_width,*,factor=2,bounds=None):
    """Read-only continuous-pose counterpart of native body/subcell SAT.

    Return (unsafe, known). Missing fine witnesses retain the coarse square;
    unknown occupancy or a footprint outside the map never becomes free.
    Used by the display/probe, never as a replacement for native control.
    """
    occupied=np.asarray(occupied);bits=np.asarray(bits)
    if (factor not in (2,4) or occupied.ndim!=2 or bits.shape!=occupied.shape or
            not np.isfinite([*origin,resolution,x,y,yaw,front,back,half_width]).all() or
            min(resolution,front,back,half_width)<=0):raise ValueError('invalid body inspection geometry')
    c,s=np.cos(yaw),np.sin(yaw);a=(front+back)/2;off=(front-back)/2
    cx,cy=x+off*c,y+off*s;hx=abs(c)*a+abs(s)*half_width;hy=abs(s)*a+abs(c)*half_width
    if bounds is not None and np.asarray(bounds).shape!=(factor*factor,*occupied.shape):raise ValueError('invalid subcell bounds shape')
    def overlaps(px,py,size,size_y=None):
        rx,ry=px-cx,py-cy;bx=size/2;by=(size if size_y is None else size_y)/2
        return (abs(rx)<=hx+bx)&(abs(ry)<=hy+by)&(abs(c*rx+s*ry)<=a+bx*abs(c)+by*abs(s))&(abs(-s*rx+c*ry)<=half_width+bx*abs(s)+by*abs(c))
    x0,x1=int(np.floor((cx-hx-origin[0])/resolution)),int(np.floor((cx+hx-origin[0])/resolution))
    y0,y1=int(np.floor((cy-hy-origin[1])/resolution)),int(np.floor((cy+hy-origin[1])/resolution))
    unknown=False;rows,cols=occupied.shape
    for r in range(y0,y1+1):
        for col in range(x0,x1+1):
            px,py=origin[0]+(col+.5)*resolution,origin[1]+(r+.5)*resolution
            if not overlaps(px,py,resolution):continue
            if not (0<=r<rows and 0<=col<cols) or not np.isfinite(occupied[r,col]):unknown=True;continue
            if occupied[r,col]<.5:continue
            full=(1<<(factor*factor))-1
            b=bits[r,col];b=int(b) if np.isfinite(b) and 1<=b<=full and b==int(b) else full
            for sy in range(factor):
                for sx in range(factor):
                    if not b&(1<<(sy*factor+sx)):continue
                    fine=resolution/factor;lo_x=lo_y=0.;hi_x=hi_y=1.
                    if bounds is not None:
                        code=float(bounds[sy*factor+sx,r,col])
                        if np.isfinite(code) and 0<=code<=65535 and code==int(code):
                            code=int(code);v=[(code>>(4*j))&15 for j in range(4)]
                            if v[0]<v[1] and v[2]<v[3]:lo_x,hi_x,lo_y,hi_y=np.array(v)/15.
                    ox=px-resolution/2+sx*fine;oy=py-resolution/2+sy*fine
                    if overlaps(ox+(lo_x+hi_x)*fine/2,oy+(lo_y+hi_y)*fine/2,(hi_x-lo_x)*fine,(hi_y-lo_y)*fine):return True,True
    return (True,False) if unknown else (False,True)


def occupied_column_extents(points,occupied,origin,resolution):
    """Measured vertical extents for DISPLAY, never inferred ground/support.

    Empty columns stay NaN. Do not erase a positive occupied column merely
    because the separate terrain elevation has no valid height there.
    """
    occupied=np.asarray(occupied,bool)
    if occupied.ndim!=2 or not np.isfinite(resolution) or resolution<=0:
        raise ValueError('invalid occupied extent geometry')
    low=np.full(occupied.shape,np.inf,np.float32);high=-low.copy()
    if points is not None:
        raw=np.asarray(points)
        if raw.ndim!=2 or raw.shape[1]<3:raise ValueError('expected measured XYZ')
        raw=raw[np.isfinite(raw[:,:3]).all(axis=1)]
        xy=np.floor((raw[:,:2]-np.asarray(origin))/resolution).astype(int)
        rows,cols=occupied.shape
        keep=(xy[:,0]>=0)&(xy[:,0]<cols)&(xy[:,1]>=0)&(xy[:,1]<rows)
        raw,xy=raw[keep],xy[keep];hit=occupied[xy[:,1],xy[:,0]]
        np.minimum.at(low,(xy[hit,1],xy[hit,0]),raw[hit,2])
        np.maximum.at(high,(xy[hit,1],xy[hit,0]),raw[hit,2])
    return np.where(np.isfinite(low),low,np.nan),np.where(np.isfinite(high),high,np.nan)


def collision_subcells(points,occupied,origin,resolution,*,xy_padding_m=.02,
                       measured_floor=None,measured_floor_gradient=None,floor_tolerance_m=.04,max_step_m=.18,factor=2,return_bounds=False,backend='cpu'):
    """Fine collision raster INSIDE each existing coarse occupied cell.

    By default keep every finite original return in an occupied column. An
    optional ORIGINAL measured-floor raster may identify floor returns which
    are not obstacle extent. Raised tops beside lower measured ground cannot
    supply that witness. Ambiguous, below-floor and all non-floor returns stay.
    No predicted height is allowed here, and no coarse occupied cell is cleared.
    Padding touches all subcells inside the original coarse square that the
    return's XY box intersects. Empty/missing witnesses fall back to the WHOLE
    coarse square. This is a representation contract, not a calibrated sensor
    error bound or a ground-support certificate.
    return_bounds adds outward-rounded XY rectangles inside the SAME bits,
    retaining the original witness padding with at most fine/15 extra error.
    """
    occupied=np.asarray(occupied,bool)
    if backend not in ('cpu','cupy') or factor not in (2,4) or occupied.ndim!=2 or not np.isfinite(resolution) or resolution<=0 or not 0<=xy_padding_m<=resolution/2:
        raise ValueError('invalid collision subcell geometry')
    if (not np.isfinite([floor_tolerance_m,max_step_m]).all() or
            not 0<floor_tolerance_m<=.05 or max_step_m<=floor_tolerance_m):
        raise ValueError('invalid measured-floor tolerance')
    floor=None
    gradient=None
    if measured_floor_gradient is not None:
        gradient=np.asarray(measured_floor_gradient)
        if measured_floor is None or gradient.shape!=(2,*occupied.shape) or not np.isfinite(gradient).all():
            raise ValueError('invalid measured floor gradient')
    if measured_floor is not None:
        from scipy.ndimage import minimum_filter
        floor=np.asarray(measured_floor)
        if floor.shape!=occupied.shape:raise ValueError('measured floor must match occupancy')
        lower=minimum_filter(np.where(np.isfinite(floor),floor,np.inf),size=3,mode='constant',cval=np.inf)
        floor=np.where(np.isfinite(floor)&(floor-lower<=max_step_m),floor,np.nan)
    if backend=='cupy':
        from .obstacle_columns_cuda import collision_subcells_cuda
        return collision_subcells_cuda(points,occupied,origin,resolution,xy_padding_m=xy_padding_m,
            measured_floor=floor,gradient=gradient,floor_tolerance_m=floor_tolerance_m,
            factor=factor,return_bounds=return_bounds)
    result=np.zeros(occupied.shape,np.uint16)
    bounds=np.full((factor*factor,*occupied.shape),0xF0F0,np.uint16) if return_bounds else None
    if points is not None:
        raw=np.asarray(points)
        if raw.ndim!=2 or raw.shape[1]<3:raise ValueError('expected original XYZ returns')
        raw=raw[np.isfinite(raw[:,:3]).all(axis=1)]
        xy=(raw[:,:2].astype(np.float64)-np.asarray(origin,np.float64))/resolution
        ij=np.floor(xy).astype(np.int64);rows,cols=occupied.shape
        inside=(ij[:,0]>=0)&(ij[:,0]<cols)&(ij[:,1]>=0)&(ij[:,1]<rows)
        ij,xy,raw=ij[inside],xy[inside],raw[inside]
        selected=occupied[ij[:,1],ij[:,0]];ij,xy=ij[selected],xy[selected]
        if floor is not None:
            z=raw[selected,2];ground=floor[ij[:,1],ij[:,0]]
            if gradient is not None:
                offset=(xy-ij-.5)*resolution
                ground=ground+gradient[0,ij[:,1],ij[:,0]]*offset[:,0]+gradient[1,ij[:,1],ij[:,0]]*offset[:,1]
            nonfloor=~np.isfinite(ground)|(np.abs(z-ground)>floor_tolerance_m)
            ij,xy=ij[nonfloor],xy[nonfloor]
        fraction=xy-ij;pad=xy_padding_m/resolution
        for sy in range(factor):
            for sx in range(factor):
                hit=((fraction[:,0]+pad>=sx/factor)&(fraction[:,0]-pad<=(sx+1)/factor)&
                     (fraction[:,1]+pad>=sy/factor)&(fraction[:,1]-pad<=(sy+1)/factor))
                np.bitwise_or.at(result,(ij[hit,1],ij[hit,0]),np.uint16(1<<(sy*factor+sx)))
                if return_bounds and hit.any():
                    # Preserve the SAME 2 cm XY witness box, clipped to this
                    # fine cell; do not expand it to the entire fine square.
                    # Four outward-rounded 4-bit coordinates are exactly
                    # representable in the existing Float32 layer format.
                    low=np.maximum(fraction[hit]-pad,np.array([sx,sy])/factor)
                    high=np.minimum(fraction[hit]+pad,(np.array([sx,sy])+1)/factor)
                    low=np.clip(np.floor((low*factor-[sx,sy])*15.-1e-7),0,15).astype(np.uint16)
                    high=np.clip(np.ceil((high*factor-[sx,sy])*15.+1e-7),0,15).astype(np.uint16)
                    limits=[np.full(occupied.shape,15,np.uint16),np.zeros(occupied.shape,np.uint16),
                            np.full(occupied.shape,15,np.uint16),np.zeros(occupied.shape,np.uint16)]
                    indices=(ij[hit,1],ij[hit,0])
                    np.minimum.at(limits[0],indices,low[:,0]);np.maximum.at(limits[1],indices,high[:,0])
                    np.minimum.at(limits[2],indices,low[:,1]);np.maximum.at(limits[3],indices,high[:,1])
                    present=(result&(1<<(sy*factor+sx)))!=0
                    code=limits[0]|(limits[1]<<4)|(limits[2]<<8)|(limits[3]<<12)
                    bounds[sy*factor+sx,present]=code[present]
    result[occupied&(result==0)]=(1<<(factor*factor))-1
    return (result,bounds) if return_bounds else result


_pose_kernel=None


def collision_pose_masks(bits,resolution,front,back,half_width,yaw_bins,*,backend='cpu',factor=2):
    """Representative-yaw body vs the SAME exact fine squares used by control.

    Height/uncertainty remain on the original grid. This deterministic obstacle
    overlay is separate from the Gaussian ground event. Native control still
    checks the actual continuous heading and every stopping footprint.
    """
    bits=np.asarray(bits)
    if (factor not in (2,4) or bits.ndim!=2 or not np.isfinite(bits).all() or (bits<0).any() or (bits>((1<<(factor*factor))-1)).any() or
            (bits!=np.floor(bits)).any() or not np.isfinite([resolution,front,back,half_width]).all() or
            min(resolution,front,back,half_width)<=0 or max(front,back,half_width)>3 or resolution<.05 or
            not 1<=yaw_bins<=180):
        raise ValueError('invalid collision pose geometry')
    rows,cols=bits.shape;fine=resolution/factor
    if not bits.any():return np.zeros((yaw_bins,rows,cols),bool)
    reach=int(np.ceil(np.hypot(max(front,back),half_width)/fine))+2
    if backend=='cupy':
        from .collision_pose_cuda import cached_collision_poses
        cached=cached_collision_poses(bits,resolution,front,back,half_width,yaw_bins,factor)
        if cached is not None:return cached
        import cupy as cp
        global _pose_kernel
        if _pose_kernel is None:
            _pose_kernel=cp.RawKernel(r'''
            extern "C" __global__ void collision_poses(const unsigned int *bits,unsigned char *out,
                int rows,int cols,int bins,int reach,int factor,float fine,float front,float back,float width) {
              int i=blockDim.x*blockIdx.x+threadIdx.x;
              if(i>=rows*cols*bins)return;
              int col=i%cols,row=(i/cols)%rows,k=i/(rows*cols);
              float yaw=(k+.5f)*6.283185307179586f/bins,c=cosf(yaw),s=sinf(yaw);
              float a=(front+back)*.5f,off=(front-back)*.5f,h=fine*.5f;
              float hx=fabsf(c)*a+fabsf(s)*width,hy=fabsf(s)*a+fabsf(c)*width;
              float projected=h*(fabsf(c)+fabsf(s));out[i]=0;
              for(int dy=-reach;dy<=reach;++dy)for(int dx=-reach;dx<=reach;++dx) {
                int x=factor*col+dx,y=factor*row+dy;
                if(x<0||y<0||x>=factor*cols||y>=factor*rows)continue;
                int b=bits[(y/factor)*cols+x/factor];if(!(b&(1<<((y%factor)*factor+x%factor))))continue;
                float rx=(dx+.5f-.5f*factor)*fine-off*c,ry=(dy+.5f-.5f*factor)*fine-off*s;
                if(fabsf(rx)<=hx+h+1e-6f && fabsf(ry)<=hy+h+1e-6f && fabsf(c*rx+s*ry)<=a+projected+1e-6f &&
                   fabsf(-s*rx+c*ry)<=width+projected+1e-6f){out[i]=1;return;}
              }
            }''','collision_poses')
        values=cp.asarray(bits,dtype=cp.uint32);out=cp.empty((yaw_bins,rows,cols),cp.uint8)
        _pose_kernel(((out.size+255)//256,),(256,),(values,out,np.int32(rows),np.int32(cols),np.int32(yaw_bins),
            np.int32(reach),np.int32(factor),np.float32(fine),np.float32(front),np.float32(back),np.float32(half_width)))
        return cp.asnumpy(out).astype(bool)
    if backend!='cpu':raise ValueError('collision pose backend must be cpu or cupy')
    from scipy.ndimage import correlate
    occupancy=np.zeros((factor*rows,factor*cols),np.int32)
    for sy in range(factor):
        for sx in range(factor):occupancy[sy::factor,sx::factor]=(bits.astype(np.uint32)&(1<<(sy*factor+sx)))!=0
    dy,dx=np.mgrid[-reach:reach+1,-reach:reach+1]
    a=(front+back)/2;off=(front-back)/2;h=fine/2;out=[]
    for k in range(yaw_bins):
        yaw=(k+.5)*2*np.pi/yaw_bins;c,s=np.cos(yaw),np.sin(yaw)
        rx=(dx+.5-factor/2)*fine-off*c;ry=(dy+.5-factor/2)*fine-off*s
        kernel=((abs(rx)<=abs(c)*a+abs(s)*half_width+h+1e-6)&(abs(ry)<=abs(s)*a+abs(c)*half_width+h+1e-6)&
                (abs(c*rx+s*ry)<=a+h*(abs(c)+abs(s))+1e-6)&(abs(-s*rx+c*ry)<=half_width+h*(abs(c)+abs(s))+1e-6))
        out.append(correlate(occupancy,kernel.astype(np.int32),mode='constant',cval=0)[::factor,::factor]>0)
    return np.asarray(out)


def vertical_columns(points, shape, origin, resolution, *, min_height=0.32,
                     min_span=0.18, min_hits=2, neighbor_radius=0.6,
                     ground_elevation=None, xp=np):
    """Current returns above observed ground; predicted/hidden heights are forbidden.

    With trusted observed ground, neighboring upper returns can corroborate a
    sparse column without requiring ground and top returns in the same cell.
    Without it the conservative within-column span test is retained.
    """
    if xp.__name__ == 'cupy':
        from .obstacle_columns_cuda import vertical_columns_cuda
        return vertical_columns_cuda(points, shape, origin, resolution, min_height=min_height,
            min_span=min_span, min_hits=min_hits, neighbor_radius=neighbor_radius,
            ground_elevation=ground_elevation)
    rows, cols = shape
    points = xp.asarray(points, dtype=xp.float32)
    low = xp.full(rows * cols, xp.inf, dtype=xp.float32)
    high = xp.full(rows * cols, -xp.inf, dtype=xp.float32)
    if points.size == 0:
        return xp.zeros(shape, dtype=bool), low.reshape(shape)
    points = points[xp.isfinite(points).all(axis=1)]
    # Match native control and collision_subcells exactly. A Float32 return
    # is a measured coordinate, but map origins/resolutions are not rounded
    # to Float32 before subtraction: near-boundary disagreement created a
    # coarse occupied neighbour with NO matching fine witness, triggering an
    # erroneous full-square fallback there.
    xy=points[:,:2].astype(xp.float64)
    ix = xp.floor((xy[:, 0] - np.float64(origin[0])) / np.float64(resolution)).astype(xp.int32)
    iy = xp.floor((xy[:, 1] - np.float64(origin[1])) / np.float64(resolution)).astype(xp.int32)
    inside = (ix >= 0) & (ix < cols) & (iy >= 0) & (iy < rows)
    ids = iy[inside] * cols + ix[inside]
    z = points[inside, 2]
    xp.minimum.at(low, ids, z)
    xp.maximum.at(high, ids, z)
    low, high = low.reshape(shape), high.reshape(shape)
    ground = low.copy()
    radius = max(1, int(np.ceil(neighbor_radius / resolution)))
    # Bounded shifts avoid wrap-around contamination at rolling-map edges.
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            y0, y1 = max(0, -dy), min(rows, rows - dy)
            x0, x1 = max(0, -dx), min(cols, cols - dx)
            ground[y0:y1, x0:x1] = xp.minimum(
                ground[y0:y1, x0:x1], low[y0+dy:y1+dy, x0+dx:x1+dx])
    hits = xp.zeros(rows * cols, dtype=xp.int32)
    relative = z - ground.ravel()[ids]
    # Do not classify a high canopy as a bumper-height obstacle.
    colliding = (relative >= min_height) & (relative <= 1.5)
    xp.add.at(hits, ids[colliding], 1)
    observed = xp.isfinite(low) & xp.isfinite(high)
    span = xp.where(observed, high, 0) - xp.where(observed, low, 0)
    occupied = observed & (span >= min_span) & (hits.reshape(shape) >= min_hits)
    if ground_elevation is not None:
        trusted = xp.asarray(ground_elevation, dtype=xp.float32)
        if trusted.shape != shape:
            raise ValueError("observed ground must match the occupancy grid")
        original = trusted
        # A low-variance obstacle top is still a measured surface, not a ground
        # witness. Only a nearby measured lower surface can reject that role;
        # no GP height or assumed free ground is introduced here.
        rejected = xp.zeros(shape, dtype=bool)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                y0, y1 = max(0, -dy), min(rows, rows - dy)
                x0, x1 = max(0, -dx), min(cols, cols - dx)
                lower = trusted[y0+dy:y1+dy, x0+dx:x1+dx]
                difference = trusted[y0:y1, x0:x1] - lower
                # The distance term preserves continuous slopes up to 45 deg,
                # beyond the configured passable pitch/roll envelope.
                rejected[y0:y1, x0:x1] |= xp.isfinite(difference) & (
                    difference > max(min_height, np.hypot(dx, dy) * resolution))
        trusted = xp.where(rejected, xp.nan, trusted)
        ground = trusted.copy()
        best_distance = xp.where(xp.isfinite(trusted), 0, xp.inf)
        original_ground = original.copy()
        original_distance = xp.where(xp.isfinite(original), 0, xp.inf)
        # Nearest measured ground, not a neighborhood minimum on a slope.
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                distance = dx * dx + dy * dy
                if distance == 0 or distance > radius * radius:
                    continue
                y0, y1 = max(0, -dy), min(rows, rows - dy)
                x0, x1 = max(0, -dx), min(cols, cols - dx)
                candidate = trusted[y0+dy:y1+dy, x0+dx:x1+dx]
                closer = xp.isfinite(candidate) & (distance < best_distance[y0:y1, x0:x1])
                ground[y0:y1, x0:x1] = xp.where(closer, candidate, ground[y0:y1, x0:x1])
                best_distance[y0:y1, x0:x1] = xp.where(closer, distance, best_distance[y0:y1, x0:x1])
                candidate = original[y0+dy:y1+dy, x0+dx:x1+dx]
                closer = xp.isfinite(candidate) & (distance < original_distance[y0:y1, x0:x1])
                original_ground[y0:y1, x0:x1] = xp.where(closer, candidate, original_ground[y0:y1, x0:x1])
                original_distance[y0:y1, x0:x1] = xp.where(closer, distance, original_distance[y0:y1, x0:x1])
        relative = z - ground.ravel()[ids]
        colliding = xp.isfinite(relative) & (relative >= min_height) & (relative <= 1.5)
        # A second ground hypothesis may add hazards, never remove an existing
        # upper-return witness by moving it above the canopy cutoff.
        relative = z - original_ground.ravel()[ids]
        colliding |= xp.isfinite(relative) & (relative >= min_height) & (relative <= 1.5)
        supported_hits = xp.zeros(rows * cols, dtype=xp.int32)
        xp.add.at(supported_hits, ids[colliding], 1)
        supported_hits = supported_hits.reshape(shape)
        corroborated = supported_hits.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                y0, y1 = max(0, -dy), min(rows, rows - dy)
                x0, x1 = max(0, -dx), min(cols, cols - dx)
                corroborated[y0:y1, x0:x1] += supported_hits[y0+dy:y1+dy, x0+dx:x1+dx]
        # A cell needs its own upper return; nearby evidence cannot dilate risk.
        occupied |= (supported_hits > 0) & (corroborated >= min_hits)
    return occupied, ground


def preserve_collision_risk(result, occupied):
    """Hard geometry cannot be averaged away by BGK/SEB fusion weights."""
    result.baseline_risk = np.maximum(result.baseline_risk, occupied.astype(np.float32))
    result.final_risk_slices = [np.where(occupied, 1.0, risk).astype(np.float32)
                               for risk in result.final_risk_slices]
