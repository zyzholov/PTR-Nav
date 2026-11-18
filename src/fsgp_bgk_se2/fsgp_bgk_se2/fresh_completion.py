"""Current-scan-conditioned interpolation; unknowns are never training heights."""
import numpy as np
from scipy.spatial import cKDTree
from .feature_sgp import TerrainPrediction
from .ground_evidence import temporal_std_bound


def fresh_local_prediction(predictor, elevation, variance, observed, x, y, points,
                           age, resolution, *, blocked, max_age=.25, max_step=.18,
                           history=None, stamp=None, frame=None, pose_xy=None,
                           process_std_mps=.01):
    shape = elevation.shape
    empty = lambda value: np.full(shape, value)
    prediction = TerrainPrediction(empty(np.nan), empty(np.inf), empty(np.inf),
        empty(np.inf), empty(False), empty(False), str(predictor.device), 0., 0., 0,
        local_conditioned=empty(False),conditioning_age=empty(np.inf))
    if not np.isfinite(age) or not 0 <= age <= max_age or points is None:
        return prediction
    points = np.asarray(points)
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        return prediction
    raw_points=points
    left, bottom = x[0, 0] - resolution/2, y[0, 0] - resolution/2
    def point_cells(values):
        col=np.floor((values[:,0]-left)/resolution).astype(int)
        row=np.floor((values[:,1]-bottom)/resolution).astype(int)
        inside=(row>=0)&(row<shape[0])&(col>=0)&(col<shape[1])
        return row[inside],col[inside],inside
    usable = observed & np.isfinite(elevation) & np.isfinite(variance) & (variance >= 0)
    usable &= variance <= predictor.params.noise_std_m ** 2
    tolerance = np.minimum(.05, np.maximum(.03, 2*np.sqrt(np.maximum(variance, 0))))
    row,col,inside=point_cells(points)
    points=points[inside]
    agrees=usable[row,col]&(np.abs(points[:,2]-elevation[row,col])<=tolerance[row,col])
    evidence_age=np.zeros(len(points))
    training_height=elevation.copy()
    persistent_veto=np.zeros(shape,bool)
    if history is not None:
        if not np.isfinite(process_std_mps) or process_std_mps<=0:
            raise ValueError('history requires a positive uncertainty growth rate')
        # Admit raw returns only when their original scan corroborates measured
        # ground. Occlusion later does not make that original observation fake.
        points,evidence_age=history.update(points[agrees&~blocked[row,col]],
            stamp,frame,pose_xy,source_age_s=age)
        row,col,inside=point_cells(points)
        points=points[inside];evidence_age=evidence_age[inside]
        agrees=~blocked[row,col]
        agrees &= ~usable[row,col] | (np.abs(points[:,2]-elevation[row,col])<=tolerance[row,col])
        counts=np.bincount(row*shape[1]+col,minlength=elevation.size).reshape(shape)
        total=np.bincount(row*shape[1]+col,weights=points[:,2],minlength=elevation.size).reshape(shape)
        previous_height=np.divide(total,counts,out=np.full(shape,np.nan),where=counts>0)
        rr,cc,raw_inside=point_cells(raw_points)
        difference=raw_points[raw_inside,2]-previous_height[rr,cc]
        contradiction=np.isfinite(difference)&((difference < -max_step)|
                       ((difference>max_step)&(difference<1.5)))
        veto=np.bincount(rr[contradiction]*shape[1]+cc[contradiction],
                         minlength=elevation.size).reshape(shape)>=2
        raw_ground=usable[rr,cc]&~blocked[rr,cc]&(
            np.abs(raw_points[raw_inside,2]-elevation[rr,cc])<=tolerance[rr,cc])
        clear_counts=np.bincount(rr[raw_ground]*shape[1]+cc[raw_ground],
                                 minlength=elevation.size).reshape(shape)
        # Negative measured evidence must survive even if no GP query can run.
        persistent_veto=history.veto_mask(x,y,veto,clear_counts>=2,stamp-age)
        agrees &= ~persistent_veto[row,col]
        keep=np.zeros(len(history.points),bool);keep[np.flatnonzero(inside)[agrees]]=True
        history.retain(keep)
        counts=np.bincount(row[agrees]*shape[1]+col[agrees],minlength=elevation.size).reshape(shape)
        total=np.bincount(row[agrees]*shape[1]+col[agrees],weights=points[agrees,2],
                          minlength=elevation.size).reshape(shape)
        retained_height=np.divide(total,counts,out=np.full(shape,np.nan),where=counts>0)
        training_height=np.where(usable,elevation,retained_height)
    fresh = np.zeros(shape, bool)
    fresh[row[agrees], col[agrees]] = True
    if np.count_nonzero(fresh) < 10:
        return prediction
    query = ~observed & ~blocked & ~persistent_veto
    if not query.any():
        return prediction
    # A return near a cell edge must not move to the cell center for the
    # physical support-distance gate. Heights remain one estimate per cell.
    evidence_tree=cKDTree(points[agrees,:2])
    distance=np.full(shape,np.inf)
    distance[query]=evidence_tree.query(np.column_stack((x[query],y[query])),workers=1)[0]
    query &= distance <= predictor.params.local_nearest_m
    if not query.any():
        return prediction
    local = predictor.condition_local(np.column_stack((x[fresh], y[fresh], training_height[fresh])),
                                      np.column_stack((x[query], y[query])),
                                      current_support_distance=distance[query])
    ages=np.zeros(shape)
    np.maximum.at(ages,(row[agrees],col[agrees]),evidence_age[agrees])
    if history is not None and local.conditioning_indices is not None:
        local_xy=predictor.local_tree.data
        lc=np.rint((local_xy[:,0]-left-resolution/2)/resolution).astype(int)
        lr=np.rint((local_xy[:,1]-bottom-resolution/2)/resolution).astype(int)
        # Reuse the exact selected sources, including angular-sector fallback.
        # Repeating k-NN here costs time and can select different float ties.
        ids=local.conditioning_indices
        source_ages=ages[lr,lc][ids]
        source_age=source_ages.max(axis=1)
        # Worst-case correlated temporal drift: add standard deviations, not
        # independent variances. GP weights can be negative: max(age) is not
        # a valid substitute for their absolute-weighted source bound.
        drift_std=temporal_std_bound(local.conditioning_weights,source_ages,process_std_mps)
        local.latent_var=(np.sqrt(local.latent_var)+drift_std)**2
        local.observation_var=local.latent_var+predictor.params.noise_std_m**2
        local.accepted &= local.latent_var<=predictor.params.accept_std_m**2
    else:source_age=np.zeros(np.count_nonzero(query))
    prediction.conditioning_age[query]=source_age
    for name in ("mean", "latent_var", "observation_var", "nearest_distance", "supported",
                 "accepted", "local_conditioned"):
        getattr(prediction, name)[query] = getattr(local, name)
    prediction.predict_ms = local.predict_ms
    prediction.training_points = local.training_points
    # Raw returns in hidden cells are veto-only evidence, never GP training.
    # Low steps/pits can be dangerous without meeting the tall-column detector.
    col=np.floor((raw_points[:,0]-left)/resolution).astype(int)
    row=np.floor((raw_points[:,1]-bottom)/resolution).astype(int)
    inside=(row>=0)&(row<shape[0])&(col>=0)&(col<shape[1])
    row,col,raw_points=row[inside],col[inside],raw_points[inside]
    difference = raw_points[:, 2]-prediction.mean[row, col]
    contradictory = query[row, col] & np.isfinite(difference) & (
        (difference < -max_step) | ((difference > max_step) & (difference < 1.5)))
    counts = np.bincount((row[contradictory]*shape[1]+col[contradictory]), minlength=elevation.size)
    veto = counts.reshape(shape) >= 2
    if history is not None:
        ground=usable[row,col]&~blocked[row,col]&(
            np.abs(raw_points[:,2]-elevation[row,col])<=tolerance[row,col])
        ground_counts=np.bincount(row[ground]*shape[1]+col[ground],minlength=elevation.size).reshape(shape)
        veto=history.veto_mask(x,y,veto,ground_counts>=2,stamp-age)
    prediction.accepted[veto] = False
    return prediction
