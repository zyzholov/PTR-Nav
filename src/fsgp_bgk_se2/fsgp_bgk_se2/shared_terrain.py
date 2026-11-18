"""Local finite-feature GP with a shared posterior for geometric contrasts.

The likelihood is an explicit Gaussian observation model, not an empirical
calibration claim. Only the frontend's original measured ground conditions the
model. Missing cells require nearby original measured support. New contradictory
returns veto old ground; no predicted or assumed height enters training.
"""
from dataclasses import dataclass,replace
from functools import lru_cache
import math
import time

import numpy as np
from scipy.ndimage import minimum_filter,maximum_filter
from scipy.spatial import ConvexHull,QhullError,cKDTree
import torch

from .feature_sgp import spatial_sample_indices
from .probabilistic_risk import _geometry


@dataclass(frozen=True)
class SharedTerrainParams:
    radius_m: float = 3.
    basis_spacing_m: float = 1.
    lengthscale_m: float = .8
    residual_std_m: float = .04
    measurement_std_m: float = .02
    discrepancy_std_m: float = .01
    discrepancy_lengthscale_m: float = .2
    max_std_m: float = .04
    max_distance_m: float = .75
    max_extrapolation_m: float = 0.
    max_sources: int = 2048
    flat_initial_basis: bool = False
    max_detail_sources: int = 0
    detail_lengthscale_m: float = .20
    detail_trigger_m: float = .025
    observation_halo_m: float = 0.
    # Runtime can complete the already conditioned footprint halo with the
    # SAME distance, covariance and negative-evidence gates as the core.
    # False defaults retain exact archived model/replay semantics.
    complete_footprint_halo: bool = False
    current_floor_column_seams: bool = False
    condition_measured_column_seams: bool = False
    predict_ground_under_columns: bool = False
    # Retire a discontinuity's broad shadow ONLY under a positive column
    # with an independent, enclosed measured floor. No likelihood/covariance
    # substitution; actual occupied volume remains a separate hard constraint.
    column_shadow_floor_reference: bool = False
    condition_local_residual: bool = False
    conditioned_measurement_std_m: float = .01
    # Explicit alternative kernel: ALL unresolved cell-scale terrain is a
    # conditionable local latent variable. False preserves archived models
    # whose spatial RBF discrepancy floor was added after conditioning.
    condition_full_residual: bool = False
    # A FIC local RBF kernel: low-rank correlated component plus the EXACT
    # prior diagonal remainder. Zero retains archived cell-white models.
    residual_inducing_points: int = 0
    residual_lengthscale_m: float = .35
    # Do not solve/query a posterior outside its ALREADY declared admissible
    # tile. False retains full-field diagnostics in archived replays.
    bounded_queries: bool = False

    def __post_init__(self):
        values=[self.radius_m,self.basis_spacing_m,self.lengthscale_m,self.residual_std_m,
                self.measurement_std_m,self.discrepancy_std_m,self.discrepancy_lengthscale_m,self.max_std_m,self.max_distance_m]
        if (not np.isfinite(values).all() or min(values)<=0 or not 1<=self.radius_m<=5 or
                not .5<=self.basis_spacing_m<=2 or not 0<self.max_distance_m<=1.25 or
                not 32<=self.max_sources<=4096 or not math.isfinite(self.max_extrapolation_m) or
                not 0<=self.max_extrapolation_m<=min(.75,self.max_distance_m) or
                not isinstance(self.flat_initial_basis,bool) or
                not isinstance(self.condition_local_residual,bool) or
                not isinstance(self.condition_full_residual,bool) or
                not isinstance(self.complete_footprint_halo,bool) or
                not isinstance(self.current_floor_column_seams,bool) or
                not isinstance(self.condition_measured_column_seams,bool) or
                not isinstance(self.predict_ground_under_columns,bool) or
                not isinstance(self.column_shadow_floor_reference,bool) or
                self.column_shadow_floor_reference and not self.predict_ground_under_columns or
                not isinstance(self.bounded_queries,bool) or
                self.condition_full_residual and not self.condition_local_residual or
                not isinstance(self.residual_inducing_points,int) or not 0<=self.residual_inducing_points<=256 or
                self.residual_inducing_points>0 and not self.condition_full_residual or
                not math.isfinite(self.residual_lengthscale_m) or not .1<=self.residual_lengthscale_m<=.8 or
                not math.isfinite(self.conditioned_measurement_std_m) or
                not 0<self.conditioned_measurement_std_m<=self.measurement_std_m or
                not isinstance(self.max_detail_sources,int) or not 0<=self.max_detail_sources<=128 or
                not math.isfinite(self.detail_lengthscale_m) or not .1<=self.detail_lengthscale_m<=.5 or
                not math.isfinite(self.detail_trigger_m) or not .01<=self.detail_trigger_m<=.10 or
                not math.isfinite(self.observation_halo_m) or not 0<=self.observation_halo_m<=1.5):
            raise ValueError('invalid bounded shared-terrain model')


@dataclass
class SharedTerrainPrediction:
    mean: np.ndarray
    variance: np.ndarray
    valid: np.ndarray
    completed: np.ndarray
    nearest: np.ndarray
    factor: torch.Tensor
    discrepancy: float
    training_points: int
    fit_ms: float
    assumed: np.ndarray | None = None
    discrepancy_lengthscale_m: float = .2
    candidate: np.ndarray | None = None
    contradicted: np.ndarray | None = None
    discontinuity_veto: np.ndarray | None = None
    flat_initial_basis_used: bool = False
    detail_features: int = 0
    local_variance: np.ndarray | None = None
    initial_plane_used: bool = False
    relative_variance: np.ndarray | None = None
    # The stored factor still carries the conservative sqrt(2) two-source
    # envelope. Keep the split so each scalar geometric contrast can use the
    # tighter triangle bound, without assuming source independence.
    plane_components: int = 0
    column_shadow_floor_supported: np.ndarray | None = None
    # A deterministic mean family, NOT an additional Gaussian covariance.
    bounded_mean_factor: torch.Tensor | None = None
    bounded_pending: np.ndarray | None = None
    admission_domain: np.ndarray | None = None


def prediction_on_device(prediction,device):
    """Move covariance/bounded-mean factors without refitting or adding support.

    Geometry may run on CUDA even when the small posterior was fit on CPU.
    Keep all measured-source identities and NumPy fields; never mutate the
    producer's prediction. Blocking copies make the result ready for consumers
    outside the current CUDA stream, just as the fit's return contract requires.
    """
    target=torch.device(device)
    factor=prediction.factor.to(target)
    bounded=(None if prediction.bounded_mean_factor is None else
             prediction.bounded_mean_factor.to(target))
    if target.type=='cuda':torch.cuda.current_stream(target).synchronize()
    return replace(prediction,factor=factor,bounded_mean_factor=bounded)


@dataclass(frozen=True)
class InitialPlaneContext:
    anchor_xy: np.ndarray
    coefficients: np.ndarray
    coefficient_std_bound: np.ndarray
    area: np.ndarray
    coefficient_covariance_bound: np.ndarray | None = None


def plane_covariance_envelope(coefficients,sigma):
    """A rotation-aware PSD upper envelope for arbitrary source correlation.

    With b_j=A_j*sigma_j and positive l_j, C=(sum l_j)*sum b_j b_j'/l_j
    bounds every admissible source covariance by weighted Cauchy-Schwarz.
    The lengths use the original measurement geometry only, never truth.
    """
    design=np.linalg.pinv(coefficients)
    scale=np.r_[np.maximum(.5,np.max(abs(design[:,:2]),axis=0)),1.]
    b=coefficients*sigma[None];length=np.linalg.norm(scale[:,None]*b,axis=0)
    keep=length>1e-15
    return length.sum()*(b[:,keep]/length[keep][None])@b[:,keep].T


def initial_plane_context(model,params):
    """Original measured plane sources, not extra GP observations.

    A weighted Cauchy envelope retains coefficient correlation geometry;
    it does not divide source uncertainty by the number of LiDAR returns.
    """
    sigma=np.sqrt(model.measurement_sigma**2+params.height_std_floor**2/params.variance_scale)
    return InitialPlaneContext(np.array([model.anchor.x,model.anchor.y]),np.array(model.plane),
        np.abs(model.plane_weights)@sigma,model.raster_area.copy(),plane_covariance_envelope(model.plane_weights,sigma))


def relative_height_variance(factor,local,discrepancy,lengthscale,query,anchor_ids):
    """Variance bound of h(q)-h(nearest REAL conditioning cell).

    Keep the absolute covariance for every geometric chance calculation.
    A common vertical datum must not itself disqualify shape support.
    """
    ids=torch.as_tensor(anchor_ids,device=factor.device)
    difference=(factor-factor[ids]).square().sum(dim=1)
    xy=torch.as_tensor(query,dtype=factor.dtype,device=factor.device)
    distance=(xy-xy[ids]).square().sum(dim=1)
    difference+=2*discrepancy**2*(1-torch.exp(-distance/(2*lengthscale**2)))
    if local is not None:
        same=torch.arange(len(ids),device=factor.device)==ids
        difference+=torch.where(same,torch.zeros_like(local),local+local[ids])
    return difference


def _features(xy,center,knots,p,gate=None,detail_knots=None):
    uv=(xy-center)/p.radius_m
    a,b=uv.unbind(dim=1)
    trend=torch.stack((torch.ones_like(a),a,b,a*a,a*b,b*b),dim=1)
    radial=torch.exp(-torch.cdist(xy,knots).square()/(2*p.lengthscale_m**2))
    radial=radial/torch.linalg.vector_norm(radial,dim=1,keepdim=True).clamp_min(1e-12)
    parts=[trend,radial]
    if detail_knots is not None and len(detail_knots):
        detail=torch.exp(-torch.cdist(xy,detail_knots).square()/(2*p.detail_lengthscale_m**2))
        # A local component must decay away from its measured detail sites.
        # Unit normalization there would invent a long-range fine component.
        detail=detail/torch.linalg.vector_norm(detail,dim=1,keepdim=True).clamp_min(1.)
        parts.append(detail)
    values=torch.cat(parts,dim=1)
    if gate is not None:
        # A simulation-only planar initial hypothesis is structural prior
        # information, not extra observations. Define ONE input-dependent GP
        # basis for training and queries; retain its full posterior covariance.
        values[:,3:]*=gate[:,None]
    return values


def residual_fic_features(training,query,*,count,lengthscale,variance,training_gate=None,query_gate=None,
                          training_xy=None):
    """One proper finite GP prior, with no missing prior variance discarded.

    Phi=K_xu chol(K_uu+jitter I)^-T * sigma, and D=sigma²-diag(Phi Phi').
    k_FIC(x,x')=Phi(x)Phi(x')'+1[x=x']D(x). Inducing sites are a bounded
    spatial subset of REAL likelihood sites, never assumptions/predictions.
    Query/training share the SAME features and diagonal correction.
    training_xy, when supplied, must contain the same ordered training rows
    on the CPU; it only avoids copying those coordinates back from the device.
    """
    xy=training.detach().cpu().numpy() if training_xy is None else training_xy
    ids=spatial_sample_indices(np.c_[xy,np.zeros(len(xy))],min(count,len(training)))
    inducing=training[torch.as_tensor(ids,device=training.device)]
    gram=torch.exp(-torch.cdist(inducing,inducing).square()/(2*lengthscale**2))
    gram=gram+torch.eye(len(inducing),dtype=training.dtype,device=training.device)*1e-6
    root=torch.linalg.cholesky(gram)
    def evaluate(xy,gate):
        kernel=torch.exp(-torch.cdist(xy,inducing).square()/(2*lengthscale**2))
        features=torch.linalg.solve_triangular(root,kernel.T,upper=False).T*math.sqrt(variance)
        diagonal=torch.full((len(xy),),variance,dtype=xy.dtype,device=xy.device)
        if gate is not None:features=features*gate[:,None];diagonal=diagonal*gate.square()
        remainder=diagonal-features.square().sum(dim=1)
        if remainder.min().item() < -max(1e-12,variance*1e-7):
            raise ValueError('FIC residual prior lost positive diagonal remainder')
        return features,remainder.clamp_min(0)
    # The SAME inducing root/variance applies to likelihood and query rows.
    # One batched triangular solve and one diagonal-validity synchronization
    # avoid two GPU launch/readback chains; the prior and its exact diagonal
    # remainder do not change. This also preserves cross-set covariance.
    combined_gate=None
    if training_gate is not None or query_gate is not None:
        combined_gate=torch.cat((training_gate if training_gate is not None else
            torch.ones(len(training),dtype=training.dtype,device=training.device),
            query_gate if query_gate is not None else
            torch.ones(len(query),dtype=query.dtype,device=query.device)))
    features,diagonal=evaluate(torch.cat((training,query)),combined_gate)
    n=len(training)
    return features[:n],features[n:],diagonal[:n],diagonal[n:]


def raw_measured_evidence(elevation,observed,x,y,points,resolution,max_step,positive_veto_height=1.5):
    """Return independent contradiction and measured-ground clearing masks."""
    if positive_veto_height is not None and (not math.isfinite(positive_veto_height) or positive_veto_height<=max_step):
        raise ValueError('invalid positive support-veto height')
    result=np.zeros(elevation.shape,bool)
    if points is None:return result,result.copy()
    raw=np.asarray(points);raw=raw[np.isfinite(raw).all(axis=1)]
    c=np.floor((raw[:,0]-x[0,0]+resolution/2)/resolution).astype(int)
    r=np.floor((raw[:,1]-y[0,0]+resolution/2)/resolution).astype(int)
    keep=(r>=0)&(r<elevation.shape[0])&(c>=0)&(c<elevation.shape[1])
    r,c,raw=r[keep],c[keep],raw[keep]
    delta=raw[:,2]-elevation[r,c]
    positive=(delta>max_step)&(delta<positive_veto_height) if positive_veto_height is not None else np.zeros(len(delta),bool)
    bad=observed[r,c]&np.isfinite(delta)&((delta < -max_step)|positive)
    counts=np.bincount(r[bad]*elevation.shape[1]+c[bad],minlength=elevation.size)
    clear=observed[r,c]&np.isfinite(delta)&(np.abs(delta)<=.05)
    clear_counts=np.bincount(r[clear]*elevation.shape[1]+c[clear],minlength=elevation.size)
    # Registered clouds may contain only one voxel-filtered return per cell.
    # This is a per-scan witness, not permission to clear immediately: the
    # history requires three distinct consecutive scans and veto takes priority.
    counts=counts.reshape(elevation.shape)
    # A contradictory return against measured ground immediately removes that
    # support. Filtering can leave just one point from a thin obstacle per cell.
    return counts>=1,(clear_counts.reshape(elevation.shape)>=1)&(counts==0)


def raw_measured_veto(elevation,observed,x,y,points,resolution,max_step):
    """Negative evidence must survive even when no GP can be fitted."""
    return raw_measured_evidence(elevation,observed,x,y,points,resolution,max_step)[0]


def shared_terrain_prediction(elevation,variance,observed,x,y,points,age,resolution,
                              blocked,*,device='cpu',params=None,max_age=.25,max_step=.18,
                              initial_support=None,initial_height=None,positive_veto_height=1.5,
                              plane_context=None,columns=None,diagnostics=None):
    p=params or SharedTerrainParams()
    if diagnostics is not None:diagnostics.clear()
    def reject(reason):
        if diagnostics is not None:diagnostics['reason']=reason
        return None
    if positive_veto_height is not None and (not math.isfinite(positive_veto_height) or positive_veto_height<=max_step):
        raise ValueError('invalid positive support-veto height')
    shape=elevation.shape
    if (x.shape!=shape or y.shape!=shape or variance.shape!=shape or observed.shape!=shape or
            blocked.shape!=shape or not np.isfinite(resolution) or resolution<=0):
        raise ValueError('inconsistent shared-terrain geometry')
    column=np.zeros(shape,bool) if columns is None else np.asarray(columns,dtype=bool)
    if column.shape!=shape:raise ValueError('column mask must match the terrain grid')
    if not p.predict_ground_under_columns:
        blocked=blocked|column
    else:
        # A vertical column and the latent ground below it are separate fields.
        # Column/side returns are NOT measured ground. A conditional ground
        # prediction can extend below the column, but cannot make its occupied
        # XY volume traversable: the native full-body geometry remains hard.
        observed=observed&~column
    if points is None or not np.isfinite(age) or not 0<=age<=max_age:return reject('missing_or_stale_scan')
    raw=np.asarray(points)
    if raw.ndim!=2 or raw.shape[1]!=3:raise ValueError('expected raw XYZ')
    raw=raw[np.isfinite(raw).all(axis=1)]
    left,bottom=x[0,0]-resolution/2,y[0,0]-resolution/2
    def cells(points):
        c=np.floor((points[:,0]-left)/resolution).astype(int)
        r=np.floor((points[:,1]-bottom)/resolution).astype(int)
        keep=(r>=0)&(r<shape[0])&(c>=0)&(c<shape[1])
        return r[keep],c[keep],keep
    rr,cc,inside=cells(raw);raw=raw[inside]
    measured=observed&np.isfinite(elevation)&np.isfinite(variance)&(variance>=0)&~blocked&~column
    measured &= variance<=p.measurement_std_m**2
    # Use the same observation-noise floor as the likelihood below. Applying
    # only the frontend's smaller fused-cell variance rejected ordinary
    # observations that this GP explicitly models as noisy.
    tolerance=np.minimum(.05,np.maximum(.03,2*np.sqrt(np.maximum(variance,p.measurement_std_m**2))))
    agrees=measured[rr,cc]&(np.abs(raw[:,2]-elevation[rr,cc])<=tolerance[rr,cc])
    difference=raw[:,2]-elevation[rr,cc]
    positive=((difference>max_step)&(difference<positive_veto_height)
              if positive_veto_height is not None else np.zeros(len(difference),bool))
    contradiction=measured[rr,cc]&((difference < -max_step)|positive)
    counts=np.bincount(rr[contradiction]*shape[1]+cc[contradiction],minlength=elevation.size)
    contradicted=counts.reshape(shape)>=2
    # Occlusion by the moving chassis does not erase the frontend's fused
    # measurements. A fresh scan supplies negative evidence; both fused real
    # measurements and current returns bound the interpolation domain.
    fresh=measured&~contradicted
    # Raised obstacle tops can have small within-cell variance too. Do not
    # condition a smooth ground field on a top when fresh nearby lower ground
    # demonstrates a discontinuity. Their original measured heights remain
    # available below to veto model support; no return is erased.
    local_floor=minimum_filter(np.where(fresh,elevation,np.inf),size=3,mode='constant',cval=np.inf)
    ground_witness=~maximum_filter(blocked|column,size=3,mode='constant',cval=False)
    if p.current_floor_column_seams:
        # A positive current floor return in an UNoccupied cell is independent
        # support even beside a column. Do not erode away that real ground with
        # a blanket one-cell exclusion. No inferred height can supply this.
        witnessed_counts=np.bincount(rr[agrees]*shape[1]+cc[agrees],minlength=elevation.size).reshape(shape)
        ground_witness |= (witnessed_counts>0)&measured&~blocked&~contradicted
    if p.condition_measured_column_seams:
        # The frontend's original measured floor does not become unobserved
        # merely because a nearby wall/chassis occludes this scan. Excluding
        # it from the likelihood but replacing it with an UNconditioned local
        # GP residual manufactured high-uncertainty bands beside walls/behind
        # the vehicle. Use that real likelihood once, with its original noise.
        # Occupied cells, raised tops, pits and pending/contradicted evidence
        # remain excluded. Neither GP predictions nor the initial assumption
        # supply this witness or acquire a fresh observation timestamp.
        ground_witness |= measured&~blocked&~contradicted&(elevation-local_floor<=max_step)
    fresh &= (elevation-local_floor<=max_step)&ground_witness
    center=np.array([(x[0,0]+x[0,-1])/2,(y[0,0]+y[-1,0])/2])
    region=(np.abs(x-center[0])<=p.radius_m)&(np.abs(y-center[1])<=p.radius_m)
    conditioning_region=(np.abs(x-center[0])<=p.radius_m+p.observation_halo_m)&(np.abs(y-center[1])<=p.radius_m+p.observation_halo_m)
    # Whole-pose stencils require their REAL observations just beyond the core
    # tile too. The optional footprint halo below can query missing cells too,
    # with unchanged statistical and source-evidence gates.
    fresh &= conditioning_region
    details=np.zeros(shape,bool)
    if p.max_detail_sources:
        # Choose only real measured, non-occupied surface details. A planar
        # grade has zero second difference; a narrow ledge supplies new local
        # basis sites. Final fitting still uses the original likelihood once.
        for axis in (0,1):
            a=[slice(None),slice(None)];b=a.copy();c=a.copy()
            a[axis]=slice(None,-2);b[axis]=slice(1,-1);c[axis]=slice(2,None)
            a,b,c=tuple(a),tuple(b),tuple(c)
            good=fresh[a]&fresh[b]&fresh[c]
            edge=good&(np.abs(elevation[a]-2*elevation[b]+elevation[c])>p.detail_trigger_m)
            details[b]|=edge
        sites=np.c_[x[details],y[details],elevation[details]]
        if len(sites)>p.max_detail_sources:sites=sites[spatial_sample_indices(sites,p.max_detail_sources)]
    else:sites=np.empty((0,3))
    xyz=np.c_[x[fresh],y[fresh],elevation[fresh]]
    if diagnostics is not None:
        diagnostics.update(training_candidates=int(len(xyz)),measured_cells=int(measured.sum()),
                           current_floor_returns=int(np.count_nonzero(agrees)))
    # Only the three affine trend coefficients are unregularized. With
    # positive likelihood noise and proper curvature/residual priors, their
    # 2-D rank (checked below), not an arbitrary 32-cell count, identifies the
    # posterior. Sparse data retains its ORIGINAL, often wider covariance.
    if len(xyz)<3:return reject('insufficient_measured_training_cells')
    ids=spatial_sample_indices(xyz,p.max_sources)
    likelihood_floor=p.conditioned_measurement_std_m if p.condition_local_residual else p.measurement_std_m
    xyz=xyz[ids];noise=np.maximum(variance[fresh][ids],likelihood_floor**2)
    current_floor=agrees&fresh[rr,cc]
    raw_support=raw[current_floor,:2]
    source_cells=len(np.unique(rr[current_floor]*shape[1]+cc[current_floor]))
    if diagnostics is not None:diagnostics.update(training_points=len(xyz),raw_support_points=len(raw_support),
                                                 current_support_cells=source_cells)
    if source_cells<3:return reject('insufficient_current_floor_cells')
    query=np.c_[x.ravel(),y.ravel()]
    query_region=(conditioning_region if p.complete_footprint_halo else
                  region|(conditioning_region&measured))
    query_ids=np.flatnonzero(query_region) if p.bounded_queries else np.arange(len(query))
    query_xy=query[query_ids]
    # Every conditioning cell is in the same admitted tile, including the
    # measured footprint halo. Compact indices must preserve those exact
    # likelihood locations for the conditional local-residual update.
    query_lookup=np.full(len(query),-1,np.int64);query_lookup[query_ids]=np.arange(len(query_ids))
    flat_tree=None
    if plane_context is not None:
        if (plane_context.area.shape!=shape or plane_context.anchor_xy.shape!=(2,) or
            plane_context.coefficients.shape!=(3,) or plane_context.coefficient_std_bound.shape!=(3,) or
            not np.isfinite(np.r_[plane_context.anchor_xy,plane_context.coefficients,plane_context.coefficient_std_bound]).all() or
            np.any(plane_context.coefficient_std_bound<0)):
            raise ValueError('invalid measured initial-plane context')
        if plane_context.coefficient_covariance_bound is not None:
            covariance=plane_context.coefficient_covariance_bound
            if (covariance.shape!=(3,3) or not np.isfinite(covariance).all() or
                not np.allclose(covariance,covariance.T,atol=1e-10) or np.linalg.eigvalsh(covariance).min()<-1e-10):
                raise ValueError('invalid initial-plane covariance envelope')
        flat=plane_context.area
        if flat.any():flat_tree=cKDTree(np.c_[x[flat],y[flat]])
    elif p.flat_initial_basis and initial_support is not None:
        if initial_height is None or initial_support.shape!=shape or initial_height.shape!=shape:
            raise ValueError('fixed initial support requires matching height geometry')
        flat=initial_support&np.isfinite(initial_height)
        if flat.any():flat_tree=cKDTree(np.c_[x[flat],y[flat]])
    def gate(points):
        if flat_tree is None:return reject('initial_basis_without_measured_plane')
        distance=flat_tree.query(points,workers=1)[0]
        t=np.clip((distance-.5*resolution)/p.lengthscale_m,0.,1.)
        return torch.as_tensor(t*t*(3-2*t),dtype=torch.float64,device=device)
    support_xy=np.r_[raw_support,np.c_[x[fresh],y[fresh]]]
    nearest=cKDTree(support_xy).query(query,workers=1)[0].reshape(shape)
    try:
        hull=ConvexHull(xyz[:,:2]).equations
        enclosed=np.all(query@hull[:,:2].T+hull[:,2]<=1e-7,axis=1).reshape(shape)
    except QhullError:return reject('training_xy_not_two_dimensional')
    dtype=torch.float64
    target=torch.device(device)
    # This function creates its tensors from CPU arrays on the current
    # Torch stream. Timing/readiness needs that stream, not a device-wide
    # barrier which also waits for unrelated planner/renderer streams.
    if target.type=='cuda':torch.cuda.current_stream(target).synchronize()
    started=time.perf_counter()
    center_t=torch.as_tensor(center,dtype=dtype,device=target)
    reach=p.radius_m+p.observation_halo_m
    reach=math.ceil(reach/p.basis_spacing_m)*p.basis_spacing_m
    axis=np.arange(-reach,reach+p.basis_spacing_m/2,p.basis_spacing_m)
    kx,ky=np.meshgrid(axis+center[0],axis+center[1])
    knots=torch.as_tensor(np.c_[kx.ravel(),ky.ravel()],dtype=dtype,device=target)
    detail_knots=torch.as_tensor(sites[:,:2],dtype=dtype,device=target)
    training=torch.as_tensor(xyz[:,:2],dtype=dtype,device=target)
    training_gate=gate(xyz[:,:2])
    design=_features(training,center_t,knots,p,training_gate,detail_knots)
    if plane_context is not None:design=design[:,3:]
    elif torch.linalg.matrix_rank(design[:,:3]).item()<3:return reject('affine_trend_rank_deficient')
    weight=torch.as_tensor(1/noise,dtype=dtype,device=target)
    precision=design.T@(weight[:,None]*design)
    prior=torch.zeros(len(precision),dtype=dtype,device=target)
    # Only the affine datum/slope is an unpenalized trend. Unbounded quadratic
    # coefficients fitted to one-sided tilted-LiDAR support can bow through the
    # unseen initial footprint even beside a locally flat, measured stance.
    # Give curvature the same proper zero-mean Gaussian prior as local residual
    # features. This changes the terrain model, not the physical risk limits;
    # assumptions/predictions still contribute no training observations.
    prior[3:]=1/p.residual_std_m**2
    if plane_context is not None:prior[:]=1/p.residual_std_m**2
    precision=precision+torch.diag(prior)
    chol,info=torch.linalg.cholesky_ex(precision)
    if info.item()!=0:return reject('posterior_factorization_failed')
    measured_z=torch.as_tensor(xyz[:,2],dtype=dtype,device=target)
    z=measured_z
    train_plane=None;query_plane=None
    if plane_context is not None:
        origin=torch.as_tensor(plane_context.anchor_xy,dtype=dtype,device=target)
        train_plane=torch.cat((training-origin,torch.ones((len(training),1),device=target,dtype=dtype)),dim=1)
        query_plane=torch.cat((torch.as_tensor(query_xy,dtype=dtype,device=target)-origin,
            torch.ones((len(query_xy),1),device=target,dtype=dtype)),dim=1)
        plane_coeff=torch.as_tensor(plane_context.coefficients,dtype=dtype,device=target)
        z=z-train_plane@plane_coeff
    z_center=z.mean() if plane_context is None else torch.zeros((),dtype=dtype,device=target)
    mean_weights=torch.cholesky_solve((design.T@(weight*(z-z_center)))[:,None],chol)[:,0]
    query_gate=gate(query_xy)
    query_design=_features(torch.as_tensor(query_xy,dtype=dtype,device=target),center_t,knots,p,query_gate,detail_knots)
    if plane_context is not None:query_design=query_design[:,3:]
    residual=z-(design@mean_weights+z_center)
    discrepancy=max(p.discrepancy_std_m,torch.sqrt(torch.mean(residual.square())).item())
    local_variance=None
    local_prior=None
    if p.condition_local_residual:
        # Model unresolved cell-scale structure as a Gaussian latent residual,
        # and condition it on the SAME real measurements as the trend. Merely
        # adding its fitted RMS AFTER conditioning makes already measured
        # ledges uncertain forever. This is an empirical-Bayes model choice,
        # not a calibration claim and not a reduction of physical thresholds.
        residual2=(discrepancy**2 if p.condition_full_residual else
                   max(0.,discrepancy**2-p.discrepancy_std_m**2))
        local_train=torch.full_like(weight,residual2)
        if training_gate is not None:local_train*=training_gate.square()
        residual_query=None
        if p.residual_inducing_points:
            extra_train,extra_query,local_train,residual_query=residual_fic_features(
                training,torch.as_tensor(query_xy,dtype=dtype,device=target),
                count=p.residual_inducing_points,lengthscale=p.residual_lengthscale_m,variance=residual2,
                training_gate=training_gate,query_gate=query_gate,training_xy=xyz[:,:2])
            design=torch.cat((design,extra_train),dim=1)
            query_design=torch.cat((query_design,extra_query),dim=1)
            prior=torch.cat((prior,torch.ones(extra_train.shape[1],dtype=dtype,device=target)))
        noise_t=torch.as_tensor(noise,dtype=dtype,device=target)
        weight=1/(noise_t+local_train)
        precision=design.T@(weight[:,None]*design)+torch.diag(prior)
        chol,info=torch.linalg.cholesky_ex(precision)
        if info.item()!=0:return reject('residual_factorization_failed')
        mean_weights=torch.cholesky_solve((design.T@(weight*(z-z_center)))[:,None],chol)[:,0]
        local_prior=torch.full((len(query_xy),),residual2,dtype=dtype,device=target)
        if query_gate is not None:local_prior*=query_gate.square()
        if residual_query is not None:local_prior=residual_query
        # The full model retains the same >=1cm residual PRIOR at unobserved
        # cells; it conditions it only where the original likelihood exists.
        # It does not remove sensor noise or any post-fit geometric floor.
        # This is a different cell-local Gaussian kernel, not a claim that
        # the legacy spatial RBF covariance was algebraically simplified.
        discrepancy=0. if p.condition_full_residual else p.discrepancy_std_m
    mean=query_design@mean_weights+z_center
    if plane_context is not None:mean+=query_plane@plane_coeff
    # B B^T is the shared latent covariance. No per-query independent model.
    factor=torch.linalg.solve_triangular(chol,query_design.T,upper=False).T
    plane_factor=None
    if plane_context is not None:
        # Sensitivity to the estimated plane includes its subtraction from
        # the REAL training heights. Dq alone would double-count datum error.
        sensitivity=query_plane-query_design@torch.cholesky_solve(design.T@(weight[:,None]*train_plane),chol)
        bound=torch.as_tensor(plane_context.coefficient_std_bound,dtype=dtype,device=target)
        if plane_context.coefficient_covariance_bound is None:
            plane_factor=math.sqrt(3.)*sensitivity*bound[None]
        else:
            eigen,vectors=np.linalg.eigh(plane_context.coefficient_covariance_bound)
            root=torch.as_tensor(vectors*np.sqrt(np.maximum(eigen,0))[None],dtype=dtype,device=target)
            plane_factor=sensitivity@root
    if local_prior is not None:
        # Conditional h_i = alpha*y_i + (1-alpha)*phi_i*w + independent error.
        # Cross-covariance is retained through the scaled shared factor, with
        # diagonal local variance carried separately. Only selected REAL
        # training cells receive this update; no assumptions/predictions do.
        tr,tc=np.where(fresh);mapped=query_lookup[tr[ids]*shape[1]+tc[ids]]
        if (mapped<0).any():raise ValueError('conditioning source outside posterior query tile')
        qids=torch.as_tensor(mapped,device=target)
        alpha=local_train/(noise_t+local_train)
        flat_mean=mean.reshape(-1)
        flat_mean[qids]=alpha*measured_z+(1-alpha)*flat_mean[qids]
        factor[qids]*=(1-alpha)[:,None]
        if plane_factor is not None:plane_factor[qids]*=(1-alpha)[:,None]
        local_prior[qids]=local_train*noise_t/(noise_t+local_train)
    if plane_factor is not None:
        # Unknown cross covariance of plane estimate / GP estimator errors:
        # Cov(e_G+e_I) <= 2 Cov(e_G) + 2 Cov(e_I). This deliberately inflates
        # the certificate; it never declares the two sources independent.
        factor=math.sqrt(2.)*torch.cat((factor,plane_factor),dim=1)
        discrepancy*=math.sqrt(2.)
        if local_prior is not None:
            local_prior*=2
    variance_out=(factor.square().sum(dim=1)+discrepancy**2+
                  (local_prior if local_prior is not None else 0))
    def expand(values,fill=np.nan):
        result=np.full(len(query),fill,dtype=values.dtype)
        result[query_ids]=values
        return result.reshape(shape)
    mean=expand(mean.cpu().numpy());variance_out=expand(variance_out.cpu().numpy())
    if local_prior is not None:local_variance=expand(local_prior.cpu().numpy()).astype(np.float32)
    relative_variance=None
    if plane_context is not None:
        tr,tc=np.where(fresh);training_ids=tr[ids]*shape[1]+tc[ids]
        nearest_training=cKDTree(xyz[:,:2]).query(query_xy,workers=1)[1]
        relative_variance=relative_height_variance(factor,local_prior,discrepancy,
            p.discrepancy_lengthscale_m,query_xy,query_lookup[training_ids[nearest_training]]).cpu().numpy()
        relative_variance=expand(relative_variance)
    # Optional bounded extrapolation is a model assumption, not newly measured
    # ground. It retains the full shared covariance, uncertainty gate, independent
    # measured-step veto and subsequent whole-vehicle chance constraints.
    domain=enclosed | (nearest<=p.max_extrapolation_m) if p.max_extrapolation_m>0 else enclosed
    fixed_initial=np.zeros(shape,bool)
    if p.max_extrapolation_m>0 and initial_support is not None:
        if initial_height is None or initial_support.shape!=shape or initial_height.shape!=shape:
            raise ValueError('fixed initial support requires matching height geometry')
        # The explicitly authorized fixed stance can also admit the existing
        # shared posterior at its original footprint. It contributes NO training
        # samples. A common uncertain height offset can cancel in slope/step
        # contrasts; do not substitute unrelated cell errors for that covariance.
        # Keep the measured-plane agreement check and retain the actual, possibly
        # larger (up to 8cm), absolute sigma for display and probability propagation.
        fixed_initial=initial_support&np.isfinite(initial_height)&(np.abs(mean-initial_height)<=.05)
        domain |= fixed_initial
    common=query_region&domain&~blocked&~contradicted&np.isfinite(mean)&np.isfinite(variance_out)
    assumed=np.zeros(shape,bool)
    if initial_support is not None:
        if initial_height is None or initial_support.shape!=shape or initial_height.shape!=shape:
            raise ValueError('fixed initial support requires matching height geometry')
        assumed=initial_support&common&np.isfinite(initial_height)&(np.abs(mean-initial_height)<=.05)
    valid=common&((nearest<=p.max_distance_m)|measured|assumed)
    confidence_variance=variance_out if relative_variance is None else relative_variance
    if plane_context is None:
        valid &= (variance_out<=p.max_std_m**2)|(fixed_initial&(variance_out<=.08**2))
    # A source-aware field is admitted for EVALUATION by support provenance,
    # distance, contradictions and finite covariance, not by an absolute
    # height-sigma threshold. Full-pose chance constraints consume that same
    # unreduced covariance and still reject unsafe predictions.
    # Measurements retain their raw geometry. A discrepant model cannot erase
    # a narrow step/rock merely because the basis is smooth.
    valid &= ~observed|(np.abs(mean-elevation)<=tolerance)
    difference=raw[:,2]-mean[rr,cc]
    positive=((difference>max_step)&(difference<positive_veto_height)
              if positive_veto_height is not None else np.zeros(len(difference),bool))
    # An unbounded extrapolation is not evidence against real measured ground.
    # Model-relative vetoes apply only to an actually admissible completion
    # proposal. Measured-ground contradictions were checked independently above.
    contradiction=((difference < -max_step)|positive)&valid[rr,cc]&~observed[rr,cc]
    counts=np.bincount(rr[contradiction]*shape[1]+cc[contradiction],minlength=elevation.size)
    contradicted |= counts.reshape(shape)>=2
    valid &= ~contradicted
    # A smooth posterior must not bridge a missing patch beside a measured
    # discontinuity larger than the vehicle's step limit. These original
    # surface samples are witnesses, including ones excluded from GP training.
    # This is a completion-domain veto, never an inflated collision layer or
    # permission to erase observed ground. The fixed, explicitly authorized
    # initial stance remains separately tagged and independently contradicted.
    # Occupied vertical columns are not a 2.5D ground surface. Their scattered
    # low/top returns must not cast a completion shadow across the corridor
    # floor; the independent full-body column check still forbids the columns.
    # Match the training-ground exclusion above: a one-cell column/ground
    # registration seam is not independent terrain evidence. This mask is
    # used only to select witnesses, never to inflate or clear collision cells.
    witnessed=observed&ground_witness&np.isfinite(elevation)&np.isfinite(variance)&(variance>=0)&(variance<=p.measurement_std_m**2)
    edge=np.zeros(shape,bool)
    for axis in (0,1):
        a=[slice(None),slice(None)];b=a.copy();a[axis]=slice(None,-1);b[axis]=slice(1,None)
        a,b=tuple(a),tuple(b)
        jump=witnessed[a]&witnessed[b]&(np.abs(elevation[a]-elevation[b])>max_step)
        edge[a]|=jump;edge[b]|=jump
    shadow=np.zeros(shape,bool)
    column_floor_supported=np.zeros(shape,bool)
    if edge.any():
        distance=cKDTree(np.c_[x[edge],y[edge]]).query(query,workers=1)[0].reshape(shape)
        shadow=(distance<=p.max_distance_m)&~observed&~assumed
        if p.column_shadow_floor_reference and np.any(shadow&column):
            from .obstacle_columns import measured_collision_floor_witness
            # A 2.5D grid square can contain a narrow column AND road. A jump
            # elsewhere must not erase the latter by radial dilation. Only
            # ORIGINAL admissible floor neighbours enter this enclosed local
            # reference; neither the GP nor the column candidate enters it.
            reference,_=measured_collision_floor_witness(elevation,variance,fresh,column,resolution,max_step_m=max_step)
            column_floor_supported=shadow&column&valid&np.isfinite(reference)&(
                np.abs(mean-reference)<=min(.03,.5*max_step))
            shadow &= ~column_floor_supported
        valid &= ~shadow
    factor=factor.to(torch.float32)
    if p.bounded_queries:
        expanded_factor=torch.zeros((len(query),factor.shape[1]),dtype=factor.dtype,device=target)
        expanded_factor[torch.as_tensor(query_ids,device=target)]=factor
        factor=expanded_factor
    factor=factor.reshape(*shape,-1)
    if target.type=='cuda':torch.cuda.current_stream(target).synchronize()
    if diagnostics is not None:diagnostics['reason']='posterior_ready'
    return SharedTerrainPrediction(mean.astype(np.float32),variance_out.astype(np.float32),
        valid,valid&~observed&~assumed,nearest.astype(np.float32),factor,discrepancy,
        len(xyz),(time.perf_counter()-started)*1000,assumed&valid,p.discrepancy_lengthscale_m,
        common&~contradicted&~shadow&((confidence_variance<=.08**2)|(fixed_initial if plane_context is not None else False)),contradicted,shadow,flat_tree is not None,len(sites),local_variance,
        plane_context is not None,relative_variance,0 if plane_factor is None else plane_factor.shape[1],column_floor_supported,
        admission_domain=(query_region&domain&~contradicted&~shadow&np.isfinite(mean)&
            np.isfinite(variance_out)&(nearest<=p.max_distance_m)&(variance_out<=p.max_std_m**2)))


@lru_cache(maxsize=256)
def _contrast_discrepancy(params,yaw_index,lengthscale):
    offsets,weights,_=_geometry(params,yaw_index)
    xy=np.asarray(offsets,dtype=np.float64)*params.resolution
    covariance=np.exp(-np.sum((xy[:,None]-xy[None])**2,axis=2)/(2*lengthscale**2))
    projected=weights.astype(np.float64)@covariance@weights.T
    count=len(offsets);pi,pj=np.triu_indices(count,1)
    diagonal=np.diag(projected)
    pair=diagonal[2+pi]+diagonal[2+pj]-2*projected[2+pi,2+pj]
    return np.maximum(diagonal,0).astype(np.float32),np.maximum(pair,0).astype(np.float32)


@lru_cache(maxsize=256)
def _device_geometry(params,k,lengthscale,device):
    offsets,weights,limits=_geometry(params,k)
    count=len(offsets);pi,pj=np.triu_indices(count,1)
    disc,pair_disc=_contrast_discrepancy(params,k,lengthscale)
    values=(np.asarray(offsets),weights,limits,pi,pj,
        np.abs(weights[2+pi]-weights[2+pj]).sum(axis=1),disc,pair_disc)
    return offsets,tuple(torch.as_tensor(v,device=device) for v in values)


def integrate_independent_model_floor(prediction,params):
    """Explicit diagonal nugget prior; not valid for unknown floor covariance.

    The original plane/GP source cross-correlation still uses a triangle
    envelope. Fold the independent nugget into the GP group, conservatively
    retaining that triangle. The sqrt(2) source-envelope storage convention
    requires doubling this group's diagonal when plane components are present.
    """
    if not params.independent_model_floor or not params.height_std_floor:return prediction,params
    amount=params.height_std_floor**2/params.variance_scale*(2 if prediction.plane_components else 1)
    local=np.full(prediction.mean.shape,amount,np.float32) if prediction.local_variance is None else prediction.local_variance+amount
    return replace(prediction,local_variance=local),replace(params,height_std_floor=0.)


def shared_geometric_probability(prediction,params,collision,*,fused=True,tighten_span=True,query_mask=None):
    """Full-pose Gaussian contrast bounds using the same latent terrain field.

    The opt-in model discrepancy is a Gaussian spatial RBF field independent
    of the finite-feature posterior. Its lengthscale is a model assumption,
    not calibration. The additional sensor floor still allows arbitrary
    correlation via an L1 bound. The residual-span event has two valid union
    bounds: half-span individual residuals and full-span pair differences.
    Their minimum tightens the proof, not the physical limit or GP covariance.
    Path events are never multiplied.
    """
    prediction,params=integrate_independent_model_floor(prediction,params)
    factor=prediction.factor
    bias=prediction.bounded_mean_factor
    plane_components=prediction.plane_components
    if not 0<=plane_components<factor.shape[-1]:
        raise ValueError('invalid source covariance partition')
    device=factor.device
    if query_mask is not None:
        query_mask=np.asarray(query_mask,dtype=bool)
        if query_mask.shape!=(params.yaw_bins,*prediction.mean.shape):
            raise ValueError('risk refinement mask must match heading and map geometry')
    if fused and device.type=='cuda':
        from .shared_terrain_cuda import shared_probability_cuda
        return shared_probability_cuda(prediction,params,collision,tighten_span=tighten_span,query_mask=query_mask)
    height=torch.as_tensor(prediction.mean,dtype=torch.float32,device=device)
    valid=torch.as_tensor(prediction.valid,dtype=torch.bool,device=device)
    occupied=torch.as_tensor(collision,dtype=torch.bool,device=device)
    rr,cc=torch.where(valid)
    rows,cols=height.shape
    floor=params.height_std_floor
    local=None if prediction.local_variance is None else torch.as_tensor(
        prediction.local_variance,dtype=torch.float32,device=device)
    updates=[]
    for k in range(params.yaw_bins):
        offsets,tensors=_device_geometry(params,k,prediction.discrepancy_lengthscale_m,str(device))
        delta,w,bound,pair_i,pair_j,pair_l1,disc,pair_disc=tensors
        r=rr[:,None]+delta[:,0];c=cc[:,None]+delta[:,1]
        inside=((r>=0)&(r<rows)&(c>=0)&(c<cols)).all(dim=1)
        r=r.clamp(0,rows-1);c=c.clamp(0,cols-1)
        eligible=inside&valid[r,c].all(dim=1)&~occupied[r,c].any(dim=1)
        if query_mask is not None:
            eligible &= torch.as_tensor(query_mask[k],device=device)[rr,cc]
        r=r[eligible];c=c[eligible];sr=rr[eligible];sc=cc[eligible]
        risk=[]
        count=len(offsets)
        disc=disc*prediction.discrepancy**2
        pair_disc=pair_disc*prediction.discrepancy**2
        batch=2048 if device.type=='cuda' else 128
        for begin in range(0,len(r),batch):
            ro,co=r[begin:begin+batch],c[begin:begin+batch]
            h=height[ro,co];h=h-h[:,offsets.index((0,0)),None]
            means=h@w.T
            shifts=(torch.einsum('mn,bnp->bmp',w,bias[ro,co]).abs().sum(dim=2)
                    if bias is not None else 0.)
            worst=means.abs()+shifts
            transformed=torch.einsum('mn,bnp->bmp',w,factor[ro,co])
            local_contrast=0. if local is None else local[ro,co]@w.square().T
            if plane_components:
                gp_variance=transformed[:,:,:-plane_components].square().sum(dim=2)+disc[None]+local_contrast
                plane_variance=transformed[:,:,-plane_components:].square().sum(dim=2)
                sigma=(gp_variance.clamp_min(0).sqrt()+plane_variance.clamp_min(0).sqrt())*math.sqrt(.5*params.variance_scale)
            else:
                contrast_variance=transformed.square().sum(dim=2)+disc[None]+local_contrast
                sigma=contrast_variance.clamp_min(0).sqrt()*math.sqrt(params.variance_scale)
            sigma=sigma+w.abs().sum(dim=1)[None]*floor
            sigma=sigma.clamp_min(1e-12)
            tails=.5*torch.special.erfc((bound+worst)/(math.sqrt(2)*sigma))+.5*torch.special.erfc(
                (bound-worst)/(math.sqrt(2)*sigma))
            tails=torch.where(worst>=bound,torch.ones_like(tails),tails)
            # Direct residual-span certificate, not the stricter sufficient
            # condition |r_i| <= h_max/2. Project covariance once, then form
            # pair variances by a small Gram matrix, avoiding P work per pair.
            span_upper=tails[:,2:2+count].sum(dim=1)
            # An optional tighter proof is unnecessary when the cheap proof
            # is already tight. Skipping it cannot lower a probability bound.
            # A tighter cheap bound can fall below the old refinement trigger
            # while its predecessor used the better pair proof. Refine source
            # partitions down to numerical slack, avoiding that discontinuity.
            tighten=(span_upper>(1e-7 if plane_components else 1e-3)) if tighten_span else torch.zeros_like(span_upper,dtype=torch.bool)
            if tighten.any().item():
                residual_factor=transformed[tighten,2:2+count]
                def pair_variance(values,include_local):
                    gram=values@values.transpose(1,2)
                    if include_local and local is not None:
                        residual_weights=w[2:2+count]
                        gram+=(residual_weights[None]*local[ro,co][tighten,None,:])@residual_weights.T
                    norm=gram.diagonal(dim1=1,dim2=2)
                    return (norm[:,pair_i]+norm[:,pair_j]-2*gram[:,pair_i,pair_j]).clamp_min(0)
                if plane_components:
                    vg=pair_variance(residual_factor[:,:,:-plane_components],True)+pair_disc[None]
                    vi=pair_variance(residual_factor[:,:,-plane_components:],False)
                    pair_sigma=(vg.clamp_min(0).sqrt()+vi.sqrt())*math.sqrt(.5*params.variance_scale)
                else:
                    pair_sigma=(pair_variance(residual_factor,True)+pair_disc[None]).sqrt()*math.sqrt(params.variance_scale)
                pair_sigma+=pair_l1[None]*floor
                pair_sigma=pair_sigma.clamp_min(1e-12)
                selected=means[tighten]
                pair_mean=selected[:,2+pair_i]-selected[:,2+pair_j]
                if bias is not None:
                    projected_bias=torch.einsum('mn,bnp->bmp',w[2:2+count],bias[ro,co][tighten])
                    pair_mean=pair_mean.abs()+(projected_bias[:,pair_i]-projected_bias[:,pair_j]).abs().sum(dim=2)
                pair_tail=.5*torch.special.erfc((params.roughness_limit+pair_mean)/(math.sqrt(2)*pair_sigma))+.5*torch.special.erfc(
                    (params.roughness_limit-pair_mean)/(math.sqrt(2)*pair_sigma))
                pair_tail=torch.where(pair_mean.abs()>=params.roughness_limit,torch.ones_like(pair_tail),pair_tail)
                span_upper[tighten]=torch.minimum(span_upper[tighten],pair_tail.sum(dim=1))
            upper=(tails[:,:2].sum(dim=1)+span_upper+tails[:,2+count:].sum(dim=1)).clamp_max(1)
            risk.append((upper+1e-7).clamp_max(1))
        updates.append((sr,sc,torch.cat(risk) if risk else torch.empty(0,device=device)))
    # One transfer after all headings, instead of three synchronizations per
    # heading. Coordinates are small exact integers representable in float32.
    counts=[len(r) for r,_,_ in updates]
    packed=torch.cat([torch.stack((r,c,u),dim=1) for r,c,u in updates]).cpu().numpy()
    result=[];start=0
    for count in counts:
        a=packed[start:start+count];start+=count
        result.append((a[:,0].astype(int),a[:,1].astype(int),a[:,2]))
    return result
