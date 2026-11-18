"""Discrete SE(2) geometric exceedance bounds, not calibrated vehicle failures.

Height errors are assumed jointly Gaussian with supplied marginal variances.
Unknown correlations are bounded by Cauchy-Schwarz; no independence is needed
for either the feature bounds or their union. See docs/PROBABILISTIC_NAVIGATION.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np


@dataclass(frozen=True)
class ProbabilityParams:
    resolution: float = 0.2
    yaw_bins: int = 36
    front: float = 0.42
    back: float = 0.42
    half_width: float = 0.313
    margin: float = 0.10
    max_pitch: float = 0.55
    max_roll: float = 0.45
    max_step: float = 0.18
    height_std_floor: float = 0.005
    variance_scale: float = 1.0
    internal_steps: bool = False
    internal_step_model: str = 'second_difference'
    exact_footprint: bool = False  # legacy recordings retain their original stencil
    # Local wheel-crossing discontinuity and full-body residual span are
    # different physical constraints. None preserves recorded legacy maps.
    max_roughness: float | None = None
    # Explicit, opt-in joint Gaussian nugget. False retains the legacy bound
    # for arbitrary model-floor correlation. Its standard deviation is NOT cut.
    independent_model_floor: bool = False

    @property
    def roughness_limit(self):
        return self.max_step if self.max_roughness is None else self.max_roughness

    def __post_init__(self):
        positive = (self.resolution, self.front, self.back, self.half_width,
                    self.max_pitch, self.max_roll, self.max_step, self.variance_scale)
        if not all(math.isfinite(v) and v > 0 for v in positive):
            raise ValueError("probability geometry and variance scale must be finite and positive")
        if (self.yaw_bins < 1 or not math.isfinite(self.margin) or
                not math.isfinite(self.height_std_floor) or self.margin < 0 or self.height_std_floor < 0):
            raise ValueError("invalid probability support parameters")
        if max(self.max_pitch, self.max_roll) >= math.pi / 2:
            raise ValueError("pitch/roll limits must be below pi/2")
        if self.internal_step_model not in ('second_difference','height_jump'):
            raise ValueError('unknown internal terrain constraint model')
        if not math.isfinite(self.roughness_limit) or self.roughness_limit<=0:
            raise ValueError('roughness span must be finite and positive')


@lru_cache(maxsize=256)
def _geometry(p: ProbabilityParams, yaw_index: int):
    yaw = (yaw_index + 0.5) * 2 * math.pi / p.yaw_bins
    c, s = math.cos(yaw), math.sin(yaw)
    # Conservatively cover all grid squares intersecting the oriented footprint,
    # rather than sparsely sampling corners and missing a one-cell obstacle.
    front, back, width = p.front+p.margin, p.back+p.margin, p.half_width+p.margin
    extent = math.ceil(math.hypot(max(front,back),width)/p.resolution)+1
    pad = 0.5*p.resolution*(abs(c)+abs(s))
    half=(front+back)/2;offset=(front-back)/2
    extent_x=abs(c)*half+abs(s)*width+.5*p.resolution
    extent_y=abs(s)*half+abs(c)*width+.5*p.resolution
    offsets = []
    for dr in range(-extent,extent+1):
        for dc in range(-extent,extent+1):
            x, y = p.resolution*(c*dc+s*dr), p.resolution*(-s*dc+c*dr)
            world_axes=(abs(p.resolution*dc-offset*c)<=extent_x+1e-9 and
                        abs(p.resolution*dr-offset*s)<=extent_y+1e-9)
            if (-back-pad <= x <= front+pad and abs(y) <= width+pad and
                    (not p.exact_footprint or world_axes)):
                offsets.append((dr,dc))
    xy = np.array([(p.resolution*(c*dc+s*dr), p.resolution*(-s*dc+c*dr))
                   for dr, dc in offsets])
    design = np.column_stack((xy, np.ones(len(xy))))
    if np.linalg.matrix_rank(design) < 3:
        raise ValueError("footprint grid is too coarse to estimate an oriented plane")
    plane = np.linalg.pinv(design)
    residual = np.eye(len(xy)) - design @ plane
    # |residual_i| <= roughness/2 implies span <= roughness. A local vertical
    # step is separately constrained by max_step, including under contract 7.
    # This is a sufficient, deliberately conservative geometric condition.
    extra=[]
    if p.internal_steps:
        lookup={offset:i for i,offset in enumerate(offsets)}
        for r,c in offsets:
            for dr,dc in ((0,1),(1,0)):
                if p.internal_step_model=='height_jump':
                    other=(r+dr,c+dc)
                    if other in lookup:
                        row=np.zeros(len(offsets));row[lookup[other]]=1.;row[lookup[(r,c)]]=-1.
                        extra.append(row)
                    continue
                triplet=((r-dr,c-dc),(r,c),(r+dr,c+dc))
                if all(q in lookup for q in triplet):
                    row=np.zeros(len(offsets))
                    for q,w in zip(triplet,(1.,-2.,1.)): row[lookup[q]]=w
                    extra.append(row)
    weights = np.vstack((plane[:2], residual,*extra)).astype(np.float32)
    limits = np.array([math.tan(p.max_pitch), math.tan(p.max_roll)] +
                      [p.roughness_limit/2]*len(xy)+[p.max_step]*len(extra), dtype=np.float32)
    return offsets, weights, limits


def _gather(array, offsets, xp, fill):
    pad = max(max(abs(r), abs(c)) for r, c in offsets)
    padded = xp.pad(array, pad, mode="constant", constant_values=fill)
    h, w = array.shape
    return xp.stack([padded[pad+r:pad+r+h, pad+c:pad+c+w] for r, c in offsets])


@lru_cache(maxsize=256)
def _span_pairs(p, yaw_index):
    offsets, weights, _ = _geometry(p, yaw_index)
    i,j=np.triu_indices(len(offsets),1)
    return weights[2+i]-weights[2+j]


@lru_cache(maxsize=256)
def _envelope_geometry(p, yaw_index):
    offsets, weights, _ = _geometry(p,yaw_index)
    residual=weights[2:2+len(offsets)].astype(np.float64)
    span_norm=max(np.abs(a-b).sum() for i,a in enumerate(residual) for b in residual[:i])
    support=set(offsets)
    if not p.internal_steps:
        for r,c in offsets:
            for dr,dc in ((0,1),(1,0)):
                support.update((r+k*dr,c+k*dc) for k in (-1,0,1,2))
    return float(span_norm),len(support),max(max(abs(r),abs(c)) for r,c in support)


def height_support_radius(p):
    return max(_envelope_geometry(p,k)[2] for k in range(p.yaw_bins))


def _interval_upper(mean, std_upper, limit, xp, ndtr):
    denom = xp.maximum(std_upper, xp.float32(1e-12))
    tails = ndtr((-limit-mean)/denom) + ndtr((mean-limit)/denom)
    # Increasing variance increases this tail only when the mean is inside.
    return xp.where(xp.abs(mean) >= limit, 1, tails)


def _step_edges(height, sigma, valid, params, xp, ndtr):
    upper = xp.zeros_like(height)
    supported = valid.copy()
    standardized_margin=xp.full_like(height,xp.inf)
    absolute_radius=xp.full_like(height,xp.inf)
    for dr, dc in ((0, 1), (1, 0)):
        offsets = [(-dr,-dc),(0,0),(dr,dc),(2*dr,2*dc)]
        h = _gather(xp.where(valid,height,0),offsets,xp,0)
        s = _gather(xp.where(valid,sigma,0),offsets,xp,0)
        support = xp.all(_gather(valid,offsets,xp,False),axis=0)
        # Compare an edge increment with its two same-side increments. Planes
        # and quadratics cancel; an isolated vertical jump is preserved.
        mean=(h[2]-h[1])-0.5*((h[1]-h[0])+(h[3]-h[2]))
        std_upper=0.5*s[0]+1.5*s[1]+1.5*s[2]+0.5*s[3]
        upper += _interval_upper(mean,std_upper,params.max_step,xp,ndtr)
        standardized_margin=xp.minimum(standardized_margin,
            (params.max_step-xp.abs(mean))/xp.maximum(std_upper,1e-12))
        absolute_radius=xp.minimum(absolute_radius,(params.max_step-xp.abs(mean))/4)
        supported &= support
    return upper,supported,standardized_margin,absolute_radius


def geometric_probability(elevation, variance, observed, params: ProbabilityParams,
                          *, collision=None, xp=np, diagnostics=False, bounded_radius=None):
    """Return SO2 layers; unsupported poses are NaN unless occupancy proves risk.

    All footprint cells and edge stencils need admissible height AND variance
    support. The caller controls whether qualified GP support is admissible.
    CuPy executes gathering, feature projection and Gaussian tails on the GPU;
    only cached small geometry matrices are formed on the CPU.
    """
    if xp is np:
        from scipy.special import ndtr
        to_host = np.asarray
    else:
        from cupyx.scipy.special import ndtr
        to_host = xp.asnumpy
    h = xp.asarray(elevation, dtype=xp.float32)
    v = xp.asarray(variance, dtype=xp.float32)
    valid = (xp.asarray(observed, dtype=bool) & xp.isfinite(h) & xp.isfinite(v) & (v >= 0))
    if h.ndim != 2 or h.shape != v.shape or h.shape != valid.shape:
        raise ValueError("probability input shapes must be equal 2D grids")
    occupied = xp.zeros_like(valid) if collision is None else xp.asarray(collision,dtype=bool)
    if occupied.shape != h.shape:
        raise ValueError("collision mask must match elevation geometry")
    sigma = xp.sqrt(xp.maximum(v, 0)*params.variance_scale + params.height_std_floor**2)
    rho=None
    if bounded_radius is not None:
        rho=xp.asarray(bounded_radius,dtype=xp.float32)
        if (rho.shape!=h.shape or not params.internal_steps or
                bool(xp.any(~xp.isfinite(rho)|(rho<0)))):
            raise ValueError('bounded mean radii need finite nonnegative internal-step geometry')
    if xp is not np and not diagnostics:
        from .probability_cuda import fused_probability
        fused=fused_probability(h,sigma,valid,occupied,params,bounded_radius=rho)
        if fused is not None: return fused
    if params.internal_steps:
        step_upper=xp.zeros_like(h);step_support=valid;edge_margin=xp.full_like(h,xp.inf)
        edge_radius=xp.full_like(h,xp.inf)
    else:
        step_upper, step_support, edge_margin, edge_radius = _step_edges(h,sigma,valid,params,xp,ndtr)
    layers = {"failure_probability_upper": [], "probability_valid": [],
              "height_error_tail_upper": [], "height_error_radius_m": []}
    if diagnostics:
        layers.update({k: [] for k in ("pitch_probability_upper", "roll_probability_upper",
                                      "step_probability_upper", "probability_support")})
    for yaw_index in range(params.yaw_bins):
        offsets, host_weights, host_limits = _geometry(params, yaw_index)
        sample_valid = _gather(valid, offsets, xp, False)
        supported = xp.all(sample_valid, axis=0) & xp.all(_gather(step_support,offsets,xp,False),axis=0)
        footprint_occupied = xp.any(_gather(occupied,offsets,xp,False),axis=0)
        heights = _gather(xp.where(valid, h, 0), offsets, xp, 0)
        stds = _gather(xp.where(valid, sigma, 0), offsets, xp, 0)
        weights = xp.asarray(host_weights)
        # Translation-invariant contrasts suppress cancellation at large map Z.
        heights = heights - heights[offsets.index((0, 0))][None]
        means = xp.tensordot(weights, heights, axes=(1, 0))
        bound_std = xp.tensordot(xp.abs(weights), stds, axes=(1, 0))
        limits = xp.asarray(host_limits)[:, None, None]
        worst=(means if rho is None else xp.abs(means)+xp.tensordot(
            xp.abs(weights),_gather(rho,offsets,xp,0),axes=(1,0)))
        tails = _interval_upper(worst,bound_std,limits,xp,ndtr)
        edge_upper = xp.sum(_gather(step_upper,offsets,xp,0),axis=0)
        total = xp.minimum(1, xp.sum(tails, axis=0)+edge_upper)
        if rho is not None:
            # Conservative marginal bound for EVERY family member. The shared
            # covariance calculation may tighten it. Do not reuse the old
            # pure-Gaussian absolute-height certificate for this mean family.
            total=xp.where(footprint_occupied,1,xp.where(supported,xp.minimum(1,total+1e-7),xp.nan))
            layers['failure_probability_upper'].append(total.astype(xp.float32))
            layers['probability_valid'].append((supported|footprint_occupied).astype(xp.float32))
            for name in ('height_error_tail_upper','height_error_radius_m'):
                layers[name].append(xp.where(supported&~footprint_occupied,1.,xp.nan).astype(xp.float32))
            if diagnostics:
                for name,values in (('pitch_probability_upper',tails[0]),('roll_probability_upper',tails[1]),
                    ('step_probability_upper',xp.minimum(1,xp.sum(tails[2:],axis=0))),
                    ('probability_support',xp.mean(sample_valid,axis=0))):
                    layers[name].append(xp.where(supported,values,xp.nan).astype(xp.float32))
            continue
        # A single marginal-height confidence envelope certifies ALL geometric
        # contrasts. Its source events can be reused across overlapping poses.
        # The physical roughness constraint is residual max-min <= max_step;
        # unlike the old sufficient half-step condition, it need not reject a
        # small asymmetric residual solely because it exceeds max_step/2.
        span_norm,support_count,_=_envelope_geometry(params,yaw_index)
        n=len(offsets)
        span_mean=xp.max(means[2:2+n],axis=0)-xp.min(means[2:2+n],axis=0)
        span_margin=(params.roughness_limit-span_mean)/xp.maximum(span_norm*xp.max(stds,axis=0),1e-12)
        slope_margin=xp.min((limits[:2]-xp.abs(means[:2]))/xp.maximum(bound_std[:2],1e-12),axis=0)
        margin=xp.minimum(xp.minimum(span_margin,slope_margin),
                          xp.min(_gather(edge_margin,offsets,xp,-xp.inf),axis=0))
        if params.internal_steps and len(host_weights)>n+2:
            margin=xp.minimum(margin,xp.min((limits[n+2:]-xp.abs(means[n+2:]))/
                xp.maximum(bound_std[n+2:],1e-12),axis=0))
        norm=xp.sum(xp.abs(weights),axis=1)[:,None,None]
        absolute_radius=xp.minimum((params.roughness_limit-span_mean)/span_norm,
            xp.min((limits[:2]-xp.abs(means[:2]))/xp.maximum(norm[:2],1e-12),axis=0))
        absolute_radius=xp.minimum(absolute_radius,xp.min(_gather(edge_radius,offsets,xp,-xp.inf),axis=0))
        if params.internal_steps and len(host_weights)>n+2:
            absolute_radius=xp.minimum(absolute_radius,xp.min((limits[n+2:]-xp.abs(means[n+2:]))/
                xp.maximum(norm[n+2:],1e-12),axis=0))
        tail=xp.minimum(1,2*ndtr(-xp.maximum(margin,0)))
        total=xp.minimum(total,xp.minimum(1,support_count*tail))
        # The half-absolute-residual condition above is only sufficient for
        # physical residual span. A second union bound uses its exact pair
        # contrasts. Keep a conservative work floor for this extra proof:
        # min(old, max(floor, alternative)) is continuous at the work gate
        # and remains monotone in uncertainty, unlike a data-dependent blend.
        selected=supported & ~footprint_occupied & (total>.005) & (span_mean<params.roughness_limit)
        indices=xp.flatnonzero(selected)
        if indices.size:
            pairs=xp.asarray(_span_pairs(params,yaw_index))
            z=heights.reshape(n,-1)[:,indices]
            s=stds.reshape(n,-1)[:,indices]
            pair_upper=xp.sum(_interval_upper(pairs@z,xp.abs(pairs)@s,params.roughness_limit,xp,ndtr),axis=0)
            other=xp.sum(tails[:2],axis=0)+xp.sum(tails[2+n:],axis=0)+edge_upper
            original=total.ravel()[indices]
            alternative=xp.minimum(1,pair_upper+other.ravel()[indices])
            total.ravel()[indices]=xp.minimum(original,xp.maximum(.005,alternative))
        # Confirmed occupancy is a deterministic violated constraint in this
        # map model, even when the hidden ground beneath it is unsupported.
        total = xp.where(footprint_occupied,1,xp.where(supported,total,xp.nan))
        layers["failure_probability_upper"].append(total.astype(xp.float32))
        layers["probability_valid"].append((supported | footprint_occupied).astype(xp.float32))
        tail=xp.where(supported & ~footprint_occupied,tail,xp.nan)
        layers["height_error_tail_upper"].append(tail.astype(xp.float32))
        layers["height_error_radius_m"].append(xp.where(supported & ~footprint_occupied,
            xp.maximum(0,absolute_radius),xp.nan).astype(xp.float32))
        if diagnostics:
            for name, values in (("pitch_probability_upper", tails[0]),
                                 ("roll_probability_upper", tails[1]),
                                 ("step_probability_upper", xp.minimum(1, xp.sum(tails[2:], axis=0)+edge_upper)),
                                 ("probability_support", xp.mean(sample_valid, axis=0))):
                if name != "probability_support":
                    values = xp.where(supported,values,xp.nan)
                layers[name].append(values.astype(xp.float32))
    # Transfer whole layers, not each yaw slice. Per-slice host copies force
    # more than a hundred stream synchronizations per map on CUDA.
    return {name:list(to_host(xp.stack(slices))) for name,slices in layers.items()}
