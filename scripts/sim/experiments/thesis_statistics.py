"""Source-group statistics for evidence plots, not point-wise independence claims."""

from collections import defaultdict
import math

import numpy as np
from scipy.special import ndtr, ndtri


def source_values(rows, key):
    groups=defaultdict(list)
    for row in rows:
        value=row.get(key)
        if row.get("status")=="ok" and value is not None and np.isfinite(value):
            groups[row["cluster_id"]].append(float(value))
    return {group:float(np.mean(values)) for group,values in sorted(groups.items())}


def box_summary(values):
    a=np.asarray(values,dtype=float)
    if not len(a) or not np.isfinite(a).all():
        raise ValueError("boxplot needs finite nonempty source values")
    q1,median,q3=np.quantile(a,[.25,.5,.75]);iqr=q3-q1
    inside=a[(a>=q1-1.5*iqr)&(a<=q3+1.5*iqr)]
    low,high=float(inside.min()),float(inside.max())
    return {"n":len(a),"q1":float(q1),"median":float(median),"q3":float(q3),
        "whisker_low":low,"whisker_high":high,"outliers":int(((a<low)|(a>high)).sum()),
        "mean":float(a.mean()),"p95":float(np.quantile(a,.95)),"max":float(a.max())}


def source_mean_interval(values,resamples=2000,seed=2026101413):
    a=np.asarray(values,dtype=float)
    if a.ndim!=2 or len(a)<1 or not np.isfinite(a).all():
        raise ValueError("expected finite source x metric matrix")
    rng=np.random.default_rng(seed)
    means=a[rng.integers(0,len(a),(resamples,len(a)))].mean(axis=1)
    return np.quantile(means,[.025,.975],axis=0)


def threshold_counts(scores, unsafe, supported, thresholds):
    scores=np.asarray(scores)
    unsafe=np.asarray(unsafe,dtype=bool);supported=np.asarray(supported,dtype=bool)
    if scores.shape!=unsafe.shape or scores.shape!=supported.shape:
        raise ValueError("risk, hazard and support must have identical shape")
    selected=scores[supported]
    if not np.isfinite(selected).all() or (selected<0).any() or (selected>1).any():
        raise ValueError("risk score outside finite [0,1] contract")
    # Match NumPy's float32 scalar comparisons used by the original scorer.
    thresholds=np.asarray(thresholds,dtype=scores.dtype)
    positive=np.sort(scores[supported&unsafe]);negative=np.sort(scores[supported&~unsafe])
    tp=len(positive)-np.searchsorted(positive,thresholds,side="left")
    fp=len(negative)-np.searchsorted(negative,thresholds,side="left")
    return np.stack((tp,len(positive)-tp,fp,len(negative)-fp),axis=-1)


def equal_source_curve(group_counts):
    counts=np.asarray(group_counts,dtype=float)
    if counts.ndim!=3 or counts.shape[-1]!=4 or (counts<0).any():
        raise ValueError("expected source x threshold x four confusion counts")
    totals=counts.sum(axis=-1)
    if not (totals>0).all() or not np.allclose(totals,totals[:,0,None]):
        raise ValueError("each source must retain its reference denominator across thresholds")
    mass=(counts/totals[:,:,None]).mean(axis=0)
    tp,fn,fp,tn=mass.T
    def ratio(a,b):
        return np.divide(a,b,out=np.full_like(a,np.nan),where=b>0)
    return {"recall":ratio(tp,tp+fn),"fpr":ratio(fp,fp+tn),
        "precision":ratio(tp,tp+fp),"source_groups":len(counts),
        "weighted_prevalence":float((tp+fn)[0])}


def height_diagnostics(mean, variance, reference, observed, nominal):
    mean=np.asarray(mean,dtype=float).ravel();variance=np.asarray(variance,dtype=float).ravel()
    reference=np.asarray(reference,dtype=float).ravel();observed=np.asarray(observed,dtype=bool).ravel()
    if not (mean.shape==variance.shape==reference.shape==observed.shape):
        raise ValueError("height fields must share grid dimensions")
    if not np.isfinite(mean).all() or not np.isfinite(variance).all() or (variance<0).any():
        raise ValueError("nonfinite height or invalid posterior variance")
    mask=np.isfinite(reference)&~observed
    error=mean[mask]-reference[mask];std=np.sqrt(variance[mask]);nominal=np.asarray(nominal)
    if not ((nominal>0)&(nominal<1)).all():raise ValueError("nominal coverage outside (0,1)")
    widths=2*ndtri((1+nominal)/2)
    coverage=(np.abs(error)[:,None]<=std[:,None]*widths/2).sum(axis=0)
    crps=np.abs(error).copy();regular=std>1e-12
    u=error[regular]/std[regular]
    crps[regular]=std[regular]*(u*(2*ndtr(u)-1)+2*np.exp(-u*u/2)/np.sqrt(2*np.pi)-1/np.sqrt(np.pi))
    return {"n":len(error),"coverage_counts":coverage,"width_sums":std.sum()*widths,
        "crps_sum_m":float(crps.sum()),"signed_error_sum_m":float(error.sum())}


def zero_failure_upper(n,alpha=.05):
    if n<1 or not 0<alpha<1:raise ValueError("invalid binomial bound parameters")
    return float(-math.expm1(math.log(alpha)/n))


def zero_failure_required(limit,alpha=.05):
    if not 0<limit<1 or not 0<alpha<1:raise ValueError("invalid target bound")
    return math.ceil(math.log(alpha)/math.log1p(-limit))
