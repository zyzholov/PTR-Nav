"""Fused shared-field bounds with selective direct residual-span certification."""
from functools import lru_cache
import numpy as np


SOURCE=r'''
#include <math_constants.h>
__device__ float sum_warp(float value) {
  for(int d=16;d;d>>=1)value+=__shfl_down_sync(0xffffffff,value,d);
  return value;
}
__device__ float source_std(float gp,float plane,float disc,float scale,int plane_components) {
  if(plane_components)return (sqrtf(fmaxf(0.f,gp+disc))+sqrtf(fmaxf(0.f,plane)))*sqrtf(.5f*scale);
  return sqrtf(fmaxf(0.f,(gp+disc)*scale));
}
extern "C" __global__ void supported(const int *centers,int candidates,int rows,int cols,
    const unsigned char *valid,const unsigned char *occupied,const int *offsets,
    const int *counts,int stride,const unsigned char *query_mask,unsigned char *eligible) {
  int i=blockIdx.x*blockDim.x+threadIdx.x,k=blockIdx.y;
  if(i>=candidates)return;
  if(!query_mask[k*rows*cols+centers[i]]){eligible[k*candidates+i]=0;return;}
  int r=centers[i]/cols,c=centers[i]%cols;bool ok=true;
  for(int j=0;j<counts[k];j++) {
    int y=r+offsets[(k*stride+j)*2],x=c+offsets[(k*stride+j)*2+1];
    if(y<0||y>=rows||x<0||x>=cols){ok=false;break;}
    int q=y*cols+x;if(!valid[q]||occupied[q]){ok=false;break;}
  }
  eligible[k*candidates+i]=ok;
}
extern "C" __global__ void contrasts(const float *height,const float *factor,const float *local,int components,int plane_components,int bounded_components,
    const int *centers,int candidates,int cols,const unsigned char *eligible,
    const int *offsets,const float *weights,const float *limits,const float *l1,
    const float *disc,const int *counts,const int *features,int stride,int max_features,
    float discrepancy2,float scale,float floor,float *tails) {
  int lane=threadIdx.x%32;
  int task=blockIdx.x*8+threadIdx.x/32,k=blockIdx.y;
  int i=task/max_features,t=task%max_features;
  if(i>=candidates || t>=features[k] || !eligible[k*candidates+i])return;
  int center=centers[i],r=center/cols,c=center%cols,n=counts[k];
  const float *w=weights+(k*max_features+t)*stride;
  float squared=0.f,plane_squared=0.f,shift=0.f;
  for(int p=lane;p<components;p+=32) {
    float value=0.f;
    for(int j=0;j<n;j++) {
      int q=(r+offsets[(k*stride+j)*2])*cols+c+offsets[(k*stride+j)*2+1];
      value+=w[j]*factor[q*components+p];
    }
    if(p>=components-bounded_components)shift+=fabsf(value);
    else if(plane_components && p>=components-bounded_components-plane_components)plane_squared+=value*value;else squared+=value*value;
  }
  float mean=0.f;
  for(int j=lane;j<n;j+=32) {
    int q=(r+offsets[(k*stride+j)*2])*cols+c+offsets[(k*stride+j)*2+1];
    mean+=w[j]*(height[q]-height[center]);
      squared+=w[j]*w[j]*local[q];
  }
  squared=sum_warp(squared);plane_squared=sum_warp(plane_squared);mean=sum_warp(mean);shift=sum_warp(shift);
  if(lane==0) {
    mean=fabsf(mean)+shift;
    int f=k*max_features+t;
    float sigma=source_std(squared,plane_squared,discrepancy2*disc[f],scale,plane_components)+floor*l1[f];
    sigma=fmaxf(sigma,1e-12f)*1.4142135623730951f;
    tails[(k*candidates+i)*max_features+t]=fabsf(mean)>=limits[f]?1.f:
      .5f*(erfcf((limits[f]+mean)/sigma)+erfcf((limits[f]-mean)/sigma));
  }
}
extern "C" __global__ void reduce_tail(const float *tails,const unsigned char *eligible,
    int candidates,int max_features,const int *features,const int *counts,
    float *upper,float *span,float *other) {
  int i=blockIdx.x*blockDim.x+threadIdx.x,k=blockIdx.y;
  if(i>=candidates)return;
  int index=k*candidates+i;
  if(!eligible[index]){upper[index]=CUDART_NAN_F;span[index]=0.f;other[index]=1.f;return;}
  float residual=0.f,remaining=0.f;
  for(int t=0;t<features[k];t++) {
    float value=tails[index*max_features+t];
    if(t>=2 && t<2+counts[k])residual+=value;else remaining+=value;
  }
  span[index]=residual;other[index]=remaining;
  upper[index]=fminf(1.f,residual+remaining+1e-7f);
}
extern "C" __global__ void span_project(const int *jobs,int job_count,
    const int *centers,int candidates,int cols,const float *height,const float *factor,
    int components,const int *offsets,const float *weights,const int *counts,
    int stride,int max_features,float *projected,float *means) {
  int lane=threadIdx.x%32,task=blockIdx.x*8+threadIdx.x/32;
  int job=task/stride,j=task%stride;
  if(job>=job_count)return;
  int index=jobs[job],k=index/candidates,i=index%candidates,n=counts[k];
  if(j>=n)return;
  int center=centers[i],r=center/cols,c=center%cols;
  const float *w=weights+(k*max_features+2+j)*stride;
  for(int p=lane;p<components;p+=32) {
    float value=0.f;
    for(int a=0;a<n;a++) {
      int q=(r+offsets[(k*stride+a)*2])*cols+c+offsets[(k*stride+a)*2+1];
      value+=w[a]*factor[q*components+p];
    }
    projected[(job*stride+j)*components+p]=value;
  }
  float mean=0.f;
  for(int a=lane;a<n;a+=32) {
    int q=(r+offsets[(k*stride+a)*2])*cols+c+offsets[(k*stride+a)*2+1];
    mean+=w[a]*(height[q]-height[center]);
  }
  mean=sum_warp(mean);
  if(lane==0)means[job*stride+j]=mean;
}
extern "C" __global__ void span_pairs(const int *jobs,int job_count,int candidates,
    const float *projected,const float *means,int components,int plane_components,int bounded_components,int stride,
    const int *pair_i,const int *pair_j,const int *pair_counts,int max_pairs,
    const float *pair_disc,const float *pair_l1,float discrepancy2,float scale,
    float floor,float limit,const int *centers,int cols,const int *offsets,
    const float *weights,const int *counts,int max_features,const float *local,float *sums) {
  int lane=threadIdx.x%32,task=blockIdx.x*8+threadIdx.x/32;
  int job=task/max_pairs,pair=task%max_pairs;
  if(job>=job_count)return;
  int k=jobs[job]/candidates;
  if(pair>=pair_counts[k])return;
  int index=k*max_pairs+pair,a=pair_i[index],b=pair_j[index];
  float squared=0.f,plane_squared=0.f,shift=0.f;
  for(int p=lane;p<components;p+=32) {
    float delta=projected[(job*stride+a)*components+p]-projected[(job*stride+b)*components+p];
    if(p>=components-bounded_components)shift+=fabsf(delta);
    else if(plane_components && p>=components-bounded_components-plane_components)plane_squared+=delta*delta;else squared+=delta*delta;
  }
  int center=centers[jobs[job]%candidates],r=center/cols,c=center%cols;
  for(int j=lane;j<counts[k];j+=32) {
    int q=(r+offsets[(k*stride+j)*2])*cols+c+offsets[(k*stride+j)*2+1];
    float delta=weights[(k*max_features+2+a)*stride+j]-weights[(k*max_features+2+b)*stride+j];
    squared+=delta*delta*local[q];
  }
  squared=sum_warp(squared);plane_squared=sum_warp(plane_squared);shift=sum_warp(shift);
  if(lane==0) {
    float mean=fabsf(means[job*stride+a]-means[job*stride+b])+shift;
    float sigma=source_std(squared,plane_squared,discrepancy2*pair_disc[index],scale,plane_components)+
      floor*pair_l1[index];
    sigma=fmaxf(sigma,1e-12f)*1.4142135623730951f;
    float tail=fabsf(mean)>=limit ? 1.f :
      .5f*(erfcf((limit+mean)/sigma)+erfcf((limit-mean)/sigma));
    atomicAdd(sums+job,tail);
  }
}
extern "C" __global__ void span_apply(const int *jobs,int job_count,const float *sums,
    const float *span,const float *other,float *upper) {
  int job=blockIdx.x*blockDim.x+threadIdx.x;
  if(job>=job_count)return;
  int index=jobs[job];
  float proposed=other[index]+fminf(span[index],sums[job]);
  if(isfinite(sums[job]))upper[index]=fminf(upper[index],fminf(1.f,proposed+1e-7f));
}
'''

# R = I - X X^+: project the three plane coefficients once per pose.
# Residual factors then cost O(n*P), instead of O(n*n*P). Keep the original
# CUDA kernels as a numerical qualification reference, not a different model.
SOURCE+=r'''
extern "C" __global__ void plane_factor(const float *factor,int components,
 const int *centers,int candidates,int cols,const unsigned char *eligible,
 const int *offsets,const float *fit,const int *counts,int stride,float *projected) {
 int lane=threadIdx.x%32,job=blockIdx.x*8+threadIdx.x/32,k=blockIdx.y;
 if(job>=candidates||!eligible[k*candidates+job])return;
 int center=centers[job],r=center/cols,c=center%cols,n=counts[k];
 const float *w=fit+k*3*stride;
 for(int p=lane;p<components;p+=32) {
  float a=0.f,b=0.f,d=0.f;
  for(int j=0;j<n;j++) {
   int q=(r+offsets[(k*stride+j)*2])*cols+c+offsets[(k*stride+j)*2+1];
   float v=factor[q*components+p];
   a+=w[j]*v;b+=w[stride+j]*v;d+=w[2*stride+j]*v;
  }
  float *out=projected+(k*candidates+job)*3*components;
  out[p]=a;out[components+p]=b;out[2*components+p]=d;
 }
}
__device__ __forceinline__ float residual_component(const float *factor,const float *plane,
 const float *xy,const int *offsets,int center,int cols,int k,int j,int p,int components,int stride) {
 int q=(center/cols+offsets[(k*stride+j)*2])*cols+center%cols+offsets[(k*stride+j)*2+1];
 return factor[q*components+p]-xy[(k*stride+j)*2]*plane[p]-
  xy[(k*stride+j)*2+1]*plane[components+p]-plane[2*components+p];
}
extern "C" __global__ void factored_contrasts(const float *height,const float *factor,const float *local,int components,int plane_components,int bounded_components,
 const int *centers,int candidates,int cols,const unsigned char *eligible,
 const int *offsets,const float *weights,const float *limits,const float *l1,
 const float *disc,const int *counts,const int *features,int stride,int max_features,
 float discrepancy2,float scale,float floor,float *fit_factor,const float *xy,const float *fit_weights,
 const int *sparse_counts,const int *sparse_ids,const float *sparse_values,float *tails) {
 int lane=threadIdx.x%32,i=blockIdx.x,k=blockIdx.y;
 if(i>=candidates||!eligible[k*candidates+i])return;
 int center=centers[i],r=center/cols,c=center%cols,n=counts[k];
 extern __shared__ float plane[];
 const float *fit=fit_weights+k*3*stride;
 for(int p=threadIdx.x;p<components;p+=blockDim.x) {
  float a=0.f,b=0.f,d=0.f;
  for(int j=0;j<n;j++) {
   int q=(r+offsets[(k*stride+j)*2])*cols+c+offsets[(k*stride+j)*2+1];
   float v=factor[q*components+p];
   a+=fit[j]*v;b+=fit[stride+j]*v;d+=fit[2*stride+j]*v;
  }
  plane[p]=a;plane[components+p]=b;plane[2*components+p]=d;
  float *out=fit_factor+(k*candidates+i)*3*components;
  out[p]=a;out[components+p]=b;out[2*components+p]=d;
 }
 __syncthreads();
 // A block owns one pose, so all its features reuse the same shared plane.
 for(int t=threadIdx.x/32;t<features[k];t+=blockDim.x/32) {
 const float *w=weights+(k*max_features+t)*stride;
 // These coefficients depend only on immutable body/yaw geometry. Cache
 // their ORIGINAL float32 nonzero order once, instead of every pose/lane.
 const int sparse_index=k*max_features+t;
 const int sparse_count=sparse_counts[sparse_index];
 float squared=0.f,plane_squared=0.f,shift=0.f;
 for(int p=lane;p<components;p+=32) {
  float v=0.f;
  if(t<2)v=plane[t*components+p];
  else if(t<2+n)v=residual_component(factor,plane,xy,offsets,center,cols,k,t-2,p,components,stride);
  else if(sparse_count<=3) {
   for(int j=0;j<sparse_count;j++) {
    int a=sparse_ids[sparse_index*3+j],q=(r+offsets[(k*stride+a)*2])*cols+c+offsets[(k*stride+a)*2+1];
    v+=sparse_values[sparse_index*3+j]*factor[q*components+p];
   }
  } else for(int j=0;j<n;j++)if(w[j]!=0.f) {
    int q=(r+offsets[(k*stride+j)*2])*cols+c+offsets[(k*stride+j)*2+1];
    v+=w[j]*factor[q*components+p];
   }
  if(p>=components-bounded_components)shift+=fabsf(v);
  else if(plane_components && p>=components-bounded_components-plane_components)plane_squared+=v*v;else squared+=v*v;
 }
 // Keep the original mean summation; only covariance work is factored.
 float mean=0.f;
 for(int j=lane;j<n;j+=32) {
  int q=(r+offsets[(k*stride+j)*2])*cols+c+offsets[(k*stride+j)*2+1];
  mean+=w[j]*(height[q]-height[center]);
  squared+=w[j]*w[j]*local[q];
 }
 squared=sum_warp(squared);plane_squared=sum_warp(plane_squared);mean=sum_warp(mean);shift=sum_warp(shift);
 if(lane==0) {
  mean=fabsf(mean)+shift;
    int f=k*max_features+t;
  float sigma=source_std(squared,plane_squared,discrepancy2*disc[f],scale,plane_components)+floor*l1[f];
  sigma=fmaxf(sigma,1e-12f)*1.4142135623730951f;
  tails[(k*candidates+i)*max_features+t]=fabsf(mean)>=limits[f]?1.f:
   .5f*(erfcf((limits[f]+mean)/sigma)+erfcf((limits[f]-mean)/sigma));
 }
 }
}
extern "C" __global__ void factored_span_project(const int *jobs,int job_count,
 const int *centers,int candidates,int cols,const float *height,const float *factor,
 int components,const int *offsets,const float *weights,const int *counts,
 int stride,int max_features,const float *fit_factor,const float *xy,float *projected,float *means) {
 int lane=threadIdx.x%32,task=blockIdx.x*8+threadIdx.x/32;
 int job=task/stride,j=task%stride;
 if(job>=job_count)return;
 int index=jobs[job],k=index/candidates,i=index%candidates,n=counts[k];
 if(j>=n)return;
 int center=centers[i],r=center/cols,c=center%cols;
 const float *plane=fit_factor+index*3*components;
 for(int p=lane;p<components;p+=32)
  projected[(job*stride+j)*components+p]=residual_component(factor,plane,xy,offsets,center,cols,k,j,p,components,stride);
 const float *w=weights+(k*max_features+2+j)*stride;
 float mean=0.f;
 for(int a=lane;a<n;a+=32) {
  int q=(r+offsets[(k*stride+a)*2])*cols+c+offsets[(k*stride+a)*2+1];
  mean+=w[a]*(height[q]-height[center]);
 }
 mean=sum_warp(mean);if(lane==0)means[job*stride+j]=mean;
}

// One block owns one pose. Reuse its projected covariance rows in shared
// memory and reduce the pair tails inside the block. The old kernel launched
// many blocks for each pose and contended on hundreds of global atomics.
extern "C" __global__ void cached_span_pairs(const int *jobs,int job_count,int candidates,
    const float *projected,const float *means,int components,int plane_components,int bounded_components,int stride,
    const int *pair_i,const int *pair_j,const int *pair_counts,int max_pairs,
    const float *pair_disc,const float *pair_l1,float discrepancy2,float scale,
    float floor,float limit,const int *centers,int cols,const int *offsets,
    const float *weights,const int *counts,int max_features,const float *local,float *sums,
    const float *span,const float *height,const float *factor,const float *fit_factor,
    const float *xy,int direct) {
  int job=blockIdx.x,lane=threadIdx.x%32,warp=threadIdx.x/32;
  if(job>=job_count)return;
  int k=jobs[job]/candidates,n=counts[k];
  extern __shared__ float memory[];
  float *cache=memory,*pose_means=cache+stride*components,*partial=pose_means+stride;
  int center=centers[jobs[job]%candidates],r=center/cols,c=center%cols;
  const float *plane=fit_factor+jobs[job]*3*components;
  // Retain the same residual factors and the same mean summation, but
  // write directly to this block's shared cache. No global projection
  // tensor or extra projection launch is required on the direct path.
  for(int p=threadIdx.x;p<n*components;p+=blockDim.x) {
    int j=p/components,component=p%components;
    cache[p]=direct?residual_component(factor,plane,xy,offsets,center,cols,k,j,component,components,stride):
      projected[job*stride*components+p];
  }
  for(int j=warp;j<n;j+=blockDim.x/32) {
    float mean=0.f;
    if(direct) {
      const float *w=weights+(k*max_features+2+j)*stride;
      for(int a=lane;a<n;a+=32) {
        int q=(r+offsets[(k*stride+a)*2])*cols+c+offsets[(k*stride+a)*2+1];
        mean+=w[a]*(height[q]-height[center]);
      }
      mean=sum_warp(mean);
    } else if(lane==0)mean=means[job*stride+j];
    if(lane==0)pose_means[j]=mean;
  }
  __syncthreads();
  float total=0.f;
  const int warps=blockDim.x/32;
  // The all-pairs sum only tightens the existing residual-span bound.
  // Once a nonnegative partial sum reaches that bound, omitted pairs
  // cannot improve it. Check in tiles to amortize the block reduction.
  for(int tile=0;tile<pair_counts[k];tile+=warps*8) {
   for(int pair=tile+warp;pair<min(tile+warps*8,pair_counts[k]);pair+=warps) {
    int index=k*max_pairs+pair,a=pair_i[index],b=pair_j[index];
    float squared=0.f,plane_squared=0.f,shift=0.f;
    for(int p=lane;p<components;p+=32) {
      float delta=cache[a*components+p]-cache[b*components+p];
      if(p>=components-bounded_components)shift+=fabsf(delta);
    else if(plane_components && p>=components-bounded_components-plane_components)plane_squared+=delta*delta;else squared+=delta*delta;
    }
    for(int j=lane;j<n;j+=32) {
      int q=(r+offsets[(k*stride+j)*2])*cols+c+offsets[(k*stride+j)*2+1];
      float delta=weights[(k*max_features+2+a)*stride+j]-weights[(k*max_features+2+b)*stride+j];
      squared+=delta*delta*local[q];
    }
    squared=sum_warp(squared);plane_squared=sum_warp(plane_squared);shift=sum_warp(shift);
    if(lane==0) {
      float mean=fabsf(pose_means[a]-pose_means[b])+shift;
      float sigma=source_std(squared,plane_squared,discrepancy2*pair_disc[index],scale,plane_components)+floor*pair_l1[index];
      sigma=fmaxf(sigma,1e-12f)*1.4142135623730951f;
      total+=fabsf(mean)>=limit?1.f:.5f*(erfcf((limit+mean)/sigma)+erfcf((limit-mean)/sigma));
    }
   }
   if(lane==0)partial[warp]=total;
   __syncthreads();
   if(threadIdx.x==0) {
     float sum=0.f;for(int w=0;w<warps;w++)sum+=partial[w];sums[job]=sum;
   }
   __syncthreads();
   if(sums[job]>=span[jobs[job]])break;
  }
  if(lane==0)partial[warp]=total;
  __syncthreads();
  if(threadIdx.x==0) {
    float sum=0.f;for(int w=0;w<blockDim.x/32;w++)sum+=partial[w];sums[job]=sum;
  }
}
'''


@lru_cache(maxsize=32)
def geometry(params,lengthscale,device):
    import cupy as cp
    from .probabilistic_risk import _geometry
    from .shared_terrain import _contrast_discrepancy
    parts=[_geometry(params,k) for k in range(params.yaw_bins)]
    n=max(len(a) for a,_,_ in parts);f=max(len(b) for _,b,_ in parts)
    offsets=np.zeros((params.yaw_bins,n,2),np.int32)
    body_xy=np.zeros((params.yaw_bins,n,2),np.float32)
    fit_weights=np.zeros((params.yaw_bins,3,n),np.float32)
    weights=np.zeros((params.yaw_bins,f,n),np.float32)
    limits=np.ones((params.yaw_bins,f),np.float32);disc=np.zeros_like(limits)
    pairs=max(len(a)*(len(a)-1)//2 for a,_,_ in parts)
    pair_i=np.zeros((params.yaw_bins,pairs),np.int32);pair_j=np.zeros_like(pair_i)
    pair_disc=np.zeros((params.yaw_bins,pairs),np.float32);pair_l1=np.zeros_like(pair_disc)
    pair_counts=[]
    counts=[];features=[]
    for k,(off,w,lim) in enumerate(parts):
        theta=(k+.5)*2*np.pi/params.yaw_bins;c,s=np.cos(theta),np.sin(theta)
        xy=np.array([(params.resolution*(c*dc+s*dr),params.resolution*(-s*dc+c*dr)) for dr,dc in off])
        body_xy[k,:len(off)]=xy
        fit_weights[k,:2,:len(off)]=w[:2]
        fit_weights[k,2,:len(off)]=np.linalg.pinv(np.c_[xy,np.ones(len(xy))])[2]
        offsets[k,:len(off)]=off;weights[k,:len(w),:len(off)]=w;limits[k,:len(lim)]=lim
        individual,pair=_contrast_discrepancy(params,k,lengthscale)
        disc[k,:len(lim)]=individual
        i,j=np.triu_indices(len(off),1);size=len(i)
        pair_i[k,:size]=i;pair_j[k,:size]=j;pair_disc[k,:size]=pair
        pair_l1[k,:size]=np.abs(w[2+i]-w[2+j]).sum(axis=1)
        pair_counts.append(size)
        counts.append(len(off));features.append(len(w))
    # Derive from the SAME rounded float32 weights passed to the kernel.
    # Computing sparsity on float64 geometry could change a near-zero row.
    sparse_counts=np.count_nonzero(weights,axis=2).astype(np.int32)
    sparse_ids=np.zeros((params.yaw_bins,f,3),np.int32)
    sparse_values=np.zeros((params.yaw_bins,f,3),np.float32)
    for k in range(params.yaw_bins):
        for t in range(features[k]):
            ids=np.flatnonzero(weights[k,t])
            if len(ids)<=3:
                sparse_ids[k,t,:len(ids)]=ids
                sparse_values[k,t,:len(ids)]=weights[k,t,ids]
    module=cp.RawModule(code=SOURCE,options=('--std=c++11',))
    return n,f,pairs,tuple(cp.asarray(a) for a in (offsets,weights,limits,np.abs(weights).sum(axis=2),disc,
        np.asarray(counts,np.int32),np.asarray(features,np.int32),pair_i,pair_j,
        np.asarray(pair_counts,np.int32),pair_disc,pair_l1,fit_weights,body_xy,sparse_counts,sparse_ids,sparse_values)),tuple(module.get_function(name)
        for name in ('supported','contrasts','reduce_tail','span_project','span_pairs','span_apply',
                     'plane_factor','factored_contrasts','factored_span_project','cached_span_pairs'))


@lru_cache(maxsize=64)
def opposite_heading_count(params):
    """Compute once per identical pi-periodic terrain event, not per heading.

    A front/back symmetric footprint has the same support after a half-turn.
    Its fitted pitch and roll change sign, while residual span and vertical
    jumps do not. All these constraints are absolute, so their joint event is
    identical, including covariance and bounded mean uncertainty. Check the
    discrete supports as well: a numerical stencil boundary must fall back to
    all headings. This does not apply to asymmetric vehicle footprints.
    """
    from .probabilistic_risk import _geometry
    half=params.yaw_bins//2
    if params.yaw_bins%2 or params.front!=params.back:
        return params.yaw_bins
    for k in range(half):
        a,_,limits_a=_geometry(params,k)
        b,_,limits_b=_geometry(params,k+half)
        if a!=b or not np.array_equal(limits_a,limits_b):
            return params.yaw_bins
    return half


def shared_probability_cuda(prediction,params,collision,*,tighten_span=True,factorized=True,query_mask=None,direct_span=True,
                            heading_symmetry=True):
    from .shared_terrain import integrate_independent_model_floor
    prediction,params=integrate_independent_model_floor(prediction,params)
    import cupy as cp
    device=prediction.factor.device.index
    with cp.cuda.Device(device):
        rows,cols=prediction.mean.shape
        # Only requested centres can become eligible. Compact the host-side
        # work list before allocating per-centre covariance scratch/launching
        # kernels; all footprint neighbours still use the FULL valid field.
        # The supported kernel retains its per-heading mask and collision
        # checks. Unrequested poses return no tightening, never zero risk.
        domain=prediction.valid
        requested=None
        if query_mask is not None:
            query_mask=np.asarray(query_mask,dtype=bool)
            if query_mask.shape!=(params.yaw_bins,rows,cols):
                raise ValueError('risk refinement mask must match heading and map geometry')
            requested=query_mask
            domain=domain&query_mask.any(axis=0)
        centers=np.flatnonzero(domain).astype(np.int32);count=len(centers)
        if not count:return [(np.empty(0,int),np.empty(0,int),np.empty(0,np.float32)) for _ in range(params.yaw_bins)]
        n,f,pairs,arrays,kernels=geometry(params,prediction.discrepancy_lengthscale_m,device)
        offsets,weights,limits,l1,disc,counts,features,pair_i,pair_j,pair_counts,pair_disc,pair_l1,fit_weights,body_xy,sparse_counts,sparse_ids,sparse_values=arrays
        positions=cp.asarray(centers);height=cp.asarray(prediction.mean)
        factor=cp.from_dlpack(prediction.factor.contiguous())
        bounded=0
        if prediction.bounded_mean_factor is not None:
            nuisance=cp.from_dlpack(prediction.bounded_mean_factor.contiguous())
            bounded=nuisance.shape[-1]
            factor=cp.concatenate((factor,nuisance),axis=-1)
        local=cp.zeros((rows,cols),cp.float32) if prediction.local_variance is None else cp.asarray(prediction.local_variance)
        valid=cp.asarray(prediction.valid,dtype=cp.uint8);occupied=cp.asarray(collision,dtype=cp.uint8)
        headings=opposite_heading_count(params) if heading_symmetry else params.yaw_bins
        if requested is not None and headings<params.yaw_bins:
            query_mask=requested[:headings]|requested[headings:]
        eligible=cp.empty((headings,count),cp.uint8)
        query_mask=(cp.ones((headings,rows,cols),cp.uint8) if requested is None
                    else cp.asarray(query_mask,dtype=cp.uint8))
        tails=cp.empty((headings,count,f),cp.float32);upper=cp.empty_like(eligible,dtype=cp.float32)
        span=cp.empty_like(upper);other=cp.empty_like(upper)
        kernels[0](((count+127)//128,headings),(128,),
            (positions,np.int32(count),np.int32(rows),np.int32(cols),valid,occupied,offsets,counts,np.int32(n),query_mask,eligible))
        extra=()
        if factorized:
            plane=cp.empty((headings,count,3,factor.shape[2]),cp.float32)
            extra=(plane,body_xy)
        if not 0<=prediction.plane_components<factor.shape[2]:raise ValueError('invalid source covariance partition')
        contrast_args=(height,factor,local,np.int32(factor.shape[2]),np.int32(prediction.plane_components),np.int32(bounded),positions,np.int32(count),np.int32(cols),eligible,
             offsets,weights,limits,l1,disc,counts,features,np.int32(n),np.int32(f),
             np.float32(prediction.discrepancy**2),np.float32(params.variance_scale),
             np.float32(params.height_std_floor))
        if factorized:
            kernels[7]((count,headings),(128,),(*contrast_args,*extra,fit_weights,sparse_counts,sparse_ids,sparse_values,tails),
                       shared_mem=3*factor.shape[2]*4)
        else:kernels[1](((count*f+7)//8,headings),(256,),(*contrast_args,tails))
        kernels[2](((count+127)//128,headings),(128,),
            (tails,eligible,np.int32(count),np.int32(f),features,counts,upper,span,other))
        if tighten_span:
            # Keep extra projection storage local to nontrivial certificates;
            # flat ground needs neither all-pairs work nor a full-map tensor.
            threshold=1e-7 if prediction.plane_components else 1e-3
            jobs=cp.flatnonzero((eligible!=0)&(span>threshold)&(other<1)).astype(cp.int32)
            size=len(jobs)
            if size:
                components=factor.shape[2]
                batch=min(size,2048)
                cache_bytes=(n*components+n+8)*4
                cached=factorized and cache_bytes<=48*1024
                direct=cached and direct_span
                projected=cp.empty(1 if direct else (batch,n,components),cp.float32)
                means=cp.empty(1 if direct else (batch,n),cp.float32)
                sums=cp.empty(batch,cp.float32)
                for begin in range(0,size,batch):
                    selected=jobs[begin:begin+batch];number=len(selected);sums.fill(0)
                    if not direct:
                        kernels[8 if factorized else 3](((number*n+7)//8,),(256,),
                            (selected,np.int32(number),positions,np.int32(count),np.int32(cols),height,factor,
                             np.int32(components),offsets,weights,counts,np.int32(n),np.int32(f),*extra,projected,means))
                    pair_args=(selected,np.int32(number),np.int32(count),projected,means,np.int32(components),np.int32(prediction.plane_components),np.int32(bounded),np.int32(n),
                         pair_i,pair_j,pair_counts,np.int32(pairs),pair_disc,pair_l1,
                         np.float32(prediction.discrepancy**2),np.float32(params.variance_scale),
                         np.float32(params.height_std_floor),np.float32(params.roughness_limit),positions,np.int32(cols),
                         offsets,weights,counts,np.int32(f),local,sums)
                    if cached:
                        kernels[9]((number,),(256,),(*pair_args,span,height,factor,plane,body_xy,np.int32(direct)),shared_mem=cache_bytes)
                    else:kernels[4](((number*pairs+7)//8,),(256,),pair_args)
                    kernels[5](((number+127)//128,),(128,),
                        (selected,np.int32(number),sums,span,other,upper))
        values=cp.asnumpy(upper)
    result=[]
    for k in range(params.yaw_bins):
        row=values[k%headings]
        keep=np.isfinite(row)
        # The proof may have been requested by the opposite heading only.
        # Preserve each ORIGINAL query mask; never return an unrequested pose.
        if requested is not None:keep&=requested[k].ravel()[centers]
        result.append((centers[keep]//cols,centers[keep]%cols,row[keep]))
    return result
