"""Fused CUDA execution of the CPU probability reference, not a new risk model."""
from functools import lru_cache
import numpy as np


SOURCE=r'''
#include <math_constants.h>
#define INFINITY CUDART_INF_F
#define NAN CUDART_NAN_F
__device__ float exceed(float m,float s,float l) {
  if (fabsf(m)>=l) return 1.f;
  float den=fmaxf(s,1.e-12f)*1.4142135623730951f;
  return .5f*(erfcf((l+m)/den)+erfcf((l-m)/den));
}
extern "C" __global__ void edges(const float *h,const float *s,const unsigned char *v,
    int rows,int cols,float limit,float *upper,float *margin,float *radius,unsigned char *supported) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*cols)return;
  int r=i/cols,c=i%cols;float total=0.f,minimum=INFINITY,absolute=INFINITY;bool ok=true;
  for(int axis=0;axis<2;axis++) {
    float a[4],b[4];bool good=true;
    for(int j=0;j<4;j++) {
      int y=r+(axis?j-1:0),x=c+(axis?0:j-1);
      if(y<0||y>=rows||x<0||x>=cols||!v[y*cols+x]){good=false;a[j]=b[j]=0.f;}
      else {a[j]=h[y*cols+x];b[j]=s[y*cols+x];}
    }
    ok &= good;
    float mean=(a[2]-a[1])-.5f*((a[1]-a[0])+(a[3]-a[2]));
    float sigma=.5f*b[0]+1.5f*b[1]+1.5f*b[2]+.5f*b[3];
    total+=exceed(mean,sigma,limit);
    minimum=fminf(minimum,(limit-fabsf(mean))/fmaxf(sigma,1.e-12f));
    absolute=fminf(absolute,(limit-fabsf(mean))*.25f);
  }
  upper[i]=total;margin[i]=minimum;radius[i]=absolute;supported[i]=ok;
}
extern "C" __global__ void poses(const float *h,const float *s,const unsigned char *v,
    const unsigned char *occupied,int rows,int cols,const int *offsets,const float *weights,
    const float *limits,const int *counts,const int *features,const float *span_norm,
    const int *support_count,int stride,int internal,float max_step,const float *edge_upper,
    const float *edge_margin,const float *edge_radius,const unsigned char *edge_valid,
    const float *group_weights,const int *group_ends,const int *group_counts,
    const int *pair_indices,const float *pair_corrections,
    const float *rho,int bounded,float *prob,float *tail,float *valid,float *radius) {
  int i=blockIdx.x*blockDim.x+threadIdx.x,k=blockIdx.y;if(i>=rows*cols)return;
  int r=i/cols,c=i%cols,n=counts[k],f=features[k],out=k*rows*cols+i;
  bool ok=true,hit=false;float a[96],b[96],bias[96],max_sigma=0.f,edge_sum=0.f,margin=INFINITY,absolute=INFINITY;
  const int *off=offsets+k*stride*2;
  float center=h[i];
  for(int j=0;j<n;j++) {
    int y=r+off[2*j],x=c+off[2*j+1];
    if(y<0||y>=rows||x<0||x>=cols){ok=false;a[j]=b[j]=0.f;continue;}
    int q=y*cols+x;hit |= occupied[q];ok &= v[q];
    a[j]=v[q]?h[q]-center:0.f;b[j]=v[q]?s[q]:0.f;bias[j]=bounded?rho[q]:0.f;
    max_sigma=fmaxf(max_sigma,b[j]);
    if(!internal){ok &= edge_valid[q];edge_sum+=edge_upper[q];margin=fminf(margin,edge_margin[q]);absolute=fminf(absolute,edge_radius[q]);}
  }
  if(hit){prob[out]=valid[out]=1.f;tail[out]=radius[out]=NAN;return;}
  if(!ok){prob[out]=tail[out]=radius[out]=NAN;valid[out]=0.f;return;}
  const float *w=weights+k*MAX_FEATURES*stride,*lim=limits+k*MAX_FEATURES;
  float total=edge_sum,other=edge_sum,low=INFINITY,high=-INFINITY,residual[96];
  for(int t=0;t<f;t++) {
    float mean=0.f,sigma=0.f,norm=0.f,shift=0.f;
    for(int j=0;j<n;j++){float z=w[t*stride+j];mean+=z*a[j];sigma+=fabsf(z)*b[j];norm+=fabsf(z);if(bounded)shift+=fabsf(z)*bias[j];}
    const float probability=exceed(bounded?fabsf(mean)+shift:mean,sigma,lim[t]);total+=probability;
    if(t>=2&&t<2+n){residual[t-2]=mean;low=fminf(low,mean);high=fmaxf(high,mean);}
    else {other+=probability;margin=fminf(margin,(lim[t]-fabsf(mean))/fmaxf(sigma,1.e-12f));
      absolute=fminf(absolute,(lim[t]-fabsf(mean))/fmaxf(norm,1.e-12f));}
  }
  if(bounded){prob[out]=fminf(1.f,total+1e-7f);valid[out]=1.f;tail[out]=radius[out]=1.f;return;}
  margin=fminf(margin,(max_step-(high-low))/fmaxf(span_norm[k]*max_sigma,1.e-12f));
  absolute=fminf(absolute,(max_step-(high-low))/span_norm[k]);
  float u=erfcf(fmaxf(margin,0.f)*.7071067811865475f);
  float bound=fminf(1.f,fminf(total,support_count[k]*u));
  if(bound>.005f && high-low<max_step && other<bound) {
    const float *base=group_weights+k*MAX_SPAN_GROUPS*stride;
    const int *ends=group_ends+k*MAX_SPAN_GROUPS,*ids=pair_indices+k*MAX_SPAN_PAIRS*2;
    const float *correction=pair_corrections+k*MAX_SPAN_PAIRS*2;
    float alternative=other;int t=0;
    for(int g=0;g<group_counts[k] && alternative<bound;g++) {
      float shared=0.f;
      for(int j=0;j<n;j++)shared+=base[g*stride+j]*b[j];
      for(;t<ends[g] && alternative<bound;t++) {
        const int i0=ids[2*t],i1=ids[2*t+1];
        // Exact L1 contrast norm, regrouped by equal spatial displacement.
        // Guard accumulated float roundoff conservatively.
        const float sigma=fmaxf(0.f,shared+correction[2*t]*b[i0]+correction[2*t+1]*b[i1])*(1.f+2.e-6f)+1.e-10f;
        alternative+=exceed(residual[i0]-residual[i1],sigma,max_step);
      }
    }
    bound=fminf(bound,fmaxf(.005f,alternative));
  }
  tail[out]=u;prob[out]=bound;valid[out]=1.f;radius[out]=fmaxf(absolute,0.f);
}
extern "C" __global__ void conditioned(const float *h,const float *sigma,const unsigned char *valid,
    const unsigned char *occupied,const unsigned char *prior,const float *dx,const float *dy,
    const float *coefficient,const float *source_sigma,int sources,int rows,int cols,
    const int *offsets,const float *weights,const float *limits,const int *counts,
    const int *features,int stride,float model_floor,float *prob,float *mode) {
  int i=blockIdx.x*blockDim.x+threadIdx.x,k=blockIdx.y;if(i>=rows*cols)return;
  int r=i/cols,c=i%cols,n=counts[k],f=features[k],out=k*rows*cols+i;
  const int *off=offsets+k*stride*2;
  bool affected=false;
  for(int j=0;j<n;j++) {
    int y=r+off[2*j],x=c+off[2*j+1];
    if(y<0||y>=rows||x<0||x>=cols)return;
    int q=y*cols+x;if(!valid[q]||occupied[q])return;
    affected |= prior[q];
  }
  if(!affected||!isfinite(prob[out]))return;
  float a[96],s[96],px[96],py[96],p0[96];
  for(int j=0;j<n;j++) {
    int q=(r+off[2*j])*cols+c+off[2*j+1];
    a[j]=h[q]-h[i];s[j]=prior[q]?model_floor:sigma[q];
    p0[j]=prior[q]?1.f:0.f;px[j]=p0[j]*dx[q];py[j]=p0[j]*dy[q];
  }
  const float *w=weights+k*MAX_FEATURES*stride,*lim=limits+k*MAX_FEATURES;
  const int pair_end=2+n*(n-1)/2,residual_begin=f-n;
  float low=INFINITY,high=-INFINITY;
  for(int t=residual_begin;t<f;t++) {
    float mean=0.f;for(int j=0;j<n;j++)mean+=w[t*stride+j]*a[j];
    low=fminf(low,mean);high=fmaxf(high,mean);
  }
  const float center=.5f*(low+high);
  float common=0.f,pairs=0.f,residuals=0.f;
  // Evaluate the O(n) residual-interval proof before the O(n^2) pair
  // proof. All tails are nonnegative: once the pair sum reaches the
  // residual sum, no remaining pair can change their minimum. This is an
  // exact short circuit of the SAME two sufficient proofs, not truncation
  // of a union bound or removal of any common constraint.
  for(int pass=0;pass<4;pass++) {
   int begin=pass==0?0:pass==1?pair_end:pass==2?residual_begin:2;
   int end=pass==0?2:pass==1?residual_begin:pass==2?f:pair_end;
   for(int t=begin;t<end;t++) {
    if(pass==3 && pairs>=residuals)break;
    float mean=0.f,local=0.f,bx=0.f,by=0.f,b0=0.f;
    for(int j=0;j<n;j++) {
      float z=w[t*stride+j];mean+=z*a[j];local+=fabsf(z)*s[j];
      bx+=z*px[j];by+=z*py[j];b0+=z*p0[j];
    }
    float shared=0.f;
    for(int j=0;j<sources;j++) {
      float effective=bx*coefficient[j]+by*coefficient[sources+j]+b0*coefficient[2*sources+j];
      shared+=fabsf(effective)*source_sigma[j];
    }
    const float tail=exceed(mean-(t>=residual_begin?center:0.f),shared+local,lim[t]);
    if(t>=residual_begin) residuals+=tail;
    else if(t>=2 && t<pair_end) pairs+=tail;
    else common+=tail;
   }
  }
  prob[out]=fminf(prob[out],fminf(1.f,common+fminf(pairs,residuals)));mode[out]=1.f;
}
'''


@lru_cache(maxsize=256)
def _span_groups(p,k):
    from .probabilistic_risk import _geometry as geometry
    offsets,weights,_=geometry(p,k)
    yaw=(k+.5)*2*np.pi/p.yaw_bins;c,s=np.cos(yaw),np.sin(yaw)
    groups={}
    for i,(r0,c0) in enumerate(offsets):
        for j in range(i+1,len(offsets)):
            r1,c1=offsets[j]
            groups.setdefault((r0-r1,c0-c1),[]).append((i,j))
    bases=[];ends=[];indices=[];corrections=[]
    for (dr,dc),pairs in groups.items():
        dx=p.resolution*(c*dc+s*dr);dy=p.resolution*(-s*dc+c*dr)
        projection=dx*weights[0].astype(np.float64)+dy*weights[1].astype(np.float64)
        bases.append(np.abs(projection))
        for i,j in pairs:
            indices.append((i,j))
            corrections.append((abs(1-projection[i])-abs(projection[i]),
                                abs(-1-projection[j])-abs(projection[j])))
        ends.append(len(indices))
    return np.asarray(bases,np.float32),np.asarray(ends,np.int32),np.asarray(indices,np.int32),np.asarray(corrections,np.float32)


@lru_cache(maxsize=16)
def _geometry(p,device,conditional=False):
    import cupy as cp
    from .probabilistic_risk import _geometry as geometry,_envelope_geometry
    if conditional:
        from .conditioned_probability import conditional_evaluation_geometry as geometry
    parts=[geometry(p,k) for k in range(p.yaw_bins)]
    n=max(len(q[0]) for q in parts);f=max(len(q[1]) for q in parts)
    if n>96:return None
    off=np.zeros((p.yaw_bins,n,2),np.int32)
    weights=np.zeros((p.yaw_bins,f,n),np.float32);limits=np.zeros((p.yaw_bins,f),np.float32)
    counts=[];features=[];spans=[];support=[]
    for k,(o,w,l) in enumerate(parts):
        off[k,:len(o)]=o;weights[k,:len(w),:len(o)]=w;limits[k,:len(l)]=l
        counts.append(len(o));features.append(len(w))
        span,count,_=_envelope_geometry(p,k);spans.append(span);support.append(count)
    pair_arrays=[];pair_count=1;group_count=1
    if not conditional:
        groups=[_span_groups(p,k) for k in range(p.yaw_bins)]
        group_count=max(len(q[0]) for q in groups);pair_count=max(len(q[2]) for q in groups)
        bases=np.zeros((p.yaw_bins,group_count,n),np.float32)
        ends=np.zeros((p.yaw_bins,group_count),np.int32)
        ids=np.zeros((p.yaw_bins,pair_count,2),np.int32)
        corrections=np.zeros((p.yaw_bins,pair_count,2),np.float32)
        for k,(base,end,index,correction) in enumerate(groups):
            bases[k,:len(base),:base.shape[1]]=base;ends[k,:len(end)]=end
            ids[k,:len(index)]=index;corrections[k,:len(correction)]=correction
        pair_arrays=[cp.asarray(q) for q in (bases,ends,np.array([len(q[0]) for q in groups],np.int32),ids,corrections)]
    module=cp.RawModule(code=f"#define MAX_FEATURES {f}\n#define MAX_SPAN_PAIRS {pair_count}\n#define MAX_SPAN_GROUPS {group_count}\n"+SOURCE,options=("--std=c++11",),
        name_expressions=("edges","poses","conditioned"))
    arrays=[cp.asarray(q) for q in (off,weights,limits,np.array(counts,np.int32),
        np.array(features,np.int32),np.array(spans,np.float32),np.array(support,np.int32))]
    return module,arrays+pair_arrays,n


def prepare_conditioned_geometry(p):
    """Compile immutable geometry before a first goal can activate the stance.

    This allocates constants only: no synthetic observations, pose permission
    or navigation result. A later real footprint reuses the same kernel.
    """
    import cupy as cp
    geometry=_geometry(p,cp.cuda.runtime.getDevice(),True)
    if geometry is not None:geometry[0].get_function('conditioned')


def fused_probability(h,sigma,valid,occupied,p,*,bounded_radius=None):
    import cupy as cp
    geometry=_geometry(p,cp.cuda.runtime.getDevice())
    if geometry is None:return None
    module,arrays,stride=geometry
    h,sigma=cp.ascontiguousarray(h),cp.ascontiguousarray(sigma)
    valid,occupied=cp.ascontiguousarray(valid,dtype=cp.uint8),cp.ascontiguousarray(occupied,dtype=cp.uint8)
    rows,cols=h.shape
    upper=cp.empty_like(h);margin=cp.empty_like(h);radius=cp.empty_like(h);supported=cp.empty_like(valid)
    if not p.internal_steps:
        module.get_function("edges")(((h.size+127)//128,),(128,),
            (h,sigma,valid,np.int32(rows),np.int32(cols),np.float32(p.max_step),upper,margin,radius,supported))
    outputs=[cp.empty((p.yaw_bins,rows,cols),cp.float32) for _ in range(4)]
    module.get_function("poses")(((h.size+127)//128,p.yaw_bins),(128,),
        (h,sigma,valid,occupied,np.int32(rows),np.int32(cols),*arrays[:7],np.int32(stride),
         np.int32(p.internal_steps),np.float32(p.roughness_limit),upper,margin,radius,supported,*arrays[7:],
         cp.zeros(1,cp.float32) if bounded_radius is None else cp.ascontiguousarray(bounded_radius),
         np.int32(bounded_radius is not None),*outputs))
    return {name:list(cp.asnumpy(value)) for name,value in zip(
        ("failure_probability_upper","height_error_tail_upper","probability_valid","height_error_radius_m"),outputs)}


def fused_conditioned(layers,h,sigma,valid,occupied,prior,dx,dy,coefficient,source_sigma,p):
    import cupy as cp
    geometry=_geometry(p,cp.cuda.runtime.getDevice(),True)
    if geometry is None:return None
    module,arrays,stride=geometry
    rows,cols=h.shape
    floats=[cp.ascontiguousarray(a,dtype=cp.float32) for a in
            (h,sigma,dx,dy,coefficient,source_sigma)]
    h,sigma,dx,dy,coefficient,source_sigma=floats
    valid,occupied,prior=[cp.ascontiguousarray(a,dtype=cp.uint8) for a in (valid,occupied,prior)]
    prob=cp.asarray(np.stack(layers["failure_probability_upper"]),dtype=cp.float32)
    mode=cp.zeros_like(prob)
    module.get_function("conditioned")(((h.size+127)//128,p.yaw_bins),(128,),
        (h,sigma,valid,occupied,prior,dx,dy,coefficient,source_sigma,np.int32(source_sigma.size),
         np.int32(rows),np.int32(cols),*arrays[:5],np.int32(stride),
         np.float32(p.height_std_floor),prob,mode))
    layers["failure_probability_upper"]=list(cp.asnumpy(prob))
    layers["probability_pose_event"]=list(cp.asnumpy(mode))
    return layers
