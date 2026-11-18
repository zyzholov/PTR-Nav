"""Cached exact float32 SAT stencils for measured subcell collision overlays.

The stencil uses the same device predicates as the original per-pose kernel.
It changes no obstacle bits, body envelope, height posterior or risk budget.
"""
from functools import lru_cache
import numpy as np

_GEOMETRY = r'''
extern "C" __global__ void stencil(unsigned char *out,int bins,int reach,int factor,
    float fine,float front,float back,float width) {
  int i=blockDim.x*blockIdx.x+threadIdx.x;
  int side=2*reach+1,area=side*side;if(i>=bins*area)return;
  int k=i/area,dx=i%side-reach,dy=(i/side)%side-reach;
  float yaw=(k+.5f)*6.283185307179586f/bins,c=cosf(yaw),s=sinf(yaw);
  float a=(front+back)*.5f,off=(front-back)*.5f,h=fine*.5f;
  float hx=fabsf(c)*a+fabsf(s)*width,hy=fabsf(s)*a+fabsf(c)*width;
  float projected=h*(fabsf(c)+fabsf(s));
  float rx=(dx+.5f-.5f*factor)*fine-off*c,ry=(dy+.5f-.5f*factor)*fine-off*s;
  out[i]=fabsf(rx)<=hx+h+1e-6f && fabsf(ry)<=hy+h+1e-6f &&
    fabsf(c*rx+s*ry)<=a+projected+1e-6f && fabsf(-s*rx+c*ry)<=width+projected+1e-6f;
}'''
_APPLY = r'''
extern "C" __global__ void apply(const unsigned int *bits,unsigned char *out,
    const int *dx,const int *dy,const unsigned int *mask,const int *counts,
    int rows,int cols,int bins,int stride) {
  int i=blockDim.x*blockIdx.x+threadIdx.x;if(i>=rows*cols*bins)return;
  int x=i%cols,y=(i/cols)%rows,k=i/(rows*cols);out[i]=0;
  for(int n=0;n<counts[k];++n) {
    int j=k*stride+n,c=x+dx[j],r=y+dy[j];
    if(c>=0&&r>=0&&c<cols&&r<rows&&(bits[r*cols+c]&mask[j])){out[i]=1;return;}
  }
}'''


@lru_cache(maxsize=4)
def _kernels(device):
    import cupy as cp
    with cp.cuda.Device(device):
        return cp.RawKernel(_GEOMETRY,'stencil'),cp.RawKernel(_APPLY,'apply')


@lru_cache(maxsize=4)
def _stencil(device,resolution,front,back,width,bins,factor):
    import cupy as cp
    fine=resolution/factor
    reach=int(np.ceil(np.hypot(max(front,back),width)/fine))+2
    # Bound cache construction for unusual large envelopes/fine resolutions.
    # The caller retains the original exact kernel as a fallback.
    area=(2*reach+1)**2
    if bins*area>8_000_000:return None
    with cp.cuda.Device(device):
        geometry,_=_kernels(device);values=cp.empty((bins,area),cp.uint8)
        geometry(((values.size+255)//256,),(256,),(values,np.int32(bins),np.int32(reach),
            np.int32(factor),np.float32(fine),np.float32(front),np.float32(back),np.float32(width)))
        accepted=cp.asnumpy(values);groups=[]
        for k in range(bins):
            masks={}
            for q in np.flatnonzero(accepted[k]):
                dy,dx=divmod(int(q),2*reach+1);dx-=reach;dy-=reach
                ox,sx=divmod(dx,factor);oy,sy=divmod(dy,factor)
                masks[ox,oy]=masks.get((ox,oy),0)|(1<<(sy*factor+sx))
            groups.append(masks)
        stride=max(map(len,groups),default=0)
        if not stride:return None
        dx=np.zeros((bins,stride),np.int32);dy=dx.copy();mask=np.zeros((bins,stride),np.uint32)
        counts=np.array([len(group)for group in groups],np.int32)
        for k,group in enumerate(groups):
            for n,((x,y),bits)in enumerate(group.items()):dx[k,n]=x;dy[k,n]=y;mask[k,n]=bits
        return cp.asarray(dx),cp.asarray(dy),cp.asarray(mask),cp.asarray(counts),stride


def cached_collision_poses(bits,resolution,front,back,width,bins,factor):
    import cupy as cp
    device=cp.cuda.runtime.getDevice();stencil=_stencil(device,float(resolution),float(front),float(back),
        float(width),int(bins),int(factor))
    if stencil is None:return None
    dx,dy,mask,counts,stride=stencil;rows,cols=bits.shape
    values=cp.asarray(bits,dtype=cp.uint32);out=cp.empty((bins,rows,cols),cp.uint8)
    if not out.size:return np.zeros(out.shape,bool)
    _,apply=_kernels(device)
    apply(((out.size+255)//256,),(256,),(values,out,dx,dy,mask,counts,
        np.int32(rows),np.int32(cols),np.int32(bins),np.int32(stride)))
    return cp.asnumpy(out).astype(bool)
