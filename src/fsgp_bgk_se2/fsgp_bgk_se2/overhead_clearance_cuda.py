"""Fused CUDA reference-plane solve and non-cascading normal transfer.

The CPU supplies the same measured witness/component masks and offset order.
No GP heights, simulator truth, probability thresholds or body sizes are changed.
"""
from functools import lru_cache
import numpy as np


SOURCE=r'''
#define INFINITY __longlong_as_double(0x7ff0000000000000ULL)
extern "C" __global__ void query_halo(const unsigned char* query,const int* offsets,
 int count,int rows,int cols,unsigned char* targets) {
 int id=blockDim.x*blockIdx.x+threadIdx.x;if(id>=rows*cols)return;
 int r=id/cols,c=id%cols;bool found=false;
 for(int k=0;k<count&&!found;k++) {
  int rr=r+offsets[2*k],cc=c+offsets[2*k+1];
  if(rr>=0&&rr<rows&&cc>=0&&cc<cols)found=query[rr*cols+cc];
 }
 targets[id]=found;
}
extern "C" __global__ void fit(const double* z,const unsigned char* floor,
 const unsigned char* witness,const int* labels,const int* owner,const int* offsets,
 const unsigned char* fit_targets,int count,int rows,int cols,double resolution,double residual_limit,double* model,unsigned char* seeds) {
 int id=blockDim.x*blockIdx.x+threadIdx.x;if(id>=rows*cols)return;
 int r=id/cols,c=id%cols,n=0,q[4]={0,0,0,0};
 seeds[id]=0;double* out=model+4*id;out[0]=out[1]=out[2]=out[3]=0;
 if(!fit_targets[id])return;
 double su=0,sv=0,suu=0,suv=0,svv=0,b0=0,b1=0,b2=0;
 for(int k=0;k<count;k++) {
  int dy=offsets[2*k],dx=offsets[2*k+1],rr=r+dy,cc=c+dx;
  if(rr<0||rr>=rows||cc<0||cc>=cols)continue;int j=rr*cols+cc;
  if(!witness[j]||labels[j]!=owner[id])continue;
  double u=dx*resolution,v=dy*resolution,h=z[j];
  n++;su+=u;sv+=v;suu+=u*u;suv+=u*v;svv+=v*v;b0+=h;b1+=h*u;b2+=h*v;
  if(dx<0&&dy<0)q[0]++;if(dx>0&&dy<0)q[1]++;
  if(dx<0&&dy>0)q[2]++;if(dx>0&&dy>0)q[3]++;
 }
 double a=suu*svv-suv*suv,b=sv*suv-su*svv,d=su*suv-sv*suu;
 double det=n*a+su*b+sv*d;
 if(n<8||!q[0]||!q[1]||!q[2]||!q[3]||!(det>1e-6))return;
 double p0=(a*b0+b*b1+d*b2)/det;
 double p1=(b*b0+(n*svv-sv*sv)*b1+(su*sv-n*suv)*b2)/det;
 double p2=(d*b0+(su*sv-n*suv)*b1+(n*suu-su*su)*b2)/det;
 double residual=0;
 for(int k=0;k<count;k++) {
  int dy=offsets[2*k],dx=offsets[2*k+1],rr=r+dy,cc=c+dx;
  if(rr<0||rr>=rows||cc<0||cc>=cols)continue;int j=rr*cols+cc;
  if(witness[j]&&labels[j]==owner[id])
   residual=fmax(residual,fabs(z[j]-p0-p1*(dx*resolution)-p2*(dy*resolution)));
 }
 out[0]=p0;out[1]=p1;out[2]=p2;out[3]=residual;
 seeds[id]=(p1*p1+p2*p2<=1.0&&residual<=residual_limit&&(!floor[id]||fabs(p0-z[id])<=.05));
}
extern "C" __global__ void envelope(const double* z,const unsigned char* floor,
 const unsigned char* direct,const unsigned char* columns,const double* model,
 const unsigned char* seeds,const int* nearby,int count,int rows,int cols,
 const unsigned char* query,double resolution,double height,double margin,double reserve,float* ceiling) {
 int id=blockDim.x*blockIdx.x+threadIdx.x;if(id>=rows*cols)return;
 if(!query[id]){ceiling[id]=INFINITY;return;}
 int r=id/cols,c=id%cols;bool valid=seeds[id];
 double p0=model[4*id],p1=model[4*id+1],p2=model[4*id+2],res=model[4*id+3];
 if(!valid&&floor[id]&&direct[id]&&r>0&&r<rows-1&&c>0&&c<cols-1) {
  for(int k=0;k<count;k++) {
   int dy=nearby[2*k],dx=nearby[2*k+1],rr=r+dy,cc=c+dx;
   if(rr<0||rr>=rows||cc<0||cc>=cols)continue;int j=rr*cols+cc;
   if(!seeds[j])continue;
   double a=model[4*j+1],b=model[4*j+2];
   double predicted=model[4*j]-a*(dx*resolution)-b*(dy*resolution);
   if(!(fabs(predicted-z[id])<=.04))continue;bool good=true;
   for(int oy=-1;oy<=1&&good;oy++)for(int ox=-1;ox<=1;ox++) {
    int nr=r+oy,nc=c+ox;if(nr<0||nr>=rows||nc<0||nc>=cols)continue;
    int n=nr*cols+nc;
    if(floor[n]&&!columns[n]&&!(fabs(z[n]-predicted-a*(ox*resolution)-b*(oy*resolution))<=.04)) {
     good=false;break;
    }
   }
   if(good){p0=predicted;p1=a;p2=b;res=model[4*j+3];valid=true;break;}
  }
 }
 ceiling[id]=valid ? fmax(p0+res,floor[id]?z[id]:-INFINITY)+
  height*sqrt(1+p1*p1+p2*p2)+margin+reserve : INFINITY;
}
'''


@lru_cache(maxsize=1)
def kernels():
    import cupy as cp
    module=cp.RawModule(code=SOURCE,options=('--std=c++14','--fmad=false'),
                        name_expressions=('fit','envelope','query_halo'))
    return tuple(module.get_function(name)for name in ('fit','envelope','query_halo'))


@lru_cache(maxsize=32)
def geometry(resolution,reference_radius_m,device):
    """Immutable offsets only; no measurements, fitted planes or query masks."""
    import cupy as cp
    radius=max(1,int(np.ceil(reference_radius_m/resolution)))
    offsets=[(dy,dx) for dy in range(-radius,radius+1) for dx in range(-radius,radius+1)
             if np.hypot(dx,dy)*resolution<=reference_radius_m]
    nearby=[offset for offset in sorted(offsets,key=lambda v:v[0]**2+v[1]**2)
            if (offset[0]*resolution)**2+(offset[1]*resolution)**2<=.4**2+1e-9]
    return cp.asarray(offsets,dtype=cp.int32),cp.asarray(nearby,dtype=cp.int32)


def resident_plane_envelopes(z,floor,witness,direct,columns,labels,owner,resolution,
                            height,margin,reserve,reference_radius_m,max_residual_m,*,measured_fallback):
    """Same two proofs on one current input; only final heights cross to CPU.

    The broad query and its complete normal-transfer halo are built on device.
    Even an empty broad query can run without a host scalar synchronization.
    No plane/query is retained for another scan, and no evidence is discarded.
    """
    import cupy as cp
    rows,cols=z.shape;device=cp.cuda.runtime.getDevice()
    zz=cp.asarray(z,dtype=cp.float64,order='C')
    ff,ww,dd,oo=(cp.asarray(v,dtype=cp.uint8,order='C')for v in (floor,witness,direct,columns))
    ll,owners=(cp.asarray(v,dtype=cp.int32,order='C')for v in (labels,owner))
    fit,envelope,halo=kernels();blocks=((rows*cols+127)//128,)
    def evaluate(radius,residual,query,local):
        offsets,nearby=geometry(resolution,radius,device)
        targets=cp.ones((rows,cols),cp.uint8)if local else cp.empty((rows,cols),cp.uint8)
        if not local:
            halo(blocks,(128,),(query,nearby,np.int32(len(nearby)),np.int32(rows),np.int32(cols),targets))
        model=cp.empty((rows*cols,4),cp.float64);seeds=cp.empty(rows*cols,cp.uint8)
        result=cp.empty((rows,cols),cp.float32)
        fit(blocks,(128,),(zz,ff,ww,ll,owners,offsets,targets,np.int32(len(offsets)),
            np.int32(rows),np.int32(cols),np.float64(resolution),np.float64(residual),model,seeds))
        envelope(blocks,(128,),(zz,ff,dd,oo,model,seeds,nearby,np.int32(len(nearby)),
            np.int32(rows),np.int32(cols),query,np.float64(resolution),np.float64(height),
            np.float64(margin),np.float64(reserve),result))
        return result
    local=evaluate(reference_radius_m,max_residual_m,cp.ones((rows,cols),cp.uint8),True)
    if measured_fallback:
        missing=~cp.isfinite(local)&(ff!=0)
        broad=evaluate(3.,.10,missing,False)
        local=cp.where(missing,broad,local)
    return cp.asnumpy(local)


def plane_envelope(z,floor,witness,direct,columns,labels,owner,resolution,
                   height,margin,reserve,reference_radius_m,max_residual_m,*,query_mask=None):
    import cupy as cp
    from scipy.ndimage import binary_dilation
    rows,cols=z.shape
    all_offsets,small_offsets=geometry(resolution,reference_radius_m,cp.cuda.runtime.getDevice())
    if query_mask is None:query_mask=np.ones(z.shape,bool)
    query_mask=np.asarray(query_mask,bool)
    if query_mask.shape!=z.shape:raise ValueError('clearance query mask shape differs from terrain')
    # A target may copy a normal from any ORIGINAL seed in this exact halo.
    # Fitting only the requested cells would silently remove that evidence.
    halo=max(1,int(np.ceil(.4/resolution)));dy,dx=np.mgrid[-halo:halo+1,-halo:halo+1]
    structure=(dy*resolution)**2+(dx*resolution)**2<=.4**2+1e-9
    fit_mask=binary_dilation(query_mask,structure=structure)
    zz=cp.asarray(z,dtype=cp.float64,order='C')
    ff,ww,dd,oo=(cp.asarray(v,dtype=cp.uint8,order='C') for v in (floor,witness,direct,columns))
    ll,owners=(cp.asarray(v,dtype=cp.int32,order='C') for v in (labels,owner))
    queries,fit_targets=(cp.asarray(v,dtype=cp.uint8,order='C')for v in (query_mask,fit_mask))
    model=cp.empty((rows*cols,4),cp.float64);seeds=cp.empty(rows*cols,cp.uint8)
    result=cp.empty((rows,cols),cp.float32);fit,envelope,_=kernels()
    blocks=((rows*cols+127)//128,)
    fit(blocks,(128,),(zz,ff,ww,ll,owners,all_offsets,fit_targets,np.int32(len(all_offsets)),
        np.int32(rows),np.int32(cols),np.float64(resolution),np.float64(max_residual_m),model,seeds))
    envelope(blocks,(128,),(zz,ff,dd,oo,model,seeds,small_offsets,np.int32(len(small_offsets)),
        np.int32(rows),np.int32(cols),queries,np.float64(resolution),np.float64(height),
        np.float64(margin),np.float64(reserve),result))
    return cp.asnumpy(result)
