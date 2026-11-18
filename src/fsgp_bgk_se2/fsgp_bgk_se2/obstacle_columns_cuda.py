"""Fused CUDA implementation of the measured-ground occupancy detector."""
from functools import lru_cache
import math

import numpy as np


SOURCE = r'''
__device__ void atomic_lower(float *address, float value) {
  int *bits=reinterpret_cast<int*>(address);int old=*bits, assumed;
  while(value<__int_as_float(old)) {
    assumed=old;old=atomicCAS(bits,assumed,__float_as_int(value));
    if(old==assumed)break;
  }
}
__device__ void atomic_upper(float *address, float value) {
  int *bits=reinterpret_cast<int*>(address);int old=*bits, assumed;
  while(value>__int_as_float(old)) {
    assumed=old;old=atomicCAS(bits,assumed,__float_as_int(value));
    if(old==assumed)break;
  }
}
__device__ int cell(const float *p,int rows,int cols,double ox,double oy,double resolution) {
  if(!isfinite(p[0])||!isfinite(p[1])||!isfinite(p[2]))return -1;
  const double x=floor((double(p[0])-ox)/resolution),y=floor((double(p[1])-oy)/resolution);
  if(x<0||x>=cols||y<0||y>=rows)return -1;
  return static_cast<int>(y)*cols+static_cast<int>(x);
}
extern "C" __global__ void extrema(const float *points,int count,int rows,int cols,
    double ox,double oy,double resolution,float *low,float *high) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
  const float *p=points+3*i;const int q=cell(p,rows,cols,ox,oy,resolution);
  if(q>=0){atomic_lower(low+q,p[2]);atomic_upper(high+q,p[2]);}
}
extern "C" __global__ void ground_witness(const float *measured,int rows,int cols,
    float resolution,float min_height,float *trusted) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*cols)return;
  const int r=i/cols,c=i%cols;const float value=measured[i];bool rejected=false;
  for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++) {
    const int y=r+dy,x=c+dx;
    if((!dx&&!dy)||y<0||y>=rows||x<0||x>=cols)continue;
    const float difference=value-measured[y*cols+x];
    rejected|=isfinite(difference)&&difference>fmaxf(min_height,hypotf(dx,dy)*resolution);
  }
  trusted[i]=rejected?__int_as_float(0x7fffffff):value;
}
extern "C" __global__ void ground(const float *low,const float *trusted,const float *measured,int use_trusted,
    int rows,int cols,int radius,float *local,float *result,float *original) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*cols)return;
  const int r=i/cols,c=i%cols;float z=low[i];
  float near=use_trusted?trusted[i]:__int_as_float(0x7fffffff);
  int best=isfinite(near)?0:2147483647;
  float old_near=use_trusted?measured[i]:__int_as_float(0x7fffffff);
  int old_best=isfinite(old_near)?0:2147483647;
  for(int dy=-radius;dy<=radius;dy++)for(int dx=-radius;dx<=radius;dx++) {
    const int distance=dx*dx+dy*dy,y=r+dy,x=c+dx;
    if(distance>radius*radius||y<0||y>=rows||x<0||x>=cols)continue;
    const int q=y*cols+x;z=fminf(z,low[q]);
    if(use_trusted&&distance>0&&distance<best&&isfinite(trusted[q])) {
      near=trusted[q];best=distance;
    }
    if(use_trusted&&distance>0&&distance<old_best&&isfinite(measured[q])) {
      old_near=measured[q];old_best=distance;
    }
  }
  local[i]=z;result[i]=use_trusted?near:z;original[i]=old_near;
}
extern "C" __global__ void count_hits(const float *points,int count,int rows,int cols,
    double ox,double oy,double resolution,const float *local,const float *trusted,
    const float *original,int use_trusted,float min_height,int *hits,int *supported) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
  const float *p=points+3*i;const int q=cell(p,rows,cols,ox,oy,resolution);if(q<0)return;
  float relative=p[2]-local[q];
  if(relative>=min_height&&relative<=1.5f)atomicAdd(hits+q,1);
  if(use_trusted){relative=p[2]-trusted[q];
    const float old_relative=p[2]-original[q];
    if((isfinite(relative)&&relative>=min_height&&relative<=1.5f)||
       (isfinite(old_relative)&&old_relative>=min_height&&old_relative<=1.5f))atomicAdd(supported+q,1);}
}
extern "C" __global__ void occupied(const float *low,const float *high,const int *hits,
    const int *supported,int use_trusted,int rows,int cols,float min_span,int min_hits,bool *mask) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*cols)return;
  bool value=isfinite(low[i])&&isfinite(high[i])&&high[i]-low[i]>=min_span&&hits[i]>=min_hits;
  if(use_trusted&&supported[i]>0){
    int sum=0;const int r=i/cols,c=i%cols;
    for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++) {
      const int y=r+dy,x=c+dx;if(y>=0&&y<rows&&x>=0&&x<cols)sum+=supported[y*cols+x];
    }
    value|=sum>=min_hits;
  }
  mask[i]=value;
}
'''


@lru_cache(maxsize=8)
def _kernels(device):
    import cupy as cp
    module = cp.RawModule(code=SOURCE, options=('--std=c++11', '--fmad=false'),
                          name_expressions=('extrema', 'ground_witness', 'ground', 'count_hits', 'occupied'))
    return {name:module.get_function(name) for name in ('extrema', 'ground_witness', 'ground', 'count_hits', 'occupied')}


def vertical_columns_cuda(points, shape, origin, resolution, *, min_height, min_span,
                          min_hits, neighbor_radius, ground_elevation):
    import cupy as cp
    rows, cols = shape
    low = cp.full(shape, cp.inf, dtype=cp.float32)
    high = cp.full(shape, -cp.inf, dtype=cp.float32)
    points = cp.ascontiguousarray(cp.asarray(points, dtype=cp.float32))
    if not points.size:
        return cp.zeros(shape, dtype=bool), low
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError('point cloud must contain XYZ columns')
    if points.shape[1] != 3:
        points = cp.ascontiguousarray(points[cp.isfinite(points).all(axis=1), :3])
        if not len(points):
            return cp.zeros(shape, dtype=bool), low
    trusted = cp.empty((1,), dtype=cp.float32) if ground_elevation is None else cp.ascontiguousarray(cp.asarray(ground_elevation, dtype=cp.float32))
    if ground_elevation is not None and trusted.shape != shape:
        raise ValueError('observed ground must match the occupancy grid')
    local, result, original_ground = cp.empty_like(low), cp.empty_like(low), cp.empty_like(low)
    hits, supported = cp.zeros(shape, dtype=cp.int32), cp.zeros(shape, dtype=cp.int32)
    mask = cp.empty(shape, dtype=cp.bool_)
    kernels = _kernels(cp.cuda.runtime.getDevice())
    point_grid = ((len(points)+127)//128,)
    cell_grid = ((rows*cols+127)//128,)
    geometry = (np.int32(rows), np.int32(cols), np.float64(origin[0]), np.float64(origin[1]), np.float64(resolution))
    use_trusted = np.int32(ground_elevation is not None)
    measured = trusted
    kernels['extrema'](point_grid, (128,), (points, np.int32(len(points)), *geometry, low, high))
    if ground_elevation is not None:
        filtered = cp.empty_like(trusted)
        kernels['ground_witness'](cell_grid, (128,), (trusted, np.int32(rows), np.int32(cols),
            np.float32(resolution), np.float32(min_height), filtered))
        trusted = filtered
    kernels['ground'](cell_grid, (128,), (low, trusted, measured, use_trusted, np.int32(rows), np.int32(cols),
        np.int32(max(1, math.ceil(neighbor_radius/resolution))), local, result, original_ground))
    kernels['count_hits'](point_grid, (128,), (points, np.int32(len(points)), *geometry,
        local, result, original_ground, use_trusted, np.float32(min_height), hits, supported))
    kernels['occupied'](cell_grid, (128,), (low, high, hits, supported, use_trusted,
        np.int32(rows), np.int32(cols), np.float32(min_span), np.int32(min_hits), mask))
    return mask, result


SUBCELL_SOURCE=r'''
extern "C" __global__ void subcell_hits(const double *points,int count,
    const bool *occupied,int rows,int cols,double ox,double oy,double res,
    double pad,double tolerance,const double *floor_z,const double *gradient,int use_floor,int use_gradient,
    int factor,int use_bounds,unsigned int *bits,unsigned int *limits) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
  double px=points[3*i],py=points[3*i+1],pz=points[3*i+2];
  if(!isfinite(px)||!isfinite(py)||!isfinite(pz))return;
  double x=(px-ox)/res,y=(py-oy)/res,fx=floor(x),fy=floor(y);
  if(!isfinite(fx)||!isfinite(fy))return;
  if(fx<0||fx>=cols||fy<0||fy>=rows)return;
  int c=(int)fx,r=(int)fy,q=r*cols+c,area=rows*cols;
  if(!occupied[q])return;
  if(use_floor){
    double ground=floor_z[q];
    if(use_gradient)ground=ground+gradient[q]*((x-fx-.5)*res)+gradient[area+q]*((y-fy-.5)*res);
    if(isfinite(ground)&&fabs(pz-ground)<=tolerance)return;
  }
  x-=fx;y-=fy;
  for(int sy=0;sy<factor;++sy)for(int sx=0;sx<factor;++sx) {
    if(!(x+pad>=double(sx)/factor&&x-pad<=double(sx+1)/factor&&
         y+pad>=double(sy)/factor&&y-pad<=double(sy+1)/factor))continue;
    int cell=sy*factor+sx;atomicOr(bits+q,1u<<cell);
    if(!use_bounds)continue;
    double lx=fmax(x-pad,double(sx)/factor),hx=fmin(x+pad,double(sx+1)/factor);
    double ly=fmax(y-pad,double(sy)/factor),hy=fmin(y+pad,double(sy+1)/factor);
    unsigned int a=(unsigned int)fmin(15.,fmax(0.,floor((lx*factor-sx)*15.-1e-7)));
    unsigned int b=(unsigned int)fmin(15.,fmax(0.,ceil((hx*factor-sx)*15.+1e-7)));
    unsigned int c=(unsigned int)fmin(15.,fmax(0.,floor((ly*factor-sy)*15.-1e-7)));
    unsigned int d=(unsigned int)fmin(15.,fmax(0.,ceil((hy*factor-sy)*15.+1e-7)));
    int base=4*(cell*area+q);
    atomicMin(limits+base,a);atomicMax(limits+base+1,b);
    atomicMin(limits+base+2,c);atomicMax(limits+base+3,d);
  }
}
extern "C" __global__ void subcell_finish(const bool *occupied,const unsigned int *bits,
    const unsigned int *limits,int area,int factor,int use_bounds,unsigned short *out,unsigned short *bounds) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=area)return;
  unsigned int value=bits[i];out[i]=occupied[i]&&value==0?((1u<<(factor*factor))-1):value;
  if(use_bounds)for(int cell=0;cell<factor*factor;++cell) {
    int base=4*(cell*area+i);unsigned short code=0xF0F0;
    if(value&(1u<<cell))code=limits[base]|(limits[base+1]<<4)|(limits[base+2]<<8)|(limits[base+3]<<12);
    bounds[cell*area+i]=code;
  }
}
'''


@lru_cache(maxsize=8)
def _subcell_kernels(device):
    import cupy as cp
    module=cp.RawModule(code=SUBCELL_SOURCE,options=('--std=c++11','--fmad=false'),
        name_expressions=('subcell_hits','subcell_finish'))
    return module.get_function('subcell_hits'),module.get_function('subcell_finish')


def collision_subcells_cuda(points,occupied,origin,resolution,*,xy_padding_m,
                           measured_floor,gradient,floor_tolerance_m,factor,return_bounds):
    """Same padded boxes, floor veto and outward rounding as the CPU oracle.

    Double coordinates preserve ORIGINAL Float32/Float64 cell addressing;
    integer atomics make the result independent of point/thread ordering.
    No measured point is replaced by a centroid or by a GP height.
    """
    import cupy as cp
    rows,cols=occupied.shape;area=rows*cols
    if not area:
        result=np.zeros(occupied.shape,np.uint16)
        return (result,np.full((factor*factor,*occupied.shape),0xF0F0,np.uint16)) if return_bounds else result
    if points is None:raw=np.empty((0,3),np.float64)
    else:
        raw=np.asarray(points)
        if raw.ndim!=2 or raw.shape[1]<3:raise ValueError('expected original XYZ returns')
        raw=np.ascontiguousarray(raw[:,:3],dtype=np.float64)
    values=cp.zeros(occupied.shape,cp.uint32);mask=cp.ascontiguousarray(cp.asarray(occupied))
    floor=cp.empty(1,cp.float64) if measured_floor is None else cp.ascontiguousarray(cp.asarray(measured_floor,dtype=cp.float64))
    slope=cp.empty(1,cp.float64) if gradient is None else cp.ascontiguousarray(cp.asarray(gradient,dtype=cp.float64))
    limits=cp.empty((factor*factor,rows,cols,4) if return_bounds else (1,1,1,4),cp.uint32)
    if return_bounds:
        limits[...,0]=15;limits[...,1]=0;limits[...,2]=15;limits[...,3]=0
    out=cp.empty(occupied.shape,cp.uint16);bounds=cp.empty((factor*factor,rows,cols) if return_bounds else (1,),cp.uint16)
    hits,finish=_subcell_kernels(cp.cuda.runtime.getDevice())
    if len(raw):
        device_points=cp.asarray(raw)
        hits(((len(raw)+127)//128,),(128,),
            (device_points,np.int32(len(raw)),mask,np.int32(rows),np.int32(cols),
             np.float64(origin[0]),np.float64(origin[1]),np.float64(resolution),
             np.float64(xy_padding_m/resolution),np.float64(floor_tolerance_m),floor,slope,
             np.int32(measured_floor is not None),np.int32(gradient is not None),np.int32(factor),
             np.int32(return_bounds),values,limits))
    finish(((area+127)//128,),(128,),
        (mask,values,limits,np.int32(area),np.int32(factor),np.int32(return_bounds),out,bounds))
    result=cp.asnumpy(out)
    return (result,cp.asnumpy(bounds)) if return_bounds else result
