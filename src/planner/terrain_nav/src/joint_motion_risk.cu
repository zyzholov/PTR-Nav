#include "terrain_nav/joint_motion_kernel.hpp"
#include "terrain_nav/motion_buffer_capacity.hpp"
#include <cuda_runtime.h>
#include <atomic>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

namespace terrain_nav {
bool jointMotionCudaHits(const std::vector<joint_detail::Actor> &,const std::vector<joint_detail::Pose> &,
  size_t,uint64_t,std::vector<unsigned> &,std::string &);
bool jointMotionCudaSharedBatchHits(const std::vector<joint_detail::Actor> &,const std::vector<joint_detail::Pose> &,
  const std::vector<int> &,const std::vector<double> &,const std::vector<int> &,
  size_t,uint64_t,std::vector<unsigned> &,std::string &);
namespace {
std::atomic<uint64_t> buffer_growths{0};
__global__ void jointHits(const joint_detail::Actor *actors,const joint_detail::Pose *poses,const int *offsets,int objects,
    unsigned samples,uint64_t seed,unsigned *hits)
{
  const unsigned sample=blockIdx.x*blockDim.x+threadIdx.x,object=blockIdx.y,candidate=blockIdx.z;
  if(sample<samples&&joint_detail::trajectoryHit(actors[object],poses+offsets[candidate],offsets[candidate+1]-offsets[candidate],
       seed^(0x9e3779b97f4a7c15ULL*candidate),sample,object))atomicAdd(hits+candidate*objects+object,1u);
}
__global__ void sampleForecasts(const joint_detail::Actor *actors,const double *times,int count,
    unsigned samples,uint64_t seed,joint_detail::Segment *forecast)
{
  const unsigned sample=blockIdx.x*blockDim.x+threadIdx.x,object=blockIdx.y;
  if(sample>=samples)return;
  double state[6],previous=0.;auto rng=joint_detail::initialState(actors[object],seed,sample,object,state);
  for(int t=0;t<count;++t) {
    joint_detail::propagate(state,times[t]-previous,actors[object].density,rng,actors[object].constant_acceleration);previous=times[t];
    forecast[(size_t(object)*count+t)*samples+sample]=joint_detail::sampledSegment(actors[object],state,times[t]);
  }
}
__global__ void sharedHits(const joint_detail::Actor *actors,const joint_detail::Pose *poses,const int *offsets,
    const int *indices,int objects,const joint_detail::Segment *forecast,int count,unsigned samples,unsigned *hits)
{
  const unsigned sample=blockIdx.x*blockDim.x+threadIdx.x,object=blockIdx.y,candidate=blockIdx.z;
  if(sample<samples&&joint_detail::sharedTrajectoryHit(actors[object],poses+offsets[candidate],indices+offsets[candidate],
      offsets[candidate+1]-offsets[candidate],forecast+size_t(object)*count*samples,samples,sample))
    atomicAdd(hits+candidate*objects+object,1u);
}
struct Buffers {
  joint_detail::Actor *actors=nullptr;joint_detail::Pose *poses=nullptr;unsigned *hits=nullptr;
  int *offsets=nullptr;
  size_t actor_capacity=0,pose_capacity=0,hit_capacity=0,offset_capacity=0;cudaStream_t stream=nullptr;
  ~Buffers(){cudaFree(actors);cudaFree(poses);cudaFree(hits);cudaFree(offsets);if(stream)cudaStreamDestroy(stream);}
};
struct SharedBuffers:Buffers {
  double *times=nullptr;int *indices=nullptr;joint_detail::Segment *forecast=nullptr;
  size_t time_capacity=0,index_capacity=0,forecast_capacity=0;
  ~SharedBuffers(){cudaFree(times);cudaFree(indices);cudaFree(forecast);}
};
template<class T,class Check>bool grow(T *&pointer,size_t &capacity,size_t count,Check ok,
    size_t maximum=std::numeric_limits<size_t>::max()/sizeof(T))
{
  if(capacity>=count)return true;
  if(count>maximum)return false;
  const size_t reserved=reservedMotionCapacity(count,maximum);
  cudaFree(pointer);pointer=nullptr;capacity=0;
  if(!ok(cudaMalloc(&pointer,reserved*sizeof(T))))return false;
  capacity=reserved;++buffer_growths;return true;
}
}
// Read-only microbenchmark instrumentation; not a safety/performance permit.
uint64_t jointMotionCudaBufferGrowths(){return buffer_growths.load();}
bool jointMotionCudaAvailable(std::string &error)
{
  int devices=0;const auto e=cudaGetDeviceCount(&devices);
  if(e!=cudaSuccess||devices<1){error=e==cudaSuccess?"no CUDA device":cudaGetErrorString(e);return false;}
  // JIT/context/buffer initialization belongs to startup, never the first
  // emergency interaction. A failed warmup prevents this mode from starting.
  static std::once_flag once;static bool ready=false;static std::string warmup_error;
  std::call_once(once,[] {
    joint_detail::Actor actor{};actor.radius=1.;joint_detail::Pose pose{};pose.c=1.;pose.front=pose.back=pose.width=1.;
    std::vector<unsigned> hits(1,0);
    ready=jointMotionCudaHits({actor},{pose},32,0,hits,warmup_error)&&hits[0]==32;
    if(ready) {
      hits.assign(2,0);
      ready=jointMotionCudaSharedBatchHits({actor},{pose,pose},{0,1,2},{0.},{0,0},32,0,hits,warmup_error)&&
        hits[0]==32&&hits[1]==32;
    }
    if(!ready&&warmup_error.empty())warmup_error="CUDA joint-motion warmup parity failed";
  });
  error=warmup_error;return ready;
}
bool jointMotionCudaBatchHits(const std::vector<joint_detail::Actor> &actors,const std::vector<joint_detail::Pose> &poses,
  const std::vector<int> &offsets,size_t samples,uint64_t seed,std::vector<unsigned> &hits,std::string &error)
{
  static std::mutex mutex;static Buffers b;std::lock_guard<std::mutex> lock(mutex);
  const auto ok=[&](cudaError_t e){if(e==cudaSuccess)return true;error=cudaGetErrorString(e);return false;};
  if(actors.empty()||poses.empty()||offsets.size()<2||hits.size()!=actors.size()*(offsets.size()-1))
    {error="invalid CUDA inputs";return false;}
  if(!b.stream&&!ok(cudaStreamCreateWithFlags(&b.stream,cudaStreamNonBlocking)))return false;
  if(!grow(b.actors,b.actor_capacity,actors.size(),ok)||!grow(b.hits,b.hit_capacity,hits.size(),ok)||
     !grow(b.offsets,b.offset_capacity,offsets.size(),ok)||!grow(b.poses,b.pose_capacity,poses.size(),ok))return false;
  if(!ok(cudaMemcpyAsync(b.actors,actors.data(),actors.size()*sizeof(*b.actors),cudaMemcpyHostToDevice,b.stream))||
     !ok(cudaMemcpyAsync(b.poses,poses.data(),poses.size()*sizeof(*b.poses),cudaMemcpyHostToDevice,b.stream))||
     !ok(cudaMemcpyAsync(b.offsets,offsets.data(),offsets.size()*sizeof(*b.offsets),cudaMemcpyHostToDevice,b.stream))||
     !ok(cudaMemsetAsync(b.hits,0,hits.size()*sizeof(*b.hits),b.stream)))return false;
  jointHits<<<dim3((samples+127)/128,actors.size(),offsets.size()-1),128,0,b.stream>>>(b.actors,b.poses,b.offsets,actors.size(),samples,seed,b.hits);
  if(!ok(cudaGetLastError())||!ok(cudaMemcpyAsync(hits.data(),b.hits,hits.size()*sizeof(*b.hits),cudaMemcpyDeviceToHost,b.stream))||
     !ok(cudaStreamSynchronize(b.stream)))return false;
  return true;
}
bool jointMotionCudaHits(const std::vector<joint_detail::Actor> &actors,const std::vector<joint_detail::Pose> &poses,
  size_t samples,uint64_t seed,std::vector<unsigned> &hits,std::string &error)
{
  return jointMotionCudaBatchHits(actors,poses,{0,int(poses.size())},samples,seed,hits,error);
}
bool jointMotionCudaSharedBatchHits(const std::vector<joint_detail::Actor> &actors,const std::vector<joint_detail::Pose> &poses,
  const std::vector<int> &offsets,const std::vector<double> &times,const std::vector<int> &indices,
  size_t samples,uint64_t seed,std::vector<unsigned> &hits,std::string &error)
{
  static std::mutex mutex;static SharedBuffers b;std::lock_guard<std::mutex> lock(mutex);
  const auto ok=[&](cudaError_t e){if(e==cudaSuccess)return true;error=cudaGetErrorString(e);return false;};
  const size_t count=actors.size()*times.size()*samples;
  if(actors.empty()||poses.empty()||times.empty()||offsets.size()<2||indices.size()!=poses.size()||
     hits.size()!=actors.size()*(offsets.size()-1)||count>256*1024*1024/sizeof(joint_detail::Segment))
    {error="invalid shared CUDA inputs or memory budget";return false;}
  if(!b.stream&&!ok(cudaStreamCreateWithFlags(&b.stream,cudaStreamNonBlocking)))return false;
  if(!grow(b.actors,b.actor_capacity,actors.size(),ok)||!grow(b.poses,b.pose_capacity,poses.size(),ok)||
     !grow(b.hits,b.hit_capacity,hits.size(),ok)||!grow(b.offsets,b.offset_capacity,offsets.size(),ok)||
     !grow(b.indices,b.index_capacity,indices.size(),ok)||!grow(b.times,b.time_capacity,times.size(),ok)||
     !grow(b.forecast,b.forecast_capacity,count,ok,256*1024*1024/sizeof(joint_detail::Segment)))return false;
  if(!ok(cudaMemcpyAsync(b.actors,actors.data(),actors.size()*sizeof(*b.actors),cudaMemcpyHostToDevice,b.stream))||
     !ok(cudaMemcpyAsync(b.poses,poses.data(),poses.size()*sizeof(*b.poses),cudaMemcpyHostToDevice,b.stream))||
     !ok(cudaMemcpyAsync(b.offsets,offsets.data(),offsets.size()*sizeof(*b.offsets),cudaMemcpyHostToDevice,b.stream))||
     !ok(cudaMemcpyAsync(b.indices,indices.data(),indices.size()*sizeof(*b.indices),cudaMemcpyHostToDevice,b.stream))||
     !ok(cudaMemcpyAsync(b.times,times.data(),times.size()*sizeof(*b.times),cudaMemcpyHostToDevice,b.stream))||
     !ok(cudaMemsetAsync(b.hits,0,hits.size()*sizeof(*b.hits),b.stream)))return false;
  sampleForecasts<<<dim3((samples+127)/128,actors.size()),128,0,b.stream>>>(
      b.actors,b.times,times.size(),samples,seed,b.forecast);
  if(!ok(cudaGetLastError()))return false;
  sharedHits<<<dim3((samples+127)/128,actors.size(),offsets.size()-1),128,0,b.stream>>>(
      b.actors,b.poses,b.offsets,b.indices,actors.size(),b.forecast,times.size(),samples,b.hits);
  if(!ok(cudaGetLastError())||!ok(cudaMemcpyAsync(hits.data(),b.hits,hits.size()*sizeof(*b.hits),cudaMemcpyDeviceToHost,b.stream))||
     !ok(cudaStreamSynchronize(b.stream)))return false;
  return true;
}
} // namespace terrain_nav
