#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "terrain_analyzer/ground_fusion.hpp"
#include "ground_witness_cases.hpp"
using namespace terrain_analyzer;
__global__ void sparseReturnReplay(WitnessResult *out) {out[threadIdx.x]=sparseReturnCase(threadIdx.x);}
TEST(GroundFusionCuda,SparseCurrentRealReturnAndNegativeEnclosuresMatchHost) {
  WitnessResult *gpu=nullptr,actual[12];ASSERT_EQ(cudaMalloc(&gpu,sizeof(actual)),cudaSuccess);
  sparseReturnReplay<<<1,12>>>(gpu);ASSERT_EQ(cudaGetLastError(),cudaSuccess);
  ASSERT_EQ(cudaMemcpy(actual,gpu,sizeof(actual),cudaMemcpyDeviceToHost),cudaSuccess);
  for(int i=0;i<12;++i) {
    const auto expected=sparseReturnCase(i);EXPECT_EQ(actual[i].qualified,expected.qualified)<<i;
    EXPECT_FLOAT_EQ(actual[i].scan.height,expected.scan.height);
    EXPECT_FLOAT_EQ(actual[i].scan.variance,expected.scan.variance);
    EXPECT_EQ(actual[i].scan.count,expected.scan.count);EXPECT_EQ(actual[i].scan.centered,expected.scan.centered);
  }
  cudaFree(gpu);
}
__global__ void freeSpaceReplay(FreeSpaceResult *out) {out[threadIdx.x]=freeSpaceCase(threadIdx.x);}
TEST(GroundFusionCuda,FreeSpaceRetirementAndAllNegativeCasesMatchHost) {
  FreeSpaceResult *gpu=nullptr,actual[14];ASSERT_EQ(cudaMalloc(&gpu,sizeof(actual)),cudaSuccess);
  freeSpaceReplay<<<1,14>>>(gpu);ASSERT_EQ(cudaGetLastError(),cudaSuccess);
  ASSERT_EQ(cudaMemcpy(actual,gpu,sizeof(actual),cudaMemcpyDeviceToHost),cudaSuccess);
  for(int i=0;i<14;++i) {
    const auto expected=freeSpaceCase(i);EXPECT_EQ(actual[i].retired,expected.retired)<<i;
    EXPECT_FLOAT_EQ(actual[i].reference,expected.reference)<<i;
  }
  cudaFree(gpu);
}
__global__ void witnessReplay(WitnessResult *out) {out[threadIdx.x]=witnessCase(threadIdx.x);}
TEST(GroundFusionCuda,EnclosedRealFloorAndAllNegativeCasesMatchHost) {
  WitnessResult *gpu=nullptr,actual[13];ASSERT_EQ(cudaMalloc(&gpu,sizeof(actual)),cudaSuccess);
  witnessReplay<<<1,13>>>(gpu);ASSERT_EQ(cudaGetLastError(),cudaSuccess);
  ASSERT_EQ(cudaMemcpy(actual,gpu,sizeof(actual),cudaMemcpyDeviceToHost),cudaSuccess);
  for(int i=0;i<13;++i) {
    const auto expected=witnessCase(i);EXPECT_EQ(actual[i].qualified,expected.qualified)<<i;
    EXPECT_EQ(actual[i].qualified,i==0||i==1)<<i;
    if(expected.qualified) {
      EXPECT_NEAR(actual[i].scan.height,expected.scan.height,1e-6);
      EXPECT_NEAR(actual[i].scan.variance,expected.scan.variance,1e-7);
      EXPECT_EQ(actual[i].scan.count,expected.scan.count);EXPECT_EQ(actual[i].scan.coverage,expected.scan.coverage);
    }
  }
  cudaFree(gpu);
}
__host__ __device__ GroundState enclosedReplayCase(int mode,bool *qualified) {
  GroundState old;old.observed=true;old.height=.4f;old.variance=.0001f;old.pending_count=1;
  GroundScan scan{0,.0001f,1,1,1U,true};GroundState neighbors[9];
  for(int k=0;k<9;++k){neighbors[k].observed=true;neighbors[k].variance=.0001f;}
  if(mode==1)scan.count=0;
  if(mode==2)scan.centered=false;
  if(mode==3)for(int k=3;k<9;++k)neighbors[k].observed=false;
  if(mode==4)for(int k=0;k<9;++k)neighbors[k].pending_count=1;
  if(mode==5){old.height=0;scan.height=-.4f;}
  const bool q=enclosedLowerGround(old,scan,neighbors);if(qualified)*qualified=q;
  return fuseGroundScan(old,scan,.0001f,2.5f,true,.18f,q);
}
__global__ void enclosedReplay(GroundState *out) {out[threadIdx.x]=enclosedReplayCase(threadIdx.x,nullptr);}
TEST(GroundFusionCuda,LowerGroundRetirementAndPitUnknownVetoMatchHost) {
  GroundState *gpu=nullptr,actual[6];ASSERT_EQ(cudaMalloc(&gpu,sizeof(actual)),cudaSuccess);
  enclosedReplay<<<1,6>>>(gpu);ASSERT_EQ(cudaGetLastError(),cudaSuccess);
  ASSERT_EQ(cudaMemcpy(actual,gpu,sizeof(actual),cudaMemcpyDeviceToHost),cudaSuccess);
  for(int k=0;k<6;++k) {
    bool qualified=false;const auto expected=enclosedReplayCase(k,&qualified);
    EXPECT_EQ(qualified,k==0);EXPECT_FLOAT_EQ(actual[k].height,expected.height);
    EXPECT_FLOAT_EQ(actual[k].variance,expected.variance);EXPECT_EQ(actual[k].pending_count,expected.pending_count);
  }
  cudaFree(gpu);
}
struct DenseSamples {
  const float *h;int n;bool spatial=false;
  __host__ __device__ int first() const {return n?0:-1;}
  __host__ __device__ int next(int i) const {return ++i<n?i:-1;}
  __host__ __device__ float height(int i) const {return h[i];}
  __host__ __device__ float variance(int) const {return .0001f;}
  __host__ __device__ float x(int i)const{return -.4f+.2f*(i%5);}
  __host__ __device__ float y(int i)const{return -.4f+.2f*(i/5);}
  __host__ __device__ unsigned int coverage(int i) const {
    return spatial?groundCoverageBit(.125f+.25f*(i%4),.125f+.25f*(i/4)):1U;
  }
};
__global__ void replay(const float *h,GroundState *states,bool spatial) {
  GroundState s;s.observed=true;s.variance=.0001f;
  for(int i=0;i<6;++i) {
    s=fuseGroundScan(s,robustGroundScan(DenseSamples{h+5*i,5,spatial},.0001f),.0001f,2.5f,true,.18f);
    states[i]=s;
  }
}
TEST(GroundFusionCuda,MatchesHostAcrossOutlierRecoveryAndHighObstacleSequence) {
  const float h[]={-.08f,-.001f,0.f,.001f,.002f, .06f,.06f,.06f,.06f,.06f,
    .06f,.06f,.06f,.06f,.06f, .06f,.06f,.06f,.06f,.06f,
    .7f,.7f,.7f,.7f,.7f, -.3f,-.3f,-.3f,-.3f,-.3f};
  float *gpu=nullptr;GroundState *output=nullptr,actual[6];
  ASSERT_EQ(cudaMalloc(&gpu,sizeof(h)),cudaSuccess);
  ASSERT_EQ(cudaMalloc(&output,sizeof(actual)),cudaSuccess);
  ASSERT_EQ(cudaMemcpy(gpu,h,sizeof(h),cudaMemcpyHostToDevice),cudaSuccess);
  for(bool spatial:{false,true}) {
    replay<<<1,1>>>(gpu,output,spatial);ASSERT_EQ(cudaGetLastError(),cudaSuccess);
    ASSERT_EQ(cudaMemcpy(actual,output,sizeof(actual),cudaMemcpyDeviceToHost),cudaSuccess);
    GroundState reference;reference.observed=true;reference.variance=.0001f;
    for(int i=0;i<6;++i) {
      reference=fuseGroundScan(reference,robustGroundScan(DenseSamples{h+5*i,5,spatial},.0001f),.0001f,2.5f,true,.18f);
      EXPECT_NEAR(actual[i].height,reference.height,1e-6f);
      EXPECT_NEAR(actual[i].variance,reference.variance,1e-7f);
      EXPECT_EQ(actual[i].pending_count,reference.pending_count);EXPECT_EQ(actual[i].observed,reference.observed);
    }
    EXPECT_EQ(actual[1].pending_count,spatial?0:1);
  }
  cudaFree(output);cudaFree(gpu);
}

__global__ void centreReplay(const float *h,GroundScan *output) {
  output[0]=centeredGroundScan(DenseSamples{h,25,true},.0001f);
}
TEST(GroundFusionCuda,CellCentrePlaneAndStepFallbackMatchHost) {
  float h[25],*gpu=nullptr;GroundScan *output=nullptr,actual;
  ASSERT_EQ(cudaMalloc(&gpu,sizeof(h)),cudaSuccess);
  ASSERT_EQ(cudaMalloc(&output,sizeof(actual)),cudaSuccess);
  for(bool step:{false,true}) {
    for(int i=0;i<25;++i)h[i]=step?(i%5>2?.06f:0.f):.12f+.13f*(-.4f+.2f*(i%5))+.04f*(-.4f+.2f*(i/5));
    ASSERT_EQ(cudaMemcpy(gpu,h,sizeof(h),cudaMemcpyHostToDevice),cudaSuccess);
    centreReplay<<<1,1>>>(gpu,output);ASSERT_EQ(cudaGetLastError(),cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&actual,output,sizeof(actual),cudaMemcpyDeviceToHost),cudaSuccess);
    const auto expected=centeredGroundScan(DenseSamples{h,25,true},.0001f);
    EXPECT_NEAR(actual.height,expected.height,1e-6f);EXPECT_NEAR(actual.variance,expected.variance,1e-7f);
    EXPECT_EQ(actual.count,expected.count);EXPECT_EQ(actual.inliers,expected.inliers);
  }
  cudaFree(output);cudaFree(gpu);
}

__global__ void patchReplay(const float *h,GroundScan *output) {
  output[0]=patchCenteredGroundScan(DenseSamples{h,1,true},DenseSamples{h,25,true},.0001f);
}
TEST(GroundFusionCuda,SparseRealCellAndPatchStepVetoMatchHost) {
  float h[25],*gpu=nullptr;GroundScan *output=nullptr,actual;
  ASSERT_EQ(cudaMalloc(&gpu,sizeof(h)),cudaSuccess);
  ASSERT_EQ(cudaMalloc(&output,sizeof(actual)),cudaSuccess);
  for(bool step:{false,true}) {
    for(int i=0;i<25;++i)h[i]=step?(i%5>2?.06f:0.f):.12f+.13f*(-.4f+.2f*(i%5))+.04f*(-.4f+.2f*(i/5));
    ASSERT_EQ(cudaMemcpy(gpu,h,sizeof(h),cudaMemcpyHostToDevice),cudaSuccess);
    patchReplay<<<1,1>>>(gpu,output);ASSERT_EQ(cudaGetLastError(),cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&actual,output,sizeof(actual),cudaMemcpyDeviceToHost),cudaSuccess);
    const auto expected=patchCenteredGroundScan(DenseSamples{h,1,true},DenseSamples{h,25,true},.0001f);
    EXPECT_NEAR(actual.height,expected.height,1e-6f);EXPECT_NEAR(actual.variance,expected.variance,1e-7f);
    EXPECT_EQ(actual.count,1);EXPECT_EQ(actual.inliers,1);EXPECT_EQ(actual.coverage,expected.coverage);
    EXPECT_EQ(actual.centered,!step);
  }
  cudaFree(output);cudaFree(gpu);
}
