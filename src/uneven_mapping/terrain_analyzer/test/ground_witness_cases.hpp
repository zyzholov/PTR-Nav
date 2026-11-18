#pragma once
#include "terrain_analyzer/ground_fusion.hpp"
#ifdef __CUDACC__
#define TEST_HD __host__ __device__
#else
#define TEST_HD
#endif
struct WitnessSamples {
  int mode;bool own;
  TEST_HD int first()const{return mode==4&&own?-1:0;}
  TEST_HD int next(int i)const{return ++i<(own?1:25)?i:-1;}
  TEST_HD float height(int i)const {
    if(mode==3||mode==11)return own?-.4f:(i==12?-.4f:0.f);
    if(mode==9&&own)return .1f;
    if(i==0&&!own){if(mode==1)return .7f;if(mode==2)return .12f;if(mode==12)return .31f;}
    return 0.f;
  }
  TEST_HD float variance(int)const{return .0001f;}
  TEST_HD float x(int i)const {return own?0.f:mode==8?(.3f+.2f*(i%5)):(-.9f+.45f*(i%5));}
  TEST_HD float y(int i)const {return own?0.f:(-.9f+.45f*(i/5));}
  TEST_HD unsigned int coverage(int)const{return 1U;}
};
struct WitnessReferences {
  int mode;
  TEST_HD int size()const{return 81;}
  TEST_HD int dx(int k)const{return k%9-4;}
  TEST_HD int dy(int k)const{return k/9-4;}
  TEST_HD terrain_analyzer::GroundState state(int k)const {
    terrain_analyzer::GroundState s;s.observed=mode!=10 && !(mode==5&&dx(k)>0);
    s.height=0;s.variance=mode==7?.01f:.0001f;s.pending_count=mode==6?1:0;
    // The old column covers the immediate neighbors. A larger enclosure
    // cannot substitute a height: the CURRENT query return is still required.
    if(dx(k)*dx(k)+dy(k)*dy(k)<=6)s.height=.4f;
    return s;
  }
};
struct WitnessResult {terrain_analyzer::GroundScan scan;bool qualified;};
TEST_HD inline WitnessResult witnessCase(int mode) {
  WitnessResult r{};
  r.qualified=terrain_analyzer::reobservedGroundPlane(WitnessSamples{mode,true},WitnessSamples{mode,false},
      WitnessReferences{mode},.0001f,r.scan);
  return r;
}
struct FreeSpaceReferences {
  int mode;
  TEST_HD int size()const{return 81;}
  TEST_HD int dx(int k)const{return k%9-4;}
  TEST_HD int dy(int k)const{return k/9-4;}
  TEST_HD terrain_analyzer::GroundState state(int k)const {
    terrain_analyzer::GroundState s;s.observed=mode!=6 && !(mode==3&&dx(k)>0);
    s.height=mode==11?(k%2?.12f:0.f):mode==12&&k==0?-.2f:0.f;
    s.variance=mode==5?.01f:.0001f;s.pending_count=mode==4?1:0;return s;
  }
};
struct FreeSpaceResult {bool retired;float reference;};
TEST_HD inline FreeSpaceResult freeSpaceCase(int mode) {
  terrain_analyzer::GroundState old;old.observed=mode!=10;
  old.height=mode==7?-.4f:mode==8?.10f:.39f;
  old.variance=mode==9?.01f:mode==13?.00055f:.0001f;
  FreeSpaceResult r{false,99.f};
  r.retired=terrain_analyzer::freeSpaceGroundReference(old,FreeSpaceReferences{mode},
      mode==1?0:mode==2?2:3,3,.18f,r.reference);
  return r;
}
struct SparseReturnSamples {
  int mode;
  TEST_HD int first()const{return mode==1?-1:0;}
  TEST_HD int next(int i)const{return ++i<(mode==10?500:mode==9?2:1)?i:-1;}
  TEST_HD float height(int i)const{return mode==2?-.4f:mode==3?.06f:mode==9&&i?.035f:0.f;}
  TEST_HD float variance(int)const{return mode==8?.001f:.0001f;}
  TEST_HD unsigned int coverage(int)const{return 1U;}
};
struct SparseReturnReferences {
  int mode;
  TEST_HD int size()const{return 81;}
  TEST_HD int dx(int k)const{return k%9-4;}
  TEST_HD int dy(int k)const{return k/9-4;}
  TEST_HD terrain_analyzer::GroundState state(int k)const {
    terrain_analyzer::GroundState s;s.observed=!(mode==4&&dx(k)>0);
    s.height=mode==7&&k%2?.15f:mode==11?(k%2?.024f:-.024f):0.f;
    s.variance=mode==6?.01f:.0001f;s.pending_count=mode==5?1:0;return s;
  }
};
TEST_HD inline WitnessResult sparseReturnCase(int mode) {
  terrain_analyzer::GroundState old;old.observed=true;old.height=.39f;old.variance=.00055f;
  WitnessResult r{};
  r.qualified=terrain_analyzer::reobservedGroundReturn(old,SparseReturnSamples{mode},
      SparseReturnReferences{mode},.0001f,r.scan);
  return r;
}
#undef TEST_HD
