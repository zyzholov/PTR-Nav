#pragma once
#include <cmath>

#ifdef __CUDACC__
#define TERRAIN_HD __host__ __device__
#else
#define TERRAIN_HD
#endif

namespace terrain_analyzer {
struct GroundScan {float height=0,variance=0;int count=0,inliers=0;unsigned int coverage=0;bool centered=false;};
struct GroundState {
  float height=0,variance=0,pending_height=0,pending_variance=0;
  int pending_count=0;
  bool observed=false;
};

TERRAIN_HD inline unsigned int groundCoverageBit(float x_fraction,float y_fraction) {
  const int x=static_cast<int>(fminf(3.f,fmaxf(0.f,x_fraction*4)));
  const int y=static_cast<int>(fminf(3.f,fmaxf(0.f,y_fraction*4)));
  return 1U<<(4*y+x);
}

// The accessor exposes first/next/height/variance/coverage. No fixed-size truncation,
// point-order dependence, or averaging independent errors within a scan.
template<class Samples> TERRAIN_HD GroundScan robustGroundScan(const Samples &s,float minimum_variance) {
  float lower=INFINITY,upper=-INFINITY;int count=0;
  for(int i=s.first();i>=0;i=s.next(i)) {
    const float h=s.height(i),v=s.variance(i);
    if(!std::isfinite(h)||!std::isfinite(v)||v<0)continue;
    lower=fminf(lower,h);upper=fmaxf(upper,h);++count;
  }
  if(!count)return {};
  const int rank=(count+1)/2;
  // Lower median to <=10 micrometres; bounded work even for a dense cell.
  for(int pass=0;pass<24 && upper-lower>1e-5f;++pass) {
    const float middle=(lower+upper)*.5f;int below=0;
    for(int i=s.first();i>=0;i=s.next(i))if(std::isfinite(s.height(i))&&
        std::isfinite(s.variance(i))&&s.variance(i)>=0&&s.height(i)<=middle)++below;
    if(below>=rank)upper=middle;else lower=middle;
  }
  const float median=upper;float worst=minimum_variance,scatter=0;int inliers=0;unsigned int coverage=0;
  for(int i=s.first();i>=0;i=s.next(i)) {
    const float h=s.height(i),v=s.variance(i),d=h-median;
    if(!std::isfinite(h)||!std::isfinite(v)||v<0||fabsf(d)>.04f)continue;
    worst=fmaxf(worst,v);scatter+=d*d;++inliers;coverage|=s.coverage(i);
  }
  return {median,worst+(inliers?scatter/inliers:0),count,inliers,coverage};
}

// A scan median lives at the sampled XY positions, not necessarily at the
// grid centre. On a grade, changing beam phase can shift that median by
// several centimetres even though the surface has not moved. When an entire
// well-spread scan supports one plane, evaluate that plane at the cell centre.
// Otherwise keep the robust median, including for edges and multiple surfaces.
// x/y are cell fractions relative to its centre, in [-.5,.5].
template<class Samples> TERRAIN_HD GroundScan fitGroundPlaneAtCentre(
    const Samples &s,float minimum_variance,GroundScan fallback) {
  if(!fallback.count)return fallback;
  float xx=0,xy=0,yy=0,x=0,y=0,xz=0,yz=0,z=0;
  float xmin=INFINITY,xmax=-INFINITY,ymin=INFINITY,ymax=-INFINITY;
  int n=0;unsigned int coverage=0;
  for(int i=s.first();i>=0;i=s.next(i)) {
    const float a=s.x(i),b=s.y(i),h=s.height(i),v=s.variance(i);
    if(!std::isfinite(a)||!std::isfinite(b)||!std::isfinite(h)||!std::isfinite(v)||v<0)return fallback;
    xx+=a*a;xy+=a*b;yy+=b*b;x+=a;y+=b;xz+=a*h;yz+=b*h;z+=h;++n;
    xmin=fminf(xmin,a);xmax=fmaxf(xmax,a);ymin=fminf(ymin,b);ymax=fmaxf(ymax,b);
    coverage|=s.coverage(i);
  }
  // Do not extrapolate a plane from a narrow row or one-sided cell edge.
  if(n<6||xmin>-.15f||xmax<.15f||ymin>-.15f||ymax<.15f)return fallback;
  const float mx=x/n,my=y/n,mz=z/n;
  const float a=xx-x*mx,b=xy-x*my,c=yy-y*my,det=a*c-b*b;
  if(det<=1e-4f)return fallback;
  const float rx=xz-x*mz,ry=yz-y*mz;
  const float sx=(c*rx-b*ry)/det,sy=(a*ry-b*rx)/det,centre=mz-sx*mx-sy*my;
  // The first row of the least-squares evaluation operator. L1 propagation
  // bounds arbitrary correlation between returns and never divides sensor
  // uncertainty by ray count.
  const float wx=(-c*mx+b*my)/det,wy=(b*mx-a*my)/det;
  float std_bound=0,residual=0,l1=0;
  for(int i=s.first();i>=0;i=s.next(i)) {
    const float weight=1.f/n+wx*(s.x(i)-mx)+wy*(s.y(i)-my);
    l1+=fabsf(weight);
    std_bound+=fabsf(weight)*sqrtf(fmaxf(minimum_variance,s.variance(i)));
    residual=fmaxf(residual,fabsf(s.height(i)-centre-sx*s.x(i)-sy*s.y(i)));
  }
  // Millimetre planarity tolerance is a conservative qualification gate,
  // not an instruction to smooth a step into a ramp. No return is discarded.
  if(residual>.008f||l1>1.5f||!std::isfinite(centre))return fallback;
  return {centre,fmaxf(minimum_variance,std_bound*std_bound+residual*residual),n,n,coverage,true};
}

template<class Samples> TERRAIN_HD GroundScan centeredGroundScan(const Samples &s,float minimum_variance) {
  return fitGroundPlaneAtCentre(s,minimum_variance,robustGroundScan(s,minimum_variance));
}

// Sparse tilted scans often have ONE return per cell: no local plane exists,
// and changing its XY phase biases a grade's median height. A small current
// scan patch can supply the slope/centre interpolation, but cannot create an
// observation in an empty cell, add confirmation scans or fabricate coverage.
// ALL patch returns must satisfy the same planarity and L1 uncertainty gates.
template<class Own,class Patch> TERRAIN_HD GroundScan patchCenteredGroundScan(
    const Own &own,const Patch &patch,float minimum_variance) {
  const auto local=centeredGroundScan(own,minimum_variance);
  if(!local.count||local.centered)return local;
  auto result=fitGroundPlaneAtCentre(patch,minimum_variance,local);
  result.count=local.count;result.inliers=local.inliers;result.coverage=local.coverage;
  return result;
}

// Ring-buffer wrapping is physical storage only. Opposite logical map edges
// are never neighboring terrain. k enumerates the 3x3 scan neighborhood.
TERRAIN_HD inline int groundNeighborhoodCell(int cell,int k,int width,int height,int start_x,int start_y,int side) {
  const int lx=(cell%width-start_x+width)%width,ly=(cell/width-start_y+height)%height;
  const int x=lx+k%side-side/2,y=ly+k/side-side/2;
  if(x<0||y<0||x>=width||y>=height)return -1;
  return ((y+start_y)%height)*width+(x+start_x)%width;
}
TERRAIN_HD inline int groundPatchCell(int cell,int k,int width,int height,int start_x,int start_y) {
  return groundNeighborhoodCell(cell,k,width,height,start_x,start_y,3);
}
TERRAIN_HD inline int groundPatchDelta(int other,int centre,int size) {
  int delta=other-centre;
  if(delta>size/2)delta-=size;
  if(delta<-size/2)delta+=size;
  return delta;
}

// A band of CURRENT real returns, used only after independently observed
// lower terrain encloses the query. Upper objects remain in raw collision
// columns. No empty query, neighbor height or prediction becomes a return.
template<class Samples> struct GroundBandSamples {
  Samples source;float floor;
  TERRAIN_HD int seek(int i)const {
    while(i>=0) {
      const float h=source.height(i),v=source.variance(i);
      if(std::isfinite(h)&&std::isfinite(v)&&v>=0&&fabsf(h-floor)<=.04f)return i;
      i=source.next(i);
    }
    return -1;
  }
  TERRAIN_HD int first()const{return seek(source.first());}
  TERRAIN_HD int next(int i)const{return seek(source.next(i));}
  TERRAIN_HD float height(int i)const{return source.height(i);}
  TERRAIN_HD float variance(int i)const{return source.variance(i);}
  TERRAIN_HD unsigned int coverage(int i)const{return source.coverage(i);}
  TERRAIN_HD float x(int i)const{return source.x(i);}
  TERRAIN_HD float y(int i)const{return source.y(i);}
};

// References are an immutable, already MEASURED 9x9 neighborhood. This is a
// deliberately near-horizontal recovery gate, not a general ground fill.
// A new pit differs from its surroundings. Steps and low roughness returns
// must not be removed by selecting only the lower of multiple surfaces.
template<class Own,class Patch,class References> TERRAIN_HD bool reobservedGroundPlane(
    const Own &own,const Patch &patch,const References &references,
    float minimum_variance,GroundScan &scan) {
  float floor=INFINITY;
  for(int i=own.first();i>=0;i=own.next(i))
    if(std::isfinite(own.height(i))&&std::isfinite(own.variance(i))&&own.variance(i)>=0)
      floor=fminf(floor,own.height(i));
  if(!std::isfinite(floor))return false;
  int count=0;unsigned int quadrants=0;
  for(int k=0;k<references.size();++k) {
    const auto n=references.state(k);const int dx=references.dx(k),dy=references.dy(k);
    if((!dx&&!dy)||!n.observed||n.pending_count||!std::isfinite(n.height)||
        !std::isfinite(n.variance)||n.variance<0||n.variance>.0004f||fabsf(n.height-floor)>.03f)continue;
    ++count;
    if(dx&&dy)quadrants|=1U<<((dx>0?1:0)+(dy>0?2:0));
  }
  int distinct=0;for(;quadrants;quadrants&=quadrants-1)++distinct;
  if(count<6||distinct<4)return false;
  for(int i=patch.first();i>=0;i=patch.next(i)) {
    const float h=patch.height(i),v=patch.variance(i),delta=h-floor;
    if(!std::isfinite(h)||!std::isfinite(v)||v<0||delta<-.04f||
       (delta>.04f&&delta<.32f))return false;
  }
  const GroundBandSamples<Own> lower_own{own,floor};
  const GroundBandSamples<Patch> lower_patch{patch,floor};
  const auto qualified=patchCenteredGroundScan(lower_own,lower_patch,minimum_variance);
  if(!qualified.count||!qualified.centered||!std::isfinite(qualified.height)||
      !std::isfinite(qualified.variance)||qualified.variance>.0004f||fabsf(qualified.height-floor)>.03f)return false;
  scan=qualified;return true;
}

TERRAIN_HD inline bool spatialGroundEvidence(GroundScan scan) {
  unsigned int bits=scan.coverage;int distinct=0;
  for(;bits;bits&=bits-1)++distinct;
  return distinct>=3 && scan.inliers>=3 && 2*scan.inliers>scan.count && scan.variance<=.0004f;
}

// Sparse scans need not revisit the same cell three times to distinguish an
// OLD obstacle side from a CURRENT real floor return. A pre-update measured
// flat enclosure supplies corroboration, not synthetic samples or a height.
// Keep the actual own-cell median, variance, count and centred provenance.
// A pit/small ledge disagrees with the enclosing surface and cannot pass.
template<class Own,class References> TERRAIN_HD bool reobservedGroundReturn(
    const GroundState &old,const Own &own,const References &references,
    float minimum_variance,GroundScan &scan) {
  if(!old.observed||!std::isfinite(old.height)||!std::isfinite(old.variance)||old.variance<0)return false;
  const auto measured=robustGroundScan(own,minimum_variance);
  if(!measured.count||measured.inliers!=measured.count||!std::isfinite(measured.height)||
     !std::isfinite(measured.variance)||measured.variance>.0004f||
     old.height-measured.height<=fmaxf(.05f,3*sqrtf(old.variance+measured.variance)))return false;
  int reliable=0,count=0;float low=INFINITY,high=-INFINITY;unsigned int quadrants=0;
  for(int k=0;k<references.size();++k) {
    const int dx=references.dx(k),dy=references.dy(k);const auto n=references.state(k);
    if((!dx&&!dy)||!n.observed||n.pending_count||!std::isfinite(n.height)||
       !std::isfinite(n.variance)||n.variance<0||n.variance>.0004f)continue;
    ++reliable;
    if(fabsf(n.height-measured.height)>.025f)continue;
    ++count;low=fminf(low,n.height);high=fmaxf(high,n.height);
    if(dx&&dy)quadrants|=1U<<((dx>0?1:0)+(dy>0?2:0));
  }
  if(count<6||2*count<=reliable||quadrants!=15U||high-low>.03f)return false;
  for(int i=own.first();i>=0;i=own.next(i))
    if(!std::isfinite(own.height(i))||fabsf(own.height(i)-measured.height)>.015f)return false;
  scan=measured;return true;
}

// Retire an old elevated return only when a REAL return in this cell belongs
// to a fully qualified current scan plane, enclosed by already observed lower
// terrain. A new pit disagrees with that surrounding terrain and fails. Empty,
// one-sided, pending or uncertain neighbors cannot create this evidence.
// The nine neighbors must be an immutable PRE-UPDATE snapshot (centre at 4).
TERRAIN_HD inline bool enclosedLowerGround(const GroundState &old,const GroundScan &scan,
                                         const GroundState *neighbors) {
  if(!old.observed || !std::isfinite(old.height) || !scan.count || !scan.centered || !std::isfinite(scan.height) ||
     !std::isfinite(scan.variance) || scan.variance<0 || scan.variance>.0004f ||
     !std::isfinite(old.variance) || old.variance<0 ||
     old.height-scan.height<fmaxf(.05f,3*sqrtf(old.variance+scan.variance)))return false;
  unsigned int quadrants=0;int count=0;
  for(int k=0;k<9;++k) {
    if(k==4)continue;
    const auto &n=neighbors[k];
    if(!n.observed || n.pending_count || !std::isfinite(n.height) ||
       !std::isfinite(n.variance) || n.variance<0 || n.variance>.0004f || fabsf(n.height-scan.height)>.03f)continue;
    const int dx=k%3-1,dy=k/3-1;
    quadrants|=1U<<((dx>=0?1:0)+(dy>=0?2:0));++count;
  }
  int distinct=0;for(;quadrants;quadrants&=quadrants-1)++distinct;
  return count>=4 && distinct>=3;
}

// Free rays disprove an OLD elevated label without measuring the floor below.
// This reference is for UNKNOWN display/inpainting only, never a GroundScan.
// Four quadrants and a strict majority of reliable pre-update sources must
// agree on a low flat band; one-sided/uncertain sources cannot retire a label.
TERRAIN_HD inline bool freeRayIntersectsGround(const GroundState &old,float height) {
  return old.observed && std::isfinite(old.height) && std::isfinite(height) &&
      std::isfinite(old.variance) && old.variance>=0 &&
      fabsf(old.height-height)<=fminf(.10f,fmaxf(.03f,3*sqrtf(old.variance)));
}
template<class References> TERRAIN_HD bool freeSpaceGroundReference(
    const GroundState &old,const References &references,int free_scans,
    int required_scans,float minimum_rise,float &reference) {
  if(!old.observed || free_scans<required_scans || required_scans<1 ||
     !std::isfinite(minimum_rise) || minimum_rise<=0 ||
     !std::isfinite(old.height) || !std::isfinite(old.variance) ||
     old.variance<0)return false;
  float low=INFINITY;int reliable=0;
  for(int k=0;k<references.size();++k) {
    const auto n=references.state(k);
    if((!references.dx(k)&&!references.dy(k))||!n.observed||n.pending_count||
       !std::isfinite(n.height)||!std::isfinite(n.variance)||n.variance<0||n.variance>.0004f)continue;
    low=fminf(low,n.height);++reliable;
  }
  if(!std::isfinite(low))return false;
  float sum=0,variance=0;int count=0;unsigned int quadrants=0;
  for(int k=0;k<references.size();++k) {
    const auto n=references.state(k);const int dx=references.dx(k),dy=references.dy(k);
    if((!dx&&!dy)||!n.observed||n.pending_count||!std::isfinite(n.height)||
       !std::isfinite(n.variance)||n.variance<0||n.variance>.0004f||n.height-low>.03f)continue;
    sum+=n.height;variance=fmaxf(variance,n.variance);++count;
    if(dx&&dy)quadrants|=1U<<((dx>0?1:0)+(dy>0?2:0));
  }
  if(count<6 || 2*count<=reliable || quadrants!=15U)return false;
  const float lower=sum/count;
  if(old.height-lower<=minimum_rise+3*sqrtf(old.variance+variance))return false;
  reference=lower;return true;
}

// Weak innovations need three consistent scans. A coherent median supported
// by three distinct XY subcells establishes ONE geometric observation; its
// variance is not reduced by ray count. Duplicate rays cannot create coverage.
// Unconfirmed support remains pending/unknown.
// Raw points must still enter collision/pit checks without this delay.
TERRAIN_HD inline GroundState fuseGroundScan(GroundState old,GroundScan scan,
    float min_variance,float mahalanobis,bool preserve_lower,float recovery_max_rise,
    bool enclosed_lower_evidence=false) {
  if(!scan.count)return old;
  scan.variance=fmaxf(min_variance,scan.variance);
  if(!old.observed) {
    old.height=scan.height;old.variance=scan.variance;old.observed=true;old.pending_count=0;
    return old;
  }
  const float delta=scan.height-old.height;
  if(enclosed_lower_evidence && delta<-.05f) {
    old.height=scan.height;old.variance=scan.variance;old.pending_count=0;return old;
  }
  if(preserve_lower && delta>fmaxf(.05f,recovery_max_rise)) {
    // A high object does not replace a measured lower floor. Its returns
    // remain in the independent current collision column layer.
    old.pending_count=0;return old;
  }
  const float gate=fmaxf(.025f,mahalanobis*sqrtf(fmaxf(old.variance,min_variance)+scan.variance));
  if(fabsf(delta)>gate) {
    if(fabsf(delta)<=recovery_max_rise && spatialGroundEvidence(scan)) {
      old.height=scan.height;old.variance=scan.variance;old.pending_count=0;return old;
    }
    const float agreement=fmaxf(.02f,2*sqrtf(scan.variance));
    if(!old.pending_count||fabsf(scan.height-old.pending_height)>agreement) {
      old.pending_height=scan.height;old.pending_variance=scan.variance;old.pending_count=1;
    } else {
      ++old.pending_count;
      old.pending_height+=(scan.height-old.pending_height)/old.pending_count;
      old.pending_variance=fmaxf(old.pending_variance,scan.variance);
    }
    if(old.pending_count>=3) {
      old.height=old.pending_height;old.variance=old.pending_variance;old.pending_count=0;
    }
    return old;
  }
  old.pending_count=0;
  const float variance=fmaxf(old.variance,min_variance);
  old.height=(variance*scan.height+scan.variance*old.height)/(variance+scan.variance);
  old.variance=fmaxf(min_variance,variance*scan.variance/(variance+scan.variance));
  return old;
}
} // namespace terrain_analyzer
#undef TERRAIN_HD
