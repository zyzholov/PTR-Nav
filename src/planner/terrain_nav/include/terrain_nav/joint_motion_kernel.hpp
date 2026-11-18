#pragma once
#include <cmath>
#include <cstdint>
#include <cstddef>
#ifdef __CUDACC__
#define TERRAIN_MOTION_HD __host__ __device__
#else
#define TERRAIN_MOTION_HD
#endif
namespace terrain_nav::joint_detail {
struct Actor {
  double mean[4],root[16],age,density,radius,acceleration,maximum_speed;
  unsigned analytical_only=0,constant_acceleration=0;
  double acceleration_mean[6],acceleration_root[36];
};
struct Pose {double t,x,y,c,s,front,back,width,sweep;};
struct Segment {double ax,ay,bx,by;};
// Counter-seeded independent sample streams, identical CPU/CUDA algorithm.
// Arithmetic is double; numerical parity is checked separately from the
// statistical model's validity. This is a PRNG, not external actor truth.
TERRAIN_MOTION_HD inline uint64_t randomWord(uint64_t &state)
{
  uint64_t z=(state+=0x9e3779b97f4a7c15ULL);
  z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;z=(z^(z>>27))*0x94d049bb133111ebULL;return z^(z>>31);
}
TERRAIN_MOTION_HD inline void normalPair(uint64_t &rng,double &a,double &b)
{
  const double u=(double(randomWord(rng)>>11)+.5)*0x1.0p-53;
  const double v=(double(randomWord(rng)>>11)+.5)*0x1.0p-53;
  const double r=::sqrt(-2.*::log(u)),angle=6.2831853071795864769*v;
  a=r*::cos(angle);b=r*::sin(angle);
}
TERRAIN_MOTION_HD inline void propagate(double *state,double dt,double density,uint64_t &rng,bool ca=false)
{
  if(dt<=0)return;
  if(ca) {
    const double noise=::sqrt(density*dt);
    for(int axis=0;axis<2;++axis) {
      double z0=0,z1=0,z2=0,unused=0;
      if(density>0){normalPair(rng,z0,z1);normalPair(rng,z2,unused);}
      state[axis]+=state[axis+2]*dt+.5*state[axis+4]*dt*dt+noise*dt*dt*.22360679774997896964*z0;
      state[axis+2]+=state[axis+4]*dt+noise*dt*(.55901699437494742410*z0+.14433756729740644113*z1);
      state[axis+4]+=noise*(.74535599249992989880*z0+.57735026918962576451*z1+z2/3.);
    }
    return;
  }
  if(density==0){state[0]+=state[2]*dt;state[1]+=state[3]*dt;return;}
  const double v=::sqrt(density*dt),p=v*dt;
  for(int axis=0;axis<2;++axis) {
    double a,b;normalPair(rng,a,b);
    state[axis]+=state[axis+2]*dt+p*(.5*a+.28867513459481288225*b);
    state[axis+2]+=v*a;
  }
}
TERRAIN_MOTION_HD inline uint64_t initialState(const Actor &actor,uint64_t seed,uint64_t sample,
    uint64_t object,double *state)
{
  uint64_t rng=seed^(0xd1b54a32d192ed03ULL*(sample+1))^(0x94d049bb133111ebULL*(object+1));
  double z[6];normalPair(rng,z[0],z[1]);normalPair(rng,z[2],z[3]);
  if(actor.constant_acceleration) {
    normalPair(rng,z[4],z[5]);
    for(int i=0;i<6;++i){state[i]=actor.acceleration_mean[i];for(int j=0;j<6;++j)state[i]+=actor.acceleration_root[6*i+j]*z[j];}
  } else for(int i=0;i<4;++i){state[i]=actor.mean[i];for(int j=0;j<4;++j)state[i]+=actor.root[4*i+j]*z[j];}
  propagate(state,actor.age,actor.density,rng,actor.constant_acceleration);return rng;
}
TERRAIN_MOTION_HD inline Segment sampledSegment(const Actor &actor,const double *state,double future)
{
  const double speed=::hypot(actor.mean[2],actor.mean[3]);
  if(actor.acceleration<=0 || speed<=1e-9)return {state[0],state[1],state[0],state[1]};
  // Bounded mean acceleration and the SAME correlated Gaussian residual.
  const double t=actor.age+future,a=actor.acceleration;
  // Signed longitudinal acceleration may cross zero and reverse. Clamping
  // at rest under-approximated the declared envelope in turning encounters.
  const double reverse=t<(speed+actor.maximum_speed)/a?t:(speed+actor.maximum_speed)/a;
  const double minimum=speed*reverse-.5*a*reverse*reverse-actor.maximum_speed*(t-reverse);
  const double accelerate=t<(actor.maximum_speed-speed)/a?t:(actor.maximum_speed-speed)/a;
  const double maximum=speed*accelerate+.5*a*accelerate*accelerate+actor.maximum_speed*(t-accelerate);
  const double ux=actor.mean[2]/speed,uy=actor.mean[3]/speed;
  return {state[0]+ux*(minimum-speed*t),state[1]+uy*(minimum-speed*t),
          state[0]+ux*(maximum-speed*t),state[1]+uy*(maximum-speed*t)};
}
TERRAIN_MOTION_HD inline double meanDepartureRadius(const Actor &actor,double future)
{
  // At zero conditional mean velocity there is no observable longitudinal
  // direction. A bounded mean departure can point anywhere, so cover its
  // reachable disk rather than treating rest as a promise to remain still.
  if(actor.acceleration<=0 || ::hypot(actor.mean[2],actor.mean[3])>1e-9)return 0.;
  const double t=actor.age+future;
  const double accelerate=t<actor.maximum_speed/actor.acceleration?t:actor.maximum_speed/actor.acceleration;
  return .5*actor.acceleration*accelerate*accelerate+actor.maximum_speed*(t-accelerate);
}
TERRAIN_MOTION_HD inline bool collision(double x,double y,const Pose &p,double radius)
{
  const double dx=x-p.x,dy=y-p.y;
  const double u=p.c*dx+p.s*dy,v=-p.s*dx+p.c*dy;
  const double ax=u>p.front?u-p.front:(u< -p.back?-p.back-u:0.);
  const double ay=::fabs(v)>p.width?::fabs(v)-p.width:0.;
  const double r=radius+p.sweep;return ax*ax+ay*ay<=r*r;
}
TERRAIN_MOTION_HD inline double pointSegmentSquared(double x,double y,double ax,double ay,double bx,double by)
{
  const double dx=bx-ax,dy=by-ay,d=dx*dx+dy*dy;
  double t=d>0?((x-ax)*dx+(y-ay)*dy)/d:0.;t=t<0?0:(t>1?1:t);
  const double ex=x-ax-t*dx,ey=y-ay-t*dy;return ex*ex+ey*ey;
}
TERRAIN_MOTION_HD inline bool collisionSegment(double ax,double ay,double bx,double by,const Pose &p,double radius)
{
  const double adx=ax-p.x,ady=ay-p.y,bdx=bx-p.x,bdy=by-p.y;
  const double a[2]={p.c*adx+p.s*ady,-p.s*adx+p.c*ady},b[2]={p.c*bdx+p.s*bdy,-p.s*bdx+p.c*bdy};
  const double low[2]={-p.back,-p.width},high[2]={p.front,p.width};
  double lo=0.,hi=1.;bool intersects=true;
  for(int k=0;k<2;++k) {
    const double d=b[k]-a[k];
    if(::fabs(d)<1e-15){if(a[k]<low[k]||a[k]>high[k])intersects=false;}
    else {
      double x=(low[k]-a[k])/d,y=(high[k]-a[k])/d;
      if(x>y){const double tmp=x;x=y;y=tmp;}
      lo=lo>x?lo:x;hi=hi<y?hi:y;if(lo>hi)intersects=false;
    }
  }
  if(intersects)return true;
  double distance=1e300;
  for(int endpoint=0;endpoint<2;++endpoint) {
    const double x=endpoint?b[0]:a[0],y=endpoint?b[1]:a[1];
    const double dx=x<low[0]?low[0]-x:(x>high[0]?x-high[0]:0.);
    const double dy=y<low[1]?low[1]-y:(y>high[1]?y-high[1]:0.);
    const double d=dx*dx+dy*dy;distance=distance<d?distance:d;
  }
  for(int ix=0;ix<2;++ix)for(int iy=0;iy<2;++iy) {
    const double d=pointSegmentSquared(ix?high[0]:low[0],iy?high[1]:low[1],a[0],a[1],b[0],b[1]);
    distance=distance<d?distance:d;
  }
  const double r=radius+p.sweep;return distance<=r*r;
}
TERRAIN_MOTION_HD inline bool trajectoryHit(const Actor &actor,const Pose *poses,int count,
    uint64_t seed,uint64_t sample,uint64_t object)
{
  if(actor.analytical_only)return false; // Its analytic event bound is accounted on the host.
  double state[6];uint64_t rng=initialState(actor,seed,sample,object,state);
  double previous=0.;
  for(int k=0;k<count;++k) {
    propagate(state,poses[k].t-previous,actor.density,rng,actor.constant_acceleration);previous=poses[k].t;
    if(actor.acceleration>0) {
      // A bounded mean longitudinal acceleration is NOT white Gaussian
      // process noise. Keep the entire physically possible FUTURE distance
      // segment, including bounded reversal; do not pin its rear to the source
      // position forever. The same correlated Gaussian residual shifts both
      // endpoints. Heading changes remain outside this conditional model.
      const auto s=sampledSegment(actor,state,poses[k].t);
      if(collisionSegment(s.ax,s.ay,s.bx,s.by,poses[k],
          actor.radius+meanDepartureRadius(actor,poses[k].t)))return true;
    } else if(collision(state[0],state[1],poses[k],actor.radius))return true;
  }
  return false;
}
TERRAIN_MOTION_HD inline bool sharedTrajectoryHit(const Actor &actor,const Pose *poses,
    const int *indices,int count,const Segment *forecast,size_t samples,size_t sample)
{
  for(int k=0;k<count;++k) {
    const auto &s=forecast[size_t(indices[k])*samples+sample];
    if(actor.acceleration>0 ? collisionSegment(s.ax,s.ay,s.bx,s.by,poses[k],
                               actor.radius+meanDepartureRadius(actor,poses[k].t)) :
                             collision(s.ax,s.ay,poses[k],actor.radius))return true;
  }
  return false;
}
} // namespace terrain_nav::joint_detail
#undef TERRAIN_MOTION_HD
