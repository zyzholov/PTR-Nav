#pragma once
// Conditional Gaussian kinematics. Noise parameters are explicit MODEL
// assumptions, not calibrated real-human collision probabilities.
#include "terrain_nav/object_observation.hpp"
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

namespace terrain_nav {
struct GaussianMotion {
  bool valid=false;
  double stamp=0.,process_density=.04,observation_std=.03;
  double oldest_stamp=0.,longitudinal_acceleration_bound=0.;
  size_t observation_count=0;
  Eigen::Vector4d mean=Eigen::Vector4d::Zero(); // x,y,vx,vy, at source stamp.
  Eigen::Matrix4d covariance=Eigen::Matrix4d::Zero();
  bool constant_acceleration=false;
  Eigen::Matrix<double,6,1> acceleration_mean=Eigen::Matrix<double,6,1>::Zero();
  Eigen::Matrix<double,6,6> acceleration_covariance=Eigen::Matrix<double,6,6>::Zero();
};
struct GaussianPosition {
  bool valid=false;
  Eigen::Vector2d mean=Eigen::Vector2d::Zero();
  Eigen::Matrix2d covariance=Eigen::Matrix2d::Zero();
};
inline bool validGaussianMotion(const GaussianMotion &m)
{
  if(!m.valid || !std::isfinite(m.stamp) || !std::isfinite(m.process_density) || m.process_density<0 ||
     !std::isfinite(m.observation_std) || m.observation_std<=0 ||
     !std::isfinite(m.longitudinal_acceleration_bound)||m.longitudinal_acceleration_bound<0 ||
     !m.mean.allFinite() || !m.covariance.allFinite() ||
     (m.covariance-m.covariance.transpose()).cwiseAbs().maxCoeff()>1e-9)return false;
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> e(m.covariance);
  if(e.info()!=Eigen::Success||e.eigenvalues().minCoeff()< -1e-10)return false;
  if(m.constant_acceleration) {
    if(m.longitudinal_acceleration_bound!=0.||!m.acceleration_mean.allFinite()||!m.acceleration_covariance.allFinite()||
       (m.acceleration_covariance-m.acceleration_covariance.transpose()).cwiseAbs().maxCoeff()>1e-9||
       (m.mean-m.acceleration_mean.head<4>()).cwiseAbs().maxCoeff()>1e-10||
       (m.covariance-m.acceleration_covariance.topLeftCorner<4,4>()).cwiseAbs().maxCoeff()>1e-10)return false;
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,6,6>> a(m.acceleration_covariance);
    if(a.info()!=Eigen::Success||a.eigenvalues().minCoeff()< -1e-10)return false;
  }
  return true;
}
inline Eigen::Matrix4d gaussianMotionTransition(double t)
{
  Eigen::Matrix4d f=Eigen::Matrix4d::Identity();f(0,2)=f(1,3)=t;return f;
}
inline Eigen::Matrix4d gaussianMotionProcess(double t,double density)
{
  Eigen::Matrix4d q=Eigen::Matrix4d::Zero();
  for(int i=0;i<2;++i) {
    q(i,i)=density*t*t*t/3.;q(i,i+2)=q(i+2,i)=density*t*t/2.;q(i+2,i+2)=density*t;
  }
  return q;
}
// State x,y,vx,vy,ax,ay; continuous white jerk. Density has units m^2/s^5,
// unlike the CV white-acceleration density (m^2/s^3). They are separate models.
inline Eigen::Matrix<double,6,6> gaussianAccelerationTransition(double t)
{
  auto f=Eigen::Matrix<double,6,6>::Identity().eval();
  for(int k=0;k<2;++k){f(k,k+2)=f(k+2,k+4)=t;f(k,k+4)=.5*t*t;}
  return f;
}
inline Eigen::Matrix<double,6,6> gaussianJerkProcess(double t,double density)
{
  Eigen::Matrix<double,6,6> q=Eigen::Matrix<double,6,6>::Zero();
  for(int k=0;k<2;++k) {
    q(k,k)=density*std::pow(t,5)/20.;q(k,k+2)=q(k+2,k)=density*std::pow(t,4)/8.;
    q(k,k+4)=q(k+4,k)=density*t*t*t/6.;q(k+2,k+2)=density*t*t*t/3.;
    q(k+2,k+4)=q(k+4,k+2)=density*t*t/2.;q(k+4,k+4)=density*t;
  }
  return q;
}
inline GaussianMotion updateGaussianAccelerationMotion(const GaussianMotion &previous,
    const TimedCenter &observation,double observation_std,double innovation_limit)
{
  const double dt=observation.stamp_s-previous.stamp;
  const auto f=gaussianAccelerationTransition(dt);GaussianMotion out=previous;
  out.acceleration_mean=f*previous.acceleration_mean;
  out.acceleration_covariance=f*previous.acceleration_covariance*f.transpose()+gaussianJerkProcess(dt,previous.process_density);
  Eigen::Matrix<double,2,6> h=Eigen::Matrix<double,2,6>::Zero();h(0,0)=h(1,1)=1.;
  const Eigen::Matrix2d r=observation_std*observation_std*Eigen::Matrix2d::Identity();
  const Eigen::Matrix2d s=h*out.acceleration_covariance*h.transpose()+r;const auto factor=s.ldlt();
  if(factor.info()!=Eigen::Success||!factor.isPositive())return {};
  const Eigen::Vector2d innovation=Eigen::Vector2d(observation.center[0],observation.center[1])-h*out.acceleration_mean;
  if(!std::isfinite(innovation.squaredNorm())||innovation.dot(factor.solve(innovation))>innovation_limit)return {};
  const Eigen::Matrix<double,6,2> k=out.acceleration_covariance*h.transpose()*factor.solve(Eigen::Matrix2d::Identity());
  out.acceleration_mean+=k*innovation;
  const Eigen::Matrix<double,6,6> a=Eigen::Matrix<double,6,6>::Identity()-k*h;
  out.acceleration_covariance=a*out.acceleration_covariance*a.transpose()+k*r*k.transpose();
  out.acceleration_covariance=.5*(out.acceleration_covariance+out.acceleration_covariance.transpose()).eval();
  out.mean=out.acceleration_mean.head<4>();out.covariance=out.acceleration_covariance.topLeftCorner<4,4>();
  out.stamp=observation.stamp_s;out.observation_std=observation_std;++out.observation_count;
  return validGaussianMotion(out)?out:GaussianMotion{};
}
// A known identity consumes each NEW measured centre once. Re-fitting the
// overlapping three-scan window from a broad velocity prior on every scan
// discards evidence and makes the forecast jump during an executing escape.
// Keep the same CV process density; neither missed returns nor predictions
// are observations. An innovation inconsistent with that model rejects this
// update so the caller can restart from its independently fitted local window.
inline GaussianMotion updateGaussianCenterMotion(const GaussianMotion &previous,
    const TimedCenter &observation,double observation_std,double innovation_limit=13.815510557964274,
    double maximum_observation_gap=.400001)
{
  if(!validGaussianMotion(previous)||!std::isfinite(observation.stamp_s)||
     !std::isfinite(maximum_observation_gap)||maximum_observation_gap<=0||maximum_observation_gap>6.||
     observation.stamp_s<=previous.stamp||observation.stamp_s-previous.stamp>maximum_observation_gap||
     !std::isfinite(observation.center[0])||!std::isfinite(observation.center[1])||
     !std::isfinite(observation_std)||observation_std<=0||
     !std::isfinite(innovation_limit)||innovation_limit<=0)return {};
  if(previous.constant_acceleration)return updateGaussianAccelerationMotion(previous,observation,observation_std,innovation_limit);
  const auto f=gaussianMotionTransition(observation.stamp_s-previous.stamp);
  GaussianMotion result=previous;
  result.mean=f*previous.mean;
  result.covariance=f*previous.covariance*f.transpose()+
    gaussianMotionProcess(observation.stamp_s-previous.stamp,previous.process_density);
  Eigen::Matrix<double,2,4> h=Eigen::Matrix<double,2,4>::Zero();h(0,0)=h(1,1)=1.;
  const Eigen::Matrix2d r=observation_std*observation_std*Eigen::Matrix2d::Identity();
  const Eigen::Matrix2d s=h*result.covariance*h.transpose()+r;
  const auto factor=s.ldlt();if(factor.info()!=Eigen::Success||!factor.isPositive())return {};
  const Eigen::Vector2d innovation=Eigen::Vector2d(observation.center[0],observation.center[1])-h*result.mean;
  const double nis=innovation.dot(factor.solve(innovation));
  if(!std::isfinite(nis)||nis>innovation_limit)return {};
  const Eigen::Matrix<double,4,2> k=result.covariance*h.transpose()*factor.solve(Eigen::Matrix2d::Identity());
  result.mean+=k*innovation;
  const Eigen::Matrix4d a=Eigen::Matrix4d::Identity()-k*h;
  result.covariance=a*result.covariance*a.transpose()+k*r*k.transpose();
  result.covariance=.5*(result.covariance+result.covariance.transpose()).eval();
  result.stamp=observation.stamp_s;result.observation_std=observation_std;
  ++result.observation_count;
  return validGaussianMotion(result)?result:GaussianMotion{};
}
// Consume only actual, previously unused measured centres in time order.
// Across an occlusion, one position can fit a broad prior while implying a
// velocity contradicted by the next actual returns. The complete causal
// window supplies that check without inventing an independent velocity
// measurement from the SAME positions or double-counting old observations.
inline GaussianMotion updateGaussianCenterWindow(const GaussianMotion &previous,
    const std::vector<TimedCenter> &history,double observation_std,
    double innovation_limit=13.815510557964274,double maximum_observation_gap=.400001,
    size_t minimum_new_observations=1)
{
  if(!validGaussianMotion(previous)||history.empty()||history.size()>32||
     !minimum_new_observations||minimum_new_observations>32)return {};
  auto result=previous;size_t consumed=0;
  for(size_t i=0;i<history.size();++i) {
    const auto &s=history[i];
    if(!std::isfinite(s.stamp_s)||!std::isfinite(s.center[0])||!std::isfinite(s.center[1])||
       !std::isfinite(s.radius)||s.radius<.15||s.radius>1.5||
       !std::isfinite(s.resolution)||s.resolution<=0||
       (i&&(s.stamp_s<=history[i-1].stamp_s||s.resolution!=history[0].resolution)))return {};
    if(s.stamp_s<=previous.stamp)continue;
    result=updateGaussianCenterMotion(result,s,observation_std,innovation_limit,maximum_observation_gap);
    if(!result.valid)return {};
    ++consumed;
  }
  return consumed>=minimum_new_observations?result:GaussianMotion{};
}

// Independent position observation noise and continuous white acceleration.
// A finite causal history initializes a CV Kalman posterior. Invalid history
// is rejected, never silently skipped or filled by future/predicted samples.
inline GaussianMotion fitGaussianCenterMotion(const std::vector<TimedCenter> &history,
    double observation_std=.03,double process_density=.04,size_t maximum_observations=4)
{
  GaussianMotion result;
  if(history.size()<3 || history.size()>32 || !std::isfinite(observation_std) || observation_std<=0 ||
     !std::isfinite(process_density) || process_density<0||maximum_observations<3||maximum_observations>32)return result;
  const double span=history.back().stamp_s-history.front().stamp_s;
  if(span<.15-1e-6 || span>.4+1e-6)return result;
  double radius_low=INFINITY,radius_high=-INFINITY;
  for(size_t i=0;i<history.size();++i) {
    const auto &s=history[i];
    if(!std::isfinite(s.stamp_s) || !std::isfinite(s.center[0]) || !std::isfinite(s.center[1]) ||
       !std::isfinite(s.radius) || s.radius<.15 || s.radius>1.5 ||
       !std::isfinite(s.resolution) || s.resolution<=0 ||
       (i && (s.stamp_s<=history[i-1].stamp_s || s.resolution!=history[0].resolution)))return result;
    radius_low=std::min(radius_low,s.radius);radius_high=std::max(radius_high,s.radius);
    if(radius_high-radius_low>2*s.resolution)return result;
  }
  const double variance=observation_std*observation_std;
  const size_t begin=history.size()>maximum_observations?history.size()-maximum_observations:0;
  if(history.back().stamp_s-history[begin].stamp_s<.15-1e-6)return result;
  result.mean<<history[begin].center[0],history[begin].center[1],0.,0.;
  // First measured position is consumed once; broad independent velocity prior.
  result.covariance.diagonal()<<variance,variance,4.5*4.5,4.5*4.5;
  Eigen::Matrix<double,2,4> h=Eigen::Matrix<double,2,4>::Zero();h(0,0)=h(1,1)=1.;
  const Eigen::Matrix2d r=variance*Eigen::Matrix2d::Identity();
  for(size_t i=begin+1;i<history.size();++i) {
    const double dt=history[i].stamp_s-history[i-1].stamp_s;
    const auto f=gaussianMotionTransition(dt);
    result.mean=f*result.mean;
    result.covariance=f*result.covariance*f.transpose()+gaussianMotionProcess(dt,process_density);
    const Eigen::Matrix2d s=h*result.covariance*h.transpose()+r;
    const auto factor=s.ldlt();if(factor.info()!=Eigen::Success || !factor.isPositive())return GaussianMotion{};
    const Eigen::Matrix<double,4,2> k=result.covariance*h.transpose()*factor.solve(Eigen::Matrix2d::Identity());
    const Eigen::Vector2d measurement(history[i].center[0],history[i].center[1]);
    result.mean+=k*(measurement-h*result.mean);
    const Eigen::Matrix4d a=Eigen::Matrix4d::Identity()-k*h;
    result.covariance=a*result.covariance*a.transpose()+k*r*k.transpose(); // Joseph form.
    result.covariance=.5*(result.covariance+result.covariance.transpose()).eval();
  }
  result.stamp=history.back().stamp_s;result.process_density=process_density;
  result.observation_std=observation_std;result.valid=true;
  result.oldest_stamp=history[begin].stamp_s;result.observation_count=history.size()-begin;
  if(!validGaussianMotion(result))return GaussianMotion{};
  return result;
}
inline GaussianMotion fitGaussianAccelerationCenterMotion(const std::vector<TimedCenter> &history,
    double observation_std=.03,double jerk_density=.04,size_t maximum_observations=4)
{
  // Reuse the CV initializer's full history/shape admission, NOT its posterior.
  // Each original measured position is then consumed once by this CA model.
  if(!fitGaussianCenterMotion(history,observation_std,jerk_density,maximum_observations).valid)return {};
  const size_t begin=history.size()>maximum_observations?history.size()-maximum_observations:0;
  GaussianMotion out;out.valid=out.constant_acceleration=true;out.process_density=jerk_density;
  out.observation_std=observation_std;out.stamp=out.oldest_stamp=history[begin].stamp_s;out.observation_count=1;
  out.acceleration_mean<<history[begin].center[0],history[begin].center[1],0.,0.,0.,0.;
  out.acceleration_covariance.diagonal()<<observation_std*observation_std,observation_std*observation_std,
    4.5*4.5,4.5*4.5,1.2*1.2,1.2*1.2;
  out.mean=out.acceleration_mean.head<4>();out.covariance=out.acceleration_covariance.topLeftCorner<4,4>();
  for(size_t i=begin+1;i<history.size();++i) {
    out=updateGaussianCenterMotion(out,history[i],observation_std);
    if(!out.valid)return {};
  }
  return out;
}
namespace gaussian_motion_detail {
inline GaussianPosition predictValidated(const GaussianMotion &m,double t)
{
  GaussianPosition out;
  if(!std::isfinite(t)||t<0||t>10)return out;
  if(m.constant_acceleration) {
    const auto f=gaussianAccelerationTransition(t);
    out.mean=(f*m.acceleration_mean).head<2>();
    out.covariance=(f*m.acceleration_covariance*f.transpose()+gaussianJerkProcess(t,m.process_density)).topLeftCorner<2,2>();
    out.valid=out.mean.allFinite()&&out.covariance.allFinite();return out;
  }
  const auto f=gaussianMotionTransition(t);
  out.mean=(f*m.mean).head<2>();
  out.covariance=(f*m.covariance*f.transpose()+gaussianMotionProcess(t,m.process_density)).topLeftCorner<2,2>();
  out.valid=out.mean.allFinite()&&out.covariance.allFinite();return out;
}
}
inline GaussianPosition predictGaussianPosition(const GaussianMotion &m,double t)
{
  return validGaussianMotion(m)?gaussian_motion_detail::predictValidated(m,t):GaussianPosition{};
}

// Collision with a rectangle Minkowski-summed with the physical object disk
// implies collision with ANY supporting halfspace. The minimum Gaussian
// one-sided tail over these valid halfspaces is therefore an UPPER bound.
// No x/y or inter-time independence assumption is made.
inline double gaussianBodyCollisionUpper(double x,double y,double yaw,double front,double back,double half_width,
    double radius,const GaussianPosition &p)
{
  if(!p.valid || !p.mean.allFinite() || !p.covariance.allFinite() ||
     !std::isfinite(x)||!std::isfinite(y)||!std::isfinite(yaw)||!std::isfinite(front)||front<0||
     !std::isfinite(back)||back<0||!std::isfinite(half_width)||half_width<0||!std::isfinite(radius)||radius<0 ||
     (p.covariance-p.covariance.transpose()).cwiseAbs().maxCoeff()>1e-9)return 1.;
  const double c=std::cos(yaw),s=std::sin(yaw);
  Eigen::Matrix2d r;r<<c,s,-s,c;
  const Eigen::Vector2d mean=r*(p.mean-Eigen::Vector2d(x,y));
  const Eigen::Matrix2d cov=r*p.covariance*r.transpose();
  const double minimum=.5*(cov.trace()-std::hypot(cov(0,0)-cov(1,1),2*cov(0,1)));
  if(minimum< -1e-9)return 1.;
  double upper=1.;
  const auto axis=[&](const Eigen::Vector2d &u) {
    const double support=(u.x()>=0?front:back)*std::abs(u.x())+half_width*std::abs(u.y());
    const double distance=u.dot(mean)-support-radius;
    const double variance=u.dot(cov*u);
    if(!std::isfinite(variance)||variance< -1e-9)return;
    const double tail=variance<=1e-18?(distance>0?0.:1.):.5*std::erfc(distance/std::sqrt(2*variance));
    upper=std::min(upper,tail);
  };
  axis(Eigen::Vector2d(1,0));axis(Eigen::Vector2d(-1,0));
  axis(Eigen::Vector2d(0,1));axis(Eigen::Vector2d(0,-1));
  const Eigen::Vector2d closest(std::clamp(mean.x(),-back,front),std::clamp(mean.y(),-half_width,half_width));
  const auto normal=mean-closest;if(normal.norm()>1e-12)axis(normal/normal.norm());
  return std::clamp(upper,0.,1.);
}
} // namespace terrain_nav
