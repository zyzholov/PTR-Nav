#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <array>
#include <cmath>
#include <deque>
#include <stdexcept>

namespace fast_lio {
// IMU-only rest check. Constant translational acceleration remains
// unobservable; the vehicle must be at rest during initialization.
class QuietImuWindow {
  struct Sample {double time;std::array<double,3> acc,gyr;};
  std::deque<Sample> samples_;
  double duration_;
public:
  explicit QuietImuWindow(double duration=1.):duration_(duration) {
    if(!std::isfinite(duration) || duration<=0 || duration>5.)
      throw std::invalid_argument("IMU rest duration must be in (0,5] seconds");
  }
  void add(double time,const Eigen::Vector3d &acc,const Eigen::Vector3d &gyr) {
    if(!std::isfinite(time) || !acc.allFinite() || !gyr.allFinite() || acc.norm()<1.) {samples_.clear();return;}
    if(!samples_.empty()) {
      if(time==samples_.back().time)return;
      if(time<samples_.back().time || time-samples_.back().time>.05)samples_.clear();
    }
    samples_.push_back({time,{acc.x(),acc.y(),acc.z()},{gyr.x(),gyr.y(),gyr.z()}});
    while(samples_.size()>1 && samples_[1].time<=time-duration_)samples_.pop_front();
    if(samples_.size()>4096)samples_.clear();
  }
  bool estimate(Eigen::Vector3d &acc,Eigen::Vector3d &gyr) const {
    if(samples_.size()<10 || samples_.back().time-samples_.front().time<duration_-1e-6)return false;
    acc.setZero();gyr.setZero();
    for(const auto &s:samples_) {acc+=Eigen::Map<const Eigen::Vector3d>(s.acc.data());gyr+=Eigen::Map<const Eigen::Vector3d>(s.gyr.data());}
    acc/=samples_.size();gyr/=samples_.size();
    if(std::abs(acc.norm()-9.81)>.5)return false;
    // Contact vibration can alternate strongly between consecutive IMU
    // samples while its sustained specific force is stable. Require five
    // temporal block means to agree; individual rays/samples do not become
    // independent confidence evidence. A gross transient is still rejected.
    std::array<Eigen::Vector3d,5> blocks;std::array<int,5> counts{};
    for(auto &b:blocks)b.setZero();
    for(const auto &s:samples_) {
      const Eigen::Map<const Eigen::Vector3d> a(s.acc.data()),g(s.gyr.data());
      if((a-acc).norm()>1.5 || g.norm()>.08)return false;
      const int bin=std::min(4,static_cast<int>(5*(s.time-samples_.front().time)/duration_));
      blocks[bin]+=a;++counts[bin];
    }
    for(int i=0;i<5;++i)if(!counts[i] || (blocks[i]/counts[i]-acc).norm()>.05)return false;
    return true;
  }
};

inline Eigen::Matrix3d gravityAlignment(const Eigen::Vector3d &specific_force) {
  if(!specific_force.allFinite() || specific_force.norm()<1.)
    throw std::invalid_argument("invalid initial specific force");
  return Eigen::Quaterniond::FromTwoVectors(specific_force.normalized(),Eigen::Vector3d::UnitZ()).toRotationMatrix();
}
} // namespace fast_lio
