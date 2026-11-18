#include "imu_initialization.hpp"
#include <cassert>
#include <limits>

int main() {
  using Eigen::Vector3d;
  const Vector3d g(0,0,9.81),zero=Vector3d::Zero();Vector3d a,w;
  fast_lio::QuietImuWindow window;
  // A brief quiet period is insufficient, and a transient must age out.
  for(int i=0;i<100;++i)window.add(i*.005,g,zero);
  assert(!window.estimate(a,w));
  fast_lio::QuietImuWindow vibration;
  for(int i=0;i<=200;++i)vibration.add(i*.005,g+Vector3d(0,i%2?.9:-.9,0),zero);
  assert(vibration.estimate(a,w));assert(std::abs(a.y())<.005);
  // Slow force changes must not be mistaken for zero-mean contact vibration.
  fast_lio::QuietImuWindow changing;
  for(int i=0;i<=200;++i)changing.add(i*.005,g+Vector3d(i<100?.2:-.2,0,0),zero);
  assert(!changing.estimate(a,w));
  window.add(.5,g+Vector3d(2,0,0),zero);
  for(int i=101;i<=299;++i)window.add(i*.005,g,zero);
  assert(!window.estimate(a,w));
  for(int i=300;i<=302;++i)window.add(i*.005,g,zero);
  assert(window.estimate(a,w));assert((a-g).norm()<1e-10);
  // Duplicate stamps do not manufacture a full second. Rewind and gaps reset.
  for(int i=0;i<1000;++i)window.add(1.51,g,zero);
  assert(window.estimate(a,w));
  window.add(0,g,zero);assert(!window.estimate(a,w));
  window.add(1,g,zero);assert(!window.estimate(a,w));
  window.add(std::numeric_limits<double>::quiet_NaN(),g,zero);assert(!window.estimate(a,w));
  for(int i=0;i<=200;++i)window.add(2+i*.005,g,Vector3d(0,0,.2));
  assert(!window.estimate(a,w));
  // Gravity alignment works for a tilted body and keeps SO(3) proper.
  const auto R=(Eigen::AngleAxisd(.2,Vector3d::UnitY())*Eigen::AngleAxisd(-.1,Vector3d::UnitX())).toRotationMatrix();
  const auto force=R.transpose()*g;
  const auto aligned=fast_lio::gravityAlignment(force);
  assert((aligned*force-g).norm()<1e-10);
  assert((aligned.transpose()*aligned-Eigen::Matrix3d::Identity()).norm()<1e-10);
  assert(std::abs(aligned.determinant()-1.)<1e-10);
  assert((fast_lio::gravityAlignment(g)-Eigen::Matrix3d::Identity()).norm()<1e-10);
  bool rejected=false;try {fast_lio::gravityAlignment(zero);}catch(const std::invalid_argument&){rejected=true;}
  assert(rejected);
}
