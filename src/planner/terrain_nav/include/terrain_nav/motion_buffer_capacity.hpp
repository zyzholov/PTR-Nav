#pragma once
#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace terrain_nav {
// Reserve bounded headroom when a workload first grows, rather than calling
// cudaFree/cudaMalloc for every slightly larger scan/candidate batch. Counts,
// samples, arithmetic and the shared-forecast memory limit do not change.
inline size_t reservedMotionCapacity(size_t requested,size_t maximum)
{
  if(!requested||requested>maximum)throw std::invalid_argument("invalid motion buffer capacity");
  return requested+std::min(requested/2,maximum-requested);
}
}
