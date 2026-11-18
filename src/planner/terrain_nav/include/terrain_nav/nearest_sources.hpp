#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace terrain_nav {
// Exact squared Euclidean nearest source, lexicographic (distance,row,column).
// Two linear lower-envelope passes replace source*disk scatter. Integer costs
// encode distance*(N+1)+source_index, so equal-distance witnesses keep EXACTLY
// the original row-major tie order, without floating intersection rounding.
// This selects an existing finite source; it does not interpolate any height.
inline std::vector<int> nearestFiniteSources(const std::vector<float> &values,
    int width,int height,int64_t maximum_squared_distance) {
  const size_t cells=size_t(std::max(0,width))*std::max(0,height);
  const int extent=std::max(width,height);
  constexpr int64_t infinity=std::numeric_limits<int64_t>::max()/4;
  if(width<=0||height<=0||values.size()!=cells||cells>size_t(std::numeric_limits<int>::max())||
      maximum_squared_distance<0||4.L*(cells+1)*extent*extent>=infinity)
    throw std::invalid_argument("invalid nearest-source geometry");
  std::vector<int> result(cells,-1);
  size_t finite=0;for(float z:values)finite+=std::isfinite(z);
  if(!finite)return result;
  if(finite==cells){for(size_t i=0;i<cells;++i)result[i]=int(i);return result;}
  const int64_t scale=int64_t(cells)+1;
  std::vector<int64_t> costs(cells,infinity),f(extent),out(extent),starts(extent);
  std::vector<int> envelope(extent);
  const auto transform=[&](int n) {
    int count=0;
    for(int q=0;q<n;++q) {
      if(f[q]==infinity)continue;
      int64_t first=std::numeric_limits<int64_t>::min();
      while(count) {
        const int p=envelope[count-1];
        const int64_t numerator=f[q]-f[p]+scale*(int64_t(q)*q-int64_t(p)*p);
        const int64_t denominator=2*scale*(q-p);
        // floor division, including a negative numerator (C++ truncates).
        first=numerator/denominator-(numerator<0&&numerator%denominator!=0)+1;
        if(first>starts[count-1])break;
        --count;
      }
      if(!count)first=std::numeric_limits<int64_t>::min();
      envelope[count]=q;starts[count++]=first;
    }
    if(!count){std::fill(out.begin(),out.begin()+n,infinity);return;}
    int segment=0;
    for(int x=0;x<n;++x) {
      while(segment+1<count&&starts[segment+1]<=x)++segment;
      const int p=envelope[segment];const int64_t dx=x-p;
      out[x]=f[p]+scale*dx*dx;
    }
  };
  for(int y=0;y<height;++y) {
    for(int x=0;x<width;++x){const int i=y*width+x;f[x]=std::isfinite(values[i])?i:infinity;}
    transform(width);std::copy(out.begin(),out.begin()+width,costs.begin()+y*width);
  }
  for(int x=0;x<width;++x) {
    for(int y=0;y<height;++y)f[y]=costs[y*width+x];
    transform(height);
    for(int y=0;y<height;++y)if(out[y]!=infinity&&out[y]/scale<=maximum_squared_distance)
      result[y*width+x]=int(out[y]%scale);
  }
  return result;
}
}
