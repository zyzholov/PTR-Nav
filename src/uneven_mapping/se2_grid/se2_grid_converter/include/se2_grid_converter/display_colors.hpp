#pragma once

#include <algorithm>
#include <array>
#include <cmath>

namespace se2_grid_converter
{
using RGB = std::array<float, 3>;

inline RGB mixColor(const RGB &a, const RGB &b, float u)
{
  u=std::clamp(u,0.F,1.F);
  return {a[0]+u*(b[0]-a[0]),a[1]+u*(b[1]-a[1]),a[2]+u*(b[2]-a[2])};
}

inline RGB probabilityColor(float p)
{
  if(!std::isfinite(p))return {.30F,.34F,.39F};
  constexpr float stops[]={0.F,.001F,.01F,.05F,1.F};
  constexpr RGB colors[]={{{.23F,.37F,.40F}},{{.29F,.51F,.49F}},
    {{.57F,.66F,.55F}},{{.91F,.70F,.36F}},{{.84F,.29F,.36F}}};
  p=std::clamp(p,0.F,1.F);int i=0;
  while(i<3 && p>stops[i+1])++i;
  return mixColor(colors[i],colors[i+1],(p-stops[i])/(stops[i+1]-stops[i]));
}

inline RGB lidarReturnColor(float range)
{
  if(!std::isfinite(range))return {.30F,.34F,.39F};
  if(range<8.F)return mixColor({.98F,.76F,.34F},{.37F,.89F,.72F},range/8.F);
  return mixColor({.37F,.89F,.72F},{.24F,.64F,.96F},(range-8.F)/22.F);
}

inline RGB lidarMapColor(float relative_height)
{
  if(!std::isfinite(relative_height))return {.30F,.34F,.39F};
  return mixColor({.29F,.40F,.48F},{.64F,.77F,.80F},(relative_height+.4F)/3.4F);
}
}  // namespace se2_grid_converter
