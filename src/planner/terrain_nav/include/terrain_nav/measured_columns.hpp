#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
#include "terrain_nav/nearest_sources.hpp"

namespace terrain_nav
{
inline std::vector<uint8_t> inflateColumns(const std::vector<uint8_t> &occupied,
    int width,int height,double resolution,double radius)
{
  if(width<=0 || height<=0 || occupied.size()!=static_cast<size_t>(width*height) ||
     !std::isfinite(resolution) || resolution<=0 || !std::isfinite(radius) || radius<0)
    throw std::invalid_argument("invalid column inflation geometry");
  auto result=occupied;
  if(radius==0)return result;
  const int reach=static_cast<int>(std::ceil(radius/resolution))+1;
  for(int y=0;y<height;++y)for(int x=0;x<width;++x) {
    if(!occupied[y*width+x])continue;
    for(int dy=-reach;dy<=reach;++dy)for(int dx=-reach;dx<=reach;++dx) {
      const int nx=x+dx,ny=y+dy;
      if(nx<0 || ny<0 || nx>=width || ny>=height)continue;
      // Rasterize the Minkowski sum of the ENTIRE occupied square with a
      // disk. Center-only distance under-inflates partially intersected cells.
      const double gx=std::max(0.,(std::abs(dx)-1)*resolution);
      const double gy=std::max(0.,(std::abs(dy)-1)*resolution);
      if(std::hypot(gx,gy)<=radius+1e-12)result[ny*width+nx]=1;
    }
  }
  return result;
}

struct ColumnEvidence {
  std::vector<float> minimum, ground,occupied_minimum;
  std::vector<uint8_t> hits, occupied;
  std::vector<float> maximum;
};

// Actual above-floor returns belonging to the existing measured mask. This
// only supplies shape fitting: it neither adds/deletes occupied cells nor
// turns an overhead/witness-only return into vehicle collision occupancy.
inline std::vector<std::array<double,2>> measuredMotionSurfacePoints(
    const std::vector<std::array<float,3>>&points,const ColumnEvidence&columns,
    const std::vector<uint8_t>&tracking,int width,int height,double min_x,double min_y,
    double resolution,double minimum_height,double witness_height,const std::vector<float>&ceiling={})
{
  const size_t count=size_t(width)*height;
  if(width<=0||height<=0||columns.occupied.size()!=count||columns.ground.size()!=count||tracking.size()!=count||
     !std::isfinite(min_x)||!std::isfinite(min_y)||!std::isfinite(resolution)||resolution<=0||!std::isfinite(minimum_height)||minimum_height<=0||
     !std::isfinite(witness_height)||witness_height<0||(!ceiling.empty()&&ceiling.size()!=count))
    throw std::invalid_argument("invalid raw motion surface geometry");
  std::vector<std::array<double,2>> result;
  for(const auto &p:points) {
    if(!std::isfinite(p[0])||!std::isfinite(p[1])||!std::isfinite(p[2]))continue;
    const double x=std::floor((p[0]-min_x)/resolution),y=std::floor((p[1]-min_y)/resolution);
    if(x<0||y<0||x>=width||y>=height)continue;
    const size_t i=size_t(y)*width+size_t(x);
    if(!tracking[i]||!std::isfinite(columns.ground[i]))continue;
    const double above=p[2]-columns.ground[i];
    if(columns.occupied[i]) {
      if(above<=minimum_height||(!ceiling.empty()&&std::isfinite(ceiling[i])&&p[2]>ceiling[i]))continue;
    } else if(above<=.30||above>witness_height)continue;
    result.push_back({double(p[0]),double(p[1])});
  }
  return result;
}

struct ColumnRectangle {
  float x0=NAN,y0=NAN,x1=NAN,y1=NAN;
  bool valid() const {return std::isfinite(x0)&&std::isfinite(y0)&&std::isfinite(x1)&&std::isfinite(y1)&&x0<x1&&y0<y1;}
};

// Refine ONLY the representation inside an already occupied column. Every
// original finite XYZ return in that column contributes, including floor
// returns. Missing witnesses remain invalid and require a whole-cell fallback.
inline std::vector<ColumnRectangle> columnWitnessRectangles(
    const std::vector<std::array<float,3>> &points,const std::vector<uint8_t> &occupied,
    int width,int height,double min_x,double min_y,double resolution,double padding=.02)
{
  if(width<=0||height<=0||occupied.size()!=size_t(width*height)||
      !std::isfinite(min_x)||!std::isfinite(min_y)||!std::isfinite(resolution)||resolution<=0||
      !std::isfinite(padding)||padding<=0||padding>1.)throw std::invalid_argument("invalid column witness geometry");
  std::vector<ColumnRectangle> result(occupied.size());
  for(const auto &p:points) {
    if(!std::isfinite(p[0])||!std::isfinite(p[1])||!std::isfinite(p[2]))continue;
    const int x=int(std::floor((p[0]-min_x)/resolution)),y=int(std::floor((p[1]-min_y)/resolution));
    if(x<0||y<0||x>=width||y>=height||!occupied[y*width+x])continue;
    const double ox=min_x+x*resolution,oy=min_y+y*resolution;
    const ColumnRectangle q{
      std::nextafter(float(std::max(ox,double(p[0])-padding)),-INFINITY),
      std::nextafter(float(std::max(oy,double(p[1])-padding)),-INFINITY),
      std::nextafter(float(std::min(ox+resolution,double(p[0])+padding)),INFINITY),
      std::nextafter(float(std::min(oy+resolution,double(p[1])+padding)),INFINITY)};
    auto &r=result[y*width+x];
    if(!r.valid())r=q;
    else {r.x0=std::min(r.x0,q.x0);r.y0=std::min(r.y0,q.y0);r.x1=std::max(r.x1,q.x1);r.y1=std::max(r.y1,q.y1);}
  }
  return result;
}

// A fine column can contradict old ROUTING occupancy only after two distinct,
// recent scans contain actual floor returns there. Missing rays, GP-only floor,
// obstacle tops, mixed vertical returns and negative drops cannot clear it.
// This is not a collision certificate: the independent execution guard stays.
inline uint8_t routingClearStreak(uint8_t previous,double previous_stamp,double stamp,
                                 bool floor_witness) {
  if(!std::isfinite(stamp) || stamp<=0 || !floor_witness)return 0;
  if(stamp==previous_stamp)return previous;
  if(stamp<previous_stamp || stamp-previous_stamp>.3)return 1;
  return std::min<int>(2,previous+1);
}

inline bool routingFloorWitness(const ColumnEvidence &columns,const std::vector<float> &measured,size_t i) {
  return i<measured.size() && i<columns.maximum.size() && !columns.hits[i] && !columns.occupied[i] &&
    std::isfinite(measured[i]) && std::isfinite(columns.ground[i]) &&
    std::isfinite(columns.minimum[i]) && std::isfinite(columns.maximum[i]) &&
    std::abs(columns.minimum[i]-measured[i])<=.04F &&
    std::abs(columns.maximum[i]-measured[i])<=.04F &&
    std::abs(columns.minimum[i]-columns.ground[i])<=.04F &&
    columns.maximum[i]-columns.minimum[i]<=.04F;
}

inline ColumnEvidence measuredColumns(const std::vector<std::array<float,3>> &points,
    int width,int height,double min_x,double min_y,double resolution,
    double neighbor_radius,float min_height,int min_hits,const std::vector<float> &measured,
    const std::vector<float> &ceiling={})
{
  if(width<=0 || height<=0 || resolution<=0 || neighbor_radius<=0 ||
      !std::isfinite(resolution) || !std::isfinite(neighbor_radius) ||
      min_hits<1 || min_hits>255 ||
      (!measured.empty() && measured.size()!=static_cast<size_t>(width*height)) ||
      (!ceiling.empty() && ceiling.size()!=static_cast<size_t>(width*height)))
    throw std::invalid_argument("invalid measured-column geometry");
  const size_t cells=static_cast<size_t>(width*height);
  const float inf=std::numeric_limits<float>::infinity();
  ColumnEvidence result{std::vector<float>(cells,inf),std::vector<float>(cells,inf),std::vector<float>(cells,inf),
                        std::vector<uint8_t>(cells,0),std::vector<uint8_t>(cells,0),{}};
  std::vector<float> maximum(cells,-inf),local(cells,inf),original_ground(cells,inf);
  const auto cell_for=[&](const auto &point) {
    if(!std::isfinite(point[0]) || !std::isfinite(point[1]) || !std::isfinite(point[2]))return -1;
    const double x=std::floor((point[0]-min_x)/resolution),y=std::floor((point[1]-min_y)/resolution);
    if(x<0 || y<0 || x>=width || y>=height)return -1;
    const int i=static_cast<int>(y)*width+static_cast<int>(x);
    if(!ceiling.empty() && std::isfinite(ceiling[i]) && point[2]>ceiling[i])return -1;
    return i;
  };
  for(const auto &point:points) {
    const int i=cell_for(point);if(i<0)continue;
    result.minimum[i]=std::min(result.minimum[i],point[2]);
    maximum[i]=std::max(maximum[i],point[2]);
  }
  auto ground_witness=measured;
  if(!measured.empty())for(int row=0;row<height;++row)for(int col=0;col<width;++col) {
    const int i=row*width+col;
    // Low variance alone cannot turn an isolated obstacle top into ground.
    // Reject that role only against original nearby lower measurements.
    for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx) {
      const int y=row+dy,x=col+dx;
      if((!dx&&!dy)||x<0||y<0||x>=width||y>=height)continue;
      const float difference=measured[i]-measured[y*width+x];
      if(std::isfinite(difference)&&difference>std::max(double(min_height),std::hypot(dx,dy)*resolution))
        ground_witness[i]=std::numeric_limits<float>::quiet_NaN();
    }
  }
  const int radius=std::max(1,static_cast<int>(std::ceil(neighbor_radius/resolution)));
  // Keep the original nearest finite witness and row-major distance tie. A
  // separable exact transform avoids scattering every measured floor source
  // over a large fine-grid disk (the dominant dense-map callback cost).
  if(!measured.empty()) {
    result.ground=ground_witness;original_ground=measured;
    const auto nearest=nearestFiniteSources(ground_witness,width,height,int64_t(radius)*radius);
    bool same_sources=true;
    for(size_t i=0;i<cells;++i)if(std::isfinite(measured[i])!=std::isfinite(ground_witness[i])){same_sources=false;break;}
    const auto original_nearest=same_sources?std::vector<int>{}:
      nearestFiniteSources(measured,width,height,int64_t(radius)*radius);
    for(size_t i=0;i<cells;++i) {
      if(nearest[i]>=0)result.ground[i]=ground_witness[nearest[i]];
      const int j=same_sources?nearest[i]:original_nearest[i];
      if(j>=0)original_ground[i]=measured[j];
    }
  }
  // The scan's minimum-in-disk still uses ALL actual returns and the SAME
  // disk, including sparse/unknown terrain. No source or height is dropped.
  struct Offset {int dx,dy,distance;};
  std::vector<Offset> offsets;
  for(int dy=-radius;dy<=radius;++dy)for(int dx=-radius;dx<=radius;++dx)
    if(dx*dx+dy*dy<=radius*radius)offsets.push_back({dx,dy,dx*dx+dy*dy});
  for(int row=0;row<height;++row)for(int col=0;col<width;++col) {
    const int source=row*width+col;
    const float scan_floor=result.minimum[source];
    if(!std::isfinite(scan_floor))continue;
    for(const auto &offset:offsets) {
      const int x=col+offset.dx,y=row+offset.dy;
      if(x<0 || y<0 || x>=width || y>=height)continue;
      const int target=y*width+x;
      local[target]=std::min(local[target],scan_floor);
    }
  }
  if(measured.empty())result.ground=local;
  std::vector<uint8_t> raw(cells,0),supported(cells,0);
  const auto upper=[&](float relative){return std::isfinite(relative)&&relative>=min_height&&relative<=1.5F;};
  for(const auto &point:points) {
    const int i=cell_for(point);if(i<0)continue;
    const bool raw_hit=upper(point[2]-local[i]);
    const bool supported_hit=!measured.empty() && (upper(point[2]-result.ground[i]) || upper(point[2]-original_ground[i]));
    if(raw_hit && raw[i]<255)++raw[i];
    if(supported_hit && supported[i]<255)++supported[i];
    if(raw_hit || supported_hit)result.occupied_minimum[i]=std::min(result.occupied_minimum[i],point[2]);
  }
  for(int row=0;row<height;++row)for(int col=0;col<width;++col) {
    const int i=row*width+col;
    result.hits[i]=std::max(raw[i],supported[i]);
    result.occupied[i]=std::isfinite(result.minimum[i]) && std::isfinite(maximum[i]) &&
      maximum[i]-result.minimum[i]>=.18F && raw[i]>=min_hits;
    if(!supported[i])continue;
    int corroborated=0;
    for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx) {
      const int y=row+dy,x=col+dx;
      if(x>=0 && y>=0 && x<width && y<height)corroborated+=supported[y*width+x];
    }
    result.occupied[i]|=corroborated>=min_hits;
  }
  result.maximum=std::move(maximum);
  return result;
}
}  // namespace terrain_nav
