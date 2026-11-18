// Standalone microbenchmark; no timing assertions and no physical qualification.
// Compile identical code/flags against each frozen header, compare ALL evidence
// buffers, and retain source/compiler/hardware provenance alongside the output.
#include "terrain_nav/measured_columns.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <random>
#include <fstream>

template<class T>void checksum(uint64_t &h,const std::vector<T> &v) {
  for(size_t i=0;i<v.size()*sizeof(T);++i){h^=reinterpret_cast<const uint8_t*>(v.data())[i];h*=1099511628211ULL;}
}
template<class T>void dump(std::ofstream &out,const std::vector<T> &v) {
  out.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(T));
}
int main(int argc,char **argv) {
  constexpr int width=416,height=416;
  std::cout<<"ground_percent,median_ms,min_ms,evidence_fnv64\n";
  for(int density:{5,25,75}) {
    std::mt19937 rng(20261107);std::vector<float> ground(width*height,NAN);
    for(int i=0;i<width*height;++i)if(int(rng()%100)<density)ground[i]=.01F*float(i%17);
    std::vector<std::array<float,3>> points;
    for(int i=0;i<10000;++i) {
      const int x=rng()%width,y=rng()%height;
      points.push_back({float((x+.5)*.05),float((y+.5)*.05),float(rng()%100<15?.6:0.)});
    }
    std::vector<double> elapsed;uint64_t hash=0;
    for(int iteration=0;iteration<12;++iteration) {
      const auto start=std::chrono::steady_clock::now();
      const auto result=terrain_nav::measuredColumns(points,width,height,0,0,.05,.85,.100001F,1,ground);
      const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
      if(iteration<3)continue;
      elapsed.push_back(ms);hash=14695981039346656037ULL;
      checksum(hash,result.minimum);checksum(hash,result.ground);checksum(hash,result.occupied_minimum);
      checksum(hash,result.hits);checksum(hash,result.occupied);checksum(hash,result.maximum);
      if(argc==2&&iteration==11) {
        std::ofstream data(std::string(argv[1])+"_"+std::to_string(density)+".bin",std::ios::binary);
        dump(data,result.minimum);dump(data,result.ground);dump(data,result.occupied_minimum);
        dump(data,result.hits);dump(data,result.occupied);dump(data,result.maximum);
        if(!data)return 2;
      }
    }
    std::sort(elapsed.begin(),elapsed.end());
    std::cout<<density<<','<<std::fixed<<std::setprecision(3)<<elapsed[elapsed.size()/2]<<','<<elapsed[0]<<','<<hash<<'\n';
  }
}
