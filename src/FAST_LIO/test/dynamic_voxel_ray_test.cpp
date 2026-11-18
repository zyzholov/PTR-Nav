#include "dynamic_voxel_ray.hpp"

#include <array>
#include <cassert>
#include <unordered_set>
#include <random>
#include <vector>

int main() {
  using dynamic_voxel_ray::Key;
  {
    using dynamic_voxel_ray::fresh_registration_label;
    assert(fresh_registration_label(true,10.1,10.,true));
    assert(!fresh_registration_label(false,10.1,10.,true));
    assert(!fresh_registration_label(true,10.1,10.,false));
    assert(!fresh_registration_label(true,10.,10.1,true)); // Future source / clock reset.
    assert(!fresh_registration_label(true,10.3,10.,true)); // Historical membership is not current evidence.
    assert(!fresh_registration_label(true,10.,-1.,true));
    assert(!fresh_registration_label(true,NAN,10.,true));
    assert(!fresh_registration_label(true,10.,INFINITY,true));
  }
  {
    // Reproduce the old omission: the same measured Float32 point has label
    // 9, but the accumulated double-coordinate grid stores it in cell 8.
    const float point=.9f,size=.1f;
    const int64_t label=std::floor(point/size),stored=std::floor(double(point)/.1);
    const int64_t old_begin=std::floor(double(label*size)/.1);
    assert(label==9 && stored==8 && old_begin>stored);
    for(float resolution:{.05f,.1f,.2f,1.f})
    for(int64_t id:{-1000000LL,-101LL,-25LL,-13LL,-1LL,0LL,1LL,13LL,25LL,61LL,101LL,1000000LL}) {
      std::array<float,2> bounds;
      assert(dynamic_voxel_ray::float_label_interval(id,resolution,bounds));
      const auto key=[&](float value){return std::floor(double(float(value/resolution)));};
      assert(key(bounds[0])==id);
      assert(key(std::nextafter(bounds[0],-INFINITY))<id);
      assert(key(std::nextafter(bounds[1],-INFINITY))==id);
      assert(key(bounds[1])>id);
      float value=float(double(id)*resolution);
      for(int i=0;i<64;++i)value=std::nextafter(value,-INFINITY);
      for(int i=0;i<129;++i,value=std::nextafter(value,INFINITY))
        assert((key(value)==id)==(value>=bounds[0]&&value<bounds[1]));
    }
    std::array<float,2> bounds;
    assert(!dynamic_voxel_ray::float_label_interval(0,NAN,bounds));
    assert(!dynamic_voxel_ray::float_label_interval(0,-.1f,bounds));
    assert(!dynamic_voxel_ray::float_label_interval(16777217,1.f,bounds)); // Unrepresentable label.
  }
  dynamic_voxel_ray::GroundHistory ground_history;
  float low,high;
  ground_history.begin(10.);ground_history.observe({0,0,0},-.5f,-.48f);
  ground_history.begin(12.);
  assert(ground_history.lookup({0,0,0},low,high));
  assert(low>-.5f && high>-.48f);  // Aging makes the ground veto stricter.
  ground_history.observe({0,0,0},-.5f,1.5f);  // An additive tall object is not a new floor.
  assert(ground_history.lookup({0,0,0},low,high));
  ground_history.begin(13.01);assert(!ground_history.lookup({0,0,0},low,high));
  ground_history.observe({0,0,0},-.5f,-.48f);ground_history.begin(2.);
  assert(!ground_history.lookup({0,0,0},low,high));  // Clock rewind cannot reuse old support.
  ground_history.observe({0,0,0},-.5f,-.48f);ground_history.begin(2.1,.5);
  assert(!ground_history.lookup({0,0,0},low,high));  // Display-map budget may change the reference grid scale.
  std::unordered_set<Key, dynamic_voxel_ray::Hash> passed;
  const auto scan = [&](std::array<double, 3> endpoint) {
    passed.clear();
    dynamic_voxel_ray::traverse({0.0, 0.0, 0.0}, endpoint, 0.2, 20.0, 0.65,
                                [&](const Key &key) { passed.insert(key); });
  };

  scan({10.0, 0.1, 0.1});
  assert(passed.count({25, 0, 0}) == 1);  // A farther return proves the old cell free.
  assert(passed.count({49, 0, 0}) == 0);  // The endpoint is occupied, not free.

  scan({4.0, 0.1, 0.1});
  assert(passed.count({25, 0, 0}) == 0);  // An occluding return cannot clear behind it.

  scan({10.0, 2.0, 0.1});
  assert(passed.count({25, 0, 0}) == 0);  // A nearby, nonintersecting ray is not proof.

  // Averaging endpoints in one registration voxel does NOT produce an
  // observed ray. Two returns miss this old cell while their centroid hits
  // it: a free-space clearer must consume the actual return positions.
  const Key old{20,0,20};
  const auto hits=[&](const std::array<double,3> &endpoint) {
    bool hit=false;
    dynamic_voxel_ray::traverse({0.,-.055,1.025},endpoint,.05,20.,.65,
      [&](const Key &key){hit|=key==old;});
    return hit;
  };
  assert(!hits({2.,.015,1.025}));assert(!hits({2.,.175,1.025}));
  assert(hits({2.,.095,1.025})); // Unobserved voxel centroid would erase it.

  dynamic_voxel_ray::EvidenceWindow evidence;
  assert(!evidence.observe(1, 3, 3));
  assert(!evidence.observe(1, 3, 3));  // Repeated rays in one scan count once.
  assert(!evidence.observe(5, 3, 3));  // Old isolated evidence expires.
  assert(!evidence.observe(6, 3, 3));
  assert(evidence.observe(8, 3, 3));
  evidence.reset();                     // A current hit makes the cell occupied again.
  assert(!evidence.observe(9, 3, 3));

  std::unordered_set<Key, dynamic_voxel_ray::Hash> cells{
      {0, 0, 0}, {1, 0, 0}, {2, 1, 0}, {5, 0, 0}, {6, 0, 0}};
  std::unordered_set<Key, dynamic_voxel_ray::Hash> connected;
  dynamic_voxel_ray::flood_connected(
      {0, 0, 0}, [&](const Key &key) { return cells.count(key) != 0; },
      [&](const Key &key) { connected.insert(key); }, 20);
  assert(connected.count({2, 1, 0}) == 1);
  assert(connected.count({5, 0, 0}) == 0);  // A separated static object is not part of the cluster.

  dynamic_voxel_ray::VoxelPresence index;
  dynamic_voxel_ray::VoxelPresence::Cache cache;
  std::unordered_set<Key,dynamic_voxel_ray::Hash> reference;
  for (int x=-24;x<25;x++) for(int y=-12;y<13;y++) for(int z=-9;z<10;z++) {
    Key key{x,y,z};
    if ((x+y+z)%7==0) {index.insert(key);reference.insert(key);}
  }
  for (int x=-25;x<26;x++) for(int y=-13;y<14;y++) for(int z=-10;z<11;z++) {
    Key key{x,y,z};
    assert(index.contains(key,cache)==(reference.count(key)!=0));
    index.erase(key);reference.erase(key);
    assert(!index.contains(key,cache));
  }
  // Cached misses must be invalidated when a block is created, including
  // after a clear/rebuild. Negative block boundaries use floor division.
  const Key negative{-1,-8,-9};
  assert(!index.contains(negative,cache));index.insert(negative);
  assert(index.contains(negative,cache));index.clear();
  assert(!index.contains(negative,cache));index.insert(negative);
  assert(index.contains(negative,cache));
  // Differential check: retain exactly the same ordered present-voxel visits,
  // including ties, negative boundaries and deletion of indexed blocks.
  std::mt19937 rng(42);std::uniform_real_distribution<double> coord(-12.,12.);
  for(int trial=0;trial<1200;++trial) {
    std::array<double,3> origin{coord(rng),coord(rng),coord(rng)};
    std::array<double,3> endpoint{coord(rng),coord(rng),coord(rng)};
    if(trial%4==0) {origin={0.,0.,0.};endpoint={8.,8.,trial%8?0.:8.};}
    if(trial%4==1) {origin={-1.6,-.8,0.};endpoint={-12.,-.8,0.};}
    const double voxel=trial%2?.1:.2;
    std::vector<Key> expected,actual,ray;
    dynamic_voxel_ray::traverse(origin,endpoint,voxel,20.,.65,[&](const Key &k){ray.push_back(k);});
    dynamic_voxel_ray::VoxelPresence after;
    std::unordered_set<Key,dynamic_voxel_ray::Hash> expected_set;
    for(const auto &k:ray)if(rng()%5==0){expected_set.insert(k);after.insert(k);}
    dynamic_voxel_ray::VoxelPresence::Cache second;
    dynamic_voxel_ray::traverse(origin,endpoint,voxel,20.,.65,[&](const Key &k){
      if(expected_set.erase(k)){expected.push_back(k);}
    });
    dynamic_voxel_ray::traverse_candidates(origin,endpoint,voxel,20.,.65,
      [&](const Key &k){return after.contains(k,second);},
      [&](const Key &k){actual.push_back(k);after.erase(k);});
    assert(expected==actual);
  }
  for (int fixture=0;fixture<32;fixture++) {
    const auto lookup=[&](int64_t x,int64_t y,float &low,float &high) {
      low=static_cast<float>(((x+3)*(y+3)+fixture)%13)*.1f;
      high=low+(((x+y+fixture)%3)==0?.3f:.1f);
      return ((x-y+fixture)%5)!=0;
    };
    const float floor=dynamic_voxel_ray::enclosing_ground_floor(0,0,lookup);
    for (int height=0;height<40;height++) {
      const double z=height*.05;
      unsigned quadrants=0;
      for(int y=-2;y<=2;y++) for(int x=-2;x<=2;x++) {
        float low,high;
        if (lookup(x,y,low,high) && high-low<=.20f && z-low>=.30)
          quadrants|=1u<<((x>=0?1:0)+(y>=0?2:0));
      }
      assert((z-floor>=.30)==(__builtin_popcount(quadrants)>=3));
    }
  }
}
