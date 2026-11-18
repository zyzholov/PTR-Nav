// Static randomized queries through production planning; no controller or plant.
// Geometry validation is timed separately from the synchronous planning call.
#define TERRAIN_NAV_BENCHMARK
#include "../src/terrain_nav_node.cpp"
#include <fstream>
#include <filesystem>
#include <cstring>
#include <nlohmann/json.hpp>
#include <rclcpp/serialization.hpp>
#include <rcl/time.h>
#include <Eigen/Cholesky>

using Json = nlohmann::json;

struct TerrainNavBenchmark
{
  static void clock(TerrainNavNode &node, double t)
  {
    auto *handle = node.get_clock()->get_clock_handle();
    if (rcl_enable_ros_time_override(handle) != RCL_RET_OK ||
        rcl_set_ros_time_override(handle, static_cast<int64_t>(t * 1e9)) != RCL_RET_OK)
      throw std::runtime_error("sandbox ROS clock override failed");
  }

  static se2_grid_msgs::msg::SE2Grid::SharedPtr loadMap(const std::filesystem::path &path)
  {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("missing recorded SE2Grid CDR");
    const size_t length = static_cast<size_t>(stream.tellg());
    if (!length || length > 32 * 1024 * 1024) throw std::runtime_error("invalid CDR size");
    rclcpp::SerializedMessage bytes(length);
    auto &buffer = bytes.get_rcl_serialized_message();
    stream.seekg(0); stream.read(reinterpret_cast<char *>(buffer.buffer), length);
    if (!stream) throw std::runtime_error("incomplete CDR input");
    buffer.buffer_length = length;
    auto msg = std::make_shared<se2_grid_msgs::msg::SE2Grid>();
    rclcpp::Serialization<se2_grid_msgs::msg::SE2Grid> decoder;
    decoder.deserialize_message(&bytes, msg.get());
    return msg;
  }

  static void odometry(TerrainNavNode &node, const OdomState &state, double t)
  {
    auto msg = std::make_shared<nav_msgs::msg::Odometry>();
    msg->header.frame_id = "map"; msg->header.stamp = rclcpp::Time(static_cast<int64_t>(t * 1e9));
    msg->pose.pose.position.x = state.x; msg->pose.pose.position.y = state.y;
    tf2::Quaternion q; q.setRPY(0, 0, state.yaw); msg->pose.pose.orientation = tf2::toMsg(q);
    msg->twist.twist.linear.x = state.v; msg->twist.twist.angular.z = state.yaw_rate;
    node.odomCallback(msg);
  }

  static std::vector<Json> obstacles(const Json &case_data, double t)
  {
    std::vector<Json> result;
    for (auto box : case_data["obstacles"]) {
      if (t < box.value("appear_s", 0.0) || t >= box.value("leave_s", 1e6)) continue;
      box["y"] = box.at("y").get<double>() + box.value("vy", 0.0) * t;
      result.push_back(box);
    }
    return result;
  }

  static void liveCloud(TerrainNavNode &node, const std::vector<Json> &boxes, const Json &terrain, double t)
  {
    std::vector<std::array<float, 3>> points;
    for (double x = -2; x <= 11; x += .35)
      for (double y = -5; y <= 5; y += .35) points.push_back({float(x), float(y), float(height(terrain,x,y))});
    for (const auto &b : boxes) {
      const double cx = b["x"], cy = b["y"], hx = b["hx"], hy = b["hy"];
      for (double x = cx-hx; x <= cx+hx+.01; x += .12)
        for (double y = cy-hy; y <= cy+hy+.01; y += .12) {
          const float ground=height(terrain,x,y);
          points.push_back({float(x), float(y), ground+.45f});
          points.push_back({float(x), float(y), ground+.95f});
        }
    }
    auto cloud = std::make_shared<sensor_msgs::msg::PointCloud2>();
    cloud->header.frame_id = "map"; cloud->header.stamp = rclcpp::Time(static_cast<int64_t>(t * 1e9));
    sensor_msgs::PointCloud2Modifier modifier(*cloud);
    modifier.setPointCloud2FieldsByString(1, "xyz"); modifier.resize(points.size());
    sensor_msgs::PointCloud2Iterator<float> x(*cloud, "x"), y(*cloud, "y"), z(*cloud, "z");
    for (const auto &p : points) { *x=p[0]; *y=p[1]; *z=p[2]; ++x; ++y; ++z; }
    node.cloudCallback(cloud);
  }

  static bool collision(const OdomState &s, const std::vector<Json> &boxes)
  {
    // Independent rectangle separating-axis check, not the planner's grid mask.
    const double c=std::cos(s.yaw), sn=std::sin(s.yaw);
    for (const auto &b : boxes) {
      const double dx=b["x"].get<double>()-s.x, dy=b["y"].get<double>()-s.y;
      const double hx=b["hx"], hy=b["hy"];
      if (std::abs(dx) > hx+.42*std::abs(c)+.313*std::abs(sn)) continue;
      if (std::abs(dy) > hy+.42*std::abs(sn)+.313*std::abs(c)) continue;
      if (std::abs(c*dx+sn*dy) > .42+hx*std::abs(c)+hy*std::abs(sn)) continue;
      if (std::abs(-sn*dx+c*dy) > .313+hx*std::abs(sn)+hy*std::abs(c)) continue;
      return true;
    }
    return false;
  }

  static double height(const Json &terrain, double x, double y)
  {
    const std::string kind=terrain["kind"];
    const double scale=terrain.value("scale", 0.0);
    if (kind=="ramp") return scale*x;
    if (kind=="cross_slope") return scale*y;
    if (kind=="step") return x >= terrain.value("edge", 3.5) ? scale : 0;
    if (kind=="steep_ramp") return scale*std::max(0.0,x-terrain.value("edge",3.5));
    if (kind=="waves") return scale*std::sin(x*.65)*std::cos(y*.45);
    return 0;
  }

  static std::array<double, 3> terrainState(const Json &terrain, const OdomState &s)
  {
    Eigen::Matrix<double, 143, 3> design;
    Eigen::Matrix<double, 143, 1> z;
    const double c=std::cos(s.yaw), sn=std::sin(s.yaw);
    int k=0;
    for (int i=0; i<13; ++i) for (int j=0; j<11; ++j) {
      const double u=-.42+.84*i/12, v=-.313+.626*j/10;
      design.row(k) << u, v, 1;
      z[k]=height(terrain,s.x+c*u-sn*v,s.y+sn*u+c*v); ++k;
    }
    const auto coef=(design.transpose()*design).ldlt().solve(design.transpose()*z).eval();
    const auto residual=(z-design*coef).eval();
    return {std::atan(std::abs(coef[0])), std::atan(std::abs(coef[1])), residual.maxCoeff()-residual.minCoeff()};
  }


  static Json validate(const Json &data, const std::vector<Pose2> &path)
  {
    size_t samples=0, collision_samples=0, terrain_samples=0, unknown_samples=0, outside_samples=0;
    std::array<double,3> peak{0,0,0};
    const auto boxes=obstacles(data,0);
    for (size_t i=0;i<path.size();++i) {
      const auto &a=path[i?i-1:0], &b=path[i];
      const double ds=std::hypot(b.x-a.x,b.y-a.y), dyaw=wrapAngle(b.yaw-a.yaw);
      const int steps=std::max(1,static_cast<int>(std::ceil(std::max(ds/.05,std::abs(dyaw)/.05))));
      for (int j=(i?1:0);j<=steps;++j) {
        const double f=double(j)/steps;
        OdomState s; s.x=a.x+f*(b.x-a.x);s.y=a.y+f*(b.y-a.y);s.yaw=wrapAngle(a.yaw+f*dyaw);
        ++samples;if (collision(s,boxes)) ++collision_samples;
        const auto pose=terrainState(data["terrain"],s);
        for (int k=0;k<3;++k) peak[k]=std::max(peak[k],pose[k]);
        if (pose[0]>.55 || pose[1]>.45 || pose[2]>.18) ++terrain_samples;
        const double ex=.42*std::abs(std::cos(s.yaw))+.313*std::abs(std::sin(s.yaw));
        const double ey=.42*std::abs(std::sin(s.yaw))+.313*std::abs(std::cos(s.yaw));
        if (s.x-ex < -5.2 || s.x+ex >= 15.6 || s.y-ey < -10.4 || s.y+ey >= 10.4) ++outside_samples;
        if (data.value("unknown_band",false) && s.x+ex>=data.value("unknown_edge",3.5)) ++unknown_samples;
      }
    }
    return {{"independent_validation_samples",samples},{"collision_samples",collision_samples},
      {"terrain_violation_samples",terrain_samples},{"unknown_violation_samples",unknown_samples},
      {"outside_map_samples",outside_samples},{"peak_pitch_rad",peak[0]},{"peak_roll_rad",peak[1]},
      {"peak_residual_span_m",peak[2]},
      {"independent_path_safe",samples>0 && !collision_samples && !terrain_samples && !unknown_samples && !outside_samples}};
  }

  static Json run(const Json &data,const Json &parameters,const std::filesystem::path &root,int repeat)
  {
    std::vector<rclcpp::Parameter> overrides;
    for (auto it=parameters.begin();it!=parameters.end();++it) {
      if (it.value().is_boolean()) overrides.emplace_back(it.key(),it.value().get<bool>());
      else if (it.value().is_number_integer()) overrides.emplace_back(it.key(),it.value().get<int64_t>());
      else if (it.value().is_number()) overrides.emplace_back(it.key(),it.value().get<double>());
      else if (it.value().is_string()) overrides.emplace_back(it.key(),it.value().get<std::string>());
    }
    rclcpp::NodeOptions options;options.parameter_overrides(overrides);
    TerrainNavNode node(options,false);node.rng_.seed(data.at("seed").get<unsigned>());
    OdomState state;state.x=data["start"][0];state.y=data["start"][1];state.yaw=data["start"][2];state.valid=true;
    Json row={{"id",data["id"]},{"scenario",data["scenario"]},{"repeat",repeat},
      {"expected_stop",data["expected_stop"]},{"layout_id",data["layout_id"]},{"status","ok"}};
    Pose2 initial;initial.x=state.x;initial.y=state.y;initial.yaw=state.yaw;
    const bool valid_start=validate(data,{initial})["independent_path_safe"];
    row["valid_initial_state"]=valid_start;
    if (!valid_start) {row["status"]="not_evaluable";row["reason"]="invalid independent initial footprint";return row;}
    auto map=loadMap(root/data.at("map_file").get<std::string>());
    clock(node,10);odometry(node,state,10);map->info.header.stamp=node.now();node.mapCallback(map);
    if (!node.map_ready_) throw std::runtime_error("map did not enter production planner");
    liveCloud(node,obstacles(data,0),data["terrain"],10);
    auto goal=std::make_shared<geometry_msgs::msg::PoseStamped>();goal->header.frame_id="map";
    goal->pose.position.x=data["goal"][0];goal->pose.position.y=data["goal"][1];goal->pose.orientation.w=1;
    node.goalCallback(goal);
    const auto begin=std::chrono::steady_clock::now();
    const bool ok=node.executePlannerJob("benchmark initial query");
    const double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
    std::vector<Pose2> path;
    {std::lock_guard<std::mutex> guard(node.path_mutex_);path=node.path_;}
    row.update(validate(data,path));
    row["planner_wall_ms"]=elapsed;row["returned"]=ok;row["path_points"]=path.size();
    Json points=Json::array();double length=0,variation=0,curvature_length=0;int gears=0;
    for (size_t i=0;i<path.size();++i) {
      const auto &p=path[i];points.push_back({p.x,p.y,p.yaw,p.direction});
      if (i) {
        const auto &a=path[i-1];const double ds=std::hypot(p.x-a.x,p.y-a.y);length+=ds;
        if (p.direction!=a.direction) ++gears;
        else if (ds>1e-8) {variation+=std::abs(wrapAngle(p.yaw-a.yaw));curvature_length+=ds;}
      }
    }
    row["path"]=points;row["path_length_m"]=length;row["gear_switches"]=gears;
    row["mean_abs_curvature_per_m"]=curvature_length>0?Json(variation/curvature_length):Json();
    row["start_residual_m"]=path.empty()?Json():Json(std::hypot(path.front().x-state.x,path.front().y-state.y));
    row["full_goal_connector"]=ok && !path.empty() && std::hypot(path.front().x-state.x,path.front().y-state.y)<=.25 &&
      std::hypot(path.back().x-data["goal"][0].get<double>(),path.back().y-data["goal"][1].get<double>())<=.6;
    row["goal_residual_m"]=path.empty()?Json():Json(std::hypot(path.back().x-data["goal"][0].get<double>(),path.back().y-data["goal"][1].get<double>()));
    return row;
  }
};

int main(int argc,char **argv)
{
  if (argc==2 && std::string(argv[1])=="--self-test") {
    Json data={{"terrain",{{"kind","flat"},{"scale",0}}},{"obstacles",Json::array()}};
    Pose2 p;p.x=0;p.y=0;p.yaw=0;
    if (!TerrainNavBenchmark::validate(data,{p})["independent_path_safe"].get<bool>()) return 1;
    data["obstacles"].push_back({{"x",.2},{"y",0},{"hx",.1},{"hy",.1}});
    if (TerrainNavBenchmark::validate(data,{p})["independent_path_safe"].get<bool>()) return 2;
    data["obstacles"]=Json::array();data["terrain"]={{"kind","ramp"},{"scale",1.}};
    if (TerrainNavBenchmark::validate(data,{p})["independent_path_safe"].get<bool>()) return 3;
    data["terrain"]={{"kind","step"},{"scale",.4},{"edge",0.}};
    if (TerrainNavBenchmark::validate(data,{p})["independent_path_safe"].get<bool>()) return 4;
    data["terrain"]={{"kind","flat"},{"scale",0}};data["unknown_band"]=true;data["unknown_edge"]=.3;
    if (TerrainNavBenchmark::validate(data,{p})["independent_path_safe"].get<bool>()) return 5;
    std::cout<<"independent query geometry checks passed"<<std::endl;return 0;
  }
  if (argc==2 && std::string(argv[1])=="--build-info") {
    std::cout<<Json({{"src/planner/terrain_nav/src/terrain_nav_node.cpp",BENCH_NODE_SHA},
      {"src/planner/terrain_nav/test/navigation_query_benchmark.cpp",BENCH_HARNESS_SHA},
      {"src/uneven_mapping/se2_grid/se2_grid_core/include/se2_grid_core/SE2Grid.hpp",BENCH_GRID_SHA}}).dump()<<std::endl;
    return 0;
  }
  if (argc!=3) {std::cerr<<"usage: terrain_nav_query_benchmark cases.json output-directory\n";return 2;}
  std::ifstream input(argv[1]);Json plan;input>>plan;
  const auto root=std::filesystem::path(argv[1]).parent_path(), output=std::filesystem::path(argv[2]);
  std::filesystem::create_directories(output);
  const auto config=output/"settings.json";
  if (std::filesystem::exists(config)) {
    Json prior;std::ifstream stream(config);stream>>prior;
    if (prior!=plan["settings"]) throw std::runtime_error("frozen query settings differ");
  } else {std::ofstream stream(config);stream<<plan["settings"].dump(2)<<"\n";}
  const char *args[]={"terrain_nav_query_benchmark","--ros-args","--log-level","error"};
  rclcpp::init(4,args);Json runs=Json::array();
  const int repeats=plan["settings"]["timing_repeats"];
  for (int repeat=0;repeat<repeats;++repeat) for (const auto &data:plan["cases"]) {
    if (!rclcpp::ok()) throw std::runtime_error("query batch interrupted");
    const auto path=output/(data["id"].get<std::string>()+"__r"+std::to_string(repeat)+".json");
    Json row;
    if (std::filesystem::exists(path)) {
      std::ifstream stream(path);stream>>row;
      if (row.at("id")!=data.at("id") || row.at("repeat")!=repeat) throw std::runtime_error("checkpoint mismatch");
    } else {
      try {row=TerrainNavBenchmark::run(data,plan["parameters"],root,repeat);}
      catch(const std::exception &e) {row={{"id",data["id"]},{"scenario",data["scenario"]},{"repeat",repeat},{"status","error"},{"reason",e.what()}};}
      std::ofstream stream(path);stream<<row.dump(2)<<"\n";
    }
    runs.push_back(row);
    if (runs.size()%100==0) std::cout<<"[query] "<<runs.size()<<"/"<<plan["cases"].size()*repeats<<std::endl;
  }
  std::ofstream stream(output/"results.json");stream<<Json({{"settings",plan["settings"]},{"runs",runs}}).dump(2)<<"\n";
  rclcpp::shutdown();return 0;
}

