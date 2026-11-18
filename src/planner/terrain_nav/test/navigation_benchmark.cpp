// Accelerated kinematic sandbox using the production planner/control callbacks.
// This deliberately does not claim to replace Isaac/PhysX or sensor SLAM trials.
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

  static double quantile(std::vector<double> values, double q)
  {
    if (values.empty()) return 0;
    std::sort(values.begin(),values.end());
    return values[std::min(values.size()-1,static_cast<size_t>(q*(values.size()-1)))];
  }

  static void safetyTests(const Json &plan, const std::filesystem::path &root)
  {
    rclcpp::NodeOptions options;
    options.parameter_overrides({rclcpp::Parameter("use_sim_time",true),
      rclcpp::Parameter("allow_unknown",false),rclcpp::Parameter("unknown_robot_bubble_radius",1.25)});
    TerrainNavNode node(options,false);
    OdomState s; s.valid=true;
    clock(node,10);odometry(node,s,10);
    auto message=loadMap(root/plan["cases"][0]["map_file"].get<std::string>());
    message->info.header.stamp=node.now();node.mapCallback(message);
    if (!node.map_ready_) throw std::runtime_error("self-test map rejected");
    auto snapshot=std::make_shared<ControlMapSnapshot>(*std::atomic_load(&node.control_map_snapshot_));
    snapshot->costs.assign(snapshot->costs.size(),255);
    snapshot->observed_ground_until.assign(snapshot->costs.size(),0);
    snapshot->unknown_baseline.assign(snapshot->unknown_baseline.size(),0);
    s.x=3;odometry(node,s,10.1);
    if (!node.queryCostLocked(3,0,0,true,snapshot.get()).lethal)
      throw std::runtime_error("unknown exemption moved with the vehicle");
    if (node.queryCostLocked(0,0,0,false,snapshot.get()).lethal)
      throw std::runtime_error("initial underbody exemption was lost");
    snapshot->observed_ground_until.assign(snapshot->costs.size(),13);
    if (node.queryCostLocked(3,0,0,false,snapshot.get()).lethal)
      throw std::runtime_error("recent observed underbody evidence was ignored");
    if (!node.queryCostLocked(5,0,0,false,snapshot.get()).lethal)
      throw std::runtime_error("underbody evidence released distant unknown terrain");
    clock(node,14);
    if (!node.queryCostLocked(3,0,0,false,snapshot.get()).lethal)
      throw std::runtime_error("expired ground evidence remained usable");
    snapshot->costs.assign(snapshot->costs.size(),0);
    snapshot->clearance.assign(snapshot->clearance.size(),kInf);
    for (int iy=0;iy<snapshot->height;++iy) for(int ix=0;ix<snapshot->width;++ix)
      if (snapshot->min_x+(ix+.5)*snapshot->resolution>=1.2)
        snapshot->costs[iy*snapshot->width+ix]=254;
    std::atomic_store(&node.control_map_snapshot_,std::shared_ptr<const ControlMapSnapshot>(snapshot));
    s.x=0;s.v=1.5;s.yaw_rate=0;geometry_msgs::msg::Twist command;command.linear.x=1.5;
    if (node.commandBrakingEnvelopeSafe(s,command))
      throw std::runtime_error("braking envelope missed a future collision");
    s.v=0;command.linear.x=0;
    if (!node.commandBrakingEnvelopeSafe(s,command))
      throw std::runtime_error("stationary safe footprint rejected");
    Json box={{"x",0.1},{"y",0.1},{"hx",.1},{"hy",.1}};
    if (!collision(s,{box})) throw std::runtime_error("independent rectangle collision check failed");
    const auto slope=terrainState(Json{{"kind","ramp"},{"scale",.2}},s);
    if (std::abs(slope[0]-std::atan(.2))>1e-8 || slope[2]>1e-8)
      throw std::runtime_error("independent slope reference failed");
    s.x=0;odometry(node,s,14);node.ground_evidence_.clear();
    node.costs_.assign(node.costs_.size(),50);
    node.updateGroundEvidenceLocked(14);
    const auto key=node.evidenceKey(0,0);
    if (!node.ground_evidence_.count(key))
      throw std::runtime_error("observed soft-risk ground was not recorded");
    node.costs_.assign(node.costs_.size(),255);node.unknown_baseline_.assign(node.costs_.size(),.2);
    node.updateGroundEvidenceLocked(16.5);node.updateGroundEvidenceLocked(17.5);
    if (!node.ground_evidence_.count(key) || node.ground_evidence_.at(key)<=17.5)
      throw std::runtime_error("continuously occluded underbody ground expired");
    node.costs_.assign(node.costs_.size(),254);node.updateGroundEvidenceLocked(18);
    if (node.ground_evidence_.count(key))
      throw std::runtime_error("new occupied ground did not invalidate remembered support");
    node.risk_query_mode_="se2_yaw";node.se2_risk_lethal_=true;
    snapshot->costs.assign(snapshot->costs.size(),0);
    for (auto &slice:snapshot->grid["risk"]) slice.setZero();
    Eigen::Array3i center,corner;
    snapshot->grid.pos2Index(Eigen::Vector3d(0,0,0),center);
    snapshot->grid.pos2Index(Eigen::Vector3d(.1,.1,0),corner);
    snapshot->grid.at("risk",corner)=1;
    if (snapshot->grid.at("risk",center)!=0 || !node.queryCostLocked(0,0,0,false,snapshot.get()).lethal)
      throw std::runtime_error("half-cell hazard was not queried conservatively");
    std::cout<<"[sandbox safety] 13 checks passed"<<std::endl;
  }

  static Json run(const Json &case_data, const Json &parameters, const std::filesystem::path &root)
  {
    std::vector<rclcpp::Parameter> overrides;
    for (auto it=parameters.begin(); it!=parameters.end(); ++it) {
      if (it.value().is_boolean()) overrides.emplace_back(it.key(),it.value().get<bool>());
      else if (it.value().is_number_integer()) overrides.emplace_back(it.key(),it.value().get<int64_t>());
      else if (it.value().is_number()) overrides.emplace_back(it.key(),it.value().get<double>());
      else if (it.value().is_string()) overrides.emplace_back(it.key(),it.value().get<std::string>());
    }
    rclcpp::NodeOptions options; options.parameter_overrides(overrides);
    TerrainNavNode node(options,false);
    node.rng_.seed(case_data.at("seed").get<unsigned>());
    auto map=loadMap(root/case_data.at("map_file").get<std::string>());
    OdomState state; state.x=case_data["start"][0]; state.y=case_data["start"][1]; state.yaw=case_data["start"][2]; state.valid=true;
    const double dt=.02, duration=case_data.value("duration_s",35.0), lag=case_data.value("speed_tau_s",.2);
    double steering=0, traveled=0, sim_time=0;
    clock(node,10); odometry(node,state,10); map->info.header.stamp=node.now(); node.mapCallback(map);
    if (!node.map_ready_ || node.width_!=104 || node.height_!=104)
      throw std::runtime_error("recorded terrain map did not enter the production planner");
    auto goal=std::make_shared<geometry_msgs::msg::PoseStamped>(); goal->header.frame_id="map";
    goal->pose.position.x=case_data["goal"][0]; goal->pose.position.y=case_data["goal"][1]; goal->pose.orientation.w=1;
    node.goalCallback(goal);
    std::vector<double> plans, controls, tracking;
    Json trajectory=Json::array(), planning=Json::array(); bool collided=false, unsafe=false, reached=false, policy_violation=false;
    int reverse_samples=0, steering_samples=0, stops=0, replans=0;
    const bool expected_stop=case_data.value("expected_stop",false);
    double unsafe_time=-1, event_response=-1;
    std::array<double,3> peak_pose{0,0,0},first_unsafe_pose{0,0,0};
    const double event=case_data.value("event_s",-1.0);
    for (int i=0; sim_time<duration; ++i, sim_time+=dt) {
      if (!rclcpp::ok()) throw std::runtime_error("sandbox interrupted");
      const double t=10+sim_time; clock(node,t); odometry(node,state,t);
      const auto boxes=obstacles(case_data,sim_time);
      if (i%25==0) {
        if (!case_data.value("stale_map",false) || sim_time<event) {
          map->info.header.stamp=node.now(); node.mapCallback(map);
        }
      }
      if (i%5==0) {
        liveCloud(node,boxes,case_data["terrain"],t);
      }
      auto started=std::chrono::steady_clock::now(); node.controlTimer();
      controls.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count());
      if (node.planner_wakeup_) {
        std::string reason;
        {std::lock_guard<std::mutex> guard(node.planner_mutex_); reason=node.planner_reason_;node.planner_wakeup_=false;}
        started=std::chrono::steady_clock::now(); const bool ok=node.executePlannerJob(reason);
        const double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
        plans.push_back(elapsed);node.last_plan_elapsed_ms_=elapsed;
        planning.push_back({{"time_s",sim_time},{"reason",reason},{"success",ok},{"elapsed_ms",elapsed},
          {"path_points",node.path_.size()}});
        if (ok) ++replans;
      }
      const double command=node.last_cmd_v_, steer=node.last_cmd_steer_/node.steer_output_sign_;
      if (command<-.01) ++reverse_samples;
      if (std::abs(steer)>.03) ++steering_samples;
      if (std::abs(command)<.01) ++stops;
      if (event>=0 && sim_time>=event && event_response<0 && std::abs(command)<.01) event_response=sim_time-event;
      state.v+=std::clamp((command-state.v)*std::min(1.0,dt/std::max(.01,lag)),
                          -node.max_accel_*dt,node.max_accel_*dt);
      steering+=std::clamp(steer-steering,-node.max_steer_rate_*dt,node.max_steer_rate_*dt);
      state.yaw_rate=state.v*std::tan(steering)/node.wheel_base_;
      const double mid=state.yaw+.5*state.yaw_rate*dt;
      state.x+=state.v*std::cos(mid)*dt;state.y+=state.v*std::sin(mid)*dt;state.yaw=wrapAngle(state.yaw+state.yaw_rate*dt);
      traveled+=std::abs(state.v)*dt;
      collided=collision(state,boxes);
      const auto pose=terrainState(case_data["terrain"],state);
      for(int k=0;k<3;++k) peak_pose[k]=std::max(peak_pose[k],pose[k]);
      unsafe=pose[0]>.55 || pose[1]>.45 || pose[2]>.18;
      if (case_data.value("unknown_band",false) && state.x+.42>=case_data.value("unknown_edge",3.5)) policy_violation=true;
      if (unsafe || collided || policy_violation) {unsafe_time=sim_time;first_unsafe_pose=pose;break;}
      const double error=std::hypot(state.x-goal->pose.position.x,state.y-goal->pose.position.y);
      if (i%10==0) {
        trajectory.push_back({sim_time,state.x,state.y,state.yaw,state.v,steering,command});
        if (!node.path_.empty()) {double distance=0;node.nearestPathIndex(node.path_,state.x,state.y,&distance);tracking.push_back(distance);}
      }
      if (error<=node.goal_reached_distance_ && std::abs(state.v)<.05 && !node.have_goal_) {reached=true;break;}
      if (expected_stop && sim_time>12 && std::abs(state.v)<.01 && command==0) break;
    }
    const bool passed=!collided && !unsafe && !policy_violation && (expected_stop ? !reached && std::abs(state.v)<.05 : reached);
    return {{"id",case_data["id"]},{"scenario",case_data["scenario"]},{"seed",case_data["seed"]},
      {"status",passed?"passed":"failed"},{"reached",reached},{"collision",collided},{"unsafe_terrain",unsafe},
      {"expected_stop",expected_stop},{"sim_elapsed_s",sim_time},{"goal_error_m",std::hypot(state.x-goal->pose.position.x,state.y-goal->pose.position.y)},
      {"unknown_policy_violation",policy_violation},
      {"peak_pitch_rad",peak_pose[0]},{"peak_roll_rad",peak_pose[1]},{"peak_residual_span_m",peak_pose[2]},
      {"first_unsafe_pose",first_unsafe_pose},
      {"path_length_m",traveled},{"plan_count",plans.size()},{"replans",replans},{"planner_p95_ms",quantile(plans,.95)},
      {"final_speed_mps",state.v},
      {"planner_max_ms",quantile(plans,1)},{"control_p95_ms",quantile(controls,.95)},{"tracking_p95_m",quantile(tracking,.95)},
      {"reverse_samples",reverse_samples},{"steering_samples",steering_samples},{"stopped_samples",stops},
      {"event_stop_response_s",event_response},{"first_unsafe_time_s",unsafe_time},{"trajectory",trajectory},{"planning",planning},
      {"scope","production callbacks + first-order bicycle sandbox; not Isaac dynamics, SLAM or DDS concurrency"}};
  }
};

int main(int argc,char **argv)
{
  if (argc==2 && std::string(argv[1])=="--build-info") {
    std::cout<<Json({{"src/planner/terrain_nav/src/terrain_nav_node.cpp",BENCH_NODE_SHA},
      {"src/planner/terrain_nav/test/navigation_benchmark.cpp",BENCH_HARNESS_SHA},
      {"src/uneven_mapping/se2_grid/se2_grid_core/include/se2_grid_core/SE2Grid.hpp",BENCH_GRID_SHA}}).dump()<<std::endl;
    return 0;
  }
  if (argc<3) {std::cerr<<"usage: terrain_nav_benchmark cases.json output-directory\n";return 2;}
  std::ifstream input(argv[1]);Json plan;input>>plan;
  std::filesystem::path root=std::filesystem::path(argv[1]).parent_path(), output=argv[2];
  std::filesystem::create_directories(output);
  const auto configuration=output/"settings.json";
  if (std::filesystem::exists(configuration)) {
    std::ifstream prior(configuration);Json settings;prior>>settings;
    if (settings!=plan["settings"]) throw std::runtime_error("changed sandbox plan; use a new output directory");
  } else {std::ofstream settings(configuration);settings<<plan["settings"].dump(2)<<"\n";}
  const char *level=std::getenv("SANDBOX_LOG_LEVEL");
  int ros_argc=4;const char *ros_argv[]={"terrain_nav_benchmark","--ros-args","--log-level",level?level:"error"};
  rclcpp::init(ros_argc,ros_argv);
  if (argc>3 && std::string(argv[3])=="--self-test") {
    TerrainNavBenchmark::safetyTests(plan,root);rclcpp::shutdown();return 0;
  }
  Json runs=Json::array();int passed=0;
  for (const auto &data : plan["cases"]) {
    if (!rclcpp::ok()) break;
    const auto checkpoint=output/(data["id"].get<std::string>()+".json");
    Json result;
    if (std::filesystem::exists(checkpoint)) {std::ifstream prior(checkpoint);prior>>result;
      if (result.at("id")!=data.at("id") || result.at("scenario")!=data.at("scenario"))
        throw std::runtime_error("checkpoint identity differs from frozen case");}
    else {try {result=TerrainNavBenchmark::run(data,plan["parameters"],root);}
      catch(const std::exception &e) {result={{"id",data["id"]},{"scenario",data["scenario"]},{"status","error"},{"reason",e.what()}};}
      std::ofstream artifact(checkpoint);artifact<<result.dump(2)<<"\n";}
    if (result["status"]=="passed") ++passed;runs.push_back(result);
    if (runs.size()%20==0) std::cout<<"[sandbox] "<<runs.size()<<"/"<<plan["cases"].size()<<"; passed="<<passed<<std::endl;
  }
  std::ofstream artifact(output/"results.json");artifact<<Json({{"settings",plan["settings"]},{"runs",runs},{"passed",passed}}).dump(2)<<"\n";
  rclcpp::shutdown();return 0;
}
