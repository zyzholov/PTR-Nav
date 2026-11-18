// Read-only classification through production queries, never a reference-path
// planner/controller. Authored geometry labels stay in the Python assessor.
#define TERRAIN_NAV_BENCHMARK
#include "../src/terrain_nav_node.cpp"
#include <fstream>
#include <filesystem>
#include <numeric>
#include <nlohmann/json.hpp>
#include <rclcpp/serialization.hpp>
#include <rcl/time.h>
using J=nlohmann::json;

struct TerrainNavBenchmark {
  static std::vector<double> numbers(const std::string &text,char separator=',') {
    std::istringstream input(text);std::string item;std::vector<double> result;
    while(std::getline(input,item,separator))if(!item.empty())result.push_back(std::stod(item));
    return result;
  }
  static J dynamicRefusal(TerrainNavNode &n,const std::filesystem::path &root,const J &event) {
    // Replay the recorded conditional model, not simulated actor truth. A
    // single captured scan cannot recreate source-owned historic map labels.
    const auto &s=event.at("original_status");
    const auto value=[&](const char *key){return std::stod(s.at(key).get<std::string>());};
    const double origin=value("control_state_stamp_s");
    auto *clock=n.get_clock()->get_clock_handle();
    if(rcl_enable_ros_time_override(clock)!=RCL_RET_OK||
       rcl_set_ros_time_override(clock,int64_t(std::llround(origin*1e9)))!=RCL_RET_OK)
      throw std::runtime_error("recorded root clock");
    n.odomCallback(std::make_shared<nav_msgs::msg::Odometry>(read<nav_msgs::msg::Odometry>(root/"nearby_odom.cdr")));
    n.mapCallback(std::make_shared<se2_grid_msgs::msg::SE2Grid>(read<se2_grid_msgs::msg::SE2Grid>(root/"map.cdr")));
    auto map=std::atomic_load(&n.control_map_snapshot_);
    if(!map)throw std::runtime_error("recorded map contract rejected");
    n.installSearchMapLocked(*map);
    n.cloudCallback(std::make_shared<sensor_msgs::msg::PointCloud2>(read<sensor_msgs::msg::PointCloud2>(root/"live_cloud.cdr")));
    const auto scan=std::atomic_load(&n.live_snapshot_);
    if(!scan)throw std::runtime_error("recorded live scan rejected");
    auto live=*scan;live.tracks.clear();
    for(auto it=s.begin();it!=s.end();++it) {
      const bool provisional=it.key().rfind("motion_provisional_",0)==0;
      if(!provisional&&it.key().rfind("motion_object_",0)!=0&&it.key().rfind("motion_candidate_",0)!=0)continue;
      std::map<std::string,std::string> fields;std::istringstream words(it.value().get<std::string>());std::string item;
      while(words>>item) {const auto equal=item.find('=');if(equal!=std::string::npos)fields[item.substr(0,equal)]=item.substr(equal+1);}
      const auto f=[&](const char *key){return std::stod(fields.at(key));};
      terrain_nav::MotionTrackSnapshot t;t.id=std::stoull(fields.at("identity"));t.stamp=f("source_stamp_s");
      t.resolution=value("motion_grid_resolution_m");t.circular=fields.at("model")=="measured_circle";
      t.center_motion_fit=f("causal_center_fit");t.motion_confirmed=f("confirmed_identity");
      t.center_x=f("center_x");t.center_y=f("center_y");t.radius_bound=f("radius_bound_m");
      t.vx=f("vx");t.vy=f("vy");t.velocity_error=f("velocity_reserve_mps");
      t.acceleration_bound=f("longitudinal_acceleration_bound_mps2");t.prediction_maximum_speed=f("prediction_profile_speed_mps");
      std::istringstream cells(fields.at("measured_cells"));
      while(std::getline(cells,item,';'))if(!item.empty()) {auto xy=numbers(item);t.cells.push_back({xy.at(0),xy.at(1)});}
      auto &g=t.gaussian;g.valid=f("gaussian_valid");g.constant_acceleration=f("gaussian_constant_acceleration");
      g.stamp=f("gaussian_source_stamp_s");g.observation_count=f("gaussian_observations");
      g.observation_std=f("gaussian_sigma_m");g.process_density=f("gaussian_process_density");
      g.longitudinal_acceleration_bound=f("gaussian_mean_acceleration_bound");
      if(g.valid) {
        auto mean=numbers(fields.at("gaussian_mean")),covariance=numbers(fields.at("gaussian_covariance"));
        for(int i=0;i<4;++i){g.mean[i]=mean.at(i);for(int j=0;j<4;++j)g.covariance(i,j)=covariance.at(4*i+j);}
        if(g.constant_acceleration) {
          mean=numbers(fields.at("gaussian_acceleration_mean"));covariance=numbers(fields.at("gaussian_acceleration_covariance"));
          for(int i=0;i<6;++i){g.acceleration_mean[i]=mean.at(i);for(int j=0;j<6;++j)g.acceleration_covariance(i,j)=covariance.at(6*i+j);}
        }
        if(!terrain_nav::validGaussianMotion(g))throw std::runtime_error("invalid recorded Gaussian");
      }
      if(provisional)live.provisional_tracks.push_back(t);else live.tracks.push_back(t);
    }
    const auto require=[&](double actual,const char *key) {
      if(std::abs(actual-value(key))>1e-7)throw std::runtime_error(std::string("configuration mismatch: ")+key);
    };
    const auto body=n.collisionEnvelope(n.footprint_margin_);
    require(body.front,"collision_front_m");require(body.back,"collision_back_m");
    require(2.*body.half_width,"collision_protected_width_m");
    require(n.dynamic_yield_delay_,"dynamic_actuator_delay_s");require(n.motion_joint_risk_limit_,"dynamic_joint_risk_limit");
    require(n.chance_state_limit_,"state_limit");require(n.chance_path_budget_,"local_path_budget");
    require(n.chance_braking_budget_,"braking_budget");require(live.stamp_sec,"reported_live_source_stamp_s");
    require(map->stamp.seconds(),"reported_map_source_stamp_s");
    // Only actually published commands at/before the recorded root are causal.
    std::istringstream queue(s.at("issued_command_pipeline").get<std::string>());std::string item;
    while(std::getline(queue,item,';'))if(!item.empty()) {
      const auto c=numbers(item);if(c.at(0)<=origin)n.issued_commands_.record({c[0],c[1],c[2]});
    }
    terrain_nav::YieldState state{0.,value("control_state_x_m"),value("control_state_y_m"),
      value("control_state_yaw_rad"),value("control_state_speed_mps"),0.};
    state.steering=std::abs(state.speed)>.05?clampAbs(std::atan(value("control_state_yaw_rate_radps")*n.wheel_base_/state.speed),n.max_steer_):0.;
    const auto model=n.configuredYieldModel();
    if(s.contains("dynamic_speed_dead_time_s"))require(terrain_nav::yieldSpeedDelay(model),"dynamic_speed_dead_time_s");
    if(s.contains("dynamic_steering_dead_time_s"))require(terrain_nav::yieldSteeringDelay(model),"dynamic_steering_dead_time_s");
    std::vector<std::vector<terrain_nav::YieldCommand>> commands{{{0,0,0}}};
    for(const auto &p:terrain_nav::forwardYieldProposals(state,model,n.max_speed_,n.max_steer_,6.,true))commands.push_back(p.commands);
    for(const auto &p:terrain_nav::turnReleaseYieldProposals(n.max_speed_,n.max_steer_))commands.push_back(p.commands);
    for(const auto &p:terrain_nav::reverseYieldProposals(state,model,n.reverse_max_speed_,n.max_steer_,6.,true))commands.push_back(p.commands);
    if(event.contains("additional_commands")) {
      const auto &extra=event.at("additional_commands");
      if(!extra.is_array()||extra.size()>512)throw std::runtime_error("bounded diagnostic command list required");
      for(const auto &candidate:extra) {
        if(!candidate.is_array()||candidate.empty()||candidate.size()>32)
          throw std::runtime_error("bounded diagnostic schedule required");
        std::vector<terrain_nav::YieldCommand> sequence;double previous=-1.;
        for(const auto &row:candidate) {
          if(!row.is_array()||row.size()!=3)throw std::runtime_error("command must be [time,speed,steering]");
          const double t=row.at(0),v=row.at(1),steer=row.at(2);
          if(!std::isfinite(t)||!std::isfinite(v)||!std::isfinite(steer)||t<0||t<previous||t>6.||
             v>n.max_speed_+1e-9||v < -n.reverse_max_speed_-1e-9||std::abs(steer)>n.max_steer_+1e-9)
            throw std::runtime_error("diagnostic command exceeds explicit configured limits");
          sequence.push_back({t,v,steer});previous=t;
        }
        commands.push_back(std::move(sequence));
      }
    }
    std::vector<std::vector<terrain_nav::YieldState>> paths;
    for(const auto &c:commands) {
      auto path=terrain_nav::finiteJointRollout(n.issuedYieldRollout(state,c,model,6.,origin));
      if(path.empty())throw std::runtime_error("diagnostic policy lacks complete stop and tail within six seconds");
      paths.push_back(std::move(path));
    }
    const terrain_nav::MotionCheckGeometry geometry{body.front,body.back,body.half_width,n.max_steer_,n.wheel_base_,n.live_obstacle_inflation_radius_};
    const auto screen=terrain_nav::screenJointMotionBatch(paths,origin,live.tracks,geometry,512,873612,true);
    if(!screen.valid)throw std::runtime_error("recorded-model screen: "+screen.error);
    std::vector<size_t> order(paths.size());std::iota(order.begin(),order.end(),0);
    std::stable_sort(order.begin(),order.end(),[&](size_t a,size_t b){return screen.estimates[a]<screen.estimates[b];});
    J rows=J::array();YieldFootprintCache cache;n.bindYieldCollisionViews(*map,live,cache);
    for(size_t k=0;k<order.size();++k) {
      const auto i=order[k];double path_upper=0.,brake_upper=0.;std::string reason,details;
      const auto start=std::chrono::steady_clock::now();
      double terminal_cost=0.,terminal_clearance=kInf;CostQuery terminal_query;
      const auto &end=paths[i].back();
      const bool terminal_ok=n.footprintCostLocked(end.x,end.y,end.yaw,terminal_cost,&terminal_clearance,
          nullptr,false,map.get(),cache.future_view.get(),&terminal_query);
      const bool safe=n.timedYieldSafe(paths[i],origin,commands[i],*map,live,&path_upper,&brake_upper,&reason,&details,&cache);
      J c=J::array();for(const auto &u:commands[i])c.push_back({u.t,u.speed,u.steering});
      rows.push_back({{"index",i},{"commands",c},{"screen_estimate",screen.estimates[i]},
        {"terminal_xyyaw",{end.x,end.y,end.yaw}},{"terminal_footprint_ok",terminal_ok},
        {"terminal_query_known",terminal_query.known},{"terminal_state_upper",terminal_query.probability_upper},
        {"full_proof",safe},{"reason",reason},{"detail",details},
        {"elapsed_ms",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()},
        {"dynamic_upper",n.motion_joint_last_upper_.load()},{"path_upper",path_upper},{"brake_upper",brake_upper}});
    }
    return {{"scope","recorded root, published queue and conditional motion models; production map/one-scan footprint queries; independent new proof samples; not bit-exact callback replay or physical qualification"},
      {"limitation","held occupancy and source-owned map labels are not serialized; no invented ownership exclusions or ground support"},
      {"candidate_count",paths.size()},{"all_candidates_checked",rows.size()==paths.size()},
      {"screen_ms",screen.elapsed_ms},{"rows",rows}};
  }
  template<class T>static T read(const std::filesystem::path &path) {
    std::ifstream file(path,std::ios::binary|std::ios::ate);
    if(!file)throw std::runtime_error("missing CDR: "+path.string());
    const auto size=file.tellg();if(size<=0 || size>32*1024*1024)throw std::runtime_error("CDR size");
    rclcpp::SerializedMessage buffer(static_cast<size_t>(size));auto &b=buffer.get_rcl_serialized_message();
    file.seekg(0);file.read(reinterpret_cast<char*>(b.buffer),size);b.buffer_length=size;
    if(!file)throw std::runtime_error("incomplete CDR");
    T result;rclcpp::Serialization<T>().deserialize_message(&buffer,&result);return result;
  }
  static J run(const std::filesystem::path &root,const J &poses) {
    TerrainNavNode n(rclcpp::NodeOptions(),false);
    if(poses.contains("dynamic_capture"))return dynamicRefusal(n,root,poses.at("dynamic_capture").at("event"));
    if(poses.is_object()&&poses.value("background_diagnosis",false))n.background_guide_runtime_=true;
    auto map=read<se2_grid_msgs::msg::SE2Grid>(root/"map.cdr");auto odom=read<nav_msgs::msg::Odometry>(root/"odom.cdr");
    auto *clock=n.get_clock()->get_clock_handle();
    const auto recorded_clock=poses.is_object()&&poses.contains("clock_s")?
      int64_t(std::llround(poses.at("clock_s").get<double>()*1e9)):rclcpp::Time(odom.header.stamp).nanoseconds();
    if(rcl_enable_ros_time_override(clock)!=RCL_RET_OK ||
       rcl_set_ros_time_override(clock,recorded_clock)!=RCL_RET_OK)
      throw std::runtime_error("recorded source clock");
    n.odomCallback(std::make_shared<nav_msgs::msg::Odometry>(odom));
    n.mapCallback(std::make_shared<se2_grid_msgs::msg::SE2Grid>(map));
    if(const auto snapshot=std::atomic_load(&n.control_map_snapshot_))n.installSearchMapLocked(*snapshot);
    if(!n.map_ready_)throw std::runtime_error("recorded contract rejected");
    const auto live_start=std::chrono::steady_clock::now();
    const bool newer_live=poses.is_object()&&poses.value("use_recorded_live_cloud",false);
    if(n.live_obstacle_enable_)n.cloudCallback(std::make_shared<sensor_msgs::msg::PointCloud2>(
      read<sensor_msgs::msg::PointCloud2>(root/(newer_live?"live_cloud.cdr":"completion_cloud.cdr"))));
    const double live_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-live_start).count();
    J rows=J::array();
    for(const auto &p:poses.is_array()?poses:poses.at("poses")) {
      const double x=p.at(0),y=p.at(1),yaw=p.at(2);Eigen::Array3i index;
      if(!n.map_.pos2Index(Eigen::Vector3d(x,y,yaw),index)) {rows.push_back({{"xyyaw",p},{"in_map",false}});continue;}
      const auto q=n.queryCostLocked(x,y,yaw);bool known=false;
      const bool collision=n.collisionBodyUnsafe(n.map_,x,y,yaw,known);
      double cost=0,clearance=0;const bool footprint=n.footprintCostLocked(x,y,yaw,cost,&clearance);
      terrain_nav::ChanceBudget account;const bool state=n.addChancePose(x,y,yaw,account,n.chance_state_limit_);
      J row={{"xyyaw",p},{"in_map",true},{"body_collision",collision},{"body_geometry_known",known},
        {"pose_known",q.known},{"pose_lethal",q.lethal},{"state_upper",q.probability_upper},
        {"footprint_ok",footprint},{"state_budget_ok",state},{"cost",cost},{"clearance",clearance}};
      if(const auto live=std::atomic_load(&n.live_snapshot_)) {
        row["live_overlap_all"]=n.liveBodyOverlapCount(p[0],p[1],p[2],*live);
        auto present=*live;present.hard=live->present_hard;
        row["live_overlap_present"]=n.liveBodyOverlapCount(p[0],p[1],p[2],present);
        present.hard=live->uninflated_present;present.inflation_radius=0.;
        row["live_overlap_raw"]=n.liveBodyOverlapCount(p[0],p[1],p[2],present);
        row["live_raw_raster_overlap_cells"]=J::array();row["live_raw_rectangle_candidates"]=J::array();
        const auto body=n.collisionEnvelope(n.footprint_margin_);
        for(int y=0;y<live->height;++y)for(int x=0;x<live->width;++x) {
          const double cx=live->min_x+(x+.5)*live->resolution,cy=live->min_y+(y+.5)*live->resolution;
          if(live->uninflated_present[y*live->width+x] && bodyOverlapsCell(p[0],p[1],p[2],body.front,body.back,body.half_width,cx,cy,live->resolution))
          {
            row["live_raw_raster_overlap_cells"].push_back({cx,cy,live->resolution});
            row["live_raw_rectangle_candidates"].push_back(live->rectangle(x,y));
          }
        }
      }
      for(const char *name:{"collision_risk","historical_obstacle_advisory","probability_valid","probability_pose_event",
          "failure_probability_upper","terrain_failure_probability_upper","terrain_observed","ground_pending",
          "gp_elevation_observed","probability_elevation","probability_variance","probability_gp_used","probability_raw_veto"})
        if(n.map_.exists(name))row[name]=n.map_.at(name,index);
      rows.push_back(row);
    }
    J result={{"scope","frozen map/one-scan classification; optional recorded-path continuation diagnosis, no physical execution or truth input"},
      {"rows",rows},{"live_callback_ms",live_ms},
      {"live_source",newer_live?"recorded newer live scan; no reconstruction of held-occupancy history":"conditioning scan only"},
      {"clock_s",double(recorded_clock)*1e-9}};
    if(poses.is_object() && poses.contains("recorded_external_goal")) {
      const auto g=poses.at("recorded_external_goal");
      auto goal=std::make_shared<geometry_msgs::msg::PoseStamped>();goal->header.frame_id=n.frame_id_;
      goal->header.stamp=odom.header.stamp;goal->pose.position.x=g[0];goal->pose.position.y=g[1];goal->pose.orientation.w=1.;
      n.goalCallback(goal);const bool planned=n.planPath("read-only frozen source diagnosis");
      result["planning"]={{"found",planned},{"goal",g},{"path",J::array()},
        {"staged_gears",n.recovery_staged_gears_},{"queued_legs",n.recovery_legs_.size()}};
      for(const auto &p:n.path_)result["planning"]["path"].push_back({p.x,p.y,p.yaw,p.s,p.direction});
      if(poses.value("background_diagnosis",false)&&n.background_guide_.valid()) {
        auto proposal=n.background_guide_.get();
        result["background_diagnosis"]={{"reached",proposal.route.reached},{"reason",proposal.route.reason},
          {"candidate",proposal.candidate},{"objective",proposal.objective},
          {"geometry_reached",proposal.geometry_reached},{"risk_reached",proposal.risk_reached},
          {"expanded",proposal.route.expanded},{"elapsed_ms",proposal.elapsed_ms},
          {"path",proposal.route.path},{"yaws",proposal.route.yaws},
          {"scope","frozen source clock, complete guidance search; not live navigation or a current-map certificate"}};
        std::promise<TerrainNavNode::BackgroundGuideResult> ready;n.background_guide_=ready.get_future();ready.set_value(std::move(proposal));
        const auto sequence=n.path_seq_;terrain_nav::RouteMemory::Result selected;
        result["background_diagnosis"]["admitted_as_guidance"]=n.takeBackgroundGuideLocked(n.odom_,Pose2{g[0],g[1]},selected);
        result["background_diagnosis"]["execution_path_sequence_unchanged"]=sequence==n.path_seq_;
      }
      result["planning"]["queued_certification"]=J::array();
      if(poses.value("execution_diagnosis",false)) {
        if(poses.contains("recorded_recovery_attempts"))n.reverse_recovery_count_=poses.at("recorded_recovery_attempts").get<int>();
        if(poses.contains("recorded_recovery_total"))n.reverse_recovery_total_=poses.at("recorded_recovery_total").get<int>();
        const bool dispatched=n.executePlannerJob("read-only completed-guide handoff diagnosis");
        result["execution_diagnosis"]={{"found",dispatched},{"path_points",n.path_.size()},
          {"first_gear",n.path_.empty()?0:n.path_.front().direction},{"queued_legs",n.recovery_legs_.size()},
          {"staged",n.recovery_staged_gears_},{"unplanned_attempts",n.reverse_recovery_count_.load()},
          {"reverse_total",n.reverse_recovery_total_.load()},
          {"scope","production job on frozen sources; no ROS commands or physical execution"}};
      }
      for(const auto &leg:n.recovery_legs_) {
        std::vector<Pose2> prefix;
        const bool certified=n.buildCertifiedGuidePrefixLocked(leg,prefix);
        J entry={{"gear",leg.front().direction},{"length",leg.back().s},{"prefix_available",certified},
          {"prefix_length",prefix.empty()?0.:prefix.back().s},{"first_rejected_pose",nullptr}};
        for(const auto &p:leg) {
          const auto q=n.queryCostLocked(p.x,p.y,p.yaw);
          double cost=0,clearance=0;
          if(n.footprintCostLocked(p.x,p.y,p.yaw,cost,&clearance))continue;
          bool known=false;const bool collision=n.collisionBodyUnsafe(n.map_,p.x,p.y,p.yaw,known);
          const auto live=std::atomic_load(&n.live_snapshot_);
          entry["first_rejected_pose"]={{"pose",{p.x,p.y,p.yaw,p.s}},{"support_known",q.known},
            {"upper",q.probability_upper},{"collision",collision},{"geometry_known",known},
            {"live_collision",live&&n.liveBodyUnsafe(p.x,p.y,p.yaw,*live)}};
          break;
        }
        result["planning"]["queued_certification"].push_back(entry);
      }
      if(poses.value("routing_diagnosis",false)) {
        // PRE-GOAL frames may have no qualified stance. Populate geometry for
        // read-only optimistic routing, never as an execution permission.
        Pose2 local;std::vector<Pose2> unused;
        n.resolveTopologyGoalLocked(n.odom_,Pose2{g[0],g[1]},local,unused,true);
        result["routing_searches"]=J::array();
        const auto budgets=poses.value("routing_budgets_ms",std::vector<double>{24.,250.,1000.});
        for(bool reverse:{false,true})for(double milliseconds:budgets) {
          terrain_nav::RouteOptions options;options.rear_axle_offset=n.rear_axle_offset_;options.heuristic_weight=2.5;options.allow_partial=true;
          options.advisory_cost=poses.value("routing_advisory_cost",2.);
          options.allow_advisory_start_overlap=n.topologyAdvisoryStartQualified(n.odom_.x,n.odom_.y,n.odom_.yaw);
          const auto began=std::chrono::steady_clock::now();
          const auto route=n.route_memory_.route({n.odom_.x,n.odom_.y},{g[0],g[1]},n.odom_.yaw,
            .5,n.maxReferenceCurvature(),.68,.58,[&](double x,double y,double yaw){return poses.value("routing_flat_cost",false)?1.:n.topologyTerrainCostLocked(x,y,yaw);},
            true,reverse,1,milliseconds,options);
          int switches=0;for(size_t i=1;i<route.directions.size();++i)if(route.directions[i]!=route.directions[i-1])++switches;
          result["routing_searches"].push_back({{"allow_reverse",reverse},{"budget_ms",milliseconds},
            {"reached",route.reached},{"partial",route.partial},{"expanded",route.expanded},{"reason",route.reason},
            {"elapsed_ms",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-began).count()},
            {"gear_switches",switches},{"path",route.path},{"yaws",route.yaws},{"directions",route.directions}});
        }
      }
      if(poses.value("seed_diagnosis",false)) {
        Pose2 local;std::vector<Pose2> unused;
        n.resolveTopologyGoalLocked(n.odom_,Pose2{g[0],g[1]},local,unused,true);
        result["forward_seed_diagnosis"]=J::array();
        for(double reserve:{.05,.02,0.}) {
          terrain_nav::RouteOptions options;options.rear_axle_offset=n.rear_axle_offset_;options.heuristic_weight=2.5;
          options.advisory_cost=poses.value("routing_advisory_cost",2.);
          options.allow_advisory_start_overlap=n.topologyAdvisoryStartQualified(n.odom_.x,n.odom_.y,n.odom_.yaw);
          auto route=n.route_memory_.route({n.odom_.x,n.odom_.y},{g[0],g[1]},n.odom_.yaw,
            .5+reserve,n.maxReferenceCurvature(),.68+reserve,.58+reserve,{},true,false,1,100.,options);
          J hazards=J::array();std::vector<Pose2> guide,certificate;double length=0.;
          for(size_t i=0;i<route.path.size();++i) {
            if(i)length+=std::hypot(route.path[i][0]-route.path[i-1][0],route.path[i][1]-route.path[i-1][1]);
            Pose2 p{route.path[i][0],route.path[i][1],route.yaws[i],length};
            if(length<=n.local_goal_distance_)guide.push_back(p);
            Eigen::Array3i index;if(!n.map_.pos2Index(Eigen::Vector3d(p.x,p.y,p.yaw),index))continue;
            const char *field=n.map_.exists("terrain_failure_probability_upper")?"terrain_failure_probability_upper":"failure_probability_upper";
            const double v=n.map_.at("probability_valid",index),upper=n.map_.at(field,index);
            if(std::isfinite(v)&&v>=.5&&std::isfinite(upper)&&upper>n.chance_state_limit_)
              hazards.push_back({{"pose",{p.x,p.y,p.yaw,p.s}},{"upper",upper}});
          }
          const bool prefix=n.buildCertifiedGuidePrefixLocked(guide,certificate);
          result["forward_seed_diagnosis"].push_back({{"reserve",reserve},{"reached",route.reached},
            {"reason",route.reason},{"expanded",route.expanded},{"future_hazards",hazards},
            {"prefix_available",prefix},{"prefix_length",certificate.empty()?0.:certificate.back().s},
            {"admitted",n.forwardSeedAdmittedLocked(route,n.odom_,Pose2{g[0],g[1]})},
            {"path",route.path},{"yaws",route.yaws}});
        }
      }
      if(poses.value("recovery_diagnosis",false)) {
        const Pose2 start{n.odom_.x,n.odom_.y,n.odom_.yaw,0.,-1},target{g[0],g[1],0.,0.};
        for(bool topology:{false,true}) {
          std::vector<Pose2> retreat,exit;
          const auto began=std::chrono::steady_clock::now();
          const bool found=topology?n.buildTopologyRecoveryLocked(start,target,retreat,exit):
            n.buildRecoveryManeuverLocked(start,target,retreat,&exit);
          J r={{"found",found},{"elapsed_ms",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-began).count()},
            {"retreat",J::array()},{"forward_exit",J::array()}};
          for(const auto &p:retreat)r["retreat"].push_back({p.x,p.y,p.yaw,p.s,p.direction});
          for(const auto &p:exit)r["forward_exit"].push_back({p.x,p.y,p.yaw,p.s,p.direction});
          result[topology?"topology_recovery":"local_arc_recovery"]=r;
        }
        result["certified_reverse_reach"]=J::array();
        std::vector<Pose2> fallback;
        const bool backed=n.buildCertifiedDeadendRetreatLocked(start,fallback);
        result["bounded_retreat"]={{"found",backed},{"path",J::array()}};
        for(const auto &p:fallback)result["bounded_retreat"]["path"].push_back({p.x,p.y,p.yaw,p.s,p.direction});
        const bool dispatched=n.planReverseRecovery(n.goal_seq_,n.input_epoch_.load());
        result["recovery_dispatch"]={{"found",dispatched},{"replan_only",n.recovery_replan_only_},
          {"retreat_length",n.path_.empty()?0.:n.path_.back().s},{"attempts",n.reverse_recovery_count_.load()}};
        for(double steer:n.steerSamples()) {
          auto p=start;terrain_nav::ChanceBudget budget;double reached=0.,cost=0.,clearance=0.;
          bool geometry=true,risk=n.addChancePose(p.x,p.y,p.yaw,budget,n.chance_path_budget_);
          for(int i=1;risk&&geometry&&i<=160;++i) {
            p=n.advanceArc(p,n.primitiveCurvature(steer),-.025);
            geometry=n.footprintCostLocked(p.x,p.y,p.yaw,cost,&clearance);
            risk=n.addChancePose(p.x,p.y,p.yaw,budget,n.chance_path_budget_);
            if(geometry&&risk)reached=i*.025;
          }
          result["certified_reverse_reach"].push_back({{"steer",steer},{"distance",reached},{"geometry",geometry},{"budget_ok",risk}});
        }
      }
    }
    if(std::filesystem::exists(root/"braking.json")) {
      J event;std::ifstream(root/"braking.json")>>event;
      const auto values=event.at("root"),command=event.at("command"),sample=event.at("rejected_pose");
      OdomState state=n.odom_;state.x=values[0];state.y=values[1];state.yaw=values[2];
      state.v=values[3];state.yaw_rate=values[4];
      if(rcl_set_ros_time_override(clock,event.at("evaluation_clock_ns").get<int64_t>())!=RCL_RET_OK)
        throw std::runtime_error("braking evaluation clock");
      geometry_msgs::msg::Twist cmd;cmd.linear.x=command[0];cmd.angular.z=command[1];
      double upper=0;const bool accepted=n.commandBrakingEnvelopeSafe(state,cmd,&upper);
      const auto q=n.queryCostLocked(sample[0],sample[1],sample[2]);bool known=false;
      const bool body=n.collisionBodyUnsafe(n.map_,sample[0],sample[1],sample[2],known);
      result["recorded_braking"]={{"accepted",accepted},{"upper",upper},{"rejected_pose",sample},
        {"pose_known",q.known},{"pose_upper",q.probability_upper},{"body_collision",body},
        {"geometry_known",known},{"limitation",event.at("limitation")}};
    }
    const auto event_path=root/"rejection.json";
    if(std::filesystem::exists(event_path)) {
      J event;std::ifstream(event_path)>>event;
      auto pose=event.at("evaluated_pose");OdomState state=n.odom_;
      state.x=pose[0];state.y=pose[1];state.yaw=pose[2];
      if(rcl_set_ros_time_override(clock,event.at("evaluation_clock_ns").get<int64_t>())!=RCL_RET_OK)
        throw std::runtime_error("evaluation clock");
      auto published=read<nav_msgs::msg::Path>(root/"path.cdr");std::vector<Pose2> old;
      for(const auto &p:published.poses) {
        const auto &q=p.pose.orientation;const double yaw=std::atan2(2.*(q.w*q.z+q.x*q.y),1.-2.*(q.y*q.y+q.z*q.z));
        old.push_back(Pose2{p.pose.position.x,p.pose.position.y,yaw,0.});
      }
      n.updatePathArcLength(old);double distance=0.;const size_t nearest=n.nearestPathIndex(old,state.x,state.y,&distance);
      std::vector<Pose2> remaining(old.begin()+nearest,old.end());remaining.insert(remaining.begin(),Pose2{state.x,state.y,state.yaw,0.});
      result["retention"]={{"distance",distance},{"nearest",nearest},{"remaining",old.back().s-old[nearest].s},
        {"current_path_footprint",n.pathFootprintSafeLocked(remaining,.025)},
        {"current_joint_budget",n.chancePathSafe(remaining,nullptr,n.chance_path_budget_)},
        {"hazard_in_straight_extension",n.knownHazardAheadLocked(old.back(),1.)},
        {"retained",n.canRetainCertifiedForwardPathLocked(state,old,false)}};
      // Identify the first CURRENT rejection on the original, unchanged
      // curve. Do not infer a historical failure from the final trial map.
      result["retention"]["first_rejected_sample"]=nullptr;
      terrain_nav::ChanceBudget path_account;
      const auto inspect=[&](const Pose2 &p) {
        const auto q=n.queryCostLocked(p.x,p.y,p.yaw);
        double cost=0.,clearance=kInf;bool known=false;
        const bool collision=n.collisionBodyUnsafe(n.map_,p.x,p.y,p.yaw,known);
        const bool footprint=n.footprintCostLocked(p.x,p.y,p.yaw,cost,&clearance);
        const bool joint=footprint && n.addChancePose(p.x,p.y,p.yaw,path_account,n.chance_path_budget_);
        if(footprint&&joint)return true;
        J detail={{"xyyaws",{p.x,p.y,p.yaw,p.s}},{"body_collision",collision},
          {"body_geometry_known",known},{"footprint_ok",footprint},{"pose_known",q.known},
          {"state_upper",q.probability_upper},{"path_budget_ok",joint},{"clearance",clearance}};
        const auto live=std::atomic_load(&n.live_snapshot_);
        detail["live_collision"]=live&&n.liveBodyUnsafe(p.x,p.y,p.yaw,*live);
        result["retention"]["first_rejected_sample"]=detail;return false;
      };
      if(inspect(Pose2{state.x,state.y,state.yaw})&&inspect(old[nearest]))
        for(double s=old[nearest].s+.025;s<=old.back().s+1e-8;s+=.025)
          if(!inspect(n.samplePathAt(old,std::min(s,old.back().s))))break;
      std::vector<Pose2> prefix;
      const auto snapshot=std::atomic_load(&n.control_map_snapshot_);
      const auto live=std::atomic_load(&n.live_snapshot_);
      const auto repair_start=std::chrono::steady_clock::now();
      const bool repaired=snapshot && n.certifyUnchangedPathPrefix(state,old,*snapshot,live.get(),prefix);
      result["finite_prefix_repair"]={{"found",repaired},{"path",J::array()},
        {"elapsed_ms",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-repair_start).count()}};
      if(repaired)for(const auto &p:prefix)result["finite_prefix_repair"]["path"].push_back({p.x,p.y,p.yaw,p.s});
      // This is the requested external goal in the recorded planner frame,
      // copied explicitly into the diagnostic input, never authored truth.
      if(event.contains("recorded_external_goal")) {
        const auto g=event["recorded_external_goal"];Pose2 goal{g[0],g[1],0.,0.};
        for(bool use_reference:{false,true}) {
          Pose2 active;std::vector<Pose2> guide,certificate;
          const bool found=n.resolveTopologyGoalLocked(state,goal,active,guide,false,use_reference?old:std::vector<Pose2>{});
          J output={{"found",found},{"certified",found&&n.buildCertifiedGuidePrefixLocked(guide,certificate)}};
          output["path"]=J::array();for(const auto &p:certificate)output["path"].push_back({p.x,p.y,p.yaw,p.s});
          result[use_reference?"continuity_proposal":"fresh_proposal"]=output;
        }
      }
    }
    return result;
  }
};
int main(int argc,char**argv) {
  rclcpp::init(argc,argv);
  try {const auto args=rclcpp::remove_ros_arguments(argc,argv);
    if(args.size()!=4)throw std::runtime_error("audit_frozen_map SNAPSHOT POSES_JSON OUTPUT_JSON [--ros-args ...]");
    J poses;std::ifstream(args[2])>>poses;auto out=TerrainNavBenchmark::run(args[1],poses);
    std::ofstream(args[3])<<out.dump(2)<<'\n';rclcpp::shutdown();return 0;
  } catch(const std::exception&e) {std::cerr<<e.what()<<'\n';rclcpp::shutdown();return 1;}
}
