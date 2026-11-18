// Offline qualification on recorded maps. No simulation, live obstacles or plant.
#define TERRAIN_NAV_BENCHMARK
#include "../src/terrain_nav_node.cpp"
#include <fstream>
#include <filesystem>
#include <iostream>
#include <nlohmann/json.hpp>
#include <rclcpp/serialization.hpp>
#include <rcl/time.h>

using Json = nlohmann::json;

struct TerrainNavBenchmark
{
  static Json odometryConsistency(const std::filesystem::path &run_dir)
  {
    std::ifstream input(run_dir/"navigation_data.json");Json trace;input>>trace;
    const auto &poses=trace.at("localization_poses");const auto &attitudes=trace.at("localization_attitudes");
    if(poses.size()!=attitudes.size() || poses.size()<20)throw std::runtime_error("missing complete pose/quaternion pairing");
    const auto final=readCdr<nav_msgs::msg::Odometry>(run_dir/"probability_after_trial"/"odom.cdr");
    rclcpp::NodeOptions options;options.parameter_overrides({rclcpp::Parameter("cmd_topic","/qualification/odometry_guard/cmd_vel")});
    TerrainNavNode node(options,false);auto *clock=node.get_clock()->get_clock_handle();
    if(rcl_enable_ros_time_override(clock)!=RCL_RET_OK)throw std::runtime_error("clock enable failed");
    Json first=nullptr;size_t accepted=0;
    for(size_t i=0;i<poses.size();++i) {
      const auto &p=poses[i],&q=attitudes[i];double stamp=p[0];
      if(std::abs(stamp-q[0].get<double>())>1e-8)throw std::runtime_error("source-pose pairing mismatch");
      const auto time=rclcpp::Time(static_cast<int64_t>(stamp*1e9));
      if(rcl_set_ros_time_override(clock,time.nanoseconds())!=RCL_RET_OK)throw std::runtime_error("clock update failed");
      auto message=std::make_shared<nav_msgs::msg::Odometry>();message->header=final.header;
      message->header.stamp=time;message->pose.pose.position.x=p[1];message->pose.pose.position.y=p[2];message->pose.pose.position.z=p[3];
      message->pose.pose.orientation.x=q[1];message->pose.pose.orientation.y=q[2];message->pose.pose.orientation.z=q[3];message->pose.pose.orientation.w=q[4];
      node.odomCallback(message);
      if(!node.localization_fault_)++accepted;
      else if(first.is_null())first={{"index",i},{"source_stamp_s",stamp},{"input_pose",p}};
      if(node.localization_fault_) {
        geometry_msgs::msg::Twist command;command.linear.x=.8;node.publishCommand(command);
        if(node.last_cmd_v_!=0.)throw std::runtime_error("nonzero command after localization fault");
      }
    }
    return {{"scope","recorded source pose/quaternion replay through actual odometry callback; no truth input, map correction or physical execution; slow consistent drift is not detected by this kinematic gate"},
      {"samples",poses.size()},{"accepted_before_fault",accepted},{"fault_latched",node.localization_fault_.load()},{"first_fault",first}};
  }
  template<class Message> static Message readCdr(const std::filesystem::path &path)
  {
    std::ifstream file(path,std::ios::binary|std::ios::ate);
    if(!file) throw std::runtime_error("missing CDR: "+path.string());
    const auto size=file.tellg();
    if(size<=0 || size>32*1024*1024) throw std::runtime_error("invalid CDR size");
    rclcpp::SerializedMessage bytes(static_cast<size_t>(size));
    auto &buffer=bytes.get_rcl_serialized_message();
    file.seekg(0);file.read(reinterpret_cast<char*>(buffer.buffer),size);
    if(!file) throw std::runtime_error("incomplete CDR");
    buffer.buffer_length=static_cast<size_t>(size);
    Message message;rclcpp::Serialization<Message>().deserialize_message(&bytes,&message);
    return message;
  }

  static Json inspect(TerrainNavNode &node,const Pose2 &start,const Pose2 &goal)
  {
    auto arc=terrain_nav::reverseArc(start.x,start.y,start.yaw,goal.x,goal.y,
      node.reverse_max_distance_,node.maxReferenceCurvature());
    Json row={{"target",{goal.x,goal.y,goal.yaw}},{"geometric_points",arc.size()}};
    terrain_nav::ChanceBudget budget;
    std::vector<Pose2> samples;
    Json diagnostics=Json::array();
    double previous=-1;
    for(const auto &point:arc) {
      Pose2 p{point[0],point[1],wrapAngle(point[2]),0,-1};
      if(!samples.empty()) p.s=samples.back().s+std::hypot(p.x-samples.back().x,p.y-samples.back().y);
      samples.push_back(p);
      const auto query=node.queryCostLocked(p.x,p.y,p.yaw);
      double cost=0,clearance=kInf;
      const bool footprint=node.footprintCostLocked(p.x,p.y,p.yaw,cost,&clearance,nullptr,true);
      const bool accepted=node.addChancePose(p.x,p.y,p.yaw,budget,node.chance_path_budget_);
      if(!accepted || !footprint || budget.upper()!=previous) {
        diagnostics.push_back({{"s",p.s},{"pose",{p.x,p.y,p.yaw}},
          {"known",query.known},{"lethal",query.lethal},{"p",query.probability_upper},
          {"height_radius",query.height_radius},{"footprint",footprint},
          {"clearance",clearance},{"budget_upper",budget.upper()},{"budget_ok",accepted}});
        previous=budget.upper();
      }
      if(!accepted || !footprint || (std::isfinite(clearance)&&clearance<node.hardClearanceFloor())) break;
    }
    row["samples"]=diagnostics;
    std::vector<Pose2> path;
    row["certified_arc"]=node.buildCertifiedReverseArcLocked(start,goal,path);
    row["arc_length"]=path.empty()?0:path.back().s;
    row["certified_pose"]=node.buildCertifiedReversePoseLocked(start,goal,path);
    row["pose_length"]=path.empty()?0:path.back().s;
    return row;
  }

  static Json run(const std::filesystem::path &run_dir)
  {
    const auto root=run_dir/"probability_after_return";
    auto map=std::make_shared<se2_grid_msgs::msg::SE2Grid>(readCdr<se2_grid_msgs::msg::SE2Grid>(root/"map.cdr"));
    const auto odom=readCdr<nav_msgs::msg::Odometry>(root/"odom.cdr");
    std::ifstream trace_file(run_dir/"return_goal_trace.json");Json trace;trace_file>>trace;
    const auto goal_xy=trace.at("summary").at("goal");
    TerrainNavNode node(rclcpp::NodeOptions(),false);
    if(!node.chance_constrained_) throw std::runtime_error("replay requires chance_constrained");
    auto *clock=node.get_clock()->get_clock_handle();
    if(rcl_enable_ros_time_override(clock)!=RCL_RET_OK ||
       rcl_set_ros_time_override(clock,rclcpp::Time(odom.header.stamp).nanoseconds())!=RCL_RET_OK)
      throw std::runtime_error("cannot set replay clock");
    node.odomCallback(std::make_shared<nav_msgs::msg::Odometry>(odom));
    node.mapCallback(map);
    if(!node.map_ready_) throw std::runtime_error("map contract rejected");
    const Pose2 start{node.odom_.x,node.odom_.y,node.odom_.yaw,0,-1};
    const Pose2 goal{goal_xy[0],goal_xy[1],0,0,-1};
    std::ifstream forward_file(run_dir/"goal_trace.json");Json forward;forward_file>>forward;
    std::vector<Pose2> trail;
    for(const auto &event:forward.at("events")) if(event.at("kind")=="odometry") {
      Pose2 p{event.at("x"),event.at("y"),event.at("yaw"),0};
      if(trail.empty() || std::hypot(p.x-trail.back().x,p.y-trail.back().y)>.07) trail.push_back(p);
    }
    Pose2 guide;const bool guided=node.reverseGuideTarget(start,goal,trail,guide);
    Json result={{"scope","offline recorded map; no live-cloud guard or physical execution"},
      {"start",{start.x,start.y,start.yaw}},{"direct",inspect(node,start,goal)},
      {"has_guide",guided},{"map_age",(node.now()-rclcpp::Time(map->info.header.stamp)).seconds()}};
    if(guided) result["guide"]=inspect(node,start,guide);
    result["local_arcs"]=Json::array();
    for(int k=-4;k<=4;++k) {
      const double curvature=node.maxReferenceCurvature()*k/4,travel=-.8;
      Pose2 local=start;
      local.yaw=wrapAngle(start.yaw+travel*curvature);
      if(k==0) {local.x+=travel*std::cos(start.yaw);local.y+=travel*std::sin(start.yaw);}
      else {local.x+=(std::sin(local.yaw)-std::sin(start.yaw))/curvature;
        local.y-=(std::cos(local.yaw)-std::cos(start.yaw))/curvature;}
      auto probe=inspect(node,start,local);probe["curvature"]=curvature;
      result["local_arcs"].push_back(probe);
    }
    node.forward_trail_.assign(trail.begin(),trail.end());
    node.have_goal_=true;node.raw_goal_=goal;
    result["planner_returned"]=node.planPath("recorded reverse qualification");
    result["path_points"]=node.path_.size();
    return result;
  }

  static Json initialDetour(const std::filesystem::path &run_dir)
  {
    const auto root=run_dir/"probability_inputs";
    auto map=std::make_shared<se2_grid_msgs::msg::SE2Grid>(readCdr<se2_grid_msgs::msg::SE2Grid>(root/"map.cdr"));
    const auto odom=readCdr<nav_msgs::msg::Odometry>(root/"odom.cdr");
    TerrainNavNode node(rclcpp::NodeOptions(),false);
    auto *clock=node.get_clock()->get_clock_handle();
    if(rcl_enable_ros_time_override(clock)!=RCL_RET_OK ||
       rcl_set_ros_time_override(clock,rclcpp::Time(odom.header.stamp).nanoseconds())!=RCL_RET_OK)
      throw std::runtime_error("cannot set replay clock");
    node.odomCallback(std::make_shared<nav_msgs::msg::Odometry>(odom));node.mapCallback(map);
    if(!node.map_ready_ || !node.chance_constrained_)throw std::runtime_error("probability map required");
    const Pose2 start{node.odom_.x,node.odom_.y,node.odom_.yaw,0};
    node.have_goal_=true;node.raw_goal_=Pose2{start.x+7*std::cos(start.yaw),start.y+7*std::sin(start.yaw),start.yaw,0};
    const bool planned=node.planPath("recorded initial static detour");
    Json points=Json::array();
    for(const auto &p:node.path_)points.push_back({p.x,p.y,p.yaw,p.s});
    return Json{{"scope","initial recorded map; no live-cloud guard or physical execution"},
      {"planner_returned",planned},{"path",points},
      {"chance_safe",planned&&node.chancePathSafe(node.path_,nullptr,node.chance_path_budget_)}};
  }

  static Json terminalMotions(const std::filesystem::path &run_dir)
  {
    const auto root=run_dir/"probability_after_trial";
    auto map=std::make_shared<se2_grid_msgs::msg::SE2Grid>(readCdr<se2_grid_msgs::msg::SE2Grid>(root/"map.cdr"));
    const auto odom=readCdr<nav_msgs::msg::Odometry>(root/"odom.cdr");
    TerrainNavNode node(rclcpp::NodeOptions(),false);
    auto *clock=node.get_clock()->get_clock_handle();
    if(rcl_enable_ros_time_override(clock)!=RCL_RET_OK ||
       rcl_set_ros_time_override(clock,rclcpp::Time(odom.header.stamp).nanoseconds())!=RCL_RET_OK)
      throw std::runtime_error("cannot set replay clock");
    node.odomCallback(std::make_shared<nav_msgs::msg::Odometry>(odom));node.mapCallback(map);
    if(!node.map_ready_ || !node.chance_constrained_)throw std::runtime_error("probability map required");
    if(node.live_obstacle_enable_) {
      const auto cloud=readCdr<sensor_msgs::msg::PointCloud2>(root/"completion_cloud.cdr");
      node.cloudCallback(std::make_shared<sensor_msgs::msg::PointCloud2>(cloud));
      if(!std::atomic_load(&node.live_snapshot_))throw std::runtime_error("recorded live cloud rejected");
    }
    const Pose2 start{node.odom_.x,node.odom_.y,node.odom_.yaw,0};
    Json body_hits=Json::array();
    const double c=std::cos(start.yaw),s=std::sin(start.yaw);
    const double pad=.5*node.resolution_*(std::abs(c)+std::abs(s));
    const double front=std::max(node.footprint_front_,node.body_collision_front_)+node.footprint_margin_;
    const double back=std::max(node.footprint_back_,node.body_collision_back_)+node.footprint_margin_;
    const double half_width=std::max(node.footprint_half_width_,node.body_collision_half_width_)+node.footprint_margin_;
    const auto live=std::atomic_load(&node.live_snapshot_);
    for(int iy=0;iy<node.height_;++iy) for(int ix=0;ix<node.width_;++ix) {
      const double x=node.min_x_+(ix+.5)*node.resolution_,y=node.min_y_+(iy+.5)*node.resolution_;
      const double bx=c*(x-start.x)+s*(y-start.y),by=-s*(x-start.x)+c*(y-start.y);
      if(bx < -back-pad || bx > front+pad || std::abs(by)>half_width+pad)continue;
      const auto q=node.queryCostLocked(x,y,start.yaw,false,nullptr,false);
      if(q.lethal)body_hits.push_back({{"xy",{x,y}},{"local_xy",{bx,by}},
        {"collision_risk",node.map_.getValue("collision_risk",Eigen::Vector3d(x,y,start.yaw),se2_grid::InterpolationMethods::INTER_NEAREST)},
        {"live",live&&live->occupied(x,y)}});
    }
    Json motions=Json::array();
    const auto collision_evidence=[&](const Pose2 &pose) {
      const auto envelope=node.collisionEnvelope(node.footprint_margin_);
      Json coarse=Json::array(),fine=Json::array();
      for(int y=0;y<node.height_;++y)for(int x=0;x<node.width_;++x) {
        const double wx=node.min_x_+(x+.5)*node.resolution_,wy=node.min_y_+(y+.5)*node.resolution_;
        if(!bodyOverlapsCell(pose.x,pose.y,pose.yaw,envelope.front,envelope.back,envelope.half_width,
            wx,wy,node.resolution_))continue;
        const auto q=node.queryCostLocked(wx,wy,pose.yaw,false,nullptr,false,false);
        if(q.lethal)coarse.push_back({wx,wy,node.resolution_});
      }
      if(live)for(int y=0;y<live->height;++y)for(int x=0;x<live->width;++x) {
        if(!live->hard[y*live->width+x])continue;
        const double wx=live->min_x+(x+.5)*live->resolution,wy=live->min_y+(y+.5)*live->resolution;
        if(bodyOverlapsCell(pose.x,pose.y,pose.yaw,envelope.front,envelope.back,envelope.half_width,
            wx,wy,live->resolution))fine.push_back({wx,wy,live->resolution});
      }
      return Json{{"coarse_collision_cells",coarse},{"one_scan_live_collision_cells",fine},
        {"full_body",{envelope.front,envelope.back,envelope.half_width}},
        {"live_overlap",live&&node.liveBodyUnsafe(pose.x,pose.y,pose.yaw,*live)}};
    };
    for(int direction:{1,-1}) for(double steer:node.steerSamples()) {
      Pose2 pose=start;terrain_nav::ChanceBudget budget;
      Json probes=Json::array();double safe_length=0;
      bool contiguous=true;
      Json first_collision=nullptr;
      for(int i=0;i<=40;++i) {
        if(i)pose=advanceArc(pose,node.primitiveCurvature(steer),direction*.025);
        const auto q=node.queryCostLocked(pose.x,pose.y,pose.yaw);
        double cost=0,clearance=kInf;
        const bool footprint=node.footprintCostLocked(pose.x,pose.y,pose.yaw,cost,&clearance);
        const bool chance=node.addChancePose(pose.x,pose.y,pose.yaw,budget,node.chance_path_budget_);
        const bool safe=footprint && chance && (!std::isfinite(clearance)||clearance>=node.hardClearanceFloor());
        if(!footprint && first_collision.is_null())first_collision=collision_evidence(pose);
        if(contiguous && safe)safe_length=i*.025;else contiguous=false;
        probes.push_back({{"distance",i*.025},{"pose",{pose.x,pose.y,pose.yaw}},
          {"known",q.known},{"lethal",q.lethal},{"probability",q.probability_upper},
          {"footprint",footprint},{"budget_ok",chance},{"budget_upper",budget.upper()}});
      }
      motions.push_back({{"direction",direction},{"steering",steer},
        {"safe_contiguous_length",safe_length},{"probes",probes},{"first_collision_evidence",first_collision}});
    }
    std::vector<Pose2> recovery;
    std::ifstream trial_file(run_dir/"navigation_data.json");Json trial;trial_file>>trial;
    const auto &summary=trial.at("summary");
    // Preparation can move the car before the main goal is issued.
    Pose2 goal;
    std::string goal_source;
    if(trial.contains("requested_goal")) {
      const auto &request=trial.at("requested_goal");
      goal=Pose2{request.at("x"),request.at("y"),request.at("yaw"),0.};
      goal_source="recorded single external requested_goal; no authored route";
    } else {
    const auto &map_start=trial.at("map_start");
    if(!map_start.is_array() || map_start.size()!=4)throw std::runtime_error("missing main-trial map pose");
    for(const auto &value:map_start)if(!value.is_number() || !std::isfinite(value.get<double>()))
      throw std::runtime_error("nonfinite main-trial map pose");
    const double yaw=map_start.at(3);
    const double direction=summary.at("goal_mode")=="behind" ? -1. : 1.;
    const double distance=summary.at("goal_distance_m"),lateral=summary.at("goal_lateral_m");
    goal=Pose2{map_start.at(0).get<double>()+direction*distance*std::cos(yaw)-lateral*std::sin(yaw),
      map_start.at(1).get<double>()+direction*distance*std::sin(yaw)+lateral*std::cos(yaw),yaw,0.};
    goal_source="navigation_data.map_start and main-trial goal arguments";
    }
    const auto recovery_start=std::chrono::steady_clock::now();
    const bool found=node.buildRecoveryManeuverLocked(start,goal,recovery);
    const double recovery_ms=std::chrono::duration<double,std::milli>(
      std::chrono::steady_clock::now()-recovery_start).count();
    Json recovery_path=Json::array();
    for(const auto &p:recovery)recovery_path.push_back({p.x,p.y,p.yaw,p.s});
    node.have_goal_=true;node.raw_goal_=goal;
    const bool planned=node.planPath("recorded terminal forward qualification");
    std::vector<Pose2> topology_reverse,topology_exit;
    const bool topology_recovery=node.buildTopologyRecoveryLocked(start,goal,topology_reverse,topology_exit);
    Json optimistic_routes=Json::array();
    for(bool fine:{false,true}) {
      const auto route=node.route_memory_.route({start.x,start.y},{goal.x,goal.y},start.yaw,
          std::max(node.footprint_half_width_,node.body_collision_half_width_)+node.footprint_margin_,node.maxReferenceCurvature(),
          node.body_collision_front_+node.footprint_margin_,node.body_collision_back_+node.footprint_margin_,
          [&node](double x,double y,double yaw){return node.topologyTerrainCostLocked(x,y,yaw);},fine,true,-1,80.);
      optimistic_routes.push_back({{"fine",fine},{"reached",route.reached},{"reason",route.reason},{"expanded",route.expanded},
          {"path",route.path},{"yaw",route.yaws},{"gears",route.directions}});
    }
    Json forward_path=Json::array();
    for(const auto &p:node.path_)forward_path.push_back({p.x,p.y,p.yaw,p.s});
    node.goal_seq_=1;
    const bool reserve_selected=node.reserveRecoveryControl(node.odom_);
    Json reserve_commands=Json::array();
    for(const auto &command:node.reserve_commands_)reserve_commands.push_back({command.t,command.speed,command.steering});
    Json reserve_candidates=Json::array();
    terrain_nav::YieldState state;state.x=start.x;state.y=start.y;state.yaw=start.yaw;state.speed=node.odom_.v;
    terrain_nav::YieldModel model;model.delay=node.dynamic_yield_delay_;model.deceleration=node.dynamic_yield_deceleration_;
    model.acceleration=node.max_accel_;model.wheelbase=node.wheel_base_;model.steering_rate=node.max_steer_rate_;
    const auto current_map=std::atomic_load(&node.control_map_snapshot_);
    if(live)for(double distance:{.12,.25,.4,.6,.9,1.2,1.5})for(double steer:{0.,-node.max_steer_,node.max_steer_}) {
      const auto commands=terrain_nav::makeYieldCommands(state,model,distance,.3,steer,6.5);
      const auto states=terrain_nav::rolloutYield(state,state,commands,model,6.5);
      std::string reason;
      const bool safe=node.reserveRolloutSafe(states,node.now().seconds(),commands,*current_map,*live,nullptr,nullptr,&reason);
      std::vector<Pose2> exit;double joint=kInf;
      const auto exit_begin=std::chrono::steady_clock::now();
      const bool has_exit=safe&&node.reserveForwardExit(states,goal,*current_map,*live,exit,&joint);
      const double exit_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-exit_begin).count();
      Json exit_probes=Json::array();
      if(safe)for(double exit_steer:{0.,-node.max_steer_,node.max_steer_}) {
        const auto &end=states.back();Pose2 p{end.x,end.y,end.yaw,0.,1};
        double last_safe=0;std::string why="no sufficient progress";
        for(int i=1;i<=80;++i) {
          p=advanceArc(p,node.primitiveCurvature(exit_steer),.025);
          double cost=0,clearance=kInf;
          if(!node.footprintCostLocked(p.x,p.y,p.yaw,cost,&clearance,nullptr,false,current_map.get(),live.get())) {
            why="footprint";break;
          }
          if(clearance<node.hardClearanceFloor()){why="clearance";break;}
          last_safe=i*.025;
        }
        exit_probes.push_back({{"steer",exit_steer},{"length",last_safe},{"stop_reason",why}});
      }
      auto physical=*live;physical.hard=physical.present_hard=live->uninflated_present;physical.inflation_radius=0;
      terrain_nav::ChanceBudget stopping;bool stopping_ok=true;
      size_t peak=0;double raw_failure=-1;
      for(const auto &pose:states) {
        peak=std::max(peak,node.liveBodyOverlapCount(pose.x,pose.y,pose.yaw,*live));
        if(stopping_ok)stopping_ok=node.addChancePose(pose.x,pose.y,pose.yaw,stopping,node.chance_braking_budget_,current_map.get(),&physical);
        double cost=0;
        if(raw_failure<0 && !node.footprintCostLocked(pose.x,pose.y,pose.yaw,cost,nullptr,nullptr,false,current_map.get(),&physical))raw_failure=pose.t;
      }
      reserve_candidates.push_back({{"distance",distance},{"steer",steer},{"safe",safe},{"reason",reason},
        {"forward_exit",has_exit},{"joint_upper",joint},{"exit_ms",exit_ms},
        {"exit_geometry_probes",exit_probes},
        {"initial_reserve_cells",node.liveBodyOverlapCount(state.x,state.y,state.yaw,*live)},
        {"peak_reserve_cells",peak},{"whole_retreat_braking_budget",stopping.upper()},
        {"whole_retreat_braking_ok",stopping_ok},{"first_raw_footprint_failure_s",raw_failure}});
    }
    return Json{{"scope","recorded terminal map; optional one-scan obstacle reconstruction; no physical execution or reconstructed motion history"},
      {"recorded_cloud_enabled",node.live_obstacle_enable_},
      {"body_collision_cells",body_hits},
      {"start",{start.x,start.y,start.yaw}},{"motions",motions},
      {"goal",{goal.x,goal.y,goal.yaw}},{"goal_source",goal_source},
      {"recovery_ms",recovery_ms},
      {"recovery_found",found},{"recovery_path",recovery_path},
      {"topology_recovery_found",topology_recovery},
      {"optimistic_routes",optimistic_routes},
      {"topology_reverse_length",topology_reverse.empty()?0:topology_reverse.back().s},
      {"topology_exit_length",topology_exit.empty()?0:topology_exit.back().s},
      {"forward_planner_returned",planned},{"forward_path",forward_path},
      {"reserve_recovery_selected",reserve_selected},{"reserve_recovery_active",node.reserve_active_},
      {"reserve_commands",reserve_commands},{"reserve_path_upper",node.reserve_path_upper_},
      {"reserve_braking_upper",node.reserve_braking_upper_},
      {"reserve_candidates",reserve_candidates},
      {"forward_path_probability_safe",planned&&node.chancePathSafe(node.path_,nullptr,node.chance_path_budget_)}};
  }
};

int main(int argc,char **argv)
{
  rclcpp::init(argc,argv);
  try {
    const auto args=rclcpp::remove_ros_arguments(argc,argv);
    if(args.size()!=3 && !(args.size()==4 && (args[3]=="--initial-detour" || args[3]=="--terminal-motions" || args[3]=="--odometry-guard")))
      throw std::runtime_error("usage: replay_reverse_snapshot RUN_DIRECTORY OUTPUT_JSON [--initial-detour|--terminal-motions] --ros-args --params-file CONFIG -p chance_constrained:=true ...");
    const auto result=args.size()==4 && args[3]=="--odometry-guard"?TerrainNavBenchmark::odometryConsistency(args[1]):args.size()==3?TerrainNavBenchmark::run(args[1]):
      args[3]=="--terminal-motions"?TerrainNavBenchmark::terminalMotions(args[1]):TerrainNavBenchmark::initialDetour(args[1]);
    std::ofstream output(args[2]);if(!output) throw std::runtime_error("cannot open output");
    output<<result.dump(2)<<'\n';std::cout<<result.dump(2)<<std::endl;
    rclcpp::shutdown();return 0;
  } catch(const std::exception &error) {
    std::cerr<<error.what()<<std::endl;rclcpp::shutdown();return 1;
  }
}
