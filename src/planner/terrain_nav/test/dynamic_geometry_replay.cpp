// Recorded whole-scene motion masks -> SAME online body/time predicate.
// Root reconstruction is disclosed; no terrain admission or physical claim.
#include "terrain_nav/motion_prediction.hpp"
#include "terrain_nav/motion_risk.hpp"
#include "terrain_nav/joint_motion_risk.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>

using Json=nlohmann::json;
Json read(const std::filesystem::path &p){std::ifstream f(p);if(!f)throw std::runtime_error(p.string());Json d;f>>d;return d;}
int main(int argc,char **argv)
{
  try {
    if((argc<3 || argc>15) || std::filesystem::exists(argv[2]))throw std::runtime_error("usage: dynamic_geometry_replay RUN NEW_JSON [MODEL_DELAY [PREDICTION_SPEED_PROFILE [MODEL [ESCAPE_SPEED [HORIZON [OBSERVATIONS [REPORT_STAMP [RISK_LIMIT [IDENTITY_MEMORY [TURNING_REACQUISITION [KNOWN_SLOW_CENTERS [REVERSE_SPEED]]]]]]]]]]]]");
    const std::filesystem::path run=argv[1];
    const auto nav=read(run/"navigation_data.json"),raw=read(run/"motion_grids.json"),stream=read(run/"motion_streams.json");
    if(raw.at("overflow").get<bool>() || raw.at("schema_version")!=1)throw std::runtime_error("incomplete raw history");
    double end=0;
    for(const auto &r:nav.at("safety_status")) {
      const auto &v=r["status"][0]["values"];
      if(v.value("localization_fault",std::string("0"))=="1")break;
      if(std::stoi(v.value("dynamic_track_count",std::string("0")))>0)end=r.at("stamp_s");
    }
    if(!end)throw std::runtime_error("no pre-fault motion interaction");
    terrain_nav::GridMotionGuard guard;guard.acceleration_bound_mps2=1.2;
    if(argc>=5)guard.prediction_maximum_speed_mps=std::stod(argv[4]);
    const std::string prediction_model=argc>=6?argv[5]:"reachable";
    const double escape_speed=argc>=7?std::stod(argv[6]):1.33;
    const double joint_horizon=argc>=8?std::stod(argv[7]):3.;
    const double selected_report=argc>=10?std::stod(argv[9]):NAN;
    const double risk_limit=argc>=11?std::stod(argv[10]):.02;
    guard.identity_memory_seconds=argc>=12?std::stod(argv[11]):3.;
    const std::string turning=argc>=13?argv[12]:"false";
    if(turning!="true"&&turning!="false")throw std::runtime_error("turning identity flag must be true or false");
    guard.turning_identity_reacquisition=turning=="true";
    const std::string slow=argc>=14?argv[13]:"false";
    if(slow!="true"&&slow!="false")throw std::runtime_error("known slow centres flag must be true or false");
    guard.track_known_slow_centers=slow=="true";
    const double reverse_speed=argc>=15?std::stod(argv[14]):.55;
    if(!std::isfinite(reverse_speed)||reverse_speed<=0||reverse_speed>1.33)throw std::runtime_error("counterfactual reverse speed outside disclosed simulation operating bound (0,1.33]");
    if((argc>=10&&!std::isfinite(selected_report))||!std::isfinite(risk_limit)||risk_limit<=0||risk_limit>=1.)
      throw std::runtime_error("invalid disclosed report selection or risk limit");
    if(!std::isfinite(joint_horizon)||joint_horizon<3||joint_horizon>6)throw std::runtime_error("invalid disclosed proof horizon");
    if(!std::isfinite(escape_speed)||escape_speed<=0||escape_speed>20./3.6)throw std::runtime_error("escape speed outside disclosed datasheet maximum");
    if(prediction_model!="reachable" && prediction_model!="gaussian_cv" && prediction_model!="gaussian_joint_cuda" &&
       prediction_model!="gaussian_joint_cpu" && prediction_model!="gaussian_joint_accel_cuda" &&
       prediction_model!="gaussian_joint_sim_cuda" && prediction_model!="gaussian_joint_accel_sim_cuda")throw std::runtime_error("invalid prediction model");
    const bool joint=prediction_model.find("gaussian_joint_")==0;
    guard.fit_gaussian_motion=prediction_model!="reachable";
    guard.gaussian_acceleration_envelope=prediction_model=="gaussian_joint_accel_cuda" || prediction_model=="gaussian_joint_accel_sim_cuda";
    guard.sim_observation_calibration=prediction_model=="gaussian_joint_sim_cuda" || prediction_model=="gaussian_joint_accel_sim_cuda";
    const auto &settings=nav.at("safety_status").at(0).at("status").at(0).at("values");
    if(settings.contains("dynamic_gaussian_constant_acceleration"))
      guard.gaussian_constant_acceleration=std::stod(settings.at("dynamic_gaussian_constant_acceleration").get<std::string>())>.5;
    if(guard.gaussian_constant_acceleration) {
      if(!joint||guard.gaussian_acceleration_envelope)throw std::runtime_error("CA replay requires matching joint model, not bounded CV");
      guard.gaussian_jerk_density=std::stod(settings.at("dynamic_gaussian_jerk_density").get<std::string>());
    }
    if(argc>=9)guard.gaussian_maximum_observations=std::stoi(argv[8]);
    if(guard.gaussian_maximum_observations<3||guard.gaussian_maximum_observations>4)throw std::runtime_error("invalid observation window");
    if(joint&&prediction_model!="gaussian_joint_cpu") {std::string e;if(!terrain_nav::jointMotionGpuAvailable(&e))throw std::runtime_error(e);}
    std::vector<std::pair<double,std::vector<terrain_nav::MotionTrackSnapshot>>> history;
    Json hypothesis_timeline=Json::array();
    const bool tracking=raw.contains("tracking_frames")&&!raw.at("tracking_frames").empty();
    for(const auto &sample:raw.at(tracking?"tracking_frames":"frames")) {
      const double t=sample.at("stamp_s");if(t>end)break;
      const auto &g=sample.at("geometry");const int w=g[0],h=g[1];
      std::vector<uint8_t> cells(size_t(w)*h,0);
      for(int id:sample.at("occupied")){if(id<0||id>=int(cells.size())||cells[id])throw std::runtime_error("invalid occupancy");cells[id]=1;}
      guard.update(cells,w,h,g[2],g[3],g[4],t);
      auto actions=guard.actionHypotheses(true);
      if(!actions.empty())hypothesis_timeline.push_back({{"stamp",t},{"confirmed",guard.snapshot().size()},
        {"action_hypotheses",actions.size()},{"remaining_brake_only",guard.brakeOnlyHypotheses(true).size()}});
      history.push_back({t,std::move(actions)});
    }
    Json rows=Json::array();const auto &poses=nav.at("localization_poses");
    terrain_nav::YieldModel model;model.wheelbase=.430;model.rear_axle_offset=.215;
    if(settings.contains("dynamic_speed_dead_time_s"))model.speed_dead_time=std::stod(settings.at("dynamic_speed_dead_time_s").get<std::string>());
    if(settings.contains("dynamic_steering_dead_time_s"))model.steering_dead_time=std::stod(settings.at("dynamic_steering_dead_time_s").get<std::string>());
    if(argc>=4)model.delay=std::stod(argv[3]);
    if(!std::isfinite(model.delay)||model.delay<0||model.delay>1.)throw std::runtime_error("invalid disclosed model delay");
    terrain_nav::MotionCheckGeometry geometry;
    for(const auto &r:nav.at("safety_status")) {
      const auto &values=r["status"][0]["values"];const double report_time=r.at("stamp_s");
      if(std::isfinite(selected_report)&&std::abs(report_time-selected_report)>1e-6)continue;
      const bool exact_origin=values.contains("dynamic_control_callback_origin_s") &&
        values.value("dynamic_evaluation_current",std::string("0"))=="1";
      double time=exact_origin?std::stod(values.at("dynamic_control_callback_origin_s").get<std::string>()):report_time;
      const double risk_origin=values.contains("dynamic_joint_last_origin_s")?std::stod(values.at("dynamic_joint_last_origin_s").get<std::string>()):NAN;
      const bool exact_risk_origin=exact_origin&&std::isfinite(risk_origin)&&risk_origin<=report_time&&report_time-risk_origin<.2;
      if(exact_risk_origin)time=risk_origin;
      if(values.value("localization_fault",std::string("0"))=="1")break;
      if(std::stoi(values.value("dynamic_track_count",std::string("0")))<1 &&
         (hypothesis_timeline.empty() || time<hypothesis_timeline[0]["stamp"].get<double>()+.10))continue;
      // The observer's actual odometry reception clock chooses a source,
      // rather than accidentally using a future delivered pose from disk.
      double source=-1;
      for(const auto &s:stream.at("samples").at("/Odometry")) {
        const double stamp=s[1].get<double>()+s[2].get<double>()*1e-9;
        if(stamp+s[3].get<double>()<=time+1e-8)source=stamp;
      }
      size_t pose_id=0;double delta=INFINITY;
      for(size_t i=0;i<poses.size();++i)if(std::abs(poses[i][0].get<double>()-source)<delta) {
        pose_id=i;delta=std::abs(poses[i][0].get<double>()-source);
      }
      if(delta>1e-5)continue;
      const auto &p=poses[pose_id];terrain_nav::YieldState state;
      state.x=p[1];state.y=p[2];state.yaw=p[4];state.speed=std::stod(values.value("control_state_speed_mps",values.at("measured_speed_mps").get<std::string>()));
      double yaw_rate=0;
      if(pose_id){const auto &old=poses[pose_id-1];yaw_rate=std::remainder(state.yaw-old[4].get<double>(),2*M_PI)/(source-old[0].get<double>());}
      state=terrain_nav::integrateBodyVelocity(state,state.speed,yaw_rate,
          std::clamp(.08+std::max(0.,time-source),0.,.25),.215);
      state.steering=std::abs(state.speed)>.05?std::clamp(std::atan(yaw_rate*.430/state.speed),-geometry.max_steering,geometry.max_steering):0.;
      const bool exact_root=values.contains("control_state_x_m");
      if(exact_root) {
        state.x=std::stod(values.at("control_state_x_m").get<std::string>());
        state.y=std::stod(values.at("control_state_y_m").get<std::string>());
        state.yaw=std::stod(values.at("control_state_yaw_rad").get<std::string>());
        yaw_rate=std::stod(values.value("control_state_yaw_rate_radps",values.at("measured_yaw_rate_radps").get<std::string>()));
        state.steering=std::abs(state.speed)>.05?std::clamp(std::atan(yaw_rate*.430/state.speed),-geometry.max_steering,geometry.max_steering):
          std::stod(values.at("command_steering_rad").get<std::string>());
      }
      double motion_source=-1;
      for(auto it=values.begin();it!=values.end();++it)if(it.key().find("motion_object_")==0) {
        const auto text=it.value().get<std::string>();const auto pos=text.find("source_age_s=");
        if(pos!=std::string::npos){motion_source=report_time-std::stod(text.substr(pos+13));break;}
      }
      const bool exact_motion_source=motion_source>=0;
      if(!exact_motion_source)for(const auto &s:stream.at("samples").at("/cloud_registered")) {
        const double stamp=s[1].get<double>()+s[2].get<double>()*1e-9;
        // Diagnostic reconstruction only: reserve 20ms after the observer's
        // measured receipt for motion processing. Never an online timestamp.
        if(stamp+s[3].get<double>()+.02<=time)motion_source=stamp;
      }
      const std::vector<terrain_nav::MotionTrackSnapshot> *tracks=nullptr;double match=INFINITY,matched_stamp=0;
      for(const auto &[t,ts]:history)if(std::abs(t-motion_source)<match) {match=std::abs(t-motion_source);tracks=&ts;matched_stamp=t;}
      if(!tracks || tracks->empty() || match>.011)continue;
      const double horizon=joint?joint_horizon:terrain_nav::interactionHorizon(state,model,.55,1.2);
      Json candidates=Json::array();std::map<std::string,int> counts;
      const auto check=[&](const std::string &family,double length,double speed,double steering,std::vector<terrain_nav::YieldCommand> commands) {
        if(commands.empty()){++counts["empty_commands"];return;}
        auto states=terrain_nav::rolloutYield(state,state,commands,model,horizon);
        if(!joint&&(family=="advance"||family=="bypass"))states=terrain_nav::finiteAdvanceRollout(std::move(states));
        if(joint&&joint_horizon>3.)states=terrain_nav::finiteJointRollout(std::move(states));
        if(joint&&(states.size()<11 || std::abs(states[states.size()-11].speed)>.001))states.clear();
        if(states.empty()){++counts["insufficient_stopped_tail"];return;}
        terrain_nav::MotionConflict conflict;
        const auto risk=guard.fit_gaussian_motion?terrain_nav::timedGaussianMotionRisk(states,time,*tracks,geometry):terrain_nav::MotionRisk{};
        const auto joint_risk=joint?terrain_nav::jointGaussianMotionRisk(states,time,*tracks,geometry,8192,1e-6,789+rows.size()*100+candidates.size(),prediction_model!="gaussian_joint_cpu"):terrain_nav::JointMotionRisk{};
        const bool unsafe=joint?(!joint_risk.valid||joint_risk.upper>risk_limit):guard.fit_gaussian_motion?(!risk.valid || risk.upper>risk_limit):
          terrain_nav::timedMotionUnsafe(states,time,*tracks,geometry,0.,0.,&conflict);
        ++counts[unsafe?"motion_rejected":"motion_admissible"];
        candidates.push_back({{"family",family},{"length",length},{"speed",speed},{"steering",steering},
          {"horizon",states.back().t},{"end",{states.back().x,states.back().y,states.back().yaw}},
          {"commands",Json::array()},{"states",Json::array()},
          {"motion_rejected",unsafe},{"conditional_gaussian_union_upper",guard.fit_gaussian_motion?Json(risk.upper):Json(nullptr)},
          {"conditional_joint_valid",joint_risk.valid},{"conditional_joint_upper",joint?Json(joint_risk.upper):Json(nullptr)},
          {"conditional_joint_estimate",joint?Json(joint_risk.estimate):Json(nullptr)},
          {"joint_elapsed_ms",joint_risk.elapsed_ms},{"joint_error",joint_risk.error},
          {"conditional_gaussian_valid",risk.valid},
          {"conditional_gaussian_max_pose",risk.pose_upper.empty()?Json(nullptr):Json(*std::max_element(risk.pose_upper.begin(),risk.pose_upper.end()))},
          {"gaussian_objects",risk.gaussian_objects},{"reachable_objects",risk.reachable_objects},
          {"conflict",{{"reason",conflict.reason},{"id",conflict.track_id},
            {"t",conflict.state.t},{"ego",{conflict.state.x,conflict.state.y}},
            {"future_delta",{conflict.end_dx,conflict.end_dy}},{"reserve",conflict.reserve}}}});
        auto &record=candidates.back();
        for(const auto &c:commands)record["commands"].push_back({c.t,c.speed,c.steering});
        for(const auto &s:states)record["states"].push_back({s.t,s.x,s.y,s.yaw,s.speed,s.steering});
      };
      check("brake",0,0,0,{{0,0,0}});
      // Use the ACTUAL online library. A denser hand-authored grid can
      // falsely claim an escape the running controller never generates.
      for(auto &proposal:terrain_nav::forwardYieldProposals(state,model,escape_speed,
          geometry.max_steering,horizon,joint)) {
        const double steering=proposal.commands.front().steering;
        const bool bypass=proposal.commands.size()>2 && proposal.commands[1].speed!=0.;
        const double requested_speed=proposal.commands.front().speed;
        check(bypass?"bypass":"advance",proposal.distance,requested_speed,steering,std::move(proposal.commands));
      }
      for(auto &proposal:terrain_nav::reverseYieldProposals(state,model,reverse_speed,geometry.max_steering,horizon,joint)) {
        const double steering=proposal.commands.front().steering;
        const double speed=proposal.commands.front().speed;
        check("retreat",proposal.distance,speed,steering,std::move(proposal.commands));
      }
      Json objects=Json::array();for(const auto &t:*tracks)objects.push_back({{"id",t.id},{"circular",t.circular},
        {"center",{t.center_x,t.center_y}},{"velocity",{t.vx,t.vy}},{"radius",t.radius_bound},
        {"confirmed",t.motion_confirmed},{"center_motion_fit",t.center_motion_fit},{"velocity_error",t.velocity_error},
        {"gaussian_valid",t.gaussian.valid},{"gaussian_stamp",t.gaussian.stamp},{"gaussian_observation_std_m",t.gaussian.observation_std},
        {"gaussian_mean",{t.gaussian.mean[0],t.gaussian.mean[1],t.gaussian.mean[2],t.gaussian.mean[3]}},
        {"gaussian_diagonal",{t.gaussian.covariance(0,0),t.gaussian.covariance(1,1),t.gaussian.covariance(2,2),t.gaussian.covariance(3,3)}}});
      rows.push_back({{"time",time},{"report_time",report_time},{"exact_logged_callback_origin",exact_origin},{"exact_logged_risk_origin",exact_risk_origin},
        {"pose_source",source},{"exact_logged_control_root",exact_root},
        {"motion_source",matched_stamp},{"exact_logged_motion_source",exact_motion_source},{"motion_source_match_error",match},
        {"reconstructed_ego",{state.x,state.y,state.yaw,state.speed,state.steering}},
        {"online_message",r["status"][0]["message"]},{"objects",objects},{"counts",counts},{"candidates",candidates}});
    }
    Json out={{"scope","whole measured-mask replay and identical online motion geometry; NOT terrain, command or physical acceptance"},
      {"root_limitation","per-row exact_logged_control_root/exact_logged_motion_source identify actual diagnostics; otherwise observer receipt plus disclosed pose lead/20ms processing reconstruction is NOT claimed identical"},
      {"candidate_method","production forwardYieldProposals library; current qualified-center hypotheses enabled; old executed actions and outcomes are unchanged"},
      {"model_actuator_delay_s",model.delay},
      {"prediction_model",prediction_model},{"conditional_gaussian_budget",risk_limit},
      {"selected_report_stamp_s",std::isfinite(selected_report)?Json(selected_report):Json(nullptr)},
      {"gaussian_observation_window",guard.gaussian_maximum_observations},{"gaussian_acceleration_stress",guard.gaussian_acceleration_envelope},
      {"gaussian_constant_acceleration",guard.gaussian_constant_acceleration},{"white_jerk_density_m2_s5",guard.gaussian_jerk_density},
      {"sim_observation_calibration",guard.sim_observation_calibration},{"sim_domain_observation_std_m",guard.sim_observation_std},
      {"identity_memory_seconds",guard.identity_memory_seconds},{"turning_identity_reacquisition",guard.turning_identity_reacquisition},
      {"track_known_slow_centers",guard.track_known_slow_centers},
      {"counterfactual_escape_speed_mps",escape_speed},
      {"counterfactual_reverse_speed_mps",reverse_speed},
      {"joint_proposal_horizon_s",joint?Json(joint_horizon):Json(nullptr)},
      {"joint_tail_policy",joint_horizon>3.?"complete finite stop + .20s predicted zero-speed tail, minimum 3s":"fixed 3s, at least .20s predicted stop"},
      {"gaussian_assumption",{{"observation_std_m",guard.gaussian_observation_std},
         {"state_model",guard.gaussian_constant_acceleration?"CA6_white_jerk":"CV4_white_acceleration"},
         {"process_density_m2_s3",guard.gaussian_constant_acceleration?Json(nullptr):Json(guard.gaussian_process_density)},
         {"jerk_density_m2_s5",guard.gaussian_constant_acceleration?Json(guard.gaussian_jerk_density):Json(nullptr)},
         {"calibration","unqualified model assumption; replay is development, not held-out calibration"}}},
      {"prediction_speed_profile_mps",guard.prediction_maximum_speed_mps},
      {"source_run",run.string()},{"new_hypothesis_timeline",hypothesis_timeline},{"rows",rows}};
    out["input_stream"]=tracking?"tracking_frames":"frames (legacy body-only input)";
    std::ofstream output(argv[2]);if(!output)throw std::runtime_error("output creation failed");output<<out.dump(2)<<'\n';
    std::cout<<"replayed "<<rows.size()<<" interaction records\n";return 0;
  }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
